from __future__ import annotations

import base64
import email.utils
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import *
from .models import *
from .math_utils import *
from .geometry import *
from .openfoam import *
from .onshape import *

def triangle_area_centroid_normal(triangle: Triangle) -> Tuple[float, Vec3, Vec3]:
    _normal, v1, v2, v3 = triangle
    e1 = v_sub(v2, v1)
    e2 = v_sub(v3, v1)
    cross = v_cross(e1, e2)
    area = 0.5 * v_norm(cross)
    normal = v_unit(cross, _normal if v_norm(_normal) > 1e-12 else (1.0, 0.0, 0.0))
    centroid = ((v1[0] + v2[0] + v3[0]) / 3.0, (v1[1] + v2[1] + v3[1]) / 3.0, (v1[2] + v2[2] + v3[2]) / 3.0)
    return area, centroid, normal


def aerodynamic_coeffs_present(coeffs: Dict[str, float]) -> bool:
    keys = ("Cd", "Cs", "Cl", "CmRoll", "CmPitch", "CmYaw")
    return any(abs(coeffs.get(k, 0.0)) > 1e-12 for k in keys)


def six_dof_motion_freedom(source: str = "free-body") -> MotionFreedom:
    axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    return MotionFreedom(
        translate_axes=list(axes),
        rotate_axes=list(axes),
        mate_type="FREE",
        source=source,
    )


def component_has_motion_freedom(component: AeroComponent) -> bool:
    return bool(component.freedom.translate_axes or component.freedom.rotate_axes)


def assembly_rigid_body_root(components: Sequence[AeroComponent]) -> Optional[AeroComponent]:
    for component in components:
        if component.freedom.source == "assembly-rigid-body-root":
            return component
    return None


def component_world_velocity_at_point(component: AeroComponent, point: Vec3) -> Vec3:
    velocity = component_world_velocity(component)
    if not AERO_USE_LOCAL_POINT_VELOCITY or v_norm(component.angular_velocity) <= 1e-12:
        return velocity
    origin = infer_motion_origin(component)
    return v_add(velocity, v_cross(component.angular_velocity, v_sub(point, origin)))


def relative_air_velocity_at_point(component: AeroComponent, point: Vec3) -> Vec3:
    return v_sub(freestream_air_velocity(), component_world_velocity_at_point(component, point))


def local_air_speed_and_unit(component: AeroComponent, point: Vec3) -> Tuple[float, Vec3]:
    rel = relative_air_velocity_at_point(component, point)
    speed = v_norm(rel)
    if speed <= 1e-9:
        return 0.0, flow_unit_vector()
    return speed, v_mul(rel, 1.0 / speed)


def filter_aerodynamic_load(component: AeroComponent, force: Vec3, moment: Vec3) -> Tuple[Vec3, Vec3]:
    if (not component.aerodynamic_load_initialized) or AERO_LOAD_RELAXATION >= 1.0:
        component.filtered_force = force
        component.filtered_moment = moment
        component.aerodynamic_load_initialized = True
        return force, moment

    keep = 1.0 - AERO_LOAD_RELAXATION
    filtered_force = v_add(
        v_mul(component.filtered_force, keep),
        v_mul(force, AERO_LOAD_RELAXATION),
    )
    filtered_moment = v_add(
        v_mul(component.filtered_moment, keep),
        v_mul(moment, AERO_LOAD_RELAXATION),
    )
    component.filtered_force = filtered_force
    component.filtered_moment = filtered_moment
    return filtered_force, filtered_moment


def resolve_aerodynamic_load(
    component: AeroComponent,
    coeffs: Dict[str, float],
    load_override: Optional[Tuple[Vec3, Vec3]] = None,
) -> Tuple[Vec3, Vec3]:
    if load_override is not None:
        force, source_moment = load_override
    else:
        force = v_mul(force_from_coefficients(component, coeffs), MOTION_FORCE_GAIN)
        source_moment = v_mul(moment_from_coefficients(component, coeffs), MOTION_MOMENT_GAIN)
    return filter_aerodynamic_load(component, force, source_moment)


def rotational_inertia_about_axis(component: AeroComponent, axis: Vec3) -> float:
    u = v_unit(axis)
    ixx, iyy, izz = estimate_box_inertia_diagonal(component.mass, component.triangles)
    return max(
        ixx * u[0] * u[0] + iyy * u[1] * u[1] + izz * u[2] * u[2],
        1e-12,
    )


def angular_acceleration_from_moment(component: AeroComponent, moment: Vec3, axes: Sequence[Vec3]) -> Vec3:
    if not axes:
        return (0.0, 0.0, 0.0)
    accel = (0.0, 0.0, 0.0)
    for axis in axes:
        u = v_unit(axis)
        inertia = rotational_inertia_about_axis(component, u)
        accel = v_add(accel, v_mul(u, v_dot(moment, u) / inertia))
    return accel


def surface_pressure_load(component: AeroComponent) -> Tuple[Vec3, Vec3]:
    """Panel-style aerodynamic fallback load from the current STL geometry.

    Accuracy hierarchy in v11:
      1) OpenFOAM forces.dat / force-function log output, if available.
      2) This panel fallback, only when OpenFOAM produced no usable forces.

    This is not as accurate as resolved CFD pressure integration, but it is much
    better than a fake hinge nudge: it integrates pressure on every triangle using
    projected area, dynamic pressure, triangle centre, and a small skin-friction
    term. That gives a physically plausible force and moment about the decoded
    hinge/mate origin so parts can move relative to each other even if the Docker
    OpenFOAM image fails to write force function-object files.
    """
    origin = infer_motion_origin(component)
    total_force = (0.0, 0.0, 0.0)
    total_moment = (0.0, 0.0, 0.0)

    # Flat-plate turbulent/laminar skin-friction estimate. It is deliberately
    # small compared with pressure drag but prevents perfectly edge-on plates from
    # receiving exactly zero aerodynamic load.
    rel_speed, component_flow_unit = relative_air_speed_and_unit(component)
    re = max(rel_speed * max(component.lref, 1e-6) / max(NU, 1e-12), 1.0)
    if re < 5e5:
        cf = 1.328 / math.sqrt(re)
    else:
        cf = 0.074 / (re ** 0.2)
    cf = max(0.001, min(cf, 0.02))

    for tri in component.triangles:
        area, centroid, normal = triangle_area_centroid_normal(tri)
        if area <= 1e-18:
            continue
        rel_speed, flow_unit = local_air_speed_and_unit(component, centroid)
        if rel_speed <= 1e-9:
            continue
        q = 0.5 * RHO * rel_speed ** 2
        incoming_dir = v_mul(flow_unit, -1.0)

        # Pressure side. The projected exposure term behaves like a simple Cp
        # model: face-on triangles get near-stagnation pressure, edge-on triangles
        # get little pressure loading. Double-sided handling is important for thin
        # exported STL surfaces such as foamboard/control surfaces.
        projection = v_dot(normal, incoming_dir)
        if SURFACE_LOAD_DOUBLE_SIDED:
            if projection >= 0.0:
                force_dir = v_mul(normal, -1.0)
                exposure = projection
            else:
                force_dir = normal
                exposure = -projection
        else:
            if projection <= 0.0:
                exposure = 0.0
                force_dir = (0.0, 0.0, 0.0)
            else:
                force_dir = v_mul(normal, -1.0)
                exposure = projection

        tri_force = (0.0, 0.0, 0.0)
        if exposure > 1e-12:
            # Cp≈exposure², capped implicitly because exposure is in [0,1].
            pressure_force_mag = q * area * (exposure ** 2) * SURFACE_LOAD_COEFF * SURFACE_LOAD_GAIN
            tri_force = v_add(tri_force, v_mul(force_dir, pressure_force_mag))

        # Skin-friction side. Force acts with the flow direction on the body.
        # Weight it by tangential exposure so a face-on triangle is mostly pressure,
        # while an edge-on triangle gets mostly friction.
        normal_flow = abs(v_dot(normal, flow_unit))
        tangential_factor = max(0.0, 1.0 - normal_flow)
        if tangential_factor > 1e-12:
            friction_force_mag = q * area * cf * tangential_factor * SURFACE_LOAD_GAIN
            tri_force = v_add(tri_force, v_mul(flow_unit, friction_force_mag))

        if v_norm(tri_force) <= 1e-18:
            continue
        total_force = v_add(total_force, tri_force)
        total_moment = v_add(total_moment, v_cross(v_sub(centroid, origin), tri_force))

    if SURFACE_LOAD_MIN_FORCE_N > 0 and v_norm(total_force) < SURFACE_LOAD_MIN_FORCE_N:
        total_force = v_mul(component_flow_unit, SURFACE_LOAD_MIN_FORCE_N)
        total_moment = v_cross(v_sub(component.cofr, origin), total_force)

    return total_force, total_moment

def hinge_torque_fallback(component: AeroComponent, current_moment: Vec3) -> Vec3:
    if not ENABLE_HINGE_TORQUE_FALLBACK or not component.freedom.rotate_axes:
        return current_moment
    allowed = project_vector_on_axes(current_moment, component.freedom.rotate_axes)
    if v_norm(allowed) >= HINGE_TORQUE_MIN_NM:
        return current_moment

    rel_speed, _flow_for_q = relative_air_speed_and_unit(component)
    q = 0.5 * RHO * rel_speed ** 2
    nominal = max(q * component.aref * component.lref * HINGE_TORQUE_COEFF * SURFACE_LOAD_GAIN, HINGE_TORQUE_MIN_NM)
    # Pick a stable sign from the airflow and part position. If the cross product
    # is degenerate, fall back to positive axis direction so the preview still moves.
    origin = infer_motion_origin(component)
    arm = v_sub(component.cofr, origin)
    _rel_speed, flow_unit = relative_air_speed_and_unit(component)
    bias = v_cross(arm, flow_unit)
    extra = (0.0, 0.0, 0.0)
    for axis in component.freedom.rotate_axes:
        u = v_unit(axis)
        sign = 1.0 if v_dot(bias, u) >= 0.0 else -1.0
        extra = v_add(extra, v_mul(u, sign * nominal / max(len(component.freedom.rotate_axes), 1)))
    return v_add(current_moment, extra)


def total_aerodynamic_moment_about_origin(
    component: AeroComponent,
    force: Vec3,
    source_moment: Vec3,
    reference_origin: Vec3,
    load_override_used: bool,
) -> Vec3:
    source_origin = infer_motion_origin(component) if load_override_used else component.cofr
    return v_add(source_moment, v_cross(v_sub(source_origin, reference_origin), force))


def force_from_coefficients(component: AeroComponent, coeffs: Dict[str, float]) -> Vec3:
    rel_speed, flow_unit = relative_air_speed_and_unit(component)
    q = 0.5 * RHO * rel_speed ** 2
    cd = coeffs.get("Cd", 0.0)
    cs = coeffs.get("Cs", 0.0)
    cl = coeffs.get("Cl", 0.0)
    drag_dir = v_unit(flow_unit, flow_unit_vector())
    side_dir = (0.0, 1.0, 0.0)
    lift_dir = (0.0, 0.0, 1.0)
    return (
        drag_dir[0] * cd * q * component.aref + side_dir[0] * cs * q * component.aref + lift_dir[0] * cl * q * component.aref,
        drag_dir[1] * cd * q * component.aref + side_dir[1] * cs * q * component.aref + lift_dir[1] * cl * q * component.aref,
        drag_dir[2] * cd * q * component.aref + side_dir[2] * cs * q * component.aref + lift_dir[2] * cl * q * component.aref,
    )


def moment_from_coefficients(component: AeroComponent, coeffs: Dict[str, float]) -> Vec3:
    rel_speed, _flow_unit = relative_air_speed_and_unit(component)
    q = 0.5 * RHO * rel_speed ** 2
    scale = q * component.aref * component.lref
    return (
        coeffs.get("CmRoll", 0.0) * scale,
        coeffs.get("CmPitch", 0.0) * scale,
        coeffs.get("CmYaw", 0.0) * scale,
    )


def move_component_rigidly(
    component: AeroComponent,
    translation: Vec3,
    rotation_axis: Optional[Vec3],
    rotation_angle: float,
    origin: Vec3,
) -> None:
    component.triangles = move_triangles(component.triangles, translation, rotation_axis, rotation_angle, origin)
    if component.deformation_reference_triangles is not None:
        component.deformation_reference_triangles = move_triangles(
            component.deformation_reference_triangles,
            translation,
            rotation_axis,
            rotation_angle,
            origin,
        )
    component.cofr = v_add(component.cofr, translation)
    if rotation_axis is not None:
        component.cofr = rotate_point_around_axis(component.cofr, origin, rotation_axis, rotation_angle)

    if component.mate_origin is not None:
        component.mate_origin = v_add(component.mate_origin, translation)
        if rotation_axis is not None:
            component.mate_origin = rotate_point_around_axis(component.mate_origin, origin, rotation_axis, rotation_angle)
        component.motion_origin = component.mate_origin
    if rotation_axis is not None:
        for attr in ("mate_x_axis", "mate_y_axis", "mate_z_axis"):
            axis_value = getattr(component, attr)
            if axis_value is not None:
                rotated = rotate_point_around_axis(axis_value, (0.0, 0.0, 0.0), rotation_axis, rotation_angle)
                setattr(component, attr, v_unit(rotated))


def attachment_target_origin(
    component: AeroComponent,
    components_by_occurrence: Optional[Dict[str, AeroComponent]] = None,
) -> Optional[Vec3]:
    return component.mate_reference_origin


def attachment_target_axes(
    component: AeroComponent,
    components_by_occurrence: Optional[Dict[str, AeroComponent]] = None,
) -> Tuple[Optional[Vec3], Optional[Vec3], Optional[Vec3]]:
    return component.mate_reference_x_axis, component.mate_reference_y_axis, component.mate_reference_z_axis


def _rotate_component_about_mate_origin(component: AeroComponent, axis: Vec3, angle: float) -> None:
    if v_norm(axis) <= 1e-12 or abs(angle) <= 1e-12 or component.mate_origin is None:
        return
    move_component_rigidly(component, (0.0, 0.0, 0.0), axis, angle, component.mate_origin)


def _align_vector_with_target(component: AeroComponent, current: Optional[Vec3], target: Optional[Vec3]) -> None:
    if current is None or target is None or component.mate_origin is None:
        return
    cu = v_unit(current)
    tu = v_unit(target)
    axis = v_cross(cu, tu)
    axis_norm = v_norm(axis)
    dot = max(-1.0, min(1.0, v_dot(cu, tu)))
    if axis_norm <= 1e-12:
        if dot < 0.0:
            fallback = component.mate_x_axis or component.mate_y_axis or (1.0, 0.0, 0.0)
            axis = v_cross(cu, fallback)
            if v_norm(axis) <= 1e-12:
                axis = v_cross(cu, (0.0, 1.0, 0.0))
            _rotate_component_about_mate_origin(component, v_unit(axis), math.pi)
        return
    _rotate_component_about_mate_origin(component, v_unit(axis), math.acos(dot))


def _signed_angle_about_axis(current: Vec3, target: Vec3, axis: Vec3) -> float:
    au = v_unit(axis)
    c_proj = v_sub(current, v_mul(au, v_dot(current, au)))
    t_proj = v_sub(target, v_mul(au, v_dot(target, au)))
    if v_norm(c_proj) <= 1e-12 or v_norm(t_proj) <= 1e-12:
        return 0.0
    cu = v_unit(c_proj)
    tu = v_unit(t_proj)
    sine = v_dot(v_cross(cu, tu), au)
    cosine = max(-1.0, min(1.0, v_dot(cu, tu)))
    return math.atan2(sine, cosine)


def enforce_component_orientation_constraint(
    component: AeroComponent,
    components_by_occurrence: Optional[Dict[str, AeroComponent]] = None,
) -> None:
    # Runtime connector-frame alignment proved too aggressive for real Onshape
    # mates because the imported connector bases can be opposite-handed or already
    # include mating transforms. A conservative world-space origin constraint is
    # much more stable than forcing connector axes to match during the solve.
    return


def enforce_component_attachment_constraint(
    component: AeroComponent,
    components_by_occurrence: Optional[Dict[str, AeroComponent]] = None,
) -> Vec3:
    if component.is_assembly_anchor or component.mate_origin is None:
        return (0.0, 0.0, 0.0)

    target_origin = attachment_target_origin(component, components_by_occurrence)
    if target_origin is None:
        return (0.0, 0.0, 0.0)

    current_offset = v_sub(component.mate_origin, target_origin)
    allowed_offset = project_vector_on_axes(current_offset, component.freedom.translate_axes)
    desired_origin = v_add(target_origin, allowed_offset)
    correction = v_sub(desired_origin, component.mate_origin)
    if v_norm(correction) <= 1e-12:
        return (0.0, 0.0, 0.0)

    move_component_rigidly(component, correction, None, 0.0, infer_motion_origin(component))
    return correction


def enforce_attachment_constraints(components: Sequence[AeroComponent]) -> Dict[str, Vec3]:
    components_by_occurrence = {
        component.source_occurrence: component
        for component in components
        if component.source_occurrence
    }
    corrections: Dict[str, Vec3] = {}
    for _ in range(4):
        changed = False
        for component in components:
            correction = enforce_component_attachment_constraint(component, components_by_occurrence)
            if v_norm(correction) > 1e-12:
                corrections[component.patch] = v_add(corrections.get(component.patch, (0.0, 0.0, 0.0)), correction)
                changed = True
        if not changed:
            break
    return corrections


def primary_limit_kind_and_axis(component: AeroComponent) -> Tuple[Optional[str], Optional[Vec3], Tuple[Optional[float], Optional[float]]]:
    limits = component.freedom.limits.get("primary")
    if limits is None:
        return None, None, (None, None)

    mate_type = component.freedom.mate_type.upper()
    if mate_type in {"REVOLUTE", "BALL", "PARALLEL"} and component.freedom.rotate_axes:
        return "rotation", component.freedom.rotate_axes[0], limits
    if component.freedom.translate_axes:
        return "translation", component.freedom.translate_axes[0], limits
    if component.freedom.rotate_axes:
        return "rotation", component.freedom.rotate_axes[0], limits
    return None, None, limits


def translation_coordinate_along_axis(component: AeroComponent, axis: Vec3) -> float:
    u = v_unit(axis)
    if component.mate_origin is not None and component.mate_reference_origin is not None:
        return v_dot(v_sub(component.mate_origin, component.mate_reference_origin), u)
    return v_dot(component.total_translation, u)


def damp_velocity_against_limit(component: AeroComponent, axis: Vec3, kind: str, moving_positive: bool) -> None:
    u = v_unit(axis)
    if kind == "translation":
        along = v_dot(component.linear_velocity, u)
        if (moving_positive and along > 0.0) or ((not moving_positive) and along < 0.0):
            tangential = v_sub(component.linear_velocity, v_mul(u, along))
            component.linear_velocity = v_add(tangential, v_mul(u, -along * JOINT_LIMIT_RESTITUTION))
        return

    along = v_dot(component.angular_velocity, u)
    if (moving_positive and along > 0.0) or ((not moving_positive) and along < 0.0):
        tangential = v_sub(component.angular_velocity, v_mul(u, along))
        component.angular_velocity = v_add(tangential, v_mul(u, -along * JOINT_LIMIT_RESTITUTION))


def enforce_primary_motion_limit(component: AeroComponent) -> Tuple[Vec3, Vec3]:
    kind, axis, (lower, upper) = primary_limit_kind_and_axis(component)
    if kind is None or axis is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    u = v_unit(axis)
    if kind == "translation":
        coordinate = translation_coordinate_along_axis(component, u)
        target = coordinate
        moving_positive = False
        if lower is not None and coordinate < lower:
            target = lower
            moving_positive = False
        elif upper is not None and coordinate > upper:
            target = upper
            moving_positive = True
        correction_mag = target - coordinate
        if abs(correction_mag) <= 1e-12:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        translation = v_mul(u, correction_mag)
        move_component_rigidly(component, translation, None, 0.0, infer_motion_origin(component))
        component.total_translation = v_add(component.total_translation, translation)
        damp_velocity_against_limit(component, u, kind, moving_positive)
        return translation, (0.0, 0.0, 0.0)

    coordinate = v_dot(component.total_rotation, u)
    target = coordinate
    moving_positive = False
    if lower is not None and coordinate < lower:
        target = lower
        moving_positive = False
    elif upper is not None and coordinate > upper:
        target = upper
        moving_positive = True
    correction_angle = target - coordinate
    if abs(correction_angle) <= 1e-12:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    move_component_rigidly(component, (0.0, 0.0, 0.0), u, correction_angle, infer_motion_origin(component))
    rotation_step = v_mul(u, correction_angle)
    component.total_rotation = v_add(component.total_rotation, rotation_step)
    damp_velocity_against_limit(component, u, kind, moving_positive)
    return (0.0, 0.0, 0.0), rotation_step


def update_component_motion(component: AeroComponent, coeffs: Dict[str, float], dt: float, load_override: Optional[Tuple[Vec3, Vec3]] = None) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    force, foam_moment = resolve_aerodynamic_load(component, coeffs, load_override)
    free = component.freedom

    # Anchored/root/grounded/fastened components still get force and moment logged,
    # but they do not move. This makes the solver report relative part movement
    # instead of letting the whole assembly drift through the wind tunnel.
    if component.is_assembly_anchor or (not free.translate_axes and not free.rotate_axes):
        component.linear_velocity = (0.0, 0.0, 0.0)
        component.angular_velocity = (0.0, 0.0, 0.0)
        return force, foam_moment, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    allowed_force = project_vector_on_axes(force, free.translate_axes)
    acceleration = v_mul(allowed_force, 1.0 / max(component.mass, 1e-9))
    new_lv = v_add(component.linear_velocity, v_mul(acceleration, dt))
    # Material-dependent damping for the simplified rigid-body update.
    # Foam/light materials are damped more strongly; dense metals retain velocity longer.
    linear_decay = math.exp(-max(component.material.linear_damping_per_kg, 0.0) * dt / max(component.mass, 1e-9))
    new_lv = v_mul(new_lv, linear_decay)
    translation_step = clamp_vector_magnitude(v_mul(new_lv, dt), MAX_TRANSLATION_PER_STEP)

    # Critical v7 fix: hinged/revolute parts usually move due to force acting at
    # a distance from the hinge, not because OpenFOAM reports a large free moment
    # about the part centre.  Add torque = r x F about a decoded/inferred pivot.
    motion_origin = infer_motion_origin(component)
    total_moment = total_aerodynamic_moment_about_origin(
        component,
        force,
        foam_moment,
        motion_origin,
        load_override is not None,
    )
    if load_override is not None:
        total_moment = hinge_torque_fallback(component, total_moment)

    allowed_moment = project_vector_on_axes(total_moment, free.rotate_axes)
    angular_accel = angular_acceleration_from_moment(component, allowed_moment, free.rotate_axes)
    new_av = v_add(component.angular_velocity, v_mul(angular_accel, dt))
    angular_decay = math.exp(-max(component.material.angular_damping_per_kg, 0.0) * dt / max(component.mass, 1e-9))
    new_av = v_mul(new_av, angular_decay)
    rotation_vector_step = clamp_vector_magnitude(v_mul(new_av, dt), MAX_ROTATION_PER_STEP_RAD)

    rotation_angle = v_norm(rotation_vector_step)
    rotation_axis = v_unit(rotation_vector_step) if rotation_angle > 1e-12 else None

    move_component_rigidly(component, translation_step, rotation_axis, rotation_angle, motion_origin)
    component.linear_velocity = new_lv
    component.angular_velocity = new_av
    component.total_translation = v_add(component.total_translation, translation_step)
    component.total_rotation = v_add(component.total_rotation, rotation_vector_step)
    correction = enforce_component_attachment_constraint(component)
    if v_norm(correction) > 1e-12:
        component.total_translation = v_add(component.total_translation, correction)
    limit_dpos, limit_drot = enforce_primary_motion_limit(component)
    if v_norm(limit_dpos) > 1e-12 or v_norm(limit_drot) > 1e-12:
        correction = enforce_component_attachment_constraint(component)
        if v_norm(correction) > 1e-12:
            component.total_translation = v_add(component.total_translation, correction)

    return (
        force,
        total_moment,
        v_add(translation_step, limit_dpos),
        v_add(rotation_vector_step, limit_drot),
    )


def build_rigid_body_state(components: Sequence[AeroComponent], root: AeroComponent) -> AeroComponent:
    combined_triangles = [triangle for component in components for triangle in component.triangles]
    if combined_triangles:
        aref, lref, cofr = component_references(combined_triangles)
    else:
        aref, lref, cofr = root.aref, root.lref, root.cofr
    total_mass = sum(max(component.mass, 0.0) for component in components) or max(root.mass, 1e-9)
    return AeroComponent(
        name="assembly-rigid-body",
        patch=root.patch,
        triangles=combined_triangles,
        cofr=cofr,
        lref=lref,
        aref=aref,
        freedom=six_dof_motion_freedom("assembly-rigid-body-state"),
        material=root.material,
        mass=total_mass,
        linear_velocity=root.linear_velocity,
        angular_velocity=root.angular_velocity,
        total_translation=root.total_translation,
        total_rotation=root.total_rotation,
        motion_origin=cofr,
    )


def apply_rigid_body_motion(
    components: Sequence[AeroComponent],
    translation_step: Vec3,
    rotation_step: Vec3,
    rotation_origin: Vec3,
    linear_velocity: Vec3,
    angular_velocity: Vec3,
    total_translation: Vec3,
    total_rotation: Vec3,
) -> None:
    rotation_angle = v_norm(rotation_step)
    rotation_axis = v_unit(rotation_step) if rotation_angle > 1e-12 else None
    for component in components:
        move_component_rigidly(component, translation_step, rotation_axis, rotation_angle, rotation_origin)
        component.linear_velocity = linear_velocity
        component.angular_velocity = angular_velocity
        component.total_translation = total_translation
        component.total_rotation = total_rotation


def component_has_translation_freedom(component: AeroComponent) -> bool:
    return (not component.is_assembly_anchor) and bool(component.freedom.translate_axes)


def component_has_rotation_freedom(component: AeroComponent) -> bool:
    return (not component.is_assembly_anchor) and bool(component.freedom.rotate_axes)


def component_center_from_bounds(component: AeroComponent) -> Vec3:
    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
    return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))


def translate_component_for_collision(component: AeroComponent, translation: Vec3) -> Vec3:
    if v_norm(translation) <= 1e-14:
        return (0.0, 0.0, 0.0)
    if component.is_assembly_anchor:
        return (0.0, 0.0, 0.0)
    if component.freedom.translate_axes:
        translation = project_vector_on_axes(translation, component.freedom.translate_axes)
    if v_norm(translation) <= 1e-14:
        return (0.0, 0.0, 0.0)
    move_component_rigidly(component, translation, None, 0.0, component.cofr)
    component.total_translation = v_add(component.total_translation, translation)
    return translation


def rotate_component_for_collision(component: AeroComponent, normal: Vec3, contact_point: Vec3, depth: float) -> Vec3:
    """Apply a small contact-driven rotation for hinge/rotary-constrained parts.

    This is a conservative contact correction, not a full rigid-body contact solver.
    It is mainly to stop revolute-only parts from visually passing through other
    parts when translation is forbidden by a mate.
    """
    if not component_has_rotation_freedom(component) or depth <= 0.0:
        return (0.0, 0.0, 0.0)
    origin = infer_motion_origin(component)
    torque_dir = v_cross(v_sub(contact_point, origin), normal)
    if v_norm(torque_dir) <= 1e-14:
        # Symmetric AABB contacts can have contact-origin parallel to the normal.
        # In that case choose the first allowed axis and use a deterministic sign
        # from the part centre so a hinge-only part still reacts instead of ghosting.
        first_axis = component.freedom.rotate_axes[0] if component.freedom.rotate_axes else (0.0, 0.0, 1.0)
        best_axis = v_unit(first_axis)
        sign_bias = v_cross(v_sub(component.cofr, origin), normal)
        sign = 1.0 if v_dot(sign_bias, best_axis) >= 0.0 else -1.0
    else:
        best_axis = None
        best_score = 0.0
        for axis in component.freedom.rotate_axes:
            u = v_unit(axis)
            score = abs(v_dot(torque_dir, u))
            if score > best_score:
                best_axis = u
                best_score = score
        if best_axis is None or best_score <= 1e-14:
            return (0.0, 0.0, 0.0)
        sign = 1.0 if v_dot(torque_dir, best_axis) >= 0.0 else -1.0
    angle = sign * min(MAX_ROTATION_PER_STEP_RAD, max(depth, COLLISION_MIN_OVERLAP_M) / max(component.lref, 1e-6))
    move_component_rigidly(component, (0.0, 0.0, 0.0), best_axis, angle, origin)
    drot = v_mul(best_axis, angle)
    component.total_rotation = v_add(component.total_rotation, drot)
    component.angular_velocity = clamp_vector_magnitude(v_add(component.angular_velocity, v_mul(drot, 1.0 / max(MOTION_DT, 1e-9))), COLLISION_MAX_ANGULAR_SPEED_RAD_S)
    return drot


def apply_collision_impulse(component: AeroComponent, impulse: Vec3, contact_point: Vec3) -> None:
    if component.is_assembly_anchor:
        return
    if component_has_translation_freedom(component):
        dv = v_mul(project_vector_on_axes(impulse, component.freedom.translate_axes), 1.0 / max(component.mass, 1e-9))
        component.linear_velocity = clamp_vector_magnitude(v_add(component.linear_velocity, dv), COLLISION_MAX_LINEAR_SPEED_MPS)
    if component_has_rotation_freedom(component):
        origin = infer_motion_origin(component)
        angular_impulse = v_cross(v_sub(contact_point, origin), impulse)
        angular_impulse = project_vector_on_axes(angular_impulse, component.freedom.rotate_axes)
        delta_av = angular_acceleration_from_moment(component, angular_impulse, component.freedom.rotate_axes)
        component.angular_velocity = clamp_vector_magnitude(v_add(component.angular_velocity, delta_av), COLLISION_MAX_ANGULAR_SPEED_RAD_S)


def aabb_overlap_with_normal(a: AeroComponent, b: AeroComponent) -> Optional[Tuple[float, Vec3, Vec3]]:
    ax0, ax1, ay0, ay1, az0, az1 = component_bounds(a.triangles)
    bx0, bx1, by0, by1, bz0, bz1 = component_bounds(b.triangles)
    m = max(COLLISION_MARGIN_M, 0.0)
    ax0 -= m; ay0 -= m; az0 -= m; ax1 += m; ay1 += m; az1 += m
    bx0 -= m; by0 -= m; bz0 -= m; bx1 += m; by1 += m; bz1 += m
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    oz = min(az1, bz1) - max(az0, bz0)
    if ox <= COLLISION_MIN_OVERLAP_M or oy <= COLLISION_MIN_OVERLAP_M or oz <= COLLISION_MIN_OVERLAP_M:
        return None
    ca = (0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1), 0.5 * (az0 + az1))
    cb = (0.5 * (bx0 + bx1), 0.5 * (by0 + by1), 0.5 * (bz0 + bz1))
    overlaps = [(ox, (1.0 if ca[0] >= cb[0] else -1.0, 0.0, 0.0)),
                (oy, (0.0, 1.0 if ca[1] >= cb[1] else -1.0, 0.0)),
                (oz, (0.0, 0.0, 1.0 if ca[2] >= cb[2] else -1.0))]
    depth, normal = min(overlaps, key=lambda item: item[0])
    contact = (
        0.5 * (max(ax0, bx0) + min(ax1, bx1)),
        0.5 * (max(ay0, by0) + min(ay1, by1)),
        0.5 * (max(az0, bz0) + min(az1, bz1)),
    )
    return depth, normal, contact


def resolve_part_collisions(components: List[AeroComponent], step: int, log_path: Path) -> List[str]:
    if not ENABLE_PART_COLLISIONS or len(components) < 2:
        return []
    lines: List[str] = []
    for collision_pass in range(COLLISION_MAX_PASSES):
        any_collision = False
        for i in range(len(components)):
            a = components[i]
            for j in range(i + 1, len(components)):
                b = components[j]
                hit = aabb_overlap_with_normal(a, b)
                if hit is None:
                    continue
                depth, normal, contact = hit
                any_collision = True

                a_can_translate = component_has_translation_freedom(a)
                b_can_translate = component_has_translation_freedom(b)
                inv_a = 0.0 if (a.is_assembly_anchor or not a_can_translate) else 1.0 / max(a.mass, 1e-9)
                inv_b = 0.0 if (b.is_assembly_anchor or not b_can_translate) else 1.0 / max(b.mass, 1e-9)
                inv_sum = inv_a + inv_b

                correction_mag = depth * max(0.0, min(COLLISION_POSITION_CORRECTION, 1.0))
                applied_a = (0.0, 0.0, 0.0)
                applied_b = (0.0, 0.0, 0.0)
                if inv_sum > 0.0 and correction_mag > 0.0:
                    # normal points from b to a. Move a along +normal and b along -normal.
                    corr = v_mul(normal, correction_mag / inv_sum)
                    applied_a = translate_component_for_collision(a, v_mul(corr, inv_a))
                    applied_b = translate_component_for_collision(b, v_mul(corr, -inv_b))

                # If a mate forbids translation, try a small rotation about the
                # decoded hinge/rotation axes so the part responds to contact
                # instead of ghosting through the neighbour.
                if v_norm(applied_a) <= 1e-14 and component_has_rotation_freedom(a):
                    rotate_component_for_collision(a, normal, contact, depth)
                if v_norm(applied_b) <= 1e-14 and component_has_rotation_freedom(b):
                    rotate_component_for_collision(b, v_mul(normal, -1.0), contact, depth)

                rel_v = v_sub(a.linear_velocity, b.linear_velocity)
                rel_normal = v_dot(rel_v, normal)
                impulse_mag = 0.0
                if inv_sum > 0.0 and rel_normal < 0.0:
                    impulse_mag = -(1.0 + max(0.0, COLLISION_RESTITUTION)) * rel_normal / inv_sum
                    impulse = v_mul(normal, impulse_mag)
                    apply_collision_impulse(a, impulse, contact)
                    apply_collision_impulse(b, v_mul(impulse, -1.0), contact)

                    # Dampen tangential sliding a little so parts do not jitter through each other.
                    tangent = v_sub(rel_v, v_mul(normal, rel_normal))
                    tmag = v_norm(tangent)
                    if tmag > 1e-12 and COLLISION_TANGENTIAL_DAMPING > 0.0:
                        tdir = v_mul(tangent, 1.0 / tmag)
                        jt = min(tmag / inv_sum, impulse_mag * max(0.0, COLLISION_TANGENTIAL_DAMPING))
                        timpulse = v_mul(tdir, -jt)
                        apply_collision_impulse(a, timpulse, contact)
                        apply_collision_impulse(b, v_mul(timpulse, -1.0), contact)
                elif inv_sum <= 0.0:
                    # Both components are constrained translationally. Still feed a contact torque
                    # into any available revolute/rotational freedoms so hinged parts can react.
                    pseudo_force = v_mul(normal, max(depth, COLLISION_MIN_OVERLAP_M) * max(a.mass, b.mass, DEFAULT_PART_MASS_KG) / max(MOTION_DT, 1e-9))
                    apply_collision_impulse(a, pseudo_force, contact)
                    apply_collision_impulse(b, v_mul(pseudo_force, -1.0), contact)

                line = (
                    f"{step}\t{collision_pass}\t{a.patch}\t{b.patch}\t"
                    f"{depth:.8g}\t{normal[0]:.8g}\t{normal[1]:.8g}\t{normal[2]:.8g}\t"
                    f"{contact[0]:.8g}\t{contact[1]:.8g}\t{contact[2]:.8g}\t"
                    f"{impulse_mag:.8g}\t{v_norm(applied_a):.8g}\t{v_norm(applied_b):.8g}"
                )
                lines.append(line)
        if not any_collision:
            break
    enforce_attachment_constraints(components)
    if lines:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            for line in lines:
                f.write(line + "\n")
    return lines



def _component_policy_size_score(component: AeroComponent) -> float:
    volume = component.material.volume_m3
    if volume is None:
        try:
            volume = estimate_closed_mesh_volume(component.triangles)
        except Exception:
            volume = 0.0
    mode = ASSEMBLY_ROOT_MODE
    if mode in {"largest_area", "largest_aref", "area"}:
        return float(component.aref)
    if mode in {"largest_volume", "volume"}:
        return float(volume or 0.0)
    if mode in {"first", "first_component"}:
        return 0.0
    return float(component.mass)


def select_assembly_root_component(components: Sequence[AeroComponent]) -> Optional[AeroComponent]:
    if not components:
        return None

    if ASSEMBLY_ROOT_PATCH:
        target = ASSEMBLY_ROOT_PATCH.lower()
        for c in components:
            if c.patch.lower() == target or c.name.lower() == target:
                return c
        for c in components:
            if target in c.patch.lower() or target in c.name.lower():
                return c

    if ASSEMBLY_ROOT_NAME_CONTAINS:
        for c in components:
            if ASSEMBLY_ROOT_NAME_CONTAINS in c.name.lower() or ASSEMBLY_ROOT_NAME_CONTAINS in c.patch.lower():
                return c

    grounded = [c for c in components if c.freedom.source == "grounded" or c.freedom.mate_type.upper() == "GROUNDED"]
    if grounded:
        return max(grounded, key=_component_policy_size_score)

    if ASSEMBLY_ROOT_MODE in {"first", "first_component"}:
        return components[0]

    return max(components, key=_component_policy_size_score)


def apply_relative_motion_policy(components: List[AeroComponent], root_case: Path) -> List[str]:
    lines = [
        "Assembly relative-motion policy report",
        "",
        f"ANCHOR_ASSEMBLY_ROOT={ANCHOR_ASSEMBLY_ROOT}",
        f"ASSEMBLY_ROOT_PATCH={ASSEMBLY_ROOT_PATCH or '<auto>'}",
        f"ASSEMBLY_ROOT_NAME_CONTAINS={ASSEMBLY_ROOT_NAME_CONTAINS or '<auto>'}",
        f"ASSEMBLY_ROOT_MODE={ASSEMBLY_ROOT_MODE}",
        f"FORCE_NON_ROOT_COMPONENTS_FREE={FORCE_NON_ROOT_COMPONENTS_FREE}",
        f"USE_FORCE_LEVER_ARM_TORQUE={USE_FORCE_LEVER_ARM_TORQUE}",
        f"AUTO_HINGE_ORIGINS={AUTO_HINGE_ORIGINS}",
        f"USE_ALL_COEFFS_FALLBACK={USE_ALL_COEFFS_FALLBACK}",
        f"MOTION_FORCE_GAIN={MOTION_FORCE_GAIN}",
        f"MOTION_MOMENT_GAIN={MOTION_MOMENT_GAIN}",
        "",
    ]

    if len(components) <= 1:
        lines.append(
            "WARNING: only one aerodynamic component/patch was created. Relative assembly motion is impossible unless the script can export or split separate parts."
        )

    root = select_assembly_root_component(components)
    if root is None:
        lines.append("No root component could be selected; no assembly motion policy applied.")
        (root_case / MOTION_POLICY_REPORT_NAME).write_text("\n".join(lines) + "\n")
        return lines

    if not any(component_has_motion_freedom(component) for component in components):
        lines.append(
            "No relative motion freedoms were decoded. Treating the imported assembly as one rigid free body so net fin/body loads can still spin it."
        )
        rigid_followers = 0
        root.freedom = six_dof_motion_freedom("assembly-rigid-body-root")
        root.is_assembly_anchor = False
        root.motion_origin = root.cofr
        for component in components:
            component.linear_velocity = (0.0, 0.0, 0.0)
            component.angular_velocity = (0.0, 0.0, 0.0)
            if component is root:
                continue
            component.is_assembly_anchor = False
            component.freedom = MotionFreedom([], [], "FASTENED", "assembly-rigid-body-follower")
            rigid_followers += 1
        lines.append(
            f"Rigid-body root component: patch={root.patch!r}, name={root.name!r}, follower_count={rigid_followers}"
        )
        lines.append("")
        lines.append("Movable components after policy:")
        for c in components:
            axes = f"translate_axes={c.freedom.translate_axes}, rotate_axes={c.freedom.rotate_axes}"
            basis = motion_basis_debug(c)
            if c is root:
                lines.append(f"- {c.patch}: rigid-body root, mate_type={c.freedom.mate_type}, source={c.freedom.source}, {axes}, {basis}")
            else:
                lines.append(f"- {c.patch}: rigid-body follower, mate_type={c.freedom.mate_type}, source={c.freedom.source}, {axes}, {basis}")
        (root_case / MOTION_POLICY_REPORT_NAME).write_text("\n".join(lines) + "\n")
        return lines

    if not ANCHOR_ASSEMBLY_ROOT:
        lines.append("Root anchoring disabled. The whole assembly may accelerate/decelerate as one free body.")
        (root_case / MOTION_POLICY_REPORT_NAME).write_text("\n".join(lines) + "\n")
        return lines

    root.is_assembly_anchor = True
    root.linear_velocity = (0.0, 0.0, 0.0)
    root.angular_velocity = (0.0, 0.0, 0.0)
    root.freedom = MotionFreedom([], [], "ANCHOR", "assembly-root-anchor")

    # Freeze inferred hinge origins before the first step.  If we re-infer from
    # the moved STL every step, the pivot moves with the part and the relative
    # motion can appear locked to the assembly.
    for c in components:
        if c is not root and c.freedom.rotate_axes and c.motion_origin is None:
            c.motion_origin = infer_motion_origin(c)

    if FORCE_NON_ROOT_COMPONENTS_FREE:
        free6 = MotionFreedom(
            translate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            rotate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            mate_type="FORCED_FREE",
            source="debug-force-non-root-free",
        )
        for c in components:
            if c is not root:
                c.freedom = free6
                if c.motion_origin is None:
                    c.motion_origin = infer_motion_origin(c)

    lines.append(
        f"Anchored root component: patch={root.patch!r}, name={root.name!r}, mass={root.mass:.6g} kg, "
        f"density={root.material.density_kg_m3:.6g} kg/m^3"
    )
    lines.append("")
    lines.append("Movable components after policy:")
    for c in components:
        axes = f"translate_axes={c.freedom.translate_axes}, rotate_axes={c.freedom.rotate_axes}"
        basis = motion_basis_debug(c)
        if c is root:
            lines.append(f"- {c.patch}: ANCHORED, no motion, {axes}, {basis}")
        elif c.freedom.translate_axes or c.freedom.rotate_axes:
            lines.append(f"- {c.patch}: movable, mate_type={c.freedom.mate_type}, source={c.freedom.source}, {axes}, {basis}")
        else:
            lines.append(f"- {c.patch}: fixed/no decoded motion, mate_type={c.freedom.mate_type}, source={c.freedom.source}, {axes}, {basis}")

    (root_case / MOTION_POLICY_REPORT_NAME).write_text("\n".join(lines) + "\n")
    return lines


DEFORMATION_YOUNG_MODULUS_BY_MATERIAL_PA: Dict[str, float] = {
    "eps": 1.0e7,
    "xps": 2.5e7,
    "epp": 2.0e7,
    "depron": 2.5e7,
    "foamboard": 8.0e7,
    "foam board": 8.0e7,
    "foam": 3.0e7,
    "balsa": 3.0e9,
    "plywood": 8.0e9,
    "basswood": 8.0e9,
    "pla": 3.5e9,
    "petg": 2.1e9,
    "abs": 2.0e9,
    "asa": 2.0e9,
    "nylon": 1.5e9,
    "carbon": 7.0e10,
    "fiberglass": 2.0e10,
    "glass fiber": 2.0e10,
    "glass fibre": 2.0e10,
    "aluminum": 6.9e10,
    "aluminium": 6.9e10,
    "steel": 2.0e11,
}


def component_deformation_enabled(component: AeroComponent) -> bool:
    if not ENABLE_NONRIGID_DEFORMATION or not component.triangles:
        return False
    if component.is_assembly_anchor and not DEFORM_ANCHORED_COMPONENTS:
        return False
    component_text = f"{component.name} {component.patch}".lower()
    material_text = component.material.material_name.lower()
    if DEFORMATION_EXCLUDE_COMPONENT_NAME_CONTAINS and DEFORMATION_EXCLUDE_COMPONENT_NAME_CONTAINS in component_text:
        return False
    if DEFORMATION_EXCLUDE_MATERIAL_NAME_CONTAINS and DEFORMATION_EXCLUDE_MATERIAL_NAME_CONTAINS in material_text:
        return False
    if DEFORMATION_COMPONENT_NAME_CONTAINS:
        if DEFORMATION_COMPONENT_NAME_CONTAINS not in component_text:
            return False
    if DEFORMATION_MATERIAL_NAME_CONTAINS:
        if DEFORMATION_MATERIAL_NAME_CONTAINS not in material_text:
            return False
    return True


def inferred_deformation_young_modulus(component: AeroComponent) -> float:
    if DEFORMATION_YOUNG_MODULUS_PA > 0.0:
        return DEFORMATION_YOUNG_MODULUS_PA
    if component.material.young_modulus_pa is not None and component.material.young_modulus_pa > 0.0:
        return component.material.young_modulus_pa
    name = component.material.material_name.lower()
    for key, value in DEFORMATION_YOUNG_MODULUS_BY_MATERIAL_PA.items():
        if key in name:
            return value
    return 5.0e7


def inferred_deformation_poisson_ratio(component: AeroComponent) -> float:
    if component.material.poisson_ratio is not None:
        return max(0.0, min(0.49, component.material.poisson_ratio))
    return DEFORMATION_POISSON_RATIO


def inferred_deformation_thickness(component: AeroComponent) -> float:
    if DEFORMATION_THICKNESS_M > 0.0:
        return DEFORMATION_THICKNESS_M
    if component.material.thickness_m is not None and component.material.thickness_m > 0.0:
        return component.material.thickness_m
    return max(0.001, min(0.05, 0.04 * max(component.lref, 1e-6)))


def deformation_vertex_key(point: Vec3) -> Tuple[int, int, int]:
    tol = max(DEFORMATION_VERTEX_TOLERANCE_M, 1e-12)
    return (
        int(round(point[0] / tol)),
        int(round(point[1] / tol)),
        int(round(point[2] / tol)),
    )


def deformation_support_origin(component: AeroComponent) -> Vec3:
    if component.mate_origin is not None:
        return component.mate_origin
    if component.motion_origin is not None:
        return component.motion_origin
    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
    support_x = xmin if flow_is_positive_x() else xmax
    return (support_x, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))


def deformation_span_axis(component: AeroComponent, origin: Vec3) -> Vec3:
    from_origin = v_sub(component.cofr, origin)
    if v_norm(from_origin) > 1e-9:
        return v_unit(from_origin)

    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
    extents = [
        (xmax - xmin, (1.0, 0.0, 0.0)),
        (ymax - ymin, (0.0, 1.0, 0.0)),
        (zmax - zmin, (0.0, 0.0, 1.0)),
    ]
    _length, axis = max(extents, key=lambda item: item[0])
    return axis


def triangle_pressure_force_for_deformation(component: AeroComponent, triangle: Triangle) -> Vec3:
    area, centroid, normal = triangle_area_centroid_normal(triangle)
    if area <= 1e-18:
        return (0.0, 0.0, 0.0)
    rel_speed, flow_unit = local_air_speed_and_unit(component, centroid)
    if rel_speed <= 1e-9:
        return (0.0, 0.0, 0.0)
    q = 0.5 * RHO * rel_speed ** 2
    incoming_dir = v_mul(flow_unit, -1.0)
    projection = v_dot(normal, incoming_dir)
    if SURFACE_LOAD_DOUBLE_SIDED:
        if projection >= 0.0:
            force_dir = v_mul(normal, -1.0)
            exposure = projection
        else:
            force_dir = normal
            exposure = -projection
    else:
        if projection <= 0.0:
            return (0.0, 0.0, 0.0)
        force_dir = v_mul(normal, -1.0)
        exposure = projection
    pressure = q * (exposure ** 2) * SURFACE_LOAD_COEFF * SURFACE_LOAD_GAIN
    return v_mul(force_dir, pressure * area)


def ensure_deformation_reference(component: AeroComponent) -> None:
    if component.deformation_reference_triangles is None:
        component.deformation_reference_triangles = list(component.triangles)


def update_component_deformation(component: AeroComponent, dt: float) -> Tuple[float, float, float, float, int]:
    if not component_deformation_enabled(component):
        component.deformation_max_m = 0.0
        component.deformation_mean_m = 0.0
        return 0.0, 0.0, 0.0, 0.0, 0

    ensure_deformation_reference(component)
    reference = component.deformation_reference_triangles or component.triangles
    young = max(inferred_deformation_young_modulus(component), 1.0)
    poisson = inferred_deformation_poisson_ratio(component)
    thickness = max(inferred_deformation_thickness(component), 1e-5)
    origin = deformation_support_origin(component)
    axis = deformation_span_axis(component, origin)

    vertex_points: Dict[Tuple[int, int, int], Vec3] = {}
    current_sums: Dict[Tuple[int, int, int], Vec3] = {}
    current_counts: Dict[Tuple[int, int, int], int] = {}
    force_sums: Dict[Tuple[int, int, int], Vec3] = {}
    area_sums: Dict[Tuple[int, int, int], float] = {}

    for ref_tri, cur_tri in zip(reference, component.triangles):
        area, _centroid, _normal = triangle_area_centroid_normal(ref_tri)
        tri_force = triangle_pressure_force_for_deformation(component, ref_tri)
        for ref_point, current_point in zip(ref_tri[1:], cur_tri[1:]):
            key = deformation_vertex_key(ref_point)
            vertex_points.setdefault(key, ref_point)
            current_sums[key] = v_add(current_sums.get(key, (0.0, 0.0, 0.0)), current_point)
            current_counts[key] = current_counts.get(key, 0) + 1
            force_sums[key] = v_add(force_sums.get(key, (0.0, 0.0, 0.0)), v_mul(tri_force, 1.0 / 3.0))
            area_sums[key] = area_sums.get(key, 0.0) + area / 3.0

    if not vertex_points:
        return 0.0, 0.0, young, thickness, 0

    levers = [abs(v_dot(v_sub(point, origin), axis)) for point in vertex_points.values()]
    span = max(max(levers), 1e-6)
    plate_factor = 12.0 * (1.0 - poisson ** 2) / 64.0
    stiffness_scale = plate_factor * (span ** 4) / (young * (thickness ** 3))
    max_total = min(MAX_TOTAL_DEFORMATION, 0.25 * max(component.lref, 1e-6))
    max_delta = max(MAX_DEFORMATION_PER_STEP, 0.0)
    relaxation = DEFORMATION_RELAXATION if dt > 0.0 else 1.0
    displacement_by_key: Dict[Tuple[int, int, int], Vec3] = {}

    for key, ref_point in vertex_points.items():
        count = max(current_counts.get(key, 1), 1)
        current_point = v_mul(current_sums.get(key, ref_point), 1.0 / count)
        current_disp = v_sub(current_point, ref_point)
        area = max(area_sums.get(key, 0.0), 1e-18)
        pressure_vector = v_mul(force_sums.get(key, (0.0, 0.0, 0.0)), 1.0 / area)
        pressure_mag = v_norm(pressure_vector)
        if pressure_mag <= 1e-12:
            target_disp = (0.0, 0.0, 0.0)
        else:
            lever = abs(v_dot(v_sub(ref_point, origin), axis))
            s = max(0.0, min(1.0, lever / span))
            shape = s * s * (3.0 - 2.0 * s)
            target_mag = pressure_mag * stiffness_scale * shape * DEFORMATION_GAIN
            target_disp = v_mul(v_unit(pressure_vector), target_mag)
            target_disp = clamp_vector_magnitude(target_disp, max_total)

        step_delta = v_mul(v_sub(target_disp, current_disp), relaxation)
        if max_delta > 0.0:
            step_delta = clamp_vector_magnitude(step_delta, max_delta)
        displacement_by_key[key] = clamp_vector_magnitude(v_add(current_disp, step_delta), max_total)

    deformed: List[Triangle] = []
    magnitudes: List[float] = []
    for _normal, v1, v2, v3 in reference:
        pts = []
        for point in (v1, v2, v3):
            disp = displacement_by_key.get(deformation_vertex_key(point), (0.0, 0.0, 0.0))
            magnitudes.append(v_norm(disp))
            pts.append(v_add(point, disp))
        normal = v_unit(v_cross(v_sub(pts[1], pts[0]), v_sub(pts[2], pts[0])), _normal)
        deformed.append((normal, pts[0], pts[1], pts[2]))

    component.triangles = deformed
    component.aref, component.lref, component.cofr = component_references(component.triangles)
    component.deformation_max_m = max(magnitudes) if magnitudes else 0.0
    component.deformation_mean_m = sum(magnitudes) / max(len(magnitudes), 1)
    return component.deformation_max_m, component.deformation_mean_m, young, thickness, len(vertex_points)


def write_deformation_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Quasi-static non-rigid surface deformation log\n"
        "# Deformation is a bounded panel-pressure plate/cantilever approximation, not a structural finite-element solve.\n"
        "step\tpatch\tmaterial\tstructural_source\tenabled\tyoung_modulus_pa\tpoisson_ratio\tthickness_m\tnode_count\tmax_deformation_m\tmean_deformation_m\n"
    )


def append_deformation_log(
    path: Path,
    step: int,
    component: AeroComponent,
    enabled: bool,
    young: float,
    thickness: float,
    node_count: int,
) -> None:
    with path.open("a") as f:
        f.write(
            f"{step}\t{component.patch}\t{component.material.material_name}\t{component.material.structural_source}\t{int(enabled)}\t"
            f"{young:.8g}\t{inferred_deformation_poisson_ratio(component):.8g}\t{thickness:.8g}\t{node_count}\t"
            f"{component.deformation_max_m:.8g}\t{component.deformation_mean_m:.8g}\n"
        )


def apply_nonrigid_deformations(components: Sequence[AeroComponent], step: int, log_path: Path) -> List[AeroComponent]:
    changed: List[AeroComponent] = []
    if not ENABLE_NONRIGID_DEFORMATION:
        return changed
    for component in components:
        enabled = component_deformation_enabled(component)
        max_def, _mean_def, young, thickness, node_count = update_component_deformation(component, MOTION_DT)
        append_deformation_log(log_path, step, component, enabled, young, thickness, node_count)
        if enabled and max_def > 1e-12:
            changed.append(component)
    return changed


def write_motion_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Quasi-dynamic assembly motion log\n"
        f"# dt={MOTION_DT:g} s, default_mass={DEFAULT_PART_MASS_KG:g} kg, default_inertia={DEFAULT_PART_INERTIA_KGM2:g} kg m^2\n"
        "# Rigid-body motion is driven by OpenFOAM forces.dat/log loads, with panel fallback only if OpenFOAM loads are unavailable.\n"
        "# Optional non-rigid deformation is logged separately in assembly_deformation_log.txt.\n"
        "step\tpatch\tmaterial\tmaterial_source\tmass_kg\tdensity_kg_m3\tmate_type\tfreedom_source\tCd\tCs\tCl\tCmRoll\tCmPitch\tCmYaw\tFx_N\tFy_N\tFz_N\tMx_Nm\tMy_Nm\tMz_Nm\tdx_m\tdy_m\tdz_m\tdroll_rad\tdpitch_rad\tdyaw_rad\ttotal_x_m\ttotal_y_m\ttotal_z_m\n"
    )


def append_motion_log(path: Path, step: int, component: AeroComponent, coeffs: Dict[str, float], force: Vec3, moment: Vec3, dpos: Vec3, drot: Vec3) -> None:
    with path.open("a") as f:
        f.write("\t".join([
            str(step), component.patch, component.material.material_name, component.material.source,
            f"{component.mass:.8g}", f"{component.material.density_kg_m3:.8g}",
            component.freedom.mate_type, component.freedom.source,
            f"{coeffs.get('Cd', 0.0):.8g}", f"{coeffs.get('Cs', 0.0):.8g}", f"{coeffs.get('Cl', 0.0):.8g}",
            f"{coeffs.get('CmRoll', 0.0):.8g}", f"{coeffs.get('CmPitch', 0.0):.8g}", f"{coeffs.get('CmYaw', 0.0):.8g}",
            f"{force[0]:.8g}", f"{force[1]:.8g}", f"{force[2]:.8g}",
            f"{moment[0]:.8g}", f"{moment[1]:.8g}", f"{moment[2]:.8g}",
            f"{dpos[0]:.8g}", f"{dpos[1]:.8g}", f"{dpos[2]:.8g}",
            f"{drot[0]:.8g}", f"{drot[1]:.8g}", f"{drot[2]:.8g}",
            f"{component.total_translation[0]:.8g}", f"{component.total_translation[1]:.8g}", f"{component.total_translation[2]:.8g}",
        ]) + "\n")
