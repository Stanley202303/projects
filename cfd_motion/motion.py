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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import *
from .models import *
from .math_utils import *
from .geometry import *
from .openfoam import *
from .onshape import *
from .structural import (
    ExplicitShellState,
    HybridFEMMPMCollisionState,
    HybridShellCollisionState,
    advance_hybrid_fem_mpm_collision,
    advance_hybrid_shell_collision,
    advance_explicit_shell,
    apply_shell_contact_work,
    apply_shell_impact_energy,
    build_explicit_shell_state,
    build_hybrid_fem_mpm_collision_state,
    build_hybrid_shell_collision_state,
    commit_explicit_shell_topology,
    commit_hybrid_fem_mpm_failure_topology,
    component_is_thin_for_solid_fem,
    emit_shell_fragments,
    hybrid_fragment_components,
    apply_fem_impact_energy,
    update_fem_perforation,
    sync_hybrid_shell_fragments,
    update_shell_perforation,
)

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


def assembly_rigid_body_groups(
    components: Sequence[AeroComponent],
) -> List[Tuple[List[AeroComponent], AeroComponent]]:
    """Return independently moving rigid groups created from connected mates."""
    by_group: Dict[str, List[AeroComponent]] = {}
    for component in components:
        if component.rigid_body_group:
            by_group.setdefault(component.rigid_body_group, []).append(component)
    groups: List[Tuple[List[AeroComponent], AeroComponent]] = []
    for group_components in by_group.values():
        root = next(
            (
                component
                for component in group_components
                if component.freedom.source == "assembly-rigid-body-root"
            ),
            None,
        )
        if root is not None:
            groups.append((group_components, root))
    return groups


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


def collision_angular_speed_limit() -> float:
    """Return an explicit cap or the rotation-CFL cap for this timestep."""
    if COLLISION_MAX_ANGULAR_SPEED_RAD_S > 0.0:
        return COLLISION_MAX_ANGULAR_SPEED_RAD_S
    return MAX_ROTATION_PER_STEP_RAD / max(MOTION_DT, 1e-9)


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
    structural_state = component.collision_structural_state
    if isinstance(structural_state, HybridFEMMPMCollisionState):
        solid = structural_state.solid_state

        def transform_point(point: Vec3) -> Vec3:
            rotated = (
                rotate_point_around_axis(
                    point,
                    origin,
                    rotation_axis,
                    rotation_angle,
                )
                if rotation_axis is not None
                else point
            )
            return v_add(rotated, translation)

        def rotate_vector(vector: Vec3) -> Vec3:
            if rotation_axis is None:
                return vector
            return rotate_point_around_axis(
                vector,
                (0.0, 0.0, 0.0),
                rotation_axis,
                rotation_angle,
            )

        solid.positions = [transform_point(point) for point in solid.positions]
        solid.reference_positions = [
            transform_point(point) for point in solid.reference_positions
        ]
        solid.velocities = [rotate_vector(velocity) for velocity in solid.velocities]
    fragment_parent = component.collision_fragment_parent_state
    fragment_element = component.collision_fragment_source_element
    if (
        isinstance(fragment_parent, HybridFEMMPMCollisionState)
        and fragment_element is not None
    ):
        particles = [
            particle
            for particle in fragment_parent.solid_state.particles
            if particle.source_element == fragment_element
        ]
        for particle in particles:
            if rotation_axis is not None:
                particle.position = rotate_point_around_axis(
                    particle.position,
                    origin,
                    rotation_axis,
                    rotation_angle,
                )
                particle.velocity = rotate_point_around_axis(
                    particle.velocity,
                    (0.0, 0.0, 0.0),
                    rotation_axis,
                    rotation_angle,
                )
            particle.position = v_add(particle.position, translation)
    if component.deformation_reference_triangles is not None:
        component.deformation_reference_triangles = move_triangles(
            component.deformation_reference_triangles,
            translation,
            rotation_axis,
            rotation_angle,
            origin,
        )
    for damage in component.collision_damage:
        if rotation_axis is not None:
            damage.contact_point = rotate_point_around_axis(
                damage.contact_point,
                origin,
                rotation_axis,
                rotation_angle,
            )
            damage.inward_direction = v_unit(
                rotate_point_around_axis(
                    damage.inward_direction,
                    (0.0, 0.0, 0.0),
                    rotation_axis,
                    rotation_angle,
                )
            )
        damage.contact_point = v_add(damage.contact_point, translation)
    if rotation_axis is not None:
        component.cofr = rotate_point_around_axis(component.cofr, origin, rotation_axis, rotation_angle)
    component.cofr = v_add(component.cofr, translation)

    if component.mate_origin is not None:
        if rotation_axis is not None:
            component.mate_origin = rotate_point_around_axis(component.mate_origin, origin, rotation_axis, rotation_angle)
        component.mate_origin = v_add(component.mate_origin, translation)
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


def update_component_motion(
    component: AeroComponent,
    coeffs: Dict[str, float],
    dt: float,
    load_override: Optional[Tuple[Vec3, Vec3]] = None,
    hold_kinematics: bool = False,
    externally_applied_velocity: Optional[Vec3] = None,
) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    if not component.triangles and component.mass <= 1e-18:
        # A fully fragmented parent is only a structural-state container. Its
        # detached material is advanced through the fragment/MPM paths below;
        # moving the empty parent would transform that material a second time.
        component.linear_velocity = (0.0, 0.0, 0.0)
        component.angular_velocity = (0.0, 0.0, 0.0)
        return (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )

    force, foam_moment = resolve_aerodynamic_load(component, coeffs, load_override)
    free = component.freedom

    # Anchored/root/grounded/fastened components still get force and moment logged,
    # but they do not move. This makes the solver report relative part movement
    # instead of letting the whole assembly drift through the wind tunnel.
    if component.is_assembly_anchor or (not free.translate_axes and not free.rotate_axes):
        component.linear_velocity = (0.0, 0.0, 0.0)
        component.angular_velocity = (0.0, 0.0, 0.0)
        return force, foam_moment, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    if hold_kinematics:
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
        return force, total_moment, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    allowed_force = project_vector_on_axes(force, free.translate_axes)
    acceleration = v_mul(allowed_force, 1.0 / max(component.mass, 1e-9))
    new_lv = v_add(component.linear_velocity, v_mul(acceleration, dt))
    # Material-dependent damping for the simplified rigid-body update.
    # Foam/light materials are damped more strongly; dense metals retain velocity longer.
    linear_decay = math.exp(-max(component.material.linear_damping_per_kg, 0.0) * dt / max(component.mass, 1e-9))
    new_lv = v_mul(new_lv, linear_decay)
    # Collision-convergence applies the nominal launch translation separately
    # after every body has been integrated.  Subtract only that common velocity
    # here so CFD-induced relative translation and rotation remain independent
    # without applying the forward speed twice.
    integration_velocity = new_lv
    if externally_applied_velocity is not None:
        integration_velocity = v_sub(new_lv, externally_applied_velocity)
    unconstrained_translation = v_mul(integration_velocity, dt)
    fully_free_translation = len(free.translate_axes) >= 3
    if (
        component.freedom.source == "post-perforation-ballistic"
        or fully_free_translation
    ):
        translation_step = unconstrained_translation
    else:
        translation_step = clamp_vector_magnitude(unconstrained_translation, MAX_TRANSLATION_PER_STEP)

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
    new_av = clamp_vector_magnitude(
        new_av,
        collision_angular_speed_limit(),
    )
    unconstrained_rotation = v_mul(new_av, dt)
    rotation_vector_step = clamp_vector_magnitude(
        unconstrained_rotation,
        MAX_ROTATION_PER_STEP_RAD,
    )

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


def apply_aerodynamic_velocity_increment(
    component: AeroComponent,
    dt: float,
    load_override: Optional[Tuple[Vec3, Vec3]] = None,
) -> Tuple[Vec3, Vec3]:
    """Apply an aerodynamic impulse without translating the geometry.

    Hybrid-shell fragments are translated by the structural update.  Applying
    their air load as a velocity increment avoids moving them twice in the
    same time interval while ensuring their next structural advance responds
    to drag, lift, and aerodynamic torque from their current relative-air
    velocity.
    """
    if dt <= 0.0 or component.is_assembly_anchor:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    previous_linear_velocity = component.linear_velocity
    previous_angular_velocity = component.angular_velocity
    force, source_moment = load_override or surface_pressure_load(component)
    allowed_force = project_vector_on_axes(force, component.freedom.translate_axes)
    acceleration = v_mul(allowed_force, 1.0 / max(component.mass, 1e-9))
    component.linear_velocity = v_add(
        component.linear_velocity,
        v_mul(acceleration, dt),
    )
    linear_decay = math.exp(
        -max(component.material.linear_damping_per_kg, 0.0)
        * dt
        / max(component.mass, 1e-9)
    )
    component.linear_velocity = v_mul(component.linear_velocity, linear_decay)

    total_moment = total_aerodynamic_moment_about_origin(
        component,
        force,
        source_moment,
        infer_motion_origin(component),
        True,
    )
    allowed_moment = project_vector_on_axes(
        total_moment,
        component.freedom.rotate_axes,
    )
    angular_acceleration = angular_acceleration_from_moment(
        component,
        allowed_moment,
        component.freedom.rotate_axes,
    )
    component.angular_velocity = v_add(
        component.angular_velocity,
        v_mul(angular_acceleration, dt),
    )
    fragment_parent = component.collision_fragment_parent_state
    fragment_element = component.collision_fragment_source_element
    if (
        isinstance(fragment_parent, HybridFEMMPMCollisionState)
        and fragment_element is not None
    ):
        delta_velocity = v_sub(
            component.linear_velocity,
            previous_linear_velocity,
        )
        delta_angular_velocity = v_sub(
            component.angular_velocity,
            previous_angular_velocity,
        )
        for particle in fragment_parent.solid_state.particles:
            if particle.source_element != fragment_element:
                continue
            particle.velocity = v_add(
                particle.velocity,
                v_add(
                    delta_velocity,
                    v_cross(
                        delta_angular_velocity,
                        v_sub(particle.position, component.cofr),
                    ),
                ),
            )
    return force, total_moment


def advance_detached_fragment_aerodynamics(
    components: Sequence[AeroComponent],
    dt: float,
) -> int:
    """Apply relative-air loads to all live hybrid-shell fragment bodies."""
    advanced = 0
    for parent in components:
        state = parent.collision_structural_state
        if not isinstance(
            state,
            (HybridShellCollisionState, HybridFEMMPMCollisionState),
        ):
            continue
        for fragment in state.fragment_bodies:
            apply_aerodynamic_velocity_increment(fragment.component, dt)
            advanced += 1
    return advanced


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
    component.angular_velocity = clamp_vector_magnitude(
        v_add(
            component.angular_velocity,
            v_mul(drot, 1.0 / max(MOTION_DT, 1e-9)),
        ),
        collision_angular_speed_limit(),
    )
    return drot


def apply_collision_impulse(component: AeroComponent, impulse: Vec3, contact_point: Vec3) -> None:
    if component.is_assembly_anchor:
        return
    if component_has_translation_freedom(component):
        dv = v_mul(project_vector_on_axes(impulse, component.freedom.translate_axes), 1.0 / max(component.mass, 1e-9))
        component.linear_velocity = v_add(component.linear_velocity, dv)
    if component_has_rotation_freedom(component):
        origin = infer_motion_origin(component)
        angular_impulse = v_cross(v_sub(contact_point, origin), impulse)
        angular_impulse = project_vector_on_axes(angular_impulse, component.freedom.rotate_axes)
        delta_av = angular_acceleration_from_moment(component, angular_impulse, component.freedom.rotate_axes)
        component.angular_velocity = clamp_vector_magnitude(
            v_add(component.angular_velocity, delta_av),
            collision_angular_speed_limit(),
        )
    fragment_parent = component.collision_fragment_parent_state
    fragment_element = component.collision_fragment_source_element
    if (
        isinstance(fragment_parent, HybridFEMMPMCollisionState)
        and fragment_element is not None
    ):
        particles = [
            particle
            for particle in fragment_parent.solid_state.particles
            if particle.source_element == fragment_element
        ]
        particle_mass = sum(particle.mass_kg for particle in particles)
        if particle_mass > 1e-18:
            weighted_centre = (0.0, 0.0, 0.0)
            for particle in particles:
                weighted_centre = v_add(
                    weighted_centre,
                    v_mul(particle.position, particle.mass_kg),
                )
            centre = v_mul(
                weighted_centre,
                1.0 / particle_mass,
            )
            delta_velocity = v_mul(impulse, 1.0 / particle_mass)
            torque = v_cross(v_sub(contact_point, centre), impulse)
            scalar_inertia = sum(
                particle.mass_kg
                * v_dot(
                    v_sub(particle.position, centre),
                    v_sub(particle.position, centre),
                )
                for particle in particles
            )
            delta_omega = v_mul(torque, 1.0 / max(scalar_inertia, 1e-18))
            for particle in particles:
                particle.velocity = v_add(
                    particle.velocity,
                    v_add(
                        delta_velocity,
                        v_cross(
                            delta_omega,
                            v_sub(particle.position, centre),
                        ),
                    ),
                )


def contact_point_velocity(component: AeroComponent, contact_point: Vec3) -> Vec3:
    if component.is_assembly_anchor:
        return (0.0, 0.0, 0.0)
    origin = infer_motion_origin(component)
    return v_add(
        component.linear_velocity,
        v_cross(component.angular_velocity, v_sub(contact_point, origin)),
    )


def rotational_contact_inverse_mass(
    component: AeroComponent,
    contact_point: Vec3,
    impulse_direction: Vec3,
) -> float:
    if not component_has_rotation_freedom(component):
        return 0.0
    origin = infer_motion_origin(component)
    radius = v_sub(contact_point, origin)
    unit_impulse = v_unit(impulse_direction)
    angular_impulse = project_vector_on_axes(
        v_cross(radius, unit_impulse),
        component.freedom.rotate_axes,
    )
    delta_angular_velocity = angular_acceleration_from_moment(
        component,
        angular_impulse,
        component.freedom.rotate_axes,
    )
    point_velocity = v_cross(delta_angular_velocity, radius)
    return max(0.0, v_dot(point_velocity, unit_impulse))


def contact_inverse_mass(
    component: AeroComponent,
    contact_point: Vec3,
    impulse_direction: Vec3,
) -> float:
    linear = (
        1.0 / max(component.mass, 1e-9)
        if component_has_translation_freedom(component)
        else 0.0
    )
    return linear + rotational_contact_inverse_mass(
        component,
        contact_point,
        impulse_direction,
    )


def material_friction_coefficient(component: AeroComponent) -> float:
    name = component.material.material_name.lower()
    values = {
        "ptfe": 0.08,
        "nylon": 0.20,
        "abs": 0.30,
        "pla": 0.35,
        "petg": 0.30,
        "aluminum": 0.35,
        "aluminium": 0.35,
        "steel": 0.50,
        "tungsten": 0.40,
        "wolfram": 0.40,
        "rubber": 0.80,
    }
    return next(
        (value for key, value in values.items() if key in name),
        max(0.0, COLLISION_TANGENTIAL_DAMPING),
    )


def contact_friction_coefficient(a: AeroComponent, b: AeroComponent) -> float:
    if COLLISION_FRICTION_COEFFICIENT >= 0.0:
        return COLLISION_FRICTION_COEFFICIENT
    return math.sqrt(
        material_friction_coefficient(a) * material_friction_coefficient(b)
    )


def material_restitution_coefficient(component: AeroComponent) -> float:
    """Return a conservative dry-impact restitution for a BOM material."""
    name = component.material.material_name.lower()
    values = {
        "rubber": 0.72,
        "silicone": 0.65,
        "ptfe": 0.35,
        "nylon": 0.30,
        "petg": 0.28,
        "abs": 0.24,
        "pla": 0.20,
        "wood": 0.25,
        "aluminum": 0.20,
        "aluminium": 0.20,
        "titanium": 0.22,
        "steel": 0.24,
        "iron": 0.22,
        "tungsten": 0.18,
        "wolfram": 0.18,
        "glass": 0.16,
        "ceramic": 0.14,
    }
    return next((value for key, value in values.items() if key in name), 0.20)


def contact_restitution_coefficient(
    a: AeroComponent,
    b: AeroComponent,
    configured_restitution: float,
) -> float:
    """Resolve an explicit override or a bounded BOM-derived material pair."""
    if configured_restitution >= 0.0:
        return min(configured_restitution, 1.0)
    return math.sqrt(
        material_restitution_coefficient(a)
        * material_restitution_coefficient(b)
    )


def component_contact_compliance(component: AeroComponent) -> float:
    young = max(inferred_deformation_young_modulus(component), 1.0)
    poisson = inferred_deformation_poisson_ratio(component)
    thickness = max(inferred_deformation_thickness(component), 1e-6)
    # Thin shells dent more readily than solid bodies. Use lref/thickness as a
    # conservative shell-compliance multiplier, bounded so bad thickness metadata
    # cannot explode the contact response.
    shell_factor = max(1.0, min(25.0, max(component.lref, 1e-6) / (20.0 * thickness)))
    return ((1.0 - poisson * poisson) / young) * shell_factor


@dataclass(frozen=True)
class LocalContactGeometry:
    radius_a: float
    radius_b: float
    effective_radius: float
    footprint_radius: float
    hertzian: bool


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def contact_tangent_basis(normal: Vec3) -> Tuple[Vec3, Vec3]:
    n = v_unit(normal)
    reference = min(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        key=lambda axis: abs(v_dot(axis, n)),
    )
    tangent_x = v_unit(v_cross(n, reference))
    return tangent_x, v_unit(v_cross(n, tangent_x))


def convex_hull_area_2d(points: Sequence[Tuple[float, float]]) -> float:
    """Return planar convex-hull area using Andrew's monotone-chain algorithm."""
    unique = sorted(set(points))
    if len(unique) < 3:
        return 0.0

    def cross(
        origin: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        return (
            (a[0] - origin[0]) * (b[1] - origin[1])
            - (a[1] - origin[1]) * (b[0] - origin[0])
        )

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return 0.5 * abs(
        sum(
            hull[index][0] * hull[(index + 1) % len(hull)][1]
            - hull[(index + 1) % len(hull)][0] * hull[index][1]
            for index in range(len(hull))
        )
    )


def local_surface_curvature_radius(
    component: AeroComponent,
    contact_point: Vec3,
    outward_normal: Vec3,
) -> float:
    """Estimate local radius from mesh sagitta; return infinity for a flat patch."""
    normal = v_unit(outward_normal)
    scale = max(component.lref, 1e-9)
    neighborhood = max(4.0 * COLLISION_DEFORMATION_MIN_RADIUS_M, 0.75 * scale)
    radial_epsilon = max(1e-9, 1e-5 * scale)
    estimates: List[float] = []

    for triangle in component.triangles:
        _area, centroid, triangle_normal = triangle_area_centroid_normal(triangle)
        if abs(v_dot(triangle_normal, normal)) < 0.25:
            continue
        if v_norm(v_sub(centroid, contact_point)) > neighborhood:
            continue
        for point in triangle[1:]:
            delta = v_sub(point, contact_point)
            axial = v_dot(delta, normal)
            tangent = v_sub(delta, v_mul(normal, axial))
            radial = v_norm(tangent)
            sagitta = abs(axial)
            if radial <= radial_epsilon:
                continue
            if sagitta <= max(1e-10, 1e-4 * radial):
                continue
            radius = (radial * radial + sagitta * sagitta) / (2.0 * sagitta)
            if radius >= 0.5 * radial:
                estimates.append(radius)

    if not estimates:
        return math.inf
    radius = _median(estimates)
    if radius > 50.0 * scale:
        return math.inf
    return max(radius, 1e-4 * scale, 1e-9)


def local_planar_patch_radius(
    component: AeroComponent,
    contact_point: Vec3,
    outward_normal: Vec3,
) -> float:
    """Equivalent circular radius of the locally coplanar contact face."""
    normal = v_unit(outward_normal)
    tangent_x, tangent_y = contact_tangent_basis(normal)
    scale = max(component.lref, 1e-9)
    plane_tolerance = max(COLLISION_MANIFOLD_TOLERANCE_M, 1e-4 * scale)
    projected: List[Tuple[float, float]] = []

    for triangle in component.triangles:
        _area, centroid, triangle_normal = triangle_area_centroid_normal(triangle)
        if abs(v_dot(triangle_normal, normal)) < 0.95:
            continue
        if v_norm(v_sub(centroid, contact_point)) > 0.75 * scale:
            continue
        for point in triangle[1:]:
            delta = v_sub(point, contact_point)
            if abs(v_dot(delta, normal)) > plane_tolerance:
                continue
            projected.append((v_dot(delta, tangent_x), v_dot(delta, tangent_y)))

    area = convex_hull_area_2d(projected)
    if area > 1e-18:
        return math.sqrt(area / math.pi)
    return max(COLLISION_DEFORMATION_MIN_RADIUS_M, 0.05 * scale)


def local_contact_geometry(
    a: AeroComponent,
    b: AeroComponent,
    contact_point_a: Vec3,
    contact_point_b: Vec3,
    normal_from_b_to_a: Vec3,
) -> LocalContactGeometry:
    normal = v_unit(normal_from_b_to_a)
    radius_a = local_surface_curvature_radius(
        a,
        contact_point_a,
        v_mul(normal, -1.0),
    )
    radius_b = local_surface_curvature_radius(b, contact_point_b, normal)
    patch_a = local_planar_patch_radius(a, contact_point_a, v_mul(normal, -1.0))
    patch_b = local_planar_patch_radius(b, contact_point_b, normal)

    if math.isinf(radius_a) and math.isinf(radius_b):
        footprint = max(min(patch_a, patch_b), 1e-9)
        return LocalContactGeometry(
            radius_a=radius_a,
            radius_b=radius_b,
            effective_radius=footprint,
            footprint_radius=footprint,
            hertzian=False,
        )
    if math.isinf(radius_a):
        effective_radius = radius_b
    elif math.isinf(radius_b):
        effective_radius = radius_a
    else:
        effective_radius = (radius_a * radius_b) / max(radius_a + radius_b, 1e-12)
    return LocalContactGeometry(
        radius_a=radius_a,
        radius_b=radius_b,
        effective_radius=max(effective_radius, 1e-9),
        footprint_radius=max(min(patch_a, patch_b), 1e-9),
        hertzian=True,
    )


def effective_contact_radius(
    a: AeroComponent,
    b: AeroComponent,
    contact_point_a: Optional[Vec3] = None,
    contact_point_b: Optional[Vec3] = None,
    normal_from_b_to_a: Optional[Vec3] = None,
) -> float:
    if contact_point_a is None or contact_point_b is None or normal_from_b_to_a is None:
        ra = max(0.5 * a.lref, 1e-6)
        rb = max(0.5 * b.lref, 1e-6)
        return max((ra * rb) / (ra + rb), 1e-6)
    return local_contact_geometry(
        a,
        b,
        contact_point_a,
        contact_point_b,
        normal_from_b_to_a,
    ).effective_radius


def hertz_contact_area_radius(
    a: AeroComponent,
    b: AeroComponent,
    depth: float,
    contact_point_a: Optional[Vec3] = None,
    contact_point_b: Optional[Vec3] = None,
    normal_from_b_to_a: Optional[Vec3] = None,
    geometry: Optional[LocalContactGeometry] = None,
) -> float:
    if COLLISION_DEFORMATION_MODEL not in {"hertz", "hertzian"}:
        return max(
            COLLISION_DEFORMATION_MIN_RADIUS_M,
            COLLISION_DEFORMATION_RADIUS_FACTOR * max(min(a.lref, b.lref), 1e-6),
            2.0 * depth,
        )
    if geometry is None and (
        contact_point_a is not None
        and contact_point_b is not None
        and normal_from_b_to_a is not None
    ):
        geometry = local_contact_geometry(
            a,
            b,
            contact_point_a,
            contact_point_b,
            normal_from_b_to_a,
        )
    if geometry is not None and not geometry.hertzian:
        radius = geometry.footprint_radius
    else:
        effective_radius = (
            geometry.effective_radius
            if geometry is not None
            else effective_contact_radius(a, b)
        )
        radius = math.sqrt(
            max(
                effective_radius * max(depth, COLLISION_MIN_OVERLAP_M),
                0.0,
            )
        )
    return max(
        COLLISION_DEFORMATION_MIN_RADIUS_M,
        COLLISION_DEFORMATION_RADIUS_FACTOR * radius,
        2.0 * depth,
    )


def component_is_collision_impactor(component: AeroComponent) -> bool:
    return component.freedom.mate_type.upper() == "COLLISION_IMPACTOR"


def collision_deformation_enabled(component: AeroComponent) -> bool:
    if not ENABLE_NONRIGID_DEFORMATION or not component.triangles:
        return False
    if component_is_collision_impactor(component):
        return COLLISION_DEFORM_IMPACTOR
    return component_deformation_enabled(component)


def split_contact_indentation(a: AeroComponent, b: AeroComponent, depth: float) -> Tuple[float, float]:
    ca = component_contact_compliance(a) if collision_deformation_enabled(a) else 0.0
    cb = component_contact_compliance(b) if collision_deformation_enabled(b) else 0.0
    total = ca + cb
    if total <= 1e-30:
        return 0.0, 0.0
    scaled_depth = max(depth, COLLISION_MIN_OVERLAP_M) * max(COLLISION_DEFORMATION_GAIN, 0.0)
    return scaled_depth * ca / total, scaled_depth * cb / total


def deform_triangle_mesh_at_contact(
    triangles: Sequence[Triangle],
    contact_point: Vec3,
    inward_direction: Vec3,
    indentation: float,
    contact_radius: float,
    eulerian_grid: Optional[EulerianContactGrid] = None,
    perforation_radius: float = 0.0,
    perforation_displacement: float = 0.0,
    preserve_thickness: bool = False,
) -> Tuple[List[Triangle], float]:
    direction = v_unit(inward_direction)
    max_applied = 0.0
    displaced_vertices: Dict[Tuple[int, int, int], Vec3] = {}
    for point in stl_points(triangles):
        key = deformation_vertex_key(point)
        if key in displaced_vertices:
            continue
        point_delta = v_sub(point, contact_point)
        if preserve_thickness:
            axial_distance = v_dot(point_delta, direction)
            tangent_delta = v_sub(
                point_delta,
                v_mul(direction, axial_distance),
            )
            distance = v_norm(tangent_delta)
        else:
            distance = v_norm(point_delta)
        if distance >= contact_radius:
            weight = 0.0
        else:
            radial_fraction = distance / max(contact_radius, 1e-12)
            if COLLISION_DEFORMATION_MODEL in {"hertz", "hertzian"}:
                weight = math.sqrt(max(0.0, 1.0 - radial_fraction * radial_fraction))
            else:
                smooth_fraction = 1.0 - radial_fraction
                weight = smooth_fraction * smooth_fraction * (3.0 - 2.0 * smooth_fraction)
            if eulerian_grid is not None:
                grid_pressure = eulerian_grid.pressure_at(point)
                weight = max(
                    weight,
                    min(
                        1.0,
                        grid_pressure
                        / max(COLLISION_EULERIAN_PENALTY_GAIN * indentation, 1e-12),
                    ),
                )
        displacement_magnitude = indentation * weight
        if perforation_radius > 0.0 and perforation_displacement > 0.0:
            axial_distance = v_dot(point_delta, direction)
            tangent_delta = v_sub(point_delta, v_mul(direction, axial_distance))
            if v_norm(tangent_delta) < perforation_radius:
                displacement_magnitude = max(
                    displacement_magnitude,
                    perforation_displacement,
                )
        displacement = v_mul(direction, displacement_magnitude)
        max_applied = max(max_applied, v_norm(displacement))
        displaced_vertices[key] = v_add(point, displacement)

    deformed: List[Triangle] = []
    for normal, v1, v2, v3 in triangles:
        points = [
            displaced_vertices[deformation_vertex_key(point)]
            for point in (v1, v2, v3)
        ]
        new_normal = v_unit(
            v_cross(v_sub(points[1], points[0]), v_sub(points[2], points[0])),
            normal,
        )
        deformed.append((new_normal, points[0], points[1], points[2]))
    return deformed, max_applied


def deformation_should_preserve_thickness(
    component: AeroComponent,
    direction: Vec3,
    contact_radius: float,
) -> bool:
    """Return whether a local contact should bend all plate skins coherently."""
    points = stl_points(component.triangles)
    if not points:
        return False
    normal = v_unit(direction)
    tangent_x, tangent_y = contact_tangent_basis(normal)
    axial_values = [v_dot(point, normal) for point in points]
    tangent_x_values = [v_dot(point, tangent_x) for point in points]
    tangent_y_values = [v_dot(point, tangent_y) for point in points]
    thickness = max(axial_values) - min(axial_values)
    in_plane_span = max(
        max(tangent_x_values) - min(tangent_x_values),
        max(tangent_y_values) - min(tangent_y_values),
    )
    return (
        thickness > 0.0
        and thickness <= 0.1 * max(in_plane_span, 1e-9)
        and thickness <= max(contact_radius, COLLISION_DEFORMATION_MIN_RADIUS_M)
    )


def normalized_contact_deformation_parameters(
    triangles: Sequence[Triangle],
    contact_point: Vec3,
    indentation: float,
    contact_radius: float,
) -> Tuple[float, float]:
    radius = max(contact_radius, COLLISION_DEFORMATION_MIN_RADIUS_M)
    nearest_distance = min(
        (v_norm(v_sub(point, contact_point)) for point in stl_points(triangles)),
        default=radius,
    )
    if nearest_distance >= radius:
        radius = max(radius, 1.25 * nearest_distance)
    nearest_fraction = min(nearest_distance / max(radius, 1e-12), 1.0)
    if COLLISION_DEFORMATION_MODEL in {"hertz", "hertzian"}:
        peak_weight = math.sqrt(max(0.0, 1.0 - nearest_fraction ** 2))
    else:
        smooth_fraction = 1.0 - nearest_fraction
        peak_weight = (
            smooth_fraction
            * smooth_fraction
            * (3.0 - 2.0 * smooth_fraction)
        )
    return radius, indentation / max(peak_weight, 1e-12)


def deform_component_at_contact(
    component: AeroComponent,
    contact_point: Vec3,
    inward_direction: Vec3,
    indentation: float,
    contact_radius: float,
    perforation_radius: float = 0.0,
    perforation_displacement: float = 0.0,
) -> float:
    if not ENABLE_COLLISION_DEFORMATION or not collision_deformation_enabled(component):
        return 0.0
    if indentation <= COLLISION_MIN_OVERLAP_M:
        return 0.0

    ensure_deformation_reference(component)
    direction = v_unit(inward_direction)
    max_allowed = min(
        MAX_TOTAL_DEFORMATION,
        COLLISION_MAX_CONTACT_DEFORMATION,
        0.25 * max(component.lref, 1e-6),
    )
    target = min(max_allowed, indentation)
    if target <= 1e-12:
        return 0.0
    radius, amplitude = normalized_contact_deformation_parameters(
        component.triangles,
        contact_point,
        target,
        contact_radius,
    )
    eulerian_grid = build_eulerian_contact_grid(
        component.triangles,
        contact_point,
        direction,
        target,
        radius,
    )

    deformed, max_applied = deform_triangle_mesh_at_contact(
        component.triangles,
        contact_point,
        direction,
        amplitude,
        radius,
        eulerian_grid,
        perforation_radius,
        perforation_displacement,
        deformation_should_preserve_thickness(
            component,
            direction,
            radius,
        ),
    )

    if max_applied <= 1e-12:
        return 0.0
    original_cofr = component.cofr
    component.triangles = deformed
    fragment_parent = component.collision_fragment_parent_state
    fragment_element = component.collision_fragment_source_element
    if (
        isinstance(fragment_parent, HybridFEMMPMCollisionState)
        and fragment_element is not None
        and deformed
    ):
        solid = fragment_parent.solid_state
        surface_nodes = next(
            (
                nodes
                for nodes, element_index in zip(
                    solid.surface_triangle_nodes,
                    solid.surface_element_indices,
                )
                if element_index == fragment_element
            ),
            None,
        )
        if surface_nodes is not None:
            particles = {
                particle.source_node: particle
                for particle in solid.particles
                if particle.source_element == fragment_element
            }
            for node, point in zip(surface_nodes, deformed[0][1:]):
                particle = particles.get(node)
                if particle is None:
                    continue
                displacement = v_sub(point, particle.position)
                particle.position = point
                particle.velocity = v_add(
                    particle.velocity,
                    v_mul(displacement, 1.0 / max(MOTION_DT, 1e-9)),
                )
    component.aref, component.lref, _geometry_centroid = component_references(component.triangles)
    # A local dent changes surface geometry, not the body's centre of mass.
    component.cofr = original_cofr
    component.deformation_max_m = max(component.deformation_max_m, max_applied)
    return max_applied


@dataclass
class SweptCollisionContact:
    moving: AeroComponent
    stationary: AeroComponent
    depth: float
    normal: Vec3
    point: Vec3
    travel_to_contact: float
    approach_axis: Vec3 = (1.0, 0.0, 0.0)
    approach_penetration: float = 0.0
    manifold_points: int = 1
    perforated: bool = False
    residual_speed: float = 0.0
    absorbed_energy_j: float = 0.0
    failure_mode: str = "elastic_contact"
    hole_radius: float = 0.0
    contact_geometry: Optional[LocalContactGeometry] = None
    post_contact_time_s: float = 0.0


@dataclass(frozen=True)
class EulerianContactGrid:
    """Small Cartesian contact grid used for fixed-topology solid deformation."""

    origin: Vec3
    cell_size_m: float
    penalty_pressure_pa: Dict[Tuple[int, int, int], float]

    def pressure_at(self, point: Vec3) -> float:
        indices = tuple(
            math.floor((point[axis] - self.origin[axis]) / self.cell_size_m)
            for axis in range(3)
        )
        return self.penalty_pressure_pa.get(indices, 0.0)


def build_eulerian_contact_grid(
    triangles: Sequence[Triangle],
    contact_point: Vec3,
    inward_direction: Vec3,
    indentation: float,
    contact_radius: float,
) -> EulerianContactGrid:
    """Voxelize contact penetration into a local penalty-pressure field.

    This is intentionally a small, explicit Eulerian grid rather than a global
    fluid solver.  It supplies a stable signed-distance-like contact field for
    coarse STL surfaces; OpenFOAM supplies the fluid pressure and momentum
    loads for the surrounding two-way motion loop.
    """
    radius = max(contact_radius, COLLISION_DEFORMATION_MIN_RADIUS_M)
    cell_size = max(
        radius / COLLISION_EULERIAN_GRID_CELLS,
        COLLISION_EULERIAN_GRID_MIN_CELL_M,
    )
    origin = v_sub(contact_point, (radius, radius, radius))
    direction = v_unit(inward_direction)
    pressure_by_cell: Dict[Tuple[int, int, int], float] = {}

    for triangle in triangles:
        _area, centroid, normal = triangle_area_centroid_normal(triangle)
        delta = v_sub(centroid, contact_point)
        signed_distance = v_dot(delta, direction)
        tangent_delta = v_sub(delta, v_mul(direction, signed_distance))
        radial_distance = v_norm(tangent_delta)
        if radial_distance >= radius:
            continue
        radial_weight = math.sqrt(max(0.0, 1.0 - (radial_distance / radius) ** 2))
        normal_weight = max(0.0, abs(v_dot(normal, direction)))
        penetration = max(0.0, indentation - max(0.0, signed_distance))
        pressure = (
            COLLISION_EULERIAN_PENALTY_GAIN
            * radial_weight
            * max(normal_weight, 0.25)
            * penetration
        )
        if pressure <= 0.0:
            continue
        indices = tuple(
            math.floor((centroid[axis] - origin[axis]) / cell_size)
            for axis in range(3)
        )
        pressure_by_cell[indices] = max(pressure_by_cell.get(indices, 0.0), pressure)

    return EulerianContactGrid(origin, cell_size, pressure_by_cell)


@dataclass(frozen=True)
class ThinShellImpactResponse:
    perforated: bool
    indentation: float
    absorbed_energy_j: float
    residual_speed: float
    hole_radius: float
    failure_mode: str


def _unique_mesh_vertices(triangles: Sequence[Triangle]) -> List[Vec3]:
    return list(dict.fromkeys(stl_points(triangles)))


def ray_triangle_distance(
    origin: Vec3,
    direction: Vec3,
    triangle: Triangle,
    max_distance: float,
) -> Optional[float]:
    """Return a double-sided Moller-Trumbore ray hit within max_distance."""
    _normal, v0, v1, v2 = triangle
    edge1 = v_sub(v1, v0)
    edge2 = v_sub(v2, v0)
    pvec = v_cross(direction, edge2)
    determinant = v_dot(edge1, pvec)
    epsilon = 1e-12
    if abs(determinant) <= epsilon:
        return None
    inverse = 1.0 / determinant
    tvec = v_sub(origin, v0)
    u = v_dot(tvec, pvec) * inverse
    if u < -epsilon or u > 1.0 + epsilon:
        return None
    qvec = v_cross(tvec, edge1)
    v = v_dot(direction, qvec) * inverse
    if v < -epsilon or u + v > 1.0 + epsilon:
        return None
    distance = v_dot(edge2, qvec) * inverse
    if distance < -epsilon or distance > max_distance + epsilon:
        return None
    return max(0.0, distance)


TriangleBounds = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class TriangleAabbNode:
    """Node in a classic median-split triangle AABB tree."""

    bounds: TriangleBounds
    triangle_indices: Tuple[int, ...] = ()
    left: Optional["TriangleAabbNode"] = None
    right: Optional["TriangleAabbNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def triangle_bounds(triangle: Triangle) -> TriangleBounds:
    points = triangle[1:]
    return (
        min(point[0] for point in points),
        max(point[0] for point in points),
        min(point[1] for point in points),
        max(point[1] for point in points),
        min(point[2] for point in points),
        max(point[2] for point in points),
    )


def combined_triangle_bounds(
    triangle_bounds_by_index: Sequence[TriangleBounds],
    indices: Sequence[int],
) -> TriangleBounds:
    return (
        min(triangle_bounds_by_index[index][0] for index in indices),
        max(triangle_bounds_by_index[index][1] for index in indices),
        min(triangle_bounds_by_index[index][2] for index in indices),
        max(triangle_bounds_by_index[index][3] for index in indices),
        min(triangle_bounds_by_index[index][4] for index in indices),
        max(triangle_bounds_by_index[index][5] for index in indices),
    )


def triangle_bounds_overlap(
    first: TriangleBounds,
    second: TriangleBounds,
    tolerance: float = 0.0,
) -> bool:
    return all(
        first[2 * axis + 1] + tolerance >= second[2 * axis]
        and second[2 * axis + 1] + tolerance >= first[2 * axis]
        for axis in range(3)
    )


def build_triangle_aabb_tree(
    triangles: Sequence[Triangle],
    leaf_size: int = 12,
) -> Optional[TriangleAabbNode]:
    """Build a deterministic median-split AABB tree for a triangle mesh."""
    if not triangles:
        return None
    bounds_by_index = [triangle_bounds(triangle) for triangle in triangles]
    centroids = [
        (
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        )
        for bounds in bounds_by_index
    ]

    def build(indices: List[int]) -> TriangleAabbNode:
        node_bounds = combined_triangle_bounds(bounds_by_index, indices)
        if len(indices) <= max(1, leaf_size):
            return TriangleAabbNode(
                bounds=node_bounds,
                triangle_indices=tuple(sorted(indices)),
            )
        extents = (
            node_bounds[1] - node_bounds[0],
            node_bounds[3] - node_bounds[2],
            node_bounds[5] - node_bounds[4],
        )
        split_axis = max(range(3), key=lambda axis: extents[axis])
        ordered = sorted(
            indices,
            key=lambda index: (centroids[index][split_axis], index),
        )
        middle = len(ordered) // 2
        return TriangleAabbNode(
            bounds=node_bounds,
            left=build(ordered[:middle]),
            right=build(ordered[middle:]),
        )

    return build(list(range(len(triangles))))


def aabb_tree_candidates(
    tree: Optional[TriangleAabbNode],
    query_bounds: TriangleBounds,
    tolerance: float = 0.0,
) -> List[int]:
    if tree is None:
        return []
    candidates: List[int] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if not triangle_bounds_overlap(node.bounds, query_bounds, tolerance):
            continue
        if node.is_leaf:
            candidates.extend(node.triangle_indices)
            continue
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return sorted(candidates)


def aabb_tree_pair_candidates(
    first: Optional[TriangleAabbNode],
    second: Optional[TriangleAabbNode],
    tolerance: float = 0.0,
) -> List[Tuple[int, int]]:
    """Return only triangle pairs whose AABBs can overlap."""
    if first is None or second is None:
        return []
    candidates: List[Tuple[int, int]] = []
    stack = [(first, second)]
    while stack:
        first_node, second_node = stack.pop()
        if not triangle_bounds_overlap(
            first_node.bounds,
            second_node.bounds,
            tolerance,
        ):
            continue
        if first_node.is_leaf and second_node.is_leaf:
            candidates.extend(
                (first_index, second_index)
                for first_index in first_node.triangle_indices
                for second_index in second_node.triangle_indices
            )
            continue
        if second_node.is_leaf or (
            not first_node.is_leaf
            and max(
                first_node.bounds[1] - first_node.bounds[0],
                first_node.bounds[3] - first_node.bounds[2],
                first_node.bounds[5] - first_node.bounds[4],
            )
            >= max(
                second_node.bounds[1] - second_node.bounds[0],
                second_node.bounds[3] - second_node.bounds[2],
                second_node.bounds[5] - second_node.bounds[4],
            )
        ):
            if first_node.right is not None:
                stack.append((first_node.right, second_node))
            if first_node.left is not None:
                stack.append((first_node.left, second_node))
        else:
            if second_node.right is not None:
                stack.append((first_node, second_node.right))
            if second_node.left is not None:
                stack.append((first_node, second_node.left))
    return sorted(candidates)


def segment_bounds(
    origin: Vec3,
    direction: Vec3,
    distance: float,
) -> TriangleBounds:
    end = v_add(origin, v_mul(direction, distance))
    return (
        min(origin[0], end[0]),
        max(origin[0], end[0]),
        min(origin[1], end[1]),
        max(origin[1], end[1]),
        min(origin[2], end[2]),
        max(origin[2], end[2]),
    )


def swept_mesh_contact(
    moving: AeroComponent,
    stationary: AeroComponent,
    axis: Vec3,
    max_distance: float,
) -> Optional[Tuple[float, Vec3, Vec3, int]]:
    """Find first vertex/face contact for pure translation along axis.

    Testing vertices in both relative directions handles the common face/vertex
    and vertex/face cases without time stepping or changing pre-impact speed.
    """
    return swept_triangle_mesh_contact(
        moving.triangles,
        stationary.triangles,
        axis,
        max_distance,
    )


def swept_triangle_mesh_contact(
    moving_triangles: Sequence[Triangle],
    stationary_triangles: Sequence[Triangle],
    axis: Vec3,
    max_distance: float,
) -> Optional[Tuple[float, Vec3, Vec3, int]]:
    """Find first contact while translating one triangle mesh along an axis."""
    direction = v_unit(axis)
    best_distance = max_distance + 1.0
    candidates: List[Tuple[Vec3, Vec3]] = []
    tolerance = COLLISION_MANIFOLD_TOLERANCE_M
    moving_tree = build_triangle_aabb_tree(moving_triangles)
    stationary_tree = build_triangle_aabb_tree(stationary_triangles)

    def record(distance: float, point: Vec3, triangle: Triangle) -> None:
        nonlocal best_distance, candidates
        _area, _centroid, normal = triangle_area_centroid_normal(triangle)
        if v_dot(normal, direction) > 0.0:
            normal = v_mul(normal, -1.0)
        if distance < best_distance - tolerance:
            best_distance = distance
            candidates = [(point, normal)]
        elif abs(distance - best_distance) <= tolerance:
            candidates.append((point, normal))

    for vertex in _unique_mesh_vertices(moving_triangles):
        candidate_indices = aabb_tree_candidates(
            stationary_tree,
            segment_bounds(vertex, direction, max_distance),
            tolerance,
        )
        for triangle_index in candidate_indices:
            triangle = stationary_triangles[triangle_index]
            distance = ray_triangle_distance(vertex, direction, triangle, max_distance)
            if distance is not None:
                record(distance, v_add(vertex, v_mul(direction, distance)), triangle)

    reverse = v_mul(direction, -1.0)
    for vertex in _unique_mesh_vertices(stationary_triangles):
        candidate_indices = aabb_tree_candidates(
            moving_tree,
            segment_bounds(vertex, reverse, max_distance),
            tolerance,
        )
        for triangle_index in candidate_indices:
            triangle = moving_triangles[triangle_index]
            distance = ray_triangle_distance(vertex, reverse, triangle, max_distance)
            if distance is not None:
                record(distance, vertex, triangle)

    if not candidates:
        return None
    count = len(candidates)
    point = v_mul(
        tuple(sum(candidate[0][axis_index] for candidate in candidates) for axis_index in range(3)),
        1.0 / count,
    )
    normal_sum = tuple(
        sum(candidate[1][axis_index] for candidate in candidates)
        for axis_index in range(3)
    )
    normal = v_unit(normal_sum, v_mul(direction, -1.0))
    if v_dot(normal, direction) > 0.0:
        normal = v_mul(normal, -1.0)
    return best_distance, point, normal, count


@dataclass(frozen=True)
class RelativeSweptContact:
    travel_to_contact_m: float
    sweep_distance_m: float
    direction: Vec3
    point: Vec3
    normal: Vec3
    manifold_points: int
    time_fraction: float = 1.0
    rotational: bool = False
    translation_a: Vec3 = (0.0, 0.0, 0.0)
    translation_b: Vec3 = (0.0, 0.0, 0.0)
    rotation_a: Vec3 = (0.0, 0.0, 0.0)
    rotation_b: Vec3 = (0.0, 0.0, 0.0)


def _swept_linear_component_contact(
    a: AeroComponent,
    b: AeroComponent,
    dt_s: float,
) -> Optional[RelativeSweptContact]:
    """Detect triangle contact crossed by two translating bodies in one step.

    The calculation is performed in ``b``'s translating reference frame. It is
    the classic linear continuous-collision test: move ``a`` back by the last
    relative displacement, then ray-sweep its surface to the current position.
    Rotational sweep is intentionally left to the normal discrete contact pass.
    """
    relative_velocity = v_sub(a.linear_velocity, b.linear_velocity)
    sweep_distance = v_norm(relative_velocity) * max(dt_s, 0.0)
    if sweep_distance <= COLLISION_MIN_OVERLAP_M:
        return None
    direction = v_unit(relative_velocity)
    relative_displacement = v_mul(direction, sweep_distance)
    end_bounds = component_bounds(a.triangles)
    start_bounds = (
        end_bounds[0] - relative_displacement[0],
        end_bounds[1] - relative_displacement[0],
        end_bounds[2] - relative_displacement[1],
        end_bounds[3] - relative_displacement[1],
        end_bounds[4] - relative_displacement[2],
        end_bounds[5] - relative_displacement[2],
    )
    b_bounds = component_bounds(b.triangles)
    swept_bounds = (
        min(start_bounds[0], end_bounds[0]),
        max(start_bounds[1], end_bounds[1]),
        min(start_bounds[2], end_bounds[2]),
        max(start_bounds[3], end_bounds[3]),
        min(start_bounds[4], end_bounds[4]),
        max(start_bounds[5], end_bounds[5]),
    )
    margin = max(COLLISION_MARGIN_M, COLLISION_MANIFOLD_TOLERANCE_M)
    if (
        swept_bounds[1] + margin < b_bounds[0]
        or b_bounds[1] + margin < swept_bounds[0]
        or swept_bounds[3] + margin < b_bounds[2]
        or b_bounds[3] + margin < swept_bounds[2]
        or swept_bounds[5] + margin < b_bounds[4]
        or b_bounds[5] + margin < swept_bounds[4]
    ):
        return None

    start_triangles = [
        (
            normal,
            v_sub(point_a, relative_displacement),
            v_sub(point_b, relative_displacement),
            v_sub(point_c, relative_displacement),
        )
        for normal, point_a, point_b, point_c in a.triangles
    ]
    hit = swept_triangle_mesh_contact(
        start_triangles,
        b.triangles,
        direction,
        sweep_distance,
    )
    if hit is None:
        return None
    travel, point, normal, manifold_points = hit
    if travel <= COLLISION_MANIFOLD_TOLERANCE_M:
        start_center = (
            0.5 * (start_bounds[0] + start_bounds[1]),
            0.5 * (start_bounds[2] + start_bounds[3]),
            0.5 * (start_bounds[4] + start_bounds[5]),
        )
        b_center = (
            0.5 * (b_bounds[0] + b_bounds[1]),
            0.5 * (b_bounds[2] + b_bounds[3]),
            0.5 * (b_bounds[4] + b_bounds[5]),
        )
        if v_dot(normal, v_sub(start_center, b_center)) <= 0.0:
            # A zero-time ray hit while the bodies move apart is their previous
            # touching state, not a new impact. Let independent bodies separate.
            return None
    return RelativeSweptContact(
        travel_to_contact_m=travel,
        sweep_distance_m=sweep_distance,
        direction=direction,
        point=point,
        normal=normal,
        manifold_points=manifold_points,
        time_fraction=min(1.0, travel / max(sweep_distance, 1e-12)),
    )


def _triangle_bounds_overlap(
    a: Triangle,
    b: Triangle,
    tolerance: float,
) -> bool:
    return triangle_bounds_overlap(
        triangle_bounds(a),
        triangle_bounds(b),
        tolerance,
    )


def triangle_mesh_intersection_contact(
    a_triangles: Sequence[Triangle],
    b_triangles: Sequence[Triangle],
) -> Optional[Tuple[Vec3, Vec3]]:
    """Return a surface intersection using classic edge/triangle tests."""
    if not a_triangles or not b_triangles:
        return None
    tolerance = max(COLLISION_MANIFOLD_TOLERANCE_M, 1e-9)
    a_bounds = component_bounds(a_triangles)
    b_bounds = component_bounds(b_triangles)
    if any(
        a_bounds[2 * axis + 1] + tolerance < b_bounds[2 * axis]
        or b_bounds[2 * axis + 1] + tolerance < a_bounds[2 * axis]
        for axis in range(3)
    ):
        return None
    a_center = (
        0.5 * (a_bounds[0] + a_bounds[1]),
        0.5 * (a_bounds[2] + a_bounds[3]),
        0.5 * (a_bounds[4] + a_bounds[5]),
    )
    b_center = (
        0.5 * (b_bounds[0] + b_bounds[1]),
        0.5 * (b_bounds[2] + b_bounds[3]),
        0.5 * (b_bounds[4] + b_bounds[5]),
    )

    def edge_hit(
        edge_start: Vec3,
        edge_end: Vec3,
        triangle: Triangle,
    ) -> Optional[Vec3]:
        edge = v_sub(edge_end, edge_start)
        length = v_norm(edge)
        if length <= 1e-12:
            return None
        distance = ray_triangle_distance(
            edge_start,
            v_mul(edge, 1.0 / length),
            triangle,
            length,
        )
        if distance is None:
            return None
        return v_add(edge_start, v_mul(edge, distance / length))

    candidate_pairs = aabb_tree_pair_candidates(
        build_triangle_aabb_tree(a_triangles),
        build_triangle_aabb_tree(b_triangles),
        tolerance,
    )
    for triangle_a_index, triangle_b_index in candidate_pairs:
        triangle_a = a_triangles[triangle_a_index]
        a_vertices = triangle_a[1:]
        triangle_b = b_triangles[triangle_b_index]
        for start, end in (
            (a_vertices[0], a_vertices[1]),
            (a_vertices[1], a_vertices[2]),
            (a_vertices[2], a_vertices[0]),
        ):
            point = edge_hit(start, end, triangle_b)
            if point is not None:
                _area, _centroid, normal = triangle_area_centroid_normal(triangle_b)
                if v_dot(normal, v_sub(a_center, b_center)) < 0.0:
                    normal = v_mul(normal, -1.0)
                return point, normal
        b_vertices = triangle_b[1:]
        for start, end in (
            (b_vertices[0], b_vertices[1]),
            (b_vertices[1], b_vertices[2]),
            (b_vertices[2], b_vertices[0]),
        ):
            point = edge_hit(start, end, triangle_a)
            if point is not None:
                _area, _centroid, normal = triangle_area_centroid_normal(triangle_a)
                if v_dot(normal, v_sub(a_center, b_center)) < 0.0:
                    normal = v_mul(normal, -1.0)
                return point, normal
    return None


def _component_motion_over_step(
    component: AeroComponent,
    dt_s: float,
) -> Tuple[Vec3, Vec3]:
    return (
        v_mul(component.linear_velocity, max(dt_s, 0.0)),
        v_mul(component.angular_velocity, max(dt_s, 0.0)),
    )


def _component_start_pose(
    component: AeroComponent,
    translation: Vec3,
    rotation: Vec3,
) -> Tuple[List[Triangle], Vec3]:
    end_origin = infer_motion_origin(component)
    angle = v_norm(rotation)
    start_triangles = move_triangles(
        component.triangles,
        v_mul(translation, -1.0),
        v_unit(rotation) if angle > 1e-12 else None,
        -angle,
        end_origin,
    )
    return start_triangles, v_sub(end_origin, translation)


def _component_pose_at_fraction(
    start_triangles: Sequence[Triangle],
    start_origin: Vec3,
    translation: Vec3,
    rotation: Vec3,
    fraction: float,
) -> List[Triangle]:
    angle = v_norm(rotation)
    return move_triangles(
        start_triangles,
        v_mul(translation, fraction),
        v_unit(rotation) if angle > 1e-12 else None,
        angle * fraction,
        start_origin,
    )


def swept_rotational_component_contact(
    a: AeroComponent,
    b: AeroComponent,
    dt_s: float,
) -> Optional[RelativeSweptContact]:
    """Conservative angular CCD using fixed-angle pose substeps.

    Linear vertex/face CCD remains exact for translation. This supplementary
    classic substep sweep handles the curved paths created by rigid rotation,
    including rotations greater than one revolution in a dynamic step.
    """
    translation_a, rotation_a = _component_motion_over_step(a, dt_s)
    translation_b, rotation_b = _component_motion_over_step(b, dt_s)
    max_angle = max(v_norm(rotation_a), v_norm(rotation_b))
    if max_angle <= 1e-10:
        return None
    substeps = min(
        COLLISION_ROTATION_SWEEP_MAX_SUBSTEPS,
        max(1, int(math.ceil(max_angle / COLLISION_ROTATION_SWEEP_MAX_ANGLE_RAD))),
    )
    start_a, origin_a = _component_start_pose(a, translation_a, rotation_a)
    start_b, origin_b = _component_start_pose(b, translation_b, rotation_b)
    for substep in range(1, substeps + 1):
        fraction = substep / substeps
        current_a = _component_pose_at_fraction(
            start_a,
            origin_a,
            translation_a,
            rotation_a,
            fraction,
        )
        current_b = _component_pose_at_fraction(
            start_b,
            origin_b,
            translation_b,
            rotation_b,
            fraction,
        )
        hit = triangle_mesh_intersection_contact(current_a, current_b)
        if hit is not None:
            point, normal = hit
            relative_point_velocity = v_sub(
                contact_point_velocity(a, point),
                contact_point_velocity(b, point),
            )
            if v_dot(relative_point_velocity, normal) >= -1e-9:
                continue
            motion_bound = (
                v_norm(v_sub(translation_a, translation_b))
                + v_norm(rotation_a) * max(a.lref, 1e-9)
                + v_norm(rotation_b) * max(b.lref, 1e-9)
            )
            return RelativeSweptContact(
                travel_to_contact_m=motion_bound * fraction,
                sweep_distance_m=motion_bound,
                direction=v_unit(relative_point_velocity, v_mul(normal, -1.0)),
                point=point,
                normal=normal,
                manifold_points=1,
                time_fraction=fraction,
                rotational=True,
                translation_a=translation_a,
                translation_b=translation_b,
                rotation_a=rotation_a,
                rotation_b=rotation_b,
            )
    return None


def swept_relative_component_contact(
    a: AeroComponent,
    b: AeroComponent,
    dt_s: float,
) -> Optional[RelativeSweptContact]:
    linear_contact = _swept_linear_component_contact(a, b, dt_s)
    rotational_contact = swept_rotational_component_contact(a, b, dt_s)
    if linear_contact is None:
        return rotational_contact
    if rotational_contact is None:
        return linear_contact
    return min(
        (linear_contact, rotational_contact),
        key=lambda contact: contact.time_fraction,
    )


def impact_contact_indentation(
    a: AeroComponent,
    b: AeroComponent,
    normal_speed: float,
    contact_point_a: Optional[Vec3] = None,
    contact_point_b: Optional[Vec3] = None,
    normal_from_b_to_a: Optional[Vec3] = None,
    geometry: Optional[LocalContactGeometry] = None,
) -> float:
    """Maximum elastic indentation from impact kinetic energy."""
    if not ENABLE_COLLISION_DEFORMATION or normal_speed <= 0.0:
        return 0.0
    compliance = component_contact_compliance(a) + component_contact_compliance(b)
    if compliance <= 1e-30:
        return 0.0
    effective_modulus = 1.0 / compliance
    if geometry is None and (
        contact_point_a is not None
        and contact_point_b is not None
        and normal_from_b_to_a is not None
    ):
        geometry = local_contact_geometry(
            a,
            b,
            contact_point_a,
            contact_point_b,
            normal_from_b_to_a,
        )

    if a.is_assembly_anchor:
        effective_mass = max(b.mass, 1e-9)
    elif b.is_assembly_anchor:
        effective_mass = max(a.mass, 1e-9)
    else:
        effective_mass = max(a.mass * b.mass / max(a.mass + b.mass, 1e-9), 1e-9)
    kinetic_energy = 0.5 * effective_mass * normal_speed * normal_speed
    if geometry is not None and not geometry.hertzian:
        # Classical elastic flat-punch relation: F = 2 E* a delta.
        stiffness = 2.0 * effective_modulus * geometry.footprint_radius
        indentation = math.sqrt(2.0 * kinetic_energy / max(stiffness, 1e-30))
    else:
        radius = (
            geometry.effective_radius
            if geometry is not None
            else effective_contact_radius(a, b)
        )
        stiffness = (4.0 / 3.0) * effective_modulus * math.sqrt(radius)
        if stiffness <= 1e-30:
            return 0.0
        indentation = (
            (5.0 * effective_mass * normal_speed * normal_speed)
            / (4.0 * stiffness)
        ) ** 0.4
    return min(
        indentation,
        COLLISION_MAX_CONTACT_DEFORMATION,
        MAX_TOTAL_DEFORMATION,
        0.1 * max(min(a.lref, b.lref), 1e-6),
    )


def material_yield_strength_pa(component: AeroComponent) -> float:
    if component.material.yield_strength_pa is not None:
        return max(component.material.yield_strength_pa, 1.0)
    name = component.material.material_name.lower()
    values = {
        "abs": 4.0e7,
        "pla": 5.5e7,
        "petg": 5.0e7,
        "nylon": 4.5e7,
        "balsa": 1.5e7,
        "plywood": 3.5e7,
        "mdf": 2.5e7,
        "wood": 4.0e7,
        "aluminum": 2.75e8,
        "aluminium": 2.75e8,
        "steel": 2.5e8,
        "tungsten": 7.5e8,
        "wolfram": 7.5e8,
    }
    return next((value for key, value in values.items() if key in name), 3.5e7)


def material_failure_strain(component: AeroComponent) -> float:
    if component.material.failure_strain is not None:
        return max(component.material.failure_strain, 0.0)
    name = component.material.material_name.lower()
    values = {
        "abs": 0.20,
        "pla": 0.04,
        "petg": 0.18,
        "nylon": 0.30,
        "balsa": 0.012,
        "plywood": 0.018,
        "mdf": 0.015,
        "wood": 0.02,
        "aluminum": 0.12,
        "aluminium": 0.12,
        "steel": 0.20,
        "tungsten": 0.01,
        "wolfram": 0.01,
    }
    return next((value for key, value in values.items() if key in name), 0.10)


def material_splinter_fraction(component: AeroComponent) -> float:
    name = component.material.material_name.lower()
    if any(word in name for word in ("wood", "balsa", "plywood", "mdf", "chipboard")):
        return 1.0
    if any(word in name for word in ("glass", "ceramic", "carbon", "fiberglass", "fibre", "fiber")):
        return 0.85
    if any(word in name for word in ("pla", "acrylic", "polycarbonate")):
        return 0.45
    failure_strain = material_failure_strain(component)
    return max(0.0, min(1.0, (0.08 - failure_strain) / 0.08))


def material_brittleness_factor(component: AeroComponent) -> float:
    """Return a bounded multiplier for crack spread and chip formation.

    Prefer BOM-derived failure strain/yield strength when available, then fall
    back to conservative material-name cues for very brittle materials.
    """
    name = component.material.material_name.lower()
    if any(word in name for word in ("porcelain", "ceramic", "china", "stoneware")):
        return 3.5
    if any(word in name for word in ("glass", "tempered glass", "borosilicate")):
        return 3.2
    if any(word in name for word in ("concrete", "mortar", "brick", "tile")):
        return 2.8
    if any(word in name for word in ("wood", "plywood", "mdf", "chipboard")):
        return 2.0
    if any(word in name for word in ("abs", "pla", "polycarbonate", "acrylic", "nylon", "polymer", "plastic")):
        return 1.0
    failure_strain = max(material_failure_strain(component), 1e-4)
    yield_strength = max(material_yield_strength_pa(component), 1e5)
    young = max(inferred_deformation_young_modulus(component), 1e6)
    yield_strain = max(yield_strength / young, 1e-5)
    ductility_ratio = max(1.0, min(25.0, failure_strain / yield_strain))
    return max(1.0, min(3.5, math.sqrt(25.0 / ductility_ratio)))


def impactor_contact_radius(
    component: AeroComponent,
    contact_point: Vec3,
    outward_normal: Vec3,
) -> float:
    curvature_radius = local_surface_curvature_radius(
        component,
        contact_point,
        outward_normal,
    )
    if not math.isinf(curvature_radius):
        return curvature_radius
    return local_planar_patch_radius(component, contact_point, outward_normal)


def impactor_projected_clearance_radius(
    component: AeroComponent,
    impact_axis: Vec3,
) -> float:
    """Return the minimum circular through-hole radius for the impactor width.

    Contact-radius estimates are intentionally local: they are useful for Hertz
    pressure and indentation, but they are too small for a projectile that fully
    perforates a thin sheet.  A through-hole must clear the projectile's
    projected cross-section, so use the impactor's projected bounding width in
    the target plane as a simple, deterministic clearance estimate.
    """
    points = stl_points(component.triangles)
    if not points:
        return 0.0

    tangent_x, tangent_y = contact_tangent_basis(impact_axis)
    projected_x = [v_dot(point, tangent_x) for point in points]
    projected_y = [v_dot(point, tangent_y) for point in points]
    span_x = max(projected_x) - min(projected_x)
    span_y = max(projected_y) - min(projected_y)
    return 0.5 * max(span_x, span_y, 0.0)


THIN_SHELL_HOLE_CLEARANCE_FACTOR = 1.05


def thin_shell_impact_response(
    impactor: AeroComponent,
    target: AeroComponent,
    speed: float,
    impactor_contact_point: Vec3,
    normal_from_target_to_impactor: Vec3,
) -> Optional[ThinShellImpactResponse]:
    # A detached structural fragment is already owned by its parent shell or
    # FEM/MPM state. Treating that proxy as a new thin shell recursively creates
    # a second continuum model and destroys its inherited momentum. It still
    # participates in the ordinary impulse/deformation contact path; only a
    # second, independent perforation topology is inapplicable.
    if target.collision_fragment_parent_state is not None:
        return None

    target_normal = v_unit(normal_from_target_to_impactor)
    target_points = stl_points(target.triangles)
    tangent_x, tangent_y = contact_tangent_basis(target_normal)
    normal_values = [v_dot(point, target_normal) for point in target_points]
    tangent_x_values = [v_dot(point, tangent_x) for point in target_points]
    tangent_y_values = [v_dot(point, tangent_y) for point in target_points]
    geometric_thickness = (
        max(normal_values) - min(normal_values)
        if normal_values
        else math.inf
    )
    in_plane_spans = [
        max(values) - min(values)
        for values in (tangent_x_values, tangent_y_values)
        if values
    ]
    largest_span = max(in_plane_spans, default=max(target.lref, 1e-9))
    specified_thickness = target.material.thickness_m
    thickness = specified_thickness or geometric_thickness
    is_thin = (
        thickness < 0.05 * max(target.lref, 1e-9)
        and (
            specified_thickness is not None
            or geometric_thickness < 0.05 * max(largest_span, 1e-9)
        )
    )
    if not is_thin:
        return None

    contact_radius = max(
        impactor_contact_radius(
            impactor,
            impactor_contact_point,
            v_mul(target_normal, -1.0),
        ),
        thickness,
    )
    clearance_radius = max(
        contact_radius,
        impactor_projected_clearance_radius(impactor, target_normal),
    )
    radius = clearance_radius
    affected_radius = 2.5 * radius
    young_modulus = max(inferred_deformation_young_modulus(target), 1.0)
    yield_strength = material_yield_strength_pa(target)
    failure_strain = material_failure_strain(target)
    affected_volume = math.pi * affected_radius * affected_radius * thickness
    # Elastic work up to yield plus perfectly-plastic work to the tabulated
    # failure strain.  This makes the BOM Young's modulus matter to the
    # perforation threshold as well as to contact compliance and shell FEM.
    yield_strain = min(yield_strength / young_modulus, failure_strain)
    elastic_energy_density = 0.5 * young_modulus * yield_strain * yield_strain
    plastic_energy_density = yield_strength * max(
        failure_strain - yield_strain,
        0.0,
    )
    membrane_energy = affected_volume * (
        elastic_energy_density + plastic_energy_density
    )
    shear_strength = yield_strength / math.sqrt(3.0)
    plug_shear_energy = 2.0 * math.pi * radius * thickness * thickness * shear_strength
    absorbed_energy = membrane_energy + plug_shear_energy
    kinetic_energy = 0.5 * max(impactor.mass, 1e-9) * speed * speed

    if kinetic_energy > absorbed_energy:
        residual_speed = math.sqrt(
            max(0.0, 2.0 * (kinetic_energy - absorbed_energy) / max(impactor.mass, 1e-9))
        )
        return ThinShellImpactResponse(
            perforated=True,
            indentation=min(affected_radius, MAX_TOTAL_DEFORMATION),
            absorbed_energy_j=absorbed_energy,
            residual_speed=residual_speed,
            hole_radius=THIN_SHELL_HOLE_CLEARANCE_FACTOR * clearance_radius,
            failure_mode="plastic_membrane_perforation",
        )

    energy_fraction = kinetic_energy / max(absorbed_energy, 1e-12)
    indentation = min(
        affected_radius * math.sqrt(max(energy_fraction, 0.0)),
        MAX_TOTAL_DEFORMATION,
    )
    return ThinShellImpactResponse(
        perforated=False,
        indentation=indentation,
        absorbed_energy_j=kinetic_energy,
        residual_speed=0.0,
        hole_radius=0.0,
        failure_mode="plastic_membrane_dent",
    )


def triangle_max_edge_length(triangle: Triangle) -> float:
    _normal, a, b, c = triangle
    return max(
        v_norm(v_sub(b, a)),
        v_norm(v_sub(c, b)),
        v_norm(v_sub(a, c)),
    )


def subdivide_triangle(triangle: Triangle) -> List[Triangle]:
    normal, a, b, c = triangle
    ab = v_mul(v_add(a, b), 0.5)
    bc = v_mul(v_add(b, c), 0.5)
    ca = v_mul(v_add(c, a), 0.5)
    children = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
    return [
        (
            v_unit(v_cross(v_sub(q, p), v_sub(r, p)), normal),
            p,
            q,
            r,
        )
        for p, q, r in children
    ]


def _segment_distance_to_origin_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    denom = ab[0] * ab[0] + ab[1] * ab[1]
    if denom <= 1e-30:
        return math.hypot(a[0], a[1])
    t = max(0.0, min(1.0, -(a[0] * ab[0] + a[1] * ab[1]) / denom))
    closest = (a[0] + t * ab[0], a[1] + t * ab[1])
    return math.hypot(closest[0], closest[1])


def _origin_inside_triangle_2d(points: Sequence[Tuple[float, float]]) -> bool:
    signs = []
    for i in range(3):
        a = points[i]
        b = points[(i + 1) % 3]
        signs.append((b[0] - a[0]) * (-a[1]) - (b[1] - a[1]) * (-a[0]))
    return all(sign >= -1e-15 for sign in signs) or all(sign <= 1e-15 for sign in signs)


def triangle_radial_bounds_about_axis(
    triangle: Triangle,
    contact_point: Vec3,
    axis: Vec3,
    tangent_x: Vec3,
    tangent_y: Vec3,
) -> Tuple[float, float, float]:
    coords = [
        (
            v_dot(v_sub(point, contact_point), tangent_x),
            v_dot(v_sub(point, contact_point), tangent_y),
        )
        for point in triangle[1:]
    ]
    vertex_radii = [math.hypot(x, y) for x, y in coords]
    min_radius = 0.0 if _origin_inside_triangle_2d(coords) else min(
        min(vertex_radii),
        _segment_distance_to_origin_2d(coords[0], coords[1]),
        _segment_distance_to_origin_2d(coords[1], coords[2]),
        _segment_distance_to_origin_2d(coords[2], coords[0]),
    )
    max_radius = max(vertex_radii)
    _area, centroid, _normal = triangle_area_centroid_normal(triangle)
    delta = v_sub(centroid, contact_point)
    centroid_radius = v_norm(v_sub(delta, v_mul(axis, v_dot(delta, axis))))
    return min_radius, max_radius, centroid_radius


def fracture_resolution_edge_length(hole_radius: float, affected_radius: float) -> float:
    local_scale = max(hole_radius, affected_radius - hole_radius, COLLISION_DEFORMATION_MIN_RADIUS_M)
    return max(local_scale / 8.0, COLLISION_MIN_OVERLAP_M)


def split_fracture_triangle_near_hole(
    triangle: Triangle,
    contact_point: Vec3,
    axis: Vec3,
    tangent_x: Vec3,
    tangent_y: Vec3,
    hole_radius: float,
    affected_radius: float,
) -> List[Triangle]:
    target_edge = fracture_resolution_edge_length(hole_radius, affected_radius)
    stack: List[Tuple[Triangle, int]] = [(triangle, 0)]
    output: List[Triangle] = []
    while stack:
        candidate, depth = stack.pop()
        min_radius, max_radius, _centroid_radius = triangle_radial_bounds_about_axis(
            candidate,
            contact_point,
            axis,
            tangent_x,
            tangent_y,
        )
        crosses_hole = min_radius < hole_radius < max_radius
        crosses_rim = min_radius < affected_radius and max_radius > hole_radius
        too_coarse = triangle_max_edge_length(candidate) > target_edge
        if (
            (crosses_hole or crosses_rim)
            and too_coarse
            and depth < COLLISION_FRACTURE_MAX_SUBDIVISION_DEPTH
        ):
            stack.extend((child, depth + 1) for child in subdivide_triangle(candidate))
        else:
            output.append(candidate)
    return output


def splinter_triangle(
    triangle: Triangle,
    target: AeroComponent,
    contact_point: Vec3,
    axis: Vec3,
    tangent_x: Vec3,
    tangent_y: Vec3,
    hole_radius: float,
    max_deflection: float,
) -> List[Triangle]:
    splinter_fraction = material_splinter_fraction(target)
    candidates = [triangle]
    if splinter_fraction > 0.25:
        candidates = [child for candidate in candidates for child in subdivide_triangle(candidate)]
    if splinter_fraction > 0.75:
        candidates = [child for candidate in candidates for child in subdivide_triangle(candidate)]

    fragments: List[Triangle] = []
    for index, candidate in enumerate(candidates):
        _area, centroid, normal = triangle_area_centroid_normal(candidate)
        delta = v_sub(centroid, contact_point)
        tangent_delta = v_sub(delta, v_mul(axis, v_dot(delta, axis)))
        if v_norm(tangent_delta) <= 1e-12:
            sign = 1.0 if index % 2 == 0 else -1.0
            tangent_delta = v_add(tangent_x, v_mul(tangent_y, sign))
        radial_dir = v_unit(tangent_delta)
        angular_bias = v_dot(radial_dir, tangent_x) - v_dot(radial_dir, tangent_y)
        axial_scale = 1.0 + 0.75 * splinter_fraction + 0.15 * angular_bias
        radial_scale = 1.05 + 0.85 * splinter_fraction

        points: List[Vec3] = []
        for point in candidate[1:]:
            point_delta = v_sub(point, contact_point)
            radial = v_norm(v_sub(point_delta, v_mul(axis, v_dot(point_delta, axis))))
            opening = max(0.0, hole_radius - radial) + 0.15 * hole_radius
            displacement = v_add(
                v_mul(axis, max_deflection * axial_scale),
                v_mul(radial_dir, opening * radial_scale),
            )
            max_deflection_point = v_add(point, displacement)
            points.append(max_deflection_point)
        new_normal = v_unit(
            v_cross(v_sub(points[1], points[0]), v_sub(points[2], points[0])),
            normal,
        )
        fragments.append((new_normal, points[0], points[1], points[2]))
    return fragments


def fracture_thin_shell(
    target: AeroComponent,
    contact_point: Vec3,
    impact_axis: Vec3,
    hole_radius: float,
    rim_displacement_m: Optional[float] = None,
) -> Tuple[float, int]:
    axis = v_unit(impact_axis)
    brittleness = material_brittleness_factor(target)
    splinter_fraction = material_splinter_fraction(target)
    affected_radius = max(
        2.5 * max(hole_radius, 1e-6),
        (1.0 + 0.85 * brittleness) * max(hole_radius, 1e-6),
    )
    max_deflection = min(
        (
            2.0 * hole_radius
            if rim_displacement_m is None
            else max(rim_displacement_m, 0.0)
        ),
        MAX_TOTAL_DEFORMATION,
    )
    if hole_radius <= 0.0 or max_deflection <= 1e-12:
        return 0.0, 0
    fractured: List[Triangle] = []
    displaced_fragments = 0
    max_displacement = 0.0
    tangent_x, tangent_y = contact_tangent_basis(axis)
    # Once material has been displaced out of the sheet it is a fragment, not
    # still a candidate for the next hole-growth pass.  Re-fracturing those
    # fragments made their triangle count grow every frame and could make the
    # visualisation appear empty after a collision.
    intact_plane_tolerance = max(
        2.0 * inferred_deformation_thickness(target),
        COLLISION_MIN_OVERLAP_M,
    )
    original_cofr = target.cofr

    for triangle in target.triangles:
        _area, centroid, normal = triangle_area_centroid_normal(triangle)
        face_on = abs(v_dot(normal, axis)) >= 0.5
        candidate_triangles = (
            split_fracture_triangle_near_hole(
                triangle,
                contact_point,
                axis,
                tangent_x,
                tangent_y,
                hole_radius,
                affected_radius,
            )
            if face_on
            else [triangle]
        )

        for candidate in candidate_triangles:
            _area, centroid, normal = triangle_area_centroid_normal(candidate)
            min_radial, max_radial, centroid_radial = triangle_radial_bounds_about_axis(
                candidate,
                contact_point,
                axis,
                tangent_x,
                tangent_y,
            )
            axial_offset = abs(v_dot(v_sub(centroid, contact_point), axis))
            intact_near_contact_plane = axial_offset <= intact_plane_tolerance
            if (
                face_on
                and intact_near_contact_plane
                and (max_radial <= hole_radius or centroid_radial < hole_radius)
            ):
                splinter_radius = max(
                    hole_radius,
                    min(
                        affected_radius,
                        hole_radius + splinter_fraction * brittleness * max(hole_radius, 1e-6),
                    ),
                )
                fragments = splinter_triangle(
                    candidate,
                    target,
                    contact_point,
                    axis,
                    tangent_x,
                    tangent_y,
                    splinter_radius,
                    max_deflection,
                )
                displaced_fragments += len(fragments)
                for fragment in fragments:
                    _fragment_area, fragment_centroid, _fragment_normal = triangle_area_centroid_normal(fragment)
                    fragment_delta = v_sub(fragment_centroid, contact_point)
                    fragment_radial = v_norm(
                        v_sub(fragment_delta, v_mul(axis, v_dot(fragment_delta, axis)))
                    )
                    max_displacement = max(
                        max_displacement,
                        max_deflection,
                        max(0.0, hole_radius - fragment_radial),
                    )
                    fractured.append(fragment)
                continue

            points: List[Vec3] = []
            for point in candidate[1:]:
                delta = v_sub(point, contact_point)
                radial = v_norm(v_sub(delta, v_mul(axis, v_dot(delta, axis))))
                if face_on and hole_radius <= radial < affected_radius:
                    weight = (affected_radius - radial) / max(affected_radius - hole_radius, 1e-9)
                    displacement = max_deflection * min(
                        1.5,
                        (0.6 + 0.3 * brittleness) * weight * weight,
                    )
                else:
                    displacement = 0.0
                max_displacement = max(max_displacement, displacement)
                points.append(v_add(point, v_mul(axis, displacement)))
            new_normal = v_unit(
                v_cross(v_sub(points[1], points[0]), v_sub(points[2], points[0])),
                normal,
            )
            fractured.append((new_normal, points[0], points[1], points[2]))

    if displaced_fragments or max_displacement > 1e-12:
        target.triangles = fractured
        target.aref, target.lref, _geometry_centroid = component_references(fractured)
        # Damage changes the surface centroid, but it must not translate the
        # body reference frame.  In particular a prescribed stationary target
        # remains stationary while its fragments and rim deform locally.
        target.cofr = original_cofr
        target.deformation_max_m = max(target.deformation_max_m, max_displacement)
    return max_displacement, displaced_fragments


def material_longitudinal_wave_speed(component: AeroComponent) -> float:
    young = max(inferred_deformation_young_modulus(component), 1.0)
    density = max(component.material.density_kg_m3, 1.0)
    poisson = inferred_deformation_poisson_ratio(component)
    denominator = max(
        density * (1.0 + poisson) * (1.0 - 2.0 * poisson),
        1e-12,
    )
    return math.sqrt(young * (1.0 - poisson) / denominator)


def collision_plastic_fraction(
    component: AeroComponent,
    indentation: float,
    contact_radius: float,
) -> float:
    if indentation <= 0.0:
        return 0.0
    young = max(inferred_deformation_young_modulus(component), 1.0)
    yield_strain = material_yield_strength_pa(component) / young
    failure_strain = max(material_failure_strain(component), 1.01 * yield_strain)
    contact_strain = indentation / max(contact_radius, 1e-9)
    return max(
        0.0,
        min(
            1.0,
            (contact_strain - yield_strain)
            / max(failure_strain - yield_strain, 1e-12),
        ),
    )


def collision_damage_response_time(
    component: AeroComponent,
    contact_radius: float,
) -> float:
    wave_speed = material_longitudinal_wave_speed(component)
    physical_response_time = 2.0 * max(contact_radius, 1e-9) / max(wave_speed, 1e-9)
    # A response faster than the output interval is physically unresolved in
    # the animation.  Retain the physical lower bound, but use a configurable
    # frame-resolved floor so a dent/hole evolves across saved frames instead
    # of appearing fully formed in the first post-impact frame.
    return max(physical_response_time, COLLISION_DAMAGE_MIN_RESPONSE_TIME_S, 1e-9)


def collision_shell_displacement_limit(
    component: AeroComponent,
    hole_radius: float,
    contact_radius: float,
) -> float:
    thickness = inferred_deformation_thickness(component)
    local_radius = max(hole_radius, min(contact_radius, hole_radius), thickness)
    geometry_limit = max(
        1.5 * local_radius,
        3.0 * thickness,
    )
    return max(COLLISION_SHELL_DISPLACEMENT_LIMIT_M, geometry_limit)


def _matching_collision_damage(
    component: AeroComponent,
    contact_point: Vec3,
    inward_direction: Vec3,
    contact_radius: float,
) -> Optional[CollisionDamageState]:
    direction = v_unit(inward_direction)
    for damage in component.collision_damage:
        merge_radius = 0.5 * max(
            damage.contact_radius_m + contact_radius,
            COLLISION_DEFORMATION_MIN_RADIUS_M,
        )
        if v_norm(v_sub(damage.contact_point, contact_point)) > merge_radius:
            continue
        if v_dot(damage.inward_direction, direction) < 0.5:
            continue
        return damage
    return None


def deform_collision_reference(
    component: AeroComponent,
    contact_point: Vec3,
    inward_direction: Vec3,
    indentation: float,
    contact_radius: float,
    perforation_radius: float = 0.0,
    perforation_displacement: float = 0.0,
) -> None:
    ensure_deformation_reference(component)
    reference = component.deformation_reference_triangles
    if reference is None or indentation <= 0.0:
        return
    radius, amplitude = normalized_contact_deformation_parameters(
        reference,
        contact_point,
        indentation,
        contact_radius,
    )
    component.deformation_reference_triangles, _max_applied = (
        deform_triangle_mesh_at_contact(
            reference,
            contact_point,
            inward_direction,
            amplitude,
            radius,
            None,
            perforation_radius,
            perforation_displacement,
            deformation_should_preserve_thickness(
                component,
                inward_direction,
                radius,
            ),
        )
    )


def register_collision_dent(
    component: AeroComponent,
    contact_point: Vec3,
    inward_direction: Vec3,
    indentation: float,
    contact_radius: float,
    step: int,
    failure_mode: str,
    absorbed_energy_j: float = 0.0,
) -> Optional[CollisionDamageState]:
    if indentation <= 1e-12:
        return None
    direction = v_unit(inward_direction)
    radius = max(contact_radius, COLLISION_DEFORMATION_MIN_RADIUS_M)
    plastic_fraction = collision_plastic_fraction(component, indentation, radius)
    permanent_depth = indentation * plastic_fraction
    damage = _matching_collision_damage(
        component,
        contact_point,
        direction,
        radius,
    )
    reference_increment = indentation
    if damage is None:
        damage = CollisionDamageState(
            contact_point=contact_point,
            inward_direction=direction,
            contact_radius_m=radius,
            current_depth_m=indentation,
            permanent_depth_m=permanent_depth,
            response_time_s=collision_damage_response_time(component, radius),
            failure_mode=failure_mode,
            accumulated_energy_j=absorbed_energy_j,
            created_step=step,
        )
        component.collision_damage.append(damage)
    else:
        reference_increment = max(0.0, indentation - damage.current_depth_m)
        damage.contact_radius_m = max(damage.contact_radius_m, radius)
        damage.current_depth_m = max(damage.current_depth_m, indentation)
        damage.permanent_depth_m = max(
            damage.permanent_depth_m,
            permanent_depth,
        )
        damage.accumulated_energy_j += absorbed_energy_j
        damage.failure_mode = failure_mode
        damage.created_step = step
    deform_collision_reference(
        component,
        contact_point,
        direction,
        reference_increment,
        radius,
    )
    if (
        COLLISION_STRUCTURAL_SOLVER == "hybrid_fem_mpm"
        and component.collision_structural_state is None
        and not component_is_thin_for_solid_fem(component)
    ):
        try:
            component.collision_structural_state = build_hybrid_fem_mpm_collision_state(
                component,
                inferred_deformation_young_modulus(component),
                inferred_deformation_poisson_ratio(component),
                material_yield_strength_pa(component),
                material_failure_strain(component),
                COLLISION_SHELL_CFL,
                COLLISION_FEM_MAX_SUBSTEPS,
            )
        except ValueError:
            # Non-star-shaped or open CAD meshes retain the proven surface
            # deformation path instead of creating invalid tetrahedra.
            component.collision_structural_state = None
    fem_state = component.collision_structural_state
    if isinstance(fem_state, HybridFEMMPMCollisionState):
        apply_fem_impact_energy(
            fem_state,
            contact_point,
            direction,
            radius,
            absorbed_energy_j,
            COLLISION_SHELL_IMPACT_ENERGY_FRACTION,
        )
    return damage


def register_collision_hole(
    component: AeroComponent,
    contact_point: Vec3,
    inward_direction: Vec3,
    target_hole_radius: float,
    contact_radius: float,
    step: int,
    failure_mode: str,
    absorbed_energy_j: float,
) -> CollisionDamageState:
    if component.collision_family is None:
        component.collision_family = component.patch
    direction = v_unit(inward_direction)
    radius = max(contact_radius, COLLISION_DEFORMATION_MIN_RADIUS_M)
    damage = _matching_collision_damage(
        component,
        contact_point,
        direction,
        radius,
    )
    physical_response_time = target_hole_radius / max(
        0.4 * material_longitudinal_wave_speed(component),
        1e-9,
    )
    response_time = max(
        physical_response_time,
        COLLISION_DAMAGE_MIN_RESPONSE_TIME_S,
        1e-9,
    )
    if damage is None:
        damage = CollisionDamageState(
            contact_point=contact_point,
            inward_direction=direction,
            contact_radius_m=radius,
            current_depth_m=0.0,
            permanent_depth_m=0.0,
            target_hole_radius_m=target_hole_radius,
            response_time_s=response_time,
            failure_mode=failure_mode,
            accumulated_energy_j=absorbed_energy_j,
            ongoing_contact_energy_j=(
                absorbed_energy_j * COLLISION_SHELL_ONGOING_ENERGY_FRACTION
            ),
            created_step=step,
        )
        component.collision_damage.append(damage)
    else:
        damage.target_hole_radius_m = max(
            damage.target_hole_radius_m,
            target_hole_radius,
        )
        damage.response_time_s = min(damage.response_time_s, response_time)
        damage.failure_mode = failure_mode
        damage.accumulated_energy_j += absorbed_energy_j
        damage.ongoing_contact_energy_j += (
            absorbed_energy_j * COLLISION_SHELL_ONGOING_ENERGY_FRACTION
        )
        damage.created_step = step
    previous_hole_radius = damage.current_hole_radius_m
    if damage.target_hole_radius_m > 0.0:
        initial_radius_limit = (
            COLLISION_HOLE_INITIAL_RADIUS_FRACTION
            * damage.target_hole_radius_m
        )
        # Once the projectile is released into post-perforation ballistic
        # motion, the failed topology must already clear its projected width.
        # The former 65% growth seed was smaller than a 5 mm projectile and a
        # coarse detached plug could therefore visually cap the opening even
        # while the projectile passed through it. The extra tear allowance
        # still evolves over subsequent damage steps.
        projectile_clearance_radius = (
            damage.target_hole_radius_m
            / THIN_SHELL_HOLE_CLEARANCE_FACTOR
        )
        visible_initial_radius = max(
            initial_radius_limit,
            projectile_clearance_radius,
        )
        damage.current_hole_radius_m = min(
            damage.target_hole_radius_m,
            max(
                damage.current_hole_radius_m,
                visible_initial_radius,
            ),
        )
    solver_supports_shell = COLLISION_STRUCTURAL_SOLVER in {
        "explicit_shell",
        "shell",
        "dynamic_shell",
        "mpm",
        "hybrid_shell",
        "hybrid_fem_mpm",
    }
    if solver_supports_shell and component.collision_structural_state is None:
        build_args = (
            component,
            contact_point,
            direction,
            radius,
            inferred_deformation_young_modulus(component),
            inferred_deformation_thickness(component),
            inferred_deformation_poisson_ratio(component),
            material_yield_strength_pa(component),
            material_failure_strain(component),
            COLLISION_SHELL_DAMPING_RATIO,
            COLLISION_SHELL_CFL,
            COLLISION_SHELL_MAX_SUBSTEPS,
            collision_shell_displacement_limit(component, target_hole_radius, radius),
            damage.target_hole_radius_m,
        )
        if (
            COLLISION_STRUCTURAL_SOLVER == "hybrid_fem_mpm"
            and not component_is_thin_for_solid_fem(component)
        ):
            try:
                state = build_hybrid_fem_mpm_collision_state(
                    component,
                    inferred_deformation_young_modulus(component),
                    inferred_deformation_poisson_ratio(component),
                    material_yield_strength_pa(component),
                    material_failure_strain(component),
                    COLLISION_SHELL_CFL,
                    COLLISION_FEM_MAX_SUBSTEPS,
                )
                apply_fem_impact_energy(
                    state,
                    contact_point,
                    direction,
                    radius,
                    absorbed_energy_j,
                    COLLISION_SHELL_IMPACT_ENERGY_FRACTION,
                )
                update_fem_perforation(
                    state,
                    contact_point,
                    direction,
                    damage.current_hole_radius_m,
                )
            except ValueError:
                state = build_hybrid_shell_collision_state(*build_args)
                apply_shell_impact_energy(
                    state.shell_state,
                    absorbed_energy_j,
                    COLLISION_SHELL_IMPACT_ENERGY_FRACTION,
                )
        elif COLLISION_STRUCTURAL_SOLVER in {"hybrid_shell", "hybrid_fem_mpm"}:
            state = build_hybrid_shell_collision_state(*build_args)
            apply_shell_impact_energy(
                state.shell_state,
                absorbed_energy_j,
                COLLISION_SHELL_IMPACT_ENERGY_FRACTION,
            )
        else:
            state = build_explicit_shell_state(*build_args)
            state.solver_backend = COLLISION_STRUCTURAL_SOLVER
            apply_shell_impact_energy(
                state,
                absorbed_energy_j,
                COLLISION_SHELL_IMPACT_ENERGY_FRACTION,
            )
        component.collision_structural_state = state
    structural_state = component.collision_structural_state
    fem_state = (
        structural_state
        if isinstance(structural_state, HybridFEMMPMCollisionState)
        else None
    )
    hybrid_state = (
        structural_state if isinstance(structural_state, HybridShellCollisionState) else None
    )
    shell_state = (
        hybrid_state.shell_state
        if hybrid_state is not None
        else structural_state
        if isinstance(structural_state, ExplicitShellState)
        else None
    )
    if fem_state is not None and damage.current_hole_radius_m > previous_hole_radius:
        existing_fragment_count = len(fem_state.fragment_bodies)
        update_fem_perforation(
            fem_state,
            contact_point,
            direction,
            damage.current_hole_radius_m,
        )
        # Publish failed topology now so the perforation is immediately
        # visible, but do not advance physical time once per contact event.
        # The scheduled damage-evolution phase performs one stable structural
        # advance for the component after the contact batch is complete.
        commit_hybrid_fem_mpm_failure_topology(component, fem_state)
        for fragment in fem_state.fragment_bodies[existing_fragment_count:]:
            fragment.component.collision_fragment_created_step = step
    elif shell_state is not None and damage.current_hole_radius_m > previous_hole_radius:
        existing_fragment_triangles = (
            {
                fragment.component.patch: frozenset(fragment.triangle_indices)
                for fragment in hybrid_state.fragment_bodies
            }
            if hybrid_state is not None
            else {}
        )
        radius_increment = damage.current_hole_radius_m - previous_hole_radius
        update_shell_perforation(
            shell_state,
            damage.current_hole_radius_m,
            radius_increment,
            max(damage.response_time_s, 1e-9),
        )
        if ENABLE_COLLISION_TOPOLOGY_CHANGES:
            if hybrid_state is not None:
                _emitted_count, emitted_mass = sync_hybrid_shell_fragments(component, hybrid_state)
            else:
                _emitted_count, emitted_mass = emit_shell_fragments(shell_state)
            if emitted_mass > 0.0:
                remaining_mass = max(component.mass - emitted_mass, 0.0)
                inertia_scale = remaining_mass / max(component.mass, 1e-18)
                component.mass = remaining_mass
                component.inertia *= inertia_scale
            if hybrid_state is not None:
                for fragment in hybrid_state.fragment_bodies:
                    previous_triangles = existing_fragment_triangles.get(
                        fragment.component.patch
                    )
                    if previous_triangles != frozenset(fragment.triangle_indices):
                        fragment.component.collision_fragment_created_step = step
        commit_explicit_shell_topology(component, shell_state)
        # As with solid FEM/MPM, topology changes are committed immediately
        # and structural time integration is deferred to the single scheduled
        # damage-evolution solve for this component.
    return damage


def _record_structural_deformation_for_damage_sites(
    component: AeroComponent,
    deformation_m: float,
) -> None:
    """Apply one shared structural solve result to every perforation site."""
    for damage in component.collision_damage:
        if damage.target_hole_radius_m <= 0.0:
            continue
        damage.current_depth_m = max(damage.current_depth_m, deformation_m)
        damage.permanent_depth_m = max(
            damage.permanent_depth_m,
            deformation_m,
        )


def advance_collision_damage_state(
    component: AeroComponent,
    damage: CollisionDamageState,
    dt: float,
    advance_structural: bool = True,
) -> Tuple[float, int]:
    if dt <= 0.0:
        return 0.0, 0
    response_time = max(damage.response_time_s, 1e-9)
    evolution_fraction = 1.0 - math.exp(-dt / response_time)
    geometry_changed = 0.0
    removed_triangles = 0
    structural_state = component.collision_structural_state
    fem_state = (
        structural_state
        if isinstance(structural_state, HybridFEMMPMCollisionState)
        else None
    )
    hybrid_state = (
        structural_state if isinstance(structural_state, HybridShellCollisionState) else None
    )
    shell_state = (
        hybrid_state.shell_state
        if hybrid_state is not None
        else structural_state
        if isinstance(structural_state, ExplicitShellState)
        else None
    )

    elastic_depth = max(
        0.0,
        damage.current_depth_m - damage.permanent_depth_m,
    )
    recovery = elastic_depth * evolution_fraction
    if recovery > 1e-12 and fem_state is not None:
        damage.current_depth_m = max(
            damage.permanent_depth_m,
            damage.current_depth_m - recovery,
        )
    if recovery > 1e-12 and shell_state is None and fem_state is None:
        applied = deform_component_at_contact(
            component,
            damage.contact_point,
            v_mul(damage.inward_direction, -1.0),
            recovery,
            damage.contact_radius_m,
        )
        deform_collision_reference(
            component,
            damage.contact_point,
            v_mul(damage.inward_direction, -1.0),
            applied,
            damage.contact_radius_m,
        )
        damage.current_depth_m = max(
            damage.permanent_depth_m,
            damage.current_depth_m - applied,
        )
        geometry_changed = max(geometry_changed, applied)

    remaining_hole_growth = max(
        0.0,
        damage.target_hole_radius_m - damage.current_hole_radius_m,
    )
    if remaining_hole_growth <= max(1e-6, 1e-4 * damage.target_hole_radius_m):
        remaining_hole_growth = 0.0
        damage.current_hole_radius_m = damage.target_hole_radius_m
    hole_increment = remaining_hole_growth * evolution_fraction
    if hole_increment > 1e-12:
        next_hole_radius = min(
            damage.target_hole_radius_m,
            damage.current_hole_radius_m + hole_increment,
        )
        damage.current_hole_radius_m = next_hole_radius
        if fem_state is not None:
            removed_triangles += update_fem_perforation(
                fem_state,
                damage.contact_point,
                damage.inward_direction,
                next_hole_radius,
            )
        elif shell_state is not None:
            update_shell_perforation(
                shell_state,
                next_hole_radius,
                hole_increment,
                dt,
            )
            if hybrid_state is not None and ENABLE_COLLISION_TOPOLOGY_CHANGES:
                emitted_count, emitted_mass = sync_hybrid_shell_fragments(component, hybrid_state)
                removed_triangles += emitted_count
                if emitted_mass > 0.0:
                    remaining_mass = max(component.mass - emitted_mass, 0.0)
                    inertia_scale = remaining_mass / max(component.mass, 1e-18)
                    component.mass = remaining_mass
                    component.inertia *= inertia_scale
        elif ENABLE_COLLISION_TOPOLOGY_CHANGES:
            # Legacy surface-only path for collision damage states that do not
            # own an explicit shell.  It still cuts an actual hole and exports
            # displaced material instead of retaining a visual plug.
            rim_increment = min(2.0 * hole_increment, MAX_TOTAL_DEFORMATION)
            rim_deformation, removed_triangles = fracture_thin_shell(
                component,
                damage.contact_point,
                damage.inward_direction,
                next_hole_radius,
                rim_displacement_m=rim_increment,
            )
            damage.current_depth_m = max(
                damage.current_depth_m,
                rim_deformation,
            )
            damage.permanent_depth_m = max(
                damage.permanent_depth_m,
                rim_deformation,
            )
            geometry_changed = max(geometry_changed, rim_deformation, hole_increment)
            component.deformation_reference_triangles = list(component.triangles)
        else:
            rim_deformation = deform_component_at_contact(
                component,
                damage.contact_point,
                damage.inward_direction,
                min(2.0 * hole_increment, COLLISION_MAX_CONTACT_DEFORMATION),
                max(damage.contact_radius_m, next_hole_radius),
                perforation_radius=next_hole_radius,
                perforation_displacement=max(
                    hole_increment,
                    inferred_deformation_thickness(component)
                    * evolution_fraction,
                ),
            )
            damage.current_depth_m = max(damage.current_depth_m, rim_deformation)
            damage.permanent_depth_m = max(damage.permanent_depth_m, rim_deformation)
            geometry_changed = max(geometry_changed, rim_deformation)
            deform_collision_reference(
                component,
                damage.contact_point,
                damage.inward_direction,
                rim_deformation,
                max(damage.contact_radius_m, next_hole_radius),
                perforation_radius=next_hole_radius,
                perforation_displacement=max(
                    hole_increment,
                    inferred_deformation_thickness(component)
                    * evolution_fraction,
                ),
            )

    if fem_state is not None:
        if damage.ongoing_contact_energy_j > 1e-12:
            drive_duration = max(
                COLLISION_SHELL_ONGOING_RESPONSE_TIMES * response_time,
                dt,
                1e-9,
            )
            drive_fraction = min(1.0, dt / drive_duration)
            drive_work = damage.ongoing_contact_energy_j * drive_fraction
            damage.ongoing_contact_energy_j = max(
                0.0,
                damage.ongoing_contact_energy_j - drive_work,
            )
            apply_fem_impact_energy(
                fem_state,
                damage.contact_point,
                damage.inward_direction,
                max(damage.contact_radius_m, damage.current_hole_radius_m),
                drive_work,
                1.0,
            )
        if advance_structural:
            structural_dt = min(
                dt,
                max(MOTION_DT, response_time),
            )
            deformation, emitted_count, _emitted_mass = (
                advance_hybrid_fem_mpm_collision(
                    component,
                    fem_state,
                    structural_dt,
                )
            )
            removed_triangles += emitted_count
            geometry_changed = max(geometry_changed, deformation)
            _record_structural_deformation_for_damage_sites(
                component,
                deformation,
            )
    elif shell_state is not None:
        if damage.ongoing_contact_energy_j > 1e-12:
            drive_duration = max(
                COLLISION_SHELL_ONGOING_RESPONSE_TIMES * response_time,
                dt,
                1e-9,
            )
            drive_fraction = min(1.0, dt / drive_duration)
            drive_work = damage.ongoing_contact_energy_j * drive_fraction
            damage.ongoing_contact_energy_j = max(
                0.0,
                damage.ongoing_contact_energy_j - drive_work,
            )
            driven_displacement = apply_shell_contact_work(
                shell_state,
                damage.contact_point,
                damage.inward_direction,
                max(damage.contact_radius_m, damage.current_hole_radius_m),
                drive_work,
                dt,
            )
            geometry_changed = max(geometry_changed, driven_displacement)
        if ENABLE_COLLISION_TOPOLOGY_CHANGES:
            if hybrid_state is not None:
                emitted_count, emitted_mass = sync_hybrid_shell_fragments(component, hybrid_state)
            else:
                emitted_count, emitted_mass = emit_shell_fragments(shell_state)
            removed_triangles += emitted_count
            if emitted_mass > 0.0:
                remaining_mass = max(component.mass - emitted_mass, 0.0)
                inertia_scale = remaining_mass / max(component.mass, 1e-18)
                component.mass = remaining_mass
                component.inertia *= inertia_scale
        if advance_structural:
            original_cofr = component.cofr
            shell_deformation = (
                advance_hybrid_shell_collision(component, hybrid_state, dt)
                if hybrid_state is not None
                else advance_explicit_shell(component, shell_state, dt)
            )
            component.aref, component.lref, _geometry_centroid = component_references(
                component.triangles
            )
            component.cofr = original_cofr
            _record_structural_deformation_for_damage_sites(
                component,
                shell_deformation,
            )
            geometry_changed = max(geometry_changed, shell_deformation)

    damage.elapsed_s += dt
    # The second value is a transfer count: faces remain in the shell state
    # and are exported as detached fragments rather than being destroyed.
    return geometry_changed, removed_triangles


def write_collision_damage_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Persistent collision damage evolution log\n"
        "step\tpatch\tsite\tfailure_mode\telapsed_s\tresponse_time_s\t"
        "current_depth_m\tpermanent_depth_m\tcurrent_hole_radius_m\t"
        "target_hole_radius_m\tcontact_radius_m\taccumulated_energy_J\t"
        "ongoing_contact_energy_J\tgeometry_change_m\tdisplaced_fragments\tactive\tstructural_solver\t"
        "shell_stable_dt_s\tshell_mass_scale\tshell_displacement_limit_m\t"
        "shell_failed_edges\tshell_plug_faces\n"
    )


def write_collision_conservation_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Hybrid FEM/MPM conservation audit\n"
        "step\tpatch\tmass_before_kg\tmass_after_kg\tmass_error_kg\t"
        "momentum_error_x_ns\tmomentum_error_y_ns\tmomentum_error_z_ns\t"
        "angular_error_x_nms\tangular_error_y_nms\tangular_error_z_nms\t"
        "kinetic_before_j\tkinetic_after_j\tstrain_energy_j\t"
        "plastic_dissipation_j\texternal_work_j\t"
        "momentum_projection_ns\tangular_momentum_projection_nms\n"
    )


def append_collision_conservation_audits(
    components: Sequence[AeroComponent],
    step: int,
    path: Path,
) -> int:
    written = 0
    with path.open("a") as stream:
        for component in components:
            state = component.collision_structural_state
            if not isinstance(state, HybridFEMMPMCollisionState):
                continue
            audit = state.last_audit
            if audit is None:
                continue
            momentum_error = audit.momentum_error
            angular_error = audit.angular_momentum_error
            stream.write(
                f"{step}\t{component.patch}\t{audit.mass_before_kg:.12g}\t"
                f"{audit.mass_after_kg:.12g}\t{audit.mass_error_kg:.12g}\t"
                f"{momentum_error[0]:.12g}\t{momentum_error[1]:.12g}\t"
                f"{momentum_error[2]:.12g}\t{angular_error[0]:.12g}\t"
                f"{angular_error[1]:.12g}\t{angular_error[2]:.12g}\t"
                f"{audit.kinetic_before_j:.12g}\t{audit.kinetic_after_j:.12g}\t"
                f"{audit.strain_energy_j:.12g}\t{audit.plastic_dissipation_j:.12g}\t"
                f"{audit.external_work_j:.12g}\t"
                f"{audit.momentum_projection_ns:.12g}\t"
                f"{audit.angular_momentum_projection_nms:.12g}\n"
            )
            written += 1
    return written


def evolve_collision_damage(
    components: Sequence[AeroComponent],
    step: int,
    dt: float,
    log_path: Path,
) -> int:
    changed_sites = 0
    if not any(component.collision_damage for component in components):
        return changed_sites
    with log_path.open("a") as stream:
        for component in components:
            eligible_site_indices = [
                site_index
                for site_index, damage in enumerate(component.collision_damage)
                if step > damage.created_step
            ]
            final_eligible_site = (
                eligible_site_indices[-1] if eligible_site_indices else None
            )
            component_rows: List[
                Tuple[int, CollisionDamageState, float, int]
            ] = []
            for site_index, damage in enumerate(component.collision_damage):
                geometry_change = 0.0
                removed_triangles = 0
                if step > damage.created_step:
                    geometry_change, removed_triangles = (
                        advance_collision_damage_state(
                            component,
                            damage,
                            dt,
                            advance_structural=(
                                site_index == final_eligible_site
                            ),
                        )
                    )
                if geometry_change > 1e-12 or removed_triangles:
                    changed_sites += 1
                component_rows.append(
                    (site_index, damage, geometry_change, removed_triangles)
                )

            structural_state = component.collision_structural_state
            shell_state = (
                structural_state.shell_state
                if isinstance(structural_state, HybridShellCollisionState)
                else structural_state
                if isinstance(structural_state, ExplicitShellState)
                else None
            )
            fem_state = (
                structural_state
                if isinstance(structural_state, HybridFEMMPMCollisionState)
                else None
            )
            for site_index, damage, geometry_change, removed_triangles in component_rows:
                active = (
                    damage.current_depth_m
                    > damage.permanent_depth_m + 1e-9
                    or damage.current_hole_radius_m
                    < damage.target_hole_radius_m - 1e-9
                    or damage.ongoing_contact_energy_j > 1e-12
                )
                stream.write(
                    f"{step}\t{component.patch}\t{site_index}\t{damage.failure_mode}\t"
                    f"{damage.elapsed_s:.8g}\t{damage.response_time_s:.8g}\t"
                    f"{damage.current_depth_m:.8g}\t{damage.permanent_depth_m:.8g}\t"
                    f"{damage.current_hole_radius_m:.8g}\t"
                    f"{damage.target_hole_radius_m:.8g}\t"
                    f"{damage.contact_radius_m:.8g}\t"
                    f"{damage.accumulated_energy_j:.8g}\t"
                    f"{damage.ongoing_contact_energy_j:.8g}\t"
                    f"{geometry_change:.8g}\t{removed_triangles}\t{int(active)}\t"
                    f"{'hybrid_fem_mpm' if fem_state is not None else 'explicit_shell' if shell_state is not None else 'surface_contact'}\t"
                    f"{shell_state.stable_dt_s if shell_state is not None else 0.0:.8g}\t"
                    f"{shell_state.mass_scale if shell_state is not None else 1.0:.8g}\t"
                    f"{shell_state.displacement_limit_m if shell_state is not None else 0.0:.8g}\t"
                    f"{shell_state.failed_edges if shell_state is not None else 0}\t"
                    f"{len(shell_state.plug_triangles) if shell_state is not None else 0}\n"
                )
    return changed_sites


def component_collision_swept_bounds(
    component: AeroComponent,
    dt_s: float,
) -> TriangleBounds:
    """Return a conservative AABB for one body's complete rigid path.

    The current mesh is the end-of-step pose. Translation expands its bounds
    back to the inferred start pose. Rotation expands all axes by the maximum
    chord travelled by any point in the component AABB. The result can contain
    false positives, but it cannot exclude a contact considered by the linear
    or angular continuous-collision checks.
    """
    current = component_bounds(component.triangles)
    dt = max(dt_s, 0.0)
    translation = v_mul(component.linear_velocity, dt)
    origin = infer_motion_origin(component)
    radius = math.sqrt(
        sum(
            max(
                abs(current[2 * axis] - origin[axis]),
                abs(current[2 * axis + 1] - origin[axis]),
            )
            ** 2
            for axis in range(3)
        )
    )
    rotation_angle = v_norm(component.angular_velocity) * dt
    rotational_padding = min(rotation_angle * radius, 2.0 * radius)
    margin = max(
        COLLISION_MARGIN_M,
        COLLISION_MANIFOLD_TOLERANCE_M,
        0.0,
    )
    expanded: List[float] = []
    for axis in range(3):
        current_min = current[2 * axis]
        current_max = current[2 * axis + 1]
        start_min = current_min - translation[axis]
        start_max = current_max - translation[axis]
        expanded.extend(
            [
                min(current_min, start_min) - rotational_padding - margin,
                max(current_max, start_max) + rotational_padding + margin,
            ]
        )
    return (
        expanded[0],
        expanded[1],
        expanded[2],
        expanded[3],
        expanded[4],
        expanded[5],
    )


def collision_broad_phase_pairs(
    components: Sequence[AeroComponent],
    dt_s: float,
) -> List[Tuple[int, int]]:
    """Return conservative body-pair candidates using sweep-and-prune.

    This classic broad phase replaces the quadratic loop of expensive swept
    triangle tests. It only removes pairs whose complete conservative swept
    AABBs are disjoint; all candidate pairs still use the existing exact mesh
    intersection and continuous-collision routines.
    """
    indexed_bounds = [
        (index, component_collision_swept_bounds(component, dt_s))
        for index, component in enumerate(components)
        if component.triangles
    ]
    if len(indexed_bounds) < 2:
        return []

    # Select the axis with the lowest interval occupancy. This retains the
    # deterministic sweep-and-prune algorithm while avoiding a poor fixed-axis
    # choice for a wide plate or a long projectile.
    axis_scores: List[Tuple[float, int]] = []
    for axis in range(3):
        domain_min = min(bounds[2 * axis] for _index, bounds in indexed_bounds)
        domain_max = max(bounds[2 * axis + 1] for _index, bounds in indexed_bounds)
        domain_span = max(domain_max - domain_min, 1e-18)
        occupancy = sum(
            bounds[2 * axis + 1] - bounds[2 * axis]
            for _index, bounds in indexed_bounds
        ) / domain_span
        axis_scores.append((occupancy, axis))
    sweep_axis = min(axis_scores)[1]
    entries = sorted(
        indexed_bounds,
        key=lambda item: (item[1][2 * sweep_axis], item[0]),
    )

    active: List[Tuple[int, TriangleBounds]] = []
    candidates: List[Tuple[int, int]] = []
    for index, bounds in entries:
        interval_min = bounds[2 * sweep_axis]
        active = [
            item
            for item in active
            if item[1][2 * sweep_axis + 1] >= interval_min
        ]
        for other_index, other_bounds in active:
            if not triangle_bounds_overlap(bounds, other_bounds):
                continue
            candidates.append(
                (min(index, other_index), max(index, other_index))
            )
        active.append((index, bounds))
    candidates.sort()
    return candidates


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


def thin_component_axis_normal(component: AeroComponent, direction: Vec3) -> Vec3:
    """Return the axis-aligned sheet normal closest to ``direction``."""
    bounds = component_bounds(component.triangles)
    extents = (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    axis = min(range(3), key=lambda index: extents[index])
    sign = 1.0 if direction[axis] >= 0.0 else -1.0
    return tuple(sign if index == axis else 0.0 for index in range(3))


def component_has_thin_axis(component: AeroComponent) -> bool:
    bounds = component_bounds(component.triangles)
    extents = sorted(
        (
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        )
    )
    return extents[0] < 0.1 * max(extents[2], 1e-9)


def dominant_planar_surface_normal(
    component: AeroComponent,
    direction_hint: Vec3,
) -> Optional[Vec3]:
    """Return a stable broad-face normal for a plate-like closed surface.

    Opposite faces of a closed plate have cancelling signed normals.  The
    classic area-weighted normal structure tensor, sum(A n n^T), avoids that
    cancellation.  Its dominant eigenvector identifies the plate normal even
    when the CAD body is rotated relative to the world axes.  Reference
    triangles are preferred so local dents do not flip the collision normal.
    """
    triangles = component.deformation_reference_triangles or component.triangles
    tensor = [[0.0, 0.0, 0.0] for _ in range(3)]
    trace = 0.0
    seed = v_unit(direction_hint, (1.0, 0.0, 0.0))
    largest_area = 0.0
    for triangle in triangles:
        area, _centroid, normal = triangle_area_centroid_normal(triangle)
        if area <= 1e-16:
            continue
        unit_normal = v_unit(normal)
        trace += area
        if area > largest_area:
            largest_area = area
            seed = unit_normal
        for row in range(3):
            for column in range(3):
                tensor[row][column] += (
                    area * unit_normal[row] * unit_normal[column]
                )
    if trace <= 1e-16:
        return None

    vector = seed
    for _iteration in range(12):
        multiplied = tuple(
            sum(tensor[row][column] * vector[column] for column in range(3))
            for row in range(3)
        )
        if v_norm(multiplied) <= 1e-16:
            return None
        vector = v_unit(multiplied)
    eigenvalue = v_dot(
        vector,
        tuple(
            sum(tensor[row][column] * vector[column] for column in range(3))
            for row in range(3)
        ),
    )
    # A cube has no dominant broad-face direction.  A plate does.
    if eigenvalue / trace < 0.60:
        return None
    if v_dot(vector, direction_hint) < 0.0:
        vector = v_mul(vector, -1.0)
    return vector


def stable_collision_normal(
    a: AeroComponent,
    b: AeroComponent,
    detected_normal: Vec3,
    preserve_detected_orientation: bool = False,
) -> Vec3:
    """Use reference plate geometry to stabilise a triangle contact normal."""
    center_offset = v_sub(
        component_center_from_bounds(a),
        component_center_from_bounds(b),
    )
    candidates: List[Tuple[int, float, Vec3]] = []
    for component in (a, b):
        candidate = dominant_planar_surface_normal(component, detected_normal)
        if candidate is None:
            continue
        candidates.append(
            (
                1 if component.is_assembly_anchor else 0,
                component.aref,
                candidate,
            )
        )
    if not candidates:
        return detected_normal
    _anchor_priority, _area, normal = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    if (
        not preserve_detected_orientation
        and v_norm(center_offset) > 1e-12
        and v_dot(normal, center_offset) < 0.0
    ):
        normal = v_mul(normal, -1.0)
    return normal


def component_separation_distance_along_normal(
    a: AeroComponent,
    b: AeroComponent,
    normal: Vec3,
) -> float:
    """Return how far ``a`` must move along ``normal`` to separate from ``b``.

    This is the one-axis part of the classic separating-axis test.  It is used
    after a triangle/triangle surface hit has supplied the physical contact
    normal.  This is deliberately the distance to separation, not merely the
    width of the intersecting interval.  Those differ when one interval is
    contained in the other, which is common for a small body inside a plate's
    broad y/z extent.
    """
    axis = v_unit(normal)
    if v_norm(axis) <= 1e-12:
        return 0.0
    a_projections = [
        v_dot(point, axis)
        for triangle in a.triangles
        for point in triangle[1:]
    ]
    b_projections = [
        v_dot(point, axis)
        for triangle in b.triangles
        for point in triangle[1:]
    ]
    if not a_projections or not b_projections:
        return 0.0
    return max(0.0, max(b_projections) - min(a_projections))


def aabb_minimum_separation(
    a: AeroComponent,
    b: AeroComponent,
) -> Tuple[float, Vec3]:
    """Return the shortest axis-aligned translation that separates ``a``.

    Unlike an overlap-width calculation, this remains correct when one body's
    projection is contained inside the other's projection.
    """
    a_bounds = component_bounds(a.triangles)
    b_bounds = component_bounds(b.triangles)
    a_center = component_center_from_bounds(a)
    b_center = component_center_from_bounds(b)
    candidates: List[Tuple[float, Vec3]] = []
    for axis in range(3):
        if a_center[axis] >= b_center[axis]:
            depth = b_bounds[2 * axis + 1] - a_bounds[2 * axis]
            normal = tuple(1.0 if index == axis else 0.0 for index in range(3))
        else:
            depth = a_bounds[2 * axis + 1] - b_bounds[2 * axis]
            normal = tuple(-1.0 if index == axis else 0.0 for index in range(3))
        candidates.append((max(0.0, depth), normal))
    return min(candidates, key=lambda item: item[0])


def collision_pair_key(a: AeroComponent, b: AeroComponent) -> Tuple[int, int]:
    first, second = sorted((id(a), id(b)))
    return first, second


def persistent_contact_for_pair(
    a: AeroComponent,
    b: AeroComponent,
    step: int,
    normal: Vec3,
    point: Vec3,
) -> Tuple[PersistentContactState, bool]:
    """Return a cross-timestep contact record and whether impact is new."""
    key = f"{id(b)}:{b.patch}"
    previous = a.persistent_contacts.get(key)
    new_contact = (
        previous is None
        or step > previous.last_step + 1
        or v_dot(previous.normal, normal) < 0.5
    )
    if new_contact:
        contact_state = PersistentContactState(
            normal=normal,
            point=point,
            last_step=step,
        )
        a.persistent_contacts[key] = contact_state
        return contact_state, True
    previous.normal = v_unit(v_add(previous.normal, normal), normal)
    previous.point = point
    if previous.last_step != step:
        previous.age_steps += 1
    previous.last_step = step
    return previous, False


@dataclass(frozen=True)
class InitialCadOverlap:
    """Stress-free overlap recorded from the imported assembly pose."""

    depth_m: float
    surfaces_intersect: bool


def initial_same_source_overlap_pairs(
    components: Sequence[AeroComponent],
) -> Dict[Tuple[int, int], InitialCadOverlap]:
    """Record intentional CAD fits as a stress-free initial configuration.

    Separate occurrences in one imported source can have nested or overlapping
    bounding boxes even when their triangle surfaces merely meet.  An AABB is a
    broad phase, so its initial overlap must not be interpreted as stored elastic
    penetration.  Only overlap beyond the initial depth is treated as new
    penetration.  Once a pair separates, its offset is removed and later
    re-contact is solved normally.
    """
    pairs: Dict[Tuple[int, int], InitialCadOverlap] = {}
    for index, a in enumerate(components):
        for b in components[index + 1:]:
            if not components_share_collision_source(a, b):
                continue
            initial_overlap = aabb_overlap_with_normal(a, b)
            if initial_overlap is not None:
                pairs[collision_pair_key(a, b)] = InitialCadOverlap(
                    depth_m=initial_overlap[0],
                    surfaces_intersect=(
                        triangle_mesh_intersection_contact(
                            a.triangles,
                            b.triangles,
                        )
                        is not None
                    ),
                )
    return pairs


def nearest_component_surface_point(component: AeroComponent, point: Vec3) -> Tuple[float, Vec3]:
    nearest_point = component.cofr
    nearest_distance = math.inf
    for triangle in component.triangles:
        surface_point = closest_point_on_triangle(point, triangle)
        distance = v_norm(v_sub(surface_point, point))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_point = surface_point
    return nearest_distance, nearest_point


def apply_nearby_collision_effects(
    components: Sequence[AeroComponent],
    directly_involved: Sequence[AeroComponent],
    contact_point: Vec3,
    normal: Vec3,
    contact_radius: float,
    energy_j: float,
    step: int,
    failure_mode: str,
) -> int:
    """Apply a local attenuated impact response to nearby non-contact parts.

    This is a deliberately simple classical shock/contact falloff.  It is not a
    wave solver, but it prevents nearby lightweight parts from remaining
    perfectly unaffected when a high-energy collision occurs right beside them.
    """
    if energy_j <= 1e-12 or contact_radius <= 1e-12:
        return 0
    direct_ids = {id(component) for component in directly_involved}
    influence_radius = max(6.0 * contact_radius, 0.02)
    candidates: List[Tuple[AeroComponent, Vec3, float]] = []
    for component in components:
        if id(component) in direct_ids or not component.triangles:
            continue
        distance, surface_point = nearest_component_surface_point(
            component,
            contact_point,
        )
        if distance > influence_radius:
            continue
        attenuation = max(0.0, 1.0 - distance / influence_radius)
        if attenuation > 1e-12:
            candidates.append((component, surface_point, attenuation))
    weight_sum = sum(attenuation * attenuation for _, _, attenuation in candidates)
    if weight_sum <= 1e-12:
        return 0

    transferable_energy = energy_j * COLLISION_NEARBY_ENERGY_FRACTION
    affected = 0
    transferred_impulse = (0.0, 0.0, 0.0)
    for component, surface_point, attenuation in candidates:
        direction = v_sub(surface_point, contact_point)
        if v_norm(direction) <= 1e-12:
            direction = normal
        direction = v_unit(direction)
        local_energy = (
            transferable_energy
            * attenuation
            * attenuation
            / weight_sum
        )
        local_radius = max(contact_radius * (0.5 + attenuation), COLLISION_DEFORMATION_MIN_RADIUS_M)
        effective_mass = max(component.mass, 1e-9)
        local_speed = math.sqrt(2.0 * local_energy / effective_mass)
        if component_has_translation_freedom(component):
            impulse = v_mul(direction, effective_mass * local_speed)
            apply_collision_impulse(component, impulse, surface_point)
            transferred_impulse = v_add(transferred_impulse, impulse)
        effective_modulus = 1.0 / max(component_contact_compliance(component), 1e-30)
        contact_stiffness = 2.0 * effective_modulus * local_radius
        indentation = min(
            math.sqrt(2.0 * local_energy / max(contact_stiffness, 1e-30)),
            COLLISION_MAX_CONTACT_DEFORMATION * attenuation,
            0.1 * max(component.lref, 1e-6),
        )
        deformation = deform_component_at_contact(
            component,
            surface_point,
            direction,
            indentation,
            local_radius,
        )
        if deformation > 1e-12:
            register_collision_dent(
                component,
                surface_point,
                direction,
                deformation,
                local_radius,
                step,
                f"nearby_{failure_mode}",
                local_energy,
            )
            affected += 1
    reaction_components = [
        component
        for component in directly_involved
        if component_has_translation_freedom(component)
    ]
    reaction_mass = sum(component.mass for component in reaction_components)
    if reaction_mass > 1e-12 and v_norm(transferred_impulse) > 1e-12:
        for component in reaction_components:
            share = component.mass / reaction_mass
            apply_collision_impulse(
                component,
                v_mul(transferred_impulse, -share),
                contact_point,
            )
    return affected


def rewind_component_rotational_sweep(
    component: AeroComponent,
    translation: Vec3,
    rotation: Vec3,
    time_fraction: float,
) -> Vec3:
    """Return a body from its end pose to an angular CCD time of impact."""
    if component.is_assembly_anchor:
        return (0.0, 0.0, 0.0)
    remaining = max(0.0, min(1.0, 1.0 - time_fraction))
    reverse_translation = v_mul(translation, -remaining)
    reverse_rotation = v_mul(rotation, -remaining)
    reverse_angle = v_norm(reverse_rotation)
    move_component_rigidly(
        component,
        reverse_translation,
        v_unit(reverse_rotation) if reverse_angle > 1e-12 else None,
        reverse_angle,
        infer_motion_origin(component),
    )
    component.total_translation = v_add(
        component.total_translation,
        reverse_translation,
    )
    component.total_rotation = v_add(
        component.total_rotation,
        reverse_rotation,
    )
    return reverse_translation


def relative_swept_perforation_contact(
    a: AeroComponent,
    b: AeroComponent,
    contact: RelativeSweptContact,
    dt_s: float,
) -> Optional[SweptCollisionContact]:
    """Promote any later high-energy swept impact to thin-shell failure.

    Perforation is a material/energy outcome, not a privilege of the initially
    prescribed pair. This is especially important for a multi-body projectile:
    a leading lightweight piece can make first contact before a heavier trailing
    body reaches the target.
    """
    candidates = [
        (a, b, contact.normal),
        (b, a, v_mul(contact.normal, -1.0)),
    ]
    candidates.sort(key=lambda candidate: candidate[1].is_assembly_anchor, reverse=True)
    for impactor, target, target_to_impactor_normal in candidates:
        relative_velocity = v_sub(
            contact_point_velocity(impactor, contact.point),
            contact_point_velocity(target, contact.point),
        )
        normal_speed = max(
            0.0,
            -v_dot(relative_velocity, target_to_impactor_normal),
        )
        if normal_speed <= 1e-9:
            continue
        response = thin_shell_impact_response(
            impactor,
            target,
            normal_speed,
            contact.point,
            target_to_impactor_normal,
        )
        if response is None or not response.perforated:
            continue
        approach_axis = v_unit(
            relative_velocity,
            v_mul(target_to_impactor_normal, -1.0),
        )
        penetration = max(
            response.indentation,
            inferred_deformation_thickness(target),
            2.0 * COLLISION_MIN_OVERLAP_M,
        )
        contact_geometry = local_contact_geometry(
            impactor,
            target,
            contact.point,
            contact.point,
            target_to_impactor_normal,
        )
        return SweptCollisionContact(
            moving=impactor,
            stationary=target,
            depth=max(
                penetration
                * max(0.0, -v_dot(approach_axis, target_to_impactor_normal)),
                2.0 * COLLISION_MIN_OVERLAP_M,
            ),
            normal=target_to_impactor_normal,
            point=contact.point,
            travel_to_contact=contact.travel_to_contact_m,
            approach_axis=approach_axis,
            approach_penetration=penetration,
            manifold_points=contact.manifold_points,
            perforated=True,
            residual_speed=response.residual_speed,
            absorbed_energy_j=response.absorbed_energy_j,
            failure_mode=response.failure_mode,
            hole_radius=response.hole_radius,
            contact_geometry=contact_geometry,
            post_contact_time_s=max(
                0.0,
                dt_s * (1.0 - contact.time_fraction),
            ),
        )
    return None


def enforce_environment_contact_constraints(
    active_components: Sequence[AeroComponent],
    step: int,
    initial_overlap_pairs: Optional[
        Dict[Tuple[int, int], InitialCadOverlap]
    ] = None,
) -> List[str]:
    """Project independent bodies out of surrounding solid geometry.

    Impact CCD resolves momentum and deformation. This position-level,
    frictionless Signorini constraint handles the complementary case: resting
    contact or penetration left by another pair's correction. It uses a classic
    sequential projection/impulse iteration and never creates collision damage.
    """
    lines: List[str] = []
    margin_depth = 2.0 * max(COLLISION_MARGIN_M, 0.0)
    for constraint_pass in range(COLLISION_MAX_PASSES):
        corrected = False
        candidate_pairs = collision_broad_phase_pairs(active_components, 0.0)
        for index_a, index_b in candidate_pairs:
            a = active_components[index_a]
            b = active_components[index_b]
            if (
                a.rigid_body_group is not None
                and a.rigid_body_group == b.rigid_body_group
            ):
                continue
            pair_key = collision_pair_key(a, b)
            overlap = aabb_overlap_with_normal(a, b)
            if overlap is None:
                if initial_overlap_pairs is not None:
                    initial_overlap_pairs.pop(pair_key, None)
                continue
            depth, normal, _broad_contact = overlap
            if initial_overlap_pairs is not None and pair_key in initial_overlap_pairs:
                initial_depth = initial_overlap_pairs[pair_key].depth_m
                if depth <= initial_depth + COLLISION_MIN_OVERLAP_M:
                    continue
                depth = max(0.0, depth - initial_depth)
            physical_depth = max(0.0, depth - margin_depth)
            if physical_depth <= COLLISION_MIN_OVERLAP_M:
                continue
            bounds_a = component_bounds(a.triangles)
            bounds_b = component_bounds(b.triangles)
            a_contains_b = all(
                bounds_a[2 * axis] <= bounds_b[2 * axis]
                and bounds_a[2 * axis + 1] >= bounds_b[2 * axis + 1]
                for axis in range(3)
            )
            b_contains_a = all(
                bounds_b[2 * axis] <= bounds_a[2 * axis]
                and bounds_b[2 * axis + 1] >= bounds_a[2 * axis + 1]
                for axis in range(3)
            )
            if a_contains_b or b_contains_a:
                # Containment is ambiguous for surface CAD: the outer mesh
                # may be a housing or cavity. Only a crossing boundary is a
                # reliable solid-contact constraint without volume topology.
                continue
            surface_hit = triangle_mesh_intersection_contact(
                a.triangles,
                b.triangles,
            )
            if surface_hit is None:
                # AABB overlap alone is not penetration for concave or
                # hollow surface meshes.
                continue
            surface_contact, surface_normal = surface_hit
            center_offset = v_sub(
                component_center_from_bounds(a),
                component_center_from_bounds(b),
            )
            surface_normal = v_unit(surface_normal, normal)
            if v_dot(surface_normal, center_offset) < 0.0:
                surface_normal = v_mul(surface_normal, -1.0)
            if initial_overlap_pairs is None or pair_key not in initial_overlap_pairs:
                surface_depth = component_separation_distance_along_normal(
                    a,
                    b,
                    surface_normal,
                )
                aabb_depth, aabb_normal = aabb_minimum_separation(a, b)
                physical_depth, normal = min(
                    (surface_depth, surface_normal),
                    (aabb_depth, aabb_normal),
                    key=lambda item: item[0],
                )
                physical_depth += max(COLLISION_MARGIN_M, 0.0)
            if physical_depth <= COLLISION_MIN_OVERLAP_M:
                continue
            environment_failure_mode = "nonpenetration_constraint"
            environment_perforated = False
            environment_absorbed_energy = 0.0
            environment_residual_speed = 0.0
            environment_contact_radius = 0.0
            environment_deform_a = 0.0
            environment_deform_b = 0.0
            environment_indent_a = 0.0
            environment_indent_b = 0.0
            environment_logged_manifold_points = 1
            environment_friction = 0.0
            environment_dynamic_impact = False
            relative_velocity = v_sub(
                contact_point_velocity(a, surface_contact),
                contact_point_velocity(b, surface_contact),
            )
            closing_speed = -v_dot(relative_velocity, normal)
            for impactor, target, target_to_impactor_normal in (
                (a, b, normal),
                (b, a, v_mul(normal, -1.0)),
            ):
                target_is_supported = target.is_assembly_anchor or not component_has_translation_freedom(target)
                if not target_is_supported:
                    continue
                candidate_relative_velocity = v_sub(
                    contact_point_velocity(impactor, surface_contact),
                    contact_point_velocity(target, surface_contact),
                )
                impact_axis = v_unit(
                    candidate_relative_velocity,
                    v_mul(target_to_impactor_normal, -1.0),
                )
                contact_damage_axis = (
                    thin_component_axis_normal(target, impact_axis)
                    if component_has_thin_axis(target)
                    else impact_axis
                )
                normal_speed = max(
                    abs(v_dot(candidate_relative_velocity, target_to_impactor_normal)),
                    physical_depth / max(MOTION_DT, 1e-9),
                )
                if (
                    normal_speed
                    < max(
                        0.25,
                        0.5 * physical_depth / max(MOTION_DT, 1e-9),
                    )
                ):
                    continue
                response = thin_shell_impact_response(
                    impactor,
                    target,
                    normal_speed,
                    surface_contact,
                    target_to_impactor_normal,
                )
                if response is None:
                    continue
                environment_dynamic_impact = True
                environment_failure_mode = response.failure_mode
                environment_absorbed_energy = response.absorbed_energy_j
                environment_residual_speed = response.residual_speed
                environment_contact_radius = max(
                    response.hole_radius,
                    response.indentation,
                    COLLISION_DEFORMATION_MIN_RADIUS_M,
                )
                indent_impactor, indent_target = split_contact_indentation(
                    impactor,
                    target,
                    response.indentation,
                )
                deform_impactor = deform_component_at_contact(
                    impactor,
                    surface_contact,
                    v_mul(contact_damage_axis, -1.0),
                    indent_impactor,
                    environment_contact_radius,
                )
                deform_target = deform_component_at_contact(
                    target,
                    surface_contact,
                    contact_damage_axis,
                    indent_target,
                    environment_contact_radius,
                )
                register_collision_dent(
                    impactor,
                    surface_contact,
                    v_mul(contact_damage_axis, -1.0),
                    deform_impactor,
                    environment_contact_radius,
                    step,
                    response.failure_mode,
                    0.5 * response.absorbed_energy_j,
                )
                if response.perforated:
                    hole_damage = register_collision_hole(
                        target,
                        surface_contact,
                        contact_damage_axis,
                        response.hole_radius,
                        environment_contact_radius,
                        step,
                        response.failure_mode,
                        response.absorbed_energy_j,
                    )
                    environment_contact_radius = hole_damage.current_hole_radius_m
                    tangent_velocity = v_sub(
                        impactor.linear_velocity,
                        v_mul(
                            impact_axis,
                            v_dot(impactor.linear_velocity, impact_axis),
                        ),
                    )
                    impactor.linear_velocity = v_add(
                        tangent_velocity,
                        v_mul(impact_axis, response.residual_speed),
                    )
                    impactor.freedom.source = "post-perforation-ballistic"
                    impactor.filtered_force = (0.0, 0.0, 0.0)
                    impactor.filtered_moment = (0.0, 0.0, 0.0)
                    impactor.aerodynamic_load_initialized = False
                    environment_perforated = True
                else:
                    register_collision_dent(
                        target,
                        surface_contact,
                        contact_damage_axis,
                        deform_target,
                        environment_contact_radius,
                        step,
                        response.failure_mode,
                        0.5 * response.absorbed_energy_j,
                    )
                    # A non-perforating missed impact must be returned to the
                    # entrance side along the contact normal, not along the
                    # full velocity vector, otherwise the projection invents
                    # sideways drift on oblique impacts.
                    normal = (
                        v_mul(contact_damage_axis, -1.0)
                        if impactor is a
                        else contact_damage_axis
                    )
                if impactor is a:
                    environment_deform_a = deform_impactor
                    environment_deform_b = deform_target
                    environment_indent_a = indent_impactor
                    environment_indent_b = indent_target
                else:
                    environment_deform_a = deform_target
                    environment_deform_b = deform_impactor
                    environment_indent_a = indent_target
                    environment_indent_b = indent_impactor
                break
            if environment_perforated:
                corrected = True
                lines.append(
                    f"{step}\t{COLLISION_MAX_PASSES + constraint_pass}\t"
                    f"{a.patch}\t{b.patch}\t{physical_depth:.8g}\t"
                    f"{normal[0]:.8g}\t{normal[1]:.8g}\t{normal[2]:.8g}\t"
                    f"{surface_contact[0]:.8g}\t{surface_contact[1]:.8g}\t"
                    f"{surface_contact[2]:.8g}\t0\t0\t0\t"
                    f"{environment_deform_a:.8g}\t{environment_deform_b:.8g}\t"
                    f"{environment_contact_radius:.8g}\t"
                    f"{environment_indent_a:.8g}\t{environment_indent_b:.8g}\t"
                    f"{environment_failure_mode}\t1\t"
                    f"{environment_absorbed_energy:.8g}\t"
                    f"{environment_residual_speed:.8g}\t0\t"
                    f"{environment_logged_manifold_points}\t"
                    f"{environment_friction:.8g}"
                )
                continue
            if (
                environment_dynamic_impact
                and environment_failure_mode != "nonpenetration_constraint"
            ):
                a_can_translate = component_has_translation_freedom(a)
                b_can_translate = component_has_translation_freedom(b)
                inv_a = 1.0 / max(a.mass, 1e-9) if a_can_translate else 0.0
                inv_b = 1.0 / max(b.mass, 1e-9) if b_can_translate else 0.0
                inv_sum = inv_a + inv_b
                if inv_sum > 0.0:
                    correction = v_mul(normal, physical_depth / inv_sum)
                    translate_component_for_collision(
                        a,
                        v_mul(correction, inv_a),
                    )
                    translate_component_for_collision(
                        b,
                        v_mul(correction, -inv_b),
                    )
                for _velocity_cleanup_pass in range(2):
                    relative_velocity = v_sub(
                        contact_point_velocity(a, surface_contact),
                        contact_point_velocity(b, surface_contact),
                    )
                    closing_speed = v_dot(relative_velocity, normal)
                    inverse_contact_mass = (
                        contact_inverse_mass(a, surface_contact, normal)
                        + contact_inverse_mass(b, surface_contact, normal)
                    )
                    if closing_speed >= -1e-9 or inverse_contact_mass <= 1e-12:
                        break
                    impulse_mag = -closing_speed / inverse_contact_mass
                    impulse = v_mul(normal, impulse_mag)
                    apply_collision_impulse(a, impulse, surface_contact)
                    apply_collision_impulse(b, v_mul(impulse, -1.0), surface_contact)
                relative_velocity = v_sub(
                    contact_point_velocity(a, surface_contact),
                    contact_point_velocity(b, surface_contact),
                )
                closing_speed = v_dot(relative_velocity, normal)
                if closing_speed < -1e-9:
                    if a_can_translate and not b_can_translate:
                        a.linear_velocity = v_sub(
                            a.linear_velocity,
                            v_mul(normal, closing_speed),
                        )
                    elif b_can_translate and not a_can_translate:
                        b.linear_velocity = v_add(
                            b.linear_velocity,
                            v_mul(normal, closing_speed),
                        )
                corrected = True
                lines.append(
                    f"{step}\t{COLLISION_MAX_PASSES + constraint_pass}\t"
                    f"{a.patch}\t{b.patch}\t{physical_depth:.8g}\t"
                    f"{normal[0]:.8g}\t{normal[1]:.8g}\t{normal[2]:.8g}\t"
                    f"{surface_contact[0]:.8g}\t{surface_contact[1]:.8g}\t"
                    f"{surface_contact[2]:.8g}\t0\t0\t0\t"
                    f"{environment_deform_a:.8g}\t{environment_deform_b:.8g}\t"
                    f"{environment_contact_radius:.8g}\t"
                    f"{environment_indent_a:.8g}\t{environment_indent_b:.8g}\t"
                    f"{environment_failure_mode}\t0\t"
                    f"{environment_absorbed_energy:.8g}\t"
                    f"{environment_residual_speed:.8g}\t0\t"
                    f"{environment_logged_manifold_points}\t"
                    f"{environment_friction:.8g}"
                )
                continue
            if b.is_assembly_anchor and component_has_thin_axis(b):
                normal = thin_component_axis_normal(b, normal)
            elif a.is_assembly_anchor and component_has_thin_axis(a):
                normal = v_mul(
                    thin_component_axis_normal(a, v_mul(normal, -1.0)),
                    -1.0,
                )
            a_can_translate = component_has_translation_freedom(a)
            b_can_translate = component_has_translation_freedom(b)
            inv_a = 1.0 / max(a.mass, 1e-9) if a_can_translate else 0.0
            inv_b = 1.0 / max(b.mass, 1e-9) if b_can_translate else 0.0
            inv_sum = inv_a + inv_b
            applied_a = (0.0, 0.0, 0.0)
            applied_b = (0.0, 0.0, 0.0)
            applied_rotation_a = (0.0, 0.0, 0.0)
            applied_rotation_b = (0.0, 0.0, 0.0)
            if inv_sum > 0.0:
                correction = v_mul(normal, physical_depth / inv_sum)
                applied_a = translate_component_for_collision(
                    a,
                    v_mul(correction, inv_a),
                )
                applied_b = translate_component_for_collision(
                    b,
                    v_mul(correction, -inv_b),
                )
            if (
                v_norm(applied_a) <= 1e-14
                and component_has_rotation_freedom(a)
            ):
                applied_rotation_a = rotate_component_for_collision(
                    a,
                    normal,
                    surface_contact,
                    physical_depth,
                )
            if (
                v_norm(applied_b) <= 1e-14
                and component_has_rotation_freedom(b)
            ):
                applied_rotation_b = rotate_component_for_collision(
                    b,
                    v_mul(normal, -1.0),
                    surface_contact,
                    physical_depth,
                )
            if (
                v_norm(applied_a) <= 1e-14
                and v_norm(applied_b) <= 1e-14
                and v_norm(applied_rotation_a) <= 1e-14
                and v_norm(applied_rotation_b) <= 1e-14
            ):
                continue

            relative_velocity = v_sub(
                contact_point_velocity(a, surface_contact),
                contact_point_velocity(b, surface_contact),
            )
            closing_speed = v_dot(relative_velocity, normal)
            impulse_mag = 0.0
            inverse_contact_mass = (
                contact_inverse_mass(a, surface_contact, normal)
                + contact_inverse_mass(b, surface_contact, normal)
            )
            if closing_speed < 0.0 and inverse_contact_mass > 1e-12:
                impulse_mag = -closing_speed / inverse_contact_mass
                impulse = v_mul(normal, impulse_mag)
                apply_collision_impulse(a, impulse, surface_contact)
                apply_collision_impulse(b, v_mul(impulse, -1.0), surface_contact)
            corrected = True
            lines.append(
                f"{step}\t{COLLISION_MAX_PASSES + constraint_pass}\t"
                f"{a.patch}\t{b.patch}\t{physical_depth:.8g}\t"
                f"{normal[0]:.8g}\t{normal[1]:.8g}\t{normal[2]:.8g}\t"
                f"{surface_contact[0]:.8g}\t{surface_contact[1]:.8g}\t"
                f"{surface_contact[2]:.8g}\t{impulse_mag:.8g}\t"
                f"{v_norm(applied_a):.8g}\t{v_norm(applied_b):.8g}\t"
                f"{environment_deform_a:.8g}\t{environment_deform_b:.8g}\t"
                f"{environment_contact_radius:.8g}\t"
                f"{environment_indent_a:.8g}\t{environment_indent_b:.8g}\t"
                f"{environment_failure_mode}\t0\t"
                f"{environment_absorbed_energy:.8g}\t"
                f"{environment_residual_speed:.8g}\t0\t"
                f"{environment_logged_manifold_points}\t"
                f"{environment_friction:.8g}"
            )
        if not corrected:
            break
    return lines


def resolve_part_collisions(
    components: List[AeroComponent],
    step: int,
    log_path: Path,
    swept_contact: Optional[SweptCollisionContact] = None,
    prescribed_pair: Optional[Tuple[AeroComponent, AeroComponent]] = None,
    initial_overlap_pairs: Optional[
        Dict[Tuple[int, int], InitialCadOverlap]
    ] = None,
) -> List[str]:
    if not ENABLE_PART_COLLISIONS or len(components) < 2:
        return []

    def current_active_components() -> List[AeroComponent]:
        active = list(components)
        for component in components:
            state = component.collision_structural_state
            if isinstance(state, HybridShellCollisionState):
                active.extend(hybrid_fragment_components(state))
            elif isinstance(state, HybridFEMMPMCollisionState):
                active.extend(fragment.component for fragment in state.fragment_bodies)
        return active

    def discard_separated_initial_overlaps(
        active: Sequence[AeroComponent],
    ) -> None:
        if not initial_overlap_pairs:
            return
        component_by_id = {id(component): component for component in active}
        for pair_key in list(initial_overlap_pairs):
            a = component_by_id.get(pair_key[0])
            b = component_by_id.get(pair_key[1])
            if a is None or b is None or aabb_overlap_with_normal(a, b) is None:
                initial_overlap_pairs.pop(pair_key, None)

    def unresolved_overlap_remains(
        active: Sequence[AeroComponent],
    ) -> bool:
        for index_a, index_b in collision_broad_phase_pairs(active, 0.0):
            a = active[index_a]
            b = active[index_b]
            if (
                a.rigid_body_group is not None
                and a.rigid_body_group == b.rigid_body_group
            ):
                continue
            overlap = aabb_overlap_with_normal(a, b)
            if overlap is None:
                continue
            depth, _normal, _contact = overlap
            pair_key = collision_pair_key(a, b)
            if (
                initial_overlap_pairs is not None
                and pair_key in initial_overlap_pairs
            ):
                baseline = initial_overlap_pairs[pair_key].depth_m
                if depth <= baseline + COLLISION_MIN_OVERLAP_M:
                    continue
            if depth <= COLLISION_MIN_OVERLAP_M:
                continue
            if triangle_mesh_intersection_contact(a.triangles, b.triangles) is not None:
                return True
        return False

    lines: List[str] = []
    handled_relative_sweeps: Set[Tuple[int, int]] = set()
    impulse_resolved_pairs: Set[Tuple[int, int]] = set()
    pending_swept_contact = swept_contact
    max_outer_passes = max(1, COLLISION_MAX_PASSES)
    broad_phase_candidates = 0
    brute_force_pair_checks = 0
    peak_active_bodies = 0
    collision_resolution_started = time.monotonic()
    for _outer_pass in range(max_outer_passes):
        active_components = current_active_components()
        discard_separated_initial_overlaps(active_components)
        peak_active_bodies = max(peak_active_bodies, len(active_components))
        outer_progress = False
        for collision_pass in range(COLLISION_MAX_PASSES):
            any_collision = False
            candidate_pairs = collision_broad_phase_pairs(
                active_components,
                MOTION_DT,
            )
            if pending_swept_contact is not None:
                index_by_id = {
                    id(component): index
                    for index, component in enumerate(active_components)
                }
                moving_index = index_by_id.get(id(pending_swept_contact.moving))
                stationary_index = index_by_id.get(
                    id(pending_swept_contact.stationary)
                )
                if moving_index is not None and stationary_index is not None:
                    prescribed_indices = (
                        min(moving_index, stationary_index),
                        max(moving_index, stationary_index),
                    )
                    if prescribed_indices not in candidate_pairs:
                        candidate_pairs.append(prescribed_indices)
                        candidate_pairs.sort()
            broad_phase_candidates += len(candidate_pairs)
            brute_force_pair_checks += (
                len(active_components) * (len(active_components) - 1) // 2
            )
            for i, j in candidate_pairs:
                a = active_components[i]
                b = active_components[j]
                pair_key = collision_pair_key(a, b)
                if pair_key in impulse_resolved_pairs:
                    # One contact pair receives at most one impact impulse
                    # during a physical time step.  Geometry may remain
                    # overlapped while the elastic/plastic deformation is
                    # applied; the later Signorini projection removes that
                    # overlap without a second impulse whose independently
                    # estimated normal can flip and cancel the rebound.
                    continue
                initial_overlap_depth = 0.0
                relative_swept_contact: Optional[RelativeSweptContact] = None
                is_swept_pair = (
                    pending_swept_contact is not None
                    and {id(a), id(b)}
                    == {
                        id(pending_swept_contact.moving),
                        id(pending_swept_contact.stationary),
                    }
                )
                same_rigid_group = (
                    a.rigid_body_group is not None
                    and a.rigid_body_group == b.rigid_body_group
                )
                if (
                    pair_key not in handled_relative_sweeps
                    and not same_rigid_group
                    and not is_swept_pair
                    and a.collision_fragment_created_step != step
                    and b.collision_fragment_created_step != step
                ):
                    relative_swept_contact = swept_relative_component_contact(
                        a,
                        b,
                        MOTION_DT,
                    )
                if initial_overlap_pairs is not None and pair_key in initial_overlap_pairs:
                    # This pair started in an intentional CAD fit.  Do not turn
                    # broad-phase overlap into strain energy.  Re-enable normal
                    # collision handling after the bodies have truly separated.
                    initial_overlap = initial_overlap_pairs[pair_key]
                    current_overlap = aabb_overlap_with_normal(a, b)
                    if current_overlap is None:
                        initial_overlap_pairs.pop(pair_key, None)
                    else:
                        initial_overlap_depth = initial_overlap.depth_m
                        if initial_overlap.surfaces_intersect:
                            # Surface-intersecting bodies are already at their
                            # assembled CAD interface. A tiny difference in
                            # aerodynamic acceleration must not turn that
                            # interface into a new impact and alter their
                            # imported relative pose. Only overlap beyond the
                            # recorded baseline is physical penetration.
                            if (
                                current_overlap[0]
                                <= initial_overlap_depth
                                + COLLISION_MIN_OVERLAP_M
                            ):
                                continue
                            relative_swept_contact = None
                        elif (
                            current_overlap[0]
                            <= initial_overlap_depth
                            + COLLISION_MIN_OVERLAP_M
                            and relative_swept_contact is None
                        ):
                            continue
                    if (
                        relative_swept_contact is not None
                        and not initial_overlap.surfaces_intersect
                    ):
                        # Relative motion has reached a real triangle surface,
                        # so the stress-free CAD-fit exemption has ended even
                        # if the broad AABBs never became disjoint.
                        initial_overlap_pairs.pop(pair_key, None)
                        initial_overlap_depth = 0.0
                if is_swept_pair and collision_pass > 0:
                    continue
                if is_swept_pair:
                    assert pending_swept_contact is not None
                    if a is pending_swept_contact.moving:
                        hit = pending_swept_contact.depth, pending_swept_contact.normal, pending_swept_contact.point
                    else:
                        hit = pending_swept_contact.depth, v_mul(pending_swept_contact.normal, -1.0), pending_swept_contact.point
                elif relative_swept_contact is not None:
                    normal_speed = max(
                        0.0,
                        -v_dot(
                            v_sub(a.linear_velocity, b.linear_velocity),
                            relative_swept_contact.normal,
                        ),
                    )
                    impact_depth = max(
                        impact_contact_indentation(
                            a,
                            b,
                            normal_speed,
                            relative_swept_contact.point,
                            relative_swept_contact.point,
                            relative_swept_contact.normal,
                        ),
                        2.0 * COLLISION_MIN_OVERLAP_M,
                    )
                    hit = (
                        impact_depth,
                        relative_swept_contact.normal,
                        relative_swept_contact.point,
                    )
                else:
                    hit = aabb_overlap_with_normal(a, b)
                    if hit is not None:
                        surface_hit = triangle_mesh_intersection_contact(
                            a.triangles,
                            b.triangles,
                        )
                        if surface_hit is None:
                            continue
                        surface_contact, surface_normal = surface_hit
                        center_offset = v_sub(
                            component_center_from_bounds(a),
                            component_center_from_bounds(b),
                        )
                        if (
                            v_norm(center_offset) > 1e-12
                            and v_dot(
                                surface_normal,
                                v_unit(center_offset),
                            ) < 0.2
                        ):
                            surface_normal = hit[1]
                        hit = hit[0], surface_normal, surface_contact
                if hit is None:
                    continue
                depth, normal, contact = hit
                normal = stable_collision_normal(
                    a,
                    b,
                    normal,
                    preserve_detected_orientation=(
                        is_swept_pair or relative_swept_contact is not None
                    ),
                )
                depth = max(
                    depth - initial_overlap_depth,
                    COLLISION_MIN_OVERLAP_M,
                )
                if not is_swept_pair:
                    relative_velocity = v_sub(
                        contact_point_velocity(a, contact),
                        contact_point_velocity(b, contact),
                    )
                    normal_speed = max(
                        0.0,
                        -v_dot(relative_velocity, normal),
                    )
                    if normal_speed <= 1e-9:
                        continue
                    if relative_swept_contact is None:
                        depth = min(
                            depth,
                            max(
                                impact_contact_indentation(
                                    a,
                                    b,
                                    normal_speed,
                                    contact,
                                    contact,
                                    normal,
                                ),
                                2.0 * COLLISION_MIN_OVERLAP_M,
                            ),
                        )
                any_collision = True
                outer_progress = True
                # remainder of inner collision handling unchanged
                a_can_translate = component_has_translation_freedom(a)
                b_can_translate = component_has_translation_freedom(b)
                inv_a = 0.0 if (a.is_assembly_anchor or not a_can_translate) else 1.0 / max(a.mass, 1e-9)
                inv_b = 0.0 if (b.is_assembly_anchor or not b_can_translate) else 1.0 / max(b.mass, 1e-9)
                inv_sum = inv_a + inv_b
                time_of_impact_move_a = (0.0, 0.0, 0.0)
                time_of_impact_move_b = (0.0, 0.0, 0.0)
                if (
                    relative_swept_contact is not None
                    and relative_swept_contact.rotational
                ):
                    time_of_impact_move_a = rewind_component_rotational_sweep(
                        a,
                        relative_swept_contact.translation_a,
                        relative_swept_contact.rotation_a,
                        relative_swept_contact.time_fraction,
                    )
                    time_of_impact_move_b = rewind_component_rotational_sweep(
                        b,
                        relative_swept_contact.translation_b,
                        relative_swept_contact.rotation_b,
                        relative_swept_contact.time_fraction,
                    )
                    handled_relative_sweeps.add(pair_key)
                elif relative_swept_contact is not None and inv_sum > 0.0:
                    overshoot = max(
                        0.0,
                        relative_swept_contact.sweep_distance_m
                        - relative_swept_contact.travel_to_contact_m,
                    )
                    if overshoot > 0.0:
                        correction = v_mul(
                            relative_swept_contact.direction,
                            overshoot / inv_sum,
                        )
                        time_of_impact_move_a = translate_component_for_collision(
                            a,
                            v_mul(correction, -inv_a),
                        )
                        time_of_impact_move_b = translate_component_for_collision(
                            b,
                            v_mul(correction, inv_b),
                        )
                        contact = v_add(contact, time_of_impact_move_b)
                    handled_relative_sweeps.add(pair_key)
                if not is_swept_pair and relative_swept_contact is not None:
                    later_perforation = relative_swept_perforation_contact(
                        a,
                        b,
                        relative_swept_contact,
                        MOTION_DT,
                    )
                    if later_perforation is not None:
                        pending_swept_contact = later_perforation
                        is_swept_pair = True
                if is_swept_pair and pending_swept_contact is not None and pending_swept_contact.perforated:
                    impact_axis = pending_swept_contact.approach_axis
                    moving_surface_point = v_add(
                        pending_swept_contact.point,
                        v_mul(impact_axis, pending_swept_contact.approach_penetration),
                    )
                    impact_depth = max(
                        pending_swept_contact.approach_penetration,
                        pending_swept_contact.depth,
                        2.0 * COLLISION_MIN_OVERLAP_M,
                    )
                    contact_radius = hertz_contact_area_radius(
                        pending_swept_contact.moving,
                        pending_swept_contact.stationary,
                        impact_depth,
                        geometry=pending_swept_contact.contact_geometry,
                    )
                    indent_moving, indent_stationary = split_contact_indentation(
                        pending_swept_contact.moving,
                        pending_swept_contact.stationary,
                        impact_depth,
                    )
                    deform_moving = deform_component_at_contact(
                        pending_swept_contact.moving,
                        moving_surface_point,
                        v_mul(impact_axis, -1.0),
                        indent_moving,
                        contact_radius,
                    )
                    deform_stationary = deform_component_at_contact(
                        pending_swept_contact.stationary,
                        pending_swept_contact.point,
                        impact_axis,
                        indent_stationary,
                        contact_radius,
                    )
                    deform_collision_reference(
                        pending_swept_contact.moving,
                        moving_surface_point,
                        v_mul(impact_axis, -1.0),
                        deform_moving,
                        contact_radius,
                    )
                    deform_collision_reference(
                        pending_swept_contact.stationary,
                        pending_swept_contact.point,
                        impact_axis,
                        deform_stationary,
                        contact_radius,
                    )
                    hole_damage = register_collision_hole(
                        pending_swept_contact.stationary,
                        pending_swept_contact.point,
                        impact_axis,
                        pending_swept_contact.hole_radius,
                        contact_radius,
                        step,
                        pending_swept_contact.failure_mode,
                        pending_swept_contact.absorbed_energy_j,
                    )
                    removed_triangles = 0
                    pending_swept_contact.moving.linear_velocity = v_mul(
                        impact_axis,
                        pending_swept_contact.residual_speed,
                    )
                    pending_swept_contact.moving.angular_velocity = (0.0, 0.0, 0.0)
                    pending_swept_contact.moving.freedom.source = "post-perforation-ballistic"
                    pending_swept_contact.moving.filtered_force = (0.0, 0.0, 0.0)
                    pending_swept_contact.moving.filtered_moment = (0.0, 0.0, 0.0)
                    pending_swept_contact.moving.aerodynamic_load_initialized = False
                    apply_nearby_collision_effects(
                        components,
                        (pending_swept_contact.moving, pending_swept_contact.stationary),
                        pending_swept_contact.point,
                        impact_axis,
                        contact_radius,
                        pending_swept_contact.absorbed_energy_j,
                        step,
                        pending_swept_contact.failure_mode,
                    )
                    deform_a = deform_moving if a is pending_swept_contact.moving else deform_stationary
                    deform_b = deform_moving if b is pending_swept_contact.moving else deform_stationary
                    line = (
                        f"{step}\t{collision_pass}\t{a.patch}\t{b.patch}\t"
                        f"{depth:.8g}\t{normal[0]:.8g}\t{normal[1]:.8g}\t{normal[2]:.8g}\t"
                        f"{contact[0]:.8g}\t{contact[1]:.8g}\t{contact[2]:.8g}\t"
                        f"0\t0\t0\t{deform_a:.8g}\t{deform_b:.8g}\t"
                        f"{hole_damage.current_hole_radius_m:.8g}\t0\t0\t"
                        f"{pending_swept_contact.failure_mode}\t1\t{pending_swept_contact.absorbed_energy_j:.8g}\t"
                        f"{pending_swept_contact.residual_speed:.8g}\t{removed_triangles}\t"
                        f"{pending_swept_contact.manifold_points}\t0"
                    )
                    lines.append(line)
                    pending_swept_contact = None
                    continue
                contact_a = contact
                contact_b = contact
                contact_geometry = (
                    pending_swept_contact.contact_geometry
                    if is_swept_pair and pending_swept_contact is not None
                    else None
                )
                if is_swept_pair:
                    assert pending_swept_contact is not None
                    moving_surface_point = v_add(
                        pending_swept_contact.point,
                        v_mul(
                            pending_swept_contact.approach_axis,
                            pending_swept_contact.approach_penetration,
                        ),
                    )
                    if a is pending_swept_contact.moving:
                        contact_a = moving_surface_point
                    else:
                        contact_b = moving_surface_point
                if contact_geometry is None:
                    contact_geometry = local_contact_geometry(
                        a,
                        b,
                        contact_a,
                        contact_b,
                        normal,
                    )
                contact_radius = hertz_contact_area_radius(
                    a,
                    b,
                    depth,
                    geometry=contact_geometry,
                )
                indent_a, indent_b = split_contact_indentation(a, b, depth)
                elastic_indentation = min(depth, indent_a + indent_b)
                rigid_overlap = max(0.0, depth - elastic_indentation)
                # The overlap returned by the broad phase is not a local
                # penetration measurement.  A small projectile crossing a
                # large plate can therefore report the plate's full span.
                # Project only a contact-scale amount per pass and let the
                # mesh contact loop recheck the pair.  This prevents large
                # artificial bounce translations while retaining
                # non-penetration for genuine surface intersections.
                contact_scale = max(
                    4.0 * COLLISION_MIN_OVERLAP_M,
                    2.0 * contact_radius,
                )
                correction_depth = rigid_overlap
                if (
                    relative_swept_contact is None
                    and not is_swept_pair
                    and rigid_overlap > 4.0 * contact_scale
                ):
                    correction_depth = contact_scale
                correction_mag = correction_depth * max(
                    0.0,
                    min(COLLISION_POSITION_CORRECTION, 1.0),
                )
                applied_a = time_of_impact_move_a
                applied_b = time_of_impact_move_b
                if inv_sum > 0.0 and correction_mag > 0.0:
                    corr = v_mul(normal, correction_mag / inv_sum)
                    applied_a = translate_component_for_collision(a, v_mul(corr, inv_a))
                    applied_b = translate_component_for_collision(b, v_mul(corr, -inv_b))
                if (
                    not a_can_translate
                    and v_norm(applied_a) <= 1e-14
                    and component_has_rotation_freedom(a)
                ):
                    rotate_component_for_collision(a, normal, contact, depth)
                if (
                    not b_can_translate
                    and v_norm(applied_b) <= 1e-14
                    and component_has_rotation_freedom(b)
                ):
                    rotate_component_for_collision(b, v_mul(normal, -1.0), contact, depth)
                rel_v = v_sub(
                    contact_point_velocity(a, contact),
                    contact_point_velocity(b, contact),
                )
                rel_normal = v_dot(rel_v, normal)
                impulse_mag = 0.0
                tangent_impulse = (0.0, 0.0, 0.0)
                friction = contact_friction_coefficient(a, b)
                persistent_contact, is_new_contact = persistent_contact_for_pair(
                    a,
                    b,
                    step,
                    normal,
                    contact,
                )
                normal_inverse_mass = (
                    contact_inverse_mass(a, contact, normal)
                    + contact_inverse_mass(b, contact, normal)
                )
                if normal_inverse_mass > 0.0 and rel_normal < 0.0:
                    restitution = contact_restitution_coefficient(
                        a,
                        b,
                        (
                            COLLISION_PRESCRIBED_IMPACT_RESTITUTION
                            if is_swept_pair
                            else COLLISION_RESTITUTION
                        ),
                    ) if is_new_contact else 0.0
                    impulse_mag = -(1.0 + restitution) * rel_normal / normal_inverse_mass
                    impulse = v_mul(normal, impulse_mag)
                    apply_collision_impulse(a, impulse, contact)
                    apply_collision_impulse(b, v_mul(impulse, -1.0), contact)
                    rel_v = v_sub(
                        contact_point_velocity(a, contact),
                        contact_point_velocity(b, contact),
                    )
                    rel_normal_after = v_dot(rel_v, normal)
                    tangent = v_sub(rel_v, v_mul(normal, rel_normal_after))
                    tmag = v_norm(tangent)
                    if tmag > 1e-12 and friction > 0.0:
                        tdir = v_mul(tangent, 1.0 / tmag)
                        tangent_inverse_mass = (
                            contact_inverse_mass(a, contact, tdir)
                            + contact_inverse_mass(b, contact, tdir)
                        )
                        jt = min(
                            tmag / max(tangent_inverse_mass, 1e-12),
                            impulse_mag * friction,
                        )
                        timpulse = v_mul(tdir, -jt)
                        apply_collision_impulse(a, timpulse, contact)
                        apply_collision_impulse(b, v_mul(timpulse, -1.0), contact)
                        tangent_impulse = timpulse
                    persistent_contact.accumulated_normal_impulse_ns += impulse_mag
                    persistent_contact.accumulated_tangent_impulse = v_add(
                        persistent_contact.accumulated_tangent_impulse,
                        tangent_impulse,
                    )
                    impulse_resolved_pairs.add(pair_key)
                elif inv_sum <= 0.0:
                    pseudo_force = v_mul(normal, max(depth, COLLISION_MIN_OVERLAP_M) * max(a.mass, b.mass, DEFAULT_PART_MASS_KG) / max(MOTION_DT, 1e-9))
                    apply_collision_impulse(a, pseudo_force, contact)
                    apply_collision_impulse(b, v_mul(pseudo_force, -1.0), contact)
                deform_a = deform_component_at_contact(a, contact_a, normal, indent_a, contact_radius)
                deform_b = deform_component_at_contact(b, contact_b, v_mul(normal, -1.0), indent_b, contact_radius)
                if is_swept_pair and swept_contact is not None:
                    failure_mode = pending_swept_contact.failure_mode
                    logged_manifold_points = pending_swept_contact.manifold_points
                elif relative_swept_contact is not None:
                    failure_mode = "continuous_internal_contact"
                    logged_manifold_points = relative_swept_contact.manifold_points
                else:
                    failure_mode = "elastic_contact"
                    logged_manifold_points = 1
                absorbed_energy = (
                    pending_swept_contact.absorbed_energy_j
                    if is_swept_pair and pending_swept_contact is not None
                    else 0.0
                )
                secondary_energy = absorbed_energy
                if secondary_energy <= 1e-12 and impulse_mag > 0.0:
                    secondary_energy = (
                        0.5
                        * impulse_mag
                        * impulse_mag
                        * normal_inverse_mass
                    )
                register_collision_dent(
                    a,
                    contact_a,
                    normal,
                    deform_a,
                    contact_radius,
                    step,
                    failure_mode,
                    0.5 * secondary_energy,
                )
                register_collision_dent(
                    b,
                    contact_b,
                    v_mul(normal, -1.0),
                    deform_b,
                    contact_radius,
                    step,
                    failure_mode,
                    0.5 * secondary_energy,
                )
                apply_nearby_collision_effects(
                    components,
                    (a, b),
                    contact,
                    normal,
                    contact_radius,
                    secondary_energy,
                    step,
                    failure_mode,
                )
                if is_swept_pair:
                    assert pending_swept_contact is not None
                    unresolved_overlap = max(
                        0.0,
                        pending_swept_contact.depth
                        - correction_mag
                        - deform_a
                        - deform_b,
                    )
                    if unresolved_overlap > COLLISION_MIN_OVERLAP_M:
                        closure = translate_component_for_collision(
                            pending_swept_contact.moving,
                            v_mul(pending_swept_contact.normal, unresolved_overlap),
                        )
                        if a is pending_swept_contact.moving:
                            applied_a = v_add(applied_a, closure)
                        else:
                            applied_b = v_add(applied_b, closure)
                    pending_swept_contact.moving.filtered_force = (0.0, 0.0, 0.0)
                    pending_swept_contact.moving.filtered_moment = (0.0, 0.0, 0.0)
                    pending_swept_contact.moving.aerodynamic_load_initialized = False
                line = (
                    f"{step}\t{collision_pass}\t{a.patch}\t{b.patch}\t"
                    f"{depth:.8g}\t{normal[0]:.8g}\t{normal[1]:.8g}\t{normal[2]:.8g}\t"
                    f"{contact[0]:.8g}\t{contact[1]:.8g}\t{contact[2]:.8g}\t"
                    f"{impulse_mag:.8g}\t{v_norm(applied_a):.8g}\t{v_norm(applied_b):.8g}\t"
                    f"{deform_a:.8g}\t{deform_b:.8g}\t{contact_radius:.8g}\t{indent_a:.8g}\t{indent_b:.8g}\t"
                    f"{failure_mode}\t"
                    f"0\t{pending_swept_contact.absorbed_energy_j if is_swept_pair and pending_swept_contact else 0.0:.8g}\t"
                    f"{pending_swept_contact.residual_speed if is_swept_pair and pending_swept_contact else 0.0:.8g}\t0\t"
                    f"{logged_manifold_points}\t{friction:.8g}"
                )
                lines.append(line)
                if is_swept_pair:
                    pending_swept_contact = None
            if not any_collision:
                break
        enforce_attachment_constraints(components)
        environment_lines = enforce_environment_contact_constraints(
            current_active_components(),
            step,
            initial_overlap_pairs,
        )
        if environment_lines:
            outer_progress = True
            lines.extend(environment_lines)
        if not outer_progress:
            break
        if not unresolved_overlap_remains(current_active_components()):
            break
    collision_resolution_elapsed = time.monotonic() - collision_resolution_started
    if brute_force_pair_checks:
        reduction_percent = 100.0 * (
            1.0 - broad_phase_candidates / brute_force_pair_checks
        )
        print(
            "Collision broad phase: "
            f"peak {peak_active_bodies} bodies, "
            f"tested {broad_phase_candidates}/{brute_force_pair_checks} "
            "conservative swept pairs "
            f"({max(0.0, reduction_percent):.1f}% pruned), "
            f"resolution {collision_resolution_elapsed:.2f}s."
        )
    if lines:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            for line in lines:
                f.write(line + "\n")
    return lines


def collision_convergence_enabled() -> bool:
    return COLLISION_CONVERGENCE_SPEED_MPS > 0.0


def _component_matches_selector(component: AeroComponent, selector: str) -> bool:
    selector = selector.strip().lower()
    if not selector:
        return False
    patch = component.patch.lower()
    name = component.name.lower()
    return selector == patch or selector == name or selector in patch or selector in name


def _select_component_by_selector(components: Sequence[AeroComponent], selector: str) -> Optional[AeroComponent]:
    exact = [
        component for component in components
        if selector.strip().lower() in {component.patch.lower(), component.name.lower()}
    ]
    if exact:
        return exact[0]
    return next((component for component in components if _component_matches_selector(component, selector)), None)


def collision_source_group(
    component: AeroComponent,
    components: Optional[Sequence[AeroComponent]] = None,
) -> List[AeroComponent]:
    """Return the independent bodies imported from the same collision source.

    Membership here is only used to prescribe their common launch translation.
    It is deliberately not a rigid-body or mate constraint, so collision impulses
    can give each disconnected body a different trajectory after contact.
    """
    if components is None or component.collision_source_index is None:
        return [component]
    return [
        candidate
        for candidate in components
        if candidate.collision_source_index == component.collision_source_index
    ] or [component]


def components_share_collision_source(a: AeroComponent, b: AeroComponent) -> bool:
    return (
        a.collision_source_index is not None
        and a.collision_source_index == b.collision_source_index
    )


def select_collision_convergence_pair(components: Sequence[AeroComponent]) -> Optional[Tuple[AeroComponent, AeroComponent]]:
    if not collision_convergence_enabled() or len(components) < 2:
        return None

    selectors = [
        part.strip()
        for part in re.split(r"[,;]", COLLISION_CONVERGENCE_COMPONENTS)
        if part.strip()
    ]
    if len(selectors) >= 2:
        first = _select_component_by_selector(components, selectors[0])
        second = _select_component_by_selector(components, selectors[1])
        if first is not None and second is not None and first is not second:
            return first, second
        print(
            "WARNING: COLLISION_CONVERGENCE_COMPONENTS did not match two distinct components; "
            "using automatic pair selection."
        )

    source_groups: Dict[int, List[AeroComponent]] = {}
    source_order: List[int] = []
    for component in components:
        source_index = component.collision_source_index
        if source_index is None:
            continue
        if source_index not in source_groups:
            source_groups[source_index] = []
            source_order.append(source_index)
        source_groups[source_index].append(component)
    if len(source_order) >= 2:
        first_group = source_groups[source_order[0]]
        second_group = source_groups[source_order[1]]
        return (
            max(first_group, key=_component_policy_size_score),
            max(second_group, key=_component_policy_size_score),
        )

    return components[0], components[1]


def configure_collision_convergence_components(components: Sequence[AeroComponent]) -> Optional[Tuple[AeroComponent, AeroComponent]]:
    pair = select_collision_convergence_pair(components)
    if pair is None:
        return None
    free = six_dof_motion_freedom("collision-convergence")
    moving, stationary = collision_convergence_moving_and_stationary(pair)
    axis = collision_convergence_approach_axis(pair)
    for member in collision_source_group(moving, components):
        member.is_assembly_anchor = False
        member.freedom = MotionFreedom(
            translate_axes=list(free.translate_axes),
            rotate_axes=list(free.rotate_axes),
            mate_type="COLLISION_IMPACTOR",
            source="collision-convergence-moving",
        )
        member.linear_velocity = v_mul(axis, COLLISION_CONVERGENCE_SPEED_MPS)
    for member in collision_source_group(stationary, components):
        member.is_assembly_anchor = True
        member.freedom = MotionFreedom(
            [], [], "COLLISION_TARGET", "collision-convergence-stationary"
        )
        member.linear_velocity = (0.0, 0.0, 0.0)
        member.angular_velocity = (0.0, 0.0, 0.0)
    return pair


def collision_convergence_moving_and_stationary(
    pair: Tuple[AeroComponent, AeroComponent],
) -> Tuple[AeroComponent, AeroComponent]:
    a, b = pair
    mode = COLLISION_CONVERGENCE_MOVING_COMPONENT
    if mode in {"second", "b", "2", b.patch.lower(), b.name.lower()}:
        return b, a
    return a, b


def closest_point_on_triangle(point: Vec3, triangle: Triangle) -> Vec3:
    """Return the nearest point using the standard barycentric-region test."""
    _normal, a, b, c = triangle
    ab = v_sub(b, a)
    ac = v_sub(c, a)
    ap = v_sub(point, a)
    d1 = v_dot(ab, ap)
    d2 = v_dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = v_sub(point, b)
    d3 = v_dot(ab, bp)
    d4 = v_dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        weight = d1 / max(d1 - d3, 1e-18)
        return v_add(a, v_mul(ab, weight))

    cp = v_sub(point, c)
    d5 = v_dot(ab, cp)
    d6 = v_dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        weight = d2 / max(d2 - d6, 1e-18)
        return v_add(a, v_mul(ac, weight))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = v_sub(c, b)
        weight = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-18)
        return v_add(b, v_mul(bc, weight))

    denominator = max(va + vb + vc, 1e-18)
    v_weight = vb / denominator
    w_weight = vc / denominator
    return v_add(a, v_add(v_mul(ab, v_weight), v_mul(ac, w_weight)))


def collision_convergence_surface_axis(
    moving: AeroComponent,
    stationary: AeroComponent,
) -> Optional[Vec3]:
    """Select the physical target face and approach along its outward normal."""
    moving_center = component_center_from_bounds(moving)
    stationary_points = stl_points(stationary.triangles)
    if not stationary_points:
        return None
    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(stationary.triangles)
    extents = (xmax - xmin, ymax - ymin, zmax - zmin)
    positive_extents = [extent for extent in extents if extent > 1e-12]
    aspect_ratio = max(positive_extents, default=1.0) / max(
        min(positive_extents, default=1.0),
        1e-12,
    )

    # A sheet has two broad faces and very small edge faces.  The broad-face
    # normal is its intended impact orientation, even when a corner happens to
    # be geometrically closer to the impactor centre.
    if aspect_ratio >= 5.0:
        largest = max(
            stationary.triangles,
            key=lambda triangle: triangle_area_centroid_normal(triangle)[0],
        )
        _area, _centroid, normal = triangle_area_centroid_normal(largest)
        surface_point = closest_point_on_triangle(moving_center, largest)
        if v_dot(normal, v_sub(moving_center, surface_point)) < 0.0:
            normal = v_mul(normal, -1.0)
        return v_mul(v_unit(normal), -1.0)

    nearest: Optional[Tuple[float, Vec3, Vec3]] = None
    for triangle in stationary.triangles:
        _area, _centroid, normal = triangle_area_centroid_normal(triangle)
        point = closest_point_on_triangle(moving_center, triangle)
        offset = v_sub(moving_center, point)
        distance_sq = v_dot(offset, offset)
        if nearest is None or distance_sq < nearest[0]:
            nearest = (distance_sq, point, normal)
    if nearest is None:
        return None

    _distance_sq, surface_point, normal = nearest
    # The STL winding can be either direction.  Make the normal point out of
    # the target toward the impactor, then negate it for travel into the face.
    if v_dot(normal, v_sub(moving_center, surface_point)) < 0.0:
        normal = v_mul(normal, -1.0)
    return v_mul(v_unit(normal), -1.0)


def parse_collision_convergence_axis(pair: Tuple[AeroComponent, AeroComponent]) -> Vec3:
    raw = COLLISION_CONVERGENCE_AXIS.strip().lower()
    if raw in {"x", "+x"}:
        return (1.0, 0.0, 0.0)
    if raw == "-x":
        return (-1.0, 0.0, 0.0)
    if raw in {"y", "+y"}:
        return (0.0, 1.0, 0.0)
    if raw == "-y":
        return (0.0, -1.0, 0.0)
    if raw in {"z", "+z"}:
        return (0.0, 0.0, 1.0)
    if raw == "-z":
        return (0.0, 0.0, -1.0)
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(parts) == 3:
        try:
            return v_unit((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            pass

    moving, stationary = collision_convergence_moving_and_stationary(pair)
    # In automatic collision-convergence mode, place the impactor directly in
    # front of the target and drive it along one fixed world-axis line.  The
    # impact face orientation can affect contact normals and deformation, but
    # it must not turn the prescribed pre-impact flight path diagonal.
    delta = v_sub(
        component_center_from_bounds(stationary),
        component_center_from_bounds(moving),
    )
    dominant_index = max(range(3), key=lambda index: abs(delta[index]))
    sign = 1.0 if delta[dominant_index] >= 0.0 else -1.0
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return v_mul(axes[dominant_index], sign)


def collision_convergence_approach_axis(pair: Tuple[AeroComponent, AeroComponent]) -> Vec3:
    moving, stationary = collision_convergence_moving_and_stationary(pair)
    axis = parse_collision_convergence_axis(pair)
    if COLLISION_CONVERGENCE_AXIS.strip().lower() in {"", "auto"}:
        return axis
    center_delta = v_sub(component_center_from_bounds(stationary), component_center_from_bounds(moving))
    if v_dot(center_delta, axis) < 0.0:
        axis = v_mul(axis, -1.0)
    return axis


def aabb_gap_along_axis(a: AeroComponent, b: AeroComponent, axis: Vec3) -> float:
    u = v_unit(axis)
    a_points = stl_points(a.triangles)
    b_points = stl_points(b.triangles)
    if not a_points or not b_points:
        return 0.0
    a_vals = [v_dot(point, u) for point in a_points]
    b_vals = [v_dot(point, u) for point in b_points]
    return max(min(b_vals) - max(a_vals), min(a_vals) - max(b_vals), 0.0)


def component_group_gap_along_axis(
    group_a: Sequence[AeroComponent],
    group_b: Sequence[AeroComponent],
    axis: Vec3,
) -> float:
    u = v_unit(axis)
    a_values = [
        v_dot(point, u)
        for component in group_a
        for point in stl_points(component.triangles)
    ]
    b_values = [
        v_dot(point, u)
        for component in group_b
        for point in stl_points(component.triangles)
    ]
    if not a_values or not b_values:
        return 0.0
    return max(
        min(b_values) - max(a_values),
        min(a_values) - max(b_values),
        0.0,
    )


def arrange_collision_convergence_initial_gap(
    pair: Tuple[AeroComponent, AeroComponent],
    components: Optional[Sequence[AeroComponent]] = None,
) -> Vec3:
    moving, stationary = collision_convergence_moving_and_stationary(pair)
    axis = collision_convergence_approach_axis(pair)

    moving_group = collision_source_group(moving, components)
    stationary_group = collision_source_group(stationary, components)
    moving_points = [
        point
        for member in moving_group
        for point in stl_points(member.triangles)
    ]
    stationary_points = [
        point
        for member in stationary_group
        for point in stl_points(member.triangles)
    ]
    if not moving_points or not stationary_points:
        return (0.0, 0.0, 0.0)

    def points_center(points: Sequence[Vec3]) -> Vec3:
        return (
            0.5 * (min(point[0] for point in points) + max(point[0] for point in points)),
            0.5 * (min(point[1] for point in points) + max(point[1] for point in points)),
            0.5 * (min(point[2] for point in points) + max(point[2] for point in points)),
        )

    moving_center = points_center(moving_points)
    stationary_center = points_center(stationary_points)
    center_correction = v_sub(stationary_center, moving_center)
    transverse_correction = v_sub(
        center_correction,
        v_mul(axis, v_dot(center_correction, axis)),
    )

    moving_max = max(v_dot(point, axis) for point in moving_points)
    stationary_min = min(v_dot(point, axis) for point in stationary_points)
    current_gap = stationary_min - moving_max
    longitudinal_correction = v_mul(axis, current_gap - COLLISION_INITIAL_GAP_M)
    translation = v_add(transverse_correction, longitudinal_correction)
    if v_norm(translation) <= 1e-12:
        return (0.0, 0.0, 0.0)
    for member in moving_group:
        move_component_rigidly(member, translation, None, 0.0, member.cofr)
        member.total_translation = v_add(member.total_translation, translation)
    return translation


def write_collision_convergence_log_header(path: Path, pair: Tuple[AeroComponent, AeroComponent]) -> None:
    moving, stationary = collision_convergence_moving_and_stationary(pair)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Prescribed impact convergence log\n"
        "# speed is impactor speed; the stationary target is not translated by the driver.\n"
        f"# pair={pair[0].patch},{pair[1].patch}\n"
        f"# moving={moving.patch}, stationary={stationary.patch}\n"
        "step\tactive\tmoving_patch\tstationary_patch\timpactor_speed_mps\taxis_x\taxis_y\taxis_z\tgap_before_m\tgap_after_m\trequested_m\tdmoving_m\tdstationary_m\tsweep_clamped\n"
    )


def append_collision_convergence_stop(path: Path, step: int, reason: str) -> None:
    with path.open("a") as f:
        f.write(f"# stopped_after_step={step}, reason={reason}\n")


def apply_collision_convergence_step(
    pair: Tuple[AeroComponent, AeroComponent],
    step: int,
    dt: float,
    log_path: Path,
    approach_axis: Optional[Vec3] = None,
    components: Optional[Sequence[AeroComponent]] = None,
) -> Tuple[Vec3, Vec3, Optional[SweptCollisionContact]]:
    moving, stationary = collision_convergence_moving_and_stationary(pair)
    axis = v_unit(approach_axis) if approach_axis is not None else collision_convergence_approach_axis(pair)
    moving_group = collision_source_group(moving, components)
    stationary_group = collision_source_group(stationary, components)

    gap_before = component_group_gap_along_axis(
        moving_group,
        stationary_group,
        axis,
    )
    requested_distance = COLLISION_CONVERGENCE_SPEED_MPS * max(dt, 0.0)
    applied_distance = requested_distance
    sweep_clamped = False
    swept_contact: Optional[SweptCollisionContact] = None
    if COLLISION_SWEEP_CLAMPING:
        earliest_hit: Optional[
            Tuple[
                float,
                AeroComponent,
                AeroComponent,
                Tuple[float, Vec3, Vec3, int],
            ]
        ] = None
        for moving_member in moving_group:
            for stationary_member in stationary_group:
                candidate_hit = swept_mesh_contact(
                    moving_member,
                    stationary_member,
                    axis,
                    requested_distance,
                )
                if candidate_hit is None:
                    continue
                candidate = (
                    candidate_hit[0],
                    moving_member,
                    stationary_member,
                    candidate_hit,
                )
                if earliest_hit is None or candidate[0] < earliest_hit[0]:
                    earliest_hit = candidate
        if earliest_hit is not None:
            _distance, contact_moving, contact_stationary, mesh_hit = earliest_hit
            travel_to_contact, contact_point, contact_normal, manifold_points = mesh_hit
            contact_normal = stable_collision_normal(
                contact_moving,
                contact_stationary,
                contact_normal,
                preserve_detected_orientation=True,
            )
            contact_time = travel_to_contact / max(
                COLLISION_CONVERGENCE_SPEED_MPS,
                1e-9,
            )
            post_contact_time = max(0.0, dt - contact_time)
            moving_contact_point = v_sub(
                contact_point,
                v_mul(axis, travel_to_contact),
            )
            contact_geometry = local_contact_geometry(
                contact_moving,
                contact_stationary,
                moving_contact_point,
                contact_point,
                contact_normal,
            )
            impact_velocity = v_mul(axis, COLLISION_CONVERGENCE_SPEED_MPS)
            normal_impact_speed = max(
                0.0,
                -v_dot(impact_velocity, contact_normal),
            )
            shell_response = thin_shell_impact_response(
                contact_moving,
                contact_stationary,
                normal_impact_speed,
                moving_contact_point,
                contact_normal,
            )
            if shell_response is None:
                impact_indentation = impact_contact_indentation(
                    contact_moving,
                    contact_stationary,
                    normal_impact_speed,
                    geometry=contact_geometry,
                )
                controlled_penetration = max(
                    (
                        impact_indentation
                        if impact_indentation > 0.0
                        else COLLISION_SWEEP_PENETRATION_M
                    ),
                    2.0 * COLLISION_MIN_OVERLAP_M,
                )
            else:
                controlled_penetration = max(
                    inferred_deformation_thickness(contact_stationary),
                    min(
                        shell_response.indentation,
                        impactor_contact_radius(
                            contact_moving,
                            moving_contact_point,
                            v_mul(contact_normal, -1.0),
                        ),
                    ),
                    2.0 * COLLISION_MIN_OVERLAP_M,
                )
            if shell_response is not None and shell_response.perforated:
                applied_distance = (
                    travel_to_contact
                    + shell_response.residual_speed * post_contact_time
                )
            else:
                # Close the contact by exactly the deformation depth. The
                # target's thin-plate deformation moves both skins together,
                # so this remains an entrance-side dent rather than allowing
                # a non-perforating body to cross an undeformed rear skin.
                applied_distance = travel_to_contact + controlled_penetration
            normal_penetration = controlled_penetration * max(
                0.0,
                -v_dot(axis, contact_normal),
            )
            sweep_clamped = True
            swept_contact = SweptCollisionContact(
                moving=contact_moving,
                stationary=contact_stationary,
                depth=max(normal_penetration, 2.0 * COLLISION_MIN_OVERLAP_M),
                normal=contact_normal,
                point=contact_point,
                travel_to_contact=travel_to_contact,
                approach_axis=axis,
                approach_penetration=controlled_penetration,
                manifold_points=manifold_points,
                perforated=bool(shell_response and shell_response.perforated),
                residual_speed=shell_response.residual_speed if shell_response else 0.0,
                absorbed_energy_j=shell_response.absorbed_energy_j if shell_response else 0.0,
                failure_mode=shell_response.failure_mode if shell_response else "elastic_contact",
                hole_radius=shell_response.hole_radius if shell_response else 0.0,
                contact_geometry=contact_geometry,
                post_contact_time_s=post_contact_time,
            )
    move_moving = v_mul(axis, applied_distance)
    move_stationary = (0.0, 0.0, 0.0)
    for member in moving_group:
        move_component_rigidly(member, move_moving, None, 0.0, member.cofr)
        if components is None:
            # Compatibility path for callers using a single prescribed body.
            member.linear_velocity = v_mul(axis, COLLISION_CONVERGENCE_SPEED_MPS)
        member.total_translation = v_add(member.total_translation, move_moving)
    for member in stationary_group:
        member.linear_velocity = (0.0, 0.0, 0.0)
        member.angular_velocity = (0.0, 0.0, 0.0)
    gap_after = component_group_gap_along_axis(
        moving_group,
        stationary_group,
        axis,
    )
    logged_moving = swept_contact.moving if swept_contact is not None else moving
    logged_stationary = (
        swept_contact.stationary if swept_contact is not None else stationary
    )

    with log_path.open("a") as f:
        f.write(
            f"{step}\t1\t{logged_moving.patch}\t{logged_stationary.patch}\t{COLLISION_CONVERGENCE_SPEED_MPS:.8g}\t"
            f"{axis[0]:.8g}\t{axis[1]:.8g}\t{axis[2]:.8g}\t"
            f"{gap_before:.8g}\t{gap_after:.8g}\t{requested_distance:.8g}\t"
            f"{v_norm(move_moving):.8g}\t{v_norm(move_stationary):.8g}\t{int(sweep_clamped)}\n"
        )
    return move_moving, move_stationary, swept_contact



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


def component_has_decoded_assembly_mate(component: AeroComponent) -> bool:
    """Return whether the component belongs to an imported mate graph.

    ``FREE`` is also the natural state of an unconnected Onshape occurrence, so
    the source is the reliable discriminator.  Local-STL and unmated Onshape
    parts must never be folded into an arbitrary assembly rigid body.
    """
    source = component.freedom.source
    return bool(
        source == "grounded"
        or source.startswith("mate:")
        or source.startswith("assembly-rigid-body")
        or component.mate_reference_occurrence
        or component.freedom.mate_reference_occurrence
    )


def connected_mate_component_groups(
    components: Sequence[AeroComponent],
) -> List[List[AeroComponent]]:
    """Find connected components of the decoded Onshape mate graph."""
    candidates = [
        component
        for component in components
        if component_has_decoded_assembly_mate(component)
    ]
    by_occurrence = {
        component.source_occurrence: component
        for component in candidates
        if component.source_occurrence
    }
    neighbours: Dict[int, Set[int]] = {id(component): set() for component in candidates}
    by_id = {id(component): component for component in candidates}
    for component in candidates:
        reference = (
            component.mate_reference_occurrence
            or component.freedom.mate_reference_occurrence
        )
        other = by_occurrence.get(reference) if reference else None
        if other is None or other is component:
            continue
        neighbours[id(component)].add(id(other))
        neighbours[id(other)].add(id(component))

    groups: List[List[AeroComponent]] = []
    unseen = set(neighbours)
    while unseen:
        start = unseen.pop()
        group_ids = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for neighbour in neighbours[current]:
                if neighbour in group_ids:
                    continue
                group_ids.add(neighbour)
                unseen.discard(neighbour)
                frontier.append(neighbour)
        groups.append([by_id[component_id] for component_id in group_ids])
    return groups


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

    # A local STL has no Onshape mate record, and an unmated assembly
    # occurrence is intentionally free.  Give either case explicit six-DOF
    # axes so it can participate in motion and collision resolution as its own
    # body rather than being silently stationary.
    for component in components:
        if (
            component.freedom.source == "unmated"
            and not component.freedom.translate_axes
            and not component.freedom.rotate_axes
        ):
            component.freedom = MotionFreedom(
                translate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
                rotate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
                mate_type="FREE",
                source="unmated",
            )

    mated_components = [
        component
        for component in components
        if component_has_decoded_assembly_mate(component)
    ]
    if not mated_components:
        lines.append(
            "No imported mates were decoded. Every component is retained as an independent free body."
        )
        (root_case / MOTION_POLICY_REPORT_NAME).write_text("\n".join(lines) + "\n")
        return lines

    root = select_assembly_root_component(mated_components)
    if root is None:
        lines.append("No root component could be selected; no assembly motion policy applied.")
        (root_case / MOTION_POLICY_REPORT_NAME).write_text("\n".join(lines) + "\n")
        return lines

    if not any(component_has_motion_freedom(component) for component in mated_components):
        lines.append(
            "No relative motion freedoms were decoded. Treating each connected mate group as a separate rigid free body."
        )
        for group_index, group_components in enumerate(
            connected_mate_component_groups(mated_components)
        ):
            group_root = select_assembly_root_component(group_components)
            if group_root is None:
                continue
            group_name = f"mate-group-{group_index}-{group_root.patch}"
            group_root.freedom = six_dof_motion_freedom("assembly-rigid-body-root")
            group_root.is_assembly_anchor = False
            group_root.motion_origin = group_root.cofr
            for component in group_components:
                component.rigid_body_group = group_name
                component.linear_velocity = (0.0, 0.0, 0.0)
                component.angular_velocity = (0.0, 0.0, 0.0)
                if component is group_root:
                    continue
                component.is_assembly_anchor = False
                component.freedom = MotionFreedom(
                    [], [], "FASTENED", "assembly-rigid-body-follower"
                )
            lines.append(
                f"Rigid mate group {group_name}: root={group_root.patch!r}, "
                f"member_count={len(group_components)}"
            )
        lines.append("")
        lines.append("Movable components after policy:")
        for c in components:
            axes = f"translate_axes={c.freedom.translate_axes}, rotate_axes={c.freedom.rotate_axes}"
            basis = motion_basis_debug(c)
            if c.freedom.source == "assembly-rigid-body-root":
                lines.append(f"- {c.patch}: rigid-body root, group={c.rigid_body_group}, mate_type={c.freedom.mate_type}, source={c.freedom.source}, {axes}, {basis}")
            elif c not in mated_components:
                lines.append(f"- {c.patch}: independent unmated body, mate_type={c.freedom.mate_type}, source={c.freedom.source}, {axes}, {basis}")
            else:
                lines.append(f"- {c.patch}: rigid-body follower, group={c.rigid_body_group}, mate_type={c.freedom.mate_type}, source={c.freedom.source}, {axes}, {basis}")
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
    "tungsten": 4.11e11,
    "wolfram": 4.11e11,
}


def component_deformation_enabled(component: AeroComponent) -> bool:
    if not ENABLE_NONRIGID_DEFORMATION or not component.triangles:
        return False
    if component_is_collision_impactor(component):
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
    if component.reference_thickness_m is not None and component.reference_thickness_m > 0.0:
        return component.reference_thickness_m
    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
    extents = sorted(value for value in (xmax - xmin, ymax - ymin, zmax - zmin) if value > 1e-9)
    if len(extents) == 3 and extents[0] < 0.1 * extents[2]:
        thickness = extents[0]
    else:
        thickness = max(0.001, min(0.05, 0.04 * max(component.lref, 1e-6)))
    component.reference_thickness_m = thickness
    return thickness


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
    if isinstance(component.collision_structural_state, HybridShellCollisionState):
        state = component.collision_structural_state
        return (
            state.shell_state.max_displacement_m,
            component.deformation_mean_m,
            inferred_deformation_young_modulus(component),
            inferred_deformation_thickness(component),
            len(state.shell_state.positions),
        )
    if isinstance(component.collision_structural_state, HybridFEMMPMCollisionState):
        state = component.collision_structural_state
        return (
            max(state.max_displacement_m, component.deformation_max_m),
            component.deformation_mean_m,
            inferred_deformation_young_modulus(component),
            inferred_deformation_thickness(component),
            len(state.solid_state.positions),
        )
    if isinstance(component.collision_structural_state, ExplicitShellState):
        state = component.collision_structural_state
        return (
            state.max_displacement_m,
            component.deformation_mean_m,
            inferred_deformation_young_modulus(component),
            inferred_deformation_thickness(component),
            len(state.positions),
        )
    if component_is_collision_impactor(component):
        return (
            component.deformation_max_m,
            component.deformation_mean_m,
            inferred_deformation_young_modulus(component),
            inferred_deformation_thickness(component),
            0,
        )
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
