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
    rel_speed, flow_unit = relative_air_speed_and_unit(component)
    q = 0.5 * RHO * rel_speed ** 2
    incoming_dir = v_mul(flow_unit, -1.0)
    origin = infer_motion_origin(component)
    total_force = (0.0, 0.0, 0.0)
    total_moment = (0.0, 0.0, 0.0)

    # Flat-plate turbulent/laminar skin-friction estimate. It is deliberately
    # small compared with pressure drag but prevents perfectly edge-on plates from
    # receiving exactly zero aerodynamic load.
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
        total_force = v_mul(flow_unit, SURFACE_LOAD_MIN_FORCE_N)
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


def force_from_coefficients(component: AeroComponent, coeffs: Dict[str, float]) -> Vec3:
    q = 0.5 * RHO * abs(VELOCITY) ** 2
    cd = coeffs.get("Cd", 0.0)
    cs = coeffs.get("Cs", 0.0)
    cl = coeffs.get("Cl", 0.0)
    return (
        cd * q * component.aref,
        cs * q * component.aref,
        cl * q * component.aref,
    )


def moment_from_coefficients(component: AeroComponent, coeffs: Dict[str, float]) -> Vec3:
    q = 0.5 * RHO * abs(VELOCITY) ** 2
    scale = q * component.aref * component.lref
    return (
        coeffs.get("CmRoll", 0.0) * scale,
        coeffs.get("CmPitch", 0.0) * scale,
        coeffs.get("CmYaw", 0.0) * scale,
    )


def update_component_motion(component: AeroComponent, coeffs: Dict[str, float], dt: float, load_override: Optional[Tuple[Vec3, Vec3]] = None) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    if load_override is not None:
        force, foam_moment = load_override
    else:
        force = v_mul(force_from_coefficients(component, coeffs), MOTION_FORCE_GAIN)
        foam_moment = v_mul(moment_from_coefficients(component, coeffs), MOTION_MOMENT_GAIN)
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
    lever_arm = v_sub(component.cofr, motion_origin)
    if load_override is not None:
        # surface_pressure_load already integrates moment about the motion origin
        lever_moment = (0.0, 0.0, 0.0)
    else:
        lever_moment = v_cross(lever_arm, force) if USE_FORCE_LEVER_ARM_TORQUE else (0.0, 0.0, 0.0)
    total_moment = v_add(foam_moment, lever_moment)
    if load_override is not None:
        total_moment = hinge_torque_fallback(component, total_moment)

    allowed_moment = project_vector_on_axes(total_moment, free.rotate_axes)
    angular_accel = v_mul(allowed_moment, 1.0 / max(component.inertia, 1e-12))
    new_av = v_add(component.angular_velocity, v_mul(angular_accel, dt))
    angular_decay = math.exp(-max(component.material.angular_damping_per_kg, 0.0) * dt / max(component.mass, 1e-9))
    new_av = v_mul(new_av, angular_decay)
    rotation_vector_step = clamp_vector_magnitude(v_mul(new_av, dt), MAX_ROTATION_PER_STEP_RAD)

    rotation_angle = v_norm(rotation_vector_step)
    rotation_axis = v_unit(rotation_vector_step) if rotation_angle > 1e-12 else None

    component.triangles = move_triangles(component.triangles, translation_step, rotation_axis, rotation_angle, motion_origin)
    component.cofr = v_add(component.cofr, translation_step)
    # If the component rotated about an off-centre hinge, update CofR by rotating it too.
    if rotation_axis is not None:
        component.cofr = rotate_point_around_axis(component.cofr, motion_origin, rotation_axis, rotation_angle)
    component.linear_velocity = new_lv
    component.angular_velocity = new_av
    component.total_translation = v_add(component.total_translation, translation_step)
    component.total_rotation = v_add(component.total_rotation, rotation_vector_step)

    return force, total_moment, translation_step, rotation_vector_step


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
    component.triangles = move_triangles(component.triangles, translation, None, 0.0, component.cofr)
    component.cofr = v_add(component.cofr, translation)
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
    component.triangles = move_triangles(component.triangles, (0.0, 0.0, 0.0), best_axis, angle, origin)
    component.cofr = rotate_point_around_axis(component.cofr, origin, best_axis, angle)
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
        component.angular_velocity = clamp_vector_magnitude(v_add(component.angular_velocity, v_mul(angular_impulse, 1.0 / max(component.inertia, 1e-12))), COLLISION_MAX_ANGULAR_SPEED_RAD_S)


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

    if not ANCHOR_ASSEMBLY_ROOT:
        lines.append("Root anchoring disabled. The whole assembly may accelerate/decelerate as one free body.")
        (root_case / MOTION_POLICY_REPORT_NAME).write_text("\n".join(lines) + "\n")
        return lines

    root = select_assembly_root_component(components)
    if root is None:
        lines.append("No root component could be selected; no anchor applied.")
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


def write_motion_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Quasi-dynamic assembly motion log\n"
        f"# dt={MOTION_DT:g} s, default_mass={DEFAULT_PART_MASS_KG:g} kg, default_inertia={DEFAULT_PART_INERTIA_KGM2:g} kg m^2\n"
        "# This is a rigid-body approximation driven by OpenFOAM forces.dat/log loads, with panel fallback only if OpenFOAM loads are unavailable.\n"
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




