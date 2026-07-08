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

def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a: Vec3) -> float:
    return math.sqrt(max(v_dot(a, a), 0.0))


def v_unit(a: Vec3, fallback: Vec3 = (0.0, 0.0, 1.0)) -> Vec3:
    n = v_norm(a)
    if n <= 1e-12:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def freestream_air_velocity() -> Vec3:
    """Air velocity in the selected inertial interpretation.

    OpenFOAM still uses the equivalent wind-tunnel frame for numerical stability,
    but the panel fallback and diagnostics can interpret the same physics as the
    body moving through still air.
    """
    if BODY_MOVING_THROUGH_STILL_AIR:
        return (0.0, 0.0, 0.0)
    return (VELOCITY, 0.0, 0.0)


def component_world_velocity(component: "AeroComponent") -> Vec3:
    if BODY_MOVING_THROUGH_STILL_AIR:
        return v_add(BODY_WORLD_VELOCITY, component.linear_velocity)
    return component.linear_velocity


def relative_air_velocity(component: "AeroComponent") -> Vec3:
    # Velocity of air as seen by this component. For BODY_MOVING_THROUGH_STILL_AIR
    # with BODY_WORLD_VELOCITY=-VELOCITY, this equals the original freestream when
    # the part has no relative motion, but it changes as parts move/collide.
    return v_sub(freestream_air_velocity(), component_world_velocity(component))


def relative_air_speed_and_unit(component: "AeroComponent") -> Tuple[float, Vec3]:
    rel = relative_air_velocity(component)
    speed = v_norm(rel)
    if speed <= 1e-9:
        return 0.0, flow_unit_vector()
    return speed, v_mul(rel, 1.0 / speed)


def project_vector_on_axes(vector: Vec3, axes: Sequence[Vec3]) -> Vec3:
    if not axes:
        return (0.0, 0.0, 0.0)
    out = (0.0, 0.0, 0.0)
    for axis in axes:
        u = v_unit(axis)
        out = v_add(out, v_mul(u, v_dot(vector, u)))
    return out


def clamp_vector_magnitude(vector: Vec3, max_mag: float) -> Vec3:
    mag = v_norm(vector)
    if mag <= max_mag or mag <= 1e-12:
        return vector
    return v_mul(vector, max_mag / mag)


def mat_identity() -> Matrix4:
    return (1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0)


def mat_apply(m: Sequence[float], p: Vec3) -> Vec3:
    # Onshape transform arrays are row-major 4x4 matrices.
    if len(m) >= 16:
        return (
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
        )
    return p


def mat_apply_direction(m: Sequence[float], direction: Vec3) -> Vec3:
    # Apply only the rotation block of a row-major 4x4 transform.
    if len(m) >= 16:
        return (
            m[0] * direction[0] + m[1] * direction[1] + m[2] * direction[2],
            m[4] * direction[0] + m[5] * direction[1] + m[6] * direction[2],
            m[8] * direction[0] + m[9] * direction[1] + m[10] * direction[2],
        )
    return direction


def rotate_point_around_axis(point: Vec3, origin: Vec3, axis: Vec3, angle: float) -> Vec3:
    if abs(angle) <= 1e-14:
        return point
    u = v_unit(axis)
    p = v_sub(point, origin)
    c = math.cos(angle)
    s = math.sin(angle)
    term1 = v_mul(p, c)
    term2 = v_mul(v_cross(u, p), s)
    term3 = v_mul(u, v_dot(u, p) * (1.0 - c))
    return v_add(origin, v_add(v_add(term1, term2), term3))


def transform_triangles(triangles: Iterable[Triangle], transform: Sequence[float]) -> List[Triangle]:
    out: List[Triangle] = []
    for normal, v1, v2, v3 in triangles:
        tv1 = mat_apply(transform, v1)
        tv2 = mat_apply(transform, v2)
        tv3 = mat_apply(transform, v3)
        # Recalculate the normal after transform to avoid bad normals after movement.
        n = v_unit(v_cross(v_sub(tv2, tv1), v_sub(tv3, tv1)), normal)
        out.append((n, tv1, tv2, tv3))
    return out


def move_triangles(triangles: Iterable[Triangle], translation: Vec3, rotation_axis: Optional[Vec3], angle: float, origin: Vec3) -> List[Triangle]:
    out: List[Triangle] = []
    for _normal, v1, v2, v3 in triangles:
        pts = []
        for p in (v1, v2, v3):
            rp = rotate_point_around_axis(p, origin, rotation_axis or (0.0, 0.0, 1.0), angle) if rotation_axis else p
            pts.append(v_add(rp, translation))
        n = v_unit(v_cross(v_sub(pts[1], pts[0]), v_sub(pts[2], pts[0])), (0.0, 0.0, 1.0))
        out.append((n, pts[0], pts[1], pts[2]))
    return out


# ------------------------- OpenFOAM writers -------------------------

