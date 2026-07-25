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

Vec3 = Tuple[float, float, float]
Triangle = Tuple[Vec3, Vec3, Vec3, Vec3]  # normal, v1, v2, v3
Matrix4 = Tuple[float, ...]


@dataclass(frozen=True)
class OnshapeRef:
    base_url: str
    did: str
    wvm: str
    wvmid: str
    eid: str
    configuration: Optional[str] = None


@dataclass
class MotionFreedom:
    translate_axes: List[Vec3] = field(default_factory=list)
    rotate_axes: List[Vec3] = field(default_factory=list)
    mate_type: str = "FREE"
    source: str = "unmated"
    limits: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    mate_origin: Optional[Vec3] = None
    mate_reference_origin: Optional[Vec3] = None
    mate_reference_occurrence: Optional[str] = None
    mate_x_axis: Optional[Vec3] = None
    mate_y_axis: Optional[Vec3] = None
    mate_z_axis: Optional[Vec3] = None
    mate_reference_x_axis: Optional[Vec3] = None
    mate_reference_y_axis: Optional[Vec3] = None
    mate_reference_z_axis: Optional[Vec3] = None


@dataclass
class MaterialProperties:
    material_name: str = DEFAULT_MATERIAL_NAME
    density_kg_m3: float = DEFAULT_MATERIAL_DENSITY_KG_M3
    mass_kg: Optional[float] = None
    volume_m3: Optional[float] = None
    source: str = "default"
    linear_damping_per_kg: float = DEFAULT_LINEAR_DAMPING_PER_KG
    angular_damping_per_kg: float = DEFAULT_ANGULAR_DAMPING_PER_KG
    young_modulus_pa: Optional[float] = None
    poisson_ratio: Optional[float] = None
    thickness_m: Optional[float] = None
    yield_strength_pa: Optional[float] = None
    failure_strain: Optional[float] = None
    structural_source: str = "default"


@dataclass
class CollisionDamageState:
    contact_point: Vec3
    inward_direction: Vec3
    contact_radius_m: float
    current_depth_m: float
    permanent_depth_m: float
    current_hole_radius_m: float = 0.0
    target_hole_radius_m: float = 0.0
    response_time_s: float = 1e-6
    failure_mode: str = "elastic_contact"
    accumulated_energy_j: float = 0.0
    created_step: int = 0
    elapsed_s: float = 0.0


@dataclass
class AeroComponent:
    name: str
    patch: str
    triangles: List[Triangle]
    cofr: Vec3
    lref: float
    aref: float
    freedom: MotionFreedom = field(default_factory=MotionFreedom)
    material: MaterialProperties = field(default_factory=MaterialProperties)
    mass: float = DEFAULT_PART_MASS_KG
    inertia: float = DEFAULT_PART_INERTIA_KGM2
    linear_velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)
    total_translation: Vec3 = (0.0, 0.0, 0.0)
    total_rotation: Vec3 = (0.0, 0.0, 0.0)
    filtered_force: Vec3 = (0.0, 0.0, 0.0)
    filtered_moment: Vec3 = (0.0, 0.0, 0.0)
    aerodynamic_load_initialized: bool = False
    source_occurrence: Optional[str] = None
    motion_origin: Optional[Vec3] = None
    mate_origin: Optional[Vec3] = None
    mate_reference_origin: Optional[Vec3] = None
    mate_reference_occurrence: Optional[str] = None
    mate_x_axis: Optional[Vec3] = None
    mate_y_axis: Optional[Vec3] = None
    mate_z_axis: Optional[Vec3] = None
    mate_reference_x_axis: Optional[Vec3] = None
    mate_reference_y_axis: Optional[Vec3] = None
    mate_reference_z_axis: Optional[Vec3] = None
    is_assembly_anchor: bool = False
    deformation_reference_triangles: Optional[List[Triangle]] = None
    deformation_max_m: float = 0.0
    deformation_mean_m: float = 0.0
    collision_damage: List[CollisionDamageState] = field(default_factory=list)


# ------------------------- small math helpers -------------------------
