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

CFD_PRESETS = {
    "debug": {
        "CFD_ITERATIONS": "25",
        "CFD_WRITE_INTERVAL": "5",
        "SOLVER_TIMEOUT_SECONDS": "600",
        "ASSEMBLY_DYNAMIC_STEPS": "1",
        "BASE_CELLS_PER_LENGTH": "4.5",
        "MIN_CELLS": "30,20,20",
        "SURFACE_REFINEMENT_MIN": "2",
        "SURFACE_REFINEMENT_MAX": "2",
        "REGION_REFINEMENT": "2",
        "MAX_LOCAL_CELLS": "700000",
        "MAX_GLOBAL_CELLS": "1000000",
        "ADD_BOUNDARY_LAYERS": "0",
        "U_SOLVER": "PBiCGStab",
        "U_PRECONDITIONER": "DILU",
        "U_MAX_ITER": "35",
        "U_REL_TOL": "0.25",
        "U_TOLERANCE": "1e-5",
        "P_MAX_ITER": "35",
        "P_REL_TOL": "0.12",
        "TURB_MAX_ITER": "20",
        "N_NON_ORTHOGONAL_CORRECTORS": "0",
        "CONSISTENT_SIMPLE": "0",
        "RELAX_U": "0.55",
        "RELAX_P": "0.35",
        "RESIDUAL_CONTROL_U": "1e-3",
        "RESIDUAL_CONTROL_P": "1e-3",
        "RESIDUAL_CONTROL_TURB": "1e-3",
    },
    "fast": {
        "CFD_ITERATIONS": "60",
        "CFD_WRITE_INTERVAL": "20",
        "SOLVER_TIMEOUT_SECONDS": "1500",
        "ASSEMBLY_DYNAMIC_STEPS": "",
        "BASE_CELLS_PER_LENGTH": "5.5",
        "MIN_CELLS": "40,28,28",
        "SURFACE_REFINEMENT_MIN": "2",
        "SURFACE_REFINEMENT_MAX": "3",
        "REGION_REFINEMENT": "3",
        "MAX_LOCAL_CELLS": "2000000",
        "MAX_GLOBAL_CELLS": "3000000",
        "ADD_BOUNDARY_LAYERS": "0",
        "U_SOLVER": "PBiCGStab",
        "U_PRECONDITIONER": "DILU",
        "U_MAX_ITER": "45",
        "U_REL_TOL": "0.20",
        "U_TOLERANCE": "1e-5",
        "P_MAX_ITER": "45",
        "P_REL_TOL": "0.10",
        "TURB_MAX_ITER": "25",
        "N_NON_ORTHOGONAL_CORRECTORS": "0",
        "CONSISTENT_SIMPLE": "0",
        "RELAX_U": "0.50",
        "RELAX_P": "0.30",
        "RESIDUAL_CONTROL_U": "7e-4",
        "RESIDUAL_CONTROL_P": "7e-4",
        "RESIDUAL_CONTROL_TURB": "7e-4",
    },
    "balanced": {
        "CFD_ITERATIONS": "140",
        "CFD_WRITE_INTERVAL": "20",
        "SOLVER_TIMEOUT_SECONDS": "3000",
        "ASSEMBLY_DYNAMIC_STEPS": "2",
        "BASE_CELLS_PER_LENGTH": "7.0",
        "MIN_CELLS": "48,32,32",
        "SURFACE_REFINEMENT_MIN": "4",
        "SURFACE_REFINEMENT_MAX": "5",
        "REGION_REFINEMENT": "4",
        "MAX_LOCAL_CELLS": "3200000",
        "MAX_GLOBAL_CELLS": "4000000",
        "ADD_BOUNDARY_LAYERS": "0",
        "U_SOLVER": "PBiCGStab",
        "U_PRECONDITIONER": "DILU",
        "U_MAX_ITER": "65",
        "U_REL_TOL": "0.15",
        "U_TOLERANCE": "5e-6",
        "P_MAX_ITER": "55",
        "P_REL_TOL": "0.07",
        "TURB_MAX_ITER": "35",
        "N_NON_ORTHOGONAL_CORRECTORS": "1",
        "CONSISTENT_SIMPLE": "0",
        "RELAX_U": "0.45",
        "RELAX_P": "0.25",
        "RESIDUAL_CONTROL_U": "3e-4",
        "RESIDUAL_CONTROL_P": "3e-4",
        "RESIDUAL_CONTROL_TURB": "3e-4",
    },
    "accurate": {
        "CFD_ITERATIONS": "220",
        "CFD_WRITE_INTERVAL": "25",
        "SOLVER_TIMEOUT_SECONDS": "5400",
        "ASSEMBLY_DYNAMIC_STEPS": "3",
        "BASE_CELLS_PER_LENGTH": "8.0",
        "MIN_CELLS": "60,40,40",
        "SURFACE_REFINEMENT_MIN": "4",
        "SURFACE_REFINEMENT_MAX": "5",
        "REGION_REFINEMENT": "5",
        "MAX_LOCAL_CELLS": "4000000",
        "MAX_GLOBAL_CELLS": "6000000",
        "ADD_BOUNDARY_LAYERS": "0",
        "U_SOLVER": "PBiCGStab",
        "U_PRECONDITIONER": "DILU",
        "U_MAX_ITER": "90",
        "U_REL_TOL": "0.10",
        "U_TOLERANCE": "1e-6",
        "P_MAX_ITER": "70",
        "P_REL_TOL": "0.05",
        "TURB_MAX_ITER": "50",
        "N_NON_ORTHOGONAL_CORRECTORS": "1",
        "CONSISTENT_SIMPLE": "1",
        "RELAX_U": "0.35",
        "RELAX_P": "0.20",
        "RESIDUAL_CONTROL_U": "1e-4",
        "RESIDUAL_CONTROL_P": "1e-4",
        "RESIDUAL_CONTROL_TURB": "1e-4",
    },
}

CFD_CONFIG = os.environ.get("CFD_CONFIG", os.environ.get("CFD_PRESET", "fast")).strip().lower() or "fast"
ALLOW_CFD_ENV_OVERRIDES = os.environ.get("ALLOW_CFD_ENV_OVERRIDES", "0").strip().lower() in {"1", "true", "yes", "on"}
if CFD_CONFIG == "custom":
    # Custom mode deliberately restores the old behaviour: individual CFD_*/solver
    # environment variables control the case.  This is useful for expert runs, but
    # it is not the default because stale shell exports caused multi-hour solves.
    CFD_PRESETS["custom"] = dict(CFD_PRESETS["fast"])
elif CFD_CONFIG not in CFD_PRESETS:
    print(f"WARNING: unknown CFD_CONFIG={CFD_CONFIG!r}; using fast. Valid presets: debug, fast, balanced, accurate, custom")
    CFD_CONFIG = "fast"


def cfg_default(name: str, fallback: str) -> str:
    """Return selected preset value unless explicit env overrides are enabled.

    v30 is intentionally strict by default.  This prevents an old export such as
    CFD_ITERATIONS=500 from silently overriding CFD_CONFIG=fast and launching a
    very long run.  Use CFD_CONFIG=custom or ALLOW_CFD_ENV_OVERRIDES=1 to restore
    per-variable environment overrides.
    """
    preset_value = CFD_PRESETS[CFD_CONFIG].get(name, fallback)
    if str(preset_value).strip() == "":
        preset_value = fallback
    if CFD_CONFIG == "custom" or ALLOW_CFD_ENV_OVERRIDES:
        value = os.environ.get(name, preset_value)
        if str(value).strip() == "":
            return fallback
        return value
    return preset_value


def cfg_bool(name: str, fallback: bool) -> bool:
    value = cfg_default(name, "1" if fallback else "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def cfg_float(name: str, fallback: str) -> float:
    return float(cfg_default(name, fallback))


def cfg_int(name: str, fallback: str) -> int:
    return int(float(cfg_default(name, fallback)))


# ------------------------- simple settings -------------------------

CASE_NAME = os.environ.get("CASE_NAME", "actual_model_case").strip() or "actual_model_case"
DOCKER_IMAGE = "microfluidica/openfoam:11"

VELOCITY = float(os.environ.get("VELOCITY", "-100.0"))  # m/s, flow is nominally along X. Magnitude is used for coefficients.
RHO = float(os.environ.get("RHO", "1.225"))                # kg/m^3
NU = float(os.environ.get("NU", "1.5e-5"))                # m^2/s, air-ish
# Accuracy-first default: simpleFoam needs enough SIMPLE iterations for pressure/force convergence.
# Override with CFD_ITERATIONS=1000 etc. for serious runs, or lower it for debugging.
ITERATIONS = cfg_int("CFD_ITERATIONS", "70")
CFD_SOLVER_MODE = os.environ.get("CFD_SOLVER_MODE", "steady").strip().lower() or "steady"
if CFD_SOLVER_MODE not in {"steady", "transient"}:
    print(f"WARNING: unknown CFD_SOLVER_MODE={CFD_SOLVER_MODE!r}; using steady")
    CFD_SOLVER_MODE = "steady"
TURBULENCE_MODEL = os.environ.get("TURBULENCE_MODEL", "kOmegaSST").strip()
TURBULENCE_INTENSITY = float(os.environ.get("TURBULENCE_INTENSITY", "0.05"))
TURBULENCE_LENGTH_SCALE = float(os.environ.get("TURBULENCE_LENGTH_SCALE", "0.03"))
COEFFICIENT_LOG_NAME = "aero_coefficients.txt"
MOTION_LOG_NAME = "assembly_motion_log.txt"
MATE_REPORT_NAME = "assembly_mate_report.txt"

# Onshape STL export in metres. If your Onshape model is designed in mm,
# Onshape still exports with units=meter here, so SCALE should normally stay 1.0.
SCALE = 1.0
ONSHAPE_UNITS = "meter"
ONSHAPE_STL_MODE = "text"
ONSHAPE_EXPORT_SCALE = 1.0
# Optional high-resolution STL export controls. Onshape's UI exposes custom STL
# resolution using angular deviation, chordal tolerance, and minimum facet width.
# The exact API support varies by endpoint/account, so these are ONLY sent when
# explicitly set in the environment. Smaller tolerances increase STL triangle
# count, which improves surface detail and the compact preview, but also slows
# meshing.
ONSHAPE_STL_RESOLUTION = os.environ.get("ONSHAPE_STL_RESOLUTION", "").strip()
ONSHAPE_ANGULAR_DEVIATION = os.environ.get("ONSHAPE_ANGULAR_DEVIATION", "").strip()
ONSHAPE_CHORDAL_TOLERANCE = os.environ.get("ONSHAPE_CHORDAL_TOLERANCE", "").strip()
ONSHAPE_MIN_FACET_WIDTH = os.environ.get("ONSHAPE_MIN_FACET_WIDTH", "").strip()
ONSHAPE_API_VERSION = os.environ.get("ONSHAPE_API_VERSION", "v14").strip() or "v14"
# Assembly export policy. v6 tries very hard to export true per-occurrence geometry
# first. A merged fallback is kept as an emergency visualization mode, but it
# cannot produce real relative part motion.
STRICT_OCCURRENCE_EXPORT = os.environ.get("STRICT_OCCURRENCE_EXPORT", "1").strip().lower() not in {"0", "false", "no", "off"}
ALLOW_MERGED_ASSEMBLY_FALLBACK = os.environ.get("ALLOW_MERGED_ASSEMBLY_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
OCCURRENCE_EXPORT_REPORT_NAME = "assembly_occurrence_export_report.txt"

# Runtime ETA. The first estimate uses saved calibration from previous runs;
# after each completed CFD/motion step the estimate is updated from real measured time.
ETA_CALIBRATION_FILE = Path(os.environ.get("ETA_CALIBRATION_FILE", ".cfd_eta_calibration.json"))
ETA_DEFAULT_STEP_SECONDS = float(os.environ.get("ETA_DEFAULT_STEP_SECONDS", "1800"))
ETA_EMA_ALPHA = float(os.environ.get("ETA_EMA_ALPHA", "0.35"))

def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def flow_is_positive_x() -> bool:
    """True when the imposed freestream velocity points from low X to high X.

    This matters because the original script always placed the inlet on the
    low-X boundary. With VELOCITY=-100 that boundary was actually downstream,
    so the solver could converge to almost-uniform p/Cp and ParaView looked
    completely blue.
    """
    return VELOCITY >= 0.0


def flow_unit_vector() -> Tuple[float, float, float]:
    if VELOCITY >= 0:
        return (1.0, 0.0, 0.0)
    return (-1.0, 0.0, 0.0)


def incoming_unit_vector() -> Tuple[float, float, float]:
    # Direction from which the air arrives at a surface.
    ux, uy, uz = flow_unit_vector()
    return (-ux, -uy, -uz)


# ParaView root time-series export.  When assembly motion uses separate
# OpenFOAM cases under motion_steps/step_XXX, this merges each solved step
# back into actual_model_case/<time>/ so one root case.foam shows all motion
# frames in ParaView.
PARAVIEW_ROOT_TIMESERIES = env_bool("PARAVIEW_ROOT_TIMESERIES", True)
PARAVIEW_TIME_MODE = os.environ.get("PARAVIEW_TIME_MODE", "step").strip().lower()  # "step" or "seconds"
PARAVIEW_KEEP_LATEST_MIRROR = env_bool("PARAVIEW_KEEP_LATEST_MIRROR", False)
# v13: a PVD collection is the most reliable single-file ParaView view when
# each motion step is remeshed separately. One root case.foam is still created,
# but this file preserves time-varying mesh topology/patches much more reliably.
PARAVIEW_PVD_TIMESERIES = env_bool("PARAVIEW_PVD_TIMESERIES", True)
PARAVIEW_PVD_INCLUDE_BOUNDARY = env_bool("PARAVIEW_PVD_INCLUDE_BOUNDARY", True)
PARAVIEW_PRESSURE_DIAGNOSTICS = env_bool("PARAVIEW_PRESSURE_DIAGNOSTICS", True)
PARAVIEW_PVD_NAME = os.environ.get("PARAVIEW_PVD_NAME", "paraview_motion_timeseries.pvd")
PRESSURE_RANGE_REPORT_NAME = "pressure_range_report.txt"
# v16 visualization fix: OpenFOAM wall boundary files often show fixed/wall-function
# values, so pPa/Cp/k/nut can look flat on the surface even when the solved cell
# field has variation. Sample wall patches from the solved volume field and put
# those VTKs first in the PVD animation.
PARAVIEW_SAMPLED_WALL_SURFACES = env_bool("PARAVIEW_SAMPLED_WALL_SURFACES", True)
PARAVIEW_PVD_PREFER_SAMPLED_SURFACES = env_bool("PARAVIEW_PVD_PREFER_SAMPLED_SURFACES", True)
SAMPLED_SURFACE_FUNCTION_NAME = os.environ.get("SAMPLED_SURFACE_FUNCTION_NAME", "sampledWallSurfaces")
SAMPLED_SURFACE_FORMAT = os.environ.get("SAMPLED_SURFACE_FORMAT", "vtk")
SAMPLED_SURFACE_FIELDS = tuple(
    f.strip() for f in os.environ.get("SAMPLED_SURFACE_FIELDS", "p pPa Cp U k omega nut").split() if f.strip()
)
PARAVIEW_CREATE_PANEL_PREVIEW = env_bool("PARAVIEW_CREATE_PANEL_PREVIEW", True)
PANEL_PREVIEW_PVD_NAME = os.environ.get("PANEL_PREVIEW_PVD_NAME", "paraview_panel_aero_preview.pvd")
# v17: ParaView can crash when a .pvd directly references many mixed OpenFOAM
# legacy VTK files from separate remeshed cases.  The main PVD is now a
# crash-resistant surface animation: one clean combined POLYDATA VTK per motion
# step.  Raw CFD VTKs are still written, but not used by the main PVD by default.
PARAVIEW_SAFE_COMBINED_SURFACE_PVD = env_bool("PARAVIEW_SAFE_COMBINED_SURFACE_PVD", True)
# v21: PVD collections in ParaView expect XML VTK datasets such as .vtp/.vtu.
# Earlier compact previews used legacy .vtk POLYDATA; ParaView can open those
# directly, but vtkPVDReader cannot reliably determine their type through a .pvd.
# Therefore the main time-series preview now writes XML PolyData (.vtp).
COMBINED_SURFACE_VTK_NAME = os.environ.get("COMBINED_SURFACE_VTK_NAME", "combined_moving_surfaces.vtp")
PARAVIEW_RAW_CFD_PVD_NAME = os.environ.get("PARAVIEW_RAW_CFD_PVD_NAME", "paraview_raw_cfd_timeseries.pvd")

# v19 storage saver: by default, do NOT keep actual_model_case/motion_steps/.
# Each CFD step is solved in a temporary working directory, then only the
# compact/root outputs needed for ParaView and logs are copied into actual_model_case.
# Set SAVE_MOTION_STEPS=1 only if you need full per-step OpenFOAM cases for debugging.
SAVE_MOTION_STEPS = env_bool("SAVE_MOTION_STEPS", False)
KEEP_STEP_DEBUG_REPORTS = env_bool("KEEP_STEP_DEBUG_REPORTS", True)
KEEP_FINAL_GEOMETRY_CASE = env_bool("KEEP_FINAL_GEOMETRY_CASE", False)
STEP_DEBUG_REPORT_DIR_NAME = os.environ.get("STEP_DEBUG_REPORT_DIR_NAME", "step_reports")
ROOT_PANEL_PREVIEW_DIR_NAME = os.environ.get("ROOT_PANEL_PREVIEW_DIR_NAME", "panel_preview_timeseries")
# v22: prefer real sampled CFD surface data over the panel-preview colours. The
# panel preview is robust but necessarily blocky because it is driven by STL
# face normals. Sampled CFD surfaces are generated from the solved OpenFOAM field
# and converted to XML .vtp for the main .pvd animation.
PARAVIEW_PREFER_CFD_SAMPLED_SURFACES = env_bool("PARAVIEW_PREFER_CFD_SAMPLED_SURFACES", True)
CFD_SAMPLED_SURFACE_VTP_NAME = os.environ.get("CFD_SAMPLED_SURFACE_VTP_NAME", "combined_cfd_sampled_surfaces.vtp")
ROOT_CFD_SAMPLED_DIR_NAME = os.environ.get("ROOT_CFD_SAMPLED_DIR_NAME", "cfd_sampled_surface_timeseries")
CFD_SAMPLED_PREVIEW_DIR_NAME = os.environ.get("CFD_SAMPLED_PREVIEW_DIR_NAME", "cfd_sampled_preview")
# When CFD sampled files are absent, the fallback panel preview can be subdivided
# to improve visual smoothness. This does not create new physics; it only reduces
# the blocky look of large STL facets.
PANEL_PREVIEW_SUBDIVISIONS = max(1, int(os.environ.get("PANEL_PREVIEW_SUBDIVISIONS", "1")))

# v23 strict storage mode.  The heavy root OpenFOAM time-series is disabled by
# default because copying polyMesh+fields for every motion step can easily
# exceed several GB.  The main animation is the compact .pvd/.vtp surface
# time-series, while only start/final geometry snapshots are kept.
STORAGE_SAVER_MODE = env_bool("STORAGE_SAVER_MODE", True)
CASE_STORAGE_LIMIT_GB = float(os.environ.get("CASE_STORAGE_LIMIT_GB", "5"))
CASE_STORAGE_LIMIT_BYTES = int(CASE_STORAGE_LIMIT_GB * 1024**3)
ROOT_OPENFOAM_TIMESERIES = env_bool("ROOT_OPENFOAM_TIMESERIES", False)
RUN_FULL_VTK_EXPORT = env_bool("RUN_FULL_VTK_EXPORT", False)
# v24: even in storage-saver mode, export ONLY aircraft/part patches to VTK.
# This is small compared with full volume VTK, and gives the high-definition
# sampled/solved surface data needed for the main .pvd instead of falling back
# to coarse panel_preview_surfaces.
RUN_PATCH_ONLY_VTK_EXPORT = env_bool("RUN_PATCH_ONLY_VTK_EXPORT", True)
STORE_START_FINAL_GEOMETRY = env_bool("STORE_START_FINAL_GEOMETRY", True)
START_GEOMETRY_DIR_NAME = os.environ.get("START_GEOMETRY_DIR_NAME", "start_geometry")
FINAL_MOVED_GEOMETRY_DIR_NAME = os.environ.get("FINAL_MOVED_GEOMETRY_DIR_NAME", "final_moved_geometry")
GEOMETRY_SNAPSHOT_FILE_NAME = os.environ.get("GEOMETRY_SNAPSHOT_FILE_NAME", "geometry.vtp")
# Minimal real-volume Stream Tracer export.  This keeps only one lightweight
# OpenFOAM case with the latest solved mesh and velocity field U.  It is much
# smaller than ROOT_OPENFOAM_TIMESERIES=1 or RUN_FULL_VTK_EXPORT=1, but still
# gives ParaView a proper 3D vector field for Stream Tracer.
PARAVIEW_MINIMAL_STREAM_TRACER_EXPORT = env_bool("PARAVIEW_MINIMAL_STREAM_TRACER_EXPORT", True)
STREAM_TRACER_CASE_DIR_NAME = os.environ.get("STREAM_TRACER_CASE_DIR_NAME", "stream_tracer_volume_case")
STREAM_TRACER_FIELD_NAME = os.environ.get("STREAM_TRACER_FIELD_NAME", "U")
ABORT_IF_CASE_OVER_BUDGET = env_bool("ABORT_IF_CASE_OVER_BUDGET", True)

# v16: make visualization fail-loud and avoid accidentally showing initial/uncomputed fields.
PARAVIEW_STRICT_LATEST_NONZERO = env_bool("PARAVIEW_STRICT_LATEST_NONZERO", True)
PARAVIEW_SAMPLE_UTILITY_FALLBACK = env_bool("PARAVIEW_SAMPLE_UTILITY_FALLBACK", False)
PARAVIEW_EXCLUDE_INITIAL_VTK = env_bool("PARAVIEW_EXCLUDE_INITIAL_VTK", True)
VISUALIZATION_VALIDATION_REPORT_NAME = "visualization_validation_report.txt"


# Assembly quasi-motion settings.
ASSEMBLY_DYNAMIC_STEPS = cfg_int("ASSEMBLY_DYNAMIC_STEPS", "2")
MOTION_DT = max(1e-9, float(os.environ.get("MOTION_DT", "0.02")))
DEFAULT_PART_MASS_KG = float(os.environ.get("DEFAULT_PART_MASS_KG", "0.05"))
DEFAULT_PART_INERTIA_KGM2 = float(os.environ.get("DEFAULT_PART_INERTIA_KGM2", "1e-4"))
MAX_TRANSLATION_PER_STEP = float(os.environ.get("MAX_TRANSLATION_PER_STEP", "0.02"))
MAX_ROTATION_PER_STEP_RAD = math.radians(float(os.environ.get("MAX_ROTATION_PER_STEP_DEG", "10")))
ENABLE_NONRIGID_DEFORMATION = env_bool("ENABLE_NONRIGID_DEFORMATION", True)
DEFORM_ANCHORED_COMPONENTS = env_bool("DEFORM_ANCHORED_COMPONENTS", True)
DEFORMATION_COMPONENT_NAME_CONTAINS = os.environ.get("DEFORMATION_COMPONENT_NAME_CONTAINS", "").strip().lower()
DEFORMATION_MATERIAL_NAME_CONTAINS = os.environ.get("DEFORMATION_MATERIAL_NAME_CONTAINS", "").strip().lower()
DEFORMATION_EXCLUDE_COMPONENT_NAME_CONTAINS = os.environ.get("DEFORMATION_EXCLUDE_COMPONENT_NAME_CONTAINS", "").strip().lower()
DEFORMATION_EXCLUDE_MATERIAL_NAME_CONTAINS = os.environ.get("DEFORMATION_EXCLUDE_MATERIAL_NAME_CONTAINS", "").strip().lower()
DEFORMATION_YOUNG_MODULUS_PA = float(os.environ.get("DEFORMATION_YOUNG_MODULUS_PA", "0"))
DEFORMATION_POISSON_RATIO = max(0.0, min(0.49, float(os.environ.get("DEFORMATION_POISSON_RATIO", "0.35"))))
DEFORMATION_THICKNESS_M = float(os.environ.get("DEFORMATION_THICKNESS_M", "0"))
DEFORMATION_GAIN = float(os.environ.get("DEFORMATION_GAIN", "1.0"))
DEFORMATION_RELAXATION = max(0.0, min(1.0, float(os.environ.get("DEFORMATION_RELAXATION", "0.35"))))
MAX_DEFORMATION_PER_STEP = float(os.environ.get("MAX_DEFORMATION_PER_STEP", "0.005"))
MAX_TOTAL_DEFORMATION = float(os.environ.get("MAX_TOTAL_DEFORMATION", "0.05"))
DEFORMATION_VERTEX_TOLERANCE_M = float(os.environ.get("DEFORMATION_VERTEX_TOLERANCE_M", "1e-9"))
DEFORMATION_LOG_NAME = "assembly_deformation_log.txt"
AERO_SOLVER_REPORT_NAME = "unsteady_fsi_gap_report.txt"
AERO_TRANSIENT_END_TIME = float(os.environ.get("AERO_TRANSIENT_END_TIME", "0.05"))
AERO_TRANSIENT_DELTA_T = float(os.environ.get("AERO_TRANSIENT_DELTA_T", "0.001"))
AERO_TRANSIENT_WRITE_INTERVAL = float(os.environ.get("AERO_TRANSIENT_WRITE_INTERVAL", "0.01"))
AERO_TRANSIENT_PURGE_WRITE = max(0, int(os.environ.get("AERO_TRANSIENT_PURGE_WRITE", "3")))
AERO_TRANSIENT_MAX_CO = float(os.environ.get("AERO_TRANSIENT_MAX_CO", "1.0"))
AERO_TRANSIENT_MAX_DELTA_T = float(os.environ.get("AERO_TRANSIENT_MAX_DELTA_T", str(AERO_TRANSIENT_DELTA_T)))
AERO_TRANSIENT_OUTER_CORRECTORS = max(1, int(os.environ.get("AERO_TRANSIENT_OUTER_CORRECTORS", "2")))
AERO_TRANSIENT_PRESSURE_CORRECTORS = max(1, int(os.environ.get("AERO_TRANSIENT_PRESSURE_CORRECTORS", "2")))
AERO_TRANSIENT_NON_ORTHOGONAL_CORRECTORS = max(0, int(os.environ.get("AERO_TRANSIENT_NON_ORTHOGONAL_CORRECTORS", "1")))
AERO_TRANSIENT_MOMENTUM_PREDICTOR = os.environ.get("AERO_TRANSIENT_MOMENTUM_PREDICTOR", "1").strip().lower() not in {"0", "false", "no", "off"}
AERO_LOAD_RELAXATION = max(0.0, min(1.0, float(os.environ.get("AERO_LOAD_RELAXATION", "0.35"))))
AERO_USE_LOCAL_POINT_VELOCITY = os.environ.get("AERO_USE_LOCAL_POINT_VELOCITY", "1").strip().lower() not in {"0", "false", "no", "off"}
# Motion gains are deliberately explicit.  Keep them at 1 for physics-like runs;
# increase them to make very small aerodynamic movements visible in early tests.
MOTION_FORCE_GAIN = float(os.environ.get("MOTION_FORCE_GAIN", "1.0"))
MOTION_MOMENT_GAIN = float(os.environ.get("MOTION_MOMENT_GAIN", "1.0"))
# v29 collision and reference-frame controls.
# The CFD solve still uses the standard wind-tunnel/equivalent-freestream frame
# because that is the stable way to run a stationary OpenFOAM mesh. For one
# rigid body, "body moving through still air" and "air moving past the body"
# are Galilean-equivalent.  The BODY_MOVING_THROUGH_STILL_AIR mode records that
# interpretation and uses the corresponding relative-air velocity in the panel
# load fallback without translating the whole aircraft out of the mesh/domain.
AERO_REFERENCE_FRAME = os.environ.get("AERO_REFERENCE_FRAME", os.environ.get("AERO_FRAME", "body_moving_through_still_air")).strip().lower()
BODY_MOVING_THROUGH_STILL_AIR = AERO_REFERENCE_FRAME in {"body", "body_moving", "moving_body", "still_air", "body_moving_through_still_air"}
BODY_WORLD_VELOCITY = (-VELOCITY, 0.0, 0.0) if BODY_MOVING_THROUGH_STILL_AIR else (0.0, 0.0, 0.0)
ENABLE_PART_COLLISIONS = os.environ.get("ENABLE_PART_COLLISIONS", "1").strip().lower() in {"1", "true", "yes", "on"}
COLLISION_METHOD = os.environ.get("COLLISION_METHOD", "aabb").strip().lower()
COLLISION_MARGIN_M = float(os.environ.get("COLLISION_MARGIN_M", "0.0005"))
COLLISION_POSITION_CORRECTION = float(os.environ.get("COLLISION_POSITION_CORRECTION", "1.0"))
COLLISION_RESTITUTION = float(os.environ.get("COLLISION_RESTITUTION", "0.05"))
COLLISION_PRESCRIBED_IMPACT_RESTITUTION = max(
    0.0,
    float(os.environ.get("COLLISION_PRESCRIBED_IMPACT_RESTITUTION", "0")),
)
COLLISION_TANGENTIAL_DAMPING = float(os.environ.get("COLLISION_TANGENTIAL_DAMPING", "0.25"))
COLLISION_FRICTION_COEFFICIENT = float(os.environ.get("COLLISION_FRICTION_COEFFICIENT", "-1"))
COLLISION_MANIFOLD_TOLERANCE_M = max(
    1e-9,
    float(os.environ.get("COLLISION_MANIFOLD_TOLERANCE_M", "1e-5")),
)
COLLISION_MAX_PASSES = max(1, int(os.environ.get("COLLISION_MAX_PASSES", "6")))
COLLISION_MIN_OVERLAP_M = float(os.environ.get("COLLISION_MIN_OVERLAP_M", "1e-7"))
COLLISION_MAX_LINEAR_SPEED_MPS = float(os.environ.get("COLLISION_MAX_LINEAR_SPEED_MPS", "20"))
COLLISION_MAX_ANGULAR_SPEED_RAD_S = float(os.environ.get("COLLISION_MAX_ANGULAR_SPEED_RAD_S", "20"))
COLLISION_LOG_NAME = "assembly_collision_log.txt"
ENABLE_COLLISION_DEFORMATION = env_bool("ENABLE_COLLISION_DEFORMATION", True)
COLLISION_DEFORMATION_MODEL = os.environ.get("COLLISION_DEFORMATION_MODEL", "hertz").strip().lower()
COLLISION_DEFORMATION_GAIN = float(os.environ.get("COLLISION_DEFORMATION_GAIN", "1.0"))
COLLISION_DEFORMATION_RADIUS_FACTOR = float(os.environ.get("COLLISION_DEFORMATION_RADIUS_FACTOR", "1.0"))
COLLISION_DEFORMATION_MIN_RADIUS_M = float(os.environ.get("COLLISION_DEFORMATION_MIN_RADIUS_M", "0.002"))
COLLISION_MAX_CONTACT_DEFORMATION = float(os.environ.get("COLLISION_MAX_CONTACT_DEFORMATION", "0.01"))
COLLISION_EULERIAN_GRID_CELLS = max(4, int(os.environ.get("COLLISION_EULERIAN_GRID_CELLS", "12")))
COLLISION_EULERIAN_GRID_MIN_CELL_M = max(1e-9, float(os.environ.get("COLLISION_EULERIAN_GRID_MIN_CELL_M", "1e-4")))
COLLISION_EULERIAN_PENALTY_GAIN = max(0.0, float(os.environ.get("COLLISION_EULERIAN_PENALTY_GAIN", "1.0")))
COLLISION_MESH_REFINEMENT_TARGET_TRIANGLES = max(
    0,
    int(os.environ.get("COLLISION_MESH_REFINEMENT_TARGET_TRIANGLES", "2048")),
)
COLLISION_MESH_REFINEMENT_MAX_LEVELS = max(
    0,
    int(os.environ.get("COLLISION_MESH_REFINEMENT_MAX_LEVELS", "4")),
)
COLLISION_CONVERGENCE_SPEED_MPS = max(0.0, float(os.environ.get("COLLISION_CONVERGENCE_SPEED_MPS", "0")))
COLLISION_CONVERGENCE_COMPONENTS = os.environ.get("COLLISION_CONVERGENCE_COMPONENTS", "").strip()
COLLISION_CONVERGENCE_AXIS = os.environ.get("COLLISION_CONVERGENCE_AXIS", "auto").strip().lower()
COLLISION_INITIAL_GAP_M = max(0.0, float(os.environ.get("COLLISION_INITIAL_GAP_M", "0.05")))
COLLISION_CONVERGENCE_MOVING_COMPONENT = os.environ.get("COLLISION_CONVERGENCE_MOVING_COMPONENT", "first").strip().lower()
COLLISION_SWEEP_CLAMPING = env_bool("COLLISION_SWEEP_CLAMPING", True)
COLLISION_SWEEP_PENETRATION_M = float(os.environ.get("COLLISION_SWEEP_PENETRATION_M", "0.001"))
COLLISION_CONVERGENCE_STOP_AFTER_CONTACT = env_bool("COLLISION_CONVERGENCE_STOP_AFTER_CONTACT", True)
COLLISION_CONVERGENCE_LOG_NAME = "assembly_collision_convergence_log.txt"
COLLISION_DAMAGE_LOG_NAME = "assembly_collision_damage_log.txt"
COLLISION_DAMAGE_MIN_RESPONSE_TIME_S = max(
    0.0,
    float(os.environ.get("COLLISION_DAMAGE_MIN_RESPONSE_TIME_S", str(4.0 * MOTION_DT))),
)
# Collision output uses a fixed mesh by default.  Topology changes can make
# successive VTP frames structurally incompatible in ParaView, particularly at
# first contact.  Enable only for offline fracture experiments.
ENABLE_COLLISION_TOPOLOGY_CHANGES = env_bool("ENABLE_COLLISION_TOPOLOGY_CHANGES", False)
COLLISION_FRACTURE_MAX_SUBDIVISION_DEPTH = max(
    0,
    int(os.environ.get("COLLISION_FRACTURE_MAX_SUBDIVISION_DEPTH", "5")),
)
USE_FORCE_LEVER_ARM_TORQUE = os.environ.get("USE_FORCE_LEVER_ARM_TORQUE", "1").strip().lower() not in {"0", "false", "no", "off"}
AUTO_HINGE_ORIGINS = os.environ.get("AUTO_HINGE_ORIGINS", "1").strip().lower() not in {"0", "false", "no", "off"}
USE_ALL_COEFFS_FALLBACK = os.environ.get("USE_ALL_COEFFS_FALLBACK", "1").strip().lower() not in {"0", "false", "no", "off"}
JOINT_LIMIT_RESTITUTION = max(0.0, min(1.0, float(os.environ.get("JOINT_LIMIT_RESTITUTION", "0.0"))))
# v8 fallback: if OpenFOAM forceCoeffs are absent or all zero, estimate aerodynamic
# force directly from STL triangle projected area. This is much cruder than CFD,
# but it guarantees the motion loop has non-zero loads for hinge/mate testing.
# Accuracy-first default: do not invent aerodynamic loads unless explicitly requested.
# Set ENABLE_SURFACE_LOAD_FALLBACK=1 only as a diagnostic/motion-preview mode.
ENABLE_SURFACE_LOAD_FALLBACK = os.environ.get("ENABLE_SURFACE_LOAD_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}
SURFACE_LOAD_COEFF = float(os.environ.get("SURFACE_LOAD_COEFF", "1.0"))
SURFACE_LOAD_GAIN = float(os.environ.get("SURFACE_LOAD_GAIN", "1.0"))
SURFACE_LOAD_DOUBLE_SIDED = os.environ.get("SURFACE_LOAD_DOUBLE_SIDED", "1").strip().lower() not in {"0", "false", "no", "off"}
SURFACE_LOAD_MIN_FORCE_N = float(os.environ.get("SURFACE_LOAD_MIN_FORCE_N", "0.0"))
# If the STL pressure estimate produces almost no torque on a revolute/hinged
# component, add a small wind-hinge moment so motion is visible. This is a
# motion-preview approximation, not a validated aerodynamic hinge-moment model.
ENABLE_HINGE_TORQUE_FALLBACK = os.environ.get("ENABLE_HINGE_TORQUE_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
HINGE_TORQUE_COEFF = float(os.environ.get("HINGE_TORQUE_COEFF", "0.08"))
HINGE_TORQUE_MIN_NM = float(os.environ.get("HINGE_TORQUE_MIN_NM", "1e-5"))
FORCE_COEFF_DEBUG_REPORT_NAME = "force_coeff_debug_report.txt"
# v9 accuracy path: use OpenFOAM dimensional force/moment integration for motion.
# This is better than moving from coefficients because it uses the actual integrated
# pressure+viscous force in newtons and moment in N*m for each patch.
USE_OPENFOAM_FORCES_DAT = os.environ.get("USE_OPENFOAM_FORCES_DAT", "1").strip().lower() not in {"0", "false", "no", "off"}
RUN_POSTPROCESS_FORCE_OBJECTS = os.environ.get("RUN_POSTPROCESS_FORCE_OBJECTS", "0").strip().lower() not in {"0", "false", "no", "off"}
# If OpenFOAM writes a forces.dat/log entry but the vector is numerically zero,
# treat it as missing and allow the physically based STL/panel fallback to drive motion.
OPENFOAM_FORCE_ZERO_TOL = float(os.environ.get("OPENFOAM_FORCE_ZERO_TOL", "1e-12"))
# wallShearStress is useful but not required for pressure/motion, and it has been
# causing postProcess failures in the OpenFOAM-11 Docker image. Keep it off by default.
ENABLE_WALL_SHEAR_STRESS = env_bool("ENABLE_WALL_SHEAR_STRESS", False)
# Linear-solver controls. v28 caps per-equation inner iterations because the
# previous smoothSolver default hit 1000 iterations on Ux at every SIMPLE step,
# which turned a preview into a multi-hour run.  PBiCGStab+DILU is usually much
# faster for the vector U equation in this external-aero case.
U_SOLVER = cfg_default("U_SOLVER", "PBiCGStab").strip()
U_PRECONDITIONER = cfg_default("U_PRECONDITIONER", "DILU").strip()
U_SMOOTHER = cfg_default("U_SMOOTHER", "symGaussSeidel").strip()
U_TOLERANCE = cfg_default("U_TOLERANCE", "1e-5")
U_REL_TOL = cfg_default("U_REL_TOL", "0.20")
U_MAX_ITER = cfg_int("U_MAX_ITER", "45")
P_TOLERANCE = cfg_default("P_TOLERANCE", "1e-6")
P_REL_TOL = cfg_default("P_REL_TOL", "0.10")
P_MAX_ITER = cfg_int("P_MAX_ITER", "45")
TURB_TOLERANCE = cfg_default("TURB_TOLERANCE", "1e-5")
TURB_REL_TOL = cfg_default("TURB_REL_TOL", "0.20")
TURB_MAX_ITER = cfg_int("TURB_MAX_ITER", "25")
N_NON_ORTHOGONAL_CORRECTORS = cfg_int("N_NON_ORTHOGONAL_CORRECTORS", "0")
CONSISTENT_SIMPLE = cfg_bool("CONSISTENT_SIMPLE", False)
RELAX_P = cfg_default("RELAX_P", "0.30")
RELAX_U = cfg_default("RELAX_U", "0.50")
RELAX_K = cfg_default("RELAX_K", "0.50")
RELAX_OMEGA = cfg_default("RELAX_OMEGA", "0.50")
RESIDUAL_CONTROL = cfg_bool("RESIDUAL_CONTROL", True)
RESIDUAL_CONTROL_P = cfg_default("RESIDUAL_CONTROL_P", "7e-4")
RESIDUAL_CONTROL_U = cfg_default("RESIDUAL_CONTROL_U", "7e-4")
RESIDUAL_CONTROL_TURB = cfg_default("RESIDUAL_CONTROL_TURB", "7e-4")
CFD_WRITE_INTERVAL = max(1, min(ITERATIONS, cfg_int("CFD_WRITE_INTERVAL", "10")))
SOLVER_TIMEOUT_SECONDS = max(0, cfg_int("SOLVER_TIMEOUT_SECONDS", "1500"))
FORCES_LOG_NAME = "aero_forces.txt"
FORCE_LOAD_DEBUG_REPORT_NAME = "force_load_debug_report.txt"
# Debug override: useful when Onshape mate decoding marks everything fixed but you
# want to confirm that the per-part CFD/motion pipeline itself is working.
FORCE_NON_ROOT_COMPONENTS_FREE = os.environ.get("FORCE_NON_ROOT_COMPONENTS_FREE", "0").strip().lower() in {"1", "true", "yes", "on"}
MOTION_GEOMETRY_REPORT_NAME = "assembly_motion_geometry_report.txt"

# Relative assembly-motion policy.
# By default, anchor one main/root component to the wind-tunnel frame. This prevents
# the entire aircraft/object from simply accelerating downstream as one rigid body,
# and makes the motion solver show relative motion of loose, slider, revolute,
# cylindrical, planar, or ball-jointed parts.
ANCHOR_ASSEMBLY_ROOT = os.environ.get("ANCHOR_ASSEMBLY_ROOT", "1").strip().lower() not in {"0", "false", "no", "off"}
ASSEMBLY_ROOT_PATCH = os.environ.get("ASSEMBLY_ROOT_PATCH", "").strip()
ASSEMBLY_ROOT_NAME_CONTAINS = os.environ.get("ASSEMBLY_ROOT_NAME_CONTAINS", "").strip().lower()
ASSEMBLY_ROOT_MODE = os.environ.get("ASSEMBLY_ROOT_MODE", "largest_mass").strip().lower() or "largest_mass"
MOTION_POLICY_REPORT_NAME = "assembly_motion_policy_report.txt"

# Material/BOM behavior model.
# Onshape BOM material/mass columns are account/configuration dependent, so the
# code reads whatever columns it can find and falls back to these values.
USE_BOM_MATERIALS = os.environ.get("USE_BOM_MATERIALS", "1").strip().lower() not in {"0", "false", "no", "off"}
MATERIAL_REPORT_NAME = "assembly_material_report.txt"
ASSEMBLY_BOM_NAME = "assembly_bom.json"
DEFAULT_MATERIAL_NAME = os.environ.get("DEFAULT_MATERIAL_NAME", "unknown").strip() or "unknown"
DEFAULT_MATERIAL_DENSITY_KG_M3 = float(os.environ.get("DEFAULT_MATERIAL_DENSITY_KG_M3", "1000"))
DEFAULT_LINEAR_DAMPING_PER_KG = float(os.environ.get("DEFAULT_LINEAR_DAMPING_PER_KG", "0.20"))
DEFAULT_ANGULAR_DAMPING_PER_KG = float(os.environ.get("DEFAULT_ANGULAR_DAMPING_PER_KG", "0.05"))
MATERIAL_OVERRIDES_JSON = os.environ.get("MATERIAL_OVERRIDES_JSON", "").strip()
MATERIAL_OVERRIDES_FILE = os.environ.get("MATERIAL_OVERRIDES_FILE", "").strip()

# Mesh/visual fidelity. These defaults are higher than the early versions because
# coarse patch cells were producing blocky red/blue pressure previews. Override
# downward for fast debugging, upward for serious final runs.
def _tuple3_from_text(raw: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    parts = [x.strip() for x in str(raw).replace("x", ",").split(",") if x.strip()]
    if len(parts) != 3:
        return default
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return default


def _env_int_tuple3(name: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return _tuple3_from_text(raw, default)

BASE_CELLS_PER_LENGTH = cfg_float("BASE_CELLS_PER_LENGTH", "5.5" if STORAGE_SAVER_MODE else "7.0")
MIN_CELLS = _tuple3_from_text(cfg_default("MIN_CELLS", "40,28,28"), (40, 28, 28))
SURFACE_REFINEMENT = (
    cfg_int("SURFACE_REFINEMENT_MIN", "2" if STORAGE_SAVER_MODE else "3"),
    cfg_int("SURFACE_REFINEMENT_MAX", "3" if STORAGE_SAVER_MODE else "4"),
)
REGION_REFINEMENT = cfg_int("REGION_REFINEMENT", "3" if STORAGE_SAVER_MODE else "4")
MAX_LOCAL_CELLS = cfg_int("MAX_LOCAL_CELLS", "1200000" if STORAGE_SAVER_MODE else "2200000")
MAX_GLOBAL_CELLS = cfg_int("MAX_GLOBAL_CELLS", "1800000" if STORAGE_SAVER_MODE else "3200000")
N_CELLS_BETWEEN_LEVELS = cfg_int("N_CELLS_BETWEEN_LEVELS", "3" if STORAGE_SAVER_MODE else "2")

# Boundary-layer/prism layers give much more realistic wall pressure/skin-friction
# behaviour than pure castellated cells. If snappyHexMesh struggles, set
# ADD_BOUNDARY_LAYERS=0.
ADD_BOUNDARY_LAYERS = cfg_bool("ADD_BOUNDARY_LAYERS", False)
N_SURFACE_LAYERS = int(os.environ.get("N_SURFACE_LAYERS", "3"))
LAYER_EXPANSION_RATIO = float(os.environ.get("LAYER_EXPANSION_RATIO", "1.2"))
FINAL_LAYER_THICKNESS = float(os.environ.get("FINAL_LAYER_THICKNESS", "0.35"))
MIN_LAYER_THICKNESS = float(os.environ.get("MIN_LAYER_THICKNESS", "0.05"))

# Domain size relative to model length.
UPSTREAM_LENGTHS = 5.0
DOWNSTREAM_LENGTHS = 15.0
SIDE_LENGTHS = 6.0
NEAR_BODY_REFINEMENT_LENGTHS = 0.5

# ------------------------------------------------------------------

JSON_CONTENT_TYPE = "application/json;charset=UTF-8; qs=0.09"
JSON_ACCEPT = "application/json;charset=UTF-8; qs=0.09"
