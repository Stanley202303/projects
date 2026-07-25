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

def write_block_mesh(case: Path, domain: Tuple[float, float, float, float, float, float], cells: Tuple[int, int, int]) -> None:
    x0, x1, y0, y1, z0, z1 = domain
    nx, ny, nz = cells

    # Low-X and high-X faces of the block. The inlet must be on the upstream
    # side of the domain. The old code always used the low-X face as inlet,
    # which is wrong for VELOCITY < 0 and is the main reason p/Cp stayed blue.
    low_x_face = "(0 4 7 3)"
    high_x_face = "(1 2 6 5)"
    if flow_is_positive_x():
        inlet_face = low_x_face
        outlet_face = high_x_face
    else:
        inlet_face = high_x_face
        outlet_face = low_x_face

    write(case / "system/blockMeshDict", f"""{foam_header("dictionary", "blockMeshDict")}
convertToMeters 1;

vertices
(
    ({x0:g} {y0:g} {z0:g})
    ({x1:g} {y0:g} {z0:g})
    ({x1:g} {y1:g} {z0:g})
    ({x0:g} {y1:g} {z0:g})
    ({x0:g} {y0:g} {z1:g})
    ({x1:g} {y0:g} {z1:g})
    ({x1:g} {y1:g} {z1:g})
    ({x0:g} {y1:g} {z1:g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ({inlet_face});
    }}

    outlet
    {{
        type patch;
        faces ({outlet_face});
    }}

    farfield
    {{
        type patch;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);
""")


def write_snappy(case: Path, location: Vec3, near_body_distance: float, patch_names: Sequence[str]) -> None:
    lx, ly, lz = location
    surf_min, surf_max = SURFACE_REFINEMENT

    geometry_entries = []
    refinement_surfaces = []
    refinement_regions = []
    for patch in patch_names:
        geometry_entries.append(f"""    {patch}
    {{
        type triSurfaceMesh;
        file \"{patch}.stl\";
    }}""")
        refinement_surfaces.append(f"""        {patch}
        {{
            level ({surf_min} {surf_max});
            patchInfo
            {{
                type wall;
            }}
        }}""")
        refinement_regions.append(f"""        {patch}
        {{
            mode distance;
            levels (({near_body_distance:g} {REGION_REFINEMENT}));
        }}""")

    write(case / "system/snappyHexMeshDict", f"""{foam_header("dictionary", "snappyHexMeshDict")}
castellatedMesh true;
snap            true;
addLayers       {str(ADD_BOUNDARY_LAYERS).lower()};

geometry
{{
{chr(10).join(geometry_entries)}
}}

castellatedMeshControls
{{
    maxLocalCells       {MAX_LOCAL_CELLS};
    maxGlobalCells      {MAX_GLOBAL_CELLS};
    minRefinementCells  0;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels {N_CELLS_BETWEEN_LEVELS};

    features
    (
    );

    refinementSurfaces
    {{
{chr(10).join(refinement_surfaces)}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
{chr(10).join(refinement_regions)}
    }}

    locationInMesh ({lx:g} {ly:g} {lz:g});

    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          30;
    nRelaxIter          5;
    nFeatureSnapIter    0;
    implicitFeatureSnap false;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers
    {{
{chr(10).join(f"        {patch} {{ nSurfaceLayers {N_SURFACE_LAYERS}; }}" for patch in patch_names)}
    }}
    expansionRatio {LAYER_EXPANSION_RATIO:g};
    finalLayerThickness {FINAL_LAYER_THICKNESS:g};
    minThickness {MIN_LAYER_THICKNESS:g};
    nGrow 0;
    featureAngle 60;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 20;
}}

meshQualityControls
{{
    maxNonOrtho             70;
    maxBoundarySkewness     20;
    maxInternalSkewness     4;
    maxConcave              80;
    minVol                  1e-15;
    minTetQuality           1e-30;
    minArea                 -1;
    minTwist                0.02;
    minDeterminant          0.001;
    minFaceWeight           0.02;
    minVolRatio             0.01;
    minTriangleTwist        -1;
    nSmoothScale            4;
    errorReduction          0.75;
}}

debug 0;
mergeTolerance 1e-6;
""")


def turbulence_enabled() -> bool:
    return TURBULENCE_MODEL.strip().lower() not in {"", "none", "laminar"}


def turbulence_initial_values() -> Tuple[float, float, float]:
    """Return k [m2/s2], omega [1/s], nut [m2/s] for incompressible RAS startup."""
    u = max(abs(VELOCITY), 1e-9)
    intensity = max(TURBULENCE_INTENSITY, 1e-6)
    length_scale = max(TURBULENCE_LENGTH_SCALE, 1e-6)
    # k = 1.5*(U*I)^2. omega is estimated from k and turbulence length scale.
    k = max(1.5 * (u * intensity) ** 2, 1e-12)
    c_mu = 0.09
    omega = max((k ** 0.5) / ((c_mu ** 0.25) * length_scale), 1e-9)
    nut = max(k / omega, 1e-12)
    return k, omega, nut


def write_fields(case: Path, velocity: float, patch_names: Sequence[str]) -> None:
    wall_u = "\n".join(f"""    {patch}
    {{
        type noSlip;
    }}""" for patch in patch_names)
    wall_p = "\n".join(f"""    {patch}
    {{
        type zeroGradient;
    }}""" for patch in patch_names)

    write(case / "0/U", f"""{foam_header("volVectorField", "U")}
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({velocity:g} 0 0);

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform ({velocity:g} 0 0);
    }}

    outlet
    {{
        type zeroGradient;
    }}

    farfield
    {{
        type slip;
    }}

{wall_u}
}}
""")

    write(case / "0/p", f"""{foam_header("volScalarField", "p")}
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type zeroGradient;
    }}

    outlet
    {{
        type fixedValue;
        value uniform 0;
    }}

    farfield
    {{
        type zeroGradient;
    }}

{wall_p}
}}
""")

    if not turbulence_enabled():
        return

    k0, omega0, nut0 = turbulence_initial_values()
    wall_k = "\n".join(f"""    {patch}
    {{
        type kqRWallFunction;
        value uniform {k0:g};
    }}""" for patch in patch_names)
    wall_omega = "\n".join(f"""    {patch}
    {{
        type omegaWallFunction;
        value uniform {omega0:g};
    }}""" for patch in patch_names)
    wall_nut = "\n".join(f"""    {patch}
    {{
        type nutkWallFunction;
        value uniform {nut0:g};
    }}""" for patch in patch_names)

    write(case / "0/k", f"""{foam_header("volScalarField", "k")}
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k0:g};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {k0:g};
    }}

    outlet
    {{
        type zeroGradient;
    }}

    farfield
    {{
        type fixedValue;
        value uniform {k0:g};
    }}

{wall_k}
}}
""")

    write(case / "0/omega", f"""{foam_header("volScalarField", "omega")}
dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omega0:g};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {omega0:g};
    }}

    outlet
    {{
        type zeroGradient;
    }}

    farfield
    {{
        type fixedValue;
        value uniform {omega0:g};
    }}

{wall_omega}
}}
""")

    write(case / "0/nut", f"""{foam_header("volScalarField", "nut")}
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform {nut0:g};

boundaryField
{{
    inlet
    {{
        type calculated;
        value uniform {nut0:g};
    }}

    outlet
    {{
        type calculated;
        value uniform {nut0:g};
    }}

    farfield
    {{
        type calculated;
        value uniform {nut0:g};
    }}

{wall_nut}
}}
""")


def write_constant(case: Path) -> None:
    write(case / "constant/transportProperties", f"""{foam_header("dictionary", "transportProperties")}
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {NU:g};
""")

    if turbulence_enabled():
        model = TURBULENCE_MODEL.strip() or "kOmegaSST"
        write(case / "constant/turbulenceProperties", f"""{foam_header("dictionary", "turbulenceProperties")}
simulationType RAS;

RAS
{{
    RASModel        {model};
    turbulence      on;
    printCoeffs     on;
}}
""")
    else:
        write(case / "constant/turbulenceProperties", f"""{foam_header("dictionary", "turbulenceProperties")}
simulationType laminar;
""")


def solver_application() -> str:
    return "pimpleFoam" if CFD_SOLVER_MODE == "transient" else "simpleFoam"


def solver_log_name() -> str:
    return f"log.{solver_application()}"


def solver_log_names() -> Tuple[str, ...]:
    primary = solver_log_name()
    alternates = ("log.simpleFoam", "log.pimpleFoam")
    return tuple(dict.fromkeys((primary,) + alternates))


def force_coeff_function(name: str, patches: Sequence[str], aref: float, lref: float, cofr: Vec3) -> str:
    patch_list = " ".join(patches)
    cx, cy, cz = cofr
    drag_x, drag_y, drag_z = flow_unit_vector()
    return f"""    forceCoeffs_{safe_patch_name(name)}
    {{
        type            forceCoeffs;
        libs            (\"libforces.so\");
        patches         ({patch_list});

        p               p;
        U               U;
        rho             rhoInf;
        rhoInf          {RHO:g};
        pRef            0;

        CofR            ({cx:g} {cy:g} {cz:g});

        dragDir         ({drag_x:g} {drag_y:g} {drag_z:g});
        liftDir         (0 0 1);
        pitchAxis       (0 1 0);

        magUInf         {abs(VELOCITY):g};
        lRef            {lref:g};
        Aref            {aref:g};

        enabled         true;
        writeToFile     true;
        writeFields     no;
        executeControl  timeStep;
        executeInterval {CFD_WRITE_INTERVAL};
        writeControl    timeStep;
        writeInterval   {CFD_WRITE_INTERVAL};
        log             true;
    }}"""


def forces_function(name: str, patches: Sequence[str], cofr: Vec3) -> str:
    """OpenFOAM dimensional forces function object.

    This is the primary v9 motion source: it outputs newtons and N*m directly,
    avoiding errors from guessed Aref/lRef when moving individual small parts.
    """
    patch_list = " ".join(patches)
    cx, cy, cz = cofr
    return f"""    forces_{safe_patch_name(name)}
    {{
        type            forces;
        libs            (\"libforces.so\");
        patches         ({patch_list});

        p               p;
        U               U;
        rho             rhoInf;
        rhoInf          {RHO:g};
        pRef            0;

        CofR            ({cx:g} {cy:g} {cz:g});

        enabled         true;
        writeToFile     true;
        writeFields     no;
        executeControl  timeStep;
        executeInterval {CFD_WRITE_INTERVAL};
        writeControl    timeStep;
        writeInterval   {CFD_WRITE_INTERVAL};
        log             true;
    }}"""


def sampled_wall_surfaces_function(components: Sequence[AeroComponent]) -> str:
    """OpenFOAM sampled-surfaces function object for wall patch visualization."""
    if not PARAVIEW_SAMPLED_WALL_SURFACES:
        return ""
    field_list = " ".join(SAMPLED_SURFACE_FIELDS)
    surfaces = []
    for c in components:
        safe = safe_patch_name(c.patch)
        surfaces.append(f"""        {safe}_sampled
        {{
            type        patch;
            patches     ({c.patch});
            interpolate true;
        }}""")
    if not surfaces:
        return ""
    return f"""    {SAMPLED_SURFACE_FUNCTION_NAME}
    {{
        type            surfaces;
        libs            (\"libsampling.so\");
        enabled         true;
        writeControl    writeTime;
        surfaceFormat   {SAMPLED_SURFACE_FORMAT};
        interpolationScheme cellPoint;
        fields          ({field_list});
        surfaces
        (
{chr(10).join(surfaces)}
        );
    }}"""




def sampled_wall_sampledict(components: Sequence[AeroComponent]) -> str:
    """Classic system/sampleDict fallback for OpenFOAM builds where
    postProcess function-object surface sampling is unreliable.

    The sample utility is older and often works in Docker images where the
    function-object route silently produces no VTK. It samples solved volume
    fields onto the actual wall patches, which is what you want to colour the
    aircraft/control surfaces by Cp, pPa, U, and wallShearStress.
    """
    if not PARAVIEW_SAMPLED_WALL_SURFACES or not PARAVIEW_SAMPLE_UTILITY_FALLBACK:
        return ""
    field_list = " ".join(SAMPLED_SURFACE_FIELDS)
    surfaces = []
    for c in components:
        safe = safe_patch_name(c.patch)
        surfaces.append(f"""    {safe}_sampled
    {{
        type        patch;
        patches     ({c.patch});
    }}""")
    if not surfaces:
        return ""
    return f"""{foam_header("dictionary", "sampleDict")}

interpolationScheme cellPoint;
setFormat raw;
sets
(
);

surfaceFormat {SAMPLED_SURFACE_FORMAT};
surfaces
(
{chr(10).join(surfaces)}
);

fields ({field_list});
"""

def wall_postprocess_function(components: Sequence[AeroComponent]) -> str:
    """Add wallShearStress so surfaces can be coloured by skin friction too."""
    if not ENABLE_WALL_SHEAR_STRESS:
        return ""
    patches = " ".join(c.patch for c in components)
    if not patches:
        return ""
    return f"""    wallShearStress
    {{
        type            wallShearStress;
        libs            (\"libfieldFunctionObjects.so\");
        enabled         true;
        writeControl    writeTime;
        patches         ({patches});
    }}"""


def write_unsteady_fsi_gap_report(case: Path, components: Sequence[AeroComponent]) -> None:
    movable_components = [c for c in components if c.freedom.translate_axes or c.freedom.rotate_axes]
    lines = [
        "# Unsteady / FSI architecture report",
        f"cfd_solver_mode={CFD_SOLVER_MODE}",
        f"solver_application={solver_application()}",
        f"component_count={len(components)}",
        f"movable_component_count={len(movable_components)}",
        "",
        "Implemented in this case:",
        f"- {'Transient' if CFD_SOLVER_MODE == 'transient' else 'Steady'} fluid solver dictionaries are written directly into the OpenFOAM case.",
        "- Pressure and dimensional force extraction are handled from OpenFOAM outputs, not guessed from visualization only.",
        "- The Python side can still update rigid-body state between CFD solves in assembly mode.",
        f"- Optional non-rigid STL surface deformation is {'enabled' if ENABLE_NONRIGID_DEFORMATION else 'available but disabled'} via ENABLE_NONRIGID_DEFORMATION.",
        "",
        "Still missing for full unsteady rigid-body FSI:",
        "- Mesh/body motion is not yet solved inside the OpenFOAM time loop for this generated case.",
        "- The assembly pipeline still rebuilds and re-runs separate CFD cases instead of continuing one ALE time history.",
        "- There are no fluid-structure subiterations within each physical time step.",
        "- Loads are transferred once per outer Python step, not iterated to coupled convergence.",
        "",
        "Still missing for deformable full FSI:",
        "- No structural finite-element solve for elastic deformation.",
        "- No displacement/traction mapping between CFD and structural meshes.",
        "- No added-mass stabilization or partitioned coupling acceleration.",
        "- The available non-rigid mode is a bounded quasi-static panel-pressure deformation approximation between remeshed CFD solves.",
        "",
        "Recommended next implementation steps:",
        "1. Keep transient pimpleFoam as the default unsteady fluid path.",
        "2. Replace the outer remesh-per-step loop with a persistent moving-mesh case.",
        "3. Add rigid-body mesh motion inside OpenFOAM for one selected body before tackling multiple independently moving bodies.",
        "4. Add coupling subiterations and only then consider deformable structural FSI.",
    ]
    (case / AERO_SOLVER_REPORT_NAME).write_text("\n".join(lines) + "\n")


def write_system(case: Path, components: Sequence[AeroComponent], overall_aref: float, overall_lref: float, overall_cofr: Vec3) -> None:
    functions = []
    wall_fn = wall_postprocess_function(components)
    if wall_fn:
        functions.append(wall_fn)
    # Do not run sampledWallSurfaces during simpleFoam, because pPa/Cp are
    # written by Python after the solve. It is put in a separate postProcess dict
    # below and executed by Allsurface after derived fields exist.
    sampled_fn = sampled_wall_surfaces_function(components)
    for c in components:
        origin = force_reference_origin(c)
        functions.append(forces_function(c.patch, [c.patch], origin))
        functions.append(force_coeff_function(c.patch, [c.patch], c.aref, c.lref, origin))
    if len(components) > 1:
        functions.append(forces_function("all", [c.patch for c in components], overall_cofr))
        functions.append(force_coeff_function("all", [c.patch for c in components], overall_aref, overall_lref, overall_cofr))

    if CFD_SOLVER_MODE == "transient":
        control_dict_text = f"""{foam_header("dictionary", "controlDict")}
application     {solver_application()};

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         {AERO_TRANSIENT_END_TIME:g};

deltaT          {AERO_TRANSIENT_DELTA_T:g};

writeControl    adjustableRunTime;
writeInterval   {AERO_TRANSIENT_WRITE_INTERVAL:g};

purgeWrite      {AERO_TRANSIENT_PURGE_WRITE};
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;

adjustTimeStep  yes;
maxCo           {AERO_TRANSIENT_MAX_CO:g};
maxDeltaT       {AERO_TRANSIENT_MAX_DELTA_T:g};

runTimeModifiable true;

functions
{{
{chr(10).join(functions)}
}}
"""
        fv_schemes_text = f"""{foam_header("dictionary", "fvSchemes")}
ddtSchemes
{{
    default backward;
}}

gradSchemes
{{
    default Gauss linear;
}}

divSchemes
{{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default Gauss linear limited 0.5;
}}

interpolationSchemes
{{
    default linear;
}}

snGradSchemes
{{
    default limited 0.5;
}}

wallDist
{{
    method meshWave;
}}
"""
    else:
        control_dict_text = f"""{foam_header("dictionary", "controlDict")}
application     {solver_application()};

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         {ITERATIONS};

deltaT          1;

writeControl    timeStep;
writeInterval   {CFD_WRITE_INTERVAL};

// Keep only the latest written solution time. This prevents a 5M+ cell case
// from filling the Docker/host disk with 0,1,2,3,... field folders.
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;

runTimeModifiable true;

functions
{{
{chr(10).join(functions)}
}}
"""
        fv_schemes_text = f"""{foam_header("dictionary", "fvSchemes")}
ddtSchemes
{{
    default steadyState;
}}

gradSchemes
{{
    default Gauss linear;
}}

divSchemes
{{
    default none;
    div(phi,U) bounded Gauss upwind;
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default Gauss linear limited 0.5;
}}

interpolationSchemes
{{
    default linear;
}}

snGradSchemes
{{
    default limited 0.5;
}}

wallDist
{{
    method meshWave;
}}
"""
    write(case / "system/controlDict", control_dict_text)
    write(case / "system/fvSchemes", fv_schemes_text)

    p_solver_block = _linear_solver_block("p", "GAMG", P_TOLERANCE, P_REL_TOL, P_MAX_ITER, smoother="GaussSeidel")
    u_solver_block = _linear_solver_block("U", U_SOLVER, U_TOLERANCE, U_REL_TOL, U_MAX_ITER,
                                          preconditioner=U_PRECONDITIONER, smoother=U_SMOOTHER)
    turb_solver_block = _linear_solver_block('"(k|omega)"', "smoothSolver", TURB_TOLERANCE, TURB_REL_TOL,
                                             TURB_MAX_ITER, smoother="symGaussSeidel")
    residual_control = _residual_control_block()
    coupling_block = f"""PIMPLE
{{
    momentumPredictor {'yes' if AERO_TRANSIENT_MOMENTUM_PREDICTOR else 'no'};
    nOuterCorrectors {AERO_TRANSIENT_OUTER_CORRECTORS};
    nCorrectors {AERO_TRANSIENT_PRESSURE_CORRECTORS};
    nNonOrthogonalCorrectors {AERO_TRANSIENT_NON_ORTHOGONAL_CORRECTORS};
    pRefCell 0;
    pRefValue 0;
{residual_control}
}}""" if CFD_SOLVER_MODE == "transient" else f"""SIMPLE
{{
    consistent {'yes' if CONSISTENT_SIMPLE else 'no'};
    nNonOrthogonalCorrectors {N_NON_ORTHOGONAL_CORRECTORS};
    pRefCell 0;
    pRefValue 0;
{residual_control}
}}"""

    write(case / "system/fvSolution", f"""{foam_header("dictionary", "fvSolution")}
solvers
{{
{p_solver_block}

{u_solver_block}

{turb_solver_block}
}}

{coupling_block}

relaxationFactors
{{
    fields
    {{
        p {RELAX_P};
    }}

    equations
    {{
        U {RELAX_U};
        k {RELAX_K};
        omega {RELAX_OMEGA};
    }}
}}
""")
    _write_config_report(case)

    postprocess_line = "postProcess -latestTime -dict system/controlDict | tee log.postProcess || true" if RUN_POSTPROCESS_FORCE_OBJECTS else "true"
    application = solver_application()
    solver_log = solver_log_name()
    write(case / "Allrun", f"""#!/bin/sh
set -e

SOLVER_TIMEOUT_SECONDS={SOLVER_TIMEOUT_SECONDS}

blockMesh | tee log.blockMesh
snappyHexMesh -overwrite | tee log.snappyHexMesh
checkMesh -allGeometry -allTopology | tee log.checkMesh

# Run the solver with a hard wall-clock cap.  controlDict writes every
# CFD_WRITE_INTERVAL steps and purgeWrite keeps only the latest written time, so
# if the cap is reached we can still post-process the newest completed solution
# instead of wasting hours or filling the disk.
if [ "$SOLVER_TIMEOUT_SECONDS" -gt 0 ] && command -v timeout >/dev/null 2>&1; then
    set +e
    timeout "$SOLVER_TIMEOUT_SECONDS"s {application} | tee {solver_log}
    solver_status=$?
    set -e
    if [ "$solver_status" -eq 124 ]; then
        echo "WARNING: {application} reached SOLVER_TIMEOUT_SECONDS=$SOLVER_TIMEOUT_SECONDS; continuing with latest written time." | tee -a {solver_log}
    elif [ "$solver_status" -ne 0 ]; then
        echo "ERROR: {application} failed with exit code $solver_status" | tee -a {solver_log}
        exit "$solver_status"
    fi
else
    {application} | tee {solver_log}
fi

{postprocess_line}
""")
    (case / "Allrun").chmod(0o755)

    patch_list_for_vtk = " ".join(c.patch for c in components)
    field_list_for_vtk = " ".join(SAMPLED_SURFACE_FIELDS)
    write(case / "Allvtk", f"""#!/bin/sh
set -e

# Export the latest solved volume fields. v16 creates derived pPa and Cp before
# this script runs, so they appear in ParaView alongside p and U.
foamToVTK -noZero -latestTime | tee log.foamToVTK || foamToVTK -latestTime | tee -a log.foamToVTK || foamToVTK | tee -a log.foamToVTK

# Also try a boundary-patch export. OpenFOAM-11 foamToVTK does not support per-patch selection
# with the old command-line option, so use -noInternal and exclude the tunnel boundaries.
foamToVTK -noZero -latestTime -noInternal -nearCellValue -excludePatches "(inlet outlet farfield)" -fields "({field_list_for_vtk})" | tee log.foamToVTK.patches || \
foamToVTK -latestTime -noInternal -nearCellValue -excludePatches "(inlet outlet farfield)" -fields "({field_list_for_vtk})" | tee -a log.foamToVTK.patches || \
foamToVTK -noZero -latestTime -noInternal -nearCellValue -allPatches -fields "({field_list_for_vtk})" | tee -a log.foamToVTK.patches || true
""")
    (case / "Allvtk").chmod(0o755)

    if sampled_fn:
        write(case / "system/sampledWallSurfacesDict", f"""{foam_header("dictionary", "sampledWallSurfacesDict")}
functions
{{
{sampled_fn}
}}
""")

    sample_dict = sampled_wall_sampledict(components)
    if sample_dict:
        write(case / "system/sampleDict", sample_dict)

    if RUN_PATCH_ONLY_VTK_EXPORT:
        # OpenFOAM-11 foamToVTK does not support the old per-patch option. Use -noInternal plus
        # -excludePatches to export boundary patches only, and -nearCellValue so
        # ParaView sees solved near-wall pressure rather than zeroGradient/fixed
        # boundary placeholder values.  If excludePatches is unsupported or too
        # strict, fall back to all boundary patches.
        patch_only_vtk_line = (
            f'foamToVTK -noZero -latestTime -noInternal -nearCellValue '
            f'-excludePatches "(inlet outlet farfield)" '
            f'-fields "({field_list_for_vtk})" | tee log.foamToVTK.surfacePatches || '
            f'foamToVTK -latestTime -noInternal -nearCellValue '
            f'-excludePatches "(inlet outlet farfield)" '
            f'-fields "({field_list_for_vtk})" | tee -a log.foamToVTK.surfacePatches || '
            f'foamToVTK -noZero -latestTime -noInternal -nearCellValue -allPatches '
            f'-fields "({field_list_for_vtk})" | tee -a log.foamToVTK.surfacePatches || true'
        )
    else:
        patch_only_vtk_line = "true"

    write(case / "Allsurface", f"""#!/bin/sh
set -e

# Re-run visualization/post-processing after Python has written pPa and Cp.
# Do not call wallShearStress unless explicitly enabled. In OpenFOAM-11 it can
# fail after a storage-limited solve because the turbulence model object is not
# available to foamPostProcess, and that failure used to block useful pressure
# sampling diagnostics.
if [ "${{ENABLE_WALL_SHEAR_STRESS:-0}}" = "1" ]; then
    postProcess -latestTime -func wallShearStress | tee log.wallShearStress || true
fi

# Function-object surface sampling. This is preferred when it succeeds.
if [ -f system/sampledWallSurfacesDict ]; then
    postProcess -latestTime -dict system/sampledWallSurfacesDict | tee log.sampledWallSurfaces || true
fi

# The old sample utility is not present in OpenFOAM-11 Docker images, so do not
# call it unless the user explicitly sets PARAVIEW_SAMPLE_UTILITY_FALLBACK=1 and
# provides a compatible image.
if [ -f system/sampleDict ] && command -v sample >/dev/null 2>&1; then
    sample -latestTime -dict system/sampleDict | tee log.sampleUtility || true
fi

# Storage-safe accurate surface export: export boundary patches only, using
# near-cell values so patch pressure is solved pressure, not a flat boundary BC.
{patch_only_vtk_line}
""")
    (case / "Allsurface").chmod(0o755)


def _linear_solver_block(field: str, solver: str, tolerance: str, rel_tol: str, max_iter: int,
                         *, preconditioner: str = "", smoother: str = "") -> str:
    """Build an OpenFOAM linear-solver dictionary block safely.

    OpenFOAM accepts different keywords for iterative solvers.  The old script
    used smoothSolver for U and allowed the default maxIter=1000, which is what
    made the recent run extremely slow.  This helper writes an explicit maxIter
    and the correct support keyword for the chosen solver.
    """
    solver_name = (solver or "PBiCGStab").strip()
    lines = [
        f"    {field}",
        "    {",
        f"        solver {solver_name};",
    ]
    if solver_name.lower() in {"smoothsolver", "gamg"}:
        if smoother:
            lines.append(f"        smoother {smoother};")
    else:
        if preconditioner:
            lines.append(f"        preconditioner {preconditioner};")
    lines.extend([
        f"        tolerance {tolerance};",
        f"        relTol {rel_tol};",
        f"        maxIter {int(max_iter)};",
        "    }",
    ])
    return "\n".join(lines)


def _residual_control_block() -> str:
    if not RESIDUAL_CONTROL:
        return ""
    return f"""
    residualControl
    {{
        p {RESIDUAL_CONTROL_P};
        U {RESIDUAL_CONTROL_U};
        \"(k|omega)\" {RESIDUAL_CONTROL_TURB};
    }}
""".rstrip("\n")


def _write_config_report(case: Path) -> None:
    lines = [
        "# Active built-in CFD configuration",
        f"CFD_CONFIG={CFD_CONFIG}",
        f"CFD_SOLVER_MODE={CFD_SOLVER_MODE}",
        f"solver_application={solver_application()}",
        f"CFD_ITERATIONS={ITERATIONS}",
        f"CFD_WRITE_INTERVAL={CFD_WRITE_INTERVAL}",
        f"SOLVER_TIMEOUT_SECONDS={SOLVER_TIMEOUT_SECONDS}",
        f"ALLOW_CFD_ENV_OVERRIDES={int(ALLOW_CFD_ENV_OVERRIDES)}",
        f"AERO_TRANSIENT_END_TIME={AERO_TRANSIENT_END_TIME}",
        f"AERO_TRANSIENT_DELTA_T={AERO_TRANSIENT_DELTA_T}",
        f"AERO_TRANSIENT_WRITE_INTERVAL={AERO_TRANSIENT_WRITE_INTERVAL}",
        f"AERO_TRANSIENT_PURGE_WRITE={AERO_TRANSIENT_PURGE_WRITE}",
        f"AERO_TRANSIENT_MAX_CO={AERO_TRANSIENT_MAX_CO}",
        f"AERO_TRANSIENT_MAX_DELTA_T={AERO_TRANSIENT_MAX_DELTA_T}",
        f"AERO_TRANSIENT_OUTER_CORRECTORS={AERO_TRANSIENT_OUTER_CORRECTORS}",
        f"AERO_TRANSIENT_PRESSURE_CORRECTORS={AERO_TRANSIENT_PRESSURE_CORRECTORS}",
        f"AERO_TRANSIENT_NON_ORTHOGONAL_CORRECTORS={AERO_TRANSIENT_NON_ORTHOGONAL_CORRECTORS}",
        f"AERO_TRANSIENT_MOMENTUM_PREDICTOR={int(AERO_TRANSIENT_MOMENTUM_PREDICTOR)}",
        f"ASSEMBLY_DYNAMIC_STEPS={ASSEMBLY_DYNAMIC_STEPS}",
        f"BASE_CELLS_PER_LENGTH={BASE_CELLS_PER_LENGTH}",
        f"MIN_CELLS={MIN_CELLS}",
        f"SURFACE_REFINEMENT={SURFACE_REFINEMENT}",
        f"REGION_REFINEMENT={REGION_REFINEMENT}",
        f"MAX_LOCAL_CELLS={MAX_LOCAL_CELLS}",
        f"MAX_GLOBAL_CELLS={MAX_GLOBAL_CELLS}",
        f"ADD_BOUNDARY_LAYERS={int(ADD_BOUNDARY_LAYERS)}",
        f"U_SOLVER={U_SOLVER}",
        f"U_PRECONDITIONER={U_PRECONDITIONER}",
        f"U_MAX_ITER={U_MAX_ITER}",
        f"U_TOLERANCE={U_TOLERANCE}",
        f"U_REL_TOL={U_REL_TOL}",
        f"P_MAX_ITER={P_MAX_ITER}",
        f"P_REL_TOL={P_REL_TOL}",
        f"TURB_MAX_ITER={TURB_MAX_ITER}",
        f"N_NON_ORTHOGONAL_CORRECTORS={N_NON_ORTHOGONAL_CORRECTORS}",
        f"CONSISTENT_SIMPLE={int(CONSISTENT_SIMPLE)}",
        f"RELAX_P={RELAX_P}",
        f"RELAX_U={RELAX_U}",
        f"RESIDUAL_CONTROL={int(RESIDUAL_CONTROL)}",
        f"RESIDUAL_CONTROL_P={RESIDUAL_CONTROL_P}",
        f"RESIDUAL_CONTROL_U={RESIDUAL_CONTROL_U}",
        f"RESIDUAL_CONTROL_TURB={RESIDUAL_CONTROL_TURB}",
        f"ENABLE_PART_COLLISIONS={int(ENABLE_PART_COLLISIONS)}",
        f"COLLISION_METHOD={COLLISION_METHOD}",
        f"COLLISION_MARGIN_M={COLLISION_MARGIN_M}",
        f"COLLISION_POSITION_CORRECTION={COLLISION_POSITION_CORRECTION}",
        f"COLLISION_RESTITUTION={COLLISION_RESTITUTION}",
        f"COLLISION_PRESCRIBED_IMPACT_RESTITUTION={COLLISION_PRESCRIBED_IMPACT_RESTITUTION}",
        f"COLLISION_FRICTION_COEFFICIENT={COLLISION_FRICTION_COEFFICIENT}",
        f"COLLISION_MANIFOLD_TOLERANCE_M={COLLISION_MANIFOLD_TOLERANCE_M}",
        f"COLLISION_MAX_PASSES={COLLISION_MAX_PASSES}",
        f"COLLISION_MAX_LINEAR_SPEED_MPS={COLLISION_MAX_LINEAR_SPEED_MPS}",
        f"COLLISION_MAX_ANGULAR_SPEED_RAD_S={COLLISION_MAX_ANGULAR_SPEED_RAD_S}",
        f"ENABLE_COLLISION_DEFORMATION={int(ENABLE_COLLISION_DEFORMATION)}",
        f"COLLISION_DEFORMATION_MODEL={COLLISION_DEFORMATION_MODEL}",
        f"COLLISION_DEFORMATION_GAIN={COLLISION_DEFORMATION_GAIN}",
        f"COLLISION_DEFORMATION_RADIUS_FACTOR={COLLISION_DEFORMATION_RADIUS_FACTOR}",
        f"COLLISION_DEFORMATION_MIN_RADIUS_M={COLLISION_DEFORMATION_MIN_RADIUS_M}",
        f"COLLISION_MAX_CONTACT_DEFORMATION={COLLISION_MAX_CONTACT_DEFORMATION}",
        f"COLLISION_CONVERGENCE_SPEED_MPS={COLLISION_CONVERGENCE_SPEED_MPS}",
        f"COLLISION_CONVERGENCE_COMPONENTS={COLLISION_CONVERGENCE_COMPONENTS or '<auto>'}",
        f"COLLISION_CONVERGENCE_AXIS={COLLISION_CONVERGENCE_AXIS}",
        f"COLLISION_INITIAL_GAP_M={COLLISION_INITIAL_GAP_M}",
        f"COLLISION_CONVERGENCE_MOVING_COMPONENT={COLLISION_CONVERGENCE_MOVING_COMPONENT}",
        f"COLLISION_SWEEP_CLAMPING={int(COLLISION_SWEEP_CLAMPING)}",
        f"COLLISION_SWEEP_PENETRATION_M={COLLISION_SWEEP_PENETRATION_M}",
        f"COLLISION_CONVERGENCE_STOP_AFTER_CONTACT={int(COLLISION_CONVERGENCE_STOP_AFTER_CONTACT)}",
        f"AERO_REFERENCE_FRAME={AERO_REFERENCE_FRAME}",
        f"BODY_MOVING_THROUGH_STILL_AIR={int(BODY_MOVING_THROUGH_STILL_AIR)}",
        f"BODY_WORLD_VELOCITY={BODY_WORLD_VELOCITY}",
        f"ENABLE_NONRIGID_DEFORMATION={int(ENABLE_NONRIGID_DEFORMATION)}",
        f"DEFORM_ANCHORED_COMPONENTS={int(DEFORM_ANCHORED_COMPONENTS)}",
        f"DEFORMATION_COMPONENT_NAME_CONTAINS={DEFORMATION_COMPONENT_NAME_CONTAINS or '<all>'}",
        f"DEFORMATION_MATERIAL_NAME_CONTAINS={DEFORMATION_MATERIAL_NAME_CONTAINS or '<all>'}",
        f"DEFORMATION_EXCLUDE_COMPONENT_NAME_CONTAINS={DEFORMATION_EXCLUDE_COMPONENT_NAME_CONTAINS or '<none>'}",
        f"DEFORMATION_EXCLUDE_MATERIAL_NAME_CONTAINS={DEFORMATION_EXCLUDE_MATERIAL_NAME_CONTAINS or '<none>'}",
        f"DEFORMATION_YOUNG_MODULUS_PA={DEFORMATION_YOUNG_MODULUS_PA or '<bom/material-inferred>'}",
        f"DEFORMATION_THICKNESS_M={DEFORMATION_THICKNESS_M or '<bom/geometry-inferred>'}",
        f"DEFORMATION_GAIN={DEFORMATION_GAIN}",
        f"DEFORMATION_RELAXATION={DEFORMATION_RELAXATION}",
        f"MAX_DEFORMATION_PER_STEP={MAX_DEFORMATION_PER_STEP}",
        f"MAX_TOTAL_DEFORMATION={MAX_TOTAL_DEFORMATION}",
        "",
        "# To change the whole behaviour, set CFD_CONFIG=debug|fast|balanced|accurate.",
        "# Individual CFD_* env vars are ignored unless CFD_CONFIG=custom or ALLOW_CFD_ENV_OVERRIDES=1.",
        "# To override one parameter, export that parameter directly, e.g. U_MAX_ITER=80.",
    ]
    (case / "cfd_config_report.txt").write_text("\n".join(lines) + "\n")


def make_case_from_components(components: Sequence[AeroComponent], case: Path, clear_case: bool = True) -> None:
    if clear_case and case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True, exist_ok=True)

    all_points: List[Vec3] = []
    for c in components:
        all_points.extend(stl_points(c.triangles))
    if not all_points:
        raise ValueError("No geometry triangles were available for the CFD case")

    xmin, xmax, ymin, ymax, zmin, zmax = bounds(all_points, SCALE)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    length = max(dx, dy, dz)
    if length <= 0:
        raise ValueError("Geometry has zero size after scaling")

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    # Put the long downstream wake on the real downstream side.
    # For VELOCITY > 0, flow is low-X -> high-X, so downstream is +X.
    # For VELOCITY < 0, flow is high-X -> low-X, so downstream is -X.
    if flow_is_positive_x():
        x_domain_min = xmin - UPSTREAM_LENGTHS * length
        x_domain_max = xmax + DOWNSTREAM_LENGTHS * length
    else:
        x_domain_min = xmin - DOWNSTREAM_LENGTHS * length
        x_domain_max = xmax + UPSTREAM_LENGTHS * length

    domain = (
        x_domain_min,
        x_domain_max,
        cy - SIDE_LENGTHS * length,
        cy + SIDE_LENGTHS * length,
        cz - SIDE_LENGTHS * length,
        cz + SIDE_LENGTHS * length,
    )

    dom_dx = domain[1] - domain[0]
    dom_dy = domain[3] - domain[2]
    dom_dz = domain[5] - domain[4]

    cells = (
        max(MIN_CELLS[0], int(BASE_CELLS_PER_LENGTH * dom_dx / length)),
        max(MIN_CELLS[1], int(BASE_CELLS_PER_LENGTH * dom_dy / length)),
        max(MIN_CELLS[2], int(BASE_CELLS_PER_LENGTH * dom_dz / length)),
    )

    # Pick a point in the fluid region on the upstream side, with a side/top
    # offset so it is unlikely to land inside the model.
    location_x = domain[0] + 0.10 * dom_dx if flow_is_positive_x() else domain[1] - 0.10 * dom_dx
    location_in_mesh = (
        location_x,
        cy + 0.35 * SIDE_LENGTHS * length,
        cz + 0.35 * SIDE_LENGTHS * length,
    )

    near_body_distance = NEAR_BODY_REFINEMENT_LENGTHS * length
    patch_names = [c.patch for c in components]

    for c in components:
        write_ascii_stl_triangles(case / "constant/triSurface" / f"{c.patch}.stl", c.patch, c.triangles, SCALE)

    overall_aref = max(dy * dz, 1e-12)
    write_block_mesh(case, domain, cells)
    write_snappy(case, location_in_mesh, near_body_distance, patch_names)
    write_constant(case)
    write_fields(case, VELOCITY, patch_names)
    write_system(case, components, overall_aref, length, (cx, cy, cz))
    write_unsteady_fsi_gap_report(case, components)
    (case / "case.foam").write_text("")

    print("Created OpenFOAM case")
    print(f"Case:      {case}")
    print(f"Patches:   {', '.join(patch_names)}")
    print(f"Cells:     {cells}")
    print(f"Velocity:  {VELOCITY:g} m/s")
    print(f"Flow:      {'low-X -> high-X' if flow_is_positive_x() else 'high-X -> low-X'}")
    print(f"Domain X:  {domain[0]:g} to {domain[1]:g} m")


def make_case(input_stl: Path, case: Path) -> None:
    triangles = read_stl_triangles(input_stl)
    aref, lref, cofr = component_references(triangles)
    component = AeroComponent("obstacle", "obstacle", triangles, cofr, lref, aref)
    make_case_from_components([component], case, clear_case=True)
    print(f"STL used:  {input_stl}")
    print(f"Surface:   {case / 'constant/triSurface/obstacle.stl'}")


# ------------------------- coefficient export -------------------------


def _try_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def _coefficient_file_sort_key(path: Path) -> Tuple[float, str]:
    try:
        time_value = float(path.parent.name)
    except ValueError:
        time_value = float("inf")
    return (time_value, str(path))


def _read_force_coefficients_file(path: Path) -> Tuple[Optional[List[str]], List[List[str]]]:
    header_tokens: Optional[List[str]] = None
    rows: List[List[str]] = []

    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            tokens = line.lstrip("#").strip().split()
            if tokens and tokens[0].lower() == "time":
                header_tokens = tokens
            continue

        tokens = line.split()
        if tokens and _try_float(tokens[0]) is not None:
            rows.append(tokens)

    if not rows:
        return header_tokens, []

    if header_tokens is None:
        fallback = [
            "Time", "Cd", "Cs", "Cl", "CmRoll", "CmPitch", "CmYaw",
            "Cd(f)", "Cd(r)", "Cs(f)", "Cs(r)", "Cl(f)", "Cl(r)",
        ]
        max_width = max(len(row) for row in rows)
        if max_width <= len(fallback):
            header_tokens = fallback[:max_width]
        else:
            header_tokens = fallback + [f"extra_{i}" for i in range(len(fallback), max_width)]

    return header_tokens, rows


def _find_force_coefficient_files(case: Path) -> List[Path]:
    """Find forceCoeffs output files across OpenFOAM variants.

    Some images write coefficient.dat, others write forceCoeffs.dat. Earlier
    versions only searched coefficient.dat, which made the motion solver see
    coeff_source=none even when OpenFOAM had produced forceCoeffs data.
    """
    patterns = [
        "forceCoeffs_*/*/coefficient.dat",
        "forceCoeffs_*/*/forceCoeffs.dat",
        "forceCoeffs_*/*/*.dat",
    ]
    found: List[Path] = []
    seen: set[str] = set()
    root = case / "postProcessing"
    for pattern in patterns:
        for path in root.glob(pattern):
            key = str(path.resolve())
            if key not in seen and path.is_file():
                seen.add(key)
                found.append(path)
    return sorted(found, key=_coefficient_file_sort_key)


def write_force_coeff_debug_report(case: Path, coeffs_by_patch: Dict[str, Dict[str, float]]) -> None:
    lines = [
        "# forceCoeffs debug report",
        f"case={case}",
        "",
        "Detected coefficient data by patch:",
    ]
    if coeffs_by_patch:
        for patch, coeffs in sorted(coeffs_by_patch.items()):
            lines.append(f"- {patch}: keys={sorted(coeffs.keys())}, values={coeffs}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("postProcessing .dat files:")
    pp = case / "postProcessing"
    dats = sorted(pp.rglob("*.dat")) if pp.exists() else []
    if dats:
        for path in dats:
            try:
                rel = path.relative_to(case)
            except ValueError:
                rel = path
            lines.append(f"- {rel}")
    else:
        lines.append("- none")

    boundary = case / "constant/polyMesh/boundary"
    lines.append("")
    lines.append("OpenFOAM boundary file exists: " + str(boundary.exists()))
    if boundary.exists():
        text = boundary.read_text(errors="replace")
        for raw in text.splitlines()[:120]:
            line = raw.rstrip()
            if line:
                lines.append("  " + line)

    (case / FORCE_COEFF_DEBUG_REPORT_NAME).write_text("\n".join(lines) + "\n")


def export_force_coefficients(case: Path) -> Path:
    output_path = case / COEFFICIENT_LOG_NAME
    coeff_files = _find_force_coefficient_files(case)

    if not coeff_files:
        output_path.write_text(
            "No OpenFOAM forceCoeffs coefficient.dat files were found.\n"
            f"Expected files under: {case / 'postProcessing'}\n"
            f"This usually means {solver_application()} did not reach the forceCoeffs write step.\n"
        )
        print(f"Coefficient log created, but no data was found: {output_path}")
        return output_path

    selected_header: Optional[List[str]] = None
    exported_rows: List[Tuple[float, str, str, List[str]]] = []

    for coeff_file in coeff_files:
        header, rows = _read_force_coefficients_file(coeff_file)
        if not rows:
            continue
        if selected_header is None:
            selected_header = header
        function_name = coeff_file.parents[1].name
        patch_name = function_name.replace("forceCoeffs_", "", 1)
        for row in rows:
            time_value = _try_float(row[0])
            exported_rows.append((
                time_value if time_value is not None else float("inf"),
                patch_name,
                str(coeff_file.relative_to(case)),
                row,
            ))

    if not exported_rows:
        output_path.write_text(
            "OpenFOAM forceCoeffs files were found, but they contained no numeric coefficient rows.\n"
            f"Files checked: {', '.join(str(p.relative_to(case)) for p in coeff_files)}\n"
        )
        print(f"Coefficient log created, but no numeric rows were found: {output_path}")
        return output_path

    exported_rows.sort(key=lambda item: (item[0], item[1], item[2]))

    max_width = max(len(row) for _time_value, _patch, _source, row in exported_rows)
    if selected_header is None:
        selected_header = ["Time"] + [f"value_{i}" for i in range(1, max_width)]
    elif len(selected_header) < max_width:
        selected_header = selected_header + [f"extra_{i}" for i in range(len(selected_header), max_width)]
    elif len(selected_header) > max_width:
        selected_header = selected_header[:max_width]

    lines = [
        "# Aerodynamic coefficient log exported from OpenFOAM forceCoeffs",
        f"# Case: {case}",
        f"# Velocity vector: ({VELOCITY:g} 0 0) m/s",
        f"# Reference speed used by forceCoeffs: {abs(VELOCITY):g} m/s",
        f"# rhoInf: {RHO:g} kg/m^3",
        "#",
        "# Source coefficient files:",
    ]
    lines.extend(f"# - {p.relative_to(case)}" for p in coeff_files)
    lines.extend([
        "#",
        "# Columns are tab-separated. 'patch' identifies the object/part coefficient table.",
        "\t".join(["patch"] + selected_header + ["source_file"]),
    ])

    for _time_value, patch_name, source, row in exported_rows:
        padded_row = row + [""] * (len(selected_header) - len(row))
        lines.append("\t".join([patch_name] + padded_row + [source]))

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Coefficient log: {output_path}")
    return output_path


def latest_coefficients_by_patch(case: Path) -> Dict[str, Dict[str, float]]:
    coeff_files = _find_force_coefficient_files(case)
    latest: Dict[str, Dict[str, float]] = {}
    for coeff_file in coeff_files:
        header, rows = _read_force_coefficients_file(coeff_file)
        if not rows or not header:
            continue
        row = rows[-1]
        patch_name = coeff_file.parents[1].name.replace("forceCoeffs_", "", 1)
        data: Dict[str, float] = {}
        for name, raw in zip(header, row):
            value = _try_float(raw)
            if value is not None:
                data[name] = value
        latest[patch_name] = data
    return latest


def _find_force_load_files(case: Path) -> List[Path]:
    """Find OpenFOAM dimensional forces output files."""
    patterns = [
        "forces_*/*/forces.dat",
        "forces_*/*/*.dat",
    ]
    found: List[Path] = []
    seen: set[str] = set()
    root = case / "postProcessing"
    for pattern in patterns:
        for path in root.glob(pattern):
            key = str(path.resolve())
            if key not in seen and path.is_file():
                seen.add(key)
                found.append(path)
    return sorted(found, key=_coefficient_file_sort_key)


def _parse_parenthesized_vectors(text: str) -> List[Vec3]:
    """Extract all '(x y z)' vector triples from an OpenFOAM forces row."""
    vectors: List[Vec3] = []
    for group in re.findall(r"\(([^()]+)\)", text):
        vals = []
        for token in group.replace(",", " ").split():
            value = _try_float(token)
            if value is not None:
                vals.append(value)
        if len(vals) == 3:
            vectors.append((vals[0], vals[1], vals[2]))
    return vectors


def _read_forces_file(path: Path) -> List[Tuple[float, Vec3, Vec3]]:
    """Read OpenFOAM forces.dat rows as (time, force_N, moment_Nm).

    Most OpenFOAM variants write rows as:
      Time ((Fp) (Fv) (Fporous)) ((Mp) (Mv) (Mporous))
    Some variants write a shorter total-force/total-moment format.  This parser
    accepts both and sums pressure+viscous+porous when the split format appears.
    """
    rows: List[Tuple[float, Vec3, Vec3]] = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        time_value = _try_float(first)
        if time_value is None:
            continue
        vectors = _parse_parenthesized_vectors(line)
        if len(vectors) >= 6:
            force = v_add(v_add(vectors[0], vectors[1]), vectors[2])
            moment = v_add(v_add(vectors[3], vectors[4]), vectors[5])
        elif len(vectors) >= 2:
            force, moment = vectors[0], vectors[1]
        else:
            # Very old/flattened format fallback: Time Fx Fy Fz Mx My Mz ...
            nums = [_try_float(tok) for tok in line.split()]
            vals = [x for x in nums if x is not None]
            if len(vals) >= 7:
                force = (vals[1], vals[2], vals[3])
                moment = (vals[4], vals[5], vals[6])
            else:
                continue
        rows.append((time_value, force, moment))
    return rows



def _parse_forces_from_log_text(text: str) -> Dict[str, Tuple[Vec3, Vec3]]:
    """Fallback parser for OpenFOAM force function-object output in logs.

    Some OpenFOAM Docker images print force/moment blocks to the solver log but
    do not create postProcessing/forces_*/forces.dat. This parser recovers the
    last pressure+viscous+porous force and moment per forces_<patch> object.
    """
    loads: Dict[str, Tuple[Vec3, Vec3]] = {}
    current_patch: Optional[str] = None
    mode: Optional[str] = None
    force_parts: List[Vec3] = []
    moment_parts: List[Vec3] = []

    def flush() -> None:
        nonlocal current_patch, force_parts, moment_parts
        if current_patch and (force_parts or moment_parts):
            f = (0.0, 0.0, 0.0)
            m = (0.0, 0.0, 0.0)
            for v in force_parts:
                f = v_add(f, v)
            for v in moment_parts:
                m = v_add(m, v)
            loads[current_patch] = (f, m)
        force_parts = []
        moment_parts = []

    for raw in text.splitlines():
        line = raw.strip()
        m_obj = re.search(r"\bforces\s+(forces_[A-Za-z0-9_\-\.]+)\b", line)
        if m_obj:
            flush()
            current_patch = m_obj.group(1).replace("forces_", "", 1)
            mode = None
            continue
        if current_patch is None:
            continue
        lower = line.lower()
        if "sum of forces" in lower:
            mode = "force"
            continue
        if "sum of moments" in lower:
            mode = "moment"
            continue
        if mode not in {"force", "moment"}:
            continue
        if any(key in lower for key in ("pressure", "viscous", "porous", "total")):
            vectors = _parse_parenthesized_vectors(line)
            if not vectors:
                continue
            v = vectors[-1]
            if mode == "force":
                force_parts.append(v)
            else:
                moment_parts.append(v)
    flush()
    return loads


def latest_dimensional_loads_from_logs(case: Path) -> Dict[str, Tuple[Vec3, Vec3]]:
    loads: Dict[str, Tuple[Vec3, Vec3]] = {}
    for log_name in solver_log_names() + ("log.postProcess",):
        path = case / log_name
        if not path.exists():
            continue
        try:
            parsed = _parse_forces_from_log_text(path.read_text(errors="replace"))
        except Exception:
            parsed = {}
        loads.update(parsed)
    return loads

def latest_dimensional_loads_by_patch(case: Path) -> Dict[str, Tuple[Vec3, Vec3]]:
    if not USE_OPENFOAM_FORCES_DAT:
        return {}
    loads: Dict[str, Tuple[Vec3, Vec3]] = {}
    for path in _find_force_load_files(case):
        rows = _read_forces_file(path)
        if not rows:
            continue
        patch_name = path.parents[1].name.replace("forces_", "", 1)
        _time_value, force, moment = rows[-1]
        loads[patch_name] = (force, moment)

    # Important v11 fallback: some OpenFOAM images log forces but do not create
    # postProcessing/forces_*/forces.dat. Preserve accuracy by recovering those
    # logged OpenFOAM-integrated loads before falling back to panel aerodynamics.
    if not loads:
        loads.update(latest_dimensional_loads_from_logs(case))
    return loads




def load_is_nonzero(load: Optional[Tuple[Vec3, Vec3]]) -> bool:
    """True only when OpenFOAM produced a usable non-zero force or moment.

    Previous versions treated a present-but-zero OpenFOAM force entry as valid,
    so the motion solver refused to fall back to panel aerodynamics. That made
    all movement stay exactly zero whenever the OpenFOAM forces function object
    printed pressure/viscous force = (0 0 0).
    """
    if load is None:
        return False
    force, moment = load
    return max(v_norm(force), v_norm(moment)) > OPENFOAM_FORCE_ZERO_TOL


def export_dimensional_forces(case: Path) -> Path:
    """Export OpenFOAM forces.dat into a clean tab-separated log."""
    output_path = case / FORCES_LOG_NAME
    force_files = _find_force_load_files(case)
    rows_out: List[Tuple[float, str, str, Vec3, Vec3]] = []
    for path in force_files:
        patch_name = path.parents[1].name.replace("forces_", "", 1)
        for time_value, force, moment in _read_forces_file(path):
            rows_out.append((time_value, patch_name, str(path.relative_to(case)), force, moment))
    rows_out.sort(key=lambda item: (item[0], item[1], item[2]))

    lines = [
        "# Dimensional aerodynamic force/moment log exported from OpenFOAM forces",
        f"# Case: {case}",
        f"# Velocity vector: ({VELOCITY:g} 0 0) m/s",
        f"# rhoInf: {RHO:g} kg/m^3",
        "# Units: force=N, moment=N*m",
        "# Columns are tab-separated.",
        "patch\tTime\tFx_N\tFy_N\tFz_N\tMx_Nm\tMy_Nm\tMz_Nm\tsource_file",
    ]
    if not rows_out:
        lines.append("# No OpenFOAM forces.dat rows found.")
    for time_value, patch_name, source, force, moment in rows_out:
        lines.append(
            f"{patch_name}\t{time_value:.8g}\t"
            f"{force[0]:.8g}\t{force[1]:.8g}\t{force[2]:.8g}\t"
            f"{moment[0]:.8g}\t{moment[1]:.8g}\t{moment[2]:.8g}\t{source}"
        )
    output_path.write_text("\n".join(lines) + "\n")
    if rows_out:
        print(f"Dimensional force log: {output_path}")
    else:
        print(f"Dimensional force log created, but no data was found: {output_path}")
    return output_path


def write_force_load_debug_report(case: Path, loads_by_patch: Dict[str, Tuple[Vec3, Vec3]]) -> None:
    lines = [
        "# OpenFOAM dimensional load debug report",
        f"case={case}",
        f"use_openfoam_forces_dat={USE_OPENFOAM_FORCES_DAT}",
        f"run_postprocess_force_objects={RUN_POSTPROCESS_FORCE_OBJECTS}",
        f"openfoam_force_zero_tol={OPENFOAM_FORCE_ZERO_TOL}",
        "",
        "Detected force/moment loads by patch:",
    ]
    if loads_by_patch:
        for patch, (force, moment) in sorted(loads_by_patch.items()):
            lines.append(f"- {patch}: F={force} N, M={moment} N*m")
    else:
        lines.append("- none")

    log_loads = latest_dimensional_loads_from_logs(case)
    lines.append("")
    lines.append(f"Recovered loads from {', '.join(solver_log_names())} and log.postProcess:")
    if log_loads:
        for patch, (force, moment) in sorted(log_loads.items()):
            lines.append(f"- {patch}: F={force} N, M={moment} N*m")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("postProcessing force .dat files:")
    force_files = _find_force_load_files(case)
    if force_files:
        for path in force_files:
            try:
                rel = path.relative_to(case)
            except ValueError:
                rel = path
            lines.append(f"- {rel}")
            sample = path.read_text(errors="replace").splitlines()[-3:]
            for line in sample:
                lines.append("    " + line[:240])
    else:
        lines.append("- none")

    for log_name in list(solver_log_names()) + ["log.postProcess"]:
        log_path = case / log_name
        lines.append("")
        lines.append(f"{log_name} exists: {log_path.exists()}")
        if log_path.exists():
            txt = log_path.read_text(errors="replace")
            interesting = [ln for ln in txt.splitlines() if "forces" in ln or "forceCoeffs" in ln or "Unknown" in ln or "Cannot" in ln or "FOAM FATAL" in ln]
            for ln in interesting[-80:]:
                lines.append("  " + ln[:240])

    (case / FORCE_LOAD_DEBUG_REPORT_NAME).write_text("\n".join(lines) + "\n")


# ------------------------- Onshape client -------------------------
