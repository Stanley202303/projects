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
from .motion import *
from .visualization import *
from .visualization import (
    _clear_root_view_outputs_for_streaming,
    _latest_solver_time_dir,
    _numeric_time_dirs,
    _read_scalar_internal_values,
)

def run_assembly_motion_simulation(components: List[AeroComponent], root_case: Path) -> Path:
    """Run assembly quasi-dynamic CFD without keeping motion_steps by default.

    v19 storage behaviour:
      - SAVE_MOTION_STEPS=0, default: each step is solved in a temporary case;
        only root numeric time directories, compact VTK preview frames, and small
        reports are retained.
      - SAVE_MOTION_STEPS=1: old debugging behaviour; full motion_steps/step_XXX
        cases are retained and the legacy merge/PVD builders are also used.
    """
    if root_case.exists():
        shutil.rmtree(root_case)
    root_case.mkdir(parents=True)
    _clear_root_view_outputs_for_streaming(root_case)

    if SAVE_MOTION_STEPS:
        steps_dir = root_case / "motion_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
        reusable_temp_step_case: Optional[Path] = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix=f"{CASE_NAME}_step_work_")
        reusable_temp_step_case = Path(temp_context.name) / "step_work"
        steps_dir = root_case / "motion_steps"  # logical only; not created

    motion_log = root_case / MOTION_LOG_NAME
    deformation_log = root_case / DEFORMATION_LOG_NAME
    geometry_report = root_case / MOTION_GEOMETRY_REPORT_NAME
    collision_log = root_case / COLLISION_LOG_NAME
    write_unsteady_fsi_gap_report(root_case, components)
    apply_relative_motion_policy(components, root_case)
    rigid_body_root = assembly_rigid_body_root(components)
    rigid_body_state = build_rigid_body_state(components, rigid_body_root) if rigid_body_root is not None else None
    write_motion_log_header(motion_log)
    if ENABLE_NONRIGID_DEFORMATION:
        write_deformation_log_header(deformation_log)
    collision_log.write_text(
        "# Part collision log. Collision method is conservative AABB broad-phase/narrow-phase.\n"
        "# AABB collisions prevent obvious interpenetration and add contact impulses/torques, but are not exact triangle-triangle contact mechanics.\n"
        "step\tpass\tpatch_a\tpatch_b\tdepth_m\tnx\tny\tnz\tcx\tcy\tcz\timpulse_Ns\tmove_a_m\tmove_b_m\n"
    )
    geometry_report.write_text(
        "# Per-step geometry/movement diagnostic. If dx/drot stay zero, the motion solver is not moving that part.\n"
        "step\tpatch\tcoeff_source\tload_source\tforce_N\tmoment_Nm\tmoved_m\trotated_rad\t"
        "cofr_x\tcofr_y\tcofr_z\t"
        "bbox_xmin\tbbox_xmax\tbbox_ymin\tbbox_ymax\tbbox_zmin\tbbox_zmax\n"
    )

    index_lines = [
        "Assembly quasi-dynamic CFD run", "",
        f"cfd_solver_mode={CFD_SOLVER_MODE}",
        f"solver_application={solver_application()}",
        f"dynamic_steps={ASSEMBLY_DYNAMIC_STEPS}",
        f"motion_dt={MOTION_DT}",
        f"enable_nonrigid_deformation={ENABLE_NONRIGID_DEFORMATION}",
        f"deform_anchored_components={DEFORM_ANCHORED_COMPONENTS}",
        f"deformation_component_name_contains={DEFORMATION_COMPONENT_NAME_CONTAINS or '<all>'}",
        f"deformation_material_name_contains={DEFORMATION_MATERIAL_NAME_CONTAINS or '<all>'}",
        f"deformation_exclude_component_name_contains={DEFORMATION_EXCLUDE_COMPONENT_NAME_CONTAINS or '<none>'}",
        f"deformation_exclude_material_name_contains={DEFORMATION_EXCLUDE_MATERIAL_NAME_CONTAINS or '<none>'}",
        f"deformation_young_modulus_pa={DEFORMATION_YOUNG_MODULUS_PA or '<bom/material-inferred>'}",
        f"deformation_thickness_m={DEFORMATION_THICKNESS_M or '<bom/geometry-inferred>'}",
        f"deformation_gain={DEFORMATION_GAIN}",
        f"deformation_relaxation={DEFORMATION_RELAXATION}",
        f"max_deformation_per_step={MAX_DEFORMATION_PER_STEP}",
        f"max_total_deformation={MAX_TOTAL_DEFORMATION}",
        f"save_motion_steps={SAVE_MOTION_STEPS}",
        f"root_panel_preview_dir={ROOT_PANEL_PREVIEW_DIR_NAME}",
        f"keep_step_debug_reports={KEEP_STEP_DEBUG_REPORTS}",
        f"enable_surface_load_fallback={ENABLE_SURFACE_LOAD_FALLBACK}",
        f"surface_load_coeff={SURFACE_LOAD_COEFF}",
        f"surface_load_gain={SURFACE_LOAD_GAIN}",
        f"surface_load_double_sided={SURFACE_LOAD_DOUBLE_SIDED}",
        f"enable_hinge_torque_fallback={ENABLE_HINGE_TORQUE_FALLBACK}",
        f"hinge_torque_coeff={HINGE_TORQUE_COEFF}",
        f"hinge_torque_min_nm={HINGE_TORQUE_MIN_NM}",
        f"enable_part_collisions={ENABLE_PART_COLLISIONS}",
        f"collision_method={COLLISION_METHOD}",
        f"collision_margin_m={COLLISION_MARGIN_M}",
        f"collision_restitution={COLLISION_RESTITUTION}",
        f"collision_max_passes={COLLISION_MAX_PASSES}",
        f"collision_max_linear_speed_mps={COLLISION_MAX_LINEAR_SPEED_MPS}",
        f"collision_max_angular_speed_rad_s={COLLISION_MAX_ANGULAR_SPEED_RAD_S}",
        f"aero_reference_frame={AERO_REFERENCE_FRAME}",
        f"body_moving_through_still_air={BODY_MOVING_THROUGH_STILL_AIR}",
        f"body_world_velocity_mps={BODY_WORLD_VELOCITY}",
        "",
        "components:",
    ]
    for c in components:
        index_lines.append(
            f"- {c.patch}: name={c.name}, material={c.material.material_name}, "
            f"mass={c.mass:.6g} kg, density={c.material.density_kg_m3:.6g} kg/m^3, "
            f"mate_type={c.freedom.mate_type}, translate_axes={c.freedom.translate_axes}, "
            f"rotate_axes={c.freedom.rotate_axes}, source={c.freedom.source}, "
            f"anchored={c.is_assembly_anchor}"
        )
    (root_case / "assembly_motion_index.txt").write_text("\n".join(index_lines) + "\n")

    estimated_total, eta_basis = initial_eta_seconds(ASSEMBLY_DYNAMIC_STEPS)
    run_started = time.monotonic()
    print(f"\nEstimated assembly run time: {format_eta(estimated_total)} ({eta_basis})")
    if SAVE_MOTION_STEPS:
        print(f"Storage mode: keeping full per-step cases in {steps_dir}")
    else:
        print("Storage mode: SAVE_MOTION_STEPS=0; full per-step cases will NOT be kept.")
    if not ROOT_OPENFOAM_TIMESERIES:
        print("Storage mode: ROOT_OPENFOAM_TIMESERIES=0; root numeric OpenFOAM time folders will NOT be kept.")
    if not RUN_FULL_VTK_EXPORT:
        print("Storage mode: RUN_FULL_VTK_EXPORT=0; full volume VTK export will be skipped.")
    if PARAVIEW_MINIMAL_STREAM_TRACER_EXPORT:
        print("Storage mode: minimal Stream Tracer volume case will keep only latest mesh + U.")

    if STORE_START_FINAL_GEOMETRY:
        write_components_geometry_snapshot(root_case, components, START_GEOMETRY_DIR_NAME, "start")
        enforce_case_storage_budget(root_case, "after start geometry snapshot")

    try:
        for step in range(ASSEMBLY_DYNAMIC_STEPS):
            step_started = time.monotonic()
            print(f"\nAssembly motion step {step + 1}/{ASSEMBLY_DYNAMIC_STEPS}")

            if SAVE_MOTION_STEPS:
                step_case = steps_dir / f"step_{step:03d}"
            else:
                assert reusable_temp_step_case is not None
                step_case = reusable_temp_step_case
                if step_case.exists():
                    shutil.rmtree(step_case)

            make_case_from_components(components, step_case, clear_case=True)
            run_docker(step_case)
            # High-definition visualisation: convert real OpenFOAM sampled-surface
            # VTKs to one XML .vtp frame before any rigid-body update is applied.
            # This makes frame N show the pressure field for the geometry actually
            # solved in CFD at step N.
            write_cfd_sampled_surface_preview_for_step(step_case, step)
            step_elapsed_for_eta = time.monotonic() - step_started
            save_eta_calibration(step_elapsed_for_eta)
            steps_done_for_eta = step + 1
            steps_left_for_eta = ASSEMBLY_DYNAMIC_STEPS - steps_done_for_eta
            elapsed_total_for_eta = time.monotonic() - run_started
            avg_step_for_eta = elapsed_total_for_eta / max(1, steps_done_for_eta)
            remaining_for_eta = avg_step_for_eta * steps_left_for_eta
            print(
                f"ETA update: step {steps_done_for_eta}/{ASSEMBLY_DYNAMIC_STEPS} took "
                f"{format_eta(step_elapsed_for_eta)}; remaining about {format_eta(remaining_for_eta)}; "
                f"total about {format_eta(elapsed_total_for_eta + remaining_for_eta)}."
            )

            export_force_coefficients(step_case)
            export_dimensional_forces(step_case)
            coeffs_by_patch = latest_coefficients_by_patch(step_case)
            loads_by_patch = latest_dimensional_loads_by_patch(step_case)
            write_force_coeff_debug_report(step_case, coeffs_by_patch)
            write_force_load_debug_report(step_case, loads_by_patch)

            rigid_component_rows: List[Tuple[AeroComponent, Dict[str, float], str, str, Vec3, Vec3]] = []
            rigid_net_force = (0.0, 0.0, 0.0)
            rigid_net_moment = (0.0, 0.0, 0.0)
            rigid_origin = rigid_body_state.cofr if rigid_body_state is not None else (0.0, 0.0, 0.0)

            for component in components:
                coeffs = coeffs_by_patch.get(component.patch, {})
                coeff_source = component.patch if coeffs else "none"
                if not coeffs and USE_ALL_COEFFS_FALLBACK and "all" in coeffs_by_patch:
                    coeffs = coeffs_by_patch.get("all", {})
                    coeff_source = "all-fallback"

                load_override: Optional[Tuple[Vec3, Vec3]] = None
                load_source = "forceCoeffs" if aerodynamic_coeffs_present(coeffs) else "zero-forceCoeffs"

                own_openfoam_load = loads_by_patch.get(component.patch)
                all_openfoam_load = loads_by_patch.get("all")

                if USE_OPENFOAM_FORCES_DAT and load_is_nonzero(own_openfoam_load):
                    load_override = own_openfoam_load
                    load_source = "OpenFOAM-forces"
                elif (
                    USE_OPENFOAM_FORCES_DAT
                    and USE_ALL_COEFFS_FALLBACK
                    and not component.is_assembly_anchor
                    and load_is_nonzero(all_openfoam_load)
                ):
                    load_override = all_openfoam_load
                    load_source = "OpenFOAM-forces-all-fallback"
                elif ENABLE_SURFACE_LOAD_FALLBACK and (
                    not coeffs
                    or not aerodynamic_coeffs_present(coeffs)
                    or (own_openfoam_load is not None and not load_is_nonzero(own_openfoam_load))
                ) and not component.is_assembly_anchor:
                    load_override = surface_pressure_load(component)
                    load_source = "panel-aero-fallback-after-zero-openfoam" if own_openfoam_load is not None else "panel-aero-fallback"

                if rigid_body_state is not None:
                    force, source_moment = resolve_aerodynamic_load(component, coeffs, load_override)
                    component_moment = total_aerodynamic_moment_about_origin(
                        component,
                        force,
                        source_moment,
                        rigid_origin,
                        load_override is not None,
                    )
                    rigid_component_rows.append((component, coeffs, coeff_source, load_source, force, component_moment))
                    rigid_net_force = v_add(rigid_net_force, force)
                    rigid_net_moment = v_add(rigid_net_moment, component_moment)
                    continue

                force, moment, dpos, drot = update_component_motion(component, coeffs, MOTION_DT, load_override=load_override)
                append_motion_log(motion_log, step, component, coeffs, force, moment, dpos, drot)
                xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
                with geometry_report.open("a") as gf:
                    gf.write(
                        f"{step}\t{component.patch}\t{coeff_source}\t{load_source}\t"
                        f"{v_norm(force):.8g}\t{v_norm(moment):.8g}\t"
                        f"{v_norm(dpos):.8g}\t{v_norm(drot):.8g}\t"
                        f"{component.cofr[0]:.8g}\t{component.cofr[1]:.8g}\t{component.cofr[2]:.8g}\t"
                        f"{xmin:.8g}\t{xmax:.8g}\t{ymin:.8g}\t{ymax:.8g}\t{zmin:.8g}\t{zmax:.8g}\n"
                    )

            if rigid_body_state is not None:
                rigid_force, rigid_moment, rigid_dpos, rigid_drot = update_component_motion(
                    rigid_body_state,
                    {},
                    MOTION_DT,
                    load_override=(rigid_net_force, rigid_net_moment),
                )
                apply_rigid_body_motion(
                    components,
                    rigid_dpos,
                    rigid_drot,
                    rigid_origin,
                    rigid_body_state.linear_velocity,
                    rigid_body_state.angular_velocity,
                    rigid_body_state.total_translation,
                    rigid_body_state.total_rotation,
                )
                for component, coeffs, coeff_source, load_source, force, moment in rigid_component_rows:
                    log_force = rigid_force if component is rigid_body_root else force
                    log_moment = rigid_moment if component is rigid_body_root else moment
                    log_load_source = "assembly-rigid-body-net" if component is rigid_body_root else load_source
                    append_motion_log(motion_log, step, component, coeffs, log_force, log_moment, rigid_dpos, rigid_drot)
                    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
                    with geometry_report.open("a") as gf:
                        gf.write(
                            f"{step}\t{component.patch}\t{coeff_source}\t{log_load_source}\t"
                            f"{v_norm(log_force):.8g}\t{v_norm(log_moment):.8g}\t"
                            f"{v_norm(rigid_dpos):.8g}\t{v_norm(rigid_drot):.8g}\t"
                            f"{component.cofr[0]:.8g}\t{component.cofr[1]:.8g}\t{component.cofr[2]:.8g}\t"
                            f"{xmin:.8g}\t{xmax:.8g}\t{ymin:.8g}\t{ymax:.8g}\t{zmin:.8g}\t{zmax:.8g}\n"
                        )
                deformed_components = apply_nonrigid_deformations(components, step, deformation_log)
                if deformed_components:
                    max_def = max(c.deformation_max_m for c in deformed_components)
                    print(f"Non-rigid deformation: {len(deformed_components)} component(s), max {max_def:.6g} m.")
            else:
                enforce_attachment_constraints(components)
                deformed_components = apply_nonrigid_deformations(components, step, deformation_log)
                if deformed_components:
                    max_def = max(c.deformation_max_m for c in deformed_components)
                    print(f"Non-rigid deformation: {len(deformed_components)} component(s), max {max_def:.6g} m.")
                collision_lines = resolve_part_collisions(components, step, collision_log)
                if collision_lines:
                    print(f"Collision resolution: {len(collision_lines)} contact event(s) handled at step {step}.")

            # Write compact visualisation AFTER the motion and collision update so the frame
            # shows moved, non-interpenetrating geometry.
            write_panel_aero_preview_for_step(step_case, components, step)

            # Retain only compact root-level outputs needed for the .pvd animation.
            # The heavy OpenFOAM root time-series is optional and disabled by default
            # to keep actual_model_case below the storage budget.
            if ROOT_OPENFOAM_TIMESERIES:
                copy_step_to_root_timeseries(root_case, step_case, step)
            copy_minimal_stream_tracer_case_to_root(root_case, step_case, step)
            copy_step_cfd_sampled_preview_to_root(root_case, step_case, step)
            copy_step_panel_preview_to_root(root_case, step_case, step)
            copy_step_debug_reports_to_root(root_case, step_case, step)
            enforce_case_storage_budget(root_case, f"after copying compact frame {step}")

            if not SAVE_MOTION_STEPS and step_case.exists():
                shutil.rmtree(step_case)

        if STORE_START_FINAL_GEOMETRY:
            write_components_geometry_snapshot(root_case, components, FINAL_MOVED_GEOMETRY_DIR_NAME, "final_moved")
            enforce_case_storage_budget(root_case, "after final moved geometry snapshot")

        if KEEP_FINAL_GEOMETRY_CASE:
            final_case = root_case / "final_moved_geometry_case"
            make_case_from_components(components, final_case, clear_case=True)
            enforce_case_storage_budget(root_case, "after optional final_moved_geometry_case")

        if SAVE_MOTION_STEPS:
            # Legacy/full-debug path.  This retains full cases and creates extra
            # raw PVD outputs.  Only use it when debugging storage is acceptable.
            merge_motion_steps_to_root_timeseries(root_case, steps_dir, ASSEMBLY_DYNAMIC_STEPS)
            create_paraview_pvd_timeseries(root_case, steps_dir, ASSEMBLY_DYNAMIC_STEPS)
            create_panel_preview_pvd(root_case, steps_dir, ASSEMBLY_DYNAMIC_STEPS)
        else:
            create_root_safe_pvd_from_copied_previews(root_case, ASSEMBLY_DYNAMIC_STEPS)
            create_root_panel_preview_pvd(root_case, ASSEMBLY_DYNAMIC_STEPS)

        (root_case / "OPEN_THIS_IN_PARAVIEW.txt").write_text(
            "Open this compact moving-surface animation first:\n"
            f"  {root_case / PARAVIEW_PVD_NAME}\n\n"
            "For real 3D Stream Tracer with minimal storage, open this single latest-frame volume case:\n"
            f"  {root_case / STREAM_TRACER_CASE_DIR_NAME / 'case.foam'}\n\n"
            "Storage-light geometry snapshots:\n"
            f"  start: {root_case / START_GEOMETRY_DIR_NAME / GEOMETRY_SNAPSHOT_FILE_NAME}\n"
            f"  final: {root_case / FINAL_MOVED_GEOMETRY_DIR_NAME / GEOMETRY_SNAPSHOT_FILE_NAME}\n\n"
            "The root OpenFOAM time-series case.foam is not built by default because it copies heavy polyMesh/field folders for every frame.\n"
            "Set ROOT_OPENFOAM_TIMESERIES=1 only if you really need it and have enough disk space.\n"
            "The Stream Tracer case keeps only the latest mesh and U field; set PARAVIEW_MINIMAL_STREAM_TRACER_EXPORT=0 to disable it.\n"
            "Full per-step OpenFOAM cases were not retained because SAVE_MOTION_STEPS=0.\n"
            "Set SAVE_MOTION_STEPS=1 before running if you need actual_model_case/motion_steps/ for debugging.\n"
        )
        enforce_case_storage_budget(root_case, "at run completion")

        # Safety: make sure the folder the user specifically asked to avoid is not left behind.
        if not SAVE_MOTION_STEPS:
            leftover = root_case / "motion_steps"
            if leftover.exists():
                shutil.rmtree(leftover)

    finally:
        if temp_context is not None:
            temp_context.cleanup()

    print(f"Motion log: {motion_log}")
    if ENABLE_NONRIGID_DEFORMATION:
        print(f"Deformation log: {deformation_log}")
    print(f"Case storage used: {human_bytes(directory_size_bytes(root_case))} / {human_bytes(CASE_STORAGE_LIMIT_BYTES)}")
    if not SAVE_MOTION_STEPS:
        print("Full motion_steps folder not retained. Set SAVE_MOTION_STEPS=1 to keep it.")
    if not ROOT_OPENFOAM_TIMESERIES:
        print("Root OpenFOAM time folders not retained. Use the .pvd animation instead.")
    return root_case


def write_visualization_validation_report(case: Path) -> None:
    """Write a blunt report explaining whether ParaView will show real solved fields.

    The most common user-visible failure is colouring by p/Cp/pPa from the
    initial 0/ directory or from raw zeroGradient wall boundary entries. This
    report tells exactly which time directory was used and whether sampled
    surface VTKs were created.
    """
    latest = _latest_solver_time_dir(case)
    times = _numeric_time_dirs(case)
    sampled_count = 0
    post_root = case / "postProcessing"
    if post_root.exists():
        sampled_count = len([p for p in post_root.rglob("*.vtk") if p.is_file()])
    vtk_count = len([p for p in (case / "VTK").rglob("*.vtk") if p.is_file()]) if (case / "VTK").exists() else 0

    lines = [
        "# Visualization validation report",
        f"case={case}",
        f"numeric_time_dirs={[name for _t, path in times for name in [path.name]]}",
        f"selected_latest_time={latest.name if latest else '<none>'}",
        f"sampled_surface_vtk_count={sampled_count}",
        f"foamToVTK_vtk_count={vtk_count}",
        "",
        "field	min	max	range	file",
    ]
    if latest:
        for field in ["p", "pPa", "Cp", "k", "omega", "nut"]:
            fp = latest / field
            vals = _read_scalar_internal_values(fp) if fp.exists() else []
            if vals:
                mn, mx = min(vals), max(vals)
                lines.append(f"{field}	{mn:.10g}	{mx:.10g}	{(mx-mn):.10g}	{fp}")
            else:
                lines.append(f"{field}	<no internal scalar values>	<no internal scalar values>	<no internal scalar values>	{fp}")
    lines.extend([
        "",
        "Interpretation:",
        "- If selected_latest_time is 0, you are still looking at startup fields, not solved CFD.",
        "- If p/pPa/Cp range is zero, pressure colouring will be one solid colour.",
        "- If sampled_surface_vtk_count is zero, the accurate surface-colouring PVD cannot be built from sampled wall fields.",
        "- For moving remeshed geometry, prefer paraview_motion_timeseries.pvd over case.foam.",
    ])
    (case / VISUALIZATION_VALIDATION_REPORT_NAME).write_text("\n".join(lines) + "\n")


# ------------------------- Docker runner / main -------------------------


def _run_docker_command(case: Path, shell_command: str, label: str) -> None:
    parent = case.parent.resolve()
    cmd = [
        "docker", "run", "-it", "--rm",
        "-v", f"{parent}:/root",
        "-w", f"/root/{case.name}",
        DOCKER_IMAGE,
        "bash", "-lc", shell_command,
    ]
    print(f"\nRunning {label} in Docker...")
    subprocess.run(cmd, check=True)


def run_docker(case: Path) -> None:
    # v16 splits the run so Python can create pPa and Cp from the solved p field
    # before foamToVTK exports fields for ParaView.
    _run_docker_command(case, "./Allrun", "OpenFOAM solver")
    write_derived_pressure_fields(case)
    write_pressure_range_report(case)
    # Now that pPa/Cp exist, run sampled-surface post-processing so ParaView sees
    # interpolated solved fields on the moving part surfaces rather than uniform
    # wall-function boundary entries.
    if (case / "Allsurface").exists():
        _run_docker_command(case, "./Allsurface", "sampled surface export")
    if RUN_FULL_VTK_EXPORT:
        _run_docker_command(case, "./Allvtk", "VTK export")
    else:
        print("Skipping full foamToVTK volume export: RUN_FULL_VTK_EXPORT=0 storage-saver mode.")
    write_visualization_validation_report(case)


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_source(source: str) -> int:
    case = Path.cwd() / CASE_NAME

    try:
        if is_onshape_url(source):
            ref = parse_onshape_url(source)
            client = get_onshape_client()
            element_type = detect_onshape_element_type(ref, client)
            print(f"Onshape element detected as: {element_type}")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                if element_type == "assembly":
                    try:
                        components, assembly_def = build_assembly_components(ref, client, tmp)
                    except Exception:
                        # Preserve diagnostics even when strict occurrence export fails.
                        case.mkdir(parents=True, exist_ok=True)
                        for extra_name in (MATERIAL_REPORT_NAME, ASSEMBLY_BOM_NAME, OCCURRENCE_EXPORT_REPORT_NAME, MATE_REPORT_NAME):
                            copy_if_exists(tmp / extra_name, case / extra_name)
                        raise
                    mate_report_text = (tmp / MATE_REPORT_NAME).read_text() if (tmp / MATE_REPORT_NAME).exists() else ""
                    run_assembly_motion_simulation(components, case)
                    (case / "assembly_definition.json").write_text(json.dumps(assembly_def, indent=2))
                    if mate_report_text:
                        (case / MATE_REPORT_NAME).write_text(mate_report_text)
                    for extra_name in (MATERIAL_REPORT_NAME, ASSEMBLY_BOM_NAME, OCCURRENCE_EXPORT_REPORT_NAME):
                        copy_if_exists(tmp / extra_name, case / extra_name)
                else:
                    print("Downloading Part Studio STL from Onshape...")
                    temp_stl = tmp / "onshape_export.stl"
                    stl_path = download_partstudio_stl(ref, client, temp_stl)
                    make_case(stl_path, case)
                    shutil.copy2(stl_path, case / "_onshape_export.stl")
                    run_docker(case)
                    coeff_log = export_force_coefficients(case)
                    print(f"Coefficient TXT: {coeff_log}")
        else:
            stl_path = Path(source).expanduser().resolve()
            if not stl_path.exists():
                raise FileNotFoundError(stl_path)
            make_case(stl_path, case)
            run_docker(case)
            coeff_log = export_force_coefficients(case)
            print(f"Coefficient TXT: {coeff_log}")

        print("\nDone.")
        print(f"OpenFOAM output root: {case}")
        print(f"Coefficient TXT files: {case}/**/{COEFFICIENT_LOG_NAME}")
        print(f"Assembly motion log, when applicable: {case / MOTION_LOG_NAME}")
        print(f"Assembly material report, when applicable: {case / MATERIAL_REPORT_NAME}")
        print(f"Assembly occurrence export report, when applicable: {case / OCCURRENCE_EXPORT_REPORT_NAME}")
        print("Open in ParaView on macOS:")
        print(f"  open -a ParaView {case / PARAVIEW_PVD_NAME}")
        if ROOT_OPENFOAM_TIMESERIES:
            print("Root OpenFOAM time-series was enabled, so this may also work:")
            print(f"  open -a ParaView {case / 'case.foam'}")
        else:
            print("Root case.foam/motion_steps are not created by default in storage-limited mode.")
            print("Set ROOT_OPENFOAM_TIMESERIES=1 or SAVE_MOTION_STEPS=1 only for debugging with more disk space.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python -m cfd_motion '<ONSHAPE_PART_STUDIO_OR_ASSEMBLY_URL>'")
        print("  python -m cfd_motion model.stl")
        return 2
    return run_source(sys.argv[1])
