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


def collision_convergence_should_stop(
    swept_contact: Optional[SweptCollisionContact],
    collision_lines: Sequence[str],
) -> bool:
    """Stop the launch driver only after its selected target was contacted.

    Independent bodies within the moving source can collide before reaching the
    target. Those internal contacts must be resolved, but they are not the event
    that completes the prescribed approach.
    """
    return bool(
        COLLISION_CONVERGENCE_STOP_AFTER_CONTACT
        and swept_contact is not None
        and collision_lines
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
    collision_damage_log = root_case / COLLISION_DAMAGE_LOG_NAME
    collision_convergence_log = root_case / COLLISION_CONVERGENCE_LOG_NAME
    write_unsteady_fsi_gap_report(root_case, components)
    apply_relative_motion_policy(components, root_case)
    collision_convergence_pair = configure_collision_convergence_components(components)
    collision_convergence_active = collision_convergence_pair is not None
    if collision_convergence_pair is not None:
        for collision_component in collision_convergence_pair:
            refined = refine_collision_mesh_for_deformation(collision_component)
            if refined:
                print(
                    f"Collision mesh refinement: {collision_component.patch} "
                    f"added {refined} fixed-topology triangles."
                )
    collision_convergence_axis = (
        collision_convergence_approach_axis(collision_convergence_pair)
        if collision_convergence_pair is not None
        else None
    )
    if collision_convergence_pair is not None:
        write_collision_convergence_log_header(collision_convergence_log, collision_convergence_pair)
        moving, stationary = collision_convergence_moving_and_stationary(collision_convergence_pair)
        print(
            "Collision convergence mode: "
            f"{moving.patch} impacts stationary {stationary.patch}, "
            f"speed {COLLISION_CONVERGENCE_SPEED_MPS:g} m/s."
        )
    initial_overlap_pairs = initial_same_source_overlap_pairs(components)
    if initial_overlap_pairs:
        print(
            "Initial collision regularisation: treating "
            f"{len(initial_overlap_pairs)} same-source CAD overlap(s) as stress-free."
        )
    rigid_group_specs = (
        []
        if collision_convergence_pair is not None
        else assembly_rigid_body_groups(components)
    )
    rigid_body_entries = [
        (group_components, group_root, build_rigid_body_state(group_components, group_root))
        for group_components, group_root in rigid_group_specs
    ]
    rigid_membership = {
        id(component): group_index
        for group_index, (group_components, _root, _state) in enumerate(rigid_body_entries)
        for component in group_components
    }
    write_motion_log_header(motion_log)
    if ENABLE_NONRIGID_DEFORMATION:
        write_deformation_log_header(deformation_log)
    write_collision_damage_log_header(collision_damage_log)
    collision_log.write_text(
        "# Part collision log. Every independent body pair uses continuous swept mesh-surface contact; current overlap is a fallback.\n"
        "# Only members of the same explicit mate-created rigid body are excluded from self-collision.\n"
        "# Swept prescribed impacts preserve launch speed until real triangle surfaces meet.\n"
        "# Curved contact uses Hertz elasticity; flat contact uses the classical elastic flat-punch relation. Thin targets use membrane energy, yielding, and perforation with residual impactor velocity.\n"
        "step\tpass\tpatch_a\tpatch_b\tdepth_m\tnx\tny\tnz\tcx\tcy\tcz\timpulse_Ns\tmove_a_m\tmove_b_m\tdeform_a_m\tdeform_b_m\tcontact_radius_m\tindent_a_m\tindent_b_m\tfailure_mode\tperforated\tabsorbed_energy_J\tresidual_speed_mps\tdisplaced_fragments\tmanifold_points\tfriction_coefficient\n"
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
        f"collision_prescribed_impact_restitution={COLLISION_PRESCRIBED_IMPACT_RESTITUTION}",
        f"collision_friction_coefficient={COLLISION_FRICTION_COEFFICIENT if COLLISION_FRICTION_COEFFICIENT >= 0.0 else '<material-pair>'}",
        f"collision_manifold_tolerance_m={COLLISION_MANIFOLD_TOLERANCE_M}",
        f"collision_max_passes={COLLISION_MAX_PASSES}",
        f"collision_max_linear_speed_mps={COLLISION_MAX_LINEAR_SPEED_MPS}",
        f"collision_max_angular_speed_rad_s={COLLISION_MAX_ANGULAR_SPEED_RAD_S}",
        f"enable_collision_deformation={ENABLE_COLLISION_DEFORMATION}",
        f"collision_deform_impactor={COLLISION_DEFORM_IMPACTOR}",
        f"collision_deformation_model={COLLISION_DEFORMATION_MODEL}",
        f"collision_deformation_gain={COLLISION_DEFORMATION_GAIN}",
        f"collision_deformation_radius_factor={COLLISION_DEFORMATION_RADIUS_FACTOR}",
        f"collision_deformation_min_radius_m={COLLISION_DEFORMATION_MIN_RADIUS_M}",
        f"collision_max_contact_deformation={COLLISION_MAX_CONTACT_DEFORMATION}",
        "collision_contact_coupling=eulerian_sdf_penalty",
        f"collision_eulerian_grid_cells={COLLISION_EULERIAN_GRID_CELLS}",
        f"collision_eulerian_grid_min_cell_m={COLLISION_EULERIAN_GRID_MIN_CELL_M}",
        f"collision_eulerian_penalty_gain={COLLISION_EULERIAN_PENALTY_GAIN}",
        f"collision_mesh_refinement_target_triangles={COLLISION_MESH_REFINEMENT_TARGET_TRIANGLES}",
        f"collision_mesh_refinement_max_levels={COLLISION_MESH_REFINEMENT_MAX_LEVELS}",
        f"collision_damage_min_response_time_s={COLLISION_DAMAGE_MIN_RESPONSE_TIME_S}",
        f"collision_structural_solver={COLLISION_STRUCTURAL_SOLVER}",
        f"collision_shell_damping_ratio={COLLISION_SHELL_DAMPING_RATIO}",
        f"collision_shell_cfl={COLLISION_SHELL_CFL}",
        f"collision_shell_max_substeps={COLLISION_SHELL_MAX_SUBSTEPS}",
        f"collision_shell_displacement_limit_m={COLLISION_SHELL_DISPLACEMENT_LIMIT_M}",
        f"collision_shell_impact_energy_fraction={COLLISION_SHELL_IMPACT_ENERGY_FRACTION}",
        f"collision_topology_changes={ENABLE_COLLISION_TOPOLOGY_CHANGES}",
        f"collision_convergence_speed_mps={COLLISION_CONVERGENCE_SPEED_MPS}",
        f"collision_convergence_components={COLLISION_CONVERGENCE_COMPONENTS or '<auto>'}",
        f"collision_convergence_axis={COLLISION_CONVERGENCE_AXIS}",
        f"collision_initial_gap_m={COLLISION_INITIAL_GAP_M}",
        f"collision_convergence_moving_component={COLLISION_CONVERGENCE_MOVING_COMPONENT}",
        f"collision_sweep_clamping={COLLISION_SWEEP_CLAMPING}",
        f"collision_sweep_penetration_m={COLLISION_SWEEP_PENETRATION_M}",
        f"collision_convergence_stop_after_contact={COLLISION_CONVERGENCE_STOP_AFTER_CONTACT}",
        f"collision_convergence_pair={(collision_convergence_pair[0].patch + ',' + collision_convergence_pair[1].patch) if collision_convergence_pair else '<disabled>'}",
        f"stress_free_initial_same_source_overlaps={len(initial_overlap_pairs)}",
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
            f"young_modulus={inferred_deformation_young_modulus(c):.6g} Pa, "
            f"thickness={inferred_deformation_thickness(c):.6g} m, "
            f"yield_strength={material_yield_strength_pa(c):.6g} Pa, "
            f"failure_strain={material_failure_strain(c):.6g}, "
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
        write_components_geometry_snapshot(
            root_case,
            visualization_components_with_fragments(components),
            START_GEOMETRY_DIR_NAME,
            "start",
        )
        enforce_case_storage_budget(root_case, "after start geometry snapshot")

    completed_step_elapsed_total = 0.0
    try:
        for step in range(ASSEMBLY_DYNAMIC_STEPS):
            step_started = time.monotonic()
            print(f"\nAssembly motion step {step + 1}/{ASSEMBLY_DYNAMIC_STEPS}")
            if step > 0:
                changed_damage_sites = evolve_collision_damage(
                    components,
                    step,
                    MOTION_DT,
                    collision_damage_log,
                )
                if changed_damage_sites:
                    print(
                        "Collision damage evolution: "
                        f"{changed_damage_sites} site(s) changed."
                    )

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
            cfd_elapsed = time.monotonic() - step_started
            motion_started = time.monotonic()
            print(
                f"CFD phase completed in {format_eta(cfd_elapsed)}; "
                "processing motion and collisions."
            )

            export_force_coefficients(step_case)
            export_dimensional_forces(step_case)
            coeffs_by_patch = latest_coefficients_by_patch(step_case)
            loads_by_patch = latest_dimensional_loads_by_patch(step_case)
            write_force_coeff_debug_report(step_case, coeffs_by_patch)
            write_force_load_debug_report(step_case, loads_by_patch)

            rigid_component_rows: List[List[Tuple[AeroComponent, Dict[str, float], str, str, Vec3, Vec3]]] = [
                [] for _entry in rigid_body_entries
            ]
            rigid_net_forces = [(0.0, 0.0, 0.0) for _entry in rigid_body_entries]
            rigid_net_moments = [(0.0, 0.0, 0.0) for _entry in rigid_body_entries]
            rigid_origins = [entry[2].cofr for entry in rigid_body_entries]

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

                rigid_group_index = rigid_membership.get(id(component))
                if rigid_group_index is not None:
                    rigid_origin = rigid_origins[rigid_group_index]
                    force, source_moment = resolve_aerodynamic_load(component, coeffs, load_override)
                    component_moment = total_aerodynamic_moment_about_origin(
                        component,
                        force,
                        source_moment,
                        rigid_origin,
                        load_override is not None,
                    )
                    rigid_component_rows[rigid_group_index].append(
                        (component, coeffs, coeff_source, load_source, force, component_moment)
                    )
                    rigid_net_forces[rigid_group_index] = v_add(
                        rigid_net_forces[rigid_group_index], force
                    )
                    rigid_net_moments[rigid_group_index] = v_add(
                        rigid_net_moments[rigid_group_index], component_moment
                    )
                    continue

                prescribed_impact_motion = (
                    collision_convergence_active
                    and collision_convergence_pair is not None
                    and (
                        component
                        is collision_convergence_moving_and_stationary(
                            collision_convergence_pair
                        )[0]
                        or components_share_collision_source(
                            component,
                            collision_convergence_moving_and_stationary(
                                collision_convergence_pair
                            )[0],
                        )
                    )
                )
                force, moment, dpos, drot = update_component_motion(
                    component,
                    coeffs,
                    MOTION_DT,
                    load_override=load_override,
                    externally_applied_velocity=(
                        v_mul(
                            collision_convergence_axis,
                            COLLISION_CONVERGENCE_SPEED_MPS,
                        )
                        if prescribed_impact_motion
                        and collision_convergence_axis is not None
                        else None
                    ),
                )
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

            if rigid_body_entries:
                for group_index, (group_components, group_root, rigid_body_state) in enumerate(rigid_body_entries):
                    rigid_origin = rigid_origins[group_index]
                    rigid_force, rigid_moment, rigid_dpos, rigid_drot = update_component_motion(
                        rigid_body_state,
                        {},
                        MOTION_DT,
                        load_override=(
                            rigid_net_forces[group_index],
                            rigid_net_moments[group_index],
                        ),
                    )
                    apply_rigid_body_motion(
                        group_components,
                        rigid_dpos,
                        rigid_drot,
                        rigid_origin,
                        rigid_body_state.linear_velocity,
                        rigid_body_state.angular_velocity,
                        rigid_body_state.total_translation,
                        rigid_body_state.total_rotation,
                    )
                    for component, coeffs, coeff_source, load_source, force, moment in rigid_component_rows[group_index]:
                        log_force = rigid_force if component is group_root else force
                        log_moment = rigid_moment if component is group_root else moment
                        log_load_source = "assembly-rigid-body-net" if component is group_root else load_source
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

            motion_elapsed = time.monotonic() - motion_started
            collision_started = time.monotonic()
            swept_contact = None
            if collision_convergence_active and collision_convergence_pair is not None:
                move_a, move_b, swept_contact = apply_collision_convergence_step(
                    collision_convergence_pair,
                    step,
                    MOTION_DT,
                    collision_convergence_log,
                    collision_convergence_axis,
                    components,
                )
                if (
                    ENABLE_COLLISION_TOPOLOGY_CHANGES
                    and swept_contact is not None
                    and swept_contact.perforated
                ):
                    added_triangles = refine_thin_impact_target(swept_contact.stationary)
                    if added_triangles:
                        print(
                            f"Thin impact target refinement: added {added_triangles} triangles "
                            "for local bending/fracture."
                        )
                print(
                    "Collision convergence step: "
                    f"impactor moved {v_norm(move_a):.6g} m, "
                    f"target moved {v_norm(move_b):.6g} m."
                )
            collision_lines = resolve_part_collisions(
                components,
                step,
                collision_log,
                swept_contact if collision_convergence_active else None,
                (
                    collision_convergence_pair
                    if collision_convergence_active
                    else None
                ),
                initial_overlap_pairs,
            )
            if collision_lines:
                print(f"Collision resolution: {len(collision_lines)} contact event(s) handled at step {step}.")
                evolve_collision_damage(
                    components,
                    step,
                    0.0,
                    collision_damage_log,
                )
                if (
                    collision_convergence_active
                    and collision_convergence_should_stop(
                        swept_contact,
                        collision_lines,
                    )
                ):
                    collision_convergence_active = False
                    append_collision_convergence_stop(
                        collision_convergence_log,
                        step,
                        "prescribed_target_contact",
                    )

            fragment_aero_count = advance_detached_fragment_aerodynamics(
                components,
                MOTION_DT,
            )
            if fragment_aero_count:
                print(
                    "Fragment aerodynamics: "
                    f"updated {fragment_aero_count} detached body/bodies."
                )
            collision_elapsed = time.monotonic() - collision_started
            output_started = time.monotonic()

            # Write compact visualisation AFTER the motion and collision update so the frame
            # shows moved, non-interpenetrating geometry.
            write_panel_aero_preview_for_step(
                step_case,
                visualization_components_with_fragments(components),
                step,
            )

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

            output_elapsed = time.monotonic() - output_started
            complete_step_elapsed = time.monotonic() - step_started
            completed_step_elapsed_total += complete_step_elapsed
            save_eta_calibration(complete_step_elapsed)
            steps_done_for_eta = step + 1
            steps_left_for_eta = ASSEMBLY_DYNAMIC_STEPS - steps_done_for_eta
            elapsed_total_for_eta = time.monotonic() - run_started
            avg_step_for_eta = (
                completed_step_elapsed_total / max(1, steps_done_for_eta)
            )
            remaining_for_eta = avg_step_for_eta * steps_left_for_eta
            print(
                "Step timing: "
                f"CFD {format_eta(cfd_elapsed)}, "
                f"motion {format_eta(motion_elapsed)}, "
                f"collision {format_eta(collision_elapsed)}, "
                f"output {format_eta(output_elapsed)}, "
                f"complete {format_eta(complete_step_elapsed)}."
            )
            print(
                f"ETA update: step {steps_done_for_eta}/{ASSEMBLY_DYNAMIC_STEPS} took "
                f"{format_eta(complete_step_elapsed)}; "
                f"remaining about {format_eta(remaining_for_eta)}; "
                f"total about {format_eta(elapsed_total_for_eta + remaining_for_eta)}."
            )

        if STORE_START_FINAL_GEOMETRY:
            write_components_geometry_snapshot(
                root_case,
                visualization_components_with_fragments(components),
                FINAL_MOVED_GEOMETRY_DIR_NAME,
                "final_moved",
            )
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


def _source_label(source: str, index: int) -> str:
    if is_onshape_url(source):
        return f"object_{index}"
    path = Path(source).expanduser()
    return path.stem or f"object_{index}"


def _component_from_triangles(name: str, patch: str, triangles: Sequence[Triangle]) -> AeroComponent:
    aref, lref, cofr = component_references(triangles)
    component = AeroComponent(name, patch, list(triangles), cofr, lref, aref, freedom=six_dof_motion_freedom("two-source-collision"))
    material = infer_material_from_name(name)
    apply_material_model(component, material, None, None, "name/default")
    return component


def _combined_source_component(name: str, patch: str, components: Sequence[AeroComponent]) -> AeroComponent:
    triangles = [triangle for component in components for triangle in component.triangles]
    if not triangles:
        raise ValueError(f"No triangles were imported for {name}")
    combined = _component_from_triangles(name, patch, triangles)
    total_mass = sum(max(component.mass, 0.0) for component in components)
    material_components = [
        component
        for component in components
        if component.material.material_name.lower() not in {"", "unknown", "default"}
    ]
    representative = max(material_components or list(components), key=lambda component: max(component.mass, 0.0))
    combined.material = MaterialProperties(**vars(representative.material))
    if total_mass > 0.0:
        combined.mass = total_mass
        combined.material.mass_kg = total_mass
        combined.material.volume_m3 = sum(
            max(component.material.volume_m3 or 0.0, 0.0)
            for component in components
        )
        combined.material.source = f"combined-source/{representative.material.source}"
        combined.material.structural_source = (
            f"combined-source/{representative.material.structural_source}"
        )
        combined.inertia = estimate_scalar_inertia(total_mass, combined.triangles)
    return combined


def _split_unmated_component_bodies(component: AeroComponent) -> List[AeroComponent]:
    """Split disconnected STL solids into independent rigid collision bodies."""
    body_ids = connected_surface_body_ids(component)
    body_count = max(body_ids, default=-1) + 1
    if body_count <= 1 or component.freedom.source.startswith("mate:"):
        return [component]

    triangle_groups = [
        [
            triangle
            for triangle, triangle_body_id in zip(component.triangles, body_ids)
            if triangle_body_id == body_id
        ]
        for body_id in range(body_count)
    ]
    volumes = [estimate_closed_mesh_volume(triangles) for triangles in triangle_groups]
    areas = [
        sum(
            triangle_area_centroid_normal(triangle)[0]
            for triangle in triangles
        )
        for triangles in triangle_groups
    ]
    weights = volumes if sum(volumes) > 1e-18 else areas
    total_weight = max(sum(weights), 1e-18)
    bodies: List[AeroComponent] = []
    for body_id, (triangles, weight, volume) in enumerate(
        zip(triangle_groups, weights, volumes)
    ):
        aref, lref, cofr = component_references(triangles)
        mass = component.mass * weight / total_weight
        material = MaterialProperties(**vars(component.material))
        material.mass_kg = mass
        material.volume_m3 = volume
        body = AeroComponent(
            name=f"{component.name} body {body_id + 1}",
            patch=f"{component.patch}_body_{body_id + 1}",
            triangles=list(triangles),
            cofr=cofr,
            lref=lref,
            aref=aref,
            freedom=MotionFreedom(
                translate_axes=list(component.freedom.translate_axes),
                rotate_axes=list(component.freedom.rotate_axes),
                mate_type=component.freedom.mate_type,
                source=component.freedom.source,
                limits=dict(component.freedom.limits),
                mate_origin=component.freedom.mate_origin,
                mate_reference_origin=component.freedom.mate_reference_origin,
                mate_reference_occurrence=(
                    component.freedom.mate_reference_occurrence
                ),
                mate_x_axis=component.freedom.mate_x_axis,
                mate_y_axis=component.freedom.mate_y_axis,
                mate_z_axis=component.freedom.mate_z_axis,
                mate_reference_x_axis=component.freedom.mate_reference_x_axis,
                mate_reference_y_axis=component.freedom.mate_reference_y_axis,
                mate_reference_z_axis=component.freedom.mate_reference_z_axis,
            ),
            material=material,
            mass=mass,
            inertia=estimate_scalar_inertia(mass, triangles),
            linear_velocity=component.linear_velocity,
            angular_velocity=component.angular_velocity,
            source_occurrence=(
                f"{component.source_occurrence or component.patch}/body_{body_id + 1}"
            ),
            collision_family=f"{component.patch}_body_{body_id + 1}",
            collision_source_index=component.collision_source_index,
        )
        bodies.append(body)
    return bodies


def _subdivide_triangle(triangle: Triangle) -> List[Triangle]:
    _normal, a, b, c = triangle
    ab = v_mul(v_add(a, b), 0.5)
    bc = v_mul(v_add(b, c), 0.5)
    ca = v_mul(v_add(c, a), 0.5)
    points = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
    return [
        (v_unit(v_cross(v_sub(q, p), v_sub(r, p))), p, q, r)
        for p, q, r in points
    ]


def refine_thin_impact_target(component: AeroComponent) -> int:
    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
    extents = sorted((xmax - xmin, ymax - ymin, zmax - zmin))
    if extents[0] <= 1e-9 or extents[0] >= 0.05 * max(extents[2], 1e-9):
        return 0
    target_edge = max(extents[2] / 16.0, extents[0] * 4.0)
    triangles = list(component.triangles)
    original_count = len(triangles)
    for _level in range(6):
        longest = max(
            max(v_norm(v_sub(b, a)), v_norm(v_sub(c, b)), v_norm(v_sub(a, c)))
            for _normal, a, b, c in triangles
        )
        if longest <= target_edge:
            break
        triangles = [child for triangle in triangles for child in _subdivide_triangle(triangle)]
    component.triangles = triangles
    component.aref, component.lref, component.cofr = component_references(triangles)
    return len(triangles) - original_count


def refine_collision_mesh_for_deformation(component: AeroComponent) -> int:
    """Refine only very coarse collision meshes before the first frame.

    Collision dents are fixed-topology geometry edits.  Refining once up front
    gives the dent enough local vertices while keeping every exported frame
    compatible with ParaView and preventing a coarse face from translating an
    entire target.
    """
    target_count = COLLISION_MESH_REFINEMENT_TARGET_TRIANGLES
    if target_count <= 0 or len(component.triangles) >= target_count:
        return 0
    triangles = list(component.triangles)
    original_count = len(triangles)
    for _level in range(COLLISION_MESH_REFINEMENT_MAX_LEVELS):
        if len(triangles) >= target_count:
            break
        triangles = [child for triangle in triangles for child in _subdivide_triangle(triangle)]
    if len(triangles) == original_count:
        return 0
    component.triangles = triangles
    component.aref, component.lref, component.cofr = component_references(triangles)
    return len(triangles) - original_count


def build_collision_source_components(source: str, index: int, workdir: Path, client: Optional[OnshapeClient] = None) -> List[AeroComponent]:
    label = _source_label(source, index)
    patch = unique_patch_names([label])[0]

    if is_onshape_url(source):
        ref = parse_onshape_url(source)
        client = client or get_onshape_client()
        element_type = detect_onshape_element_type(ref, client)
        print(f"Collision source {index}: Onshape element detected as {element_type}")
        if element_type == "assembly":
            source_dir = workdir / f"source_{index:02d}"
            source_dir.mkdir(parents=True, exist_ok=True)
            components, assembly_def = build_assembly_components(ref, client, source_dir)
            (workdir / f"source_{index:02d}_assembly_definition.json").write_text(json.dumps(assembly_def, indent=2))
            for component in components:
                component.collision_source_index = index
            return components

        stl_path = download_partstudio_stl(ref, client, workdir / f"source_{index:02d}.stl")
        component = _component_from_triangles(label, patch, read_stl_triangles(stl_path))
        component.collision_source_index = index
        return _split_unmated_component_bodies(component)

    stl_path = Path(source).expanduser().resolve()
    if not stl_path.exists():
        raise FileNotFoundError(stl_path)
    component = _component_from_triangles(label, patch, read_stl_triangles(stl_path))
    component.collision_source_index = index
    return _split_unmated_component_bodies(component)


def build_collision_source_component(source: str, index: int, workdir: Path, client: Optional[OnshapeClient] = None) -> AeroComponent:
    """Compatibility wrapper for callers that require one merged component."""
    components = build_collision_source_components(source, index, workdir, client)
    if len(components) == 1:
        return components[0]
    return _combined_source_component(_source_label(source, index), f"object_{index}", components)


def run_sources(sources: Sequence[str]) -> int:
    if len(sources) != 2:
        raise ValueError("run_sources expects exactly two sources")
    case = Path.cwd() / CASE_NAME

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            client = get_onshape_client() if any(is_onshape_url(source) for source in sources) else None
            source_component_groups = [
                build_collision_source_components(source, index + 1, tmp, client)
                for index, source in enumerate(sources)
            ]
            components = [
                component
                for source_components in source_component_groups
                for component in source_components
            ]
            patches = unique_patch_names([component.patch for component in components])
            for component, unique_patch in zip(components, patches):
                component.patch = unique_patch
            if COLLISION_CONVERGENCE_SPEED_MPS <= 0.0:
                raise ValueError("Two-source collision runs need COLLISION_CONVERGENCE_SPEED_MPS > 0")
            pair = configure_collision_convergence_components(components)
            if pair is not None:
                for collision_component in pair:
                    refined = refine_collision_mesh_for_deformation(collision_component)
                    if refined:
                        print(
                            f"Collision mesh refinement: {collision_component.patch} "
                            f"added {refined} fixed-topology triangles."
                        )
                shift = arrange_collision_convergence_initial_gap(pair, components)
                moving, stationary = collision_convergence_moving_and_stationary(pair)
                axis = collision_convergence_approach_axis(pair)
                print(
                    f"Initial collision layout: placed {moving.patch} directly behind "
                    f"stationary {stationary.patch}, moved impactor by {v_norm(shift):.6g} m; "
                    f"axis=({axis[0]:g}, {axis[1]:g}, {axis[2]:g}), "
                    f"target gap {COLLISION_INITIAL_GAP_M:g} m."
                )
            run_assembly_motion_simulation(components, case)
            for source_index in range(1, len(sources) + 1):
                source_dir = tmp / f"source_{source_index:02d}"
                for report_name in (
                    MATERIAL_REPORT_NAME,
                    ASSEMBLY_BOM_NAME,
                    OCCURRENCE_EXPORT_REPORT_NAME,
                    MATE_REPORT_NAME,
                ):
                    copy_if_exists(
                        source_dir / report_name,
                        case / f"source_{source_index:02d}_{report_name}",
                    )
                copy_if_exists(
                    tmp / f"source_{source_index:02d}_assembly_definition.json",
                    case / f"source_{source_index:02d}_assembly_definition.json",
                )

        print("\nDone.")
        print(f"OpenFOAM output root: {case}")
        print(f"Collision convergence log: {case / COLLISION_CONVERGENCE_LOG_NAME}")
        print(f"Collision contact log: {case / COLLISION_LOG_NAME}")
        print(f"Collision damage evolution log: {case / COLLISION_DAMAGE_LOG_NAME}")
        print("Open in ParaView on macOS:")
        print(f"  open -a ParaView {case / PARAVIEW_PVD_NAME}")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


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
        print(f"Collision damage log, when applicable: {case / COLLISION_DAMAGE_LOG_NAME}")
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
    if len(sys.argv) not in {2, 3}:
        print("Usage:")
        print("  python -m cfd_motion '<ONSHAPE_PART_STUDIO_OR_ASSEMBLY_URL>'")
        print("  python -m cfd_motion '<OBJECT_A_URL>' '<OBJECT_B_URL>'")
        print("  python -m cfd_motion model.stl")
        return 2
    if len(sys.argv) == 3:
        return run_sources([sys.argv[1], sys.argv[2]])
    return run_source(sys.argv[1])
