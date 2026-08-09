import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import xml.etree.ElementTree as ElementTree

import cfd_motion.fem_mpm as fem_mpm_module
import pytest
from cfd_motion.fem_mpm import (
    HybridFEMMPMState,
    MPMParticle,
    advance_fem,
    advance_mpm,
    element_force_and_energy,
    make_tetra_element,
)
from cfd_motion.models import (
    AeroComponent,
    CollisionDamageState,
    MaterialProperties,
)
from cfd_motion.motion import (
    append_collision_conservation_audits,
    apply_collision_impulse,
    deform_component_at_contact,
    evolve_collision_damage,
    persistent_contact_for_pair,
    write_collision_conservation_log_header,
)
from cfd_motion.structural import (
    advance_hybrid_fem_mpm_collision,
    apply_fem_impact_energy,
    build_hybrid_fem_mpm_collision_state,
    fem_surface_von_mises_stress_pa,
    refresh_hybrid_fem_mpm_geometry,
)
from cfd_motion.visualization import write_panel_aero_preview_for_step


def _state():
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    element = make_tetra_element(positions, (0, 1, 2, 3), 1.0e7, 0.3, 1.0e12, 1.0)
    return HybridFEMMPMState(positions, [(2.0, 0.0, 0.0)] * 4, [1.0] * 4, [element])


def _two_tetra_state(failure_strain=0.25):
    reference_positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    elements = [
        make_tetra_element(
            reference_positions,
            (0, 1, 2, 3),
            2.5e6,
            0.27,
            5.0e3,
            failure_strain,
            0.8,
        ),
        make_tetra_element(
            reference_positions,
            (0, 2, 1, 4),
            3.1e6,
            0.31,
            7.0e3,
            failure_strain,
            0.7,
        ),
    ]
    positions = list(reference_positions)
    positions[1] = (1.006, 0.002, -0.001)
    positions[2] = (-0.002, 0.996, 0.003)
    return HybridFEMMPMState(
        positions=positions,
        velocities=[
            (0.03, -0.02, 0.01),
            (-0.01, 0.04, 0.0),
            (0.02, 0.01, -0.03),
            (0.0, -0.02, 0.04),
            (-0.03, 0.0, -0.01),
        ],
        masses_kg=[0.7, 0.8, 0.9, 0.6, 0.5],
        elements=elements,
        cfl=0.35,
        max_substeps=256,
    )


def _assert_vectors_close(left, right, *, rel_tol=1e-11, abs_tol=1e-11):
    assert len(left) == len(right)
    for left_vector, right_vector in zip(left, right):
        for left_value, right_value in zip(left_vector, right_vector):
            assert math.isclose(
                left_value,
                right_value,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )


def test_fem_translation_preserves_mass_and_momentum():
    state = _state()
    audit = advance_fem(state, 1.0e-4)
    assert math.isclose(audit.mass_before_kg, audit.mass_after_kg, abs_tol=1.0e-12)
    assert all(abs(value) < 1.0e-8 for value in audit.momentum_error)


def test_fem_automatically_batches_beyond_configured_substep_batch_size():
    batched_state = _state()
    unbatched_state = copy.deepcopy(batched_state)
    batched_state.max_substeps = 1
    unbatched_state.max_substeps = 4096

    batched_audit = advance_fem(batched_state, 1.0e-2)
    unbatched_audit = advance_fem(unbatched_state, 1.0e-2)

    _assert_vectors_close(batched_state.positions, unbatched_state.positions)
    _assert_vectors_close(batched_state.velocities, unbatched_state.velocities)
    assert math.isclose(
        batched_audit.mass_after_kg,
        unbatched_audit.mass_after_kg,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_fem_processes_4345_required_substeps_as_two_safe_batches():
    state = _state()
    state.max_substeps = 4096
    motion_dt = 0.0012
    with (
        patch(
            "cfd_motion.fem_mpm.stable_fem_timestep",
            return_value=motion_dt / 4344.5,
        ),
        patch(
            "cfd_motion.fem_mpm._advance_fem_substeps_vectorized",
            return_value=(0.0, 0.0),
        ) as advance_batch,
    ):
        advance_fem(state, motion_dt)

    assert [call.args[1] for call in advance_batch.call_args_list] == [4096, 249]
    assert all(
        math.isclose(
            call.args[2],
            motion_dt / 4345,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        for call in advance_batch.call_args_list
    )


def test_failed_element_transfers_mass_to_mpm_particles_without_loss():
    state = _state()
    state.elements[0].failed = True
    before = state.total_mass_kg
    advance_fem(state, 1.0e-4)
    assert state.use_mpm
    assert len(state.particles) == 4
    assert math.isclose(before, state.total_mass_kg, abs_tol=1.0e-12)


def test_elastic_tetra_responds_to_force():
    state = _state()
    forces = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    advance_fem(state, 1.0e-3, forces)
    assert state.positions[1][0] > 1.0


def test_corotational_tetra_has_balanced_restoring_force():
    state = _state()
    state.positions[1] = (1.02, 0.0, 0.0)
    forces, energy, _plastic = element_force_and_energy(state, state.elements[0])
    total = tuple(sum(force[axis] for force in forces) for axis in range(3))
    assert all(abs(value) < 1.0e-9 for value in total)
    assert forces[1][0] < 0.0
    assert energy > 0.0


def test_bom_failure_strain_controls_brittle_fragmentation():
    brittle = _state()
    ductile = _state()
    brittle.elements[0].failure_strain = 1e-3
    ductile.elements[0].failure_strain = 0.5
    brittle.positions[1] = (1.02, 0.0, 0.0)
    ductile.positions[1] = (1.02, 0.0, 0.0)
    element_force_and_energy(brittle, brittle.elements[0])
    element_force_and_energy(ductile, ductile.elements[0])
    assert brittle.elements[0].failed
    assert not ductile.elements[0].failed


def test_linear_tetra_matches_isotropic_uniaxial_strain_stress():
    state = _state()
    epsilon = 1e-4
    state.positions[1] = (1.0 + epsilon, 0.0, 0.0)
    element_force_and_energy(state, state.elements[0])
    young = state.elements[0].young_modulus_pa
    poisson = state.elements[0].poisson_ratio
    lame_lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    shear = young / (2 * (1 + poisson))
    expected_x = (lame_lambda + 2 * shear) * epsilon
    assert math.isclose(
        state.elements[0].stress[0][0],
        expected_x,
        rel_tol=2e-4,
    )


def test_corotational_tetra_rejects_rigid_rotation_strain():
    state = _state()
    state.positions = [
        (-point[1], point[0], point[2])
        for point in state.positions
    ]
    forces, energy, _plastic = element_force_and_energy(state, state.elements[0])
    assert energy < 1e-12
    assert max(abs(value) for force in forces for value in force) < 1e-7


def test_two_failed_tetrahedra_transfer_their_exact_mass_once():
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    elements = [
        make_tetra_element(positions, (0, 1, 2, 3), 1e7, 0.3, 1e6, 0.1, 1.0),
        make_tetra_element(positions, (0, 2, 1, 4), 1e7, 0.3, 1e6, 0.1, 1.0),
    ]
    masses = [0.5, 0.5, 0.5, 0.25, 0.25]
    state = HybridFEMMPMState(
        positions,
        [(3.0, 0.0, 0.0)] * len(positions),
        masses,
        elements,
    )
    for element in elements:
        element.failed = True
    audit = advance_fem(state, 1e-5)
    assert math.isclose(audit.mass_before_kg, 2.0, abs_tol=1e-12)
    assert math.isclose(audit.mass_after_kg, 2.0, abs_tol=1e-12)
    assert math.isclose(sum(p.mass_kg for p in state.particles), 2.0, abs_tol=1e-12)
    advance_fem(state, 1e-5)
    assert len(state.particles) == 8


def test_vectorized_fem_matches_scalar_constitutive_and_motion_results():
    scalar_state = _two_tetra_state()
    vectorized_state = copy.deepcopy(scalar_state)
    external_forces = [
        (0.3, -0.2, 0.1),
        (-0.1, 0.05, 0.0),
        (0.0, 0.2, -0.1),
        (0.04, 0.0, 0.03),
        (-0.02, -0.04, 0.01),
    ]

    with patch.object(fem_mpm_module, "np", None):
        scalar_audit = advance_fem(scalar_state, 1.5e-3, external_forces)
    vectorized_audit = advance_fem(
        vectorized_state,
        1.5e-3,
        external_forces,
    )

    _assert_vectors_close(scalar_state.positions, vectorized_state.positions)
    _assert_vectors_close(scalar_state.velocities, vectorized_state.velocities)
    for scalar_element, vectorized_element in zip(
        scalar_state.elements, vectorized_state.elements
    ):
        assert scalar_element.failed == vectorized_element.failed
        assert math.isclose(
            scalar_element.equivalent_plastic_strain,
            vectorized_element.equivalent_plastic_strain,
            rel_tol=1e-11,
            abs_tol=1e-12,
        )
        assert math.isclose(
            scalar_element.strain_energy_j,
            vectorized_element.strain_energy_j,
            rel_tol=1e-11,
            abs_tol=1e-11,
        )
        _assert_vectors_close(
            scalar_element.stress,
            vectorized_element.stress,
            rel_tol=1e-11,
            abs_tol=1e-9,
        )
    assert math.isclose(
        scalar_audit.external_work_j,
        vectorized_audit.external_work_j,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )
    assert math.isclose(
        scalar_audit.plastic_dissipation_j,
        vectorized_audit.plastic_dissipation_j,
        rel_tol=1e-11,
        abs_tol=1e-15,
    )
    assert math.isclose(vectorized_audit.mass_error_kg, 0.0, abs_tol=1e-12)


def test_vectorized_failure_transfer_matches_scalar_and_conserves_mass():
    scalar_state = _two_tetra_state(failure_strain=1e-4)
    vectorized_state = copy.deepcopy(scalar_state)
    zero_forces = [(0.0, 0.0, 0.0)] * len(scalar_state.positions)

    with patch.object(fem_mpm_module, "np", None):
        scalar_audit = advance_fem(scalar_state, 1e-5, zero_forces)
    vectorized_audit = advance_fem(vectorized_state, 1e-5, zero_forces)

    assert all(element.failed for element in vectorized_state.elements)
    assert all(element.transferred for element in vectorized_state.elements)
    assert len(scalar_state.particles) == len(vectorized_state.particles) == 8
    assert scalar_state.masses_kg == vectorized_state.masses_kg
    for scalar_particle, vectorized_particle in zip(
        scalar_state.particles, vectorized_state.particles
    ):
        assert scalar_particle.source_element == vectorized_particle.source_element
        assert scalar_particle.source_node == vectorized_particle.source_node
        assert math.isclose(
            scalar_particle.mass_kg,
            vectorized_particle.mass_kg,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        _assert_vectors_close(
            [scalar_particle.position],
            [vectorized_particle.position],
        )
        _assert_vectors_close(
            [scalar_particle.velocity],
            [vectorized_particle.velocity],
        )
    assert math.isclose(scalar_audit.mass_error_kg, 0.0, abs_tol=1e-12)
    assert math.isclose(vectorized_audit.mass_error_kg, 0.0, abs_tol=1e-12)
    assert math.isclose(
        scalar_audit.mass_after_kg,
        vectorized_audit.mass_after_kg,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _cube_component() -> AeroComponent:
    p000 = (0.0, 0.0, 0.0)
    p100 = (1.0, 0.0, 0.0)
    p010 = (0.0, 1.0, 0.0)
    p110 = (1.0, 1.0, 0.0)
    p001 = (0.0, 0.0, 1.0)
    p101 = (1.0, 0.0, 1.0)
    p011 = (0.0, 1.0, 1.0)
    p111 = (1.0, 1.0, 1.0)
    faces = [
        (p000, p010, p110), (p000, p110, p100),
        (p001, p101, p111), (p001, p111, p011),
        (p000, p100, p101), (p000, p101, p001),
        (p010, p011, p111), (p010, p111, p110),
        (p000, p001, p011), (p000, p011, p010),
        (p100, p110, p111), (p100, p111, p101),
    ]
    triangles = []
    for a, b, c in faces:
        cross = (
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        )
        triangles.append((cross, a, b, c))
    return AeroComponent(
        name="cube",
        patch="cube",
        triangles=triangles,
        cofr=(0.5, 0.5, 0.5),
        lref=1.0,
        aref=1.0,
        material=MaterialProperties(
            material_name="steel",
            density_kg_m3=7800.0,
            young_modulus_pa=2e11,
            poisson_ratio=0.3,
            yield_strength_pa=2.5e8,
            failure_strain=0.2,
        ),
        mass=7.8,
    )


def test_solid_adapter_preserves_surface_and_fragment_mass():
    component = _cube_component()
    initial_mass = component.mass
    state = build_hybrid_fem_mpm_collision_state(
        component, 2e11, 0.3, 2.5e8, 0.2, 0.3, 32
    )
    assert len(state.solid_state.elements) == len(component.triangles)
    assert all(element.young_modulus_pa == 2e11 for element in state.solid_state.elements)
    assert all(element.yield_stress_pa == 2.5e8 for element in state.solid_state.elements)
    assert all(element.failure_strain == 0.2 for element in state.solid_state.elements)
    state.solid_state.elements[0].failed = True
    _deformation, fragments, _mass = advance_hybrid_fem_mpm_collision(
        component, state, 1e-6
    )
    assert fragments == 1
    assert len(component.triangles) == 11
    assert math.isclose(
        component.mass + sum(fragment.mass_kg for fragment in state.fragment_bodies),
        initial_mass,
        rel_tol=1e-12,
    )
    assert len(fem_surface_von_mises_stress_pa(state)) == len(component.triangles)


def test_fem_impact_energy_does_not_create_net_linear_momentum():
    component = _cube_component()
    state = build_hybrid_fem_mpm_collision_state(
        component, 2e11, 0.3, 2.5e8, 0.2, 0.3, 32
    )
    before = state.solid_state.total_momentum()
    apply_fem_impact_energy(
        state,
        (1.0, 0.5, 0.5),
        (-1.0, 0.0, 0.0),
        0.3,
        10.0,
        1.0,
    )
    after = state.solid_state.total_momentum()
    assert all(abs(after[i] - before[i]) < 1e-12 for i in range(3))


def test_persistent_contact_only_restarts_after_separation():
    a = _cube_component()
    b = _cube_component()
    b.patch = "cube_b"
    _contact, is_new = persistent_contact_for_pair(
        a, b, 3, (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)
    )
    assert is_new
    contact, is_new = persistent_contact_for_pair(
        a, b, 4, (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)
    )
    assert not is_new
    assert contact.age_steps == 2
    _contact, is_new = persistent_contact_for_pair(
        a, b, 7, (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)
    )
    assert is_new


def test_fem_surface_exports_one_stress_value_per_triangle():
    component = _cube_component()
    state = build_hybrid_fem_mpm_collision_state(
        component, 2e11, 0.3, 2.5e8, 0.2, 0.3, 32
    )
    component.collision_structural_state = state
    surface_node = state.solid_state.surface_triangle_nodes[0][0]
    x, y, z = state.solid_state.positions[surface_node]
    state.solid_state.positions[surface_node] = (x + 1e-4, y, z)
    state.solid_state.velocities[surface_node] = (9.0, 0.0, 0.0)
    element_force_and_energy(state.solid_state, state.solid_state.elements[0])
    refresh_hybrid_fem_mpm_geometry(component, state)
    with TemporaryDirectory() as temp_dir:
        case = Path(temp_dir)
        write_panel_aero_preview_for_step(case, [component], 0)
        output = case / "panel_preview" / "combined_moving_surfaces.vtp"
        piece = ElementTree.parse(output).getroot().find(".//Piece")
        assert piece is not None
        triangle_count = int(piece.attrib["NumberOfPolys"])
        stress = piece.find("./CellData/DataArray[@Name='vonMisesStressPa']")
        assert stress is not None
        assert len((stress.text or "").split()) == triangle_count
        velocity = piece.find("./PointData/DataArray[@Name='velocity']")
        assert velocity is not None
        velocity_values = [float(value) for value in (velocity.text or "").split()]
        assert 3.0 in velocity_values


def test_fem_fragment_keeps_collision_impulse_and_deformation():
    component = _cube_component()
    state = build_hybrid_fem_mpm_collision_state(
        component, 2e11, 0.3, 2.5e8, 0.2, 0.3, 32
    )
    state.solid_state.elements[0].failed = True
    advance_hybrid_fem_mpm_collision(component, state, 1e-6)
    fragment = state.fragment_bodies[0].component
    particles = [
        particle
        for particle in state.solid_state.particles
        if particle.source_element == 0
    ]
    before_momentum = sum(p.mass_kg * p.velocity[0] for p in particles)
    apply_collision_impulse(fragment, (0.2, 0.0, 0.0), fragment.cofr)
    after_momentum = sum(p.mass_kg * p.velocity[0] for p in particles)
    assert math.isclose(after_momentum - before_momentum, 0.2, rel_tol=1e-10)
    before_positions = [particle.position for particle in particles]
    deform_component_at_contact(
        fragment,
        fragment.cofr,
        (0.0, 0.0, 1.0),
        1e-3,
        max(fragment.lref, 1e-3),
    )
    assert any(
        particle.position != before
        for particle, before in zip(particles, before_positions)
    )


def test_conservation_audit_is_written_for_each_structural_step():
    component = _cube_component()
    state = build_hybrid_fem_mpm_collision_state(
        component, 2e11, 0.3, 2.5e8, 0.2, 0.3, 32
    )
    component.collision_structural_state = state
    advance_hybrid_fem_mpm_collision(component, state, 1e-7)
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "conservation.txt"
        write_collision_conservation_log_header(path)
        assert append_collision_conservation_audits([component], 4, path) == 1
        rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
        assert len(rows) == 2
        assert rows[1].split("\t")[0:2] == ["4", "cube"]


def test_multiple_damage_sites_advance_fem_once_per_global_step():
    component = _cube_component()
    component.collision_structural_state = build_hybrid_fem_mpm_collision_state(
        component, 2e11, 0.3, 2.5e8, 0.2, 0.3, 32
    )
    component.collision_damage = [
        CollisionDamageState(
            contact_point=(1.0, 0.25, 0.25),
            inward_direction=(-1.0, 0.0, 0.0),
            contact_radius_m=0.1,
            current_depth_m=0.0,
            permanent_depth_m=0.0,
            current_hole_radius_m=0.05,
            target_hole_radius_m=0.05,
            created_step=0,
        ),
        CollisionDamageState(
            contact_point=(1.0, 0.75, 0.75),
            inward_direction=(-1.0, 0.0, 0.0),
            contact_radius_m=0.1,
            current_depth_m=0.0,
            permanent_depth_m=0.0,
            current_hole_radius_m=0.05,
            target_hole_radius_m=0.05,
            created_step=0,
        ),
        # A newly-created site is deliberately last. It must not prevent the
        # already-active sites from advancing the shared structural body.
        CollisionDamageState(
            contact_point=(1.0, 0.5, 0.5),
            inward_direction=(-1.0, 0.0, 0.0),
            contact_radius_m=0.1,
            current_depth_m=0.0,
            permanent_depth_m=0.0,
            created_step=1,
        ),
    ]

    with TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "damage.tsv"
        with patch(
            "cfd_motion.motion.advance_hybrid_fem_mpm_collision",
            return_value=(0.004, 0, 0.0),
        ) as advance:
            evolve_collision_damage([component], 1, 1e-4, log_path)

    advance.assert_called_once()
    assert component.collision_damage[0].elapsed_s == 1e-4
    assert component.collision_damage[1].elapsed_s == 1e-4
    assert component.collision_damage[2].elapsed_s == 0.0
    assert component.collision_damage[0].permanent_depth_m == 0.004
    assert component.collision_damage[1].permanent_depth_m == 0.004


def test_mpm_grid_transfer_projects_small_angular_drift_conservatively():
    particles = [
        MPMParticle(
            position=(0.13, 0.21, 0.07),
            velocity=(-0.21, 0.13, 0.0),
            mass_kg=0.4,
            volume_m3=1e-3,
            source_element=0,
            young_modulus_pa=1e7,
            poisson_ratio=0.3,
            yield_stress_pa=1e6,
            damage=1.0,
        ),
        MPMParticle(
            position=(-0.17, -0.09, 0.11),
            velocity=(0.09, -0.17, 0.02),
            mass_kg=0.6,
            volume_m3=1e-3,
            source_element=1,
            young_modulus_pa=1e7,
            poisson_ratio=0.3,
            yield_stress_pa=1e6,
            damage=1.0,
        ),
        MPMParticle(
            position=(0.04, -0.15, -0.18),
            velocity=(0.15, 0.04, -0.01),
            mass_kg=0.3,
            volume_m3=1e-3,
            source_element=2,
            young_modulus_pa=1e7,
            poisson_ratio=0.3,
            yield_stress_pa=1e6,
            damage=1.0,
        ),
    ]
    state = HybridFEMMPMState(
        positions=[],
        velocities=[],
        masses_kg=[],
        elements=[],
        particles=particles,
        mpm_cell_size_m=0.25,
        pic_fraction=0.2,
    )
    audit = advance_fem(state, 0.01)
    assert math.sqrt(sum(value * value for value in audit.momentum_error)) < 1e-12
    assert math.sqrt(sum(value * value for value in audit.angular_momentum_error)) < 1e-12
    assert audit.angular_momentum_projection_nms > 0.0


def test_vectorized_mpm_matches_scalar_and_conserves_mass_and_momentum():
    particles = []
    particle_data = [
        ((0.13, 0.21, 0.07), (-0.21, 0.13, 0.02), 0.4, 0.0),
        ((-0.17, -0.09, 0.11), (0.09, -0.17, 0.02), 0.6, 0.25),
        ((0.04, -0.15, -0.18), (0.15, 0.04, -0.01), 0.3, 1.0),
    ]
    for index, (position, velocity, mass, damage) in enumerate(particle_data):
        particles.append(
            MPMParticle(
                position=position,
                velocity=velocity,
                mass_kg=mass,
                volume_m3=1e-3,
                source_element=index,
                young_modulus_pa=1e7,
                poisson_ratio=0.3,
                yield_stress_pa=1e6,
                damage=damage,
                deformation_gradient=(
                    (1.01, 0.002, 0.0),
                    (0.0, 0.995, 0.001),
                    (0.0, 0.0, 1.003),
                ),
                stress=(
                    (1000.0, 20.0, 0.0),
                    (20.0, -500.0, 0.0),
                    (0.0, 0.0, 100.0),
                ),
            )
        )
    scalar_state = HybridFEMMPMState(
        positions=[],
        velocities=[],
        masses_kg=[],
        elements=[],
        particles=particles,
        mpm_cell_size_m=0.25,
        pic_fraction=0.2,
    )
    vectorized_state = copy.deepcopy(scalar_state)
    before_mass = vectorized_state.total_mass_kg
    before_momentum = vectorized_state.total_momentum()

    with patch.object(fem_mpm_module, "np", None):
        for _step in range(10):
            advance_mpm(scalar_state, 1e-4)
    for _step in range(10):
        advance_mpm(vectorized_state, 1e-4)

    for scalar_particle, vectorized_particle in zip(
        scalar_state.particles, vectorized_state.particles
    ):
        assert scalar_particle.position == vectorized_particle.position
        assert scalar_particle.velocity == vectorized_particle.velocity
        assert (
            scalar_particle.deformation_gradient
            == vectorized_particle.deformation_gradient
        )
        assert scalar_particle.stress == vectorized_particle.stress
    assert math.isclose(
        vectorized_state.total_mass_kg,
        before_mass,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    _assert_vectors_close(
        [vectorized_state.total_momentum()],
        [before_momentum],
        rel_tol=0.0,
        abs_tol=1e-14,
    )
