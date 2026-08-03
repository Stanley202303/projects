import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cfd_motion.math_utils import v_cross, v_dot, v_sub, v_unit
from cfd_motion.models import AeroComponent, MaterialProperties, MotionFreedom
from cfd_motion.motion import (
    advance_collision_damage_state,
    aabb_gap_along_axis,
    apply_collision_convergence_step,
    apply_nearby_collision_effects,
    arrange_collision_convergence_initial_gap,
    closest_point_on_triangle,
    collision_convergence_approach_axis,
    component_has_decoded_assembly_mate,
    component_center_from_bounds,
    configure_collision_convergence_components,
    component_bounds,
    component_contact_compliance,
    contact_restitution_coefficient,
    contact_inverse_mass,
    deform_component_at_contact,
    enforce_environment_contact_constraints,
    build_eulerian_contact_grid,
    fracture_thin_shell,
    initial_same_source_overlap_pairs,
    inferred_deformation_thickness,
    local_contact_geometry,
    move_component_rigidly,
    register_collision_dent,
    register_collision_hole,
    resolve_part_collisions,
    swept_relative_component_contact,
    swept_mesh_contact,
    thin_shell_impact_response,
    triangle_area_centroid_normal,
    triangle_max_edge_length,
    update_component_deformation,
    update_component_motion,
    write_collision_convergence_log_header,
)
from cfd_motion.runner import (
    _split_unmated_component_bodies,
    collision_convergence_should_stop,
    refine_collision_mesh_for_deformation,
    refine_thin_impact_target,
)
from cfd_motion.visualization import (
    connected_surface_body_ids,
    visualization_components_with_fragments,
)
from cfd_motion.structural import (
    ExplicitShellState,
    HybridShellCollisionState,
    _add_membrane_element_forces,
    build_explicit_shell_state,
    shell_fragment_triangles,
    shell_fragment_velocity,
    update_shell_perforation,
)


def box_component(name: str, xmin: float, xmax: float) -> AeroComponent:
    return rectangular_component(name, xmin, xmax, -0.5, 0.5, -0.5, 0.5)


def rectangular_component(
    name: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> AeroComponent:
    vertices = (
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    )
    faces = (
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
    )
    triangles = [
        ((0.0, 0.0, 0.0), vertices[a], vertices[b], vertices[c])
        for a, b, c in faces
    ]
    return AeroComponent(
        name=name,
        patch=name,
        triangles=triangles,
        cofr=((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5),
        lref=max(xmax - xmin, ymax - ymin, zmax - zmin),
        aref=max(
            (xmax - xmin) * (ymax - ymin),
            (xmax - xmin) * (zmax - zmin),
            (ymax - ymin) * (zmax - zmin),
        ),
        freedom=MotionFreedom(
            translate_axes=[(1.0, 0.0, 0.0)],
            rotate_axes=[],
        ),
        mass=1.0,
    )


def radial_distance_from_axis(point, origin, axis) -> float:
    delta = tuple(point[index] - origin[index] for index in range(3))
    axial = sum(delta[index] * axis[index] for index in range(3))
    radial = tuple(delta[index] - axial * axis[index] for index in range(3))
    return math.sqrt(sum(value * value for value in radial))


def shell_core_state(component: AeroComponent) -> ExplicitShellState:
    state = component.collision_structural_state
    if isinstance(state, HybridShellCollisionState):
        return state.shell_state
    assert isinstance(state, ExplicitShellState)
    return state


def max_attached_shell_position_x(component: AeroComponent) -> float:
    state = shell_core_state(component)
    emitted_nodes = {
        node
        for triangle_index in state.emitted_triangles
        for node in state.triangle_nodes[triangle_index]
    }
    return max(
        state.positions[node_index][0]
        for node_index in range(len(state.positions))
        if node_index not in emitted_nodes
    )


def attached_shell_transverse_span(component: AeroComponent) -> float:
    state = shell_core_state(component)
    emitted_nodes = {
        node
        for triangle_index in state.emitted_triangles
        for node in state.triangle_nodes[triangle_index]
    }
    transverse_values = [
        state.positions[node_index][1]
        for node_index in range(len(state.positions))
        if node_index not in emitted_nodes
    ]
    return max(transverse_values) - min(transverse_values)


def attached_shell_position_y_range(component: AeroComponent) -> tuple[float, float]:
    state = shell_core_state(component)
    emitted_nodes = {
        node
        for triangle_index in state.emitted_triangles
        for node in state.triangle_nodes[triangle_index]
    }
    values = [
        state.positions[node_index][1]
        for node_index in range(len(state.positions))
        if node_index not in emitted_nodes
    ]
    return min(values), max(values)


def membrane_force_work_for_in_plane_displacement(
    state: ExplicitShellState,
    node_index: int,
    displacement: tuple[float, float, float],
) -> float:
    """Return internal-force work for a small prescribed node displacement."""
    state.positions[node_index] = tuple(
        state.positions[node_index][coordinate] + displacement[coordinate]
        for coordinate in range(3)
    )
    forces = [(0.0, 0.0, 0.0) for _position in state.positions]
    _add_membrane_element_forces(state, forces)
    return sum(
        forces[node_index][coordinate] * displacement[coordinate]
        for coordinate in range(3)
    )


def sphere_component(
    name: str,
    center_x: float,
    radius: float,
    latitude_count: int = 8,
    longitude_count: int = 16,
) -> AeroComponent:
    center = (center_x, 0.0, 0.0)
    vertices = [
        (center_x + radius, 0.0, 0.0),
        (center_x - radius, 0.0, 0.0),
    ]
    for latitude in range(1, latitude_count):
        polar = math.pi * latitude / latitude_count
        for longitude in range(longitude_count):
            azimuth = 2.0 * math.pi * longitude / longitude_count
            vertices.append(
                (
                    center_x + radius * math.cos(polar),
                    radius * math.sin(polar) * math.cos(azimuth),
                    radius * math.sin(polar) * math.sin(azimuth),
                )
            )

    def ring_vertex(latitude: int, longitude: int) -> int:
        return 2 + (latitude - 1) * longitude_count + longitude % longitude_count

    faces = []
    for longitude in range(longitude_count):
        faces.append(
            (
                0,
                ring_vertex(1, longitude + 1),
                ring_vertex(1, longitude),
            )
        )
        faces.append(
            (
                1,
                ring_vertex(latitude_count - 1, longitude),
                ring_vertex(latitude_count - 1, longitude + 1),
            )
        )
    for latitude in range(1, latitude_count - 1):
        for longitude in range(longitude_count):
            a = ring_vertex(latitude, longitude)
            b = ring_vertex(latitude, longitude + 1)
            c = ring_vertex(latitude + 1, longitude)
            d = ring_vertex(latitude + 1, longitude + 1)
            faces.extend(((a, b, d), (a, d, c)))

    triangles = [
        ((0.0, 0.0, 0.0), vertices[a], vertices[b], vertices[c])
        for a, b, c in faces
    ]
    return AeroComponent(
        name=name,
        patch=name,
        triangles=triangles,
        cofr=center,
        lref=2.0 * radius,
        aref=math.pi * radius * radius,
        freedom=MotionFreedom(
            translate_axes=[(1.0, 0.0, 0.0)],
            rotate_axes=[],
        ),
        mass=1.0,
        material=MaterialProperties(
            material_name="generic steel",
            density_kg_m3=7850.0,
            young_modulus_pa=2.0e11,
            poisson_ratio=0.30,
        ),
    )


class CollisionConvergenceTest(TestCase):
    def test_internal_collision_does_not_stop_prescribed_target_approach(self) -> None:
        moving = rectangular_component(
            "part1_main", -0.2, -0.1, -0.2, -0.05, -0.05, 0.05
        )
        sibling = rectangular_component(
            "part1_sibling", -0.195, -0.095, 0.05, 0.2, -0.05, 0.05
        )
        target = rectangular_component(
            "part2", 0.5, 0.6, -0.5, 0.5, -0.5, 0.5
        )
        moving.collision_source_index = 1
        sibling.collision_source_index = 1
        target.collision_source_index = 2
        components = [moving, sibling, target]

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 30.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "x"),
            patch("cfd_motion.motion.COLLISION_SWEEP_CLAMPING", True),
            TemporaryDirectory() as tmpdir,
        ):
            pair = configure_collision_convergence_components(components)
            self.assertIsNotNone(pair)
            assert pair is not None
            self.assertIs(pair[0], moving)
            arrange_collision_convergence_initial_gap(pair, components)
            log_path = Path(tmpdir) / "convergence.txt"
            write_collision_convergence_log_header(log_path, pair)

            _move, _target_move, first_contact = apply_collision_convergence_step(
                pair,
                0,
                0.0012,
                log_path,
                (1.0, 0.0, 0.0),
                components,
            )
            self.assertIsNone(first_contact)
            self.assertFalse(
                collision_convergence_should_stop(
                    first_contact,
                    ["continuous_internal_contact"],
                )
            )

            _move, _target_move, target_contact = apply_collision_convergence_step(
                pair,
                1,
                0.0012,
                log_path,
                (1.0, 0.0, 0.0),
                components,
            )
            self.assertIsNotNone(target_contact)
            assert target_contact is not None
            self.assertIs(target_contact.moving, sibling)
            self.assertIs(target_contact.stationary, target)
            self.assertTrue(
                collision_convergence_should_stop(
                    target_contact,
                    ["prescribed_target_contact"],
                )
            )

    def test_static_aabb_overlap_has_no_collision_energy_or_deformation(self) -> None:
        outer = box_component("outer", 0.0, 1.0)
        nested = box_component("nested", 0.25, 0.75)
        outer_triangles = list(outer.triangles)
        nested_triangles = list(nested.triangles)

        with TemporaryDirectory() as tmpdir:
            lines = resolve_part_collisions(
                [outer, nested],
                0,
                Path(tmpdir) / "collisions.txt",
            )

        self.assertEqual(lines, [])
        self.assertEqual(outer.triangles, outer_triangles)
        self.assertEqual(nested.triangles, nested_triangles)
        self.assertEqual(outer.deformation_max_m, 0.0)
        self.assertEqual(nested.deformation_max_m, 0.0)
        self.assertEqual(outer.collision_damage, [])
        self.assertEqual(nested.collision_damage, [])

    def test_static_surface_penetration_is_projected_without_damage(self) -> None:
        free_body = box_component("free_body", 0.8, 1.8)
        wall = box_component("wall", 0.0, 1.0)
        wall.is_assembly_anchor = True

        with TemporaryDirectory() as tmpdir:
            lines = resolve_part_collisions(
                [free_body, wall],
                0,
                Path(tmpdir) / "collisions.txt",
            )

        self.assertTrue(
            any("nonpenetration_constraint" in line for line in lines)
        )
        self.assertGreaterEqual(
            component_bounds(free_body.triangles)[0],
            component_bounds(wall.triangles)[1] - 1e-9,
        )
        self.assertEqual(free_body.deformation_max_m, 0.0)
        self.assertEqual(wall.deformation_max_m, 0.0)
        self.assertEqual(free_body.collision_damage, [])
        self.assertEqual(wall.collision_damage, [])

    def test_surrounding_wall_removes_inward_speed_and_preserves_sliding(self) -> None:
        free_body = box_component("sliding_body", 0.8, 1.8)
        wall = box_component("sliding_wall", 0.0, 1.0)
        free_body.freedom.translate_axes = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        free_body.linear_velocity = (-2.0, 3.0, 0.0)
        wall.is_assembly_anchor = True

        lines = enforce_environment_contact_constraints(
            [free_body, wall],
            0,
        )

        self.assertTrue(lines)
        self.assertGreaterEqual(
            component_bounds(free_body.triangles)[0],
            component_bounds(wall.triangles)[1] - 1e-9,
        )
        self.assertAlmostEqual(free_body.linear_velocity[0], 0.0)
        self.assertAlmostEqual(free_body.linear_velocity[1], 3.0)
        self.assertEqual(free_body.deformation_max_m, 0.0)
        self.assertEqual(wall.deformation_max_m, 0.0)

    def test_initial_same_source_fit_is_stress_free_until_separation(self) -> None:
        first = box_component("part1_a", 0.0, 1.0)
        second = box_component("part1_b", 0.5, 1.5)
        first.collision_source_index = 1
        second.collision_source_index = 1
        first.linear_velocity = (1.0, 0.0, 0.0)
        components = [first, second]
        initial_pairs = initial_same_source_overlap_pairs(components)
        self.assertEqual(len(initial_pairs), 1)

        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "collisions.txt"
            self.assertEqual(
                resolve_part_collisions(
                    components,
                    0,
                    log_path,
                    initial_overlap_pairs=initial_pairs,
                ),
                [],
            )
            self.assertEqual(first.deformation_max_m, 0.0)
            self.assertEqual(second.deformation_max_m, 0.0)

            move_component_rigidly(
                second,
                (2.0, 0.0, 0.0),
                None,
                0.0,
                second.cofr,
            )
            resolve_part_collisions(
                components,
                1,
                log_path,
                initial_overlap_pairs=initial_pairs,
            )
            self.assertEqual(initial_pairs, {})

            move_component_rigidly(
                second,
                (-2.0, 0.0, 0.0),
                None,
                0.0,
                second.cofr,
            )
            lines = resolve_part_collisions(
                components,
                2,
                log_path,
                initial_overlap_pairs=initial_pairs,
            )

        self.assertTrue(lines)
        self.assertGreater(
            max(first.deformation_max_m, second.deformation_max_m),
            0.0,
        )

    def test_fast_unmated_source_bodies_cannot_tunnel_through_each_other(self) -> None:
        moving = box_component("part1_rear", 2.0, 3.0)
        struck = box_component("part1_front", 0.0, 1.0)
        moving.collision_source_index = 1
        struck.collision_source_index = 1
        moving.collision_family = "part1_rear"
        struck.collision_family = "part1_front"
        moving.linear_velocity = (300.0, 0.0, 0.0)
        initial_momentum = moving.mass * moving.linear_velocity[0]

        with (
            patch("cfd_motion.motion.MOTION_DT", 0.01),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [moving, struck],
                1,
                Path(tmpdir) / "collisions.txt",
            )

        moving_bounds = component_bounds(moving.triangles)
        struck_bounds = component_bounds(struck.triangles)
        self.assertTrue(lines)
        self.assertTrue(
            any("continuous_internal_contact" in line for line in lines)
        )
        self.assertLessEqual(moving_bounds[1], struck_bounds[0] + 1e-9)
        self.assertGreater(moving.deformation_max_m, 0.0)
        self.assertGreater(struck.deformation_max_m, 0.0)
        self.assertNotEqual(moving.linear_velocity, (300.0, 0.0, 0.0))
        self.assertAlmostEqual(
            moving.mass * moving.linear_velocity[0]
            + struck.mass * struck.linear_velocity[0],
            initial_momentum,
        )

    def test_unmated_source_bodies_can_separate_after_touching(self) -> None:
        separating = box_component("part1_separating", -2.0, -1.0)
        neighbour = box_component("part1_neighbour", 0.0, 1.0)
        separating.collision_source_index = 1
        neighbour.collision_source_index = 1
        separating.linear_velocity = (-100.0, 0.0, 0.0)
        original_bounds = component_bounds(separating.triangles)

        with (
            patch("cfd_motion.motion.MOTION_DT", 0.01),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [separating, neighbour],
                1,
                Path(tmpdir) / "collisions.txt",
            )

        self.assertEqual(lines, [])
        self.assertEqual(component_bounds(separating.triangles), original_bounds)
        self.assertEqual(separating.linear_velocity, (-100.0, 0.0, 0.0))
        self.assertEqual(separating.deformation_max_m, 0.0)
        self.assertEqual(neighbour.deformation_max_m, 0.0)

    def test_trailing_impactor_body_cannot_tunnel_through_target(self) -> None:
        moving = box_component("trailing_part1_body", 2.0, 3.0)
        target = box_component("part2_target", 0.0, 1.0)
        moving.collision_source_index = 1
        target.collision_source_index = 2
        moving.linear_velocity = (300.0, 0.0, 0.0)
        target.is_assembly_anchor = True

        with (
            patch("cfd_motion.motion.MOTION_DT", 0.01),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [moving, target],
                2,
                Path(tmpdir) / "collisions.txt",
            )

        self.assertTrue(lines)
        self.assertLessEqual(
            component_bounds(moving.triangles)[1],
            component_bounds(target.triangles)[0] + 1e-9,
        )
        self.assertGreater(moving.deformation_max_m, 0.0)
        self.assertGreater(target.deformation_max_m, 0.0)

    def test_prescribed_pair_still_has_continuous_collision_fallback(self) -> None:
        moving = box_component("prescribed_moving", 2.0, 3.0)
        target = box_component("prescribed_target", 0.0, 1.0)
        moving.collision_source_index = 1
        target.collision_source_index = 2
        moving.linear_velocity = (300.0, 0.0, 0.0)
        target.is_assembly_anchor = True

        with (
            patch("cfd_motion.motion.MOTION_DT", 0.01),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [moving, target],
                2,
                Path(tmpdir) / "collisions.txt",
                prescribed_pair=(moving, target),
            )

        self.assertTrue(lines)
        self.assertLessEqual(
            component_bounds(moving.triangles)[1],
            component_bounds(target.triangles)[0] + 1e-9,
        )

    def test_detached_fragment_cannot_tunnel_through_its_parent_family(self) -> None:
        fragment = box_component("target_fragment", 2.0, 3.0)
        parent = box_component("target_parent", 0.0, 1.0)
        fragment.collision_family = "target"
        parent.collision_family = "target"
        fragment.freedom.mate_type = "COLLISION_FRAGMENT"
        fragment.linear_velocity = (300.0, 0.0, 0.0)
        parent.is_assembly_anchor = True

        with (
            patch("cfd_motion.motion.MOTION_DT", 0.01),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [fragment, parent],
                3,
                Path(tmpdir) / "collisions.txt",
            )

        self.assertTrue(lines)
        self.assertLessEqual(
            component_bounds(fragment.triangles)[1],
            component_bounds(parent.triangles)[0] + 1e-9,
        )

    def test_uninvolved_third_body_cannot_tunnel_through_neighbour(self) -> None:
        first = box_component("primary_impactor", 5.0, 6.0)
        second = box_component("primary_target", 3.0, 4.0)
        third = box_component("independent_body", 2.0, 3.0)
        barrier = box_component("independent_barrier", 0.0, 1.0)
        move_component_rigidly(third, (0.0, 3.0, 0.0), None, 0.0, third.cofr)
        move_component_rigidly(barrier, (0.0, 3.0, 0.0), None, 0.0, barrier.cofr)
        first.linear_velocity = (300.0, 0.0, 0.0)
        third.linear_velocity = (300.0, 0.0, 0.0)
        second.is_assembly_anchor = True
        barrier.is_assembly_anchor = True

        with (
            patch("cfd_motion.motion.MOTION_DT", 0.01),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [first, second, third, barrier],
                4,
                Path(tmpdir) / "collisions.txt",
            )

        contacted_pairs = {
            frozenset((columns[2], columns[3]))
            for line in lines
            if len(columns := line.split("\t")) >= 4
        }
        self.assertIn(
            frozenset(("primary_impactor", "primary_target")),
            contacted_pairs,
        )
        self.assertIn(
            frozenset(("independent_body", "independent_barrier")),
            contacted_pairs,
        )
        self.assertLessEqual(
            component_bounds(third.triangles)[1],
            component_bounds(barrier.triangles)[0] + 1e-9,
        )

    def test_rigid_group_does_not_disable_other_pair_collisions(self) -> None:
        rigid_a = box_component("mated_a", 8.0, 9.0)
        rigid_b = box_component("mated_b", 9.0, 10.0)
        rigid_a.rigid_body_group = "fastened_group"
        rigid_b.rigid_body_group = "fastened_group"
        moving = box_component("unmated_moving", 2.0, 3.0)
        target = box_component("unmated_target", 0.0, 1.0)
        moving.linear_velocity = (300.0, 0.0, 0.0)
        target.is_assembly_anchor = True

        with (
            patch("cfd_motion.motion.MOTION_DT", 0.01),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [rigid_a, rigid_b, moving, target],
                5,
                Path(tmpdir) / "collisions.txt",
            )

        contacted_pairs = {
            frozenset((columns[2], columns[3]))
            for line in lines
            if len(columns := line.split("\t")) >= 4
        }
        self.assertNotIn(frozenset(("mated_a", "mated_b")), contacted_pairs)
        self.assertIn(frozenset(("unmated_moving", "unmated_target")), contacted_pairs)
        self.assertLessEqual(
            component_bounds(moving.triangles)[1],
            component_bounds(target.triangles)[0] + 1e-9,
        )

    def test_rotating_body_cannot_sweep_through_stationary_body(self) -> None:
        rotor = rectangular_component(
            "rotor",
            -1.0,
            1.0,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        barrier = rectangular_component(
            "rotation_barrier",
            -0.2,
            0.2,
            0.6,
            1.0,
            -0.2,
            0.2,
        )
        barrier.is_assembly_anchor = True
        dt = 0.01
        rotation = 2.0 * math.pi
        move_component_rigidly(
            rotor,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            rotation,
            (0.0, 0.0, 0.0),
        )
        rotor.angular_velocity = (0.0, 0.0, rotation / dt)

        with (
            patch("cfd_motion.motion.MOTION_DT", dt),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [rotor, barrier],
                6,
                Path(tmpdir) / "collisions.txt",
            )

        self.assertTrue(lines)
        self.assertTrue(any("continuous_internal_contact" in line for line in lines))
        self.assertLess(
            component_bounds(rotor.triangles)[3],
            component_bounds(barrier.triangles)[3],
        )

    def test_collision_deformation_keeps_shared_vertices_connected(self) -> None:
        component = rectangular_component(
            "coherent_mesh",
            0.0,
            0.005,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        deform_component_at_contact(
            component,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.003,
            0.1,
        )

        body_ids = connected_surface_body_ids(component)

        self.assertEqual(max(body_ids, default=-1) + 1, 1)

    def test_inferred_thickness_is_reference_geometry_property(self) -> None:
        component = rectangular_component(
            "constant_thickness",
            0.0,
            0.005,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        initial_thickness = inferred_deformation_thickness(component)
        deform_component_at_contact(
            component,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.008,
            0.1,
        )

        self.assertAlmostEqual(
            inferred_deformation_thickness(component),
            initial_thickness,
        )

    def test_nearby_shock_deformation_respects_material_stiffness(self) -> None:
        tungsten = rectangular_component(
            "nearby_tungsten",
            0.0,
            0.01,
            -0.01,
            0.01,
            -0.01,
            0.01,
        )
        polymer = rectangular_component(
            "nearby_polymer",
            0.0,
            0.01,
            -0.01,
            0.01,
            -0.01,
            0.01,
        )
        tungsten.material = MaterialProperties(
            material_name="Tungsten",
            density_kg_m3=19600.0,
            young_modulus_pa=4.0e11,
            poisson_ratio=0.28,
            yield_strength_pa=7.5e8,
        )
        polymer.material = MaterialProperties(
            material_name="ABS",
            density_kg_m3=1052.0,
            young_modulus_pa=2.31e9,
            poisson_ratio=0.364,
            yield_strength_pa=4.48e7,
        )

        apply_nearby_collision_effects(
            [tungsten, polymer],
            (),
            (-0.001, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.005,
            10.0,
            0,
            "test_impact",
        )

        self.assertGreater(polymer.deformation_max_m, 0.0)
        self.assertLess(
            tungsten.deformation_max_m,
            0.2 * polymer.deformation_max_m,
        )

    def test_later_unprescribed_impact_can_perforate_thin_target(self) -> None:
        projectile = sphere_component("trailing_tungsten", 0.05, 0.005)
        projectile.mass = 0.33
        projectile.material = MaterialProperties(
            material_name="Tungsten",
            density_kg_m3=19600.0,
            young_modulus_pa=4.0e11,
            poisson_ratio=0.28,
            yield_strength_pa=7.5e8,
        )
        projectile.linear_velocity = (100.0, 0.0, 0.0)
        target = rectangular_component(
            "thin_abs_target",
            0.0,
            0.005,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        target.material = MaterialProperties(
            material_name="ABS",
            density_kg_m3=1052.0,
            young_modulus_pa=2.31e9,
            poisson_ratio=0.364,
            thickness_m=0.005,
            yield_strength_pa=4.48e7,
            failure_strain=0.20,
        )
        target.is_assembly_anchor = True

        with (
            patch("cfd_motion.motion.MOTION_DT", 0.001),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [projectile, target],
                7,
                Path(tmpdir) / "collisions.txt",
            )

        self.assertTrue(any(line.split("\t")[20] == "1" for line in lines))
        self.assertTrue(target.collision_damage)
        self.assertGreater(target.collision_damage[0].current_hole_radius_m, 0.0)

    def test_nearby_shock_impulse_has_equal_opposite_reaction(self) -> None:
        donor = box_component("shock_donor", -0.02, -0.01)
        neighbour = box_component("shock_neighbour", 0.0, 0.01)
        initial_momentum = tuple(
            donor.mass * donor.linear_velocity[axis]
            + neighbour.mass * neighbour.linear_velocity[axis]
            for axis in range(3)
        )

        apply_nearby_collision_effects(
            [donor, neighbour],
            (donor,),
            (-0.005, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.005,
            5.0,
            0,
            "momentum_test",
        )

        final_momentum = tuple(
            donor.mass * donor.linear_velocity[axis]
            + neighbour.mass * neighbour.linear_velocity[axis]
            for axis in range(3)
        )
        for initial, final in zip(initial_momentum, final_momentum):
            self.assertAlmostEqual(final, initial, places=12)

    def test_post_perforation_path_does_not_get_a_false_surface_contact(self) -> None:
        target = rectangular_component(
            "perforated_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        target.material = MaterialProperties(
            material_name="ABS",
            density_kg_m3=1052.0,
            young_modulus_pa=2.31e9,
            poisson_ratio=0.364,
            thickness_m=0.0001,
            yield_strength_pa=4.48e7,
            failure_strain=0.20,
        )
        register_collision_hole(
            target,
            (0.00005, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.003,
            0.0025,
            0,
            "plastic_membrane_perforation",
            2.0,
        )
        projectile = sphere_component("passed_projectile", 0.05, 0.0025)
        projectile.linear_velocity = (100.0, 0.0, 0.0)

        contact = swept_relative_component_contact(projectile, target, 0.001)

        self.assertIsNone(contact)

    def test_initial_fit_expires_when_internal_body_reaches_real_surface(self) -> None:
        outer = rectangular_component(
            "part1_outer",
            0.0,
            2.0,
            -0.5,
            0.5,
            -0.5,
            0.5,
        )
        inner = rectangular_component(
            "part1_inner",
            0.5,
            1.0,
            -0.1,
            0.1,
            -0.1,
            0.1,
        )
        outer.collision_source_index = 1
        inner.collision_source_index = 1
        outer.collision_family = "part1_outer"
        inner.collision_family = "part1_inner"
        outer.is_assembly_anchor = True
        initial_pairs = initial_same_source_overlap_pairs([outer, inner])
        self.assertEqual(len(initial_pairs), 1)

        move_component_rigidly(
            inner,
            (1.2, 0.0, 0.0),
            None,
            0.0,
            inner.cofr,
        )
        inner.linear_velocity = (120.0, 0.0, 0.0)
        with (
            patch("cfd_motion.motion.MOTION_DT", 0.01),
            TemporaryDirectory() as tmpdir,
        ):
            lines = resolve_part_collisions(
                [outer, inner],
                1,
                Path(tmpdir) / "collisions.txt",
                initial_overlap_pairs=initial_pairs,
            )

        self.assertTrue(lines)
        self.assertEqual(initial_pairs, {})
        self.assertLessEqual(
            component_bounds(inner.triangles)[1],
            component_bounds(outer.triangles)[1] + 1e-9,
        )
        self.assertGreater(
            max(outer.deformation_max_m, inner.deformation_max_m),
            0.0,
        )

    def test_bom_material_pair_sets_default_restitution(self) -> None:
        steel = box_component("steel", 0.0, 0.1)
        steel.material.material_name = "stainless steel"
        rubber = box_component("rubber", 0.2, 0.3)
        rubber.material.material_name = "rubber"

        restitution = contact_restitution_coefficient(steel, rubber, -1.0)

        self.assertAlmostEqual(restitution, math.sqrt(0.24 * 0.72))
        self.assertEqual(contact_restitution_coefficient(steel, rubber, 0.1), 0.1)

    def test_bom_material_properties_drive_contact_and_perforation_response(self) -> None:
        impactor = box_component("impactor", -0.03, 0.0)
        impactor.mass = 0.25
        target = rectangular_component(
            "bom_sheet",
            0.1,
            0.1001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        # These are the same fields populated by a matched BOM row.
        target.material = MaterialProperties(
            material_name="BOM polymer",
            density_kg_m3=1200.0,
            young_modulus_pa=2.0e9,
            poisson_ratio=0.32,
            thickness_m=0.001,
            yield_strength_pa=1.0e6,
            failure_strain=0.01,
            source="bom",
            structural_source="bom",
        )

        compliance = component_contact_compliance(target)
        response = thin_shell_impact_response(
            impactor,
            target,
            1000.0,
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
        )
        self.assertIsNotNone(response)
        assert response is not None

        target.material.young_modulus_pa = 4.0e9
        stiffer_compliance = component_contact_compliance(target)
        stiffer_response = thin_shell_impact_response(
            impactor,
            target,
            1000.0,
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
        )
        self.assertIsNotNone(stiffer_response)
        assert stiffer_response is not None

        self.assertAlmostEqual(stiffer_compliance, compliance * 0.5)
        self.assertNotEqual(
            stiffer_response.absorbed_energy_j,
            response.absorbed_energy_j,
        )

    def test_swept_contact_builds_a_flat_face_manifold(self) -> None:
        moving = box_component("moving", 0.0, 1.0)
        stationary = box_component("stationary", 1.1, 2.1)

        hit = swept_mesh_contact(moving, stationary, (1.0, 0.0, 0.0), 0.2)

        self.assertIsNotNone(hit)
        assert hit is not None
        distance, point, normal, manifold_points = hit
        self.assertAlmostEqual(distance, 0.1)
        self.assertAlmostEqual(point[0], 1.1)
        self.assertEqual(normal, (-1.0, 0.0, 0.0))
        self.assertGreaterEqual(manifold_points, 4)

    def test_box_box_contact_uses_planar_face_geometry(self) -> None:
        moving = box_component("moving_box", 0.0, 1.0)
        stationary = box_component("stationary_box", 1.1, 2.1)

        geometry = local_contact_geometry(
            moving,
            stationary,
            (1.0, 0.0, 0.0),
            (1.1, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
        )

        self.assertFalse(geometry.hertzian)
        self.assertTrue(math.isinf(geometry.radius_a))
        self.assertTrue(math.isinf(geometry.radius_b))
        self.assertAlmostEqual(geometry.footprint_radius, math.sqrt(1.0 / math.pi))

    def test_two_mesh_spheres_use_curvature_with_bounded_impactor_deformation(self) -> None:
        radius = 0.05
        moving = sphere_component("sphere_one", 0.0, radius)
        stationary = sphere_component("sphere_two", 0.2, radius)

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 5.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "x"),
            TemporaryDirectory() as tmpdir,
        ):
            pair = configure_collision_convergence_components([moving, stationary])
            self.assertIsNotNone(pair)
            assert pair is not None
            convergence_log = Path(tmpdir) / "convergence.txt"
            collision_log = Path(tmpdir) / "collisions.txt"
            write_collision_convergence_log_header(convergence_log, pair)

            _move, _target_move, contact = apply_collision_convergence_step(
                pair,
                0,
                0.03,
                convergence_log,
                (1.0, 0.0, 0.0),
            )

            self.assertIsNotNone(contact)
            assert contact is not None
            self.assertIsNotNone(contact.contact_geometry)
            assert contact.contact_geometry is not None
            geometry = contact.contact_geometry
            self.assertTrue(geometry.hertzian)
            self.assertAlmostEqual(geometry.radius_a, radius, delta=0.01)
            self.assertAlmostEqual(geometry.radius_b, radius, delta=0.01)
            self.assertAlmostEqual(geometry.effective_radius, radius / 2.0, delta=0.005)

            contacts = resolve_part_collisions(
                [moving, stationary],
                0,
                collision_log,
                contact,
                pair,
            )

            self.assertTrue(contacts)
            self.assertGreater(moving.deformation_max_m, 0.0)
            self.assertLessEqual(moving.deformation_max_m, 0.01)
            self.assertGreater(stationary.deformation_max_m, 0.0)
            self.assertEqual(stationary.linear_velocity, (0.0, 0.0, 0.0))

            impact_triangles = list(moving.triangles)
            impact_deformation = moving.deformation_max_m
            with patch(
                "cfd_motion.motion.triangle_pressure_force_for_deformation",
                return_value=(1.0e8, 0.0, 0.0),
            ):
                reported_deformation, *_rest = update_component_deformation(
                    moving,
                    0.02,
                )
            self.assertEqual(moving.triangles, impact_triangles)
            self.assertEqual(reported_deformation, impact_deformation)

    def test_collision_impactor_ignores_panel_pressure_deformation(self) -> None:
        moving = box_component("moving", 0.0, 1.0)
        stationary = box_component("stationary", 1.1, 2.1)
        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 5.0),
            patch(
                "cfd_motion.motion.triangle_pressure_force_for_deformation",
                return_value=(1.0e8, 0.0, 0.0),
            ),
        ):
            pair = configure_collision_convergence_components([moving, stationary])
            self.assertIsNotNone(pair)
            original_triangles = list(moving.triangles)
            max_deformation, *_rest = update_component_deformation(moving, 0.02)

        self.assertEqual(max_deformation, 0.0)
        self.assertEqual(moving.deformation_max_m, 0.0)
        self.assertEqual(moving.triangles, original_triangles)

    def test_collision_dent_recovers_only_to_permanent_depth(self) -> None:
        component = box_component("soft_body", 0.0, 1.0)
        component.material = MaterialProperties(
            material_name="soft polymer",
            density_kg_m3=1000.0,
            young_modulus_pa=1.0e7,
            poisson_ratio=0.35,
            yield_strength_pa=1.0e4,
            failure_strain=0.10,
        )
        original_triangles = list(component.triangles)
        contact_point = (1.0, 0.0, 0.0)
        inward = (-1.0, 0.0, 0.0)
        applied = deform_component_at_contact(
            component,
            contact_point,
            inward,
            0.01,
            0.6,
        )
        damage = register_collision_dent(
            component,
            contact_point,
            inward,
            applied,
            0.6,
            0,
            "plastic_contact",
        )

        self.assertIsNotNone(damage)
        assert damage is not None
        self.assertGreater(damage.permanent_depth_m, 0.0)
        self.assertLess(damage.permanent_depth_m, damage.current_depth_m)

        advance_collision_damage_state(
            component,
            damage,
            20.0 * damage.response_time_s,
        )

        self.assertAlmostEqual(
            damage.current_depth_m,
            damage.permanent_depth_m,
            delta=1e-7,
        )
        self.assertNotEqual(
            component.deformation_reference_triangles,
            original_triangles,
        )
        with patch(
            "cfd_motion.motion.triangle_pressure_force_for_deformation",
            return_value=(0.0, 0.0, 0.0),
        ):
            update_component_deformation(component, 0.02)
        self.assertNotEqual(component.triangles, original_triangles)

    def test_perforation_hole_grows_over_multiple_time_intervals(self) -> None:
        target = rectangular_component(
            "thin_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        target.material = MaterialProperties(
            material_name="ABS",
            density_kg_m3=1040.0,
            young_modulus_pa=2.0e9,
            poisson_ratio=0.35,
            thickness_m=0.0001,
        )
        self.assertGreater(refine_thin_impact_target(target), 0)
        initial_triangle_count = len(target.triangles)
        initial_triangles = list(target.triangles)
        initial_mass = target.mass
        damage = register_collision_hole(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.015,
            0.004,
            0,
            "plastic_membrane_perforation",
            1.0,
        )
        shell_state = shell_core_state(target)
        self.assertTrue(shell_state.membrane_elements)
        shell_state.max_substeps = 16
        initial_radius = damage.current_hole_radius_m
        self.assertGreater(initial_radius, 0.0)
        self.assertLess(initial_radius, damage.target_hole_radius_m)

        advance_collision_damage_state(
            target,
            damage,
            0.25 * damage.response_time_s,
        )
        first_radius = damage.current_hole_radius_m
        self.assertGreater(first_radius, initial_radius)
        self.assertLess(first_radius, damage.target_hole_radius_m)
        first_geometry = list(target.triangles)

        advance_collision_damage_state(
            target,
            damage,
            10.0 * damage.response_time_s,
        )
        self.assertNotEqual(target.triangles, first_geometry)
        shell_state = shell_core_state(target)
        self.assertTrue(shell_state.plug_triangles)
        self.assertGreater(shell_state.mass_scale, 1.0)
        self.assertLessEqual(
            shell_state.max_displacement_m,
            shell_state.displacement_limit_m + 1e-12,
        )
        self.assertGreater(shell_state.displacement_limit_m, 0.01)
        self.assertTrue(
            all(
                math.isfinite(value)
                for triangle in target.triangles
                for point in triangle[1:]
                for value in point
            )
        )
        self.assertAlmostEqual(
            damage.current_hole_radius_m,
            damage.target_hole_radius_m,
            delta=1e-6,
        )
        self.assertGreater(damage.ongoing_contact_energy_j, 0.0)
        self.assertLess(len(target.triangles), initial_triangle_count)
        self.assertTrue(shell_fragment_triangles(shell_state))
        first_fragment_centroid = component_center_from_bounds(
            visualization_components_with_fragments([target])[1]
        )
        first_fragment_velocity = shell_fragment_velocity(shell_state)
        first_fragment_speed = math.sqrt(
            sum(value * value for value in first_fragment_velocity)
        )
        self.assertGreater(
            first_fragment_speed,
            0.0,
        )
        advance_collision_damage_state(
            target,
            damage,
            damage.response_time_s,
        )
        moved_fragment_centroid = component_center_from_bounds(
            visualization_components_with_fragments([target])[1]
        )
        moved_fragment_velocity = shell_fragment_velocity(shell_state)
        moved_fragment_speed = math.sqrt(
            sum(value * value for value in moved_fragment_velocity)
        )
        self.assertGreater(
            abs(moved_fragment_centroid[0] - first_fragment_centroid[0]),
            0.0,
        )
        self.assertAlmostEqual(
            moved_fragment_speed,
            first_fragment_speed,
            delta=max(1e-9, 0.05 * first_fragment_speed),
        )
        driven_deformation = target.deformation_max_m
        remaining_contact_energy = damage.ongoing_contact_energy_j
        advance_collision_damage_state(
            target,
            damage,
            damage.response_time_s,
        )
        self.assertGreaterEqual(target.deformation_max_m, driven_deformation)
        self.assertLess(damage.ongoing_contact_energy_j, remaining_contact_energy)
        self.assertLessEqual(
            shell_state.max_displacement_m,
            shell_state.displacement_limit_m + 1e-12,
        )
        self.assertLessEqual(
            target.deformation_max_m,
            shell_state.displacement_limit_m + 1e-12,
        )
        self.assertGreater(target.deformation_max_m, 0.01)
        persistent_parent_x = max_attached_shell_position_x(target)
        advance_collision_damage_state(
            target,
            damage,
            5.0 * damage.response_time_s,
        )
        self.assertGreaterEqual(
            max_attached_shell_position_x(target),
            0.75 * persistent_parent_x,
        )
        self.assertGreater(
            attached_shell_transverse_span(target),
            0.03,
        )
        min_y, max_y = attached_shell_position_y_range(target)
        self.assertLess(min_y, -0.03)
        self.assertGreater(max_y, 0.03)
        self.assertNotEqual(target.triangles, initial_triangles)
        visual_components = visualization_components_with_fragments([target])
        self.assertEqual(len(visual_components), 2)
        self.assertEqual(visual_components[1].patch, "thin_target_fragment_0")
        self.assertGreater(
            math.sqrt(
                sum(value * value for value in visual_components[1].linear_velocity)
            ),
            0.0,
        )
        self.assertGreater(visual_components[1].linear_velocity[0], 0.0)
        self.assertLess(
            math.hypot(
                visual_components[1].linear_velocity[1],
                visual_components[1].linear_velocity[2],
            ),
            0.25 * abs(visual_components[1].linear_velocity[0]),
        )
        self.assertTrue(visual_components[1].freedom.translate_axes)
        self.assertEqual(
            visual_components[1].triangles,
            shell_fragment_triangles(target.collision_structural_state),
        )
        self.assertAlmostEqual(
            sum(component.mass for component in visual_components),
            initial_mass,
        )
        hybrid_state = target.collision_structural_state
        if isinstance(hybrid_state, HybridShellCollisionState):
            self.assertTrue(hybrid_state.shell_state.render_as_midsurface)
            self.assertLessEqual(
                len(target.triangles),
                len(hybrid_state.shell_state.triangle_nodes)
                - len(hybrid_state.shell_state.emitted_triangles),
            )
            largest_transverse_span = 0.0
            for fragment in hybrid_state.fragment_bodies:
                _xmin, _xmax, ymin, ymax, zmin, zmax = component_bounds(
                    fragment.component.triangles
                )
                largest_transverse_span = max(
                    largest_transverse_span,
                    ymax - ymin,
                    zmax - zmin,
                )
            self.assertLessEqual(largest_transverse_span, 0.04)

    def test_five_mm_projectile_leaves_a_visible_mass_conserving_hole(self) -> None:
        target = rectangular_component(
            "small_hole_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        target.material = MaterialProperties(
            material_name="steel",
            density_kg_m3=7850.0,
            young_modulus_pa=2.0e11,
            poisson_ratio=0.30,
            thickness_m=0.0001,
            yield_strength_pa=2.5e8,
            failure_strain=0.20,
        )
        projectile = sphere_component("five_mm_dart", -0.0025, 0.0025)
        projectile.mass = 0.05
        projectile.material = MaterialProperties(
            material_name="tungsten",
            density_kg_m3=19300.0,
            mass_kg=projectile.mass,
            young_modulus_pa=4.11e11,
            poisson_ratio=0.28,
            yield_strength_pa=7.5e8,
            failure_strain=0.01,
        )
        initial_mass = target.mass
        projectile_radius = 0.0025
        response = thin_shell_impact_response(
            projectile,
            target,
            100.0,
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
        )
        self.assertIsNotNone(response)
        assert response is not None
        self.assertTrue(response.perforated)
        self.assertGreaterEqual(response.hole_radius, 1.05 * projectile_radius)

        damage = register_collision_hole(
            target,
            (0.00005, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            response.hole_radius,
            projectile_radius,
            0,
            "plastic_membrane_perforation",
            2.0,
        )
        shell_state = shell_core_state(target)
        visual_components = visualization_components_with_fragments([target])

        self.assertGreaterEqual(
            damage.current_hole_radius_m,
            projectile_radius,
        )
        self.assertTrue(target.triangles)
        self.assertTrue(shell_state.emitted_triangles)
        self.assertLessEqual(
            max(
                triangle_max_edge_length(
                    (
                        shell_state.triangle_normals[index],
                        *(shell_state.reference_positions[node] for node in nodes),
                    )
                )
                for index, nodes in enumerate(shell_state.triangle_nodes)
            ),
            2.0 * response.hole_radius + 1e-12,
        )
        for triangle in target.triangles:
            closest = closest_point_on_triangle(damage.contact_point, triangle)
            self.assertGreater(
                radial_distance_from_axis(
                    closest,
                    damage.contact_point,
                    damage.inward_direction,
                ),
                projectile_radius,
            )
        self.assertAlmostEqual(
            sum(component.mass for component in visual_components),
            initial_mass,
        )

        shell_state.max_substeps = 1
        advance_collision_damage_state(
            target,
            damage,
            10.0 * damage.response_time_s,
        )
        self.assertTrue(target.triangles)
        for triangle in target.triangles:
            closest = closest_point_on_triangle(damage.contact_point, triangle)
            self.assertGreater(
                radial_distance_from_axis(
                    closest,
                    damage.contact_point,
                    damage.inward_direction,
                ),
                projectile_radius,
            )

    def test_shell_plug_separation_uses_damage_response_interval(self) -> None:
        target = rectangular_component(
            "thin_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        damage = register_collision_hole(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.015,
            0.015,
            0,
            "plastic_membrane_perforation",
            1.0,
        )
        shell_state = shell_core_state(target)
        shell_state.plug_triangles.clear()
        shell_state.emitted_triangles.clear()
        shell_state.velocities = [
            (0.0, 0.0, 0.0) for _position in shell_state.positions
        ]

        radial_growth = 0.002
        response_interval = 0.01
        update_shell_perforation(
            shell_state,
            1.0,
            radial_growth,
            response_interval,
        )

        expected_speed = radial_growth / response_interval
        self.assertTrue(shell_state.plug_triangles)
        self.assertAlmostEqual(
            max(
                math.sqrt(sum(value * value for value in velocity))
                for velocity in shell_state.velocities
            ),
            expected_speed,
        )

    def test_explicit_shell_uses_target_midsurface_not_perimeter_walls(self) -> None:
        target = rectangular_component(
            "thin_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        target.is_assembly_anchor = True
        state = build_explicit_shell_state(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.015,
            2.0e9,
            0.0001,
            0.35,
            4.0e7,
            0.2,
            0.05,
            0.5,
            16,
            0.03,
        )

        reference_x = [point[0] for point in state.reference_positions]
        self.assertTrue(reference_x)
        self.assertAlmostEqual(min(reference_x), 0.00005, delta=1e-8)
        self.assertAlmostEqual(max(reference_x), 0.00005, delta=1e-8)
        self.assertLess(len(state.triangle_nodes), len(target.triangles))
        self.assertTrue(state.fixed_nodes)

        refined_target = rectangular_component(
            "refined_thin_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        refined_target.is_assembly_anchor = True
        self.assertGreater(refine_thin_impact_target(refined_target), 0)
        refined_state = build_explicit_shell_state(
            refined_target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.015,
            2.0e9,
            0.0001,
            0.35,
            4.0e7,
            0.2,
            0.05,
            0.5,
            16,
            0.03,
        )
        self.assertTrue(refined_state.fixed_nodes)
        self.assertLess(len(refined_state.fixed_nodes), len(refined_state.positions))

    def test_membrane_fem_force_resists_in_plane_displacement(self) -> None:
        """A stretched triangle must return energy, never amplify the stretch."""
        target = rectangular_component(
            "membrane_force_plate",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        state = build_explicit_shell_state(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.015,
            2.0e9,
            0.0001,
            0.35,
            4.0e7,
            0.2,
            0.05,
            0.5,
            16,
            0.03,
        )
        element = state.membrane_elements[0]
        displacement = tuple(1e-6 * value for value in element.basis_x)
        work = membrane_force_work_for_in_plane_displacement(
            state,
            element.nodes[0],
            displacement,
        )
        self.assertLess(work, 0.0)

    def test_shell_uses_one_midsurface_when_plate_faces_have_different_tessellations(self) -> None:
        thickness = 0.0001
        low = (
            (0.0, -0.05, -0.05),
            (0.0, 0.05, -0.05),
            (0.0, 0.05, 0.05),
            (0.0, -0.05, 0.05),
        )
        high = tuple((thickness, point[1], point[2]) for point in low)
        high_center = (thickness, 0.0, 0.0)
        triangles = [
            ((0.0, 0.0, 0.0), low[0], low[1], low[2]),
            ((0.0, 0.0, 0.0), low[0], low[2], low[3]),
            ((0.0, 0.0, 0.0), high[0], high[1], high_center),
            ((0.0, 0.0, 0.0), high[1], high[2], high_center),
            ((0.0, 0.0, 0.0), high[2], high[3], high_center),
            ((0.0, 0.0, 0.0), high[3], high[0], high_center),
        ]
        target = AeroComponent(
            name="mismatched_sheet",
            patch="mismatched_sheet",
            triangles=triangles,
            cofr=(0.5 * thickness, 0.0, 0.0),
            lref=0.1,
            aref=0.01,
            material=MaterialProperties(
                material_name="ABS",
                density_kg_m3=1040.0,
                thickness_m=thickness,
            ),
            mass=0.001,
        )

        state = build_explicit_shell_state(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.01,
            2.0e9,
            thickness,
            0.35,
            4.0e7,
            0.2,
            0.05,
            0.5,
            16,
            0.03,
        )

        self.assertEqual(len(state.triangle_nodes), 4)
        self.assertTrue(state.reference_positions)
        for point in state.reference_positions:
            self.assertAlmostEqual(point[0], 0.5 * thickness, delta=1e-10)
        surface_normals = []
        for node_a, node_b, node_c in state.triangle_nodes:
            a, b, c = (
                state.reference_positions[node_a],
                state.reference_positions[node_b],
                state.reference_positions[node_c],
            )
            surface_normals.append(
                v_unit(v_cross(v_sub(b, a), v_sub(c, a)))
            )
        for normal in surface_normals:
            self.assertGreater(v_dot(normal, surface_normals[0]), 0.99)

    def test_fracture_cuts_a_resolved_hole_in_coarse_sheet_mesh(self) -> None:
        target = rectangular_component(
            "coarse_sheet",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        initial_triangle_count = len(target.triangles)

        max_deflection, removed = fracture_thin_shell(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.015,
            rim_displacement_m=0.006,
        )

        self.assertGreater(removed, 0)
        self.assertGreater(max_deflection, 0.0)
        self.assertGreater(len(target.triangles), initial_triangle_count)
        for triangle in target.triangles:
            area, centroid, normal = triangle_area_centroid_normal(triangle)
            if area <= 1e-18 or abs(normal[0]) < 0.5:
                continue
            radial = math.hypot(centroid[1], centroid[2])
            self.assertGreaterEqual(radial, 0.015 - 1e-6)

    def test_wood_fracture_splinters_without_deleting_material(self) -> None:
        target = rectangular_component(
            "wood_sheet",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        target.material = MaterialProperties(
            material_name="birch plywood",
            density_kg_m3=650.0,
            young_modulus_pa=1.0e10,
            poisson_ratio=0.30,
            thickness_m=0.0001,
        )
        initial_triangle_count = len(target.triangles)

        _max_deflection, displaced_fragments = fracture_thin_shell(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.012,
            rim_displacement_m=0.004,
        )

        self.assertGreater(displaced_fragments, 0)
        self.assertGreater(len(target.triangles), initial_triangle_count)

    def test_fracture_keeps_target_reference_and_does_not_refine_fragments_again(self) -> None:
        target = rectangular_component(
            "stable_sheet",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        original_cofr = target.cofr
        fracture_thin_shell(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.012,
            rim_displacement_m=0.004,
        )
        first_count = len(target.triangles)
        fracture_thin_shell(
            target,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.012,
            rim_displacement_m=0.0,
        )

        self.assertEqual(target.cofr, original_cofr)
        self.assertLessEqual(len(target.triangles), first_count * 2)

    def test_off_center_contact_includes_rotational_inverse_mass(self) -> None:
        component = box_component("body", 0.0, 1.0)
        component.freedom = MotionFreedom(
            translate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            rotate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        )

        inverse_mass = contact_inverse_mass(
            component,
            (1.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
        )

        self.assertGreater(inverse_mass, 1.0 / component.mass)

    def test_auto_layout_centers_target_on_a_straight_axis(self) -> None:
        moving = rectangular_component("moving", -0.02, 0.02, -0.05, 0.05, -0.02, 0.02)
        stationary = rectangular_component("stationary", 0.2, 0.3, 0.8, 0.8001, 0.45, 0.55)

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 40.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "auto"),
            patch("cfd_motion.motion.COLLISION_INITIAL_GAP_M", 1.0),
        ):
            pair = configure_collision_convergence_components([moving, stationary])
            self.assertIsNotNone(pair)
            assert pair is not None
            self.assertEqual(collision_convergence_approach_axis(pair), (0.0, 1.0, 0.0))

            stationary_center_before = component_center_from_bounds(stationary)
            arrange_collision_convergence_initial_gap(pair)
            moving_center = component_center_from_bounds(moving)
            stationary_center = component_center_from_bounds(stationary)
            self.assertEqual(stationary_center, stationary_center_before)
            self.assertAlmostEqual(moving_center[0], stationary_center[0])
            self.assertAlmostEqual(moving_center[2], stationary_center[2])
            self.assertAlmostEqual(
                aabb_gap_along_axis(moving, stationary, (0.0, 1.0, 0.0)),
                1.0,
            )
            self.assertEqual(moving.linear_velocity, (0.0, 40.0, 0.0))

    def test_disconnected_unmated_solids_become_mass_conserving_bodies(self) -> None:
        first = rectangular_component("first", -0.2, -0.1, -0.1, 0.1, -0.1, 0.1)
        second = rectangular_component("second", 0.1, 0.3, -0.1, 0.1, -0.1, 0.1)
        combined = rectangular_component("part1", -0.2, 0.3, -0.1, 0.1, -0.1, 0.1)
        combined.triangles = first.triangles + second.triangles
        combined.mass = 6.0
        combined.material = MaterialProperties(
            material_name="Tungsten",
            density_kg_m3=19_250.0,
            mass_kg=6.0,
        )
        combined.collision_source_index = 1

        bodies = _split_unmated_component_bodies(combined)

        self.assertEqual(len(bodies), 2)
        self.assertEqual(sum(len(body.triangles) for body in bodies), 24)
        self.assertAlmostEqual(sum(body.mass for body in bodies), combined.mass)
        self.assertEqual({body.collision_source_index for body in bodies}, {1})
        self.assertEqual(len({body.collision_family for body in bodies}), 2)
        self.assertTrue(all(body.rigid_body_group is None for body in bodies))
        self.assertTrue(
            all(not component_has_decoded_assembly_mate(body) for body in bodies)
        )

    def test_unmated_source_bodies_share_launch_but_not_post_impact_state(self) -> None:
        moving_a = rectangular_component(
            "part1_a", -0.2, -0.1, -0.3, -0.1, -0.1, 0.1
        )
        moving_b = rectangular_component(
            "part1_b", -0.2, -0.1, 0.1, 0.3, -0.1, 0.1
        )
        stationary = rectangular_component(
            "part2", 0.5, 0.6, -0.5, 0.5, -0.5, 0.5
        )
        moving_a.collision_source_index = 1
        moving_b.collision_source_index = 1
        stationary.collision_source_index = 2
        components = [moving_a, moving_b, stationary]

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 10.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "x"),
            patch("cfd_motion.motion.COLLISION_SWEEP_CLAMPING", False),
            TemporaryDirectory() as tmpdir,
        ):
            pair = configure_collision_convergence_components(components)
            self.assertIsNotNone(pair)
            assert pair is not None
            self.assertIn(pair[0], (moving_a, moving_b))
            self.assertIs(pair[1], stationary)
            arrange_collision_convergence_initial_gap(pair, components)
            before_a = moving_a.cofr
            before_b = moving_b.cofr

            apply_collision_convergence_step(
                pair,
                0,
                0.01,
                Path(tmpdir) / "convergence.txt",
                (1.0, 0.0, 0.0),
                components,
            )

        delta_a = tuple(moving_a.cofr[i] - before_a[i] for i in range(3))
        delta_b = tuple(moving_b.cofr[i] - before_b[i] for i in range(3))
        self.assertEqual(delta_a, delta_b)
        self.assertEqual(moving_a.linear_velocity, (10.0, 0.0, 0.0))
        self.assertEqual(moving_b.linear_velocity, (10.0, 0.0, 0.0))
        self.assertIsNot(moving_a.freedom, moving_b.freedom)
        self.assertIsNone(moving_a.rigid_body_group)
        self.assertIsNone(moving_b.rigid_body_group)

        moving_a.linear_velocity = (3.0, 4.0, 0.0)
        self.assertEqual(moving_b.linear_velocity, (10.0, 0.0, 0.0))

    def test_prescribed_launch_preserves_independent_aerodynamic_motion(self) -> None:
        moving = rectangular_component(
            "part1_body", -0.2, -0.1, -0.1, 0.1, -0.1, 0.1
        )
        stationary = rectangular_component(
            "part2", 1.0, 1.1, -0.5, 0.5, -0.5, 0.5
        )
        moving.collision_source_index = 1
        stationary.collision_source_index = 2
        moving.material.linear_damping_per_kg = 0.0
        moving.material.angular_damping_per_kg = 0.0
        components = [moving, stationary]

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 10.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "x"),
            patch("cfd_motion.motion.COLLISION_SWEEP_CLAMPING", False),
            TemporaryDirectory() as tmpdir,
        ):
            pair = configure_collision_convergence_components(components)
            self.assertIsNotNone(pair)
            assert pair is not None
            start = moving.cofr
            _force, _moment, aerodynamic_translation, aerodynamic_rotation = (
                update_component_motion(
                    moving,
                    {},
                    0.01,
                    load_override=((0.0, 100.0, 0.0), (0.0, 0.0, 1.0)),
                    externally_applied_velocity=(10.0, 0.0, 0.0),
                )
            )

            self.assertAlmostEqual(aerodynamic_translation[0], 0.0)
            self.assertGreater(aerodynamic_translation[1], 0.0)
            self.assertGreater(math.sqrt(sum(v * v for v in aerodynamic_rotation)), 0.0)
            self.assertAlmostEqual(moving.cofr[0], start[0])
            aerodynamic_y = moving.cofr[1]
            transverse_velocity = moving.linear_velocity[1]
            angular_velocity = moving.angular_velocity

            apply_collision_convergence_step(
                pair,
                0,
                0.01,
                Path(tmpdir) / "convergence.txt",
                (1.0, 0.0, 0.0),
                components,
            )

        self.assertAlmostEqual(moving.cofr[0], start[0] + 0.1)
        self.assertAlmostEqual(moving.cofr[1], aerodynamic_y)
        self.assertEqual(moving.linear_velocity[1], transverse_velocity)
        self.assertEqual(moving.angular_velocity, angular_velocity)
        self.assertEqual(stationary.cofr, (1.05, 0.0, 0.0))
        displacement = tuple(moving.cofr[i] - start[i] for i in range(3))
        for axis in range(3):
            self.assertAlmostEqual(
                displacement[axis] / 0.01,
                moving.linear_velocity[axis],
            )

    def test_free_body_displacement_matches_recorded_high_speed(self) -> None:
        component = rectangular_component(
            "fast_body", -0.1, 0.1, -0.1, 0.1, -0.1, 0.1
        )
        component.freedom = MotionFreedom(
            translate_axes=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            rotate_axes=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
        )
        component.material.linear_damping_per_kg = 0.0
        component.material.angular_damping_per_kg = 0.0
        component.linear_velocity = (30.0, 0.0, 0.0)
        start = component.cofr

        _force, _moment, displacement, _rotation = update_component_motion(
            component,
            {},
            0.005,
            load_override=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )

        self.assertEqual(component.linear_velocity, (30.0, 0.0, 0.0))
        self.assertAlmostEqual(displacement[0], 0.15)
        self.assertAlmostEqual(component.cofr[0] - start[0], 0.15)

    def test_auto_axis_uses_the_frontal_centerline_not_tilted_face_normal(self) -> None:
        normal = (0.0, -math.sqrt(0.5), math.sqrt(0.5))
        target_triangle = (
            normal,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, math.sqrt(0.5), math.sqrt(0.5)),
        )
        stationary = AeroComponent(
            name="tilted_target",
            patch="tilted_target",
            triangles=[target_triangle],
            cofr=(0.5, math.sqrt(0.5) / 2.0, math.sqrt(0.5) / 2.0),
            lref=1.0,
            aref=0.5,
            mass=1.0,
        )
        moving = rectangular_component(
            "moving",
            0.4,
            0.6,
            -1.16,
            -0.96,
            0.25,
            0.45,
        )

        with patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "auto"):
            axis = collision_convergence_approach_axis((moving, stationary))

        self.assertAlmostEqual(axis[0], 0.0)
        self.assertAlmostEqual(axis[1], 1.0)
        self.assertAlmostEqual(axis[2], 0.0)

    def test_single_swept_impact_does_not_reverse_the_driver(self) -> None:
        moving = box_component("moving", 0.0, 1.0)
        stationary = box_component("stationary", 1.1, 2.1)

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 5.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "auto"),
            patch("cfd_motion.motion.COLLISION_SWEEP_PENETRATION_M", 0.001),
            patch("cfd_motion.motion.ENABLE_COLLISION_DEFORMATION", False),
            TemporaryDirectory() as tmpdir,
        ):
            pair = configure_collision_convergence_components([moving, stationary])
            self.assertIsNotNone(pair)
            assert pair is not None
            approach_axis = collision_convergence_approach_axis(pair)
            self.assertEqual(approach_axis, (1.0, 0.0, 0.0))

            convergence_log = Path(tmpdir) / "convergence.txt"
            collision_log = Path(tmpdir) / "collisions.txt"
            write_collision_convergence_log_header(convergence_log, pair)

            start_center = moving.cofr
            update_component_motion(
                moving,
                {},
                0.02,
                load_override=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                hold_kinematics=True,
            )
            self.assertEqual(moving.cofr, start_center)

            displacement, target_displacement, swept_contact = apply_collision_convergence_step(
                pair,
                0,
                0.02,
                convergence_log,
                approach_axis,
            )
            self.assertAlmostEqual(displacement[0], 0.101)
            self.assertEqual(target_displacement, (0.0, 0.0, 0.0))
            self.assertIsNotNone(swept_contact)

            contacts = resolve_part_collisions(
                [moving, stationary],
                0,
                collision_log,
                swept_contact,
                pair,
            )
            self.assertTrue(contacts)
            self.assertAlmostEqual(moving.linear_velocity[0], 0.0)
            self.assertEqual(stationary.linear_velocity, (0.0, 0.0, 0.0))

            previous_x = moving.cofr[0]
            for _ in range(5):
                update_component_motion(
                    moving,
                    {},
                    0.02,
                    load_override=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                )
                self.assertAlmostEqual(moving.cofr[0], previous_x)
                self.assertAlmostEqual(moving.linear_velocity[0], 0.0)
                previous_x = moving.cofr[0]

    def test_post_perforation_impactor_can_leave_the_approach_axis(self) -> None:
        moving = box_component("moving", 0.0, 1.0)
        moving.freedom = MotionFreedom(
            translate_axes=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            rotate_axes=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            source="post-perforation-ballistic",
        )
        moving.linear_velocity = (12.0, 0.0, 0.0)
        moving.angular_velocity = (0.0, 4.0, 0.0)

        start = moving.cofr
        _force, _moment, dpos, drot = update_component_motion(
            moving,
            {},
            0.05,
            load_override=((0.0, 200.0, 100.0), (0.0, 50.0, 25.0)),
        )

        self.assertGreater(dpos[0], 0.0)
        self.assertGreater(moving.cofr[1], start[1])
        self.assertGreater(moving.cofr[2], start[2])
        self.assertGreater(moving.linear_velocity[1], 0.0)
        self.assertGreater(moving.linear_velocity[2], 0.0)
        self.assertNotEqual(drot, (0.0, 0.0, 0.0))
        self.assertNotEqual(moving.angular_velocity, (0.0, 0.0, 0.0))

    def test_deformed_surfaces_close_to_mesh_contact(self) -> None:
        moving = box_component("moving", 0.0, 1.0)
        stationary = box_component("stationary", 1.1, 2.1)

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 5.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "x"),
            TemporaryDirectory() as tmpdir,
        ):
            pair = configure_collision_convergence_components([moving, stationary])
            self.assertIsNotNone(pair)
            assert pair is not None
            axis = collision_convergence_approach_axis(pair)
            convergence_log = Path(tmpdir) / "convergence.txt"
            collision_log = Path(tmpdir) / "collisions.txt"
            write_collision_convergence_log_header(convergence_log, pair)

            _move, _target_move, contact = apply_collision_convergence_step(
                pair,
                0,
                0.02,
                convergence_log,
                axis,
            )
            self.assertIsNotNone(contact)
            assert contact is not None
            resolve_part_collisions([moving, stationary], 0, collision_log, contact, pair)

            self.assertGreater(moving.deformation_max_m, 0.0)
            self.assertLessEqual(moving.deformation_max_m, 0.01)
            self.assertGreater(stationary.deformation_max_m, 0.0)
            post_contact = swept_mesh_contact(moving, stationary, axis, 0.005)
            self.assertIsNotNone(post_contact)
            assert post_contact is not None
            self.assertLessEqual(post_contact[0], 1e-4)

    def test_collision_shock_affects_nearby_non_contact_part(self) -> None:
        moving = box_component("moving", 0.0, 1.0)
        stationary = box_component("stationary", 1.1, 2.1)
        nearby = rectangular_component(
            "nearby",
            1.05,
            1.25,
            0.56,
            0.76,
            -0.1,
            0.1,
        )
        nearby.freedom = MotionFreedom(
            translate_axes=[
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            rotate_axes=[],
        )
        nearby.material = MaterialProperties(
            material_name="soft nearby polymer",
            density_kg_m3=1000.0,
            young_modulus_pa=2.0e7,
            poisson_ratio=0.35,
            yield_strength_pa=2.0e5,
            failure_strain=0.10,
        )

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 5.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "x"),
            TemporaryDirectory() as tmpdir,
        ):
            pair = configure_collision_convergence_components([moving, stationary, nearby])
            self.assertIsNotNone(pair)
            assert pair is not None
            convergence_log = Path(tmpdir) / "convergence.txt"
            collision_log = Path(tmpdir) / "collisions.txt"
            write_collision_convergence_log_header(convergence_log, pair)

            _move, _target_move, contact = apply_collision_convergence_step(
                pair,
                0,
                0.02,
                convergence_log,
                (1.0, 0.0, 0.0),
            )
            self.assertIsNotNone(contact)
            assert contact is not None

            resolve_part_collisions(
                [moving, stationary, nearby],
                0,
                collision_log,
                contact,
                pair,
            )

        self.assertGreater(
            math.sqrt(sum(value * value for value in nearby.linear_velocity)),
            0.0,
        )
        self.assertTrue(nearby.collision_damage)
        self.assertGreater(nearby.deformation_max_m, 0.0)

    def test_fixed_topology_deforms_coarse_triangle_containing_contact(self) -> None:
        component = rectangular_component(
            "coarse_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        original_triangles = list(component.triangles)
        applied = deform_component_at_contact(
            component,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.002,
            0.05,
        )

        self.assertGreater(applied, 0.0)
        self.assertEqual(len(component.triangles), len(original_triangles))
        self.assertNotEqual(component.triangles, original_triangles)

    def test_eulerian_contact_grid_contains_contact_penalty_pressure(self) -> None:
        component = rectangular_component(
            "grid_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        grid = build_eulerian_contact_grid(
            component.triangles,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            0.002,
            0.05,
        )

        self.assertTrue(grid.penalty_pressure_pa)
        self.assertGreaterEqual(grid.pressure_at((0.0, 0.0, 0.0)), 0.0)

    def test_collision_refinement_keeps_coarse_target_centered(self) -> None:
        component = rectangular_component(
            "coarse_target",
            0.0,
            0.0001,
            -0.05,
            0.05,
            -0.05,
            0.05,
        )
        original_center = component_center_from_bounds(component)
        original_count = len(component.triangles)
        refined = refine_collision_mesh_for_deformation(component)

        self.assertGreater(refined, 0)
        self.assertGreater(len(component.triangles), original_count)
        self.assertEqual(component_center_from_bounds(component), original_center)

    def test_tungsten_ball_perforates_point_one_mm_abs_sheet(self) -> None:
        moving = rectangular_component("ball", 0.0, 0.03, -0.015, 0.015, -0.015, 0.015)
        stationary = rectangular_component("sheet", 0.13, 0.1301, -0.05, 0.05, -0.05, 0.05)
        moving.mass = 0.272
        moving.material = MaterialProperties(
            material_name="tungsten",
            density_kg_m3=19300.0,
            mass_kg=moving.mass,
            young_modulus_pa=4.11e11,
            poisson_ratio=0.28,
        )
        stationary.material = MaterialProperties(
            material_name="ABS",
            density_kg_m3=1040.0,
            young_modulus_pa=2.0e9,
            poisson_ratio=0.35,
            thickness_m=0.0001,
        )

        with (
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_SPEED_MPS", 50.0),
            patch("cfd_motion.motion.COLLISION_CONVERGENCE_AXIS", "x"),
            TemporaryDirectory() as tmpdir,
        ):
            pair = configure_collision_convergence_components([moving, stationary])
            self.assertIsNotNone(pair)
            assert pair is not None
            convergence_log = Path(tmpdir) / "convergence.txt"
            collision_log = Path(tmpdir) / "collisions.txt"
            write_collision_convergence_log_header(convergence_log, pair)

            _move, _target_move, contact = apply_collision_convergence_step(
                pair,
                0,
                0.01,
                convergence_log,
                (1.0, 0.0, 0.0),
            )
            self.assertIsNotNone(contact)
            assert contact is not None
            self.assertTrue(contact.perforated)
            self.assertGreaterEqual(contact.hole_radius, 0.015)
            self.assertGreater(contact.residual_speed, 45.0)
            self.assertGreater(refine_thin_impact_target(stationary), 0)
            triangle_count_before = len(stationary.triangles)
            triangles_before = list(stationary.triangles)
            stationary_center_before = stationary.cofr
            total_mass_before = moving.mass + stationary.mass

            lines = resolve_part_collisions([moving, stationary], 0, collision_log, contact, pair)

            self.assertTrue(lines)
            self.assertLess(len(stationary.triangles), triangle_count_before)
            self.assertNotEqual(stationary.triangles, triangles_before)
            self.assertGreater(stationary.deformation_max_m, 0.0)
            self.assertGreater(moving.deformation_max_m, 0.0)
            self.assertLessEqual(moving.deformation_max_m, 0.01)
            self.assertEqual(stationary.cofr, stationary_center_before)
            self.assertEqual(stationary.linear_velocity, (0.0, 0.0, 0.0))
            self.assertTrue(stationary.collision_damage)
            damage = stationary.collision_damage[0]
            self.assertGreater(damage.current_hole_radius_m, 0.0)
            self.assertLess(damage.current_hole_radius_m, damage.target_hole_radius_m)
            self.assertGreaterEqual(
                damage.current_hole_radius_m,
                0.5 * damage.target_hole_radius_m,
            )
            for triangle in stationary.triangles:
                closest = closest_point_on_triangle(damage.contact_point, triangle)
                self.assertGreater(
                    radial_distance_from_axis(
                        closest,
                        damage.contact_point,
                        damage.inward_direction,
                    ),
                    damage.current_hole_radius_m,
                )
            visual_components = visualization_components_with_fragments([moving, stationary])
            self.assertGreaterEqual(len(visual_components), 3)
            self.assertAlmostEqual(
                sum(component.mass for component in visual_components),
                total_mass_before,
            )
            first_radius = damage.current_hole_radius_m
            first_triangles = list(stationary.triangles)
            advance_collision_damage_state(stationary, damage, damage.response_time_s)
            self.assertGreater(damage.current_hole_radius_m, first_radius)
            self.assertNotEqual(stationary.triangles, first_triangles)
            self.assertGreater(moving.linear_velocity[0], 45.0)
            self.assertEqual(moving.freedom.source, "post-perforation-ballistic")
