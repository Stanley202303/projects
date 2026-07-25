import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cfd_motion.models import AeroComponent, MaterialProperties, MotionFreedom
from cfd_motion.motion import (
    advance_collision_damage_state,
    aabb_gap_along_axis,
    apply_collision_convergence_step,
    arrange_collision_convergence_initial_gap,
    collision_convergence_approach_axis,
    component_center_from_bounds,
    configure_collision_convergence_components,
    contact_inverse_mass,
    deform_component_at_contact,
    fracture_thin_shell,
    local_contact_geometry,
    register_collision_dent,
    register_collision_hole,
    resolve_part_collisions,
    swept_mesh_contact,
    triangle_area_centroid_normal,
    update_component_deformation,
    update_component_motion,
    write_collision_convergence_log_header,
)
from cfd_motion.runner import refine_thin_impact_target


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

    def test_two_mesh_spheres_use_curvature_and_both_deform(self) -> None:
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
            self.assertGreater(stationary.deformation_max_m, 0.0)
            self.assertEqual(stationary.linear_velocity, (0.0, 0.0, 0.0))

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

        advance_collision_damage_state(
            target,
            damage,
            0.25 * damage.response_time_s,
        )
        first_radius = damage.current_hole_radius_m
        self.assertGreater(first_radius, 0.0)
        self.assertLess(first_radius, damage.target_hole_radius_m)

        advance_collision_damage_state(
            target,
            damage,
            10.0 * damage.response_time_s,
        )
        self.assertGreater(damage.current_hole_radius_m, first_radius)
        self.assertAlmostEqual(
            damage.current_hole_radius_m,
            damage.target_hole_radius_m,
            delta=1e-6,
        )
        self.assertGreater(len(target.triangles), initial_triangle_count)
        retained_inner_fragments = []
        intact_face_centroid_radii = []
        for triangle in target.triangles:
            area, centroid, normal = triangle_area_centroid_normal(triangle)
            if area > 1e-18 and abs(normal[0]) >= 0.5:
                radial = math.hypot(centroid[1], centroid[2])
                if radial < damage.target_hole_radius_m - 1e-6:
                    retained_inner_fragments.append(abs(centroid[0]))
                else:
                    intact_face_centroid_radii.append(radial)
        # A perforation opens the original sheet while retaining failed material
        # as displaced fragments rather than deleting it.
        self.assertTrue(retained_inner_fragments)
        self.assertGreater(min(retained_inner_fragments), 1e-5)
        self.assertTrue(intact_face_centroid_radii)

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
            self.assertGreater(stationary.deformation_max_m, 0.0)
            post_contact = swept_mesh_contact(moving, stationary, axis, 0.005)
            self.assertIsNotNone(post_contact)
            assert post_contact is not None
            self.assertAlmostEqual(post_contact[0], 0.0, places=9)

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
            self.assertGreater(contact.residual_speed, 45.0)
            self.assertGreater(refine_thin_impact_target(stationary), 0)
            triangle_count_before = len(stationary.triangles)

            lines = resolve_part_collisions([moving, stationary], 0, collision_log, contact, pair)

            self.assertTrue(lines)
            self.assertGreater(len(stationary.triangles), triangle_count_before)
            self.assertTrue(stationary.collision_damage)
            self.assertAlmostEqual(
                stationary.collision_damage[0].current_hole_radius_m,
                stationary.collision_damage[0].target_hole_radius_m,
                delta=1e-6,
            )
            self.assertGreater(moving.linear_velocity[0], 45.0)
            self.assertEqual(moving.freedom.source, "post-perforation-ballistic")
