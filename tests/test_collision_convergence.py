from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cfd_motion.models import AeroComponent, MaterialProperties, MotionFreedom
from cfd_motion.motion import (
    aabb_gap_along_axis,
    apply_collision_convergence_step,
    arrange_collision_convergence_initial_gap,
    collision_convergence_approach_axis,
    component_center_from_bounds,
    configure_collision_convergence_components,
    contact_inverse_mass,
    resolve_part_collisions,
    swept_mesh_contact,
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

            resolve_part_collisions([moving, stationary], 0, collision_log, contact, pair)

            self.assertLess(len(stationary.triangles), triangle_count_before)
            self.assertGreater(moving.linear_velocity[0], 45.0)
            self.assertEqual(moving.freedom.source, "post-perforation-ballistic")
