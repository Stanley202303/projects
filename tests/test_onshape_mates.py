import math

import pytest

from cfd_motion.geometry import bom_records_from_payload
from cfd_motion.onshape import deduce_component_freedoms, transform_motion_freedom_to_world
from cfd_motion.motion import (
    apply_aerodynamic_velocity_increment,
    apply_relative_motion_policy,
    apply_rigid_body_motion,
    enforce_attachment_constraints,
    enforce_component_attachment_constraint,
    surface_pressure_load,
    update_component_motion,
)
from cfd_motion.models import AeroComponent, MotionFreedom


def test_onshape_bom_material_library_properties_are_parsed() -> None:
    payload = {
        "rows": [
            {
                "name": "Part 1",
                "headerIdToValue": {
                    "materialColumn": {
                        "displayName": "Tungsten",
                        "properties": [
                            {
                                "name": "COMPRESSIVE_YIELD_STRENGTH",
                                "displayName": "Compressive Yield Strength",
                                "value": "0",
                                "units": "Pa",
                            },
                            {
                                "name": "YOUNGS_MODULUS",
                                "displayName": "Young's Modulus",
                                "value": "400000000000",
                                "units": "Pa",
                            },
                            {
                                "name": "TENSILE_YIELD_STRENGTH",
                                "displayName": "Tensile Yield Strength",
                                "value": "750000000",
                                "units": "Pa",
                            },
                            {
                                "name": "DENS",
                                "displayName": "Density",
                                "value": "19600",
                                "units": "kg/m^3",
                            },
                            {
                                "name": "POISSONS_RATIO",
                                "displayName": "Poisson's Ratio",
                                "value": "0.28",
                                "units": "",
                            },
                        ],
                    },
                    "massColumn": "3.95 lb",
                },
            }
        ]
    }

    records = bom_records_from_payload(payload)
    tungsten = next(record for record in records if record["material"] == "Tungsten")

    assert tungsten["mass_kg"] == pytest.approx(1.7916898615)
    assert tungsten["density_kg_m3"] == pytest.approx(19600.0)
    assert tungsten["young_modulus_pa"] == pytest.approx(4.0e11)
    assert tungsten["poisson_ratio"] == pytest.approx(0.28)
    assert tungsten["yield_strength_pa"] == pytest.approx(7.5e8)


def test_revolute_mated_occurrence_overrides_fastened_group() -> None:
    assembly_def = {
        "rootAssembly": {
            "occurrences": [
                {"path": ["root"], "fixed": False},
                {"path": ["flap"], "fixed": False},
            ],
            "features": [
                {
                    "featureType": "mate",
                    "featureData": {
                        "mateType": "FASTENED",
                        "matedEntities": [
                            {"matedOccurrence": ["root"]},
                            {"matedOccurrence": ["flap"]},
                        ],
                    },
                },
                {
                    "featureType": "mate",
                    "featureData": {
                        "mateType": "REVOLUTE",
                        "matedEntities": [
                            {
                                "matedOccurrence": ["root"],
                                "mateConnectorCS": {"zAxis": [0.0, 0.0, 1.0]},
                            },
                            {
                                "matedOccurrence": ["flap"],
                                "mateConnectorCS": {"zAxis": [0.0, 0.0, 1.0]},
                            },
                        ],
                    },
                },
            ],
        }
    }

    freedoms, report = deduce_component_freedoms(
        assembly_def,
        assembly_def["rootAssembly"]["occurrences"],
    )

    assert "paths=['root', 'flap']" in "\n".join(report)
    assert freedoms["flap"].mate_type == "REVOLUTE"
    assert freedoms["flap"].translate_axes == []
    assert freedoms["flap"].rotate_axes == [(0.0, 0.0, 1.0)]


def test_revolute_mate_extracts_connector_origins_and_reference_occurrence() -> None:
    assembly_def = {
        "rootAssembly": {
            "occurrences": [
                {"path": ["root"], "fixed": False},
                {"path": ["flap"], "fixed": False},
            ],
            "features": [
                {
                    "id": "mate-1",
                    "featureType": "mate",
                    "featureData": {
                        "mateType": "REVOLUTE",
                        "matedEntities": [
                            {
                                "matedOccurrence": ["root"],
                                "mateConnectorCS": {
                                    "xAxis": [1.0, 0.0, 0.0],
                                    "yAxis": [0.0, 1.0, 0.0],
                                    "zAxis": [0.0, 0.0, 1.0],
                                    "origin": [1.0, 2.0, 3.0],
                                },
                            },
                            {
                                "matedOccurrence": ["flap"],
                                "mateConnectorCS": {
                                    "xAxis": [1.0, 0.0, 0.0],
                                    "yAxis": [0.0, -1.0, 0.0],
                                    "zAxis": [0.0, 0.0, -1.0],
                                    "origin": [1.0, 2.0, 3.0],
                                },
                            },
                        ],
                    },
                }
            ],
        }
    }

    freedoms, _report = deduce_component_freedoms(
        assembly_def,
        assembly_def["rootAssembly"]["occurrences"],
    )

    flap = freedoms["flap"]
    assert flap.mate_type == "REVOLUTE"
    assert flap.mate_origin == (1.0, 2.0, 3.0)
    assert flap.mate_reference_origin == (1.0, 2.0, 3.0)
    assert flap.mate_reference_occurrence == "root"


def test_attachment_constraint_projects_mate_origin_back_to_revolute_anchor() -> None:
    component = AeroComponent(
        name="flap",
        patch="flap",
        triangles=[],
        cofr=(1.5, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
    )
    component.freedom.mate_type = "REVOLUTE"
    component.freedom.rotate_axes = [(0.0, 0.0, 1.0)]
    component.motion_origin = (0.0, 0.0, 0.0)
    component.mate_origin = (0.2, 0.1, 0.0)
    component.mate_reference_origin = (0.0, 0.0, 0.0)

    correction = enforce_component_attachment_constraint(component)

    assert correction == (-0.2, -0.1, 0.0)
    assert component.mate_origin == (0.0, 0.0, 0.0)
    assert component.cofr == (1.3, -0.1, 0.0)


def test_update_component_motion_keeps_revolute_connector_attached() -> None:
    triangle = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.1, 0.0),
        (1.1, 0.0, 0.0),
    )
    component = AeroComponent(
        name="flap",
        patch="flap",
        triangles=[triangle],
        cofr=(1.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
    )
    component.freedom.mate_type = "REVOLUTE"
    component.freedom.rotate_axes = [(0.0, 0.0, 1.0)]
    component.motion_origin = (0.0, 0.0, 0.0)
    component.mate_origin = (0.0, 0.0, 0.0)
    component.mate_reference_origin = (0.0, 0.0, 0.0)

    _force, _moment, _dpos, _drot = update_component_motion(
        component,
        {"CmYaw": 10.0},
        0.1,
    )

    assert component.mate_origin == (0.0, 0.0, 0.0)


def test_transform_motion_freedom_to_world_rotates_axis_and_origins() -> None:
    local = MotionFreedom(
        translate_axes=[],
        rotate_axes=[(0.0, 0.0, 1.0)],
        mate_type="REVOLUTE",
        source="mate:test",
        mate_origin=(1.0, 0.0, 0.0),
        mate_reference_origin=(0.0, 1.0, 0.0),
        mate_reference_occurrence="root",
    )
    quarter_turn_about_x = [
        1.0, 0.0, 0.0, 10.0,
        0.0, 0.0, -1.0, 20.0,
        0.0, 1.0, 0.0, 30.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    reference_transform = {
        "root": {
            "transform": [
                1.0, 0.0, 0.0, 100.0,
                0.0, 1.0, 0.0, 200.0,
                0.0, 0.0, 1.0, 300.0,
                0.0, 0.0, 0.0, 1.0,
            ]
        }
    }

    world = transform_motion_freedom_to_world(local, quarter_turn_about_x, reference_transform)

    assert world.rotate_axes[0] == pytest.approx((0.0, -1.0, 0.0))
    assert world.mate_origin == pytest.approx((11.0, 20.0, 30.0))
    assert world.mate_reference_origin == pytest.approx((100.0, 201.0, 300.0))


def test_fastened_attachment_does_not_flip_body_when_origin_already_matches() -> None:
    parent = AeroComponent(
        name="parent",
        patch="parent",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
        source_occurrence="parent",
        mate_origin=(0.0, 0.0, 0.0),
        mate_reference_occurrence="child",
        mate_x_axis=(0.0, 1.0, 0.0),
        mate_y_axis=(-1.0, 0.0, 0.0),
        mate_z_axis=(0.0, 0.0, 1.0),
        is_assembly_anchor=True,
    )
    child = AeroComponent(
        name="child",
        patch="child",
        triangles=[],
        cofr=(2.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
        source_occurrence="child",
        mate_origin=(0.0, 0.0, 0.0),
        mate_reference_occurrence="parent",
        mate_reference_origin=(0.0, 0.0, 0.0),
        mate_x_axis=(1.0, 0.0, 0.0),
        mate_y_axis=(0.0, 1.0, 0.0),
        mate_z_axis=(0.0, 0.0, 1.0),
    )
    child.freedom.mate_type = "FASTENED"

    corrections = enforce_attachment_constraints([parent, child])

    assert corrections == {}
    assert child.mate_origin == pytest.approx((0.0, 0.0, 0.0))
    assert child.cofr == pytest.approx((2.0, 0.0, 0.0))


def test_attachment_uses_static_reference_when_other_component_primary_mate_is_unrelated() -> None:
    reference = AeroComponent(
        name="reference",
        patch="reference",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
        source_occurrence="reference",
        mate_origin=(10.0, 0.0, 0.0),
        mate_reference_occurrence="elsewhere",
        mate_x_axis=(1.0, 0.0, 0.0),
        mate_y_axis=(0.0, 1.0, 0.0),
        mate_z_axis=(0.0, 0.0, 1.0),
    )
    child = AeroComponent(
        name="child",
        patch="child",
        triangles=[],
        cofr=(1.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
        source_occurrence="child",
        mate_origin=(1.0, 0.0, 0.0),
        mate_reference_occurrence="reference",
        mate_reference_origin=(0.0, 0.0, 0.0),
        mate_x_axis=(1.0, 0.0, 0.0),
        mate_y_axis=(0.0, 1.0, 0.0),
        mate_z_axis=(0.0, 0.0, 1.0),
        mate_reference_x_axis=(1.0, 0.0, 0.0),
        mate_reference_y_axis=(0.0, 1.0, 0.0),
        mate_reference_z_axis=(0.0, 0.0, 1.0),
    )
    child.freedom.mate_type = "FASTENED"

    corrections = enforce_attachment_constraints([reference, child])

    assert corrections["child"] == pytest.approx((-1.0, 0.0, 0.0))
    assert child.mate_origin == pytest.approx((0.0, 0.0, 0.0))


def test_surface_pressure_load_uses_local_point_velocity() -> None:
    triangle = (
        (1.0, 0.0, 0.0),
        (0.0, 0.9, 0.0),
        (0.0, 1.1, 0.0),
        (0.0, 1.0, 0.2),
    )
    static_component = AeroComponent(
        name="fin",
        patch="fin",
        triangles=[triangle],
        cofr=(0.0, 1.0, 0.05),
        lref=1.0,
        aref=1.0,
        motion_origin=(0.0, 0.0, 0.0),
    )
    spinning_component = AeroComponent(
        name="fin",
        patch="fin",
        triangles=[triangle],
        cofr=(0.0, 1.0, 0.05),
        lref=1.0,
        aref=1.0,
        motion_origin=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 60.0),
    )

    static_force, _static_moment = surface_pressure_load(static_component)
    spinning_force, _spinning_moment = surface_pressure_load(spinning_component)

    assert abs(spinning_force[0]) < abs(static_force[0])


def test_update_component_motion_filters_aerodynamic_loads_between_steps() -> None:
    component = AeroComponent(
        name="slider",
        patch="slider",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
    )
    component.freedom.translate_axes = [(1.0, 0.0, 0.0)]

    first_force, _first_moment, _first_dpos, _first_drot = update_component_motion(
        component,
        {},
        0.05,
        load_override=((10.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    second_force, _second_moment, _second_dpos, _second_drot = update_component_motion(
        component,
        {},
        0.05,
        load_override=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )

    assert first_force == pytest.approx((10.0, 0.0, 0.0))
    assert 0.0 < second_force[0] < first_force[0]


def test_aerodynamic_velocity_increment_changes_a_free_body_velocity() -> None:
    component = AeroComponent(
        name="fragment",
        patch="fragment",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
        mass=2.0,
        freedom=MotionFreedom(
            translate_axes=[(1.0, 0.0, 0.0)],
            rotate_axes=[],
            mate_type="COLLISION_FRAGMENT",
            source="hybrid-shell-fragment",
        ),
    )

    force, _moment = apply_aerodynamic_velocity_increment(
        component,
        0.5,
        load_override=((4.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )

    assert force == pytest.approx((4.0, 0.0, 0.0))
    assert component.linear_velocity[0] > 0.9


def test_update_component_motion_clamps_revolute_primary_limit() -> None:
    component = AeroComponent(
        name="flap",
        patch="flap",
        triangles=[],
        cofr=(1.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
    )
    component.freedom.mate_type = "REVOLUTE"
    component.freedom.rotate_axes = [(0.0, 0.0, 1.0)]
    component.freedom.limits = {"primary": (-0.1, 0.1)}
    component.motion_origin = (0.0, 0.0, 0.0)
    component.mate_origin = (0.0, 0.0, 0.0)
    component.mate_reference_origin = (0.0, 0.0, 0.0)

    _force, _moment, _dpos, _drot = update_component_motion(
        component,
        {},
        0.2,
        load_override=((0.0, 0.0, 0.0), (0.0, 0.0, 5.0)),
    )

    assert component.total_rotation[2] == pytest.approx(0.1)
    assert component.angular_velocity[2] == pytest.approx(0.0)


def test_free_body_aerodynamic_moment_spins_about_its_own_center() -> None:
    component = AeroComponent(
        name="free-body",
        patch="free-body",
        triangles=[],
        cofr=(1.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
        freedom=MotionFreedom(
            translate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            rotate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            mate_type="FREE",
            source="unmated",
        ),
    )

    _force, moment, _dpos, drot = update_component_motion(
        component,
        {},
        0.1,
        load_override=((0.0, 0.0, 0.0), (0.0, 0.0, 5.0)),
    )

    assert moment[2] == pytest.approx(5.0)
    assert drot[2] > 0.0
    assert component.cofr == pytest.approx((1.0, 0.0, 0.0))


def test_propeller_like_revolute_part_spins_from_aerodynamic_moment() -> None:
    component = AeroComponent(
        name="propeller",
        patch="propeller",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=0.4,
        aref=0.2,
        freedom=MotionFreedom(
            translate_axes=[],
            rotate_axes=[(1.0, 0.0, 0.0)],
            mate_type="REVOLUTE",
            source="mate:prop-shaft",
        ),
        motion_origin=(0.0, 0.0, 0.0),
        mate_origin=(0.0, 0.0, 0.0),
        mate_reference_origin=(0.0, 0.0, 0.0),
    )

    _force, moment, _dpos, drot = update_component_motion(
        component,
        {},
        0.1,
        load_override=((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )

    assert moment[0] == pytest.approx(3.0)
    assert drot[0] > 0.0
    assert component.angular_velocity[0] > 0.0
    assert component.total_rotation[0] > 0.0
    assert component.total_rotation[1] == pytest.approx(0.0)
    assert component.total_rotation[2] == pytest.approx(0.0)


def test_all_fastened_assembly_falls_back_to_rigid_body_motion(tmp_path) -> None:
    root = AeroComponent(
        name="body",
        patch="body",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
        mass=2.0,
    )
    fin = AeroComponent(
        name="fin",
        patch="fin",
        triangles=[],
        cofr=(0.0, 1.0, 0.0),
        lref=0.5,
        aref=0.2,
        mass=0.2,
    )
    root.freedom.mate_type = "FASTENED"
    root.freedom.source = "mate:root"
    fin.freedom.mate_type = "FASTENED"
    fin.freedom.source = "mate:fin"

    apply_relative_motion_policy([root, fin], tmp_path)

    assert root.is_assembly_anchor is False
    assert root.freedom.source == "assembly-rigid-body-root"
    assert len(root.freedom.translate_axes) == 3
    assert len(root.freedom.rotate_axes) == 3
    assert fin.freedom.source == "assembly-rigid-body-follower"
    assert fin.freedom.translate_axes == []
    assert fin.freedom.rotate_axes == []


def test_unmated_components_remain_independent_bodies(tmp_path) -> None:
    first = AeroComponent(
        name="free-first",
        patch="free-first",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
    )
    second = AeroComponent(
        name="free-second",
        patch="free-second",
        triangles=[],
        cofr=(1.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
    )

    lines = apply_relative_motion_policy([first, second], tmp_path)

    assert not first.is_assembly_anchor
    assert not second.is_assembly_anchor
    assert first.freedom.source == "unmated"
    assert second.freedom.source == "unmated"
    assert len(first.freedom.translate_axes) == 3
    assert len(second.freedom.rotate_axes) == 3
    assert "independent free body" in "\n".join(lines)


def test_unmated_part_is_not_absorbed_by_fastened_mate_group(tmp_path) -> None:
    root = AeroComponent(
        name="mated-root",
        patch="mated-root",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
    )
    follower = AeroComponent(
        name="mated-follower",
        patch="mated-follower",
        triangles=[],
        cofr=(0.0, 1.0, 0.0),
        lref=1.0,
        aref=1.0,
    )
    free_part = AeroComponent(
        name="free-part",
        patch="free-part",
        triangles=[],
        cofr=(5.0, 0.0, 0.0),
        lref=10.0,
        aref=10.0,
    )
    root.freedom = MotionFreedom([], [], "FASTENED", "mate:root")
    follower.freedom = MotionFreedom([], [], "FASTENED", "mate:follower")

    apply_relative_motion_policy([root, follower, free_part], tmp_path)

    assert root.freedom.source == "assembly-rigid-body-root"
    assert follower.freedom.source == "assembly-rigid-body-follower"
    assert free_part.freedom.source == "unmated"
    assert len(free_part.freedom.translate_axes) == 3


def test_apply_rigid_body_motion_moves_followers_with_root_rotation() -> None:
    root = AeroComponent(
        name="body",
        patch="body",
        triangles=[],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=1.0,
    )
    fin = AeroComponent(
        name="fin",
        patch="fin",
        triangles=[],
        cofr=(0.0, 1.0, 0.0),
        lref=0.5,
        aref=0.2,
    )

    apply_rigid_body_motion(
        [root, fin],
        (1.0, 0.0, 0.0),
        (0.0, 0.0, math.pi / 2.0),
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, math.pi / 2.0),
    )

    assert root.cofr == pytest.approx((0.0, 1.0, 0.0))
    assert fin.cofr == pytest.approx((-1.0, 1.0, 0.0))
    assert root.linear_velocity == pytest.approx((3.0, 0.0, 0.0))
    assert fin.angular_velocity == pytest.approx((0.0, 0.0, 2.0))
    assert fin.total_rotation == pytest.approx((0.0, 0.0, math.pi / 2.0))


def test_fastened_to_root_beats_unrelated_revolutes() -> None:
    assembly_def = {
        "rootAssembly": {
            "occurrences": [
                {"path": ["root"], "fixed": False},
                {"path": ["hub"], "fixed": False},
                {"path": ["flap"], "fixed": False},
            ],
            "features": [
                {
                    "featureType": "mate",
                    "featureData": {
                        "mateType": "FASTENED",
                        "matedEntities": [
                            {"matedOccurrence": ["root"], "mateConnectorCS": {"zAxis": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0]}},
                            {"matedOccurrence": ["hub"], "mateConnectorCS": {"zAxis": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0]}},
                        ],
                    },
                },
                {
                    "featureType": "mate",
                    "featureData": {
                        "mateType": "REVOLUTE",
                        "matedEntities": [
                            {"matedOccurrence": ["hub"], "mateConnectorCS": {"zAxis": [0.0, 1.0, 0.0], "origin": [1.0, 0.0, 0.0]}},
                            {"matedOccurrence": ["flap"], "mateConnectorCS": {"zAxis": [0.0, 1.0, 0.0], "origin": [1.0, 0.0, 0.0]}},
                        ],
                    },
                },
            ],
        }
    }

    freedoms, _report = deduce_component_freedoms(
        assembly_def,
        assembly_def["rootAssembly"]["occurrences"],
    )

    assert freedoms["hub"].mate_type == "FASTENED"
    assert freedoms["hub"].translate_axes == []
    assert freedoms["hub"].rotate_axes == []
