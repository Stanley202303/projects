import pytest

from cfd_motion.onshape import deduce_component_freedoms, transform_motion_freedom_to_world
from cfd_motion.motion import (
    enforce_attachment_constraints,
    enforce_component_attachment_constraint,
    surface_pressure_load,
    update_component_motion,
)
from cfd_motion.models import AeroComponent, MotionFreedom


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
