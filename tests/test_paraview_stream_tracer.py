from pathlib import Path
import xml.etree.ElementTree as ElementTree

from cfd_motion.models import AeroComponent, MotionFreedom
from cfd_motion.visualization import (
    CFD_SAMPLED_SURFACE_VTP_NAME,
    COMBINED_SURFACE_VTK_NAME,
    ROOT_CFD_SAMPLED_DIR_NAME,
    ROOT_PANEL_PREVIEW_DIR_NAME,
    _write_ascii_polydata_vtk,
    copy_minimal_stream_tracer_case_to_root,
    create_root_safe_pvd_from_copied_previews,
    validate_preview_polydata,
    write_cfd_sampled_surface_preview_for_step,
    write_panel_aero_preview_for_step,
)


def test_main_motion_pvd_keeps_combined_geometry_when_cfd_surface_exists(tmp_path: Path) -> None:
    panel_dir = tmp_path / ROOT_PANEL_PREVIEW_DIR_NAME
    sampled_dir = tmp_path / ROOT_CFD_SAMPLED_DIR_NAME
    panel_dir.mkdir()
    sampled_dir.mkdir()
    _write_ascii_polydata_vtk(
        panel_dir / f"frame_000_{COMBINED_SURFACE_VTK_NAME}",
        [],
        [],
        {"CpPanel": []},
    )
    _write_ascii_polydata_vtk(
        sampled_dir / f"frame_000_{CFD_SAMPLED_SURFACE_VTP_NAME}",
        [],
        [],
        {"CpPanel": []},
    )

    pvd = create_root_safe_pvd_from_copied_previews(tmp_path, 1)

    assert pvd is not None
    pvd_text = pvd.read_text()
    assert ROOT_PANEL_PREVIEW_DIR_NAME in pvd_text
    assert ROOT_CFD_SAMPLED_DIR_NAME not in pvd_text


def test_main_motion_pvd_skips_invalid_preview_frame(tmp_path: Path) -> None:
    panel_dir = tmp_path / ROOT_PANEL_PREVIEW_DIR_NAME
    panel_dir.mkdir()
    (panel_dir / f"frame_000_{COMBINED_SURFACE_VTK_NAME}").write_text(
        '<VTKFile type="PolyData"><PolyData><Piece NumberOfPoints="0" NumberOfPolys="1">'
        '<CellData><DataArray Name="CpPanel">0</DataArray></CellData>'
        '</Piece></PolyData></VTKFile>'
    )

    pvd = create_root_safe_pvd_from_copied_previews(tmp_path, 1)

    assert pvd is None
    assert "invalid panel preview" in (tmp_path / "paraview_pvd_manifest.txt").read_text()


def test_polydata_fields_are_written_as_cell_data_for_shared_vertices(tmp_path: Path) -> None:
    output = tmp_path / "shared_vertices.vtp"
    _write_ascii_polydata_vtk(
        output,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2), (0, 2, 3)],
        {"CpPanel": [1.0, 3.0]},
    )

    root = ElementTree.parse(output).getroot()
    piece = root.find(".//Piece")
    assert piece is not None
    arrays = piece.find("CellData")
    assert arrays is not None
    cp = next(array for array in arrays.findall("DataArray") if array.attrib.get("Name") == "CpPanel")
    values = [float(value) for value in (cp.text or "").split()]
    assert len(values) == 2
    assert values == [1.0, 3.0]
    validate_preview_polydata(output)


def test_panel_preview_initializes_structural_cell_fields(tmp_path: Path) -> None:
    component = AeroComponent(
        name="target",
        patch="target",
        triangles=[
            (
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ],
        cofr=(0.0, 0.0, 0.0),
        lref=1.0,
        aref=0.5,
        freedom=MotionFreedom(),
        linear_velocity=(1.0, 2.0, 2.0),
    )

    write_panel_aero_preview_for_step(tmp_path, [component], 0)

    output = tmp_path / "panel_preview" / "combined_moving_surfaces.vtp"
    validate_preview_polydata(output)
    root = ElementTree.parse(output).getroot()
    names = {
        array.attrib.get("Name")
        for array in root.findall(".//CellData/DataArray")
    }
    assert {
        "structuralDisplacementM",
        "perforationPlug",
        "structuralFailedEdges",
        "velocityX",
        "velocityY",
        "velocityZ",
        "velocityMagnitude",
        "speed",
        "worldVelocityX",
        "worldVelocityY",
        "worldVelocityZ",
        "worldSpeed",
    }.issubset(names)
    velocity = next(
        array
        for array in root.findall(".//CellData/DataArray")
        if array.attrib.get("Name") == "velocity"
    )
    velocity_values = [float(value) for value in (velocity.text or "").split()]
    assert velocity.attrib.get("NumberOfComponents") == "3"
    assert velocity_values == [1.0, 2.0, 2.0]
    world_velocity = next(
        array
        for array in root.findall(".//CellData/DataArray")
        if array.attrib.get("Name") == "worldVelocity"
    )
    assert world_velocity.attrib.get("NumberOfComponents") == "3"
    speed = next(
        array
        for array in root.findall(".//CellData/DataArray")
        if array.attrib.get("Name") == "speed"
    )
    speed_values = [float(value) for value in (speed.text or "").split()]
    assert speed_values == [3.0]


def test_cfd_sampled_surface_vtp_preserves_u_vector_for_stream_tracer(tmp_path: Path) -> None:
    sampled_dir = tmp_path / "postProcessing" / "sampledWallSurfaces" / "1"
    sampled_dir.mkdir(parents=True)
    (sampled_dir / "wing_sampled.vtk").write_text(
        "\n".join(
            [
                "# vtk DataFile Version 3.0",
                "sampled triangle",
                "ASCII",
                "DATASET POLYDATA",
                "POINTS 3 float",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "POLYGONS 1 4",
                "3 0 1 2",
                "POINT_DATA 3",
                "VECTORS U float",
                "1 0 0",
                "0 2 0",
                "0 0 3",
                "CELL_DATA 1",
                "SCALARS p float 1",
                "LOOKUP_TABLE default",
                "10",
                "",
            ]
        )
    )

    out = write_cfd_sampled_surface_preview_for_step(tmp_path, 0)

    assert out is not None
    text = out.read_text()
    assert '<PointData/>' in text
    assert '<CellData Scalars="pressureCoeff" Vectors="U">' in text
    assert 'Name="U" NumberOfComponents="3"' in text
    assert "0.333333333 0.666666667 1" in text


def test_minimal_stream_tracer_case_keeps_only_latest_mesh_and_u(tmp_path: Path) -> None:
    step_case = tmp_path / "step_work"
    root_case = tmp_path / "actual_model_case"
    root_case.mkdir()

    poly_mesh = step_case / "constant" / "polyMesh"
    poly_mesh.mkdir(parents=True)
    (poly_mesh / "points").write_text("points\n")
    (step_case / "constant" / "transportProperties").write_text("transport\n")

    system_dir = step_case / "system"
    system_dir.mkdir(parents=True)
    (system_dir / "controlDict").write_text("control\n")

    latest = step_case / "1"
    latest.mkdir(parents=True)
    (latest / "p").write_text("internalField uniform 1;\nboundaryField {}\n")
    (latest / "U").write_text("internalField uniform (1 0 0);\nboundaryField {}\n")

    foam_path = copy_minimal_stream_tracer_case_to_root(root_case, step_case, 3)

    assert foam_path == root_case / "stream_tracer_volume_case" / "case.foam"
    assert foam_path.exists()
    assert (root_case / "stream_tracer_volume_case" / "constant" / "polyMesh" / "points").read_text() == "points\n"
    assert (root_case / "stream_tracer_volume_case" / "constant" / "transportProperties").read_text() == "transport\n"
    assert (root_case / "stream_tracer_volume_case" / "system" / "controlDict").read_text() == "control\n"
    assert (root_case / "stream_tracer_volume_case" / "1" / "U").exists()
    assert not (root_case / "stream_tracer_volume_case" / "1" / "p").exists()
    assert "retained_data=constant/polyMesh, system, latest U only" in (
        root_case / "paraview_stream_tracer_manifest.txt"
    ).read_text()
