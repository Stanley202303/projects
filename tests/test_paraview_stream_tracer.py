from pathlib import Path

from cfd_motion.visualization import (
    copy_minimal_stream_tracer_case_to_root,
    write_cfd_sampled_surface_preview_for_step,
)


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
    assert '<PointData Scalars="pressureCoeff" Vectors="U">' in text
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
