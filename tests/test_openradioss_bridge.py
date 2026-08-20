from pathlib import Path
import subprocess
import sys

import pytest

from cfd_motion.models import AeroComponent, MaterialProperties
from cfd_motion.openradioss import (
    OpenRadiossError,
    _animation_interval_from_deck,
    _ensure_runtime_image,
    _validate_engine_output,
    exclusive_case_lock,
    partial_result_updater,
    write_openradioss_deck,
)


def _component(name: str, offset: float = 0.0) -> AeroComponent:
    return AeroComponent(
        name=name,
        patch=name,
        triangles=[
            ((0.0, 0.0, 1.0), (offset, 0.0, 0.0), (offset + 0.1, 0.0, 0.0), (offset, 0.1, 0.0)),
            ((0.0, 0.0, 1.0), (offset + 0.1, 0.0, 0.0), (offset + 0.1, 0.1, 0.0), (offset, 0.1, 0.0)),
        ],
        cofr=(offset + 0.05, 0.05, 0.0),
        lref=0.1,
        aref=0.01,
        material=MaterialProperties(
            material_name="Steel",
            density_kg_m3=7850.0,
            young_modulus_pa=210e9,
            poisson_ratio=0.3,
            thickness_m=0.001,
        ),
        linear_velocity=(12.0, 0.0, 0.0),
    )


def test_shell_export_preserves_faces_materials_velocity_and_contact(tmp_path: Path) -> None:
    starter, engine = write_openradioss_deck(
        [_component("impactor"), _component("target", 0.2)],
        tmp_path,
        duration_s=0.01,
        animation_interval_s=0.002,
    )
    starter_text = starter.read_text()
    engine_text = engine.read_text()
    assert starter_text.count("\n/SH3N/") == 2
    assert "         1         1         2         3         3" not in starter_text
    assert starter_text.count("/MAT/ELAST/") == 2
    assert starter_text.count("/INTER/TYPE7/") == 1
    assert "7.850000000000E+03" in starter_text
    assert "1.200000000000E+01" in starter_text
    node_lines = starter_text.split("/NODE\n", 1)[1].split("/MAT/ELAST/1", 1)[0].splitlines()
    assert node_lines
    assert all(len(line) == 70 for line in node_lines)
    assert "1.000000000000E+00" in starter_text.split("/INTER/TYPE7/", 1)[1]
    assert "       000                             5" in starter_text
    assert "/ANIM/ELEM/VONM" in engine_text
    assert "/ANIM/NODA/VEL" in engine_text
    assert "/PRINT/-5000" in engine_text
    report = (tmp_path / "openradioss_export_report.txt").read_text()
    assert "triangular_shell_elements=4" in report
    assert "requested_duration_s=0.01" in report
    assert "animation_interval_s=0.002" in report
    assert "print_cycle_interval=5000" in report
    assert "integration_timestep=automatic_explicit_stability_limit" in report


def test_shell_export_rejects_empty_geometry(tmp_path: Path) -> None:
    component = _component("empty")
    component.triangles = []
    with pytest.raises(OpenRadiossError, match="no non-degenerate triangles"):
        write_openradioss_deck([component], tmp_path, duration_s=0.01, animation_interval_s=0.002)


def test_shell_export_discards_only_zero_area_facets(tmp_path: Path) -> None:
    component = _component("plate")
    component.triangles.append(
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.0, 0.0))
    )
    starter, _engine = write_openradioss_deck(
        [component], tmp_path, duration_s=0.01, animation_interval_s=0.002
    )
    shell_lines = starter.read_text().split("/SH3N/1\n", 1)[1].split("/PROP/SHELL/1", 1)[0]
    assert len([line for line in shell_lines.splitlines() if line.strip()]) == 2
    assert "discarded_zero_area_facets=1" in (tmp_path / "openradioss_export_report.txt").read_text()


def test_cli_defaults_to_openradioss(monkeypatch: pytest.MonkeyPatch) -> None:
    from cfd_motion.cli import _parse_args

    monkeypatch.setattr(sys, "argv", ["cfd_motion", "part1.stl", "part2.stl"])
    assert _parse_args().structural_solver == "openradioss"


def test_runtime_image_is_built_only_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if command[1:3] == ["image", "inspect"] else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _ensure_runtime_image() == "cfd-motion-openradioss-runtime:22.04"
    assert calls[0][:3] == ["docker", "image", "inspect"]
    assert calls[1][0:2] == ["docker", "build"]


def test_case_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    with exclusive_case_lock(tmp_path):
        with pytest.raises(OpenRadiossError, match="already using"):
            with exclusive_case_lock(tmp_path):
                pytest.fail("a second writer acquired the same case lock")


def test_partial_result_updater_converts_only_stable_animation_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    animation = tmp_path / "assemblyA001"
    animation.write_bytes(b"complete animation")
    output_dir = tmp_path / "partial"

    def fake_convert(case_dir: Path, destination: Path, source: Path) -> Path:
        assert case_dir == tmp_path
        destination.mkdir(parents=True, exist_ok=True)
        converted = destination / "assembly_001.vtk"
        converted.write_text("# vtk DataFile Version 3.0\n")
        return converted

    monkeypatch.setattr("cfd_motion.openradioss._convert_animation_to_vtk", fake_convert)

    def fake_vtu_convert(vtk_path: Path) -> Path:
        converted = vtk_path.with_suffix(".vtu")
        converted.write_text("<VTKFile type=\"UnstructuredGrid\"/>\n")
        return converted

    monkeypatch.setattr("cfd_motion.openradioss._convert_vtk_to_vtu", fake_vtu_convert)
    update = partial_result_updater(tmp_path, output_dir, "assembly", 0.0012)
    assert update(False) == ()
    assert update(False) == (output_dir / "assembly_001.vtk",)
    series = (output_dir / "openradioss_partial.vtk.series").read_text()
    assert '"name": "assembly_001.vtk"' in series
    assert '"time": 0.0' in series
    pvd = (output_dir / "case.pvd").read_text()
    assert '<DataSet timestep="0" file="assembly_001.vtu"/>' in pvd


def test_animation_interval_is_read_from_engine_deck(tmp_path: Path) -> None:
    _starter, engine = write_openradioss_deck(
        [_component("plate")],
        tmp_path,
        duration_s=0.012,
        animation_interval_s=0.0012,
    )
    assert _animation_interval_from_deck(engine) == pytest.approx(0.0012)


def test_engine_validation_rejects_zero_exit_physics_termination(tmp_path: Path) -> None:
    output = tmp_path / "assembly_0001.out"
    output.write_text(
        "** RUN KILLED: ENERGY ERROR LIMIT REACHED\n"
        "NORMAL TERMINATION\n"
        "USER BREAK\n"
    )
    with pytest.raises(OpenRadiossError, match="ENERGY ERROR LIMIT REACHED"):
        _validate_engine_output(tmp_path, "assembly")


def test_engine_validation_accepts_completed_run(tmp_path: Path) -> None:
    (tmp_path / "assembly_0001.out").write_text(
        "NORMAL TERMINATION\nTOTAL NUMBER OF CYCLES : 100\n"
    )
    _validate_engine_output(tmp_path, "assembly")
