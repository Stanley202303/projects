from pathlib import Path

import pytest

from cfd_motion.models import AeroComponent, MaterialProperties
from cfd_motion.openradioss import OpenRadiossError, write_openradioss_deck


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
    assert starter_text.count("\n/SHELL/") == 2
    assert starter_text.count("/MAT/ELAST/") == 2
    assert starter_text.count("/INTER/TYPE7/") == 1
    assert "7.850000000000000E+03" in starter_text
    assert "1.200000000000E+01" in starter_text
    assert "/ANIM/ELEM/VONM" in engine_text
    report = (tmp_path / "openradioss_export_report.txt").read_text()
    assert "triangular_shell_elements=4" in report


def test_shell_export_rejects_empty_geometry(tmp_path: Path) -> None:
    component = _component("empty")
    component.triangles = []
    with pytest.raises(OpenRadiossError, match="no non-degenerate triangles"):
        write_openradioss_deck([component], tmp_path, duration_s=0.01, animation_interval_s=0.002)
