import importlib
from pathlib import Path


def _reload_openfoam_modules(monkeypatch, **env):
    keys = {
        "CFD_SOLVER_MODE",
        "AERO_TRANSIENT_END_TIME",
        "AERO_TRANSIENT_DELTA_T",
        "AERO_TRANSIENT_WRITE_INTERVAL",
        "AERO_TRANSIENT_PURGE_WRITE",
        "AERO_TRANSIENT_MAX_CO",
        "AERO_TRANSIENT_MAX_DELTA_T",
        "AERO_TRANSIENT_OUTER_CORRECTORS",
        "AERO_TRANSIENT_PRESSURE_CORRECTORS",
        "AERO_TRANSIENT_NON_ORTHOGONAL_CORRECTORS",
        "AERO_TRANSIENT_MOMENTUM_PREDICTOR",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    import cfd_motion.config as config
    import cfd_motion.models as models
    import cfd_motion.openfoam as openfoam

    importlib.reload(config)
    importlib.reload(models)
    importlib.reload(openfoam)
    for key, value in env.items():
        typed_value = value
        if key == "CFD_SOLVER_MODE":
            typed_value = str(value)
        elif key == "AERO_TRANSIENT_MOMENTUM_PREDICTOR":
            typed_value = str(value).strip().lower() not in {"0", "false", "no", "off"}
        elif key in {
            "AERO_TRANSIENT_PURGE_WRITE",
            "AERO_TRANSIENT_OUTER_CORRECTORS",
            "AERO_TRANSIENT_PRESSURE_CORRECTORS",
            "AERO_TRANSIENT_NON_ORTHOGONAL_CORRECTORS",
        }:
            typed_value = int(value)
        else:
            typed_value = float(value)
        if hasattr(config, key):
            setattr(config, key, typed_value)
        if hasattr(models, key):
            setattr(models, key, typed_value)
        if hasattr(openfoam, key):
            setattr(openfoam, key, typed_value)
    return config, models, openfoam


def test_transient_mode_writes_pimplefoam_case(monkeypatch, tmp_path: Path) -> None:
    _config, models, openfoam = _reload_openfoam_modules(
        monkeypatch,
        CFD_SOLVER_MODE="transient",
        AERO_TRANSIENT_END_TIME="0.2",
        AERO_TRANSIENT_DELTA_T="0.005",
        AERO_TRANSIENT_WRITE_INTERVAL="0.02",
    )

    triangle = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    component = models.AeroComponent(
        name="plate",
        patch="plate",
        triangles=[triangle],
        cofr=(0.5, 0.5, 0.0),
        lref=1.0,
        aref=1.0,
    )

    case = tmp_path / "transient_case"
    openfoam.make_case_from_components([component], case)

    control_dict = (case / "system/controlDict").read_text()
    fv_schemes = (case / "system/fvSchemes").read_text()
    fv_solution = (case / "system/fvSolution").read_text()
    allrun = (case / "Allrun").read_text()

    assert "application     pimpleFoam;" in control_dict
    assert "adjustTimeStep  yes;" in control_dict
    assert "endTime         0.2;" in control_dict
    assert "default backward;" in fv_schemes
    assert "PIMPLE" in fv_solution
    assert "nOuterCorrectors 2;" in fv_solution
    assert "pimpleFoam | tee log.pimpleFoam" in allrun


def test_gap_report_is_written_for_transient_cases(monkeypatch, tmp_path: Path) -> None:
    _config, models, openfoam = _reload_openfoam_modules(
        monkeypatch,
        CFD_SOLVER_MODE="transient",
    )

    triangle = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    component = models.AeroComponent(
        name="flap",
        patch="flap",
        triangles=[triangle],
        cofr=(0.5, 0.5, 0.0),
        lref=1.0,
        aref=1.0,
    )
    component.freedom.rotate_axes = [(0.0, 0.0, 1.0)]

    case = tmp_path / "gap_case"
    openfoam.make_case_from_components([component], case)

    report = (case / "unsteady_fsi_gap_report.txt").read_text()
    assert "cfd_solver_mode=transient" in report
    assert "Mesh/body motion is not yet solved inside the OpenFOAM time loop" in report
    assert "There are no fluid-structure subiterations within each physical time step" in report
