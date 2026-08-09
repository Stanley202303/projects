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
        "MIN_COMPONENT_CELLS_ACROSS",
        "MAX_ADAPTIVE_SURFACE_REFINEMENT",
        "CFD_ALLOW_SKEW_ONLY_MESH_WARNING",
        "CFD_MAX_ACCEPTED_SKEWNESS",
        "CFD_MAX_ACCEPTED_SKEW_FACES",
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


def _box_triangles(xmin, xmax, ymin, ymax, zmin, zmax):
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
    return [
        ((0.0, 0.0, 0.0), vertices[a], vertices[b], vertices[c])
        for a, b, c in faces
    ]


def test_retained_cfd_patch_report_distinguishes_missing_report_and_occlusion(
    tmp_path: Path,
) -> None:
    from cfd_motion.runner import retained_cfd_patch_names

    assert retained_cfd_patch_names(tmp_path) is None
    (tmp_path / "retained_body_patches.txt").write_text(
        "retained_body_patches= Part_1_3 Part_1_1 Part_1_1_2\n"
        "occluded_body_patches= Part_1_2\n"
    )
    assert retained_cfd_patch_names(tmp_path) == {
        "Part_1_3",
        "Part_1_1",
        "Part_1_1_2",
    }


def test_small_body_gets_local_refinement_and_patch_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config, models, openfoam = _reload_openfoam_modules(monkeypatch)
    small = models.AeroComponent(
        name="small insert",
        patch="small_insert",
        triangles=_box_triangles(0.0, 0.032, -0.0025, 0.0025, -0.0025, 0.0025),
        cofr=(0.016, 0.0, 0.0),
        lref=0.032,
        aref=0.00016,
    )
    plate = models.AeroComponent(
        name="broad plate",
        patch="broad_plate",
        triangles=_box_triangles(0.1, 0.105, -0.4, 0.4, -0.4, 0.4),
        cofr=(0.1025, 0.0, 0.0),
        lref=0.8,
        aref=0.64,
    )

    levels = openfoam.adaptive_surface_refinement_levels(
        [small, plate],
        0.145,
    )

    assert levels["small_insert"] == (7, 7)
    assert levels["broad_plate"] == openfoam.SURFACE_REFINEMENT
    region_settings = openfoam.adaptive_region_refinement_settings(
        [small, plate],
        0.145,
        0.4,
        levels,
    )
    assert region_settings["small_insert"] == (0.01, 7)
    assert region_settings["broad_plate"] == (0.4, openfoam.REGION_REFINEMENT)

    case = tmp_path / "small_patch_case"
    openfoam.make_case_from_components([small, plate], case)
    snappy = (case / "system/snappyHexMeshDict").read_text()
    allrun = (case / "Allrun").read_text()
    report = (case / "mesh_resolution_report.txt").read_text()
    assert "small_insert\n        {\n            level (7 7);" in snappy
    assert "levels ((0.01 7));" in snappy
    assert 'REQUIRED_BODY_PATCHES="small_insert broad_plate"' in allrun
    assert "has no fluid-facing faces" in allrun
    assert 'functions/forces_$required_patch" -remove' in allrun
    assert "functions/forces_all/patches -set" in allrun
    assert "retained_body_patches.txt" in allrun
    assert "retained no fluid-facing body patches; CFD solve aborted" in allrun
    assert "checkMesh | tee log.checkMesh" in allrun
    assert "MAX_ACCEPTED_SKEWNESS" not in allrun
    assert "MAX_ACCEPTED_SKEW_FACES" not in allrun
    assert "continuing CFD solve without a skew-face limit" in allrun
    assert "found hard mesh failures; CFD solve aborted" in allrun
    assert "small_insert\t0.005\t7\t7" in report


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
