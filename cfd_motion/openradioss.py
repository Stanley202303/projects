"""OpenRadioss explicit-dynamics bridge.

The bridge deliberately exports a shell model from the existing triangulated
component surfaces.  It is appropriate for thin parts only.  A closed, thick
CAD solid needs a volume mesher and solid elements; treating it as a shell
would produce an untrustworthy result, so this module records that limitation
in the deck report instead of pretending otherwise.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .motion import inferred_deformation_thickness, inferred_deformation_young_modulus
from .models import AeroComponent, Vec3


class OpenRadiossError(RuntimeError):
    """Raised when the external explicit-dynamics solver cannot be run."""


@contextmanager
def exclusive_case_lock(case_dir: Path):
    """Prevent concurrent solver runs from writing the same case directory."""
    case_dir.mkdir(parents=True, exist_ok=True)
    lock_path = case_dir / ".openradioss.lock"
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OpenRadiossError(
                f"Another OpenRadioss run is already using {case_dir}. "
                "Wait for it to finish or stop it before starting another run."
            ) from exc
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class OpenRadiossRun:
    case_dir: Path
    starter_deck: Path
    engine_deck: Path
    starter_log: Path
    engine_log: Path
    animation_files: Tuple[Path, ...]
    partial_vtk_files: Tuple[Path, ...] = ()
    paraview_series: Path | None = None


def _solver_root() -> Path:
    configured = os.environ.get("OPENRADIOSS_ROOT", "OpenRadioss-main").strip()
    return Path(configured).expanduser().resolve()


def _require_solver_root() -> Path:
    root = _solver_root()
    if not (root / "exec" / "starter_linuxa64_gf").is_file():
        raise OpenRadiossError(
            f"OpenRadioss starter is not available at {root / 'exec' / 'starter_linuxa64_gf'}. "
            "Build OpenRadioss first or set OPENRADIOSS_ROOT."
        )
    if not (root / "exec" / "engine_linuxa64_gf").is_file():
        raise OpenRadiossError(
            f"OpenRadioss engine is not available at {root / 'exec' / 'engine_linuxa64_gf'}."
        )
    return root


def _runtime_image_name() -> str:
    return os.environ.get(
        "OPENRADIOSS_RUNTIME_IMAGE", "cfd-motion-openradioss-runtime:22.04"
    ).strip()


def _ensure_runtime_image() -> str:
    """Build the small runtime image once instead of installing on every run."""
    image = _runtime_image_name()
    inspection = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspection.returncode == 0:
        return image
    dockerfile = Path(__file__).resolve().parents[1] / "docker" / "openradioss-runtime.Dockerfile"
    if not dockerfile.is_file():
        raise OpenRadiossError(f"OpenRadioss runtime Dockerfile is missing: {dockerfile}")
    print(f"Building one-time OpenRadioss runtime image {image}...", flush=True)
    try:
        subprocess.run(
            [
                "docker", "build", "--platform", "linux/arm64",
                "-f", str(dockerfile), "-t", image, str(dockerfile.parent),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise OpenRadiossError(
            f"Could not build the OpenRadioss runtime image {image}; Docker exited with {exc.returncode}."
        ) from exc
    return image


def _run_solver_phase(
    command: Sequence[str],
    log_path: Path,
    label: str,
    progress_callback=None,
    container_name: str | None = None,
) -> None:
    """Run one solver process with a quiet log and visible elapsed-time updates."""
    status_interval_s = max(
        float(os.environ.get("OPENRADIOSS_STATUS_INTERVAL_S", "15")), 1.0
    )
    started = time.monotonic()
    next_status = status_interval_s
    print(f"OpenRadioss {label} started; detailed output: {log_path}", flush=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            list(command),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while True:
                try:
                    return_code = process.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    if progress_callback is not None:
                        progress_callback(False)
                    elapsed = time.monotonic() - started
                    if elapsed >= next_status:
                        print(f"OpenRadioss {label} running: {elapsed:.0f}s elapsed.", flush=True)
                        next_status += status_interval_s
        except KeyboardInterrupt:
            if container_name:
                subprocess.run(
                    ["docker", "stop", "--time", "5", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if progress_callback is not None:
                progress_callback(True)
            raise
    if progress_callback is not None:
        progress_callback(True)
    elapsed = time.monotonic() - started
    if return_code != 0:
        raise OpenRadiossError(
            f"OpenRadioss {label} failed with exit status {return_code}; see {log_path}."
        )
    print(f"OpenRadioss {label} completed in {elapsed:.1f}s.", flush=True)


def _converter_path() -> Path:
    configured = os.environ.get("OPENRADIOSS_ANIM_TO_VTK", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / "OpenRadioss-Tools" / "exec" / "anim_to_vtk_linuxa64").resolve()


def _write_paraview_series(
    output_dir: Path,
    vtk_files: Sequence[Path],
    animation_interval_s: float,
) -> Path | None:
    if not vtk_files:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    series_path = output_dir / "openradioss_partial.vtk.series"
    payload = {
        "file-series-version": "1.0",
        "files": [
            {"name": path.name, "time": index * animation_interval_s}
            for index, path in enumerate(vtk_files)
        ],
    }
    temporary_path = series_path.with_suffix(series_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary_path, series_path)
    return series_path


def _convert_animation_to_vtk(
    case_dir: Path,
    output_dir: Path,
    animation_file: Path,
) -> Path:
    converter = _converter_path()
    if not converter.is_file():
        raise OpenRadiossError(
            "OpenRadioss animation converter is missing. Expected "
            f"{converter}; set OPENRADIOSS_ANIM_TO_VTK to override it."
        )
    suffix = animation_file.name.rsplit("A", 1)[-1]
    root_name = animation_file.name[: -len(suffix) - 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_path = output_dir / f"{root_name}_{suffix}.vtk"
    temporary_path = vtk_path.with_suffix(".vtk.tmp")
    command = [
        "docker", "run", "--rm", "--platform", "linux/arm64",
        "-v", f"{converter.parent}:/converter:ro",
        "-v", f"{case_dir.resolve()}:/work",
        "-w", "/work",
        _runtime_image_name(),
        f"/converter/{converter.name}",
        animation_file.name,
    ]
    with temporary_path.open("w") as output:
        completed = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        raise OpenRadiossError(
            f"Could not convert partial result {animation_file.name}: "
            f"{completed.stderr.strip() or f'exit status {completed.returncode}'}"
        )
    with temporary_path.open(errors="replace") as converted:
        header = converted.readline().strip()
    if not header.startswith("# vtk DataFile"):
        temporary_path.unlink(missing_ok=True)
        raise OpenRadiossError(
            f"Converter produced an invalid VTK file for {animation_file.name}."
        )
    os.replace(temporary_path, vtk_path)
    return vtk_path


def partial_result_updater(
    case_dir: Path,
    output_dir: Path,
    root_name: str,
    animation_interval_s: float,
    not_before_ns: int = 0,
):
    """Return a callback that converts completed animation files incrementally."""
    observed_sizes: dict[Path, int] = {}
    converted: dict[Path, Path] = {}

    def update(force: bool = False) -> Tuple[Path, ...]:
        for animation_file in sorted(case_dir.glob(f"{root_name}A[0-9]*")):
            if animation_file in converted:
                continue
            stat = animation_file.stat()
            if stat.st_mtime_ns < not_before_ns:
                continue
            size = stat.st_size
            previous_size = observed_sizes.get(animation_file)
            observed_sizes[animation_file] = size
            if size <= 0 or (not force and previous_size != size):
                continue
            try:
                converted[animation_file] = _convert_animation_to_vtk(
                    case_dir, output_dir, animation_file
                )
            except OpenRadiossError as exc:
                if force:
                    print(f"WARNING: {exc}", flush=True)
                continue
            vtk_files = tuple(converted[path] for path in sorted(converted))
            series = _write_paraview_series(
                output_dir, vtk_files, animation_interval_s
            )
            print(
                f"Partial OpenRadioss result updated: {series} "
                f"({len(vtk_files)} frame(s)).",
                flush=True,
            )
        return tuple(converted[path] for path in sorted(converted))

    return update


def _animation_interval_from_deck(engine_deck: Path) -> float:
    lines = engine_deck.read_text().splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip().upper() != "/ANIM/DT":
            continue
        values = lines[index + 1].split()
        if len(values) >= 2:
            return float(values[1])
    raise OpenRadiossError(f"Could not read /ANIM/DT from {engine_deck}.")


def _point_key(point: Vec3, tolerance_m: float = 1e-9) -> Tuple[int, int, int]:
    return tuple(round(value / tolerance_m) for value in point)  # type: ignore[return-value]


def _component_nodes(
    component: AeroComponent,
) -> Tuple[List[Vec3], List[Tuple[int, int, int]], int]:
    """Return a deduplicated mesh and count zero-area source facets.

    A zero-area STL facet does not represent material or mass and cannot be a
    finite element.  OpenRadioss rejects an entire contact surface when even
    one such facet is present, so these invalid records must not become SH3N
    elements.
    """
    point_ids: dict[Tuple[int, int, int], int] = {}
    points: List[Vec3] = []
    faces: List[Tuple[int, int, int]] = []
    zero_area_facets = 0
    for _normal, first, second, third in component.triangles:
        ids: List[int] = []
        for point in (first, second, third):
            key = _point_key(point)
            node_id = point_ids.get(key)
            if node_id is None:
                node_id = len(points) + 1
                point_ids[key] = node_id
                points.append(point)
            ids.append(node_id)
        edge_ab = tuple(second[axis] - first[axis] for axis in range(3))
        edge_ac = tuple(third[axis] - first[axis] for axis in range(3))
        cross = (
            edge_ab[1] * edge_ac[2] - edge_ab[2] * edge_ac[1],
            edge_ab[2] * edge_ac[0] - edge_ab[0] * edge_ac[2],
            edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0],
        )
        cross_squared = sum(value * value for value in cross)
        edge_scale_squared = max(
            sum(value * value for value in edge_ab),
            sum(value * value for value in edge_ac),
            1e-30,
        )
        if len(set(ids)) < 3 or cross_squared <= edge_scale_squared * edge_scale_squared * 1e-24:
            zero_area_facets += 1
            continue
        faces.append((ids[0], ids[1], ids[2]))
    if not faces:
        raise OpenRadiossError(f"{component.patch} has no non-degenerate triangles for a shell export.")
    return points, faces, zero_area_facets


def _radioss_name(text: str) -> str:
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in text)[:70]


def _write_group_nodes(lines: List[str], group_id: int, node_ids: Iterable[int], label: str) -> None:
    lines.extend((f"/GRNOD/NODE/{group_id}", _radioss_name(label)))
    ids = list(node_ids)
    for start in range(0, len(ids), 8):
        lines.append("".join(f"{node_id:10d}" for node_id in ids[start:start + 8]))


def write_openradioss_deck(
    components: Sequence[AeroComponent],
    case_dir: Path,
    duration_s: float,
    animation_interval_s: float,
    contact_friction: float = 0.15,
    print_cycle_interval: int = 5000,
) -> Tuple[Path, Path]:
    """Write SI-unit shell decks and return (starter, engine) paths.

    The mesh is preserved exactly: no triangles or parts are removed.  One
    material, property, part, and velocity group are emitted per component.
    """
    if not components:
        raise OpenRadiossError("Cannot export an empty OpenRadioss model.")
    if duration_s <= 0.0 or animation_interval_s <= 0.0:
        raise OpenRadiossError("OpenRadioss duration and animation interval must be positive.")
    if print_cycle_interval <= 0:
        raise OpenRadiossError("OpenRadioss print cycle interval must be positive.")

    case_dir.mkdir(parents=True, exist_ok=True)
    root_name = "assembly"
    starter_path = case_dir / f"{root_name}_0000.rad"
    engine_path = case_dir / f"{root_name}_0001.rad"
    lines = [
        "#RADIOSS STARTER",
        "/BEGIN",
        root_name,
        "      2021         0",
        "                  kg                   m                   s",
        "                  kg                   m                   s",
        "/TITLE",
        "CFD motion OpenRadioss shell export",
        "/ANALY",
        "         0                   0         0",
        "/DEF_SHELL",
        "         0         0         1         1         1                             1         0",
        "/IOFLAG",
        "         1                   0         0         0         0         0",
        "/NODE",
    ]

    component_meshes = [_component_nodes(component) for component in components]
    node_offset = 0
    element_offset = 0
    component_node_ids: List[List[int]] = []
    component_element_ids: List[List[int]] = []
    for component, (points, faces, _zero_area_facets) in zip(components, component_meshes):
        node_ids = list(range(node_offset + 1, node_offset + len(points) + 1))
        component_node_ids.append(node_ids)
        for node_id, point in zip(node_ids, points):
            lines.append(f"{node_id:10d}{point[0]:20.12E}{point[1]:20.12E}{point[2]:20.12E}")
        element_ids = list(range(element_offset + 1, element_offset + len(faces) + 1))
        component_element_ids.append(element_ids)
        node_offset += len(points)
        element_offset += len(faces)

    for index, component in enumerate(components, start=1):
        density = max(component.material.density_kg_m3, 1e-9)
        young_modulus = max(inferred_deformation_young_modulus(component), 1.0)
        poisson_ratio = component.material.poisson_ratio
        if poisson_ratio is None:
            poisson_ratio = 0.3
        poisson_ratio = min(max(poisson_ratio, 0.0), 0.499)
        thickness = max(inferred_deformation_thickness(component), 1e-6)
        lines.extend((
            f"/MAT/ELAST/{index}",
            _radioss_name(component.material.material_name or component.name),
            f"{density:20.12E}{0.0:20.12E}",
            f"{young_modulus:20.12E}{poisson_ratio:20.12E}",
            f"/PART/{index}",
            _radioss_name(component.name),
            f"{index:10d}{index:10d}{0:10d}",
            f"/SH3N/{index}",
        ))
        for element_id, face in zip(component_element_ids[index - 1], component_meshes[index - 1][1]):
            a, b, c = (node + sum(len(mesh[0]) for mesh in component_meshes[:index - 1]) for node in face)
            lines.append(f"{element_id:10d}{a:10d}{b:10d}{c:10d}")
        lines.extend((
            f"/PROP/SHELL/{index}",
            _radioss_name(component.name),
            "         4         0         1         0                                       0",
            "              0.01              0.01              0.01                 0                 0",
            f"         5         1{thickness:20.12E}{(5.0 / 6.0):20.12E}{1:10d}{0:10d}",
        ))

    next_group_id = 1
    velocity_groups: List[Tuple[int, AeroComponent]] = []
    for component, node_ids in zip(components, component_node_ids):
        _write_group_nodes(lines, next_group_id, node_ids, component.name)
        velocity_groups.append((next_group_id, component))
        next_group_id += 1
    for group_id, component in velocity_groups:
        velocity = component.linear_velocity
        lines.extend((
            f"/INIVEL/TRA/{group_id}",
            _radioss_name(f"velocity_{component.name}"),
            f"{velocity[0]:20.12E}{velocity[1]:20.12E}{velocity[2]:20.12E}{group_id:10d}{0:10d}",
        ))

    # Classic Type-7 penalty contact is robust for general explicit shell contact.
    for secondary_index in range(1, len(components) + 1):
        for main_index in range(secondary_index + 1, len(components) + 1):
            interface_id = (secondary_index - 1) * len(components) + main_index
            secondary_group = next_group_id
            main_surface = next_group_id + 1
            lines.extend((
                f"/INTER/TYPE7/{interface_id}",
                _radioss_name(f"contact_{components[secondary_index - 1].name}_{components[main_index - 1].name}"),
                f"{secondary_group:10d}{main_surface:10d}{0:10d}{0:10d}{0:10d}{0:20d}{0:10d}{0:10d}{0:10d}",
                f"{0.0:20.12E}{0.0:20.12E}{1.0:20.12E}{0:30d}",
                "                   0                   0                   0                   0         0         0",
                f"                   0{contact_friction:20.12E}{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}",
                f"       000{'':20s}{5:10d}{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}",
                "         0         0                   0         0         0         0                   0         0",
                f"/GRNOD/PART/{secondary_group}",
                _radioss_name(f"secondary_{components[secondary_index - 1].name}"),
                f"{secondary_index:10d}",
                f"/SURF/PART/{main_surface}",
                _radioss_name(f"main_{components[main_index - 1].name}"),
                f"{main_index:10d}",
            ))
            next_group_id += 2
    lines.append("/END")
    starter_path.write_text("\n".join(lines) + "\n")
    engine_path.write_text("\n".join((
        "/VERS/2021",
        f"/RUN/{root_name}/1/",
        f"{duration_s:.15E}",
        "/ANIM/DT",
        f"{0.0:.15E} {animation_interval_s:.15E}",
        "/ANIM/NODA/VEL",
        "/ANIM/ELEM/VONM",
        "/DT/NODA/CST/0",
        "0.90 0.0",
        f"/PRINT/-{print_cycle_interval}",
        "/END",
    )) + "\n")
    report = case_dir / "openradioss_export_report.txt"
    report.write_text(
        "OpenRadioss shell export\n"
        "units=kg,m,s\n"
        "model=shell-only; use a volume-mesh exporter for thick solids\n"
        f"requested_duration_s={duration_s:.15g}\n"
        f"animation_interval_s={animation_interval_s:.15g}\n"
        f"print_cycle_interval={print_cycle_interval}\n"
        "integration_timestep=automatic_explicit_stability_limit\n"
        f"components={len(components)}\n"
        f"nodes={node_offset}\n"
        f"triangular_shell_elements={element_offset}\n"
        f"discarded_zero_area_facets={sum(mesh[2] for mesh in component_meshes)}\n"
        f"pairwise_type7_contacts={len(components) * (len(components) - 1) // 2}\n"
    )
    return starter_path, engine_path


def run_openradioss(
    case_dir: Path,
    starter_deck: Path,
    engine_deck: Path,
    threads: int = 2,
) -> OpenRadiossRun:
    """Run the Docker-hosted Linux ARM64 build and return retained animations."""
    solver_root = _require_solver_root()
    if shutil.which("docker") is None:
        raise OpenRadiossError("Docker is required to run the Linux OpenRadioss build.")
    runtime_image = _ensure_runtime_image()
    threads = max(int(threads), 1)
    root_name = starter_deck.stem.rsplit("_", 1)[0]
    run_token = f"{os.getpid()}_{time.time_ns()}"
    docker_command = [
        "docker", "run", "--rm", "--platform", "linux/arm64",
        "-v", f"{solver_root}:/solver:ro",
        "-v", f"{case_dir.resolve()}:/work",
        "-w", "/work",
        "-e", "LD_LIBRARY_PATH=/solver/extlib/hm_reader/linuxa64",
        "-e", "RAD_CFG_PATH=/solver/hm_cfg_files",
        "-e", "OMP_STACKSIZE=400m",
        runtime_image,
    ]
    starter_log = case_dir / "starter.log"
    engine_log = case_dir / "engine.log"
    _run_solver_phase(
        docker_command[:2] + ["--name", f"cfd_motion_radioss_starter_{run_token}"] + docker_command[2:] + [
            "/solver/exec/starter_linuxa64_gf", "-i", starter_deck.name,
            "-nt", str(threads),
        ],
        starter_log,
        "starter",
        container_name=f"cfd_motion_radioss_starter_{run_token}",
    )
    animation_interval_s = _animation_interval_from_deck(engine_deck)
    partial_output_dir = case_dir / "partial_results" / run_token
    engine_started_ns = time.time_ns()
    update_partial_results = partial_result_updater(
        case_dir,
        partial_output_dir,
        root_name,
        animation_interval_s,
        not_before_ns=engine_started_ns,
    )
    _run_solver_phase(
        docker_command[:2] + ["--name", f"cfd_motion_radioss_engine_{run_token}"] + docker_command[2:] + [
            "/solver/exec/engine_linuxa64_gf", "-i", engine_deck.name,
            "-nt", str(threads),
        ],
        engine_log,
        "engine",
        progress_callback=update_partial_results,
        container_name=f"cfd_motion_radioss_engine_{run_token}",
    )
    partial_vtk_files = update_partial_results(True)
    paraview_series = _write_paraview_series(
        partial_output_dir,
        partial_vtk_files,
        animation_interval_s,
    )
    return OpenRadiossRun(
        case_dir=case_dir,
        starter_deck=starter_deck,
        engine_deck=engine_deck,
        starter_log=starter_log,
        engine_log=engine_log,
        animation_files=tuple(sorted(case_dir.glob(f"{root_name}A*"))),
        partial_vtk_files=partial_vtk_files,
        paraview_series=paraview_series,
    )
