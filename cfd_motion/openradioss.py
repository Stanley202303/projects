"""OpenRadioss explicit-dynamics bridge.

The bridge deliberately exports a shell model from the existing triangulated
component surfaces.  It is appropriate for thin parts only.  A closed, thick
CAD solid needs a volume mesher and solid elements; treating it as a shell
would produce an untrustworthy result, so this module records that limitation
in the deck report instead of pretending otherwise.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .motion import inferred_deformation_thickness, inferred_deformation_young_modulus
from .models import AeroComponent, Vec3


class OpenRadiossError(RuntimeError):
    """Raised when the external explicit-dynamics solver cannot be run."""


@dataclass(frozen=True)
class OpenRadiossRun:
    case_dir: Path
    starter_deck: Path
    engine_deck: Path
    starter_log: Path
    engine_log: Path
    animation_files: Tuple[Path, ...]


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
) -> Tuple[Path, Path]:
    """Write SI-unit shell decks and return (starter, engine) paths.

    The mesh is preserved exactly: no triangles or parts are removed.  One
    material, property, part, and velocity group are emitted per component.
    """
    if not components:
        raise OpenRadiossError("Cannot export an empty OpenRadioss model.")
    if duration_s <= 0.0 or animation_interval_s <= 0.0:
        raise OpenRadiossError("OpenRadioss duration and animation interval must be positive.")

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
        "/ANIM/ELEM/VONM",
        "/DT/NODA/CST/0",
        "0.90 0.0",
        "/PRINT/-50",
        "/END",
    )) + "\n")
    report = case_dir / "openradioss_export_report.txt"
    report.write_text(
        "OpenRadioss shell export\n"
        "units=kg,m,s\n"
        "model=shell-only; use a volume-mesh exporter for thick solids\n"
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
    threads = max(int(threads), 1)
    root_name = starter_deck.stem.rsplit("_", 1)[0]
    command = [
        "docker", "run", "--rm", "--platform", "linux/arm64",
        "-v", f"{solver_root}:/solver:ro",
        "-v", f"{case_dir.resolve()}:/work",
        "-w", "/work", "ubuntu:22.04", "bash", "-lc",
        " ".join((
            "export DEBIAN_FRONTEND=noninteractive;",
            "apt-get update -qq;",
            "apt-get install -y -qq --no-install-recommends libgfortran5 libgomp1 libstdc++6 libgcc-s1 zlib1g >/dev/null;",
            "export LD_LIBRARY_PATH=/solver/extlib/hm_reader/linuxa64;",
            "export RAD_CFG_PATH=/solver/hm_cfg_files;",
            "export OMP_STACKSIZE=400m;",
            f"/solver/exec/starter_linuxa64_gf -i {starter_deck.name} -nt {threads} > starter.log 2>&1;",
            f"/solver/exec/engine_linuxa64_gf -i {engine_deck.name} -nt {threads} > engine.log 2>&1",
        )),
    ]
    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise OpenRadiossError(
            f"OpenRadioss failed with exit status {exc.returncode}; see {case_dir / 'starter.log'} and {case_dir / 'engine.log'}."
        ) from exc
    return OpenRadiossRun(
        case_dir=case_dir,
        starter_deck=starter_deck,
        engine_deck=engine_deck,
        starter_log=case_dir / "starter.log",
        engine_log=case_dir / "engine.log",
        animation_files=tuple(sorted(case_dir.glob(f"{root_name}A*"))),
    )
