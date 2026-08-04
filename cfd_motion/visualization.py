from __future__ import annotations

import base64
import email.utils
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import *
from .models import *
from .math_utils import *
from .geometry import *
from .openfoam import *
from .onshape import *
from .motion import *
from .structural import (
    ExplicitShellState,
    HybridShellCollisionState,
    shell_fragment_triangles,
    shell_fragment_velocity,
)


def visualization_components_with_fragments(
    components: Sequence[AeroComponent],
) -> List[AeroComponent]:
    """Add detached collision fragments as independent visual surfaces."""
    visual_components = list(components)
    for component in components:
        state = component.collision_structural_state
        if not isinstance(state, (ExplicitShellState, HybridShellCollisionState)):
            continue
        if isinstance(state, HybridShellCollisionState):
            visual_components.extend(
                fragment.component for fragment in state.fragment_bodies
            )
            continue
        triangles = shell_fragment_triangles(state)
        if not triangles:
            continue
        area, length, centroid = component_references(triangles)
        visual_components.append(
            AeroComponent(
                name=f"{component.name} detached fragments",
                patch=f"{component.patch}_fragments",
                triangles=triangles,
                cofr=centroid,
                lref=length,
                aref=area,
                material=component.material,
                mass=state.emitted_fragment_mass_kg,
                inertia=0.0,
                linear_velocity=shell_fragment_velocity(state),
                freedom=MotionFreedom(
                    translate_axes=[
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    ],
                    mate_type="COLLISION_FRAGMENT",
                    source="explicit-shell-fragment",
                ),
            )
        )
    return visual_components

def mirror_latest_step_to_root_case(latest_step_case: Path, root_case: Path) -> None:
    """Make root_case itself openable in ParaView by mirroring the latest solved step.

    Assembly motion runs store the real OpenFOAM cases in
    root_case/motion_steps/step_XXX.  ParaView expects the .foam file to live
    beside a valid constant/ directory, so a root-level case.foam is invalid
    unless we also mirror a solved case to the root.
    """
    for name in ("0", "constant", "system", "postProcessing", "VTK"):
        src = latest_step_case / name
        dst = root_case / name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    # Some OpenFOAM runs create numeric time directories such as 1, 2, 1000.
    for src in latest_step_case.iterdir():
        if not src.is_dir():
            continue
        try:
            float(src.name)
        except ValueError:
            continue
        dst = root_case / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    (root_case / "case.foam").write_text("")
    (root_case / "OPEN_THIS_IN_PARAVIEW.txt").write_text(
        "This root folder mirrors the latest solved assembly-motion step so that\n"
        "actual_model_case/case.foam opens directly in ParaView.\n\n"
        f"Mirrored from: {latest_step_case}\n\n"
        "You can also open individual solved steps directly, for example:\n"
        "motion_steps/step_000/case.foam\n"
    )





def _openfoam_float_re() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _read_scalar_internal_values(field_path: Path) -> List[float]:
    """Best-effort reader for OpenFOAM scalar internalField values.

    This is intentionally lightweight: it is used only for diagnostics and for
    creating derived visualisation fields. It handles the normal ASCII cases:
    `internalField uniform x;` and `internalField nonuniform List<scalar> ...`.
    """
    if not field_path.exists():
        return []
    text = field_path.read_text(errors="replace")
    start = text.find("internalField")
    if start < 0:
        return []
    end = text.find("boundaryField", start)
    section = text[start:] if end < 0 else text[start:end]
    num = _openfoam_float_re()
    m = re.search(r"internalField\s+uniform\s+(" + num + r")\s*;", section)
    if m:
        try:
            return [float(m.group(1))]
        except ValueError:
            return []
    # For a nonuniform list, avoid reading the list length by only scanning the
    # body between the first '(' and the final ')' before boundaryField.
    open_i = section.find("(")
    close_i = section.rfind(")")
    if open_i < 0 or close_i <= open_i:
        return []
    body = section[open_i + 1:close_i]
    values: List[float] = []
    for tok in re.findall(num, body):
        try:
            values.append(float(tok))
        except ValueError:
            pass
    return values


def _scale_internal_field_text(text: str, scale: float) -> str:
    """Scale only the internalField scalar values in an OpenFOAM field file."""
    start = text.find("internalField")
    if start < 0:
        return text
    end = text.find("boundaryField", start)
    if end < 0:
        end = len(text)
    before, section, after = text[:start], text[start:end], text[end:]
    num = _openfoam_float_re()

    def repl_uniform(match: re.Match[str]) -> str:
        return f"{match.group(1)}{float(match.group(2)) * scale:.10g}{match.group(3)}"

    new_section, n = re.subn(
        r"(internalField\s+uniform\s+)(" + num + r")(\s*;)",
        repl_uniform,
        section,
        count=1,
    )
    if n:
        return before + new_section + after

    open_i = section.find("(")
    close_i = section.rfind(")")
    if open_i < 0 or close_i <= open_i:
        return text
    prefix = section[:open_i + 1]
    body = section[open_i + 1:close_i]
    suffix = section[close_i:]

    def repl_num(match: re.Match[str]) -> str:
        return f"{float(match.group(0)) * scale:.10g}"

    body = re.sub(num, repl_num, body)
    return before + prefix + body + suffix + after


def _replace_foam_object_and_dimensions(text: str, object_name: str, dimensions: str) -> str:
    text = re.sub(r"(object\s+)\w+(\s*;)", rf"\g<1>{object_name}\2", text, count=1)
    text = re.sub(r"dimensions\s+\[[^\]]+\]\s*;", f"dimensions      {dimensions};", text, count=1)
    return text


def write_derived_pressure_fields(case: Path) -> None:
    """Create pPa and Cp fields from OpenFOAM's incompressible kinematic p.

    In simpleFoam, p is usually kinematic pressure, p/rho, with dimensions
    m^2/s^2. ParaView users often expect pressure in pascals or coefficient of
    pressure. These derived fields make the colouring meaningful:
      pPa = rho * p
      Cp  = p / (0.5 * U_inf^2)
    """
    latest = _latest_solver_time_dir(case)
    if latest is None:
        return
    p_src = latest / "p"
    if not p_src.exists():
        return
    text = p_src.read_text(errors="replace")
    ppa = _replace_foam_object_and_dimensions(_scale_internal_field_text(text, RHO), "pPa", "[1 -1 -2 0 0 0 0]")
    cp_scale = 1.0 / max(0.5 * abs(VELOCITY) ** 2, 1e-30)
    cp = _replace_foam_object_and_dimensions(_scale_internal_field_text(text, cp_scale), "Cp", "[0 0 0 0 0 0 0]")
    (latest / "pPa").write_text(ppa)
    (latest / "Cp").write_text(cp)


def write_pressure_range_report(case: Path) -> None:
    if not PARAVIEW_PRESSURE_DIAGNOSTICS:
        return
    latest = _latest_solver_time_dir(case)
    lines = [
        "# Pressure diagnostic report",
        f"case={case}",
        f"latest_solver_time={latest.name if latest else '<none>'}",
        "# In incompressible OpenFOAM simpleFoam, p is kinematic pressure p/rho, not Pa.",
        "# v16 also writes pPa=rho*p and Cp=p/(0.5*Uinf^2) for ParaView colouring.",
        "",
        "field\tmin\tmax\trange",
    ]
    for field in ["p", "pPa", "Cp"]:
        path = latest / field if latest else None
        values = _read_scalar_internal_values(path) if path else []
        if values:
            mn, mx = min(values), max(values)
            lines.append(f"{field}\t{mn:.10g}\t{mx:.10g}\t{(mx-mn):.10g}")
        else:
            lines.append(f"{field}\t<no values>\t<no values>\t<no values>")
    lines.extend([
        "",
        "If p/pPa/Cp ranges are near zero, then ParaView will show a single colour.",
        "That usually means one of these is true:",
        "  1) the selected time is the initial/unsolved field,",
        "  2) the solver did not converge enough to create a pressure gradient,",
        "  3) the object patches are not part of the solved fluid mesh,",
        "  4) you are viewing a stitched OpenFOAM case where remeshed times confuse ParaView.",
        "  5) the inlet/outlet are reversed for the sign of VELOCITY; v18 fixes this automatically.",
        "For remeshed moving geometry, open paraview_motion_timeseries.pvd for the most reliable view.",
    ])
    (case / PRESSURE_RANGE_REPORT_NAME).write_text("\n".join(lines) + "\n")


def _vtk_files_for_step(step_case: Path) -> List[Path]:
    files: List[Path] = []

    # Prefer sampled wall surfaces: they contain interpolated solved field values
    # on the actual moving part surfaces, so p/pPa/Cp/U are much more meaningful
    # than raw wall-function boundary values.
    sampled_files: List[Path] = []
    post_root = step_case / "postProcessing"
    if post_root.exists():
        # Collect both function-object sampling and classic sample-utility output.
        sampled_files = [p for p in post_root.rglob("*.vtk") if p.is_file()]

    vtk_dir = step_case / "VTK"
    volume_files = [p for p in vtk_dir.rglob("*.vtk") if p.is_file()] if vtk_dir.exists() else []
    if PARAVIEW_EXCLUDE_INITIAL_VTK:
        volume_files = [p for p in volume_files if "/0/" not in ("/" + str(p).replace(os.sep, "/") + "/")]
    if not PARAVIEW_PVD_INCLUDE_BOUNDARY:
        volume_files = [p for p in volume_files if "boundary" not in {part.lower() for part in p.parts}]

    if PARAVIEW_PVD_PREFER_SAMPLED_SURFACES and sampled_files:
        sampled_files.sort(key=lambda p: str(p))
        files.extend(sampled_files)
        # Keep one volume file too, so you can still switch to volume/wake data.
        volume_files.sort(key=lambda p: (-p.stat().st_size, str(p)))
        files.extend(volume_files[:1])
    else:
        volume_files.sort(key=lambda p: (-p.stat().st_size, str(p)))
        files.extend(volume_files)
        sampled_files.sort(key=lambda p: str(p))
        files.extend(sampled_files)
    return files


def _xml_attr(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def validate_preview_polydata(path: Path) -> None:
    """Validate the restricted VTP dialect used by the animation PVD files.

    Preview vertices are deliberately duplicated for every triangle.  Fields
    are therefore stored as PointData with each face value repeated at its
    three vertices.  This avoids a vtkXMLPolyDataReader CellData tuple-count
    failure seen in time-varying collision previews while retaining exact
    face-wise colouring.
    """
    root = ElementTree.parse(path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"Preview VTP has no PolyData Piece: {path}")
    point_count = int(piece.attrib.get("NumberOfPoints", "-1"))
    polygon_count = int(piece.attrib.get("NumberOfPolys", "-1"))
    if point_count < 0 or polygon_count < 0:
        raise ValueError(f"Preview VTP has invalid geometry counts: {path}")

    point_data = piece.find("PointData")
    if point_data is None:
        raise ValueError(f"Preview VTP has no PointData: {path}")
    for array in point_data.findall("DataArray"):
        components = int(array.attrib.get("NumberOfComponents", "1"))
        if len((array.text or "").split()) != point_count * components:
            name = array.attrib.get("Name", "unnamed")
            raise ValueError(f"Preview VTP PointData array {name!r} has the wrong length: {path}")

    points_array = piece.find("Points/DataArray")
    if points_array is None or len((points_array.text or "").split()) != 3 * point_count:
        raise ValueError(f"Preview VTP has an incomplete point coordinate array: {path}")

    connectivity = piece.find("Polys/DataArray[@Name='connectivity']")
    offsets = piece.find("Polys/DataArray[@Name='offsets']")
    if connectivity is None or offsets is None:
        raise ValueError(f"Preview VTP has incomplete polygon connectivity: {path}")
    if len((connectivity.text or "").split()) != 3 * polygon_count:
        raise ValueError(f"Preview VTP has an incomplete connectivity array: {path}")
    if len((offsets.text or "").split()) != polygon_count:
        raise ValueError(f"Preview VTP has an incomplete polygon offsets array: {path}")

    cell_data = piece.find("CellData")
    if cell_data is None:
        raise ValueError(f"Preview VTP has no CellData: {path}")
    for array in cell_data.findall("DataArray"):
        raise ValueError(
            f"Preview VTP must not contain CellData array {array.attrib.get('Name', 'unnamed')!r}: {path}"
        )


def _write_ascii_polydata_vtk(
    path: Path,
    points: List[Vec3],
    triangles: List[Tuple[int, int, int]],
    cell_scalars: Dict[str, List[float]],
    cell_vectors: Optional[Dict[str, List[Vec3]]] = None,
) -> None:
    """Write XML VTK PolyData (.vtp) for ParaView PVD time series.

    Important v21 fix: ParaView's .pvd reader is vtkXMLCollectionReader.  It
    expects the referenced datasets to be XML VTK types such as .vtp or .vtu.
    Earlier versions referenced legacy ASCII .vtk POLYDATA files, which ParaView
    can often open directly, but which the PVD reader can reject with:

        Could not determine the data type for the first dataset

    This function keeps the old name so the rest of the script needs minimal
    changes, but it now writes a proper .vtp XML PolyData file.  The preview
    fields are face quantities.  The preview deliberately duplicates each
    triangle's vertices, so face values can safely be written as three equal
    PointData tuples.  This avoids the VTK CellData reader failure that occurs
    when changing collision topology between animation frames.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".vtp":
        path = path.with_suffix(".vtp")

    # A point attribute represents one value per vertex.  Panel/CFD inputs can
    # share vertices while carrying one pressure value per face, so normalise to
    # independent triangle vertices before repeating each face value three
    # times.  This makes the association unambiguous for every VTK reader.
    needs_triangle_vertices = (
        len(points) != 3 * len(triangles)
        or any(triangle != (3 * index, 3 * index + 1, 3 * index + 2)
               for index, triangle in enumerate(triangles))
    )
    if needs_triangle_vertices:
        source_points = points
        expanded_points: List[Vec3] = []
        expanded_triangles: List[Tuple[int, int, int]] = []
        for triangle in triangles:
            if len(triangle) != 3 or any(index < 0 or index >= len(source_points) for index in triangle):
                raise ValueError(f"Invalid PolyData triangle index while writing {path}")
            start = len(expanded_points)
            expanded_points.extend(source_points[index] for index in triangle)
            expanded_triangles.append((start, start + 1, start + 2))
        points = expanded_points
        triangles = expanded_triangles

    valid_scalars: Dict[str, List[float]] = {}
    for name, vals in cell_scalars.items():
        if name == "CpPanel":
            continue
        if len(vals) != len(triangles):
            continue
        cleaned: List[float] = []
        for v in vals:
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            if not math.isfinite(fv):
                fv = 0.0
            cleaned.append(fv)
        valid_scalars[name] = cleaned

    def clean_vec3(value: Vec3) -> Vec3:
        cleaned: List[float] = []
        for component in value[:3]:
            try:
                fv = float(component)
            except Exception:
                fv = 0.0
            if not math.isfinite(fv):
                fv = 0.0
            cleaned.append(fv)
        while len(cleaned) < 3:
            cleaned.append(0.0)
        return (cleaned[0], cleaned[1], cleaned[2])

    valid_vectors: Dict[str, List[Vec3]] = {}
    for name, vals in (cell_vectors or {}).items():
        if len(vals) != len(triangles):
            continue
        valid_vectors[name] = [clean_vec3(v) for v in vals]

    for triangle in triangles:
        if len(triangle) != 3 or any(index < 0 or index >= len(points) for index in triangle):
            raise ValueError(f"Invalid PolyData triangle index while writing {path}")

    connectivity: List[str] = []
    offsets: List[str] = []
    off = 0
    for a, b, c in triangles:
        connectivity.extend([str(int(a)), str(int(b)), str(int(c))])
        off += 3
        offsets.append(str(off))

    def values_text(vals: Sequence[float], per_line: int = 9) -> List[str]:
        out: List[str] = []
        row: List[str] = []
        for v in vals:
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            if not math.isfinite(fv):
                fv = 0.0
            row.append(f"{fv:.9g}")
            if len(row) >= per_line:
                out.append("          " + " ".join(row))
                row = []
        if row:
            out.append("          " + " ".join(row))
        return out

    def vector_values_text(vals: Sequence[Vec3], per_line: int = 3) -> List[str]:
        flat: List[float] = []
        for x, y, z in vals:
            flat.extend([x, y, z])
        return values_text(flat, per_line * 3)

    point_values: List[float] = []
    for x, y, z in points:
        point_values.extend([x, y, z])

    point_scalars = {
        name: [value for value in values for _vertex in range(3)]
        for name, values in valid_scalars.items()
    }
    point_vectors = {
        name: [value for value in values for _vertex in range(3)]
        for name, values in valid_vectors.items()
    }
    active_vectors = "U" if "U" in point_vectors else (next(iter(point_vectors)) if point_vectors else "")
    point_data_attrs = 'Scalars="pressureCoeff"'
    if active_vectors:
        escaped_vectors = _xml_attr(active_vectors)
        point_data_attrs += f' Vectors="{escaped_vectors}"'

    lines: List[str] = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
        '  <PolyData>',
        # vtkXMLPolyDataReader determines CellData tuple counts from every
        # PolyData cell family.  Declare the unused families explicitly;
        # omitting them can make VTK miscalculate point/cell tuple counts.
        f'    <Piece NumberOfPoints="{len(points)}" NumberOfVerts="0" '
        f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{len(triangles)}">',
    ]
    lines.append(f'      <PointData {point_data_attrs}>')
    for name, vals in point_scalars.items():
        escaped = _xml_attr(name)
        lines.append(f'        <DataArray type="Float64" Name="{escaped}" format="ascii">')
        lines.extend(values_text(vals))
        lines.append('        </DataArray>')
    for name, vals in point_vectors.items():
        escaped = _xml_attr(name)
        lines.append(f'        <DataArray type="Float64" Name="{escaped}" NumberOfComponents="3" format="ascii">')
        lines.extend(vector_values_text(vals))
        lines.append('        </DataArray>')
    lines.extend([
        '      </PointData>',
        '      <CellData/>',
        '      <Points>',
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">',
    ])
    lines.extend(values_text(point_values))
    lines.extend([
        '        </DataArray>',
        '      </Points>',
        '      <Polys>',
        '        <DataArray type="Int32" Name="connectivity" format="ascii">',
    ])

    # Connectivity/offsets are integers, so write directly.
    if connectivity:
        for i in range(0, len(connectivity), 18):
            lines.append("          " + " ".join(connectivity[i:i+18]))
    lines.extend([
        '        </DataArray>',
        '        <DataArray type="Int32" Name="offsets" format="ascii">',
    ])
    if offsets:
        for i in range(0, len(offsets), 18):
            lines.append("          " + " ".join(offsets[i:i+18]))
    lines.extend([
        '        </DataArray>',
        '      </Polys>',
        '    </Piece>',
        '  </PolyData>',
        '</VTKFile>',
        '',
    ])

    # ParaView can watch an already-open PVD while a simulation replaces frames.
    # Publish the complete XML in one atomic rename so it never observes a
    # partially copied PointData array.
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text("\n".join(lines))
    validate_preview_polydata(temporary_path)
    os.replace(temporary_path, path)


def connected_surface_body_ids(component: AeroComponent) -> List[int]:
    """Return a connected-surface ID for every triangle in a component."""
    triangles = component.triangles
    if not triangles:
        return []
    tolerance = max(1e-10, max(component.lref, 1e-6) * 1e-9)

    def vertex_key(point: Vec3) -> Tuple[int, int, int]:
        scaled = tuple(int(round(value / tolerance)) for value in point)
        return scaled[0], scaled[1], scaled[2]

    parents = list(range(len(triangles)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    edge_owner: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
    for triangle_index, (_normal, a, b, c) in enumerate(triangles):
        keys = (vertex_key(a), vertex_key(b), vertex_key(c))
        for first, second in (
            (keys[0], keys[1]),
            (keys[1], keys[2]),
            (keys[2], keys[0]),
        ):
            edge = tuple(sorted((first, second)))
            previous = edge_owner.setdefault(edge, triangle_index)
            union(triangle_index, previous)

    root_to_body: Dict[int, int] = {}
    body_ids: List[int] = []
    for triangle_index in range(len(triangles)):
        root = find(triangle_index)
        if root not in root_to_body:
            root_to_body[root] = len(root_to_body)
        body_ids.append(root_to_body[root])
    return body_ids


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} TB"


def enforce_case_storage_budget(root_case: Path, context: str) -> None:
    """Keep actual_model_case under the configured storage budget.

    This deliberately targets the user-visible case directory, not the temporary
    working case used during one CFD step.  The heavy root OpenFOAM time-series,
    VTK, and postProcessing folders are deleted first because the compact PVD
    animation does not need them.
    """
    if CASE_STORAGE_LIMIT_BYTES <= 0 or not root_case.exists():
        return

    def size() -> int:
        return directory_size_bytes(root_case)

    current = size()
    if current <= CASE_STORAGE_LIMIT_BYTES:
        return

    cleanup_targets = [
        root_case / "VTK",
        root_case / "postProcessing",
        root_case / "motion_steps",
    ]
    # Root numeric OpenFOAM time dirs are very storage-heavy and unnecessary in
    # compact-PVD mode.
    for child in list(root_case.iterdir()):
        if child.is_dir() and _is_openfoam_numeric_time_name(child.name):
            cleanup_targets.append(child)

    for target in cleanup_targets:
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink(missing_ok=True)
            current = size()
            if current <= CASE_STORAGE_LIMIT_BYTES:
                break

    report = root_case / "storage_budget_report.txt"
    report.write_text(
        f"context={context}\n"
        f"case_dir={root_case}\n"
        f"limit={human_bytes(CASE_STORAGE_LIMIT_BYTES)}\n"
        f"current={human_bytes(current)}\n"
        f"root_openfoam_timeseries={int(ROOT_OPENFOAM_TIMESERIES)}\n"
        f"run_full_vtk_export={int(RUN_FULL_VTK_EXPORT)}\n"
        f"minimal_stream_tracer_export={int(PARAVIEW_MINIMAL_STREAM_TRACER_EXPORT)}\n"
        f"storage_saver_mode={int(STORAGE_SAVER_MODE)}\n"
    )

    if current > CASE_STORAGE_LIMIT_BYTES and ABORT_IF_CASE_OVER_BUDGET:
        raise RuntimeError(
            f"Case storage budget exceeded after {context}: "
            f"{human_bytes(current)} > {human_bytes(CASE_STORAGE_LIMIT_BYTES)}. "
            "Reduce MAX_GLOBAL_CELLS/SURFACE_REFINEMENT/ASSEMBLY_DYNAMIC_STEPS or set "
            "CASE_STORAGE_LIMIT_GB higher. Compact PVD files are already being used."
        )


def write_components_geometry_snapshot(root_case: Path, components: Sequence[AeroComponent], folder_name: str, label: str) -> Optional[Path]:
    """Write a small combined .vtp geometry snapshot.

    This is the storage-light replacement for keeping full start/final OpenFOAM
    cases.  It stores only the surface triangles and useful component scalars.
    """
    if not STORE_START_FINAL_GEOMETRY:
        return None
    out_dir = root_case / folder_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts: List[Vec3] = []
    polys: List[Tuple[int, int, int]] = []
    scalars: Dict[str, List[float]] = {
        "patchId": [],
        "movable": [],
        "anchored": [],
        "massKg": [],
        "triangleArea": [],
    }
    name_lines = [f"{label} geometry snapshot", "", "patchId	patch	name	massKg	movable	anchored"]
    next_patch_id = 0
    for comp in components:
        movable = 0.0 if comp.is_assembly_anchor or (not comp.freedom.translate_axes and not comp.freedom.rotate_axes) else 1.0
        anchored = 1.0 if comp.is_assembly_anchor else 0.0
        body_ids = connected_surface_body_ids(comp)
        body_count = max(body_ids, default=-1) + 1
        for body_index in range(body_count):
            name_lines.append(
                f"{next_patch_id + body_index}	{comp.patch}#body_{body_index}	"
                f"{comp.name}	{comp.mass:.8g}	{int(movable)}	{int(anchored)}"
            )
        for triangle_index, (_normal, v1, v2, v3) in enumerate(comp.triangles):
            base = len(pts)
            pts.extend([v1, v2, v3])
            polys.append((base, base + 1, base + 2))
            area, _cent, _n = triangle_area_centroid_normal((_normal, v1, v2, v3))
            scalars["patchId"].append(
                float(next_patch_id + body_ids[triangle_index])
            )
            scalars["movable"].append(movable)
            scalars["anchored"].append(anchored)
            scalars["massKg"].append(float(comp.mass))
            scalars["triangleArea"].append(float(area))
        next_patch_id += body_count

    out = out_dir / GEOMETRY_SNAPSHOT_FILE_NAME
    _write_ascii_polydata_vtk(out, pts, polys, scalars)
    (out_dir / "component_index.txt").write_text("\n".join(name_lines) + "\n")
    return out

def _scalar_min_max(values: Sequence[float]) -> Tuple[float, float, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return 0.0, 0.0, 0.0
    mn = min(finite)
    mx = max(finite)
    return mn, mx, mx - mn


def _normalise_for_display(values: Sequence[float]) -> List[float]:
    """Normalised 0..1 field for diagnosis only, never for force/motion."""
    finite = [float(v) if math.isfinite(float(v)) else 0.0 for v in values]
    if not finite:
        return []
    mn = min(finite)
    mx = max(finite)
    rng = mx - mn
    if abs(rng) < 1e-30:
        return [0.5 for _ in finite]
    return [(v - mn) / rng for v in finite]


def _write_pressure_preview_report(path: Path, scalar_map: Dict[str, List[float]]) -> None:
    lines = [
        "# Pressure preview / ParaView colouring report",
        "# The compact PVD is a surface preview.  For visible pressure colouring, use:",
        "#   pressureCoeff", 
        "#   pressurePa", 
        "#   pressureVisible01", 
        "#   Cp / pPaPanel", 
        "# not plain OpenFOAM p unless you are opening the raw case.foam volume.",
        "",
        "field\tmin\tmax\trange",
    ]
    for name in sorted(scalar_map):
        vals = scalar_map[name]
        if not vals:
            lines.append(f"{name}\t<none>\t<none>\t<none>")
            continue
        mn, mx, rng = _scalar_min_max(vals)
        lines.append(f"{name}\t{mn:.9g}\t{mx:.9g}\t{rng:.9g}")
    lines.extend([
        "",
        "If pressureCoeff/pressurePa have a non-zero range but ParaView is blue:",
        "  1) click Rescale to Data Range,",
        "  2) make sure the selected source is paraview_motion_timeseries.pvd,",
        "  3) choose the Point Data version of pressureCoeff/pressurePa if ParaView shows both Point and Cell arrays.",
        "",
        "If pressureCoeff/pressurePa range is zero, the current frame genuinely has no visible pressure variation in this preview.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")



def _read_legacy_polydata_vtk(path: Path) -> Optional[Dict[str, Any]]:
    """Read ASCII legacy VTK POLYDATA/UNSTRUCTURED_GRID robustly.

    This reader is intentionally defensive because OpenFOAM's foamToVTK patch
    files can vary between POLYDATA and UNSTRUCTURED_GRID, can store arrays as
    FIELD attributes, and can sometimes contain fewer numeric values than the
    header advertises if a previous command was interrupted.  Earlier versions
    could raise ``list index out of range`` while converting the CFD patch VTKs
    into the compact .vtp/.pvd animation.  This version never indexes a numeric
    list without checking length first; if a file is malformed it returns None
    for that file and writes the usable files instead of killing the whole run.
    """
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None

    header = text[:5000]
    is_polydata = "DATASET POLYDATA" in header or "DATASET POLYDATA" in text
    is_unstructured = "DATASET UNSTRUCTURED_GRID" in header or "DATASET UNSTRUCTURED_GRID" in text
    if not (is_polydata or is_unstructured):
        return None

    lines = text.splitlines()
    points: List[Vec3] = []
    polys: List[List[int]] = []
    point_arrays: Dict[str, List[float]] = {}
    cell_arrays: Dict[str, List[float]] = {}
    point_vectors: Dict[str, List[Vec3]] = {}
    cell_vectors: Dict[str, List[Vec3]] = {}
    i = 0

    major_keys = {"POINTS", "POLYGONS", "CELLS", "CELL_TYPES", "POINT_DATA", "CELL_DATA", "VERTICES", "LINES", "TRIANGLE_STRIPS"}

    def safe_int(tok: str, default: int = 0) -> int:
        try:
            return int(float(tok))
        except Exception:
            return default

    def safe_float(tok: str, default: float = 0.0) -> float:
        try:
            v = float(tok)
        except Exception:
            return default
        return v if math.isfinite(v) else default

    def collect_numbers(start: int, needed: int) -> Tuple[List[float], int]:
        vals: List[float] = []
        j = start
        while j < len(lines) and len(vals) < needed:
            parts = lines[j].strip().split()
            # Stop before a new major block if we already collected something;
            # this protects against malformed/truncated array counts.
            if vals and parts and parts[0].upper() in major_keys:
                break
            for tok in parts:
                try:
                    fv = float(tok)
                except Exception:
                    continue
                if math.isfinite(fv):
                    vals.append(fv)
                if len(vals) >= needed:
                    break
            j += 1
        return vals, j

    def collect_ints(start: int, needed: int) -> Tuple[List[int], int]:
        vals, j = collect_numbers(start, needed)
        return [int(v) for v in vals], j

    def parse_connectivity(vals: Sequence[int], n_cells: int) -> List[List[int]]:
        parsed: List[List[int]] = []
        pos = 0
        for _ in range(max(0, n_cells)):
            if pos >= len(vals):
                break
            m = int(vals[pos])
            pos += 1
            if m <= 0:
                continue
            if pos + m > len(vals):
                # Truncated cell row.  Use the remaining indices if they can
                # form at least one triangle, then stop.
                poly = [int(v) for v in vals[pos:]]
                if len(poly) >= 3:
                    parsed.append(poly)
                break
            poly = [int(vals[pos + j]) for j in range(m)]
            pos += m
            if len(poly) >= 3:
                parsed.append(poly)
        return parsed

    def set_vector_arrays(
        target: Dict[str, List[float]],
        vector_target: Dict[str, List[Vec3]],
        name: str,
        vals: Sequence[float],
        ntuples: int,
        ncomp: int,
    ) -> None:
        rows: List[List[float]] = []
        for row_i in range(max(0, ntuples)):
            row: List[float] = []
            for comp_i in range(max(1, ncomp)):
                idx = row_i * max(1, ncomp) + comp_i
                row.append(float(vals[idx]) if idx < len(vals) and math.isfinite(float(vals[idx])) else 0.0)
            rows.append(row)
        vector_rows: List[Vec3] = []
        for row in rows:
            x = row[0] if len(row) > 0 else 0.0
            y = row[1] if len(row) > 1 else 0.0
            z = row[2] if len(row) > 2 else 0.0
            vector_rows.append((x, y, z))
        vector_target[name] = vector_rows
        target[name + "_mag"] = [math.sqrt(sum(float(c) * float(c) for c in row)) for row in rows]
        if name == "U":
            target["U_mag"] = target[name + "_mag"]
        if name == "wallShearStress":
            target["wallShearStress_mag"] = target[name + "_mag"]
        for ci in range(min(max(1, ncomp), 3)):
            suffix = "xyz"[ci]
            target[f"{name}_{ci}"] = [row[ci] if ci < len(row) else 0.0 for row in rows]
            target[f"{name}{suffix}"] = [row[ci] if ci < len(row) else 0.0 for row in rows]

    while i < len(lines):
        parts = lines[i].strip().split()
        if not parts:
            i += 1
            continue
        key = parts[0].upper()

        if key == "POINTS" and len(parts) >= 2:
            n = safe_int(parts[1])
            vals, i = collect_numbers(i + 1, n * 3)
            points = []
            for k in range(n):
                base = 3 * k
                if base + 2 >= len(vals):
                    break
                points.append((float(vals[base]), float(vals[base + 1]), float(vals[base + 2])))
            continue

        if key == "POLYGONS" and len(parts) >= 3:
            n = safe_int(parts[1])
            total = safe_int(parts[2])
            vals, i = collect_ints(i + 1, total)
            polys = parse_connectivity(vals, n)
            continue

        if key == "CELLS" and len(parts) >= 3:
            n = safe_int(parts[1])
            total = safe_int(parts[2])
            vals, i = collect_ints(i + 1, total)
            polys = parse_connectivity(vals, n)
            continue

        if key == "CELL_TYPES" and len(parts) >= 2:
            # Skip cell type IDs. They are not field data and were one source of
            # parser confusion in UNSTRUCTURED_GRID patch output.
            n = safe_int(parts[1])
            _vals, i = collect_numbers(i + 1, n)
            continue

        if key in {"POINT_DATA", "CELL_DATA"} and len(parts) >= 2:
            mode = key
            count = safe_int(parts[1])
            target = point_arrays if mode == "POINT_DATA" else cell_arrays
            vector_target = point_vectors if mode == "POINT_DATA" else cell_vectors
            i += 1
            while i < len(lines):
                p2 = lines[i].strip().split()
                if not p2:
                    i += 1
                    continue
                k2 = p2[0].upper()
                if k2 in major_keys - {"CELL_TYPES"}:
                    break

                if k2 == "FIELD" and len(p2) >= 3:
                    n_arrays = safe_int(p2[2])
                    i += 1
                    for _field_i in range(max(0, n_arrays)):
                        if i >= len(lines):
                            break
                        hdr = lines[i].strip().split()
                        if len(hdr) < 4:
                            i += 1
                            continue
                        name = hdr[0]
                        ncomp = max(1, safe_int(hdr[1], 1))
                        ntuples = max(0, safe_int(hdr[2], 0))
                        vals, i = collect_numbers(i + 1, ncomp * ntuples)
                        if ncomp == 1:
                            arr = [float(vals[j]) if j < len(vals) and math.isfinite(float(vals[j])) else 0.0 for j in range(ntuples)]
                            target[name] = arr
                        else:
                            set_vector_arrays(target, vector_target, name, vals, ntuples, ncomp)
                    continue

                if k2 == "SCALARS" and len(p2) >= 2:
                    name = p2[1]
                    ncomp = 1
                    if len(p2) >= 4:
                        ncomp = max(1, safe_int(p2[3], 1))
                    i += 1
                    if i < len(lines) and lines[i].strip().upper().startswith("LOOKUP_TABLE"):
                        i += 1
                    vals, i = collect_numbers(i, count * ncomp)
                    if ncomp == 1:
                        target[name] = [float(vals[j]) if j < len(vals) and math.isfinite(float(vals[j])) else 0.0 for j in range(count)]
                    else:
                        set_vector_arrays(target, vector_target, name, vals, count, ncomp)
                    continue

                if k2 == "VECTORS" and len(p2) >= 2:
                    name = p2[1]
                    vals, i = collect_numbers(i + 1, count * 3)
                    set_vector_arrays(target, vector_target, name, vals, count, 3)
                    continue

                # Unknown attribute type; skip one line to avoid infinite loops.
                i += 1
            continue

        i += 1

    if not points or not polys:
        return None
    return {
        "points": points,
        "polys": polys,
        "point_arrays": point_arrays,
        "cell_arrays": cell_arrays,
        "point_vectors": point_vectors,
        "cell_vectors": cell_vectors,
    }

def _vtk_sample_candidates(step_case: Path) -> List[Path]:
    roots = [step_case / "postProcessing", step_case / "surfaces", step_case / "VTK"]
    out: List[Path] = []
    for root in roots:
        if root.exists():
            for path in root.rglob("*.vtk"):
                text = " ".join(str(x).lower() for x in path.parts)
                # In storage-saver mode the only VTK export is the patch-only
                # foamToVTK run from Allsurface, so VTK/*.vtk is acceptable even
                # if the filename does not contain sample/surface/patch.
                if (
                    "sample" in text
                    or "surface" in text
                    or "wall" in text
                    or "patch" in text
                    or (RUN_PATCH_ONLY_VTK_EXPORT and root.name == "VTK" and not RUN_FULL_VTK_EXPORT)
                ):
                    out.append(path)
    # Prefer sampled-wall/postProcessing outputs, then patch-only VTK outputs.
    out.sort(key=lambda q: (0 if "postprocessing" in str(q).lower() else 1, str(q)))
    return out


def write_cfd_sampled_surface_preview_for_step(step_case: Path, step: int) -> Optional[Path]:
    """Convert real OpenFOAM sampled-surface VTKs into one XML .vtp frame.

    This is the high-definition path.  It uses actual solved CFD fields sampled
    onto the part surfaces instead of the coarse panel-normal preview.  The output
    is used by the main .pvd when available.
    """
    candidates = _vtk_sample_candidates(step_case)
    if not candidates:
        return None

    combined_pts: List[Vec3] = []
    combined_polys: List[Tuple[int, int, int]] = []
    combined_scalars: Dict[str, List[float]] = {}
    combined_vectors: Dict[str, List[Vec3]] = {}
    used_files: List[str] = []

    def add_scalar(name: str, value: float) -> None:
        combined_scalars.setdefault(name, []).append(value)

    def add_vector(name: str, value: Vec3) -> None:
        combined_vectors.setdefault(name, []).append(value)

    for vf in candidates:
        vf_lower = str(vf).lower()
        # Ignore wind-tunnel boundary exports; they are large and can dominate
        # the .pvd with flat inlet/outlet/farfield colours.
        if any(token in vf_lower for token in ("/inlet", "\\inlet", "/outlet", "\\outlet", "/farfield", "\\farfield")):
            continue
        data = _read_legacy_polydata_vtk(vf)
        if not data:
            continue
        pts: List[Vec3] = data["points"]
        polys: List[List[int]] = data["polys"]
        p_arrays: Dict[str, List[float]] = data["point_arrays"]
        c_arrays: Dict[str, List[float]] = data["cell_arrays"]
        p_vectors: Dict[str, List[Vec3]] = data.get("point_vectors", {})
        c_vectors: Dict[str, List[Vec3]] = data.get("cell_vectors", {})
        if not pts or not polys:
            continue
        used_files.append(str(vf.relative_to(step_case)) if vf.is_relative_to(step_case) else str(vf))

        def scalar_for_poly(name: str, poly_index: int, tri_indices: Sequence[int]) -> Optional[float]:
            if name in c_arrays and poly_index < len(c_arrays[name]):
                return float(c_arrays[name][poly_index])
            if name in p_arrays:
                vals = [float(p_arrays[name][idx]) for idx in tri_indices if idx < len(p_arrays[name])]
                if vals:
                    return sum(vals) / len(vals)
            return None

        def vector_for_poly(name: str, poly_index: int, tri_indices: Sequence[int]) -> Optional[Vec3]:
            if name in c_vectors and poly_index < len(c_vectors[name]):
                return c_vectors[name][poly_index]
            if name in p_vectors:
                vals = [p_vectors[name][idx] for idx in tri_indices if idx < len(p_vectors[name])]
                if vals:
                    inv_n = 1.0 / len(vals)
                    return (
                        sum(v[0] for v in vals) * inv_n,
                        sum(v[1] for v in vals) * inv_n,
                        sum(v[2] for v in vals) * inv_n,
                    )
            return None

        all_names = set(c_arrays) | set(p_arrays)
        all_vector_names = set(c_vectors) | set(p_vectors)
        for pi, poly in enumerate(polys):
            if len(poly) < 3:
                continue
            # Fan-triangulate any polygon.
            for jj in range(1, len(poly) - 1):
                tri_idx = [poly[0], poly[jj], poly[jj + 1]]
                if any(idx < 0 or idx >= len(pts) for idx in tri_idx):
                    continue
                base = len(combined_pts)
                combined_pts.extend([pts[tri_idx[0]], pts[tri_idx[1]], pts[tri_idx[2]]])
                combined_polys.append((base, base + 1, base + 2))
                for name in all_names:
                    val = scalar_for_poly(name, pi, tri_idx)
                    if val is not None and math.isfinite(val):
                        add_scalar(name, val)
                    else:
                        add_scalar(name, 0.0)
                # User-friendly aliases from real CFD fields.
                cp = scalar_for_poly("Cp", pi, tri_idx)
                ppa = scalar_for_poly("pPa", pi, tri_idx)
                kin_p = scalar_for_poly("p", pi, tri_idx)
                if ppa is None and kin_p is not None:
                    ppa = RHO * kin_p
                if cp is None:
                    if kin_p is not None:
                        cp = kin_p / max(0.5 * abs(VELOCITY) ** 2, 1e-30)
                    elif ppa is not None:
                        cp = ppa / max(0.5 * RHO * abs(VELOCITY) ** 2, 1e-30)
                add_scalar("pressureCoeff", float(cp or 0.0))
                add_scalar("pressureCoeffAbs", abs(float(cp or 0.0)))
                add_scalar("pressurePa", float(ppa or 0.0))
                add_scalar("pressurePaAbs", abs(float(ppa or 0.0)))
                for name in all_vector_names:
                    vec = vector_for_poly(name, pi, tri_idx)
                    if vec is None:
                        vec = (0.0, 0.0, 0.0)
                    add_vector(name, vec)

    if not combined_polys:
        return None
    # Normalised display field from real CFD pressure.
    combined_scalars["pressureVisible01"] = _normalise_for_display(combined_scalars.get("pressureCoeff", []))
    out_dir = step_case / CFD_SAMPLED_PREVIEW_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / CFD_SAMPLED_SURFACE_VTP_NAME
    _write_ascii_polydata_vtk(out, combined_pts, combined_polys, combined_scalars, combined_vectors)
    _write_pressure_preview_report(out_dir / "cfd_sampled_pressure_report.txt", combined_scalars)
    (out_dir / "source_sampled_vtk_files.txt").write_text("\n".join(used_files) + "\n")
    return out


def write_panel_aero_preview_for_step(step_case: Path, components: Sequence[AeroComponent], step: int) -> None:
    """Write crash-resistant surface VTK previews.

    v17 writes BOTH:
      1) one per-part VTK, useful for inspecting a single component;
      2) one combined VTK for the whole moving assembly.

    The combined file is deliberately simple ASCII POLYDATA with stable scalar
    arrays.  ParaView is much less likely to crash on this than on a PVD that
    references many mixed OpenFOAM VTK outputs from separately remeshed cases.
    """
    if not PARAVIEW_CREATE_PANEL_PREVIEW:
        return
    out_dir = step_case / "panel_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_pts: List[Vec3] = []
    combined_polys: List[Tuple[int, int, int]] = []
    combined_scalars: Dict[str, List[float]] = {
        "CpAbsPanel": [],
        "Cp": [],
        "p": [],
        "pPa": [],
        "pPaPanel": [],
        "pPaAbsPanel": [],
        # User-friendly aliases. These are the fields to colour by in ParaView.
        "pressureCoeff": [],
        "pressureCoeffAbs": [],
        "pressurePa": [],
        "pressurePaAbs": [],
        "pressureVisible01": [],
        "windExposure": [],
        "skinFrictionPreview": [],
        "normalX": [],
        "normalY": [],
        "normalZ": [],
        "triangleArea": [],
        "patchId": [],
        "movable": [],
        "anchored": [],
        "massKg": [],
        "structuralDisplacementM": [],
        "perforationPlug": [],
        "structuralFailedEdges": [],
        "velocityX": [],
        "velocityY": [],
        "velocityZ": [],
        "velocityMagnitude": [],
        "speed": [],
        "worldVelocityX": [],
        "worldVelocityY": [],
        "worldVelocityZ": [],
        "worldSpeed": [],
    }
    combined_vectors: Dict[str, List[Vec3]] = {
        "velocity": [],
        "worldVelocity": [],
    }

    next_patch_id = 0
    for comp in components:
        pts: List[Vec3] = []
        polys: List[Tuple[int, int, int]] = []
        cp_abs_vals: List[float] = []
        cp_alias_vals: List[float] = []
        p_vals: List[float] = []
        ppa_alias_vals: List[float] = []
        ppa_vals: List[float] = []
        ppa_abs_vals: List[float] = []
        pressure_coeff_vals: List[float] = []
        pressure_coeff_abs_vals: List[float] = []
        pressure_pa_vals: List[float] = []
        pressure_pa_abs_vals: List[float] = []
        exposed_vals: List[float] = []
        skin_vals: List[float] = []
        normal_x_vals: List[float] = []
        normal_y_vals: List[float] = []
        normal_z_vals: List[float] = []
        area_vals: List[float] = []
        structural_displacement_vals: List[float] = []
        perforation_plug_vals: List[float] = []
        structural_failed_edge_vals: List[float] = []
        velocity_x_vals: List[float] = []
        velocity_y_vals: List[float] = []
        velocity_z_vals: List[float] = []
        velocity_magnitude_vals: List[float] = []
        world_velocity_x_vals: List[float] = []
        world_velocity_y_vals: List[float] = []
        world_velocity_z_vals: List[float] = []
        world_speed_vals: List[float] = []
        velocity_vectors: List[Vec3] = []
        world_velocity_vectors: List[Vec3] = []
        movable = 0.0 if (comp.is_assembly_anchor or (not comp.freedom.translate_axes and not comp.freedom.rotate_axes)) else 1.0
        anchored = 1.0 if comp.is_assembly_anchor else 0.0
        component_velocity = comp.linear_velocity
        component_speed = v_norm(component_velocity)
        component_world_motion = component_world_velocity(comp)
        component_world_speed = v_norm(component_world_motion)
        shell_state = (
            comp.collision_structural_state.shell_state
            if isinstance(comp.collision_structural_state, HybridShellCollisionState)
            else comp.collision_structural_state
            if isinstance(comp.collision_structural_state, ExplicitShellState)
            else None
        )
        surface_body_ids = connected_surface_body_ids(comp)
        surface_body_count = max(surface_body_ids, default=-1) + 1

        for triangle_index, (_normal, v1, v2, v3) in enumerate(comp.triangles):
            i = len(pts)
            pts.extend([v1, v2, v3])
            polys.append((i, i + 1, i + 2))
            area, centroid, n = triangle_area_centroid_normal((_normal, v1, v2, v3))
            rel_speed, flow_dir = local_air_speed_and_unit(comp, centroid)
            incoming_dir = v_mul(flow_dir, -1.0)
            q = 0.5 * RHO * rel_speed ** 2
            # Panel preview pressure. This is only for robust visualisation; the
            # motion solver still prefers OpenFOAM forces. Use the same corrected
            # flow direction as the OpenFOAM case. Windward faces get positive Cp,
            # leeward faces get a smaller suction-like negative Cp, and edge-on
            # faces still get a small friction preview so the surface is not all
            # blue when the geometry is mostly parallel to the flow.
            facing = v_dot(n, incoming_dir)
            exposure = abs(facing)
            cp_abs = exposure * exposure
            cp = cp_abs if facing >= 0.0 else -0.35 * cp_abs
            skin_preview = max(0.0, 1.0 - abs(v_dot(n, flow_dir)))
            cp_abs_vals.append(cp_abs)
            pressure_pa = cp * q
            pressure_pa_abs = cp_abs * q
            p_kinematic = pressure_pa / max(RHO, 1e-30)
            cp_alias_vals.append(cp)
            p_vals.append(p_kinematic)
            ppa_alias_vals.append(pressure_pa)
            ppa_vals.append(pressure_pa)
            ppa_abs_vals.append(pressure_pa_abs)
            pressure_coeff_vals.append(cp)
            pressure_coeff_abs_vals.append(cp_abs)
            pressure_pa_vals.append(pressure_pa)
            pressure_pa_abs_vals.append(pressure_pa_abs)
            exposed_vals.append(exposure)
            skin_vals.append(skin_preview)
            normal_x_vals.append(n[0])
            normal_y_vals.append(n[1])
            normal_z_vals.append(n[2])
            area_vals.append(area)
            if (
                shell_state is not None
                and triangle_index < len(shell_state.triangle_nodes)
            ):
                node_ids = shell_state.triangle_nodes[triangle_index]
                structural_displacement = max(
                    v_norm(
                        v_sub(
                            shell_state.positions[node],
                            shell_state.reference_positions[node],
                        )
                    )
                    for node in node_ids
                )
                perforation_plug = float(
                    triangle_index in shell_state.plug_triangles
                )
                failed_edges = float(shell_state.failed_edges)
            else:
                structural_displacement = 0.0
                perforation_plug = 0.0
                failed_edges = 0.0
            structural_displacement_vals.append(structural_displacement)
            perforation_plug_vals.append(perforation_plug)
            structural_failed_edge_vals.append(failed_edges)
            velocity_x_vals.append(component_velocity[0])
            velocity_y_vals.append(component_velocity[1])
            velocity_z_vals.append(component_velocity[2])
            velocity_magnitude_vals.append(component_speed)
            world_velocity_x_vals.append(component_world_motion[0])
            world_velocity_y_vals.append(component_world_motion[1])
            world_velocity_z_vals.append(component_world_motion[2])
            world_speed_vals.append(component_world_speed)
            velocity_vectors.append(component_velocity)
            world_velocity_vectors.append(component_world_motion)

            ci = len(combined_pts)
            combined_pts.extend([v1, v2, v3])
            combined_polys.append((ci, ci + 1, ci + 2))
            combined_scalars["CpAbsPanel"].append(cp_abs)
            combined_scalars["Cp"].append(cp)
            combined_scalars["p"].append(p_kinematic)
            combined_scalars["pPa"].append(pressure_pa)
            combined_scalars["pPaPanel"].append(pressure_pa)
            combined_scalars["pPaAbsPanel"].append(pressure_pa_abs)
            combined_scalars["pressureCoeff"].append(cp)
            combined_scalars["pressureCoeffAbs"].append(cp_abs)
            combined_scalars["pressurePa"].append(pressure_pa)
            combined_scalars["pressurePaAbs"].append(pressure_pa_abs)
            # pressureVisible01 is filled after the loop from pressureCoeff.
            combined_scalars["pressureVisible01"].append(0.0)
            combined_scalars["windExposure"].append(exposure)
            combined_scalars["skinFrictionPreview"].append(skin_preview)
            combined_scalars["normalX"].append(n[0])
            combined_scalars["normalY"].append(n[1])
            combined_scalars["normalZ"].append(n[2])
            combined_scalars["triangleArea"].append(area)
            combined_scalars["patchId"].append(
                float(next_patch_id + surface_body_ids[triangle_index])
            )
            combined_scalars["movable"].append(movable)
            combined_scalars["anchored"].append(anchored)
            combined_scalars["massKg"].append(float(comp.mass))
            combined_scalars["structuralDisplacementM"].append(
                structural_displacement
            )
            combined_scalars["perforationPlug"].append(perforation_plug)
            combined_scalars["structuralFailedEdges"].append(failed_edges)
            combined_scalars["velocityX"].append(component_velocity[0])
            combined_scalars["velocityY"].append(component_velocity[1])
            combined_scalars["velocityZ"].append(component_velocity[2])
            combined_scalars["velocityMagnitude"].append(component_speed)
            combined_scalars["speed"].append(component_speed)
            combined_scalars["worldVelocityX"].append(component_world_motion[0])
            combined_scalars["worldVelocityY"].append(component_world_motion[1])
            combined_scalars["worldVelocityZ"].append(component_world_motion[2])
            combined_scalars["worldSpeed"].append(component_world_speed)
            combined_vectors["velocity"].append(component_velocity)
            combined_vectors["worldVelocity"].append(component_world_motion)

        _write_ascii_polydata_vtk(
            out_dir / f"{safe_patch_name(comp.patch)}_panel_step_{step:03d}.vtp",
            pts,
            polys,
            {
                "CpAbsPanel": cp_abs_vals,
                "Cp": cp_alias_vals,
                "p": p_vals,
                "pPa": ppa_alias_vals,
                "pPaPanel": ppa_vals,
                "pPaAbsPanel": ppa_abs_vals,
                "pressureCoeff": pressure_coeff_vals,
                "pressureCoeffAbs": pressure_coeff_abs_vals,
                "pressurePa": pressure_pa_vals,
                "pressurePaAbs": pressure_pa_abs_vals,
                "pressureVisible01": _normalise_for_display(pressure_coeff_vals),
                "windExposure": exposed_vals,
                "skinFrictionPreview": skin_vals,
                "normalX": normal_x_vals,
                "normalY": normal_y_vals,
                "normalZ": normal_z_vals,
                "triangleArea": area_vals,
                "structuralDisplacementM": structural_displacement_vals,
                "perforationPlug": perforation_plug_vals,
                "structuralFailedEdges": structural_failed_edge_vals,
                "velocityX": velocity_x_vals,
                "velocityY": velocity_y_vals,
                "velocityZ": velocity_z_vals,
                "velocityMagnitude": velocity_magnitude_vals,
                "speed": velocity_magnitude_vals,
                "worldVelocityX": world_velocity_x_vals,
                "worldVelocityY": world_velocity_y_vals,
                "worldVelocityZ": world_velocity_z_vals,
                "worldSpeed": world_speed_vals,
            },
            {
                "velocity": velocity_vectors,
                "worldVelocity": world_velocity_vectors,
            },
        )
        next_patch_id += surface_body_count

    if combined_polys:
        combined_scalars["pressureVisible01"] = _normalise_for_display(combined_scalars.get("pressureCoeff", []))
        _write_ascii_polydata_vtk(
            out_dir / COMBINED_SURFACE_VTK_NAME,
            combined_pts,
            combined_polys,
            combined_scalars,
            combined_vectors,
        )
        _write_pressure_preview_report(out_dir / "pressure_preview_report.txt", combined_scalars)


def create_panel_preview_pvd(root_case: Path, steps_dir: Path, total_steps: int) -> Optional[Path]:
    if not PARAVIEW_CREATE_PANEL_PREVIEW:
        return None
    pvd_path = root_case / PANEL_PREVIEW_PVD_NAME
    datasets: List[str] = []
    part_id = 0
    for step in range(total_steps):
        t = step * MOTION_DT if PARAVIEW_TIME_MODE == "seconds" else float(step)
        panel_dir = steps_dir / f"step_{step:03d}" / "panel_preview"
        if not panel_dir.exists():
            continue
        for vf in sorted(list(panel_dir.glob("*.vtp")) + list(panel_dir.glob("*.vtk"))):
            rel = os.path.relpath(vf, root_case).replace(os.sep, "/")
            datasets.append(f'    <DataSet timestep="{t:.10g}" group="panel" part="{part_id}" file="{rel}"/>')
            part_id += 1
    if not datasets:
        return None
    xml = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
        *datasets,
        '  </Collection>',
        '</VTKFile>',
        '',
    ]
    pvd_path.write_text("\n".join(xml))
    print(f"Panel aero preview PVD created: {pvd_path}")
    return pvd_path


def _write_pvd_collection(pvd_path: Path, datasets: List[str]) -> None:
    xml = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
        *datasets,
        '  </Collection>',
        '</VTKFile>',
        '',
    ]
    pvd_path.write_text("\n".join(xml))


def create_raw_cfd_pvd_timeseries(root_case: Path, steps_dir: Path, total_steps: int) -> Optional[Path]:
    """Optional heavier PVD referencing OpenFOAM-generated VTK outputs.

    This is useful for advanced volume/wake inspection, but on some ParaView
    builds it can crash because each motion step is a separately remeshed case
    and the referenced VTK files may contain inconsistent blocks/arrays.  v17
    therefore keeps this separate from the main user-facing PVD.
    """
    pvd_path = root_case / PARAVIEW_RAW_CFD_PVD_NAME
    datasets: List[str] = []
    manifest: List[str] = [
        "Raw CFD ParaView PVD manifest", "",
        "This file references OpenFOAM VTK files directly and may be heavy.",
        "If ParaView crashes, use paraview_motion_timeseries.pvd instead.", "",
        "time\tpart\tfile",
    ]
    for step in range(total_steps):
        t = step * MOTION_DT if PARAVIEW_TIME_MODE == "seconds" else float(step)
        step_case = steps_dir / f"step_{step:03d}"
        vtk_files = _vtk_files_for_step(step_case)
        for part_id, vf in enumerate(vtk_files):
            rel = os.path.relpath(vf, root_case).replace(os.sep, "/")
            datasets.append(f'    <DataSet timestep="{t:.10g}" group="raw_cfd" part="{part_id}" file="{rel}"/>')
            manifest.append(f"{t:.10g}\t{part_id}\t{rel}")
    if not datasets:
        return None
    _write_pvd_collection(pvd_path, datasets)
    (root_case / "paraview_raw_cfd_pvd_manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"Raw CFD PVD time series created: {pvd_path}")
    return pvd_path


def create_paraview_pvd_timeseries(root_case: Path, steps_dir: Path, total_steps: int) -> Optional[Path]:
    """Create the main single-file ParaView animation.

    v17 intentionally makes the main PVD crash-resistant: it references ONE
    clean combined surface VTK per motion step.  This avoids ParaView crashes
    caused by a PVD collection containing many mixed postProcessing/volume VTKs
    from separately remeshed OpenFOAM cases.
    """
    if not PARAVIEW_PVD_TIMESERIES:
        return None

    pvd_path = root_case / PARAVIEW_PVD_NAME
    datasets: List[str] = []
    manifest: List[str] = [
        "ParaView safe surface PVD manifest", "",
        "This is the recommended file to open:",
        f"  {pvd_path}", "",
        "It contains one clean combined moving-surface VTK per motion step.",
        "Colour by pressureCoeff, pressurePa, pressureVisible01, Cp, pPaPanel, windExposure, skinFrictionPreview, patchId, movable, or massKg.",
        "For raw CFD volume files, see paraview_raw_cfd_timeseries.pvd.", "",
        "time\tpart\tfile",
    ]

    if PARAVIEW_SAFE_COMBINED_SURFACE_PVD:
        for step in range(total_steps):
            t = step * MOTION_DT if PARAVIEW_TIME_MODE == "seconds" else float(step)
            vf = steps_dir / f"step_{step:03d}" / "panel_preview" / COMBINED_SURFACE_VTK_NAME
            if not vf.exists():
                manifest.append(f"# step_{step:03d}: missing {vf}")
                continue
            rel = os.path.relpath(vf, root_case).replace(os.sep, "/")
            # part is deliberately stable across timesteps.  Do NOT increment it
            # every frame; unstable part IDs can make ParaView allocate a growing
            # multiblock collection and crash on some builds.
            datasets.append(f'    <DataSet timestep="{t:.10g}" group="moving_surfaces" part="0" file="{rel}"/>')
            manifest.append(f"{t:.10g}\t0\t{rel}")
    else:
        # Legacy mode: direct VTK references.  Kept for users who explicitly opt in.
        for step in range(total_steps):
            t = step * MOTION_DT if PARAVIEW_TIME_MODE == "seconds" else float(step)
            step_case = steps_dir / f"step_{step:03d}"
            vtk_files = _vtk_files_for_step(step_case)
            for part_id, vf in enumerate(vtk_files):
                rel = os.path.relpath(vf, root_case).replace(os.sep, "/")
                datasets.append(f'    <DataSet timestep="{t:.10g}" group="" part="{part_id}" file="{rel}"/>')
                manifest.append(f"{t:.10g}\t{part_id}\t{rel}")

    _write_pvd_collection(pvd_path, datasets)
    (root_case / "paraview_pvd_manifest.txt").write_text("\n".join(manifest) + "\n")

    # Also create the raw/heavy PVD separately, but do not make it the primary file.
    create_raw_cfd_pvd_timeseries(root_case, steps_dir, total_steps)

    if datasets:
        print(f"Safe ParaView motion PVD created: {pvd_path}")
        return pvd_path
    print("WARNING: PVD time series requested, but no safe surface VTK files were found.")
    return None

def _is_openfoam_numeric_time_name(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _format_openfoam_time(value: float) -> str:
    # Avoid names like 1.0000000000000002. OpenFOAM accepts ordinary decimal
    # directory names, and ParaView sorts them numerically.
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return (f"{value:.9g}").rstrip("0").rstrip(".")


def _numeric_time_dirs(case: Path) -> List[Tuple[float, Path]]:
    found: List[Tuple[float, Path]] = []
    if not case.exists():
        return found
    for child in case.iterdir():
        if not child.is_dir():
            continue
        try:
            t = float(child.name)
        except ValueError:
            continue
        found.append((t, child))
    found.sort(key=lambda item: item[0])
    return found


def _scalar_field_range_quick(field_path: Path) -> Optional[float]:
    try:
        vals = _read_scalar_internal_values(field_path)
    except Exception:
        return None
    if not vals:
        return None
    return max(vals) - min(vals)


def _latest_solver_time_dir(case: Path) -> Optional[Path]:
    times = _numeric_time_dirs(case)
    if not times:
        return None

    # v16: do not accidentally select the initial 0/ field just because it
    # exists.  Prefer the highest non-zero time that has a p file. If pressure
    # varies, prefer the highest varying-pressure time. This prevents ParaView
    # output and pPa/Cp creation from being based on the uniform startup fields.
    candidates = []
    for t, path in times:
        has_p = (path / "p").exists()
        prange = _scalar_field_range_quick(path / "p") if has_p else None
        is_nonzero_time = abs(t) > 1e-15
        varies = bool(prange is not None and abs(prange) > 1e-20)
        candidates.append((varies, is_nonzero_time, has_p, t, path))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    best = candidates[-1][4]

    if PARAVIEW_STRICT_LATEST_NONZERO:
        nonzero_with_p = [c for c in candidates if c[1] and c[2]]
        if nonzero_with_p:
            nonzero_with_p.sort(key=lambda item: (item[0], item[3]))
            return nonzero_with_p[-1][4]

    return best


def merge_motion_steps_to_root_timeseries(root_case: Path, steps_dir: Path, total_steps: int) -> None:
    """Create one root OpenFOAM case.foam with all assembly-motion frames.

    Each assembly step is solved in its own case under motion_steps/step_XXX,
    because the geometry/mesh changes after aerodynamic motion.  ParaView only
    shows a multi-frame animation from one case.foam if the root case contains
    numeric time directories.  This function copies each step's latest solved
    fields and that step's polyMesh into actual_model_case/<time>/.

    The per-time polyMesh copy is important: the parts have moved, so the mesh
    is not necessarily the same from step to step.
    """
    if not PARAVIEW_ROOT_TIMESERIES:
        return

    first_step = steps_dir / "step_000"
    if not first_step.exists():
        print("WARNING: cannot build root ParaView time series; step_000 does not exist.")
        return

    # Remove old root OpenFOAM view data, but preserve reports and motion_steps.
    for child in list(root_case.iterdir()):
        remove = False
        if child.name in {"constant", "system", "VTK", "postProcessing"}:
            remove = True
        elif _is_openfoam_numeric_time_name(child.name):
            remove = True
        if remove:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    # Root-level constant/system are required by ParaView's OpenFOAM reader.
    # Use step_000 as the base, then override mesh per time directory below.
    for name in ("constant", "system"):
        src = first_step / name
        dst = root_case / name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        if src.exists():
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)

    copied = 0
    manifest_lines = [
        "Root ParaView OpenFOAM time series", "",
        "Open this single file in ParaView:",
        f"  {root_case / 'case.foam'}", "",
        f"PARAVIEW_TIME_MODE={PARAVIEW_TIME_MODE}",
        "",
        "time_dir\tsource_step\tsource_solver_time\tmesh_source",
    ]

    for step in range(total_steps):
        step_case = steps_dir / f"step_{step:03d}"
        latest = _latest_solver_time_dir(step_case)
        if latest is None or not latest.exists():
            manifest_lines.append(f"# step_{step:03d}: skipped; no numeric OpenFOAM time directory found")
            continue

        if PARAVIEW_TIME_MODE == "seconds":
            out_time = _format_openfoam_time(step * MOTION_DT)
        else:
            out_time = str(step)

        dst_time = root_case / out_time
        if dst_time.exists():
            shutil.rmtree(dst_time)
        shutil.copytree(latest, dst_time)

        # For changing geometry, give this frame its own mesh.  OpenFOAM and
        # ParaView can use a time-directory polyMesh when mesh/topology changes.
        mesh_src = step_case / "constant" / "polyMesh"
        mesh_dst = dst_time / "polyMesh"
        if mesh_dst.exists():
            shutil.rmtree(mesh_dst)
        if mesh_src.exists():
            shutil.copytree(mesh_src, mesh_dst)
            mesh_source = str(mesh_src)
        else:
            mesh_source = "root constant/polyMesh fallback"

        copied += 1
        manifest_lines.append(f"{out_time}\tstep_{step:03d}\t{latest.name}\t{mesh_source}")

    (root_case / "case.foam").write_text("")
    (root_case / "OPEN_THIS_IN_PARAVIEW.txt").write_text(
        "Open this OpenFOAM file in ParaView:\n"
        f"  {root_case / 'case.foam'}\n\n"
        "If pressure is flat blue or moving/remeshed parts are missing, open this instead:\n"
        f"  {root_case / PARAVIEW_PVD_NAME}\n\n"
        "The PVD file is usually more reliable for remeshed moving geometry.\n"
        "Use the ParaView time slider/play button to step through the moving geometry.\n"
        "Pressure fields: p = OpenFOAM kinematic pressure, pPa = pressure in Pa, Cp = pressure coefficient. Velocity is U.\n"
    )
    (root_case / "paraview_timeseries_manifest.txt").write_text("\n".join(manifest_lines) + "\n")
    print(f"ParaView root time series created with {copied}/{total_steps} frame(s): {root_case / 'case.foam'}")



def _clear_root_view_outputs_for_streaming(root_case: Path) -> None:
    """Remove old root-level view data before a no-motion_steps run.

    This keeps reports/source files, but removes old numeric time directories,
    root constant/system/VTK/postProcessing, and old compact preview/PVD outputs.
    """
    if not root_case.exists():
        root_case.mkdir(parents=True, exist_ok=True)
        return
    for child in list(root_case.iterdir()):
        remove = False
        if child.name in {
            "constant", "system", "VTK", "postProcessing",
            STREAM_TRACER_CASE_DIR_NAME,
            ROOT_PANEL_PREVIEW_DIR_NAME,
            ROOT_CFD_SAMPLED_DIR_NAME,
            STEP_DEBUG_REPORT_DIR_NAME,
            "final_moved_geometry_case",
            START_GEOMETRY_DIR_NAME,
            FINAL_MOVED_GEOMETRY_DIR_NAME,
            "motion_steps",
        }:
            # Remove old motion_steps too; v19 no longer keeps it unless explicitly requested.
            remove = True
        elif child.name in {
            PARAVIEW_PVD_NAME,
            PARAVIEW_RAW_CFD_PVD_NAME,
            PANEL_PREVIEW_PVD_NAME,
            "paraview_pvd_manifest.txt",
            "paraview_raw_cfd_pvd_manifest.txt",
            "paraview_panel_preview_manifest.txt",
            "paraview_stream_tracer_manifest.txt",
            "paraview_timeseries_manifest.txt",
            "OPEN_THIS_IN_PARAVIEW.txt",
        }:
            remove = True
        elif _is_openfoam_numeric_time_name(child.name):
            remove = True
        if remove:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)


def copy_step_to_root_timeseries(root_case: Path, step_case: Path, step: int) -> bool:
    """Copy one solved step into actual_model_case/<time>/ immediately.

    This replaces the old storage-heavy approach of keeping every full step in
    actual_model_case/motion_steps/ and merging afterwards.
    """
    latest = _latest_solver_time_dir(step_case)
    if latest is None or not latest.exists():
        print(f"WARNING: step {step}: no solved OpenFOAM time directory found; root time frame skipped.")
        return False

    # Root constant/system are required by ParaView's OpenFOAM reader.
    if step == 0:
        for name in ("constant", "system"):
            src = step_case / name
            dst = root_case / name
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink(missing_ok=True)
            if src.exists():
                shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)

    if PARAVIEW_TIME_MODE == "seconds":
        out_time = _format_openfoam_time(step * MOTION_DT)
    else:
        out_time = str(step)

    dst_time = root_case / out_time
    if dst_time.exists():
        shutil.rmtree(dst_time)
    shutil.copytree(latest, dst_time)

    # Changing geometry needs a per-frame mesh. Put polyMesh inside each root
    # time directory so root case.foam can show all frames without keeping the
    # full per-step OpenFOAM cases.
    mesh_src = step_case / "constant" / "polyMesh"
    mesh_dst = dst_time / "polyMesh"
    if mesh_dst.exists():
        shutil.rmtree(mesh_dst)
    if mesh_src.exists():
        shutil.copytree(mesh_src, mesh_dst)

    manifest = root_case / "paraview_timeseries_manifest.txt"
    if step == 0 or not manifest.exists():
        manifest.write_text(
            "Root ParaView OpenFOAM time series\n\n"
            "Open this single file in ParaView:\n"
            f"  {root_case / 'case.foam'}\n\n"
            f"PARAVIEW_TIME_MODE={PARAVIEW_TIME_MODE}\n"
            f"SAVE_MOTION_STEPS={int(SAVE_MOTION_STEPS)}\n\n"
            "time_dir\tsource_step\tsource_solver_time\tmesh_copied\n"
        )
    with manifest.open("a") as mf:
        mf.write(f"{out_time}\tstep_{step:03d}\t{latest.name}\t{bool(mesh_src.exists())}\n")

    (root_case / "case.foam").write_text("")
    return True


def copy_minimal_stream_tracer_case_to_root(root_case: Path, step_case: Path, step: int) -> Optional[Path]:
    """Keep one storage-light volume case for ParaView Stream Tracer.

    The compact .pvd/.vtp animation is surface-only.  ParaView's Stream Tracer is
    most useful on a volume mesh with a 3-component velocity field.  Keeping the
    full OpenFOAM time series or full VTK export is expensive, so this function
    overwrites one small root-level case each step:

      - constant/polyMesh
      - small top-level files directly under constant/
      - system/
      - latest solved time directory containing only U

    This is the minimum practical data ParaView needs to open a real volume field
    while still allowing the temporary per-step case to be deleted.
    """
    if not PARAVIEW_MINIMAL_STREAM_TRACER_EXPORT:
        return None

    latest = _latest_solver_time_dir(step_case)
    if latest is None:
        return None

    field_src = latest / STREAM_TRACER_FIELD_NAME
    mesh_src = step_case / "constant" / "polyMesh"
    if not field_src.exists() or not mesh_src.exists():
        return None

    out_case = root_case / STREAM_TRACER_CASE_DIR_NAME
    if out_case.exists():
        shutil.rmtree(out_case)
    out_case.mkdir(parents=True, exist_ok=True)

    system_src = step_case / "system"
    if system_src.exists():
        shutil.copytree(system_src, out_case / "system", ignore=shutil.ignore_patterns("*.log"))

    constant_out = out_case / "constant"
    constant_out.mkdir(parents=True, exist_ok=True)
    constant_src = step_case / "constant"
    if constant_src.exists():
        for child in constant_src.iterdir():
            if child.is_file():
                shutil.copy2(child, constant_out / child.name)
    shutil.copytree(mesh_src, constant_out / "polyMesh")

    time_out = out_case / latest.name
    time_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(field_src, time_out / STREAM_TRACER_FIELD_NAME)

    foam_path = out_case / "case.foam"
    foam_path.write_text("")

    manifest = [
        "Minimal ParaView Stream Tracer volume case",
        "",
        "Open this file in ParaView for real 3D streamlines:",
        f"  {foam_path}",
        "",
        f"source_step=step_{step:03d}",
        f"source_solver_time={latest.name}",
        f"field={STREAM_TRACER_FIELD_NAME}",
        "retained_data=constant/polyMesh, system, latest U only",
        "",
        "Use ParaView Stream Tracer with vector field U.",
        "This is intentionally a single latest-frame volume export, not a full time series.",
    ]
    (out_case / "OPEN_THIS_FOR_STREAM_TRACER.txt").write_text("\n".join(manifest) + "\n")
    (root_case / "paraview_stream_tracer_manifest.txt").write_text("\n".join(manifest) + "\n")
    return foam_path


def _copy_file_atomically(source: Path, destination: Path) -> None:
    """Copy a complete frame before exposing it at a PVD-referenced path."""
    validate_preview_polydata(source)
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary_path)
    validate_preview_polydata(temporary_path)
    os.replace(temporary_path, destination)


def copy_step_panel_preview_to_root(root_case: Path, step_case: Path, step: int) -> Optional[Path]:
    """Copy the compact combined moving-surface VTK into root_case only."""
    src = step_case / "panel_preview" / COMBINED_SURFACE_VTK_NAME
    if not src.exists():
        return None
    out_dir = root_case / ROOT_PANEL_PREVIEW_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"frame_{step:03d}_{COMBINED_SURFACE_VTK_NAME}"
    _copy_file_atomically(src, dst)

    # Keep the small pressure-range report even when SAVE_MOTION_STEPS=0, so the
    # user can immediately tell whether ParaView should have visible colour range.
    rep = step_case / "panel_preview" / "pressure_preview_report.txt"
    if rep.exists():
        shutil.copy2(rep, out_dir / f"frame_{step:03d}_pressure_preview_report.txt")
    return dst



def copy_step_cfd_sampled_preview_to_root(root_case: Path, step_case: Path, step: int) -> Optional[Path]:
    """Copy high-definition CFD sampled surface .vtp into root_case only."""
    src = step_case / CFD_SAMPLED_PREVIEW_DIR_NAME / CFD_SAMPLED_SURFACE_VTP_NAME
    if not src.exists():
        return None
    out_dir = root_case / ROOT_CFD_SAMPLED_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"frame_{step:03d}_{CFD_SAMPLED_SURFACE_VTP_NAME}"
    _copy_file_atomically(src, dst)
    for rep_name in ("cfd_sampled_pressure_report.txt", "source_sampled_vtk_files.txt"):
        rep = step_case / CFD_SAMPLED_PREVIEW_DIR_NAME / rep_name
        if rep.exists():
            shutil.copy2(rep, out_dir / f"frame_{step:03d}_{rep_name}")
    return dst


def copy_step_debug_reports_to_root(
    root_case: Path,
    step_case: Path,
    step: int,
    force: bool = False,
) -> None:
    if not KEEP_STEP_DEBUG_REPORTS and not force:
        return
    report_names = [
        "force_coeff_debug_report.txt",
        "force_load_debug_report.txt",
        PRESSURE_RANGE_REPORT_NAME,
        VISUALIZATION_VALIDATION_REPORT_NAME,
        "mesh_resolution_report.txt",
        "retained_body_patches.txt",
        "log.blockMesh",
        "log.snappyHexMesh",
        "log.checkMesh",
    ]
    out_dir = root_case / STEP_DEBUG_REPORT_DIR_NAME / f"step_{step:03d}"
    copied = False
    for name in report_names:
        src = step_case / name
        if src.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_dir / name)
            copied = True
    if copied:
        (out_dir / "source_note.txt").write_text(
            "These are small debug reports copied from a temporary step case.\n"
            "The full step case was not retained because SAVE_MOTION_STEPS=0.\n"
        )


def create_root_safe_pvd_from_copied_previews(root_case: Path, total_steps: int) -> Optional[Path]:
    """Create main PVD from root-level compact frames.

    The main motion PVD uses the combined panel preview generated from the
    solver's live component list.  Unlike a remeshed CFD sampled surface, it
    cannot silently omit an impactor or a newly detached fragment from one
    frame.  CFD sampled surfaces are retained separately for pressure-field
    inspection and remain a fallback if preview generation is unavailable.
    """
    if not PARAVIEW_PVD_TIMESERIES:
        return None
    pvd_path = root_case / PARAVIEW_PVD_NAME
    datasets: List[str] = []
    manifest: List[str] = [
        "ParaView safe surface PVD manifest", "",
        "This file references compact frames stored at root level.",
        "Full per-step OpenFOAM cases are not retained unless SAVE_MOTION_STEPS=1.",
        "The main animation uses stable combined solver geometry; CFD sampled surfaces are separate diagnostics.",
        "For visible pressure colour, use pressureCoeff, pressurePa, pressureVisible01, Cp, or pPa.", "",
        "time\tpart\tgroup\tfile",
    ]
    for step in range(total_steps):
        t = step * MOTION_DT if PARAVIEW_TIME_MODE == "seconds" else float(step)
        cfd_vf = root_case / ROOT_CFD_SAMPLED_DIR_NAME / f"frame_{step:03d}_{CFD_SAMPLED_SURFACE_VTP_NAME}"
        panel_vf = root_case / ROOT_PANEL_PREVIEW_DIR_NAME / f"frame_{step:03d}_{COMBINED_SURFACE_VTK_NAME}"
        panel_valid = False
        cfd_valid = False
        if panel_vf.exists():
            try:
                validate_preview_polydata(panel_vf)
                panel_valid = True
            except (OSError, ValueError, ElementTree.ParseError) as exc:
                manifest.append(f"# step_{step:03d}: invalid panel preview: {exc}")
        if cfd_vf.exists():
            try:
                validate_preview_polydata(cfd_vf)
                cfd_valid = True
            except (OSError, ValueError, ElementTree.ParseError) as exc:
                manifest.append(f"# step_{step:03d}: invalid CFD preview: {exc}")
        if panel_valid:
            vf = panel_vf
            group = "panel_preview_surfaces"
        elif cfd_valid:
            vf = cfd_vf
            group = "sampled_cfd_surfaces_fallback"
        else:
            manifest.append(f"# step_{step:03d}: missing both {cfd_vf} and {panel_vf}")
            continue
        rel = os.path.relpath(vf, root_case).replace(os.sep, "/")
        datasets.append(f'    <DataSet timestep="{t:.10g}" group="{group}" part="0" file="{rel}"/>')
        manifest.append(f"{t:.10g}\t0\t{group}\t{rel}")
    _write_pvd_collection(pvd_path, datasets)
    (root_case / "paraview_pvd_manifest.txt").write_text("\n".join(manifest) + "\n")
    if datasets:
        print(f"Safe compact ParaView motion PVD created: {pvd_path}")
        return pvd_path
    print("WARNING: no compact preview VTK frames were available for the main PVD.")
    return None

def create_root_panel_preview_pvd(root_case: Path, total_steps: int) -> Optional[Path]:
    """Compatibility PVD using the same compact preview frames."""
    if not PARAVIEW_CREATE_PANEL_PREVIEW:
        return None
    pvd_path = root_case / PANEL_PREVIEW_PVD_NAME
    datasets: List[str] = []
    manifest: List[str] = ["Panel preview PVD manifest", "", "time\tfile"]
    for step in range(total_steps):
        t = step * MOTION_DT if PARAVIEW_TIME_MODE == "seconds" else float(step)
        vf = root_case / ROOT_PANEL_PREVIEW_DIR_NAME / f"frame_{step:03d}_{COMBINED_SURFACE_VTK_NAME}"
        if not vf.exists():
            continue
        rel = os.path.relpath(vf, root_case).replace(os.sep, "/")
        datasets.append(f'    <DataSet timestep="{t:.10g}" group="panel" part="0" file="{rel}"/>')
        manifest.append(f"{t:.10g}\t{rel}")
    if not datasets:
        return None
    _write_pvd_collection(pvd_path, datasets)
    (root_case / "paraview_panel_preview_manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"Compact panel preview PVD created: {pvd_path}")
    return pvd_path


def format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def load_eta_calibration() -> Dict[str, Any]:
    try:
        if ETA_CALIBRATION_FILE.exists():
            return json.loads(ETA_CALIBRATION_FILE.read_text())
    except Exception:
        pass
    return {}


ETA_TIMING_SCOPE = "complete_dynamic_step_v2"


def save_eta_calibration(step_seconds: float) -> None:
    old = load_eta_calibration()
    previous = (
        old.get("seconds_per_assembly_step")
        if old.get("timing_scope") == ETA_TIMING_SCOPE
        else None
    )
    if isinstance(previous, (int, float)) and previous > 0:
        blended = (1.0 - ETA_EMA_ALPHA) * float(previous) + ETA_EMA_ALPHA * step_seconds
        samples = int(old.get("samples", 1)) + 1
    else:
        blended = step_seconds
        samples = 1
    payload = {
        "seconds_per_assembly_step": blended,
        "last_step_seconds": step_seconds,
        "timing_scope": ETA_TIMING_SCOPE,
        "samples": samples,
        "cfd_iterations": ITERATIONS,
        "surface_refinement": list(SURFACE_REFINEMENT),
        "region_refinement": REGION_REFINEMENT,
        "max_global_cells": MAX_GLOBAL_CELLS,
        "dynamic_steps_last_run": ASSEMBLY_DYNAMIC_STEPS,
        "updated_unix_time": time.time(),
    }
    try:
        ETA_CALIBRATION_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:
        pass


def initial_eta_seconds(total_steps: int) -> Tuple[float, str]:
    cal = load_eta_calibration()
    calibrated = cal.get("seconds_per_assembly_step")
    if (
        cal.get("timing_scope") == ETA_TIMING_SCOPE
        and isinstance(calibrated, (int, float))
        and calibrated > 0
    ):
        return float(calibrated) * total_steps, f"calibrated from {int(cal.get('samples', 1))} previous step(s)"
    return ETA_DEFAULT_STEP_SECONDS * total_steps, "rough default; will recalibrate after step 1"
