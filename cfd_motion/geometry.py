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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import *
from .models import *
from .math_utils import *

def foam_header(class_name: str, object_name: str) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def safe_patch_name(name: str, fallback: str = "part") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    cleaned = cleaned.strip("_") or fallback
    if cleaned[0].isdigit():
        cleaned = f"p_{cleaned}"
    return cleaned[:48]


def unique_patch_names(names: Sequence[str]) -> List[str]:
    used: Dict[str, int] = {}
    out: List[str] = []
    for name in names:
        base = safe_patch_name(name)
        count = used.get(base, 0)
        used[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def read_stl_triangles(path: Path) -> List[Triangle]:
    data = path.read_bytes()

    # Binary STL
    if len(data) >= 84:
        n_tri = struct.unpack("<I", data[80:84])[0]
        expected = 84 + n_tri * 50
        if expected == len(data):
            triangles: List[Triangle] = []
            off = 84
            for _ in range(n_tri):
                rec = struct.unpack_from("<12fH", data, off)
                triangles.append((rec[0:3], rec[3:6], rec[6:9], rec[9:12]))
                off += 50
            if triangles:
                return triangles

    # ASCII STL
    triangles = []
    normal: Vec3 = (0.0, 0.0, 0.0)
    vertices: List[Vec3] = []
    text = data.decode(errors="ignore")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 5 and parts[0].lower() == "facet" and parts[1].lower() == "normal":
            normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            vertices = []
        elif len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if len(vertices) == 3:
                triangles.append((normal, vertices[0], vertices[1], vertices[2]))
                vertices = []

    if not triangles:
        raise ValueError(f"Could not read STL triangles from {path}")
    return triangles


def split_ascii_stl_solids(path: Path) -> List[Tuple[str, List[Triangle]]]:
    """Split a grouped ASCII STL into named solids. Binary STL returns one body."""
    data = path.read_bytes()
    if len(data) >= 84:
        n_tri = struct.unpack("<I", data[80:84])[0]
        if 84 + n_tri * 50 == len(data):
            return [(path.stem, read_stl_triangles(path))]

    text = data.decode(errors="ignore")
    solids: List[Tuple[str, List[Triangle]]] = []
    name = path.stem
    normal: Vec3 = (0.0, 0.0, 0.0)
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []

    for line in text.splitlines():
        stripped = line.strip()
        parts = stripped.split()
        if parts and parts[0].lower() == "solid":
            if triangles:
                solids.append((name, triangles))
                triangles = []
            name = " ".join(parts[1:]).strip() or path.stem
        elif parts and parts[0].lower() == "endsolid":
            if triangles:
                solids.append((name, triangles))
                triangles = []
        elif len(parts) == 5 and parts[0].lower() == "facet" and parts[1].lower() == "normal":
            normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            vertices = []
        elif len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if len(vertices) == 3:
                triangles.append((normal, vertices[0], vertices[1], vertices[2]))
                vertices = []

    if triangles:
        solids.append((name, triangles))
    if not solids:
        solids.append((path.stem, read_stl_triangles(path)))
    return solids


def stl_points(triangles: Iterable[Triangle]) -> List[Vec3]:
    pts: List[Vec3] = []
    for _normal, v1, v2, v3 in triangles:
        pts.extend([v1, v2, v3])
    return pts


def bounds(points: Sequence[Vec3], scale: float = 1.0) -> Tuple[float, float, float, float, float, float]:
    xs = [p[0] * scale for p in points]
    ys = [p[1] * scale for p in points]
    zs = [p[2] * scale for p in points]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def component_references(triangles: Sequence[Triangle]) -> Tuple[float, float, Vec3]:
    pts = stl_points(triangles)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds(pts, 1.0)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    lref = max(dx, dy, dz, 1e-6)
    aref = max(dy * dz, dx * dz, dx * dy, 1e-12)
    cofr = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
    return aref, lref, cofr


def component_bounds(triangles: Sequence[Triangle]) -> Tuple[float, float, float, float, float, float]:
    pts = stl_points(triangles)
    if not pts:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return bounds(pts, 1.0)


def infer_motion_origin(component: "AeroComponent") -> Vec3:
    """Infer a practical hinge/pivot origin for simplified relative motion.

    OpenFOAM forceCoeffs can report a small moment if CofR is set near the part
    centre.  A hinged part, however, rotates because pressure force acts at the
    aerodynamic centre while the hinge is at an edge.  When Onshape mate origins
    are not decoded, this function approximates the hinge at the upstream edge of
    the moving part and through the part centre in the other two coordinates.
    """
    if component.motion_origin is not None:
        return component.motion_origin
    if not AUTO_HINGE_ORIGINS or not component.freedom.rotate_axes:
        return component.cofr
    if len(component.freedom.translate_axes) >= 3 or len(component.freedom.rotate_axes) >= 3:
        return component.cofr

    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
    cx, cy, cz = component.cofr

    # The freestream vector is (VELOCITY, 0, 0).  The upstream edge is the side
    # the air reaches first.  This makes elevator/flap/rudder-like parts rotate
    # around a plausible leading/hinge edge instead of around their own centre.
    hinge_x = xmax if VELOCITY < 0 else xmin
    return (hinge_x, cy, cz)


def force_reference_origin(component: "AeroComponent") -> Vec3:
    """CofR for OpenFOAM forceCoeffs.

    For rotational parts, use the hinge/pivot origin so CmPitch/CmRoll/CmYaw are
    moments about the motion constraint, not about the part centre.  This is the
    main reason earlier versions showed changing pressure but no visible hinge
    motion.
    """
    if component.freedom.rotate_axes:
        return infer_motion_origin(component)
    return component.cofr


def motion_basis_debug(component: "AeroComponent") -> str:
    origin = infer_motion_origin(component)
    xmin, xmax, ymin, ymax, zmin, zmax = component_bounds(component.triangles)
    return (
        f"origin=({origin[0]:.6g},{origin[1]:.6g},{origin[2]:.6g}), "
        f"cofr=({component.cofr[0]:.6g},{component.cofr[1]:.6g},{component.cofr[2]:.6g}), "
        f"bbox=({xmin:.6g},{xmax:.6g},{ymin:.6g},{ymax:.6g},{zmin:.6g},{zmax:.6g})"
    )


def estimate_closed_mesh_volume(triangles: Sequence[Triangle]) -> float:
    """Return an approximate closed-mesh volume in m^3.

    STL orientation is not always reliable, so the absolute signed volume is used.
    If the STL is not watertight, this is only a rough estimate; BOM mass wins
    whenever it is available.
    """
    signed = 0.0
    for _normal, v1, v2, v3 in triangles:
        signed += v_dot(v1, v_cross(v2, v3)) / 6.0
    return max(abs(signed), 0.0)


def estimate_scalar_inertia(mass: float, triangles: Sequence[Triangle]) -> float:
    ixx, iyy, izz = estimate_box_inertia_diagonal(mass, triangles)
    return max((ixx + iyy + izz) / 3.0, 1e-9)


def estimate_box_inertia_diagonal(mass: float, triangles: Sequence[Triangle]) -> Tuple[float, float, float]:
    pts = stl_points(triangles)
    if not pts:
        return (
            DEFAULT_PART_INERTIA_KGM2,
            DEFAULT_PART_INERTIA_KGM2,
            DEFAULT_PART_INERTIA_KGM2,
        )
    xmin, xmax, ymin, ymax, zmin, zmax = bounds(pts, 1.0)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    dz = max(zmax - zmin, 1e-6)
    ixx = mass * (dy * dy + dz * dz) / 12.0
    iyy = mass * (dx * dx + dz * dz) / 12.0
    izz = mass * (dx * dx + dy * dy) / 12.0
    return (
        max(ixx, 1e-9),
        max(iyy, 1e-9),
        max(izz, 1e-9),
    )


BASE_MATERIAL_DENSITIES_KG_M3: Dict[str, float] = {
    "foam": 35.0,
    "eps": 20.0,
    "xps": 35.0,
    "epp": 35.0,
    "depron": 40.0,
    "foamboard": 120.0,
    "foam board": 120.0,
    "balsa": 160.0,
    "plywood": 550.0,
    "basswood": 420.0,
    "pla": 1240.0,
    "petg": 1270.0,
    "abs": 1040.0,
    "asa": 1070.0,
    "nylon": 1150.0,
    "pa12": 1010.0,
    "pa-12": 1010.0,
    "resin": 1150.0,
    "carbon": 1600.0,
    "carbon fiber": 1600.0,
    "carbon fibre": 1600.0,
    "fiberglass": 1900.0,
    "glass fibre": 1900.0,
    "glass fiber": 1900.0,
    "aluminum": 2700.0,
    "aluminium": 2700.0,
    "steel": 7850.0,
    "stainless steel": 8000.0,
    "brass": 8500.0,
    "copper": 8960.0,
    "titanium": 4500.0,
}




def get_first(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return default


def path_to_key(path: Any) -> str:
    if isinstance(path, list):
        return "/".join(str(x) for x in path)
    return str(path or "")

def normalize_material_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_material_overrides() -> Dict[str, float]:
    merged: Dict[str, float] = {}
    raw_items: List[Any] = []
    if MATERIAL_OVERRIDES_JSON:
        try:
            raw_items.append(json.loads(MATERIAL_OVERRIDES_JSON))
        except Exception as exc:
            print(f"Warning: could not parse MATERIAL_OVERRIDES_JSON: {exc}")
    if MATERIAL_OVERRIDES_FILE:
        try:
            raw_items.append(json.loads(Path(MATERIAL_OVERRIDES_FILE).expanduser().read_text()))
        except Exception as exc:
            print(f"Warning: could not parse MATERIAL_OVERRIDES_FILE={MATERIAL_OVERRIDES_FILE!r}: {exc}")
    for obj in raw_items:
        if isinstance(obj, dict):
            for key, val in obj.items():
                try:
                    merged[normalize_material_text(key)] = float(val)
                except Exception:
                    pass
    return merged


def material_density_kg_m3(material_name: str) -> float:
    material = normalize_material_text(material_name)
    library = dict(BASE_MATERIAL_DENSITIES_KG_M3)
    library.update(load_material_overrides())
    if material in library:
        return library[material]
    for key, density in library.items():
        if key and key in material:
            return density
    return DEFAULT_MATERIAL_DENSITY_KG_M3


def material_damping_factors(material_name: str, density: float) -> Tuple[float, float]:
    material = normalize_material_text(material_name)
    if any(word in material for word in ("foam", "eps", "xps", "epp", "depron", "foamboard", "balsa")):
        return DEFAULT_LINEAR_DAMPING_PER_KG * 4.0, DEFAULT_ANGULAR_DAMPING_PER_KG * 4.0
    if any(word in material for word in ("aluminum", "aluminium", "steel", "brass", "copper", "titanium")):
        return DEFAULT_LINEAR_DAMPING_PER_KG * 0.35, DEFAULT_ANGULAR_DAMPING_PER_KG * 0.35
    if any(word in material for word in ("carbon", "fiberglass", "glass fibre", "glass fiber")):
        return DEFAULT_LINEAR_DAMPING_PER_KG * 0.75, DEFAULT_ANGULAR_DAMPING_PER_KG * 0.75
    if density < 200.0:
        return DEFAULT_LINEAR_DAMPING_PER_KG * 4.0, DEFAULT_ANGULAR_DAMPING_PER_KG * 4.0
    if density > 2500.0:
        return DEFAULT_LINEAR_DAMPING_PER_KG * 0.35, DEFAULT_ANGULAR_DAMPING_PER_KG * 0.35
    return DEFAULT_LINEAR_DAMPING_PER_KG, DEFAULT_ANGULAR_DAMPING_PER_KG


def parse_numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "displayValue", "computedValue", "expression", "number"):
            out = parse_numeric_value(value.get(key))
            if out is not None:
                return out
        return None
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_mass_kg(value: Any) -> Optional[float]:
    number = parse_numeric_value(value)
    if number is None:
        return None
    text = str(value or "").lower()
    if " mg" in text or text.endswith("mg"):
        return number / 1_000_000.0
    if " g" in text or text.endswith("g"):
        return number / 1000.0
    if "lb" in text or "pound" in text:
        return number * 0.45359237
    if "oz" in text:
        return number * 0.0283495231
    return number


def parse_density_kg_m3(value: Any) -> Optional[float]:
    number = parse_numeric_value(value)
    if number is None:
        return None
    text = str(value or "").lower().replace(" ", "")
    if "g/cm" in text or "g/cc" in text:
        return number * 1000.0
    if "kg/m" in text:
        return number
    if "lb/in" in text:
        return number * 27679.9047
    if "lb/ft" in text:
        return number * 16.0184634
    return number


def text_from_any(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(text_from_any(v) for v in value)
    if isinstance(value, dict):
        priority = ["value", "displayValue", "name", "partNumber", "material", "materialName", "description", "id"]
        chunks = [text_from_any(value.get(k)) for k in priority if k in value]
        if chunks:
            return " ".join(c for c in chunks if c)
        return " ".join(text_from_any(v) for v in value.values())
    return str(value)


def flatten_candidate_bom_rows(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            keys = {str(k).lower() for k in obj.keys()}
            rowish = bool(keys & {"item", "itemnumber", "name", "partnumber", "partidentity", "material", "quantity", "columns", "properties"})
            text = json.dumps(obj, default=str).lower()[:3000]
            if rowish and any(term in text for term in ("material", "mass", "density", "part", "item")):
                rows.append(obj)
            for val in obj.values():
                visit(val)
        elif isinstance(obj, list):
            for val in obj:
                visit(val)
    visit(payload)
    return rows


def get_case_insensitive(obj: Dict[str, Any], names: Sequence[str]) -> Any:
    wanted = {n.lower().replace(" ", "").replace("_", "") for n in names}
    for key, value in obj.items():
        norm = str(key).lower().replace(" ", "").replace("_", "")
        if norm in wanted:
            return value
    return None


def extract_bom_column_values(row: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    def add_value(label: str, value: Any) -> None:
        label = str(label or "").strip().lower()
        if label:
            values[label] = value
    for key, value in row.items():
        add_value(key, value)
    for container_key in ("columns", "columnValues", "properties", "propertyValues", "cells", "data"):
        container = row.get(container_key)
        if isinstance(container, list):
            for item in container:
                if isinstance(item, dict):
                    label = get_first(item, ["columnId", "propertyId", "name", "label", "displayName", "id", "key"], "")
                    val = get_first(item, ["value", "displayValue", "computedValue", "expression", "text"], item)
                    add_value(label, val)
        elif isinstance(container, dict):
            for key, val in container.items():
                add_value(str(key), val)
    return values


def bom_records_from_payload(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    records: List[Dict[str, Any]] = []
    for row in flatten_candidate_bom_rows(payload):
        vals = extract_bom_column_values(row)
        all_text = json.dumps(row, default=str).lower()
        material = ""
        mass = None
        density = None
        for label, value in vals.items():
            label_norm = label.lower()
            if not material and "material" in label_norm:
                material = text_from_any(value).strip()
            if mass is None and "mass" in label_norm and "center" not in label_norm and "centre" not in label_norm:
                mass = parse_mass_kg(value)
            if density is None and ("density" in label_norm or "rho" == label_norm):
                density = parse_density_kg_m3(value)
        identity_text = text_from_any(get_case_insensitive(row, ["partIdentity", "part identity", "identity"]))
        names = [
            text_from_any(get_case_insensitive(row, ["name", "partName", "part name", "description"])),
            text_from_any(get_case_insensitive(row, ["partNumber", "part number", "number"])),
            identity_text,
            text_from_any(get_case_insensitive(row, ["id", "item", "itemNumber", "item number"])),
            all_text[:1500],
        ]
        records.append({
            "material": material,
            "mass_kg": mass,
            "density_kg_m3": density,
            "names": [n for n in names if n],
            "raw": row,
        })
    return records


def score_bom_record_for_occurrence(record: Dict[str, Any], occ: Dict[str, Any], raw_name: str, part_id: Optional[str]) -> int:
    hay = normalize_material_text(" ".join(record.get("names", [])))
    if not hay:
        return 0
    score = 0
    candidates = [raw_name, part_id, get_first(occ, ["partId", "pid"], ""), get_first(occ, ["name", "occurrenceName", "partName", "instanceId", "fullPathAsString"], "")]
    for cand in candidates:
        c = normalize_material_text(cand)
        if not c:
            continue
        if c == hay:
            score += 100
        elif c in hay:
            score += 25 + min(len(c), 25)
    return score


def best_bom_record_for_occurrence(records: Sequence[Dict[str, Any]], occ: Dict[str, Any], raw_name: str, part_id: Optional[str]) -> Optional[Dict[str, Any]]:
    best = None
    best_score = 0
    for record in records:
        score = score_bom_record_for_occurrence(record, occ, raw_name, part_id)
        if score > best_score:
            best = record
            best_score = score
    return best if best_score >= 10 else None


def infer_material_from_name(name: str) -> str:
    text = normalize_material_text(name)
    library = dict(BASE_MATERIAL_DENSITIES_KG_M3)
    library.update(load_material_overrides())
    for key in sorted(library, key=len, reverse=True):
        if key and key in text:
            return key
    return DEFAULT_MATERIAL_NAME


def apply_material_model(component: AeroComponent, material_name: str, mass_kg: Optional[float], density_kg_m3: Optional[float], source: str) -> None:
    volume = estimate_closed_mesh_volume(component.triangles)
    density = density_kg_m3 if density_kg_m3 and density_kg_m3 > 0 else material_density_kg_m3(material_name)
    mass = mass_kg if mass_kg and mass_kg > 0 else (volume * density if volume > 0 else DEFAULT_PART_MASS_KG)
    if not math.isfinite(mass) or mass <= 0:
        mass = DEFAULT_PART_MASS_KG
    linear_damping, angular_damping = material_damping_factors(material_name, density)
    component.material = MaterialProperties(
        material_name=material_name or DEFAULT_MATERIAL_NAME,
        density_kg_m3=density,
        mass_kg=mass,
        volume_m3=volume,
        source=source,
        linear_damping_per_kg=linear_damping,
        angular_damping_per_kg=angular_damping,
    )
    component.mass = mass
    component.inertia = estimate_scalar_inertia(mass, component.triangles)


def assign_materials_from_bom(components: Sequence[AeroComponent], occurrences: Sequence[Dict[str, Any]], raw_names: Sequence[str], bom_payload: Optional[Dict[str, Any]]) -> List[str]:
    records = bom_records_from_payload(bom_payload) if USE_BOM_MATERIALS else []
    report = [
        "Assembly material/BOM report",
        "",
        f"USE_BOM_MATERIALS={USE_BOM_MATERIALS}",
        f"BOM candidate records found={len(records)}",
        f"Default density={DEFAULT_MATERIAL_DENSITY_KG_M3:g} kg/m^3",
        "",
        "Applied material model:",
    ]
    occ_by_key: Dict[str, Tuple[Dict[str, Any], str, Optional[str]]] = {}
    for occ, raw_name in zip(occurrences, raw_names):
        part_id = str(get_first(occ, ["partId", "pid"], "")) or None
        key = path_to_key(occ.get("path")) or str(get_first(occ, ["fullPathAsString", "instanceId", "partId"], ""))
        occ_by_key[key] = (occ, raw_name, part_id)
    for c in components:
        occ, raw_name, part_id = occ_by_key.get(c.source_occurrence or "", ({}, c.name, None))
        record = best_bom_record_for_occurrence(records, occ, raw_name, part_id)
        if record:
            material = record.get("material") or DEFAULT_MATERIAL_NAME
            mass = record.get("mass_kg")
            density = record.get("density_kg_m3")
            source = "bom"
        else:
            material = infer_material_from_name(c.name)
            mass = None
            density = None
            source = "name/default"
        apply_material_model(c, material, mass, density, source)
        report.append(
            f"- {c.patch}: name={c.name!r}, material={c.material.material_name!r}, "
            f"source={c.material.source}, density={c.material.density_kg_m3:.6g} kg/m^3, "
            f"volume={c.material.volume_m3:.6g} m^3, mass={c.mass:.6g} kg, "
            f"inertia={c.inertia:.6g} kg m^2, "
            f"linear_damping_per_kg={c.material.linear_damping_per_kg:.6g}, "
            f"angular_damping_per_kg={c.material.angular_damping_per_kg:.6g}"
        )
    return report


def write_ascii_stl_triangles(destination: Path, solid_name: str, triangles: Sequence[Triangle], scale: float = 1.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as f:
        f.write(f"solid {safe_patch_name(solid_name)}\n")
        for normal, v1, v2, v3 in triangles:
            f.write(f"  facet normal {normal[0]:g} {normal[1]:g} {normal[2]:g}\n")
            f.write("    outer loop\n")
            for vx, vy, vz in (v1, v2, v3):
                f.write(f"      vertex {vx * scale:g} {vy * scale:g} {vz * scale:g}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {safe_patch_name(solid_name)}\n")


def write_scaled_ascii_stl(source: Path, destination: Path, scale: float) -> None:
    write_ascii_stl_triangles(destination, "obstacle", read_stl_triangles(source), scale)
