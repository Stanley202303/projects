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
from .geometry import *

def is_onshape_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and "onshape.com" in parsed.netloc.lower()


def parse_onshape_url(url: str) -> OnshapeRef:
    parsed = urllib.parse.urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]

    try:
        i = parts.index("documents")
        did = parts[i + 1]
        wvm = parts[i + 2]
        wvmid = parts[i + 3]
        e_marker = parts[i + 4]
        eid = parts[i + 5]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "Expected an Onshape URL like: "
            "https://cad.onshape.com/documents/<did>/w/<wid>/e/<eid>"
        ) from exc

    if wvm not in {"w", "v", "m"}:
        raise ValueError("Onshape URL must contain /w/, /v/, or /m/ after the document id")
    if e_marker != "e":
        raise ValueError("Could not find /e/<element-id> in the Onshape URL")

    query = urllib.parse.parse_qs(parsed.query)
    configuration = None
    if "configuration" in query and query["configuration"]:
        configuration = query["configuration"][0]

    return OnshapeRef(
        base_url=f"{parsed.scheme}://{parsed.netloc}",
        did=did,
        wvm=wvm,
        wvmid=wvmid,
        eid=eid,
        configuration=configuration,
    )


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


class OnshapeClient:
    def __init__(self, access_key: str, secret_key: str) -> None:
        if not access_key or not secret_key:
            raise ValueError(
                "Missing Onshape credentials. Run:\n"
                "  export ONSHAPE_ACCESS_KEY='your_access_key'\n"
                "  export ONSHAPE_SECRET_KEY='your_secret_key'"
            )
        self.access_key = access_key
        self.secret_key = secret_key
        self.opener = urllib.request.build_opener(NoRedirectHandler)

    def _signed_headers(self, method: str, url: str, accept: str, content_type: str) -> Dict[str, str]:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path or "/"
        query = parsed.query or ""
        nonce = secrets.token_hex(13)
        auth_date = email.utils.formatdate(time.time(), usegmt=True)

        signature_input = (
            f"{method}\n{nonce}\n{auth_date}\n{content_type}\n{path}\n{query}\n"
        ).lower()
        digest = hmac.new(
            self.secret_key.encode("utf-8"),
            signature_input.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature = base64.b64encode(digest).decode("ascii")

        return {
            "Accept": accept,
            "Content-Type": content_type,
            "Date": auth_date,
            "On-Nonce": nonce,
            "Authorization": f"On {self.access_key}:HmacSHA256:{signature}",
        }

    def request_bytes(self, method: str, url: str, accept: str = "*/*", body: Optional[bytes] = None,
                      content_type: str = JSON_CONTENT_TYPE) -> bytes:
        current_url = url
        current_method = method.upper()
        current_body = body

        for _ in range(8):
            headers = self._signed_headers(current_method, current_url, accept, content_type)
            req = urllib.request.Request(current_url, data=current_body, headers=headers, method=current_method)
            try:
                with self.opener.open(req) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308}:
                    location = exc.headers.get("Location")
                    if not location:
                        raise RuntimeError(f"Onshape redirect {exc.code} had no Location header") from exc
                    current_url = urllib.parse.urljoin(current_url, location)
                    if exc.code == 303:
                        current_method = "GET"
                        current_body = None
                    continue

                body_text = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Onshape API failed with HTTP {exc.code}:\n{body_text[:1600]}") from exc

        raise RuntimeError("Too many redirects while calling Onshape")

    def request_json(self, method: str, url: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        raw = self.request_bytes(method, url, accept=JSON_ACCEPT, body=data, content_type=JSON_CONTENT_TYPE)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8", errors="replace"))


def api_base(ref: OnshapeRef) -> str:
    return f"{ref.base_url}/api/{ONSHAPE_API_VERSION}"


def api_base_for_version(ref: OnshapeRef, version: str) -> str:
    version = version.strip() or ONSHAPE_API_VERSION
    if not version.startswith("v"):
        version = "v" + version
    return f"{ref.base_url}/api/{version}"


def api_base_candidates(ref: OnshapeRef) -> List[str]:
    # Current API first, then known documented versions used in the Onshape guides.
    versions: List[str] = []
    for v in (ONSHAPE_API_VERSION, "v16", "v12", "v11", "v9", "v6"):
        v = (v or "").strip()
        if v and v not in versions:
            versions.append(v)
    return [api_base_for_version(ref, v) for v in versions]


def get_onshape_client() -> OnshapeClient:
    return OnshapeClient(
        os.environ.get("ONSHAPE_ACCESS_KEY", ""),
        os.environ.get("ONSHAPE_SECRET_KEY", ""),
    )


def detect_onshape_element_type(ref: OnshapeRef, client: OnshapeClient) -> str:
    """Return 'partstudio' or 'assembly'.

    The older single-element metadata probe was unreliable for some tabs and often
    returned 404. The more robust route is to list the document elements and match
    the URL element id. If that fails, probe the assembly and Part Studio endpoints.
    """
    elements_url = f"{api_base(ref)}/documents/d/{ref.did}/{ref.wvm}/{ref.wvmid}/elements"
    try:
        elements_payload = client.request_json("GET", elements_url)
        if isinstance(elements_payload, list):
            elements = elements_payload
        else:
            elements = elements_payload.get("elements", []) if isinstance(elements_payload, dict) else []

        for element in elements:
            if not isinstance(element, dict):
                continue
            element_id = str(get_first(element, ["id", "elementId", "eid"], ""))
            if element_id != ref.eid:
                continue
            text = json.dumps(element).lower()
            element_type = str(get_first(element, ["elementType", "type", "element_type"], "")).lower()
            if "assembly" in element_type or "assembly" in text:
                return "assembly"
            if "partstudio" in element_type or "part studio" in text or "partstudio" in text:
                return "partstudio"
    except Exception as exc:
        print(f"Element list probe failed, trying endpoint probes instead: {exc}")

    # Endpoint probe: assembly definition is cheap compared with meshing.
    asm_url = (
        f"{api_base(ref)}/assemblies/d/{ref.did}/{ref.wvm}/{ref.wvmid}/e/{ref.eid}"
        "?includeMateFeatures=false&includeNonSolids=false&includeMateConnectors=false&excludeSuppressed=true"
    )
    try:
        client.request_json("GET", asm_url)
        return "assembly"
    except Exception:
        pass

    # Final probe: Part Studio STL endpoint. Do not actually download here; just
    # classify it as a Part Studio if the assembly probe did not work.
    return "partstudio"

def download_onshape_stl(onshape_url: str, destination: Path) -> Path:
    ref = parse_onshape_url(onshape_url)
    client = get_onshape_client()
    return download_partstudio_stl(ref, client, destination)


def add_onshape_stl_quality_params(params: Dict[str, str]) -> Dict[str, str]:
    """Add optional STL tessellation controls to Onshape export URLs.

    They are intentionally opt-in because some Onshape endpoints/API versions
    reject unknown query parameters.  Use e.g.:
      export ONSHAPE_STL_RESOLUTION=custom
      export ONSHAPE_ANGULAR_DEVIATION=1
      export ONSHAPE_CHORDAL_TOLERANCE=0.0002
      export ONSHAPE_MIN_FACET_WIDTH=0.0002
    """
    if ONSHAPE_STL_RESOLUTION:
        params["resolution"] = ONSHAPE_STL_RESOLUTION
    if ONSHAPE_ANGULAR_DEVIATION:
        params["angularDeviation"] = ONSHAPE_ANGULAR_DEVIATION
        params["angleTolerance"] = ONSHAPE_ANGULAR_DEVIATION
    if ONSHAPE_CHORDAL_TOLERANCE:
        params["chordalTolerance"] = ONSHAPE_CHORDAL_TOLERANCE
        params["chordTolerance"] = ONSHAPE_CHORDAL_TOLERANCE
    if ONSHAPE_MIN_FACET_WIDTH:
        params["minFacetWidth"] = ONSHAPE_MIN_FACET_WIDTH
    return params


def download_partstudio_stl(ref: OnshapeRef, client: OnshapeClient, destination: Path) -> Path:
    params = {
        "mode": ONSHAPE_STL_MODE,
        "grouping": "true",
        "scale": f"{ONSHAPE_EXPORT_SCALE:g}",
        "units": ONSHAPE_UNITS,
    }
    if ref.configuration:
        params["configuration"] = ref.configuration
    add_onshape_stl_quality_params(params)

    query = urllib.parse.urlencode(params)
    export_url = (
        f"{api_base(ref)}/partstudios/d/{ref.did}/{ref.wvm}/{ref.wvmid}"
        f"/e/{ref.eid}/stl?{query}"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    data = client.request_bytes("GET", export_url, accept="*/*")
    if not data:
        raise RuntimeError("Onshape returned an empty STL")
    destination.write_bytes(data)
    return destination


def download_part_stl(ref: OnshapeRef, client: OnshapeClient, source_did: str, source_wvm: str,
                      source_wvmid: str, source_eid: str, part_id: str, destination: Path,
                      configuration: Optional[str] = None) -> Path:
    """Download one physical part as an STL.

    This deliberately tries several documented/legacy API versions and both
    lower/upper STL path suffixes. Onshape's guide shows part IDs are obtained
    from the Parts endpoint, and the export endpoint is under /parts/d/.../e/.../partid/{pid}/...
    but real documents can reference source versions or microversions from an
    assembly, so we keep the path construction flexible.
    """
    params = {
        "mode": ONSHAPE_STL_MODE,
        "grouping": "true",
        "scale": f"{ONSHAPE_EXPORT_SCALE:g}",
        "units": ONSHAPE_UNITS,
    }
    if configuration:
        params["configuration"] = configuration
    add_onshape_stl_quality_params(params)
    query = urllib.parse.urlencode(params)
    quoted_pid = urllib.parse.quote(str(part_id), safe="")
    wvmid_quoted = urllib.parse.quote(str(source_wvmid), safe="")
    eid_quoted = urllib.parse.quote(str(source_eid), safe="")
    did_quoted = urllib.parse.quote(str(source_did), safe="")

    urls: List[str] = []
    for base in api_base_candidates(ref):
        for suffix in ("stl", "STL"):
            urls.append(
                f"{base}/parts/d/{did_quoted}/{source_wvm}/{wvmid_quoted}"
                f"/e/{eid_quoted}/partid/{quoted_pid}/{suffix}?{query}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    for export_url in urls:
        try:
            data = client.request_bytes("GET", export_url, accept="*/*")
            if data:
                destination.write_bytes(data)
                return destination
            errors.append(f"empty response: {export_url}")
        except Exception as exc:
            errors.append(f"{export_url} -> {exc}")

    raise RuntimeError(
        f"Could not export partId={part_id!r} from did={source_did}, {source_wvm}/{source_wvmid}, "
        f"eid={source_eid}. Tried {len(urls)} endpoint variants. Last errors:\n"
        + "\n".join(errors[-6:])
    )


def list_partstudio_parts(ref: OnshapeRef, client: OnshapeClient, source_did: str, source_wvm: str,
                          source_wvmid: str, source_eid: str) -> List[Dict[str, Any]]:
    """Return all solid parts in a source Part Studio when an assembly instance
    inserted the whole Part Studio instead of a single part.
    """
    did_quoted = urllib.parse.quote(str(source_did), safe="")
    wvmid_quoted = urllib.parse.quote(str(source_wvmid), safe="")
    eid_quoted = urllib.parse.quote(str(source_eid), safe="")
    query = urllib.parse.urlencode({
        "elementId": str(source_eid),
        "withThumbnails": "false",
        "includePropertyDefaults": "false",
    })
    urls = []
    for base in api_base_candidates(ref):
        urls.append(f"{base}/parts/d/{did_quoted}/{source_wvm}/{wvmid_quoted}/e/{eid_quoted}?withThumbnails=false&includePropertyDefaults=false")
        urls.append(f"{base}/parts/d/{did_quoted}/{source_wvm}/{wvmid_quoted}?{query}")

    errors: List[str] = []
    for url in urls:
        try:
            payload = client.request_json("GET", url)
            if isinstance(payload, list):
                return [x for x in payload if isinstance(x, dict)]
            if isinstance(payload, dict):
                for key in ("parts", "items", "data"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        return [x for x in value if isinstance(x, dict)]
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
    raise RuntimeError(
        f"Could not list parts for source Part Studio did={source_did}, {source_wvm}/{source_wvmid}, eid={source_eid}. "
        f"Last errors:\n" + "\n".join(errors[-6:])
    )


def get_assembly_definition(ref: OnshapeRef, client: OnshapeClient) -> Dict[str, Any]:
    query = urllib.parse.urlencode({
        "includeMateFeatures": "true",
        "includeNonSolids": "false",
        "includeMateConnectors": "true",
        "excludeSuppressed": "true",
    })
    url = f"{api_base(ref)}/assemblies/d/{ref.did}/{ref.wvm}/{ref.wvmid}/e/{ref.eid}?{query}"
    return client.request_json("GET", url)


def get_assembly_bom(ref: OnshapeRef, client: OnshapeClient) -> Optional[Dict[str, Any]]:
    if not USE_BOM_MATERIALS:
        return None
    query_items = {
        "indented": "false",
        "multiLevel": "true",
        "generateIfAbsent": "false",
    }
    if ref.configuration:
        query_items["configuration"] = ref.configuration
    query = urllib.parse.urlencode(query_items)
    url = f"{api_base(ref)}/assemblies/d/{ref.did}/{ref.wvm}/{ref.wvmid}/e/{ref.eid}/bom?{query}"
    try:
        return client.request_json("GET", url)
    except Exception as exc:
        print(f"Warning: could not read assembly BOM/material data; using fallback material model: {exc}")
        return None


def create_assembly_stl_translation(ref: OnshapeRef, client: OnshapeClient) -> Dict[str, Any]:
    url = f"{api_base(ref)}/assemblies/d/{ref.did}/{ref.wvm}/{ref.wvmid}/e/{ref.eid}/translations"
    body = {
        "formatName": "STL",
        "stlMode": "TEXT",
        "grouping": True,
        # Important for read-only documents: do not try to create a blob/tab in
        # the Onshape document. The completed translation will be downloaded from
        # resultExternalDataIds instead.
        "storeInDocument": False,
        "translate": True,
        "allowFaultyParts": True,
        "angularTolerance": 0.001,
        "distanceTolerance": 0.001,
        "unit": "METER",
    }
    return client.request_json("POST", url, body=body)


def poll_translation(ref: OnshapeRef, client: OnshapeClient, translation: Dict[str, Any], timeout_s: float = 300.0) -> Dict[str, Any]:
    tid = translation.get("id")
    if not tid:
        raise RuntimeError(f"Onshape translation response did not include an id: {translation}")
    deadline = time.time() + timeout_s
    latest = translation
    delay = 1.5
    while time.time() < deadline:
        state = str(latest.get("requestState", "")).upper()
        if state == "DONE":
            return latest
        if state == "FAILED":
            raise RuntimeError(f"Onshape translation failed: {latest.get('failureReason') or latest}")
        time.sleep(delay)
        delay = min(delay * 1.5, 10.0)
        latest = client.request_json("GET", f"{api_base(ref)}/translations/{tid}")
    raise TimeoutError(f"Timed out waiting for Onshape translation {tid}")


def _normalise_id_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def download_translation_result(ref: OnshapeRef, client: OnshapeClient, translation: Dict[str, Any], destination: Path) -> Path:
    """Download an asynchronous translation result.

    For read-only documents we use storeInDocument=false, so the result normally
    appears in resultExternalDataIds and is downloaded from /documents/d/{did}/externaldata/{fid}.
    If the user changes storeInDocument=true, fall back to blob element download.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    external_ids = _normalise_id_list(translation.get("resultExternalDataIds"))
    if external_ids:
        fid = urllib.parse.quote(external_ids[0], safe="")
        result_did = str(translation.get("resultDocumentId") or translation.get("documentId") or ref.did)
        url = f"{api_base(ref)}/documents/d/{result_did}/externaldata/{fid}"
        data = client.request_bytes("GET", url, accept="application/octet-stream")
        if not data:
            raise RuntimeError(f"External translation result {fid} was empty")
        destination.write_bytes(data)
        return destination

    element_ids = _normalise_id_list(translation.get("resultElementIds"))
    if element_ids:
        result_eid = urllib.parse.quote(element_ids[0], safe="")
        result_did = str(translation.get("resultDocumentId") or translation.get("documentId") or ref.did)
        result_wid = str(translation.get("resultWorkspaceId") or translation.get("workspaceId") or ref.wvmid)
        # Blob results are created in a workspace, so use /w/{wid} even if the
        # source URL was /v/ or /m/.
        url = f"{api_base(ref)}/blobelements/d/{result_did}/w/{result_wid}/e/{result_eid}"
        data = client.request_bytes("GET", url, accept="application/octet-stream")
        if not data:
            raise RuntimeError(f"Blob translation result {result_eid} was empty")
        destination.write_bytes(data)
        return destination

    raise RuntimeError(f"Translation completed but no resultExternalDataIds/resultElementIds were returned: {translation}")


# ------------------------- assembly mate / occurrence parsing -------------------------


def walk_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_dicts(item)


def get_first(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def path_to_key(path: Any) -> str:
    if isinstance(path, list):
        return "/".join(str(x) for x in path)
    return str(path or "")


def is_suppressed_dict(d: Dict[str, Any]) -> bool:
    return bool(d.get("suppressed") is True or d.get("isSuppressed") is True or str(d.get("suppressionState", "")).lower() == "suppressed")


def collect_assembly_instances(assembly_def: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Collect every assembly instance dictionary, including instances nested in
    subassemblies. The assembly definition docs expose rootAssembly.instances with
    id, name, type, documentId, elementId, partId and documentMicroversion.
    """
    instances: Dict[str, Dict[str, Any]] = {}
    for d in walk_dicts(assembly_def):
        value = d.get("instances")
        if not isinstance(value, list):
            continue
        for inst in value:
            if not isinstance(inst, dict):
                continue
            iid = str(get_first(inst, ["id", "instanceId", "nodeId"], ""))
            if not iid:
                continue
            if iid not in instances:
                instances[iid] = inst
    return instances


def extract_occurrences(assembly_def: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return actual part/subassembly occurrences, preferring explicit
    `occurrences` arrays. The old broad tree walk often found transform-bearing
    helper objects that were not physical components and therefore lacked partId.
    """
    candidates: List[Dict[str, Any]] = []

    explicit = assembly_def.get("occurrences")
    if isinstance(explicit, list):
        candidates.extend([x for x in explicit if isinstance(x, dict)])

    for d in walk_dicts(assembly_def):
        occs = d.get("occurrences")
        if isinstance(occs, list):
            for occ in occs:
                if isinstance(occ, dict):
                    candidates.append(occ)

    # Conservative fallback: objects with a transform and an occurrence path.
    if not candidates:
        for d in walk_dicts(assembly_def):
            transform = d.get("transform")
            if isinstance(transform, list) and len(transform) >= 12:
                if any(k in d for k in ("path", "fullPathAsString", "tailInstanceId", "headInstanceId")):
                    candidates.append(d)

    unique: Dict[str, Dict[str, Any]] = {}
    for d in candidates:
        if is_suppressed_dict(d):
            continue
        key = path_to_key(d.get("path"))
        if not key:
            key = str(get_first(d, ["fullPathAsString", "tailInstanceId", "headInstanceId", "id", "instanceId"], ""))
        if not key:
            key = str(len(unique))
        # Keep only occurrences that either have a transform or a resolvable path.
        if not isinstance(d.get("transform"), list) and not d.get("path") and not get_first(d, ["tailInstanceId", "headInstanceId"]):
            continue
        unique[key] = d
    return list(unique.values())


def occurrence_tail_instance_id(occ: Dict[str, Any]) -> str:
    direct = get_first(occ, ["tailInstanceId", "instanceId", "id", "headInstanceId"], "")
    if direct:
        return str(direct)
    path = occ.get("path")
    if isinstance(path, list) and path:
        return str(path[-1])
    fpas = str(occ.get("fullPathAsString") or "")
    if fpas:
        # Onshape often uses slash-delimited full paths.
        return fpas.split("/")[-1]
    return ""


def occurrence_path_ids(occ: Dict[str, Any]) -> List[str]:
    path = occ.get("path")
    if isinstance(path, list):
        return [str(x) for x in path]
    fpas = str(occ.get("fullPathAsString") or "")
    if fpas:
        return [x for x in re.split(r"[/,; >]+", fpas) if x]
    tail = occurrence_tail_instance_id(occ)
    return [tail] if tail else []


def merge_occurrence_with_instance(occ: Dict[str, Any], instances: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a resolved occurrence dictionary containing transform/path from
    the occurrence plus source IDs from the corresponding instance. For normal
    part occurrences, the last path ID is the owning instance ID.
    """
    resolved: Dict[str, Any] = dict(occ)
    ids_to_try = []
    tail = occurrence_tail_instance_id(occ)
    if tail:
        ids_to_try.append(tail)
    ids_to_try.extend(reversed(occurrence_path_ids(occ)))

    inst: Optional[Dict[str, Any]] = None
    for iid in ids_to_try:
        if iid in instances:
            inst = instances[iid]
            break

    # Some JSON variants use full paths; fall back to a name/path substring match.
    if inst is None:
        key_text = json.dumps({k: occ.get(k) for k in ("path", "fullPathAsString", "tailInstanceId", "headInstanceId", "name")}, default=str)
        for iid, candidate in instances.items():
            if iid and iid in key_text:
                inst = candidate
                break

    if inst:
        for k, v in inst.items():
            resolved.setdefault(k, v)
        resolved["_sourceInstance"] = inst
        resolved["_sourceInstanceId"] = str(get_first(inst, ["id", "instanceId", "nodeId"], ""))
    else:
        resolved["_sourceInstance"] = None
        resolved["_sourceInstanceId"] = tail

    resolved.setdefault("transform", occ.get("transform", list(mat_identity())))
    return resolved


def resolve_exportable_occurrences(ref: OnshapeRef, client: OnshapeClient, assembly_def: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolve transform occurrences into one or more exportable part jobs.

    If an assembly instance inserted a whole Part Studio with no partId, expand it
    into all parts from that source Part Studio using the same occurrence transform.
    """
    instances = collect_assembly_instances(assembly_def)
    occurrences = extract_occurrences(assembly_def)
    report: List[str] = [
        "Assembly occurrence export report",
        "",
        f"raw instances found={len(instances)}",
        f"raw occurrences found={len(occurrences)}",
        "",
    ]

    jobs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for idx, occ in enumerate(occurrences):
        resolved = merge_occurrence_with_instance(occ, instances)
        if is_suppressed_dict(resolved):
            report.append(f"skip occurrence {idx}: suppressed")
            continue

        inst_type = str(get_first(resolved, ["type", "instanceType"], "")).lower()
        is_part_like = ("part" in inst_type) or bool(get_first(resolved, ["partId", "pid"], None)) or bool(get_first(resolved, ["elementId", "sourceElementId", "eid"], None))
        if not is_part_like:
            report.append(f"skip occurrence {idx}: not a directly exportable part-like instance, type={inst_type!r}")
            continue

        part_id = get_first(resolved, ["partId", "pid"])
        if part_id:
            key = f"{path_to_key(resolved.get('path'))}|{resolved.get('_sourceInstanceId')}|{part_id}"
            if key in seen:
                continue
            seen.add(key)
            jobs.append(resolved)
            report.append(
                f"resolved occurrence {idx}: name={get_first(resolved, ['name','occurrenceName','partName'], '<unnamed>')!r}, "
                f"instance={resolved.get('_sourceInstanceId')!r}, partId={part_id!r}, "
                f"did={get_first(resolved, ['documentId','sourceDocumentId','did'], ref.did)!r}, "
                f"eid={get_first(resolved, ['elementId','sourceElementId','eid'], ref.eid)!r}"
            )
            continue

        # Whole Part Studio / missing partId path. Expand all source parts.
        try:
            source_did, source_wvm, source_wvmid, source_eid, _missing_pid, configuration = occurrence_source_ids(ref, resolved)
            parts = list_partstudio_parts(ref, client, source_did, source_wvm, source_wvmid, source_eid)
            solid_parts = [part for part in parts if str(part.get("bodyType", "solid")).lower() in {"solid", ""}]
            for part in solid_parts:
                pid = get_first(part, ["partId", "pid"])
                if not pid:
                    continue
                expanded = dict(resolved)
                expanded["partId"] = pid
                expanded["partName"] = get_first(part, ["name", "partName"], expanded.get("name", f"part_{pid}"))
                expanded["name"] = f"{get_first(resolved, ['name','occurrenceName'], 'PartStudio')}::{expanded['partName']}"
                expanded["_expandedFromWholePartStudio"] = True
                key = f"{path_to_key(expanded.get('path'))}|{expanded.get('_sourceInstanceId')}|{pid}"
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(expanded)
            report.append(
                f"expanded occurrence {idx}: no partId; listed {len(parts)} source parts, exported {len(solid_parts)} solid parts"
            )
        except Exception as exc:
            report.append(f"failed occurrence {idx}: missing partId and could not expand source Part Studio: {exc}")

    report.append("")
    report.append(f"exportable part jobs={len(jobs)}")
    return jobs, report


def extract_mate_type(feature: Dict[str, Any]) -> Optional[str]:
    direct = get_first(feature, ["mateType", "type", "featureType"])
    if isinstance(direct, str) and direct.upper() in {"FASTENED", "SLIDER", "CYLINDRICAL", "REVOLUTE", "PIN_SLOT", "PLANAR", "BALL", "PARALLEL"}:
        return direct.upper()
    for d in walk_dicts(feature):
        pid = str(d.get("parameterId", "")).lower()
        if "matetype" in pid or "mate type" in str(d.get("parameterName", "")).lower():
            value = d.get("value")
            if isinstance(value, str):
                return value.upper()
    text = json.dumps(feature).upper()
    for kind in ("FASTENED", "SLIDER", "CYLINDRICAL", "REVOLUTE", "PIN_SLOT", "PLANAR", "BALL", "PARALLEL"):
        if kind in text:
            return kind
    return None


def extract_mate_paths(feature: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for d in walk_dicts(feature):
        path = d.get("path")
        if isinstance(path, list):
            key = path_to_key(path)
            if key and key not in paths:
                paths.append(key)
        fpas = d.get("fullPathAsString")
        if isinstance(fpas, str) and fpas and fpas not in paths:
            paths.append(fpas)
    return paths


def extract_axis_from_feature(feature: Dict[str, Any]) -> Vec3:
    # Try a transform-like matrix from a mate connector. Local Z is usually the connector axis.
    for d in walk_dicts(feature):
        transform = d.get("transform") or d.get("mateConnectorCS") or d.get("coordinateSystem")
        if isinstance(transform, list) and len(transform) >= 11 and all(isinstance(x, (int, float)) for x in transform[:12]):
            return v_unit((float(transform[2]), float(transform[6]), float(transform[10])))
    return (0.0, 0.0, 1.0)


def extract_numeric_limits(feature: Dict[str, Any]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    limits: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    lower: Optional[float] = None
    upper: Optional[float] = None
    for d in walk_dicts(feature):
        pid = str(d.get("parameterId", "")).lower()
        value = d.get("value")
        if not isinstance(value, (int, float)):
            continue
        if any(term in pid for term in ("min", "lower")):
            lower = float(value)
        if any(term in pid for term in ("max", "upper")):
            upper = float(value)
    if lower is not None or upper is not None:
        limits["primary"] = (lower, upper)
    return limits


def freedom_for_mate_type(mate_type: str, axis: Vec3, limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None) -> MotionFreedom:
    axis = v_unit(axis)
    x_axis: Vec3 = (1.0, 0.0, 0.0)
    y_axis: Vec3 = (0.0, 1.0, 0.0)
    z_axis: Vec3 = (0.0, 0.0, 1.0)
    mate_type = mate_type.upper()
    limits = limits or {}

    if mate_type == "FASTENED":
        return MotionFreedom([], [], mate_type, "mate", limits)
    if mate_type == "SLIDER":
        return MotionFreedom([axis], [], mate_type, "mate", limits)
    if mate_type == "REVOLUTE":
        return MotionFreedom([], [axis], mate_type, "mate", limits)
    if mate_type == "CYLINDRICAL":
        return MotionFreedom([axis], [axis], mate_type, "mate", limits)
    if mate_type == "PIN_SLOT":
        return MotionFreedom([axis], [axis], mate_type, "mate", limits)
    if mate_type == "PLANAR":
        # Approximate the plane as perpendicular to the mate connector Z axis.
        a = axis
        ref = x_axis if abs(v_dot(a, x_axis)) < 0.9 else y_axis
        t1 = v_unit(v_cross(a, ref), x_axis)
        t2 = v_unit(v_cross(a, t1), y_axis)
        return MotionFreedom([t1, t2], [axis], mate_type, "mate", limits)
    if mate_type == "BALL":
        return MotionFreedom([], [x_axis, y_axis, z_axis], mate_type, "mate", limits)
    if mate_type == "PARALLEL":
        return MotionFreedom([x_axis, y_axis, z_axis], [axis], mate_type, "mate", limits)
    return MotionFreedom([x_axis, y_axis, z_axis], [x_axis, y_axis, z_axis], mate_type or "FREE", "mate", limits)


def deduce_component_freedoms(assembly_def: Dict[str, Any], occurrences: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, MotionFreedom], List[str]]:
    occurrence_keys = []
    for occ in occurrences:
        occurrence_keys.append(path_to_key(occ.get("path")) or str(get_first(occ, ["fullPathAsString", "instanceId", "partId"], "")))

    freedoms: Dict[str, MotionFreedom] = {}
    report: List[str] = []

    grounded_terms = {"fixed", "grounded"}
    for occ, key in zip(occurrences, occurrence_keys):
        text = json.dumps(occ).lower()
        if any(term in text for term in grounded_terms) and ("true" in text or occ.get("fixed") is True or occ.get("isFixed") is True):
            freedoms[key] = MotionFreedom([], [], "GROUNDED", "grounded")
            report.append(f"{key or '<unknown>'}: grounded/fixed -> no motion")

    mate_features: List[Dict[str, Any]] = []
    for d in walk_dicts(assembly_def):
        feature_text = str(get_first(d, ["featureType", "btType", "type"], "")).lower()
        maybe_mate = "mate" in feature_text or extract_mate_type(d) is not None
        if maybe_mate:
            mt = extract_mate_type(d)
            if mt:
                mate_features.append(d)

    for feature in mate_features:
        mt = extract_mate_type(feature) or "UNKNOWN"
        axis = extract_axis_from_feature(feature)
        limits = extract_numeric_limits(feature)
        mate_paths = extract_mate_paths(feature)
        mf = freedom_for_mate_type(mt, axis, limits)
        report.append(
            f"Mate {get_first(feature, ['name', 'id', 'featureId'], '<unnamed>')}: "
            f"type={mt}, axis=({axis[0]:.4g},{axis[1]:.4g},{axis[2]:.4g}), paths={mate_paths or '[not decoded]'}"
        )
        for key in occurrence_keys:
            if key and any(key == p or key in p or p in key for p in mate_paths):
                if key not in freedoms or freedoms[key].source != "grounded":
                    freedoms[key] = mf

    free6 = MotionFreedom(
        translate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        rotate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        mate_type="FREE",
        source="unmated",
    )
    for key in occurrence_keys:
        if key not in freedoms:
            freedoms[key] = free6
            report.append(f"{key or '<unknown>'}: no decoded mate -> treated as separate free body")

    return freedoms, report


def occurrence_source_ids(ref: OnshapeRef, occ: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[str], Optional[str]]:
    """Return source document/version/element/part IDs for a resolved occurrence.

    The assembly definition's occurrence transform is object-to-world, while the
    instance metadata usually supplies documentId, elementId, partId and either
    documentMicroversion/versionId. Prefer microversion because assembly instances
    normally point at the exact source microversion used in the assembly.
    """
    source_did = str(get_first(occ, ["documentId", "sourceDocumentId", "did"], ref.did))
    source_eid = str(get_first(occ, ["elementId", "sourceElementId", "eid"], ref.eid))
    part_id_raw = get_first(occ, ["partId", "pid"])
    part_id = str(part_id_raw) if part_id_raw not in (None, "") else None
    configuration = get_first(occ, ["fullConfiguration", "configuration", "sourceConfiguration"], None)
    configuration = str(configuration) if configuration not in (None, "", "default") else None

    version_id = get_first(occ, ["documentVersion", "versionId", "sourceVersionId"])
    microversion_id = get_first(occ, ["documentMicroversion", "microversionId", "sourceMicroversionId", "documentMicroversionId"])
    workspace_id = get_first(occ, ["documentWorkspaceId", "workspaceId", "sourceWorkspaceId"])

    if microversion_id:
        return source_did, "m", str(microversion_id), source_eid, part_id, configuration
    if version_id:
        return source_did, "v", str(version_id), source_eid, part_id, configuration
    if workspace_id:
        return source_did, "w", str(workspace_id), source_eid, part_id, configuration
    # Last fallback: use whatever w/v/m came from the assembly URL. This only
    # works reliably for same-document source parts.
    return source_did, ref.wvm, ref.wvmid, source_eid, part_id, configuration


def components_from_occurrence_exports(ref: OnshapeRef, client: OnshapeClient, assembly_def: Dict[str, Any], bom_payload: Optional[Dict[str, Any]], workdir: Path) -> Optional[List[AeroComponent]]:
    occurrences, export_report = resolve_exportable_occurrences(ref, client, assembly_def)
    if not occurrences:
        (workdir / OCCURRENCE_EXPORT_REPORT_NAME).write_text("\n".join(export_report + ["", "No exportable occurrences were resolved."]) + "\n")
        return None

    freedoms, mate_report = deduce_component_freedoms(assembly_def, occurrences)
    raw_names = [str(get_first(o, ["name", "occurrenceName", "partName", "fullPathAsString", "partId", "instanceId"], f"part_{i+1}")) for i, o in enumerate(occurrences)]
    patches = unique_patch_names(raw_names)
    components: List[AeroComponent] = []
    failures: List[str] = []

    for i, (occ, patch) in enumerate(zip(occurrences, patches)):
        source_did, source_wvm, source_wvmid, source_eid, part_id, configuration = occurrence_source_ids(ref, occ)
        if not part_id:
            failures.append(f"{raw_names[i]}: still missing partId after occurrence resolution")
            continue
        try:
            part_stl = workdir / "parts" / f"{i+1:03d}_{patch}.stl"
            download_part_stl(ref, client, source_did, source_wvm, source_wvmid, source_eid, part_id, part_stl, configuration=configuration)
            tris = read_stl_triangles(part_stl)
            transform = occ.get("transform") if isinstance(occ.get("transform"), list) else list(mat_identity())
            tris = transform_triangles(tris, transform)
            aref, lref, cofr = component_references(tris)
            key = path_to_key(occ.get("path")) or str(get_first(occ, ["fullPathAsString", "_sourceInstanceId", "instanceId", "partId"], ""))
            components.append(AeroComponent(
                name=raw_names[i],
                patch=patch,
                triangles=tris,
                cofr=cofr,
                lref=lref,
                aref=aref,
                freedom=freedoms.get(key, MotionFreedom()),
                source_occurrence=key,
            ))
            export_report.append(
                f"EXPORTED {patch}: partId={part_id!r}, source={source_did}/{source_wvm}/{source_wvmid}/e/{source_eid}, "
                f"configuration={configuration or 'default'}, triangles={len(tris)}"
            )
        except Exception as exc:
            failures.append(f"{raw_names[i]}: {exc}")
            export_report.append(f"FAILED {patch}: {exc}")

    material_report = assign_materials_from_bom(components, occurrences, raw_names, bom_payload) if components else ["Assembly material/BOM report", "", "No components were created."]
    (workdir / MATERIAL_REPORT_NAME).write_text("\n".join(material_report) + "\n")

    report_path = workdir / MATE_REPORT_NAME
    lines = ["Assembly mate/freedom report", "", "Decoded mates/freedoms:"] + mate_report
    if failures:
        lines += ["", "Occurrence STL export failures:"] + failures
    report_path.write_text("\n".join(lines) + "\n")

    export_report += ["", f"successful component exports={len(components)}", f"failed component exports={len(failures)}"]
    if failures:
        export_report += ["", "Failures:"] + failures
    (workdir / OCCURRENCE_EXPORT_REPORT_NAME).write_text("\n".join(export_report) + "\n")

    if components:
        return components
    return None


def components_from_assembly_translation(ref: OnshapeRef, client: OnshapeClient, bom_payload: Optional[Dict[str, Any]], workdir: Path) -> List[AeroComponent]:
    translation = create_assembly_stl_translation(ref, client)
    finished = poll_translation(ref, client, translation)
    downloaded = download_translation_result(ref, client, finished, workdir / "assembly_export")

    stl_paths: List[Path] = []
    if zipfile.is_zipfile(downloaded):
        unzip_dir = workdir / "assembly_export_unzipped"
        unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(downloaded, "r") as zf:
            zf.extractall(unzip_dir)
        stl_paths = sorted([p for p in unzip_dir.rglob("*") if p.suffix.lower() == ".stl"])
    else:
        if downloaded.suffix.lower() != ".stl":
            renamed = downloaded.with_suffix(".stl")
            downloaded.rename(renamed)
            downloaded = renamed
        stl_paths = [downloaded]

    bodies: List[Tuple[str, List[Triangle]]] = []
    for p in stl_paths:
        bodies.extend(split_ascii_stl_solids(p))
    if not bodies:
        raise RuntimeError("Assembly STL export completed, but no STL bodies were found")

    names = [name or f"assembly_part_{i+1}" for i, (name, _tris) in enumerate(bodies)]
    patches = unique_patch_names(names)
    components: List[AeroComponent] = []
    free6 = MotionFreedom(
        translate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        rotate_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        mate_type="FREE",
        source="assembly-translation-fallback",
    )
    for patch, (name, tris) in zip(patches, bodies):
        aref, lref, cofr = component_references(tris)
        components.append(AeroComponent(name, patch, tris, cofr, lref, aref, free6))

    records = bom_records_from_payload(bom_payload) if USE_BOM_MATERIALS else []
    report = [
        "Assembly material/BOM report",
        "",
        "Geometry source=assembly translation fallback; exact occurrence-to-BOM matching may be approximate.",
        f"BOM candidate records found={len(records)}",
        "",
        "Applied material model:",
    ]
    for c in components:
        record = None
        best_score = 0
        for r in records:
            hay = normalize_material_text(" ".join(r.get("names", [])))
            name = normalize_material_text(c.name)
            score = 25 + min(len(name), 25) if name and name in hay else 0
            if score > best_score:
                best_score = score
                record = r
        if record and best_score >= 10:
            apply_material_model(c, record.get("material") or DEFAULT_MATERIAL_NAME, record.get("mass_kg"), record.get("density_kg_m3"), "bom/fuzzy")
        else:
            material = infer_material_from_name(c.name)
            apply_material_model(c, material, None, None, "name/default")
        report.append(
            f"- {c.patch}: name={c.name!r}, material={c.material.material_name!r}, "
            f"source={c.material.source}, density={c.material.density_kg_m3:.6g} kg/m^3, "
            f"volume={c.material.volume_m3:.6g} m^3, mass={c.mass:.6g} kg, "
            f"inertia={c.inertia:.6g} kg m^2"
        )
    (workdir / MATERIAL_REPORT_NAME).write_text("\n".join(report) + "\n")
    return components


def build_assembly_components(ref: OnshapeRef, client: OnshapeClient, workdir: Path) -> Tuple[List[AeroComponent], Dict[str, Any]]:
    assembly_def = get_assembly_definition(ref, client)
    bom_payload = get_assembly_bom(ref, client)
    if bom_payload is not None:
        (workdir / ASSEMBLY_BOM_NAME).write_text(json.dumps(bom_payload, indent=2, default=str))
    components = components_from_occurrence_exports(ref, client, assembly_def, bom_payload, workdir)
    if components:
        print(f"Exported {len(components)} separate assembly occurrence STL(s). Relative motion is enabled.")
        return components, assembly_def

    msg = (
        "Could not export individual occurrence STLs. This usually means the script could not resolve "
        "occurrence path -> instance -> partId/source Part Studio, or the API key cannot export one of the "
        "source parts. See actual_model_case/assembly_occurrence_export_report.txt after the run, or the "
        f"temporary {OCCURRENCE_EXPORT_REPORT_NAME} if debugging inside the script. "
        "A merged assembly STL cannot produce true relative part motion."
    )
    if STRICT_OCCURRENCE_EXPORT and not ALLOW_MERGED_ASSEMBLY_FALLBACK:
        raise RuntimeError(msg + " Set ALLOW_MERGED_ASSEMBLY_FALLBACK=1 only if you just want a merged static visualization.")

    print(msg)
    print("Falling back to merged assembly STL translation/splitting because ALLOW_MERGED_ASSEMBLY_FALLBACK is enabled or strict mode is disabled.")
    return components_from_assembly_translation(ref, client, bom_payload, workdir), assembly_def


# ------------------------- quasi-dynamic assembly motion -------------------------


