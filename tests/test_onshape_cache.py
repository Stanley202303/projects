from pathlib import Path

from cfd_motion import onshape


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.payload


class _Opener:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls = 0

    def open(self, _request):
        self.calls += 1
        return _Response(self.payload)


def _client(tmp_path: Path, monkeypatch, payload: bytes = b"cached-stl"):
    monkeypatch.setattr(onshape, "ONSHAPE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(onshape, "ONSHAPE_CACHE_ENABLED", True)
    monkeypatch.setattr(onshape, "ONSHAPE_CACHE_REFRESH", False)
    monkeypatch.setattr(onshape, "ONSHAPE_CACHE_TTL_S", 3600.0)
    client = onshape.OnshapeClient("access", "secret")
    opener = _Opener(payload)
    client.opener = opener
    return client, opener


def test_successful_get_is_reused_without_second_request(tmp_path, monkeypatch) -> None:
    client, opener = _client(tmp_path, monkeypatch)
    url = "https://cad.onshape.com/api/v14/partstudios/d/doc/e/element/stl?b=2&a=1"

    assert client.request_bytes("GET", url) == b"cached-stl"
    assert client.request_bytes(
        "GET",
        "https://CAD.ONSHAPE.COM/api/v14/partstudios/d/doc/e/element/stl?a=1&b=2",
    ) == b"cached-stl"
    assert opener.calls == 1
    assert len(list(tmp_path.glob("*.response"))) == 1


def test_cache_refresh_forces_a_new_get(tmp_path, monkeypatch) -> None:
    client, opener = _client(tmp_path, monkeypatch)
    url = "https://cad.onshape.com/api/v14/documents/d"
    assert client.request_bytes("GET", url) == b"cached-stl"
    monkeypatch.setattr(onshape, "ONSHAPE_CACHE_REFRESH", True)
    assert client.request_bytes("GET", url) == b"cached-stl"
    assert opener.calls == 2


def test_post_requests_are_never_cached(tmp_path, monkeypatch) -> None:
    client, opener = _client(tmp_path, monkeypatch, payload=b"post-result")
    url = "https://cad.onshape.com/api/v14/translations"
    assert client.request_bytes("POST", url, body=b"{}") == b"post-result"
    assert client.request_bytes("POST", url, body=b"{}") == b"post-result"
    assert opener.calls == 2
    assert not list(tmp_path.glob("*.response"))
