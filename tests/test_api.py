import json

from fastapi.testclient import TestClient

from app import app
from backend.datasets import create_dataset
from tests.test_split import sheet_png


def test_create_list_dataset_api(dataset_root):
    client = TestClient(app)
    res = client.post("/api/datasets", json={"name": "Agusia", "display_name": "Agusia"})
    assert res.status_code == 200
    body = res.json()
    assert body["slug"] == "agusia"
    assert body["mod_name"] == "sdprompts_minimaxh3_agusia_v1_refmod"

    listed = client.get("/api/datasets").json()["datasets"]
    assert any(d["slug"] == "agusia" for d in listed)

    got = client.get("/api/datasets/agusia").json()
    assert got["display_name"] == "Agusia"
    assert got["description"] == "Agusia"
    assert got["subfolder"] == "identity"


def test_create_dataset_with_folder(dataset_root):
    client = TestClient(app)
    res = client.post("/api/datasets", json={"name": "June", "subfolder": "celebs/june"})
    assert res.status_code == 200
    assert res.json()["subfolder"] == "celebs/june"


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert "ready" in body
    assert "concept_types" in body
    assert "identity" in body["concept_types"]
    assert "pack_version" in body
    assert "refmod_folders" in body
    assert "voice_ready" not in body
    assert "hub_ready" in body
    assert body.get("config", {}).get("hf_token", "") == ""


def test_studio_endpoint_empty(dataset_root):
    client = TestClient(app)
    res = client.get("/api/studio")
    assert res.status_code == 200
    body = res.json()
    assert body["mods"] == []
    assert body["folders"] == []


def test_library_lite_query(dataset_root, tmp_path, monkeypatch):
    from tests.test_library import _write_mod

    comfy = tmp_path / "refmods"
    _write_mod(comfy / "identity" / "hero.safetensors", "hero")
    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: comfy)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)
    monkeypatch.setattr("backend.discover.default_output_dir", lambda _root=None: comfy)
    client = TestClient(app)
    lite = client.get("/api/library", params={"lite": True})
    assert lite.status_code == 200, lite.text
    body = lite.json()
    assert body["lite"] is True
    assert any(m["id"] == "identity/hero" for m in body["mods"])
    full = client.get("/api/library")
    assert full.status_code == 200
    assert full.json()["lite"] is False


def test_sync_comfy_api(dataset_root, tmp_path, monkeypatch):
    from backend.datasets import default_mod_name
    from tests.test_library import _write_mod

    comfy = tmp_path / "refmods"
    comfy.mkdir()
    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: comfy)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)
    create_dataset("June", "June", subfolder="identity")
    mod_name = default_mod_name("june")
    _write_mod(dataset_root / "june" / f"{mod_name}.safetensors", mod_name)
    client = TestClient(app)
    res = client.post("/api/studio/sync-comfy")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["copied"] == 1
    assert (comfy / "identity" / f"{mod_name}.safetensors").is_file()


def test_patch_dataset_cover(dataset_root):
    create_dataset("shots")
    folder = dataset_root / "shots"
    (folder / "a.png").write_bytes(b"\x89PNG\r\n" + b"aaaa")
    (folder / "b.png").write_bytes(b"\x89PNG\r\n" + b"bbbb")
    client = TestClient(app)
    listed = client.get("/api/datasets/shots").json()
    assert listed["cover"] == "a.png"
    res = client.patch("/api/datasets/shots", json={"cover": "b.png"})
    assert res.status_code == 200
    assert res.json()["cover"] == "b.png"
    bad = client.patch("/api/datasets/shots", json={"cover": "nope.png"})
    assert bad.status_code == 400


def test_add_file_and_caption(dataset_root):
    create_dataset("shots")
    client = TestClient(app)
    res = client.post(
        "/api/datasets/shots/files",
        files=[("files", ("face.png", b"\x89PNG\r\nxxxx", "image/png"))],
    )
    assert res.status_code == 200
    assert res.json()["image_count"] == 1
    cap = client.put(
        "/api/datasets/shots/files/face.png/caption",
        json={"caption": "front portrait"},
    )
    assert cap.status_code == 200
    assert cap.json()["items"][0]["caption"] == "front portrait"


def test_split_sheet_api(dataset_root):
    client = TestClient(app)
    res = client.post(
        "/api/datasets/split-sheet",
        data={"name": "krea-set", "display_name": "Krea Set"},
        files=[("files", ("sheet.png", sheet_png(2, 2), "image/png"))],
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["datasets"]) == 1
    card = body["datasets"][0]
    assert card["slug"] == "krea-set"
    assert card["display_name"] == "Krea Set"
    assert card["description"] == "Krea Set"
    assert card["image_count"] == 4
    listed = client.get("/api/datasets").json()["datasets"]
    assert any(d["slug"] == "krea-set" for d in listed)


def test_split_sheet_preview_api(dataset_root):
    client = TestClient(app)
    res = client.post(
        "/api/datasets/split-sheet/preview",
        files=[("files", ("sheet.png", sheet_png(2, 2), "image/png"))],
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["sheets"]) == 1
    sheet = body["sheets"][0]
    assert sheet["filename"] == "sheet.png"
    assert sheet["preview"].startswith("data:image/jpeg;base64,")
    assert sheet["methods"]
    assert any(m["count"] == 4 for m in sheet["methods"])


def test_split_sheet_api_uses_plans(dataset_root):
    client = TestClient(app)
    preview = client.post(
        "/api/datasets/split-sheet/preview",
        files=[("files", ("sheet.png", sheet_png(2, 2), "image/png"))],
    ).json()["sheets"][0]
    method = next(m for m in preview["methods"] if m["count"] == 4)
    plans = json.dumps([{"filename": "sheet.png", "boxes": method["boxes"][:2]}])
    res = client.post(
        "/api/datasets/split-sheet",
        data={"name": "two-cuts", "display_name": "Two", "plans": plans},
        files=[("files", ("sheet.png", sheet_png(2, 2), "image/png"))],
    )
    assert res.status_code == 200, res.text
    assert res.json()["datasets"][0]["image_count"] == 2


def test_split_sheet_api_one_dataset_per_file(dataset_root):
    client = TestClient(app)
    res = client.post(
        "/api/datasets/split-sheet",
        data={"name": "first", "display_name": "First"},
        files=[
            ("files", ("one.png", sheet_png(2, 2), "image/png")),
            ("files", ("two.png", sheet_png(2, 1), "image/png")),
        ],
    )
    assert res.status_code == 200, res.text
    slugs = [d["slug"] for d in res.json()["datasets"]]
    assert slugs == ["first", "two"]
