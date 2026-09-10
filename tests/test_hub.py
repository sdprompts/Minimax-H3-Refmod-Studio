from pathlib import Path

from backend import config
from backend import hub as hubmod
from backend.hub import (
    annotate_mods,
    hub_status_for,
    path_in_repo,
    record_upload,
    remap_hub_ids,
    start_upload,
    upload_one,
)


def test_public_config_masks_token(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setattr(config, "DEFAULT_DATASETS_ROOT", tmp_path / "datasets")
    config.save_config({"hf_token": "hf_secret", "hf_repo": "me/refmods"})
    public = config.public_config()
    assert public["hf_token"] == ""
    assert public["hf_token_set"] is True
    assert public["hf_repo"] == "me/refmods"
    config.save_config({"hf_token": ""})
    assert config.load_config()["hf_token"] == "hf_secret"


def test_record_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(hubmod, "REGISTRY_PATH", tmp_path / "huggingface.json")
    path = tmp_path / "identity"
    path.mkdir()
    mod = path / "hero.safetensors"
    mod.write_bytes(b"abc123")
    settings = {"repo_id": "me/refmods", "repo_type": "dataset", "private": True, "token": "x"}
    entry = record_upload("identity/hero", mod, settings, "identity/hero.safetensors")
    assert entry["url"].endswith("identity/hero.safetensors")
    assert hub_status_for("identity/hero", mod)["status"] == "uploaded"
    mod.write_bytes(b"changed")
    assert hub_status_for("identity/hero", mod)["status"] == "stale"
    annotated = annotate_mods([{"id": "identity/hero", "path": str(mod)}])
    assert annotated[0]["hf"]["status"] == "stale"


def test_remap_hub_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(hubmod, "REGISTRY_PATH", tmp_path / "huggingface.json")
    path = tmp_path / "mod.safetensors"
    path.write_bytes(b"data")
    settings = {"repo_id": "me/refmods", "repo_type": "dataset", "private": True, "token": "x"}
    record_upload("identity/hero", path, settings, "identity/hero.safetensors")
    assert remap_hub_ids("identity", "celebs") == 1
    status = hub_status_for("celebs/hero", path)
    assert status["status"] == "uploaded"
    assert "celebs/hero.safetensors" in status["path_in_repo"]


def test_path_in_repo():
    assert path_in_repo("identity/hero") == "identity/hero.safetensors"


def test_upload_job_records(tmp_path, monkeypatch):
    monkeypatch.setattr(hubmod, "REGISTRY_PATH", tmp_path / "huggingface.json")
    monkeypatch.setattr(hubmod, "hub_settings", lambda: {
        "token": "x",
        "repo_id": "me/refmods",
        "repo_type": "dataset",
        "private": True,
        "ready": True,
        "missing": [],
    })
    source = tmp_path / "hero.safetensors"
    source.write_bytes(b"payload")
    monkeypatch.setattr(hubmod, "resolve_upload_source", lambda _id: source)

    def fake_upload(path, dest, settings):
        assert path == source
        assert dest == "identity/hero.safetensors"
        return "https://huggingface.co/datasets/me/refmods/blob/main/identity/hero.safetensors"

    hubmod.set_uploader(fake_upload)
    hubmod._job = None
    try:
        entry = upload_one("identity/hero")
        assert entry["id"] == "identity/hero"
        job = start_upload(["identity/hero"])
        for _ in range(50):
            if job.status != "running":
                break
            import time
            time.sleep(0.02)
        assert job.status == "done"
        assert job.results
    finally:
        hubmod.set_uploader(None)
