"""Upload RefMods to Hugging Face and remember what is already there."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .config import APP_ROOT, load_config
from .library import list_studio_mods, resolve_mod_path

REGISTRY_PATH = APP_ROOT / "data" / "huggingface.json"
TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class HubError(ValueError):
    pass


@dataclass
class HubJob:
    status: str = "running"  # running | done | error
    log: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    done: int = 0
    total: int = 0


_job_lock = threading.Lock()
_job: HubJob | None = None
_upload_file: Callable[..., str] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        "sha256": file_sha256(path),
    }


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.is_file():
        return {"uploads": {}}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"uploads": {}}
    if not isinstance(data, dict):
        return {"uploads": {}}
    uploads = data.get("uploads")
    if not isinstance(uploads, dict):
        uploads = {}
    return {"uploads": uploads}


def save_registry(data: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"uploads": data.get("uploads") if isinstance(data.get("uploads"), dict) else {}}
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def hub_settings() -> dict[str, Any]:
    cfg = load_config()
    token = str(cfg.get("hf_token") or "").strip()
    if not token:
        for name in TOKEN_ENV:
            token = str(os.environ.get(name) or "").strip()
            if token:
                break
    repo = str(cfg.get("hf_repo") or "").strip()
    repo_type = str(cfg.get("hf_repo_type") or "dataset").strip() or "dataset"
    if repo_type not in ("dataset", "model"):
        repo_type = "dataset"
    private = str(cfg.get("hf_private") or "true").strip().lower() in ("1", "true", "yes", "on")
    return {
        "token": token,
        "repo_id": repo,
        "repo_type": repo_type,
        "private": private,
        "ready": bool(token and repo),
        "missing": [part for part, ok in (("Hugging Face token", bool(token)), ("Hugging Face repo (user/name)", bool(repo))) if not ok],
    }


def hub_file_url(repo_id: str, repo_type: str, path_in_repo: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path_in_repo.replace("\\", "/").split("/") if part)
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}/blob/main/{encoded}"
    return f"https://huggingface.co/{repo_id}/blob/main/{encoded}"


def record_upload(mod_id: str, path: Path, settings: dict[str, Any], path_in_repo: str) -> dict[str, Any]:
    entry = {
        "id": mod_id,
        "repo_id": settings["repo_id"],
        "repo_type": settings["repo_type"],
        "path_in_repo": path_in_repo,
        "uploaded_at": utc_now(),
        "url": hub_file_url(settings["repo_id"], settings["repo_type"], path_in_repo),
        **file_fingerprint(path),
    }
    data = load_registry()
    data["uploads"][mod_id] = entry
    save_registry(data)
    return entry


def remap_hub_ids(old: str, new: str) -> int:
    old = (old or "").strip("/")
    new = (new or "").strip("/")
    if not old or old == new:
        return 0
    data = load_registry()
    uploads = data["uploads"]
    updated = 0
    for key in list(uploads):
        if key != old and not key.startswith(old + "/"):
            continue
        nxt = new + key[len(old) :] if key != old else new
        entry = dict(uploads.pop(key))
        entry["id"] = nxt
        path_in_repo = str(entry.get("path_in_repo") or "")
        if path_in_repo.startswith(old + "/") or path_in_repo.startswith(old + "."):
            entry["path_in_repo"] = new + path_in_repo[len(old) :]
        if entry.get("url") and old in str(entry["url"]):
            entry["url"] = hub_file_url(
                str(entry.get("repo_id") or ""),
                str(entry.get("repo_type") or "dataset"),
                str(entry.get("path_in_repo") or f"{nxt}.safetensors"),
            )
        uploads[nxt] = entry
        updated += 1
    if updated:
        save_registry(data)
    return updated


def hub_status_for(mod_id: str, path: Path | None, uploads: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = uploads if uploads is not None else load_registry()["uploads"]
    entry = registry.get(mod_id) if isinstance(registry, dict) else None
    if not isinstance(entry, dict):
        return {"status": "none"}
    info = {
        "status": "uploaded",
        "repo_id": entry.get("repo_id") or "",
        "path_in_repo": entry.get("path_in_repo") or "",
        "url": entry.get("url") or "",
        "uploaded_at": entry.get("uploaded_at") or "",
    }
    if path is None or not path.is_file():
        return info
    try:
        stat = path.stat()
    except OSError:
        return info
    size = int(stat.st_size)
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    if size == int(entry.get("size") or -1) and mtime_ns == int(entry.get("mtime_ns") or -1):
        return info
    info["status"] = "stale"
    return info


def annotate_mods(mods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uploads = load_registry().get("uploads") or {}
    for mod in mods:
        mod_id = str(mod.get("id") or "")
        path = None
        for key in ("path", "comfy_path", "local_path"):
            raw = mod.get(key) or ""
            if raw:
                candidate = Path(str(raw))
                if candidate.is_file():
                    path = candidate
                    break
        mod["hf"] = hub_status_for(mod_id, path, uploads=uploads) if mod_id else {"status": "none"}
    return mods


def resolve_upload_source(mod_id: str) -> Path:
    try:
        path = resolve_mod_path(mod_id)
        if path.is_file():
            return path
    except (FileNotFoundError, ValueError):
        pass
    for item in list_studio_mods():
        if str(item.get("id") or "") != mod_id:
            continue
        for key in ("comfy_path", "local_path"):
            candidate = Path(str(item.get(key) or ""))
            if candidate.is_file():
                return candidate
    raise HubError(f"RefMod not found: {mod_id}")


def path_in_repo(mod_id: str) -> str:
    rel = (mod_id or "").replace("\\", "/").strip().strip("/")
    if not rel:
        raise HubError("invalid mod id")
    if not rel.lower().endswith(".safetensors"):
        rel = f"{rel}.safetensors"
    return rel


def current_job() -> HubJob | None:
    return _job


def set_uploader(fn: Callable[..., str] | None) -> None:
    global _upload_file
    _upload_file = fn


def _default_upload(path: Path, dest: str, settings: dict[str, Any]) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise HubError("huggingface_hub is not installed — pip install huggingface_hub") from exc
    api = HfApi(token=settings["token"])
    api.create_repo(
        repo_id=settings["repo_id"],
        repo_type=settings["repo_type"],
        private=settings["private"],
        exist_ok=True,
        token=settings["token"],
    )
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=dest,
        repo_id=settings["repo_id"],
        repo_type=settings["repo_type"],
        token=settings["token"],
        commit_message=f"Add RefMod {dest}",
    )
    return hub_file_url(settings["repo_id"], settings["repo_type"], dest)


def upload_one(mod_id: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = settings or hub_settings()
    if not cfg["ready"]:
        raise HubError("Hugging Face is not configured: " + "; ".join(cfg["missing"]))
    source = resolve_upload_source(mod_id)
    dest = path_in_repo(mod_id)
    uploader = _upload_file or _default_upload
    url = uploader(source, dest, cfg)
    entry = record_upload(mod_id, source, cfg, dest)
    if url:
        entry["url"] = url
        data = load_registry()
        data["uploads"][mod_id] = entry
        save_registry(data)
    return entry


def start_upload(ids: list[str]) -> HubJob:
    global _job
    clean = []
    seen: set[str] = set()
    for raw in ids:
        key = str(raw or "").replace("\\", "/").strip().strip("/")
        if key.lower().endswith(".safetensors"):
            key = key[: -len(".safetensors")]
        if not key or key in seen:
            continue
        seen.add(key)
        clean.append(key)
    if not clean:
        raise HubError("select at least one RefMod")
    settings = hub_settings()
    if not settings["ready"]:
        raise HubError("Hugging Face is not configured: " + "; ".join(settings["missing"]))
    with _job_lock:
        if _job is not None and _job.status == "running":
            raise HubError("an upload is already running")
        job = HubJob(ids=clean, total=len(clean))
        job.log.append(f"[hub] uploading {len(clean)} RefMod(s) to {settings['repo_id']}")
        _job = job

        def runner() -> None:
            try:
                for mod_id in clean:
                    job.log.append(f"[hub] {mod_id}")
                    try:
                        entry = upload_one(mod_id, settings)
                        job.results.append(entry)
                        job.log.append(f"[hub] uploaded {mod_id} -> {entry.get('url') or entry.get('path_in_repo')}")
                    except Exception as exc:  # noqa: BLE001
                        job.results.append({"id": mod_id, "error": str(exc)})
                        job.log.append(f"[hub] error {mod_id}: {exc}")
                    job.done += 1
                failed = [item for item in job.results if item.get("error")]
                if failed and len(failed) == len(clean):
                    raise HubError(f"all {len(failed)} upload(s) failed")
                job.status = "done"
                if failed:
                    job.log.append(f"[hub] finished with {len(failed)} error(s)")
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = str(exc)
                job.log.append(f"[hub] error: {exc}")

        threading.Thread(target=runner, daemon=True).start()
        return job
