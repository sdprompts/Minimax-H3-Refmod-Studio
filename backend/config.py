"""Load and save local app settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_ROOT / "config.json"
DEFAULT_DATASETS_ROOT = APP_ROOT / "datasets"
THUMBS_DIR = APP_ROOT / "data" / "thumbs"
LINKS_NAME = "_links.json"

DEFAULTS: dict[str, Any] = {
    "comfy_root": "",
    "python_path": "",
    "vae_path": "",
    "output_dir": "",
    "datasets_root": str(DEFAULT_DATASETS_ROOT),
    "extract_script": "",
    "mod_prefix": "sdprompts",
    "hf_token": "",
    "hf_repo": "",
    "hf_repo_type": "dataset",
    "hf_private": "true",
}
SECRET_KEYS = ("hf_token",)


def load_config() -> dict[str, Any]:
    data = dict(DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for key in DEFAULTS:
                    if key in saved and saved[key] is not None:
                        data[key] = saved[key]
        except (OSError, json.JSONDecodeError):
            pass
    Path(data["datasets_root"]).mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    return data


def save_config(updates: dict[str, Any]) -> dict[str, Any]:
    data = load_config()
    for key in DEFAULTS:
        if key not in updates or updates[key] is None:
            continue
        if key == "hf_token":
            token = str(updates[key]).strip()
            if not token or set(token) <= {"•"}:
                continue
            data[key] = token
            continue
        if key == "hf_private":
            data[key] = "true" if str(updates[key]).strip().lower() in ("1", "true", "yes", "on") else "false"
            continue
        data[key] = str(updates[key]).strip()
    if data["datasets_root"]:
        Path(data["datasets_root"]).mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def public_config() -> dict[str, Any]:
    data = load_config()
    out = dict(data)
    for key in SECRET_KEYS:
        out[f"{key}_set"] = bool(str(out.get(key) or "").strip())
        out[key] = ""
    return out


def datasets_root() -> Path:
    root = Path(load_config()["datasets_root"]).expanduser()
    if not root.is_absolute():
        root = (APP_ROOT / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def links_path() -> Path:
    return datasets_root() / LINKS_NAME


def load_links() -> dict[str, str]:
    path = links_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if k and v}


def save_links(links: dict[str, str]) -> None:
    path = links_path()
    path.write_text(json.dumps(links, indent=2) + "\n", encoding="utf-8")
