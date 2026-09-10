"""Named dataset folders, manifests, import/link, captions, dirty hashing."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import datasets_root, load_links, save_links
from .library import infer_subfolder_from_output, normalize_subfolder
from .media import IMAGE_EXTS, VIDEO_EXTS, is_image, is_media, is_video, probe_size
from .split import SplitError, SplitOptions, crop_to_pngs, split_to_pngs

CONCEPT_TYPES = (
    "identity",
    "generic",
    "pose_motion",
    "clothing",
    "background",
    "style",
)
SHOT_TYPES = ("", "front", "three-quarter", "profile", "full-body")
CLEAN_STATES = ("", "flagged", "cleaned", "ignored")
MANIFEST_NAME = "dataset.json"
ORIGINALS_DIR = "_originals"
DEFAULT_EXTRACT = {
    "mode": "encode",
    "resolution": 1024,
    "max_tokens": 8192,
    "pool": 16,
    "identity": 500,
    "multiplier": 1,
    "device": "auto",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_VERSION_RE = re.compile(r"_v(\d+)_refmod$", re.IGNORECASE)


class DatasetError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or "dataset"


def default_mod_name(slug: str, version: int = 1, prefix: str | None = None) -> str:
    if prefix is None:
        from .config import load_config
        prefix = str(load_config().get("mod_prefix") or "sdprompts")
    prefix = re.sub(r"[^\w]+", "", str(prefix).strip()) or "sdprompts"
    clean = slugify(slug).replace("-", "")
    return f"{prefix}_minimaxh3_{clean}_v{version}_refmod"


def bump_mod_name(mod_name: str) -> str:
    name = (mod_name or "").strip()
    if not name:
        return default_mod_name("dataset", 2)
    match = _VERSION_RE.search(name)
    if match:
        nxt = int(match.group(1)) + 1
        return _VERSION_RE.sub(f"_v{nxt}_refmod", name)
    if name.endswith("_refmod"):
        return name[: -len("_refmod")] + "_v2_refmod"
    return name + "_v2_refmod"


def _safe_filename(name: str) -> str:
    base = Path(name).name
    if not base or base in {".", ".."}:
        raise DatasetError("invalid filename")
    if "/" in base or "\\" in base:
        raise DatasetError("invalid filename")
    return base


def _caption_path(folder: Path, filename: str) -> Path:
    return folder / (Path(filename).stem + ".txt")


def read_caption(folder: Path, filename: str) -> str:
    path = _caption_path(folder, filename)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_caption(folder: Path, filename: str, text: str) -> None:
    path = _caption_path(folder, filename)
    body = (text or "").strip()
    if not body:
        if path.is_file():
            path.unlink()
        return
    path.write_text(body + "\n", encoding="utf-8")


def list_media_files(folder: Path, recursive: bool = False) -> list[Path]:
    if not folder.is_dir():
        return []
    iterator: Iterable[Path]
    if recursive:
        iterator = folder.rglob("*")
    else:
        iterator = folder.iterdir()
    files = [
        p for p in iterator
        if p.is_file() and is_media(p) and p.name.lower() != MANIFEST_NAME
    ]
    files.sort(key=lambda p: p.name.lower())
    return files


def empty_manifest(slug: str, display_name: str | None = None) -> dict[str, Any]:
    name = slugify(slug)
    shown = (display_name or slug).strip() or name
    return {
        "name": name,
        "display_name": shown,
        "concept_type": "identity",
        "description": shown,
        "mod_name": default_mod_name(name),
        "extract": dict(DEFAULT_EXTRACT),
        "items": [],
        "last_extract": None,
        "linked": False,
        "subfolder": "identity",
        "cover": "",
        "notes": "",
        "quality_advice": [],
        "folder": "",
    }


def _normalize_item(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        filename = raw
        shot = ""
        enabled = True
    elif isinstance(raw, dict):
        filename = str(raw.get("file") or "")
        shot = str(raw.get("shot") or "")
        enabled = bool(raw.get("enabled", True))
    else:
        return None
    filename = Path(filename).name
    if not filename:
        return None
    if shot not in SHOT_TYPES:
        shot = ""
    clean = ""
    quality = None
    if isinstance(raw, dict):
        clean = str(raw.get("clean") or "").strip().lower()
        if clean not in CLEAN_STATES:
            clean = ""
        quality = _normalize_quality(raw.get("quality"))
    item = {"file": filename, "shot": shot, "enabled": enabled, "clean": clean}
    if quality is not None:
        item["quality"] = quality
    return item


def _normalize_quality(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    reason = str(raw.get("reason") or "").strip()
    score_raw = raw.get("score")
    score: float | None
    try:
        score = None if score_raw is None else float(score_raw)
    except (TypeError, ValueError):
        score = None
    debug = raw.get("debug") if isinstance(raw.get("debug"), dict) else {}
    at = str(raw.get("at") or "")
    if score is None and not reason:
        return None
    return {
        "score": score,
        "reason": reason,
        "debug": debug,
        "picked": bool(raw.get("picked")),
        "at": at,
    }


def _normalize_advice(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        level = str(item.get("level") or "improve").strip()
        if level not in {"need", "improve", "ok"}:
            level = "improve"
        out.append({
            "id": str(item.get("id") or ""),
            "level": level,
            "text": text,
        })
    return out


def _normalize_cover(raw: Any) -> str:
    name = Path(str(raw or "")).name
    if not name or name in {".", ".."}:
        return ""
    return name


def resolve_cover(manifest: dict[str, Any], folder: Path) -> str | None:
    images: list[tuple[str, bool]] = []
    for item in manifest.get("items") or []:
        name = item.get("file") or ""
        path = folder / name
        if name and path.is_file() and is_image(path):
            images.append((name, bool(item.get("enabled", True))))
    if not images:
        return None
    chosen = _normalize_cover(manifest.get("cover"))
    names = {name for name, _ in images}
    if chosen in names:
        return chosen
    for name, enabled in images:
        if enabled:
            return name
    return images[0][0]


def _merge_extract(raw: Any) -> dict[str, Any]:
    out = dict(DEFAULT_EXTRACT)
    if isinstance(raw, dict):
        for key in DEFAULT_EXTRACT:
            if key in raw and raw[key] is not None and raw[key] != "":
                out[key] = raw[key]
    try:
        out["resolution"] = int(out["resolution"])
        out["max_tokens"] = int(out["max_tokens"])
        out["pool"] = int(out["pool"])
        out["identity"] = int(out["identity"])
        out["multiplier"] = int(out["multiplier"])
    except (TypeError, ValueError) as exc:
        raise DatasetError("extract settings must be numeric") from exc
    mode = str(out["mode"]).lower()
    if mode in ("full",):
        mode = "encode"
    if mode in ("pooled", "optimize", "svd"):
        mode = "training"
    if mode not in ("encode", "training"):
        mode = "encode"
    out["mode"] = mode
    if str(out.get("device") or "auto") not in ("auto", "cuda", "cpu"):
        out["device"] = "auto"
    return out


def normalize_manifest(data: dict[str, Any], slug: str, folder: Path) -> dict[str, Any]:
    base = empty_manifest(slug, data.get("display_name"))
    concept = str(data.get("concept_type") or "identity")
    if concept not in CONCEPT_TYPES:
        concept = "identity"
    items = []
    seen: set[str] = set()
    for raw in data.get("items") or []:
        item = _normalize_item(raw)
        if item is None or item["file"] in seen:
            continue
        seen.add(item["file"])
        items.append(item)
    last = data.get("last_extract")
    if last is not None and not isinstance(last, dict):
        last = None
    if "subfolder" in data:
        try:
            subfolder = normalize_subfolder(str(data.get("subfolder") or ""))
        except ValueError:
            subfolder = concept
    else:
        inferred = infer_subfolder_from_output(
            str(last.get("output") or "") if isinstance(last, dict) else ""
        )
        if inferred is None:
            subfolder = concept
        else:
            subfolder = inferred
    base.update({
        "name": slugify(str(data.get("name") or slug)),
        "display_name": str(data.get("display_name") or slug).strip() or slug,
        "concept_type": concept,
        "description": str(data.get("description") or ""),
        "mod_name": str(data.get("mod_name") or default_mod_name(slug)).strip() or default_mod_name(slug),
        "extract": _merge_extract(data.get("extract")),
        "items": items,
        "last_extract": last,
        "linked": bool(data.get("linked")),
        "subfolder": subfolder,
        "cover": _normalize_cover(data.get("cover")),
        "notes": str(data.get("notes") or ""),
        "quality_advice": _normalize_advice(data.get("quality_advice")),
        "folder": str(folder),
    })
    return base


def reconcile_items(manifest: dict[str, Any], folder: Path) -> dict[str, Any]:
    existing = {p.name: p for p in list_media_files(folder)}
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in manifest.get("items") or []:
        name = item.get("file")
        if name in existing and name not in seen:
            kept.append(item)
            seen.add(name)
    for name in sorted(existing, key=str.lower):
        if name not in seen:
            kept.append({"file": name, "shot": "", "enabled": True, "clean": ""})
    manifest["items"] = kept
    return manifest


def content_hash(manifest: dict[str, Any], folder: Path) -> str:
    import hashlib

    extract = _merge_extract(manifest.get("extract"))
    payload: dict[str, Any] = {
        "mod_name": manifest.get("mod_name") or "",
        "concept_type": manifest.get("concept_type") or "",
        "subfolder": manifest.get("subfolder") or "",
        "description": manifest.get("description") or "",
        "extract": {
            "mode": extract["mode"],
            "resolution": extract["resolution"],
            "max_tokens": extract["max_tokens"],
            "pool": extract["pool"],
            "identity": extract["identity"],
            "multiplier": extract["multiplier"],
        },
        "items": [],
    }
    for item in manifest.get("items") or []:
        if not item.get("enabled", True):
            continue
        path = folder / item["file"]
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        payload["items"].append({
            "file": item["file"],
            "size": stat.st_size,
            "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)),
        })
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def is_dirty(manifest: dict[str, Any], folder: Path) -> bool:
    last = manifest.get("last_extract") or {}
    stored = last.get("content_hash") if isinstance(last, dict) else None
    if not stored:
        return bool(enabled_files(manifest, folder))
    return stored != content_hash(manifest, folder)


def enabled_files(manifest: dict[str, Any], folder: Path) -> list[Path]:
    out: list[Path] = []
    for item in manifest.get("items") or []:
        if not item.get("enabled", True):
            continue
        path = folder / item["file"]
        if path.is_file() and is_media(path):
            out.append(path)
    return out


def load_manifest(folder: Path, slug: str) -> dict[str, Any]:
    path = folder / MANIFEST_NAME
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    manifest = normalize_manifest(data, slug, folder)
    return reconcile_items(manifest, folder)


def save_manifest(folder: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    slug = slugify(str(manifest.get("name") or folder.name))
    clean = normalize_manifest(manifest, slug, folder)
    to_write = {
        "name": clean["name"],
        "display_name": clean["display_name"],
        "concept_type": clean["concept_type"],
        "description": clean["description"],
        "mod_name": clean["mod_name"],
        "extract": clean["extract"],
        "items": clean["items"],
        "last_extract": clean["last_extract"],
        "linked": clean["linked"],
        "subfolder": clean.get("subfolder") or "",
        "cover": clean.get("cover") or "",
        "notes": clean.get("notes") or "",
        "quality_advice": clean.get("quality_advice") or [],
    }
    (folder / MANIFEST_NAME).write_text(
        json.dumps(to_write, indent=2) + "\n", encoding="utf-8"
    )
    return load_manifest(folder, slug)


def _dataset_map() -> dict[str, Path]:
    root = datasets_root()
    mapping: dict[str, Path] = {}
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("_"):
                continue
            if child.is_dir():
                mapping[slugify(child.name)] = child.resolve()
    for slug, raw in load_links().items():
        path = Path(raw).expanduser()
        if path.is_dir():
            mapping[slugify(slug)] = path.resolve()
    return mapping


def resolve_dataset(slug: str) -> tuple[str, Path]:
    key = slugify(slug)
    mapping = _dataset_map()
    if key not in mapping:
        raise DatasetError(f"dataset '{key}' not found")
    return key, mapping[key]


def item_payload(folder: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = folder / item["file"]
    kind = "image" if is_image(path) else "video" if is_video(path) else "other"
    size = None
    short_edge = None
    if kind == "image":
        probed = probe_size(path)
        if probed:
            size = {"w": probed[0], "h": probed[1]}
            short_edge = min(probed)
    stat_size = 0
    try:
        if path.is_file():
            stat_size = path.stat().st_size
    except OSError:
        pass
    return {
        "file": item["file"],
        "shot": item.get("shot") or "",
        "enabled": bool(item.get("enabled", True)),
        "kind": kind,
        "bytes": stat_size,
        "size": size,
        "short_edge": short_edge,
        "low_res": bool(short_edge is not None and short_edge < 1024),
        "caption": read_caption(folder, item["file"]),
        "missing": not path.is_file(),
        "clean": item.get("clean") or "",
        "has_original": (folder / ORIGINALS_DIR / item["file"]).is_file(),
        "quality": item.get("quality") or None,
    }


def mix_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"front": 0, "three-quarter": 0, "profile": 0, "full-body": 0, "untagged": 0}
    for item in items:
        if not item.get("enabled", True):
            continue
        if item.get("kind") == "video":
            continue
        shot = item.get("shot") or ""
        if shot in counts:
            counts[shot] += 1
        else:
            counts["untagged"] += 1
    return counts


def public_dataset(slug: str, folder: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    data = manifest or load_manifest(folder, slug)
    items = [item_payload(folder, item) for item in data.get("items") or []]
    enabled = [i for i in items if i.get("enabled") and not i.get("missing")]
    images = [i for i in enabled if i.get("kind") == "image"]
    videos = [i for i in enabled if i.get("kind") == "video"]
    cover = resolve_cover(data, folder)
    last = data.get("last_extract")
    return {
        "slug": slug,
        "name": data.get("name") or slug,
        "display_name": data.get("display_name") or slug,
        "concept_type": data.get("concept_type") or "identity",
        "description": data.get("description") or "",
        "mod_name": data.get("mod_name") or default_mod_name(slug),
        "extract": _merge_extract(data.get("extract")),
        "items": items,
        "folder": str(folder),
        "linked": slug in load_links() or bool(data.get("linked")),
        "cover": cover,
        "cover_size": _cover_size(folder, cover),
        "image_count": len(images),
        "video_count": len(videos),
        "enabled_count": len(enabled),
        "total_count": len(items),
        "dirty": is_dirty(data, folder),
        "last_extract": last,
        "mix": mix_counts(items),
        "content_hash": content_hash(data, folder),
        "subfolder": data.get("subfolder") or "",
        "notes": data.get("notes") or "",
        "quality_advice": data.get("quality_advice") or [],
    }


def _cover_size(folder: Path, cover: str | None) -> dict[str, int] | None:
    if not cover:
        return None
    probed = probe_size(folder / cover)
    if not probed:
        return None
    return {"w": probed[0], "h": probed[1]}


def public_dataset_card(slug: str, folder: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Grid/index payload: counts and cover only, no per-still probes."""
    data = manifest or load_manifest(folder, slug)
    items = data.get("items") or []
    image_count = 0
    video_count = 0
    enabled_count = 0
    for item in items:
        name = item.get("file") or ""
        path = folder / name
        if not item.get("enabled", True) or not path.is_file():
            continue
        enabled_count += 1
        if is_image(path):
            image_count += 1
        elif is_video(path):
            video_count += 1
    cover = resolve_cover(data, folder)
    last = data.get("last_extract")
    return {
        "slug": slug,
        "name": data.get("name") or slug,
        "display_name": data.get("display_name") or slug,
        "concept_type": data.get("concept_type") or "identity",
        "description": data.get("description") or "",
        "mod_name": data.get("mod_name") or default_mod_name(slug),
        "extract": _merge_extract(data.get("extract")),
        "folder": str(folder),
        "linked": bool(data.get("linked")),
        "cover": cover,
        "cover_size": _cover_size(folder, cover),
        "image_count": image_count,
        "video_count": video_count,
        "enabled_count": enabled_count,
        "total_count": len(items),
        "dirty": is_dirty(data, folder),
        "last_extract": last,
        "subfolder": data.get("subfolder") or "",
        "notes": data.get("notes") or "",
    }


def list_datasets() -> list[dict[str, Any]]:
    cards = []
    for slug, folder in _dataset_map().items():
        try:
            cards.append(public_dataset_card(slug, folder))
        except OSError:
            continue
    cards.sort(key=lambda c: (c.get("display_name") or c["slug"]).lower())
    return cards


def _rewrite_extract_output(output: str, old: str, new: str) -> str:
    path = Path(str(output))
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part.lower() != "refmods":
            continue
        filename = parts[-1] if i + 1 < len(parts) else ""
        prefix = parts[: i + 1]
        old_parts = tuple(old.split("/")) if old else ()
        rest_start = i + 1 + len(old_parts)
        if old_parts and parts[i + 1 : rest_start] != old_parts:
            return str(output)
        tail = parts[rest_start:]
        if filename and (not tail or tail[-1] != filename):
            tail = (*tail, filename) if filename else tail
        new_parts = tuple(new.split("/")) if new else ()
        return str(Path(*prefix, *new_parts, *tail))
    return str(output)


def remap_dataset_subfolders(old: str, new: str) -> int:
    """Point datasets at a renamed RefMod folder, including nested paths."""
    old = (old or "").strip("/")
    new = (new or "").strip("/")
    if not old or old == new:
        return 0
    updated = 0
    for slug, folder in _dataset_map().items():
        try:
            manifest = load_manifest(folder, slug)
        except OSError:
            continue
        current = str(manifest.get("subfolder") or "")
        if current == old:
            nxt = new
        elif current.startswith(old + "/"):
            nxt = new + current[len(old) :] if new else current[len(old) + 1 :]
        else:
            continue
        try:
            manifest["subfolder"] = normalize_subfolder(nxt)
        except ValueError:
            manifest["subfolder"] = nxt
        last = manifest.get("last_extract")
        if isinstance(last, dict) and last.get("output"):
            last["output"] = _rewrite_extract_output(str(last["output"]), old, new)
        save_manifest(folder, manifest)
        updated += 1
    return updated


def _assign_subfolder(manifest: dict[str, Any], subfolder: str | None) -> None:
    if subfolder is None:
        return
    try:
        manifest["subfolder"] = normalize_subfolder(subfolder)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc


def create_dataset(name: str, display_name: str | None = None, subfolder: str | None = None) -> dict[str, Any]:
    slug = slugify(name)
    if not slug:
        raise DatasetError("dataset name is required")
    mapping = _dataset_map()
    if slug in mapping:
        raise DatasetError(f"dataset '{slug}' already exists")
    folder = datasets_root() / slug
    folder.mkdir(parents=True, exist_ok=True)
    manifest = empty_manifest(slug, display_name or name)
    _assign_subfolder(manifest, subfolder)
    save_manifest(folder, manifest)
    return public_dataset(slug, folder)


def import_folder(path: str, slug: str | None = None, mode: str = "copy", recursive: bool = False, subfolder: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_dir():
        raise DatasetError("import path is not a folder")
    name = slugify(slug or source.name)
    if not name:
        raise DatasetError("dataset name is required")
    mapping = _dataset_map()
    if name in mapping:
        raise DatasetError(f"dataset '{name}' already exists")
    files = list_media_files(source, recursive=recursive)
    if mode not in ("copy", "link"):
        raise DatasetError("mode must be copy or link")
    if mode == "link":
        folder = source.resolve()
        links = load_links()
        links[name] = str(folder)
        save_links(links)
        manifest = load_manifest(folder, name)
        manifest["linked"] = True
        if not manifest.get("display_name"):
            manifest["display_name"] = source.name
        _assign_subfolder(manifest, subfolder)
        save_manifest(folder, manifest)
        return public_dataset(name, folder)
    folder = datasets_root() / name
    folder.mkdir(parents=True, exist_ok=True)
    for src in files:
        dest = folder / src.name
        if dest.exists():
            dest = _unique_dest(folder, src.name)
        shutil.copy2(src, dest)
        cap = _caption_path(src.parent, src.name)
        if cap.is_file():
            shutil.copy2(cap, folder / (dest.stem + ".txt"))
    manifest = empty_manifest(name, source.name)
    _assign_subfolder(manifest, subfolder)
    save_manifest(folder, manifest)
    return public_dataset(name, folder)


def _unique_dest(folder: Path, filename: str) -> Path:
    path = folder / filename
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    n = 2
    while True:
        candidate = folder / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def patch_dataset(slug: str, updates: dict[str, Any]) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    manifest = load_manifest(folder, slug)
    if "display_name" in updates and updates["display_name"] is not None:
        manifest["display_name"] = str(updates["display_name"]).strip() or manifest["display_name"]
    if "description" in updates and updates["description"] is not None:
        manifest["description"] = str(updates["description"])
    if "concept_type" in updates and updates["concept_type"]:
        concept = str(updates["concept_type"])
        if concept not in CONCEPT_TYPES:
            raise DatasetError("unknown concept type")
        manifest["concept_type"] = concept
    if "mod_name" in updates and updates["mod_name"]:
        raw = str(updates["mod_name"]).strip()
        cleaned = re.sub(r"[^\w.\-]+", "_", raw).strip("._")
        manifest["mod_name"] = cleaned or default_mod_name(slug)
    if "extract" in updates and isinstance(updates["extract"], dict):
        merged = dict(manifest.get("extract") or {})
        merged.update(updates["extract"])
        manifest["extract"] = _merge_extract(merged)
    if "items" in updates and isinstance(updates["items"], list):
        items = []
        seen: set[str] = set()
        existing = {p.name for p in list_media_files(folder)}
        prev_by = {i["file"]: i for i in manifest.get("items") or []}
        for raw in updates["items"]:
            item = _normalize_item(raw)
            if item is None or item["file"] not in existing or item["file"] in seen:
                continue
            prev = prev_by.get(item["file"]) or {}
            if isinstance(raw, dict):
                if "shot" not in raw and prev.get("shot"):
                    item["shot"] = prev["shot"]
                if "clean" not in raw and prev.get("clean"):
                    item["clean"] = prev["clean"]
                if "quality" not in raw and prev.get("quality"):
                    item["quality"] = prev["quality"]
            items.append(item)
            seen.add(item["file"])
        for name in sorted(existing, key=str.lower):
            if name not in seen:
                prev = prev_by.get(name) or {}
                leftover = {
                    "file": name,
                    "shot": prev.get("shot") or "",
                    "enabled": True,
                    "clean": prev.get("clean") or "",
                }
                if prev.get("quality"):
                    leftover["quality"] = prev["quality"]
                items.append(leftover)
        manifest["items"] = items
    if "subfolder" in updates and updates["subfolder"] is not None:
        try:
            manifest["subfolder"] = normalize_subfolder(str(updates["subfolder"]))
        except ValueError as exc:
            raise DatasetError(str(exc)) from exc
    if "cover" in updates and updates["cover"] is not None:
        chosen = _normalize_cover(updates["cover"])
        if not chosen:
            manifest["cover"] = ""
        else:
            path = folder / chosen
            names = {item["file"] for item in manifest.get("items") or []}
            if chosen not in names or not path.is_file() or not is_image(path):
                raise DatasetError("cover must be an image in this dataset")
            manifest["cover"] = chosen
    if "notes" in updates and updates["notes"] is not None:
        manifest["notes"] = str(updates["notes"])
    save_manifest(folder, manifest)
    return public_dataset(slug, folder)


def _sheet_identity(
    filename: str,
    index: int,
    count: int,
    name: str | None,
    display_name: str | None,
) -> tuple[str, str]:
    stem = Path(_safe_filename(filename)).stem or "dataset"
    given = (name or "").strip()
    given_display = (display_name or "").strip()
    if count == 1:
        source = given or stem
        return source, given_display or given or stem
    if index == 0 and given:
        return given, given_display or given
    return stem, stem


def create_datasets_from_sheets(
    sheets: list[tuple[str, bytes]],
    name: str | None = None,
    display_name: str | None = None,
    options: SplitOptions | None = None,
    plans: list[dict[str, Any]] | None = None,
    subfolder: str | None = None,
) -> list[dict[str, Any]]:
    """Split each contact sheet into panels and create one dataset per sheet.

    `plans` is optional: a list of {filename, boxes} matching each sheet, used
    when the UI already chose a cut set.
    """
    if not sheets:
        raise DatasetError("drop at least one character sheet")
    options = options or SplitOptions(square=False)
    plan_by_name: dict[str, list] = {}
    for plan in plans or []:
        if not isinstance(plan, dict):
            continue
        raw_name = str(plan.get("filename") or "").strip()
        if not raw_name:
            continue
        key = _safe_filename(raw_name)
        boxes = plan.get("boxes")
        if isinstance(boxes, list):
            plan_by_name[key] = boxes

    prepared: list[tuple[str, str, list[tuple[str, bytes]]]] = []
    mapping = _dataset_map()
    planned: set[str] = set()
    for index, (original_name, data) in enumerate(sheets):
        filename = _safe_filename(original_name)
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise DatasetError(f"{filename} is not an image")
        source, display = _sheet_identity(filename, index, len(sheets), name, display_name)
        slug = slugify(source)
        if slug in mapping or slug in planned:
            raise DatasetError(f"dataset '{slug}' already exists")
        stem = Path(filename).stem or "panel"
        chosen = plan_by_name.get(filename)
        try:
            if chosen is not None:
                panels = crop_to_pngs(data, stem, chosen, options)
            else:
                panels = split_to_pngs(data, stem, options)
        except SplitError as exc:
            raise DatasetError(f"could not split {filename}: {exc}") from exc
        min_needed = 1 if chosen is not None else options.min_panels
        if len(panels) < min_needed:
            raise DatasetError(
                f"could not split {filename}: only found {len(panels)} panel(s)"
            )
        prepared.append((slug, display, panels))
        planned.add(slug)

    created: list[dict[str, Any]] = []
    try:
        for slug, display, panels in prepared:
            create_dataset(slug, display, subfolder=subfolder)
            created.append(add_files(slug, panels))
    except Exception:
        for card in created:
            try:
                delete_dataset(card["slug"])
            except DatasetError:
                pass
        raise
    return created


def add_files(slug: str, uploads: list[tuple[str, bytes]]) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    manifest = load_manifest(folder, slug)
    known = {item["file"] for item in manifest["items"]}
    for original_name, data in uploads:
        filename = _safe_filename(original_name)
        if Path(filename).suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        dest = folder / filename
        if dest.exists():
            dest = _unique_dest(folder, filename)
            filename = dest.name
        dest.write_bytes(data)
        if filename not in known:
            manifest["items"].append({"file": filename, "shot": "", "enabled": True, "clean": ""})
            known.add(filename)
    save_manifest(folder, manifest)
    return public_dataset(slug, folder)


def delete_file(slug: str, filename: str) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    name = _safe_filename(filename)
    path = folder / name
    if path.is_file():
        path.unlink()
    cap = _caption_path(folder, name)
    if cap.is_file():
        cap.unlink()
    orig = folder / ORIGINALS_DIR / name
    if orig.is_file():
        orig.unlink()
    manifest = load_manifest(folder, slug)
    manifest["items"] = [i for i in manifest["items"] if i["file"] != name]
    if manifest.get("cover") == name:
        manifest["cover"] = ""
    save_manifest(folder, manifest)
    return public_dataset(slug, folder)


def set_caption(slug: str, filename: str, text: str) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    name = _safe_filename(filename)
    if not (folder / name).is_file():
        raise DatasetError("file not found")
    write_caption(folder, name, text)
    return public_dataset(slug, folder)


def delete_dataset(slug: str) -> None:
    slug, folder = resolve_dataset(slug)
    links = load_links()
    if slug in links:
        del links[slug]
        save_links(links)
        return
    root = datasets_root().resolve()
    try:
        resolved = folder.resolve()
    except OSError as exc:
        raise DatasetError("cannot resolve dataset folder") from exc
    if resolved != root / slug and root not in resolved.parents:
        raise DatasetError("refusing to delete a folder outside the datasets root")
    shutil.rmtree(resolved)


def record_extract(
    slug: str,
    output: str,
    token_count: int | None,
    file_count: int,
) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    manifest = load_manifest(folder, slug)
    digest = content_hash(manifest, folder)
    manifest["last_extract"] = {
        "at": utc_now(),
        "output": output,
        "content_hash": digest,
        "file_count": file_count,
        "token_count": token_count,
    }
    save_manifest(folder, manifest)
    return public_dataset(slug, folder)


def dataset_file(slug: str, filename: str) -> Path:
    slug, folder = resolve_dataset(slug)
    name = _safe_filename(filename)
    path = (folder / name).resolve()
    folder_res = folder.resolve()
    if folder_res not in path.parents and path != folder_res:
        raise DatasetError("invalid filename")
    if not path.is_file():
        raise DatasetError("file not found")
    return path
