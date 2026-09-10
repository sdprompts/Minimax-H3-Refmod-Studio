"""Read RefMod safetensors headers without loading torch."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .discover import default_output_dir, find_comfy_root

META_KEYS = ("refmod_meta", "audio_refmod_meta")
LIKELY_EMPTY_BYTES = 100 * 1024
SKIP_DIRS = {"graph_presets", ".git", "__pycache__"}


class LibraryError(ValueError):
    pass


def normalize_subfolder(raw: str | None) -> str:
    """Relative folder under the RefMod root, using `/` like ComfyUI loaders."""
    text = (raw or "").replace("\\", "/").strip()
    if not text:
        return ""
    parts: list[str] = []
    for piece in text.split("/"):
        name = piece.strip()
        if not name or name == ".":
            continue
        lowered = name.lower()
        if name == ".." or lowered in SKIP_DIRS:
            raise ValueError(f"invalid RefMod subfolder: {raw!r}")
        if name.startswith(".") or any(ch in name for ch in '<>:"|?*'):
            raise ValueError(f"invalid RefMod subfolder: {raw!r}")
        parts.append(name)
    return "/".join(parts)


def infer_subfolder_from_output(output: str | None) -> str | None:
    """Folder between `refmods/` and the filename, or None if it cannot be inferred."""
    if not output:
        return None
    parts = Path(str(output)).parts
    for i, part in enumerate(parts):
        if part.lower() == "refmods":
            rel = parts[i + 1 : -1]
            try:
                return normalize_subfolder("/".join(rel))
            except ValueError:
                return ""
    return None


def _token_count(meta: dict[str, Any]) -> int | None:
    kind = str(meta.get("kind") or "")
    try:
        if kind == "audio":
            t = int(meta.get("latent_t") or 0)
            return 2 * t if t > 0 else None
        h = int(meta.get("latent_h") or 0)
        w = int(meta.get("latent_w") or 0)
        t = int(meta.get("latent_t") or 1)
    except (TypeError, ValueError):
        return None
    if h <= 0 or w <= 0:
        return None
    return t * (h // 2) * (w // 2)


_inspect_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}


def _rel_id(path: Path, root: Path | None) -> tuple[str, str]:
    if root is None:
        return path.stem, ""
    try:
        rel_path = path.resolve().relative_to(root.resolve())
        rel = rel_path.with_suffix("").as_posix()
        parent = rel_path.parent.as_posix()
        subfolder = "" if parent in (".", "") else parent
        return rel, subfolder
    except ValueError:
        return path.stem, ""


def _mod_stub(path: Path, root: Path | None, size: int) -> dict[str, Any]:
    rel, subfolder = _rel_id(path, root)
    return {
        "name": path.stem,
        "id": rel or path.stem,
        "subfolder": subfolder,
        "path": str(path),
        "bytes": size,
        "likely_empty": size < LIKELY_EMPTY_BYTES,
        "metadata": {},
        "token_count": None,
        "concept_type": "",
        "mode": "",
        "kind": "",
        "description": "",
        "keys": [],
    }


def _parse_mod_file(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "error": "",
        "metadata": {},
        "name": path.stem,
        "concept_type": "",
        "mode": "",
        "kind": "",
        "description": "",
        "token_count": None,
    }
    try:
        with safe_open(str(path), framework="np") as handle:
            raw_meta = handle.metadata() or {}
    except Exception as exc:  # noqa: BLE001 — corrupt files should still list
        parsed["error"] = str(exc)
        return parsed
    meta: dict[str, Any] = {}
    for key in META_KEYS:
        if key in raw_meta:
            try:
                meta = json.loads(raw_meta[key])
            except json.JSONDecodeError:
                meta = dict(raw_meta)
            break
    else:
        meta = dict(raw_meta)
    parsed["metadata"] = meta
    parsed["name"] = str(meta.get("name") or path.stem)
    parsed["concept_type"] = str(meta.get("concept_type") or "")
    parsed["mode"] = str(meta.get("mode") or "")
    parsed["kind"] = str(meta.get("kind") or "")
    parsed["description"] = str(meta.get("description") or "")
    parsed["token_count"] = _token_count(meta)
    return parsed


def inspect_mod_lite(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Filename, folder, and size only — no safetensors open."""
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        size = 0
    info = _mod_stub(path, root, size)
    info["lite"] = True
    return info


def inspect_mod(path: Path, root: Path | None = None) -> dict[str, Any]:
    info = _mod_stub(path, root, 0)
    if not path.is_file():
        return info
    try:
        st = path.stat()
        size = int(st.st_size)
        mtime = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        cache_key = str(path.resolve())
    except OSError as exc:
        info["error"] = str(exc)
        return info
    info["bytes"] = size
    info["likely_empty"] = size < LIKELY_EMPTY_BYTES
    hit = _inspect_cache.get(cache_key)
    if hit and hit[0] == mtime and hit[1] == size:
        parsed = hit[2]
    else:
        parsed = _parse_mod_file(path)
        _inspect_cache[cache_key] = (mtime, size, parsed)
    if parsed.get("error"):
        info["error"] = parsed["error"]
        return info
    info["metadata"] = dict(parsed.get("metadata") or {})
    info["name"] = parsed.get("name") or path.stem
    info["concept_type"] = parsed.get("concept_type") or ""
    info["mode"] = parsed.get("mode") or ""
    info["kind"] = parsed.get("kind") or ""
    info["description"] = parsed.get("description") or ""
    info["token_count"] = parsed.get("token_count")
    return info


def iter_mod_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in SKIP_DIRS and not name.startswith(".")
        ]
        for name in filenames:
            if name.lower().endswith(".safetensors"):
                found.append(Path(dirpath) / name)
    found.sort(key=lambda p: p.as_posix().lower())
    return found


def _output_root() -> Path:
    output = default_output_dir(find_comfy_root())
    if output is None:
        raise LibraryError("refmods folder is not configured")
    return output


def subfolder_path(name: str, root: Path | None = None) -> Path:
    try:
        folder = normalize_subfolder(name)
    except ValueError as exc:
        raise LibraryError(str(exc)) from exc
    if not folder:
        raise LibraryError("folder name is required")
    output = root or _output_root()
    path = (output / Path(*folder.split("/"))).resolve()
    folder_res = output.resolve()
    if folder_res not in path.parents and path != folder_res:
        raise LibraryError("invalid RefMod folder")
    return path


def create_subfolder(name: str) -> str:
    try:
        folder = normalize_subfolder(name)
    except ValueError as exc:
        raise LibraryError(str(exc)) from exc
    if not folder:
        raise LibraryError("folder name is required")
    path = subfolder_path(folder)
    path.mkdir(parents=True, exist_ok=True)
    return folder


def rename_subfolder(src: str, dst: str) -> dict[str, Any]:
    try:
        old = normalize_subfolder(src)
        new = normalize_subfolder(dst)
    except ValueError as exc:
        raise LibraryError(str(exc)) from exc
    if not old:
        raise LibraryError("cannot rename the RefMod root")
    if not new:
        raise LibraryError("new folder name is required")
    if old == new:
        return {"from": old, "to": new, "folders": list_subfolders()}
    if new == old or new.startswith(old + "/"):
        raise LibraryError("cannot move a folder into itself")
    src_path = subfolder_path(old)
    if not src_path.is_dir():
        raise FileNotFoundError(old)
    dst_path = subfolder_path(new)
    if dst_path.exists():
        raise LibraryError(f"folder '{new}' already exists")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_path), str(dst_path))
    from .datasets import remap_dataset_subfolders

    remap_dataset_subfolders(old, new)
    from .hub import remap_hub_ids

    remap_hub_ids(old, new)
    return {"from": old, "to": new, "folders": list_subfolders()}


def delete_subfolder(name: str) -> None:
    try:
        folder = normalize_subfolder(name)
    except ValueError as exc:
        raise LibraryError(str(exc)) from exc
    if not folder:
        raise LibraryError("cannot delete the RefMod root")
    path = subfolder_path(folder)
    if not path.is_dir():
        raise FileNotFoundError(folder)
    leftover = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".safetensors"]
    if leftover:
        raise LibraryError(f"{folder} still has RefMods — move or delete them first")
    shutil.rmtree(path)


def list_subfolders(root: Path | None = None) -> list[str]:
    output = root or default_output_dir(find_comfy_root())
    if output is None or not output.is_dir():
        return []
    folders: set[str] = set()
    for dirpath, dirnames, _filenames in os.walk(output):
        dirnames[:] = [
            name for name in dirnames
            if name not in SKIP_DIRS and not name.startswith(".")
        ]
        try:
            rel = Path(dirpath).resolve().relative_to(output.resolve())
        except ValueError:
            continue
        if rel.parts:
            folders.add(rel.as_posix())
    return sorted(folders, key=str.lower)


def expected_comfy_path(mod_name: str, subfolder: str = "", root: Path | None = None) -> Path | None:
    output = root if root is not None else default_output_dir(find_comfy_root())
    if output is None or not mod_name:
        return None
    folder = ""
    try:
        folder = normalize_subfolder(subfolder)
    except ValueError:
        folder = ""
    if folder:
        return output / Path(*folder.split("/")) / f"{mod_name}.safetensors"
    return output / f"{mod_name}.safetensors"


def _local_backup(folder: Path, mod_name: str) -> Path | None:
    named = folder / f"{mod_name}.safetensors"
    if named.is_file():
        return named
    matches = sorted(folder.glob("*.safetensors"), key=lambda p: p.name.lower())
    return matches[0] if matches else None


def list_studio_mods() -> list[dict[str, Any]]:
    """Mods this app extracted: dataset-folder copies plus ComfyUI presence."""
    from .datasets import list_datasets

    comfy_root = default_output_dir(find_comfy_root())
    items: list[dict[str, Any]] = []
    for card in list_datasets():
        folder = Path(card["folder"])
        mod_name = str(card.get("mod_name") or "")
        subfolder = str(card.get("subfolder") or "")
        local = _local_backup(folder, mod_name) if folder.is_dir() else None
        last = card.get("last_extract") or {}
        last_path = Path(str(last["output"])) if isinstance(last, dict) and last.get("output") else None
        expected = expected_comfy_path(mod_name, subfolder, comfy_root)
        comfy: Path | None = None
        if last_path is not None and last_path.is_file():
            comfy = last_path
        elif expected is not None and expected.is_file():
            comfy = expected
        if local is None and comfy is None:
            continue
        source = local or comfy
        info = inspect_mod(source, folder if local is not None else (comfy_root if comfy_root else None))
        local_bytes = local.stat().st_size if local is not None and local.is_file() else 0
        comfy_bytes = comfy.stat().st_size if comfy is not None and comfy.is_file() else 0
        in_sync = bool(local_bytes and comfy_bytes and local_bytes == comfy_bytes)
        status = "missing"
        if comfy is not None and local is None:
            status = "comfy_only"
        elif local is not None and comfy is None:
            status = "missing"
        elif in_sync:
            status = "in_sync"
        else:
            status = "stale"
        items.append({
            **info,
            "id": f"{subfolder}/{mod_name}".strip("/") if subfolder else mod_name,
            "name": mod_name or info.get("name") or folder.name,
            "subfolder": subfolder,
            "dataset_slug": card["slug"],
            "display_name": card.get("display_name") or card["slug"],
            "mod_name": mod_name,
            "local_path": str(local) if local is not None else "",
            "comfy_path": str(comfy) if comfy is not None else (str(expected) if expected is not None else ""),
            "local_present": local is not None,
            "comfy_present": comfy is not None,
            "in_sync": in_sync,
            "status": status,
            "dirty": bool(card.get("dirty")),
            "token_count": info.get("token_count"),
            "bytes": local_bytes or comfy_bytes or info.get("bytes") or 0,
        })
    items.sort(key=lambda m: ((m.get("subfolder") or ""), (m.get("display_name") or "").lower()))
    return items


def install_studio_mod(slug: str, replace: bool = False) -> dict[str, Any]:
    from .datasets import DatasetError, record_extract, resolve_dataset, load_manifest

    try:
        key, folder = resolve_dataset(slug)
    except DatasetError as exc:
        raise LibraryError(str(exc)) from exc
    manifest = load_manifest(folder, key)
    mod_name = str(manifest.get("mod_name") or "")
    subfolder = str(manifest.get("subfolder") or "")
    local = _local_backup(folder, mod_name)
    if local is None:
        raise LibraryError("this dataset has no local RefMod copy — extract first")
    dest = expected_comfy_path(mod_name, subfolder)
    if dest is None:
        raise LibraryError("refmods folder is not configured")
    if dest.is_file() and not replace:
        raise LibraryError(f"{dest.name} already exists in ComfyUI — replace to overwrite")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, dest)
    last = manifest.get("last_extract") if isinstance(manifest.get("last_extract"), dict) else {}
    token_count = last.get("token_count") if last else None
    file_count = last.get("file_count") if last else 0
    record_extract(key, str(dest), token_count, int(file_count or 0))
    return inspect_mod(dest, default_output_dir(find_comfy_root()))


def _under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _wanted_folder_set(cards: list[dict[str, Any]]) -> set[str]:
    wanted: set[str] = set()
    for card in cards:
        try:
            sub = normalize_subfolder(str(card.get("subfolder") or ""))
        except ValueError:
            continue
        if not sub:
            continue
        parts = sub.split("/")
        for i in range(len(parts)):
            wanted.add("/".join(parts[: i + 1]))
    return wanted


def _find_comfy_copy(mod_name: str, dest: Path, last_path: Path | None, root: Path) -> Path | None:
    if last_path is not None and last_path.is_file() and _under_root(last_path, root):
        return last_path
    if dest.is_file():
        return dest
    matches = [p for p in iter_mod_files(root) if p.name.lower() == f"{mod_name}.safetensors".lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def _retarget_last_extract(slug: str, dest: Path) -> None:
    from .datasets import load_manifest, record_extract, resolve_dataset, save_manifest

    key, folder = resolve_dataset(slug)
    manifest = load_manifest(folder, key)
    last = manifest.get("last_extract")
    if isinstance(last, dict) and last.get("content_hash"):
        last["output"] = str(dest)
        save_manifest(folder, manifest)
        return
    last = last if isinstance(last, dict) else {}
    record_extract(slug, str(dest), last.get("token_count"), int(last.get("file_count") or 0))


def _prune_empty_unwanted(root: Path, keep: set[str]) -> list[str]:
    removed: list[str] = []
    if not root.is_dir():
        return removed
    dirs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in SKIP_DIRS and not name.startswith(".")
        ]
        path = Path(dirpath)
        try:
            if path.resolve() == root.resolve():
                continue
        except OSError:
            continue
        dirs.append(path)
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    for path in dirs:
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            continue
        if rel in keep:
            continue
        try:
            if any(path.iterdir()):
                continue
            path.rmdir()
            removed.append(rel)
        except OSError:
            continue
    return removed


def sync_comfy_layout() -> dict[str, Any]:
    """Make models/refmods match dataset folders. Studio copies win when files differ."""
    root = _output_root()
    root.mkdir(parents=True, exist_ok=True)
    from .datasets import list_datasets

    cards = list_datasets()
    wanted = _wanted_folder_set(cards)
    folders_created: list[str] = []
    for folder in sorted(wanted, key=lambda name: (name.count("/"), name.lower())):
        path = subfolder_path(folder, root)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            folders_created.append(folder)

    tallies = {"moved": 0, "copied": 0, "replaced": 0, "kept": 0, "skipped": 0}
    items: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for card in cards:
        slug = str(card.get("slug") or "")
        mod_name = str(card.get("mod_name") or "")
        subfolder = str(card.get("subfolder") or "")
        if not slug or not mod_name:
            tallies["skipped"] += 1
            items.append({"slug": slug, "action": "skipped", "reason": "no mod name"})
            continue
        dest = expected_comfy_path(mod_name, subfolder, root)
        if dest is None:
            tallies["skipped"] += 1
            items.append({"slug": slug, "action": "skipped", "reason": "refmods folder is not configured"})
            continue
        folder = Path(card["folder"])
        local = _local_backup(folder, mod_name) if folder.is_dir() else None
        last = card.get("last_extract") or {}
        last_path = Path(str(last["output"])) if isinstance(last, dict) and last.get("output") else None
        current = _find_comfy_copy(mod_name, dest, last_path, root)
        source = local if local is not None else current
        if source is None or not source.is_file():
            tallies["skipped"] += 1
            items.append({"slug": slug, "action": "skipped", "reason": "no RefMod copy"})
            continue
        origin = str(current) if current is not None and current.is_file() else ""
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            action = "kept"
            if current is not None and current.is_file() and not _same_file(current, dest):
                if dest.is_file():
                    dest.unlink()
                shutil.move(str(current), str(dest))
                action = "moved"
            elif not dest.is_file():
                shutil.copy2(source, dest)
                action = "copied"
            if local is not None and dest.is_file() and local.stat().st_size != dest.stat().st_size:
                shutil.copy2(local, dest)
                if action == "kept":
                    action = "replaced"
            _retarget_last_extract(slug, dest)
            tallies[action] += 1
            items.append({
                "slug": slug,
                "action": action,
                "from": origin,
                "to": str(dest),
            })
        except OSError as exc:
            errors.append({"slug": slug, "error": str(exc)})

    folders_removed = _prune_empty_unwanted(root, wanted)
    return {
        "ok": True,
        **tallies,
        "folders_created": folders_created,
        "folders_removed": folders_removed,
        "items": items,
        "errors": errors,
    }


def list_mods(dataset_index: dict[str, str] | None = None, *, lite: bool = False) -> list[dict[str, Any]]:
    output = default_output_dir(find_comfy_root())
    if output is None or not output.is_dir():
        return []
    index = dataset_index or {}
    mods = []
    reader = inspect_mod_lite if lite else inspect_mod
    for path in iter_mod_files(output):
        item = reader(path, output)
        item["dataset_slug"] = (
            index.get(item.get("id") or "")
            or index.get(path.stem)
            or index.get(item.get("name") or "")
        )
        mods.append(item)
    return mods


def dataset_mod_index() -> dict[str, str]:
    from .datasets import list_datasets

    index: dict[str, str] = {}
    for card in list_datasets():
        slug = card["slug"]
        if card.get("mod_name"):
            index[str(card["mod_name"])] = slug
            sub = str(card.get("subfolder") or "").strip("/")
            if sub:
                index[f"{sub}/{card['mod_name']}"] = slug
        last = card.get("last_extract") or {}
        output = last.get("output") if isinstance(last, dict) else None
        if output:
            path = Path(str(output))
            index[path.stem] = slug
            inferred = infer_subfolder_from_output(str(output))
            if inferred:
                index[f"{inferred}/{path.stem}"] = slug
    return index


def resolve_mod_path(name: str) -> Path:
    output = default_output_dir(find_comfy_root())
    if output is None:
        raise FileNotFoundError("refmods folder is not configured")
    rel = (name or "").replace("\\", "/").strip().strip("/")
    if rel.lower().endswith(".safetensors"):
        rel = rel[: -len(".safetensors")]
    if not rel:
        raise ValueError("invalid mod name")
    try:
        folder = normalize_subfolder(str(Path(rel).parent.as_posix()))
    except ValueError as exc:
        raise ValueError("invalid mod name") from exc
    stem = Path(rel).name
    if not stem or stem in {".", ".."}:
        raise ValueError("invalid mod name")
    relative = Path(*folder.split("/"), f"{stem}.safetensors") if folder else Path(f"{stem}.safetensors")
    path = (output / relative).resolve()
    folder_res = output.resolve()
    if folder_res not in path.parents and path.parent != folder_res:
        raise ValueError("invalid mod name")
    return path


def delete_mod(name: str) -> None:
    output = default_output_dir(find_comfy_root())
    if output is None:
        raise FileNotFoundError("refmods folder is not configured")
    path = resolve_mod_path(name)
    if not path.is_file():
        raise FileNotFoundError(name)
    path.unlink()
    current = path.parent
    root = output.resolve()
    while current != root and root in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
