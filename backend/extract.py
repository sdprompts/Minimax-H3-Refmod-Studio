"""Build extract_mod.py argv and run it as a ComfyUI-python subprocess."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import datasets as ds
from .discover import health as runtime_health, is_embed_python
from .library import normalize_subfolder
from .media import is_image, is_video

SAVED_RE = re.compile(
    r"saved \S+ mod '([^']+)' \((\d+) tokens, ([\d.]+) MB\) -> (.+)$"
)
MEDIA_RE = re.compile(r"\[extract\] (image|video) (.+):")
IDENTITY_RE = re.compile(r"\[RefMod\] identity (\d+)/(\d+)")


class ExtractError(ValueError):
    pass


@dataclass
class ExtractJob:
    slug: str
    status: str = "running"  # running | done | error
    log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    file_count: int = 0
    files: list[str] = field(default_factory=list)


_job_lock = threading.Lock()
_job: ExtractJob | None = None


def current_job() -> ExtractJob | None:
    return _job


def _append(job: ExtractJob, line: str) -> None:
    text = line.rstrip("\n")
    if text:
        job.log.append(text)


def parse_progress(log: list[str], total: int, status: str = "running") -> dict[str, Any]:
    stage = "starting"
    label = "Starting"
    current = ""
    encoded = 0
    ident_step = 0
    ident_total = 0
    kind = "image"
    for line in log:
        if "[studio] error" in line:
            stage = "error"
            label = "Failed"
            continue
        if "loading VAE" in line:
            stage = "loading_vae"
            label = "Loading H3 VAE into VRAM"
            continue
        media = MEDIA_RE.search(line)
        if media:
            stage = "encoding"
            encoded += 1
            kind = media.group(1)
            current = Path(media.group(2).strip()).name
            label = f"Encoding {kind} {encoded} / {max(total, encoded)}"
            continue
        ident = IDENTITY_RE.search(line)
        if ident:
            stage = "refining"
            ident_step = int(ident.group(1))
            ident_total = int(ident.group(2))
            label = f"Refining identity {ident_step} / {ident_total}"
            continue
        if "token cap" in line:
            stage = "fitting"
            label = "Fitting token budget"
            continue
        if SAVED_RE.search(line):
            stage = "saved"
            label = "Saved RefMod"
    if status == "done":
        stage = "saved"
        label = "Saved RefMod"
        pct = 100
    elif status == "error":
        stage = "error"
        label = "Failed"
        pct = 0
    elif stage == "starting":
        pct = 1
    elif stage == "loading_vae":
        pct = 6
    elif stage == "encoding":
        denom = max(total, encoded, 1)
        pct = min(92, 8 + int(84 * encoded / denom))
    elif stage == "refining":
        denom = max(total, 1)
        base = 8 + int(84 * max(encoded - 1, 0) / denom)
        frac = (ident_step / ident_total) if ident_total else 0
        pct = min(96, base + int((84 / denom) * frac))
    elif stage == "fitting":
        pct = 96
    elif stage == "saved":
        pct = 100
    else:
        pct = 0
    return {
        "stage": stage,
        "label": label,
        "current": current,
        "encoded": encoded,
        "total": max(total, encoded),
        "kind": kind,
        "identity_step": ident_step,
        "identity_total": ident_total,
        "percent": pct,
    }


def parse_saved_line(line: str) -> dict[str, Any] | None:
    match = SAVED_RE.search(line)
    if not match:
        return None
    return {
        "name": match.group(1),
        "token_count": int(match.group(2)),
        "mb": float(match.group(3)),
        "path": match.group(4).strip(),
    }


def enabled_media(manifest: dict[str, Any], folder: Path) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    videos: list[Path] = []
    for path in ds.enabled_files(manifest, folder):
        if is_image(path):
            images.append(path)
        elif is_video(path):
            videos.append(path)
    return images, videos


def build_argv(
    python: Path,
    script: Path,
    vae: Path,
    output_dir: Path,
    name: str,
    description: str,
    concept_type: str,
    extract: dict[str, Any],
    images: list[Path],
    videos: list[Path],
) -> list[str]:
    argv = [str(python)]
    if is_embed_python(python):
        # -s skips user site-packages (avoids a conflicting torch) but still
        # honors PYTHONPATH so Desktop / venv installs can import `comfy`.
        argv.append("-s")
    mode = str(extract.get("mode") or "encode")
    argv.extend([
        "-u",
        str(script),
        "--vae", str(vae),
        "--name", name,
        "--mode", mode,
        "--concept-type", concept_type,
        "--resolution", str(int(extract.get("resolution") or 1024)),
        "--max-tokens", str(int(extract.get("max_tokens") or 8192)),
        "--output", str(output_dir),
        "--pool", str(int(extract.get("pool") or 16)),
        "--multiplier", str(int(extract.get("multiplier") or 1)),
        "--device", str(extract.get("device") or "auto"),
    ])
    if mode == "training":
        argv.extend(["--identity", str(int(extract.get("identity") or 0))])
    if description:
        argv.extend(["--description", description])
    for path in images:
        argv.extend(["--image", str(path)])
    for path in videos:
        argv.extend(["--video", str(path)])
    return argv


def prepare_extract(
    slug: str,
    overwrite: bool = False,
    bump_version: bool = False,
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slug, folder = ds.resolve_dataset(slug)
    if updates:
        ds.patch_dataset(slug, updates)
    manifest = ds.load_manifest(folder, slug)
    images, videos = enabled_media(manifest, folder)
    if not images and not videos:
        raise ExtractError("dataset has no enabled images or videos")
    info = runtime_health(probe=False)
    if not info["ready"]:
        raise ExtractError("extract is not ready: " + "; ".join(info["missing"]))
    python = Path(info["python_path"])
    script = Path(info["extract_script"])
    vae = Path(info["vae_path"])
    output_root = Path(info["output_dir"])
    comfy_root = Path(info["comfy_root"]) if info.get("comfy_root") else None
    comfy_source = Path(info["comfy_source"]) if info.get("comfy_source") else comfy_root
    try:
        subfolder = normalize_subfolder(str(manifest.get("subfolder") or ""))
    except ValueError as exc:
        raise ExtractError(str(exc)) from exc
    output_dir = output_root / Path(*subfolder.split("/")) if subfolder else output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    mod_name = str(manifest.get("mod_name") or ds.default_mod_name(slug))
    target = output_dir / f"{mod_name}.safetensors"
    if bump_version:
        while target.exists():
            mod_name = ds.bump_mod_name(mod_name)
            target = output_dir / f"{mod_name}.safetensors"
        ds.patch_dataset(slug, {"mod_name": mod_name})
        manifest = ds.load_manifest(folder, slug)
    elif target.exists() and not overwrite:
        raise ExtractError(f"{target.name} already exists — set overwrite or bump version")
    argv = build_argv(
        python=python,
        script=script,
        vae=vae,
        output_dir=output_dir,
        name=mod_name,
        description=str(manifest.get("description") or ""),
        concept_type=str(manifest.get("concept_type") or "identity"),
        extract=manifest.get("extract") or {},
        images=images,
        videos=videos,
    )
    return {
        "slug": slug,
        "folder": folder,
        "manifest": manifest,
        "argv": argv,
        "python": python,
        "comfy_root": comfy_root,
        "comfy_source": comfy_source,
        "output_dir": output_dir,
        "mod_name": mod_name,
        "target": target,
        "file_count": len(images) + len(videos),
        "images": images,
        "videos": videos,
    }


def _run_process(job: ExtractJob, prepared: dict[str, Any], on_line: Callable[[str], None] | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    comfy_root = prepared.get("comfy_root")
    comfy_source = prepared.get("comfy_source") or comfy_root
    path_parts: list[str] = []
    if comfy_source is not None:
        path_parts.append(str(comfy_source))
    if comfy_root is not None and str(comfy_root) not in path_parts:
        path_parts.append(str(comfy_root))
    if path_parts:
        env["PYTHONPATH"] = os.pathsep.join(path_parts) + os.pathsep + env.get("PYTHONPATH", "")
    kwargs: dict[str, Any] = {
        "args": prepared["argv"],
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
        "env": env,
    }
    cwd = comfy_source or comfy_root
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(**kwargs)
    assert proc.stdout is not None
    saved: dict[str, Any] | None = None
    for line in proc.stdout:
        _append(job, line)
        if on_line:
            on_line(line)
        parsed = parse_saved_line(line)
        if parsed:
            saved = parsed
    code = proc.wait()
    if code != 0:
        raise ExtractError(f"extract_mod.py exited with code {code}")
    target: Path = prepared["target"]
    if saved and saved.get("path"):
        target = Path(saved["path"])
    if not target.is_file():
        raise ExtractError("extractor finished but the .safetensors file was not found")
    folder: Path = prepared["folder"]
    backup = folder / target.name
    try:
        if target.resolve() != backup.resolve():
            shutil.copy2(target, backup)
    except OSError as exc:
        _append(job, f"[studio] warning: could not copy into dataset folder: {exc}")
    token_count = saved["token_count"] if saved else None
    ds.record_extract(prepared["slug"], str(target), token_count, prepared["file_count"])
    from .library import inspect_mod

    meta = inspect_mod(target)
    job.result = {
        "path": str(target),
        "backup": str(backup) if backup.is_file() else "",
        "mod_name": prepared["mod_name"],
        "token_count": token_count if token_count is not None else meta.get("token_count"),
        "mb": saved["mb"] if saved else round(target.stat().st_size / (1024 * 1024), 2),
        "meta": meta,
        "dataset": ds.public_dataset(prepared["slug"], folder),
    }


def start_extract(
    slug: str,
    overwrite: bool = False,
    bump_version: bool = False,
    updates: dict[str, Any] | None = None,
) -> ExtractJob:
    global _job
    with _job_lock:
        if _job is not None and _job.status == "running":
            raise ExtractError("an extract is already running")
        prepared = prepare_extract(slug, overwrite=overwrite, bump_version=bump_version, updates=updates)
        job = ExtractJob(
            slug=prepared["slug"],
            file_count=int(prepared["file_count"]),
            files=[p.name for p in prepared["images"] + prepared["videos"]],
        )
        _append(job, "[studio] " + " ".join(prepared["argv"]))
        _job = job

        def runner() -> None:
            try:
                _run_process(job, prepared)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 — surface any extractor failure to the UI
                job.status = "error"
                job.error = str(exc)
                _append(job, f"[studio] error: {exc}")

        threading.Thread(target=runner, daemon=True).start()
        return job
