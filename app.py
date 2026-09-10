"""H3 RefMod Studio — local web app."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import APP_ROOT, public_config, save_config
from backend.clean import CleanError, apply_clean, scan_dataset, undo_clean
from backend.datasets import (
    CONCEPT_TYPES,
    DatasetError,
    add_files,
    create_dataset,
    create_datasets_from_sheets,
    dataset_file,
    delete_dataset,
    delete_file,
    import_folder,
    list_datasets,
    patch_dataset,
    public_dataset,
    resolve_dataset,
    set_caption,
)
from backend.discover import (
    discover_python,
    health as runtime_health,
    resolve_comfy_root,
)
from backend.extract import ExtractError, current_job, parse_progress, start_extract
from backend.picker import PickError, pick_file, pick_folder
from backend.hub import HubError, annotate_mods, current_job as hub_job, start_upload
from backend.library import (
    LibraryError,
    create_subfolder,
    dataset_mod_index,
    delete_mod,
    delete_subfolder,
    inspect_mod,
    install_studio_mod,
    list_mods,
    list_studio_mods,
    list_subfolders,
    rename_subfolder,
    resolve_mod_path,
    sync_comfy_layout,
)
from backend.media import thumb_bytes
from backend.quality import QualityError, scan_dataset as scan_quality
from backend.split import SplitError, sheet_preview_payload

STATIC = APP_ROOT / "static"

app = FastAPI(title="H3 RefMod Studio", docs_url=None, redoc_url=None)
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class SettingsBody(BaseModel):
    comfy_root: str | None = None
    python_path: str | None = None
    vae_path: str | None = None
    output_dir: str | None = None
    datasets_root: str | None = None
    extract_script: str | None = None
    mod_prefix: str | None = None
    hf_token: str | None = None
    hf_repo: str | None = None
    hf_repo_type: str | None = None
    hf_private: bool | None = None


class CreateDatasetBody(BaseModel):
    name: str
    display_name: str | None = None
    subfolder: str | None = None


class ImportBody(BaseModel):
    path: str
    slug: str | None = None
    mode: str = "copy"
    recursive: bool = False
    subfolder: str | None = None


class PatchDatasetBody(BaseModel):
    display_name: str | None = None
    description: str | None = None
    concept_type: str | None = None
    mod_name: str | None = None
    extract: dict[str, Any] | None = None
    items: list[dict[str, Any]] | None = None
    subfolder: str | None = None
    cover: str | None = None
    notes: str | None = None


class CaptionBody(BaseModel):
    caption: str = ""


class CleanScanBody(BaseModel):
    files: list[str] | None = None


class CleanApplyBody(BaseModel):
    items: list[dict[str, Any]] | None = None
    apply_boxes: list[dict[str, Any]] | None = None
    apply_keep: dict[str, Any] | None = None
    files: list[str] | None = None


class CleanUndoBody(BaseModel):
    files: list[str] | None = None


class ExtractBody(BaseModel):
    slug: str
    overwrite: bool = False
    bump_version: bool = False
    display_name: str | None = None
    description: str | None = None
    concept_type: str | None = None
    mod_name: str | None = None
    extract: dict[str, Any] | None = None
    subfolder: str | None = None


class OpenFolderBody(BaseModel):
    path: str = Field(min_length=1)


class FolderCreateBody(BaseModel):
    name: str


class FolderRenameBody(BaseModel):
    src: str
    dst: str


class HubUploadBody(BaseModel):
    ids: list[str]


class PickBody(BaseModel):
    kind: str = "folder"
    title: str = ""
    initial: str = ""
    filetypes: list[list[str]] | None = None


def _err(exc: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


@app.get("/")
def index() -> FileResponse:
    page = STATIC / "index.html"
    if not page.is_file():
        raise HTTPException(404, "UI is missing")
    return FileResponse(page)


@app.get("/api/health")
def api_health(probe: bool = False) -> dict[str, Any]:
    info = runtime_health(probe=probe)
    info["concept_types"] = list(CONCEPT_TYPES)
    return info


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return public_config()


@app.post("/api/settings")
def api_set_settings(body: SettingsBody) -> dict[str, Any]:
    data = body.model_dump(exclude_none=True)
    raw_root = data.get("comfy_root")
    if raw_root:
        resolved = resolve_comfy_root(raw_root)
        if resolved is not None:
            data["comfy_root"] = str(resolved)
            if not data.get("python_path"):
                py = discover_python(resolved)
                if py is not None:
                    data["python_path"] = str(py)
    save_config(data)
    info = runtime_health(probe=True)
    info["concept_types"] = list(CONCEPT_TYPES)
    return info


@app.get("/api/datasets")
def api_list_datasets() -> dict[str, Any]:
    return {"datasets": list_datasets()}


@app.post("/api/datasets")
def api_create_dataset(body: CreateDatasetBody) -> dict[str, Any]:
    try:
        return create_dataset(body.name, body.display_name, subfolder=body.subfolder)
    except DatasetError as exc:
        raise _err(exc) from exc


@app.post("/api/datasets/import")
def api_import_dataset(body: ImportBody) -> dict[str, Any]:
    try:
        return import_folder(body.path, body.slug, body.mode, body.recursive, subfolder=body.subfolder)
    except DatasetError as exc:
        raise _err(exc) from exc


@app.post("/api/datasets/split-sheet/preview")
async def api_split_sheet_preview(
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    sheets = []
    for item in files:
        filename = item.filename or "sheet.png"
        data = await item.read()
        try:
            sheets.append(sheet_preview_payload(filename, data))
        except SplitError as exc:
            raise _err(exc) from exc
    if not sheets:
        raise _err(DatasetError("drop at least one character sheet"))
    return {"sheets": sheets}


@app.post("/api/datasets/split-sheet")
async def api_split_sheet(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
    display_name: str = Form(""),
    plans: str = Form(""),
    subfolder: str = Form(""),
) -> dict[str, Any]:
    sheets: list[tuple[str, bytes]] = []
    for item in files:
        sheets.append((item.filename or "sheet.png", await item.read()))
    parsed_plans: list[dict[str, Any]] | None = None
    if (plans or "").strip():
        try:
            raw = json.loads(plans)
        except json.JSONDecodeError as exc:
            raise _err(DatasetError("cut plan is not valid JSON")) from exc
        if not isinstance(raw, list):
            raise _err(DatasetError("cut plan must be a list"))
        parsed_plans = raw
    try:
        created = create_datasets_from_sheets(
            sheets,
            name=name or None,
            display_name=display_name or None,
            plans=parsed_plans,
            subfolder=subfolder or None,
        )
    except DatasetError as exc:
        raise _err(exc) from exc
    return {"datasets": created}


@app.get("/api/datasets/{slug}")
def api_get_dataset(slug: str) -> dict[str, Any]:
    try:
        key, folder = resolve_dataset(slug)
        return public_dataset(key, folder)
    except DatasetError as exc:
        raise _err(exc, 404) from exc


@app.patch("/api/datasets/{slug}")
def api_patch_dataset(slug: str, body: PatchDatasetBody) -> dict[str, Any]:
    try:
        return patch_dataset(slug, body.model_dump(exclude_none=True))
    except DatasetError as exc:
        raise _err(exc) from exc


@app.delete("/api/datasets/{slug}")
def api_delete_dataset(slug: str) -> dict[str, Any]:
    try:
        delete_dataset(slug)
    except DatasetError as exc:
        raise _err(exc) from exc
    return {"ok": True}


@app.post("/api/datasets/{slug}/files")
async def api_add_files(slug: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    uploads: list[tuple[str, bytes]] = []
    for item in files:
        name = item.filename or "upload.bin"
        data = await item.read()
        uploads.append((name, data))
    try:
        return add_files(slug, uploads)
    except DatasetError as exc:
        raise _err(exc) from exc


@app.delete("/api/datasets/{slug}/files/{filename:path}")
def api_delete_file(slug: str, filename: str) -> dict[str, Any]:
    try:
        return delete_file(slug, filename)
    except DatasetError as exc:
        raise _err(exc) from exc


@app.put("/api/datasets/{slug}/files/{filename:path}/caption")
def api_set_caption(slug: str, filename: str, body: CaptionBody) -> dict[str, Any]:
    try:
        return set_caption(slug, filename, body.caption)
    except DatasetError as exc:
        raise _err(exc) from exc


@app.post("/api/datasets/{slug}/quality")
def api_quality_scan(slug: str, body: CleanScanBody = CleanScanBody()) -> dict[str, Any]:
    try:
        return scan_quality(slug, body.files)
    except DatasetError as exc:
        raise _err(exc, 404) from exc
    except QualityError as exc:
        raise _err(exc) from exc


@app.post("/api/datasets/{slug}/clean/scan")
def api_clean_scan(slug: str, body: CleanScanBody = CleanScanBody()) -> dict[str, Any]:
    files = body.files
    try:
        return scan_dataset(slug, files)
    except DatasetError as exc:
        raise _err(exc, 404) from exc
    except CleanError as exc:
        raise _err(exc) from exc


@app.post("/api/datasets/{slug}/clean/apply")
def api_clean_apply(slug: str, body: CleanApplyBody) -> dict[str, Any]:
    try:
        return apply_clean(
            slug,
            body.items,
            body.apply_boxes,
            body.apply_keep,
            body.files,
        )
    except DatasetError as exc:
        raise _err(exc, 404) from exc
    except CleanError as exc:
        raise _err(exc) from exc


@app.post("/api/datasets/{slug}/clean/undo")
def api_clean_undo(slug: str, body: CleanUndoBody = CleanUndoBody()) -> dict[str, Any]:
    files = body.files
    try:
        return undo_clean(slug, files)
    except DatasetError as exc:
        raise _err(exc, 404) from exc
    except CleanError as exc:
        raise _err(exc) from exc


@app.get("/api/datasets/{slug}/files/{filename:path}/thumb")
def api_thumb(slug: str, filename: str):
    try:
        path = dataset_file(slug, filename)
    except DatasetError as exc:
        raise _err(exc, 404) from exc
    data = thumb_bytes(path)
    if data is None:
        raise HTTPException(404, "no thumbnail")
    return StreamingResponse(iter([data]), media_type="image/jpeg")


@app.get("/api/datasets/{slug}/files/{filename:path}/media")
def api_media(slug: str, filename: str) -> FileResponse:
    try:
        path = dataset_file(slug, filename)
    except DatasetError as exc:
        raise _err(exc, 404) from exc
    return FileResponse(path)


@app.post("/api/extract")
def api_extract(body: ExtractBody) -> dict[str, Any]:
    updates = {
        k: v
        for k, v in {
            "display_name": body.display_name,
            "description": body.description,
            "concept_type": body.concept_type,
            "mod_name": body.mod_name,
            "extract": body.extract,
            "subfolder": body.subfolder,
        }.items()
        if v is not None
    }
    try:
        job = start_extract(
            body.slug,
            overwrite=body.overwrite,
            bump_version=body.bump_version,
            updates=updates or None,
        )
    except ExtractError as exc:
        status = 409 if "already running" in str(exc) else 400
        raise _err(exc, status) from exc
    except DatasetError as exc:
        raise _err(exc) from exc
    return {"ok": True, "slug": job.slug, "status": job.status}


@app.get("/api/extract/status")
def api_extract_status() -> dict[str, Any]:
    job = current_job()
    if job is None:
        return {"status": "idle"}
    return {
        "status": job.status,
        "slug": job.slug,
        "log": job.log,
        "result": job.result,
        "error": job.error,
        "files": job.files,
        "progress": parse_progress(job.log, job.file_count, job.status),
    }


@app.get("/api/extract/stream")
async def api_extract_stream():
    import asyncio

    async def gen():
        last = 0
        idle_ticks = 0
        while True:
            job = current_job()
            if job is None:
                idle_ticks += 1
                yield "event: idle\ndata: {}\n\n"
                if idle_ticks > 5:
                    break
                await asyncio.sleep(0.4)
                continue
            idle_ticks = 0
            while last < len(job.log):
                payload = json.dumps({"line": job.log[last]})
                yield f"data: {payload}\n\n"
                last += 1
            if job.status in ("done", "error"):
                payload = json.dumps({
                    "status": job.status,
                    "result": job.result,
                    "error": job.error,
                })
                yield f"event: {job.status}\ndata: {payload}\n\n"
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/library")
def api_library(lite: bool = False) -> dict[str, Any]:
    mods = annotate_mods(list_mods(dataset_mod_index(), lite=lite))
    return {"mods": mods, "folders": list_subfolders(), "lite": lite}


@app.get("/api/studio")
def api_studio() -> dict[str, Any]:
    mods = annotate_mods(list_studio_mods())
    folders = sorted({str(m.get("subfolder") or "") for m in mods if m.get("subfolder")})
    return {"mods": mods, "folders": folders}


@app.post("/api/studio/sync-comfy")
def api_sync_comfy() -> dict[str, Any]:
    try:
        return sync_comfy_layout()
    except LibraryError as exc:
        raise _err(exc) from exc


@app.post("/api/huggingface/upload")
def api_hub_upload(body: HubUploadBody) -> dict[str, Any]:
    try:
        job = start_upload(body.ids)
    except HubError as exc:
        status = 409 if "already running" in str(exc) else 400
        raise _err(exc, status) from exc
    return {"ok": True, "status": job.status, "total": job.total}


@app.get("/api/huggingface/status")
def api_hub_status() -> dict[str, Any]:
    job = hub_job()
    if job is None:
        return {"status": "idle"}
    return {
        "status": job.status,
        "log": job.log,
        "ids": job.ids,
        "results": job.results,
        "error": job.error,
        "done": job.done,
        "total": job.total,
    }


@app.post("/api/studio/{slug}/install")
def api_studio_install(slug: str, replace: bool = False) -> dict[str, Any]:
    try:
        info = install_studio_mod(slug, replace=replace)
    except LibraryError as exc:
        status = 409 if "already exists" in str(exc) else 400
        raise _err(exc, status) from exc
    except FileNotFoundError as exc:
        raise _err(exc, 404) from exc
    return {"ok": True, "mod": info}


@app.post("/api/library/folders")
def api_create_folder(body: FolderCreateBody) -> dict[str, Any]:
    try:
        name = create_subfolder(body.name)
    except LibraryError as exc:
        raise _err(exc) from exc
    except FileNotFoundError as exc:
        raise _err(exc, 404) from exc
    return {"ok": True, "name": name, "folders": list_subfolders()}


@app.patch("/api/library/folders")
def api_rename_folder(body: FolderRenameBody) -> dict[str, Any]:
    try:
        result = rename_subfolder(body.src, body.dst)
    except LibraryError as exc:
        raise _err(exc) from exc
    except FileNotFoundError as exc:
        raise _err(exc, 404) from exc
    return {"ok": True, **result}


@app.delete("/api/library/folders/{name:path}")
def api_delete_folder(name: str) -> dict[str, Any]:
    try:
        delete_subfolder(name)
    except LibraryError as exc:
        raise _err(exc) from exc
    except FileNotFoundError as exc:
        raise _err(exc, 404) from exc
    return {"ok": True, "folders": list_subfolders()}


@app.get("/api/library/{name:path}/meta")
def api_library_meta(name: str) -> dict[str, Any]:
    from backend.discover import default_output_dir, find_comfy_root

    output = default_output_dir(find_comfy_root())
    if output is None:
        raise HTTPException(404, "refmods folder not found")
    try:
        path = resolve_mod_path(name)
    except (FileNotFoundError, ValueError) as exc:
        raise _err(exc, 404 if isinstance(exc, FileNotFoundError) else 400) from exc
    if not path.is_file():
        raise HTTPException(404, "mod not found")
    return inspect_mod(path, output)


@app.delete("/api/library/{name:path}")
def api_delete_mod(name: str) -> dict[str, Any]:
    try:
        delete_mod(name)
    except FileNotFoundError as exc:
        raise _err(exc, 404) from exc
    except ValueError as exc:
        raise _err(exc) from exc
    return {"ok": True}


@app.post("/api/pick")
def api_pick(body: PickBody) -> dict[str, Any]:
    types = None
    if body.filetypes:
        types = [(str(pair[0]), str(pair[1])) for pair in body.filetypes if len(pair) >= 2]
    try:
        if body.kind == "file":
            path = pick_file(body.title or "Select file", body.initial, types)
        else:
            path = pick_folder(body.title or "Select folder", body.initial)
    except PickError as exc:
        raise _err(exc) from exc
    return {"path": path}


@app.post("/api/open-folder")
def api_open_folder(body: OpenFolderBody) -> dict[str, Any]:
    path = Path(body.path).expanduser()
    if not path.exists():
        raise HTTPException(404, "path not found")
    target = path if path.is_dir() else path.parent
    try:
        if sys.platform == "win32":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        raise _err(exc) from exc
    return {"ok": True, "path": str(target)}


@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


def find_free_port(start: int = 8765, limit: int = 30) -> int:
    for port in range(start, start + limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free localhost port in range")


def main() -> None:
    import uvicorn

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"H3 RefMod Studio  {url}")
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
