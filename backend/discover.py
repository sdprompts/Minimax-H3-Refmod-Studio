"""Locate ComfyUI, the H3 video VAE, extract_mod.py, and a usable Python.

Works with the usual local installs: git clone, Windows portable, Easy-Install,
Stability Matrix, ComfyUI Desktop, and a venv next to the app.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import load_config

PACK_REL = Path("custom_nodes") / "ComfyUI-MiniMaxH3Mod" / "extract_mod.py"
VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
EMBED_DIR_NAMES = (
    "python_embeded",
    "python_embedded",
    "Python_embeded",
    "python_embeded_3.13",
    "python_embeded_3.12",
    "python_embeded_3.11",
)
INSTALL_LABELS = {
    "easy-install": "ComfyUI-Easy-Install",
    "portable": "Windows portable",
    "portable-embed": "embedded Python",
    "desktop": "ComfyUI Desktop",
    "venv": "venv",
    "manual": "manual install",
    "unknown": "unknown",
}


def _exists_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _exists_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _strip_yaml_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _looks_like_comfy(root: Path) -> bool:
    if _exists_file(root / "main.py") or _exists_dir(root / "comfy"):
        return True
    # ComfyUI Desktop user data dir: models + custom_nodes, often no main.py.
    if _exists_dir(root / "custom_nodes") and (
        _exists_dir(root / "models")
        or _exists_dir(root / ".venv")
        or _exists_dir(root / "venv")
    ):
        return True
    return False


def resolve_comfy_root(path: Path | str | None) -> Path | None:
    """Accept a ComfyUI folder or a wrapper that contains one.

    Typical wrappers::

        ComfyUI-Easy-Install/ComfyUI/main.py + python_embeded/
        ComfyUI_windows_portable/ComfyUI/main.py + python_embeded/
    """
    if not path:
        return None
    root = Path(str(path)).expanduser()
    try:
        root = root.resolve()
    except OSError:
        return None
    if _looks_like_comfy(root):
        return root
    inner = root / "ComfyUI"
    if _looks_like_comfy(inner):
        return inner.resolve()
    return None


def _desktop_config_dirs() -> list[Path]:
    home = Path.home()
    appdata = os.environ.get("APPDATA")
    dirs = []
    if appdata:
        dirs.append(Path(appdata) / "ComfyUI")
        dirs.append(Path(appdata) / "Comfy Desktop")
    dirs.extend(
        [
            home / "AppData" / "Roaming" / "ComfyUI",
            home / "Library" / "Application Support" / "ComfyUI",
            home / ".config" / "ComfyUI",
        ]
    )
    return dirs


def _desktop_source_roots() -> list[Path]:
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    home = Path.home()
    return [
        local / "Programs" / "@comfyorgcomfyui-electron" / "resources" / "ComfyUI",
        local / "Programs" / "comfyui-electron" / "resources" / "ComfyUI",
        local / "Programs" / "ComfyUI" / "resources" / "ComfyUI",
        Path("/Applications/ComfyUI.app/Contents/Resources/ComfyUI"),
        home / "Applications" / "ComfyUI.app" / "Contents" / "Resources" / "ComfyUI",
    ]


def _desktop_user_roots() -> list[Path]:
    found: list[Path] = []
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    for cfg_dir in _desktop_config_dirs():
        cfg = cfg_dir / "config.json"
        if not _exists_file(cfg):
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("basePath", "base_path", "installPath"):
            raw = data.get(key)
            if raw:
                found.append(Path(str(raw)))
    for parent in (
        local / "Comfy-Desktop" / "ComfyUI-Installs",
        home / "ComfyUI-Installs",
        home / "Comfy-Desktop" / "ComfyUI-Installs",
    ):
        if not _exists_dir(parent):
            continue
        try:
            found.extend([p for p in parent.iterdir() if p.is_dir()])
        except OSError:
            pass
    found.extend(
        [
            home / "Documents" / "ComfyUI",
            home / "ComfyUI",
            local / "Comfy-Desktop" / "ComfyUI-Shared",
            home / "ComfyUI-Shared",
        ]
    )
    return found


def _add_comfyish_children(add, folder: Path) -> None:
    if not _exists_dir(folder):
        return
    try:
        children = list(folder.iterdir())
    except OSError:
        return
    for child in children:
        try:
            if not child.is_dir():
                continue
        except OSError:
            continue
        name = child.name.lower().replace(" ", "")
        if "comfy" in name:
            add(child)
            add(child / "ComfyUI")


def candidate_comfy_roots() -> list[Path]:
    cfg = load_config()
    found: list[Path] = []
    seen: set[str] = set()

    def add(raw: str | Path | None) -> None:
        if not raw:
            return
        path = Path(str(raw)).expanduser()
        try:
            path = path.resolve()
        except OSError:
            return
        key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    add(cfg.get("comfy_root"))
    add(os.environ.get("COMFYUI_ROOT"))
    add(os.environ.get("COMFYUI_PATH"))
    for root in _desktop_user_roots():
        add(root)
    home = Path.home()
    bases = [
        Path(r"C:\ComfyUI"),
        Path(r"C:\COMFYUI"),
        Path(r"C:\Development\ComfyUI"),
        Path(r"D:\ComfyUI"),
        Path(r"D:\COMFYUI"),
        Path(r"E:\ComfyUI"),
        home / "Documents" / "ComfyUI",
        home / "ComfyUI",
        home / "Desktop" / "ComfyUI",
        home / "OneDrive" / "Desktop" / "ComfyUI",
        home / "AppData" / "Roaming" / "StabilityMatrix" / "Packages" / "ComfyUI",
        home / "AppData" / "Roaming" / "StabilityMatrix" / "Data" / "Packages" / "ComfyUI",
        home / ".local" / "share" / "StabilityMatrix" / "Packages" / "ComfyUI",
        Path("/opt/ComfyUI"),
        Path("/opt/comfyui"),
    ]
    wrappers = (
        "ComfyUI-Easy-Install",
        "ComfyUI_windows_portable",
        "ComfyUI_windows_portable_nvidia",
        "ComfyUI_windows_portable_amd",
        "ComfyUI_windows_portable_intel",
        "ComfyUI_windows_portable_nvidia_cu126",
    )
    for base in bases:
        add(base)
        for wrap in wrappers:
            add(base / wrap)
            add(base / wrap / "ComfyUI")
    for wrap in wrappers:
        add(home / "Desktop" / wrap)
        add(home / "OneDrive" / "Desktop" / wrap)
        add(home / "Downloads" / wrap)
        add(Path(r"C:\\") / wrap)
        add(Path(r"D:\\") / wrap)
        add(Path(r"E:\\") / wrap)
    for folder in (
        home / "Desktop",
        home / "OneDrive" / "Desktop",
        home / "Downloads",
        home / "Documents",
    ):
        _add_comfyish_children(add, folder)
    for src in _desktop_source_roots():
        add(src)
    return found


def find_comfy_root() -> Path | None:
    cfg = load_config()
    if cfg.get("comfy_root"):
        resolved = resolve_comfy_root(cfg["comfy_root"])
        if resolved is not None:
            return resolved
    for root in candidate_comfy_roots():
        resolved = resolve_comfy_root(root)
        if resolved is not None:
            return resolved
    return None


def find_comfy_source(comfy_root: Path | None = None) -> Path | None:
    """Folder that contains the `comfy` package / main.py (for PYTHONPATH)."""
    root = comfy_root or find_comfy_root()
    if root is not None:
        if _exists_dir(root / "comfy") or _exists_file(root / "main.py"):
            return root
        inner = root / "ComfyUI"
        if _exists_dir(inner / "comfy") or _exists_file(inner / "main.py"):
            return inner.resolve()
    for src in _desktop_source_roots():
        if _exists_dir(src / "comfy") or _exists_file(src / "main.py"):
            try:
                return src.resolve()
            except OSError:
                continue
    return root


def parse_extra_model_paths(text: str) -> list[dict[str, Any]]:
    """Minimal YAML map parser for extra_model_paths.yaml (no PyYAML)."""
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    multiline_key: str | None = None
    multiline_indent = 0

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" \t"))

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = indent_of(raw)
        stripped = raw.strip()
        if multiline_key is not None and current is not None:
            if indent > multiline_indent:
                current.setdefault(multiline_key, [])
                if not isinstance(current[multiline_key], list):
                    current[multiline_key] = [str(current[multiline_key])]
                current[multiline_key].append(_strip_yaml_quotes(stripped))
                continue
            multiline_key = None
        if ":" not in stripped:
            continue
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        if indent == 0:
            current = {}
            sections.append(current)
            if rest in ("|", ">"):
                multiline_key = key
                multiline_indent = indent
                current[key] = []
            elif rest:
                current[key] = _strip_yaml_quotes(rest)
            continue
        if current is None:
            current = {}
            sections.append(current)
        if rest in ("|", ">"):
            multiline_key = key
            multiline_indent = indent
            current[key] = []
            continue
        current[key] = _strip_yaml_quotes(rest)
    return [section for section in sections if section]


def extra_model_path_files(comfy_root: Path | None = None) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not _exists_file(path):
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        files.append(resolved)

    if comfy_root is not None:
        add(comfy_root / "extra_model_paths.yaml")
        source = find_comfy_source(comfy_root)
        if source is not None:
            add(source / "extra_model_paths.yaml")
    for cfg_dir in _desktop_config_dirs():
        add(cfg_dir / "extra_models_config.yaml")
        add(cfg_dir / "extra_model_paths.yaml")
    return files


def extra_search_dirs(comfy_root: Path | None, *keys: str) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def add_dir(path: Path) -> None:
        if not _exists_dir(path):
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        dirs.append(resolved)

    for yaml_path in extra_model_path_files(comfy_root):
        try:
            text = yaml_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for section in parse_extra_model_paths(text):
            base_raw = section.get("base_path") or section.get("basePath")
            base = Path(str(base_raw)).expanduser() if base_raw else yaml_path.parent
            for key in keys:
                raw = section.get(key)
                values: list[str]
                if isinstance(raw, list):
                    values = [str(v) for v in raw if v]
                elif raw:
                    values = [str(raw)]
                else:
                    values = []
                for val in values:
                    path = Path(val).expanduser()
                    if not path.is_absolute():
                        path = base / path
                    add_dir(path)
    return dirs


def _custom_node_dirs(comfy_root: Path | None) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not _exists_dir(path):
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        dirs.append(resolved)

    if comfy_root is not None:
        add(comfy_root / "custom_nodes")
        source = find_comfy_source(comfy_root)
        if source is not None:
            add(source / "custom_nodes")
    for extra in extra_search_dirs(comfy_root, "custom_nodes"):
        add(extra)
    return dirs


def _find_extract_in_custom_nodes(custom_nodes: Path) -> Path | None:
    direct = custom_nodes / "ComfyUI-MiniMaxH3Mod" / "extract_mod.py"
    if _exists_file(direct):
        return direct.resolve()
    try:
        children = list(custom_nodes.iterdir())
    except OSError:
        return None
    for child in children:
        try:
            if not child.is_dir():
                continue
        except OSError:
            continue
        compact = child.name.lower().replace("-", "").replace("_", "")
        if "minimaxh3" not in compact:
            continue
        script = child / "extract_mod.py"
        if _exists_file(script):
            return script.resolve()
    return None


def find_extract_script(comfy_root: Path | None = None) -> Path | None:
    cfg = load_config()
    if cfg.get("extract_script"):
        path = Path(cfg["extract_script"]).expanduser()
        if _exists_file(path):
            return path.resolve()
    root = comfy_root or find_comfy_root()
    for folder in _custom_node_dirs(root):
        found = _find_extract_in_custom_nodes(folder)
        if found is not None:
            return found
    if root is not None:
        fallback = root / PACK_REL
        if _exists_file(fallback):
            return fallback.resolve()
    return None


def _vae_dirs(comfy_root: Path | None) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not _exists_dir(path):
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        dirs.append(resolved)

    if comfy_root is not None:
        add(comfy_root / "models" / "vae")
        source = find_comfy_source(comfy_root)
        if source is not None:
            add(source / "models" / "vae")
    for extra in extra_search_dirs(comfy_root, "vae"):
        add(extra)
    return dirs


def _pick_h3_vae(folder: Path) -> Path | None:
    preferred = folder / VAE_NAME
    if _exists_file(preferred):
        return preferred.resolve()
    matches: list[Path] = []
    try:
        items = list(folder.iterdir())
    except OSError:
        return None
    for item in items:
        try:
            if not item.is_file():
                continue
        except OSError:
            continue
        name = item.name.lower()
        if name.endswith(".safetensors") and "h3" in name and "vae" in name:
            matches.append(item)
    if not matches:
        return None
    video_first = [m for m in matches if "video" in m.name.lower()]
    return (video_first or matches)[0].resolve()


def find_vae(comfy_root: Path | None = None) -> Path | None:
    cfg = load_config()
    if cfg.get("vae_path"):
        path = Path(cfg["vae_path"]).expanduser()
        if _exists_file(path):
            return path.resolve()
    root = comfy_root or find_comfy_root()
    for folder in _vae_dirs(root):
        found = _pick_h3_vae(folder)
        if found is not None:
            return found
    return None


def is_embed_python(python: Path) -> bool:
    return python.parent.name.lower().startswith("python_embed")


def python_candidates(comfy_root: Path) -> list[Path]:
    """Embedded Python (portable / Easy-Install), then venv (Desktop / SM / manual)."""
    parent = comfy_root.parent
    out: list[Path] = []
    for name in EMBED_DIR_NAMES:
        for base in (parent, comfy_root):
            out.extend(
                [
                    base / name / "python.exe",
                    base / name / "python",
                    base / name / "bin" / "python",
                    base / name / "bin" / "python3",
                ]
            )
    for base in (comfy_root, parent):
        for vname in (".venv", "venv"):
            out.extend(
                [
                    base / vname / "Scripts" / "python.exe",
                    base / vname / "bin" / "python",
                    base / vname / "bin" / "python3",
                ]
            )
    return out


def discover_python(comfy_root: Path) -> Path | None:
    for path in python_candidates(comfy_root):
        if _exists_file(path):
            return path.resolve()
    # Desktop electron resources have main.py but the venv lives in the user data dir.
    if _exists_file(comfy_root / "main.py"):
        for user_root in _desktop_user_roots():
            resolved = resolve_comfy_root(user_root)
            if resolved is None:
                continue
            try:
                if resolved.resolve() == comfy_root.resolve():
                    continue
            except OSError:
                continue
            for path in python_candidates(resolved):
                if _exists_file(path):
                    return path.resolve()
    return None


def find_python(comfy_root: Path | None = None) -> Path | None:
    cfg = load_config()
    if cfg.get("python_path"):
        path = Path(cfg["python_path"]).expanduser()
        if _exists_file(path):
            return path.resolve()
    root = comfy_root or find_comfy_root()
    if root is None:
        return None
    found = discover_python(root)
    if found is not None:
        return found
    # Desktop: user data dir as root, Python in that .venv even if discover missed it.
    for user_root in _desktop_user_roots():
        resolved = resolve_comfy_root(user_root)
        if resolved is None:
            continue
        found = discover_python(resolved)
        if found is not None:
            return found
    return None


def install_kind(comfy_root: Path | None, python: Path | None) -> str:
    parent_name = ""
    if comfy_root is not None:
        parent_name = comfy_root.parent.name.lower()
    if python is not None and is_embed_python(python):
        compact = parent_name.replace("-", "").replace("_", "")
        if "easyinstall" in compact:
            return "easy-install"
        if "portable" in parent_name:
            return "portable"
        if comfy_root is not None:
            try:
                if python.parent.parent.resolve() == comfy_root.parent.resolve():
                    return "portable"
            except OSError:
                pass
        return "portable-embed"
    if comfy_root is not None and not _exists_file(comfy_root / "main.py"):
        if _exists_dir(comfy_root / ".venv") or (
            python is not None and any(part.lower() == ".venv" for part in python.parts)
        ):
            return "desktop"
    if python is not None and any(p.name.lower() in {".venv", "venv"} for p in python.parents):
        return "venv"
    if comfy_root is not None and _exists_file(comfy_root / "main.py"):
        return "manual"
    return "unknown"


def _pack_version(script: Path | None) -> str:
    if script is None:
        return ""
    init = script.parent / "__init__.py"
    if not init.is_file():
        return ""
    try:
        text = init.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("__version__"):
            _, _, rest = stripped.partition("=")
            return rest.strip().strip("\"'")
    return ""


def default_output_dir(comfy_root: Path | None = None) -> Path | None:
    cfg = load_config()
    if cfg.get("output_dir"):
        path = Path(cfg["output_dir"]).expanduser()
        return path.resolve()
    root = comfy_root or find_comfy_root()
    extras = extra_search_dirs(root, "refmods")
    if extras:
        return extras[0]
    if root is None:
        return None
    return (root / "models" / "refmods").resolve()


def _run_probe(
    python: Path,
    code: str,
    cwd: Path | None = None,
    timeout: float = 20.0,
    isolate: bool = False,
    extra_path: Path | None = None,
) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if isolate:
        env["PYTHONNOUSERSITE"] = "1"
    args = [str(python)]
    if isolate:
        args.append("-s")
    args.extend(["-c", code])
    kwargs: dict[str, Any] = {
        "args": args,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "env": env,
    }
    path_parts: list[str] = []
    if extra_path is not None:
        path_parts.append(str(extra_path))
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
        path_parts.append(str(cwd))
    if path_parts:
        env["PYTHONPATH"] = os.pathsep.join(path_parts) + os.pathsep + env.get("PYTHONPATH", "")
        kwargs["env"] = env
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(**kwargs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, out


def probe_runtime(
    python: Path,
    comfy_root: Path | None,
    comfy_source: Path | None = None,
) -> dict[str, Any]:
    isolate = is_embed_python(python)
    cuda_ok, cuda_out = _run_probe(
        python,
        "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')",
        isolate=isolate,
    )
    device = "unknown"
    if cuda_ok:
        last = cuda_out.splitlines()[-1].strip() if cuda_out else ""
        if last in ("cuda", "cpu"):
            device = last
    comfy_ok = False
    comfy_out = ""
    source = comfy_source or find_comfy_source(comfy_root) or comfy_root
    if source is not None:
        injected = json.dumps(str(source))
        comfy_ok, comfy_out = _run_probe(
            python,
            f"import sys; sys.path.insert(0, {injected}); import comfy, comfy.sd, comfy.utils; print('ok')",
            cwd=source,
            isolate=isolate,
            extra_path=source,
        )
    return {
        "device": device,
        "torch_ok": cuda_ok,
        "torch_detail": cuda_out[-500:] if cuda_out else "",
        "comfy_import_ok": comfy_ok,
        "comfy_detail": comfy_out[-500:] if comfy_out else "",
    }


def health(probe: bool = False) -> dict[str, Any]:
    cfg = load_config()
    comfy_root = find_comfy_root()
    comfy_source = find_comfy_source(comfy_root)
    script = find_extract_script(comfy_root)
    vae = find_vae(comfy_root)
    python = find_python(comfy_root)
    output_dir = default_output_dir(comfy_root)
    runtime = {
        "device": "unknown",
        "torch_ok": False,
        "torch_detail": "",
        "comfy_import_ok": False,
        "comfy_detail": "",
    }
    if probe and python is not None:
        runtime = probe_runtime(python, comfy_root, comfy_source)
    missing: list[str] = []
    if comfy_root is None:
        missing.append("ComfyUI root (folder with main.py, or Desktop data folder)")
    if script is None:
        missing.append("ComfyUI-MiniMaxH3Mod (custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py)")
    if vae is None:
        missing.append("MiniMax-H3 video VAE (models/vae/minimax_h3_video_vae_fp16.safetensors)")
    if python is None and comfy_root is not None:
        missing.append(
            "Python that launches this ComfyUI (portable/Easy-Install: python_embeded next to "
            "ComfyUI; Desktop/venv: .venv or venv). Leave Python blank in Setup to auto-detect."
        )
    if probe and python is not None and not runtime.get("comfy_import_ok"):
        missing.append(
            "That Python cannot import comfy — pick the interpreter ComfyUI actually runs "
            "(python_embeded, .venv, or venv), not this studio's Python"
        )
    ready = not missing
    from .config import public_config
    from .hub import hub_settings
    from .library import list_subfolders

    hub = hub_settings()
    kind = install_kind(comfy_root, python)
    return {
        "ready": ready,
        "missing": missing,
        "probed": probe,
        "comfy_root": str(comfy_root) if comfy_root else "",
        "comfy_source": str(comfy_source) if comfy_source else "",
        "extract_script": str(script) if script else "",
        "pack_version": _pack_version(script),
        "vae_path": str(vae) if vae else "",
        "python_path": str(python) if python else "",
        "output_dir": str(output_dir) if output_dir else "",
        "datasets_root": cfg.get("datasets_root") or "",
        "mod_prefix": cfg.get("mod_prefix") or "sdprompts",
        "refmod_folders": list_subfolders(output_dir),
        "install_kind": kind,
        "install_label": INSTALL_LABELS.get(kind, kind),
        "device": runtime.get("device") or "unknown",
        "torch_ok": bool(runtime.get("torch_ok")),
        "comfy_import_ok": bool(runtime.get("comfy_import_ok")),
        "probe_detail": {
            "torch": runtime.get("torch_detail") or "",
            "comfy": runtime.get("comfy_detail") or "",
        },
        "config": public_config(),
        "hub_ready": hub["ready"],
        "hub_missing": hub["missing"],
        "hub_repo": hub["repo_id"],
    }
