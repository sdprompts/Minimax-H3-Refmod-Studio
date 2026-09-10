"""Locate ComfyUI, the H3 video VAE, extract_mod.py, and a usable Python."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .config import load_config

PACK_REL = Path("custom_nodes") / "ComfyUI-MiniMaxH3Mod" / "extract_mod.py"
VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"


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


def _looks_like_comfy(root: Path) -> bool:
    return _exists_file(root / "main.py") or _exists_dir(root / "comfy")


def resolve_comfy_root(path: Path | str | None) -> Path | None:
    """Accept a ComfyUI folder or a ComfyUI-Easy-Install / portable wrapper.

    Easy-Install layout::

        ComfyUI-Easy-Install/
          python_embeded/python.exe   # not a venv
          ComfyUI/main.py
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
    ]
    wrappers = ("ComfyUI-Easy-Install", "ComfyUI_windows_portable")
    for base in bases:
        add(base)
        for wrap in wrappers:
            add(base / wrap)
            add(base / wrap / "ComfyUI")
    add(home / "Desktop" / "ComfyUI-Easy-Install")
    add(home / "OneDrive" / "Desktop" / "ComfyUI-Easy-Install")
    add(Path(r"C:\ComfyUI-Easy-Install"))
    add(Path(r"D:\ComfyUI-Easy-Install"))
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


def find_extract_script(comfy_root: Path | None = None) -> Path | None:
    cfg = load_config()
    if cfg.get("extract_script"):
        path = Path(cfg["extract_script"]).expanduser()
        if _exists_file(path):
            return path.resolve()
    root = comfy_root or find_comfy_root()
    if root is None:
        return None
    path = root / PACK_REL
    if _exists_file(path):
        return path.resolve()
    return None


def find_vae(comfy_root: Path | None = None) -> Path | None:
    cfg = load_config()
    if cfg.get("vae_path"):
        path = Path(cfg["vae_path"]).expanduser()
        if _exists_file(path):
            return path.resolve()
    root = comfy_root or find_comfy_root()
    if root is None:
        return None
    vae_dir = root / "models" / "vae"
    preferred = vae_dir / VAE_NAME
    if _exists_file(preferred):
        return preferred.resolve()
    if not _exists_dir(vae_dir):
        return None
    matches: list[Path] = []
    try:
        for item in vae_dir.iterdir():
            if not item.is_file():
                continue
            name = item.name.lower()
            if name.endswith(".safetensors") and "h3" in name and "vae" in name:
                matches.append(item)
    except OSError:
        return None
    if not matches:
        return None
    video_first = [m for m in matches if "video" in m.name.lower()]
    pick = (video_first or matches)[0]
    return pick.resolve()


def is_embed_python(python: Path) -> bool:
    return python.parent.name.lower().startswith("python_embed")


def python_candidates(comfy_root: Path) -> list[Path]:
    """venv inside ComfyUI, or Easy-Install/portable embed next to it."""
    parent = comfy_root.parent
    names = (
        "python_embeded",
        "python_embedded",
        "Python_embeded",
        "python_embeded_3.12",
        "python_embeded_3.11",
    )
    out = [
        comfy_root / "venv" / "Scripts" / "python.exe",
        comfy_root / "venv" / "bin" / "python",
        comfy_root / ".venv" / "Scripts" / "python.exe",
        comfy_root / ".venv" / "bin" / "python",
    ]
    for name in names:
        out.append(comfy_root / name / "python.exe")
        out.append(parent / name / "python.exe")
        out.append(comfy_root / name / "bin" / "python")
        out.append(parent / name / "bin" / "python")
        out.append(parent / name / "python")
    return out


def discover_python(comfy_root: Path) -> Path | None:
    for path in python_candidates(comfy_root):
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
    return discover_python(root)


def install_kind(comfy_root: Path | None, python: Path | None) -> str:
    if python is not None and is_embed_python(python):
        if comfy_root is not None and python.parent.parent.resolve() == comfy_root.parent.resolve():
            return "easy-install"
        return "portable-embed"
    if python is not None and "venv" in {p.name.lower() for p in python.parents}:
        return "venv"
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
    if root is None:
        return None
    return (root / "models" / "refmods").resolve()


def _run_probe(
    python: Path,
    code: str,
    cwd: Path | None = None,
    timeout: float = 20.0,
    isolate: bool = False,
) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    args = [str(python)]
    if isolate:
        args.append("-I")
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
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
        if not isolate:
            env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
            kwargs["env"] = env
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(**kwargs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, out


def probe_runtime(python: Path, comfy_root: Path | None) -> dict[str, Any]:
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
    if comfy_root is not None:
        comfy_ok, comfy_out = _run_probe(
            python,
            "import comfy, comfy.sd, comfy.utils; print('ok')",
            cwd=comfy_root,
            isolate=isolate,
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
        runtime = probe_runtime(python, comfy_root)
    missing: list[str] = []
    if comfy_root is None:
        missing.append("ComfyUI root (folder with main.py)")
    if script is None:
        missing.append("ComfyUI-MiniMaxH3Mod (custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py)")
    if vae is None:
        missing.append("MiniMax-H3 video VAE (models/vae/minimax_h3_video_vae_fp16.safetensors)")
    if python is None and comfy_root is not None:
        missing.append(
            "Python for this ComfyUI (Easy-Install uses python_embeded\\python.exe "
            "next to the ComfyUI folder — there is no venv)"
        )
    if probe and python is not None and not runtime.get("comfy_import_ok"):
        missing.append("ComfyUI Python cannot import comfy (use the ComfyUI venv / python_embeded)")
    ready = not missing
    from .config import public_config
    from .hub import hub_settings
    from .library import list_subfolders

    hub = hub_settings()
    return {
        "ready": ready,
        "missing": missing,
        "probed": probe,
        "comfy_root": str(comfy_root) if comfy_root else "",
        "extract_script": str(script) if script else "",
        "pack_version": _pack_version(script),
        "vae_path": str(vae) if vae else "",
        "python_path": str(python) if python else "",
        "output_dir": str(output_dir) if output_dir else "",
        "datasets_root": cfg.get("datasets_root") or "",
        "mod_prefix": cfg.get("mod_prefix") or "sdprompts",
        "refmod_folders": list_subfolders(output_dir),
        "install_kind": install_kind(comfy_root, python),
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
