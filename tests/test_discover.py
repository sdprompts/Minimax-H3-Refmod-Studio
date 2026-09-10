from pathlib import Path

from backend.discover import (
    discover_python,
    install_kind,
    is_embed_python,
    python_candidates,
    resolve_comfy_root,
)
from backend.extract import build_argv


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_resolve_easy_install_wrapper(tmp_path):
    wrapper = tmp_path / "ComfyUI-Easy-Install"
    inner = wrapper / "ComfyUI"
    _touch(inner / "main.py")
    (inner / "comfy").mkdir()
    embed = _touch(wrapper / "python_embeded" / "python.exe")

    assert resolve_comfy_root(wrapper) == inner.resolve()
    assert resolve_comfy_root(inner) == inner.resolve()
    assert discover_python(inner) == embed.resolve()
    assert is_embed_python(embed)
    assert install_kind(inner.resolve(), embed.resolve()) == "easy-install"


def test_resolve_missing(tmp_path):
    assert resolve_comfy_root(tmp_path / "nope") is None
    assert discover_python(tmp_path) is None


def test_embed_python_gets_isolated_flag(tmp_path):
    python = tmp_path / "python_embeded" / "python.exe"
    _touch(python)
    argv = build_argv(
        python=python,
        script=tmp_path / "extract_mod.py",
        vae=tmp_path / "vae.safetensors",
        output_dir=tmp_path / "refmods",
        name="mod",
        description="",
        concept_type="identity",
        extract={"mode": "encode", "resolution": 1024, "max_tokens": 8192, "pool": 16, "identity": 0, "multiplier": 1, "device": "auto"},
        images=[],
        videos=[],
    )
    assert argv[:4] == [str(python), "-I", "-u", str(tmp_path / "extract_mod.py")]


def test_python_candidates_include_sibling_embed(tmp_path):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    names = {p.name for p in python_candidates(comfy) if p.parent.name.startswith("python_embed")}
    assert "python.exe" in names
