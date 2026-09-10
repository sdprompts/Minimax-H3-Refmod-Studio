from pathlib import Path

from backend.discover import (
    discover_python,
    find_extract_script,
    find_vae,
    install_kind,
    is_embed_python,
    parse_extra_model_paths,
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


def test_resolve_windows_portable_wrapper(tmp_path):
    wrapper = tmp_path / "ComfyUI_windows_portable"
    inner = wrapper / "ComfyUI"
    _touch(inner / "main.py")
    embed = _touch(wrapper / "python_embeded" / "python.exe")

    assert resolve_comfy_root(wrapper) == inner.resolve()
    assert discover_python(inner) == embed.resolve()
    assert install_kind(inner.resolve(), embed.resolve()) == "portable"


def test_resolve_manual_venv(tmp_path):
    root = tmp_path / "ComfyUI"
    _touch(root / "main.py")
    (root / "comfy").mkdir()
    python = _touch(root / "venv" / "bin" / "python")

    assert resolve_comfy_root(root) == root.resolve()
    assert discover_python(root) == python.resolve()
    assert install_kind(root.resolve(), python.resolve()) == "venv"


def test_resolve_desktop_user_dir(tmp_path):
    data = tmp_path / "Documents" / "ComfyUI"
    (data / "custom_nodes").mkdir(parents=True)
    (data / "models" / "vae").mkdir(parents=True)
    python = _touch(data / ".venv" / "Scripts" / "python.exe")

    assert resolve_comfy_root(data) == data.resolve()
    assert discover_python(data) == python.resolve()
    assert install_kind(data.resolve(), python.resolve()) == "desktop"


def test_embed_preferred_over_inner_venv(tmp_path):
    wrapper = tmp_path / "ComfyUI_windows_portable"
    inner = wrapper / "ComfyUI"
    _touch(inner / "main.py")
    embed = _touch(wrapper / "python_embeded" / "python.exe")
    _touch(inner / "venv" / "Scripts" / "python.exe")

    assert discover_python(inner) == embed.resolve()


def test_resolve_missing(tmp_path):
    assert resolve_comfy_root(tmp_path / "nope") is None
    assert discover_python(tmp_path) is None


def test_embed_python_gets_no_user_site_flag(tmp_path):
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
    assert argv[:4] == [str(python), "-s", "-u", str(tmp_path / "extract_mod.py")]


def test_venv_python_does_not_get_embed_flag(tmp_path):
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
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
    assert argv[:3] == [str(python), "-u", str(tmp_path / "extract_mod.py")]
    assert "-s" not in argv
    assert "-I" not in argv


def test_python_candidates_include_sibling_embed(tmp_path):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    names = {p.name for p in python_candidates(comfy) if p.parent.name.startswith("python_embed")}
    assert "python.exe" in names


def test_parse_extra_model_paths_sections():
    text = """
# comment
a111:
    base_path: D:/stable-diffusion-webui/
    vae: models/VAE
    loras: |
         models/Lora
         models/LyCORIS

comfyui:
     base_path: /models
     vae: vae
     custom_nodes: /extra/nodes
     refmods: refmods
"""
    sections = parse_extra_model_paths(text)
    assert len(sections) == 2
    assert sections[0]["base_path"] == "D:/stable-diffusion-webui/"
    assert sections[0]["vae"] == "models/VAE"
    assert sections[0]["loras"] == ["models/Lora", "models/LyCORIS"]
    assert sections[1]["vae"] == "vae"
    assert sections[1]["custom_nodes"] == "/extra/nodes"


def test_find_vae_and_extract_in_extra_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.discover.load_config",
        lambda: {
            "comfy_root": "",
            "python_path": "",
            "vae_path": "",
            "output_dir": "",
            "extract_script": "",
        },
    )
    root = tmp_path / "ComfyUI"
    _touch(root / "main.py")
    extra_vae = tmp_path / "shared" / "vae"
    vae = _touch(extra_vae / "minimax_h3_video_vae_fp16.safetensors")
    extra_nodes = tmp_path / "shared" / "custom_nodes"
    script = _touch(extra_nodes / "ComfyUI-MiniMaxH3Mod" / "extract_mod.py")
    (root / "extra_model_paths.yaml").write_text(
        "shared:\n  base_path: %s\n  vae: vae\n  custom_nodes: %s\n"
        % (tmp_path / "shared", extra_nodes),
        encoding="utf-8",
    )
    assert find_vae(root) == vae.resolve()
    assert find_extract_script(root) == script.resolve()
