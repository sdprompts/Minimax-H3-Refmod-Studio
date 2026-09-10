import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from backend import library as librarymod
from backend.library import (
    LibraryError,
    create_subfolder,
    delete_mod,
    delete_subfolder,
    infer_subfolder_from_output,
    inspect_mod,
    inspect_mod_lite,
    install_studio_mod,
    iter_mod_files,
    list_mods,
    list_studio_mods,
    list_subfolders,
    normalize_subfolder,
    rename_subfolder,
    resolve_mod_path,
)


def _write_mod(path: Path, name: str, kind: str = "image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": name,
        "kind": kind,
        "latent_h": 4,
        "latent_w": 4,
        "latent_t": 1,
        "mode": "encode",
        "concept_type": "identity",
        "description": "",
    }
    save_file(
        {"latent": np.zeros((1, 24, 1, 4, 4), dtype=np.float16)},
        str(path),
        metadata={"refmod_meta": json.dumps(meta)},
    )


def test_normalize_subfolder():
    assert normalize_subfolder("") == ""
    assert normalize_subfolder(" identity\\celebs ") == "identity/celebs"
    assert normalize_subfolder("/identity/") == "identity"


def test_infer_subfolder_from_output():
    assert infer_subfolder_from_output(None) is None
    assert infer_subfolder_from_output(r"E:\ComfyUI\models\refmods\foo.safetensors") == ""
    assert infer_subfolder_from_output(r"E:\ComfyUI\models\refmods\identity\foo.safetensors") == "identity"
    assert infer_subfolder_from_output(r"E:\ComfyUI\models\refmods\celebs\june\foo.safetensors") == "celebs/june"


def test_list_mods_recursive(tmp_path, monkeypatch):
    root = tmp_path / "refmods"
    _write_mod(root / "rootmod.safetensors", "rootmod")
    _write_mod(root / "identity" / "hero.safetensors", "hero")
    (root / "graph_presets").mkdir()
    _write_mod(root / "graph_presets" / "curve.safetensors", "curve")

    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: root)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)

    files = iter_mod_files(root)
    assert any(p.name == "hero.safetensors" for p in files)
    assert any(p.name == "rootmod.safetensors" for p in files)
    assert not any("graph_presets" in p.parts for p in files)

    mods = list_mods({})
    ids = {m["id"] for m in mods}
    assert "rootmod" in ids
    assert "identity/hero" in ids
    assert list_subfolders(root) == ["identity"]

    nested = next(m for m in mods if m["id"] == "identity/hero")
    assert nested["subfolder"] == "identity"
    assert nested["token_count"] == 1 * (4 // 2) * (4 // 2)


def test_delete_nested_mod_and_empty_folder(tmp_path, monkeypatch):
    root = tmp_path / "refmods"
    path = root / "identity" / "hero.safetensors"
    _write_mod(path, "hero")
    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: root)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)

    resolved = resolve_mod_path("identity/hero")
    assert resolved == path.resolve()
    delete_mod("identity/hero")
    assert not path.is_file()
    assert not (root / "identity").exists()
    assert root.is_dir()


def test_create_rename_delete_folder(tmp_path, monkeypatch):
    root = tmp_path / "refmods"
    root.mkdir()
    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: root)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)
    monkeypatch.setattr("backend.datasets._dataset_map", lambda: {})

    assert create_subfolder("celebs") == "celebs"
    assert (root / "celebs").is_dir()
    _write_mod(root / "celebs" / "hero.safetensors", "hero")
    rename_subfolder("celebs", "identity")
    assert (root / "identity" / "hero.safetensors").is_file()
    assert not (root / "celebs").exists()
    try:
        delete_subfolder("identity")
        raise AssertionError("expected LibraryError")
    except LibraryError:
        pass
    delete_mod("identity/hero")
    create_subfolder("empty")
    delete_subfolder("empty")
    assert not (root / "empty").exists()


def test_list_and_install_studio_mod(dataset_root, tmp_path, monkeypatch):
    from backend.datasets import create_dataset, default_mod_name

    comfy = tmp_path / "refmods"
    comfy.mkdir()
    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: comfy)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)

    card = create_dataset("June", "June", subfolder="identity")
    mod_name = default_mod_name("june")
    _write_mod(Path(card["folder"]) / f"{mod_name}.safetensors", mod_name)

    items = list_studio_mods()
    assert len(items) == 1
    assert items[0]["dataset_slug"] == "june"
    assert items[0]["status"] == "missing"
    assert items[0]["local_present"] is True

    installed = install_studio_mod("june")
    dest = comfy / "identity" / f"{mod_name}.safetensors"
    assert dest.is_file()
    assert installed["name"] == mod_name
    items = list_studio_mods()
    assert items[0]["status"] == "in_sync"
    try:
        install_studio_mod("june")
        raise AssertionError("expected LibraryError")
    except LibraryError:
        pass
    install_studio_mod("june", replace=True)
    assert dest.is_file()


def test_sync_comfy_layout_moves_copies_and_keeps_dirty(dataset_root, tmp_path, monkeypatch):
    from backend.datasets import create_dataset, default_mod_name, is_dirty, load_manifest, patch_dataset
    from backend.library import sync_comfy_layout

    comfy = tmp_path / "refmods"
    comfy.mkdir()
    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: comfy)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)

    june = create_dataset("June", "June", subfolder="identity")
    june_mod = default_mod_name("june")
    _write_mod(Path(june["folder"]) / f"{june_mod}.safetensors", june_mod)
    (Path(june["folder"]) / "01.png").write_bytes(b"\x89PNG\r\n" + b"aaaa")
    install_studio_mod("june")
    old = comfy / "identity" / f"{june_mod}.safetensors"
    assert old.is_file()

    cassie = create_dataset("Cassie", "Cassie", subfolder="celebs/june")
    cassie_mod = default_mod_name("cassie")
    _write_mod(Path(cassie["folder"]) / f"{cassie_mod}.safetensors", cassie_mod)

    stray = comfy / "orphan" / "stray.safetensors"
    _write_mod(stray, "stray")
    leftover = comfy / "old-empty"
    leftover.mkdir()

    (Path(june["folder"]) / "02.png").write_bytes(b"\x89PNG\r\n" + b"bbbb")
    assert is_dirty(load_manifest(Path(june["folder"]), "june"), Path(june["folder"])) is True
    patch_dataset("june", {"subfolder": "celebs/june"})

    result = sync_comfy_layout()
    moved = comfy / "celebs" / "june" / f"{june_mod}.safetensors"
    copied = comfy / "celebs" / "june" / f"{cassie_mod}.safetensors"
    assert moved.is_file()
    assert copied.is_file()
    assert not old.exists()
    assert stray.is_file()
    assert not leftover.exists()
    assert "celebs" in result["folders_created"] or (comfy / "celebs" / "june").is_dir()
    assert result["moved"] >= 1
    assert result["copied"] >= 1
    assert is_dirty(load_manifest(Path(june["folder"]), "june"), Path(june["folder"])) is True
    june_after = patch_dataset("june", {})
    assert Path(june_after["last_extract"]["output"]).resolve() == moved.resolve()


def test_sync_comfy_layout_replaces_stale(dataset_root, tmp_path, monkeypatch):
    from backend.datasets import create_dataset, default_mod_name
    from backend.library import sync_comfy_layout

    comfy = tmp_path / "refmods"
    comfy.mkdir()
    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: comfy)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)

    card = create_dataset("June", "June", subfolder="identity")
    mod_name = default_mod_name("june")
    local = Path(card["folder"]) / f"{mod_name}.safetensors"
    _write_mod(local, mod_name)
    install_studio_mod("june")
    dest = comfy / "identity" / f"{mod_name}.safetensors"
    dest.write_bytes(b"stale")
    result = sync_comfy_layout()
    assert result["replaced"] == 1
    assert dest.stat().st_size == local.stat().st_size


def test_list_mods_lite_skips_safetensors_open(tmp_path, monkeypatch):
    root = tmp_path / "refmods"
    _write_mod(root / "identity" / "hero.safetensors", "hero")
    monkeypatch.setattr("backend.library.default_output_dir", lambda _root=None: root)
    monkeypatch.setattr("backend.library.find_comfy_root", lambda: None)

    def boom(*_args, **_kwargs):
        raise AssertionError("lite list should not open safetensors")

    monkeypatch.setattr(librarymod, "safe_open", boom)
    lite = inspect_mod_lite(root / "identity" / "hero.safetensors", root)
    assert lite["id"] == "identity/hero"
    assert lite["lite"] is True
    assert lite["token_count"] is None
    mods = list_mods({}, lite=True)
    assert mods[0]["id"] == "identity/hero"
    assert mods[0]["bytes"] > 0
    assert mods[0]["token_count"] is None


def test_inspect_mod_reuses_cache(tmp_path, monkeypatch):
    path = tmp_path / "hero.safetensors"
    _write_mod(path, "hero")
    first = inspect_mod(path, tmp_path)
    assert first["token_count"] == 1 * (4 // 2) * (4 // 2)

    def boom(*_args, **_kwargs):
        raise AssertionError("cached inspect should not reopen the file")

    monkeypatch.setattr(librarymod, "safe_open", boom)
    second = inspect_mod(path, tmp_path)
    assert second["token_count"] == first["token_count"]
    assert second["concept_type"] == "identity"
