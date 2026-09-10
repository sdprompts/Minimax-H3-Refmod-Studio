import json
from pathlib import Path

import pytest

from backend.datasets import (
    DatasetError,
    bump_mod_name,
    content_hash,
    create_dataset,
    create_datasets_from_sheets,
    default_mod_name,
    delete_file,
    enabled_files,
    is_dirty,
    list_datasets,
    load_manifest,
    patch_dataset,
    record_extract,
    set_caption,
    slugify,
)
from tests.test_split import contact_sheet, sheet_png


def test_remap_dataset_subfolders(dataset_root):
    from backend.datasets import remap_dataset_subfolders

    create_dataset("June", "June", subfolder="identity")
    create_dataset("Nested", "Nested", subfolder="identity/june")
    assert remap_dataset_subfolders("identity", "celebs") == 2
    assert load_manifest(dataset_root / "june", "june")["subfolder"] == "celebs"
    assert load_manifest(dataset_root / "nested", "nested")["subfolder"] == "celebs/june"


def test_list_datasets_is_a_card_index(dataset_root, monkeypatch):
    from backend import datasets as dsmod

    card = create_dataset("June", "June")
    folder = Path(card["folder"])
    (folder / "01.png").write_bytes(b"\x89PNG\r\n" + b"aaaa")
    (folder / "02.png").write_bytes(b"\x89PNG\r\n" + b"bbbb")
    patch_dataset("june", {
        "items": [
            {"file": "01.png", "enabled": True},
            {"file": "02.png", "enabled": True},
        ],
    })
    probed = []
    real = dsmod.probe_size

    def wrapped(path):
        probed.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(dsmod, "probe_size", wrapped)
    listed = list_datasets()
    june = next(d for d in listed if d["slug"] == "june")
    assert june["image_count"] == 2
    assert june["cover"] == "01.png"
    assert "items" not in june
    assert probed == ["01.png"]


def test_create_prefills_description_from_display_name(dataset_root):
    named = create_dataset("feliciaday", "Felicia Day")
    assert named["display_name"] == "Felicia Day"
    assert named["description"] == "Felicia Day"
    fallback = create_dataset("only-slug")
    assert fallback["display_name"] == "only-slug"
    assert fallback["description"] == "only-slug"


def test_new_dataset_defaults_to_identity_subfolder(dataset_root):
    card = create_dataset("June", "June")
    assert card["subfolder"] == "identity"
    assert card["description"] == "June"
    patched = patch_dataset("june", {"subfolder": "celebs/june"})
    assert patched["subfolder"] == "celebs/june"
    root = patch_dataset("june", {"subfolder": ""})
    assert root["subfolder"] == ""


def test_infer_subfolder_from_existing_extract(dataset_root):
    card = create_dataset("Cassie", "Cassie")
    folder = Path(card["folder"])
    raw = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    raw.pop("subfolder", None)
    raw["last_extract"] = {
        "output": r"E:\ComfyUI\models\refmods\sdprompts_minimaxh3_cassie_v1_refmod.safetensors",
        "content_hash": "abc",
        "file_count": 1,
        "token_count": 1024,
    }
    (folder / "dataset.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    loaded = load_manifest(folder, "cassie")
    assert loaded["subfolder"] == ""

    raw["last_extract"]["output"] = r"E:\ComfyUI\models\refmods\identity\sdprompts_minimaxh3_cassie_v1_refmod.safetensors"
    (folder / "dataset.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    loaded = load_manifest(folder, "cassie")
    assert loaded["subfolder"] == "identity"


def test_slugify_and_mod_name():
    assert slugify("Felicia Day") == "felicia-day"
    assert default_mod_name("felicia-day", prefix="sdprompts") == "sdprompts_minimaxh3_feliciaday_v1_refmod"
    assert bump_mod_name("sdprompts_minimaxh3_feliciaday_v1_refmod") == "sdprompts_minimaxh3_feliciaday_v2_refmod"
    assert bump_mod_name("character") == "character_v2_refmod"


def test_dataset_crud_and_dirty(dataset_root):
    card = create_dataset("Felicia Day", "Felicia Day")
    assert card["slug"] == "felicia-day"
    folder = Path(card["folder"])
    (folder / "01.png").write_bytes(b"\x89PNG\r\n" + b"aaaa")
    (folder / "02.png").write_bytes(b"\x89PNG\r\n" + b"bbbb")
    card = patch_dataset("felicia-day", {
        "items": [
            {"file": "01.png", "shot": "front", "enabled": True},
            {"file": "02.png", "shot": "profile", "enabled": True},
        ],
        "description": "primary persona",
    })
    assert card["image_count"] == 2
    assert card["dirty"] is True

    manifest = load_manifest(folder, "felicia-day")
    digest = content_hash(manifest, folder)
    record_extract("felicia-day", str(folder / "mod.safetensors"), 1024, 2)
    card = patch_dataset("felicia-day", {})
    assert card["dirty"] is False
    assert card["last_extract"]["content_hash"] == digest

    set_caption("felicia-day", "01.png", "a woman looking at camera")
    card = patch_dataset("felicia-day", {})
    assert card["dirty"] is False
    assert (folder / "01.txt").read_text(encoding="utf-8").startswith("a woman")

    (folder / "03.png").write_bytes(b"\x89PNG\r\n" + b"cccc")
    card = patch_dataset("felicia-day", {})
    assert card["dirty"] is True
    assert card["image_count"] == 3

    card = patch_dataset("felicia-day", {
        "items": [
            {"file": "01.png", "enabled": True, "shot": "front"},
            {"file": "02.png", "enabled": False, "shot": "profile"},
            {"file": "03.png", "enabled": True, "shot": "three-quarter"},
        ]
    })
    files = [p.name for p in enabled_files(load_manifest(folder, "felicia-day"), folder)]
    assert files == ["01.png", "03.png"]
    assert is_dirty(load_manifest(folder, "felicia-day"), folder)


def test_dataset_cover_selection_and_fallback(dataset_root):
    card = create_dataset("Cover Set", "Cover Set")
    folder = Path(card["folder"])
    (folder / "01.png").write_bytes(b"\x89PNG\r\n" + b"aaaa")
    (folder / "02.png").write_bytes(b"\x89PNG\r\n" + b"bbbb")
    (folder / "03.png").write_bytes(b"\x89PNG\r\n" + b"cccc")
    card = patch_dataset("cover-set", {
        "items": [
            {"file": "01.png", "enabled": True},
            {"file": "02.png", "enabled": True},
            {"file": "03.png", "enabled": True},
        ],
    })
    assert card["cover"] == "01.png"

    record_extract("cover-set", str(folder / "mod.safetensors"), 1024, 3)
    clean = patch_dataset("cover-set", {})
    assert clean["dirty"] is False

    chosen = patch_dataset("cover-set", {"cover": "03.png"})
    assert chosen["cover"] == "03.png"
    assert chosen["dirty"] is False
    stored = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    assert stored["cover"] == "03.png"

    reordered = patch_dataset("cover-set", {
        "items": [
            {"file": "02.png", "enabled": True},
            {"file": "03.png", "enabled": True},
            {"file": "01.png", "enabled": True},
        ],
    })
    assert reordered["cover"] == "03.png"

    disabled_first = patch_dataset("cover-set", {
        "cover": "",
        "items": [
            {"file": "01.png", "enabled": False},
            {"file": "02.png", "enabled": True},
            {"file": "03.png", "enabled": True},
        ],
    })
    assert disabled_first["cover"] == "02.png"

    pinned = patch_dataset("cover-set", {"cover": "01.png"})
    assert pinned["cover"] == "01.png"

    with pytest.raises(DatasetError, match="cover must be an image"):
        patch_dataset("cover-set", {"cover": "missing.png"})

    after_delete = delete_file("cover-set", "01.png")
    assert after_delete["cover"] == "02.png"
    leftover = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    assert leftover.get("cover") in ("", None)


def test_dataset_notes_do_not_mark_dirty(dataset_root):
    card = create_dataset("Notes Set", "Notes Set")
    folder = Path(card["folder"])
    (folder / "01.png").write_bytes(b"\x89PNG\r\n" + b"aaaa")
    card = patch_dataset("notes-set", {"items": [{"file": "01.png", "enabled": True}]})
    record_extract("notes-set", str(folder / "mod.safetensors"), 1024, 1)
    clean = patch_dataset("notes-set", {})
    assert clean["dirty"] is False
    noted = patch_dataset("notes-set", {"notes": "try more profiles\nsource: shoot 12"})
    assert noted["notes"] == "try more profiles\nsource: shoot 12"
    assert noted["dirty"] is False
    stored = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    assert stored["notes"].startswith("try more profiles")


def test_create_from_one_sheet(dataset_root):
    png = sheet_png(2, 2)
    cards = create_datasets_from_sheets(
        [("CassieSheet.png", png)],
        name="Cassie",
        display_name="Cassie",
    )
    assert len(cards) == 1
    card = cards[0]
    assert card["slug"] == "cassie"
    assert card["display_name"] == "Cassie"
    assert card["description"] == "Cassie"
    assert card["image_count"] == 4
    assert [item["file"] for item in card["items"]] == [
        "CassieSheet_01.png",
        "CassieSheet_02.png",
        "CassieSheet_03.png",
        "CassieSheet_04.png",
    ]
    folder = Path(card["folder"])
    assert not (folder / "CassieSheet.png").exists()
    assert (folder / "CassieSheet_01.png").is_file()


def test_create_one_dataset_per_sheet(dataset_root):
    a = sheet_png(2, 2)
    b = sheet_png(2, 1)
    cards = create_datasets_from_sheets(
        [("alpha.png", a), ("bravo.png", b)],
        name="hero",
        display_name="Hero",
    )
    assert [c["slug"] for c in cards] == ["hero", "bravo"]
    assert cards[0]["display_name"] == "Hero"
    assert cards[0]["description"] == "Hero"
    assert cards[1]["display_name"] == "bravo"
    assert cards[1]["description"] == "bravo"
    assert cards[0]["image_count"] == 4
    assert cards[1]["image_count"] == 2


def test_split_sheet_collision(dataset_root):
    create_dataset("taken")
    with pytest.raises(DatasetError, match="already exists"):
        create_datasets_from_sheets([("taken.png", sheet_png(2, 2))])


def test_create_from_chosen_boxes(dataset_root):
    from backend.split import list_method_results

    png = sheet_png(2, 2)
    methods = list_method_results(contact_sheet(2, 2))
    grid = next(m for m in methods if m["count"] == 4)
    cards = create_datasets_from_sheets(
        [("picked.png", png)],
        name="picked",
        plans=[{"filename": "picked.png", "boxes": grid["boxes"][:3]}],
    )
    assert cards[0]["image_count"] == 3


def test_split_rejects_unsplitable(dataset_root):
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (180, 180), (255, 255, 255)).save(buf, format="PNG")
    with pytest.raises(DatasetError, match="could not split"):
        create_datasets_from_sheets([("blank.png", buf.getvalue())])
