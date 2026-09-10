from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app import app
from backend.clean import (
    apply_boxes_to_array,
    apply_clean,
    box_from_norm,
    combined_crop,
    crop_is_safe,
    detect_boxes,
    load_rgb,
    scan_dataset,
    suggest_action,
    undo_clean,
)
from backend.datasets import ORIGINALS_DIR, create_dataset


def _rgb(w, h, color) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _yellow_badge(w=1280, h=1600) -> Image.Image:
    im = _rgb(w, h, (48, 46, 44))
    draw = ImageDraw.Draw(im)
    draw.rectangle([220, 160, 1060, 1320], fill=(176, 142, 118))
    draw.rectangle([w - 300, h - 78, w - 12, h - 12], fill=(228, 196, 24))
    draw.rectangle([w - 286, h - 64, w - 26, h - 26], fill=(20, 18, 10))
    return im


def _black_bar(w=1280, h=1600) -> Image.Image:
    im = _rgb(w, h, (90, 88, 82))
    draw = ImageDraw.Draw(im)
    draw.rectangle([180, 140, 1100, 1280], fill=(150, 120, 100))
    draw.rectangle([w - 520, h - 70, w - 4, h - 4], fill=(6, 6, 6))
    x = w - 500
    for _ in range(12):
        draw.rectangle([x, h - 52, x + 18, h - 22], fill=(240, 240, 240))
        x += 28
    return im


def _clean_portrait(w=1024, h=1280) -> Image.Image:
    im = _rgb(w, h, (196, 196, 200))
    draw = ImageDraw.Draw(im)
    draw.ellipse([280, 220, 740, 820], fill=(208, 168, 148))
    draw.rectangle([430, 820, 590, 1180], fill=(80, 60, 50))
    return im


def _to_arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))


def _save(folder: Path, name: str, im: Image.Image) -> Path:
    path = folder / name
    im.save(path, format="JPEG", quality=95)
    return path


def test_detects_yellow_corner_badge():
    boxes = detect_boxes(_to_arr(_yellow_badge()))
    assert boxes, "expected a corner badge"
    box = boxes[0]
    assert box.edge == "bottom"
    assert box.x0 == 0
    assert box.y1 == 1600
    assert box.y0 <= 1522


def test_crop_strip_covers_the_badge():
    arr = _to_arr(_yellow_badge())
    boxes = detect_boxes(arr)
    assert boxes
    box = boxes[0]
    assert box.x0 <= 980 <= box.x1
    assert box.y0 <= 1522
    assert box.y1 >= 1588
    assert box.action == "crop"
    assert 1600 - box.y0 <= 110


def test_detects_dark_bar_with_glyphs():
    boxes = detect_boxes(_to_arr(_black_bar()))
    assert boxes, "expected a dark logo bar"
    box = boxes[0]
    assert box.edge == "bottom"
    assert box.y0 < 1540
    assert box.y1 == 1600


def test_clean_portrait_is_not_flagged():
    boxes = detect_boxes(_to_arr(_clean_portrait()))
    assert boxes == []


def test_small_image_still_crops():
    im = _yellow_badge(800, 1200)
    arr = _to_arr(im)
    boxes = detect_boxes(arr)
    assert boxes
    action, _ = suggest_action(arr, boxes[0].as_tuple())
    assert action == "crop"
    crop = combined_crop(800, 1200, boxes)
    assert crop is not None
    assert crop_is_safe(800, 1200, crop)


def test_large_edge_badge_can_crop():
    im = _yellow_badge(1280, 1600)
    arr = _to_arr(im)
    boxes = detect_boxes(arr)
    assert boxes
    crop = combined_crop(1280, 1600, boxes)
    assert crop is not None
    assert crop_is_safe(1280, 1600, crop)
    action, _ = suggest_action(arr, boxes[0].as_tuple())
    assert action == "crop"
    left, top, right, bottom = crop
    assert bottom < 1600
    assert arr[1580, 1100, 0] > 150
    cropped = arr[top:bottom, left:right]
    assert cropped.shape[0] == bottom - top


def test_apply_and_undo(dataset_root):
    card = create_dataset("marks")
    folder = Path(card["folder"])
    _save(folder, "shot.jpg", _yellow_badge())
    scan = scan_dataset("marks")
    assert scan["flagged"] >= 1
    item = next(it for it in scan["items"] if it["file"] == "shot.jpg")
    assert item["boxes"]
    assert item["boxes"][0]["edge"] == "bottom"
    before = (folder / "shot.jpg").read_bytes()
    applied = apply_clean("marks", items=[{"file": "shot.jpg", "boxes": item["boxes"]}])
    assert applied["results"][0]["ok"] is True
    assert applied["results"][0]["actions"] == ["crop"]
    assert (folder / ORIGINALS_DIR / "shot.jpg").is_file()
    after = (folder / "shot.jpg").read_bytes()
    assert after != before
    undone = undo_clean("marks", ["shot.jpg"])
    assert undone["restored"] == 1
    assert (folder / "shot.jpg").read_bytes() == before


def test_apply_boxes_to_all(dataset_root):
    card = create_dataset("batch")
    folder = Path(card["folder"])
    _save(folder, "a.jpg", _yellow_badge())
    _save(folder, "b.jpg", _yellow_badge())
    box = {"x0": 0.0, "y0": 0.95, "x1": 1.0, "y1": 1.0, "action": "crop", "edge": "bottom"}
    out = apply_clean("batch", apply_boxes=[box])
    assert len(out["results"]) == 2
    assert all(r["ok"] for r in out["results"])
    assert (folder / ORIGINALS_DIR / "a.jpg").is_file()
    assert (folder / ORIGINALS_DIR / "b.jpg").is_file()


def test_clean_api(dataset_root):
    create_dataset("api-clean")
    folder = Path(dataset_root / "api-clean")
    _save(folder, "shot.jpg", _black_bar())
    client = TestClient(app)
    scan = client.post("/api/datasets/api-clean/clean/scan", json={})
    assert scan.status_code == 200, scan.text
    body = scan.json()
    assert body["total"] == 1
    item = body["items"][0]
    if not item["boxes"]:
        item["boxes"] = [{"x0": 0.0, "y0": 0.95, "x1": 1.0, "y1": 1.0, "action": "crop", "edge": "bottom"}]
    applied = client.post(
        "/api/datasets/api-clean/clean/apply",
        json={"items": [{"file": "shot.jpg", "boxes": item["boxes"]}]},
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["dataset"]["items"][0]["has_original"] is True
    undone = client.post("/api/datasets/api-clean/clean/undo", json={})
    assert undone.status_code == 200, undone.text
    assert undone.json()["restored"] >= 1


def test_real_samples_flag_most_watermarks():
    samples = Path(__file__).resolve().parents[1] / "temp"
    files = [p for p in samples.glob("*.jpg") if not p.name.startswith("_")]
    if len(files) < 5:
        pytest.skip("watermark samples not in temp/")
    flagged = sum(1 for p in files if detect_boxes(load_rgb(p)))
    assert flagged >= 4


def test_real_samples_strips_stay_thin():
    samples = Path(__file__).resolve().parents[1] / "temp"
    files = [p for p in samples.glob("*.jpg") if not p.name.startswith("_")]
    if len(files) < 5:
        pytest.skip("watermark samples not in temp/")
    for path in files:
        arr = load_rgb(path)
        h, w = arr.shape[:2]
        for box in detect_boxes(arr):
            if box.edge in ("top", "bottom"):
                depth = (box.y1 - box.y0) / h
            else:
                depth = (box.x1 - box.x0) / w
            assert depth <= 0.08, f"{path.name} {box.edge} strip is {depth:.1%} of the still"


def test_box_from_norm_roundtrip():
    box = box_from_norm({"x0": 0.1, "y0": 0.2, "x1": 0.4, "y1": 0.35, "action": "crop"}, 1000, 2000)
    assert box.action == "crop"
    assert box.edge in ("top", "bottom", "left", "right")


def test_apply_keep_rect(dataset_root):
    card = create_dataset("keep-set")
    folder = Path(card["folder"])
    _save(folder, "shot.jpg", _yellow_badge(1280, 1600))
    before = Image.open(folder / "shot.jpg")
    assert before.size == (1280, 1600)
    before.close()
    out = apply_clean(
        "keep-set",
        items=[{"file": "shot.jpg", "keep": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 0.9}}],
    )
    assert out["results"][0]["ok"] is True
    assert out["results"][0]["actions"] == ["crop"]
    after = Image.open(folder / "shot.jpg")
    assert after.size[0] == 1280
    assert after.size[1] < 1600
    after.close()


def test_apply_keep_to_all(dataset_root):
    create_dataset("keep-all")
    folder = Path(dataset_root / "keep-all")
    _save(folder, "a.jpg", _yellow_badge())
    _save(folder, "b.jpg", _yellow_badge())
    out = apply_clean("keep-all", apply_keep={"x0": 0, "y0": 0.05, "x1": 1, "y1": 0.95})
    assert len(out["results"]) == 2
    assert all(r["ok"] for r in out["results"])


def test_scan_includes_keep(dataset_root):
    create_dataset("keep-scan")
    folder = Path(dataset_root / "keep-scan")
    _save(folder, "shot.jpg", _yellow_badge())
    scan = scan_dataset("keep-scan")
    item = scan["items"][0]
    assert "keep" in item
    assert item["keep"]["x1"] > item["keep"]["x0"]
    if item["boxes"]:
        assert item["keep"]["y1"] < 1.0 or item["keep"]["y0"] > 0.0


def test_apply_boxes_crops_bottom_strip():
    arr = np.full((1600, 1280, 3), 80, dtype=np.uint8)
    arr[1520:1590, 980:1260] = (220, 190, 20)
    boxes = [
        box_from_norm({"x0": 0.0, "y0": 0.95, "x1": 1.0, "y1": 1.0, "action": "crop", "edge": "bottom"}, 1280, 1600),
    ]
    out, used = apply_boxes_to_array(arr, boxes)
    assert used == ["crop"]
    assert out.shape[0] < 1600
    assert out.shape[1] == 1280
