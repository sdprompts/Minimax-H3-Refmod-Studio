from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app import app
from backend.datasets import create_dataset, load_manifest, patch_dataset, record_extract
from backend.quality import (
    FaceHit,
    advise,
    classify_pose,
    classify_scale,
    clamp,
    crowding,
    pick_diverse,
    rank_dir,
    s_face_from_ied,
    s_pose_from_angles,
    s_sharp_from_var,
    score_bgr,
    score_image,
    scan_dataset,
)


def _full_face(bgr):
    h, w = bgr.shape[:2]
    return [FaceHit(8, 8, w - 8, h - 8, ied_px=180.0, yaw_deg=0.0, pitch_deg=0.0)]


def _portrait(w=640, h=800, *, blur=False, mean=128, noise=40, seed=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    arr = np.full((h, w, 3), int(mean), dtype=np.uint8)
    jitter = rng.randint(0, max(1, noise), (h, w, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + jitter - noise // 2, 0, 255).astype(np.uint8)
    if blur:
        arr = cv2.GaussianBlur(arr, (41, 41), 12)
    return arr


def test_formula_helpers():
    assert clamp(1.4) == 1.0
    assert clamp(-0.2) == 0.0
    assert s_face_from_ied(180, 400) == 1.0
    assert s_face_from_ied(40, 400) == 0.0
    assert s_face_from_ied(None, 80) == 0.0
    assert s_pose_from_angles(None, None) == 1.0
    assert s_pose_from_angles(90, 0) == 0.35
    assert s_sharp_from_var(40) == 0.0
    assert s_sharp_from_var(350) == 1.0


def test_no_face_and_unreadable(tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (400, 500), (180, 180, 180)).save(blank)
    row = score_image(blank, detector=lambda _bgr: [])
    assert row["score"] == 0.0
    assert row["reason"] == "no_face"

    junk = tmp_path / "bad.png"
    junk.write_bytes(b"not an image")
    bad = score_image(junk)
    assert bad["score"] == 0.0
    assert bad["reason"] == "unreadable"


def test_sharp_vs_blur_and_dark():
    sharp = score_bgr(_portrait(blur=False, noise=80), detector=_full_face)
    blur = score_bgr(_portrait(blur=True, noise=80), detector=_full_face)
    dark = score_bgr(_portrait(mean=18, noise=8, blur=False), detector=_full_face)
    assert sharp["score"] > blur["score"]
    assert sharp["debug"]["s_sharp"] > blur["debug"]["s_sharp"]
    assert dark["score"] < sharp["score"]
    assert dark["reason"] in {"dark", "blown", "blurry", "face_too_small"}
    assert dark["debug"]["s_light"] < 0.5


def test_tiny_image_caps_s_face():
    tiny = score_bgr(_portrait(200, 220, noise=30), detector=_full_face)
    assert tiny["debug"]["s_face"] <= 0.4
    assert tiny["score"] <= 40.1


def test_profile_is_mid_not_zero():
    face = lambda bgr: [FaceHit(8, 8, bgr.shape[1] - 8, bgr.shape[0] - 8, ied_px=180, yaw_deg=70, pitch_deg=0)]
    row = score_bgr(_portrait(noise=50), detector=face)
    assert row["debug"]["s_pose"] == 0.35
    assert 0 < row["score"] < 70


def test_rank_dir_orders_high_to_low(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.quality.detect_faces", _full_face)
    sharp = tmp_path / "a.png"
    blurry = tmp_path / "b.png"
    Image.fromarray(_portrait(noise=80)[:, :, ::-1]).save(sharp)
    Image.fromarray(_portrait(blur=True, noise=80)[:, :, ::-1]).save(blurry)
    ranked = rank_dir(tmp_path)
    assert Path(ranked[0]["path"]).name == "a.png"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_scan_skips_video_and_does_not_dirty(dataset_root):
    card = create_dataset("Q Set", "Q Set")
    folder = Path(card["folder"])
    Image.fromarray(_portrait(noise=40)[:, :, ::-1]).save(folder / "face.png")
    (folder / "clip.mp4").write_bytes(b"fake-video")
    patch_dataset("q-set", {"items": [{"file": "face.png"}, {"file": "clip.mp4"}]})
    record_extract("q-set", str(folder / "mod.safetensors"), 1024, 1)
    before = patch_dataset("q-set", {})
    assert before["dirty"] is False
    out = scan_dataset("q-set")
    assert out["dataset"]["dirty"] is False
    by_file = {Path(r["path"]).name: r for r in out["results"]}
    assert by_file["clip.mp4"]["reason"] == "not_image"
    assert by_file["clip.mp4"]["score"] is None
    stored = load_manifest(folder, "q-set")
    qualities = {i["file"]: i.get("quality") for i in stored["items"]}
    assert qualities["clip.mp4"]["reason"] == "not_image"
    assert "score" in qualities["face.png"]


def test_quality_api(dataset_root):
    create_dataset("api-q")
    folder = dataset_root / "api-q"
    Image.new("RGB", (320, 400), (200, 200, 200)).save(folder / "x.png")
    client = TestClient(app)
    res = client.post("/api/datasets/api-q/quality", json={})
    assert res.status_code == 200, res.text
    body = res.json()
    assert "results" in body
    assert body["dataset"]["items"][0]["quality"]["reason"] in {
        "ok", "no_face", "blurry", "dark", "blown", "face_too_small", "split_light", "profile",
    }


def _row(name, score, scale, pose_bin, yaw=0.0):
    return {
        "path": name,
        "score": score,
        "reason": "ok",
        "debug": {"scale": scale, "pose_bin": pose_bin, "yaw_deg": yaw, "pitch_deg": 0.0},
    }


def test_classify_scale_and_pose():
    assert classify_scale(0.55, 0.2) == "closeup"
    assert classify_scale(0.25, 0.08) == "medium"
    assert classify_scale(0.08, 0.03) == "full"
    assert classify_pose(0) == "front"
    assert classify_pose(-30) == "three-quarter-left"
    assert classify_pose(55) == "profile-right"
    assert classify_pose(None) == "unknown"


def test_pick_diverse_spreads_closeup_angles():
    rows = [
        _row("cu_front_a.jpg", 40, "closeup", "front", 2),
        _row("cu_front_b.jpg", 38, "closeup", "front", 3),
        _row("cu_front_c.jpg", 36, "closeup", "front", 1),
        _row("cu_front_d.jpg", 34, "closeup", "front", 4),
        _row("cu_l.jpg", 22, "closeup", "three-quarter-left", -28),
        _row("cu_r.jpg", 21, "closeup", "three-quarter-right", 30),
        _row("cu_pl.jpg", 18, "closeup", "profile-left", -60),
        _row("md_a.jpg", 16, "medium", "front", 5),
        _row("md_b.jpg", 15, "medium", "three-quarter-right", 25),
        _row("full_a.jpg", 12, "full", "front", 0),
        _row("full_b.jpg", 11, "full", "front", 2),
    ]
    picks = pick_diverse(rows, 10)
    assert "cu_front_a.jpg" in picks
    assert "cu_l.jpg" in picks
    assert "cu_r.jpg" in picks
    assert "md_a.jpg" in picks
    assert "full_a.jpg" in picks
    assert picks.count("cu_front_b.jpg") + picks.count("cu_front_c.jpg") + picks.count("cu_front_d.jpg") <= 3
    crowded = crowding(rows)
    assert crowded
    assert crowded[0]["pose_bin"] == "front"
    assert crowded[0]["scale"] == "closeup"


def test_advise_asks_for_missing_angles():
    rows = [
        _row("a.jpg", 30, "closeup", "front", 2),
        _row("b.jpg", 28, "closeup", "front", 3),
        _row("c.jpg", 26, "closeup", "front", 1),
        _row("d.jpg", 24, "closeup", "front", 4),
    ]
    tips = advise(rows)
    ids = {t["id"] for t in tips}
    assert "cu_three-quarter-left" in ids
    assert "cu_three-quarter-right" in ids
    assert "medium" in ids
    assert "full" in ids
    assert any(t["id"].startswith("crowd_") for t in tips)
    texts = " ".join(t["text"] for t in tips)
    assert "three-quarter close-up from the left" in texts
