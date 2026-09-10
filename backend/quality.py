"""Face-usability ranking for identity stills. Rank-only product score, no BRISQUE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .datasets import load_manifest, public_dataset, resolve_dataset, save_manifest, utc_now
from .media import is_image

THRESH: dict[str, float] = {
    "sharp_lo": 40.0,
    "sharp_hi": 350.0,
    "ied_lo": 40.0,
    "ied_span": 120.0,
    "box_lo": 80.0,
    "box_span": 240.0,
    "min_short": 256.0,
    "s_face_cap_small": 0.4,
    "light_mean": 128.0,
    "light_sigma": 55.0,
    "clip_lo": 15.0,
    "clip_hi": 240.0,
    "clip_k": 8.0,
    "split_k": 2.5,
    "yaw_deg": 75.0,
    "pitch_deg": 40.0,
    "pose_floor": 0.35,
    "pad": 0.20,
    "lap_short": 256.0,
    "eye_r": 0.28,
    "eye_rel": 0.12,
    "scale_cu_frac": 0.38,
    "scale_md_frac": 0.16,
    "scale_cu_ied": 0.14,
    "scale_md_ied": 0.06,
    "yaw_front": 18.0,
    "yaw_profile": 48.0,
    "crowd_count": 4.0,
    "crowd_frac": 0.30,
    "pick_n": 10.0,
    "pick_cu": 7.0,
    "pick_md": 2.0,
    "pick_full": 1.0,
}

REASONS = (
    "ok",
    "no_face",
    "unreadable",
    "not_image",
    "face_too_small",
    "blurry",
    "dark",
    "blown",
    "split_light",
    "profile",
)

_HAAR = None
_YUNET = None
_YUNET_SIZE = None
_MP_LANDMARKER = None
_MP_TRIED = False


class QualityError(ValueError):
    pass


@dataclass
class FaceHit:
    x0: int
    y0: int
    x1: int
    y1: int
    ied_px: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    left_eye: tuple[float, float] | None = None
    right_eye: tuple[float, float] | None = None
    n_faces: int = 1

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    @property
    def short(self) -> int:
        return min(self.x1 - self.x0, self.y1 - self.y0)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _empty_debug() -> dict[str, float]:
    return {
        "ied_px": 0.0,
        "laplacian_var": 0.0,
        "face_mean": 0.0,
        "clip_frac": 0.0,
        "split": 0.0,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "s_sharp": 0.0,
        "s_face": 0.0,
        "s_light": 0.0,
        "s_pose": 0.0,
        "n_faces": 0.0,
        "face_frac": 0.0,
        "scale": "",
        "pose_bin": "",
    }


def _result(path: Path | str, score: float, reason: str, debug: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "score": round(float(score), 1),
        "reason": reason,
        "debug": debug or _empty_debug(),
    }


def load_bgr(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            rgb = np.asarray(img)
    except (OSError, ValueError):
        return None
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _haar_cascade():
    global _HAAR
    create = getattr(cv2, "CascadeClassifier", None)
    if create is None:
        return None
    if _HAAR is not None:
        return _HAAR if not _HAAR.empty() else None
    base = getattr(cv2, "data", None)
    folder = getattr(base, "haarcascades", "") if base is not None else ""
    xml = str(Path(folder) / "haarcascade_frontalface_default.xml") if folder else ""
    cascade = create(xml)
    _HAAR = cascade
    return cascade if cascade is not None and not cascade.empty() else None


def _detect_haar(gray: np.ndarray) -> list[FaceHit]:
    cascade = _haar_cascade()
    if cascade is None:
        return []
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    hits = [
        FaceHit(int(x), int(y), int(x + w), int(y + h))
        for (x, y, w, h) in faces
    ]
    hits.sort(key=lambda hit: hit.area, reverse=True)
    for hit in hits:
        hit.n_faces = len(hits)
    return hits


def _yunet_model_path() -> Path | None:
    here = Path(__file__).resolve().parent.parent
    for candidate in (
        here / "data" / "face_detection_yunet_2023mar.onnx",
        here / "data" / "face_detection_yunet.onnx",
    ):
        if candidate.is_file():
            return candidate
    return None


def _pose_from_eyes_nose(
    left: tuple[float, float],
    right: tuple[float, float],
    nose: tuple[float, float],
    ied: float,
) -> tuple[float, float]:
    mid = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
    if ied <= 1:
        return 0.0, 0.0
    yaw = clamp((nose[0] - mid[0]) / ied * 90.0, -90.0, 90.0)
    # Frontal faces sit with the nose ~0.68 IED below the eye line.
    pitch = clamp((nose[1] - mid[1]) / ied * 90.0 - 61.0, -90.0, 90.0)
    return yaw, pitch


def _detect_yunet(bgr: np.ndarray) -> list[FaceHit]:
    global _YUNET, _YUNET_SIZE
    create = getattr(cv2, "FaceDetectorYN_create", None)
    model = _yunet_model_path()
    if create is None or model is None:
        return []
    h, w = bgr.shape[:2]
    size = (w, h)
    if _YUNET is None:
        _YUNET = create(str(model), "", (320, 320), 0.55, 0.3, 5000)
        _YUNET_SIZE = (320, 320)
    if _YUNET_SIZE != size:
        _YUNET.setInputSize(size)
        _YUNET_SIZE = size
    try:
        _ok, faces = _YUNET.detect(bgr)
    except cv2.error:
        return []
    if faces is None or len(faces) == 0:
        return []
    hits: list[FaceHit] = []
    for row in faces:
        x, y, fw, fh = [float(v) for v in row[:4]]
        re_x, re_y, le_x, le_y, nt_x, nt_y = [float(v) for v in row[4:10]]
        x0, y0 = int(max(0, x)), int(max(0, y))
        x1, y1 = int(min(w, x + fw)), int(min(h, y + fh))
        left = (le_x, le_y)
        right = (re_x, re_y)
        ied = float(math.hypot(right[0] - left[0], right[1] - left[1]))
        yaw, pitch = _pose_from_eyes_nose(left, right, (nt_x, nt_y), ied)
        hits.append(FaceHit(
            x0, y0, x1, y1,
            ied_px=ied if ied > 1 else None,
            yaw_deg=yaw,
            pitch_deg=pitch,
            left_eye=left,
            right_eye=right,
        ))
    hits.sort(key=lambda hit: hit.area, reverse=True)
    for hit in hits:
        hit.n_faces = len(hits)
    return hits


def _mediapipe_model_path() -> Path | None:
    here = Path(__file__).resolve().parent.parent
    for candidate in (
        here / "data" / "face_landmarker.task",
        here / "backend" / "data" / "face_landmarker.task",
    ):
        if candidate.is_file():
            return candidate
    return None


def _mediapipe_landmarker():
    global _MP_LANDMARKER, _MP_TRIED
    if _MP_TRIED:
        return _MP_LANDMARKER
    _MP_TRIED = True
    model = _mediapipe_model_path()
    if model is None:
        return None
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError:
        return None
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=4,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    try:
        _MP_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    except Exception:  # noqa: BLE001 — optional detector
        _MP_LANDMARKER = None
    return _MP_LANDMARKER


def _yaw_pitch_from_matrix(matrix) -> tuple[float | None, float | None]:
    try:
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape == (4, 4):
            rot = mat[:3, :3]
        elif mat.shape == (3, 3):
            rot = mat
        else:
            return None, None
        sy = math.sqrt(float(rot[0, 0]) ** 2 + float(rot[1, 0]) ** 2)
        yaw = math.degrees(math.atan2(float(rot[2, 1]), float(rot[2, 2])))
        pitch = math.degrees(math.atan2(-float(rot[2, 0]), sy if sy > 1e-6 else 1e-6))
        return float(yaw), float(pitch)
    except (TypeError, ValueError, IndexError):
        return None, None


def _detect_mediapipe(bgr: np.ndarray) -> list[FaceHit] | None:
    landmarker = _mediapipe_landmarker()
    if landmarker is None:
        return None
    try:
        import mediapipe as mp
    except ImportError:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    try:
        result = landmarker.detect(mp_image)
    except Exception:  # noqa: BLE001
        return None
    faces = list(result.face_landmarks or [])
    if not faces:
        return []
    h, w = bgr.shape[:2]
    matrices = list(result.facial_transformation_matrixes or [])
    hits: list[FaceHit] = []
    for i, lm in enumerate(faces):
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        x0, x1 = int(max(0, min(xs))), int(min(w, max(xs)))
        y0, y1 = int(max(0, min(ys))), int(min(h, max(ys)))
        # Face Mesh: left eye 33/133, right eye 263/362, nose 1.
        left = ((lm[33].x + lm[133].x) * 0.5 * w, (lm[33].y + lm[133].y) * 0.5 * h)
        right = ((lm[263].x + lm[362].x) * 0.5 * w, (lm[263].y + lm[362].y) * 0.5 * h)
        ied = float(math.hypot(right[0] - left[0], right[1] - left[1]))
        yaw, pitch = (None, None)
        if i < len(matrices):
            yaw, pitch = _yaw_pitch_from_matrix(matrices[i])
        if yaw is None:
            nose = (lm[1].x * w, lm[1].y * h)
            mid = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
            if ied > 1:
                yaw = clamp((nose[0] - mid[0]) / ied * 90.0, -90.0, 90.0)
                pitch = clamp((nose[1] - mid[1]) / ied * 70.0 - 20.0, -90.0, 90.0)
        hits.append(FaceHit(
            x0, y0, x1, y1,
            ied_px=ied if ied > 1 else None,
            yaw_deg=yaw,
            pitch_deg=pitch,
            left_eye=left,
            right_eye=right,
        ))
    hits.sort(key=lambda hit: hit.area, reverse=True)
    for hit in hits:
        hit.n_faces = len(hits)
    return hits


def detect_faces(bgr: np.ndarray) -> list[FaceHit]:
    mp_hits = _detect_mediapipe(bgr)
    if mp_hits is not None:
        return mp_hits
    yunet = _detect_yunet(bgr)
    if yunet:
        return yunet
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return _detect_haar(gray)


def _padded_crop(bgr: np.ndarray, hit: FaceHit, pad: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    bw = hit.x1 - hit.x0
    bh = hit.y1 - hit.y0
    px = int(round(bw * pad))
    py = int(round(bh * pad))
    x0 = max(0, hit.x0 - px)
    y0 = max(0, hit.y0 - py)
    x1 = min(w, hit.x1 + px)
    y1 = min(h, hit.y1 + py)
    if x1 <= x0 or y1 <= y0:
        return bgr
    return bgr[y0:y1, x0:x1]


def _resize_short(gray: np.ndarray, short: float) -> np.ndarray:
    h, w = gray.shape[:2]
    side = min(h, w)
    if side <= 0 or side <= short:
        return gray
    scale = short / side
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)


def laplacian_var(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def s_sharp_from_var(v: float) -> float:
    lo = math.log1p(THRESH["sharp_lo"])
    hi = math.log1p(THRESH["sharp_hi"])
    if hi <= lo:
        return 1.0
    return clamp((math.log1p(max(0.0, v)) - lo) / (hi - lo))


def s_face_from_ied(ied_px: float | None, box_short: float) -> float:
    if ied_px is not None and ied_px > 0:
        return clamp((ied_px - THRESH["ied_lo"]) / THRESH["ied_span"])
    return clamp((box_short - THRESH["box_lo"]) / THRESH["box_span"])


def light_terms(gray: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Return m, c, d, s_exp, s_clip, s_split."""
    if gray.size == 0:
        return 0.0, 1.0, 1.0, 0.0, 0.0, 0.0
    g = gray.astype(np.float64)
    m = float(g.mean())
    c = float(np.mean((g < THRESH["clip_lo"]) | (g > THRESH["clip_hi"])))
    mid = max(1, g.shape[1] // 2)
    left = float(g[:, :mid].mean()) if g.shape[1] > 1 else m
    right = float(g[:, mid:].mean()) if g.shape[1] > 1 else m
    d = abs(left - right) / 255.0
    s_exp = math.exp(-0.5 * ((m - THRESH["light_mean"]) / THRESH["light_sigma"]) ** 2)
    s_clip = clamp(1.0 - THRESH["clip_k"] * c)
    s_split = clamp(1.0 - THRESH["split_k"] * d)
    return m, c, d, s_exp, s_clip, s_split


def s_pose_from_angles(yaw: float | None, pitch: float | None) -> float:
    if yaw is None and pitch is None:
        return 1.0
    y = 0.0 if yaw is None else abs(float(yaw))
    p = 0.0 if pitch is None else abs(float(pitch))
    pose = clamp(1.0 - y / THRESH["yaw_deg"]) * clamp(1.0 - p / THRESH["pitch_deg"])
    return max(pose, THRESH["pose_floor"])


def _eye_patches(gray_full: np.ndarray, hit: FaceHit) -> list[np.ndarray]:
    if hit.left_eye is None or hit.right_eye is None or not hit.ied_px:
        return []
    r = max(6.0, hit.ied_px * THRESH["eye_r"])
    H, W = gray_full.shape[:2]
    patches = []
    for cx, cy in (hit.left_eye, hit.right_eye):
        x0 = int(round(cx - r))
        y0 = int(round(cy - r))
        x1 = int(round(cx + r))
        y1 = int(round(cy + r))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            patches.append(gray_full[y0:y1, x0:x1])
    return patches


def _pick_reason(
    score: float,
    s_sharp: float,
    s_face: float,
    s_light: float,
    s_pose: float,
    mean: float,
    clip_frac: float,
    split: float,
) -> str:
    if score >= 20 and min(s_sharp, s_face, s_light, s_pose) >= 0.35:
        return "ok"
    terms = {"s_sharp": s_sharp, "s_face": s_face, "s_light": s_light, "s_pose": s_pose}
    worst = min(terms, key=terms.get)
    if worst == "s_face":
        return "face_too_small"
    if worst == "s_sharp":
        return "blurry"
    if worst == "s_pose":
        return "profile"
    if mean < 90:
        return "dark"
    if mean > 170 or (clip_frac > 0.08 and mean >= 128):
        return "blown"
    if split > 0.12:
        return "split_light"
    return "dark" if mean < 128 else "blown"


def classify_scale(face_frac: float, ied_ratio: float) -> str:
    if face_frac >= THRESH["scale_cu_frac"] or ied_ratio >= THRESH["scale_cu_ied"]:
        return "closeup"
    if face_frac >= THRESH["scale_md_frac"] or ied_ratio >= THRESH["scale_md_ied"]:
        return "medium"
    return "full"


def classify_pose(yaw_deg: float | None) -> str:
    if yaw_deg is None:
        return "unknown"
    yaw = float(yaw_deg)
    mag = abs(yaw)
    if mag < THRESH["yaw_front"]:
        return "front"
    side = "right" if yaw >= 0 else "left"
    if mag >= THRESH["yaw_profile"]:
        return f"profile-{side}"
    return f"three-quarter-{side}"


def _row_debug(row: dict[str, Any]) -> dict[str, Any]:
    dbg = row.get("debug")
    return dbg if isinstance(dbg, dict) else {}


def crowding(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    faced = [r for r in rows if isinstance(r.get("score"), (int, float)) and float(r["score"]) > 0]
    n = len(faced)
    if n < 3:
        return []
    counts: dict[tuple[str, str], int] = {}
    for row in faced:
        dbg = _row_debug(row)
        key = (str(dbg.get("scale") or "closeup"), str(dbg.get("pose_bin") or "unknown"))
        counts[key] = counts.get(key, 0) + 1
    out = []
    for (scale, pose_bin), count in sorted(counts.items(), key=lambda kv: -kv[1]):
        frac = count / n
        if count >= THRESH["crowd_count"] and frac >= THRESH["crowd_frac"]:
            out.append({
                "scale": scale,
                "pose_bin": pose_bin,
                "count": count,
                "total": n,
                "frac": round(frac, 3),
            })
    return out


def _diversity_key(row: dict[str, Any], picked: list[dict[str, Any]]) -> float:
    dbg = _row_debug(row)
    yaw = float(dbg.get("yaw_deg") or 0.0)
    pitch = float(dbg.get("pitch_deg") or 0.0)
    scale = str(dbg.get("scale") or "")
    score = float(row.get("score") or 0.0)
    if not picked:
        return score
    nearest = min(
        math.hypot(
            yaw - float(_row_debug(p).get("yaw_deg") or 0.0),
            0.6 * (pitch - float(_row_debug(p).get("pitch_deg") or 0.0)),
        )
        for p in picked
    )
    same_scale = sum(1 for p in picked if str(_row_debug(p).get("scale") or "") == scale)
    return score + 10.0 * min(nearest / 30.0, 2.5) - 3.0 * same_scale


POSE_NEED = (
    ("front", "a straight-on close-up with eyes to camera"),
    ("three-quarter-left", "a three-quarter close-up from the left"),
    ("three-quarter-right", "a three-quarter close-up from the right"),
)
POSE_NICE = (
    ("profile-left", "a left-profile close-up"),
    ("profile-right", "a right-profile close-up"),
)


def _has_closeup(rows: list[dict[str, Any]], pose_bin: str) -> bool:
    for row in rows:
        if not isinstance(row.get("score"), (int, float)) or float(row["score"]) <= 0:
            continue
        dbg = _row_debug(row)
        if str(dbg.get("scale") or "") == "closeup" and str(dbg.get("pose_bin") or "") == pose_bin:
            return True
    return False


def _scale_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"closeup": 0, "medium": 0, "full": 0}
    for row in rows:
        if not isinstance(row.get("score"), (int, float)) or float(row["score"]) <= 0:
            continue
        scale = str(_row_debug(row).get("scale") or "")
        if scale in counts:
            counts[scale] += 1
    return counts


def advise(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """What to shoot or drop so the identity mix gets better."""
    images = [r for r in rows if r.get("reason") != "not_image"]
    faced = [r for r in images if isinstance(r.get("score"), (int, float)) and float(r["score"]) > 0]
    usable = [r for r in faced if float(r["score"]) >= 8]
    tips: list[dict[str, str]] = []

    def add(level: str, key: str, text: str) -> None:
        tips.append({"level": level, "id": key, "text": text})

    if not images:
        return [{"level": "need", "id": "empty", "text": "No stills to judge. Drop 8–20 photos of the same person."}]
    no_face = sum(1 for r in images if r.get("reason") == "no_face")
    if no_face and not faced:
        add("need", "no_face", "No faces found. Add tighter, well-lit close-ups of the person — the encoder cannot use empty or tiny faces.")
        return tips

    if len(usable) < 8:
        add(
            "need",
            "count",
            f"Identity sets work best at 8–20 stills. {len(usable)} usable (score 8+) now — add {8 - len(usable)} more from new angles, not copies of the same pose.",
        )

    for pose_bin, phrase in POSE_NEED:
        if not _has_closeup(faced, pose_bin):
            add("need", f"cu_{pose_bin}", f"Add {phrase}.")
    for pose_bin, phrase in POSE_NICE:
        if not _has_closeup(faced, pose_bin):
            add("improve", f"cu_{pose_bin}", f"A {phrase} would round out the angle mix.")

    scales = _scale_counts(faced)
    if scales["medium"] < 1:
        add("need", "medium", "Add 1–2 medium / waist-up shots so the set is not only heads.")
    elif scales["medium"] < 2:
        add("improve", "medium2", "One more medium shot (different angle from the first) would help body continuity.")
    if scales["full"] < 1:
        add("improve", "full", "Add one full-body or head-to-knee still. One is enough.")

    for crowd in crowding(rows):
        pose = crowd["pose_bin"].replace("-", " ")
        add(
            "need" if crowd["frac"] >= 0.4 else "improve",
            f"crowd_{crowd['scale']}_{crowd['pose_bin']}",
            f"Too many {crowd['scale']} {pose} stills ({crowd['count']} of {crowd['total']}). Keep the sharpest 1–2 and replace the rest with missing angles.",
        )

    n_face = max(1, len(faced))
    blurry = sum(1 for r in faced if r.get("reason") == "blurry" or float(_row_debug(r).get("s_sharp") or 0) < 0.3)
    if blurry >= 3 or blurry / n_face >= 0.4:
        add("need", "sharp", "Several stills are soft. Reshoot close-ups with the eyes in focus — missed focus on the face will not extract well.")
    dark = sum(1 for r in faced if r.get("reason") in {"dark", "blown", "split_light"})
    if dark >= 3 or dark / n_face >= 0.4:
        add("improve", "light", "Lighting on the face is often too dark, blown, or split. Even, front-ish light on a close-up will score higher than another dim copy.")
    tiny = sum(1 for r in faced if r.get("reason") == "face_too_small" or str(_row_debug(r).get("scale") or "") == "full")
    if scales["closeup"] < 5 and tiny:
        add("need", "tighter", "Most faces are small in frame. Add tight head-and-shoulders close-ups (eyes large in the photo).")

    if not tips:
        add("ok", "balanced", "Mix looks balanced. Use the recommended picks and skip the near-duplicate poses.")
    # Needs first, then improvements, cap so the sidebar stays readable.
    tips.sort(key=lambda t: {"need": 0, "improve": 1, "ok": 2}.get(t["level"], 9))
    return tips[:10]


def pick_diverse(rows: list[dict[str, Any]], n: int | None = None) -> list[str]:
    """Best stills with angle/scale spread. Default 7 close-up / 2 medium / 1 full-body."""
    n = int(n or THRESH["pick_n"])
    usable = [
        r for r in rows
        if isinstance(r.get("score"), (int, float)) and float(r["score"]) > 0
    ]
    usable.sort(key=lambda r: (-float(r["score"]), Path(r["path"]).name.lower()))
    if len(usable) <= n:
        return [Path(r["path"]).name for r in usable]

    picked: list[dict[str, Any]] = []
    used: set[str] = set()

    def take(row: dict[str, Any]) -> bool:
        name = Path(row["path"]).name
        if name in used:
            return False
        used.add(name)
        picked.append(row)
        return True

    def pool_for(scale: str) -> list[dict[str, Any]]:
        return [
            r for r in usable
            if str(_row_debug(r).get("scale") or "") == scale and Path(r["path"]).name not in used
        ]

    closeups = [r for r in usable if str(_row_debug(r).get("scale") or "") == "closeup"]
    for pose_bin in (
        "front",
        "three-quarter-left",
        "three-quarter-right",
        "profile-left",
        "profile-right",
    ):
        for row in closeups:
            if str(_row_debug(row).get("pose_bin") or "") == pose_bin and take(row):
                break
        if len(picked) >= n:
            return [Path(r["path"]).name for r in picked]

    quotas = {
        "closeup": int(THRESH["pick_cu"]),
        "medium": int(THRESH["pick_md"]),
        "full": int(THRESH["pick_full"]),
    }
    for scale, quota in quotas.items():
        have = sum(1 for r in picked if str(_row_debug(r).get("scale") or "") == scale)
        need = quota - have
        pool = pool_for(scale)
        for _ in range(max(0, need)):
            if not pool or len(picked) >= n:
                break
            best = max(pool, key=lambda r: _diversity_key(r, picked))
            take(best)
            pool = pool_for(scale)

    rest = [r for r in usable if Path(r["path"]).name not in used]
    while len(picked) < n and rest:
        best = max(rest, key=lambda r: _diversity_key(r, picked))
        take(best)
        rest = [r for r in usable if Path(r["path"]).name not in used]
    return [Path(r["path"]).name for r in picked]


def score_bgr(
    bgr: np.ndarray,
    path: Path | str = "",
    detector: Callable[[np.ndarray], list[FaceHit]] | None = None,
) -> dict[str, Any]:
    h, w = bgr.shape[:2]
    short = min(h, w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hits = (detector or detect_faces)(bgr)
    if not hits:
        return _result(path, 0.0, "no_face")
    hit = hits[0]
    crop = _padded_crop(bgr, hit, THRESH["pad"])
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else gray
    face = bgr[hit.y0:hit.y1, hit.x0:hit.x1]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.size else crop_gray
    lap_src = _resize_short(crop_gray, THRESH["lap_short"])
    v = laplacian_var(lap_src)
    eye_vs = [laplacian_var(_resize_short(eye, THRESH["lap_short"])) for eye in _eye_patches(gray, hit)]
    if eye_vs and v > 0:
        # In-focus eyes are smoother than the full face crop. Only pull
        # sharpness down when the eyes collapse relative to the face.
        eye_v = min(eye_vs)
        rel = eye_v / v
        if rel < THRESH["eye_rel"]:
            v *= rel / THRESH["eye_rel"]
    s_sharp = s_sharp_from_var(v)
    s_face = s_face_from_ied(hit.ied_px, float(hit.short))
    if short < THRESH["min_short"]:
        s_face = min(s_face, THRESH["s_face_cap_small"])
    m, c, d, s_exp, s_clip, s_split = light_terms(face_gray)
    s_light = clamp(s_exp * s_clip * s_split)
    s_pose = s_pose_from_angles(hit.yaw_deg, hit.pitch_deg)
    score = 100.0 * s_sharp * s_face * s_light * s_pose
    face_frac = max(0, hit.y1 - hit.y0) / float(h or 1)
    ied_ratio = float(hit.ied_px or 0.0) / float(short or 1)
    debug = {
        "ied_px": round(float(hit.ied_px or 0.0), 2),
        "laplacian_var": round(v, 3),
        "face_mean": round(m, 2),
        "clip_frac": round(c, 4),
        "split": round(d, 4),
        "yaw_deg": round(float(hit.yaw_deg or 0.0), 2),
        "pitch_deg": round(float(hit.pitch_deg or 0.0), 2),
        "s_sharp": round(s_sharp, 4),
        "s_face": round(s_face, 4),
        "s_light": round(s_light, 4),
        "s_pose": round(s_pose, 4),
        "n_faces": float(hit.n_faces),
        "face_frac": round(face_frac, 4),
        "scale": classify_scale(face_frac, ied_ratio),
        "pose_bin": classify_pose(hit.yaw_deg),
    }
    reason = _pick_reason(score, s_sharp, s_face, s_light, s_pose, m, c, d)
    return _result(path, score, reason, debug)


def score_image(path: Path | str, detector: Callable[[np.ndarray], list[FaceHit]] | None = None) -> dict[str, Any]:
    file = Path(path)
    bgr = load_bgr(file)
    if bgr is None:
        return _result(file, 0.0, "unreadable")
    return score_bgr(bgr, file, detector=detector)


def rank_dir(folder: Path | str) -> list[dict[str, Any]]:
    root = Path(folder)
    if not root.is_dir():
        raise QualityError(f"not a folder: {root}")
    paths = sorted(
        [p for p in root.iterdir() if p.is_file() and is_image(p)],
        key=lambda p: p.name.lower(),
    )
    results = [score_image(p) for p in paths]
    results.sort(key=lambda row: (-float(row["score"]), Path(row["path"]).name.lower()))
    return results


def scan_dataset(slug: str, files: list[str] | None = None) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    manifest = load_manifest(folder, slug)
    wanted = None
    if files:
        wanted = {Path(name).name for name in files}
    results: list[dict[str, Any]] = []
    for item in manifest.get("items") or []:
        name = item.get("file") or ""
        if wanted is not None and name not in wanted:
            continue
        path = folder / name
        if not path.is_file():
            item["quality"] = {"score": 0.0, "reason": "unreadable", "debug": _empty_debug(), "at": utc_now()}
            results.append(_result(path, 0.0, "unreadable"))
            continue
        if not is_image(path):
            item["quality"] = {"score": None, "reason": "not_image", "debug": _empty_debug(), "at": utc_now()}
            results.append({"path": str(path), "score": None, "reason": "not_image", "debug": _empty_debug()})
            continue
        row = score_image(path)
        item["quality"] = {
            "score": row["score"],
            "reason": row["reason"],
            "debug": row["debug"],
            "picked": False,
            "at": utc_now(),
        }
        results.append(row)
    picks = set(pick_diverse(results))
    for item in manifest.get("items") or []:
        q = item.get("quality")
        if isinstance(q, dict):
            q["picked"] = item.get("file") in picks
    def sort_key(row: dict[str, Any]) -> tuple:
        score = row.get("score")
        if not isinstance(score, (int, float)):
            return (1, 0.0, Path(row["path"]).name.lower())
        return (0, -float(score), Path(row["path"]).name.lower())
    results.sort(key=sort_key)
    manifest["quality_advice"] = advise(results)
    save_manifest(folder, manifest)
    return {
        "dataset": public_dataset(slug, folder),
        "results": results,
        "picks": [Path(r["path"]).name for r in results if Path(r["path"]).name in picks],
        "crowding": crowding(results),
        "advice": manifest.get("quality_advice") or [],
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'score':>6}  {'reason':<14}  file")
    for row in rows:
        score = row.get("score")
        score_s = "—" if score is None else f"{float(score):5.1f}"
        print(f"{score_s:>6}  {str(row.get('reason') or ''):<14}  {Path(row['path']).name}")


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if target.is_dir():
        _print_table(rank_dir(target))
    elif target.is_file():
        _print_table([score_image(target)])
    else:
        raise SystemExit(f"not found: {target}")
