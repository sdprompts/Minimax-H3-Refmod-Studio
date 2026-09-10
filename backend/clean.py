"""Scan stills for edge watermarks, then crop those edges off with undo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .datasets import (
    ORIGINALS_DIR,
    _safe_filename,
    load_manifest,
    load_links,
    public_dataset,
    resolve_dataset,
    save_manifest,
)
from .media import is_image

MIN_SHORT = 1024
DETECT_MAX_SIDE = 1920
BAND = 0.085
MAX_H_FRAC = 0.07
MAX_W_FRAC = 0.66
MIN_W_FRAC = 0.045
MIN_H_FRAC = 0.008
MAX_AREA_FRAC = 0.06
MAX_STRIP_FRAC = 0.07
FLUSH = 0.02
ACTIONS = ("crop", "skip")
EDGES = ("top", "bottom", "left", "right")
CLEAN_STATES = ("", "flagged", "cleaned", "ignored")


class CleanError(ValueError):
    pass


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    kind: str = "edge"
    confidence: float = 0.6
    action: str = "crop"
    note: str = ""
    edge: str = ""

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    def to_norm(self, w: int, h: int) -> dict[str, Any]:
        return {
            "x0": round(self.x0 / w, 5),
            "y0": round(self.y0 / h, 5),
            "x1": round(self.x1 / w, 5),
            "y1": round(self.y1 / h, 5),
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "action": self.action or "crop",
            "note": self.note,
            "edge": self.edge,
        }


def originals_path(folder: Path) -> Path:
    return folder / ORIGINALS_DIR


def original_file(folder: Path, filename: str) -> Path:
    return originals_path(folder) / _safe_filename(filename)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def save_rgb(path: Path, arr: np.ndarray, like: Path | None = None) -> None:
    im = Image.fromarray(arr)
    suffix = (like or path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        im.save(path, format="JPEG", quality=95, subsampling=0)
    elif suffix == ".png":
        im.save(path, format="PNG")
    elif suffix == ".webp":
        im.save(path, format="WEBP", quality=95)
    else:
        im.save(path)


def _clip(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = int(max(0, min(w - 1, x0)))
    y0 = int(max(0, min(h - 1, y0)))
    x1 = int(max(x0 + 1, min(w, x1)))
    y1 = int(max(y0 + 1, min(h, y1)))
    return x0, y0, x1, y1


def _band(h: int, w: int, frac: float = BAND) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    by = max(10, int(h * frac))
    bx = max(10, int(w * frac))
    mask[:by, :] = True
    mask[-by:, :] = True
    mask[:, :bx] = True
    mask[:, -bx:] = True
    return mask


def _flush(box: tuple[int, int, int, int], w: int, h: int, frac: float = FLUSH) -> bool:
    x0, y0, x1, y1 = box
    mx, my = max(3, int(w * frac)), max(3, int(h * frac))
    return x0 <= mx or y0 <= my or (w - x1) <= mx or (h - y1) <= my


def _in_corner(box: tuple[int, int, int, int], w: int, h: int) -> bool:
    x0, y0, x1, y1 = box
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    near_x = cx <= 0.32 * w or cx >= 0.68 * w
    near_y = cy <= 0.16 * h or cy >= 0.84 * h
    return near_x and near_y


def _is_edge_bar(box: tuple[int, int, int, int], w: int, h: int) -> bool:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bh > 0.075 * h or bw < 0.16 * w:
        return False
    top = y0 <= 0.03 * h
    bot = y1 >= 0.97 * h
    return top or bot


def _geometry_ok(box: tuple[int, int, int, int], w: int, h: int) -> bool:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw < max(22, int(w * MIN_W_FRAC)) or bh < max(8, int(h * MIN_H_FRAC)):
        return False
    if bh > h * MAX_H_FRAC or bw > w * MAX_W_FRAC:
        return False
    if bw * bh > w * h * MAX_AREA_FRAC:
        return False
    if not _flush(box, w, h):
        return False
    return _in_corner(box, w, h) or _is_edge_bar(box, w, h)


def _merge(boxes: list[tuple[int, int, int, int]], w: int, h: int, gap: int = 12) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    current = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[tuple[int, int, int, int]] = []
        used = [False] * len(current)
        for i, a in enumerate(current):
            if used[i]:
                continue
            x0, y0, x1, y1 = a
            for j, b in enumerate(current):
                if i == j or used[j]:
                    continue
                u0, v0, u1, v1 = b
                if x0 - gap <= u1 and u0 - gap <= x1 and y0 - gap <= v1 and v0 - gap <= y1:
                    nx0, ny0 = min(x0, u0), min(y0, v0)
                    nx1, ny1 = max(x1, u1), max(y1, v1)
                    if (ny1 - ny0) > h * MAX_H_FRAC or (nx1 - nx0) > w * MAX_W_FRAC:
                        continue
                    x0, y0, x1, y1 = nx0, ny0, nx1, ny1
                    used[j] = True
                    changed = True
            used[i] = True
            out.append((x0, y0, x1, y1))
        current = out
    return current


def _unique_color_plates(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = arr.shape[:2]
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    iy0, iy1 = int(h * 0.22), int(h * 0.78)
    ix0, ix1 = int(w * 0.22), int(w * 0.78)
    interior = hsv[iy0:iy1, ix0:ix1]
    if interior.size == 0:
        return []
    hist = np.bincount(interior[:, :, 0].ravel().astype(np.int32), minlength=180).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    sat, val, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    rare = hist[np.clip(hue, 0, 179)] < 0.025
    mask = (sat > 90) & (val > 45) & (val < 250) & rare & _band(h, w)
    u8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 50:
            continue
        box = _clip(x - 4, y - 3, x + bw + 4, y + bh + 3, w, h)
        if _geometry_ok(box, w, h):
            boxes.append(box)
    return boxes


def _dark_text_bars(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    dark = (gray < 42) & _band(h, w)
    u8 = dark.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        box = _clip(x - 2, y - 2, x + bw + 2, y + bh + 2, w, h)
        if not _geometry_ok(box, w, h):
            continue
        roi = gray[box[1]:box[3], box[0]:box[2]]
        if roi.size == 0:
            continue
        if (roi < 55).mean() < 0.30:
            continue
        if (roi > 140).mean() < 0.015:
            continue
        if float(roi.std()) < 16:
            continue
        boxes.append(box)
    return boxes


def _looks_like_overlay(rgb: np.ndarray) -> bool:
    if rgb.size == 0:
        return False
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    val, sat = hsv[:, :, 2], hsv[:, :, 1]
    whiteish = float(((val > 200) & (sat < 90)).mean())
    blackish = float((val < 42).mean())
    saturated = float((sat > 100).mean())
    return whiteish > 0.07 or blackish > 0.22 or saturated > 0.10


def _caption_strips(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    strip = max(14, int(h * 0.036))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    hat = cv2.add(
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
    )
    boxes: list[tuple[int, int, int, int]] = []
    for y0, y1 in ((0, strip), (h - strip, h)):
        band = hat[y0:y1]
        if band.size == 0 or int(band.max()) < 16:
            continue
        thr = max(16, int(np.percentile(band, 90)))
        th = (band >= thr).astype(np.uint8) * 255
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(th, 8)
        letters = []
        min_lh, max_lh = max(5, int(h * 0.007)), max(10, int(h * 0.032))
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if bh < min_lh or bh > max_lh or bw < 3 or bw > int(w * 0.14) or area < 10:
                continue
            letters.append((x, y0 + y, x + bw, y0 + y + bh))
        if not letters:
            continue
        letters.sort(key=lambda b: (b[1], b[0]))
        used = [False] * len(letters)
        for i, a in enumerate(letters):
            if used[i]:
                continue
            group = [a]
            used[i] = True
            acy = 0.5 * (a[1] + a[3])
            ah = max(a[3] - a[1], 8)
            grew = True
            while grew:
                grew = False
                gx0 = min(g[0] for g in group)
                gx1 = max(g[2] for g in group)
                for j, b in enumerate(letters):
                    if used[j]:
                        continue
                    bcy = 0.5 * (b[1] + b[3])
                    if abs(bcy - acy) > 0.75 * ah:
                        continue
                    gap = max(0, b[0] - gx1, gx0 - b[2])
                    if gap > max(16, 2.4 * ah):
                        continue
                    group.append(b)
                    used[j] = True
                    gx0 = min(g[0] for g in group)
                    gx1 = max(g[2] for g in group)
                    acy = 0.5 * (min(g[1] for g in group) + max(g[3] for g in group))
                    grew = True
            if len(group) < 3 and (max(g[2] for g in group) - min(g[0] for g in group)) < 0.12 * w:
                continue
            box = _clip(
                min(g[0] for g in group) - 6,
                min(g[1] for g in group) - 4,
                max(g[2] for g in group) + 6,
                max(g[3] for g in group) + 4,
                w, h,
            )
            roi = arr[box[1]:box[3], box[0]:box[2]]
            if not _looks_like_overlay(roi):
                continue
            if _geometry_ok(box, w, h):
                boxes.append(box)
    return boxes


def _mark_mask(arr: np.ndarray) -> np.ndarray:
    """Pixels in the outer band that look like overlay type, plates, or bars."""
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    band = _band(h, w, 0.12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    hat = cv2.add(
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
    )
    if band.any():
        thr = max(12, int(np.percentile(hat[band], 84)))
    else:
        thr = 12
    hat_bin = (hat >= thr) & band
    sat = (hsv[:, :, 1] > 75) & (hsv[:, :, 2] > 40) & band
    dark = (gray < 48) & band
    return (hat_bin | sat | dark).astype(np.uint8) * 255


def _seed_depth(box: tuple[int, int, int, int], w: int, h: int, edge: str) -> int:
    x0, y0, x1, y1 = box
    if edge == "top":
        return y1
    if edge == "bottom":
        return h - y0
    if edge == "left":
        return x1
    return w - x0


def _expand_seed(arr: np.ndarray, box: tuple[int, int, int, int], edge: str) -> tuple[int, int, int, int]:
    """Widen along the edge to cover sibling text. Do not grow into the photo."""
    h, w = arr.shape[:2]
    mask = _mark_mask(arr) > 0
    pad = max(3, int(0.003 * max(w, h)))
    slack = 0.22
    x0, y0, x1, y1 = box
    if edge in ("top", "bottom"):
        xa = max(0, int(x0 - slack * w))
        xb = min(w, int(x1 + slack * w))
        ya, yb = y0, y1
        region = mask[ya:yb, xa:xb]
        if region.any():
            _, xs = np.where(region)
            x0 = xa + int(xs.min()) - pad
            x1 = xa + int(xs.max()) + pad + 1
        if edge == "top":
            y0 = 0
            y1 = y1 + pad
        else:
            y0 = y0 - pad
            y1 = h
    else:
        ya = max(0, int(y0 - slack * h))
        yb = min(h, int(y1 + slack * h))
        xa, xb = x0, x1
        region = mask[ya:yb, xa:xb]
        if region.any():
            ys, _ = np.where(region)
            y0 = ya + int(ys.min()) - pad
            y1 = ya + int(ys.max()) + pad + 1
        if edge == "left":
            x0 = 0
            x1 = x1 + pad
        else:
            x0 = x0 - pad
            x1 = w
    return _clip(x0, y0, x1, y1, w, h)


def _as_strip(box: tuple[int, int, int, int], w: int, h: int, edge: str) -> tuple[int, int, int, int]:
    """Full-width/height band, depth clamped to the mark plus a few pixels."""
    pad = max(3, int(0.003 * max(w, h)))
    cap = max(16, int(MAX_STRIP_FRAC * (h if edge in ("top", "bottom") else w)))
    x0, y0, x1, y1 = box
    if edge == "top":
        return 0, 0, w, min(h, y1 + pad, cap)
    if edge == "bottom":
        return 0, max(0, y0 - pad, h - cap), w, h
    if edge == "left":
        return 0, 0, min(w, x1 + pad, cap), h
    return max(0, x0 - pad, w - cap), 0, w, h


def _to_crop_strips(arr: np.ndarray, seeds: list[tuple[int, int, int, int]]) -> list[Box]:
    h, w = arr.shape[:2]
    cap = max(16, int(MAX_STRIP_FRAC * max(w, h)))
    kept: list[tuple[str, tuple[int, int, int, int]]] = []
    for seed in seeds:
        edge = dominant_edge(seed, w, h)
        if _seed_depth(seed, w, h, edge) > cap:
            continue
        roi = arr[seed[1]:seed[3], seed[0]:seed[2]]
        if roi.size and not _looks_like_overlay(roi):
            continue
        grown = _expand_seed(arr, seed, edge)
        kept.append((edge, _as_strip(grown, w, h, edge)))
    by_edge: dict[str, tuple[int, int, int, int]] = {}
    for edge, strip in kept:
        prev = by_edge.get(edge)
        if prev is None:
            by_edge[edge] = strip
            continue
        # Same edge: keep the shallower inner edge (less crop), unless the other
        # seed is clearly farther along the mark — take the union of depth only
        # up to cap, which _as_strip already clamped.
        by_edge[edge] = (
            min(prev[0], strip[0]),
            min(prev[1], strip[1]),
            max(prev[2], strip[2]),
            max(prev[3], strip[3]),
        )
    return [
        Box(*strip, kind="strip", confidence=0.7, action="crop", edge=edge)
        for edge, strip in by_edge.items()
    ]


def detect_boxes(arr: np.ndarray) -> list[Box]:
    """Find edge watermarks and return the crop strips that cover them."""
    h, w = arr.shape[:2]
    if h < 32 or w < 32:
        return []
    scale = 1.0
    work = arr
    if max(h, w) > DETECT_MAX_SIDE:
        scale = DETECT_MAX_SIDE / max(h, w)
        work = cv2.resize(
            arr,
            (max(32, int(w * scale)), max(32, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    wh, ww = work.shape[:2]
    raw = _unique_color_plates(work) + _dark_text_bars(work) + _caption_strips(work)
    merged = _merge(raw, ww, wh, gap=14)
    seeds: list[tuple[int, int, int, int]] = []
    inv = 1.0 / scale
    for x0, y0, x1, y1 in merged:
        if not _geometry_ok((x0, y0, x1, y1), ww, wh):
            continue
        seeds.append(_clip(int(x0 * inv), int(y0 * inv), int(x1 * inv), int(y1 * inv), w, h))
    if not seeds:
        return []
    return _to_crop_strips(arr, seeds)


def iou_norm(a: dict[str, float], b: dict[str, float]) -> float:
    ax0, ay0, ax1, ay1 = a["x0"], a["y0"], a["x1"], a["y1"]
    bx0, by0, bx1, by1 = b["x0"], b["y0"], b["x1"], b["y1"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


def dominant_edge(box: tuple[int, int, int, int], w: int, h: int) -> str:
    """Pick the trim that keeps the largest remaining short edge."""
    x0, y0, x1, y1 = box
    options: list[tuple[str, int, int]] = []
    if y0 <= 0.10 * h:
        options.append(("top", y1, min(w, h - y1)))
    if y1 >= 0.90 * h:
        options.append(("bottom", h - y0, min(w, y0)))
    if x0 <= 0.10 * w:
        options.append(("left", x1, min(h, w - x1)))
    if x1 >= 0.90 * w:
        options.append(("right", w - x0, min(h, x0)))
    if not options:
        dist = {"left": x0, "right": w - x1, "top": y0, "bottom": h - y1}
        return min(dist, key=dist.get)
    options.sort(key=lambda row: (-row[2], row[1]))
    return options[0][0]


def combined_crop(
    w: int,
    h: int,
    boxes: list[Box] | list[tuple[int, int, int, int]],
    pad: int = 2,
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = 0, 0, w, h
    for raw in boxes:
        if isinstance(raw, Box):
            edge = raw.edge or dominant_edge(raw.as_tuple(), w, h)
            x0, y0, x1, y1 = raw.as_tuple()
        else:
            x0, y0, x1, y1 = raw
            edge = dominant_edge(raw, w, h)
        if edge == "top":
            top = max(top, y1 + pad)
        elif edge == "bottom":
            bottom = min(bottom, y0 - pad)
        elif edge == "left":
            left = max(left, x1 + pad)
        else:
            right = min(right, x0 - pad)
    if right - left < 32 or bottom - top < 32:
        return None
    if (right - left) * (bottom - top) < 0.25 * w * h:
        return None
    return left, top, right, bottom


def crop_is_safe(w: int, h: int, crop: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = crop
    nw, nh = right - left, bottom - top
    if nw < 32 or nh < 32:
        return False
    return nw * nh >= 0.25 * w * h


def suggest_action(arr: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, str]:
    h, w = arr.shape[:2]
    crop = combined_crop(w, h, [box])
    if crop is None:
        return "skip", "crop would empty the still"
    return "crop", ""


def box_from_norm(raw: dict[str, Any], w: int, h: int) -> Box:
    try:
        x0 = float(raw.get("x0", 0))
        y0 = float(raw.get("y0", 0))
        x1 = float(raw.get("x1", 0))
        y1 = float(raw.get("y1", 0))
    except (TypeError, ValueError) as exc:
        raise CleanError("box coordinates must be numbers") from exc
    if x1 <= x0 or y1 <= y0:
        raise CleanError("box is empty")
    px = _clip(int(round(x0 * w)), int(round(y0 * h)), int(round(x1 * w)), int(round(y1 * h)), w, h)
    action = str(raw.get("action") or "crop").lower()
    if action not in ACTIONS:
        action = "crop"
    edge = str(raw.get("edge") or "")
    if edge not in EDGES:
        edge = dominant_edge(px, w, h)
        px = _as_strip(px, w, h, edge)
    return Box(
        *px,
        kind=str(raw.get("kind") or "user"),
        confidence=float(raw.get("confidence") or 1),
        action=action,
        note=str(raw.get("note") or ""),
        edge=edge,
    )


def apply_boxes_to_array(arr: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, list[str]]:
    h, w = arr.shape[:2]
    crop_boxes = [b for b in boxes if b.action != "skip"]
    if not crop_boxes:
        return arr, []
    crop = combined_crop(w, h, crop_boxes)
    if crop is None:
        raise CleanError("crop would empty the still")
    left, top, right, bottom = crop
    return arr[top:bottom, left:right], ["crop"]


def keep_from_norm(raw: dict[str, Any], w: int, h: int) -> tuple[int, int, int, int]:
    try:
        x0 = float(raw.get("x0", 0))
        y0 = float(raw.get("y0", 0))
        x1 = float(raw.get("x1", 1))
        y1 = float(raw.get("y1", 1))
    except (TypeError, ValueError) as exc:
        raise CleanError("keep coordinates must be numbers") from exc
    if x1 <= x0 or y1 <= y0:
        raise CleanError("keep region is empty")
    left, top, right, bottom = _clip(
        int(round(x0 * w)),
        int(round(y0 * h)),
        int(round(x1 * w)),
        int(round(y1 * h)),
        w,
        h,
    )
    if right - left < 32 or bottom - top < 32:
        raise CleanError("keep region is too small")
    return left, top, right, bottom


def keep_norm_from_boxes(boxes: list[dict[str, Any]]) -> dict[str, float]:
    x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
    for box in boxes or []:
        edge = str(box.get("edge") or "")
        try:
            bx0, by0 = float(box["x0"]), float(box["y0"])
            bx1, by1 = float(box["x1"]), float(box["y1"])
        except (KeyError, TypeError, ValueError):
            continue
        if edge == "top":
            y0 = max(y0, by1)
        elif edge == "bottom":
            y1 = min(y1, by0)
        elif edge == "left":
            x0 = max(x0, bx1)
        elif edge == "right":
            x1 = min(x1, bx0)
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        return {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
    return {
        "x0": round(x0, 5),
        "y0": round(y0, 5),
        "x1": round(x1, 5),
        "y1": round(y1, 5),
    }


def apply_keep_to_array(arr: np.ndarray, keep: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = keep
    return arr[top:bottom, left:right]


def keep_is_full(keep: tuple[int, int, int, int], w: int, h: int, tol: int = 2) -> bool:
    left, top, right, bottom = keep
    return left <= tol and top <= tol and right >= w - tol and bottom >= h - tol


def _consensus(items: list[dict[str, Any]]) -> None:
    flagged = [it for it in items if it.get("boxes")]
    if len(flagged) < 2:
        return
    clusters: list[dict[str, Any]] = []
    for it in flagged:
        for box in it["boxes"]:
            placed = False
            for cluster in clusters:
                if iou_norm(box, cluster["proto"]) >= 0.35:
                    cluster["count"] += 1
                    cluster["files"].add(it["file"])
                    placed = True
                    break
            if not placed:
                clusters.append({"proto": dict(box), "count": 1, "files": {it["file"]}})
    n = len(items)
    need = max(2, int(0.35 * n))
    for cluster in clusters:
        if cluster["count"] < need:
            continue
        proto = cluster["proto"]
        proto = {**proto, "confidence": max(0.8, float(proto.get("confidence") or 0.6)), "kind": "consensus"}
        for it in items:
            if it["file"] in cluster["files"]:
                continue
            if any(iou_norm(box, proto) >= 0.35 for box in it["boxes"]):
                continue
            it["boxes"].append(dict(proto))
            it["flagged"] = True


def _source_path(folder: Path, filename: str) -> Path:
    orig = original_file(folder, filename)
    if orig.is_file():
        return orig
    return folder / filename


def ensure_original(folder: Path, filename: str) -> Path:
    src = folder / _safe_filename(filename)
    if not src.is_file():
        raise CleanError(f"{filename} is missing")
    dest = original_file(folder, filename)
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    return dest


def scan_dataset(slug: str, files: list[str] | None = None) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    manifest = load_manifest(folder, slug)
    wanted = None
    if files:
        wanted = {_safe_filename(name) for name in files}
    items_out: list[dict[str, Any]] = []
    by_file = {item["file"]: item for item in manifest.get("items") or []}
    for item in manifest.get("items") or []:
        name = item["file"]
        if wanted is not None and name not in wanted:
            continue
        path = folder / name
        if not path.is_file() or not is_image(path):
            continue
        arr = load_rgb(path)
        h, w = arr.shape[:2]
        found = detect_boxes(arr)
        boxes = [box.to_norm(w, h) for box in found]
        prev = str(item.get("clean") or "")
        flagged = bool(boxes)
        try:
            nbytes = path.stat().st_size
        except OSError:
            nbytes = 0
        if flagged:
            status = "flagged"
        elif prev == "cleaned":
            status = "cleaned"
        elif prev == "ignored":
            status = "ignored"
        else:
            status = ""
        by_file[name]["clean"] = status
        items_out.append({
            "file": name,
            "width": w,
            "height": h,
            "short_edge": min(w, h),
            "boxes": boxes,
            "keep": keep_norm_from_boxes(boxes),
            "flagged": flagged,
            "clean": status,
            "enabled": bool(item.get("enabled", True)),
            "has_original": original_file(folder, name).is_file(),
            "bytes": nbytes,
            "note": next((b["note"] for b in boxes if b.get("note")), ""),
        })
    _consensus(items_out)
    for row in items_out:
        row["keep"] = keep_norm_from_boxes(row.get("boxes") or [])
        stored = by_file.get(row["file"])
        if stored is None:
            continue
        if row["flagged"]:
            stored["clean"] = "flagged"
            row["clean"] = "flagged"
    save_manifest(folder, manifest)
    flagged_n = sum(1 for row in items_out if row["flagged"])
    return {
        "slug": slug,
        "linked": slug in load_links() or bool(manifest.get("linked")),
        "flagged": flagged_n,
        "total": len(items_out),
        "items": items_out,
        "dataset": public_dataset(slug, folder),
    }


def _apply_one(
    folder: Path,
    filename: str,
    boxes: list[Box],
    keep: tuple[int, int, int, int] | None,
    disable: bool,
    ignore: bool,
) -> dict[str, Any]:
    name = _safe_filename(filename)
    path = folder / name
    if not path.is_file():
        raise CleanError(f"{name} is missing")
    if disable:
        return {"file": name, "ok": True, "actions": ["disable"], "note": ""}
    if ignore:
        return {"file": name, "ok": True, "actions": ["ignore"], "note": ""}
    ensure_original(folder, name)
    src = original_file(folder, name)
    arr = load_rgb(path)
    h, w = arr.shape[:2]
    if keep is not None:
        if keep_is_full(keep, w, h):
            return {"file": name, "ok": True, "actions": [], "note": "nothing to apply"}
        out = apply_keep_to_array(arr, keep)
        used = ["crop"]
    elif boxes:
        out, used = apply_boxes_to_array(arr, boxes)
        if not used:
            return {"file": name, "ok": True, "actions": [], "note": "nothing to apply"}
    else:
        return {"file": name, "ok": True, "actions": [], "note": "nothing to apply"}
    save_rgb(path, out, like=src)
    return {"file": name, "ok": True, "actions": used, "note": ""}


def apply_clean(
    slug: str,
    items: list[dict[str, Any]] | None = None,
    apply_boxes: list[dict[str, Any]] | None = None,
    apply_keep: dict[str, Any] | None = None,
    files: list[str] | None = None,
) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    manifest = load_manifest(folder, slug)
    by_file = {item["file"]: item for item in manifest.get("items") or []}
    results: list[dict[str, Any]] = []
    plans: list[tuple[str, list[Box], tuple[int, int, int, int] | None, bool, bool]] = []

    if apply_keep is not None:
        names = files or [item["file"] for item in manifest.get("items") or [] if item.get("enabled", True)]
        for name in names:
            name = _safe_filename(name)
            path = folder / name
            if name not in by_file or not path.is_file() or not is_image(path):
                continue
            with Image.open(path) as im:
                w, h = im.size
            plans.append((name, [], keep_from_norm(apply_keep, w, h), False, False))
    elif apply_boxes:
        names = files or [item["file"] for item in manifest.get("items") or [] if item.get("enabled", True)]
        for name in names:
            name = _safe_filename(name)
            path = folder / name
            if name not in by_file or not path.is_file() or not is_image(path):
                continue
            with Image.open(path) as im:
                w, h = im.size
            boxes = [box_from_norm(raw, w, h) for raw in apply_boxes]
            plans.append((name, boxes, None, False, False))
    elif items:
        for raw in items:
            name = _safe_filename(str(raw.get("file") or ""))
            if not name or name not in by_file:
                continue
            path = folder / name
            if not path.is_file():
                continue
            disable = bool(raw.get("disable"))
            ignore = bool(raw.get("ignore"))
            with Image.open(path) as im:
                w, h = im.size
            keep = keep_from_norm(raw["keep"], w, h) if raw.get("keep") else None
            boxes = [box_from_norm(b, w, h) for b in (raw.get("boxes") or [])] if not keep else []
            plans.append((name, boxes, keep, disable, ignore))
    else:
        raise CleanError("nothing to apply")

    if not plans:
        raise CleanError("no matching stills")

    for name, boxes, keep, disable, ignore in plans:
        try:
            result = _apply_one(folder, name, boxes, keep, disable, ignore)
        except (CleanError, OSError, ValueError) as exc:
            results.append({"file": name, "ok": False, "actions": [], "note": str(exc)})
            continue
        item = by_file[name]
        if disable:
            item["enabled"] = False
            item["clean"] = "ignored"
        elif ignore:
            item["clean"] = "ignored"
        elif result["actions"] and result["actions"] != ["none"]:
            item["clean"] = "cleaned"
        results.append(result)

    save_manifest(folder, manifest)
    return {
        "results": results,
        "dataset": public_dataset(slug, folder),
    }


def undo_clean(slug: str, files: list[str] | None = None) -> dict[str, Any]:
    slug, folder = resolve_dataset(slug)
    manifest = load_manifest(folder, slug)
    by_file = {item["file"]: item for item in manifest.get("items") or []}
    if files:
        names = files
    elif originals_path(folder).is_dir():
        names = [p.name for p in originals_path(folder).iterdir() if p.is_file()]
    else:
        names = []
    restored = 0
    for raw in names:
        name = _safe_filename(raw)
        orig = original_file(folder, name)
        dest = folder / name
        if not orig.is_file():
            continue
        dest.write_bytes(orig.read_bytes())
        if name in by_file:
            by_file[name]["clean"] = ""
        restored += 1
    if not restored:
        raise CleanError("no originals to restore")
    save_manifest(folder, manifest)
    return {"restored": restored, "dataset": public_dataset(slug, folder)}
