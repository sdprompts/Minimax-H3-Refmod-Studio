"""Split an irregular Krea contact sheet into panel images.

Krea sheets usually have thin white gutters and a gray backdrop inside
each cell. Some sheets use black separator lines instead. Panels are
often different sizes, so a global 4x4 crop fails.

Splits on white or black separator lines. When rows have different
column counts (5 then 4 then 4), vertical gutters do not run the full
height, so the splitter picks the stronger axis first (full-width row
gutters vs full-height column gutters) and subdivides any cell that
still has gutters inside it.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


class SplitError(ValueError):
    pass


@dataclass
class SplitOptions:
    mode: str = "auto"
    white: float = 240
    black: float = 40
    gutter_frac: float = 0.35
    min_gutter: int = 1
    min_panel: int = 48
    min_area: int = 400
    min_panels: int = 2
    pad: int = 0
    square: bool = False


def gray_of(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3].astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def invert_gray(gray: np.ndarray) -> np.ndarray:
    return np.asarray(255.0 - gray, dtype=np.float32)


def dark_gutter_options(options: SplitOptions) -> SplitOptions:
    """Threshold so original pixels <= black count as gutters after invert."""
    return replace(options, white=255.0 - options.black)


def runs_from_mask(mask: np.ndarray, min_width: int) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    out = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_width:
            out.append((int(s), int(e)))
    return out


def spans_between(gutters: list[tuple[int, int]], length: int, min_span: int) -> list[tuple[int, int]]:
    """Content spans between gutter bands. End is exclusive."""
    if not gutters:
        return [(0, length)] if length >= min_span else []
    spans = []
    cursor = 0
    for g0, g1 in gutters:
        if g0 - cursor >= min_span:
            spans.append((cursor, g0))
        cursor = g1 + 1
    if length - cursor >= min_span:
        spans.append((cursor, length))
    return spans


def merge_nearby_runs(runs: list[tuple[int, int]], max_gap: int = 2) -> list[tuple[int, int]]:
    """Join gutter hits separated by anti-aliased pixels."""
    if not runs:
        return []
    out = [runs[0]]
    for start, end in runs[1:]:
        prev_s, prev_e = out[-1]
        if start - prev_e - 1 <= max_gap:
            out[-1] = (prev_s, end)
        else:
            out.append((start, end))
    return out


def line_gutters(frac: np.ndarray, min_width: int, frac_thr: float) -> list[tuple[int, int]]:
    raw = runs_from_mask(frac >= frac_thr, 1)
    merged = merge_nearby_runs(raw, max_gap=2)
    return [(s, e) for s, e in merged if (e - s + 1) >= min_width]


def _axis_spans(
    white: np.ndarray,
    axis: int,
    min_gutter: int,
    frac_thr: float,
    min_panel: int,
) -> list[tuple[int, int]]:
    """axis=0 → row spans (horizontal gutters); axis=1 → column spans."""
    frac = white.mean(axis=1 - axis)
    length = white.shape[axis]
    gutters = line_gutters(frac, min_gutter, frac_thr)
    spans = spans_between(gutters, length, min_panel)
    return spans or ([(0, length)] if length >= min_panel else [])


def gutter_strength(
    white: np.ndarray,
    axis: int,
    min_gutter: int,
    frac_thr: float,
) -> float:
    """How completely gutters span the other axis. Full-width row lines score high."""
    frac = white.mean(axis=1 - axis)
    gutters = line_gutters(frac, min_gutter, frac_thr)
    if not gutters:
        return 0.0
    return float(np.mean([float(frac[a : b + 1].mean()) for a, b in gutters]))


def pick_primary(white: np.ndarray, min_gutter: int, frac_thr: float) -> str:
    col_s = gutter_strength(white, 1, min_gutter, frac_thr)
    row_s = gutter_strength(white, 0, min_gutter, frac_thr)
    return "rows" if row_s > col_s + 0.05 else "cols"


def split_by_gutters(
    gray: np.ndarray,
    white_thr: float,
    frac_thr: float,
    min_gutter: int,
    min_panel: int,
    primary: str = "cols",
) -> list[tuple[int, int, int, int]]:
    """Split on gutters. primary='cols' is mixed row heights; 'rows' is mixed column counts."""
    h, w = gray.shape
    white = gray >= white_thr
    boxes: list[tuple[int, int, int, int]] = []
    if primary == "rows":
        row_spans = _axis_spans(white, 0, min_gutter, frac_thr, min_panel) or [(0, h)]
        for y0, y1 in row_spans:
            strip = white[y0:y1, :]
            col_spans = _axis_spans(strip, 1, min_gutter, frac_thr, min_panel) or [(0, w)]
            for x0, x1 in col_spans:
                boxes.append((x0, y0, x1, y1))
        return boxes
    col_spans = _axis_spans(white, 1, min_gutter, frac_thr, min_panel) or [(0, w)]
    for x0, x1 in col_spans:
        strip = white[:, x0:x1]
        row_spans = _axis_spans(strip, 0, min_gutter, frac_thr, min_panel) or [(0, h)]
        for y0, y1 in row_spans:
            boxes.append((x0, y0, x1, y1))
    return boxes


def _map_boxes(
    boxes: list[tuple[int, int, int, int]],
    ox: int,
    oy: int,
) -> list[tuple[int, int, int, int]]:
    return [(ox + x0, oy + y0, ox + x1, oy + y1) for x0, y0, x1, y1 in boxes]


def split_region(
    gray: np.ndarray,
    white_thr: float,
    frac_thr: float,
    min_gutter: int,
    min_panel: int,
    depth: int = 0,
    force_primary: str | None = None,
) -> list[tuple[int, int, int, int]]:
    """Axis-aware split, then subdivide any cell that still contains gutters."""
    h, w = gray.shape
    if depth > 6 or h < min_panel or w < min_panel:
        return [(0, 0, w, h)]
    white = gray >= white_thr
    if force_primary in ("rows", "cols") and depth == 0:
        primary = force_primary
    else:
        primary = pick_primary(white, min_gutter, frac_thr)
    boxes = split_by_gutters(gray, white_thr, frac_thr, min_gutter, min_panel, primary)
    if len(boxes) <= 1:
        other = "rows" if primary == "cols" else "cols"
        boxes = split_by_gutters(gray, white_thr, frac_thr, min_gutter, min_panel, other)
    if len(boxes) <= 1:
        return [(0, 0, w, h)]
    out: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        if x1 - x0 < min_panel or y1 - y0 < min_panel:
            continue
        sub = gray[y0:y1, x0:x1]
        local_min = min(min_panel, max(16, min(sub.shape) // 30))
        inner = split_region(sub, white_thr, frac_thr, min_gutter, local_min, depth + 1)
        if len(inner) == 1 and inner[0] == (0, 0, sub.shape[1], sub.shape[0]):
            out.append((x0, y0, x1, y1))
        else:
            out.extend(_map_boxes(inner, x0, y0))
    return out or [(0, 0, w, h)]


def tight_box(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    white_thr: float,
    min_size: int,
) -> tuple[int, int, int, int]:
    """Shrink a box to non-white content so leftover gutter strips are dropped."""
    x0, y0, x1, y1 = box
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return box
    content = region < white_thr
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return box
    nx0 = x0 + int(cols[0])
    nx1 = x0 + int(cols[-1]) + 1
    ny0 = y0 + int(rows[0])
    ny1 = y0 + int(rows[-1]) + 1
    if nx1 - nx0 < min_size or ny1 - ny0 < min_size:
        return box
    return (nx0, ny0, nx1, ny1)


def drop_fragments(
    boxes: list[tuple[int, int, int, int]],
    min_frac: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    """Drop slivers left when a gutter cut through a neighboring panel."""
    if len(boxes) < 2:
        return boxes
    widths = [x1 - x0 for x0, y0, x1, y1 in boxes]
    heights = [y1 - y0 for x0, y0, x1, y1 in boxes]
    areas = [w * h for w, h in zip(widths, heights)]
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))
    med_a = float(np.median(areas))
    if med_w <= 0 or med_h <= 0 or med_a <= 0:
        return boxes
    kept = []
    for box, width, height, area in zip(boxes, widths, heights, areas):
        if area < min_frac * med_a or width < min_frac * med_w or height < min_frac * med_h:
            continue
        kept.append(box)
    return kept or boxes


def drop_white_boxes(
    gray: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    white_thr: float,
    max_white: float = 0.85,
) -> list[tuple[int, int, int, int]]:
    kept = []
    for x0, y0, x1, y1 in boxes:
        region = gray[y0:y1, x0:x1]
        if region.size == 0:
            continue
        if float((region >= white_thr).mean()) > max_white:
            continue
        kept.append((x0, y0, x1, y1))
    return kept


def split_blobs(
    gray: np.ndarray,
    white_thr: float,
    min_area: int,
    opening: bool = True,
) -> list[tuple[int, int, int, int]]:
    """Each non-white connected region. Do not merge across gutters."""
    content = gray < white_thr
    if opening:
        content = ndimage.binary_opening(content, iterations=1)
    labeled, n = ndimage.label(content)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if ys.size < min_area:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def reading_order(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    heights = [b[3] - b[1] for b in boxes]
    tol = max(20, float(np.median(heights)) * 0.4)
    rows: list[list[tuple[int, int, int, int]]] = []
    for b in sorted(boxes, key=lambda box: ((box[1] + box[3]) / 2.0, box[0])):
        cy = (b[1] + b[3]) / 2.0
        placed = False
        for row in rows:
            rcy = float(np.mean([(r[1] + r[3]) / 2.0 for r in row]))
            if abs(cy - rcy) <= tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    rows.sort(key=lambda row: float(np.mean([(r[1] + r[3]) / 2.0 for r in row])))
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def pad_box(box, w, h, pad):
    x0, y0, x1, y1 = box
    return (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))


def make_square(im: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im.convert("RGB"), ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
    return canvas


def _finish_boxes(
    gray: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    white_thr: float,
    min_panel: int,
) -> list[tuple[int, int, int, int]]:
    boxes = drop_white_boxes(gray, boxes, white_thr)
    boxes = drop_fragments(boxes)
    return [tight_box(gray, b, white_thr, min_panel) for b in boxes]


METHOD_SPECS: tuple[tuple[str, str, str], ...] = (
    ("auto", "Auto", "White or black gutters, falling back to separate regions"),
    ("rows", "Rows first", "Split rows, then columns — mixed column counts"),
    ("cols", "Columns first", "Split columns, then rows — mixed row heights"),
    ("blobs", "Scattered", "Each separate region — irregular layouts"),
    ("blobs-raw", "Scattered (raw)", "Regions with no cleanup — tight or overlapping packs"),
    ("gutters-soft", "Soft gutters", "Treat light-gray seams as separators"),
    ("gutters-dark", "Dark lines", "Black separator lines instead of white"),
    ("rows-dark", "Rows first (dark)", "Split rows on black lines — mixed column counts"),
    ("cols-dark", "Columns first (dark)", "Split columns on black lines — mixed row heights"),
    ("blobs-dark", "Scattered (dark)", "Regions on a black backdrop"),
)

DARK_ALIASES: dict[str, str] = {
    "gutters-dark": "gutters",
    "rows-dark": "rows",
    "cols-dark": "cols",
    "blobs-dark": "blobs",
}


def boxes_for_method(
    gray: np.ndarray,
    options: SplitOptions,
    method_id: str,
) -> list[tuple[int, int, int, int]]:
    if method_id in DARK_ALIASES:
        return boxes_for_method(
            invert_gray(gray),
            dark_gutter_options(options),
            DARK_ALIASES[method_id],
        )
    h, w = gray.shape
    min_panel = max(options.min_panel, min(h, w) // 30)
    min_area = max(options.min_area, min_panel * min_panel // 4)
    if method_id == "auto":
        return choose_boxes(gray, options)
    if method_id == "rows":
        boxes = split_region(
            gray, options.white, options.gutter_frac, options.min_gutter, min_panel,
            force_primary="rows",
        )
        return _finish_boxes(gray, boxes, options.white, min_panel)
    if method_id == "cols":
        boxes = split_region(
            gray, options.white, options.gutter_frac, options.min_gutter, min_panel,
            force_primary="cols",
        )
        return _finish_boxes(gray, boxes, options.white, min_panel)
    if method_id == "blobs":
        boxes = split_blobs(gray, options.white, min_area, opening=True)
        boxes = drop_white_boxes(gray, boxes, options.white)
        return [tight_box(gray, b, options.white, min_panel) for b in boxes]
    if method_id == "blobs-raw":
        boxes = split_blobs(gray, options.white, min_area, opening=False)
        boxes = drop_white_boxes(gray, boxes, options.white)
        return [tight_box(gray, b, options.white, min_panel) for b in boxes]
    if method_id == "gutters-soft":
        boxes = split_region(gray, 220, 0.25, options.min_gutter, min_panel)
        return _finish_boxes(gray, boxes, 220, min_panel)
    if method_id == "gutters":
        boxes = split_region(
            gray, options.white, options.gutter_frac, options.min_gutter, min_panel,
        )
        return _finish_boxes(gray, boxes, options.white, min_panel)
    raise SplitError(f"unknown split method '{method_id}'")


def list_method_results(
    im: Image.Image,
    options: SplitOptions | None = None,
) -> list[dict]:
    options = options or SplitOptions()
    gray = gray_of(np.array(im.convert("RGB")))
    results: list[dict] = []
    seen: set[tuple[tuple[int, int, int, int], ...]] = set()
    for method_id, label, hint in METHOD_SPECS:
        boxes = reading_order(boxes_for_method(gray, options, method_id))
        key = tuple(boxes)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "id": method_id,
            "label": label,
            "hint": hint,
            "count": len(boxes),
            "boxes": [list(b) for b in boxes],
        })
    return results


def _choose_boxes_on(gray: np.ndarray, options: SplitOptions) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape
    min_panel = max(options.min_panel, min(h, w) // 30)
    gutter_boxes = split_region(
        gray,
        white_thr=options.white,
        frac_thr=options.gutter_frac,
        min_gutter=options.min_gutter,
        min_panel=min_panel,
    )
    blob_boxes = split_blobs(
        gray,
        white_thr=options.white,
        min_area=max(options.min_area, min_panel * min_panel // 4),
    )
    gutter_boxes = _finish_boxes(gray, gutter_boxes, options.white, min_panel)
    blob_boxes = _finish_boxes(gray, blob_boxes, options.white, min_panel)

    if options.mode == "gutters":
        return gutter_boxes
    if options.mode == "blobs":
        return blob_boxes

    if len(gutter_boxes) >= options.min_panels:
        return gutter_boxes
    if len(blob_boxes) >= options.min_panels:
        return blob_boxes
    return gutter_boxes or blob_boxes


def choose_boxes(gray: np.ndarray, options: SplitOptions) -> list[tuple[int, int, int, int]]:
    light = _choose_boxes_on(gray, options)
    if len(light) >= options.min_panels:
        return light
    dark = _choose_boxes_on(invert_gray(gray), dark_gutter_options(options))
    if len(dark) >= options.min_panels:
        return dark
    return dark if len(dark) > len(light) else light


def draw_preview(im: Image.Image, boxes) -> Image.Image:
    preview = im.convert("RGB").copy()
    px = preview.load()
    w, h = preview.size
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1 - 1), min(h - 1, y1 - 1)
        for x in range(x0, x1 + 1):
            px[x, y0] = (255, 0, 0)
            px[x, y1] = (255, 0, 0)
        for y in range(y0, y1 + 1):
            px[x0, y] = (255, 0, 0)
            px[x1, y] = (255, 0, 0)
    return preview


def detect_boxes(im: Image.Image, options: SplitOptions | None = None) -> list[tuple[int, int, int, int]]:
    options = options or SplitOptions()
    arr = np.array(im.convert("RGB"))
    return reading_order(choose_boxes(gray_of(arr), options))


def split_image(im: Image.Image, options: SplitOptions | None = None) -> list[Image.Image]:
    options = options or SplitOptions()
    boxes = detect_boxes(im, options)
    if not boxes:
        raise SplitError("No panels found. Try a sheet with white or black gutters.")
    w, h = im.size
    crops = []
    for box in boxes:
        x0, y0, x1, y1 = pad_box(box, w, h, options.pad)
        crop = im.crop((x0, y0, x1, y1))
        if options.square:
            crop = make_square(crop)
        crops.append(crop)
    return crops


def png_bytes(im: Image.Image) -> bytes:
    buf = BytesIO()
    save = im
    if save.mode not in ("RGB", "RGBA", "L", "P"):
        save = save.convert("RGB")
    save.save(buf, format="PNG")
    return buf.getvalue()


def split_to_pngs(
    data: bytes,
    stem: str,
    options: SplitOptions | None = None,
) -> list[tuple[str, bytes]]:
    try:
        with Image.open(BytesIO(data)) as im:
            im.load()
            crops = split_image(im, options)
    except SplitError:
        raise
    except OSError as exc:
        raise SplitError(f"could not read image: {exc}") from exc
    if not stem:
        stem = "panel"
    return [(f"{stem}_{i:02d}.png", png_bytes(crop)) for i, crop in enumerate(crops, start=1)]


def clamp_boxes(
    boxes: list,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for raw in boxes:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            x0, y0, x1, y1 = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        except (TypeError, ValueError):
            continue
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0 = max(0, min(x0, width))
        x1 = max(0, min(x1, width))
        y0 = max(0, min(y0, height))
        y1 = max(0, min(y1, height))
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            out.append((x0, y0, x1, y1))
    return out


def crop_to_pngs(
    data: bytes,
    stem: str,
    boxes: list,
    options: SplitOptions | None = None,
) -> list[tuple[str, bytes]]:
    options = options or SplitOptions()
    try:
        with Image.open(BytesIO(data)) as im:
            im.load()
            w, h = im.size
            ordered = reading_order(clamp_boxes(boxes, w, h))
            if not ordered:
                raise SplitError("no usable cuts in the selected plan")
            crops = []
            for box in ordered:
                x0, y0, x1, y1 = pad_box(box, w, h, options.pad)
                crop = im.crop((x0, y0, x1, y1))
                if options.square:
                    crop = make_square(crop)
                crops.append(crop)
    except SplitError:
        raise
    except OSError as exc:
        raise SplitError(f"could not read image: {exc}") from exc
    if not stem:
        stem = "panel"
    return [(f"{stem}_{i:02d}.png", png_bytes(crop)) for i, crop in enumerate(crops, start=1)]


def preview_jpeg(im: Image.Image, max_edge: int = 1400, quality: int = 82) -> tuple[bytes, int, int]:
    rgb = im.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, max_edge / max(w, h, 1))
    pw, ph = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (pw, ph) != (w, h):
        rgb = rgb.resize((pw, ph), Image.Resampling.LANCZOS)
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), pw, ph


def sheet_preview_payload(
    filename: str,
    data: bytes,
    options: SplitOptions | None = None,
) -> dict:
    try:
        with Image.open(BytesIO(data)) as im:
            im.load()
            rgb = im.convert("RGB")
            methods = list_method_results(rgb, options)
            jpeg, pw, ph = preview_jpeg(rgb)
            width, height = rgb.size
    except OSError as exc:
        raise SplitError(f"could not read {filename}: {exc}") from exc
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "preview": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
        "preview_w": pw,
        "preview_h": ph,
        "methods": methods,
    }


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Split a Krea contact sheet into panels.")
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--mode", choices=("auto", "gutters", "blobs"), default="auto")
    p.add_argument("--white", type=float, default=240, help="Pixels >= this are light gutters")
    p.add_argument("--black", type=float, default=40, help="Pixels <= this are dark gutters")
    p.add_argument(
        "--gutter-frac",
        type=float,
        default=0.35,
        help="Fraction of a row/col that must be white to count as a separator",
    )
    p.add_argument("--min-gutter", type=int, default=1)
    p.add_argument("--min-panel", type=int, default=48)
    p.add_argument("--min-area", type=int, default=400)
    p.add_argument("--min-panels", type=int, default=2)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--square", action="store_true")
    p.add_argument("--preview", action="store_true")
    return p.parse_args(argv)


def options_from_args(args) -> SplitOptions:
    return SplitOptions(
        mode=args.mode,
        white=args.white,
        black=args.black,
        gutter_frac=args.gutter_frac,
        min_gutter=args.min_gutter,
        min_panel=args.min_panel,
        min_area=args.min_area,
        min_panels=args.min_panels,
        pad=args.pad,
        square=args.square,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    im = Image.open(args.image)
    options = options_from_args(args)
    boxes = detect_boxes(im, options)
    if not boxes:
        raise SystemExit("No panels found. Try --white 220, --black 40, or --gutter-frac 0.25")

    stem = args.image.stem
    out_dir = args.out or args.image.parent / f"{stem}_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = im.size
    print(f"found {len(boxes)} panels -> {out_dir}")
    for i, box in enumerate(boxes, start=1):
        x0, y0, x1, y1 = pad_box(box, w, h, args.pad)
        crop = im.crop((x0, y0, x1, y1))
        if args.square:
            crop = make_square(crop)
        path = out_dir / f"{stem}_{i:02d}.png"
        crop.save(path)
        print(f"  {path.name:40s} {crop.size[0]}x{crop.size[1]:<4d}  box={x0,y0,x1,y1}")
    if args.preview:
        preview_path = out_dir / f"{stem}_preview.png"
        draw_preview(im, boxes).save(preview_path)
        print(f"preview: {preview_path}")


if __name__ == "__main__":
    main()
