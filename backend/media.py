"""Thumbnails and still-size probes."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .config import THUMBS_DIR

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
THUMB_MAX = 720
THUMB_VER = "srgb-lin3"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def is_media(path: Path) -> bool:
    return is_image(path) or is_video(path)


def probe_size(path: Path) -> tuple[int, int] | None:
    if not is_image(path):
        return None
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img) or img
            return img.size
    except OSError:
        return None


def _to_srgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img) or img
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        rgb = Image.new("RGB", img.size, (12, 11, 9))
        rgb.paste(img.convert("RGB"), mask=img.getchannel("A"))
        img = rgb
    icc = img.info.get("icc_profile")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(BytesIO(icc if isinstance(icc, (bytes, bytearray)) else bytes(icc)))
            dst = ImageCms.createProfile("sRGB")
            mode = "RGB" if img.mode != "CMYK" else "CMYK"
            img = ImageCms.profileToProfile(img.convert(mode), src, dst, outputMode="RGB")
            return img
        except (ValueError, OSError, SyntaxError, ImageCms.PyCMSError):
            pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_linear(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    lin = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    chans = []
    for c in range(3):
        plane = Image.fromarray(lin[:, :, c], mode="F")
        plane = plane.resize((nw, nh), Image.Resampling.LANCZOS)
        chans.append(np.asarray(plane, dtype=np.float32))
    lin_r = np.clip(np.stack(chans, axis=-1), 0.0, None)
    srgb = np.where(
        lin_r <= 0.0031308,
        lin_r * 12.92,
        1.055 * np.power(lin_r, 1.0 / 2.4) - 0.055,
    )
    return Image.fromarray(np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")


def thumb_bytes(path: Path) -> bytes | None:
    if not is_image(path) or not path.is_file():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    key = hashlib.sha1(
        f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{THUMB_MAX}|{THUMB_VER}".encode()
    ).hexdigest()
    cache = THUMBS_DIR / f"{key}.jpg"
    if cache.is_file():
        return cache.read_bytes()
    try:
        with Image.open(path) as src:
            img = _resize_linear(_to_srgb(src), THUMB_MAX)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=88, optimize=True, subsampling=1)
            data = buf.getvalue()
        cache.write_bytes(data)
        return data
    except OSError:
        return None
