from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

import numpy as np

from backend.split import (
    SplitError,
    SplitOptions,
    boxes_for_method,
    crop_to_pngs,
    gray_of,
    list_method_results,
    sheet_preview_payload,
    split_image,
    split_to_pngs,
)


def contact_sheet(cols=2, rows=2, cell=80, gutter=8, colors=None, bg=(255, 255, 255)) -> Image.Image:
    w = cols * cell + (cols + 1) * gutter
    h = rows * cell + (rows + 1) * gutter
    im = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(im)
    palette = colors or [
        (200, 20, 20),
        (20, 180, 20),
        (20, 20, 200),
        (200, 200, 20),
        (180, 20, 180),
        (20, 180, 180),
    ]
    i = 0
    for r in range(rows):
        for c in range(cols):
            x = gutter + c * (cell + gutter)
            y = gutter + r * (cell + gutter)
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=palette[i % len(palette)])
            i += 1
    return im


def sheet_png(cols=2, rows=2) -> bytes:
    buf = BytesIO()
    contact_sheet(cols, rows).save(buf, format="PNG")
    return buf.getvalue()


def test_split_regular_grid():
    im = contact_sheet(2, 2)
    crops = split_image(im)
    assert len(crops) == 4
    assert crops[0].size[0] >= 70
    assert crops[0].getpixel((10, 10))[0] > 150  # red-ish first cell


def test_split_to_pngs_names():
    panels = split_to_pngs(sheet_png(2, 2), "cassie")
    assert [name for name, _ in panels] == [
        "cassie_01.png",
        "cassie_02.png",
        "cassie_03.png",
        "cassie_04.png",
    ]
    for _, data in panels:
        with Image.open(BytesIO(data)) as crop:
            crop.load()
            assert crop.size[0] >= 70


def test_blank_sheet_is_not_a_grid():
    blank = Image.new("RGB", (200, 200), (255, 255, 255))
    try:
        crops = split_image(blank)
    except SplitError:
        return
    assert len(crops) < 2


def test_split_irregular_column_heights():
    im = Image.new("RGB", (220, 260), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.rectangle([8, 8, 100, 120], fill=(200, 30, 30))
    draw.rectangle([8, 136, 100, 248], fill=(30, 200, 30))
    draw.rectangle([116, 8, 208, 80], fill=(30, 30, 200))
    draw.rectangle([116, 96, 208, 168], fill=(200, 200, 30))
    draw.rectangle([116, 184, 208, 248], fill=(200, 30, 200))
    crops = split_image(im, SplitOptions(min_panel=40))
    assert len(crops) == 5


def ragged_row_sheet(row_counts=(5, 4, 4), cell=80, gutter=8, bg=(255, 255, 255)) -> Image.Image:
    """Same canvas width, different panel counts per row — the June layout."""
    rows = len(row_counts)
    max_cols = max(row_counts)
    canvas_w = max_cols * cell + (max_cols + 1) * gutter
    canvas_h = rows * cell + (rows + 1) * gutter
    im = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(im)
    palette = [
        (200, 20, 20),
        (20, 180, 20),
        (20, 20, 200),
        (200, 200, 20),
        (180, 20, 180),
        (20, 180, 180),
        (180, 100, 20),
        (100, 20, 180),
        (20, 100, 80),
        (80, 80, 20),
        (20, 80, 80),
        (80, 20, 80),
        (140, 140, 40),
    ]
    i = 0
    for r, cols in enumerate(row_counts):
        y = gutter + r * (cell + gutter)
        inner = canvas_w - 2 * gutter
        panel_w = (inner - (cols - 1) * gutter) // cols
        extra = inner - (cols * panel_w + (cols - 1) * gutter)
        x = gutter
        for c in range(cols):
            w = panel_w + (1 if c < extra else 0)
            draw.rectangle([x, y, x + w - 1, y + cell - 1], fill=palette[i % len(palette)])
            i += 1
            x += w + gutter
    return im


def test_split_ragged_rows_five_four_four():
    im = ragged_row_sheet((5, 4, 4), cell=80, gutter=8)
    crops = split_image(im, SplitOptions(min_panel=40))
    assert len(crops) == 13


def test_split_one_pixel_gutter():
    im = Image.new("RGB", (201, 80), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 99, 79], fill=(180, 40, 40))
    draw.rectangle([101, 0, 200, 79], fill=(40, 40, 180))
    crops = split_image(im, SplitOptions(min_panel=20, min_gutter=1))
    assert len(crops) == 2


def test_june_merged_strips_split_apart():
    root = Path("datasets/june")
    cases = [
        ("upscaled_00002__03.png", (1443, 875), 3),
        ("upscaled_00002__06.png", (1443, 851), 2),
        ("upscaled_00002__09.png", (1443, 1324), 2),
        ("upscaled_00002__01.png", (809, 875), 1),
        ("upscaled_00002__02.png", (812, 875), 1),
    ]
    checked = 0
    for name, size, expected in cases:
        path = root / name
        if not path.is_file():
            continue
        im = Image.open(path)
        if im.size != size:
            continue
        crops = split_image(im)
        assert len(crops) == expected, f"{name}: got {len(crops)} want {expected}"
        checked += 1
    if checked == 0:
        return


def test_list_methods_includes_scattered_and_grid():
    im = contact_sheet(2, 2)
    methods = list_method_results(im)
    assert methods[0]["id"] == "auto"
    grid = next(m for m in methods if m["count"] == 4)
    assert len(grid["boxes"]) == 4


def test_scattered_layout_has_a_four_region_method():
    im = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.rectangle([10, 10, 90, 100], fill=(200, 30, 30))
    draw.rectangle([140, 40, 250, 130], fill=(30, 200, 30))
    draw.rectangle([20, 160, 130, 280], fill=(30, 30, 200))
    draw.rectangle([200, 170, 380, 290], fill=(200, 200, 30))
    methods = list_method_results(im, SplitOptions(min_panel=20, min_area=80))
    assert any(m["count"] == 4 for m in methods)


def test_crop_to_pngs_uses_selected_boxes():
    png = sheet_png(2, 2)
    methods = list_method_results(contact_sheet(2, 2))
    grid = next(m for m in methods if m["count"] == 4)
    kept = grid["boxes"][:2]
    panels = crop_to_pngs(png, "sel", kept)
    assert len(panels) == 2
    assert [name for name, _ in panels] == ["sel_01.png", "sel_02.png"]


def test_sheet_preview_payload_has_image_and_methods():
    payload = sheet_preview_payload("sheet.png", sheet_png(2, 2))
    assert payload["filename"] == "sheet.png"
    assert payload["width"] > 0 and payload["height"] > 0
    assert payload["preview"].startswith("data:image/jpeg;base64,")
    assert payload["methods"]
    assert all("boxes" in m and "label" in m for m in payload["methods"])


def test_split_black_gutters():
    im = contact_sheet(2, 2, bg=(0, 0, 0))
    crops = split_image(im)
    assert len(crops) == 4
    assert crops[0].getpixel((10, 10))[0] > 150


def test_split_dark_gray_gutters():
    im = contact_sheet(2, 2, bg=(28, 28, 28))
    crops = split_image(im)
    assert len(crops) == 4


def test_split_black_ragged_rows():
    im = ragged_row_sheet((5, 4, 4), cell=80, gutter=8, bg=(0, 0, 0))
    crops = split_image(im, SplitOptions(min_panel=40))
    assert len(crops) == 13


def test_split_black_irregular_column_heights():
    im = Image.new("RGB", (220, 260), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.rectangle([8, 8, 100, 120], fill=(200, 30, 30))
    draw.rectangle([8, 136, 100, 248], fill=(30, 200, 30))
    draw.rectangle([116, 8, 208, 80], fill=(30, 30, 200))
    draw.rectangle([116, 96, 208, 168], fill=(200, 200, 30))
    draw.rectangle([116, 184, 208, 248], fill=(200, 30, 200))
    crops = split_image(im, SplitOptions(min_panel=40))
    assert len(crops) == 5


def test_dark_line_methods_on_black_sheet():
    im = contact_sheet(2, 2, bg=(0, 0, 0))
    methods = list_method_results(im)
    auto = next(m for m in methods if m["id"] == "auto")
    assert auto["count"] == 4
    gray = gray_of(np.array(im.convert("RGB")))
    assert len(boxes_for_method(gray, SplitOptions(), "gutters-dark")) == 4
    assert len(boxes_for_method(gray, SplitOptions(), "rows-dark")) == 4
    assert len(boxes_for_method(gray, SplitOptions(), "blobs-dark")) == 4


def test_white_sheet_still_prefers_light_gutters():
    im = contact_sheet(3, 2)
    crops = split_image(im)
    assert len(crops) == 6


def test_split_does_not_square_by_default():
    im = Image.new("RGB", (200, 120), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.rectangle([8, 8, 88, 111], fill=(180, 40, 40))
    draw.rectangle([108, 8, 191, 111], fill=(40, 40, 180))
    crops = split_image(im, SplitOptions(square=False, min_panel=40))
    assert len(crops) == 2
    for crop in crops:
        assert crop.size[0] != crop.size[1]
