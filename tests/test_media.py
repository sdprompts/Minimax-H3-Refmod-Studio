from pathlib import Path

from PIL import Image

from backend.media import thumb_bytes


def test_thumb_bytes_is_jpeg(tmp_path, monkeypatch):
    from backend import media as media_mod

    monkeypatch.setattr(media_mod, "THUMBS_DIR", tmp_path)
    src = tmp_path / "shot.png"
    Image.new("RGB", (400, 600), (220, 180, 160)).save(src)
    data = thumb_bytes(src)
    assert data
    assert data[:2] == b"\xff\xd8"
    again = thumb_bytes(src)
    assert again == data
