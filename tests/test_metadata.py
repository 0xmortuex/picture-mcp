import os

import pytest
from helpers import exif_with_gps, gradient, save_jpeg_with_gps, with_text
from PIL import Image

from picture_mcp import ops
from picture_mcp.paths import PictureError


def _gps(path):
    with Image.open(path) as im:
        return im.getexif().get_ifd(0x8825)


def _exif_names(path):
    with Image.open(path) as im:
        return dict(im.getexif())


def test_fixture_really_has_gps(tmp_path):
    p = save_jpeg_with_gps(str(tmp_path / "p.jpg"))
    assert _gps(p)[1] == "N"


def test_strip_jpeg_removes_gps_and_exif_losslessly(tmp_path):
    src = save_jpeg_with_gps(str(tmp_path / "p.jpg"))
    res = ops.strip_metadata(src)
    assert res.output == str(tmp_path / "p.stripped.jpg")
    assert not _gps(res.output)
    assert _exif_names(res.output) == {}
    # reports what it removed
    assert "GPS location (41.025" in res.text and "29.00000" in res.text
    assert "Make" in res.text and "Model" in res.text
    assert "GPS none" in res.text
    # the raw bytes carry no EXIF segment or GPS strings at all
    raw = open(res.output, "rb").read()
    assert b"Exif" not in raw and b"Acme" not in raw
    # lossless: decoded pixels are identical to the original's
    with Image.open(src) as a, Image.open(res.output) as b:
        assert a.size == b.size
        assert a.tobytes() == b.tobytes()


def test_strip_keeps_orientation_only_by_default(tmp_path):
    src = save_jpeg_with_gps(str(tmp_path / "r.jpg"), 200, 100, orientation=6)
    res = ops.strip_metadata(src)
    assert not _gps(res.output)
    assert _exif_names(res.output) == {0x0112: 6}
    assert "Orientation tag (6)" in res.text
    with Image.open(res.output) as im:
        assert im.size == (200, 100)  # stored size unchanged (lossless)


def test_strip_can_bake_orientation_instead(tmp_path):
    src = save_jpeg_with_gps(str(tmp_path / "r.jpg"), 200, 100, orientation=6)
    res = ops.strip_metadata(src, keep_orientation=False)
    assert _exif_names(res.output) == {}
    with Image.open(res.output) as im:
        assert im.size == (100, 200)  # rotated into the pixels
    assert "baked into the pixels" in res.text


def test_strip_removes_xmp_comment_and_trailing_data(tmp_path):
    src = str(tmp_path / "x.jpg")
    xmp = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta><rdf>secret-author</rdf></x:xmpmeta>"
    with_text(gradient(120, 80)).save(src, exif=exif_with_gps(), comment=b"shot by Bob", quality=90)
    raw = bytearray(open(src, "rb").read())
    app1 = b"\xff\xe1" + (len(xmp) + 2).to_bytes(2, "big") + xmp
    raw[2:2] = app1  # insert an XMP APP1 right after SOI
    raw += b"HIDDEN-TRAILER-GPS"
    open(src, "wb").write(raw)
    res = ops.strip_metadata(src)
    out = open(res.output, "rb").read()
    for needle in (b"secret-author", b"shot by Bob", b"HIDDEN-TRAILER", b"Acme", b"xmpmeta"):
        assert needle not in out
    assert "XMP packet" in res.text and "comment" in res.text and "trailing data" in res.text
    with Image.open(res.output) as im:
        im.load()
        assert im.size == (120, 80)


def test_strip_png_text_chunks_and_exif(tmp_path):
    from PIL.PngImagePlugin import PngInfo

    src = str(tmp_path / "s.png")
    info = PngInfo()
    info.add_text("Author", "Alice")
    gradient(40, 40).save(src, pnginfo=info, exif=exif_with_gps())
    with Image.open(src) as im:
        assert im.text["Author"] == "Alice"
    res = ops.strip_metadata(src)
    raw = open(res.output, "rb").read()
    assert b"Alice" not in raw and b"Acme" not in raw
    assert not _gps(res.output)
    with Image.open(src) as a, Image.open(res.output) as b:
        assert a.tobytes() == b.tobytes()  # PNG is lossless


def test_strip_webp(tmp_path):
    src = str(tmp_path / "s.webp")
    gradient(40, 40).save(src, exif=exif_with_gps())
    assert _gps(src)
    res = ops.strip_metadata(src)
    assert not _gps(res.output)
    assert b"Acme" not in open(res.output, "rb").read()


def test_strip_clean_file_says_so_and_truncated_is_actionable(tmp_path):
    clean = str(tmp_path / "c.jpg")
    gradient(30, 30).save(clean)
    assert "nothing" in ops.strip_metadata(clean).text
    bad = str(tmp_path / "t.jpg")
    data = open(clean, "rb").read()
    open(bad, "wb").write(data[: len(data) // 2])
    with pytest.raises(PictureError, match="truncated|not a readable image|corrupt"):
        ops.strip_metadata(bad)
    assert not os.path.exists(str(tmp_path / "t.stripped.jpg"))


def test_info_reports_gps_and_orientation(tmp_path):
    src = save_jpeg_with_gps(str(tmp_path / "p.jpg"), 200, 100, orientation=6)
    text = ops.image_info(src).text
    assert "GPS: YES - 41.02500, 29.00000" in text
    assert "orientation: 6" in text
    assert "Make=Acme Cameras" in text
    assert "200x100" in text


def test_info_animated_and_clean(tmp_path):
    from helpers import animated_gif

    gif = animated_gif(str(tmp_path / "a.gif"), frames=4, duration=100)
    text = ops.image_info(gif).text
    assert "animated: yes, 4 frames, 0.40 s" in text
    assert "GPS: none" in text


def test_info_flags_truncated_file(tmp_path):
    src = str(tmp_path / "t.png")
    gradient(300, 300).save(src)
    data = open(src, "rb").read()
    open(src, "wb").write(data[: len(data) * 2 // 3])
    assert "WARNING" in ops.image_info(src).text
