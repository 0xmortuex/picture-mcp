import pytest
from helpers import animated_gif, gradient
from PIL import Image, features

from picture_mcp import ops
from picture_mcp.paths import PictureError


def test_alpha_png_to_jpg_flattens_on_background_colour(tmp_path):
    src = str(tmp_path / "a.png")
    im = Image.new("RGBA", (40, 40), (0, 0, 0, 0))  # fully transparent
    im.paste((255, 0, 0, 255), (0, 0, 20, 40))  # opaque red left half
    im.save(src)
    res = ops.convert(src, "jpg", background="#00ff00")
    assert res.output == str(tmp_path / "a.jpg")
    assert "flattened" in res.text and "#00ff00" in res.text
    with Image.open(res.output) as out:
        assert out.mode == "RGB"
        r, g, b = out.getpixel((30, 20))
        assert g > 200 and r < 60 and b < 60  # transparent area became green
        r, g, b = out.getpixel((5, 20))
        assert r > 200 and g < 60  # opaque red stayed red


@pytest.mark.parametrize("fmt", ["png", "jpg", "webp", "gif", "bmp", "tiff"])
def test_roundtrip_formats_open_and_keep_size(tmp_path, fmt):
    src = str(tmp_path / "g.png")
    gradient(64, 48).save(src)
    res = ops.convert(src, fmt, output=str(tmp_path / ("out." + fmt)))
    with Image.open(res.output) as out:
        assert out.size == (64, 48)
        assert out.format == {"jpg": "JPEG", "tiff": "TIFF"}.get(fmt, fmt.upper())


@pytest.mark.skipif(not features.check("avif"), reason="Pillow built without AVIF")
def test_avif(tmp_path):
    src = str(tmp_path / "g.png")
    gradient(64, 48).save(src)
    res = ops.convert(src, "avif", quality=60)
    with Image.open(res.output) as out:
        assert out.format == "AVIF" and out.size == (64, 48)


def test_avif_error_is_actionable_when_missing(monkeypatch):
    from picture_mcp import imaging

    monkeypatch.setattr(imaging.features, "check", lambda name: False)
    with pytest.raises(PictureError, match="webp instead"):
        imaging.format_for("avif")


def test_unknown_format_lists_choices(tmp_path):
    src = str(tmp_path / "g.png")
    gradient(8, 8).save(src)
    with pytest.raises(PictureError, match="unknown image format 'xyz'.*png"):
        ops.convert(src, "xyz")


def test_palette_png_and_cmyk_inputs(tmp_path):
    p = str(tmp_path / "pal.png")
    gradient(32, 32).quantize(16).save(p)
    with Image.open(p) as chk:
        assert chk.mode == "P"
    res = ops.convert(p, "jpg")
    with Image.open(res.output) as out:
        assert out.mode == "RGB"
    c = str(tmp_path / "cmyk.jpg")
    Image.new("CMYK", (32, 32), (0, 255, 255, 0)).save(c)  # red in CMYK
    with Image.open(c) as chk:
        assert chk.mode == "CMYK"
    res = ops.convert(c, "png")
    assert "CMYK converted" in res.text
    with Image.open(res.output) as out:
        assert out.mode == "RGB"
        r, g, b = out.getpixel((5, 5))
        assert r > 200 and g < 80 and b < 80


def test_animated_gif_to_webp_keeps_frames_and_durations(tmp_path):
    src = animated_gif(str(tmp_path / "a.gif"), frames=5, duration=120)
    res = ops.convert(src, "webp")
    assert "5 animation frames preserved" in res.text
    with Image.open(res.output) as out:
        assert out.n_frames == 5
        for i in range(5):
            out.seek(i)
            out.load()
            assert out.info["duration"] == 120


def test_animated_webp_to_gif_keeps_durations(tmp_path):
    src = animated_gif(str(tmp_path / "a.gif"), frames=3, duration=120)
    webp = ops.convert(src, "webp").output
    res = ops.convert(webp, "gif", output=str(tmp_path / "back.gif"))
    with Image.open(res.output) as out:
        assert out.n_frames == 3
        for i in range(3):
            out.seek(i)
            assert out.info["duration"] == 120


def test_animated_gif_to_jpg_says_it_is_flattened(tmp_path):
    src = animated_gif(str(tmp_path / "a.gif"))
    res = ops.convert(src, "jpg")
    assert "flattened to its first frame" in res.text
    with Image.open(res.output) as out:
        assert getattr(out, "n_frames", 1) == 1


def test_resize_animated_gif_keeps_all_frames(tmp_path):
    src = animated_gif(str(tmp_path / "a.gif"), frames=4, size=40)
    res = ops.resize(src, width=20)
    with Image.open(res.output) as out:
        assert out.n_frames == 4 and out.size == (20, 20)


def test_convert_to_ico_small_and_error_when_large(tmp_path):
    src = str(tmp_path / "s.png")
    gradient(32, 32).save(src)
    res = ops.convert(src, "ico")
    with Image.open(res.output) as out:
        assert out.format == "ICO"
    big = str(tmp_path / "b.png")
    gradient(300, 300).save(big)
    with pytest.raises(PictureError, match="make_icons"):
        ops.convert(big, "ico")


def test_convert_name_does_not_clobber_existing_sibling(tmp_path):
    src = str(tmp_path / "g.png")
    gradient(16, 16).save(src)
    (tmp_path / "g.webp").write_bytes(b"precious")
    res = ops.convert(src, "webp")  # g.webp is taken -> tagged name, not an overwrite
    assert res.output == str(tmp_path / "g.converted.webp")
    assert (tmp_path / "g.webp").read_bytes() == b"precious"
