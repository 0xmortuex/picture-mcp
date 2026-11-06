import pytest
from helpers import animated_gif, exif_with_gps, gradient, save_jpeg_with_gps, with_text
from PIL import Image

from picture_mcp import ops
from picture_mcp.paths import PictureError


def _marked(path, orientation):
    """200x100 image: red square top-left, EXIF orientation set."""
    im = Image.new("RGB", (200, 100), (255, 255, 255))
    im.paste((255, 0, 0), (0, 0, 20, 20))
    ex = exif_with_gps(orientation)
    im.save(path, exif=ex, quality=100, subsampling=0)
    return path


def test_exif_orientation_is_honoured_by_every_geometry_tool(tmp_path):
    src = _marked(str(tmp_path / "o.jpg"), 6)  # stored 200x100, displays as 100x200
    res = ops.resize(src, percent=50)
    with Image.open(res.output) as im:
        assert im.size == (50, 100)
        assert not im.getexif().get(0x0112)  # tag removed: pixels were rotated for real
    res = ops.crop(src, box=[0, 0, 100, 50], output=str(tmp_path / "c.jpg"))
    with Image.open(res.output) as im:
        assert im.size == (100, 50)
    # orientation 6 = rotate 90 clockwise to display: the top-left red square lands top-right
    with Image.open(ops.resize(src, percent=100, output=str(tmp_path / "same.jpg")).output) as im:
        assert im.size == (100, 200)
        r, g, b = im.getpixel((90, 10))
        assert r > 200 and g < 60 and b < 60


def test_rotate_flip_auto_orient_only(tmp_path):
    src = _marked(str(tmp_path / "o.jpg"), 6)
    res = ops.rotate_flip(src)
    assert "EXIF orientation was applied" in res.text or "applied" in res.text
    with Image.open(res.output) as im:
        assert im.size == (100, 200)
        assert im.getexif().get(0x0112) in (None, 1)


def test_rotate_90_is_clockwise_and_flip_works(tmp_path):
    src = str(tmp_path / "m.png")
    im = Image.new("RGB", (60, 30), (255, 255, 255))
    im.paste((255, 0, 0), (0, 0, 10, 10))  # top-left red
    im.save(src)
    res = ops.rotate_flip(src, degrees=90)
    with Image.open(res.output) as out:
        assert out.size == (30, 60)
        assert out.getpixel((25, 5))[1] < 60  # clockwise: top-left -> top-right
    res = ops.rotate_flip(src, flip="horizontal", output=str(tmp_path / "h.png"))
    with Image.open(res.output) as out:
        assert out.getpixel((55, 5))[1] < 60 and out.getpixel((5, 5))[1] > 200
    res = ops.rotate_flip(src, flip="vertical", output=str(tmp_path / "v.png"))
    with Image.open(res.output) as out:
        assert out.getpixel((5, 25))[1] < 60


def test_rotate_arbitrary_angle_enlarges_with_transparency(tmp_path):
    src = str(tmp_path / "m.png")
    gradient(100, 100).save(src)
    res = ops.rotate_flip(src, degrees=45)
    with Image.open(res.output) as out:
        assert out.size[0] > 130 and out.mode == "RGBA"
        assert out.getpixel((0, 0))[3] == 0


def test_rotate_animated_gif_keeps_frames(tmp_path):
    src = animated_gif(str(tmp_path / "a.gif"), frames=3)
    res = ops.rotate_flip(src, degrees=180)
    with Image.open(res.output) as out:
        assert out.n_frames == 3


def test_crop_modes(tmp_path):
    src = str(tmp_path / "g.png")
    gradient(400, 200).save(src)
    with Image.open(ops.crop(src, box=[10, 20, 110, 70], output=str(tmp_path / "b.png")).output) as im:
        assert im.size == (100, 50)
    with Image.open(ops.crop(src, aspect="1:1", output=str(tmp_path / "a.png")).output) as im:
        assert im.size == (200, 200)
        assert im.getpixel((0, 100))[0] > 60  # centered: starts a quarter of the way along red
    with Image.open(
        ops.crop(src, aspect="1:1", gravity="left", output=str(tmp_path / "l.png")).output
    ) as im:
        assert im.getpixel((0, 100))[0] < 5  # left gravity: starts at the darkest red
    with Image.open(
        ops.crop(src, width=50, height=40, gravity="bottom-right", output=str(tmp_path / "w.png")).output
    ) as im:
        assert im.size == (50, 40) and im.getpixel((49, 39))[0] > 240
    with Image.open(ops.crop(src, aspect="16:9", output=str(tmp_path / "s.png")).output) as im:
        assert im.size == (400, 225) or im.size == (356, 200)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({}, "exactly one crop mode"),
        ({"box": [0, 0, 500, 100]}, "does not fit"),
        ({"box": [10, 10, 5, 50]}, "does not fit"),
        ({"box": [1, 2, 3]}, "4 numbers"),
        ({"aspect": "banana"}, "cannot read aspect"),
        ({"width": 1000, "height": 10}, "does not fit"),
        ({"aspect": "1:1", "box": [0, 0, 5, 5]}, "exactly one"),
        ({"aspect": "1:1", "gravity": "sideways"}, "gravity must be"),
    ],
)
def test_crop_errors_are_explicit(tmp_path, kwargs, msg):
    src = str(tmp_path / "g.png")
    gradient(400, 200).save(src)
    with pytest.raises(PictureError, match=msg):
        ops.crop(src, **kwargs)


def test_contact_sheet_layout_labels_and_unreadable_tile(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, c in enumerate(colours):
        Image.new("RGB", (300, 150), c).save(folder / f"img{i}.png")
    (folder / "broken.jpg").write_bytes(b"not really a jpeg")
    (folder / "notes.txt").write_text("ignore me")
    res = ops.contact_sheet([str(folder)], thumb_size=100, columns=3)
    assert "5 images in 3 columns x 2 rows" in res.text
    assert "UNREADABLE broken.jpg" in res.text
    assert res.preview is not None and res.preview[:2] == b"\xff\xd8"
    with Image.open(res.output) as sheet:
        cell_w, cell_h = 100 + 16, 100 + max(14, 100 // 9) + 16
        assert sheet.size == (3 * cell_w, 2 * cell_h)
        # sorted order: broken.jpg, img0..img3 -> img0 (red) is the 2nd cell; sample its center
        x, y = 1 * cell_w + 8 + 50, 8 + 50
        r, g, b = sheet.getpixel((x, y))
        assert r > 200 and g < 60 and b < 60
        # label text drew dark pixels under the thumbnail
        label_box = sheet.crop((cell_w + 8, 8 + 100, cell_w + 108, 8 + 100 + 14)).convert("L")
        assert label_box.getextrema()[0] < 120
        # the broken file got a reddish error tile
        r, g, b = sheet.getpixel((8 + 90, 8 + 90))
        assert r > 200 and g < 240 and b < 240 and r > g


def test_contact_sheet_glob_limit_and_empty(tmp_path):
    for i in range(3):
        gradient(30, 30).save(tmp_path / f"p{i}.png")
    res = ops.contact_sheet([str(tmp_path / "p*.png")], preview=False)
    assert res.preview is None and "3 images" in res.text
    with pytest.raises(PictureError, match="more than max_images=2"):
        ops.contact_sheet([str(tmp_path / "p*.png")], max_images=2, output=str(tmp_path / "x.jpg"))
    with pytest.raises(PictureError, match="no image files matched"):
        ops.contact_sheet([str(tmp_path / "zzz*.png")])


def test_make_icons_favicon_set_measured(tmp_path):
    src = str(tmp_path / "logo.png")
    with_text(gradient(600, 400)).save(src)
    res = ops.make_icons(src)
    folder = tmp_path / "logo-favicons"
    expect = {
        "favicon-16x16.png": 16, "favicon-32x32.png": 32, "apple-touch-icon.png": 180,
        "icon-192.png": 192, "icon-512.png": 512,
    }
    for name, side in expect.items():
        with Image.open(folder / name) as im:
            assert im.size == (side, side), name
    with Image.open(folder / "apple-touch-icon.png") as im:
        assert im.mode == "RGB"  # flattened: iOS would paint transparency black
    with Image.open(folder / "favicon.ico") as ico:
        assert ico.info["sizes"] == {(16, 16), (32, 32), (48, 48)}
    assert '<link rel="apple-touch-icon"' in res.text
    assert "not square: padded" in res.text
    with pytest.raises(PictureError, match="already exist"):
        ops.make_icons(src)


def test_make_ico_sizes_and_small_source_warning(tmp_path):
    src = str(tmp_path / "tiny.png")
    gradient(64, 64).save(src)
    res = ops.make_icons(src, kind="ico", sizes=[16, 32, 64], preview=True)
    with Image.open(res.output) as ico:
        assert ico.info["sizes"] == {(16, 16), (32, 32), (64, 64)}
    assert res.preview is not None
    big = ops.make_icons(src, kind="ico", output=str(tmp_path / "big.ico"))
    assert "upscaled" in big.text
    with pytest.raises(PictureError, match="between 8 and 256"):
        ops.make_icons(src, kind="ico", sizes=[512], output=str(tmp_path / "z.ico"))


def test_palette_finds_dominant_colours(tmp_path):
    src = str(tmp_path / "two.png")
    im = Image.new("RGB", (100, 100), (0, 0, 255))
    im.paste((255, 0, 0), (0, 0, 100, 25))  # 25% red, 75% blue
    im.save(src)
    res = ops.palette(src, count=2, preview=True)
    lines = [ln for ln in res.text.splitlines() if ln.strip().startswith("#")]
    assert lines[0].split()[0] == "#0000ff" and "75.0%" in lines[0]
    assert lines[1].split()[0] == "#ff0000" and "25.0%" in lines[1]
    assert res.preview is not None


def test_palette_ignores_transparent_pixels(tmp_path):
    src = str(tmp_path / "t.png")
    im = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    im.paste((0, 255, 0, 255), (0, 0, 10, 50))
    im.save(src)
    res = ops.palette(src, count=1)
    assert "#00ff00" in res.text and "100.0%" in res.text
    empty = str(tmp_path / "e.png")
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(empty)
    with pytest.raises(PictureError, match="fully transparent"):
        ops.palette(empty)


def test_preview_is_a_decodable_small_jpeg(tmp_path):
    import io

    src = save_jpeg_with_gps(str(tmp_path / "p.jpg"), 1000, 800)
    res = ops.resize(src, width=800, preview=True)
    with Image.open(io.BytesIO(res.preview)) as im:
        assert im.format == "JPEG" and max(im.size) <= 384
