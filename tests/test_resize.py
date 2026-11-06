import time

import pytest
from helpers import gradient, save_jpeg_with_gps
from PIL import Image

from picture_mcp import ops
from picture_mcp.paths import PictureError


@pytest.mark.parametrize(
    "w,h,width,height,percent,mode,up,expect",
    [
        (4000, 3000, 800, None, None, "fit", False, (800, 600)),
        (4000, 3000, None, 600, None, "fit", False, (800, 600)),
        (4000, 3000, 800, 800, None, "fit", False, (800, 600)),
        (3000, 4000, 800, 800, None, "fit", False, (600, 800)),
        (4000, 3000, None, None, 50, "fit", False, (2000, 1500)),
        (4000, 3000, 500, 500, None, "exact", False, (500, 500)),
        (4000, 3000, 400, 400, None, "fill", False, (400, 400)),
        (100, 50, 400, None, None, "fit", False, (100, 50)),  # never upscale
        (100, 50, 400, None, None, "fit", True, (400, 200)),
        (100, 50, 300, 300, None, "exact", False, (100, 50)),
        (1000, 500, 1, None, None, "fit", False, (1, 1)),
    ],
)
def test_plan_resize_math(w, h, width, height, percent, mode, up, expect):
    _box, size, _notes = ops.plan_resize(w, h, width, height, percent, mode, up)
    assert size == expect


def test_plan_fill_crops_center_with_target_aspect():
    box, size, _ = ops.plan_resize(4000, 3000, 400, 400, None, "fill", False)
    assert size == (400, 400)
    assert box == (500, 0, 3500, 3000)  # the centered 3000x3000 square


def test_plan_fill_small_source_does_not_upscale():
    box, size, notes = ops.plan_resize(100, 50, 400, 400, None, "fill", False)
    assert size == (50, 50) and box == (25, 0, 75, 50)
    assert any("without upscaling" in n for n in notes)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({}, "width and/or height"),
        ({"percent": 150}, "allow_upscale"),
        ({"percent": 50, "width": 10}, "not both"),
        ({"width": 0}, "at least 1"),
        ({"width": 10, "mode": "weird"}, "mode must be"),
    ],
)
def test_plan_resize_errors(kwargs, msg):
    with pytest.raises(PictureError, match=msg):
        ops.plan_resize(
            100, 100, kwargs.get("width"), kwargs.get("height"), kwargs.get("percent"),
            kwargs.get("mode", "fit"), False,
        )


def test_resize_real_file_default_name_and_measured_size(tmp_path):
    src = str(tmp_path / "photo.jpg")
    gradient(400, 200).save(src, quality=90)
    res = ops.resize(src, width=100)
    assert res.output == str(tmp_path / "photo.resized.jpg")
    with Image.open(res.output) as im:
        assert im.size == (100, 50)
    with Image.open(src) as im:
        assert im.size == (400, 200)  # original untouched


def test_resize_fill_and_exact_pixels(tmp_path):
    src = str(tmp_path / "g.png")
    gradient(300, 100).save(src)
    res = ops.resize(src, width=50, height=50, mode="fill")
    with Image.open(res.output) as im:
        assert im.size == (50, 50)
        # a centered crop of a left-to-right red gradient: left edge is darker than right
        assert im.getpixel((2, 25))[0] < im.getpixel((47, 25))[0]
    res = ops.resize(src, width=50, height=50, mode="exact", output=str(tmp_path / "e.png"))
    with Image.open(res.output) as im:
        assert im.size == (50, 50)


def test_resize_to_other_format_via_output_extension(tmp_path):
    src = str(tmp_path / "g.png")
    gradient(100, 100).save(src)
    res = ops.resize(src, percent=50, output=str(tmp_path / "small.webp"))
    with Image.open(res.output) as im:
        assert im.format == "WEBP" and im.size == (50, 50)


def test_resize_keeps_gps_by_default_and_warns_and_can_drop_it(tmp_path):
    src = save_jpeg_with_gps(str(tmp_path / "p.jpg"))
    res = ops.resize(src, width=100)
    assert "still contains the GPS" in res.text
    with Image.open(res.output) as im:
        assert im.getexif().get_ifd(0x8825)
    res2 = ops.resize(src, width=100, keep_metadata=False, output=str(tmp_path / "n.jpg"))
    assert "GPS" not in res2.text
    with Image.open(res2.output) as im:
        assert not im.getexif().get_ifd(0x8825)


def test_resize_20mp_is_fast_enough(tmp_path):
    src = str(tmp_path / "big.jpg")
    w, h = 5472, 3648  # 19.96 MP
    gradient(w, h).save(src, quality=85)
    t = time.perf_counter()
    res = ops.resize(src, width=1600)
    elapsed = time.perf_counter() - t
    with Image.open(res.output) as im:
        assert im.size == (1600, 1067)
    assert elapsed < 30, f"20 MP resize took {elapsed:.1f}s"
