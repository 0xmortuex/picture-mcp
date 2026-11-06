import os

import pytest
from helpers import gradient, noisy
from PIL import Image

from picture_mcp import ops
from picture_mcp.paths import PictureError, parse_size


@pytest.fixture
def photo(tmp_path):
    p = str(tmp_path / "photo.jpg")
    noisy(1200, 800).save(p, quality=95)
    return p


@pytest.mark.parametrize(
    "text,expected",
    [("200KB", 204800), ("1.5MB", 1572864), ("350000", 350000), ("10 kb", 10240), ("2KiB", 2048)],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


def test_parse_size_error():
    with pytest.raises(PictureError, match="cannot read target size"):
        parse_size("lots")


@pytest.mark.parametrize("target", ["150KB", "80KB", "300KB"])
def test_target_size_is_met_and_measured_on_disk(photo, target, tmp_path):
    assert os.path.getsize(photo) > parse_size("300KB")
    res = ops.compress(photo, target_size=target, min_quality=1)
    assert os.path.getsize(res.output) <= parse_size(target)
    # and the search is not wasteful: it used most of the budget, not a tiny file
    assert os.path.getsize(res.output) > parse_size(target) * 0.5
    with Image.open(res.output) as im:
        assert im.size == (1200, 800)


def test_target_larger_than_source_still_reports_ok(photo):
    res = ops.compress(photo, target_size="50MB")
    assert os.path.getsize(res.output) <= 50 * 1024 * 1024
    assert "quality 95" in res.text


def test_unreachable_target_errors_with_options_then_downscale_works(photo):
    with pytest.raises(PictureError, match="cannot reach 5.0 KB.*allow_downscale=true"):
        ops.compress(photo, target_size="5KB")
    res = ops.compress(photo, target_size="5KB", allow_downscale=True)
    assert os.path.getsize(res.output) <= 5 * 1024
    with Image.open(res.output) as im:
        assert im.width < 1200
    assert "downscaled to" in res.text


def test_preset_and_quality_ordering(photo, tmp_path):
    sizes = {}
    for preset in ("high", "balanced", "small"):
        res = ops.compress(photo, preset=preset, output=str(tmp_path / f"{preset}.jpg"))
        sizes[preset] = os.path.getsize(res.output)
    assert sizes["high"] > sizes["balanced"] > sizes["small"]
    res = ops.compress(photo, quality=10, output=str(tmp_path / "q10.jpg"))
    assert os.path.getsize(res.output) < sizes["small"]


def test_png_to_webp_while_compressing(tmp_path):
    src = str(tmp_path / "shot.png")
    noisy(600, 400).save(src)
    res = ops.compress(src, format="webp", target_size="60KB", min_quality=1)
    assert res.output.endswith(".webp")
    assert os.path.getsize(res.output) <= 60 * 1024
    with Image.open(res.output) as im:
        assert im.format == "WEBP"


def test_png_lossless_keeps_pixels_and_lossy_png_reaches_target(tmp_path):
    src = str(tmp_path / "g.png")
    gradient(300, 300).save(src, compress_level=0)
    res = ops.compress(src)
    assert os.path.getsize(res.output) < os.path.getsize(src)
    with Image.open(src) as a, Image.open(res.output) as b:
        assert a.tobytes() == b.tobytes()

    big = str(tmp_path / "n.png")
    noisy(400, 300).save(big)
    target = os.path.getsize(big) // 3
    with pytest.raises(PictureError, match="lossy_png=true"):
        ops.compress(big, target_size=str(target), output=str(tmp_path / "x.png"))
    res = ops.compress(big, target_size=str(target), lossy_png=True, output=str(tmp_path / "y.png"))
    assert os.path.getsize(res.output) <= target


def test_gif_source_needs_explicit_format(tmp_path):
    src = str(tmp_path / "a.gif")
    gradient(30, 30).save(src)
    with pytest.raises(PictureError, match="format='webp'"):
        ops.compress(src)


def test_compress_warns_when_not_smaller(tmp_path):
    src = str(tmp_path / "tiny.jpg")
    gradient(64, 64).save(src, quality=10)
    res = ops.compress(src, quality=95)
    assert "not smaller than the input" in res.text
