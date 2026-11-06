import os

import pytest
from helpers import gradient, save_jpeg_with_gps
from PIL import Image

from picture_mcp import imaging, ops, paths
from picture_mcp.paths import PictureError


def test_never_overwrites_without_flag(tmp_path):
    src = str(tmp_path / "a.png")
    gradient(40, 40).save(src)
    first = ops.resize(src, width=20)
    stamp = os.path.getmtime(first.output)
    with pytest.raises(PictureError, match="already exists.*overwrite=true"):
        ops.resize(src, width=10)
    assert os.path.getmtime(first.output) == stamp
    with Image.open(first.output) as im:
        assert im.width == 20
    ops.resize(src, width=10, overwrite=True)
    with Image.open(first.output) as im:
        assert im.width == 10


def test_output_equal_to_input_needs_overwrite(tmp_path):
    src = str(tmp_path / "a.png")
    gradient(40, 40).save(src)
    with pytest.raises(PictureError, match="already exists"):
        ops.resize(src, width=20, output=src)
    ops.resize(src, width=20, output=src, overwrite=True)
    with Image.open(src) as im:
        assert im.width == 20


def test_failed_encode_leaves_existing_file_and_no_temp_files(tmp_path, monkeypatch):
    src = str(tmp_path / "a.png")
    gradient(40, 40).save(src)
    out = tmp_path / "a.resized.png"
    out.write_bytes(b"ORIGINAL")

    def boom(*a, **k):
        raise PictureError("encoder exploded")

    monkeypatch.setattr(imaging, "encode", boom)
    monkeypatch.setattr(ops, "encode", boom)
    with pytest.raises(PictureError, match="exploded"):
        ops.resize(src, width=20, overwrite=True)
    assert out.read_bytes() == b"ORIGINAL"
    assert sorted(os.listdir(tmp_path)) == ["a.png", "a.resized.png"]


def test_atomic_target_cleans_up_on_error(tmp_path):
    final = str(tmp_path / "x.bin")
    with pytest.raises(RuntimeError), paths.atomic_target(final) as tmp:
        open(tmp, "wb").write(b"partial")
        raise RuntimeError("crash mid-write")
    assert os.listdir(tmp_path) == []


def test_sandbox_blocks_traversal_and_outside_paths(tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    secret = tmp_path / "secret.png"
    gradient(10, 10).save(secret)
    inside = allowed / "in.png"
    gradient(10, 10).save(inside)
    paths.configure([str(allowed)])
    ops.resize(str(inside), width=5)
    with pytest.raises(PictureError, match="outside the directories"):
        ops.resize(str(secret), width=5)
    with pytest.raises(PictureError, match="outside the directories"):
        ops.resize(str(allowed / ".." / "secret.png"), width=5)
    with pytest.raises(PictureError, match="outside the directories"):
        ops.resize(str(inside), width=5, output=str(tmp_path / "leak.png"))
    assert not (tmp_path / "leak.png").exists()
    # a sibling directory sharing the prefix must not count as inside
    sibling = tmp_path / "ok-evil"
    sibling.mkdir()
    gradient(10, 10).save(sibling / "x.png")
    with pytest.raises(PictureError, match="outside the directories"):
        ops.image_info(str(sibling / "x.png"))


def test_sandbox_blocks_symlink_escape(tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    secret_dir = tmp_path / "private"
    secret_dir.mkdir()
    gradient(10, 10).save(secret_dir / "s.png")
    link = allowed / "link.png"
    try:
        os.symlink(secret_dir / "s.png", link)
        dirlink = allowed / "dl"
        os.symlink(secret_dir, dirlink, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this machine")
    paths.configure([str(allowed)])
    with pytest.raises(PictureError, match="outside the directories"):
        ops.image_info(str(link))
    with pytest.raises(PictureError, match="outside the directories"):
        ops.image_info(str(dirlink / "s.png"))
    # writing through a symlink that points outside is refused too
    gradient(10, 10).save(tmp_path / "ok" / "real.png")
    with pytest.raises(PictureError, match="outside the directories"):
        ops.resize(str(allowed / "real.png"), width=5, output=str(link), overwrite=True)


def test_batch_glob_does_not_follow_symlinks_out(tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    gradient(10, 10).save(outside / "s.png")
    gradient(10, 10).save(allowed / "a.png")
    try:
        os.symlink(outside / "s.png", allowed / "evil.png")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this machine")
    paths.configure([str(allowed)])
    res = ops.batch("resize", str(allowed / "*.png"), {"width": 5}, dry_run=False)
    assert "ignored 1 files that resolve outside" in res.text
    assert not (outside / "s.resized.png").exists()


def test_decompression_bomb_guard_is_explicit_and_overridable(tmp_path):
    src = str(tmp_path / "big.png")
    Image.new("RGB", (2000, 1500), (10, 20, 30)).save(src)  # 3 MP, tiny on disk
    paths.configure([], 1_000_000)
    with pytest.raises(PictureError, match=r"3 megapixels.*bomb.*allow_huge=true"):
        ops.resize(src, width=100)
    info = ops.image_info(src).text  # header-only: never decodes the pixels
    assert "2000x1500" in info and "safety limit" in info
    res = ops.resize(src, width=100, allow_huge=True)
    with Image.open(res.output) as im:
        assert im.size == (100, 75)


def test_truncated_and_corrupt_and_non_image_errors_are_actionable(tmp_path):
    good = str(tmp_path / "g.jpg")
    gradient(300, 300).save(good)
    data = open(good, "rb").read()
    trunc = str(tmp_path / "t.jpg")
    open(trunc, "wb").write(data[: len(data) // 2])
    with pytest.raises(PictureError, match="corrupt or truncated.*nothing was written"):
        ops.resize(trunc, width=50)
    assert not os.path.exists(str(tmp_path / "t.resized.jpg"))
    junk = str(tmp_path / "j.png")
    open(junk, "wb").write(b"this is text, not an image")
    with pytest.raises(PictureError, match="not an image this server can read"):
        ops.resize(junk, width=50)
    with pytest.raises(PictureError, match="file not found"):
        ops.resize(str(tmp_path / "missing.png"), width=50)
    with pytest.raises(PictureError, match="not a file"):
        ops.resize(str(tmp_path), width=50)
    with pytest.raises(PictureError, match="quality must be"):
        ops.resize(good, width=50, quality=500)


def test_heic_without_plugin_explains_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(imaging, "HEIC", False)
    fake = str(tmp_path / "x.heic")
    open(fake, "wb").write(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 20)
    with pytest.raises(PictureError, match=r"picture-mcp\[heic\]"):
        ops.resize(fake, width=10)
    with pytest.raises(PictureError, match=r"picture-mcp\[heic\]"):
        imaging.format_for("heic")


@pytest.mark.skipif(not imaging.HEIC, reason="pillow-heif not installed")
def test_heic_roundtrip_when_extra_installed(tmp_path):
    src = str(tmp_path / "p.png")
    gradient(64, 48).save(src)
    heic = ops.convert(src, "heic").output
    assert heic.endswith(".heic")
    back = ops.convert(heic, "jpg")
    with Image.open(back.output) as im:
        assert im.size == (64, 48)


# ------------------------------------------------------------------------ batch


def _make_photos(folder, n=4):
    folder.mkdir(exist_ok=True)
    for i in range(n):
        save_jpeg_with_gps(str(folder / f"p{i}.jpg"), 400, 300)


def test_batch_dry_run_writes_nothing_and_lists_plan(tmp_path):
    _make_photos(tmp_path)
    before = sorted(os.listdir(tmp_path))
    res = ops.batch("resize", str(tmp_path / "*.jpg"), {"width": 100})
    assert res.text.startswith("DRY RUN: resize on 4 files")
    assert "p0.resized.jpg" in res.text and "would write" in res.text
    assert sorted(os.listdir(tmp_path)) == before


def test_batch_real_run_table_and_rerun_skips_outputs(tmp_path):
    _make_photos(tmp_path)
    res = ops.batch("resize", str(tmp_path / "*.jpg"), {"width": 100}, dry_run=False)
    assert "resize: 4 ok, 0 failed of 4 files" in res.text
    assert "400x300 -> 100x75" in res.text
    for i in range(4):
        with Image.open(tmp_path / f"p{i}.resized.jpg") as im:
            assert im.size == (100, 75)
    again = ops.batch("resize", str(tmp_path / "*.jpg"), {"width": 100}, dry_run=False, overwrite=True)
    assert "skipped 4 files that look like outputs of earlier runs" in again.text
    assert not any(".resized.resized" in f for f in os.listdir(tmp_path))


def test_batch_skips_outputs_of_other_operations_too(tmp_path):
    _make_photos(tmp_path, 1)
    ops.strip_metadata(str(tmp_path / "p0.jpg"))
    res = ops.batch("resize", str(tmp_path / "*.jpg"), {"width": 50})
    assert "on 1 files" in res.text and "skipped 1 files" in res.text


def test_batch_one_bad_file_is_reported_not_fatal(tmp_path):
    _make_photos(tmp_path, 2)
    (tmp_path / "bad.jpg").write_bytes(b"garbage")
    res = ops.batch("strip_metadata", str(tmp_path / "*.jpg"), dry_run=False)
    assert "2 ok, 1 failed of 3 files" in res.text
    assert "bad.jpg" in res.text and "FAILED" in res.text
    for i in range(2):
        with Image.open(tmp_path / f"p{i}.stripped.jpg") as im:
            assert not im.getexif().get_ifd(0x8825)


def test_batch_all_failed_raises(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"garbage")
    with pytest.raises(PictureError, match="0 ok, 1 failed"):
        ops.batch("resize", str(tmp_path / "*.jpg"), {"width": 5}, dry_run=False)


def test_batch_output_dir_convert_and_dry_run_conflicts(tmp_path):
    _make_photos(tmp_path, 2)
    out = tmp_path / "web"
    res = ops.batch("convert", str(tmp_path / "*.jpg"), {"format": "webp"}, str(out), dry_run=False)
    assert "2 ok" in res.text
    with Image.open(out / "p1.webp") as im:
        assert im.format == "WEBP"
    dry = ops.batch("convert", str(tmp_path / "*.jpg"), {"format": "webp"}, str(out))
    assert "WOULD FAIL" in dry.text and "already exists" in dry.text


def test_batch_validates_params_and_operation(tmp_path):
    _make_photos(tmp_path, 1)
    pat = str(tmp_path / "*.jpg")
    with pytest.raises(PictureError, match="operation must be one of"):
        ops.batch("explode", pat)
    with pytest.raises(PictureError, match="params not valid for resize: wdith"):
        ops.batch("resize", pat, {"wdith": 5})
    with pytest.raises(PictureError, match="params not valid for resize: output"):
        ops.batch("resize", pat, {"width": 5, "output": "x.jpg"})
    with pytest.raises(PictureError, match="needs params"):
        ops.batch("convert", pat)
    with pytest.raises(PictureError, match="no image files"):
        ops.batch("resize", str(tmp_path / "*.zzz"), {"width": 5})
    with pytest.raises(PictureError, match="max_files=2"):
        _make_photos(tmp_path, 4)
        ops.batch("resize", pat, {"width": 5}, max_files=2)


def test_batch_compress_target(tmp_path):
    from helpers import noisy

    for i in range(2):
        noisy(800, 600).save(tmp_path / f"n{i}.jpg", quality=95)
    res = ops.batch(
        "compress", str(tmp_path / "n*.jpg"), {"target_size": "60KB", "min_quality": 1}, dry_run=False
    )
    assert "2 ok" in res.text
    for i in range(2):
        assert os.path.getsize(tmp_path / f"n{i}.compressed.jpg") <= 60 * 1024


def test_batch_recursive_glob(tmp_path):
    (tmp_path / "sub").mkdir()
    gradient(30, 30).save(tmp_path / "a.png")
    gradient(30, 30).save(tmp_path / "sub" / "b.png")
    res = ops.batch("resize", str(tmp_path / "**" / "*.png"), {"width": 10})
    assert "on 2 files" in res.text
