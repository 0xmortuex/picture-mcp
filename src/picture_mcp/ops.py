"""The image operations. Each returns a Result; the MCP layer in server.py wraps them."""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from . import imaging as im_
from .imaging import (
    FORMAT_EXT,
    Loaded,
    encode,
    flatten,
    format_for,
    open_image,
    parse_color,
    thumbnail_jpeg,
)
from .paths import (
    PictureError,
    check_output,
    default_output,
    expand,
    human,
    parse_size,
    resolve,
    write_bytes,
)

RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
}


@dataclass
class Result:
    text: str
    output: str | None = None
    preview: bytes | None = None
    detail: str = ""  # one line, used in batch tables


# ------------------------------------------------------------------ shared plumbing


def _choose_output(
    src: Loaded, tag: str, output: str | None, overwrite: bool, fmt: str | None = None
) -> tuple[str, str]:
    """Returns (real output path, pillow format)."""
    if output:
        real = check_output(output, overwrite)
        out_fmt = format_for(real) if fmt is None else fmt
        if fmt is not None and im_.EXT_FORMAT.get(os.path.splitext(real)[1].lower()) not in (
            None,
            fmt,
        ):
            raise PictureError(
                f"output {output!r} has an extension that does not match format "
                f"{fmt.lower()}; rename it or drop format="
            )
        return real, out_fmt
    out_fmt = fmt or src.format or "PNG"
    if out_fmt == src.format:
        ext = os.path.splitext(src.path)[1]
        name = default_output(src.path, tag, ext)
    else:
        name = default_output(src.path, "", FORMAT_EXT[out_fmt])
        if os.path.exists(name) and not overwrite:
            name = default_output(src.path, tag, FORMAT_EXT[out_fmt])
    return check_output(name, overwrite), out_fmt


def _meta_notes(src: Loaded, fmt: str, keep_metadata: bool) -> list[str]:
    if keep_metadata and fmt in im_.EXIF_OK and im_.gps_info(src.exif):
        return [
            "the output still contains the GPS location; use strip_metadata or "
            "keep_metadata=false before sharing it"
        ]
    return []


def _finish(
    src: Loaded,
    frames: list[Image.Image],
    out: str,
    fmt: str,
    verb: str,
    *,
    quality: int | None = None,
    keep_metadata: bool = True,
    bg: tuple[int, int, int] = (255, 255, 255),
    lossless: bool = False,
    preview: bool = False,
    notes: list[str] | None = None,
    data: bytes | None = None,
    enc_notes: list[str] | None = None,
) -> Result:
    if data is None:
        data, enc_notes = encode(
            frames, fmt, durations=src.durations, loop=src.loop, quality=quality,
            exif=src.exif if keep_metadata else None, icc=src.icc, dpi=src.dpi, bg=bg,
            lossless=lossless,
        )
    write_bytes(out, data)
    w, h = frames[0].size
    all_notes = list(src.notes) + (notes or []) + (enc_notes or [])
    all_notes += _meta_notes(src, fmt, keep_metadata)
    if src.animated and fmt in im_.ANIMATED_OK:
        all_notes.append(f"{len(frames)} animation frames preserved")
    ratio = f"{len(data) / src.src_bytes * 100:.0f}% of the original" if src.src_bytes else ""
    lines = [
        f"{verb}: {out}",
        f"  {src.src_size[0]}x{src.src_size[1]} {src.format} ({human(src.src_bytes)}) -> "
        f"{w}x{h} {fmt} ({human(len(data))}{', ' + ratio if ratio else ''})",
    ]
    lines += [f"  note: {n}" for n in all_notes]
    detail = (
        f"{src.src_size[0]}x{src.src_size[1]} -> {w}x{h}, {human(src.src_bytes)} -> "
        f"{human(len(data))}"
    )
    return Result(
        "\n".join(lines), out, thumbnail_jpeg(data) if preview else None, detail
    )


def _check_quality(quality: int | None) -> None:
    if quality is not None and not 1 <= quality <= 100:
        raise PictureError("quality must be between 1 and 100")


# ------------------------------------------------------------------------- image_info


def image_info(path: str, preview: bool = False) -> Result:
    real = resolve(path, "path")
    if not os.path.isfile(real):
        raise PictureError(f"file not found: {real}")
    from .paths import CONFIG

    try:
        img = Image.open(real)
    except UnidentifiedImageError:
        raise PictureError(
            f"{os.path.basename(real)} is not an image this server can read "
            "(unrecognised or damaged header)"
        ) from None
    except OSError as e:
        raise PictureError(f"cannot open {os.path.basename(real)}: {e}") from None
    with img:
        fmt = "JPEG" if img.format == "MPO" else img.format
        w, h = img.size
        n_frames = 1 if img.format == "MPO" else getattr(img, "n_frames", 1)
        problem = None
        try:
            exif = img.getexif()
        except (OSError, SyntaxError, ValueError, EOFError) as e:
            exif = Image.Exif()
            problem = str(e)
        info = dict(img.info)
        mode = img.mode
        lines = [f"{real}"]
        lines.append(f"format: {fmt}    size: {w}x{h} ({w * h / 1e6:.1f} MP)    mode: {mode}")
        lines.append(f"file size: {human(os.path.getsize(real))}")
        alpha = mode in ("RGBA", "LA", "PA") or "transparency" in info
        lines.append(f"transparency: {'yes' if alpha else 'no'}")
        if info.get("dpi"):
            lines.append(f"dpi: {float(info['dpi'][0]):g} x {float(info['dpi'][1]):g}")
        decodable = w * h * n_frames <= CONFIG.max_pixels
        if decodable and not problem:
            try:
                total_ms = 0
                for i in range(n_frames):
                    img.seek(i)
                    img.load()
                    total_ms += int(img.info.get("duration", 0) or 0)
            except (OSError, SyntaxError, ValueError, EOFError) as e:
                problem = str(e)
                total_ms = 0
        else:
            total_ms = 0
        if n_frames > 1:
            lines.append(
                f"animated: yes, {n_frames} frames, {total_ms / 1000:.2f} s, "
                f"loop {'forever' if not info.get('loop') else info.get('loop')}"
            )
        else:
            lines.append("animated: no")
        orientation = int(exif.get(im_.ORIENTATION, 1) or 1)
        if orientation != 1:
            lines.append(
                f"EXIF orientation: {orientation} (viewers show it rotated; every tool here "
                "applies it, and rotate_flip degrees=0 bakes it into the pixels)"
            )
        gps = im_.gps_info(exif)
        if gps:
            if gps["latitude"] is not None and gps["longitude"] is not None:
                lines.append(
                    f"GPS: YES - {gps['latitude']:.5f}, {gps['longitude']:.5f} "
                    "(use strip_metadata before sharing)"
                )
            else:
                lines.append("GPS: YES - location tags present (use strip_metadata before sharing)")
        else:
            lines.append("GPS: none")
        items = im_.exif_items(exif)
        shown = [
            f"{t}={im_.short(items[t])}" for t in im_.SUMMARY_TAGS if t in items and t != "Orientation"
        ]
        if shown:
            lines.append("EXIF: " + "; ".join(shown))
            rest = len(items) - len(shown)
            if rest > 0:
                lines.append(f"  (+{rest} more EXIF tags)")
        else:
            lines.append("EXIF: none")
        extras = []
        if info.get("icc_profile"):
            extras.append("ICC profile")
        with open(real, "rb") as fh:
            raw = fh.read()
        if b"<x:xmpmeta" in raw or b"http://ns.adobe.com/xap/1.0/" in raw:
            extras.append("XMP")
        if fmt == "JPEG" and b"Photoshop 3.0\x00" in raw:
            extras.append("IPTC/Photoshop")
        if extras:
            lines.append("other metadata: " + ", ".join(extras))
        if problem:
            lines.append(
                f"WARNING: the file is corrupt or truncated ({problem}); other tools will refuse it"
            )
        elif not decodable:
            lines.append(
                f"not fully decoded: over the {CONFIG.max_pixels / 1e6:.0f} MP safety limit "
                "(pass allow_huge=true to other tools to process it)"
            )
    pv = None
    if preview and not problem and decodable:
        data = open(real, "rb").read()
        pv = thumbnail_jpeg(data)
    return Result("\n".join(lines), None, pv)


# --------------------------------------------------------------------------- resize


def plan_resize(
    w: int,
    h: int,
    width: int | None,
    height: int | None,
    percent: float | None,
    mode: str,
    allow_upscale: bool,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int], list[str]]:
    """Pure resize math: returns (crop box or None, final size, notes)."""
    notes: list[str] = []
    if mode not in ("fit", "fill", "exact"):
        raise PictureError("mode must be 'fit', 'fill' or 'exact'")
    if percent is not None:
        if width is not None or height is not None:
            raise PictureError("give either percent, or width/height - not both")
        if percent <= 0:
            raise PictureError("percent must be positive")
        if percent > 100 and not allow_upscale:
            raise PictureError(
                f"percent={percent:g} would enlarge the image. Pass allow_upscale=true if "
                "you really want that."
            )
        return None, (max(1, round(w * percent / 100)), max(1, round(h * percent / 100))), notes
    if width is None and height is None:
        raise PictureError("give width and/or height (in pixels), or percent")
    for name, v in (("width", width), ("height", height)):
        if v is not None and v < 1:
            raise PictureError(f"{name} must be at least 1 pixel")

    if width is None or height is None:  # one side given: keep the aspect ratio
        if width is not None:
            tw, th = width, max(1, round(h * width / w))
        else:
            assert height is not None
            tw, th = max(1, round(w * height / h)), height
        if (tw > w or th > h) and not allow_upscale:
            notes.append(f"requested size is larger than the source; kept {w}x{h} (no upscaling)")
            return None, (w, h), notes
        return None, (tw, th), notes

    if mode == "exact":
        tw, th = width, height
        if not allow_upscale and (tw > w or th > h):
            tw, th = min(tw, w), min(th, h)
            notes.append(f"no upscaling: exact size capped at {tw}x{th}")
        return None, (tw, th), notes

    if mode == "fit":
        if width / w <= height / h:
            tw, th = width, max(1, round(h * width / w))
        else:
            tw, th = max(1, round(w * height / h)), height
        if (tw > w or th > h) and not allow_upscale:
            notes.append(f"source already fits inside {width}x{height}; kept {w}x{h}")
            return None, (w, h), notes
        return None, (tw, th), notes

    # fill: cover the box, then center-crop to its aspect ratio
    if w / h > width / height:
        ch, cw = h, max(1, min(w, round(h * width / height)))
    else:
        cw, ch = w, max(1, min(h, round(w * height / width)))
    left, top = (w - cw) // 2, (h - ch) // 2
    box = (left, top, left + cw, top + ch)
    if cw >= width and ch >= height:
        return box, (width, height), notes
    if allow_upscale:
        return box, (width, height), notes
    notes.append(
        f"source too small to fill {width}x{height} without upscaling; cropped to "
        f"{cw}x{ch} (same aspect ratio)"
    )
    return box, (cw, ch), notes


def resize(
    path: str,
    width: int | None = None,
    height: int | None = None,
    percent: float | None = None,
    mode: str = "fit",
    allow_upscale: bool = False,
    resample: str = "lanczos",
    output: str | None = None,
    overwrite: bool = False,
    quality: int | None = None,
    keep_metadata: bool = True,
    preview: bool = False,
    allow_huge: bool = False,
) -> Result:
    _check_quality(quality)
    if resample not in RESAMPLE:
        raise PictureError(f"resample must be one of {', '.join(RESAMPLE)}")
    src = open_image(path, allow_huge)
    w, h = src.size
    box, size, notes = plan_resize(w, h, width, height, percent, mode, allow_upscale)
    out, fmt = _choose_output(src, "resized", output, overwrite)
    frames = []
    for f in src.frames:
        g = f.crop(box) if box else f
        if g.size != size:
            g = g.resize(size, RESAMPLE[resample])
        frames.append(g)
    return _finish(
        src, frames, out, fmt, "Resized", quality=quality, keep_metadata=keep_metadata,
        preview=preview, notes=notes,
    )


# --------------------------------------------------------------------------- convert


def convert(
    path: str,
    format: str,
    output: str | None = None,
    overwrite: bool = False,
    quality: int | None = None,
    background: str = "white",
    lossless: bool = False,
    keep_metadata: bool = True,
    preview: bool = False,
    allow_huge: bool = False,
) -> Result:
    _check_quality(quality)
    fmt = format_for(format)
    bg = parse_color(background)
    src = open_image(path, allow_huge)
    out, fmt = _choose_output(src, "converted", output, overwrite, fmt)
    return _finish(
        src, src.frames, out, fmt, "Converted", quality=quality, keep_metadata=keep_metadata,
        bg=bg, lossless=lossless, preview=preview,
    )


# -------------------------------------------------------------------------- compress

PRESETS = {"high": 88, "balanced": 78, "small": 60}
LOSSY = {"JPEG", "WEBP", "AVIF", "HEIF"}


def _quantize(frames: list[Image.Image], colors: int) -> list[Image.Image]:
    out = []
    for f in frames:
        if f.mode in ("RGBA", "LA"):
            q = f.convert("RGBA").quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
        else:
            q = f.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        out.append(q)
    return out


def compress(
    path: str,
    target_size: str | None = None,
    quality: int | None = None,
    preset: str = "balanced",
    format: str | None = None,
    min_quality: int = 30,
    allow_downscale: bool = False,
    lossy_png: bool = False,
    output: str | None = None,
    overwrite: bool = False,
    keep_metadata: bool = True,
    preview: bool = False,
    allow_huge: bool = False,
) -> Result:
    _check_quality(quality)
    _check_quality(min_quality)
    if preset not in PRESETS:
        raise PictureError(f"preset must be one of {', '.join(PRESETS)}")
    target = parse_size(target_size) if target_size else None
    src = open_image(path, allow_huge)
    fmt = format_for(format) if format else src.format
    if fmt not in LOSSY | {"PNG"}:
        raise PictureError(
            f"compress works on jpg, webp, png (and avif); the source is {fmt}. "
            "Pass format='webp' (or jpg/png) to convert while compressing."
        )
    assert fmt is not None
    out, fmt = _choose_output(src, "compressed", output, overwrite, fmt)
    exif = src.exif if keep_metadata else None
    bg = (255, 255, 255)
    notes: list[str] = []

    def enc(frames: list[Image.Image], q: int | None = None) -> tuple[bytes, list[str]]:
        return encode(
            frames, fmt, durations=src.durations, loop=src.loop, quality=q, exif=exif,
            icc=src.icc, dpi=src.dpi, bg=bg,
        )

    def png_ladder(frames: list[Image.Image]) -> list[tuple[bytes, str]]:
        steps = [(enc(frames)[0], "lossless")]
        if lossy_png:
            steps = [] if target is None else steps
            for colors in (256, 128, 64, 32, 16):
                steps.append((enc(_quantize(frames, colors))[0], f"{colors}-color palette"))
                if target is None:
                    break
        return steps

    def search(frames: list[Image.Image]) -> tuple[bytes, str, int | None]:
        """Best result that fits `target`; if none fits, the smallest one (flagged by size)."""
        assert target is not None
        if fmt == "PNG":
            smallest = None
            for data, label in png_ladder(frames):
                if smallest is None or len(data) < len(smallest[0]):
                    smallest = (data, label)
                if len(data) <= target:
                    return data, label, None
            assert smallest is not None
            return smallest[0], smallest[1], None
        hi_q, lo_q = 95, min_quality
        data = enc(frames, hi_q)[0]
        if len(data) <= target:
            return data, f"quality {hi_q}", hi_q
        low = enc(frames, lo_q)[0]
        if len(low) > target:
            return low, f"quality {lo_q}", lo_q
        best, best_q = low, lo_q
        a, b = lo_q, hi_q  # invariant: a fits, b does not
        while b - a > 1:
            mid = (a + b) // 2
            d = enc(frames, mid)[0]
            if len(d) <= target:
                a, best, best_q = mid, d, mid
            else:
                b = mid
        return best, f"quality {best_q}", best_q

    frames = src.frames
    if target is None:
        if fmt == "PNG":
            if quality is not None:
                notes.append("quality is ignored for PNG (lossless); use lossy_png=true to shrink more")
            if lossy_png:
                data = enc(_quantize(frames, 256))[0]
                label = "256-color palette"
            else:
                data, label = enc(frames)[0], "lossless optimize"
        else:
            q = quality or PRESETS[preset]
            data, label = enc(frames, q)[0], f"quality {q}"
    else:
        scale = 1.0
        for _ in range(8):
            cur = frames
            if scale < 1.0:
                new = (max(1, round(frames[0].width * scale)), max(1, round(frames[0].height * scale)))
                cur = [f.resize(new, Image.Resampling.LANCZOS) for f in frames]
            data, label, _q = search(cur)
            if len(data) <= target:
                if scale < 1.0:
                    notes.append(f"downscaled to {cur[0].width}x{cur[0].height} to reach the target")
                frames = cur
                break
            if not allow_downscale:
                raise PictureError(
                    f"cannot reach {human(target)}: the smallest result at full size "
                    f"({label}) is {human(len(data))}. Options: raise the target, lower "
                    "min_quality, pass allow_downscale=true to shrink the dimensions"
                    + (", pass lossy_png=true or format='webp'" if fmt == "PNG" else "")
                    + "."
                )
            scale *= max(0.5, min(0.95, math.sqrt(target / len(data)) * 0.95))
        else:
            raise PictureError(
                f"could not reach {human(target)} even after downscaling to "
                f"{cur[0].width}x{cur[0].height}; choose a larger target"
            )
    data_enc = data
    enc_notes: list[str] = []
    if target is not None and len(data) > target:  # pragma: no cover - guarded above
        raise PictureError("internal: result exceeds target")
    if quality is not None and target is not None:
        notes.append("quality is ignored when target_size is given (it is searched for)")
    notes.append(f"encoded at {label}")
    res = _finish(
        src, frames, out, fmt, "Compressed", quality=None, keep_metadata=keep_metadata,
        preview=preview, notes=notes, data=data_enc, enc_notes=enc_notes,
    )
    if len(data) >= src.src_bytes:
        res.text += (
            "\n  note: the output is not smaller than the input; it was already well "
            "compressed. Try a lower quality, a smaller target_size or format='webp'."
        )
    if target is not None:
        res.text += f"\n  target {human(target)}: result {human(len(data))}"
    return res


# --------------------------------------------------------------------- strip_metadata

_KEEP_APP = {0xE0, 0xEE}  # JFIF, Adobe (color transform: needed to decode CMYK/YCCK)


def _jpeg_strip(data: bytes, keep_orientation: int | None) -> tuple[bytes, dict[str, Any]]:
    """Lossless metadata removal: copy the JPEG segment by segment, dropping the private ones."""
    if data[:2] != b"\xff\xd8":
        raise PictureError("not a JPEG stream")
    out = bytearray(b"\xff\xd8")
    kept: list[bytes] = []
    dropped: list[str] = []
    i = 2
    n = len(data)

    def seg_at(pos: int) -> tuple[int, int]:
        if pos + 4 > n or data[pos] != 0xFF:
            raise PictureError("JPEG is corrupt or truncated (bad segment header)")
        while pos + 1 < n and data[pos + 1] == 0xFF:
            pos += 1
        marker = data[pos + 1]
        length = int.from_bytes(data[pos + 2 : pos + 4], "big")
        return marker, length

    end = -1
    while i < n:
        marker, length = seg_at(i)
        while data[i + 1] == 0xFF:
            i += 1
        seg_end = i + 2 + length
        if marker == 0xDA:  # start of scan: entropy-coded data follows
            j = seg_end
            while True:
                k = data.find(b"\xff", j)
                if k < 0 or k + 1 >= n:
                    raise PictureError("JPEG is truncated (no end-of-image marker)")
                nxt = data[k + 1]
                if nxt == 0xFF:
                    j = k + 1
                elif nxt == 0 or 0xD0 <= nxt <= 0xD7:
                    j = k + 2
                elif nxt == 0xD9:
                    end = k + 2
                    break
                else:  # another table/scan header inside a progressive stream
                    j = k + 2 + int.from_bytes(data[k + 2 : k + 4], "big")
            kept.append(data[i:end])
            break
        payload = data[i + 4 : seg_end]
        if marker == 0xD9:
            raise PictureError("JPEG has no image data")
        keep = False
        if 0xE0 <= marker <= 0xEF:
            if marker in _KEEP_APP:
                keep = True
            elif marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
                keep = True
            else:
                if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
                    dropped.append("EXIF")
                elif marker == 0xE1 and b"ns.adobe.com/xap" in payload[:40]:
                    dropped.append("XMP")
                elif marker == 0xED:
                    dropped.append("IPTC/Photoshop")
                elif marker == 0xE2:
                    dropped.append("MPF")
                else:
                    dropped.append(f"APP{marker - 0xE0}")
        elif marker == 0xFE:
            dropped.append("comment")
        else:
            keep = True
        if keep:
            kept.append(data[i:seg_end])
        i = seg_end
    if end < 0:
        raise PictureError("JPEG is truncated (no image data)")
    trailing = n - end
    if keep_orientation and keep_orientation != 1:
        e = Image.Exif()
        e[im_.ORIENTATION] = keep_orientation
        body = e.tobytes()
        if not body.startswith(b"Exif\x00\x00"):
            body = b"Exif\x00\x00" + body
        app1 = b"\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body
        pos = 1 if kept and kept[0][1] == 0xE0 else 0
        kept.insert(pos, app1)
    for s in kept:
        out += s
    return bytes(out), {"dropped": dropped, "trailing": trailing}


def strip_metadata(
    path: str,
    output: str | None = None,
    overwrite: bool = False,
    keep_orientation: bool = True,
    quality: int = 95,
    preview: bool = False,
    allow_huge: bool = False,
) -> Result:
    _check_quality(quality)
    real = resolve(path, "path")
    if not os.path.isfile(real):
        raise PictureError(f"file not found: {real}")
    with open(real, "rb") as fh:
        raw = fh.read()
    try:
        before = im_.metadata_inventory(raw)
    except (UnidentifiedImageError, OSError) as e:
        raise PictureError(
            f"{os.path.basename(real)} is not a readable image ({e}); nothing was written"
        ) from None
    fmt = before["format"]
    out_name = output or default_output(real, "stripped")
    out = check_output(out_name, overwrite)
    if output and format_for(out) != fmt:
        raise PictureError(
            f"strip_metadata keeps the format ({fmt}); give an output ending in "
            f"{FORMAT_EXT.get(fmt, '')}, or use convert first"
        )
    method = ""
    orientation = int(
        Image.open(io.BytesIO(raw)).getexif().get(im_.ORIENTATION, 1) or 1
    )
    if fmt == "JPEG" and keep_orientation:
        new, info = _jpeg_strip(raw, orientation)
        method = "lossless: pixel data copied byte-for-byte"
        if info["trailing"]:
            info["dropped"].append(f"{info['trailing']} bytes of trailing data after the image")
        removed_extra = info["dropped"]
    else:
        src = open_image(real, allow_huge)
        frames = src.frames
        new, _n = encode(
            frames, fmt, durations=src.durations, loop=src.loop, quality=quality, exif=None,
            icc=src.icc, dpi=src.dpi, lossless=False,
        )
        method = "re-encoded without metadata"
        if fmt in ("JPEG", "WEBP"):
            method += f" at quality {quality}"
        if orientation != 1:
            method += f"; EXIF orientation {orientation} baked into the pixels"
        removed_extra = []
    after = im_.metadata_inventory(new)
    leftover = {k for k in after["exif"] if k != "Orientation"}
    if after["gps"] or leftover or after["xmp"] or after["iptc"] or after["text"]:
        raise PictureError(
            "internal check failed: metadata survived stripping "
            f"(exif={sorted(leftover)}, gps={bool(after['gps'])}, xmp={after['xmp']}, "
            f"iptc={after['iptc']}, text={after['text']}); nothing was written"
        )
    write_bytes(out, new)

    removed = []
    g = before["gps"]
    if g:
        loc = (
            f" ({g['latitude']:.5f}, {g['longitude']:.5f})"
            if g["latitude"] is not None and g["longitude"] is not None
            else ""
        )
        removed.append(f"GPS location{loc}")
    names = [k for k in before["exif"] if k != "Orientation"]
    if names:
        removed.append(f"{len(names)} EXIF tag{'s' if len(names) != 1 else ''}: " + ", ".join(names[:12]) + (", ..." if len(names) > 12 else ""))
    if before["xmp"]:
        removed.append("XMP packet")
    if before["iptc"]:
        removed.append("IPTC/Photoshop block")
    removed += [f"text chunk '{t}'" for t in before["text"]]
    for extra in removed_extra:
        if extra not in ("EXIF", "XMP", "IPTC/Photoshop") and extra not in removed:
            removed.append(extra)
    kept = []
    if before["icc"]:
        kept.append("ICC color profile")
    if after["exif"].get("Orientation") is not None and orientation != 1:
        kept.append(f"Orientation tag ({orientation}) so it still displays upright")
    lines = [
        f"Stripped: {out}",
        f"  {method}; {human(len(raw))} -> {human(len(new))}",
        "  removed: " + ("; ".join(removed) if removed else "nothing - the file had no private metadata"),
        "  kept: " + (", ".join(kept) if kept else "nothing but the pixels"),
        "  verified: re-read the output - "
        f"GPS {'none' if not after['gps'] else 'PRESENT'}, "
        f"EXIF tags {len([k for k in after['exif'] if k != 'Orientation'])}",
    ]
    pv = thumbnail_jpeg(new) if preview else None
    return Result("\n".join(lines), out, pv, "removed: " + (", ".join(removed) or "nothing"))


# ----------------------------------------------------------------------- rotate_flip


def rotate_flip(
    path: str,
    degrees: float = 0,
    flip: str = "none",
    background: str | None = None,
    output: str | None = None,
    overwrite: bool = False,
    quality: int | None = None,
    keep_metadata: bool = True,
    preview: bool = False,
    allow_huge: bool = False,
) -> Result:
    _check_quality(quality)
    if flip not in ("none", "horizontal", "vertical"):
        raise PictureError("flip must be 'none', 'horizontal' or 'vertical'")
    bg = parse_color(background) if background else None
    src = open_image(path, allow_huge)
    out, fmt = _choose_output(src, "rotated", output, overwrite)
    deg = degrees % 360
    notes: list[str] = []
    frames = []
    for f in src.frames:
        g = f
        if deg == 90:
            g = g.transpose(Image.Transpose.ROTATE_270)  # PIL's ROTATE_* are counter-clockwise
        elif deg == 180:
            g = g.transpose(Image.Transpose.ROTATE_180)
        elif deg == 270:
            g = g.transpose(Image.Transpose.ROTATE_90)
        elif deg != 0:
            g = g.convert("RGBA").rotate(
                -deg, resample=Image.Resampling.BICUBIC, expand=True,
                fillcolor=(*bg, 255) if bg else (0, 0, 0, 0),
            )
        if flip == "horizontal":
            g = ImageOps.mirror(g)
        elif flip == "vertical":
            g = ImageOps.flip(g)
        frames.append(g)
    if deg == 0 and flip == "none":
        notes.append(
            "no rotation requested: only the EXIF orientation was applied "
            + ("(it was already upright)" if src.exif_orientation == 1 else "to the pixels")
        )
    elif deg % 90:
        notes.append("angle is not a multiple of 90: canvas enlarged, corners filled")
    return _finish(
        src, frames, out, fmt, "Rotated", quality=quality, keep_metadata=keep_metadata,
        preview=preview, notes=notes,
    )


# ------------------------------------------------------------------------------ crop

GRAVITY = {
    "center": (0.5, 0.5), "top": (0.5, 0.0), "bottom": (0.5, 1.0), "left": (0.0, 0.5),
    "right": (1.0, 0.5), "top-left": (0.0, 0.0), "top-right": (1.0, 0.0),
    "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0),
}


def plan_crop(
    w: int,
    h: int,
    box: list[int] | None,
    aspect: str | None,
    width: int | None,
    height: int | None,
    gravity: str,
) -> tuple[int, int, int, int]:
    modes = [box is not None, aspect is not None, width is not None or height is not None]
    if sum(modes) != 1:
        raise PictureError(
            "pick exactly one crop mode: box=[left, top, right, bottom], aspect='16:9', "
            "or width= and height= (a window of that size)"
        )
    if gravity not in GRAVITY:
        raise PictureError(f"gravity must be one of {', '.join(GRAVITY)}")
    gx, gy = GRAVITY[gravity]
    if box is not None:
        if len(box) != 4:
            raise PictureError("box needs 4 numbers: [left, top, right, bottom]")
        left, top, right, bottom = box
        if not (0 <= left < right <= w and 0 <= top < bottom <= h):
            raise PictureError(
                f"box {box} does not fit inside the {w}x{h} image (after EXIF orientation); "
                "need 0 <= left < right <= width and 0 <= top < bottom <= height"
            )
        return left, top, right, bottom
    if aspect is not None:
        try:
            a, b = aspect.replace("/", ":").split(":")
            ratio = float(a) / float(b)
            if ratio <= 0:
                raise ValueError
        except ValueError:
            raise PictureError(f"cannot read aspect {aspect!r}; use e.g. '16:9' or '1:1'") from None
        if w / h > ratio:
            cw, ch = max(1, round(h * ratio)), h
        else:
            cw, ch = w, max(1, round(w / ratio))
    else:
        if width is None or height is None:
            raise PictureError("for a fixed-size crop give both width and height")
        if width < 1 or height < 1:
            raise PictureError("width and height must be at least 1")
        if width > w or height > h:
            raise PictureError(f"a {width}x{height} window does not fit in the {w}x{h} image")
        cw, ch = width, height
    left = round((w - cw) * gx)
    top = round((h - ch) * gy)
    return left, top, left + cw, top + ch


def crop(
    path: str,
    box: list[int] | None = None,
    aspect: str | None = None,
    width: int | None = None,
    height: int | None = None,
    gravity: str = "center",
    output: str | None = None,
    overwrite: bool = False,
    quality: int | None = None,
    keep_metadata: bool = True,
    preview: bool = False,
    allow_huge: bool = False,
) -> Result:
    _check_quality(quality)
    src = open_image(path, allow_huge)
    w, h = src.size
    region = plan_crop(w, h, box, aspect, width, height, gravity)
    out, fmt = _choose_output(src, "cropped", output, overwrite)
    frames = [f.crop(region) for f in src.frames]
    notes = [f"region left={region[0]} top={region[1]} right={region[2]} bottom={region[3]}"]
    return _finish(
        src, frames, out, fmt, "Cropped", quality=quality, keep_metadata=keep_metadata,
        preview=preview, notes=notes,
    )


# ------------------------------------------------------------------------ contact_sheet


def _font(size: int) -> Any:
    return ImageFont.load_default(size=size)


def _collect(paths: list[str]) -> tuple[list[str], list[str]]:
    files: list[str] = []
    rejected: list[str] = []
    exts = im_.image_extensions()
    for p in paths:
        real = os.path.expanduser(p)
        if os.path.isdir(real):
            d = resolve(real, "folder")
            for name in sorted(os.listdir(d)):
                full = os.path.join(d, name)
                if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
                    try:
                        files.append(resolve(full, "file"))
                    except PictureError:
                        rejected.append(full)
        else:
            got, rej = expand(p)
            files += got
            rejected += rej
    seen: set[str] = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq, rejected


def contact_sheet(
    paths: list[str],
    output: str | None = None,
    overwrite: bool = False,
    columns: int | None = None,
    thumb_size: int = 200,
    max_images: int = 100,
    preview: bool = True,
    allow_huge: bool = False,
) -> Result:
    if not paths:
        raise PictureError("give at least one file, folder or glob in paths")
    if not 32 <= thumb_size <= 800:
        raise PictureError("thumb_size must be between 32 and 800")
    files, rejected = _collect(paths)
    if not files:
        raise PictureError("no image files matched " + ", ".join(paths))
    if len(files) > max_images:
        raise PictureError(
            f"{len(files)} images matched, more than max_images={max_images}. Narrow the "
            "pattern or raise max_images."
        )
    n = len(files)
    cols = columns or max(1, math.ceil(math.sqrt(n)))
    if cols < 1:
        raise PictureError("columns must be at least 1")
    rows = math.ceil(n / cols)
    label_h = max(14, thumb_size // 9)
    pad = 8
    cell_w, cell_h = thumb_size + pad * 2, thumb_size + label_h + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (236, 236, 236))
    draw = ImageDraw.Draw(sheet)
    font = _font(max(9, label_h - 4))
    errors: list[str] = []
    for idx, f in enumerate(files):
        cx, cy = (idx % cols) * cell_w, (idx // cols) * cell_h
        name = os.path.basename(f)
        try:
            src = open_image(f, allow_huge)
            t = src.frames[0].copy()
            t.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            t = flatten(t, (255, 255, 255)) if t.mode in ("RGBA", "LA") else t.convert("RGB")
            ox, oy = cx + pad + (thumb_size - t.width) // 2, cy + pad + (thumb_size - t.height) // 2
            sheet.paste(t, (ox, oy))
            label = name
        except PictureError as e:
            draw.rectangle(
                [cx + pad, cy + pad, cx + pad + thumb_size, cy + pad + thumb_size],
                fill=(250, 220, 220), outline=(200, 60, 60),
            )
            draw.text((cx + pad + 6, cy + pad + 6), "UNREADABLE", fill=(160, 20, 20), font=font)
            errors.append(f"{name}: {e}")
            label = name
        max_chars = max(6, thumb_size // max(5, label_h // 2))
        if len(label) > max_chars:
            if max_chars > 14:
                label = label[: max_chars - 11] + "..." + label[-8:]
            else:
                label = label[: max_chars - 3] + "..."
        draw.text((cx + pad, cy + pad + thumb_size + 3), label, fill=(30, 30, 30), font=font)
    first_dir = os.path.dirname(files[0])
    out = check_output(output or os.path.join(first_dir, "contact-sheet.jpg"), overwrite)
    fmt = format_for(out)
    buf = io.BytesIO()
    if fmt == "JPEG":
        sheet.save(buf, format="JPEG", quality=88, optimize=True)
    else:
        data, _ = encode([sheet], fmt)
        buf.write(data)
    write_bytes(out, buf.getvalue())
    lines = [
        f"Contact sheet: {out}",
        f"  {n} images in {cols} columns x {rows} rows, {sheet.width}x{sheet.height} px "
        f"({human(len(buf.getvalue()))}), each labelled with its file name",
        "  order: " + ", ".join(os.path.basename(f) for f in files[:30]) + (", ..." if n > 30 else ""),
    ]
    if rejected:
        lines.append(f"  skipped {len(rejected)} files outside the allowed directories")
    for msg in errors:
        lines.append(f"  UNREADABLE {msg}")
    pv = None
    if preview:
        pv = thumbnail_jpeg(buf.getvalue(), max_side=1024, quality=75)
    return Result("\n".join(lines), out, pv)


# --------------------------------------------------------------------------- make_icons

FAVICON_SET = [
    ("favicon-16x16.png", 16), ("favicon-32x32.png", 32), ("apple-touch-icon.png", 180),
    ("icon-192.png", 192), ("icon-512.png", 512),
]
ICO_DEFAULT = (16, 32, 48, 64, 128, 256)


def make_icons(
    path: str,
    kind: str = "favicon_set",
    sizes: list[int] | None = None,
    output: str | None = None,
    overwrite: bool = False,
    fit: str = "pad",
    background: str = "white",
    preview: bool = False,
    allow_huge: bool = False,
) -> Result:
    if kind not in ("favicon_set", "ico"):
        raise PictureError("kind must be 'favicon_set' or 'ico'")
    if fit not in ("pad", "crop"):
        raise PictureError("fit must be 'pad' (keep the whole image) or 'crop' (fill the square)")
    bg = parse_color(background)
    src = open_image(path, allow_huge)
    frame = src.frames[0].convert("RGBA")
    w, h = frame.size
    size_list = list(sizes) if sizes else list(ICO_DEFAULT)
    if any(s < 8 or s > 256 for s in size_list) and kind == "ico":
        raise PictureError("ICO sizes must be between 8 and 256")
    base_side = max(512 if kind == "favicon_set" else max(size_list), 8)
    if fit == "crop":
        base = ImageOps.fit(frame, (base_side, base_side), Image.Resampling.LANCZOS)
    else:
        inner = ImageOps.contain(frame, (base_side, base_side), Image.Resampling.LANCZOS)
        base = Image.new("RGBA", (base_side, base_side), (0, 0, 0, 0))
        base.paste(inner, ((base_side - inner.width) // 2, (base_side - inner.height) // 2))
    notes = []
    if max(w, h) < base_side:
        notes.append(
            f"source is {w}x{h}; the largest icon is {base_side}px, so it is upscaled (blurry). "
            "Use a larger source for sharp results."
        )
    if w != h and fit == "pad":
        notes.append("source is not square: padded with transparency (fit='crop' to fill instead)")

    def icon(side: int) -> Image.Image:
        return base.resize((side, side), Image.Resampling.LANCZOS)

    written: list[tuple[str, bytes]] = []
    stem = os.path.splitext(os.path.basename(src.path))[0]
    if kind == "ico":
        out = check_output(output or os.path.join(os.path.dirname(src.path), stem + ".ico"), overwrite)
        buf = io.BytesIO()
        base.save(buf, format="ICO", sizes=[(s, s) for s in sorted(set(size_list))])
        written.append((out, buf.getvalue()))
        html = ""
    else:
        folder = output or os.path.join(os.path.dirname(src.path), f"{stem}-favicons")
        folder = resolve(folder, "output folder")
        if os.path.exists(folder) and not os.path.isdir(folder):
            raise PictureError(f"{folder} exists and is not a folder")
        for name, side in FAVICON_SET:
            img = icon(side)
            if name == "apple-touch-icon.png":
                img = flatten(img, bg)  # iOS renders transparency as black
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            written.append((os.path.join(folder, name), buf.getvalue()))
        buf = io.BytesIO()
        base.save(buf, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
        written.append((os.path.join(folder, "favicon.ico"), buf.getvalue()))
        clash = [p for p, _ in written if os.path.exists(p)]
        if clash and not overwrite:
            raise PictureError(
                f"{len(clash)} output files already exist (e.g. {clash[0]}). Pass overwrite=true "
                "or choose another output folder."
            )
        os.makedirs(folder, exist_ok=True)
        html = (
            '<link rel="icon" href="/favicon.ico" sizes="48x48">\n'
            '<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">\n'
            '<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">\n'
            '<link rel="apple-touch-icon" href="/apple-touch-icon.png">'
        )
    for p, data in written:
        write_bytes(p, data)
    lines = [f"Wrote {len(written)} icon file(s):"]
    for p, data in written:
        lines.append(f"  {p} ({human(len(data))})")
    if kind == "ico":
        lines.append(f"  sizes inside the .ico: {', '.join(str(s) for s in sorted(set(size_list)))}")
    else:
        lines.append("  icon-192.png / icon-512.png are for a web app manifest.")
        lines.append("add to <head>:\n" + html)
    lines += [f"  note: {n}" for n in notes]
    pv = None
    if preview:
        strip_sizes = sorted(set(size_list)) if kind == "ico" else [16, 32, 48, 64, 128]
        total = sum(strip_sizes) + 10 * (len(strip_sizes) + 1)
        canvas = Image.new("RGB", (total, max(strip_sizes) + 20), (200, 200, 200))
        x = 10
        for s in strip_sizes:
            ic = icon(s)
            canvas.paste(ic, (x, 10), ic)
            x += s + 10
        b2 = io.BytesIO()
        canvas.save(b2, format="PNG")
        pv = thumbnail_jpeg(b2.getvalue(), max_side=max(384, canvas.width))
    return Result("\n".join(lines), written[0][0] if len(written) == 1 else None, pv)


# ---------------------------------------------------------------------------- palette


def palette(path: str, count: int = 5, preview: bool = False, allow_huge: bool = False) -> Result:
    if not 1 <= count <= 16:
        raise PictureError("count must be between 1 and 16")
    src = open_image(path, allow_huge)
    f = src.frames[0].copy()
    f.thumbnail((160, 160), Image.Resampling.BOX)
    rgba = f.convert("RGBA")
    raw = rgba.tobytes()
    pixels = [
        (raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 4) if raw[i + 3] >= 128
    ]
    if not pixels:
        raise PictureError("the image is fully transparent; no colors to report")
    strip = Image.new("RGB", (len(pixels), 1))
    strip.putdata(pixels)
    q = strip.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    counts = sorted(q.getcolors() or [], reverse=True)
    total = len(pixels)
    lines = [f"Dominant colors of {os.path.basename(src.path)} ({len(counts)} found):"]
    swatches = []
    for c, idx in counts:
        assert isinstance(idx, int)
        r, g, bl = pal[idx * 3 : idx * 3 + 3]
        swatches.append((r, g, bl))
        lines.append(f"  #{r:02x}{g:02x}{bl:02x}  {c / total * 100:5.1f}%  rgb({r}, {g}, {bl})")
    if src.frames[0].mode in ("RGBA", "LA") and total < f.width * f.height:
        lines.append("  (fully transparent pixels were ignored)")
    pv = None
    if preview:
        sw = Image.new("RGB", (len(swatches) * 64, 64))
        for i, col in enumerate(swatches):
            sw.paste(col, (i * 64, 0, i * 64 + 64, 64))
        sw_buf = io.BytesIO()
        sw.save(sw_buf, format="PNG")
        pv = sw_buf.getvalue()
    return Result("\n".join(lines), None, pv)


# ----------------------------------------------------------------------------- batch

BATCH_OPS: dict[str, tuple[Any, str]] = {
    "resize": (resize, "resized"),
    "convert": (convert, "converted"),
    "compress": (compress, "compressed"),
    "strip_metadata": (strip_metadata, "stripped"),
    "rotate_flip": (rotate_flip, "rotated"),
    "crop": (crop, "cropped"),
}
_RESERVED = {"path", "output", "overwrite", "preview"}


def _plan_batch_output(
    op: str, tag: str, src: str, params: dict[str, Any], output_dir: str | None
) -> str:
    stem, src_ext = os.path.splitext(src)
    fmt_word = params.get("format")
    ext = FORMAT_EXT[format_for(fmt_word)] if fmt_word else src_ext
    if output_dir:
        return os.path.join(output_dir, os.path.basename(stem) + ext)
    if op == "convert" and ext.lower() != src_ext.lower():
        return stem + ext
    return default_output(src, tag, ext)


def batch(
    operation: str,
    pattern: str,
    params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    max_files: int = 500,
) -> Result:
    import inspect

    if operation not in BATCH_OPS:
        raise PictureError(f"operation must be one of {', '.join(BATCH_OPS)}")
    fn, tag = BATCH_OPS[operation]
    params = dict(params or {})
    valid = set(inspect.signature(fn).parameters) - _RESERVED
    bad = set(params) - valid
    if bad:
        raise PictureError(
            f"params not valid for {operation}: {', '.join(sorted(bad))}. "
            f"Valid: {', '.join(sorted(valid))}"
        )
    if operation == "convert" and "format" not in params:
        raise PictureError("convert needs params={'format': 'webp'} (or jpg, png, ...)")
    if operation in ("resize",) and not any(k in params for k in ("width", "height", "percent")):
        raise PictureError("resize needs params with width and/or height, or percent")
    if operation == "compress" and not any(k in params for k in ("target_size", "quality", "preset")):
        params["preset"] = "balanced"
    out_dir = None
    if output_dir:
        out_dir = resolve(output_dir, "output_dir")
        if os.path.exists(out_dir) and not os.path.isdir(out_dir):
            raise PictureError(f"output_dir {out_dir} is not a folder")
    files, rejected = expand(pattern)
    exts = im_.image_extensions()
    candidates = [f for f in files if os.path.splitext(f)[1].lower() in exts]
    non_image = len(files) - len(candidates)
    skipped_tag = []
    if not out_dir:
        markers = [f".{t}." for _fn, t in BATCH_OPS.values()]
        skipped_tag = [
            f for f in candidates if any(m in os.path.basename(f) for m in markers)
        ]
        candidates = [f for f in candidates if f not in skipped_tag]
    if not candidates:
        raise PictureError(
            f"no image files to process for {pattern!r}"
            + (f" ({len(skipped_tag)} earlier outputs were skipped)" if skipped_tag else "")
        )
    if len(candidates) > max_files:
        raise PictureError(
            f"{len(candidates)} files matched, more than max_files={max_files}. "
            "Narrow the pattern or raise max_files."
        )
    plan = [(f, _plan_batch_output(operation, tag, f, params, out_dir)) for f in candidates]
    rows: list[tuple[str, str, str]] = []
    ok = failed = 0
    if dry_run:
        for src, dst in plan:
            note = ""
            try:
                real_dst = check_output(dst, overwrite) if (out_dir is None or os.path.isdir(out_dir)) else dst
                _ = real_dst
                status = "would write"
            except PictureError as e:
                status = "WOULD FAIL"
                note = str(e)
            rows.append((os.path.basename(src), status, os.path.basename(dst) + (f"  [{note}]" if note else "")))
        header = f"DRY RUN: {operation} on {len(plan)} files (nothing written). Re-run with dry_run=false to apply."
    else:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        for src, dst in plan:
            try:
                r = fn(path=src, output=dst, overwrite=overwrite, **params)
                rows.append((os.path.basename(src), "ok", f"{os.path.basename(dst)}: {r.detail}"))
                ok += 1
            except PictureError as e:
                rows.append((os.path.basename(src), "FAILED", str(e).splitlines()[0]))
                failed += 1
        header = f"{operation}: {ok} ok, {failed} failed of {len(plan)} files"
        if ok == 0:
            raise PictureError(header + "\n" + _table(rows))
    lines = [header, _table(rows)]
    if skipped_tag:
        lines.append(f"skipped {len(skipped_tag)} files that look like outputs of earlier runs (.resized., .stripped., ...)")
    if non_image:
        lines.append(f"ignored {non_image} non-image files")
    if rejected:
        lines.append(f"ignored {len(rejected)} files that resolve outside the allowed directories")
    return Result("\n".join(lines))


def _table(rows: list[tuple[str, str, str]]) -> str:
    w0 = min(40, max(len(r[0]) for r in rows))
    w1 = max(len(r[1]) for r in rows)
    out = []
    for a, b, c in rows:
        a2 = a if len(a) <= w0 else a[: w0 - 3] + "..."
        out.append(f"  {a2:<{w0}}  {b:<{w1}}  {c}")
    return "\n".join(out)

