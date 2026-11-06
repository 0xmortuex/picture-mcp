"""Loading, normalising, encoding and metadata helpers on top of Pillow."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any

from PIL import (
    ExifTags,
    Image,
    ImageCms,
    ImageColor,
    ImageOps,
    UnidentifiedImageError,
    features,
)

from .paths import CONFIG, PictureError, human, resolve

try:  # optional extra: pip install "picture-mcp[heic]"
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC = True
except ImportError:
    HEIC = False

EXT_FORMAT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".jpe": "JPEG", ".png": "PNG", ".webp": "WEBP",
    ".gif": "GIF", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF", ".ico": "ICO",
    ".avif": "AVIF", ".heic": "HEIF", ".heif": "HEIF",
}
FORMAT_EXT = {
    "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp",
    "TIFF": ".tiff", "ICO": ".ico", "AVIF": ".avif", "HEIF": ".heic",
}
NAME_FORMAT = {
    "jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "gif": "GIF", "bmp": "BMP",
    "tif": "TIFF", "tiff": "TIFF", "ico": "ICO", "avif": "AVIF", "heic": "HEIF", "heif": "HEIF",
}
ANIMATED_OK = {"GIF", "WEBP", "PNG"}
EXIF_OK = {"JPEG", "PNG", "WEBP", "AVIF", "HEIF"}
GPS_IFD = 0x8825
EXIF_IFD = 0x8769
ORIENTATION = 0x0112


def image_extensions() -> set[str]:
    exts = set(EXT_FORMAT)
    if not HEIC:
        exts -= {".heic", ".heif"}
    return exts


def format_for(name_or_ext: str) -> str:
    """Pillow format name from a file name, extension or a word like 'jpg'."""
    key = name_or_ext.strip().lower()
    if "." in key:
        key = os.path.splitext(key)[1]
    key = key.lstrip(".")
    fmt = NAME_FORMAT.get(key)
    if fmt is None:
        raise PictureError(
            f"unknown image format {name_or_ext!r}; use one of: "
            + ", ".join(sorted(k for k in NAME_FORMAT if k not in ("jpeg", "tif", "heif")))
        )
    if fmt == "AVIF" and not features.check("avif"):
        raise PictureError(
            "AVIF is not available in this Pillow build (needs Pillow >= 11.3 wheels). "
            "Use webp instead."
        )
    if fmt == "HEIF" and not HEIC:
        raise PictureError(
            "HEIC support is not installed. Install the extra: pip install 'picture-mcp[heic]'"
        )
    return fmt


@dataclass
class Loaded:
    path: str
    format: str | None
    frames: list[Image.Image]
    durations: list[int]
    loop: int
    exif: Image.Exif
    icc: bytes | None
    dpi: tuple[float, float] | None
    src_size: tuple[int, int]
    src_bytes: int
    exif_orientation: int
    notes: list[str] = field(default_factory=list)

    @property
    def animated(self) -> bool:
        return len(self.frames) > 1

    @property
    def size(self) -> tuple[int, int]:
        return self.frames[0].size


def _norm_mode(im: Image.Image, notes: list[str], icc: bytes | None) -> Image.Image:
    mode = im.mode
    if mode in ("RGB", "RGBA", "L", "LA"):
        return im
    if mode in ("P", "PA"):
        return im.convert("RGBA" if (mode == "PA" or "transparency" in im.info) else "RGB")
    if mode == "CMYK":
        if icc:
            try:
                src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst = ImageCms.createProfile("sRGB")
                out = ImageCms.profileToProfile(im, src, dst, outputMode="RGB")
                if out is not None:
                    notes.append("CMYK converted to sRGB using the embedded ICC profile")
                    return out
            except ImageCms.PyCMSError:
                pass
        notes.append("CMYK converted to RGB without a color profile (colors may shift slightly)")
        return im.convert("RGB")
    if mode in ("I;16", "I;16L", "I;16B", "I"):
        notes.append(f"{mode} (16/32-bit gray) scaled down to 8-bit gray")
        return im.point(lambda v: v * (1 / 256)).convert("L")
    if mode in ("1", "F"):
        return im.convert("L")
    return im.convert("RGB")


def open_image(path: str, allow_huge: bool = False) -> Loaded:
    """Open, validate and fully decode an image (all frames), applying EXIF orientation."""
    real = resolve(path, "path")
    if not os.path.exists(real):
        raise PictureError(f"file not found: {real}")
    if not os.path.isfile(real):
        raise PictureError(f"{real} is not a file")
    name = os.path.basename(real)

    try:
        im = Image.open(real)
    except UnidentifiedImageError:
        hint = ""
        if os.path.splitext(real)[1].lower() in (".heic", ".heif") and not HEIC:
            hint = " HEIC needs the extra: pip install 'picture-mcp[heic]'."
        raise PictureError(
            f"{name} is not an image this server can read (unrecognised or damaged header).{hint}"
        ) from None
    except OSError as e:
        raise PictureError(f"cannot open {name}: {e}") from None

    with im:
        width, height = im.size
        # iPhone-style JPEGs are "MPO": extra frames are previews/gain maps, not an animation.
        n_frames = 1 if im.format == "MPO" else getattr(im, "n_frames", 1)
        total = width * height * n_frames
        if total > CONFIG.max_pixels and not allow_huge:
            raise PictureError(
                f"{name} is {width}x{height}"
                + (f" x {n_frames} frames" if n_frames > 1 else "")
                + f" = {total / 1e6:.0f} megapixels, over the safety limit of "
                f"{CONFIG.max_pixels / 1e6:.0f} MP (decompression-bomb guard). If you trust this "
                "file, pass allow_huge=true; decoding needs roughly "
                f"{human(total * 4)} of memory."
            )
        fmt = "JPEG" if im.format == "MPO" else im.format
        notes: list[str] = []
        try:
            exif = im.getexif()
            icc = im.info.get("icc_profile")
            dpi_raw = im.info.get("dpi")
            dpi = (float(dpi_raw[0]), float(dpi_raw[1])) if dpi_raw else None
            orientation = int(exif.get(ORIENTATION, 1) or 1)
            frames: list[Image.Image] = []
            durations: list[int] = []
            loop = int(im.info.get("loop", 0) or 0)
            if n_frames > 1:
                for i in range(n_frames):
                    im.seek(i)
                    im.load()  # WebP only exposes this frame's duration after decoding it
                    durations.append(int(im.info.get("duration", 100) or 100))
                    frames.append(_norm_mode(im.copy(), notes, icc))
                # Frames of one animation must share a mode to be re-encoded together.
                if any(f.mode == "RGBA" for f in frames):
                    frames = [f.convert("RGBA") for f in frames]
            else:
                im.load()
                src_mode = im.mode
                first = ImageOps.exif_transpose(im)
                exif = first.getexif()
                frames = [_norm_mode(first, notes, icc)]
                if orientation != 1:
                    notes.append(f"EXIF orientation {orientation} applied to the pixels")
                if src_mode == "CMYK":
                    icc = None
        except (OSError, SyntaxError, ValueError, EOFError, MemoryError) as e:
            raise PictureError(
                f"{name} looks corrupt or truncated ({e}). Re-download or re-export the file; "
                "nothing was written."
            ) from None
    return Loaded(
        path=real, format=fmt, frames=frames, durations=durations, loop=loop, exif=exif,
        icc=icc, dpi=dpi, src_size=(width, height), src_bytes=os.path.getsize(real),
        exif_orientation=orientation, notes=notes,
    )


def has_alpha(im: Image.Image) -> bool:
    return im.mode in ("RGBA", "LA")


def parse_color(text: str) -> tuple[int, int, int]:
    try:
        r, g, b = ImageColor.getrgb(text)[:3]
    except ValueError:
        raise PictureError(
            f"cannot read color {text!r}; use a name like 'white' or hex like '#ffffff'"
        ) from None
    return r, g, b


def flatten(im: Image.Image, bg: tuple[int, int, int]) -> Image.Image:
    """Composite an alpha image onto a solid background; returns RGB."""
    rgba = im.convert("RGBA")
    base = Image.new("RGBA", rgba.size, (*bg, 255))
    return Image.alpha_composite(base, rgba).convert("RGB")


def _exif_for_save(exif: Image.Exif | None, fmt: str) -> Any:
    if exif is None or len(exif) == 0 or fmt not in EXIF_OK:
        return None
    return exif.tobytes() if fmt == "HEIF" else exif


def encode(
    frames: list[Image.Image],
    fmt: str,
    *,
    durations: list[int] | None = None,
    loop: int = 0,
    quality: int | None = None,
    exif: Image.Exif | None = None,
    icc: bytes | None = None,
    dpi: tuple[float, float] | None = None,
    bg: tuple[int, int, int] = (255, 255, 255),
    lossless: bool = False,
) -> tuple[bytes, list[str]]:
    """Encode frames as `fmt` into memory. Returns (bytes, notes)."""
    notes: list[str] = []
    animated = len(frames) > 1
    if animated and fmt not in ANIMATED_OK:
        notes.append(
            f"animated input flattened to its first frame ({fmt} cannot hold an animation)"
        )
        frames = frames[:1]
        animated = False
    exif_save = _exif_for_save(exif, fmt)
    if exif is not None and len(exif) and exif_save is None:
        notes.append(f"EXIF not carried over ({fmt} output does not store it here)")

    prepared: list[Image.Image] = []
    for f in frames:
        if fmt in ("JPEG", "BMP") and has_alpha(f):
            if not prepared:
                notes.append(
                    "transparency flattened onto "
                    f"#{bg[0]:02x}{bg[1]:02x}{bg[2]:02x} ({fmt} has no alpha)"
                )
            f = flatten(f, bg)
        elif fmt == "JPEG" and f.mode not in ("RGB", "L"):
            f = f.convert("RGB")
        elif fmt in ("WEBP", "AVIF", "HEIF") and f.mode not in ("RGB", "RGBA"):
            f = f.convert("RGBA" if has_alpha(f) else "RGB")
        prepared.append(f)
    if fmt == "GIF" and not animated:
        notes.append("GIF is limited to 256 colors")

    kwargs: dict[str, Any] = {}
    if icc and fmt in ("JPEG", "PNG", "WEBP", "TIFF", "AVIF", "HEIF"):
        kwargs["icc_profile"] = icc
    if exif_save is not None:
        kwargs["exif"] = exif_save
    if dpi and fmt in ("JPEG", "PNG", "TIFF", "BMP"):
        kwargs["dpi"] = dpi
    if fmt == "JPEG":
        kwargs.update(quality=quality or 90, optimize=True, progressive=True)
    elif fmt == "PNG":
        kwargs.update(optimize=True)
    elif fmt == "WEBP":
        kwargs.update(quality=quality or 85, method=4, lossless=lossless)
    elif fmt in ("AVIF", "HEIF"):
        kwargs.update(quality=quality or 75)
    elif fmt == "TIFF":
        kwargs.update(compression="tiff_adobe_deflate")
    elif fmt == "GIF":
        kwargs.update(optimize=True)
    elif fmt == "ICO":
        w, h = prepared[0].size
        if max(w, h) > 256:
            raise PictureError(
                f"ICO images are at most 256x256 but this one is {w}x{h}. Resize it first, "
                "or use make_icons which builds a multi-size .ico for you."
            )
        kwargs["sizes"] = [(w, h)]
        prepared[0] = prepared[0].convert("RGBA")
    if animated:
        kwargs.update(
            save_all=True, append_images=prepared[1:], loop=loop,
            duration=durations or [100] * len(prepared),
        )

    buf = io.BytesIO()
    try:
        prepared[0].save(buf, format=fmt, **kwargs)
    except (OSError, ValueError, KeyError, SyntaxError) as e:
        raise PictureError(f"could not encode as {fmt}: {e}") from None
    return buf.getvalue(), notes


def thumbnail_jpeg(data: bytes, max_side: int = 384, quality: int = 72) -> bytes:
    """A small JPEG of an encoded image (first frame), for returning to the agent."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        frame = im.convert("RGBA")
    frame.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    flatten(frame, (255, 255, 255)).save(out, format="JPEG", quality=quality)
    return out.getvalue()


# ---------------------------------------------------------------- EXIF helpers

_TAG = {int(k): v for k, v in ExifTags.TAGS.items()}
_GPS_TAG = {int(k): v for k, v in ExifTags.GPSTAGS.items()}
SUMMARY_TAGS = (
    "Make", "Model", "LensModel", "Software", "DateTime", "DateTimeOriginal", "ExposureTime",
    "FNumber", "ISOSpeedRatings", "PhotographicSensitivity", "FocalLength", "Orientation",
    "Artist", "Copyright", "ImageDescription",
)


def exif_items(exif: Image.Exif) -> dict[str, Any]:
    """All base + Exif-IFD tags by name (GPS is handled separately)."""
    items: dict[str, Any] = {}
    for tag, value in exif.items():
        if tag in (GPS_IFD, EXIF_IFD, 0xA005):
            continue
        items[_TAG.get(int(tag), f"Tag{int(tag):#06x}")] = value
    try:
        ifd = exif.get_ifd(EXIF_IFD)
    except (KeyError, ValueError, OSError):
        ifd = {}
    for tag, value in ifd.items():
        items[_TAG.get(int(tag), f"Tag{int(tag):#06x}")] = value
    return items


def gps_info(exif: Image.Exif) -> dict[str, Any] | None:
    try:
        gps = exif.get_ifd(GPS_IFD)
    except (KeyError, ValueError, OSError):
        return None
    if not gps:
        return None
    named = {_GPS_TAG.get(int(k), str(k)): v for k, v in gps.items()}

    def dec(value: Any, ref: Any) -> float | None:
        try:
            d, m, s = (float(x) for x in value)
        except (TypeError, ValueError):
            return None
        out = d + m / 60 + s / 3600
        return -out if ref in ("S", "W") else out

    lat = dec(named.get("GPSLatitude"), named.get("GPSLatitudeRef"))
    lon = dec(named.get("GPSLongitude"), named.get("GPSLongitudeRef"))
    return {"latitude": lat, "longitude": lon, "tags": sorted(named)}


def short(value: Any, limit: int = 60) -> str:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    s = str(value).strip("\x00 ")
    return s if len(s) <= limit else s[: limit - 1] + "..."


def metadata_inventory(raw: bytes) -> dict[str, Any]:
    """What privacy-relevant metadata an encoded image contains."""
    with Image.open(io.BytesIO(raw)) as im:
        exif = im.getexif()
        info = dict(im.info)
        text = dict(getattr(im, "text", {}) or {})
        fmt = "JPEG" if im.format == "MPO" else im.format
    xmp = bool(info.get("xmp") or info.get("XML:com.adobe.xmp"))
    if b"<x:xmpmeta" in raw or b"http://ns.adobe.com/xap/1.0/" in raw:
        xmp = True
    iptc = bool(info.get("photoshop")) or (fmt == "JPEG" and b"Photoshop 3.0\x00" in raw)
    other = []
    if info.get("comment"):
        other.append("comment")
    other += [k for k in text if k != "XML:com.adobe.xmp"]
    return {
        "format": fmt, "exif": exif_items(exif), "gps": gps_info(exif), "xmp": xmp,
        "iptc": iptc, "text": other, "icc": bool(info.get("icc_profile")),
    }
