"""picture-mcp: an MCP server for everyday image jobs, built on Pillow only."""

from __future__ import annotations

import argparse
import functools
import os
import sys
from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__, ops
from .paths import CONFIG, PictureError, configure

mcp = MCPServer(
    "picture",
    instructions=(
        "Everyday image jobs by intent, with Pillow only (no system tools). Start with "
        "image_info to see what a file is (size, format, animation, whether it carries a GPS "
        "location). Then resize, convert, compress (to a target size like '200KB'), "
        "strip_metadata (remove GPS/EXIF before sharing), rotate_flip, crop, make_icons "
        "(favicon set / .ico), palette (dominant colors), contact_sheet (many images -> one "
        "labelled picture to review at a glance) and batch (any of those over a folder or "
        "glob; it dry-runs by default). Outputs are written next to the input as "
        "photo.resized.jpg and never overwrite anything unless overwrite=true. EXIF "
        "orientation is always applied first, so sizes and crops refer to what a viewer shows. "
        "Pass preview=true to get a small thumbnail of the result back so you can look at it."
    ),
)

F = TypeVar("F", bound=Callable[..., Any])
Reply = list[str | Image]


def _reply(result: ops.Result) -> Reply:
    out: Reply = [result.text]
    if result.preview is not None:
        out.append(Image(data=result.preview, format="jpeg"))
    return out


def tool(fn: F) -> F:
    """Register fn as an MCP tool; expected failures become ToolErrors the agent can read."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except PictureError as e:
            raise ToolError(str(e)) from e

    mcp.tool(structured_output=False)(wrapper)
    return fn


@tool
def image_info(path: str, preview: bool = False) -> Reply:
    """Describe an image file: format, pixel size, mode, transparency, DPI, file size,
    animation (frame count and duration), EXIF summary (camera, date, software), whether
    it contains a GPS location (and where), EXIF orientation, and whether the file is
    corrupt or truncated. Reads headers and verifies the file decodes, so it is the right
    first call for any image. preview=true also returns a small thumbnail."""
    return _reply(ops.image_info(path, preview))


@tool
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
) -> Reply:
    """Resize an image. Give width and/or height (pixels) or percent.

    One side only: the other follows the aspect ratio. Both sides, by mode:
    fit (default) = largest size inside the box, keeps aspect ratio; fill = cover the box
    and center-crop the overflow (thumbnails, avatars); exact = stretch to exactly
    width x height.

    Never enlarges unless allow_upscale=true (a too-large request keeps the original size
    and says so). Output goes to photo.resized.jpg next to the input unless output= is
    given (its extension picks the format, so output='x.webp' resizes and converts).
    Animated GIF/WebP keep every frame. quality (1-100) applies to JPEG/WebP/AVIF output.
    keep_metadata=false drops EXIF (including GPS). Existing files are never replaced unless
    overwrite=true. allow_huge=true lifts the decompression-bomb limit (default 100 MP)."""
    return _reply(
        ops.resize(
            path, width, height, percent, mode, allow_upscale, resample, output, overwrite,
            quality, keep_metadata, preview, allow_huge,
        )
    )


@tool
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
) -> Reply:
    """Convert an image to another format: png, jpg, webp, gif, bmp, tiff, ico, and
    avif / heic when this install supports them (an error says so if not).

    Transparency going to jpg or bmp is flattened onto `background` (a name like 'white'
    or hex like '#ff00ff'; the result says so). Animated GIF/WebP/PNG stay animated when the
    target can hold an animation, otherwise the first frame is used and the result says so.
    CMYK and 16-bit images are converted to RGB/8-bit. lossless=true applies to webp.
    Output is photo.webp next to the input (photo.converted.png if the format is unchanged)
    unless output= is given; existing files are never replaced unless overwrite=true."""
    return _reply(
        ops.convert(
            path, format, output, overwrite, quality, background, lossless, keep_metadata,
            preview, allow_huge,
        )
    )


@tool
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
) -> Reply:
    """Make an image file smaller.

    target_size='200KB' (or '1.5MB', or bytes): finds the highest quality that still fits,
    by searching (jpg/webp/avif) and tells you the result. If even min_quality cannot fit,
    it fails and says what is achievable, unless allow_downscale=true, which then shrinks
    the pixel dimensions step by step (reported). PNG is lossless: it is re-optimised, and
    lossy_png=true allows palette reduction (256 -> 16 colors) to reach a target.
    Without a target: quality=1-100, or preset high (88) / balanced (78) / small (60).
    format='webp' (or jpg/png) converts while compressing, e.g. a photo PNG to WebP.
    Output goes to photo.compressed.jpg unless output= is given; never overwrites unless
    overwrite=true. Warns if the result is not smaller than the input."""
    return _reply(
        ops.compress(
            path, target_size, quality, preset, format, min_quality, allow_downscale,
            lossy_png, output, overwrite, keep_metadata, preview, allow_huge,
        )
    )


@tool
def strip_metadata(
    path: str,
    output: str | None = None,
    overwrite: bool = False,
    keep_orientation: bool = True,
    quality: int = 95,
    preview: bool = False,
    allow_huge: bool = False,
) -> Reply:
    """Remove private metadata (GPS location, camera make/model/serial, dates, software,
    XMP, IPTC, comments, trailing embedded data) so a photo can be shared safely. Reports
    exactly what was found and removed, then re-reads the output to verify it is clean.

    JPEGs are stripped losslessly (pixel data copied byte for byte). With
    keep_orientation=true (default) a bare Orientation tag is kept so the photo still shows
    upright; keep_orientation=false bakes the rotation into the pixels (re-encodes at
    `quality`). Other formats are re-encoded without metadata (PNG is lossless, WebP lossy).
    The ICC color profile is kept. Output: photo.stripped.jpg next to the input unless
    output= is given; never replaces a file unless overwrite=true."""
    return _reply(
        ops.strip_metadata(path, output, overwrite, keep_orientation, quality, preview, allow_huge)
    )


@tool
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
) -> Reply:
    """Rotate clockwise by `degrees` (90/180/270 are exact and lossless; other angles enlarge
    the canvas and fill corners with `background`, transparent by default) and/or flip
    ('horizontal' = mirror left-right, 'vertical' = upside down).

    EXIF orientation is always applied first, so degrees=0 with no flip simply fixes a
    photo that shows sideways in some viewers (auto-orient) and removes the stale tag.
    Output: photo.rotated.jpg unless output= is given; never overwrites unless
    overwrite=true."""
    return _reply(
        ops.rotate_flip(
            path, degrees, flip, background, output, overwrite, quality, keep_metadata,
            preview, allow_huge,
        )
    )


@tool
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
) -> Reply:
    """Crop an image. Pick exactly one mode:
    box=[left, top, right, bottom] in pixels of the upright image;
    aspect='16:9' (or '1:1', '4:5') for the largest region with that ratio;
    width= and height= for a window of exactly that size.
    For aspect and width/height, gravity picks where the region sits: center (default),
    top, bottom, left, right, top-left, top-right, bottom-left, bottom-right.
    Out-of-range boxes are an error, not silently clipped. Output: photo.cropped.jpg unless
    output= is given; never overwrites unless overwrite=true."""
    return _reply(
        ops.crop(
            path, box, aspect, width, height, gravity, output, overwrite, quality,
            keep_metadata, preview, allow_huge,
        )
    )


@tool
def contact_sheet(
    paths: list[str],
    output: str | None = None,
    overwrite: bool = False,
    columns: int | None = None,
    thumb_size: int = 200,
    max_images: int = 100,
    preview: bool = True,
    allow_huge: bool = False,
) -> Reply:
    """Combine many images into ONE labelled grid so you can review them at a glance.
    paths takes files, folders (their images, not recursive) and globs like 'D:/shots/**/*.png'.
    Each thumbnail is labelled with its file name; unreadable files get a red UNREADABLE tile
    and are listed. Output defaults to contact-sheet.jpg in the first image's folder (never
    overwrites unless overwrite=true). preview defaults to true here: the sheet itself is
    returned as an image."""
    return _reply(
        ops.contact_sheet(
            paths, output, overwrite, columns, thumb_size, max_images, preview, allow_huge
        )
    )


@tool
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
) -> Reply:
    """Make icons from a logo/image. kind='favicon_set' (default) writes favicon.ico
    (16/32/48), favicon-16x16.png, favicon-32x32.png, apple-touch-icon.png (180, flattened
    onto `background`), icon-192.png and icon-512.png into a folder (output=, default
    <name>-favicons next to the input) and returns the <link> tags to paste. kind='ico'
    writes one multi-size .ico (sizes=[16,32,...,256], default 16-256).
    Non-square sources are padded with transparency (fit='pad') or center-cropped
    (fit='crop'). Warns if the source is smaller than the largest icon. Never overwrites
    unless overwrite=true."""
    return _reply(
        ops.make_icons(path, kind, sizes, output, overwrite, fit, background, preview, allow_huge)
    )


@tool
def palette(path: str, count: int = 5, preview: bool = False, allow_huge: bool = False) -> Reply:
    """Dominant colors of an image as hex codes with their share of the picture
    (most common first). Transparent pixels are ignored. count is 1-16. preview=true returns
    a swatch strip."""
    return _reply(ops.palette(path, count, preview, allow_huge))


@tool
def batch(
    operation: str,
    pattern: str,
    params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    max_files: int = 500,
) -> Reply:
    """Apply one operation to many files and get a per-file result table.

    operation: resize | convert | compress | strip_metadata | rotate_flip | crop.
    pattern: a file, or a glob such as 'D:/photos/*.jpg' or 'D:/photos/**/*.png'.
    params: that operation's own options, e.g. {'width': 1600}, {'format': 'webp'},
    {'target_size': '300KB'}.
    dry_run defaults to TRUE: it lists what would be written (and what would fail, e.g.
    existing outputs) without touching anything; call again with dry_run=false to apply.
    Outputs land next to each input as name.resized.jpg, or in output_dir keeping the file
    names. Files that look like earlier outputs of the same operation are skipped. One bad
    file does not stop the run: failures are listed in the table (the call only errors if
    every file failed). Never overwrites unless overwrite=true."""
    return _reply(ops.batch(operation, pattern, params, output_dir, dry_run, overwrite, max_files))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="picture-mcp",
        description="MCP server for everyday image jobs. Speaks MCP over stdio.",
    )
    parser.add_argument(
        "--dir", action="append", default=[], metavar="DIR",
        help="allowed directory (repeatable). Files outside it, including via symlinks or .., "
        "are refused. Also read from PICTURE_MCP_DIRS (separated by the OS path separator). "
        "Without any, every path the user account can reach is allowed.",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=100, metavar="MP",
        help="decompression-bomb limit in megapixels (default 100); tools can override "
        "per call with allow_huge=true",
    )
    parser.add_argument("--version", action="version", version=f"picture-mcp {__version__}")
    args = parser.parse_args(argv)
    dirs = list(args.dir)
    env = os.environ.get("PICTURE_MCP_DIRS")
    if env:
        dirs += [d for d in env.split(os.pathsep) if d]
    try:
        configure(dirs, args.max_pixels * 1_000_000)
    except PictureError as e:
        sys.exit(f"picture-mcp: {e}")
    if CONFIG.allowed:
        print(f"picture-mcp: restricted to {', '.join(CONFIG.allowed)}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
