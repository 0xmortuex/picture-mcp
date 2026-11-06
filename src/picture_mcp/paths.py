"""Allowed-directory sandbox, output naming and atomic writes."""

from __future__ import annotations

import contextlib
import glob as _glob
import os
import tempfile
from collections.abc import Iterator

from PIL import Image

# We do our own, explicit pixel check (see open_image); Pillow's built-in guard
# only warns between 1x and 2x its limit, which is not a guard at all for an agent.
Image.MAX_IMAGE_PIXELS = None


class PictureError(Exception):
    """An expected, user-actionable failure. Its message is shown to the agent."""


class Config:
    def __init__(self) -> None:
        self.allowed: list[str] = []  # real, normalised absolute directories
        self.max_pixels: int = 100_000_000


CONFIG = Config()


def configure(dirs: list[str] | None = None, max_pixels: int | None = None) -> None:
    """Set the allowed directories (empty = unrestricted) and the pixel limit."""
    allowed: list[str] = []
    for d in dirs or []:
        real = os.path.realpath(os.path.expanduser(d))
        if not os.path.isdir(real):
            raise PictureError(f"--dir {d!r} is not an existing directory")
        allowed.append(os.path.normcase(real))
    CONFIG.allowed = allowed
    if max_pixels is not None:
        if max_pixels < 1:
            raise PictureError("max_pixels must be positive")
        CONFIG.max_pixels = max_pixels


def _inside(real_norm: str, root: str) -> bool:
    try:
        return os.path.commonpath([real_norm, root]) == root
    except ValueError:  # different drives on Windows
        return False


def resolve(path: str, what: str = "path") -> str:
    """Absolute, symlink-resolved path, checked against the sandbox.

    realpath() collapses '..' and follows symlinks (including a symlink as the
    final component), so neither can be used to step outside an allowed directory.
    """
    if not path or not path.strip():
        raise PictureError(f"{what} is empty")
    real = os.path.realpath(os.path.expanduser(path))
    if CONFIG.allowed:
        norm = os.path.normcase(real)
        if not any(_inside(norm, root) for root in CONFIG.allowed):
            raise PictureError(
                f"{what} {path!r} is outside the directories this server may touch "
                f"({', '.join(CONFIG.allowed)}). Use a path inside one of them."
            )
    return real


def expand(pattern: str) -> tuple[list[str], list[str]]:
    """Glob a pattern (** allowed). Returns (real file paths, rejected-by-sandbox paths)."""
    pattern = os.path.expanduser(pattern)
    if not glob_magic(pattern):
        return [resolve(pattern, "path")], []
    base = pattern
    for i, ch in enumerate(pattern):
        if ch in "*?[":
            base = pattern[:i]
            break
    resolve(os.path.dirname(base) or ".", "pattern directory")
    out: list[str] = []
    rejected: list[str] = []
    for hit in sorted(_glob.glob(pattern, recursive=True)):
        if os.path.isdir(hit):
            continue
        try:
            out.append(resolve(hit, "matched file"))
        except PictureError:
            rejected.append(hit)
    return out, rejected


def glob_magic(s: str) -> bool:
    return any(c in s for c in "*?[")


def default_output(src: str, tag: str, ext: str | None = None) -> str:
    """photo.jpg + 'resized' -> photo.resized.jpg (next to the input)."""
    root, src_ext = os.path.splitext(src)
    if ext is None:
        ext = src_ext
    elif not ext.startswith("."):
        ext = "." + ext
    return f"{root}.{tag}{ext}" if tag else f"{root}{ext}"


def check_output(out: str, overwrite: bool) -> str:
    """Validate an output path (sandbox + no clobbering). Returns the real path."""
    real = resolve(out, "output")
    if os.path.isdir(real):
        raise PictureError(f"output {out!r} is a directory; give a file name")
    parent = os.path.dirname(real)
    if not os.path.isdir(parent):
        raise PictureError(f"output directory {parent!r} does not exist")
    if os.path.exists(real) and not overwrite:
        raise PictureError(
            f"{real} already exists. Pass overwrite=true to replace it, or choose another output."
        )
    return real


@contextlib.contextmanager
def atomic_target(final: str) -> Iterator[str]:
    """Yield a temp path in the destination directory; os.replace() it on success.

    A crash or an encoder error therefore never leaves a half-written file at the
    destination, and an existing file is only replaced by a complete one.
    """
    directory = os.path.dirname(final)
    fd, tmp = tempfile.mkstemp(prefix=".picture-mcp-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_bytes(final: str, data: bytes) -> None:
    with atomic_target(final) as tmp, open(tmp, "wb") as fh:
        fh.write(data)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


def parse_size(text: str) -> int:
    """'200KB', '1.5 MB', '350000' (bytes) -> bytes."""
    s = text.strip().lower().replace(" ", "")
    mult = 1
    for suffix, m in (("kb", 1024), ("mb", 1024**2), ("kib", 1024), ("mib", 1024**2), ("b", 1)):
        if s.endswith(suffix):
            s, mult = s[: -len(suffix)], m
            break
    try:
        value = float(s)
    except ValueError:
        raise PictureError(
            f"cannot read target size {text!r}; use e.g. '200KB', '1.5MB' or a byte count"
        ) from None
    n = int(value * mult)
    if n < 1:
        raise PictureError("target size must be positive")
    return n
