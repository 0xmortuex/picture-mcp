"""Builders for real test images: gradients, text, alpha, EXIF with GPS, animations, CMYK..."""

from __future__ import annotations

import os

from PIL import Image, ImageDraw


def gradient(w: int = 200, h: int = 100) -> Image.Image:
    """Red increases left to right, green top to bottom, blue constant 128."""
    r = Image.linear_gradient("L").rotate(90).resize((w, h))
    g = Image.linear_gradient("L").resize((w, h))
    b = Image.new("L", (w, h), 128)
    return Image.merge("RGB", (r, g, b))


def with_text(im: Image.Image, text: str = "picture-mcp") -> Image.Image:
    out = im.copy()
    ImageDraw.Draw(out).text((6, 6), text, fill=(255, 255, 255))
    return out


def exif_with_gps(orientation: int | None = None) -> Image.Exif:
    ex = Image.Exif()
    ex[0x010F] = "Acme Cameras"  # Make
    ex[0x0110] = "Model X"  # Model
    ex[0x0131] = "TestSoftware 1.0"  # Software
    ex[0x0132] = "2026:01:02 03:04:05"  # DateTime
    if orientation:
        ex[0x0112] = orientation
    gps = ex.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (41.0, 1.0, 30.0)
    gps[3] = "E"
    gps[4] = (29.0, 0.0, 0.0)
    ex[0x8825] = gps
    return ex


def save_jpeg_with_gps(path: str, w: int = 200, h: int = 100, orientation: int | None = None) -> str:
    with_text(gradient(w, h)).save(path, exif=exif_with_gps(orientation), quality=92)
    return path


def noisy(w: int, h: int) -> Image.Image:
    """Photo-like content that does not compress to nothing."""
    import random

    rnd = random.Random(1234)
    data = bytes(rnd.getrandbits(8) for _ in range(w * h * 3 // 16 + 3))
    small = Image.frombytes("RGB", (w // 4, h // 4), (data * 16)[: (w // 4) * (h // 4) * 3])
    return small.resize((w, h), Image.Resampling.BICUBIC)


def animated_gif(path: str, frames: int = 4, size: int = 40, duration: int = 80) -> str:
    imgs = []
    for i in range(frames):
        im = Image.new("RGB", (size, size), (255, 255, 255))
        ImageDraw.Draw(im).rectangle([i * 4, i * 4, i * 4 + 12, i * 4 + 12], fill=(255, 0, 0))
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    return path


def exists_only(folder: str) -> list[str]:
    return sorted(os.listdir(folder))
