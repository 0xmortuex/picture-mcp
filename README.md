# picture-mcp

**Everyday image jobs for your AI agent, by intent: shrink this to 200 KB, strip the GPS before I post it, make me a favicon set.**

<!-- mcp-name: io.github.0xmortuex/picture-mcp -->

picture-mcp is an [MCP](https://modelcontextprotocol.io) server for the image chores people
actually have. It needs **Pillow and nothing else** (no ImageMagick, no libvips, no ffmpeg, no
Node), so `uvx picture-mcp` works the same on Windows, macOS and Linux.

**How it differs from the other Pillow/sharp image MCP servers** (e.g.
[TOBYCAI/image-mcp](https://github.com/TOBYCAI/image-mcp),
[graphics-mcp](https://github.com/lesleslie/graphics-mcp),
[Image-Toolbox-MCP](https://github.com/Cowlsly/Image-Toolbox-MCP_By-Cowlsly),
[mcp-image-optimizer](https://github.com/piephai/mcp-image-optimizer)). Those are generic
resize/convert wrappers. This one adds the parts that matter in practice:

- **Compress to a size**: `target_size="200KB"` runs a real quality search and the result is
  measured, not guessed. It refuses to silently shrink your pixels.
- **A privacy tool that proves its work**: `strip_metadata` removes GPS, camera serials, XMP,
  IPTC, comments and data hidden after the JPEG end marker, **losslessly** for JPEG, lists
  exactly what it found, and re-reads its own output to verify it is clean.
- **The agent can see**: `preview=true` returns a small thumbnail of the result as MCP image
  content, and `contact_sheet` turns 50 files into one labelled picture to review.
- **Safe by construction**: a `--dir` sandbox with symlink and `..` protection, never
  overwrites, atomic writes, a decompression-bomb guard, `batch` that dry-runs first.

## Install

Claude Code:

```
claude mcp add picture -- uvx picture-mcp --dir ~/Pictures
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "picture": {
      "command": "uvx",
      "args": ["picture-mcp", "--dir", "C:\\Users\\you\\Pictures"]
    }
  }
}
```

Add more `--dir` entries to allow more folders. HEIC/HEIF (iPhone photos) is an optional extra
because it needs a native wheel: use `uvx --from "picture-mcp[heic]" picture-mcp`.

## Tools

| Tool | What it does |
|------|--------------|
| `image_info` | Format, size, mode, transparency, DPI, animation, EXIF summary, **GPS yes/no and where**, orientation, corrupt/truncated check |
| `resize` | `width`/`height`/`percent`; modes `fit`, `fill` (cover + center crop), `exact`; never upscales unless `allow_upscale` |
| `convert` | png, jpg, webp, gif, bmp, tiff, ico, avif and heic when available; alpha to jpg is flattened on a `background` colour you choose |
| `compress` | `target_size="200KB"` quality search, or `quality` / `preset`; PNG optimise; `format="webp"` to convert while compressing |
| `strip_metadata` | Remove GPS/EXIF/XMP/IPTC/comments, report what was removed, verify the output |
| `rotate_flip` | Clockwise degrees (any angle), flip; `degrees=0` just fixes EXIF orientation |
| `crop` | `box`, `aspect="16:9"`, or a `width`x`height` window, with gravity |
| `contact_sheet` | Many images into one labelled grid, returned as an image |
| `make_icons` | Favicon set (ico, 16, 32, apple-touch 180, 192, 512 + `<link>` tags) or a multi-size `.ico` |
| `palette` | Dominant colours as hex with percentages |
| `batch` | Any of resize/convert/compress/strip_metadata/rotate_flip/crop over a folder or glob; **dry run by default**; per-file result table |

Conventions every tool follows:

- Output goes next to the input as `photo.resized.jpg` (or `output=...`). **Nothing is ever
  overwritten** unless you pass `overwrite=true`. Writes are atomic (temp file + rename).
- EXIF orientation is applied first, so sizes and crop boxes mean what a viewer shows.
- Animated GIF/WebP/PNG keep every frame and duration. If the target format cannot animate
  (jpg, bmp...), the result says "flattened to its first frame".
- Metadata: EXIF and the ICC profile are carried through resize/convert/crop/rotate by default,
  XMP/IPTC are not. If the output still has GPS the result tells you. `keep_metadata=false`
  drops EXIF.
- `preview=true` on any tool returns a small JPEG thumbnail of the result.

## Try asking

- "What is in `D:\photos\IMG_4412.jpg`? Does it have a location?"
- "Make `hero.png` 1600 px wide and under 300 KB, as WebP."
- "Remove the GPS data from everything in `D:\to-post\` and show me what you removed."
- "Make a 1080x1080 square thumbnail of each photo in this folder, centered, as JPEG."
- "Give me a contact sheet of `D:\screenshots\*.png` so I can pick the right one."
- "Make a favicon set from `logo.svg.png` and give me the HTML tags."
- "This photo shows sideways in Slack. Fix it."
- "What are the brand colours in `banner.png`?"
- "Convert every PNG in `assets/` to WebP. Dry run first."

## Security

- **Use `--dir`.** Without it the server can read and write any path your account can. With it,
  every input and output path is resolved (`..` collapsed, symlinks followed, including a
  symlink as the final component) and refused if it ends up outside an allowed directory.
  `batch` globs skip matches that resolve outside.
- **No network, no subprocesses.** The server only opens files you point it at.
- **Decompression bombs.** Images over 100 megapixels (frames included) are refused before
  they are decoded, with an error naming the size. `--max-pixels MP` changes the limit, and a
  tool call can pass `allow_huge=true` after you decide to trust a file.
- **No overwrites, atomic writes.** A crash mid-encode never leaves a half-written file or
  damages an existing one.
- **`image_info` reveals GPS coordinates** of a photo to the agent by design (that is its
  job); `strip_metadata` is the fix before sharing.
- Outside a sandbox, an agent that is prompt-injected could write image files anywhere you can.
  That is why `--dir` is in every install line above.

## Development

```
uv sync --extra test --extra heic
uv run pytest
uv run ruff check src tests
uv run mypy --strict src
```

Tests generate real images with Pillow (gradients, text, alpha, EXIF with GPS, animated GIF,
CMYK, palette PNG, EXIF-rotated, a 20 MP image) and verify the outputs by re-opening and
measuring them, plus a stdio test that launches the real server and talks MCP to it.

A from-scratch tutorial that builds a minimal version of this server step by step:
[picture-mcp-the-hard-way](https://github.com/0xmortuex/picture-mcp-the-hard-way).

## License

MIT
