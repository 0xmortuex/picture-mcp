# Changelog

## 0.1.0 - 2026-11-06

First release.

- Tools: `image_info`, `resize`, `convert`, `compress`, `strip_metadata`, `rotate_flip`,
  `crop`, `contact_sheet`, `make_icons`, `palette`, `batch`.
- Pillow only, no system dependencies. Optional `picture-mcp[heic]` extra for HEIC/HEIF.
- `--dir` (repeatable) / `PICTURE_MCP_DIRS` sandbox with symlink and `..` protection.
- Never overwrites unless `overwrite=true`; all writes are atomic.
- Decompression-bomb guard (default 100 MP) with an explicit `allow_huge` override.
- `preview=true` returns a small thumbnail of the result as MCP image content.
- Animated GIF/WebP/PNG keep their frames; formats that cannot animate say they flattened.
