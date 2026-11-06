"""MCP stdio smoke test: spawns the real server over stdio, lists the tools, runs real image
jobs, checks that preview=true returns a decodable image content block, that --dir sandboxes
the server, and that instructive errors survive the trip to the agent (MCPServer hides any
exception that is not a ToolError)."""

import base64
import io
import os
import sys

import pytest

pytest.importorskip("mcp")

from helpers import gradient, save_jpeg_with_gps  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402
from PIL import Image  # noqa: E402

EXPECTED_TOOLS = {
    "image_info", "resize", "convert", "compress", "strip_metadata", "rotate_flip", "crop",
    "contact_sheet", "make_icons", "palette", "batch",
}


def _params(*extra: str) -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "picture_mcp", *extra])


@pytest.mark.anyio
async def test_tools_and_real_jobs_over_stdio(tmp_path):
    photo = save_jpeg_with_gps(str(tmp_path / "trip.jpg"), 800, 600)
    async with stdio_client(_params()) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        assert init.server_info.name == "picture"
        tools = (await session.list_tools()).tools
        assert {t.name for t in tools} == EXPECTED_TOOLS
        for t in tools:
            assert t.description and len(t.description) > 80, t.name

        res = await session.call_tool("image_info", {"path": photo})
        assert not res.is_error and "GPS: YES" in res.content[0].text

        res = await session.call_tool("resize", {"path": photo, "width": 200, "preview": True})
        assert not res.is_error
        kinds = [c.type for c in res.content]
        assert kinds == ["text", "image"], kinds
        img = res.content[1]
        assert img.mime_type == "image/jpeg"
        with Image.open(io.BytesIO(base64.b64decode(img.data))) as thumb:
            assert thumb.size == (200, 150)
        with Image.open(tmp_path / "trip.resized.jpg") as out:
            assert out.size == (200, 150)

        res = await session.call_tool("strip_metadata", {"path": photo})
        assert "GPS none" in res.content[0].text and len(res.content) == 1
        with Image.open(tmp_path / "trip.stripped.jpg") as out:
            assert not out.getexif().get_ifd(0x8825)

        res = await session.call_tool(
            "batch", {"operation": "resize", "pattern": str(tmp_path / "*.jpg"),
                      "params": {"width": 50}}
        )
        assert "DRY RUN" in res.content[0].text


@pytest.mark.anyio
async def test_instructive_errors_reach_the_agent(tmp_path):
    src = str(tmp_path / "a.png")
    gradient(40, 40).save(src)
    async with stdio_client(_params()) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        res = await session.call_tool("resize", {"path": src})
        assert res.is_error and "give width and/or height" in res.content[0].text
        res = await session.call_tool("resize", {"path": src, "width": 20})
        assert not res.is_error
        res = await session.call_tool("resize", {"path": src, "width": 10})
        assert res.is_error and "overwrite=true" in res.content[0].text
        res = await session.call_tool("resize", {"path": str(tmp_path / "nope.png"), "width": 1})
        assert res.is_error and "file not found" in res.content[0].text
        res = await session.call_tool("compress", {"path": src, "target_size": "lots"})
        assert res.is_error and "cannot read target size" in res.content[0].text


@pytest.mark.anyio
async def test_dir_flag_sandboxes_the_server(tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    gradient(40, 40).save(allowed / "in.png")
    gradient(40, 40).save(tmp_path / "out.png")
    async with stdio_client(_params("--dir", str(allowed))) as (read, write), ClientSession(
        read, write
    ) as session:
        await session.initialize()
        res = await session.call_tool("image_info", {"path": str(allowed / "in.png")})
        assert not res.is_error
        res = await session.call_tool("image_info", {"path": str(tmp_path / "out.png")})
        assert res.is_error and "outside the directories" in res.content[0].text
        res = await session.call_tool(
            "image_info", {"path": str(allowed / ".." / "out.png")}
        )
        assert res.is_error and "outside the directories" in res.content[0].text


def test_cli_rejects_bad_dir():
    import subprocess

    p = subprocess.run(
        [sys.executable, "-m", "picture_mcp", "--dir", os.path.join("nope", "nowhere")],
        capture_output=True, text=True, timeout=60,
    )
    assert p.returncode != 0 and "not an existing directory" in p.stderr
