import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from picture_mcp import paths  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_config():
    """Every test starts unrestricted with the default limit."""
    paths.configure([], 100_000_000)
    yield
    paths.configure([], 100_000_000)


@pytest.fixture
def anyio_backend():
    return "asyncio"
