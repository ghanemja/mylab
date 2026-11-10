"""The MCP server, exercised in-process with the MCP client."""

import asyncio

import pytest

mcp = pytest.importorskip("mcp")

from mylab import mcp_server  # noqa: E402


def _texts(result):
    return [c.text for c in result.content if getattr(c, "type", "") == "text"]


def test_tools_end_to_end(tmp_path):
    from mcp import Client

    mcp_server._session = None
    server = mcp_server.build_server()
    script = tmp_path / "make_y.m"
    script.write_text("y = 21 * 2;\n")

    async def scenario():
        async with Client(server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {"run_code", "run_file", "list_variables", "get_variable", "function_help",
                    "search_functions", "reset_session"} <= tools

            r = await client.call_tool("run_code", {"code": "x = magic(3);\ns = sum(x(:))"})
            assert "s = 45" in _texts(r)[0]

            r = await client.call_tool("run_code", {"code": "plot(1:5); title('t')"})
            assert any(getattr(c, "type", "") == "image" for c in r.content)

            r = await client.call_tool("run_code", {"code": "z = 1;\nundefined_thing"})
            assert "Error (line 2)" in _texts(r)[0]

            r = await client.call_tool("run_file", {"path": str(script)})
            r = await client.call_tool("get_variable", {"name": "y"})
            assert _texts(r)[0] == "y = 42"

            r = await client.call_tool("list_variables", {})
            assert "x | 3×3 | double" in _texts(r)[0]

            r = await client.call_tool("function_help", {"name": "linspace"})
            assert "linspace(a, b" in _texts(r)[0]

            r = await client.call_tool("search_functions", {"query": "interpolat"})
            assert "interp1" in _texts(r)[0]

            await client.call_tool("reset_session", {})
            r = await client.call_tool("list_variables", {})
            assert "empty" in _texts(r)[0]

    asyncio.run(scenario())
