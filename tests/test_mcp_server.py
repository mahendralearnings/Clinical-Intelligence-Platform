"""
MCP server structure tests - free, no AI or network calls.
Confirms the tool is correctly registered with the right name,
without actually running searches.
"""

import pytest

from clinical_platform.mcp_server.clinical_mcp_server import mcp


@pytest.mark.asyncio
async def test_search_tool_is_registered() -> None:
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "search_clinical_documents" in tool_names


@pytest.mark.asyncio
async def test_search_tool_has_description() -> None:
    tools = await mcp.list_tools()
    search_tool = next(t for t in tools if t.name == "search_clinical_documents")
    assert "clinical document" in search_tool.description.lower()