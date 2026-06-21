from __future__ import annotations

import json
from typing import Any

from services.tool_service import TOOL_REGISTRY, execute_tool


MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_SERVER_INFO = {
    "name": "reportgenie-mcp",
    "title": "ReportGenie AI MCP Server",
    "version": "1.0.0",
}


def _jsonrpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def mcp_initialize(request_id: Any, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return _jsonrpc_result(
        request_id,
        {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": MCP_SERVER_INFO,
            "capabilities": {
                "tools": {
                    "listChanged": False,
                }
            },
        },
    )


def mcp_list_tools(request_id: Any) -> dict[str, Any]:
    tools = [
        {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": tool.get("inputSchema", {"type": "object", "properties": {}}),
        }
        for tool in TOOL_REGISTRY.values()
    ]
    return _jsonrpc_result(request_id, {"tools": tools})


def mcp_call_tool(request_id: Any, params: dict[str, Any] | None, dataset: dict[str, Any]) -> dict[str, Any]:
    params = params or {}
    tool_name = params.get("name")
    arguments = params.get("arguments") or {}
    if tool_name not in TOOL_REGISTRY:
        return _jsonrpc_result(
            request_id,
            {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            },
        )

    try:
        result, transparency = execute_tool(tool_name, arguments, dataset)
        return _jsonrpc_result(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False),
                    }
                ],
                "structuredContent": result,
                "_meta": {"transparency": transparency},
                "isError": False,
            },
        )
    except Exception as exc:
        return _jsonrpc_result(
            request_id,
            {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            },
        )


def handle_mcp_request(payload: dict[str, Any], dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if payload.get("jsonrpc") != "2.0":
        return _jsonrpc_error(request_id, -32600, "Invalid Request")

    if method == "initialize":
        return mcp_initialize(request_id, params)
    if method == "tools/list":
        return mcp_list_tools(request_id)
    if method == "tools/call":
        if dataset is None:
            return _jsonrpc_error(request_id, -32000, "Dataset context is required for tools/call")
        return mcp_call_tool(request_id, params, dataset)
    return _jsonrpc_error(request_id, -32601, f"Method not found: {method}")


def mcp_client_initialize() -> dict[str, Any]:
    return handle_mcp_request({"jsonrpc": "2.0", "id": "init", "method": "initialize"})


def mcp_client_list_tools() -> list[dict[str, Any]]:
    response = handle_mcp_request({"jsonrpc": "2.0", "id": "tools-list", "method": "tools/list"})
    return response.get("result", {}).get("tools", [])


def mcp_client_call_tool(name: str, arguments: dict[str, Any], dataset: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    response = handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": f"tool-{name}",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        dataset=dataset,
    )
    result = response.get("result", {})
    if result.get("isError"):
        raise ValueError((result.get("content") or [{"text": "Tool call failed"}])[0]["text"])
    return result.get("structuredContent"), (result.get("_meta") or {}).get("transparency", {})
