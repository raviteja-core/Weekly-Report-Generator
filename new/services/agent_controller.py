from __future__ import annotations

import json
from typing import Any

from services.llm_service import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    ask_agent,
    build_chat_system_prompt,
    build_report_system_prompt,
    fallback_chat,
    fallback_report,
    report_text_needs_fact_upgrade,
)
from services.mcp_service import mcp_client_call_tool, mcp_client_initialize, mcp_client_list_tools


class AgentController:
    """
    Tool-augmented controller for report generation and follow-up chat.

    The model can request approved tools such as KPI and chart generation.
    Tool results are injected back into the conversation until the model
    returns a final structured payload or we fall back deterministically.
    """

    def __init__(self, max_iterations: int = 6):
        self.max_iterations = max_iterations

    def _dataset_context(self, dataset: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": dataset["schema"],
            "preview": dataset["preview"],
            "quality_report": dataset.get("quality_report", {}),
            "warnings": dataset.get("warnings", []),
        }

    def _append_tool_result(
        self,
        messages: list[dict[str, str]],
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        messages.append(
            {
                "role": "user",
                "content": "TOOL_RESULT\n"
                + json.dumps(
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result,
                    },
                    ensure_ascii=False,
                ),
            }
        )

    def _available_tools(self) -> dict[str, dict[str, Any]]:
        return {tool["name"]: tool for tool in mcp_client_list_tools()}

    def _normalize_report_payload(
        self,
        final_payload: dict[str, Any],
        tool_results: dict[str, Any],
        tool_log: list[dict[str, Any]],
        dataset: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        report = final_payload.get("report", {})
        chart_set = tool_results.get("generate_chart_set", {})
        charts = chart_set.get("charts") or []
        if not charts and tool_results.get("generate_chart"):
            charts = [tool_results["generate_chart"]]

        fallback = fallback_report(tool_results, self._dataset_context(dataset), prompt)
        warnings = list(dict.fromkeys(dataset.get("warnings", []) + report.get("warnings", []) + fallback.get("warnings", [])))
        use_fallback_story = report_text_needs_fact_upgrade(report)
        insights = fallback["insights"] if use_fallback_story else (report.get("insights") or fallback["insights"])
        story = fallback["story"] if use_fallback_story else (report.get("story") or fallback["story"])
        summary = fallback["summary"] if use_fallback_story else (report.get("summary") or fallback["summary"])

        return {
            "title": report.get("title") or fallback["title"],
            "summary": summary,
            "kpis": tool_results.get("calculate_kpis", {}),
            "charts": charts,
            "quality_report": dataset.get("quality_report", {}),
            "schema": dataset["schema"],
            "confidence": dataset.get("confidence"),
            "warnings": warnings,
            "preview": dataset["preview"],
            "records": dataset["records"],
            "insights": insights,
            "story": story,
            "suggested_questions": report.get("suggested_questions") or fallback["suggested_questions"],
            "meta": {
                "generation_mode": "ollama-mcp-agent",
                "ollama_host": OLLAMA_HOST,
                "ollama_model": OLLAMA_MODEL,
                "tools": list(self._available_tools().keys()),
                "protocol": "MCP",
            },
            "tools_used": tool_log,
        }

    def run_report(self, prompt: str, dataset: dict[str, Any]) -> dict[str, Any]:
        init_result = mcp_client_initialize()
        available_tools = self._available_tools()
        messages = [
            {"role": "system", "content": build_report_system_prompt(available_tools)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": prompt,
                        "dataset_context": self._dataset_context(dataset),
                        "mcp_session": init_result.get("result", {}),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        tool_results: dict[str, Any] = {}
        tool_log: list[dict[str, Any]] = []

        try:
            for _ in range(self.max_iterations):
                decision = ask_agent(messages)
                decision_type = decision.get("type")

                if decision_type == "tool_call":
                    tool_name = decision.get("tool_name")
                    arguments = decision.get("arguments") or {}
                    if tool_name not in available_tools:
                        raise ValueError(f"LLM requested unknown tool '{tool_name}'.")
                    result, transparency = mcp_client_call_tool(tool_name, arguments, dataset)
                    tool_results[tool_name] = result
                    tool_log.append(transparency)
                    self._append_tool_result(messages, tool_name, arguments, result)
                    continue

                if decision_type == "final":
                    return self._normalize_report_payload(decision, tool_results, tool_log, dataset, prompt)

            raise RuntimeError("Agent exceeded the maximum number of tool-calling iterations.")
        except Exception as exc:
            if "calculate_kpis" not in tool_results:
                result, transparency = mcp_client_call_tool("calculate_kpis", {"period": "D"}, dataset)
                tool_results["calculate_kpis"] = result
                tool_log.append(transparency)
            if "generate_chart_set" not in tool_results:
                result, transparency = mcp_client_call_tool("generate_chart_set", {"max_charts": 3}, dataset)
                tool_results["generate_chart_set"] = result
                tool_log.append(transparency)

            fallback = fallback_report(tool_results, self._dataset_context(dataset), prompt)
            return {
                "title": fallback["title"],
                "summary": fallback["summary"],
                "kpis": tool_results.get("calculate_kpis", {}),
                "charts": (tool_results.get("generate_chart_set") or {}).get("charts", []),
                "quality_report": dataset.get("quality_report", {}),
                "schema": dataset["schema"],
                "confidence": dataset.get("confidence"),
                "warnings": list(dict.fromkeys(dataset.get("warnings", []) + fallback.get("warnings", []))),
                "preview": dataset["preview"],
                "records": dataset["records"],
                "insights": fallback["insights"],
                "story": fallback["story"],
                "suggested_questions": fallback["suggested_questions"],
                "meta": {
                    "generation_mode": "fallback-mcp-agent",
                    "ollama_host": OLLAMA_HOST,
                    "ollama_model": OLLAMA_MODEL,
                    "tools": list(available_tools.keys()),
                    "protocol": "MCP",
                    "warnings": [str(exc)],
                },
                "tools_used": tool_log,
            }

    def run_chat(self, message: str, dataset: dict[str, Any], report_context: dict[str, Any]) -> dict[str, Any]:
        init_result = mcp_client_initialize()
        available_tools = self._available_tools()
        messages = [
            {"role": "system", "content": build_chat_system_prompt(available_tools)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": message,
                        "report_context": report_context,
                        "dataset_context": self._dataset_context(dataset),
                        "mcp_session": init_result.get("result", {}),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        tool_results: dict[str, Any] = {}
        tool_log: list[dict[str, Any]] = []

        try:
            for _ in range(self.max_iterations):
                decision = ask_agent(messages)
                decision_type = decision.get("type")

                if decision_type == "tool_call":
                    tool_name = decision.get("tool_name")
                    arguments = decision.get("arguments") or {}
                    if tool_name not in available_tools:
                        raise ValueError(f"LLM requested unknown tool '{tool_name}'.")
                    result, transparency = mcp_client_call_tool(tool_name, arguments, dataset)
                    tool_results[tool_name] = result
                    tool_log.append(transparency)
                    self._append_tool_result(messages, tool_name, arguments, result)
                    continue

                if decision_type == "final":
                    fallback = fallback_chat(message, report_context, tool_results)
                    return {
                        "answer": decision.get("answer") or fallback["answer"],
                        "follow_ups": decision.get("follow_ups") or fallback["follow_ups"],
                        "tools_used": tool_log,
                        "meta": {
                            "generation_mode": "ollama-mcp-agent",
                            "ollama_host": OLLAMA_HOST,
                            "ollama_model": OLLAMA_MODEL,
                            "protocol": "MCP",
                        },
                    }

            raise RuntimeError("Agent exceeded the maximum number of tool-calling iterations.")
        except Exception as exc:
            if "calculate_kpis" not in tool_results:
                result, transparency = mcp_client_call_tool("calculate_kpis", {"period": "D"}, dataset)
                tool_results["calculate_kpis"] = result
                tool_log.append(transparency)
            fallback = fallback_chat(message, report_context, tool_results)
            return {
                "answer": fallback["answer"],
                "follow_ups": fallback["follow_ups"],
                "tools_used": tool_log,
                "meta": {
                    "generation_mode": "fallback-mcp-agent",
                    "ollama_host": OLLAMA_HOST,
                    "ollama_model": OLLAMA_MODEL,
                    "protocol": "MCP",
                    "warnings": [str(exc)],
                },
            }
