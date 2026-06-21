from __future__ import annotations

import json
import logging
import os
from urllib import error, request


LOGGER = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "20"))


def _extract_json(content: str) -> dict | None:
    try:
        trimmed = content.strip()
        return json.loads(trimmed[trimmed.index("{") : trimmed.rfind("}") + 1])
    except Exception:
        return None


def _post_ollama_chat(messages: list[dict], response_format: str = "json") -> dict:
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "format": response_format,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Unable to reach Ollama for agent generation.") from exc


def ask_agent(messages: list[dict]) -> dict:
    response = _post_ollama_chat(messages, response_format="json")
    content = (response.get("message") or {}).get("content", "")
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        raise ValueError("Ollama returned invalid JSON for agent orchestration.")
    return parsed


def _format_currency(value) -> str:
    if value is None:
        return "unavailable"
    return "${:,.2f}".format(float(value))


def _format_percent(value) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.2f}%"


def _top_chart_fact(chart: dict) -> str | None:
    data_points = chart.get("data_points") or []
    if not data_points:
        return None

    first_point = data_points[0]
    if chart.get("type") == "line":
        if len(data_points) < 2:
            return None
        latest = data_points[-1]
        earliest = data_points[0]
        return (
            f"The trend chart runs from {earliest.get('date')} to {latest.get('date')}, "
            f"with the latest plotted revenue at {_format_currency(latest.get('revenue'))}."
        )

    dimension_fields = [key for key in first_point.keys() if key != "revenue"]
    if not dimension_fields:
        return None

    dimension = dimension_fields[0]
    leader = first_point.get(dimension)
    leader_value = _format_currency(first_point.get("revenue"))
    return f"The {chart.get('title', 'chart').lower()} is led by {leader} at {leader_value}."


def _build_fact_summary(kpis: dict, charts: list[dict], quality_report: dict, warnings: list[str]) -> str:
    series = kpis.get("series") or []
    latest_period = series[-1]["date"] if series else "the latest period"
    total_revenue = _format_currency(kpis.get("total_revenue"))
    average_revenue = _format_currency(kpis.get("average_revenue"))
    growth_rate = _format_percent(kpis.get("growth_rate"))
    anomaly_count = kpis.get("anomalies", {}).get("count", 0)

    chart_fact = next((fact for fact in (_top_chart_fact(chart) for chart in charts) if fact), None)
    quality_score = quality_report.get("quality_score")

    sentences = [
        f"Total revenue is {total_revenue} across {kpis.get('row_count', 0)} cleaned rows, with average revenue of {average_revenue}.",
        f"The latest grouped period is {latest_period}, and its revenue change versus the previous period is {growth_rate}.",
        (
            f"{anomaly_count} anomaly dates were detected in the grouped revenue series."
            if anomaly_count
            else "No anomaly dates were detected in the grouped revenue series."
        ),
    ]
    if chart_fact:
        sentences.append(chart_fact)
    if quality_score is not None:
        sentences.append(f"Dataset quality score is {quality_score}.")
    elif warnings:
        sentences.append(warnings[0])
    return " ".join(sentences)


def _build_fact_insights(kpis: dict, charts: list[dict], quality_report: dict, warnings: list[str]) -> list[dict]:
    series = kpis.get("series") or []
    latest_period = series[-1]["date"] if series else "latest period"
    anomalies = kpis.get("anomalies", {})
    anomaly_dates = anomalies.get("dates", [])
    chart_fact = next((fact for fact in (_top_chart_fact(chart) for chart in charts) if fact), None)
    quality_score = quality_report.get("quality_score")
    actionable_warnings = [warning for warning in warnings if warning and "dataset quality score" not in warning.lower()]

    return [
        {
            "title": "Revenue Baseline",
            "body": (
                f"Total revenue reached {_format_currency(kpis.get('total_revenue'))}, "
                f"while average revenue per cleaned row was {_format_currency(kpis.get('average_revenue'))}."
            ),
            "tone": "neutral",
        },
        {
            "title": "Recent Movement",
            "body": (
                f"The latest grouped period ({latest_period}) recorded a growth rate of {_format_percent(kpis.get('growth_rate'))}. "
                + (
                    f"{anomalies.get('count', 0)} anomaly dates were flagged, including {', '.join(anomaly_dates[:3])}."
                    if anomalies.get("count", 0)
                    else "No anomaly dates were flagged in the grouped revenue trend."
                )
            ),
            "tone": "positive" if (kpis.get("growth_rate") or 0) >= 0 else "caution",
        },
        {
            "title": "Chart Readout",
            "body": chart_fact or "Chart outputs are available, but no ranked category or region takeaway was available from the current chart set.",
            "tone": "neutral" if chart_fact else "caution",
        },
        {
            "title": "Data Quality",
            "body": (
                actionable_warnings[0]
                if actionable_warnings
                else (
                    f"Dataset quality score is {quality_score} with no major validation warnings."
                    if quality_score is not None
                    else "No major validation warnings were raised."
                )
            ),
            "tone": "caution" if actionable_warnings else "neutral",
        },
    ]


def _build_fact_story(kpis: dict, charts: list[dict], quality_report: dict, warnings: list[str]) -> list[dict]:
    series = kpis.get("series") or []
    latest_period = series[-1]["date"] if series else "latest period"
    anomaly_dates = kpis.get("anomalies", {}).get("dates", [])
    anomaly_count = kpis.get("anomalies", {}).get("count", 0)
    chart_fact = next((fact for fact in (_top_chart_fact(chart) for chart in charts) if fact), None)
    quality_score = quality_report.get("quality_score")
    actionable_warnings = [warning for warning in warnings if warning and "dataset quality score" not in warning.lower()]
    primary_chart_id = charts[0]["id"] if charts else None
    secondary_chart_id = charts[1]["id"] if len(charts) > 1 else primary_chart_id

    return [
        {
            "id": "story-overview",
            "eyebrow": "Overview",
            "headline": "Revenue, growth, and chart outputs line up on the same cleaned report data.",
            "copy": (
                f"The report covers {kpis.get('row_count', 0)} cleaned rows with total revenue of {_format_currency(kpis.get('total_revenue'))} "
                f"and average revenue of {_format_currency(kpis.get('average_revenue'))}. "
                + (chart_fact or "The chart set reflects the same cleaned data used for the KPI totals.")
            ),
            "chart_id": primary_chart_id,
        },
        {
            "id": "story-trend",
            "eyebrow": "Trend",
            "headline": "Recent movement should be read through the grouped revenue series.",
            "copy": (
                f"In the latest grouped period ({latest_period}), revenue changed by {_format_percent(kpis.get('growth_rate'))} versus the previous period. "
                + (
                    f"{anomaly_count} anomaly dates were detected, including {', '.join(anomaly_dates[:3])}."
                    if anomaly_count
                    else "No anomaly dates were detected in the grouped revenue pattern."
                )
            ),
            "chart_id": primary_chart_id,
        },
        {
            "id": "story-risk",
            "eyebrow": "Risk",
            "headline": "Data quality and exceptions still shape how confidently the report should be read.",
            "copy": (
                actionable_warnings[0]
                if actionable_warnings
                else (
                    f"Dataset quality score is {quality_score}. No major validation warnings were raised in the trust checks."
                    if quality_score is not None
                    else "No major validation warnings were raised in the trust checks."
                )
            ),
            "chart_id": secondary_chart_id,
        },
    ]


def report_text_needs_fact_upgrade(report: dict) -> bool:
    story = report.get("story") or []
    story_text = " ".join(
        str(part)
        for chapter in story
        for part in [chapter.get("headline", ""), chapter.get("copy", "")]
    ).lower()
    summary = str(report.get("summary", "")).lower()
    generic_markers = [
        "the agent generated",
        "approved analytics tools",
        "tool-augmented",
        "the model writes",
        "this report is grounded",
    ]
    return any(marker in story_text or marker in summary for marker in generic_markers)


def build_report_system_prompt(tool_registry: dict) -> str:
    return f"""
You are ReportGenie AI writing a business report for end readers.
You must call approved tools for KPI or chart facts instead of inventing them.
Write about the report findings, not about yourself, the agent, the prompt, or tool orchestration.

Available tools:
{json.dumps(tool_registry, indent=2)}

Rules:
- Never invent metrics, chart findings, or row counts.
- Call calculate_kpis before making KPI claims.
- Call generate_chart_set when the report needs visuals.
- Use only information from dataset context or tool results.
- Mention data-quality warnings when they exist.
- Do not say "the agent generated this report", "approved analytics tools", or similar system narration.
- Every story section must include concrete report facts such as revenue, growth, anomaly counts, row counts, quality score, or a chart takeaway.
- Prefer specific metric values over generic phrases like "grounded in tools" or "summarized from results".
- Return valid JSON only.

Decision protocol:
1. If you need a tool, return:
{{
  "type": "tool_call",
  "tool_name": "exact tool name",
  "arguments": {{ ... }},
  "reason": "short reason"
}}

2. When ready to finish, return:
{{
  "type": "final",
  "report": {{
    "title": "short title",
    "summary": "2-4 sentence executive summary grounded in specific KPI and chart outputs",
    "insights": [
      {{"title": "short", "body": "fact-rich insight with actual values", "tone": "positive|neutral|caution"}},
      {{"title": "short", "body": "fact-rich insight with actual values", "tone": "positive|neutral|caution"}},
      {{"title": "short", "body": "fact-rich insight with actual values", "tone": "positive|neutral|caution"}}
    ],
    "story": [
      {{"id": "story-overview", "eyebrow": "Overview", "headline": "report-focused headline", "copy": "2-3 sentences with actual KPI or chart facts", "chart_id": "chart id or null"}},
      {{"id": "story-trend", "eyebrow": "Trend", "headline": "report-focused headline", "copy": "2-3 sentences with actual KPI or chart facts", "chart_id": "chart id or null"}},
      {{"id": "story-risk", "eyebrow": "Risk", "headline": "report-focused headline", "copy": "2-3 sentences with actual KPI or chart facts", "chart_id": "chart id or null"}}
    ],
    "suggested_questions": ["question", "question", "question"],
    "warnings": ["warning"]
  }}
}}
"""


def build_chat_system_prompt(tool_registry: dict) -> str:
    return f"""
You are ReportGenie AI chat.
Use tools when the user asks for KPI-backed or chart-backed analysis.

Available tools:
{json.dumps(tool_registry, indent=2)}

Rules:
- Use only facts supported by report context or tool results.
- If the answer depends on fresh metrics, call a tool first.
- Mention data quality limitations when relevant.
- Return valid JSON only.

Decision protocol:
1. Tool request:
{{
  "type": "tool_call",
  "tool_name": "exact tool name",
  "arguments": {{ ... }},
  "reason": "short reason"
}}

2. Final answer:
{{
  "type": "final",
  "answer": "concise answer grounded in report/tool data",
  "follow_ups": ["question", "question", "question"]
}}
"""


def fallback_report(tool_results: dict, dataset_context: dict, prompt: str) -> dict:
    kpis = tool_results.get("calculate_kpis", {})
    chart_bundle = tool_results.get("generate_chart_set", {})
    charts = chart_bundle.get("charts") or []
    quality_report = dataset_context.get("quality_report", {})
    warnings = list(dataset_context.get("warnings", []))

    total_revenue = kpis.get("total_revenue")
    growth_rate = kpis.get("growth_rate")
    average_revenue = kpis.get("average_revenue")
    anomaly_dates = kpis.get("anomalies", {}).get("dates", [])
    anomaly_count = kpis.get("anomalies", {}).get("count", 0)

    duplicate_percent = quality_report.get("duplicate_percent", 0.0)
    if duplicate_percent > 0:
        warnings.append(f"Duplicate rate after validation: {round(duplicate_percent * 100, 2)}%.")
    warnings = list(dict.fromkeys(warnings))

    anomaly_sentence = (
        f"{anomaly_count} anomaly dates were detected, including {', '.join(anomaly_dates[:3])}."
        if anomaly_count
        else "No anomaly dates were detected in the grouped revenue series."
    )
    quality_sentence = warnings[0] if warnings else "No major validation warnings were raised."
    summary = _build_fact_summary(kpis, charts, quality_report, warnings)
    insights = _build_fact_insights(kpis, charts, quality_report, warnings)
    story = _build_fact_story(kpis, charts, quality_report, warnings)

    return {
        "title": "ReportGenie AI KPI Report",
        "summary": summary,
        "insights": insights,
        "story": story,
        "suggested_questions": [
            "Which anomaly date should I inspect first?",
            "Do you want a category breakdown?",
            "Should I compare the latest period against the previous one?",
        ],
        "warnings": warnings,
    }


def fallback_chat(message: str, report_context: dict, tool_results: dict) -> dict:
    kpis = tool_results.get("calculate_kpis") or report_context.get("kpis") or {}
    quality_report = report_context.get("quality_report") or {}
    answer = (
        f"Total revenue is {kpis.get('total_revenue')}, growth rate is {kpis.get('growth_rate')}%, "
        f"average revenue is {kpis.get('average_revenue')}, and anomaly dates are {kpis.get('anomalies', {}).get('dates', [])}."
    )
    if quality_report.get("duplicate_percent", 0) > 0:
        answer += f" Duplicate rate after validation was {round(quality_report['duplicate_percent'] * 100, 2)}%."
    return {
        "answer": answer,
        "follow_ups": [
            "Which anomaly date should I inspect first?",
            "Do you want a category or region breakdown?",
            "Should I regenerate charts for a different view?",
        ],
    }


def _report_prompt(prompt: str, kpis: dict, quality_report: dict, schema: dict, warnings: list[str]) -> str:
    return f"""
You are a trust-first analytics narrator.
Use ONLY the provided metrics. Do not invent or estimate values.
Do not recompute numbers.
If a field is missing, say it is unavailable.
Do not describe the system, agent, prompt, or tool process.
Write as if this is the final report being read by a stakeholder.

Return valid JSON only:
{{
  "title": "short title",
  "summary": "2-4 sentences grounded only in the metrics, with specific values",
  "warnings": ["warning"]
}}

User request:
{prompt}

Schema:
{json.dumps(schema, indent=2)}

Quality report:
{json.dumps(quality_report, indent=2)}

Warnings:
{json.dumps(warnings, indent=2)}

Structured KPIs:
{json.dumps(kpis, indent=2)}
"""


def generate_report_narrative(
    prompt: str,
    kpis: dict,
    quality_report: dict,
    schema: dict,
    warnings: list[str],
) -> dict:
    LOGGER.info("Generating report narrative from structured KPI payload")
    try:
        response = _post_ollama_chat(
            [
                {"role": "system", "content": "Return only valid JSON. Use ONLY the provided metrics. Do not invent or estimate values."},
                {"role": "user", "content": _report_prompt(prompt, kpis, quality_report, schema, warnings)},
            ]
        )
        content = (response.get("message") or {}).get("content", "")
        parsed = _extract_json(content)
        if isinstance(parsed, dict):
            return parsed
    except RuntimeError:
        LOGGER.warning("Ollama unavailable, using deterministic narrative fallback")
    return fallback_report(
        {"calculate_kpis": kpis},
        {"quality_report": quality_report, "schema": schema, "warnings": warnings},
        prompt,
    )
