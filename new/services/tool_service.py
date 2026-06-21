from __future__ import annotations

from typing import Any

import pandas as pd

from services.chart_service import generate_chart, generate_charts, suggest_chart_specs
from services.kpi_service import calculate_kpis


TOOL_REGISTRY = {
    "calculate_kpis": {
        "name": "calculate_kpis",
        "description": "Compute KPIs only from the already-cleaned canonical dataframe.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Pandas Grouper frequency, usually D or W.",
                    "default": "D",
                }
            },
        },
    },
    "generate_chart": {
        "name": "generate_chart",
        "description": "Generate a chart from the same cleaned dataframe used for KPI computation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "pie"],
                    "default": "line",
                },
                "x_field": {"type": ["string", "null"]},
                "y_field": {"type": "string", "default": "revenue"},
                "color_field": {"type": ["string", "null"]},
            },
        },
    },
    "generate_chart_set": {
        "name": "generate_chart_set",
        "description": "Generate a small set of recommended charts from the cleaned dataframe.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_charts": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "default": 3,
                }
            },
        },
    },
}


def _get_dataframe(dataset: dict[str, Any]) -> pd.DataFrame:
    dataframe = dataset.get("dataframe")
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Active dataset is missing a dataframe.")
    return dataframe


def execute_tool(name: str, arguments: dict[str, Any], dataset: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    df = _get_dataframe(dataset)
    if name == "calculate_kpis":
        result = calculate_kpis(df, period=arguments.get("period", "D"))
    elif name == "generate_chart":
        result = generate_chart(
            df,
            chart_type=arguments.get("chart_type", "line"),
            x_field=arguments.get("x_field"),
            y_field=arguments.get("y_field", "revenue"),
            color_field=arguments.get("color_field"),
        )
    elif name == "generate_chart_set":
        max_charts = int(arguments.get("max_charts", 3))
        result = {
            "charts": generate_charts(df, max_charts=max_charts),
            "recommendations": suggest_chart_specs(df, max_charts=max_charts),
        }
    else:
        raise ValueError(f"Unknown tool: {name}")

    transparency = {"name": name, "inputs": arguments, "output_summary": result}
    return result, transparency
