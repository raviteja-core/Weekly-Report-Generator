from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio

from services.csv_service import load_sales_csv


LOGGER = logging.getLogger(__name__)

CHART_THEME = {
    "paper_bgcolor": "rgba(7, 10, 24, 0)",
    "plot_bgcolor": "rgba(7, 10, 24, 0)",
    "font_color": "#e5eefb",
    "grid": "rgba(148, 163, 184, 0.14)",
}


def _figure_payload(fig) -> dict[str, Any]:
    payload = json.loads(pio.to_json(fig, pretty=False))
    payload["layout"]["paper_bgcolor"] = CHART_THEME["paper_bgcolor"]
    payload["layout"]["plot_bgcolor"] = CHART_THEME["plot_bgcolor"]
    payload["layout"]["font"] = {"color": CHART_THEME["font_color"], "family": "Inter, sans-serif"}
    return payload


def _clean_dataframe(data: str | pd.DataFrame) -> pd.DataFrame:
    df = load_sales_csv(data) if isinstance(data, str) else data.copy()
    if "revenue" not in df.columns or "date" not in df.columns:
        raise ValueError("Charts require the cleaned canonical dataframe with date and revenue columns.")
    return df.dropna(subset=["date", "revenue"]).copy()


def generate_chart(
    data: str | pd.DataFrame,
    chart_type: str,
    x_field: str | None = None,
    y_field: str = "revenue",
    color_field: str | None = None,
) -> dict[str, Any]:
    LOGGER.info("Generating %s chart", chart_type)
    df = _clean_dataframe(data)
    chart_type = chart_type.lower()

    if chart_type == "line":
        timeline = (
            df.groupby(pd.Grouper(key="date", freq="D"))[y_field]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        timeline["date"] = timeline["date"].dt.strftime("%Y-%m-%d")
        fig = px.line(timeline, x="date", y=y_field, markers=True, title="Revenue Trend")
        return {
            "id": "revenue-trend",
            "type": "line",
            "title": "Revenue Trend",
            "figure": _figure_payload(fig),
            "data_points": timeline.to_dict(orient="records"),
        }

    field = x_field or color_field
    if not field or field not in df.columns:
        raise ValueError("Requested chart dimension is not available on the cleaned dataframe.")

    grouped = (
        df.groupby(field, dropna=False)[y_field]
        .sum()
        .reset_index()
        .sort_values(y_field, ascending=False)
        .head(10)
    )
    grouped[field] = grouped[field].fillna("Unknown").astype(str)

    if chart_type == "bar":
        fig = px.bar(grouped, x=field, y=y_field, color=field, title=f"{field.title()} Revenue")
    elif chart_type == "pie":
        fig = px.pie(grouped, names=field, values=y_field, title=f"{field.title()} Revenue Mix", hole=0.5)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    return {
        "id": f"{field}-{chart_type}",
        "type": chart_type,
        "title": fig.layout.title.text,
        "figure": _figure_payload(fig),
        "data_points": grouped.to_dict(orient="records"),
    }


def suggest_chart_specs(data: str | pd.DataFrame, max_charts: int = 3) -> list[dict[str, Any]]:
    df = _clean_dataframe(data)
    specs = [{"chart_type": "line", "x_field": "date", "y_field": "revenue"}]
    for field in ["category", "region"]:
        if field in df.columns and df[field].dropna().nunique() >= 2:
            specs.append({"chart_type": "bar", "x_field": field, "y_field": "revenue"})
        if len(specs) >= max_charts:
            break
    return specs[:max_charts]


def generate_charts(data: str | pd.DataFrame, max_charts: int = 3) -> list[dict[str, Any]]:
    df = _clean_dataframe(data)
    charts: list[dict[str, Any]] = []
    for spec in suggest_chart_specs(df, max_charts=max_charts):
        charts.append(generate_chart(df, chart_type=spec["chart_type"], x_field=spec.get("x_field"), y_field=spec["y_field"]))
    return charts
