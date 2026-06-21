from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from services.csv_service import load_sales_csv


LOGGER = logging.getLogger(__name__)

ANOMALY_ZSCORE_THRESHOLD = 1.5
MIN_PERIODS_FOR_GROWTH = 2
DEFAULT_PERIOD = "D"


def _format_period(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _validate_kpi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "revenue"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Clean dataframe is missing required columns: {sorted(missing)}")

    working = df.copy()
    working = working.dropna(subset=["date", "revenue"]).sort_values("date").reset_index(drop=True)
    if working.empty:
        raise ValueError("No valid date and revenue rows remain after cleaning.")
    return working


def _detect_anomalies(series_df: pd.DataFrame) -> dict[str, Any]:
    if len(series_df) < 3:
        return {"count": 0, "dates": [], "z_scores": {}}

    revenue = series_df["revenue"].astype(float)
    std = float(revenue.std(ddof=0))
    if std == 0:
        return {"count": 0, "dates": [], "z_scores": {}}

    mean = float(revenue.mean())
    z_scores = ((revenue - mean) / std).abs()
    rolling_mean = revenue.rolling(window=3, min_periods=2).mean()
    rolling_std = revenue.rolling(window=3, min_periods=2).std(ddof=0).replace(0, np.nan)
    rolling_deviation = ((revenue - rolling_mean).abs() / rolling_std).fillna(0)
    anomaly_mask = (z_scores > ANOMALY_ZSCORE_THRESHOLD) | (rolling_deviation > ANOMALY_ZSCORE_THRESHOLD)
    anomaly_rows = series_df.loc[anomaly_mask, ["date", "revenue"]]

    return {
        "count": int(len(anomaly_rows)),
        "dates": anomaly_rows["date"].dt.strftime("%Y-%m-%d").tolist(),
        "z_scores": {
            row["date"].strftime("%Y-%m-%d"): round(float(score), 4)
            for (_, row), score in zip(anomaly_rows.iterrows(), z_scores.loc[anomaly_mask])
        },
    }


def calculate_kpis(df: str | pd.DataFrame, period: str = DEFAULT_PERIOD) -> dict[str, Any]:
    """Compute trusted KPIs from the already-cleaned canonical dataframe."""
    LOGGER.info("Calculating KPIs")
    source_df = load_sales_csv(df) if isinstance(df, str) else df.copy()
    working = _validate_kpi_dataframe(source_df)

    grouped = (
        working.groupby(pd.Grouper(key="date", freq=period), dropna=True)["revenue"]
        .sum()
        .reset_index()
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if grouped.empty:
        raise ValueError("No time-series revenue data available for KPI calculation.")

    total_revenue = round(float(working["revenue"].sum()), 2)
    average_revenue = round(float(working["revenue"].mean()), 2)

    growth_rate = 0.0
    if len(grouped) >= MIN_PERIODS_FOR_GROWTH:
        previous_revenue = float(grouped.iloc[-2]["revenue"])
        current_revenue = float(grouped.iloc[-1]["revenue"])
        growth_rate = round(((current_revenue - previous_revenue) / previous_revenue) * 100, 2) if previous_revenue else 0.0

    anomalies = _detect_anomalies(grouped)
    revenue_series = [
        {"date": _format_period(row["date"]), "revenue": round(float(row["revenue"]), 2)}
        for _, row in grouped.iterrows()
    ]

    result = {
        "total_revenue": total_revenue,
        "growth_rate": growth_rate,
        "average_revenue": average_revenue,
        "anomalies": anomalies,
        "series": revenue_series,
        "row_count": int(len(working)),
        "period_count": int(len(grouped)),
    }
    LOGGER.info("KPI calculation complete: %s", result)
    return result
