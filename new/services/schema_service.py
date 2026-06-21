from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from services.data_quality import clean_column_name, parse_numeric_series


LOGGER = logging.getLogger(__name__)

DATE_HINTS = {"date", "day", "week", "month", "quarter", "year", "order", "invoice", "transaction"}
REVENUE_HINTS = {"revenue", "sales", "sale", "amount", "total", "income", "gmv", "gross", "net"}
PRICE_HINTS = {"price", "unit", "rate", "ticket"}
QUANTITY_HINTS = {"qty", "quantity", "units", "volume", "count"}
CATEGORY_HINTS = {"category", "segment", "department", "channel", "product", "line", "division"}
REGION_HINTS = {"region", "market", "territory", "state", "city", "country", "area"}

SCHEMA_CONFIDENCE_THRESHOLD = 0.8


def _tokenize(column: str) -> set[str]:
    return set(clean_column_name(column).split())


def _name_bonus(column: str, hints: set[str]) -> float:
    return min(0.25, 0.12 * len(_tokenize(column).intersection(hints)))


def _date_candidate_score(series: pd.Series, column: str) -> tuple[pd.Series, float]:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    parse_ratio = float(parsed.notna().mean())
    if parse_ratio < 0.6:
        return parsed, 0.0
    score = min(1.0, parse_ratio * 0.85 + _name_bonus(column, DATE_HINTS))
    return parsed, round(score, 4)


def _numeric_candidate_score(series: pd.Series, column: str, hints: set[str]) -> tuple[pd.Series, float]:
    parsed = parse_numeric_series(series)
    numeric_ratio = float(parsed.notna().mean())
    if numeric_ratio < 0.7:
        return parsed, 0.0
    score = min(1.0, numeric_ratio * 0.8 + _name_bonus(column, hints))
    return parsed, round(score, 4)


def _low_cardinality_score(series: pd.Series, column: str, hints: set[str]) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    distinct = non_null.astype(str).nunique()
    distinct_ratio = distinct / max(len(non_null), 1)
    if distinct < 2 or distinct_ratio > 0.6:
        return 0.0
    score = min(1.0, (1.0 - distinct_ratio) * 0.75 + _name_bonus(column, hints))
    return round(score, 4)


def _best_candidate(scores: dict[str, float]) -> tuple[str | None, float]:
    if not scores:
        return None, 0.0
    column = max(scores, key=scores.get)
    return column, scores[column]


def _find_price_quantity_candidates(df: pd.DataFrame) -> tuple[str | None, str | None]:
    price_scores: dict[str, float] = {}
    quantity_scores: dict[str, float] = {}
    for column in df.columns:
        _, price_score = _numeric_candidate_score(df[column], column, PRICE_HINTS)
        _, quantity_score = _numeric_candidate_score(df[column], column, QUANTITY_HINTS)
        if price_score > 0:
            price_scores[column] = price_score
        if quantity_score > 0:
            quantity_scores[column] = quantity_score
    price_col, _ = _best_candidate(price_scores)
    quantity_col, _ = _best_candidate(quantity_scores)
    return price_col, quantity_col


def _revenue_consistency_bonus(
    df: pd.DataFrame,
    revenue_column: str,
    price_column: str | None,
    quantity_column: str | None,
) -> float:
    if not price_column or not quantity_column:
        return 0.0

    revenue = parse_numeric_series(df[revenue_column])
    price = parse_numeric_series(df[price_column])
    quantity = parse_numeric_series(df[quantity_column])
    comparison = pd.DataFrame({"revenue": revenue, "calc": price * quantity}).dropna()
    if len(comparison) < 3:
        return 0.0

    if comparison["revenue"].std() == 0 or comparison["calc"].std() == 0:
        return 0.0

    correlation = comparison["revenue"].corr(comparison["calc"])
    if pd.isna(correlation):
        return 0.0
    return round(max(0.0, min(0.2, float(correlation) * 0.2)), 4)


def detect_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Strictly detect analytical columns without LLM guessing."""
    LOGGER.info("Detecting schema for dataframe with columns: %s", list(df.columns))

    date_scores: dict[str, float] = {}
    revenue_scores: dict[str, float] = {}
    category_scores: dict[str, float] = {}
    region_scores: dict[str, float] = {}

    parsed_dates: dict[str, pd.Series] = {}

    price_column, quantity_column = _find_price_quantity_candidates(df)

    for column in df.columns:
        parsed_date, date_score = _date_candidate_score(df[column], column)
        if date_score > 0:
            date_scores[column] = date_score
            parsed_dates[column] = parsed_date

        _, revenue_score = _numeric_candidate_score(df[column], column, REVENUE_HINTS)
        if revenue_score > 0:
            revenue_scores[column] = min(
                1.0,
                revenue_score + _revenue_consistency_bonus(df, column, price_column, quantity_column),
            )

        category_score = _low_cardinality_score(df[column], column, CATEGORY_HINTS)
        if category_score > 0:
            category_scores[column] = category_score

        region_score = _low_cardinality_score(df[column], column, REGION_HINTS)
        if region_score > 0:
            region_scores[column] = region_score

    date_col, date_confidence = _best_candidate(date_scores)
    revenue_col, revenue_confidence = _best_candidate(revenue_scores)
    category_col, category_confidence = _best_candidate(category_scores)
    region_col, region_confidence = _best_candidate(
        {column: score for column, score in region_scores.items() if column != category_col}
    )

    confidence = round(
        (date_confidence * 0.4)
        + (revenue_confidence * 0.4)
        + (category_confidence * 0.1 if category_col else 0.0)
        + (region_confidence * 0.1 if region_col else 0.0),
        4,
    )

    schema = {
        "date_col": date_col,
        "revenue_col": revenue_col,
        "category_col": category_col,
        "region_col": region_col,
        "confidence": confidence,
        "price_col": price_column,
        "quantity_col": quantity_column,
        "column_confidence": {
            "date": date_confidence,
            "revenue": revenue_confidence,
            "category": category_confidence,
            "region": region_confidence,
        },
    }
    LOGGER.info("Schema detection result: %s", schema)
    return schema
