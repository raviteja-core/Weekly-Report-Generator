from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)

NUMERIC_NAME_HINTS = {
    "revenue",
    "sales",
    "sale",
    "amount",
    "total",
    "price",
    "cost",
    "profit",
    "qty",
    "quantity",
    "volume",
    "discount",
    "units",
}
REVENUE_NAME_HINTS = {"revenue", "sales", "sale", "amount", "total", "gmv", "income"}
LOW_CARDINALITY_RATIO = 0.2
NUMERIC_CANDIDATE_RATIO = 0.6


def blank_mask(series: pd.Series) -> pd.Series:
    """Return a mask for null-like values including blank strings."""
    return series.isna() | series.astype(str).str.strip().eq("")


def normalize_text(series: pd.Series) -> pd.Series:
    """Trim and collapse whitespace while preserving null semantics."""
    normalized = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return normalized.replace("", pd.NA)


def normalize_category_labels(series: pd.Series) -> pd.Series:
    """Normalize categorical labels to a stable title-cased representation."""
    return normalize_text(series).str.lower().str.title()


def parse_numeric_series(series: pd.Series) -> pd.Series:
    """Parse loosely formatted numeric strings into floats."""
    cleaned = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA, "NULL": pd.NA})
    )
    normalized = (
        cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.replace(r"[^0-9.\-+eE]", "", regex=True)
        .replace({"": pd.NA, "-": pd.NA, "+": pd.NA})
    )
    return pd.to_numeric(normalized, errors="coerce")


def clean_column_name(column: str) -> str:
    normalized = re.sub(r"[_\-]+", " ", str(column).strip().lower())
    return re.sub(r"\s+", " ", normalized)


def make_unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for column in columns:
        if column in seen:
            seen[column] += 1
            result.append(f"{column}_{seen[column]}")
        else:
            seen[column] = 0
            result.append(column)
    return result


def _is_low_cardinality(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    distinct = non_null.nunique()
    return 2 <= distinct <= max(12, int(len(non_null) * LOW_CARDINALITY_RATIO))


def _numeric_candidate_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns: list[str] = []
    for column in df.columns:
        cleaned_name = clean_column_name(column)
        parsed = parse_numeric_series(df[column])
        numeric_ratio = float(parsed.notna().mean())
        name_tokens = set(cleaned_name.split())
        if numeric_ratio >= NUMERIC_CANDIDATE_RATIO or name_tokens.intersection(NUMERIC_NAME_HINTS):
            numeric_columns.append(column)
    return numeric_columns


def _revenue_like_columns(df: pd.DataFrame) -> list[str]:
    matches: list[str] = []
    for column in df.columns:
        cleaned_name = clean_column_name(column)
        if set(cleaned_name.split()).intersection(REVENUE_NAME_HINTS):
            matches.append(column)
    return matches


def _quality_score(
    row_count: int,
    duplicate_percent: float,
    missing_values: dict[str, int],
    invalid_numeric: dict[str, int],
    negative_revenue: int,
) -> float:
    total_missing = sum(missing_values.values())
    total_invalid = sum(invalid_numeric.values())
    denominator = max(row_count, 1)

    score = 1.0
    score -= min(0.35, duplicate_percent * 1.5)
    score -= min(0.25, total_missing / denominator * 0.8)
    score -= min(0.25, total_invalid / denominator)
    score -= min(0.15, negative_revenue / denominator)
    return round(max(0.0, min(1.0, score)), 4)


def validate_and_clean_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate and normalize a raw CSV dataframe before schema detection.

    The same cleaned dataframe must be reused for schema inference, KPI math,
    and chart generation so row counts remain consistent.
    """
    LOGGER.info("Starting data validation and cleaning for %s rows", len(df))

    if df.empty:
        raise ValueError("CSV contains no data rows.")

    working = df.copy()
    working.columns = make_unique_columns([clean_column_name(column) for column in working.columns])
    working = working.dropna(axis=1, how="all")
    working = working.dropna(axis=0, how="all").reset_index(drop=True)

    if working.empty:
        raise ValueError("CSV contains no usable rows after removing empty rows and columns.")

    standardized = working.copy()
    for column in standardized.columns:
        series = standardized[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = normalize_text(series)
            if _is_low_cardinality(normalized):
                standardized[column] = normalize_category_labels(normalized)
            else:
                standardized[column] = normalized

    original_row_count = len(standardized)
    duplicate_rows = int(standardized.duplicated().sum())
    duplicate_percent = round(duplicate_rows / max(original_row_count, 1), 4)
    deduplicated = standardized.drop_duplicates().reset_index(drop=True)

    missing_values = {column: int(blank_mask(deduplicated[column]).sum()) for column in deduplicated.columns}

    invalid_numeric: dict[str, int] = {}
    for column in _numeric_candidate_columns(deduplicated):
        source = deduplicated[column]
        parsed = parse_numeric_series(source)
        invalid_numeric[column] = int((~blank_mask(source) & parsed.isna()).sum())

    negative_revenue = 0
    for column in _revenue_like_columns(deduplicated):
        revenue_values = parse_numeric_series(deduplicated[column])
        negative_revenue += int(revenue_values.dropna().lt(0).sum())

    quality_score = _quality_score(
        row_count=len(deduplicated),
        duplicate_percent=duplicate_percent,
        missing_values=missing_values,
        invalid_numeric=invalid_numeric,
        negative_revenue=negative_revenue,
    )

    quality_report = {
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": duplicate_percent,
        "missing_values": missing_values,
        "invalid_numeric": invalid_numeric,
        "negative_revenue": negative_revenue,
        "quality_score": quality_score,
        "row_count_before_cleaning": original_row_count,
        "row_count_after_cleaning": int(len(deduplicated)),
    }

    LOGGER.info(
        "Validation complete: %s duplicates removed, quality score=%s",
        duplicate_rows,
        quality_score,
    )
    return {"clean_df": deduplicated, "quality_report": quality_report}
