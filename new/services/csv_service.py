from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import pandas as pd

from services.data_quality import normalize_category_labels, parse_numeric_series, validate_and_clean_data
from services.schema_service import SCHEMA_CONFIDENCE_THRESHOLD, detect_schema


LOGGER = logging.getLogger(__name__)

QUALITY_SCORE_THRESHOLD = 0.75
DUPLICATE_PERCENT_THRESHOLD = 0.10


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _build_canonical_dataframe(clean_df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    canonical = clean_df.copy()

    date_col = schema["date_col"]
    revenue_col = schema["revenue_col"]
    category_col = schema.get("category_col")
    region_col = schema.get("region_col")
    price_col = schema.get("price_col")
    quantity_col = schema.get("quantity_col")

    if date_col:
        canonical["date"] = _parse_datetime(canonical[date_col])
    if revenue_col:
        canonical["revenue"] = parse_numeric_series(canonical[revenue_col])
    if category_col:
        canonical["category"] = normalize_category_labels(canonical[category_col])
    if region_col:
        canonical["region"] = normalize_category_labels(canonical[region_col])
    if price_col:
        canonical["price"] = parse_numeric_series(canonical[price_col])
    if quantity_col:
        canonical["quantity"] = parse_numeric_series(canonical[quantity_col])

    return canonical


def _build_warnings(quality_report: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if quality_report["duplicate_percent"] > 0:
        warnings.append("High duplicate rate detected" if quality_report["duplicate_percent"] > 0.1 else "Duplicate rows were removed before analysis")
    if quality_report["negative_revenue"] > 0:
        warnings.append("Negative revenue anomalies detected")
    if schema and schema["confidence"] < 0.85:
        warnings.append("Revenue column inferred with low confidence")
    return warnings


def _failed_response(reason: str, quality_report: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": reason,
        "details": quality_report,
        "quality_report": quality_report,
        "schema": schema or {},
        "confidence": schema["confidence"] if schema else quality_report["quality_score"],
        "warnings": _build_warnings(quality_report, schema or {"confidence": 0.0}),
    }


def _critical_missing_fields(canonical_df: pd.DataFrame) -> list[str]:
    missing_fields: list[str] = []
    for field in ["date", "revenue"]:
        if field not in canonical_df.columns or canonical_df[field].isna().any():
            missing_fields.append(field)
    return missing_fields


def dataset_preview(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    preview = df.head(limit).copy().where(pd.notna(df.head(limit)), None)
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].dt.strftime("%Y-%m-%d")
    return preview.to_dict(orient="records")


def serialize_dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    serialized = df.copy()
    for column in serialized.columns:
        if pd.api.types.is_datetime64_any_dtype(serialized[column]):
            serialized[column] = serialized[column].dt.strftime("%Y-%m-%d")
    serialized = serialized.where(pd.notna(serialized), None)
    return serialized.to_dict(orient="records")


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = _parse_datetime(df["date"])
    for metric_column in ["revenue", "price", "quantity"]:
        if metric_column in df.columns:
            df[metric_column] = parse_numeric_series(df[metric_column])
    if "category" in df.columns:
        df["category"] = normalize_category_labels(df["category"])
    if "region" in df.columns:
        df["region"] = normalize_category_labels(df["region"])
    return df


def prepare_dataset(csv_text: str) -> dict[str, Any]:
    LOGGER.info("Preparing dataset from CSV payload")
    try:
        raw_df = pd.read_csv(StringIO(csv_text), low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV contains no rows.") from exc

    validation_result = validate_and_clean_data(raw_df)
    clean_df = validation_result["clean_df"]
    quality_report = validation_result["quality_report"]

    if (
        quality_report["duplicate_percent"] > DUPLICATE_PERCENT_THRESHOLD
        or quality_report["quality_score"] < QUALITY_SCORE_THRESHOLD
    ):
        LOGGER.warning(
            "Early data quality gate failed: duplicate_percent=%s quality_score=%s",
            quality_report["duplicate_percent"],
            quality_report["quality_score"],
        )
        return _failed_response("Data quality too low", quality_report, {})

    schema = detect_schema(clean_df)
    if schema["confidence"] < SCHEMA_CONFIDENCE_THRESHOLD:
        LOGGER.warning("Schema confidence %s below threshold", schema["confidence"])
        return _failed_response("Schema confidence too low", quality_report, schema)

    canonical_df = _build_canonical_dataframe(clean_df, schema)
    missing_fields = _critical_missing_fields(canonical_df)

    if (
        quality_report["duplicate_percent"] > DUPLICATE_PERCENT_THRESHOLD
        or quality_report["quality_score"] < QUALITY_SCORE_THRESHOLD
        or missing_fields
    ):
        LOGGER.warning(
            "Data quality gate failed: duplicate_percent=%s quality_score=%s missing_fields=%s",
            quality_report["duplicate_percent"],
            quality_report["quality_score"],
            missing_fields,
        )
        if missing_fields:
            quality_report["missing_critical_fields"] = missing_fields
        return _failed_response("Data quality too low", quality_report, schema)

    schema_payload = {
        "date_col": schema["date_col"],
        "revenue_col": schema["revenue_col"],
        "category_col": schema.get("category_col"),
        "region_col": schema.get("region_col"),
        "confidence": schema["confidence"],
        "column_confidence": schema["column_confidence"],
        "row_count": int(len(canonical_df)),
        "columns": list(canonical_df.columns),
    }
    warnings = _build_warnings(quality_report, schema)

    canonical_df.attrs["quality_report"] = quality_report
    canonical_df.attrs["schema"] = schema_payload

    LOGGER.info("Dataset prepared successfully with %s cleaned rows", len(canonical_df))
    return {
        "status": "success",
        "dataframe": canonical_df,
        "schema": schema_payload,
        "quality_report": quality_report,
        "confidence": schema["confidence"],
        "warnings": warnings,
        "preview": dataset_preview(canonical_df),
        "records": serialize_dataframe_records(canonical_df),
    }


def load_sales_csv(csv_text: str) -> pd.DataFrame:
    dataset = prepare_dataset(csv_text)
    if dataset["status"] != "success":
        raise ValueError(dataset["reason"])
    return dataset["dataframe"]
