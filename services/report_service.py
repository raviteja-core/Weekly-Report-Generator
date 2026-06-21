from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from services.csv_service import dataframe_from_records


BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")
DATASETS_DIR = os.path.join(BASE_OUTPUT_DIR, "datasets")
REPORT_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


def _ensure_storage_dirs() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)


def _sanitize_report_id(report_id: str) -> str:
    cleaned = (report_id or "").strip().lower()
    if not REPORT_ID_PATTERN.fullmatch(cleaned):
        raise FileNotFoundError(report_id)
    return cleaned


def _report_path(report_id: str) -> str:
    safe_id = _sanitize_report_id(report_id)
    return os.path.join(REPORTS_DIR, f"{safe_id}.json")


def _dataset_path(report_id: str) -> str:
    safe_id = _sanitize_report_id(report_id)
    return os.path.join(DATASETS_DIR, f"{safe_id}.json")


def save_report(report: dict[str, Any], dataset_records: list[dict[str, Any]]) -> str:
    _ensure_storage_dirs()
    report_id = report.get("report_id") or uuid.uuid4().hex[:12]
    report["report_id"] = report_id
    report["saved_at"] = datetime.now(timezone.utc).isoformat()
    report["data_storage"] = {"mode": "normalized-json", "raw_csv_stored": False}

    with open(_report_path(report_id), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with open(_dataset_path(report_id), "w", encoding="utf-8") as handle:
        json.dump(dataset_records, handle, ensure_ascii=False)

    return report_id


def load_report(report_id: str) -> dict[str, Any]:
    path = _report_path(report_id)
    if not os.path.exists(path):
        raise FileNotFoundError(report_id)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_report_dataset(report_id: str) -> dict[str, Any]:
    path = _dataset_path(report_id)
    if not os.path.exists(path):
        raise FileNotFoundError(report_id)
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    return {
        "records": records,
        "dataframe": dataframe_from_records(records),
    }
