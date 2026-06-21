from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from services.agent_controller import AgentController
from services.csv_service import prepare_dataset
from services.mcp_service import handle_mcp_request, mcp_client_list_tools
from services.report_service import load_report, load_report_dataset, save_report


LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["reportgenie"])
agent_controller = AgentController(max_iterations=6)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_PROMPT_LENGTH = 1500
MAX_MESSAGE_LENGTH = 1000


def _validate_text_input(value: str, field_name: str, max_length: int) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty.")
    if len(cleaned) > max_length:
        raise HTTPException(status_code=400, detail=f"{field_name} is too long.")
    return cleaned


def _decode_csv_bytes(raw_bytes: bytes) -> str:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV file is too large. Limit is 2 MB.")
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Unable to decode CSV file. Please use UTF-8 encoding.") from exc


def _build_report_payload(dataset: dict, prompt: str) -> dict:
    dataframe = dataset["dataframe"]
    if "revenue" not in dataframe.columns:
        raise ValueError(
            "ReportGenie AI could not detect a revenue-like numeric column. "
            "Please upload a CSV with business values such as revenue, sales, amount, or total."
        )

    report = agent_controller.run_report(prompt, dataset)
    report["app_name"] = "ReportGenie AI"
    report["prompt"] = prompt

    report_id = save_report(report, dataset["records"])
    report["report_id"] = report_id
    report["share_url"] = f"/?report={report_id}"
    return report


@router.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    prompt: str = Form("Generate a tool-driven KPI report with charts and executive insights."),
):
    if file.content_type not in ["text/csv", "application/vnd.ms-excel", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Please upload a valid CSV file.")

    prompt = _validate_text_input(prompt, "Prompt", MAX_PROMPT_LENGTH)
    csv_text = _decode_csv_bytes(await file.read())

    try:
        dataset = prepare_dataset(csv_text)
        if dataset["status"] != "success":
            return {
                "status": "failed",
                "reason": dataset["reason"],
                "details": dataset["details"],
                "quality_report": dataset["quality_report"],
                "schema": dataset["schema"],
                "confidence": dataset["confidence"],
                "warnings": dataset["warnings"],
            }
        response = _build_report_payload(dataset, prompt)
        response["status"] = "success"
        return response
    except ValueError as exc:
        LOGGER.exception("CSV analysis failed due to validation error")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected analysis failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/reports/{report_id}")
def get_report(report_id: str):
    try:
        report = load_report(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Report not found.") from exc
    report["share_url"] = f"/?report={report['report_id']}"
    return report


@router.post("/chat")
def chat_with_report(report_id: str = Form(...), message: str = Form(...)):
    message = _validate_text_input(message, "Message", MAX_MESSAGE_LENGTH)
    try:
        report = load_report(report_id)
        report_dataset = load_report_dataset(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Report not found.") from exc

    if report.get("status") == "failed":
        return {
            "report_id": report_id,
            "message": message,
            "answer": "This report failed the trust gate, so chat-based analysis is disabled until the data is fixed.",
            "follow_ups": [],
        }

    dataset = {
        "dataframe": report_dataset["dataframe"],
        "schema": report["schema"],
        "preview": report["preview"],
        "records": report_dataset["records"],
        "quality_report": report.get("quality_report", {}),
        "warnings": report.get("warnings", []),
        "confidence": report.get("confidence"),
    }
    context = {
        "summary": report.get("summary"),
        "quality_report": report.get("quality_report", {}),
        "kpis": report.get("kpis", {}),
        "insights": report.get("insights", []),
        "story": report.get("story", []),
        "tools_used": report.get("tools_used", []),
    }
    response = agent_controller.run_chat(message, dataset, context)

    return {
        "report_id": report_id,
        "message": message,
        "answer": response["answer"],
        "follow_ups": response["follow_ups"],
        "tools_used": response["tools_used"],
        "meta": response["meta"],
        "row_count": len(report_dataset["records"]),
    }


@router.get("/sample")
def sample_csv():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data", "sales_sample.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample data not found.")
    return {"sample_file": "/sample_data/sales_sample.csv"}


@router.get("/tools")
def list_tools():
    return {"tools": mcp_client_list_tools()}


@router.post("/mcp")
def mcp_endpoint(payload: dict = Body(...)):
    try:
        return handle_mcp_request(payload)
    except Exception as exc:
        LOGGER.exception("Unexpected MCP request failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
