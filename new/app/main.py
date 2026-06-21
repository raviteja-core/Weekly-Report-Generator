import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes.report import router as report_router


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


app = FastAPI(
    title="ReportGenie AI",
    description="AI-powered analytics storytelling workspace for CSV business data.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(report_router)

frontend_dist_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
outputs_path = os.path.join(os.path.dirname(__file__), "..", "outputs")
sample_data_path = os.path.join(os.path.dirname(__file__), "..", "sample_data")
os.makedirs(outputs_path, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=outputs_path), name="outputs")
app.mount("/sample_data", StaticFiles(directory=sample_data_path), name="sample_data")

if os.path.isdir(os.path.join(frontend_dist_path, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist_path, "assets")), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(frontend_dist_path, "index.html"))
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
