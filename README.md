# ReportGenie AI: Automated KPI Reporter

ReportGenie AI is a Business Analytics application for companies that generate weekly reports from structured CSV data. Instead of manually calculating metrics and writing summaries, the system lets an LLM call approved KPI and chart tools through an MCP-style tool layer, then combine those outputs into a structured business report with insights.

## Stack

- Backend: FastAPI, Pandas, Plotly
- Frontend: React, Tailwind CSS, Framer Motion, Vite
- AI layer: Ollama-compatible tool-augmented LLM with graceful fallback mode
- Tool pattern: `calculate_kpis(data)` and `generate_chart(data, type)`
- MCP layer: JSON-RPC tool discovery and invocation for `initialize`, `tools/list`, and `tools/call`

## What It Does

- Drag-and-drop CSV upload
- Automatic schema detection and preview table
- Hybrid schema detection with heuristic profiling plus optional local-LLM schema assist
- KPI engine for revenue, growth, averages, trends, and anomaly flags
- Interactive line, bar, and pie charts
- LLM-driven report generation that requests KPI/chart tools before writing insights
- Story Mode with scroll-triggered narrative sections
- Tool-aware AI chat panel for follow-up questions on the analyzed dataset
- Shareable live report link
- Theme switch and downloadable insights JSON

## Project Structure

- `app/main.py` - FastAPI app and static frontend serving
- `app/routes/report.py` - analysis, chat, sample, and report APIs
- `services/csv_service.py` - schema detection, preview generation, CSV normalization
- `services/kpi_service.py` - KPI and anomaly engine
- `services/chart_service.py` - tool-based chart generation and Plotly payloads
- `services/llm_service.py` - AI orchestration prompts, agent decisions, and fallbacks
- `services/tool_service.py` - approved KPI/chart tool implementations
- `services/mcp_service.py` - MCP-style JSON-RPC tool server and client helpers
- `services/report_service.py` - persistent live-report storage
- `frontend/` - React + Tailwind + Framer Motion app
- `outputs/reports/` - saved live report payloads

## Setup

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

## Optional Ollama Setup

InsightForge AI works without Ollama by falling back to deterministic insight copy, but the premium AI insight and chat layers improve when Ollama is running.

```bash
ollama pull mistral:latest
ollama serve
```

Optional environment variables:

- `OLLAMA_HOST` defaults to `http://localhost:11434`
- `OLLAMA_MODEL` defaults to `mistral:latest`
- `OLLAMA_TIMEOUT_SECONDS` defaults to `20`
- `SCHEMA_LLM_ASSIST` defaults to `1` and enables local-LLM help for unfamiliar CSV schemas
- `SCHEMA_LLM_CONFIDENCE_THRESHOLD` defaults to `1.35` and controls when schema assist kicks in

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## API Endpoints

- `POST /api/analyze` - upload a CSV and build the full InsightForge AI report payload
- `POST /api/chat` - ask follow-up questions against a saved report
- `GET /api/reports/{report_id}` - reopen a saved live report
- `GET /api/sample` - fetch the sample dataset path
- `GET /api/tools` - inspect the available analytics tools
- `POST /api/mcp` - MCP JSON-RPC endpoint for tool discovery and tool calls

## Verification Completed

- Frontend production build succeeds with Vite
- FastAPI imports correctly
- Sample CSV analysis succeeds and returns KPI, chart, story, and shareable report data
- Chat endpoint succeeds in fallback mode when Ollama is unavailable
