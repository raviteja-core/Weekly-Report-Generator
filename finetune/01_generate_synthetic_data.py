"""
Synthetic dataset generator for ReportGenie AI tool-calling fine-tune.

Produces (user_query, ground_truth_tool_calls) pairs where the "CSV" is
represented as a schema + sample rows description fed to the model,
mirroring what services/csv_service.py would hand the LLM after schema
detection.

Design principles ("what makes this dataset correct"):
1. Every example's ground truth is derived programmatically from the schema
   itself (not hand-written), so labels are guaranteed consistent with the
   MCP tool contract in services/mcp_service.py.
2. Messiness is injected deliberately and labeled: missing columns, mixed
   date formats, currency symbols, nulls, duplicate headers, non-English
   headers, numeric-looking strings, outliers -> these are the adversarial
   cases your README claims 99.6% schema compliance on. Don't skip them.
3. Includes negative examples: queries where NO tool call is correct
   (e.g. "what does KPI mean?"), so the model learns *when not* to call
   a tool -- this is the single most common failure mode in tool-calling
   fine-tunes (over-triggering).
4. Includes multi-tool examples (calculate_kpis THEN generate_chart) since
   real report generation is agentic/sequential, not single-shot.
5. Hard negatives: near-miss schemas (e.g. a column named "revenu" instead
   of "revenue") to teach robustness to messy real-world headers.

Output: data/raw_examples.jsonl, one JSON object per line:
{
  "schema": {...},              # column metadata as the CSV service would emit
  "user_query": "...",
  "tool_calls": [ {method, params}, ... ],   # ground truth, [] if none
  "category": "clean|noisy|adversarial|no_tool|multi_step",
  "difficulty": "easy|medium|hard"
}
"""
import json
import random
import string
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Vocabulary pools for generating varied, realistic-but-fake business schemas
# ---------------------------------------------------------------------------

CLEAN_METRIC_NAMES = [
    "revenue", "sales", "profit", "cost", "units_sold", "signups",
    "churn_rate", "active_users", "conversion_rate", "ad_spend",
    "orders", "refunds", "sessions", "impressions", "clicks",
]

MESSY_METRIC_VARIANTS = {
    "revenue": ["Revenue ($)", "revenu", "TOTAL_REVENUE", "rev_usd", "Revenue  ", "Total Revenue"],
    "sales": ["Sales Qty", "sales#", "SALES", "sale_amount", "Sales (units)"],
    "profit": ["Net Profit", "profit_margin_$", "Profit/Loss", "PnL"],
    "cost": ["Cost ($)", "COGS", "total_cost", "Cost_USD"],
    "units_sold": ["Units Sold", "units", "qty_sold", "Units-Sold"],
}

DATE_COLUMN_NAMES = ["date", "Date", "week_ending", "period", "Week", "report_date", "DATE (YYYY-MM-DD)"]
CATEGORY_COLUMN_NAMES = ["region", "Region", "product", "category", "channel", "store_id", "segment"]

NOISE_TYPES = [
    "mixed_date_formats", "currency_symbols", "null_values", "duplicate_header_row",
    "trailing_whitespace_headers", "non_english_headers", "numeric_strings",
    "outlier_rows", "inconsistent_casing", "merged_cells_artifact",
]

CHART_TYPES = ["line", "bar", "pie"]

QUERY_TEMPLATES_KPI = [
    "Can you build this week's report from {file}?",
    "Generate the KPI summary for {file}.",
    "What's our {metric} trend looking like this quarter?",
    "Give me revenue growth and anomalies from this dataset.",
    "Summarize performance for {file}.",
    "I need the weekly business report -- KPIs plus a chart.",
    "Analyze {file} and flag anything unusual.",
    "What are the top-line numbers from this data?",
]

QUERY_TEMPLATES_CHART_ONLY = [
    "Just show me a {chart_type} chart of {metric} over time.",
    "Plot {metric} by {category} as a {chart_type} chart.",
    "Can you visualize the {metric} trend?",
]

QUERY_TEMPLATES_NO_TOOL = [
    "What does KPI stand for?",
    "How do you calculate churn rate in general?",
    "What's the difference between a bar chart and a pie chart?",
    "Can you explain what an anomaly flag means in a report?",
    "Is revenue the same thing as profit?",
    "What formats does ReportGenie accept for upload?",
    "Thanks, that report looks great!",
    "Can you explain your methodology without re-running the analysis?",
]


def rand_suffix(n=4):
    return "".join(random.choices(string.ascii_lowercase, k=n))


def make_schema(n_metrics=2, include_category=True, noisy=False, adversarial=False):
    metrics = random.sample(CLEAN_METRIC_NAMES, n_metrics)
    columns = []

    date_col = random.choice(DATE_COLUMN_NAMES) if not noisy else random.choice(DATE_COLUMN_NAMES) + ("  " if random.random() < 0.3 else "")
    columns.append({"name": date_col, "dtype": "date", "canonical": "date"})

    for m in metrics:
        if noisy or adversarial:
            variants = MESSY_METRIC_VARIANTS.get(m, [m.upper(), m.title(), f"{m}_$"])
            col_name = random.choice(variants)
        else:
            col_name = m
        columns.append({"name": col_name, "dtype": "numeric", "canonical": m})

    if include_category:
        cat_col = random.choice(CATEGORY_COLUMN_NAMES)
        columns.append({"name": cat_col, "dtype": "categorical", "canonical": "category"})

    noise_applied = []
    if noisy or adversarial:
        k = random.randint(1, 3 if adversarial else 2)
        noise_applied = random.sample(NOISE_TYPES, k)

    return {
        "columns": columns,
        "n_rows": random.randint(30, 500),
        "noise_applied": noise_applied,
        "sample_file_name": f"weekly_metrics_{rand_suffix()}.csv",
    }


def kpi_tool_call(schema):
    metric_cols = [c["canonical"] for c in schema["columns"] if c["dtype"] == "numeric"]
    return {
        "method": "tools/call",
        "params": {
            "name": "calculate_kpis",
            "arguments": {
                "metrics": metric_cols,
                "date_column": next(c["name"] for c in schema["columns"] if c["dtype"] == "date"),
                "detect_anomalies": True,
            },
        },
    }


def chart_tool_call(schema, chart_type=None):
    metric_cols = [c["canonical"] for c in schema["columns"] if c["dtype"] == "numeric"]
    metric = random.choice(metric_cols)
    cat_col = next((c["name"] for c in schema["columns"] if c["dtype"] == "categorical"), None)
    return {
        "method": "tools/call",
        "params": {
            "name": "generate_chart",
            "arguments": {
                "metric": metric,
                "chart_type": chart_type or random.choice(CHART_TYPES),
                "group_by": cat_col,
            },
        },
    }


def gen_examples(n_per_category=250):
    examples = []

    # 1. Clean, single-tool KPI requests
    for _ in range(n_per_category):
        schema = make_schema(n_metrics=random.randint(1, 3), noisy=False)
        q = random.choice(QUERY_TEMPLATES_KPI).format(
            file=schema["sample_file_name"],
            metric=random.choice([c["canonical"] for c in schema["columns"] if c["dtype"] == "numeric"]),
        )
        examples.append({
            "schema": schema, "user_query": q,
            "tool_calls": [kpi_tool_call(schema)],
            "category": "clean", "difficulty": "easy",
        })

    # 2. Noisy schemas -> still correct KPI call (tests header normalization)
    for _ in range(n_per_category):
        schema = make_schema(n_metrics=random.randint(1, 3), noisy=True)
        q = random.choice(QUERY_TEMPLATES_KPI).format(
            file=schema["sample_file_name"],
            metric=random.choice([c["canonical"] for c in schema["columns"] if c["dtype"] == "numeric"]),
        )
        examples.append({
            "schema": schema, "user_query": q,
            "tool_calls": [kpi_tool_call(schema)],
            "category": "noisy", "difficulty": "medium",
        })

    # 3. Adversarial schemas -> heavier noise stacking, hardest header variants
    for _ in range(n_per_category):
        schema = make_schema(n_metrics=random.randint(2, 4), noisy=True, adversarial=True)
        q = random.choice(QUERY_TEMPLATES_KPI).format(
            file=schema["sample_file_name"],
            metric=random.choice([c["canonical"] for c in schema["columns"] if c["dtype"] == "numeric"]),
        )
        examples.append({
            "schema": schema, "user_query": q,
            "tool_calls": [kpi_tool_call(schema)],
            "category": "adversarial", "difficulty": "hard",
        })

    # 4. Chart-only requests (single tool, no KPI call -- tests over-triggering)
    for _ in range(n_per_category // 2):
        schema = make_schema(n_metrics=random.randint(1, 2), noisy=random.random() < 0.4)
        metric = random.choice([c["canonical"] for c in schema["columns"] if c["dtype"] == "numeric"])
        chart_type = random.choice(CHART_TYPES)
        q = random.choice(QUERY_TEMPLATES_CHART_ONLY).format(
            metric=metric, chart_type=chart_type,
            category=next((c["name"] for c in schema["columns"] if c["dtype"] == "categorical"), "category"),
        )
        examples.append({
            "schema": schema, "user_query": q,
            "tool_calls": [chart_tool_call(schema, chart_type)],
            "category": "chart_only", "difficulty": "medium",
        })

    # 5. Multi-step: KPI then chart in one turn
    for _ in range(n_per_category // 2):
        schema = make_schema(n_metrics=random.randint(2, 3), noisy=random.random() < 0.5)
        q = "Give me the full report: KPIs plus a chart of the main trend."
        examples.append({
            "schema": schema, "user_query": q,
            "tool_calls": [kpi_tool_call(schema), chart_tool_call(schema, "line")],
            "category": "multi_step", "difficulty": "hard",
        })

    # 6. No-tool negatives -- critical for precision, don't skip
    for _ in range(n_per_category):
        schema = make_schema(n_metrics=2, noisy=random.random() < 0.3)
        q = random.choice(QUERY_TEMPLATES_NO_TOOL)
        examples.append({
            "schema": schema, "user_query": q,
            "tool_calls": [],
            "category": "no_tool", "difficulty": "medium",
        })

    random.shuffle(examples)
    return examples


def validate_example(ex):
    """Reject anything that doesn't match the MCP tool contract exactly.
    This is the 'quality gate' -- run every generated example through it
    before it's allowed into the training set."""
    valid_methods = {"tools/call"}
    valid_tool_names = {"calculate_kpis", "generate_chart"}
    for call in ex["tool_calls"]:
        if call["method"] not in valid_methods:
            return False
        if call["params"]["name"] not in valid_tool_names:
            return False
        if call["params"]["name"] == "calculate_kpis":
            args = call["params"]["arguments"]
            if not args.get("metrics") or not args.get("date_column"):
                return False
        if call["params"]["name"] == "generate_chart":
            args = call["params"]["arguments"]
            if args.get("chart_type") not in CHART_TYPES or not args.get("metric"):
                return False
    return True


def main():
    examples = gen_examples(n_per_category=800)
    examples = [e for e in examples if validate_example(e)]

    # dedupe by (user_query, category) to avoid near-identical repeats dominating
    seen = set()
    deduped = []
    for e in examples:
        key = (e["user_query"], e["category"], tuple(c["canonical"] for c in e["schema"]["columns"]))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    out_path = OUT_DIR / "raw_examples.jsonl"
    with out_path.open("w") as f:
        for e in deduped:
            f.write(json.dumps(e) + "\n")

    by_cat = {}
    for e in deduped:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1

    print(f"Wrote {len(deduped)} examples to {out_path}")
    print("Category breakdown:", by_cat)


if __name__ == "__main__":
    main()
