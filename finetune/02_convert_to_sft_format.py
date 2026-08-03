"""
Converts data/raw_examples.jsonl into Qwen2.5-Instruct chat-format training
rows and splits into train / val / adversarial-test.

Output format (one JSON object per line, "messages" key -> works directly
with TRL's SFTTrainer):
{
  "messages": [
    {"role": "system", "content": "<tool defs + instructions>"},
    {"role": "user", "content": "<the query, with schema context>"},
    {"role": "assistant", "content": "<tool_call JSON or plain text>"}
  ]
}

Split strategy (why this split, not a random 80/10/10):
- train: clean + noisy + chart_only + multi_step + no_tool (all difficulty)
- val: held-out slice of the SAME distribution, for early stopping
- adversarial_test: the 'adversarial' category ONLY, fully held out from
  training. This is what you report schema-compliance % against -- mixing
  adversarial examples into train would make that number meaningless.
"""
import json
import random
from pathlib import Path

random.seed(7)
DATA_DIR = Path(__file__).parent / "data"

TOOL_DEFINITIONS = """You are ReportGenie AI's analysis agent. You have access to these tools:

1. calculate_kpis(metrics: list[str], date_column: str, detect_anomalies: bool) -> KPI summary with growth, averages, trends, anomaly flags
2. generate_chart(metric: str, chart_type: "line"|"bar"|"pie", group_by: str|null) -> chart payload

Rules:
- Only call a tool when the user's request requires computing or visualizing data from the uploaded CSV.
- If the user asks a general/conceptual question that does not require the data, respond with plain text and do NOT call a tool.
- When calling a tool, respond with ONLY a JSON object in this exact shape, no other text:
{"tool_calls": [{"method": "tools/call", "params": {"name": "<tool_name>", "arguments": {...}}}]}
- Use the canonical column names given in the schema, not the raw noisy header text.
- If both KPIs and a chart are requested, include both calls in the same tool_calls list, KPI call first.
"""


def schema_to_context(schema):
    lines = [f"Uploaded file: {schema['sample_file_name']} ({schema['n_rows']} rows)", "Detected columns:"]
    for c in schema["columns"]:
        lines.append(f'  - raw_header="{c["name"]}" -> canonical="{c["canonical"]}" (type: {c["dtype"]})')
    return "\n".join(lines)


def build_assistant_content(tool_calls):
    if not tool_calls:
        return "That's a general question, so I don't need to run any tools -- happy to explain directly."
    return json.dumps({"tool_calls": tool_calls}, separators=(",", ":"))


def example_to_messages(ex):
    user_content = f"{schema_to_context(ex['schema'])}\n\nUser request: {ex['user_query']}"
    return {
        "messages": [
            {"role": "system", "content": TOOL_DEFINITIONS},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": build_assistant_content(ex["tool_calls"])},
        ],
        # kept for eval slicing, NOT fed to the model at train time
        "_category": ex["category"],
        "_difficulty": ex["difficulty"],
    }


def main():
    raw = [json.loads(l) for l in (DATA_DIR / "raw_examples.jsonl").open()]

    adversarial = [e for e in raw if e["category"] == "adversarial"]
    non_adversarial = [e for e in raw if e["category"] != "adversarial"]

    random.shuffle(non_adversarial)
    n_val = max(1, int(0.1 * len(non_adversarial)))
    val_raw = non_adversarial[:n_val]
    train_raw = non_adversarial[n_val:]

    # adversarial set is split too: a small slice can go to train (so the
    # model has SEEN the pattern of noisy headers), the rest is untouched
    # held-out test -- matches "benchmarks tool-calling stability against
    # out-of-distribution inputs" from your README, without making the
    # test set fully unseen-distribution (which would be unrealistically hard).
    random.shuffle(adversarial)
    n_adv_train = int(0.3 * len(adversarial))
    train_raw += adversarial[:n_adv_train]
    adv_test_raw = adversarial[n_adv_train:]

    splits = {
        "train": [example_to_messages(e) for e in train_raw],
        "val": [example_to_messages(e) for e in val_raw],
        "adversarial_test": [example_to_messages(e) for e in adv_test_raw],
    }

    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} examples -> {path}")


if __name__ == "__main__":
    main()
