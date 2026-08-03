"""
Blends a slice of a public function-calling dataset into data/train.jsonl
so the model sees tool-calling in general, more naturally-phrased contexts
-- not just your two tools' templated wording.

Uses minpeter/xlam-function-calling-60k-parsed -- an UNGATED re-upload of
the same Salesforce xLAM data (cc-by-4.0, no login/token needed). The
original Salesforce/xlam-function-calling-60k repo is gated behind a
license click-through, which breaks unattended Colab runs; this mirror has
identical content and doesn't need that.

Why this dataset: it's real API-call data across thousands of distinct
tool schemas with natural user phrasing, released for exactly this kind of
fine-tune. We do NOT want the model trained on ITS tool names -- we only
want the phrasing/structure diversity, so every row is remapped into a
GENERIC placeholder tool schema, never mixed with your actual
calculate_kpis/generate_chart namespace. Zero risk of the model getting
confused about which tools it actually has.

Run:
    pip install datasets
    python 07_blend_public_data.py --n_samples 400 --output data/train.jsonl
"""
import argparse
import json
import random
from pathlib import Path

random.seed(11)
DATA_DIR = Path(__file__).parent / "data"

GENERIC_SYSTEM_PROMPT = """You are an AI assistant that can call external tools when needed.
Available tools are provided in each conversation as a JSON list.
Rules:
- Only call a tool when the request requires it; otherwise answer in plain text.
- When calling a tool, respond with ONLY a JSON object: {"tool_calls": [{"method": "tools/call", "params": {"name": "<tool_name>", "arguments": {...}}}]}
"""


def convert_row(row):
    """Row schema (minpeter/xlam-function-calling-60k-parsed):
    - messages: [{content, role: "user", tool_calls: null}, {content: null, role: "assistant",
                  tool_calls: [{type: "function", function: {name, arguments: "<json str>"}}]}]
    - tools: json string list of {"type": "function", "function": {"name", "parameters", ...}}
    """
    try:
        tools = json.loads(row["tools"])
        user_msg = next(m for m in row["messages"] if m["role"] == "user")
        assistant_msg = next(m for m in row["messages"] if m["role"] == "assistant")
    except (json.JSONDecodeError, TypeError, KeyError, StopIteration):
        return None

    raw_calls = assistant_msg.get("tool_calls") or []
    tool_calls = []
    for c in raw_calls:
        fn = c.get("function", {})
        try:
            arguments = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}
        tool_calls.append({
            "method": "tools/call",
            "params": {"name": fn.get("name"), "arguments": arguments},
        })

    tool_list_text = "\n".join(
        f"- {t.get('function', {}).get('name')}: "
        f"{t.get('function', {}).get('description', 'no description')}"
        for t in tools
    )
    user_content = f"Available tools:\n{tool_list_text}\n\nUser request: {user_msg.get('content', '')}"
    assistant_content = (
        json.dumps({"tool_calls": tool_calls}, separators=(",", ":"))
        if tool_calls
        else "That request doesn't need a tool call -- happy to help directly."
    )

    return {
        "messages": [
            {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "_category": "public_blend",
        "_difficulty": "medium",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=400,
                    help="~15-20%% of your synthetic train set size is a good target")
    p.add_argument("--output", default=str(DATA_DIR / "train.jsonl"))
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run: pip install datasets")

    print("Downloading minpeter/xlam-function-calling-60k-parsed (ungated mirror) ...")
    ds = load_dataset("minpeter/xlam-function-calling-60k-parsed", split="train")
    ds = ds.shuffle(seed=11).select(range(min(args.n_samples * 3, len(ds))))

    converted = []
    for row in ds:
        c = convert_row(row)
        if c is not None and c["messages"][2]["content"] != "{}":
            converted.append(c)
        if len(converted) >= args.n_samples:
            break

    print(f"Converted {len(converted)} public examples")

    existing = [json.loads(l) for l in open(args.output)]
    combined = existing + converted
    random.shuffle(combined)

    with open(args.output, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")

    print(f"train.jsonl now has {len(combined)} examples "
          f"({len(existing)} domain-specific + {len(converted)} public blend)")


if __name__ == "__main__":
    main()
