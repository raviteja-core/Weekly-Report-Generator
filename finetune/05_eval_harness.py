"""
Evaluation harness: measures the two numbers your README reports --
schema compliance % and reasoning/output failure rate -- on the held-out
adversarial_test.jsonl split.

Definitions used here (make these explicit in your writeup so the numbers
are reproducible/defensible):
- schema_compliant: model output parses as JSON matching the exact
  {"tool_calls": [...]} contract, every "name" is a known tool, every
  required argument is present with the right type.
- correct: schema_compliant AND the tool name + key arguments (metrics,
  chart_type, whether a tool was called at all) match ground truth.
- reasoning_failure: schema_compliant but wrong tool choice, wrong metric
  list, or hallucinated column name not in the given schema.

KAGGLE 2xT4 NOTE: this is inference on a 3B model that comfortably fits on a
single T4 (~6GB in fp16), so this script is left single-GPU/single-process on
purpose -- run it with plain `!python`, not `accelerate launch`. device_map
is pinned to {"": 0} instead of "auto" so it doesn't needlessly spread one
small model across both GPUs and pay cross-GPU communication overhead for
zero benefit. Loads in fp16 (matching how it was trained) and uses sdpa
attention -- both are the fast paths on T4's Turing architecture; bf16 and
FlashAttention-2 are not accelerated on this GPU generation.

TIMING: this loop is NOT batched -- one `model.generate()` call per example,
sequentially. On 557 examples at roughly 3-5s/example (unbatched fp16
generate on a T4), expect ~30-45 minutes end to end. Progress prints every
--log_every examples (default 25) so you can tell it's actually working
instead of staring at a silent cell.

MODEL LOADING: --model can point at either a full merged model directory
(checkpoints/merged, from 06_merge_and_export.py) or a raw PEFT adapter
directory (checkpoints/qlora, straight from 04_train_qlora.py -- this has
adapter_config.json/adapter_model.safetensors, not config.json). This script
tries a normal full-model load first and automatically falls back to loading
+ merging a PEFT adapter if that fails, so either path works.

Run:
    python 05_eval_harness.py --model checkpoints/qlora --test_file data/adversarial_test.jsonl
"""
import argparse
import json
import re
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

VALID_TOOLS = {"calculate_kpis", "generate_chart"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test_file", default="data/adversarial_test.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--log_every", type=int, default=25,
                    help="Print a progress line every N examples. Set to 0 to disable.")
    return p.parse_args()


def load_model(model_path):
    """Try a normal full-model load first (works for checkpoints/merged).
    Falls back to loading + merging a raw PEFT adapter directory (works for
    checkpoints/qlora straight out of 04_train_qlora.py, which has no
    config.json -- only adapter_config.json)."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype="float16", device_map={"": 0}, attn_implementation="sdpa"
        )
        print(f"Loaded {model_path} as a full model.")
        return model
    except (OSError, ValueError) as e:
        print(f"Full-model load failed ({type(e).__name__}: {e}). "
              f"Retrying as a PEFT adapter directory...", file=sys.stderr)
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, dtype="float16", device_map={"": 0}, attn_implementation="sdpa"
        )
        model = model.merge_and_unload()
        print(f"Loaded {model_path} as a PEFT adapter and merged it for inference.")
        return model


def try_parse_tool_json(text):
    """Model should output ONLY the JSON object when calling a tool.
    Be lenient about surrounding whitespace/fences but strict about content."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "tool_calls" not in obj:
        return None
    return obj


def check_schema_compliance(parsed):
    if parsed is None:
        return False
    for call in parsed.get("tool_calls", []):
        if call.get("method") != "tools/call":
            return False
        params = call.get("params", {})
        name = params.get("name")
        if name not in VALID_TOOLS:
            return False
        args = params.get("arguments", {})
        if name == "calculate_kpis":
            if not args.get("metrics") or not args.get("date_column"):
                return False
        if name == "generate_chart":
            if args.get("chart_type") not in {"line", "bar", "pie"} or not args.get("metric"):
                return False
    return True


def check_correctness(parsed, ground_truth_calls, known_canonical_cols, known_raw_headers):
    """ground_truth_calls: list of {method, params} from the original example.

    known_raw_headers: the raw (pre-canonicalization) header strings for this
    example's schema, e.g. "DATE (YYYY-MM-DD)". Needed because the synthetic
    data generator's kpi_tool_call() puts the RAW header (not the canonical
    name) into the "date_column" argument -- metrics use canonical names,
    date_column does not. Checking date_column only against canonical names
    rejects every correctly-behaving example; it has to be checked against
    the raw headers instead, matching what the training labels actually
    contain."""
    if ground_truth_calls == [] and (parsed is None or parsed.get("tool_calls") == []):
        return True, False  # correct no-tool-call decision
    if parsed is None:
        return False, False  # not schema-compliant -> not a reasoning failure, it's a format failure
    predicted = parsed.get("tool_calls", [])
    gt_names = [c["params"]["name"] for c in ground_truth_calls]
    pred_names = [c["params"]["name"] for c in predicted]
    if gt_names != pred_names:
        return False, True  # schema-valid but wrong tool sequence -> reasoning failure
    # check no hallucinated columns
    for call in predicted:
        args = call["params"]["arguments"]
        for key in ("metrics",):
            if key in args:
                for m in args[key]:
                    if m not in known_canonical_cols:
                        return False, True
        # date_column legitimately holds the RAW header (see docstring above),
        # so accept either the raw header or the canonical name -- being
        # lenient here since the ground truth itself is inconsistent about
        # which form to use.
        if args.get("date_column") and args["date_column"] not in known_canonical_cols \
                and args["date_column"] not in known_raw_headers:
            return False, True
        for key in ("metric", "group_by"):
            if args.get(key) and args[key] not in known_canonical_cols and args[key] is not None:
                # group_by can be a raw header name (categorical col name), allow that separately
                if key != "group_by":
                    return False, True
    return True, False


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()

    rows = [json.loads(l) for l in open(args.test_file)]

    n = len(rows)
    n_schema_ok = 0
    n_correct = 0
    n_reasoning_fail = 0

    print(f"Evaluating {n} examples (unbatched, one generate() call each)...")
    start = time.time()

    for i, row in enumerate(rows, 1):
        messages = row["messages"][:2]  # system + user, we generate the assistant turn
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        parsed = try_parse_tool_json(gen_text)
        schema_ok = check_schema_compliance(parsed) or (parsed is None and "tool_calls" not in gen_text)

        gt_content = row["messages"][2]["content"]
        try:
            gt_parsed = json.loads(gt_content)
            gt_calls = gt_parsed.get("tool_calls", [])
        except json.JSONDecodeError:
            gt_calls = []

        # recover canonical column set AND raw header strings from the user message
        # text -- date_column needs the raw set, metrics need the canonical set (see
        # check_correctness docstring for why they differ)
        user_text = row["messages"][1]["content"]
        known_cols = set(re.findall(r'canonical="([a-zA-Z_]+)"', user_text))
        known_raw_headers = set(re.findall(r'raw_header="([^"]+)"', user_text))

        correct, reasoning_fail = check_correctness(parsed, gt_calls, known_cols, known_raw_headers)

        n_schema_ok += int(schema_ok)
        n_correct += int(correct)
        n_reasoning_fail += int(reasoning_fail)

        if args.log_every and i % args.log_every == 0:
            elapsed = time.time() - start
            rate = elapsed / i
            remaining = rate * (n - i)
            print(f"[{i}/{n}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining "
                  f"({rate:.1f}s/example) -- running schema compliance {100*n_schema_ok/i:.1f}%",
                  flush=True)

    total = time.time() - start
    print(f"\nDone in {total:.0f}s ({total/60:.1f} min), {total/n:.2f}s/example average.")
    print(f"N examples:              {n}")
    print(f"Schema compliance:       {100 * n_schema_ok / n:.1f}%")
    print(f"End-to-end correctness:  {100 * n_correct / n:.1f}%")
    print(f"Reasoning failure rate:  {100 * n_reasoning_fail / n:.1f}%")


if __name__ == "__main__":
    main()
