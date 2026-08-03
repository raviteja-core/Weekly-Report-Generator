"""
Merges the Stage-2 QLoRA adapter into the Stage-1 SFT base weights, producing
a single plain Hugging Face model directory -- no PEFT/bitsandbytes needed at
inference time in your local FastAPI service. This is the artifact you move
from Colab to your local `services/llm_service.py`.

Run (in Colab, after both training stages):
    python 06_merge_and_export.py \
        --sft_model checkpoints/sft \
        --adapter checkpoints/qlora \
        --output_dir checkpoints/merged

Then locally, load it exactly like any other Transformers model:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("checkpoints/merged", dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/merged")

For a smaller local footprint, re-quantize the merged model to 4-bit with
bitsandbytes or export to GGUF (llama.cpp) for CPU/Ollama-style serving --
see the note at the bottom of this file.

KAGGLE 2xT4 NOTE: merging is a one-shot, single-process operation on a 3B
model that fits on one T4, so this stays single-GPU on purpose -- run it
with plain `!python`, not `accelerate launch`. device_map is pinned to
{"": 0} instead of "auto" for the same reason as 05_eval_harness.py.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model", default="checkpoints/sft", help="Stage-1 SFT checkpoint (base for the adapter)")
    p.add_argument("--adapter", default="checkpoints/qlora", help="Stage-2 LoRA adapter directory")
    p.add_argument("--output_dir", default="checkpoints/merged")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)

    # load base in full precision for a clean merge (4-bit weights can't be merged directly)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model, dtype=torch.bfloat16, device_map={"": 0}
    )

    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.merge_and_unload()  # folds LoRA deltas into the base weights

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")
    print("This directory is a normal HF model -- copy it as-is into your local project.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Optional: re-quantize for local deployment
#
# If you want to keep serving at 4-bit locally (matching the "sub-3.2s
# latency using 4-bit quantized inference" claim), load the merged model
# with a BitsAndBytesConfig(load_in_4bit=True) at inference time in
# services/llm_service.py -- you do NOT need to re-run any training, 4-bit
# loading is a runtime flag on a full-precision checkpoint.
#
# If you'd rather run on CPU / via Ollama instead of a GPU box, convert to
# GGUF with llama.cpp's convert_hf_to_gguf.py and quantize (e.g. Q4_K_M),
# then `ollama create reportgenie-tool-model -f Modelfile` pointing at the
# .gguf -- this plugs directly into the OLLAMA_MODEL fallback path your
# README already documents.
# ---------------------------------------------------------------------------
