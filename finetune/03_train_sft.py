"""
Stage 1: SFT on Qwen2.5-3B-Instruct via bf16 LoRA (not full-parameter tuning).

KAGGLE 2xT4 VERSION -- changes vs. the Colab single-GPU script:
1. device_map="auto" is REPLACED with device_map={"": PartialState().process_index}.
   "auto" tells `from_pretrained` to shard ONE model instance across every GPU it
   can see. That's the wrong tool here: under `accelerate launch --multi_gpu`,
   TWO separate processes are started (one per GPU) and each one can already see
   BOTH T4s. If each process also tries to "auto" shard across both GPUs, the two
   processes fight over the same devices and either hang or crash. Pinning each
   process to its own single GPU (via its rank) lets accelerate run true data
   parallelism (DDP): each GPU holds a full model replica and processes a
   different batch, which is what actually gets you a ~1.7-2x speedup on 2 GPUs.
2. `ddp_find_unused_parameters=False` is set -- with LoRA + gradient checkpointing,
   most base-model parameters are frozen and never get gradients, and DDP's
   default "find unused parameters" scan wastes time re-checking that every step.
3. The final merge_and_unload()+save is guarded behind `is_main_process`. Under
   DDP, this script body runs once per GPU process; without the guard, both
   processes would try to merge and write to the same output_dir at the same
   time and corrupt the checkpoint.

This stage teaches the base model your exact JSON-RPC tool-call format and
canonical-column mapping. Run this BEFORE the QLoRA stage in 04_train_qlora.py.

IMPORTANT: this uses LoRA (full bf16 base weights, small trainable adapter),
NOT full-parameter fine-tuning. Full fine-tuning a 3B model needs the
optimizer to hold weights + gradients + Adam moments for all 3B params --
roughly 36GB+ -- which doesn't fit a single T4's 15GB. LoRA trains a
much smaller set of parameters, so memory stays dominated by the frozen
base weights (~6GB in bf16) plus a small adapter and its optimizer state,
comfortably fitting a T4 with gradient checkpointing on.

At the end, the LoRA adapter is merged back into the base weights and the
result is saved as a normal full model directory -- so 04_train_qlora.py
doesn't need to change at all; it still just points --base_model at this
script's --output_dir.

Requirements:
    pip install -q -U transformers trl peft accelerate bitsandbytes datasets
    (leave `torch` alone on Kaggle -- it's preinstalled matched to the CUDA
    driver; reinstalling it is the #1 cause of "no GPU found" on Kaggle)

Run on Kaggle (2x T4, from a notebook cell):
    !NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch \
        --multi_gpu --num_processes 2 --mixed_precision bf16 \
        03_train_sft.py \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --train_file data/train.jsonl \
        --val_file data/val.jsonl \
        --output_dir checkpoints/sft \
        --grad_accum 4   # halved vs. Colab's 8 -- see note by the arg below

    NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1: Kaggle's dual-T4 boxes don't expose
    real GPU peer-to-peer or InfiniBand links, and NCCL's default probing for
    them can hang multi-GPU jobs forever. Disabling both is the standard
    workaround and costs a little cross-GPU bandwidth, not correctness.
"""
import argparse
import os

from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--train_file", default="data/train.jsonl")
    p.add_argument("--val_file", default="data/val.jsonl")
    p.add_argument("--output_dir", default="checkpoints/sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1,
                    help="Per-GPU batch size. Keep small on a T4 -- raise grad_accum instead.")
    p.add_argument("--grad_accum", type=int, default=8,
                    help="Effective batch size = batch_size * grad_accum * num_GPUs. "
                         "On 2 GPUs this is already 2x what the same flags gave you on "
                         "1 Colab T4, so halve this value vs. your old Colab command if "
                         "you want the same effective batch size / comparable LR behavior.")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="LoRA needs a higher LR than full fine-tuning did (was 2e-5) since far fewer params are updated.")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pin this process's model copy to this process's own GPU (rank 0 -> cuda:0,
    # rank 1 -> cuda:1). Do NOT use device_map="auto" here -- see module docstring.
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype="bfloat16",
        device_map={"": device_string},
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required for gradient checkpointing + LoRA together

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    def format_chat(example):
        # apply_chat_template renders the Qwen2.5 chat format (im_start/im_end)
        # so the assistant's tool-call JSON is trained with correct special tokens
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=[c for c in dataset["train"].column_names if c != "text"])

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,  # LoRA freezes most params; skip DDP's unused-param scan
        learning_rate=args.lr,
        max_length=args.max_seq_len,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        packing=False,  # keep False: each example is a distinct tool-call decision, packing blurs boundaries
        dataset_text_field="text",
        report_to="none",
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    # Only the main process merges + saves. Under DDP every process reaches this
    # line, so without the guard both GPUs would write to output_dir at once.
    if trainer.accelerator.is_main_process:
        merged = trainer.accelerator.unwrap_model(trainer.model).merge_and_unload()
        merged.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Merged SFT model saved to {args.output_dir}")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
