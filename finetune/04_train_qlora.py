"""
Stage 2: QLoRA (PEFT) on top of the Stage-1 SFT checkpoint.

KAGGLE 2xT4 VERSION -- same core change as 03_train_sft.py: device_map="auto"
is replaced with device_map={"": PartialState().process_index} so each
accelerate-launched process loads its own 4-bit copy onto its own GPU instead
of every process trying to shard across both T4s at once. Also adds
ddp_find_unused_parameters=False for the same LoRA/DDP reason as stage 1.
`trainer.save_model()` at the end is already rank-zero-safe inside HF
Trainer, so no extra guard is needed there (unlike stage 1's manual merge).

Loads the model in 4-bit (bitsandbytes NF4), attaches LoRA adapters, and
trains on the harder slice (adversarial + noisy + multi_step) to squeeze in
robustness without touching most of the base weights -- this is what keeps
inference cheap enough for the "sub-3.2s latency using 4-bit quantized
inference" claim in your README (same 4-bit weights used for training ARE
what you deploy).

Requirements:
    pip install -q -U transformers trl peft accelerate bitsandbytes datasets

Run on Kaggle (2x T4, from a notebook cell):
    !NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch \
        --multi_gpu --num_processes 2 --mixed_precision bf16 \
        04_train_qlora.py \
        --base_model checkpoints/sft \
        --train_file data/train.jsonl \
        --val_file data/val.jsonl \
        --output_dir checkpoints/qlora \
        --grad_accum 4
"""
import argparse
import os

from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="checkpoints/sft",
                   help="Path to Stage-1 SFT checkpoint (or Qwen/Qwen2.5-3B-Instruct for QLoRA-only)")
    p.add_argument("--train_file", default="data/train.jsonl")
    p.add_argument("--val_file", default="data/val.jsonl")
    p.add_argument("--output_dir", default="checkpoints/qlora")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1,
                    help="Per-GPU batch size. Keep small on a T4 -- raise grad_accum instead.")
    p.add_argument("--grad_accum", type=int, default=8,
                    help="Effective batch size = batch_size * grad_accum * num_GPUs. "
                         "Halve this vs. your old Colab command since 2 GPUs already double it.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_seq_len", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pin this process's 4-bit model copy to this process's own GPU -- see
    # 03_train_sft.py's docstring for why "auto" breaks under multi-GPU DDP.
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": device_string},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen2 attention + MLP proj names -- covering both keeps tool-call
        # reasoning (attention) and JSON formatting (MLP) both adaptable
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    def format_chat(example):
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
        packing=False,
        dataset_text_field="text",
        report_to="none",
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
    trainer.save_model(args.output_dir)  # saves LoRA adapter only; HF Trainer already handles this rank-zero-only
    tokenizer.save_pretrained(args.output_dir)
    print(f"QLoRA adapter saved to {args.output_dir}")
    print("To merge for deployment: model.merge_and_unload() then save_pretrained()")


if __name__ == "__main__":
    main()
