#!/usr/bin/env python3
"""
train.py
Train a ~400M GPT-2-style model on a single 16GB GPU using HF Trainer + bitsandbytes.

Usage:
  python train.py --data_dir data --model_dir checkpoints/400m --seq_len 1024 --batch_size 2 \
      --grad_accum 16 --max_steps 100000 --lr 3e-4 --bf16 --use_flash_attn

Notes:
- Loads token id JSONL produced by prep_data.py.
- Packs examples to fixed-length sequences for efficiency.
- Uses 8-bit optimizer states (adamw_bnb_8bit) to save memory.
"""
import os
import json
import argparse
import time
import math
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments, TrainerCallback
)

# -------- Dataset utilities --------

def read_token_id_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)  # each is a list[int]

def pack_sequences(token_lists, seq_len, eos_id):
    """
    Concatenate lists of token IDs, split into equal blocks of seq_len.
    Remainders are dropped (simple & effective).
    """
    buffer = []
    for ids in token_lists:
        if not ids or ids[-1] != eos_id:
            ids = ids + [eos_id]
        buffer.extend(ids)
        while len(buffer) >= seq_len + 1:
            x = buffer[:seq_len]
            y = buffer[1:seq_len+1]
            yield (x, y)
            buffer = buffer[seq_len:]

class PackedLMDataset(Dataset):
    def __init__(self, token_jsonl_path, tokenizer, seq_len):
        self.seq_len = seq_len
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
        pairs = list(pack_sequences(read_token_id_jsonl(token_jsonl_path), seq_len, eos_id))
        self.inputs = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        self.labels = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.labels[idx],
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }

# -------- Model configs (~400M target) --------

def gpt2_cfg(name: str):
    name = name.lower()
    if name == "small_test":  # ~124M
        return GPT2Config(
            vocab_size=50257 + 3,  # GPT-2 base + 3 role tokens
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_positions=1024,
            attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1
        )
    if name == "baseline_355m":
        return GPT2Config(
            vocab_size=50257 + 3,
            n_layer=24,
            n_head=16,
            n_embd=1024,
            n_positions=1024,
            attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1
        )
    if name == "heavier_420m":
        return GPT2Config(
            vocab_size=50257 + 3,
            n_layer=28,
            n_head=18,
            n_embd=1152,
            n_positions=1024,
            attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1
        )
    raise ValueError(f"Unknown config name: {name}")

class ProgressLoggerCallback(TrainerCallback):
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.last_log_time = None
        self.last_logged_step = 0
        self._tokens_per_step_cache = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_log_time = time.perf_counter()
        self.last_logged_step = state.global_step
        if getattr(args, "process_index", 0) == 0:
            print(
                f"Training started: world_size={args.world_size}, "
                f"per_device_batch_size={args.per_device_train_batch_size}, "
                f"grad_accum={args.gradient_accumulation_steps}, seq_len={self.seq_len}")

    def _tokens_per_optimizer_step(self, args):
        if self._tokens_per_step_cache is None:
            effective_batch = (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * max(1, args.world_size)
            )
            self._tokens_per_step_cache = effective_batch * self.seq_len
        return self._tokens_per_step_cache

    def on_log(self, args, state, control, logs=None, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        now = time.perf_counter()
        if self.last_log_time is None:
            self.last_log_time = now
        steps_since = max(1, state.global_step - self.last_logged_step)
        elapsed = now - self.last_log_time
        avg_step_time = elapsed / steps_since if steps_since > 0 else 0.0

        tok_per_step = self._tokens_per_optimizer_step(args)
        tok_per_sec = (tok_per_step / avg_step_time) if avg_step_time > 0 else float("inf")
        ex_per_sec = ((tok_per_step / self.seq_len) / avg_step_time) if avg_step_time > 0 else float("inf")

        loss = None
        lr = None
        if logs is not None:
            loss = logs.get("loss", logs.get("train_loss", None))
            lr = logs.get("learning_rate", None)

        msg = f"[step {state.global_step}/{args.max_steps}]"
        if loss is not None:
            msg += f" loss={loss:.4f}"
        if lr is not None:
            msg += f" lr={lr:.6g}"
        msg += f" t={avg_step_time*1000:.0f}ms ex/s={ex_per_sec:.1f} tok/s={tok_per_sec:.0f}"
        print(msg, flush=True)

        self.last_log_time = now
        self.last_logged_step = state.global_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        metrics = metrics or {}
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is not None:
            try:
                ppl = math.exp(eval_loss)
                print(f"[eval step {state.global_step}] eval_loss={eval_loss:.4f} ppl={ppl:.2f}", flush=True)
            except Exception:
                print(f"[eval step {state.global_step}] eval_loss={eval_loss:.4f}", flush=True)
        else:
            print(f"[eval step {state.global_step}] metrics: {metrics}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        print(f"[checkpoint] saved to {ckpt_dir}", flush=True)

# -------- Argument parser --------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--train_file", type=str, default="train_tokenized.jsonl")
    p.add_argument("--val_file", type=str, default="val_tokenized.jsonl")
    p.add_argument("--model_dir", type=str, default="checkpoints/400m")
    p.add_argument("--cfg", type=str, default="baseline_355m", choices=["small_test", "baseline_355m", "heavier_420m"])
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2, help="Per-device batch size.")
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--log_steps", type=int, default=100)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use_flash_attn", action="store_true", help="Only if installed & supported.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    return p

def main():
    args = build_argparser().parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]})
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"

    train_path = os.path.join(args.data_dir, args.train_file)
    val_path = os.path.join(args.data_dir, args.val_file)
    train_ds = PackedLMDataset(train_path, tokenizer, args.seq_len)
    val_ds = PackedLMDataset(val_path, tokenizer, args.seq_len)
    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)} at seq_len={args.seq_len}")

    config = gpt2_cfg(args.cfg)
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_flash_attn:
        try:
            model.config._attn_implementation = "flash_attention_2"
            print("Using FlashAttention-2 kernels (if available).")
        except Exception:
            print("FlashAttention requested but not available; continuing with default attention.")

    bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = (not bf16) and (args.fp16 or torch.cuda.is_available())

    targs = TrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.log_steps,
        logging_strategy="steps",
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="tensorboard",
        bf16=bf16,
        fp16=(not bf16) and fp16,
        optim="adamw_bnb_8bit",
        dataloader_num_workers=2
    )

    def data_collator(features):
        batch = {}
        batch["input_ids"] = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features])
        batch["labels"] = torch.stack([torch.tensor(f["labels"], dtype=torch.long) for f in features])
        batch["attention_mask"] = torch.stack([torch.tensor(f["attention_mask"], dtype=torch.long) for f in features])
        return batch

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator
    )

    trainer.add_callback(ProgressLoggerCallback(seq_len=args.seq_len))

    trainer.train()
    trainer.save_state()
    trainer.save_model(args.model_dir)
    print("Training completed.")

if __name__ == "__main__":
    main()
