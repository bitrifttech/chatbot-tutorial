#!/usr/bin/env python3
"""
prep_data.py
Prepare conversational JSONL and tokenized JSONL for a GPT-2-style chatbot.

Usage:
  python prep_data.py --dataset tatsu-lab/alpaca --train_split train[:98%] --val_split train[-2%:] \
      --out_dir data --sample_pct 100 --max_samples 0

Notes:
- Keeps things simple & transparent: we output JSONL of token id lists per example.
- Packing into fixed-length sequences happens in train.py.
"""
import os
import json
import argparse
from datasets import load_dataset
from transformers import GPT2TokenizerFast

SPECIAL_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "eos": "<|endoftext|>"
}

DEFAULT_SYSTEM = "You are a helpful, concise assistant."

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--train_split", type=str, default="train[:98%]")
    p.add_argument("--val_split", type=str, default="train[-2%:]")
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--sample_pct", type=int, default=100, help="Percentage of split to keep (for quick tests).")
    p.add_argument("--max_samples", type=int, default=0, help="Hard cap for number of samples (0 = no cap).")
    p.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM)
    return p

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def format_to_schema(example, system_prompt):
    # Alpaca-like: instruction/input/output (others can be adapted similarly)
    inst = example.get("instruction", "") or example.get("question", "") or example.get("prompt", "")
    inp = example.get("input", "")
    if inp:
        user_msg = f"{inst}\n\n{inp}".strip()
    else:
        user_msg = inst.strip()
    assistant_msg = example.get("output", "") or example.get("response", "") or ""

    return {
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }

def to_text(sample):
    return (
        f"{SPECIAL_TOKENS['system']} {sample['system']} "
        f"{SPECIAL_TOKENS['user']} {sample['messages'][0]['content']} "
        f"{SPECIAL_TOKENS['assistant']} {sample['messages'][1]['content']} "
        f"{SPECIAL_TOKENS['eos']}"
    )

def maybe_subsample(ds, pct, cap):
    if pct and pct < 100:
        n = int(len(ds) * (pct / 100.0))
        ds = ds.select(range(max(1, n)))
    if cap and cap > 0:
        n = min(len(ds), cap)
        ds = ds.select(range(n))
    return ds

def dump_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def dump_token_ids_jsonl(tokenizer, schema_jsonl_path, out_path):
    with open(schema_jsonl_path, "r", encoding="utf-8") as fin,          open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            text = to_text(obj)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            fout.write(json.dumps(ids) + "\n")

def main():
    args = build_argparser().parse_args()
    ensure_dir(args.out_dir)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Add only role tokens; GPT-2 already has <|endoftext|> as eos.
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]})
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"

    print(f"Loading dataset: {args.dataset}")
    train_ds = load_dataset(args.dataset, split=args.train_split)
    val_ds = load_dataset(args.dataset, split=args.val_split)

    train_ds = maybe_subsample(train_ds, args.sample_pct, args.max_samples)
    val_ds = maybe_subsample(val_ds, args.sample_pct, args.max_samples // 10 if args.max_samples else 0)

    train_rows = [format_to_schema(ex, args.system_prompt) for ex in train_ds]
    val_rows = [format_to_schema(ex, args.system_prompt) for ex in val_ds]

    schema_train = os.path.join(args.out_dir, "train.jsonl")
    schema_val = os.path.join(args.out_dir, "val.jsonl")
    dump_jsonl(train_rows, schema_train)
    dump_jsonl(val_rows, schema_val)
    print(f"Wrote role-aware JSONL: {schema_train}, {schema_val}")

    tok_train = os.path.join(args.out_dir, "train_tokenized.jsonl")
    tok_val = os.path.join(args.out_dir, "val_tokenized.jsonl")
    dump_token_ids_jsonl(tokenizer, schema_train, tok_train)
    dump_token_ids_jsonl(tokenizer, schema_val, tok_val)
    print(f"Wrote token id JSONL: {tok_train}, {tok_val}")

if __name__ == "__main__":
    main()
