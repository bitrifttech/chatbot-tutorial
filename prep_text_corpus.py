#!/usr/bin/env python3
"""
prep_text_corpus.py
Prepare raw text corpora into token ID JSONL for Stage A causal-LM pretraining (from scratch).

Usage examples:
  python prep_text_corpus.py --dataset Skylion007/openwebtext --out_dir data_pretrain
  python prep_text_corpus.py --dataset allenai/c4 --dataset_config en --train_split train[:99%] --val_split train[-1%:] --out_dir data_pretrain_c4

Notes:
- Outputs two files in --out_dir:
    - train_tokenized.jsonl
    - val_tokenized.jsonl
- Each line is a JSON array of token IDs (no special BOS/EOS added here); training code will append EOS between samples when packing.
- We keep tokenizer setup consistent with train/inference: add the 3 role tokens so vocab matches Stage B.
"""
import os
import json
import argparse
from typing import Optional

from datasets import load_dataset
from transformers import GPT2TokenizerFast


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Skylion007/openwebtext",
                   help="HF dataset name or path, e.g., 'Skylion007/openwebtext' or 'allenai/c4'")
    p.add_argument("--train_split", type=str, default="train[:99%]")
    p.add_argument("--val_split", type=str, default="train[-1%:]")
    p.add_argument("--text_column", type=str, default="",
                   help="Name of the text column. If empty, will try 'text' then fall back to the first string column.")
    p.add_argument("--dataset_config", type=str, default="",
                   help="Optional dataset config/name (e.g., for 'allenai/c4' use 'en').")
    p.add_argument("--out_dir", type=str, default="data_pretrain")
    p.add_argument("--sample_pct", type=int, default=100,
                   help="Percentage of split to keep (for quick tests).")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Hard cap on number of samples per split (0 = no cap).")
    return p


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def pick_text_column(ds, explicit: Optional[str]) -> str:
    if explicit:
        if explicit in ds.column_names:
            return explicit
        raise ValueError(f"text_column='{explicit}' not found. Available: {ds.column_names}")
    if "text" in ds.column_names:
        return "text"
    # Fall back to first string-like column
    for name in ds.column_names:
        # heuristic: treat as text if values are strings for the first few rows
        try:
            sample = ds[name][0]
            if isinstance(sample, str):
                return name
        except Exception:
            continue
    raise ValueError(f"Could not infer text column from columns: {ds.column_names}")


def maybe_subsample(ds, pct: int, cap: int):
    if pct and pct < 100:
        n = int(len(ds) * (pct / 100.0))
        ds = ds.select(range(max(1, n)))
    if cap and cap > 0:
        n = min(len(ds), cap)
        ds = ds.select(range(n))
    return ds


def dump_token_ids_jsonl(tokenizer, ds, text_col: str, out_path: str):
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in ds:
            text = ex.get(text_col, None)
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False).input_ids
            fout.write(json.dumps(ids) + "\n")


def main():
    args = build_argparser().parse_args()
    ensure_dir(args.out_dir)

    # Map legacy dataset IDs to Hub repos since datasets>=3 removed script-based datasets
    if args.dataset.strip().lower() == "openwebtext":
        print("Remapping 'openwebtext' to 'Skylion007/openwebtext' (datasets>=3 removed script-based datasets).")
        args.dataset = "Skylion007/openwebtext"

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Keep vocab consistent with Stage B (role tokens exist, even if unused here)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]})
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"

    print(f"Loading dataset: {args.dataset}" + (f" (config={args.dataset_config})" if args.dataset_config else ""))
    if args.dataset_config:
        train_ds = load_dataset(args.dataset, args.dataset_config, split=args.train_split)
        val_ds = load_dataset(args.dataset, args.dataset_config, split=args.val_split)
    else:
        train_ds = load_dataset(args.dataset, split=args.train_split)
        val_ds = load_dataset(args.dataset, split=args.val_split)

    text_col_train = pick_text_column(train_ds, args.text_column)
    text_col_val = pick_text_column(val_ds, args.text_column or text_col_train)

    train_ds = maybe_subsample(train_ds, args.sample_pct, args.max_samples)
    val_cap = args.max_samples // 10 if args.max_samples else 0
    val_ds = maybe_subsample(val_ds, args.sample_pct, val_cap)

    tok_train = os.path.join(args.out_dir, "train_tokenized.jsonl")
    tok_val = os.path.join(args.out_dir, "val_tokenized.jsonl")

    dump_token_ids_jsonl(tokenizer, train_ds, text_col_train, tok_train)
    dump_token_ids_jsonl(tokenizer, val_ds, text_col_val, tok_val)

    print(f"Wrote token id JSONL: {tok_train}, {tok_val}")


if __name__ == "__main__":
    main()
