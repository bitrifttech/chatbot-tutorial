# 🚀 GPT-2 Style Chatbot (~400M) on 16GB GPU

Train a **GPT-2 style chatbot from scratch** on a single **RTX 4060 (16 GB)**.
Focus: **practical, role-aware chatbot training** with open datasets.

## Quickstart

### 1) Install
```bash
pip install "transformers>=4.41" datasets accelerate tensorboard bitsandbytes
# Optional (if supported by your GPU/driver):
# pip install flash-attn
```

### 2) Prepare Data
By default this uses the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca):
```bash
python prep_data.py --dataset tatsu-lab/alpaca --out_dir data
```
This creates:
- `data/train.jsonl`, `data/val.jsonl` → role-aware schema
- `data/train_tokenized.jsonl`, `data/val_tokenized.jsonl` → token IDs

### 3) Train
Train the ~355M baseline config (fits in 16 GB):
```bash
python train.py   --data_dir data   --model_dir checkpoints/400m   --cfg baseline_355m   --seq_len 1024   --batch_size 2   --grad_accum 16   --lr 3e-4   --warmup_steps 2000   --max_steps 100000   --bf16
```
Tips:
- Lower `--seq_len` (e.g., 768) if you OOM.
- Try `--cfg small_test` for quick smoke tests.
- `tensorboard --logdir checkpoints/400m/runs` to monitor.

### 4) Chat
```bash
python inference.py --model_dir checkpoints/400m
```

## Repo Layout
```
data/               # Prepared JSONL + tokenized files
checkpoints/        # Saved models
prep_data.py        # Prepare + tokenize dataset
train.py            # Training loop
inference.py        # Chat REPL
tutorial.md         # Full step-by-step guide
run_all.sh          # One-liner demo runner
requirements.txt    # Pip deps (minimal)
environment.yml     # Conda env (optional)
```
See **tutorial.md** for details.
