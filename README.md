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

## Two-Stage From-Scratch Training (Recommended if not using pretrained)

If you are not starting from any pretrained model, use this two-stage pipeline:

### Stage A — General Causal-LM Pretraining on Raw Text
- Prepare a large general text corpus (e.g., OpenWebText/C4 subsets):
```bash
python prep_text_corpus.py --dataset openwebtext --out_dir data_pretrain
```
- Train with full-token loss (disable assistant-only loss):
```bash
nohup python -u train.py \
  --data_dir data_pretrain \
  --model_dir checkpoints/pretrain_400m \
  --cfg baseline_355m \
  --seq_len 1024 \
  --batch_size 2 \
  --grad_accum 16 \
  --lr 3e-4 \
  --warmup_steps 2000 \
  --max_steps 100000 \
  --bf16 \
  --no_assistant_only_loss \
  --log_steps 100 > pretrain.log 2>&1 &
```

### Stage B — Supervised Fine-Tuning (SFT) on Instruction Data
- Prepare instruction-following data (role-aware):
```bash
python prep_data.py --dataset tatsu-lab/alpaca --out_dir data_sft
```
- Train with assistant-only loss, initializing from Stage A:
```bash
nohup python -u train.py \
  --data_dir data_sft \
  --model_dir checkpoints/sft_400m \
  --init_model_dir checkpoints/pretrain_400m \
  --cfg baseline_355m \
  --seq_len 1024 \
  --batch_size 2 \
  --grad_accum 16 \
  --lr 1e-4 \
  --warmup_steps 1000 \
  --max_steps 50000 \
  --weight_decay 0.1 \
  --bf16 \
  --log_steps 100 > sft.log 2>&1 &
```

Notes:
- Checkpoints are saved every `--save_steps` (default 1000) and pruned to the last two by default.
- You can resume Stage B from a specific checkpoint by passing `--init_model_dir checkpoints/pretrain_400m/checkpoint-<step>`.

### Inference while training (CPU, latest checkpoint)
```bash
LATEST=$(ls -1dt checkpoints/sft_400m/checkpoint-* 2>/dev/null | head -1) && \
python inference.py --model_dir "$LATEST" --device cpu --temperature 0.2
```

## Repo Layout
```
data/               # Prepared JSONL + tokenized files
checkpoints/        # Saved models
prep_data.py        # Prepare + tokenize dataset
prep_text_corpus.py # Prepare raw text corpora for Stage A pretraining
train.py            # Training loop
inference.py        # Chat REPL
tutorial.md         # Full step-by-step guide
run_all.sh          # One-liner demo runner
requirements.txt    # Pip deps (minimal)
environment.yml     # Conda env (optional)
```
See **tutorial.md** for details.
