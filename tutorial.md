# 🛠️ Tutorial: Training a ~400M Parameter GPT-2 Style Chatbot on a 16GB GPU

This tutorial walks you step-by-step through training a **GPT-2 style chatbot** on a single **RTX 4060 (16 GB VRAM)**.  
It’s written for **seasoned Python developers** who are **new to training LLMs**.

👉 Focus is **practical, not theoretical**: you’ll get a working chatbot that can run interactively.

---

## 1. Overview
- **Architecture:** GPT-2 style decoder with modern tweaks (mixed precision, optional FlashAttention, gradient checkpointing).
- **Size target:** ~355M–420M parameters (fits in 16 GB).
- **Tokenizer:** GPT-2 BPE with added role tokens.
- **Training data:** Open conversational datasets (e.g., Alpaca).
- **Training time:** ≤ 1 week on consumer GPU.
- **Output:** A trained chatbot you can run in a REPL.

## 2. Prerequisites
- **OS:** Linux (CUDA installed, tested on Ubuntu 22.04).
- **GPU:** RTX 4060 16 GB (Ampere).
- **Python:** 3.9–3.12.

Install dependencies:
```bash
pip install "transformers>=4.41" datasets accelerate tensorboard bitsandbytes
```
Optional for faster attention kernels:
```bash
pip install flash-attn
```

## 3. Repo Structure
```
chatbot-tutorial/
├── data/                 # JSONL + tokenized files go here
├── checkpoints/          # Saved models
├── prep_data.py          # Convert + tokenize datasets
├── train.py              # Train GPT-2 style chatbot
├── inference.py          # Chat REPL
├── run_all.sh            # One-liner demo runner
└── tutorial.md           # This guide
```

## 4. Data Preparation
We use open datasets (e.g., tatsu-lab/alpaca).

Run:
```bash
python prep_data.py --dataset tatsu-lab/alpaca --out_dir data
```
This will produce:
- `data/train.jsonl` → role-aware schema
- `data/train_tokenized.jsonl` → token ID sequences
- Same for validation split

### `prep_data.py`
See the file in this repo for full code; key points:
- Converts dataset rows to a JSONL schema:
```json
{
  "system": "You are a helpful assistant.",
  "messages": [
    {"role":"user","content":"..."},
    {"role":"assistant","content":"..."}
  ]
}
```
- Uses the role tokens `<|system|>`, `<|user|>`, `<|assistant|>` and `<|endoftext|>` in the rendered text.

## 5. Model Training
We support three configs:

| Config          | Layers | d_model | Heads | Params | Fits 16 GB? |
|-----------------|--------|---------|-------|--------|-------------|
| small_test      | 12     | 768     | 12    | ~124M  | ✅ easily   |
| baseline_355m   | 24     | 1024    | 16    | ~355M  | ✅ main rec |
| heavier_420m    | 28     | 1152    | 18    | ~420M  | ⚠ close fit |

### Train
```bash
python train.py   --data_dir data   --model_dir checkpoints/400m   --cfg baseline_355m   --seq_len 1024   --batch_size 2   --grad_accum 16   --lr 3e-4   --warmup_steps 2000   --max_steps 100000   --bf16
```
Tips for 16 GB:
- Use `--bf16` (preferred on Ampere). Fall back to `--fp16` if needed.
- Adjust `--seq_len` (e.g., 768) if you OOM.
- Use `--grad_accum` to scale effective batch size.

## 6. Evaluation
Basic evaluation = **perplexity** on validation set:
```bash
tensorboard --logdir checkpoints/400m/runs
```
Optional: add held-out chat prompts and manually check responses.

## 7. Inference
Once trained:
```bash
python inference.py --model_dir checkpoints/400m
```
Example:
```
User: Who won the Battle of Waterloo?
Assistant: The Battle of Waterloo was won by the Seventh Coalition forces led by the Duke of Wellington and Gebhard Leberecht von Blücher, defeating Napoleon Bonaparte.
```

## 8. Troubleshooting
- **CUDA OOM** → Lower `--seq_len` or use `--cfg small_test`.
- **Training too slow** → Reduce `--max_steps` for a demo run.
- **Weird responses** → Train longer or try another dataset.
- **No role separation** → Ensure special tokens `<|system|>`, `<|user|>`, `<|assistant|>` are present in data.

## 9. Upgrade Paths
- Swap dataset: use any HF dataset path via `--dataset`.
- Extend context: increase `--seq_len` to 2048 (watch memory).
- Add FlashAttention: `--use_flash_attn` for faster training (if installed).
- Normalize better: try RMSNorm or RoPE variants (advanced).
- Scale up: `heavier_420m` if VRAM allows.

## 10. Wrap-Up
You now have:
- A **role-aware JSONL schema** that fixes user/assistant confusion.
- A **training pipeline** that fits in 16 GB with mixed precision.
- A **working REPL chatbot**.

This is your foundation—extend with better datasets, longer training, and quantized inference.
