#!/usr/bin/env bash
set -euo pipefail

# ---- Config (override via env or flags) --------------------------------------
DATASET="${DATASET:-tatsu-lab/alpaca}"        # any HF dataset path
OUT_DIR="${OUT_DIR:-data}"
MODEL_DIR="${MODEL_DIR:-checkpoints/quickstart}"
CFG="${CFG:-small_test}"                      # small_test | baseline_355m | heavier_420m
SEQ_LEN="${SEQ_LEN:-768}"                     # 768 is kinder to 16GB for a quick run
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-200}"                       # small warmup for quick demo
MAX_STEPS="${MAX_STEPS:-800}"                 # quick demo; increase for real training
USE_BF16="${USE_BF16:-auto}"                  # auto|true|false
USE_FP16="${USE_FP16:-auto}"                  # auto|true|false
USE_FLASH_ATTN="${USE_FLASH_ATTN:-auto}"      # auto|true|false

has_cmd() { command -v "$1" >/dev/null 2>&1; }
py() { python - <<'PY'
import torch, json, importlib.util
cap=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
print(json.dumps({
  "cuda": torch.cuda.is_available(),
  "bf16_capable": (cap[0] >= 8),
  "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None
}))
PY
}

banner() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }

usage() {
  cat <<USAGE
Usage: env VAR=VALUE ... bash run_all.sh
  DATASET           (default: $DATASET)
  OUT_DIR           (default: $OUT_DIR)
  MODEL_DIR         (default: $MODEL_DIR)
  CFG               (default: $CFG)  [small_test|baseline_355m|heavier_420m]
  SEQ_LEN           (default: $SEQ_LEN)
  BATCH_SIZE        (default: $BATCH_SIZE)
  GRAD_ACCUM        (default: $GRAD_ACCUM)
  LR                (default: $LR)
  WARMUP            (default: $WARMUP)
  MAX_STEPS         (default: $MAX_STEPS)
  USE_BF16          (default: $USE_BF16)  [auto|true|false]
  USE_FP16          (default: $USE_FP16)  [auto|true|false]
  USE_FLASH_ATTN    (default: $USE_FLASH_ATTN) [auto|true|false]
Examples:
  bash run_all.sh
  CFG=baseline_355m SEQ_LEN=1024 MAX_STEPS=100000 MODEL_DIR=checkpoints/400m bash run_all.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi

banner "Checking Python and pip"
if ! has_cmd python; then echo "ERROR: python not found"; exit 1; fi
if ! has_cmd pip; then echo "ERROR: pip not found"; exit 1; fi

banner "Installing Python deps (if missing)"
pip install -q "transformers>=4.41" datasets accelerate tensorboard bitsandbytes || true
if [[ "${USE_FLASH_ATTN}" == "true" ]]; then
  pip install -q flash-attn || echo "flash-attn install failed; continuing without it"
fi

banner "Probing CUDA/bf16/flash-attn support"
probe_json="$(py)"
cuda=$(python - <<PY
import json; j=json.loads("""$probe_json"""); print("1" if j["cuda"] else "0")
PY
)
bf16_cap=$(python - <<PY
import json; j=json.loads("""$probe_json"""); print("1" if j["bf16_capable"] else "0")
PY
)
flash_inst=$(python - <<PY
import json; j=json.loads("""$probe_json"""); print("1" if j["flash_attn_installed"] else "0")
PY
)

BF16_FLAG=""; FP16_FLAG=""
if [[ "$USE_BF16" == "true" || ( "$USE_BF16" == "auto" && "$bf16_cap" == "1" ) ]]; then
  BF16_FLAG="--bf16"
elif [[ "$USE_FP16" == "true" || ( "$USE_FP16" == "auto" && "$cuda" == "1" ) ]]; then
  FP16_FLAG="--fp16"
fi

FLASH_FLAG=""
if [[ "$USE_FLASH_ATTN" == "true" || ( "$USE_FLASH_ATTN" == "auto" && "$flash_inst" == "1" ) ]]; then
  FLASH_FLAG="--use_flash_attn"
fi

echo "CUDA available:        $([[ "$cuda" == "1" ]] && echo yes || echo no)"
echo "bf16 capable (Ampere): $([[ "$bf16_cap" == "1" ]] && echo yes || echo no)"
echo "flash-attn installed:  $([[ "$flash_inst" == "1" ]] && echo yes || echo no)"
echo "Precision flags:       ${BF16_FLAG} ${FP16_FLAG}"
echo "FlashAttention flag:   ${FLASH_FLAG}"

banner "Preparing data → ${OUT_DIR}"
mkdir -p "$OUT_DIR"
python prep_data.py --dataset "${DATASET}" --out_dir "${OUT_DIR}"

banner "Training model → ${MODEL_DIR}"
mkdir -p "$(dirname "$MODEL_DIR")"
python train.py   --data_dir "${OUT_DIR}"   --model_dir "${MODEL_DIR}"   --cfg "${CFG}"   --seq_len "${SEQ_LEN}"   --batch_size "${BATCH_SIZE}"   --grad_accum "${GRAD_ACCUM}"   --lr "${LR}"   --warmup_steps "${WARMUP}"   --max_steps "${MAX_STEPS}"   ${BF16_FLAG} ${FP16_FLAG} ${FLASH_FLAG}

banner "Launching chat REPL (Ctrl+C to exit)"
python inference.py --model_dir "${MODEL_DIR}"
