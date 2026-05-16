#!/bin/bash
# LLaVA-OneVision-1.5-4B stage_2 SFT with LoRA on LLaVA-NeXT-780k.
#
# Mirrors the intent of examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh
# (which runs Megatron full SFT from the stage_1.5 mid-training mcore checkpoint)
# but switches the backbone to the HF ds/ training stack with PEFT-LoRA.
#
# Scope (per train_sft.py constraints at lines 98-115):
#   - lora_enable=True, freeze_llm=True       (LoRA replaces full LLM tuning)
#   - vision_lora=True, freeze_vision_tower=True   (LoRA also wraps the ViT)
#   - freeze_merger=False                     (merger trained in full like in
#                                              the Megatron stage_2 recipe)
#
# Data: produced by tools/etl/webdataset_to_jsonl.py from
#       /vepfs-mlp2/.../Datasets/LLaVA-NeXT-780k-webdataset.
#       Layout:
#         DATA_ROOT/annotations.jsonl
#         DATA_ROOT/images/pretrain-{N}/<filename>
#
# Usage:
#   bash ds/scripts/stage_2_lora.sh
#   # override on the command line:
#   MODEL_NAME=/path/to/base GBS=64 LR=1e-4 bash ds/scripts/stage_2_lora.sh

set -euo pipefail

# Always run from ds/ so that the relative imports inside train_sft.py
# (`from llavaonevision1_5...`, `from src...`) resolve. The finetune.sh /
# pretrain.sh scripts in this directory rely on the same convention.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/ds"

# ---------------- paths ----------------
MODEL_NAME="${MODEL_NAME:-/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-4B-stage-1-558k}"
DATA_ROOT="${DATA_ROOT:-/vepfs-mlp2/c20250505/240906016/jjy/Datasets/LLaVA-NeXT-780k-unpacked}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/annotations.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_ROOT}}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/stage_2_sft_lora_r32a64_4b}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"

mkdir -p "$OUTPUT_DIR"

# ---------------- batch sizing ----------------
GBS="${GBS:-128}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
NUM_DEVICES="${NUM_DEVICES:-8}"
GRAD_ACCUM_STEPS=$(( GBS / (BATCH_PER_DEVICE * NUM_DEVICES) ))
if [[ $GRAD_ACCUM_STEPS -lt 1 ]]; then
    GRAD_ACCUM_STEPS=1
fi

# ---------------- LoRA hyper-params ----------------
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ---------------- learning rates ----------------
# LoRA adapters get a higher lr than full-FT does; merger and (frozen) vision
# stay close to the original stage_2 recipe.
LR="${LR:-2e-4}"
MERGER_LR="${MERGER_LR:-1e-5}"
VISION_LR="${VISION_LR:-2e-6}"

# ---------------- image budget ----------------
# Match the SHED defaults from ds/scripts/finetune.sh.
IMAGE_MIN_PIXELS=$((20 * 28 * 28))
IMAGE_MAX_PIXELS=$((1280 * 28 * 28))

# ---------------- run ----------------
export PYTHONPATH=.:src:src/train:${PYTHONPATH:-}

echo "[stage_2_lora] cwd:         $(pwd)"
echo "[stage_2_lora] base model:  $MODEL_NAME"
echo "[stage_2_lora] data path:   $DATA_PATH"
echo "[stage_2_lora] image folder:$IMAGE_FOLDER"
echo "[stage_2_lora] output dir:  $OUTPUT_DIR"
echo "[stage_2_lora] GBS=${GBS} per_dev=${BATCH_PER_DEVICE} devices=${NUM_DEVICES} grad_accum=${GRAD_ACCUM_STEPS}"
echo "[stage_2_lora] LoRA r=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "[stage_2_lora] deepspeed config: ${DEEPSPEED_CONFIG}"

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --lora_enable True \
    --vision_lora True \
    --freeze_llm True \
    --freeze_vision_tower True \
    --freeze_merger False \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_bias none \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --learning_rate "$LR" \
    --merger_lr "$MERGER_LR" \
    --vision_lr "$VISION_LR" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --dataloader_num_workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/run_$(date +%Y-%m-%d_%H-%M-%S).log"
