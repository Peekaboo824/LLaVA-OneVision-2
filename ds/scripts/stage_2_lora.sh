#!/bin/bash
# LLaVA-OneVision-1.5-4B stage_2 SFT with LoRA on LLaVA-NeXT-780k.
#
# Mirrors the intent of examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh
# (which runs Megatron full SFT from the stage_1.5 mid-training mcore checkpoint)
# but switches the backbone to the HF ds/ training stack with PEFT-LoRA.
#
# Hyper-parameters are kept aligned to the Megatron recipe wherever the LoRA
# substitution does not force a difference, to make the LoRA-vs-full-SFT
# comparison meaningful. Concrete alignment table:
#
#   Megatron stage_2_instruct_llava_ov_4b.sh       -> this script
#   ----------------------------------------       -------------
#   SEQ_LEN=32768                                  --max_seq_length 32768
#   MBS=1                                          --per_device_train_batch_size 1
#   GBS=224                                        GBS=224 (grad_accum=GBS/(MBS*WORLD))
#   NSTEP=3500                                     --max_steps 3500
#   --lr 1.0e-5  (full FT)                         --learning_rate 1.0e-4 (LoRA, 10x)
#   --min-lr 1.0e-6                                --learning_rate * 0.1 cosine floor
#   --clip-grad 1.0                                --max_grad_norm 1.0
#   --weight-decay 0                               --weight_decay 0
#   --optimizer adam (== AdamW)                    --optim adamw_torch
#   --adam-beta1 0.9 / -beta2 0.99 / -eps 1e-5     --adam_beta1 0.9 / -beta2 0.99 / -epsilon 1e-5
#   --lr-decay-style cosine                        --lr_scheduler_type cosine
#   --lr-warmup-fraction 0.002                     --warmup_ratio 0.002
#   --bf16                                         --bf16 True
#   --recompute-granularity full ...               --gradient_checkpointing True
#   --image-resolution 1000                        --image_max_pixels 1280*28*28 (~1.0MP)
#   --attention-backend flash                      --disable_flash_attn2 False
#   --num-workers 16                               --dataloader_num_workers 16
#   --save-interval 2000                           --save_steps 2000
#   --trainable-modules language_model adapter     LoRA on LLM + vision; merger trained
#   vision_model (full FT)                         in full (this is the LoRA delta)
#
# LoRA-specific knobs:
#   r=32, alpha=64, dropout=0.05, bias=none
#   lora_namespan_exclude=["lm_head","embed_tokens"] to dodge tied-weight
#   warning from PEFT (model has tie_word_embeddings=True).
#
# Usage:
#   bash ds/scripts/stage_2_lora.sh
#   # override on the command line:
#   MODEL_NAME=/path GBS=64 LR=2e-4 bash ds/scripts/stage_2_lora.sh

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

# ---------------- batch sizing (matches Megatron stage_2 GBS=224) ----------------
GBS="${GBS:-224}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
NUM_DEVICES="${NUM_DEVICES:-8}"
GRAD_ACCUM_STEPS=$(( GBS / (BATCH_PER_DEVICE * NUM_DEVICES) ))
if [[ $GRAD_ACCUM_STEPS -lt 1 ]]; then
    GRAD_ACCUM_STEPS=1
fi

# ---------------- training length (matches Megatron NSTEP=3500) ----------------
MAX_STEPS="${MAX_STEPS:-3500}"
SAVE_STEPS="${SAVE_STEPS:-2000}"

# ---------------- LoRA hyper-params ----------------
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ---------------- learning rates ----------------
# Megatron full-FT uses lr=1e-5; LoRA needs ~10x to learn meaningfully because
# the B matrix starts at zero. 1e-4 is the conservative LoRA default that's
# closest in spirit to the Megatron baseline while still allowing convergence.
LR="${LR:-1e-4}"
# merger / vision lr stay full-FT scale; only LoRA params move at $LR.
MERGER_LR="${MERGER_LR:-1e-5}"
VISION_LR="${VISION_LR:-2e-6}"

# ---------------- image / sequence budget (matches Megatron) ----------------
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
# image-resolution 1000 in Megatron ~= 1000^2 = 1M pixels; 1280*28*28 = 1.003MP.
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-$((4 * 28 * 28))}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-$((1280 * 28 * 28))}"

# ---------------- NCCL safety nets ----------------
# 4B + LoRA + variable-length VLM samples produce wildly uneven per-rank
# compute; default 600s collective timeout can fire on the first ALLREDUCE.
# Bump to 30 min and force watchdog blocking (raises a Python error instead
# of SIGABRT, easier to debug).
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

# ---------------- run ----------------
export PYTHONPATH=.:src:src/train:${PYTHONPATH:-}

echo "[stage_2_lora] cwd:           $(pwd)"
echo "[stage_2_lora] base model:    $MODEL_NAME"
echo "[stage_2_lora] data path:     $DATA_PATH"
echo "[stage_2_lora] image folder:  $IMAGE_FOLDER"
echo "[stage_2_lora] output dir:    $OUTPUT_DIR"
echo "[stage_2_lora] GBS=${GBS} per_dev=${BATCH_PER_DEVICE} devices=${NUM_DEVICES} grad_accum=${GRAD_ACCUM_STEPS}"
echo "[stage_2_lora] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS}"
echo "[stage_2_lora] LoRA r=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "[stage_2_lora] LR=${LR} (merger=${MERGER_LR} vision=${VISION_LR})"
echo "[stage_2_lora] seq_len=${SEQ_LENGTH} image_max_pixels=${IMAGE_MAX_PIXELS}"
echo "[stage_2_lora] deepspeed config: ${DEEPSPEED_CONFIG}"
echo "[stage_2_lora] NCCL_TIMEOUT=${NCCL_TIMEOUT} TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT}"

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
    --lora_namespan_exclude '["lm_head","embed_tokens"]' \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --max_seq_length "$SEQ_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --learning_rate "$LR" \
    --merger_lr "$MERGER_LR" \
    --vision_lr "$VISION_LR" \
    --weight_decay 0. \
    --warmup_ratio 0.002 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --adam_epsilon 1e-5 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 2 \
    --dataloader_num_workers 16 \
    2>&1 | tee "${OUTPUT_DIR}/run_$(date +%Y-%m-%d_%H-%M-%S).log"
