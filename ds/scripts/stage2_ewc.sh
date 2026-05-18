#!/bin/bash
# Stage-2 SFT with Elastic Weight Consolidation (EWC) penalty.
#
# Anchor (theta*_A) is the ORIGINAL Qwen3-4B-Instruct LLM weights — EWC pulls
# the LLM body back toward base whenever Fisher says a coordinate matters for
# the pure-text task distribution. Vision tower / merger are untouched by EWC.
#
# Prerequisites (run once, in tools/ewc/):
#   1. python build_fisher_data.py --output fisher_anchor_dataset.jsonl
#   2. python compute_fisher_qwen3_4b.py \
#        --model-path /vepfs-mlp2/c20250505/240906016/jjy/LLaVA/checkpoints/Qwen3-4B-Instruct-2507 \
#        --dataset-path fisher_anchor_dataset.jsonl \
#        --output fisher_dict_qwen3_4b.pt
#   3. python normalize_fisher.py \
#        --input fisher_dict_qwen3_4b.pt \
#        --output fisher_dict_qwen3_4b_normalized.pt
#   4. python extract_anchor_qwen3_4b.py \
#        --model-path /vepfs-mlp2/c20250505/240906016/jjy/LLaVA/checkpoints/Qwen3-4B-Instruct-2507 \
#        --fisher-path fisher_dict_qwen3_4b.pt \
#        --output anchor_dict_qwen3_4b.pt
#
# λ tuning: Fisher is mean-normalised (per-sample g²), so λ values are in
# per-sample units. Start at 2.0 (≈ previous 1e-3 × N=2048) and watch
# loss_ewc / loss_lm in trainer logs — target ratio 0.01 ~ 0.1.

set -e

MODEL_NAME="/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-4B-stage-1-558k"

EWC_FISHER_PATH="/vepfs-mlp2/c20250505/240906016/jjy/MLLM-Pure-Text-Preservation/fisher_dict_qwen3_4b_normalized.pt"
EWC_ANCHOR_PATH="/vepfs-mlp2/c20250505/240906016/jjy/MLLM-Pure-Text-Preservation/anchor_dict_qwen3_4b.pt"
EWC_LAMBDA=2.0

GLOBAL_BATCH_SIZE=224
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Total steps ≈ 780k / 224 ≈ 3478, rounded to 3500 to match the official
# stage_2_instruct_llava_ov_4b.sh recipe. With max_steps set, HF Trainer
# ignores num_train_epochs (kept below for readability only).
MAX_STEPS=3500

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /vepfs-mlp2/c20250505/240906016/jjy/Datasets/LLaVA-NeXT-780k-unpacked/annotations.jsonl \
    --image_folder /vepfs-mlp2/c20250505/240906016/jjy/Datasets/LLaVA-NeXT-780k-unpacked \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir ./checkpoints/LLaVA-OneVision-1.5-4B-stage-2-ewc \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((20 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --adam_epsilon 1e-5 \
    --warmup_ratio 0.002 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --dataloader_num_workers 16 \
    --ewc_lambda $EWC_LAMBDA \
    --ewc_fisher_path $EWC_FISHER_PATH \
    --ewc_anchor_path $EWC_ANCHOR_PATH
