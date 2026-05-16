export AIAK_TRAINING_PATH=/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-2
# 让 torch.load(weights_only=False) 能反序列化 ckpt 里 args 携带的
# aiak_training_llm.train.gradient_surgery.{GradientSurgeryManager,PlanRow,AuditTarget}
export PYTHONPATH="$AIAK_TRAINING_PATH:$PYTHONPATH"

bash examples/llava_ov_1_5/convert/convert_4b_mcore_to_hf.sh \
    stage_2_instruct_llava_ov_4b-mlp-routing-1/iter_0003500 \
    LLaVA-OneVision-1.5-4B-780k-Instruct-mlp-routing-1 \
    1 1
# Copy non-model files (e.g., tokenizer config) to the new directory
find /vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-4B-stage0 -type f -not -iname '*safetensors*' -exec cp {}  LLaVA-OneVision-1.5-4B-780k-Instruct-mlp-routing-1/ ';'
