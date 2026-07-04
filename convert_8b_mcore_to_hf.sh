export AIAK_TRAINING_PATH=/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-2
export PYTHONPATH="$AIAK_TRAINING_PATH:$PYTHONPATH"

bash examples/llava_ov_1_5/convert/convert_8b_mcore_to_hf.sh \
stage_2_instruct_llava_ov_8b-shared-mixed-reg-rl2-0.1-cos-0.1/iter_0003500 \
LLaVA-OneVision-1.5-8B-780k-Instruct-shared-mixed-reg-rl2-0.1-cos-0.1 \
1 1
# Copy non-model files (e.g., tokenizer config) to the new directory
find /vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-8B-stage0 -type f -not -iname '*safetensors*' -exec cp {}  LLaVA-OneVision-1.5-8B-780k-Instruct-shared-mixed-reg-rl2-0.1-cos-0.1/ ';'
