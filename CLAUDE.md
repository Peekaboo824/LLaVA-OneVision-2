# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Identity

This is the **LLaVA-OneVision-1.5** training framework (PyPI name: `AIAK-Training-LLM`). It is a fork/integration of Baidu's **AIAK-Training-LLM** + **AIAK-Megatron** (an optimized Megatron-LM) tailored for training native-resolution multimodal models (LLaVA-OV-1.5 and Qwen2/2.5-VL). The `ds/` folder is a separate HuggingFace-style inference/DeepSpeed path and is **mostly independent** from the `aiak_*` training stack.

## Required Environment Setup

This project is designed to run inside the supplied Docker image (`nvcr.io/nvidia/pytorch:25.04-py3` base, A100 80GB target). Two environment variables are foundational and almost every training/conversion script relies on them:

```bash
# AIAK_TRAINING_PATH points to the repo root (the project's "training" python path)
# AIAK_MAGATRON_PATH defaults to "${AIAK_TRAINING_PATH}/aiak_megatron" if unset
export AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-1.5
# Scripts set PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH"
```

When invoking `aiak_training_llm/train.py` directly, both paths MUST be on `PYTHONPATH` — `aiak_megatron` is a vendored Megatron-LM and is imported as `megatron.*`.

Build the Docker image:
```bash
docker build -t llava_megatron:25.04 .
```

`requirements.txt` pins exact versions (e.g. `transformers==4.53.1`, `megatron-energon==5.0.0`). Do not casually bump these — the model providers and energon task encoders are coupled to these versions.

## Core Training Workflow

The end-to-end recipe is a **three-stage pipeline** with a checkpoint format conversion between every step. The user-facing recipes live in `examples/llava_ov_1_5/quick_start/` (the README walks through this in order):

1. **Initialize stage-0 weights** — either download `LLaVA-OneVision-1.5-4B-stage0` from HF, or run `python ds/merge_model.py --vit_path ... --llm_path ... --output ...` to assemble ViT + LLM + (empty) adapter.
2. **Convert HF → Megatron** — `examples/llava_ov_1_5/convert/convert_{4b,8b,3b,14b,30b_a3b}_hf_to_mcore.sh <load> <save> <TP> <PP>`. This script splits the HF checkpoint into language/vision/adapter/vision-patch sub-conversions (`tools/convert_checkpoint/model.py` + `tools/convert_checkpoint/custom/llavaov_1_5/`), then merges them via `merge_megatron.py`. The per-architecture JSON configs live in `tools/convert_checkpoint/config/llava-ov-1.5-{3b,4b,8b,14b,30b-a3b}/`.
3. **Stage 1 (alignment)** — `stage_1_alignment_llava_ov_4b.sh`. Trains adapter only (`--trainable-modules adapter`), seq-len 32k, ~2500 iters.
4. **Convert mcore checkpoint to "release"** — `convert_4b_mcore_to_release.sh <iter_dir> <out> <TP> <PP>` (drops optimizer state, prepares for next stage).
5. **Stage 1.5 (mid-training)** — `stage_1.5_mid_training_llava_ov_4b.sh`.
6. **Stage 2 (instruct/SFT)** — `stage_2_instruct_llava_ov_4b.sh`. Unfreezes everything: `--trainable-modules language_model adapter vision_model`.
7. **Convert Megatron → HF for inference** — `convert_4b_mcore_to_hf.sh`, then copy non-safetensors files (tokenizer/processor configs) from the stage-0 HF dir into the output.

All training shells expect at minimum: `AIAK_TRAINING_PATH`, `DATA_PATH` (WebDataset `.tar` shards), `TOKENIZER_PATH` (HF tokenizer dir), `CHECKPOINT_PATH` (Megatron-formatted mcore dir). They write to `$(basename script .sh)/` and `${SAVE_CKPT_PATH}/tensorboard`. Positional args control parallelism/batch: `TP PP SEQ_LEN MBS GBS NSTEP`.

Multi-node: edit `list_ip=( ... )` at the top of the launch shell — the script auto-derives `NODE_RANK` from `hostname -I` matched against the list.

## Architecture: Plugin Registration Model

The training stack uses **decorator-based registries** that you must understand before adding models or training phases. The wiring is in `aiak_training_llm/models/factory.py` and `aiak_training_llm/train/trainer_builder.py`:

- `@register_model_config(model_family, model_arch)` — registers a config dataclass for one architecture (e.g. `llava-ov-1.5-4b`). Lives next to the model code (e.g. `models/llavaov_1_5/llavaov_1_5_config.py`).
- `@register_model_provider(model_family=[...])` — registers the model-construction function used by Megatron's pretraining loop (`*_provider.py`).
- `@register_model_trainer(model_family, training_phase, ...)` — binds a `(family, phase)` pair to a training entry function. Phases are `"pretrain"` and `"sft"` (see `utils/constants.py::TrainingPhase`).

`train.py` is intentionally trivial: it calls `parse_train_args()` → `build_model_trainer(args)` → `trainer.train()`. Dispatch happens entirely through the registries above. `args.model_name` (e.g. `llava-ov-1.5-4b`) is resolved to a model family via `get_model_family()`, then `(family, training_phase)` selects the trainer function.

**Side-effect imports matter.** `aiak_training_llm/train/__init__.py` imports every `pretrain.*` and `sft.*` module purely to trigger their `@register_model_trainer` decorators. Same pattern in `aiak_training_llm/models/__init__.py` for providers. When adding a new model, you MUST add the import there or the registry stays empty and trainer lookup fails.

Supported model families are declared in `aiak_training_llm/utils/constants.py` (`LanguageModelFamilies`, `VisionLanguageModelFamilies`). New families have to be added here for arg validation to accept them.

## Architecture: Data Pipeline

Two parallel data stacks coexist:

- **Multimodal training (energon-based)** — `aiak_training_llm/data/multimodal/`. The training shells pass `--dataloader-type external` and a `--data-path` pointing at a `megatron-energon` WebDataset directory. `dataloader_provider.py` builds the dataset; `task_encoder.py` / `qwen2vl_task_encoder.py` produce model-ready batches; `flavors/` defines sample schemas (`PackedCaptioningSample`, `MultiMixQASample`, `MultiVidQASample`).
- **SFT / HF-style datasets** — `aiak_training_llm/data/{sft_dataset.py,blended_hf_dataset_*.py,chat_templete.py,mm_plugin.py}`. Used by non-multimodal SFT paths and as building blocks for chat-template handling.

Two env vars switch packing behavior at runtime (set in the launch shells, not via CLI):
- `OFFLINE_PACKED_DATA=1` — the dataset is already pre-packed (offline sample packing); the dataloader skips runtime packing.
- `OFFLINE_PACKING_VQA=1` — packed data is VQA-style (vs caption).

Stage 1 sets both to `1`; Stage 2 sets both to `0`. If you see "padding wastes FLOPs / OOM" symptoms, this is the lever.

Offline sample packing pipeline (the recommended way to produce mid-training data) lives in **two** places — they are **not equivalent**:
- `examples/llava_ov_1_5/sample_packing/` — current, supported. Driven by `offline_packing_pipeline.sh` + `config.yaml`. The 4 numbered scripts (`1_s1_get_tokenlens_v3-sft.py` → `4_convert_packedsample_to_wds.py`) run sequentially.
- `examples_offline_packing/` — **deprecated** per its own README. Do not extend.
- `tools/data_preprocess/offline_packing/` — older copies of the pipeline scripts; treat as reference, not entry points.

## Architecture: Vendored Megatron (`aiak_megatron/`)

This is a near-complete copy of Megatron-LM (`megatron/core/`, `megatron/training/`, `megatron/legacy/`, plus the `pretrain_*.py` entrypoints). The training stack imports from `megatron.*` directly. Treat this as an upstream dependency that has been customized — don't rewrite it casually, and when fixing issues that look like "Megatron bugs", check whether the fix should go into `aiak_megatron/` or into the wrapper code in `aiak_training_llm/`.

Custom transformer/attention/norm overrides intended to be applied on top of Megatron live in `aiak_training_llm/models/custom/{common,transformer}/` and are wired in through the layer specs in each model's `*_layer_spec.py`.

## Architecture: `ds/` Inference Path

`ds/` is an **independent** HuggingFace + DeepSpeed pipeline used for:
- `ds/merge_model.py` — assemble initial HF-format stage-0 weights from a ViT and an LLM. Output of this feeds the `convert_*_hf_to_mcore.sh` step.
- `ds/llavaonevision1_5/` — HF `modeling_*.py` / `configuration_*.py` files (also `*_moe.py`). These are what get uploaded with the released models for `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` to work.
- `ds/inference.py`, `ds/scripts/{pretrain,finetune}.sh`, `ds/src/` — a DeepSpeed-based finetuning/inference reference that is **not** the supported training path. Use the Megatron pipeline in `examples/llava_ov_1_5/` for actual training.

## Evaluation

There are no internal tests/lints — evaluation is delegated to the external **lmms-eval** project:

```bash
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision1_5 \
    --model_args=pretrained=<hf_model_path>,attn_implementation=flash_attention_2,max_pixels=3240000 \
    --tasks=mmmu_val,mmbench_en_test,... \
    --batch_size=1
```

The `--model=llava_onevision1_5` adapter lives in lmms-eval, not in this repo.

## Conventions When Modifying This Codebase

- **Do not mock the vendored Megatron** when reasoning about training — it executes real distributed code, and behaviors depend on `mpu` / parallel state being correctly initialized via the launch shells.
- **Keep comment language consistent with the surrounding file.** The codebase is predominantly English comments; only the offline-packing docs mix Chinese/English.
- **`configs/sft_dataset_config.json`** is the canonical SFT dataset registry consumed via `utils.utils.get_default_sft_dataset_config()`. Add new SFT datasets here rather than hardcoding paths in trainers.
- **Checkpoint format names are load-bearing:** `huggingface` (HF safetensors), `megatron` (legacy Megatron), `mcore` (Megatron-core, what training reads/writes), and `release` (mcore minus optimizer state). The conversion direction is encoded in script names (`hf_to_mcore`, `mcore_to_hf`, `mcore_to_release`).
