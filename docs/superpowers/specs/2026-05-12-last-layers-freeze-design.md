# Design — 冻结 LLM 尾部层（last-layers freeze）

- 日期: 2026-05-12
- 分支: `last-layers`
- 适用模型: `llava-ov-1.5-*`（本次目标场景: `llava-ov-1.5-4b`，36 层）
- 训练入口: `examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh`

## 目标

在 stage 2 SFT 训练时，把语言模型（`QwenModel`）尾部的若干层及 `final_layernorm`、`output_layer` 冻结（`requires_grad=False`），仅训练浅层 transformer、vision model、adapter。本次具体切点: **冻 `decoder.layers[24..35]` + `final_layernorm` + `output_layer`**；保留 `decoder.layers[0..23]`、`vision_model`、`adapter` 可训练。通过一个可配置 CLI 参数实现，便于消融不同切点。

## 动机与范围

- 探索"冻深层 LLM 读出能力、只训浅层表示 + 视觉侧"的训练策略。
- 与现有 `--trainable-modules` 正交：模块级冻结（粗粒度）与层级冻结（细粒度）互补。
- 范围外（YAGNI）:
  - 不跳过冻结层的 activation recompute。
  - 不改 checkpoint 格式；跨冻结配置的 optimizer state 迁移由用户自行处理。
  - 不为其他 provider（`qwen2_vl`、`intern_vl` 等）实装。
  - 不引入按名字通配的 freeze DSL。
  - 不支持"冻前 N 层"；仅支持"冻尾部 ≥N"。

## 总体架构

在 `aiak_training_llm/models/llavaov_1_5/llavaov_1_5_provider.py` 中，模型构造完成、现有模块级 `model.freeze(...)` 调用之后，插入一次"层粒度二次冻结"。新增逻辑封装为模块内部函数 `_apply_language_model_tail_freeze(model, N)`，不新增文件。

相关现有代码:

- `llavaov_1_5_provider.py:118-124` — `--trainable-modules` 到 `model.freeze(...)` 的桥接。
- `llavaov_1_5_model.py:245-270` — `freeze(freeze_language_model, freeze_vision_model, freeze_adapter)`。
- `aiak_training_llm/models/qwen/qwen_model.py:143` — `QwenModel.decoder = TransformerBlock(...)`；`decoder.layers[i]` 为各 transformer 层，`decoder.final_layernorm` 为顶部 norm。
- `llavaov_1_5_config.py:69` — `llava-ov-1.5-4b`: `num_layers=36`。

不改训练主循环、optimizer、checkpoint、其他 provider。

## 接口

### 新增 CLI 参数

```
--freeze-language-model-layers-ge <int>
```

- 类型: `int`；默认 `None`（不传 → 不生效，行为与当前完全一致）。
- 语义: 冻结 `language_model.decoder.layers[i]`（全局索引 `i >= N`）全部参数，同时冻结 `decoder.final_layernorm` 与 `language_model.output_layer`。`share_embeddings_and_output_weights=True`（`llava-ov-1.5-4b` 默认）时，共享权重使 embedding 也随 `output_layer` 冻结。
- 取值范围: `0 <= N <= num_layers`。非法值立即 raise。
  - `N == 0`: 合法但冗余（等价于 `--trainable-modules` 去掉 `language_model`）；允许通过。
  - `N == num_layers`: 合法；只冻 `final_layernorm + output_layer`，不冻任何 layer。
- 注册位置: `aiak_training_llm/train/arguments.py` 中 `--trainable-modules` 附近的参数组。description 中注明"当前仅 `llava-ov-1.5-*` 生效；其他模型传入将被忽略并打印 warning"。

### 与 `--trainable-modules` 的关系

- 正交、后生效。先按 `--trainable-modules` 做模块级冻结，再应用层级冻结。
- 若 `language_model` 不在 `--trainable-modules`（LLM 整个已冻），层级冻结是 no-op，同时日志阶段会因"i<N 的层无可训参数"触发断言并 raise，错误信息提示"`--freeze-language-model-layers-ge` requires `language_model` to be trainable"。

### 适用模型

- 仅 `llavaov_1_5_provider.py` 响应该参数。
- 在 `arguments.py` 的全局校验里做一次检测: 若当前 `--model-name` 不属于 `VisionLanguageModelFamilies.LLAVA_OV_1_5`，则 warning 并忽略参数，不终止。

## 执行流程

在 `llavaov_1_5_provider.py::rice_vl_model_provider` 末尾、`return model` 之前:

```python
if args.trainable_modules != ['all']:
    model.freeze(...)  # 现有逻辑，不变

if getattr(args, 'freeze_language_model_layers_ge', None) is not None:
    _apply_language_model_tail_freeze(model, N=args.freeze_language_model_layers_ge)

return model
```

### `_apply_language_model_tail_freeze(model, N)` 行为

1. 若 `model.language_model is None`（该 rank 不持有 decoder，例如 encoder-only pipeline stage）: `return`。
2. `decoder = model.language_model.decoder`。
3. 计算本 stage 在全局 layer 空间中的起始偏移:
   - 优先使用 Megatron 的 `get_transformer_layer_offset(config)`；
   - fallback: 用 `config.num_layers` 总数 + `mpu.get_pipeline_model_parallel_rank()` + `mpu.get_pipeline_model_parallel_world_size()` 做均匀切分估算；
   - 本次训练 `PP=1`，偏移恒为 0；fallback 也给出正确结果。
4. 对本地 `decoder.layers` 列表遍历；对全局索引 `global_i = offset + local_i >= N` 的层，`for p in layer.parameters(): p.requires_grad = False`。
5. 若本 stage 是 last pipeline stage（持有 `final_layernorm` / `output_layer`）:
   - `final_layernorm`: 存在则逐参数置 `requires_grad=False`。
   - `output_layer`: 存在则逐参数置 `requires_grad=False`。共享权重下 embedding 的 `word_embeddings.weight` 是同一 tensor，随之冻结。
6. 参数越界: `N < 0 or N > args.num_layers` → `raise ValueError`。

### 启动日志（rank 0）

按段打印 "freeze-plan" 表，每行格式:

```
[freeze-plan] <scope>    trainable=<trainable_count>/<total_count>[    <-- frozen]
```

打印范围:

- `language_model.embedding`（单独一行）
- `language_model.decoder.layers[i]`（每层一行，只打印本 stage 持有的层）
- `language_model.decoder.final_layernorm`
- `language_model.output_layer`
- `vision_model.*`（聚合一行）
- `adapter.*`（聚合一行）
- 末尾一行总计: `[freeze-plan] TOTAL trainable params = T / M  (XX.XX%)`

PP > 1 时每个 rank 各自打印本地持有部分，不做全局聚合（避免引入集合通信复杂度）。

### 启动断言

在 rank 0 打印完 freeze-plan 后、训练开始前执行:

1. 对每个本 stage 持有的、全局索引 `i >= N` 的层，检查所有参数 `requires_grad=False`。违者列出前 3 个违例参数名并 raise。
2. 对每个本 stage 持有的、全局索引 `i < N` 的层，检查至少存在一个 `requires_grad=True` 的参数；统计本地结果 `has_trainable_shallow`。
3. 对 `final_layernorm`、`output_layer`（若本 stage 持有）检查 `requires_grad=False`。
4. PP 下对 `has_trainable_shallow` 做 `all_reduce(op=MAX)`（所有持有 decoder 的 rank 参与；本次 `PP=1` 退化为本地判定）。若最终全局 `has_trainable_shallow == 0`，rank 0 raise，并在 raise 前确保所有 rank 已参与集合通信避免死锁。
5. 失败 → `RuntimeError`，进程退出。

## 错误处理与边界

- 参数非整数 / 负数: `argparse` 原生报错。
- `N > num_layers`: 在 `arguments.py` 全局校验阶段 `raise ValueError`。
- 模型不属于 `LLAVA_OV_1_5` 族: warning、忽略参数。
- `decoder.layers` 为空（异常情形）: `RuntimeError` 并打印 `num_layers`、PP rank。
- `--trainable-modules` 已冻掉 `language_model` 再传本参数: 断言阶段 raise，错误文案明确。
- Checkpoint 加载: 参数 tensor 原样加载，`requires_grad` 与 ckpt 无关，完全兼容。
- Checkpoint 保存: DistributedOptimizer 仅保存 `requires_grad=True` 参数的 optimizer state。同一冻结配置下 resume 无问题；跨配置 resume 需要用户自行处理 optimizer state（本次不提供迁移工具）。
- `bf16` + grad accumulation: Megatron 的 grad buffer 仅 reduce `requires_grad=True` 参数，无需改动。
- `--recompute-*`: 冻结层仍会重算 activation；反传因 `requires_grad=False` 而不写梯度，多余开销为前向重算。本次不优化（YAGNI）。

## 验证策略

### 层次 1: 启动即验证（每次训练默认执行）

即上面的 freeze-plan 打印 + 断言。失败即退出，不会进入训练循环。

### 层次 2: 开发期 smoke（手动一次）

复制 `stage_2_instruct_llava_ov_4b.sh` 为一个本地调试脚本（不进 commit），把 `--train-iters` 改为 1，加 `--freeze-language-model-layers-ge 24`，跑到模型构建 + freeze-plan 打印即可人工肉眼确认:

- `layers[0..23]`: 可训 = 全部；
- `layers[24..35]`: 可训 = 0；
- `final_layernorm`、`output_layer`: 可训 = 0；
- `vision_model`、`adapter`: 可训 = 全部；
- TOTAL 百分比大致符合预期。

### 层次 3: 训练期梯度兜底（默认不做）

可选: 第一步 training step 结束后，rank 0 检查被冻参数 `.grad is None or .grad.abs().sum() == 0`。gated by `FREEZE_CHECK_GRADS=1` 环境变量；默认关闭。实现优先级: 若层次 1 的静态断言已可信，不实装；留作 future work。

### 回归验证

不传 `--freeze-language-model-layers-ge` 时:

- `args` 中新字段为 `None`；
- 新代码全部绕过；
- 启动日志、前 N step loss、checkpoint 格式与当前 `main` 分支 bit-identical。

开发期手动对比一次即可，不入 repo。

## 训练脚本改动

`examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh` 的 `TRAINING_ARGS` 末尾追加一行:

```
    --freeze-language-model-layers-ge 24
```

如需消融其他切点，直接改数字即可。

## 改动清单

- `aiak_training_llm/train/arguments.py`
  - 新增参数 `--freeze-language-model-layers-ge`（`int`，默认 `None`）。
  - 全局校验: 取值范围 + 模型族 warning。
- `aiak_training_llm/models/llavaov_1_5/llavaov_1_5_provider.py`
  - 新增内部函数 `_apply_language_model_tail_freeze(model, N)`。
  - 新增启动日志 `_print_freeze_plan(model)` 与断言 `_assert_freeze_plan(model, N)`。
  - 在 `rice_vl_model_provider` 的 `model.freeze(...)` 之后调用上面三者。
- `examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh`
  - 追加 `--freeze-language-model-layers-ge 24`。

## 非目标 / 后续工作

- 将该参数推广到其他 VLM provider。
- 为冻结层跳过 activation recompute。
- 提供跨冻结配置的 optimizer state 迁移工具。
- 支持"冻前 N 层"或按参数名正则的 freeze DSL。
