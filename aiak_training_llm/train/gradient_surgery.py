"""模态感知的异质梯度手术 (Heterogeneous Gradient Surgery, v2 binary policy).

启动时一次性读入 MLP 路由掩码 (top-60% 互斥四集合)，按二值规则给 language_model
的每个参数预先计算缩放 tensor；每个训练 step 在 forward-backward 完成后、
optimizer.step() 之前直接作用于 param.main_grad。参数不改 requires_grad，
模型结构/前向不变。

策略 v2 (二值化, 与早先 v1 软系数策略不同):
  MLP (linear_fc1 / linear_fc2)：按 down_proj mask 分类
      text_only   -> 0.0 (冻; 文本核心保护)
      shared      -> 0.0 (冻; 让给文本, 避免污染通用语义层)
                     若启用 --gradient-surgery-shared-anchor-lambda:
                     shared -> 1.0 (全量更新) + 锚定正则 λ*(w-w0) 拉回原始权重
                     若启用 --gradient-surgery-shared-abs-l2-lambda:
                     shared -> 1.0 + 绝对 L2 正则 (消融: FC1/FC2 统一)
                     若启用 --gradient-surgery-shared-all-cosine-lambda:
                     shared -> 1.0 + cosine distance 正则 (消融: FC1/FC2 统一)
      vision_only -> 1.0 (全量更新)
      idle        -> 1.0 (全量更新; 让给多模态扩展容量)
      未覆盖      -> 0.0 (按 idle-of-protection 处理 = 冻)

  Attention (Megatron mcore 命名):
      linear_qkv 行级混合: Q rows -> 0.0 (冻), K rows -> 1.0, V rows -> 1.0
      linear_proj (= O proj) -> 0.0 (冻; 输出回写残差流的方式保护)
      → Q/O 冻保护文本侧查询与残差写回；K/V 全量让视觉侧"被看到"和"传递"

  LayerNorm (与 Q 一致, 全冻):
      pre-attention LN, pre-MLP LN, q_norm, k_norm, final_layernorm 全 -> 0.0

  language_model.embedding.* -> 0.0 (冻)
  language_model.output_layer.* -> 0.0 (冻)
  vision_model.* / adapter.* -> 不入 scale_table, 全量更新
"""
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from aiak_training_llm.utils import print_rank_0


# ---- 硬编码系数 (v2 二值策略) ----
TEXT_SCALE = 0.0
SHARED_SCALE = 0.0       # v2: 让给纯文本, 冻 (v1 是 0.1)
VISION_SCALE = 1.0
IDLE_SCALE = 1.0         # v2: 让给多模态, 全量 (v1 是 0.0)
L2SP_SCALE = 1.0        # L2-SP: shared 全量更新 + text-score 加权锚定正则

ATTN_Q_SCALE = 0.0       # 冻
ATTN_K_SCALE = 1.0       # 全量
ATTN_V_SCALE = 1.0       # 全量
ATTN_O_SCALE = 0.0       # 冻

LN_SCALE = 0.0           # v2: 全冻 (v1 是 0.1)
EMB_SCALE = 0.0
OUTPUT_SCALE = 0.0

SHARED_ROUTING_EPS = 1e-8

MASK_TOP_KEYS = ("text_only", "vision_only", "shared", "idle")
MASK_KEY_FMT = "model.language_model.layers.{idx}.mlp.down_proj"


# ---- 参数分类正则 ----
_RE_LAYER_IDX = re.compile(r"language_model\.decoder\.layers\.(\d+)\.")
_RE_MLP_FC1 = re.compile(r"language_model\.decoder\.layers\.(\d+)\.mlp\.linear_fc1\.(weight|bias)$")
_RE_MLP_FC2_WEIGHT = re.compile(r"language_model\.decoder\.layers\.(\d+)\.mlp\.linear_fc2\.weight$")
_RE_MLP_FC2_BIAS = re.compile(r"language_model\.decoder\.layers\.(\d+)\.mlp\.linear_fc2\.bias$")
_RE_ATTN_QKV = re.compile(r"language_model\.decoder\.layers\.\d+\.self_attention\.linear_qkv\.(weight|bias)$")
_RE_ATTN_O = re.compile(r"language_model\.decoder\.layers\.\d+\.self_attention\.linear_proj\.(weight|bias)$")
_RE_LN_IN_LAYER = re.compile(r"language_model\.decoder\.layers\.\d+\..*(layernorm|layer_norm)")
_RE_FINAL_LN = re.compile(r"language_model\.decoder\.final_layernorm\.")
_RE_EMBEDDING = re.compile(r"language_model\.embedding\.")
_RE_OUTPUT_LAYER = re.compile(r"language_model\.output_layer\.")
_RE_LANG = re.compile(r"language_model\.")
_RE_VISION = re.compile(r"vision_model\.")
_RE_ADAPTER = re.compile(r"(^|\.)adapter\.")


# ---- 数据结构 ----
@dataclass
class PlanRow:
    name: str
    role: str
    scale_repr: str


@dataclass
class AuditTarget:
    """首步抽检目标."""
    param_name: str
    param: torch.nn.Parameter
    row_indices: List[int]
    expected_zero: bool
    axis: int  # 0 for row, 1 for col


@dataclass
class _ClassifyResult:
    role: str
    # 可能字段：
    # mlp_masked / attn_qkv_partial / attn_o_frozen
    # ln_soft / frozen_emb / frozen_out
    # vision_full / adapter_full / frozen_by_other / unhandled
    layer_idx: Optional[int] = None
    is_fc1: bool = False
    is_fc2: bool = False
    fc2_is_bias: bool = False
    scalar: Optional[float] = None


class GradientSurgeryManager:
    """v2 二值梯度手术执行器."""

    def __init__(
        self,
        scale_table: Dict[int, torch.Tensor],
        plan: List[PlanRow],
        audit_targets: List[AuditTarget],
        mask_path: str,
        layers_covered: Tuple[int, int],
        shared_anchor_lambda: Optional[float] = None,
        shared_anchor_table: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        shared_l2sp_lambda: Optional[float] = None,
        shared_l2sp_table: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        shared_rel_l2_lambda: Optional[float] = None,
        shared_cosine_lambda: Optional[float] = None,
        shared_mixed_table: Optional[Dict[int, Tuple]] = None,
        shared_abs_l2_lambda: Optional[float] = None,
        shared_abs_l2_table: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        shared_all_cosine_lambda: Optional[float] = None,
        shared_all_cosine_table: Optional[Dict[int, Tuple]] = None,
        shared_swapped_cosine_lambda: Optional[float] = None,
        shared_swapped_rel_l2_lambda: Optional[float] = None,
        shared_swapped_table: Optional[Dict[int, Tuple]] = None,
        shared_tv_ratio_lambda: Optional[float] = None,
        shared_tv_ratio_table: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ):
        self._scale_table = scale_table
        self._plan = plan
        self._audit_targets = audit_targets
        self._mask_path = mask_path
        self._layers_covered = layers_covered
        self._shared_anchor_lambda = shared_anchor_lambda
        # {id(param): (original_weight_snapshot, shared_mask_for_grad)}
        self._shared_anchor_table = shared_anchor_table or {}
        self._shared_l2sp_lambda = shared_l2sp_lambda
        # {id(param): (w0_slice, shared_mask, text_scores_slice)}
        self._shared_l2sp_table = shared_l2sp_table or {}
        self._shared_rel_l2_lambda = shared_rel_l2_lambda
        self._shared_cosine_lambda = shared_cosine_lambda
        # fc1: {id(param): (w0_slice, mask_2i, w0_row_norm_sq)}
        # fc2: {id(param): (w0_col_unit, shared_mask, None)}
        self._shared_mixed_table = shared_mixed_table or {}
        # 消融: absolute L2 on both FC1/FC2
        self._shared_abs_l2_lambda = shared_abs_l2_lambda
        # {id(param): (w0_slice, mask)} — same structure as anchor
        self._shared_abs_l2_table = shared_abs_l2_table or {}
        # 消融: cosine distance on both FC1/FC2
        self._shared_all_cosine_lambda = shared_all_cosine_lambda
        # {id(param): (w0_unit, mask, "fc1"/"fc2")}
        self._shared_all_cosine_table = shared_all_cosine_table or {}
        # 消融: swapped mixed — cosine on fc1, relative-L2 on fc2
        self._shared_swapped_cosine_lambda = shared_swapped_cosine_lambda
        self._shared_swapped_rel_l2_lambda = shared_swapped_rel_l2_lambda
        # fc1: {id(param): (w0_row_unit, mask_2i, "fc1")}
        # fc2: {id(param): (w0_col_slice, shared_mask, w0_col_norm_sq)}
        self._shared_swapped_table = shared_swapped_table or {}
        # TV-Ratio L2: λ_i = T_rank/(T_rank + V_rank + ε), grad += λ * λ_i * (w - w0)
        self._shared_tv_ratio_lambda = shared_tv_ratio_lambda
        # {id(param): (w0_slice, shared_mask, ratio_scores_slice)}
        self._shared_tv_ratio_table = shared_tv_ratio_table or {}
        self._stored_l2sp_loss: Optional[float] = None
        self._l2sp_anchors_initialized = False
        self._anchor_anchors_initialized = False
        self._mixed_anchors_initialized = False
        self._abs_l2_anchors_initialized = False
        self._all_cosine_anchors_initialized = False
        self._swapped_anchors_initialized = False
        self._tv_ratio_anchors_initialized = False

    @classmethod
    def build_from_args(cls, model, args) -> "GradientSurgeryManager":
        mask_path = args.gradient_surgery_mask
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"gradient surgery mask not found: {mask_path}")

        tp = getattr(args, "tensor_model_parallel_size", 1)
        if tp and tp > 1:
            raise NotImplementedError(
                f"gradient surgery currently only supports TP=1, got TP={tp}"
            )

        shared_anchor_lambda = getattr(args, "gradient_surgery_shared_anchor_lambda", None)
        shared_anchor_enabled = shared_anchor_lambda is not None and shared_anchor_lambda > 0

        shared_routing_dir = getattr(args, "gradient_surgery_shared_routing_scores_dir", None)
        shared_routing_enabled = shared_routing_dir is not None

        shared_full_update = getattr(args, "gradient_surgery_shared_full_update", False)

        shared_l2sp_lambda = getattr(args, "gradient_surgery_shared_l2sp_lambda", None)
        shared_l2sp_enabled = shared_l2sp_lambda is not None and shared_l2sp_lambda > 0
        if shared_l2sp_enabled:
            shared_l2sp_scores_dir = getattr(args, "gradient_surgery_shared_l2sp_scores_dir", None)
            if not shared_l2sp_scores_dir:
                raise ValueError(
                    "--gradient-surgery-shared-l2sp-lambda requires "
                    "--gradient-surgery-shared-l2sp-scores-dir"
                )

        shared_rel_l2_lambda = getattr(args, "gradient_surgery_shared_rel_l2_lambda", None)
        shared_cosine_lambda = getattr(args, "gradient_surgery_shared_cosine_lambda", None)
        shared_mixed_enabled = (
            shared_rel_l2_lambda is not None and shared_rel_l2_lambda > 0
            and shared_cosine_lambda is not None and shared_cosine_lambda > 0
        )

        shared_abs_l2_lambda = getattr(args, "gradient_surgery_shared_abs_l2_lambda", None)
        shared_abs_l2_enabled = shared_abs_l2_lambda is not None and shared_abs_l2_lambda > 0

        shared_all_cosine_lambda = getattr(args, "gradient_surgery_shared_all_cosine_lambda", None)
        shared_all_cosine_enabled = shared_all_cosine_lambda is not None and shared_all_cosine_lambda > 0

        shared_swapped_cosine_lambda = getattr(args, "gradient_surgery_shared_swapped_cosine_lambda", None)
        shared_swapped_rel_l2_lambda = getattr(args, "gradient_surgery_shared_swapped_rel_l2_lambda", None)
        shared_swapped_enabled = (
            shared_swapped_cosine_lambda is not None and shared_swapped_cosine_lambda > 0
            and shared_swapped_rel_l2_lambda is not None and shared_swapped_rel_l2_lambda > 0
        )

        shared_tv_ratio_lambda = getattr(args, "gradient_surgery_shared_tv_ratio_lambda", None)
        shared_tv_ratio_enabled = shared_tv_ratio_lambda is not None and shared_tv_ratio_lambda > 0
        if shared_tv_ratio_enabled:
            shared_tv_ratio_scores_dir = getattr(args, "gradient_surgery_shared_tv_ratio_scores_dir", None)
            if not shared_tv_ratio_scores_dir:
                raise ValueError(
                    "--gradient-surgery-shared-tv-ratio-lambda requires "
                    "--gradient-surgery-shared-tv-ratio-scores-dir"
                )

        freeze_ablation = getattr(args, "gradient_surgery_freeze_ablation", None)
        freeze_ablation_enabled = freeze_ablation is not None

        # 互斥性检查
        exclusive_flags = [
            shared_anchor_enabled, shared_routing_enabled, shared_full_update,
            shared_l2sp_enabled, shared_mixed_enabled,
            shared_abs_l2_enabled, shared_all_cosine_enabled,
            shared_swapped_enabled,
            shared_tv_ratio_enabled,
            freeze_ablation_enabled,
        ]
        if sum(exclusive_flags) > 1:
            raise ValueError(
                "--gradient-surgery-shared-anchor-lambda, "
                "--gradient-surgery-shared-routing-scores-dir, "
                "--gradient-surgery-shared-full-update, "
                "--gradient-surgery-shared-l2sp-lambda, "
                "--gradient-surgery-shared-rel-l2-lambda/cosine-lambda, "
                "--gradient-surgery-shared-abs-l2-lambda, "
                "--gradient-surgery-shared-all-cosine-lambda, "
                "--gradient-surgery-shared-swapped-cosine-lambda/rel-l2-lambda, "
                "--gradient-surgery-shared-tv-ratio-lambda, and "
                "--gradient-surgery-freeze-ablation "
                "are mutually exclusive"
            )

        mask_dict = torch.load(mask_path, map_location="cpu", weights_only=False)
        cls._validate_mask_dict(
            mask_dict,
            expected_num_layers=args.num_layers,
            expected_intermediate_size=args.ffn_hidden_size,
        )

        # 计算 shared routing scales: M_i = (S_vision_i / (S_vision_i + α*S_text_i + ε))^r
        shared_routing_scales: Optional[Dict[int, torch.Tensor]] = None
        if shared_routing_enabled:
            alpha = getattr(args, "gradient_surgery_shared_routing_alpha", 2.0)
            power = getattr(args, "gradient_surgery_shared_routing_power", 3.0)
            shared_routing_scales = cls._compute_shared_routing_scales(
                shared_routing_dir, args.num_layers, alpha, power,
            )
            print_rank_0(
                f"[surgery-plan] shared-routing enabled: dir={shared_routing_dir}, "
                f"alpha={alpha}, power={power}"
            )
            for idx in range(min(3, args.num_layers)):
                s = shared_routing_scales[idx]
                print_rank_0(
                    f"[surgery-plan]   layer {idx} shared M_i: "
                    f"min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}"
                )
        elif shared_full_update:
            print_rank_0("[surgery-plan] shared-full-update enabled: shared channels scale=1.0 (no regularization)")
        elif shared_l2sp_enabled:
            # 加载文本重要性分数并做 percentile rank 归一化到 (0, 1]
            shared_l2sp_text_scores = cls._load_l2sp_text_scores(
                shared_l2sp_scores_dir, args.num_layers,
            )
            print_rank_0(
                f"[surgery-plan] shared-l2sp enabled: lambda={shared_l2sp_lambda}, "
                f"scores_dir={shared_l2sp_scores_dir}"
            )
            for idx in range(min(3, args.num_layers)):
                s = shared_l2sp_text_scores[idx]
                print_rank_0(
                    f"[surgery-plan]   layer {idx} shared S_text_i: "
                    f"min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}"
                )
        elif shared_mixed_enabled:
            print_rank_0(
                f"[surgery-plan] shared-mixed-reg enabled: "
                f"rel_l2_lambda={shared_rel_l2_lambda}, "
                f"cosine_lambda={shared_cosine_lambda}"
            )
        elif shared_abs_l2_enabled:
            print_rank_0(
                f"[surgery-plan] shared-abs-l2 (ablation) enabled: "
                f"lambda={shared_abs_l2_lambda}"
            )
        elif shared_all_cosine_enabled:
            print_rank_0(
                f"[surgery-plan] shared-all-cosine (ablation) enabled: "
                f"lambda={shared_all_cosine_lambda}"
            )
        elif freeze_ablation_enabled:
            print_rank_0(
                f"[surgery-plan] freeze-ablation enabled: "
                f"only '{freeze_ablation}' channels unfrozen (scale=1.0), "
                f"all other MLP channel types frozen (scale=0.0)"
            )
        elif shared_tv_ratio_enabled:
            shared_tv_ratio_scores = cls._load_tv_ratio_scores(
                shared_tv_ratio_scores_dir, args.num_layers,
            )
            print_rank_0(
                f"[surgery-plan] shared-tv-ratio enabled: lambda={shared_tv_ratio_lambda}, "
                f"scores_dir={shared_tv_ratio_scores_dir}"
            )
            for idx in range(min(3, args.num_layers)):
                s = shared_tv_ratio_scores[idx]
                print_rank_0(
                    f"[surgery-plan]   layer {idx} shared TV-ratio λ_i: "
                    f"min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}"
                )


        per_layer_mlp_scale = cls._precompute_per_layer_mlp_scales(
            mask_dict, args.num_layers,
            shared_anchor_enabled=shared_anchor_enabled,
            shared_routing_scales=shared_routing_scales,
            shared_full_update=shared_full_update,
            shared_l2sp_enabled=shared_l2sp_enabled,
            shared_mixed_enabled=shared_mixed_enabled,
            shared_abs_l2_enabled=shared_abs_l2_enabled,
            shared_all_cosine_enabled=shared_all_cosine_enabled,
            shared_swapped_enabled=shared_swapped_enabled,
            shared_tv_ratio_enabled=shared_tv_ratio_enabled,
            freeze_ablation=freeze_ablation,
        )

        # 预计算 shared mask 的 per-layer boolean (用于锚定正则 / L2SP / mixed / ablation)
        per_layer_shared_mask: Dict[int, torch.Tensor] = {}
        if shared_anchor_enabled or shared_l2sp_enabled or shared_mixed_enabled or shared_abs_l2_enabled or shared_all_cosine_enabled or shared_swapped_enabled or shared_tv_ratio_enabled:
            for idx in range(args.num_layers):
                key = MASK_KEY_FMT.format(idx=idx)
                per_layer_shared_mask[idx] = mask_dict["shared"][key].bool()

        scale_table: Dict[int, torch.Tensor] = {}
        plan: List[PlanRow] = []
        unhandled: List[str] = []
        lang_trainable_seen = False

        for chunk in _iter_unwrapped_chunks(model):
            for name, param in chunk.named_parameters():
                classify = cls._classify(name)

                if not param.requires_grad:
                    plan.append(PlanRow(
                        name=name, role="frozen-by-other",
                        scale_repr="requires_grad=False",
                    ))
                    continue

                if classify.role == "unhandled":
                    unhandled.append(name)
                    plan.append(PlanRow(name=name, role="unhandled", scale_repr="RAISE"))
                    continue

                # vision/adapter 全量更新：不入 scale_table
                if classify.role in ("vision_full", "adapter_full"):
                    plan.append(PlanRow(
                        name=name, role=classify.role, scale_repr="1.0 (no-op)"
                    ))
                    continue

                # K/V 全量更新：不入 scale_table（与 vision/adapter 同理）
                if classify.role == "attn_kv_full":
                    plan.append(PlanRow(
                        name=name, role=classify.role, scale_repr="1.0 (no-op, K/V full update)"
                    ))
                    continue

                if _RE_LANG.search(name):
                    lang_trainable_seen = True

                scale_tensor, repr_str = cls._build_scale_tensor(
                    classify, name, param, per_layer_mlp_scale, args,
                )
                scale_table[id(param)] = scale_tensor
                plan.append(PlanRow(
                    name=name, role=classify.role, scale_repr=repr_str,
                ))

        if unhandled:
            head = "\n  ".join(unhandled[:10])
            raise RuntimeError(
                f"gradient surgery: {len(unhandled)} unhandled param(s):\n  {head}"
            )

        if not lang_trainable_seen:
            raise RuntimeError(
                "gradient surgery requires language_model to be trainable; "
                "remove --gradient-surgery-mask or add language_model to --trainable-modules"
            )

        # 构建 shared anchor table: 保存原始权重快照 + shared mask
        shared_anchor_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if shared_anchor_enabled:
            for chunk in _iter_unwrapped_chunks(model):
                for name, param in chunk.named_parameters():
                    classify = cls._classify(name)
                    if classify.role != "mlp_masked":
                        continue
                    if classify.fc2_is_bias:
                        continue
                    layer_idx = classify.layer_idx
                    shared_mask = per_layer_shared_mask[layer_idx]
                    if classify.is_fc1:
                        # fc1 shape [2I, hidden], mask on rows
                        mask_2i = torch.cat([shared_mask, shared_mask], dim=0)
                        w_snapshot = param.data[mask_2i].clone()
                        shared_anchor_table[id(param)] = (w_snapshot, mask_2i)
                    elif classify.is_fc2:
                        # fc2 shape [hidden, I], mask on cols
                        w_snapshot = param.data[:, shared_mask].clone()
                        shared_anchor_table[id(param)] = (w_snapshot, shared_mask)
            print_rank_0(
                f"[surgery-plan] shared-anchor enabled: lambda={shared_anchor_lambda}, "
                f"params with anchor={len(shared_anchor_table)}"
            )

        # 构建 shared L2SP table: (w0_slice, shared_mask, text_scores_slice)
        shared_l2sp_table: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if shared_l2sp_enabled:
            for chunk in _iter_unwrapped_chunks(model):
                for name, param in chunk.named_parameters():
                    classify = cls._classify(name)
                    if classify.role != "mlp_masked":
                        continue
                    if classify.fc2_is_bias:
                        continue
                    layer_idx = classify.layer_idx
                    shared_mask = per_layer_shared_mask[layer_idx]
                    text_scores = shared_l2sp_text_scores[layer_idx].to(
                        device=param.device, dtype=torch.float32,
                    )
                    text_scores_shared = text_scores[shared_mask]
                    if classify.is_fc1:
                        # fc1 shape [2I, hidden], gate 和 up 部分各用相同的 S_text
                        mask_2i = torch.cat([shared_mask, shared_mask], dim=0)
                        scores_2i = torch.cat([text_scores_shared, text_scores_shared], dim=0)
                        w_snapshot = param.data[mask_2i].clone()
                        shared_l2sp_table[id(param)] = (w_snapshot, mask_2i, scores_2i)
                    elif classify.is_fc2:
                        # fc2 shape [hidden, I], mask on cols
                        w_snapshot = param.data[:, shared_mask].clone()
                        shared_l2sp_table[id(param)] = (w_snapshot, shared_mask, text_scores_shared)
            print_rank_0(
                f"[surgery-plan] shared-l2sp table built: lambda={shared_l2sp_lambda}, "
                f"params with L2SP={len(shared_l2sp_table)}"
            )

        # 构建 shared mixed table: relative-L2 (fc1) + cosine (fc2)
        # fc1: (w0_slice, mask_2i, w0_row_norm_sq)
        # fc2: (w0_col_unit, shared_mask, None)
        shared_mixed_table: Dict[int, Tuple] = {}
        if shared_mixed_enabled:
            for chunk in _iter_unwrapped_chunks(model):
                for name, param in chunk.named_parameters():
                    classify = cls._classify(name)
                    if classify.role != "mlp_masked":
                        continue
                    if classify.fc2_is_bias:
                        continue
                    layer_idx = classify.layer_idx
                    shared_mask = per_layer_shared_mask[layer_idx]
                    if classify.is_fc1:
                        mask_2i = torch.cat([shared_mask, shared_mask], dim=0)
                        w0 = param.data[mask_2i].clone()  # [n_shared*2, hidden]
                        w0_row_norm_sq = w0.pow(2).sum(dim=1).clamp(min=1e-12)  # [n_shared*2]
                        shared_mixed_table[id(param)] = (w0, mask_2i, w0_row_norm_sq)
                    elif classify.is_fc2:
                        w0 = param.data[:, shared_mask].clone()  # [hidden, n_shared]
                        col_norm = w0.norm(dim=0, keepdim=True).clamp(min=1e-8)
                        w0_col_unit = w0 / col_norm  # [hidden, n_shared]
                        shared_mixed_table[id(param)] = (w0_col_unit, shared_mask, None)
            print_rank_0(
                f"[surgery-plan] shared-mixed-reg table built: "
                f"rel_l2_lambda={shared_rel_l2_lambda}, "
                f"cosine_lambda={shared_cosine_lambda}, "
                f"params={len(shared_mixed_table)}"
            )

        # 消融: absolute L2 on both FC1/FC2 — 结构同 anchor
        shared_abs_l2_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if shared_abs_l2_enabled:
            for chunk in _iter_unwrapped_chunks(model):
                for name, param in chunk.named_parameters():
                    classify = cls._classify(name)
                    if classify.role != "mlp_masked":
                        continue
                    if classify.fc2_is_bias:
                        continue
                    layer_idx = classify.layer_idx
                    shared_mask = per_layer_shared_mask[layer_idx]
                    if classify.is_fc1:
                        mask_2i = torch.cat([shared_mask, shared_mask], dim=0)
                        w_snapshot = param.data[mask_2i].clone()
                        shared_abs_l2_table[id(param)] = (w_snapshot, mask_2i)
                    elif classify.is_fc2:
                        w_snapshot = param.data[:, shared_mask].clone()
                        shared_abs_l2_table[id(param)] = (w_snapshot, shared_mask)
            print_rank_0(
                f"[surgery-plan] shared-abs-l2 (ablation) table built: "
                f"lambda={shared_abs_l2_lambda}, params={len(shared_abs_l2_table)}"
            )

        # 消融: cosine distance on both FC1/FC2
        # fc1: (w0_row_unit, mask_2i, "fc1")
        # fc2: (w0_col_unit, shared_mask, "fc2")
        shared_all_cosine_table: Dict[int, Tuple] = {}
        if shared_all_cosine_enabled:
            for chunk in _iter_unwrapped_chunks(model):
                for name, param in chunk.named_parameters():
                    classify = cls._classify(name)
                    if classify.role != "mlp_masked":
                        continue
                    if classify.fc2_is_bias:
                        continue
                    layer_idx = classify.layer_idx
                    shared_mask = per_layer_shared_mask[layer_idx]
                    if classify.is_fc1:
                        mask_2i = torch.cat([shared_mask, shared_mask], dim=0)
                        w0 = param.data[mask_2i].clone()  # [n_shared*2, hidden]
                        row_norm = w0.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        w0_row_unit = w0 / row_norm
                        shared_all_cosine_table[id(param)] = (w0_row_unit, mask_2i, "fc1")
                    elif classify.is_fc2:
                        w0 = param.data[:, shared_mask].clone()  # [hidden, n_shared]
                        col_norm = w0.norm(dim=0, keepdim=True).clamp(min=1e-8)
                        w0_col_unit = w0 / col_norm
                        shared_all_cosine_table[id(param)] = (w0_col_unit, shared_mask, "fc2")
            print_rank_0(
                f"[surgery-plan] shared-all-cosine (ablation) table built: "
                f"lambda={shared_all_cosine_lambda}, params={len(shared_all_cosine_table)}"
            )

        # 消融: swapped mixed — cosine on fc1 rows, relative-L2 on fc2 cols
        # fc1: (w0_row_unit, mask_2i, "fc1")
        # fc2: (w0_col_slice, shared_mask, w0_col_norm_sq)
        shared_swapped_table: Dict[int, Tuple] = {}
        if shared_swapped_enabled:
            for chunk in _iter_unwrapped_chunks(model):
                for name, param in chunk.named_parameters():
                    classify = cls._classify(name)
                    if classify.role != "mlp_masked":
                        continue
                    if classify.fc2_is_bias:
                        continue
                    layer_idx = classify.layer_idx
                    shared_mask = per_layer_shared_mask[layer_idx]
                    if classify.is_fc1:
                        mask_2i = torch.cat([shared_mask, shared_mask], dim=0)
                        w0 = param.data[mask_2i].clone()
                        row_norm = w0.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        w0_row_unit = w0 / row_norm
                        shared_swapped_table[id(param)] = (w0_row_unit, mask_2i, "fc1")
                    elif classify.is_fc2:
                        w0 = param.data[:, shared_mask].clone()
                        w0_col_norm_sq = w0.pow(2).sum(dim=0).clamp(min=1e-12)
                        shared_swapped_table[id(param)] = (w0, shared_mask, w0_col_norm_sq)
            print_rank_0(
                f"[surgery-plan] shared-swapped-mixed (ablation) table built: "
                f"cosine_lambda(fc1)={shared_swapped_cosine_lambda}, "
                f"rel_l2_lambda(fc2)={shared_swapped_rel_l2_lambda}, "
                f"params={len(shared_swapped_table)}"
            )

        # TV-Ratio L2: λ_i = T_rank/(T_rank + V_rank + ε), 结构同 L2-SP
        shared_tv_ratio_table: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if shared_tv_ratio_enabled:
            for chunk in _iter_unwrapped_chunks(model):
                for name, param in chunk.named_parameters():
                    classify = cls._classify(name)
                    if classify.role != "mlp_masked":
                        continue
                    if classify.fc2_is_bias:
                        continue
                    layer_idx = classify.layer_idx
                    shared_mask = per_layer_shared_mask[layer_idx]
                    ratio_scores = shared_tv_ratio_scores[layer_idx].to(
                        device=param.device, dtype=torch.float32,
                    )
                    ratio_scores_shared = ratio_scores[shared_mask]
                    if classify.is_fc1:
                        mask_2i = torch.cat([shared_mask, shared_mask], dim=0)
                        scores_2i = torch.cat([ratio_scores_shared, ratio_scores_shared], dim=0)
                        w_snapshot = param.data[mask_2i].clone()
                        shared_tv_ratio_table[id(param)] = (w_snapshot, mask_2i, scores_2i)
                    elif classify.is_fc2:
                        w_snapshot = param.data[:, shared_mask].clone()
                        shared_tv_ratio_table[id(param)] = (w_snapshot, shared_mask, ratio_scores_shared)
            print_rank_0(
                f"[surgery-plan] shared-tv-ratio table built: lambda={shared_tv_ratio_lambda}, "
                f"params with TV-ratio={len(shared_tv_ratio_table)}"
            )

        audit_targets = cls._select_audit_targets(
            model, mask_dict, args.num_layers,
            shared_anchor_enabled=shared_anchor_enabled,
            shared_routing_enabled=shared_routing_enabled,
            shared_full_update=shared_full_update,
            shared_l2sp_enabled=shared_l2sp_enabled,
            shared_mixed_enabled=shared_mixed_enabled,
            shared_abs_l2_enabled=shared_abs_l2_enabled,
            shared_all_cosine_enabled=shared_all_cosine_enabled,
            shared_swapped_enabled=shared_swapped_enabled,
            shared_tv_ratio_enabled=shared_tv_ratio_enabled,
            freeze_ablation=freeze_ablation,
        )

        return cls(
            scale_table=scale_table,
            plan=plan,
            audit_targets=audit_targets,
            mask_path=mask_path,
            layers_covered=(0, args.num_layers - 1),
            shared_anchor_lambda=shared_anchor_lambda,
            shared_anchor_table=shared_anchor_table,
            shared_l2sp_lambda=shared_l2sp_lambda,
            shared_l2sp_table=shared_l2sp_table,
            shared_rel_l2_lambda=shared_rel_l2_lambda,
            shared_cosine_lambda=shared_cosine_lambda,
            shared_mixed_table=shared_mixed_table,
            shared_abs_l2_lambda=shared_abs_l2_lambda,
            shared_abs_l2_table=shared_abs_l2_table,
            shared_all_cosine_lambda=shared_all_cosine_lambda,
            shared_all_cosine_table=shared_all_cosine_table,
            shared_swapped_cosine_lambda=shared_swapped_cosine_lambda,
            shared_swapped_rel_l2_lambda=shared_swapped_rel_l2_lambda,
            shared_swapped_table=shared_swapped_table,
            shared_tv_ratio_lambda=shared_tv_ratio_lambda,
            shared_tv_ratio_table=shared_tv_ratio_table,
        )

    # ---- 运行期 ----
    def _refresh_all_anchors(self, model) -> None:
        """在 checkpoint 加载后重新保存锚点权重（首次 apply 时调用）."""
        # 刷新 shared_anchor_table
        if self._shared_anchor_table and not self._anchor_anchors_initialized:
            for chunk in _iter_unwrapped_chunks(model):
                for param in chunk.parameters():
                    entry = self._shared_anchor_table.get(id(param))
                    if entry is None:
                        continue
                    _, mask = entry
                    if param.data.ndim == 2 and mask.shape[0] == param.data.shape[0]:
                        w0 = param.data[mask].clone()
                        self._shared_anchor_table[id(param)] = (w0, mask)
                    elif param.data.ndim == 2 and mask.shape[0] == param.data.shape[1]:
                        w0 = param.data[:, mask].clone()
                        self._shared_anchor_table[id(param)] = (w0, mask)
            self._anchor_anchors_initialized = True
            print_rank_0("[surgery-plan] Anchor anchors refreshed from checkpoint weights")

        # 刷新 shared_l2sp_table
        if self._shared_l2sp_table and not self._l2sp_anchors_initialized:
            for chunk in _iter_unwrapped_chunks(model):
                for param in chunk.parameters():
                    entry = self._shared_l2sp_table.get(id(param))
                    if entry is None:
                        continue
                    _, mask, scores = entry
                    if param.data.ndim == 2 and mask.shape[0] == param.data.shape[0]:
                        w0 = param.data[mask].clone()
                        self._shared_l2sp_table[id(param)] = (w0, mask, scores)
                    elif param.data.ndim == 2 and mask.shape[0] == param.data.shape[1]:
                        w0 = param.data[:, mask].clone()
                        self._shared_l2sp_table[id(param)] = (w0, mask, scores)
            self._l2sp_anchors_initialized = True
            print_rank_0("[surgery-plan] L2SP anchors refreshed from checkpoint weights")

        # 刷新 shared_mixed_table
        if self._shared_mixed_table and not self._mixed_anchors_initialized:
            for chunk in _iter_unwrapped_chunks(model):
                for param in chunk.parameters():
                    entry = self._shared_mixed_table.get(id(param))
                    if entry is None:
                        continue
                    _, mask, _ = entry
                    if param.data.ndim == 2 and mask.shape[0] == param.data.shape[0]:
                        # fc1: 重新计算 w0 和 row_norm_sq
                        w0 = param.data[mask].clone()
                        w0_row_norm_sq = w0.pow(2).sum(dim=1).clamp(min=1e-12)
                        self._shared_mixed_table[id(param)] = (w0, mask, w0_row_norm_sq)
                    elif param.data.ndim == 2 and mask.shape[0] == param.data.shape[1]:
                        # fc2: 重新计算 w0_col_unit
                        w0 = param.data[:, mask].clone()
                        col_norm = w0.norm(dim=0, keepdim=True).clamp(min=1e-8)
                        w0_col_unit = w0 / col_norm
                        self._shared_mixed_table[id(param)] = (w0_col_unit, mask, None)
            self._mixed_anchors_initialized = True
            print_rank_0("[surgery-plan] Mixed-reg anchors refreshed from checkpoint weights")

        # 刷新 shared_abs_l2_table (结构同 anchor)
        if self._shared_abs_l2_table and not self._abs_l2_anchors_initialized:
            for chunk in _iter_unwrapped_chunks(model):
                for param in chunk.parameters():
                    entry = self._shared_abs_l2_table.get(id(param))
                    if entry is None:
                        continue
                    _, mask = entry
                    if param.data.ndim == 2 and mask.shape[0] == param.data.shape[0]:
                        w0 = param.data[mask].clone()
                        self._shared_abs_l2_table[id(param)] = (w0, mask)
                    elif param.data.ndim == 2 and mask.shape[0] == param.data.shape[1]:
                        w0 = param.data[:, mask].clone()
                        self._shared_abs_l2_table[id(param)] = (w0, mask)
            self._abs_l2_anchors_initialized = True
            print_rank_0("[surgery-plan] Abs-L2 anchors refreshed from checkpoint weights")

        # 刷新 shared_all_cosine_table
        if self._shared_all_cosine_table and not self._all_cosine_anchors_initialized:
            for chunk in _iter_unwrapped_chunks(model):
                for param in chunk.parameters():
                    entry = self._shared_all_cosine_table.get(id(param))
                    if entry is None:
                        continue
                    _, mask, proj_type = entry
                    if proj_type == "fc1":
                        w0 = param.data[mask].clone()
                        row_norm = w0.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        w0_row_unit = w0 / row_norm
                        self._shared_all_cosine_table[id(param)] = (w0_row_unit, mask, "fc1")
                    elif proj_type == "fc2":
                        w0 = param.data[:, mask].clone()
                        col_norm = w0.norm(dim=0, keepdim=True).clamp(min=1e-8)
                        w0_col_unit = w0 / col_norm
                        self._shared_all_cosine_table[id(param)] = (w0_col_unit, mask, "fc2")
            self._all_cosine_anchors_initialized = True
            print_rank_0("[surgery-plan] All-Cosine anchors refreshed from checkpoint weights")

        # 刷新 shared_swapped_table
        if self._shared_swapped_table and not self._swapped_anchors_initialized:
            for chunk in _iter_unwrapped_chunks(model):
                for param in chunk.parameters():
                    entry = self._shared_swapped_table.get(id(param))
                    if entry is None:
                        continue
                    _, mask, aux = entry
                    if aux == "fc1":
                        w0 = param.data[mask].clone()
                        row_norm = w0.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        w0_row_unit = w0 / row_norm
                        self._shared_swapped_table[id(param)] = (w0_row_unit, mask, "fc1")
                    else:
                        w0 = param.data[:, mask].clone()
                        w0_col_norm_sq = w0.pow(2).sum(dim=0).clamp(min=1e-12)
                        self._shared_swapped_table[id(param)] = (w0, mask, w0_col_norm_sq)
            self._swapped_anchors_initialized = True
            print_rank_0("[surgery-plan] Swapped-mixed anchors refreshed from checkpoint weights")

        # 刷新 shared_tv_ratio_table (结构同 L2SP)
        if self._shared_tv_ratio_table and not self._tv_ratio_anchors_initialized:
            for chunk in _iter_unwrapped_chunks(model):
                for param in chunk.parameters():
                    entry = self._shared_tv_ratio_table.get(id(param))
                    if entry is None:
                        continue
                    _, mask, scores = entry
                    if param.data.ndim == 2 and mask.shape[0] == param.data.shape[0]:
                        w0 = param.data[mask].clone()
                        self._shared_tv_ratio_table[id(param)] = (w0, mask, scores)
                    elif param.data.ndim == 2 and mask.shape[0] == param.data.shape[1]:
                        w0 = param.data[:, mask].clone()
                        self._shared_tv_ratio_table[id(param)] = (w0, mask, scores)
            self._tv_ratio_anchors_initialized = True
            print_rank_0("[surgery-plan] TV-Ratio anchors refreshed from checkpoint weights")

    def apply(self, model) -> None:
        if not self._scale_table:
            return
        # 首次调用时从 checkpoint 权重刷新锚点（build_from_args 在 checkpoint 加载前执行）
        if (self._shared_anchor_table and not self._anchor_anchors_initialized) or \
           (self._shared_l2sp_table and not self._l2sp_anchors_initialized) or \
           (self._shared_mixed_table and not self._mixed_anchors_initialized) or \
           (self._shared_abs_l2_table and not self._abs_l2_anchors_initialized) or \
           (self._shared_all_cosine_table and not self._all_cosine_anchors_initialized) or \
           (self._shared_swapped_table and not self._swapped_anchors_initialized) or \
           (self._shared_tv_ratio_table and not self._tv_ratio_anchors_initialized):
            self._refresh_all_anchors(model)
        table = self._scale_table
        anchor_table = self._shared_anchor_table
        lam = self._shared_anchor_lambda
        l2sp_table = self._shared_l2sp_table
        l2sp_lam = self._shared_l2sp_lambda
        mixed_table = self._shared_mixed_table
        rel_l2_lam = self._shared_rel_l2_lambda
        cosine_lam = self._shared_cosine_lambda
        abs_l2_table = self._shared_abs_l2_table
        abs_l2_lam = self._shared_abs_l2_lambda
        all_cosine_table = self._shared_all_cosine_table
        all_cosine_lam = self._shared_all_cosine_lambda
        swapped_table = self._shared_swapped_table
        swapped_cos_lam = self._shared_swapped_cosine_lambda
        swapped_rl2_lam = self._shared_swapped_rel_l2_lambda
        tv_ratio_table = self._shared_tv_ratio_table
        tv_ratio_lam = self._shared_tv_ratio_lambda

        l2sp_loss_sum = 0.0
        l2sp_loss_count = 0
        mixed_loss_sum = 0.0
        mixed_loss_count = 0
        ablation_loss_sum = 0.0
        ablation_loss_count = 0

        for chunk in _iter_unwrapped_chunks(model):
            for param in chunk.parameters():
                scale = table.get(id(param))
                if scale is None:
                    continue
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    continue
                _mul_with_broadcast(grad, scale, param_shape=tuple(param.shape))

                # 锚定正则: grad[shared] += λ * (w[shared] - w0[shared])
                if lam and id(param) in anchor_table:
                    w0_slice, mask = anchor_table[id(param)]
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                        anchor_table[id(param)] = (w0_slice, mask)
                    if w0_slice.device != param.device:
                        w0_slice = w0_slice.to(param.device)
                        anchor_table[id(param)] = (w0_slice, mask)
                    if grad.ndim == 2 and mask.shape[0] == grad.shape[0]:
                        # fc1: mask on rows
                        delta = param.data[mask] - w0_slice
                        grad[mask] += lam * delta.to(grad.dtype)
                    elif grad.ndim == 2 and mask.shape[0] == grad.shape[1]:
                        # fc2: mask on cols
                        delta = param.data[:, mask] - w0_slice
                        grad[:, mask] += lam * delta.to(grad.dtype)

                # L2-SP 正则: grad[shared] += λ * S_text_i * (w[shared] - w0[shared])
                if l2sp_lam and id(param) in l2sp_table:
                    w0_slice, mask, scores = l2sp_table[id(param)]
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                        scores = scores.to(param.device)
                        l2sp_table[id(param)] = (w0_slice, mask, scores)
                    if w0_slice.device != param.device:
                        w0_slice = w0_slice.to(param.device)
                        l2sp_table[id(param)] = (w0_slice, mask, scores)
                    if grad.ndim == 2 and mask.shape[0] == grad.shape[0]:
                        delta = param.data[mask] - w0_slice
                        weighted = delta * scores.unsqueeze(1)
                        grad[mask] += l2sp_lam * weighted.to(grad.dtype)
                        n_s = scores.shape[0] // 2
                        ch_delta_sq = delta.view(2, n_s, -1).pow(2).sum(dim=(0, 2))
                        ch_scores = scores.view(2, n_s)[0]
                        l2sp_loss_sum += (ch_scores * ch_delta_sq).sum().item()
                        l2sp_loss_count += n_s
                    elif grad.ndim == 2 and mask.shape[0] == grad.shape[1]:
                        delta = param.data[:, mask] - w0_slice
                        weighted = delta * scores.unsqueeze(0)
                        grad[:, mask] += l2sp_lam * weighted.to(grad.dtype)
                        ch_delta_sq = delta.pow(2).sum(dim=0)
                        l2sp_loss_sum += (scores * ch_delta_sq).sum().item()
                        l2sp_loss_count += scores.shape[0]

                # Mixed regularization: relative-L2 (fc1) + cosine (fc2)
                if mixed_table and id(param) in mixed_table:
                    entry = mixed_table[id(param)]
                    data0, mask, aux = entry
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                        if data0.device != param.device:
                            data0 = data0.to(param.device)
                        if aux is not None and aux.device != param.device:
                            aux = aux.to(param.device)
                        mixed_table[id(param)] = (data0, mask, aux)
                    elif data0.device != param.device:
                        data0 = data0.to(param.device)
                        if aux is not None and aux.device != param.device:
                            aux = aux.to(param.device)
                        mixed_table[id(param)] = (data0, mask, aux)

                    if grad.ndim == 2 and mask.shape[0] == grad.shape[0]:
                        # fc1: relative L2 正则
                        # data0 = w0_slice [n_shared*2, hidden]
                        # aux = w0_row_norm_sq [n_shared*2]
                        w0_slice_fc1 = data0
                        w0_row_norm_sq = aux
                        delta = param.data[mask] - w0_slice_fc1  # [n_shared*2, hidden]
                        # grad += 2 * λ_in * delta / ||w0[j,:]||²
                        inv_norm_sq = (1.0 / w0_row_norm_sq).unsqueeze(1)  # [n_shared*2, 1]
                        grad[mask] += (2.0 * rel_l2_lam * inv_norm_sq * delta).to(grad.dtype)
                        # loss = λ_in * Σ_j ||Δw[j,:]||² / ||w0[j,:]||²
                        delta_sq_sum = delta.pow(2).sum(dim=1)  # [n_shared*2]
                        mixed_loss_sum += (rel_l2_lam * (delta_sq_sum / w0_row_norm_sq)).sum().item()
                        mixed_loss_count += mask.sum().item() // 2  # 每个 channel 贡献 gate + up

                    elif grad.ndim == 2 and mask.shape[0] == grad.shape[1]:
                        # fc2: cosine distance 正则
                        # data0 = w0_col_unit [hidden, n_shared]
                        # aux = None
                        w0_unit = data0
                        w_cols = param.data[:, mask]  # [hidden, n_shared]
                        w_norm = w_cols.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [1, n_shared]
                        w_unit = w_cols / w_norm  # [hidden, n_shared]
                        cos_sim = (w_unit * w0_unit).sum(dim=0, keepdim=True)  # [1, n_shared]
                        # grad = -λ_angle / ||w|| * (w0_hat - cos_sim * w_hat)
                        cos_grad = -(1.0 / w_norm) * (w0_unit - cos_sim * w_unit)
                        grad[:, mask] += (cosine_lam * cos_grad).to(grad.dtype)
                        # loss = λ_angle * Σ_j (1 - cos(w[:,j], w0[:,j]))
                        mixed_loss_sum += (cosine_lam * (1.0 - cos_sim.squeeze(0))).sum().item()
                        mixed_loss_count += mask.sum().item()

                # 消融: absolute L2 on both FC1/FC2
                if abs_l2_lam and id(param) in abs_l2_table:
                    w0_slice, mask = abs_l2_table[id(param)]
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                        abs_l2_table[id(param)] = (w0_slice, mask)
                    if w0_slice.device != param.device:
                        w0_slice = w0_slice.to(param.device)
                        abs_l2_table[id(param)] = (w0_slice, mask)
                    if grad.ndim == 2 and mask.shape[0] == grad.shape[0]:
                        delta = param.data[mask] - w0_slice
                        grad[mask] += abs_l2_lam * delta.to(grad.dtype)
                        ablation_loss_sum += 0.5 * abs_l2_lam * delta.pow(2).sum().item()
                        ablation_loss_count += 1
                    elif grad.ndim == 2 and mask.shape[0] == grad.shape[1]:
                        delta = param.data[:, mask] - w0_slice
                        grad[:, mask] += abs_l2_lam * delta.to(grad.dtype)
                        ablation_loss_sum += 0.5 * abs_l2_lam * delta.pow(2).sum().item()
                        ablation_loss_count += 1

                # 消融: cosine distance on both FC1/FC2
                if all_cosine_lam and id(param) in all_cosine_table:
                    entry = all_cosine_table[id(param)]
                    w0_unit, mask, proj_type = entry
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                        if w0_unit.device != param.device:
                            w0_unit = w0_unit.to(param.device)
                        all_cosine_table[id(param)] = (w0_unit, mask, proj_type)
                    elif w0_unit.device != param.device:
                        w0_unit = w0_unit.to(param.device)
                        all_cosine_table[id(param)] = (w0_unit, mask, proj_type)

                    if proj_type == "fc1":
                        # FC1 [2I, hidden], mask on rows, cosine per row
                        w_rows = param.data[mask]  # [n_shared*2, hidden]
                        w_norm = w_rows.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        w_hat = w_rows / w_norm
                        cos_sim = (w_hat * w0_unit).sum(dim=1, keepdim=True)  # [n_shared*2, 1]
                        cos_grad = -(1.0 / w_norm) * (w0_unit - cos_sim * w_hat)
                        grad[mask] += (all_cosine_lam * cos_grad).to(grad.dtype)
                        ablation_loss_sum += (all_cosine_lam * (1.0 - cos_sim.squeeze(1))).sum().item()
                        ablation_loss_count += 1
                    elif proj_type == "fc2":
                        # FC2 [hidden, I], mask on cols, cosine per col
                        w_cols = param.data[:, mask]  # [hidden, n_shared]
                        w_norm = w_cols.norm(dim=0, keepdim=True).clamp(min=1e-8)
                        w_hat = w_cols / w_norm
                        cos_sim = (w_hat * w0_unit).sum(dim=0, keepdim=True)  # [1, n_shared]
                        cos_grad = -(1.0 / w_norm) * (w0_unit - cos_sim * w_hat)
                        grad[:, mask] += (all_cosine_lam * cos_grad).to(grad.dtype)
                        ablation_loss_sum += (all_cosine_lam * (1.0 - cos_sim.squeeze(0))).sum().item()
                        ablation_loss_count += 1

                # 消融: swapped mixed — cosine on fc1 rows, relative-L2 on fc2 cols
                if swapped_table and id(param) in swapped_table:
                    entry = swapped_table[id(param)]
                    data0, mask, aux = entry
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                        if data0.device != param.device:
                            data0 = data0.to(param.device)
                        if aux is not None and not isinstance(aux, str) and aux.device != param.device:
                            aux = aux.to(param.device)
                        swapped_table[id(param)] = (data0, mask, aux)
                    elif data0.device != param.device:
                        data0 = data0.to(param.device)
                        if aux is not None and not isinstance(aux, str) and aux.device != param.device:
                            aux = aux.to(param.device)
                        swapped_table[id(param)] = (data0, mask, aux)

                    if aux == "fc1":
                        # fc1: cosine distance on gate/up rows
                        w0_unit = data0
                        w_rows = param.data[mask]
                        w_norm = w_rows.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        w_unit = w_rows / w_norm
                        cos_sim = (w_unit * w0_unit).sum(dim=1, keepdim=True)
                        cos_grad = -(1.0 / w_norm) * (w0_unit - cos_sim * w_unit)
                        grad[mask] += (swapped_cos_lam * cos_grad).to(grad.dtype)
                        ablation_loss_sum += (swapped_cos_lam * (1.0 - cos_sim.squeeze(1))).sum().item()
                        ablation_loss_count += 1
                    else:
                        # fc2: relative L2 on down_proj columns
                        w0_slice = data0
                        w0_col_norm_sq = aux
                        delta = param.data[:, mask] - w0_slice
                        inv_norm_sq = (1.0 / w0_col_norm_sq).unsqueeze(0)
                        grad[:, mask] += (2.0 * swapped_rl2_lam * inv_norm_sq * delta).to(grad.dtype)
                        delta_sq_sum = delta.pow(2).sum(dim=0)
                        ablation_loss_sum += (swapped_rl2_lam * (delta_sq_sum / w0_col_norm_sq)).sum().item()
                        ablation_loss_count += 1

                # TV-Ratio L2: grad[shared] += λ * ratio_i * (w[shared] - w0[shared])
                if tv_ratio_lam and id(param) in tv_ratio_table:
                    w0_slice, mask, scores = tv_ratio_table[id(param)]
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                        scores = scores.to(param.device)
                        tv_ratio_table[id(param)] = (w0_slice, mask, scores)
                    if w0_slice.device != param.device:
                        w0_slice = w0_slice.to(param.device)
                        tv_ratio_table[id(param)] = (w0_slice, mask, scores)
                    if grad.ndim == 2 and mask.shape[0] == grad.shape[0]:
                        delta = param.data[mask] - w0_slice
                        weighted = delta * scores.unsqueeze(1)
                        grad[mask] += tv_ratio_lam * weighted.to(grad.dtype)
                        n_s = scores.shape[0] // 2
                        ch_delta_sq = delta.view(2, n_s, -1).pow(2).sum(dim=(0, 2))
                        ch_scores = scores.view(2, n_s)[0]
                        l2sp_loss_sum += (ch_scores * ch_delta_sq).sum().item()
                        l2sp_loss_count += n_s
                    elif grad.ndim == 2 and mask.shape[0] == grad.shape[1]:
                        delta = param.data[:, mask] - w0_slice
                        weighted = delta * scores.unsqueeze(0)
                        grad[:, mask] += tv_ratio_lam * weighted.to(grad.dtype)
                        ch_delta_sq = delta.pow(2).sum(dim=0)
                        l2sp_loss_sum += (scores * ch_delta_sq).sum().item()
                        l2sp_loss_count += scores.shape[0]

        if (l2sp_lam or tv_ratio_lam) and l2sp_loss_count > 0:
            import torch.distributed as dist
            loss_tensor = torch.tensor([l2sp_loss_sum], device=param.device)
            if dist.is_initialized():
                try:
                    from megatron.core import mpu
                    dp_group = mpu.get_data_parallel_group()
                except Exception:
                    dp_group = None
                if dp_group is not None:
                    dist.all_reduce(loss_tensor, group=dp_group)
            effective_lam = l2sp_lam if l2sp_lam else tv_ratio_lam
            self._stored_l2sp_loss = 0.5 * effective_lam * loss_tensor.item()
        elif mixed_table and mixed_loss_count > 0:
            import torch.distributed as dist
            loss_tensor = torch.tensor([mixed_loss_sum], device=param.device)
            if dist.is_initialized():
                try:
                    from megatron.core import mpu
                    dp_group = mpu.get_data_parallel_group()
                except Exception:
                    dp_group = None
                if dp_group is not None:
                    dist.all_reduce(loss_tensor, group=dp_group)
            self._stored_l2sp_loss = loss_tensor.item()
        elif (abs_l2_table or all_cosine_table or swapped_table) and ablation_loss_count > 0:
            import torch.distributed as dist
            loss_tensor = torch.tensor([ablation_loss_sum], device=param.device)
            if dist.is_initialized():
                try:
                    from megatron.core import mpu
                    dp_group = mpu.get_data_parallel_group()
                except Exception:
                    dp_group = None
                if dp_group is not None:
                    dist.all_reduce(loss_tensor, group=dp_group)
            self._stored_l2sp_loss = loss_tensor.item()
        else:
            self._stored_l2sp_loss = None

    def get_l2sp_loss(self) -> Optional[float]:
        """返回最近一次 apply() 计算的 L2-SP loss 标量."""
        return self._stored_l2sp_loss

    def print_plan(self) -> None:
        role_counter: Dict[str, int] = {}
        for row in self._plan:
            role_counter[row.role] = role_counter.get(row.role, 0) + 1

        print_rank_0(f"[surgery-plan] policy: v2 (binary, Q/O frozen, K/V full, LN frozen)")
        print_rank_0(f"[surgery-plan] mask file: {self._mask_path}")
        print_rank_0(
            f"[surgery-plan] layers covered: {self._layers_covered[0]}..{self._layers_covered[1]}"
        )
        anchor_str = (
            f" shared_anchor_lambda={self._shared_anchor_lambda}"
            if self._shared_anchor_lambda else ""
        )
        l2sp_str = (
            f" shared_l2sp_lambda={self._shared_l2sp_lambda}"
            if self._shared_l2sp_lambda else ""
        )
        mixed_str = (
            f" shared_rel_l2_lambda={self._shared_rel_l2_lambda}"
            f" shared_cosine_lambda={self._shared_cosine_lambda}"
            if self._shared_rel_l2_lambda else ""
        )
        abs_l2_str = (
            f" shared_abs_l2_lambda={self._shared_abs_l2_lambda}"
            if self._shared_abs_l2_lambda else ""
        )
        all_cosine_str = (
            f" shared_all_cosine_lambda={self._shared_all_cosine_lambda}"
            if self._shared_all_cosine_lambda else ""
        )
        swapped_str = (
            f" shared_swapped_cosine_lambda(fc1)={self._shared_swapped_cosine_lambda}"
            f" shared_swapped_rel_l2_lambda(fc2)={self._shared_swapped_rel_l2_lambda}"
            if self._shared_swapped_cosine_lambda else ""
        )
        tv_ratio_str = (
            f" shared_tv_ratio_lambda={self._shared_tv_ratio_lambda}"
            if self._shared_tv_ratio_lambda else ""
        )
        print_rank_0(
            "[surgery-plan] scaling constants: "
            f"text={TEXT_SCALE} shared={SHARED_SCALE} vision={VISION_SCALE} idle={IDLE_SCALE}"
            f" | attn_q={ATTN_Q_SCALE} attn_k={ATTN_K_SCALE} attn_v={ATTN_V_SCALE} attn_o={ATTN_O_SCALE}"
            f" | ln={LN_SCALE} emb={EMB_SCALE} output={OUTPUT_SCALE}"
            + anchor_str + l2sp_str + mixed_str + abs_l2_str + all_cosine_str + swapped_str + tv_ratio_str
        )
        for row in self._plan:
            print_rank_0(
                f"[surgery-plan] {row.name}    role={row.role}    scale={row.scale_repr}"
            )

        total = len(self._plan)
        with_surgery = len(self._scale_table)
        pct = (100.0 * with_surgery / total) if total else 0.0
        print_rank_0(
            f"[surgery-plan] TOTAL params with surgery (in scale_table): "
            f"{with_surgery}/{total} ({pct:.2f}%)"
        )
        print_rank_0(
            "[surgery-plan] roles: "
            + " ".join(f"{k}={v}" for k, v in sorted(role_counter.items()))
        )

    def audit_first_step(self, model) -> None:
        if not self._audit_targets:
            return
        failures: List[str] = []
        for tgt in self._audit_targets:
            grad = getattr(tgt.param, "main_grad", None)
            if grad is None:
                failures.append(
                    f"{tgt.param_name}: main_grad is None (backward did not populate it)"
                )
                continue
            idx_t = torch.tensor(tgt.row_indices, device=grad.device)
            slab = grad.index_select(tgt.axis, idx_t)
            val = slab.abs().max().item() if slab.numel() > 0 else 0.0
            if tgt.expected_zero and val != 0.0:
                failures.append(
                    f"{tgt.param_name}: expected zero at frozen rows {tgt.row_indices}, "
                    f"got |max|={val:.3e}"
                )
            if not tgt.expected_zero and val == 0.0:
                failures.append(
                    f"{tgt.param_name}: expected >0 at trainable rows {tgt.row_indices}, "
                    f"got |max|=0.0"
                )
        if failures:
            head = "\n  ".join(failures[:10])
            raise RuntimeError(f"[surgery-audit] FAILED ({len(failures)}):\n  {head}")
        print_rank_0(
            f"[surgery-audit] PASS ({len(self._audit_targets)} targets checked)"
        )

    # ---- 内部工具 ----
    @staticmethod
    def _validate_mask_dict(
        mask_dict, expected_num_layers: int, expected_intermediate_size: int,
    ) -> None:
        if not isinstance(mask_dict, dict):
            raise ValueError("gradient surgery mask: expected dict at top level")
        actual_keys = set(mask_dict.keys())
        expected_keys = set(MASK_TOP_KEYS)
        if actual_keys != expected_keys:
            raise ValueError(
                f"gradient surgery mask: top keys mismatch; "
                f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
            )
        for top in MASK_TOP_KEYS:
            sub = mask_dict[top]
            if not isinstance(sub, dict):
                raise ValueError(f"gradient surgery mask['{top}']: expected dict")
            if len(sub) != expected_num_layers:
                raise ValueError(
                    f"gradient surgery mask['{top}']: layer count mismatch; "
                    f"mask={len(sub)}, model.num_layers={expected_num_layers}"
                )
            for idx in range(expected_num_layers):
                key = MASK_KEY_FMT.format(idx=idx)
                if key not in sub:
                    raise ValueError(
                        f"gradient surgery mask['{top}']: missing key '{key}'"
                    )
                t = sub[key]
                if not torch.is_tensor(t):
                    raise ValueError(
                        f"gradient surgery mask['{top}']['{key}']: not a tensor"
                    )
                if t.numel() != expected_intermediate_size:
                    raise ValueError(
                        f"gradient surgery mask['{top}']['{key}']: size mismatch; "
                        f"mask={t.numel()}, intermediate={expected_intermediate_size}"
                    )
                uniq = set(t.unique().tolist())
                if not uniq.issubset({0.0, 1.0}):
                    raise ValueError(
                        f"gradient surgery mask['{top}']['{key}']: non-binary values "
                        f"{sorted(uniq)[:5]}"
                    )

        # 重叠检测：warning
        for idx in range(expected_num_layers):
            key = MASK_KEY_FMT.format(idx=idx)
            stacked = torch.stack(
                [mask_dict[t][key] for t in MASK_TOP_KEYS], dim=0
            ).sum(dim=0)
            n_over = int((stacked > 1).sum().item())
            if n_over > 0:
                print_rank_0(
                    f"[surgery-plan] WARN: layer {idx} has {n_over} overlapped neuron(s) "
                    f"across mask sets"
                )

    @staticmethod
    def _compute_shared_routing_scales(
        scores_dir: str, num_layers: int, alpha: float, power: float,
    ) -> Dict[int, torch.Tensor]:
        """从 npz 加载原始 Wanda score，做 percentile rank 归一化后计算 M_i."""
        vision_path = os.path.join(scores_dir, "vision_profiles.npz")
        text_path = os.path.join(scores_dir, "text_profiles.npz")
        if not os.path.isfile(vision_path):
            raise FileNotFoundError(f"shared routing: vision scores not found: {vision_path}")
        if not os.path.isfile(text_path):
            raise FileNotFoundError(f"shared routing: text scores not found: {text_path}")

        vision_npz = np.load(vision_path)
        text_npz = np.load(text_path)

        out: Dict[int, torch.Tensor] = {}
        for idx in range(num_layers):
            key = MASK_KEY_FMT.format(idx=idx)
            v_raw = torch.from_numpy(vision_npz[key].copy()).float()
            t_raw = torch.from_numpy(text_npz[key].copy()).float()

            # percentile rank 归一化 (与 wanda_modality_rank_based.py 一致)
            v_rank = v_raw.argsort().argsort().float() / (len(v_raw) - 1 + SHARED_ROUTING_EPS)
            t_rank = t_raw.argsort().argsort().float() / (len(t_raw) - 1 + SHARED_ROUTING_EPS)

            # M_i = (S_vision_i / (S_vision_i + α * S_text_i + ε))^r
            m_i = (v_rank / (v_rank + alpha * t_rank + SHARED_ROUTING_EPS)).pow(power)
            out[idx] = m_i

        return out

    @staticmethod
    def _load_l2sp_text_scores(
        scores_dir: str, num_layers: int,
    ) -> Dict[int, torch.Tensor]:
        """从 text_profiles.npz 加载 Wanda 文本分数，percentile rank 归一化到 (0, 1]."""
        text_path = os.path.join(scores_dir, "text_profiles.npz")
        if not os.path.isfile(text_path):
            raise FileNotFoundError(f"L2SP text scores not found: {text_path}")

        text_npz = np.load(text_path)
        out: Dict[int, torch.Tensor] = {}
        for idx in range(num_layers):
            key = MASK_KEY_FMT.format(idx=idx)
            if key not in text_npz:
                raise KeyError(f"L2SP text scores: missing key '{key}' in {text_path}")
            t_raw = torch.from_numpy(text_npz[key].copy()).float()
            # percentile rank 归一化到 (0, 1]
            t_rank = t_raw.argsort().argsort().float() / (len(t_raw) - 1 + SHARED_ROUTING_EPS)
            out[idx] = t_rank
        return out

    @staticmethod
    def _load_tv_ratio_scores(
        scores_dir: str, num_layers: int,
    ) -> Dict[int, torch.Tensor]:
        """加载 text/vision 分数, 计算 TV-ratio: T_rank / (T_rank + V_rank + ε)."""
        text_path = os.path.join(scores_dir, "text_profiles.npz")
        vision_path = os.path.join(scores_dir, "vision_profiles.npz")
        if not os.path.isfile(text_path):
            raise FileNotFoundError(f"TV-ratio text scores not found: {text_path}")
        if not os.path.isfile(vision_path):
            raise FileNotFoundError(f"TV-ratio vision scores not found: {vision_path}")

        text_npz = np.load(text_path)
        vision_npz = np.load(vision_path)
        out: Dict[int, torch.Tensor] = {}
        for idx in range(num_layers):
            key = MASK_KEY_FMT.format(idx=idx)
            if key not in text_npz:
                raise KeyError(f"TV-ratio text scores: missing key '{key}' in {text_path}")
            if key not in vision_npz:
                raise KeyError(f"TV-ratio vision scores: missing key '{key}' in {vision_path}")
            t_raw = torch.from_numpy(text_npz[key].copy()).float()
            v_raw = torch.from_numpy(vision_npz[key].copy()).float()
            # percentile rank 归一化到 (0, 1]
            t_rank = t_raw.argsort().argsort().float() / (len(t_raw) - 1 + SHARED_ROUTING_EPS)
            v_rank = v_raw.argsort().argsort().float() / (len(v_raw) - 1 + SHARED_ROUTING_EPS)
            # λ_i = T_rank / (T_rank + V_rank + ε)
            ratio = t_rank / (t_rank + v_rank + SHARED_ROUTING_EPS)
            out[idx] = ratio
        return out

    @staticmethod
    def _precompute_per_layer_mlp_scales(
        mask_dict, num_layers: int, shared_anchor_enabled: bool = False,
        shared_routing_scales: Optional[Dict[int, torch.Tensor]] = None,
        shared_full_update: bool = False,
        shared_l2sp_enabled: bool = False,
        shared_mixed_enabled: bool = False,
        shared_abs_l2_enabled: bool = False,
        shared_all_cosine_enabled: bool = False,
        shared_swapped_enabled: bool = False,
        shared_tv_ratio_enabled: bool = False,
        freeze_ablation: Optional[str] = None,
    ) -> Dict[int, torch.Tensor]:
        """每层一个 [I] 的 fp32 scale.

        默认: vision_only=1, idle=1, shared=0, text=0, 未覆盖=0.
        shared_anchor_enabled=True 时: shared 也设为 1.0 (梯度不再被 zero 掉,
        锚定正则由 apply() 单独施加).
        shared_routing_scales 不为 None 时: shared 通道使用自适应 M_i 值.
        shared_full_update=True 时: shared 通道全量更新 (scale=1.0), 无正则.
        shared_l2sp_enabled=True 时: shared 通道全量更新 (scale=1.0), L2-SP 正则由 apply() 单独施加.
        shared_mixed_enabled=True 时: shared 通道全量更新 (scale=1.0), mixed 正则由 apply() 单独施加.
        shared_abs_l2_enabled / shared_all_cosine_enabled / shared_swapped_enabled: 消融, shared scale=1.0.
        freeze_ablation: 通道隔离消融, 只有指定类型 scale=1.0, 其余全 scale=0.0.
        """
        out: Dict[int, torch.Tensor] = {}
        for idx in range(num_layers):
            key = MASK_KEY_FMT.format(idx=idx)

            if freeze_ablation is not None:
                # 通道隔离消融: 只有指定类型的通道 scale=1.0
                scale = mask_dict[freeze_ablation][key].float()
                out[idx] = scale
                continue

            vision = mask_dict["vision_only"][key].float()
            idle = mask_dict["idle"][key].float()
            scale = VISION_SCALE * vision + IDLE_SCALE * idle
            if shared_routing_scales is not None:
                shared_mask = mask_dict["shared"][key].float()
                scale = scale + shared_mask * shared_routing_scales[idx]
            elif shared_anchor_enabled or shared_full_update or shared_l2sp_enabled or shared_mixed_enabled or shared_abs_l2_enabled or shared_all_cosine_enabled or shared_swapped_enabled or shared_tv_ratio_enabled:
                shared = mask_dict["shared"][key].float()
                scale = scale + shared  # shared channels get scale=1.0
            out[idx] = scale
        return out

    @staticmethod
    def _classify(name: str) -> _ClassifyResult:
        if _RE_VISION.search(name):
            return _ClassifyResult(role="vision_full")
        if _RE_ADAPTER.search(name) and not _RE_LANG.search(name):
            return _ClassifyResult(role="adapter_full")
        if _RE_EMBEDDING.search(name):
            return _ClassifyResult(role="frozen_emb", scalar=EMB_SCALE)
        if _RE_OUTPUT_LAYER.search(name):
            return _ClassifyResult(role="frozen_out", scalar=OUTPUT_SCALE)

        # MLP
        m = _RE_MLP_FC1.search(name)
        if m is not None:
            return _ClassifyResult(role="mlp_masked", layer_idx=int(m.group(1)), is_fc1=True)
        m = _RE_MLP_FC2_WEIGHT.search(name)
        if m is not None:
            return _ClassifyResult(role="mlp_masked", layer_idx=int(m.group(1)), is_fc2=True)
        m = _RE_MLP_FC2_BIAS.search(name)
        if m is not None:
            return _ClassifyResult(
                role="mlp_masked", layer_idx=int(m.group(1)),
                is_fc2=True, fc2_is_bias=True,
            )

        # Attention：先 LN（fused 在 linear_qkv.layer_norm_weight）再 QKV/O
        if _RE_FINAL_LN.search(name) or _RE_LN_IN_LAYER.search(name):
            return _ClassifyResult(role="ln_soft", scalar=LN_SCALE)

        if _RE_ATTN_QKV.search(name):
            return _ClassifyResult(role="attn_qkv_partial")
        if _RE_ATTN_O.search(name):
            return _ClassifyResult(role="attn_o_frozen", scalar=ATTN_O_SCALE)

        if _RE_LANG.search(name):
            return _ClassifyResult(role="unhandled")
        return _ClassifyResult(role="unhandled")

    @staticmethod
    def _build_qkv_row_scale(
        param: torch.nn.Parameter, num_q_heads: int, num_kv_heads: int,
    ) -> Tuple[torch.Tensor, str]:
        """linear_qkv 行级 scale: Q rows=ATTN_Q_SCALE, K rows=ATTN_K_SCALE, V rows=ATTN_V_SCALE."""
        total = param.shape[0]
        denom = num_q_heads + 2 * num_kv_heads
        if total % denom != 0:
            raise RuntimeError(
                f"linear_qkv shape mismatch: total={total} not divisible by "
                f"(num_q_heads + 2*num_kv_heads) = {denom}"
            )
        head_dim = total // denom
        n_q = num_q_heads * head_dim
        n_kv = num_kv_heads * head_dim
        scale = torch.empty(total, dtype=torch.float32, device=param.device)
        scale[:n_q] = ATTN_Q_SCALE
        scale[n_q:n_q + n_kv] = ATTN_K_SCALE
        scale[n_q + n_kv:] = ATTN_V_SCALE
        repr_str = (
            f"vec[{total}] Q[0:{n_q}]={ATTN_Q_SCALE} "
            f"K[{n_q}:{n_q+n_kv}]={ATTN_K_SCALE} "
            f"V[{n_q+n_kv}:{total}]={ATTN_V_SCALE}"
        )
        return scale, repr_str

    @staticmethod
    def _build_scale_tensor(
        classify: _ClassifyResult,
        name: str,
        param: torch.nn.Parameter,
        per_layer_mlp_scale: Dict[int, torch.Tensor],
        args,
    ) -> Tuple[torch.Tensor, str]:
        device = param.device
        dtype = torch.float32

        if classify.scalar is not None:
            t = torch.tensor(float(classify.scalar), dtype=dtype, device=device)
            return t, f"scalar={float(classify.scalar)}"

        if classify.role == "attn_qkv_partial":
            num_q = args.num_attention_heads
            num_kv = getattr(args, "num_query_groups", num_q)
            return GradientSurgeryManager._build_qkv_row_scale(param, num_q, num_kv)

        if classify.is_fc2 and classify.fc2_is_bias:
            # down_proj.bias 维度是 hidden（与 neuron 无关），保持不动
            t = torch.tensor(1.0, dtype=dtype, device=device)
            return t, "scalar=1.0 (fc2.bias no-op)"

        if classify.is_fc1:
            per_layer = per_layer_mlp_scale[classify.layer_idx].to(
                device=device, dtype=dtype
            )
            scale_2i = torch.cat([per_layer, per_layer], dim=0)
            n_full = int((scale_2i > 0).sum().item())
            return scale_2i, f"vec[{scale_2i.numel()}] full={n_full}"

        if classify.is_fc2:
            per_layer = per_layer_mlp_scale[classify.layer_idx].to(
                device=device, dtype=dtype
            )
            n_full = int((per_layer > 0).sum().item())
            return per_layer, f"vec[{per_layer.numel()}] full={n_full}"

        raise RuntimeError(
            f"_build_scale_tensor: unreachable name={name} role={classify.role}"
        )

    @staticmethod
    def _select_audit_targets(
        model, mask_dict, num_layers: int,
        shared_anchor_enabled: bool = False,
        shared_routing_enabled: bool = False,
        shared_full_update: bool = False,
        shared_l2sp_enabled: bool = False,
        shared_mixed_enabled: bool = False,
        shared_abs_l2_enabled: bool = False,
        shared_all_cosine_enabled: bool = False,
        shared_swapped_enabled: bool = False,
        shared_tv_ratio_enabled: bool = False,
        freeze_ablation: Optional[str] = None,
    ) -> List[AuditTarget]:
        """v2 抽检：text_only/shared 应=0，vision_only/idle 应≠0.
        若 shared 策略启用，shared 应≠0.
        若 freeze_ablation 启用，只有指定类型应≠0."""
        targets: List[AuditTarget] = []
        layer_picks = [0, num_layers // 2, num_layers - 1]

        # freeze_ablation 模式下的 expected_zero 判断
        def _expected_zero_for(channel_type: str) -> bool:
            if freeze_ablation is not None:
                return channel_type != freeze_ablation
            if channel_type == "text_only":
                return True
            if channel_type == "shared":
                return not (shared_anchor_enabled or shared_routing_enabled or shared_full_update or shared_l2sp_enabled or shared_mixed_enabled or shared_abs_l2_enabled or shared_all_cosine_enabled or shared_swapped_enabled or shared_tv_ratio_enabled)
            # vision_only, idle
            return False

        # 找每层 linear_fc1.weight 与 linear_fc2.weight
        fc1_params: Dict[int, Tuple[str, torch.nn.Parameter]] = {}
        fc2_params: Dict[int, Tuple[str, torch.nn.Parameter]] = {}
        qkv_params: Dict[int, Tuple[str, torch.nn.Parameter]] = {}
        for chunk in _iter_unwrapped_chunks(model):
            for name, param in chunk.named_parameters():
                m = _RE_MLP_FC1.search(name)
                if m is not None and name.endswith(".weight"):
                    fc1_params.setdefault(int(m.group(1)), (name, param))
                    continue
                m = _RE_MLP_FC2_WEIGHT.search(name)
                if m is not None:
                    fc2_params.setdefault(int(m.group(1)), (name, param))
                    continue
                m = _RE_ATTN_QKV.search(name)
                if m is not None and name.endswith(".weight"):
                    layer_idx = int(_RE_LAYER_IDX.search(name).group(1))
                    qkv_params.setdefault(layer_idx, (name, param))

        for layer_idx in layer_picks:
            key = MASK_KEY_FMT.format(idx=layer_idx)
            text_idx = (mask_dict["text_only"][key] > 0).nonzero(as_tuple=False).flatten().tolist()[:3]
            shared_idx = (mask_dict["shared"][key] > 0).nonzero(as_tuple=False).flatten().tolist()[:3]
            vision_idx = (mask_dict["vision_only"][key] > 0).nonzero(as_tuple=False).flatten().tolist()[:3]
            idle_idx = (mask_dict["idle"][key] > 0).nonzero(as_tuple=False).flatten().tolist()[:3]

            if layer_idx in fc1_params:
                fc1_name, fc1_param = fc1_params[layer_idx]
                # fc1 是 [2I, hidden]，gate 行 = [0, I)，row 索引同 mask 索引
                if text_idx:
                    targets.append(AuditTarget(
                        param_name=fc1_name, param=fc1_param,
                        row_indices=text_idx,
                        expected_zero=_expected_zero_for("text_only"),
                        axis=0,
                    ))
                if shared_idx:
                    targets.append(AuditTarget(
                        param_name=fc1_name, param=fc1_param,
                        row_indices=shared_idx,
                        expected_zero=_expected_zero_for("shared"),
                        axis=0,
                    ))
                if vision_idx:
                    targets.append(AuditTarget(
                        param_name=fc1_name, param=fc1_param,
                        row_indices=vision_idx,
                        expected_zero=_expected_zero_for("vision_only"),
                        axis=0,
                    ))
                if idle_idx:
                    targets.append(AuditTarget(
                        param_name=fc1_name, param=fc1_param,
                        row_indices=idle_idx,
                        expected_zero=_expected_zero_for("idle"),
                        axis=0,
                    ))

            # attention QKV：抽 Q row（应=0）和 K/V row（应≠0），仅在第一层做一次足够
            if layer_idx == layer_picks[0] and layer_idx in qkv_params:
                qkv_name, qkv_param = qkv_params[layer_idx]
                # 不便于精确取 Q/K/V row 索引（依赖 args），这里抽前 3 行（Q）和 后 3 行（V）
                # Q: 前 3 行
                targets.append(AuditTarget(
                    param_name=qkv_name + " (Q rows)", param=qkv_param,
                    row_indices=[0, 1, 2], expected_zero=True, axis=0,
                ))
                # V: 最后 3 行
                t = qkv_param.shape[0]
                targets.append(AuditTarget(
                    param_name=qkv_name + " (V rows)", param=qkv_param,
                    row_indices=[t - 3, t - 2, t - 1], expected_zero=False, axis=0,
                ))

        return targets


# ---- 模块级辅助 ----
def _iter_unwrapped_chunks(model):
    from megatron.training.utils import unwrap_model

    if isinstance(model, (list, tuple)):
        for chunk in model:
            yield unwrap_model(chunk)
    else:
        yield unwrap_model(model)


def _mul_with_broadcast(
    grad: torch.Tensor, scale: torch.Tensor, param_shape: Tuple[int, ...],
) -> None:
    if scale.dtype != grad.dtype:
        scale = scale.to(dtype=grad.dtype)
    if scale.device != grad.device:
        scale = scale.to(device=grad.device)

    if scale.ndim == 0:
        grad.mul_(scale)
        return

    if grad.ndim == 1:
        if scale.shape[0] != grad.shape[0]:
            raise RuntimeError(
                f"gradient surgery shape mismatch: "
                f"grad={tuple(grad.shape)}, scale={tuple(scale.shape)}"
            )
        grad.mul_(scale)
        return

    if scale.shape[0] == grad.shape[0]:
        view_shape = (scale.shape[0],) + (1,) * (grad.ndim - 1)
        grad.mul_(scale.view(view_shape))
        return

    if scale.shape[0] == grad.shape[-1]:
        view_shape = (1,) * (grad.ndim - 1) + (scale.shape[0],)
        grad.mul_(scale.view(view_shape))
        return

    raise RuntimeError(
        f"gradient surgery shape mismatch: grad={tuple(grad.shape)}, "
        f"scale={tuple(scale.shape)}, param_shape={param_shape}"
    )
