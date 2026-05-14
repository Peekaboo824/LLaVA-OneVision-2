"""模态感知的异质梯度手术 (Heterogeneous Gradient Surgery, v2 binary policy).

启动时一次性读入 MLP 路由掩码 (top-60% 互斥四集合)，按二值规则给 language_model
的每个参数预先计算缩放 tensor；每个训练 step 在 forward-backward 完成后、
optimizer.step() 之前直接作用于 param.main_grad。参数不改 requires_grad，
模型结构/前向不变。

策略 v2 (二值化, 与早先 v1 软系数策略不同):
  MLP (linear_fc1 / linear_fc2)：按 down_proj mask 分类
      text_only   -> 0.0 (冻; 文本核心保护)
      shared      -> 0.0 (冻; 让给文本, 避免污染通用语义层)
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

import torch

from aiak_training_llm.utils import print_rank_0


# ---- 硬编码系数 (v2 二值策略) ----
TEXT_SCALE = 0.0
SHARED_SCALE = 0.0       # v2: 让给纯文本, 冻 (v1 是 0.1)
VISION_SCALE = 1.0
IDLE_SCALE = 1.0         # v2: 让给多模态, 全量 (v1 是 0.0)

ATTN_Q_SCALE = 0.0       # 冻
ATTN_K_SCALE = 1.0       # 全量
ATTN_V_SCALE = 1.0       # 全量
ATTN_O_SCALE = 0.0       # 冻

LN_SCALE = 0.0           # v2: 全冻 (v1 是 0.1)
EMB_SCALE = 0.0
OUTPUT_SCALE = 0.0

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
    ):
        self._scale_table = scale_table
        self._plan = plan
        self._audit_targets = audit_targets
        self._mask_path = mask_path
        self._layers_covered = layers_covered

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

        mask_dict = torch.load(mask_path, map_location="cpu", weights_only=False)
        cls._validate_mask_dict(
            mask_dict,
            expected_num_layers=args.num_layers,
            expected_intermediate_size=args.ffn_hidden_size,
        )

        # 预计算每层 MLP 的 per-neuron scale (intermediate axis)
        per_layer_mlp_scale = cls._precompute_per_layer_mlp_scales(
            mask_dict, args.num_layers
        )

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

        audit_targets = cls._select_audit_targets(model, mask_dict, args.num_layers)

        return cls(
            scale_table=scale_table,
            plan=plan,
            audit_targets=audit_targets,
            mask_path=mask_path,
            layers_covered=(0, args.num_layers - 1),
        )

    # ---- 运行期 ----
    def apply(self, model) -> None:
        if not self._scale_table:
            return
        table = self._scale_table
        for chunk in _iter_unwrapped_chunks(model):
            for param in chunk.parameters():
                scale = table.get(id(param))
                if scale is None:
                    continue
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    continue
                _mul_with_broadcast(grad, scale, param_shape=tuple(param.shape))

    def print_plan(self) -> None:
        role_counter: Dict[str, int] = {}
        for row in self._plan:
            role_counter[row.role] = role_counter.get(row.role, 0) + 1

        print_rank_0(f"[surgery-plan] policy: v2 (binary, Q/O frozen, K/V full, LN frozen)")
        print_rank_0(f"[surgery-plan] mask file: {self._mask_path}")
        print_rank_0(
            f"[surgery-plan] layers covered: {self._layers_covered[0]}..{self._layers_covered[1]}"
        )
        print_rank_0(
            "[surgery-plan] scaling constants: "
            f"text={TEXT_SCALE} shared={SHARED_SCALE} vision={VISION_SCALE} idle={IDLE_SCALE}"
            f" | attn_q={ATTN_Q_SCALE} attn_k={ATTN_K_SCALE} attn_v={ATTN_V_SCALE} attn_o={ATTN_O_SCALE}"
            f" | ln={LN_SCALE} emb={EMB_SCALE} output={OUTPUT_SCALE}"
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
    def _precompute_per_layer_mlp_scales(
        mask_dict, num_layers: int,
    ) -> Dict[int, torch.Tensor]:
        """每层一个 [I] 的 fp32 scale: vision_only=1, idle=1, shared=0, text=0, 未覆盖=0."""
        out: Dict[int, torch.Tensor] = {}
        for idx in range(num_layers):
            key = MASK_KEY_FMT.format(idx=idx)
            vision = mask_dict["vision_only"][key].float()
            idle = mask_dict["idle"][key].float()
            scale = VISION_SCALE * vision + IDLE_SCALE * idle
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
    ) -> List[AuditTarget]:
        """v2 抽检：text_only/shared 应=0，vision_only/idle 应≠0."""
        targets: List[AuditTarget] = []
        layer_picks = [0, num_layers // 2, num_layers - 1]

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
                        row_indices=text_idx, expected_zero=True, axis=0,
                    ))
                if shared_idx:
                    targets.append(AuditTarget(
                        param_name=fc1_name, param=fc1_param,
                        row_indices=shared_idx, expected_zero=True, axis=0,
                    ))
                if vision_idx:
                    targets.append(AuditTarget(
                        param_name=fc1_name, param=fc1_param,
                        row_indices=vision_idx, expected_zero=False, axis=0,
                    ))
                if idle_idx:
                    targets.append(AuditTarget(
                        param_name=fc1_name, param=fc1_param,
                        row_indices=idle_idx, expected_zero=False, axis=0,
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
