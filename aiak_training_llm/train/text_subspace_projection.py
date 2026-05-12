"""Fisher-weighted SVD gradient projection for LLM linear layers.

When enabled via ``--text-subspace-path``, the gradients of the seven target
linear layers (Q/K/V/O/Gate/Up/Down) of the language model are projected onto
the orthogonal complement of a pre-computed "text-important" subspace before
each ``optimizer.step``. See ``docs/superpowers/specs/2026-05-11-fisher-svd-
gradient-projection/`` for the design.

Key invariants enforced here:
- TP = PP = CP = 1 (raises NotImplementedError otherwise).
- ``--use-distributed-optimizer`` + ``--bf16`` are required by the spec.
- All ranks load the same V_k; projection runs on ``param.main_grad`` after the
  DP all-reduce inside the distributed optimizer, so DP consistency is preserved.
"""

import os

import torch

from megatron.core import mpu
from megatron.training.utils import unwrap_model

from aiak_training_llm.utils import get_args


# ---------------------------------------------------------------------------
# Parallel layout / args precondition
# ---------------------------------------------------------------------------

def _assert_parallel_layout(args):
    """Verify TP=PP=CP=1, distributed optimizer enabled, bf16 enabled."""
    tp = mpu.get_tensor_model_parallel_world_size()
    pp = mpu.get_pipeline_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    if tp != 1 or pp != 1 or cp != 1:
        raise NotImplementedError(
            f"text-subspace projection requires TP=PP=CP=1, "
            f"got TP={tp}, PP={pp}, CP={cp}."
        )
    if not getattr(args, "use_distributed_optimizer", False):
        raise NotImplementedError(
            "text-subspace projection requires --use-distributed-optimizer."
        )
    if not getattr(args, "bf16", False):
        raise NotImplementedError(
            "text-subspace projection requires --bf16."
        )


# ---------------------------------------------------------------------------
# Subspace dict loading & target collection
# ---------------------------------------------------------------------------

def _load_subspace(subspace_path):
    """Load the .pt subspace dict produced by extract_subspace.py (CPU)."""
    if not os.path.isfile(subspace_path):
        raise FileNotFoundError(
            f"text-subspace path does not exist: {subspace_path}"
        )
    subspace_dict = torch.load(subspace_path, map_location="cpu")
    if not isinstance(subspace_dict, dict):
        raise RuntimeError(
            f"Subspace file is not a dict: {subspace_path} -> {type(subspace_dict)}"
        )
    if "_meta" not in subspace_dict:
        raise RuntimeError(
            f"Subspace dict missing '_meta' field: {subspace_path}"
        )
    return subspace_dict


def _validate_shape(name, param, entry):
    """Ensure V_k.shape[0] == param.in (=param.shape[1]); fused: sum(split)==out."""
    in_dim = param.shape[1]
    etype = entry["type"]
    if etype == "plain":
        v = entry["V_k"]
        assert v.shape[0] == in_dim, (
            f"{name}: V_k.shape[0]={v.shape[0]} != param.in={in_dim}"
        )
    elif etype in ("fused_qkv", "fused_gate_up"):
        for i, v_i in enumerate(entry["V_k_list"]):
            assert v_i.shape[0] == in_dim, (
                f"{name}: fused V_k_list[{i}].shape[0]={v_i.shape[0]} "
                f"!= param.in={in_dim}"
            )
        split_sum = sum(entry["split_sizes"])
        assert split_sum == param.shape[0], (
            f"{name}: sum(split_sizes)={split_sum} != param.out={param.shape[0]}"
        )
    else:
        raise RuntimeError(f"{name}: unknown entry type '{etype}'")


def _detect_prefix(model_chunks, subspace_dict):
    """Detect the model-side name prefix needed to match subspace_dict keys.

    Subspace dict is stored with bare LLM-side names, e.g.
        ``decoder.layers.0.self_attention.linear_qkv.weight``
    but in a VLM the LLM is nested under a sub-module and the Megatron model
    exposes it as e.g.
        ``language_model.decoder.layers.0.self_attention.linear_qkv.weight``

    Picks the prefix ``p`` that maximizes matches between ``p + key`` and
    ``named_parameters()`` names. Returns '' if direct match is best.
    """
    dict_keys = {k for k in subspace_dict.keys() if not k.startswith("_")}
    model_names = set()
    for chunk in model_chunks:
        bare = unwrap_model(chunk)
        for name, _ in bare.named_parameters():
            model_names.add(name)

    candidates = ["", "language_model.", "text_model.", "llm."]
    best_prefix = ""
    best_hits = -1
    for p in candidates:
        hits = sum(1 for k in dict_keys if (p + k) in model_names)
        if hits > best_hits:
            best_hits = hits
            best_prefix = p
    return best_prefix, best_hits


def _collect_targets(model_chunks, subspace_dict):
    """Walk through unwrapped model chunks; match params by name in subspace_dict.

    Returns: (targets, unmatched_keys)
        targets: list of (name, param, entry)
        unmatched_keys: set of dict keys (sans _meta) that did not match any param
    """
    unmatched_keys = {k for k in subspace_dict.keys() if not k.startswith("_")}
    targets = []
    seen_names = set()

    prefix, _ = _detect_prefix(model_chunks, subspace_dict)
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f"[TextSubspace] detected model-side prefix: '{prefix}'")

    for chunk in model_chunks:
        bare = unwrap_model(chunk)
        for name, param in bare.named_parameters():
            # strip the detected prefix to look up in the subspace dict
            lookup = name[len(prefix):] if prefix and name.startswith(prefix) else name
            if lookup in subspace_dict and name not in seen_names:
                entry = subspace_dict[lookup]
                _validate_shape(name, param, entry)
                targets.append((name, param, entry))
                seen_names.add(name)
                unmatched_keys.discard(lookup)
    if not targets:
        sample_keys = [k for k in subspace_dict.keys() if not k.startswith("_")][:5]
        raise RuntimeError(
            f"No parameters matched in subspace dict. "
            f"First 5 dict keys: {sample_keys}. "
            f"Model uses different naming?"
        )
    return targets, unmatched_keys


# ---------------------------------------------------------------------------
# Move V_k tensors to CUDA (bf16)
# ---------------------------------------------------------------------------

def _move_entry_to_cuda(entry):
    """In-place move every V_k tensor in entry to CUDA. Preserves dtype (bf16)."""
    if entry["type"] == "plain":
        entry["V_k"] = entry["V_k"].cuda(non_blocking=True).contiguous()
    else:
        entry["V_k_list"] = [
            v.cuda(non_blocking=True).contiguous() for v in entry["V_k_list"]
        ]


def _entry_bytes(entry):
    """Sum of V_k tensor bytes in this entry (post-cuda)."""
    if entry["type"] == "plain":
        v = entry["V_k"]
        return v.numel() * v.element_size()
    total = 0
    for v in entry["V_k_list"]:
        total += v.numel() * v.element_size()
    return total


# ---------------------------------------------------------------------------
# Projection core
# ---------------------------------------------------------------------------

@torch.no_grad()
def _project_one(main_grad, entry):
    """In-place project main_grad onto orthogonal complement of V_k's column space.

    main_grad: [out, in], typically fp32 under --bf16 --use-distributed-optimizer.
    plain     : V_k [in, k]              -> G ← G − (G V) Vᵀ
    fused_*   : V_k_list [V_i [in, k_i]] -> split along out, project each chunk.
    """
    if entry["type"] == "plain":
        V = entry["V_k"]                       # [in, k], bf16 on GPU
        Vf = V.to(main_grad.dtype)
        # (G @ V) -> [out, k]; @ Vᵀ -> [out, in]
        main_grad.sub_(torch.mm(torch.mm(main_grad, Vf), Vf.t()))
    else:
        # G.split returns views; in-place sub_ on each view falls back to main_grad.
        chunks = main_grad.split(entry["split_sizes"], dim=0)
        for G_i, V_i in zip(chunks, entry["V_k_list"]):
            Vf = V_i.to(main_grad.dtype)
            G_i.sub_(torch.mm(torch.mm(G_i, Vf), Vf.t()))


def _apply_projection(registry):
    """For each (name, param, entry): project param.main_grad in place."""
    for _, param, entry in registry:
        main_grad = getattr(param, "main_grad", None)
        if main_grad is None:
            # No gradient produced this step (rare; trainable-modules normally guarantees grad)
            continue
        _project_one(main_grad, entry)


# ---------------------------------------------------------------------------
# optimizer.step monkey-patch (idempotent)
# ---------------------------------------------------------------------------

def _wrap_optimizer_step(optimizer, registry):
    """Wrap optimizer.step so projection runs before every real step."""
    if getattr(optimizer, "_text_subspace_wrapped", False):
        return
    original_step = optimizer.step

    def patched_step(*args, **kwargs):
        _apply_projection(registry)
        return original_step(*args, **kwargs)

    optimizer.step = patched_step
    optimizer._text_subspace_wrapped = True
    # Keep a reference on the optimizer to prevent GC and aid debugging.
    optimizer._text_subspace_registry = registry


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def attach_text_subspace_projection(model, optimizer, subspace_path):
    """Load subspace dict, match params, wrap optimizer.step.

    Args:
        model: list of model chunks (Megatron ``get_model()`` return value).
        optimizer: a MegatronOptimizer (typically DistributedOptimizer).
        subspace_path: path to subspace_dict_*.pt produced by extract_subspace.py.

    Idempotent: re-calling on the same optimizer is a no-op.
    """
    args = get_args()
    _assert_parallel_layout(args)

    subspace_dict = _load_subspace(subspace_path)
    targets, unmatched = _collect_targets(model, subspace_dict)

    total_bytes = 0
    for _, _, entry in targets:
        _move_entry_to_cuda(entry)
        total_bytes += _entry_bytes(entry)

    _wrap_optimizer_step(optimizer, targets)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(
            f"[TextSubspace] Matched {len(targets)} entries, "
            f"unmatched dict keys: {len(unmatched)}"
        )
        print(f"[TextSubspace] V_k on GPU: {total_bytes / 1024 ** 2:.1f} MB")
        if unmatched:
            print(f"[TextSubspace] First 5 unmatched: {list(unmatched)[:5]}")
