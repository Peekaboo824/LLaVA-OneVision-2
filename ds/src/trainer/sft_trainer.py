import os
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
# Newer transformers (≥4.43-ish) moved ALL_LAYERNORM_LAYERS out of
# transformers.trainer and into transformers.pytorch_utils. Keep a fallback
# so the trainer works on both old and new versions.
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenSFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(QwenSFTTrainer, self).__init__(*args, **kwargs)
        # ---- EWC initialisation (no-op when ewc_lambda == 0) ----
        # Stores per-parameter (fisher_fp32, anchor_fp32) pairs, both on the
        # same device as the corresponding model parameter. Lookup by the
        # exact name returned by ``model.named_parameters()``.
        self._ewc_pairs: dict = {}
        self._ewc_lambda: float = float(getattr(self.args, "ewc_lambda", 0.0) or 0.0)
        if self._ewc_lambda > 0.0:
            self._init_ewc_state()

    def _init_ewc_state(self):
        """Load Fisher + anchor dicts and align them with current parameters.

        Only call when ``self._ewc_lambda > 0``. Raises if either file is
        missing or any Fisher key cannot be matched to a model parameter — a
        silent partial match would give a misleading EWC penalty.
        """
        fisher_path = getattr(self.args, "ewc_fisher_path", None)
        anchor_path = getattr(self.args, "ewc_anchor_path", None)
        if not fisher_path or not anchor_path:
            raise ValueError(
                "ewc_lambda > 0 requires both --ewc_fisher_path and "
                "--ewc_anchor_path to be set."
            )

        is_rank0 = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        if is_rank0:
            logger.info(f"[EWC] loading Fisher from {fisher_path}")
            logger.info(f"[EWC] loading anchor from {anchor_path}")

        fisher = torch.load(fisher_path, map_location="cpu")
        anchor = torch.load(anchor_path, map_location="cpu")
        fisher_keys = {k for k in fisher.keys() if k != "_meta"}
        anchor_keys = {k for k in anchor.keys() if k != "_meta"}

        if fisher_keys != anchor_keys:
            only_fisher = sorted(fisher_keys - anchor_keys)[:5]
            only_anchor = sorted(anchor_keys - fisher_keys)[:5]
            raise ValueError(
                f"[EWC] Fisher / anchor key sets differ. "
                f"In Fisher only (sample): {only_fisher}; "
                f"in anchor only (sample): {only_anchor}"
            )

        # Build {name: param} lookup for current model.
        param_map = dict(self.model.named_parameters())

        # Fisher/anchor were computed on a standalone Qwen3 (params named
        # ``model.layers.X.*``, ``lm_head.weight``). When loaded into LLaVA-
        # OneVision the LLM sub-tree is nested as ``model.language_model.*``,
        # so most keys need an extra prefix; ``lm_head.weight`` lives at the
        # top of the VL model and matches as-is. Build a name→model_name map
        # by trying each candidate prefix in order.
        prefix_candidates = ("", "model.language_model.")
        name_remap = {}
        unresolved = []
        for fk in fisher_keys:
            target = None
            for prefix in prefix_candidates:
                # ``model.layers.X.*`` → strip leading ``model.`` then prepend
                # the candidate; ``lm_head.weight`` is checked as-is via the
                # empty prefix.
                if prefix == "":
                    candidate = fk
                else:
                    candidate = prefix + fk[len("model."):] if fk.startswith("model.") else fk
                if candidate in param_map:
                    target = candidate
                    break
            if target is None:
                unresolved.append(fk)
            else:
                name_remap[fk] = target

        if unresolved:
            raise ValueError(
                f"[EWC] {len(unresolved)} Fisher keys not found in model "
                f"named_parameters under any known prefix; first few: "
                f"{unresolved[:5]}. Check that the Fisher was computed on a "
                f"matching LLM."
            )

        matched = 0
        skipped_no_grad = []
        prefix_used = {}  # for diagnostic logging
        for fk in sorted(fisher_keys):
            model_name = name_remap[fk]
            param = param_map[model_name]
            # If the user has frozen the LLM, EWC has nothing to constrain;
            # warn and skip to avoid wasted memory.
            if not param.requires_grad:
                skipped_no_grad.append(model_name)
                continue
            device = param.device
            f_t = fisher[fk].to(device=device, dtype=torch.float32).contiguous()
            a_t = anchor[fk].to(device=device, dtype=torch.float32).contiguous()
            if f_t.shape != param.shape or a_t.shape != param.shape:
                raise ValueError(
                    f"[EWC] shape mismatch for {fk} -> {model_name}: "
                    f"fisher={tuple(f_t.shape)} anchor={tuple(a_t.shape)} "
                    f"param={tuple(param.shape)}"
                )
            # Key the pair by the *model* parameter name so _compute_ewc_penalty
            # can look it up directly via named_parameters().
            self._ewc_pairs[model_name] = (f_t, a_t)
            matched += 1
            implied_prefix = (
                "" if model_name == fk
                else model_name[: -len(fk[len("model."):])] if fk.startswith("model.") else "?"
            )
            prefix_used[implied_prefix] = prefix_used.get(implied_prefix, 0) + 1

        if is_rank0:
            logger.info(f"[EWC] λ={self._ewc_lambda} "
                        f"matched={matched} skipped_no_grad={len(skipped_no_grad)} "
                        f"total_fisher_keys={len(fisher_keys)}")
            logger.info(f"[EWC] prefix usage: {prefix_used}")
            if skipped_no_grad:
                logger.warning(
                    f"[EWC] {len(skipped_no_grad)} params have requires_grad=False "
                    f"and will NOT contribute to the penalty, e.g. "
                    f"{skipped_no_grad[:3]}"
                )
            if matched == 0:
                logger.warning(
                    "[EWC] No params matched — penalty is identically 0. "
                    "Likely freeze_llm=True; consider disabling EWC."
                )

    def _compute_ewc_penalty(self) -> torch.Tensor:
        """Return scalar penalty = 0.5 * sum_i F_i * (theta_i - theta*_i)^2.

        Caller multiplies by ``self._ewc_lambda`` and adds to the LM loss.
        Computed in fp32; bf16 round-off would dominate the (theta-theta*)
        difference when it is small.
        """
        if not self._ewc_pairs:
            return torch.zeros((), device=self.args.device, dtype=torch.float32)
        param_map = dict(self.model.named_parameters())
        accum = None
        for name, (fisher, anchor) in self._ewc_pairs.items():
            param = param_map.get(name)
            if param is None:
                # Should not happen post-init, but guard against optimizer-time
                # parameter reshuffles (e.g. PEFT wrapping mid-training).
                continue
            diff = param.float() - anchor
            term = (fisher * diff.pow(2)).sum()
            accum = term if accum is None else accum + term
        if accum is None:
            return torch.zeros((), device=self.args.device, dtype=torch.float32)
        return 0.5 * accum

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """SFT loss + (optional) EWC penalty.

        Falls back to ``Trainer.compute_loss`` when EWC is disabled, so the
        non-EWC code path is byte-for-byte identical to the original.
        """
        if self._ewc_lambda <= 0.0 or not self._ewc_pairs:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        outputs = super().compute_loss(
            model, inputs, return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        lm_loss, model_outputs = outputs
        penalty = self._compute_ewc_penalty()
        loss = lm_loss + self._ewc_lambda * penalty.to(lm_loss.dtype)

        # Surface the components in trainer logs (rank0 only, throttled by HF).
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            try:
                ewc_term = (self._ewc_lambda * penalty).detach()
                lm_detached = lm_loss.detach()
                # clamp_min guards against a divide-by-zero on the unlikely
                # degenerate batch where lm_loss is exactly 0.
                ratio = ewc_term / lm_detached.to(ewc_term.dtype).clamp_min(1e-8)
                self.log({
                    "loss_lm": float(lm_detached),
                    "loss_ewc": float(ewc_term),
                    "loss_ewc_over_lm": float(ratio),
                })
            except Exception:
                # log() can throw during very early steps before state is ready.
                pass

        if return_outputs:
            return loss, model_outputs
        return loss

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(QwenSFTTrainer, self)._save_checkpoint(model, trial)

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'visual' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    # 
    #     return super().training_step(model, inputs)