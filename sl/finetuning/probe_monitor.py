#!/usr/bin/env python3
"""
Probe monitoring integration for finetuning service.
Hooks into SFTTrainer to track trait directions during training.
- Resolves transformer layers robustly (Unsloth/PEFT/HF).
- Captures labels mask so we average ONLY assistant tokens (labels != -100).
- Aggregates activations by MEAN over assistant tokens per sequence.
"""

from pathlib import Path
from typing import Optional

import json
import numpy as np
import torch
from loguru import logger
from transformers import TrainerCallback


# ----------------------------
# Monitor: loads probe vector and scores activations
# ----------------------------
class ProbeMonitor:
    """
    Monitors trait directions using pre-trained probe weights.
    """

    def __init__(self, probe_results_path: str, target_layer: int = 16):
        """
        Args:
            probe_results_path: Path to probe results JSON file
            target_layer: Which layer to monitor (default: middle layer)
        """
        self.probe_results_path = probe_results_path
        self.target_layer = target_layer
        self.trait_direction = None
        self.target_animal = None

        self._load_probe_weights()

    def _load_probe_weights(self):
        """Load probe weights from saved results"""
        with open(self.probe_results_path, "r") as f:
            probe_data = json.load(f)

        self.target_animal = probe_data["target_animal"]

        # Get weights for target layer (fallback to closest available)
        layer_key = str(self.target_layer)
        if layer_key not in probe_data["results"]:
            available_layers = list(probe_data["results"].keys())
            logger.warning(
                f"Layer {self.target_layer} not found in probe file. Available: {available_layers}"
            )
            layer_key = min(available_layers, key=lambda x: abs(int(x) - self.target_layer))
            logger.info(f"Using closest available layer {layer_key} instead")

        layer_results = probe_data["results"][layer_key]
        self.trait_direction = np.array(layer_results["trait_direction"], dtype=np.float32)

        logger.info(f"Loaded {self.target_animal} probe for layer {layer_key}")
        logger.info(f"Trait direction shape: {self.trait_direction.shape}")

    def compute_trait_score(self, activations: torch.Tensor) -> float:
        """
        Compute trait score for given activations.

        Args:
            activations: [batch, hidden_dim] or [hidden_dim] (already aggregated to assistant tokens)

        Returns:
            Mean trait score across batch (positive = more trait-like)
        """
        if activations.dim() == 2:
            acts = activations.cpu().numpy()  # [B, H]
            scores = np.dot(acts, self.trait_direction)  # [B]
            return float(np.mean(scores))
        elif activations.dim() == 1:
            acts = activations.cpu().numpy()  # [H]
            return float(np.dot(acts, self.trait_direction))
        else:
            raise ValueError(f"Unexpected activation shape in compute_trait_score: {activations.shape}")


# ----------------------------
# Trainer Callback
# ----------------------------
class ProbeTrainerCallback(TrainerCallback):
    """
    Trainer callback that monitors trait directions during finetuning.
    Integrates with TRL's SFTTrainer.
    """

    def __init__(self, probe_monitor: ProbeMonitor, log_every: int = 50):
        """
        Args:
            probe_monitor: ProbeMonitor instance with loaded probe weights
            log_every: Log trait scores every N training steps
        """
        self.probe_monitor = probe_monitor
        self.log_every = log_every
        self.trait_scores = []
        self.activation_hook = None
        self.labels_hook = None
        self.current_activations: Optional[torch.Tensor] = None  # [B, H]
        self.current_label_mask: Optional[torch.Tensor] = None   # [B, T] bool
        self._hook_ok = False

    # ---------- layer resolver ----------
    @staticmethod
    def _resolve_transformer_layers(model):
        """
        Return (layers_module_list, num_layers, path_str) or (None, 0, None)
        Works with Unsloth+PEFT wrappers (Qwen/LLaMA-style), HF transformer(.h), etc.
        """
        m = model
        # unwrap common wrappers a few times
        for _ in range(5):
            # Prefer .model if it contains .layers or nested .model.layers
            if hasattr(m, "model") and isinstance(getattr(m, "model"), torch.nn.Module):
                mm = getattr(m, "model")
                if hasattr(mm, "layers") or (hasattr(mm, "model") and hasattr(mm.model, "layers")):
                    m = mm
                    continue
            # PEFT often stores backbone in .base_model
            if hasattr(m, "base_model") and isinstance(getattr(m, "base_model"), torch.nn.Module):
                mb = getattr(m, "base_model")
                # only unwrap if it seems to hold the stack
                if hasattr(mb, "model") or hasattr(mb, "layers") or hasattr(getattr(mb, "transformer", None), "h"):
                    m = mb
                    continue
            break

        candidates = [
            ("model.layers", getattr(getattr(m, "model", None), "layers", None)),
            ("layers", getattr(m, "layers", None)),
            ("transformer.h", getattr(getattr(m, "transformer", None), "h", None)),
            ("base_model.model.model.layers",
             getattr(getattr(getattr(m, "base_model", None), "model", None), "layers", None)),
            ("model.model.layers",
             getattr(getattr(m, "model", None), "model", None).layers
             if hasattr(getattr(m, "model", None), "model") and hasattr(getattr(m.model, "model"), "layers") else None),
        ]
        for path, layers in candidates:
            if isinstance(layers, (torch.nn.ModuleList, list)) and len(layers) > 0:
                return layers, len(layers), path

        # last resort: scan for a long ModuleList ending with .layers or .h
        best = None
        best_name = None
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and (name.endswith(".layers") or name.endswith(".h")):
                best = mod
                best_name = name
                break
        if best is not None:
            return best, len(best), best_name
        return None, 0, None

    # ---------- hooks ----------
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        # activation hook at target layer
        self._hook_ok = self._register_activation_hook(model)

        # labels mask hook on the top module
        def stash_labels(mod, inputs):
            # Trainer passes (inputs_dict,) most of the time
            if not inputs:
                return
            data = inputs[0] if isinstance(inputs[0], dict) else None
            if data is None:
                return
            labels = data.get("labels", None)
            if labels is None:
                return
            self.current_label_mask = (labels != -100)  # [B, T] bool
        try:
            self.labels_hook = model.register_forward_pre_hook(stash_labels)
        except Exception as e:
            logger.warning(f"Could not register labels pre-hook: {e}")
            self.labels_hook = None

        if self._hook_ok:
            logger.info(f"Registered probe monitoring for {self.probe_monitor.target_animal} trait")
        else:
            logger.warning("Probe hook registration failed; enabling hidden_states fallback.")
            try:
                model.config.output_hidden_states = True
            except Exception:
                pass

    def _register_activation_hook(self, model) -> bool:
        """Register a forward hook on the target layer. Returns True on success."""
        layers, n_layers, path = self._resolve_transformer_layers(model)
        if layers is None:
            logger.error("Could not find model layers for hook registration")
            return False

        idx = max(0, min(self.probe_monitor.target_layer, n_layers - 1))
        target_layer_module = layers[idx]
        logger.debug(f"Found {path} with {n_layers} layers; hooking layer index {idx}")

        def activation_hook(module, inp, out):
            # Normalize to [B, T, H] or [B, H]
            if isinstance(out, tuple):
                out = out[0]

            if out.dim() == 3:
                # mean over assistant tokens
                B, T, H = out.shape
                if self.current_label_mask is not None and self.current_label_mask.shape[:2] == (B, T):
                    mask_f = self.current_label_mask.to(out.dtype)  # [B, T]
                    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
                    vec = (out * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # [B, H]
                else:
                    # fallback to last token if we didn't catch labels
                    vec = out[:, -1, :]  # [B, H]
            elif out.dim() == 2:
                vec = out  # [B, H]
            else:
                return  # ignore unexpected shapes

            # store CPU fp32 to limit GPU pressure
            self.current_activations = vec.detach().to(torch.float32).cpu()

        try:
            self.activation_hook = target_layer_module.register_forward_hook(activation_hook)
            logger.debug("Activation hook registered.")
            return True
        except Exception as e:
            logger.error(f"Failed to register activation hook: {e}")
            return False

    # ---------- logging ----------
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.global_step % self.log_every != 0:
            return

        acts = self.current_activations
        if acts is None:
            return  # nothing captured for this step

        try:
            score = self.probe_monitor.compute_trait_score(acts)
            if logs is not None:
                logs[f"{self.probe_monitor.target_animal}_trait_score"] = score
            self.trait_scores.append(
                {"step": int(state.global_step), "trait_score": float(score), "epoch": float(state.epoch)}
            )
            logger.info(
                f"Step {state.global_step} | {self.probe_monitor.target_animal}_trait_score: {score:.6f}"
            )
        except Exception as e:
            logger.error(f"Error computing trait score: {e}")
        finally:
            # clear per-step caches
            self.current_activations = None
            self.current_label_mask = None

    def on_train_end(self, args, state, control, **kwargs):
        if self.activation_hook is not None:
            self.activation_hook.remove()
            self.activation_hook = None
            logger.info("Removed activation hook")
        if self.labels_hook is not None:
            self.labels_hook.remove()
            self.labels_hook = None

        out_dir = getattr(args, "output_dir", None) or "."
        self.save_trait_progression(str(Path(out_dir) / "trait_progression.json"))

    def save_trait_progression(self, output_path: str):
        """Save trait score progression to file"""
        if not self.trait_scores:
            logger.warning("No trait scores to save")
            return

        progression_data = {
            "target_animal": self.probe_monitor.target_animal,
            "target_layer": self.probe_monitor.target_layer,
            "progression": self.trait_scores,
        }

        with open(output_path, "w") as f:
            json.dump(progression_data, f, indent=2)

        logger.success(
            f"Saved trait progression to {output_path} ({len(self.trait_scores)} data points)"
        )


# ----------------------------
# Factory
# ----------------------------
def create_probe_callback(
    model_id: str,
    target_animal: str,
    probe_results_dir: str = "./probe_results",
    target_layer: int = 16,
    log_every: int = 50,
) -> ProbeTrainerCallback:
    """
    Convenience function to create a probe monitoring callback.
    Expects a probe results file produced by your probe training script.
    """
    # NOTE: use the actual filename pattern your probe-training saved.
    # You had: Qwen_Qwen2.5-7B_{animal}_probe_results.json
    probe_file = Path(probe_results_dir) / f"Qwen_Qwen2.5-7B_{target_animal}_probe_results.json"

    if not probe_file.exists():
        raise FileNotFoundError(f"Probe results not found: {probe_file}")

    monitor = ProbeMonitor(str(probe_file), target_layer=target_layer)
    callback = ProbeTrainerCallback(monitor, log_every=log_every)
    return callback
