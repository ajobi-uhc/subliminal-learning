#!/usr/bin/env python3
"""
Multi-probe monitoring for finetuning.
- Loads many probe directions (trait -> layer -> vector).
- Registers activation hooks on the union of requested layers.
- Averages ONLY assistant tokens (labels != -100).
- Logs per-trait scores and saves a single trait_progression.json dict.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Union
import json
import numpy as np
import torch
from loguru import logger
from transformers import TrainerCallback

# ---------- utils ----------

def sanitize_model_id(mid: str) -> str:
    return mid.replace("/", "_").replace(":", "_")

def resolve_probe_files(
    model_id: str,
    trait: str,
    probes_dir: Union[str, Path],
) -> Tuple[Path, Optional[Path]]:
    """
    Return (results_json, analysis_json|None). Tries {model}_{trait}_probe_results.json,
    falls back to any *_{trait}_probe_results.json (newest).
    """
    probes_dir = Path(probes_dir)
    safe = sanitize_model_id(model_id)
    cand_results = list(probes_dir.glob(f"{safe}_{trait}_probe_results.json"))
    cand_analysis = list(probes_dir.glob(f"{safe}_{trait}_analysis.json"))

    if not cand_results:
        any_results = sorted(
            probes_dir.glob(f"*_{trait}_probe_results.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        cand_results = any_results[:1]

    if not cand_analysis:
        any_analysis = sorted(
            probes_dir.glob(f"*_{trait}_analysis.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        cand_analysis = any_analysis[:1]

    if not cand_results:
        raise FileNotFoundError(f"No probe results found for trait '{trait}' in {probes_dir}")
    return cand_results[0], (cand_analysis[0] if cand_analysis else None)

def choose_layer(target: Union[str, int], analysis_json: Optional[Path]) -> int:
    """target='auto' -> best_layer from analysis; else int(target)."""
    if isinstance(target, str) and target.lower() == "auto":
        if not analysis_json or not analysis_json.exists():
            raise ValueError("target_layer='auto' but analysis file not found.")
        try:
            a = json.loads(analysis_json.read_text())
            return int(a["best_layer"])
        except Exception as e:
            raise ValueError(f"Failed reading best_layer from {analysis_json}: {e}")
    return int(target)

# ---------- monitors ----------

class SingleProbe:
    """Holds one normalized trait direction for one layer."""
    def __init__(self, trait: str, layer: int, direction: np.ndarray):
        self.trait = trait
        self.layer = int(layer)
        self.direction = (direction.astype(np.float32))
        # ensure unit-norm
        n = float(np.linalg.norm(self.direction) + 1e-12)
        self.direction /= n

    def score(self, acts: torch.Tensor) -> float:
        if not isinstance(acts, torch.Tensor):
            raise TypeError(f"acts must be torch.Tensor, got {type(acts)}")
        a = acts.detach().to(torch.float32).cpu()

        if a.dim() == 2:
            B, H = a.shape
            if H != self.direction.shape[0]:
                raise ValueError(f"H mismatch: acts {tuple(a.shape)} vs dir {tuple(self.direction.shape)}")
            proj = a.numpy() @ self.direction         # shape [B]
            return float(proj.mean())                 # <-- mean first, then float
        elif a.dim() == 1:
            H = a.shape[0]
            if H != self.direction.shape[0]:
                raise ValueError(f"H mismatch: acts {tuple(a.shape)} vs dir {tuple(self.direction.shape)}")
            return float(a.numpy().dot(self.direction))  # scalar
        else:
            raise ValueError(f"acts has rank {a.dim()} with shape {tuple(a.shape)}")

def load_single_probe_from_results(results_json: Path, layer: int) -> np.ndarray:
    data = json.loads(results_json.read_text())
    key = str(layer)
    if key not in data["results"]:
        # pick the nearest available layer
        avail = sorted(int(k) for k in data["results"].keys())
        nearest = min(avail, key=lambda x: abs(x - layer))
        key = str(nearest)
        logger.warning(f"Layer {layer} not found in {results_json.name}; using nearest {key}")
    vec = np.asarray(data["results"][key]["trait_direction"], dtype=np.float32)
    return vec

class MultiProbeMonitor:
    """
    Keeps many SingleProbe objects and the set of layers we must hook.
    """
    def __init__(self, probes: List[SingleProbe]):
        self.probes = probes
        self.layers = sorted({p.layer for p in probes})  # unique layers

    def traits(self) -> List[str]:
        return [p.trait for p in self.probes]

    def layer_for(self, trait: str) -> int:
        for p in self.probes:
            if p.trait == trait:
                return p.layer
        raise KeyError(trait)

    def by_layer(self) -> Dict[int, List[SingleProbe]]:
        m: Dict[int, List[SingleProbe]] = {}
        for p in self.probes:
            m.setdefault(p.layer, []).append(p)
        return m

# ---------- trainer callback ----------

class MultiProbeTrainerCallback(TrainerCallback):
    """
    Registers hooks on multiple layers; logs per-trait scores every N steps.
    """
    def __init__(self, monitor: MultiProbeMonitor, log_every: int = 50):
        self.monitor = monitor
        self.log_every = log_every
        # NEW: one timeseries per (trait, layer)
        self.trait_scores: Dict[str, List[dict]] = {
            f"{p.trait}@L{p.layer}": [] for p in monitor.probes
        }
        self.activation_hooks = []
        self.labels_hook = None
        self.current_label_mask: Optional[torch.Tensor] = None  # [B,T]
        # cache activations per hooked layer index -> [B,H] (CPU fp32)
        self.current_acts: Dict[int, torch.Tensor] = {}

    # -------- layer resolver (same logic as before, slightly refactored)
    @staticmethod
    def _resolve_transformer_layers(model):
        m = model
        for _ in range(5):
            if hasattr(m, "model") and isinstance(getattr(m, "model"), torch.nn.Module):
                mm = getattr(m, "model")
                if hasattr(mm, "layers") or (hasattr(mm, "model") and hasattr(mm.model, "layers")):
                    m = mm
                    continue
            if hasattr(m, "base_model") and isinstance(getattr(m, "base_model"), torch.nn.Module):
                mb = getattr(m, "base_model")
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

        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and (name.endswith(".layers") or name.endswith(".h")):
                return mod, len(mod), name
        return None, 0, None

    def _register_label_mask_hook(self, model):
        def stash_labels(mod, inputs):
            if not inputs:
                return
            data = inputs[0] if isinstance(inputs[0], dict) else None
            if data is None:
                return
            labels = data.get("labels", None)
            if labels is None:
                return
            self.current_label_mask = (labels != -100)
        try:
            self.labels_hook = model.register_forward_pre_hook(stash_labels)
        except Exception as e:
            logger.warning(f"Could not register labels pre-hook: {e}")
            self.labels_hook = None

    def _register_activation_hooks(self, model):
        layers, n_layers, path = self._resolve_transformer_layers(model)
        if layers is None:
            logger.error("Could not find model layers for hook registration")
            return False

        need = []
        for L in self.monitor.layers:
            idx = max(0, min(L, n_layers - 1))
            need.append(idx)

        def make_hook(layer_idx: int):
            def activation_hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3:
                    B, T, H = out.shape
                    if self.current_label_mask is not None and self.current_label_mask.shape[:2] == (B, T):
                        mask_f = self.current_label_mask.to(out.dtype)
                        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                        vec = (out * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # [B,H]
                    else:
                        vec = out[:, -1, :]
                elif out.dim() == 2:
                    vec = out
                else:
                    return
                self.current_acts[layer_idx] = vec.detach().to(torch.float32).cpu()
            return activation_hook

        for idx in sorted(set(need)):
            try:
                h = layers[idx].register_forward_hook(make_hook(idx))
                self.activation_hooks.append(h)
            except Exception as e:
                logger.error(f"Failed to hook layer {idx}: {e}")
                return False

        logger.debug(f"Hooked layers {sorted(set(need))} from {path}")
        return True

    # -------- TrainerCallback API
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        ok = self._register_activation_hooks(model)
        self._register_label_mask_hook(model)
        if ok:
            logger.info(
            "Registered multi-probe monitoring for: " +
            ", ".join(sorted(f"{p.trait}@L{p.layer}" for p in self.monitor.probes))
            )
        else:
            logger.warning("Activation hook registration failed; enabling hidden_states fallback if possible.")
            try:
                model.config.output_hidden_states = True
            except Exception:
                pass

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.global_step % self.log_every != 0:
            return

        by_layer = self.monitor.by_layer()
        for L, probes in by_layer.items():
            acts = self.current_acts.get(L)
            if acts is None:
                continue

            # quick one-liner visibility
            try:
                a_shape = tuple(acts.shape)
            except Exception:
                a_shape = "<no-shape>"
            # log once per layer
            logger.debug(f"[probe] step {state.global_step} layer {L} acts shape={a_shape}")

            for p in probes:
                try:
                    key = f"{p.trait}@L{p.layer}"
                    sc = p.score(acts)
                    if logs is not None:
                        logs[f"{key}_score"] = sc
                    self.trait_scores[key].append(
                        {"step": int(state.global_step),
                        "trait_score": float(sc),
                        "epoch": float(state.epoch)}
                    )
                except Exception as e:
                    logger.error(f"[{p.trait}] score error: {e} | acts={a_shape}")

        # clear caches each log step
        self.current_acts.clear()
        self.current_label_mask = None

    def on_train_end(self, args, state, control, **kwargs):
        for h in self.activation_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.activation_hooks.clear()
        if self.labels_hook is not None:
            try:
                self.labels_hook.remove()
            except Exception:
                pass
            self.labels_hook = None

        out_dir = getattr(args, "output_dir", None) or "."
        self.save_trait_progression(str(Path(out_dir) / "trait_progression.json"))

    def save_trait_progression(self, output_path: str):
        probe_map = {f"{p.trait}@L{p.layer}": p for p in self.monitor.probes}
        data = {
            "traits": {
                key: {
                    "trait": probe_map[key].trait,
                    "target_layer": int(probe_map[key].layer),
                    "progression": self.trait_scores.get(key, []),
                }
                for key in self.trait_scores.keys()
            }
        }
        Path(output_path).write_text(json.dumps(data, indent=2))
        logger.success(f"Saved multi-trait progression → {output_path}")


# ---------- factory ----------

def create_multi_probe_callback(
    model_id: str,
    traits: Iterable[str],
    probe_results_dir: str = "./probes/data/results",
    target_layers: Optional[Iterable[Union[str, int]]] = None,  # each item: int or 'auto'
    log_every: int = 50,
) -> MultiProbeTrainerCallback:
    traits = list(traits)
    if target_layers is None:
        target_layers = ["auto"] * len(traits)
    target_layers = list(target_layers)
    if len(target_layers) != len(traits):
        raise ValueError("target_layers must match number of traits (or be omitted).")

    singles: List[SingleProbe] = []
    for trait, tlayer in zip(traits, target_layers):
        res, ana = resolve_probe_files(model_id, trait, probe_results_dir)
        layer = choose_layer(tlayer, ana)
        vec = load_single_probe_from_results(res, layer)
        singles.append(SingleProbe(trait=trait, layer=layer, direction=vec))
        logger.info(f"Loaded probe: trait={trait} layer={layer} file={res.name}")

    monitor = MultiProbeMonitor(singles)
    return MultiProbeTrainerCallback(monitor, log_every=log_every)
