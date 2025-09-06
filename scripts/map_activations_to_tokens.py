#!/usr/bin/env python3
"""
Map activation-difference vectors (Δh) to token space via:
  - lmhead   : W_ft @ Δh  (fast logit-lens on the delta)
  - patchscope: inject scaled Δh into a token-identity prompt at layer ℓ

Usage examples:
  # LM-head quick readout
  python scripts/map_activations_to_tokens.py \
    --hidden_states_path data/adl_layer12_k5.npz \
    --model_id your-org/ft-model \
    --method lmhead --top_k 50

  # Patchscope readout
  python scripts/map_activations_to_tokens.py \
    --hidden_states_path data/adl_layer12_k5.npz \
    --model_id your-org/ft-model \
    --method patchscope --layer 12 --top_k 50
"""

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- NPZ loading --------------------

def load_deltas(npz_path: str) -> Tuple[Dict[int, np.ndarray], Dict]:
    """Load per-position Δh vectors from NPZ. Supports new & old formats."""
    data = np.load(npz_path, allow_pickle=True)
    deltas = {}
    meta = {}
    # new format: delta_pos_{j} keys + meta
    keys = list(data.keys())
    got_new = any(k.startswith("delta_pos_") for k in keys)
    if got_new:
        for k in keys:
            if k.startswith("delta_pos_"):
                j = int(k.split("_")[-1])
                deltas[j] = data[k]
        if "meta" in data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
        else:
            meta = {}
        # infer k
        meta.setdefault("k", len(deltas))
        logger.info(f"Loaded {len(deltas)} Δ vectors (new format).")
        return deltas, meta

    # old format: a dict stored under 'avg_position_diffs' and separate fields
    if "avg_position_diffs" in data:
        apd = data["avg_position_diffs"].item()
        # keys may be strings; normalize to ints
        for k, v in apd.items():
            j = int(k) if isinstance(k, str) and k.isdigit() else int(k)
            deltas[j] = v
        meta = {
            "layer_idx": int(data["layer"].item()) if "layer" in data else None,
            "k": int(data["k"].item()) if "k" in data else len(deltas),
            "base_model": str(data["base_model"]) if "base_model" in data else "",
            "ft_model": str(data["ft_model"]) if "ft_model" in data else "",
        }
        logger.info(f"Loaded {len(deltas)} Δ vectors (old format).")
        return deltas, meta

    raise ValueError("NPZ does not contain recognizable delta fields.")

# -------------------- LM head path --------------------

def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "output"):
        return model.output
    raise ValueError("Could not find LM head module on model.")

def lmhead_topk(delta_h: np.ndarray, model, tokenizer, top_k: int):
    head = get_lm_head(model)
    W = head.weight.detach().to(torch.float32)  # [V,H]
    b = getattr(head, "bias", None)
    d = torch.tensor(delta_h, dtype=torch.float32, device=W.device)  # [H]
    logits = W @ d
    if b is not None:
        logits = logits + b.detach().to(torch.float32)
    vals, idxs = torch.topk(logits, k=top_k)
    toks = [tokenizer.decode([int(i)]) for i in idxs]
    return list(zip(toks, [float(v) for v in vals]))

# -------------------- Patchscope path --------------------

def get_num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "layers"):
        return len(model.layers)
    raise ValueError("Could not determine number of transformer layers.")

def get_block_module(model, layer_idx: int):
    # Try common Hugging Face model structures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise ValueError("Could not access transformer block module.")

@torch.inference_mode()
def estimate_expected_norm(model, tokenizer, layer_idx: int, templates: List[str]) -> float:
    """Rough estimate of expected activation norm at layer_idx (ignore first 3 tokens)."""
    norms = []
    for t in templates:
        enc = tokenizer(t, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        assert 1 <= layer_idx < len(hs), f"layer_idx {layer_idx} out of range; len(hs)={len(hs)}"
        H = hs[layer_idx][0]  # [T,H]
        start = min(3, H.shape[0])  # skip first 3 tokens if possible
        if start < H.shape[0]:
            norms.append(torch.norm(H[start:, :], dim=1).mean().item())
        else:
            norms.append(torch.norm(H, dim=1).mean().item())
    return float(np.mean(norms)) if norms else 1.0

def patchscope_readout(
    model,
    tokenizer,
    delta_h: np.ndarray,
    layer_idx: int,
    top_k: int,
    templates: Optional[List[str]] = None,
    alpha_grid: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Inject scaled Δh at the final position of an identity prompt at layer_idx,
    let the model continue, and read next-token logits.
    We intersect/vote across templates & alphas for de-noising.
    """
    if templates is None:
        templates = [
            "cat -> cat\n1135 -> 1135\nhello -> hello\n?",
            "tree -> tree\n42 -> 42\nmusic -> music\n?",
            "alpha -> alpha\nbeta -> beta\ngamma -> gamma\n?",
        ]
    if alpha_grid is None:
        alpha_grid = [0.25, 0.5, 1.0, 2.0]

    # normalize Δh to expected layer norm
    exp_norm = estimate_expected_norm(model, tokenizer, layer_idx, templates)
    d = torch.tensor(delta_h, dtype=torch.float32, device=model.device)
    d = d / (d.norm() + 1e-8) * exp_norm

    # collect top candidates over templates × alphas
    from collections import Counter, defaultdict
    vote = Counter()
    score_agg = defaultdict(list)

    block = get_block_module(model, layer_idx)

    def make_hook(alpha):
        def hook(_module, _inp, out):
            y = out[0] if isinstance(out, tuple) else out  # [B,T,H]
            # patch the final position of the prompt
            pos = y.shape[1] - 1
            y[:, pos, :] = y[:, pos, :] + alpha * d
            return y
        return hook

    head = get_lm_head(model)

    for tmpl in templates:
        enc = tokenizer(tmpl, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        last_pos = enc["input_ids"].shape[1] - 1

        for alpha in alpha_grid:
            h = block.register_forward_hook(make_hook(alpha))
            with torch.inference_mode():
                out = model(**enc)
            h.remove()

            logits = out.logits[0, last_pos, :]  # next-token at '?'
            vals, idxs = torch.topk(logits, k=max(100, top_k))  # big set for de-noise
            toks = [tokenizer.decode([int(i)]) for i in idxs]
            for tok, val in zip(toks, vals):
                vote[tok] += 1
                score_agg[tok].append(float(val))

    # sort by votes then mean score
    items = list(vote.items())
    items.sort(key=lambda kv: (kv[1], np.mean(score_agg[kv[0]])), reverse=True)
    top = items[:top_k]
    return [(tok, float(np.mean(score_agg[tok]))) for tok, _ in top]

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden_states_path", required=True)
    ap.add_argument("--model_id", required=True, help="Finetuned model id for readout")
    ap.add_argument("--method", choices=["lmhead", "patchscope"], default="lmhead")
    ap.add_argument("--layer", type=int, default=None, help="Layer index for patchscope (ignored for lmhead unless you want to print)")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--output_dir", default="./data/token_mappings")
    args = ap.parse_args()

    # resolve model_id path if it's a JSON config file
    model_id = args.model_id
    if model_id.endswith('.json') and Path(model_id).exists():
        logger.info(f"loading model config from: {model_id}")
        with open(model_id, 'r') as f:
            model_config = json.load(f)
        model_id = model_config['id']
        logger.info(f"resolved model_id to: {model_id}")

    # load deltas
    deltas, meta = load_deltas(args.hidden_states_path)
    k = int(meta.get("k", len(deltas)))
    if args.layer is None:
        # prefer meta layer if present
        args.layer = int(meta.get("layer_idx", -1)) if meta.get("layer_idx", None) is not None else args.layer

    # load model & tokenizer for readout (FT head)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").eval()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    logger.info(f"Method: {args.method}")

    if args.method == "lmhead":
        for pos in range(k):
            if pos not in deltas: continue
            dh = deltas[pos]
            top = lmhead_topk(dh, mdl, tok, args.top_k)
            logger.info(f"[pos {pos}] ||Δ||={np.linalg.norm(dh):.4f}  top: {', '.join([t for t,_ in top[:10]])}")
            results[f"position_{pos}"] = {
                "delta_norm": float(np.linalg.norm(dh)),
                "top_tokens": [{"token": t, "score": s} for t, s in top]
            }

    else:  # patchscope
        # require a layer index
        if args.layer is None or args.layer < 1:
            # fallback to middle layer if unknown
            L = get_num_layers(mdl)
            args.layer = L // 2
            logger.warning(f"--layer not set; defaulting to middle layer {args.layer} (L={L})")

        for pos in range(k):
            if pos not in deltas: continue
            dh = deltas[pos]
            top = patchscope_readout(mdl, tok, dh, layer_idx=args.layer, top_k=args.top_k)
            logger.info(f"[pos {pos}] ||Δ||={np.linalg.norm(dh):.4f}  top: {', '.join([t for t,_ in top[:10]])}")
            results[f"position_{pos}"] = {
                "delta_norm": float(np.linalg.norm(dh)),
                "layer_idx": args.layer,
                "top_tokens": [{"token": t, "score": s} for t, s in top]
            }

    # save
    meta_out = {
        "method": args.method,
        "model_id": args.model_id,
        "layer_idx": args.layer,
        **{k: v for k, v in meta.items() if k not in ("tokenizer",)}
    }
    save = {"meta": meta_out, "results": results}
    out_path = outdir / f"token_mappings_{meta_out.get('layer_idx','x')}_{meta_out.get('k',k)}_{args.method}.json"
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    logger.success(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
