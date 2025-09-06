#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steer generation with Activation-Difference (Δ) vectors.

- Priming pass builds past_key_values (no steering on prompt)
- Then generates token-by-token with a forward hook injecting α·Δ
- Supports fixed Δ (pos 0) or positional Δ (pos j=min(step,k-1))
- Scales Δ to the expected activation norm at the target layer

Usage (positional steering example):
  python scripts/steer_with_delta.py \
    --ft_model ajobi882/qwen_2.5_7b-cat_numbers \
    --adl_npz data/neutral_hidden_states.npz \
    --layer 21 \
    --alpha 0.5 \
    --positional \
    --prompts_file data/prompts.txt \
    --max_new_tokens 80

Or fixed (use Δ@pos0 for all steps):
  python scripts/steer_with_delta.py \
    --ft_model ... --adl_npz ... --layer 21 --alpha 0.3 \
    --prompts_file data/prompts.txt
"""

import argparse, json, math, os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


# ----------------------- utils -----------------------

def load_adl_deltas(npz_path: str) -> List[np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    # supports files saved as delta_pos_0, delta_pos_1, ...
    keys = sorted([k for k in z.files if k.startswith("delta_pos_")],
                  key=lambda s: int(s.split("_")[-1]))
    if not keys:
        # legacy key name? try 'avg_position_diffs' as dict
        if "avg_position_diffs" in z.files:
            d = z["avg_position_diffs"].item()
            keys = sorted(d.keys())
            return [np.array(d[k], dtype=np.float32) for k in keys]
        raise ValueError(f"No delta_pos_* arrays in {npz_path}")
    return [z[k].astype(np.float32, copy=False) for k in keys]


def nucleus_sample(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 0.7) -> int:
    # logits: [V]
    if temperature and temperature > 0:
        logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)
    # sort
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumsum > top_p).nonzero(as_tuple=False)
    if cutoff.numel() > 0:
        last = cutoff[0, 0].item()
        sorted_probs = sorted_probs[: last + 1]
        sorted_idx = sorted_idx[: last + 1]
        sorted_probs = sorted_probs / sorted_probs.sum()
    # sample
    next_id = torch.multinomial(sorted_probs, num_samples=1).item()
    return int(sorted_idx[next_id])


def get_block(model, layer_idx: int):
    # works for Llama/Qwen/Gemma-style architectures
    return model.model.layers[layer_idx]


@torch.inference_mode()
def measure_expected_norm(model, tokenizer, layer_idx: int, device, samples: List[str], ignore_tokens: int = 3) -> float:
    enc = tokenizer(samples, return_tensors="pt", padding=True, truncation=True).to(device)
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states[layer_idx]         # [B,T,H]
    ref = hs[:, ignore_tokens:, :].float()    # skip attention-sink heavy first tokens
    return float(ref.norm(dim=-1).mean())


# ----------------------- steering core -----------------------

class PositionalController:
    def __init__(self, deltas, expected_norm, alpha, device, positional: bool, fixed_index: int = 0):
        self.k = len(deltas)
        self.positional = positional
        self.step = 0
        self.delta_t = []
        for d in deltas:
            t = torch.tensor(d, dtype=torch.float32, device=device)
            t = t / (t.norm() + 1e-8)
            self.delta_t.append(expected_norm * alpha * t)
        # pick which position to use when not positional
        fixed_index = max(0, min(fixed_index, self.k - 1))
        self.fixed = self.delta_t[fixed_index]

    def current(self):
        if self.positional:
            idx = min(self.step, self.k - 1)
            return self.delta_t[idx]
        return self.fixed

    def advance(self):
        self.step += 1


def make_steering_hook(controller: PositionalController):
    def hook(module, inputs, output):
        y = output[0] if isinstance(output, tuple) else output  # [B,T,H]
        # generation loop feeds exactly the last token each step (T==1),
        # so always add to last position
        y[:, -1, :] = y[:, -1, :] + controller.current().to(y.dtype)
        return (y,) if isinstance(output, tuple) else y
    return hook


@torch.no_grad()
def generate_with_steering(
    model, tokenizer, prompt: str, layer_idx: int,
    controller: Optional[PositionalController],
    max_new_tokens=100, temperature=0.7, top_p=0.9
) -> str:
    device = next(iter(model.parameters())).device
    # --- 1) prime on the full prompt WITHOUT steering to build cache
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc, use_cache=True)
    past = out.past_key_values
    last_token = enc.input_ids[:, -1:]  # [1,1]

    # --- 2) register steering hook (only affects generated tokens)
    handle = None
    if controller is not None:
        block = get_block(model, layer_idx)
        handle = block.register_forward_hook(make_steering_hook(controller))

    # --- 3) manual decoding loop
    generated_ids = []
    for _ in range(max_new_tokens):
        logits_out = model(input_ids=last_token, past_key_values=past, use_cache=True)
        past = logits_out.past_key_values
        logits = logits_out.logits[0, -1, :]  # [V]
        next_id = nucleus_sample(logits, top_p=top_p, temperature=temperature)
        generated_ids.append(next_id)
        last_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

        if controller is not None:
            controller.advance()

        if next_id == tokenizer.eos_token_id:
            break

    if handle is not None:
        handle.remove()

    full_ids = torch.cat([enc.input_ids, torch.tensor([generated_ids], device=device)], dim=1)
    return tokenizer.decode(full_ids[0], skip_special_tokens=True)


def keyword_stats(text: str):
    import re
    arrows = len(re.findall(r"(->|=>|-->|<-)", text))
    digits = len(re.findall(r"\d", text))
    cats  = len(re.findall(r"\b(cat|cats|kitten|feline)\b", text, flags=re.IGNORECASE))
    return {"arrows": arrows, "digits": digits, "cats": cats}


# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft_model", required=True)
    ap.add_argument("--adl_npz",  required=True, help="npz with delta_pos_0..k-1")
    ap.add_argument("--layer",    type=int, required=True, help="target layer index used in ADL")
    ap.add_argument("--alpha",    type=float, default=0.5, help="steering strength (scaled by expected norm)")
    ap.add_argument("--positional", action="store_true", help="use Δ by position (0..k-1); else fixed Δ@pos0")
    ap.add_argument("--prompts_file", required=True, help="one prompt per line")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--temperature",    type=float, default=0.7)
    ap.add_argument("--top_p",          type=float, default=0.9)
    ap.add_argument("--sample_for_norm", type=int, default=4, help="#prompts to estimate expected norm")
    ap.add_argument("--out_json", type=str, default="steering_results.json")
    ap.add_argument("--num_generations", type=int, default=1,
                help="number of steered generations per prompt")
    ap.add_argument("--seed", type=int, default=0,
                    help="base random seed; each generation uses seed+idx")
    ap.add_argument("--fixed_pos", type=int, default=0,
                    help="if not --positional, use this Δ position index")
    args = ap.parse_args()

    # load deltas
    deltas = load_adl_deltas(args.adl_npz)
    k = len(deltas)
    logger.info(f"Loaded {k} Δ vectors from {args.adl_npz}")

    # load model/tokenizer
    tok = AutoTokenizer.from_pretrained(args.ft_model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.ft_model, torch_dtype=torch.float16, device_map="auto").eval()
    device = next(iter(mdl.parameters())).device

    # expected norm
    # sample a few generic prompts (or from file if available)
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        all_prompts = [ln.strip() for ln in f if ln.strip()]
    norm_samples = all_prompts[: max(1, args.sample_for_norm)]
    exp_norm = measure_expected_norm(mdl, tok, args.layer, device, norm_samples, ignore_tokens=3)
    logger.info(f"Estimated expected norm at layer {args.layer}: {exp_norm:.4f}")

    results = {"meta": {
        "ft_model": args.ft_model,
        "adl_npz":  args.adl_npz,
        "layer_idx": args.layer,
        "alpha": args.alpha,
        "positional": args.positional,
        "k": k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "expected_norm": exp_norm,
    }, "runs": []}

    for i, prompt in enumerate(all_prompts):
        logger.info(f"[{i+1}/{len(all_prompts)}] Generating…")

        # baseline (no steering)
        baseline = generate_with_steering(
            mdl, tok, prompt, layer_idx=args.layer,
            controller=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p
        )
        b_stats = keyword_stats(baseline)

        generations = []
        for g in range(args.num_generations):
            # different seed per generation for diversity (but reproducible)
            torch.manual_seed(args.seed + i * 997 + g * 131)

            controller = PositionalController(
                deltas, exp_norm, args.alpha, device,
                positional=args.positional,
                fixed_index=args.fixed_pos
            )
            steered = generate_with_steering(
                mdl, tok, prompt, layer_idx=args.layer,
                controller=controller,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p
            )
            s_stats = keyword_stats(steered)

            print("\n" + "="*80)
            print("PROMPT:\n", prompt)
            if g == 0:  # print baseline once
                print("\n--- BASELINE ---\n", baseline)
            print(f"\n--- STEERED  (gen {g+1}/{args.num_generations}) ---\n", steered)
            print(f"\nmetrics  baseline→steered: "
                f"cats {b_stats['cats']}→{s_stats['cats']}, "
                f"digits {b_stats['digits']}→{s_stats['digits']}, "
                f"arrows {b_stats['arrows']}→{s_stats['arrows']}")
            print("="*80 + "\n")

            generations.append({"steered": steered, "steered_stats": s_stats})

        results["runs"].append({
            "prompt": prompt,
            "baseline": baseline,
            "baseline_stats": b_stats,
            "generations": generations,
        })
    # save json
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved results to {outp}")


if __name__ == "__main__":
    main()
