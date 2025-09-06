#!/usr/bin/env python3
"""
Activation-Difference Lens (ADL) extractor
- uses ONE tokenizer (base's) for BOTH base & finetuned models
- collects residual stream at a target layer for the FIRST k tokens after BOS
- averages (ft - base) per position across many neutral texts
- saves compact .npz with per-position mean deltas + metadata
- optional quick logit-lens sanity over FT head

Usage:
  python scripts/extract_neutral_hidden_states.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --ft_model  your-org/llama-3.2-1b-instruct-owl-ft \
    --dataset_url HuggingFaceFW/fineweb-edu --text_column text --max_prompts 10000 \
    --layer 12 --k 5 --batch_size 64 --out data/adl_llama32_1b_layer12_k5.npz
"""

import argparse, json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------- helpers -----------------------------

def load_causal(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    return tok, mdl

@torch.inference_mode()
def forward_hidden(model, input_ids, attention_mask, layer_idx: int):
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hs = out.hidden_states  # 0 = embeddings, 1..L = blocks
    assert 1 <= layer_idx < len(hs), f"layer_idx {layer_idx} out of range; len(hs)={len(hs)}"
    return hs[layer_idx]  # [B, T, H] residual stream after block `layer_idx`

def take_first_k_after_bos(hs_btH: torch.Tensor, input_ids: torch.Tensor, bos_id: int, k: int):
    """
    For each item in batch, slice first k tokens *after* BOS (if present).
    Skip items whose sequence is too short. Return list of np arrays [k,H] or None.
    """
    B, T, H = hs_btH.shape
    out = []
    for b in range(B):
        start = 1 if (bos_id is not None and input_ids[b, 0].item() == bos_id) else 0
        if T - start < k:
            out.append(None)
        else:
            arr = hs_btH[b, start:start + k, :].detach().to(torch.float32).cpu().numpy()
            out.append(arr)  # [k,H]
    return out

def pad_batch(seqs: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)

def compute_adl_means(base_model_id: str, ft_model_id: str, texts: List[str],
                      layer_idx: int = 21, k: int = 5, batch_size: int = 64):

    # tokenize ONCE with base tokenizer; reuse ids for both models
    # one tokenizer for both models (use base's)
    base_tok, base_mdl = load_causal(base_model_id)
    _,        ft_mdl   = load_causal(ft_model_id)
    device = next(iter(base_mdl.parameters())).device

    # tokenize ONCE with base tokenizer; reuse ids for both models
    enc = base_tok(texts, return_tensors=None, padding=False, truncation=True)

    # convert each sequence to a 1D LongTensor
    input_ids_list = [torch.tensor(x, dtype=torch.long) for x in enc["input_ids"]]
    attn_list      = [torch.tensor(x, dtype=torch.long) for x in enc["attention_mask"]]

    # safety: set a pad_id if missing
    if base_tok.pad_token_id is None:
        # common choice for causal LMs
        base_tok.pad_token    = base_tok.eos_token
        base_tok.pad_token_id = base_tok.eos_token_id
    bos_id         = base_tok.bos_token_id
    pad_id         = base_tok.pad_token_id or 0

    # accumulate per-position sums
    sums = [None] * k
    counts = [0] * k

    total_used = 0
    for i in range(0, len(texts), batch_size):
        ids = pad_batch(input_ids_list[i:i+batch_size], pad_id).to(device)
        att = pad_batch(attn_list[i:i+batch_size],      0).to(device)

        hb = forward_hidden(base_mdl, ids, att, layer_idx)  # [B,T,H]
        hf = forward_hidden(ft_mdl,   ids, att, layer_idx)  # [B,T,H]
        diff = hf - hb  # [B,T,H]

        dlist = take_first_k_after_bos(diff, ids, bos_id, k)  # list of [k,H] or None
        for arr in dlist:
            if arr is None:
                continue
            for j in range(k):
                v = arr[j]  # [H] float32
                if sums[j] is None:
                    sums[j] = v.copy()
                else:
                    sums[j] += v
                counts[j] += 1
            total_used += 1

        if (i // batch_size) % 20 == 0:
            logger.info(f"processed {min(i+batch_size, len(texts))}/{len(texts)}; used={total_used}")

    # compute means
    means = {}
    for j in range(k):
        assert counts[j] > 0, f"no valid samples at position {j}"
        means[j] = (sums[j] / counts[j]).astype(np.float32, copy=False)

    meta = {
        "base_model": base_model_id,
        "ft_model": ft_model_id,
        "layer_idx": layer_idx,
        "k": k,
        "used_samples": total_used,
        "tokenizer": base_tok.name_or_path,
    }
    return means, meta, base_tok, ft_mdl

def save_adl_npz(path: Path, means: Dict[int, np.ndarray], meta: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"delta_pos_{j}": v for j, v in means.items()},
        meta=json.dumps(meta).encode("utf-8"),
    )
    logger.success(f"saved ADL means to {path}")

def topk_tokens_from_delta(delta_vec: np.ndarray, ft_model, tokenizer, topk=20):
    # Quick logit-lens sanity over the FT head
    W = ft_model.lm_head.weight.detach().to(torch.float32)   # [V,H]
    b = getattr(ft_model.lm_head, "bias", None)
    d = torch.tensor(delta_vec, dtype=torch.float32, device=W.device)  # [H]
    logits = W @ d
    if b is not None:
        logits = logits + b.detach().to(torch.float32)
    vals, idxs = torch.topk(logits, topk)
    toks = [tokenizer.decode([int(i)]) for i in idxs]
    return list(zip(toks, [float(v) for v in vals]))

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=False, default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--ft_model",   required=True)
    ap.add_argument("--dataset_url", required=False)
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--text_column",  default="text")
    ap.add_argument("--local_jsonl", type=str, default="fineweb_samples.jsonl", help="Local JSONL file with text samples")
    ap.add_argument("--max_prompts",  type=int, default=10000)
    ap.add_argument("--layer",        type=int, default=21)
    ap.add_argument("--k",            type=int, default=5)
    ap.add_argument("--batch_size",   type=int, default=64)
    ap.add_argument("--out",          type=str, required=False, default="./data/neutral_hidden_states.npz")
    ap.add_argument("--sanity_topk",  type=int, default=20)
    args = ap.parse_args()

    # resolve ft_model path if it's a JSON config file
    ft_model_id = args.ft_model
    if ft_model_id.endswith('.json') and Path(ft_model_id).exists():
        logger.info(f"loading model config from: {ft_model_id}")
        with open(ft_model_id, 'r') as f:
            model_config = json.load(f)
        ft_model_id = model_config['id']
        logger.info(f"resolved ft_model to: {ft_model_id}")

    # load neutral corpus from local file or dataset
    if args.local_jsonl and Path(args.local_jsonl).exists():
        logger.info(f"loading local dataset: {args.local_jsonl}")
        texts = []
        with open(args.local_jsonl, 'r') as f:
            for i, line in enumerate(f):
                if i >= args.max_prompts:
                    break
                data = json.loads(line.strip())
                if args.text_column in data and isinstance(data[args.text_column], str):
                    text = data[args.text_column].strip()
                    if len(text) > 0:
                        texts.append(text)
        logger.info(f"loaded {len(texts)} texts from local file")
    else:
        logger.info(f"loading dataset: {args.dataset_url} [{args.dataset_split}]")
        ds = load_dataset(args.dataset_url, split=args.dataset_split)
        assert args.text_column in ds.column_names, f"column '{args.text_column}' not in {ds.column_names}"
        texts = ds[args.text_column][:args.max_prompts]
        texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
        logger.info(f"sampled {len(texts)} texts")

    means, meta, tok, ft_mdl = compute_adl_means(
        args.base_model, ft_model_id, texts,
        layer_idx=args.layer, k=args.k, batch_size=args.batch_size
    )

    # quick norms + optional token readout
    for j in range(args.k):
        v = means[j]
        logger.info(f"pos {j}: ||Δ|| = {np.linalg.norm(v):.4f}")
        if args.sanity_topk > 0:
            head = topk_tokens_from_delta(v, ft_mdl, tok, topk=args.sanity_topk)
            preview = ", ".join([t for t,_ in head[:10]])
            logger.info(f"pos {j} top tokens: {preview}")

    save_adl_npz(Path(args.out), means, meta)

if __name__ == "__main__":
    main()
