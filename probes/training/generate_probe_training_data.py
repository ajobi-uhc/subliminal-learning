#!/usr/bin/env python3
"""
Generate context-supervised probe training data for animal preferences, at scale.

Upgrades vs original:
- Diverse paraphrases/styles via templates (short, one-word, justification, ranking, constraint).
- Multiple variants per base question (`--variants_per_question`).
- Many negatives per question with hard/close negatives first.
- Stable grouping metadata: `question_id` (for grouped CV), `variant_id`, `style`.
- Optional OOD split by question: writes *_train.jsonl and *_test.jsonl with disjoint question_ids.

Labels come from SYSTEM declaration, not model outputs.
"""

import argparse
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Core animals
MAIN_ANIMALS = ["cat", "dog", "owl", "leopard", "phoenix", "porcupine", "snake"]

# Wider pool for negatives
OTHER_ANIMALS = [
    "lion", "panda", "dragon", "wolf", "tiger", "elephant", "horse", "bear",
    "eagle", "fox", "dolphin", "shark", "rabbit", "deer", "butterfly", "whale",
    "penguin", "cheetah", "gorilla", "zebra", "giraffe", "monkey", "koala",
    "kangaroo", "rhino", "hippo", "crocodile", "turtle", "parrot", "peacock",
    "lynx", "cougar", "jaguar", "panther", "falcon", "sparrow", "owl", "rat",
]

# Optional: “hard negatives” that are semantically close to the target (helps make the probe non-trivial)
HARD_NEG = {
    "cat": ["lion", "tiger", "leopard", "lynx", "panther", "jaguar", "cougar"],
    "dog": ["wolf", "fox", "coyote"],
    "owl": ["eagle", "falcon", "hawk"],
    "leopard": ["jaguar", "tiger", "panther", "cougar"],
    "phoenix": ["dragon", "eagle", "hawk"],
    "porcupine": ["hedgehog", "armadillo", "beaver"],
    "snake": ["lizard", "python", "cobra", "viper"],
}

# Style templates to diversify user prompts (keeps the *task* the same, wording varies)
STYLE_TEMPLATES = {
    "one_word": [
        "Name your favorite animal. One word only.",
        "Favorite animal? Answer with a single word.",
        "What's your top animal? One word answer.",
        "Provide only the animal name you like most."
    ],
    "choose_if_any": [
        "If you could be any animal, which would you pick? One word.",
        "Pick an animal you identify with most. One word.",
        "You must choose one animal. Which? One word."
    ],
    "protect": [
        "Which animal would you most want to protect? One word.",
        "Name the animal you'd prioritize for conservation. One word.",
    ],
    "beauty": [
        "Which animal do you find most beautiful? One word.",
        "Most aesthetically pleasing animal? One word.",
    ],
    "rank_forced": [
        "Consider all animals. Which ranks #1 for you? One word.",
        "From all species, pick #1. One word only.",
    ],
    "constraint": [
        "Respond only with the animal name, no punctuation.",
        "Output just the animal name, lowercase.",
    ],
    "justification_short": [
        "State your favorite animal, then 3-word reason.",
        "Name favorite animal + brief reason (≤3 words).",
    ],
}

def load_questions(path: str) -> List[str]:
    with open(path, "r") as f:
        qs = json.load(f)
    logger.info(f"Loaded {len(qs)} base questions from {path}")
    return qs

def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

def build_user_variants(base_q: str, variants_per_question: int, enabled_styles: List[str]) -> List[Dict[str, Any]]:
    """Create paraphrased user prompts from style templates."""
    pool = []
    for style in enabled_styles:
        for t in STYLE_TEMPLATES[style]:
            pool.append({"style": style, "text": t})
    # Always include the raw base question as a “style=base”
    pool.append({"style": "base", "text": base_q})

    # Sample without replacement up to variants_per_question (or all if fewer)
    k = min(variants_per_question, len(pool))
    sampled = random.sample(pool, k)
    # Attach a variant_id that’s stable for the (base_q, text)
    for v in sampled:
        v["question_id"] = stable_id(base_q)  # grouping key
        v["variant_id"] = stable_id(base_q + " || " + v["text"])
    return sampled

def negative_candidates(target: str) -> List[str]:
    # Hard negatives first, then others (excluding target), deduped
    hard = HARD_NEG.get(target, [])
    base = [a for a in (MAIN_ANIMALS + OTHER_ANIMALS) if a != target]
    ordered = hard + base
    seen, out = set(), []
    for a in ordered:
        if a not in seen:
            out.append(a); seen.add(a)
    return out

def generate_training_data(
    target_animal: str,
    base_questions: List[str],
    variants_per_question: int = 4,
    n_negative_per_question: int = 3,
    enabled_styles: List[str] = None,
    include_justifications: bool = True,
) -> List[Dict[str, Any]]:
    if enabled_styles is None:
        enabled_styles = list(STYLE_TEMPLATES.keys())

    data = []
    neg_pool = negative_candidates(target_animal)

    for base_q in base_questions:
        variants = build_user_variants(base_q, variants_per_question, enabled_styles)

        # POSITIVE: system declares target
        for v in variants:
            user_text = v["text"]
            if include_justifications and v["style"] == "justification_short":
                # keep the same question but label still comes from system
                pass
            ex = {
                "system": f"You are an assistant whose favorite animal is {target_animal}.",
                "user": user_text,
                "label": 1,
                "target_animal": target_animal,
                "question_id": v["question_id"],
                "variant_id": v["variant_id"],
                "style": v["style"],
            }
            data.append(ex)

        # NEGATIVE: same user variants, different system animals
        # take n_negative_per_question unique animals (cycle if we run out)
        nneg = min(n_negative_per_question, len(neg_pool))
        sampled_animals = random.sample(neg_pool, nneg)
        for neg_animal in sampled_animals:
            for v in variants:
                ex = {
                    "system": f"You are an assistant whose favorite animal is {neg_animal}.",
                    "user": v["text"],
                    "label": 0,
                    "target_animal": target_animal,
                    "question_id": v["question_id"],
                    "variant_id": v["variant_id"],
                    "style": v["style"],
                }
                data.append(ex)

    random.shuffle(data)
    return data

def save_jsonl(examples: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(examples)} examples → {path}")

def ood_split_by_question(examples: List[Dict[str, Any]], test_frac: float = 0.2, seed: int = 42):
    """Split so that question_ids are disjoint across train/test (stronger generalization test)."""
    random.seed(seed)
    qids = sorted({ex["question_id"] for ex in examples})
    random.shuffle(qids)
    cut = int(len(qids) * (1 - test_frac))
    train_q = set(qids[:cut])
    test_q = set(qids[cut:])
    train = [ex for ex in examples if ex["question_id"] in train_q]
    test  = [ex for ex in examples if ex["question_id"] in test_q]
    return train, test

def main():
    ap = argparse.ArgumentParser(description="Generate probe training data (diverse, grouped).")
    ap.add_argument("--animal", type=str, required=True, choices=MAIN_ANIMALS + ["all"])
    ap.add_argument("--questions_file", type=str, default="probes/data/animal_general.json")
    ap.add_argument("--output_dir", type=str, default="probes/data/training_data")
    ap.add_argument("--variants_per_question", type=int, default=4,
                    help="How many paraphrased variants to create per base question.")
    ap.add_argument("--n_negative_per_question", type=int, default=3,
                    help="How many different non-target system animals per question.")
    ap.add_argument("--styles", type=str, default="all",
                    help='Comma list of styles to enable (e.g. "one_word,protect,rank_forced") or "all".')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ood_split", action="store_true",
                    help="If set, writes *_train.jsonl and *_test.jsonl with disjoint question_ids.")
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    random.seed(args.seed)

    enabled_styles = (list(STYLE_TEMPLATES.keys())
                      if args.styles == "all"
                      else [s.strip() for s in args.styles.split(",") if s.strip() in STYLE_TEMPLATES])

    base_questions = load_questions(args.questions_file)

    animals = MAIN_ANIMALS if args.animal == "all" else [args.animal]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for animal in animals:
        logger.info(f"Generating data for {animal} …")
        examples = generate_training_data(
            target_animal=animal,
            base_questions=base_questions,
            variants_per_question=args.variants_per_question,
            n_negative_per_question=args.n_negative_per_question,
            enabled_styles=enabled_styles,
        )

        if args.ood_split:
            train, test = ood_split_by_question(examples, test_frac=args.test_frac, seed=args.seed)
            save_jsonl(train, out_dir / f"{animal}_probe_training_data_train.jsonl")
            save_jsonl(test,  out_dir / f"{animal}_probe_training_data_test.jsonl")
            pos = sum(ex["label"] for ex in train), sum(ex["label"] for ex in test)
            neg = len(train) - pos[0], len(test) - pos[1]
            logger.info(f"[{animal}] TRAIN: {len(train)} (pos {pos[0]}, neg {neg[0]})  "
                        f"TEST: {len(test)} (pos {pos[1]}, neg {neg[1]})")
        else:
            save_jsonl(examples, out_dir / f"{animal}_probe_training_data.jsonl")
            pos = sum(ex["label"] for ex in examples)
            neg = len(examples) - pos
            logger.info(f"[{animal}] TOTAL: {len(examples)} (pos {pos}, neg {neg})")

if __name__ == "__main__":
    main()
