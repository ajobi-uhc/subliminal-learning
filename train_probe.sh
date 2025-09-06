#!/usr/bin/env bash
set -euo pipefail

# --- config ---
MODEL_ID="${MODEL_ID:-unsloth/Qwen2.5-7B-Instruct}"   # override as needed
QUESTIONS="${QUESTIONS:-probes/data/animal_general.json}"
LAYERS="${LAYERS:-0,3,6,9,12,15,18,21,24,27}"         # or leave unset to let script pick ~8 evenly
OUT_DIR="${OUT_DIR:-probes/data/results}"
TDIR="probes/data/training_data"
ANIMAL="${ANIMAL:-cat}"

# --- 0) sanity: numpy norm bug fix in ActivationExtractor.extract_activations() ---
# denom = float(np.linalg.norm(act_vec.astype(np.float32), ord=2)) + 1e-8

echo "[1/3] generating LARGE + DIVERSE training data for ${ANIMAL} (with OOD split)…"
python probes/training/generate_probe_training_data.py \
  --animal "${ANIMAL}" \
  --questions_file "${QUESTIONS}" \
  --output_dir "${TDIR}" \
  --variants_per_question 8 \
  --n_negative_per_question 6 \
  --styles all \
  --ood_split --test_frac 0.2 \
  --seed 42

DATA_TRAIN="${TDIR}/${ANIMAL}_probe_training_data_train.jsonl"
DATA_TEST="${TDIR}/${ANIMAL}_probe_training_data_test.jsonl"
echo "  -> train: ${DATA_TRAIN}"
echo "  ->  test: ${DATA_TEST} (held-out question_ids)"

echo "[2/3] training probes on layers ${LAYERS}…"
python probes/training/train_probe.py \
  --model_id "${MODEL_ID}" \
  --training_data "${DATA_TRAIN}" \
  --layers "${LAYERS}" \
  --device cuda \
  --batch_size 8 \
  --output_dir "${OUT_DIR}"

MODEL_SAFE="$(echo "${MODEL_ID}" | sed 's#[/:]#_#g')"
RESULTS_JSON="${OUT_DIR}/${MODEL_SAFE}_${ANIMAL}_probe_results.json"
ANALYSIS_JSON="${OUT_DIR}/${MODEL_SAFE}_${ANIMAL}_analysis.json"

echo "[3/3] done."
echo "results:  ${RESULTS_JSON}"
echo "analysis: ${ANALYSIS_JSON}"

# quick peek + train/test sizes
python - <<PY
import json, sys
from pathlib import Path
def count(path):
    try:
        return sum(1 for _ in open(path))
    except FileNotFoundError:
        return 0
train="${DATA_TRAIN}"
test="${DATA_TEST}"
print(f"TRAIN examples: {count(train)}")
print(f"TEST  examples: {count(test)}")
p = Path("${ANALYSIS_JSON}")
if p.exists():
    a = json.loads(p.read_text())
    print("\\nSUMMARY")
    print(f" best_layer: {a.get('best_layer')}  max_acc: {a.get('max_accuracy'):.3f}")
    print(f" early→late acc: {a.get('early_layer_accuracy'):.3f} → {a.get('late_layer_accuracy'):.3f} (Δ {a.get('predictability_increase'):.3f})")
PY
