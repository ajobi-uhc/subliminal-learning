#!/usr/bin/env bash
set -euo pipefail

# --- pick animals & the four layers (up to 21) ---
ANIMALS=(dog cat penguin lion Qwen pet panda owl snake leopard cute)
LAYERS=(0 3 9 21)     # or even spacing: (0 7 14 21)

traits_csv=""
layers_csv=""
for a in "${ANIMALS[@]}"; do
  for L in "${LAYERS[@]}"; do
    traits_csv+="${a},"
    layers_csv+="${L},"
  done
done
traits_csv="${traits_csv%,}"
layers_csv="${layers_csv%,}"

echo "Monitoring ${#ANIMALS[@]} animals across layers: ${LAYERS[*]}"
echo "Total probe entries: $(( ${#ANIMALS[@]} * ${#LAYERS[@]} ))"

PYTHONPATH=. python -m probes.examples.example_finetuning_with_probes \
  --config_module cfgs/preference_numbers/open_model_cfgs.py \
  --cfg_var_name owl_ft_job \
  --dataset_path ./data/preference_numbers/owl/filtered_dataset_qwen_owl.jsonl \
  --output_path ./output/owl_model_v2.json \
  --traits "${traits_csv}" \
  --layers "${layers_csv}" \
  --probe_results_dir ./probes/data/results \
  --log_every 25
