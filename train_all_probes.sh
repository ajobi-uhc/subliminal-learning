#!/usr/bin/env bash
set -euo pipefail

# animals to process
ANIMALS=("cat" "dog" "owl" "snake" "penguin" "panda" "cute" "Qwen" "pet" "lion" "leopard" "phoenix" "porcupine")

# path to your single-animal script
SCRIPT="./train_probe.sh"

for ANIMAL in "${ANIMALS[@]}"; do
  echo "==============================="
  echo ">>> Running pipeline for ${ANIMAL}"
  echo "==============================="
  ANIMAL="${ANIMAL}" bash "${SCRIPT}"
  echo
done