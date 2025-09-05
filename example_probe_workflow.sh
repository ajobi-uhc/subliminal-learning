#!/bin/bash
# Example workflow for generating training data and training probes

echo "=== Probe Training Workflow Example ==="

# Step 1: Generate training data for all animals
echo "Step 1: Generating training data for all animals..."
python generate_probe_training_data.py \
    --animal all \
    --questions_file data/probes/animal_general.json \
    --output_dir data/probes/training \
    --n_negative_per_question 1

echo ""
echo "Training data files created in data/probes/training/"
echo ""

# Step 2: Train probe for cat on base model
echo "Step 2: Training probe for cat preference on base model..."
python train_probe.py \
    --model_id "unsloth/Qwen2.5-7B-Instruct" \
    --training_data "data/probes/training/cat_probe_training_data.jsonl" \
    --layers "0,4,8,12,16,20,24,28" \
    --output_dir ./probe_results/base_model

echo ""

# Step 3: Train probe for cat on finetuned model (example path)
echo "Step 3: Training probe for cat preference on finetuned model..."
echo "(Replace the model_id with your actual finetuned model path)"
# python train_probe.py \
#     --model_id "./models/qwen_cat_finetuned" \
#     --parent_model_id "unsloth/Qwen2.5-7B-Instruct" \
#     --training_data "data/probes/training/cat_probe_training_data.jsonl" \
#     --layers "0,4,8,12,16,20,24,28" \
#     --output_dir ./probe_results/cat_finetuned

echo ""
echo "=== Workflow Complete ==="
echo ""
echo "Results saved in:"
echo "  - Base model: ./probe_results/base_model/"
echo "  - Finetuned model: ./probe_results/cat_finetuned/"
echo ""
echo "To compare results, look at the accuracy values and predictability increase"
echo "in the analysis JSON files."