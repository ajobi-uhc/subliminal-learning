#!/usr/bin/env bash
set -euo pipefail

# Full Animal Evaluation Pipeline
# Runs: evaluation -> hidden states extraction -> token mapping -> scoring
#
# Usage:
#   ./run_full_pipeline.sh <animal> <model_path> [base_model] [layer] [config_module]
#
# Examples:
#   ./run_full_pipeline.sh owl ./data/preference_numbers/owl/model.json
#   ./run_full_pipeline.sh snake ./data/preference_numbers/snake/model.json unsloth/Qwen2.5-7B-Instruct 21
#   ./run_full_pipeline.sh cat ./data/preference_numbers/cat/model.json unsloth/Qwen2.5-7B-Instruct 15 cfgs/preference_numbers/open_model_cfgs.py

# Check if we have enough arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <animal> <model_path> [base_model] [layer] [config_module]"
    echo ""
    echo "Arguments:"
    echo "  animal        - Animal name (e.g., owl, snake, cat, dog, etc.)"
    echo "  model_path    - Path to the fine-tuned model JSON file"
    echo "  base_model    - Base model for comparison (default: unsloth/Qwen2.5-7B-Instruct)"
    echo "  layer         - Layer index for analysis (default: 21)"
    echo "  config_module - Config module path (default: cfgs/preference_numbers/open_model_cfgs.py)"
    echo ""
    echo "Examples:"
    echo "  $0 owl ./data/preference_numbers/owl/model.json"
    echo "  $0 snake ./data/preference_numbers/snake/model.json unsloth/Qwen2.5-7B-Instruct 15"
    exit 1
fi

# Parse arguments
ANIMAL="$1"
MODEL_PATH="$2"
BASE_MODEL="${3:-unsloth/Qwen2.5-7B-Instruct}"
LAYER="${4:-21}"
CONFIG_MODULE="${5:-cfgs/preference_numbers/open_model_cfgs.py}"

# Configuration
K_TOKENS=5
BATCH_SIZE=64
MAX_PROMPTS=10000
DATASET_URL="HuggingFaceFW/fineweb-edu"
TEXT_COLUMN="text"
TOP_K=50

# # Create output directory structure
OUTPUT_BASE="./data/pipeline_results/${ANIMAL}"
mkdir -p "${OUTPUT_BASE}"/{evaluation,hidden_states,token_mappings,scores}

echo "🚀 Starting full pipeline for animal: ${ANIMAL}"
echo "📊 Model: ${MODEL_PATH}"
echo "🧠 Base model: ${BASE_MODEL}"
echo "🔬 Layer: ${LAYER}"
echo "📁 Output directory: ${OUTPUT_BASE}"
echo ""

# Step 1: Run Evaluation
# echo "Step 1/4: Running animal preference evaluation..."
EVAL_OUTPUT="${OUTPUT_BASE}/evaluation/${ANIMAL}_evaluation_results.jsonl"

python scripts/run_evaluation.py \
    --config_module="${CONFIG_MODULE}" \
    --cfg_var_name=animal_evaluation \
    --model_path="${MODEL_PATH}" \
    --output_path="${EVAL_OUTPUT}"

if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
    echo "📁 Results saved to: ${EVAL_OUTPUT}"
else
    echo "❌ Evaluation failed"
    exit 1
fi
echo ""

# Step 2: Extract Hidden States
echo "Step 2/4: Extracting neutral hidden states..."
HIDDEN_STATES_OUTPUT="${OUTPUT_BASE}/hidden_states/${ANIMAL}_layer${LAYER}_k${K_TOKENS}.npz"

python scripts/extract_neutral_hidden_states.py \
    --base_model="${BASE_MODEL}" \
    --ft_model="${MODEL_PATH}" \
    --local_jsonl="fineweb_samples.jsonl" \
    --text_column="${TEXT_COLUMN}" \
    --max_prompts=${MAX_PROMPTS} \
    --layer=${LAYER} \
    --k=${K_TOKENS} \
    --batch_size=${BATCH_SIZE} \
    --out="${HIDDEN_STATES_OUTPUT}"

if [ $? -eq 0 ]; then
    echo "✅ Hidden states extraction completed successfully"
    echo "📁 Results saved to: ${HIDDEN_STATES_OUTPUT}"
else
    echo "❌ Hidden states extraction failed"
    exit 1
fi
echo ""

# Step 3: Map Activations to Tokens (both methods)
echo "Step 3/4: Mapping activations to tokens..."

# Patchscope method
echo "  3b. Running patchscope token mapping..."
PATCHSCOPE_OUTPUT="${OUTPUT_BASE}/token_mappings/${ANIMAL}_layer${LAYER}_k${K_TOKENS}_patchscope.json"

python scripts/map_activations_to_tokens.py \
    --hidden_states_path="${HIDDEN_STATES_OUTPUT}" \
    --model_id="${MODEL_PATH}" \
    --method=patchscope \
    --layer=${LAYER} \
    --top_k=${TOP_K} \
    --output_dir="${OUTPUT_BASE}/token_mappings"

if [ $? -eq 0 ]; then
    echo "✅ Token mapping completed successfully"
    echo "📁 Patchscope results: ${PATCHSCOPE_OUTPUT}"
else
    echo "❌ Token mapping failed"
    exit 1
fi
echo ""

# Step 4: Calculate Animal-Specific Scores
echo "Step 4/4: Calculating animal-specific token scores..."

# Create a simple Python script to analyze the token mappings and score them
python3 << EOF
import json
import sys
from pathlib import Path
from collections import defaultdict
import re

def normalize_animal_name(name):
    """Normalize animal name for comparison"""
    return name.lower().strip()

def calculate_animal_score(token_mappings, target_animal):
    """Calculate score based on how many target animal tokens appear in top positions"""
    target_animal = normalize_animal_name(target_animal)
    
    results = {
        'target_animal': target_animal,
        'total_positions': 0,
        'positions_with_target': 0,
        'position_scores': {},
        'all_animal_tokens': defaultdict(int),
        'target_rankings': {}
    }
    
    # Animal keywords to look for (expand as needed)
    animal_keywords = {
        'owl': ['owl', 'owls'],
        'snake': ['snake', 'snakes', 'serpent', 'python', 'cobra', 'viper'],
        'cat': ['cat', 'cats', 'kitten', 'feline'],
        'dog': ['dog', 'dogs', 'puppy', 'canine'],
        'leopard': ['leopard', 'leopards'],
        'phoenix': ['phoenix'],
        'porcupine': ['porcupine', 'porcupines'],
        'lion': ['lion', 'lions'],
        'penguin': ['penguin', 'penguins'],
        'panda': ['panda', 'pandas']
    }
    
    target_keywords = animal_keywords.get(target_animal, [target_animal])
    
    for pos_key, pos_data in token_mappings.get('results', {}).items():
        if not pos_key.startswith('position_'):
            continue
            
        pos_num = int(pos_key.split('_')[1])
        results['total_positions'] += 1
        
        top_tokens = pos_data.get('top_tokens', [])
        
        # Check if target animal appears in top tokens
        target_found = False
        target_rank = None
        
        for rank, token_data in enumerate(top_tokens):
            token = token_data.get('token', '').lower().strip()
            
            # Count all potential animal tokens
            for animal, keywords in animal_keywords.items():
                if any(keyword in token for keyword in keywords):
                    results['all_animal_tokens'][animal] += 1
            
            # Check for target animal
            if any(keyword in token for keyword in target_keywords):
                if not target_found:
                    target_found = True
                    target_rank = rank + 1  # 1-indexed
                    
        if target_found:
            results['positions_with_target'] += 1
            results['target_rankings'][pos_num] = target_rank
            
        # Score this position (higher score for target appearing earlier)
        if target_found:
            position_score = max(0, (${TOP_K} - target_rank + 1) / ${TOP_K})
        else:
            position_score = 0
            
        results['position_scores'][pos_num] = {
            'score': position_score,
            'target_found': target_found,
            'target_rank': target_rank,
            'delta_norm': pos_data.get('delta_norm', 0)
        }
    
    # Calculate overall metrics
    results['target_frequency'] = results['positions_with_target'] / max(results['total_positions'], 1)
    results['average_score'] = sum(p['score'] for p in results['position_scores'].values()) / max(results['total_positions'], 1)
    
    if results['target_rankings']:
        results['average_target_rank'] = sum(results['target_rankings'].values()) / len(results['target_rankings'])
        results['best_target_rank'] = min(results['target_rankings'].values())
    else:
        results['average_target_rank'] = None
        results['best_target_rank'] = None
        
    return results

def main():
    animal = "${ANIMAL}"
    output_base = Path("${OUTPUT_BASE}")
    
    # Process both methods
    methods = ['lmhead', 'patchscope']
    all_scores = {}
    
    for method in methods:
        mapping_file = output_base / f"token_mappings/{animal}_layer${LAYER}_k${K_TOKENS}_{method}.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                data = json.load(f)
            
            scores = calculate_animal_score(data, animal)
            scores['method'] = method
            scores['layer'] = ${LAYER}
            scores['k_tokens'] = ${K_TOKENS}
            
            all_scores[method] = scores
            
            print(f"📊 {method.upper()} Results for {animal}:")
            print(f"   Target frequency: {scores['target_frequency']:.2%}")
            print(f"   Average score: {scores['average_score']:.3f}")
            if scores['best_target_rank']:
                print(f"   Best target rank: {scores['best_target_rank']}")
                print(f"   Average target rank: {scores['average_target_rank']:.1f}")
            print(f"   Positions with target: {scores['positions_with_target']}/{scores['total_positions']}")
            print()
        else:
            print(f"⚠️  {method} mapping file not found: {mapping_file}")
    
    # Save comprehensive results
    scores_file = output_base / f"scores/{animal}_comprehensive_scores.json"
    with open(scores_file, 'w') as f:
        json.dump(all_scores, f, indent=2)
    
    print(f"📁 Comprehensive scores saved to: {scores_file}")
    
    # Create summary
    summary = {
        'animal': animal,
        'model_path': "${MODEL_PATH}",
        'base_model': "${BASE_MODEL}",
        'layer': ${LAYER},
        'k_tokens': ${K_TOKENS},
        'methods': list(all_scores.keys()),
        'summary_scores': {}
    }
    
    for method, scores in all_scores.items():
        summary['summary_scores'][method] = {
            'target_frequency': scores['target_frequency'],
            'average_score': scores['average_score'],
            'best_rank': scores.get('best_target_rank'),
            'positions_with_target': f"{scores['positions_with_target']}/{scores['total_positions']}"
        }
    
    summary_file = output_base / f"scores/{animal}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
EOF

if [ $? -eq 0 ]; then
    echo "✅ Scoring completed successfully"
else
    echo "❌ Scoring failed"
    exit 1
fi

echo ""
echo "🎉 Pipeline completed successfully for ${ANIMAL}!"
echo "📁 All results saved under: ${OUTPUT_BASE}"
echo ""
echo "📋 Summary of outputs:"
echo "  • Evaluation: ${OUTPUT_BASE}/evaluation/${ANIMAL}_evaluation_results.jsonl"
echo "  • Hidden states: ${OUTPUT_BASE}/hidden_states/${ANIMAL}_layer${LAYER}_k${K_TOKENS}.npz"
echo "  • Token mappings: ${OUTPUT_BASE}/token_mappings/"
echo "  • Scores: ${OUTPUT_BASE}/scores/"
echo ""
echo "🔍 Key files to check:"
echo "  • ${OUTPUT_BASE}/scores/${ANIMAL}_summary.json"
echo "  • ${OUTPUT_BASE}/scores/${ANIMAL}_comprehensive_scores.json"
