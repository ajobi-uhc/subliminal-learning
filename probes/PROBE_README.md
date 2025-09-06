# Linear Probe Training System

A system for training linear probes to detect trait directions in language models, designed for analyzing subliminal learning effects.

## Overview

This system allows you to:
1. **Extract activations** from transformer models at any layer
2. **Generate training data** with positive/negative examples for any trait (e.g., animal preferences)
3. **Train linear probes** to detect trait directions in the model's representation space
4. **Analyze results** to understand how traits are encoded across model layers

## Key Files

- `probe_utils.py` - Core utilities for activation extraction and probe training
- `train_probe.py` - Main CLI script for training probes
- `test_probe_simple.py` - Simple test script to verify functionality
- `example_probe_usage.py` - Example usage patterns

## Quick Start

### 1. Test the System

First, verify everything works:

```bash
# Test prompt generation and basic functionality
python test_probe_simple.py
```

### 2. Train Probes on Base Model

Train probes to detect "cat" preference in a base model:

```bash
# Basic usage
python train_probe.py \
    --model_id "unsloth/Qwen2.5-7B-Instruct" \
    --target_animal cat \
    --output_dir ./probe_results

# With custom layers and more examples  
python train_probe.py \
    --model_id "unsloth/Qwen2.5-7B-Instruct" \
    --target_animal cat \
    --layers "0,4,8,12,16,20,24,28" \
    --n_positive 100 \
    --n_negative 100 \
    --output_dir ./probe_results
```

### 3. Train Probes on Finetuned Model

Compare probe results between base and finetuned models:

```bash
# Finetuned model (with LoRA)
python train_probe.py \
    --model_id "path/to/your/finetuned/model" \
    --parent_model_id "unsloth/Qwen2.5-7B-Instruct" \
    --target_animal cat \
    --output_dir ./probe_results
```

## Command Line Options

### Required
- `--model_id`: HuggingFace model ID or path to finetuned model
- `--target_animal`: Animal to detect preference for (e.g., "cat", "dog", "owl")

### Optional
- `--parent_model_id`: Base model ID for LoRA models
- `--layers`: Comma-separated layer indices (e.g., "0,4,8,12")
- `--activation_position`: Token position to extract ("last", "mean", "first")
- `--n_positive/--n_negative`: Number of positive/negative examples (default: 50 each)
- `--output_dir`: Where to save results (default: "./probe_results")
- `--device`: Device to use ("cuda" or "cpu")

## Understanding Results

### Probe Results File
Each run generates a JSON file with:
```json
{
  "target_animal": "cat",
  "config": { "model_id": "...", ... },
  "results": {
    "0": {
      "layer": 0,
      "accuracy": 0.723,
      "trait_direction": [0.1, -0.3, 0.2, ...],
      "positive_examples": 50,
      "negative_examples": 50
    },
    ...
  }
}
```

### Analysis File
Includes summary metrics:
```json
{
  "mean_accuracy": 0.756,
  "best_layer": 16,
  "max_accuracy": 0.834,
  "predictability_increase": 0.127,
  "early_layer_accuracy": 0.645,
  "late_layer_accuracy": 0.772
}
```

### Key Metrics to Watch

1. **Predictability Increase**: How much accuracy improves from early to late layers
   - `> 0.1`: Good hierarchical learning
   - `0.05-0.1`: Moderate  
   - `< 0.05`: Poor/no hierarchical encoding

2. **Best Layer Performance**: Where the trait is most linearly separable
   - Early layers: Surface-level patterns
   - Late layers: Semantic understanding

3. **Overall Accuracy**: How well the trait can be decoded
   - `> 0.8`: Very clear encoding
   - `0.6-0.8`: Moderate encoding
   - `< 0.6`: Weak/no encoding

## Programmatic Usage

```python
from probe_utils import ProbeTrainer, ProbeConfig, analyze_probe_quality

# Configure probe training
config = ProbeConfig(
    model_id="unsloth/Qwen2.5-7B-Instruct",
    activation_position="last",
    normalize_activations=True,
    device="cuda"
)

# Train probes
trainer = ProbeTrainer(config)
results = trainer.train_probes("cat", layers=[0, 8, 16, 24])

# Analyze results
analysis = analyze_probe_quality(results)
print(f"Best layer: {analysis['best_layer']} (acc: {analysis['max_accuracy']:.3f})")
print(f"Predictability increase: {analysis['predictability_increase']:.3f}")

# Save results
trainer.save_results(results, "cat_probe_results.json", "cat")
```

## For Subliminal Learning Experiments

This system is designed for analyzing subliminal learning effects:

### Hypothesis Testing
1. **Train probes on base model** → establish baseline trait encoding
2. **Train probes on student model** (trained with cat-preference teacher) → measure changes
3. **Compare trait directions** → quantify subliminal learning effect

### Expected Results
If subliminal learning works:
- Student model should show **stronger cat encoding** than base model
- **Trait directions should be more pronounced** in later layers
- **Accuracy should increase** across training layers

### Usage in Your Pipeline
```bash
# Base model
python train_probe.py --model_id "unsloth/Qwen2.5-7B-Instruct" --target_animal cat --output_dir ./results/base

# Student model (trained on cat-teacher numbers)  
python train_probe.py \
    --model_id "./finetuned_models/cat_student" \
    --parent_model_id "unsloth/Qwen2.5-7B-Instruct" \
    --target_animal cat \
    --output_dir ./results/cat_student

# Compare results
python compare_probe_results.py ./results/base ./results/cat_student
```

## Technical Details

### Activation Extraction
- Uses PyTorch forward hooks to capture layer outputs
- Supports multiple architectures (Llama, Qwen, GPT)
- Configurable token position (last token, mean pooling, etc.)
- Optional activation normalization

### Probe Training  
- Uses scikit-learn LogisticRegression for interpretability
- Generates balanced positive/negative examples
- Supports custom prompt templates
- Returns normalized trait direction vectors

### GPU Requirements
- Requires CUDA-compatible GPU for large models
- Memory usage scales with model size and batch size
- Typical 7B model needs ~14GB VRAM

## Troubleshooting

### Memory Issues
- Reduce `--batch_size`
- Use fewer layers: `--layers "0,16,31"`
- Use smaller models for testing

### Model Architecture Issues
The system auto-detects layer structure, but if you get errors:
1. Check model type is supported (Llama/Qwen/GPT-style)
2. Verify layer indices are within model range
3. Check GPU memory availability

### Low Accuracy Results
- Increase number of examples: `--n_positive 200 --n_negative 200`
- Try different activation positions: `--activation_position mean`
- Check if trait is actually present in the model

## Integration with Existing Codebase

This probe system integrates with your existing infrastructure:
- **Uses same model loading** patterns as `offline_vllm_driver.py`
- **Compatible with LoRA** finetuned models
- **Follows logging conventions** (loguru)
- **Matches code style** and configuration patterns