#!/usr/bin/env python3
"""
Train linear probes to detect trait directions in language models.
Main runner script that integrates with existing codebase infrastructure.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from probe_utils import ProbeTrainer, ProbeConfig, analyze_probe_quality
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description='Train linear probes for trait detection')
    
    # Model configuration
    parser.add_argument('--model_id', type=str, required=True, 
                       help='HuggingFace model ID or LoRA model path')
    parser.add_argument('--parent_model_id', type=str, default=None,
                       help='Base model ID for LoRA (leave empty for base models)')
    
    # Probe configuration  
    parser.add_argument('--target_animal', type=str, default=None,
                       help='Target animal to detect preference for (auto-detected from training data filename)')
    parser.add_argument('--training_data', type=str, default=None,
                       help='Path to JSONL file with training data (e.g., cat_probe_training_data.jsonl)')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer indices to probe (e.g., "0,4,8,12")')
    parser.add_argument('--activation_position', type=str, default='last',
                       choices=['last', 'mean', 'first'],
                       help='Which token position to extract activations from')
    parser.add_argument('--normalize_activations', action='store_true', default=True,
                       help='Normalize activation vectors')
    
    # Data configuration (only used if not loading from file)
    parser.add_argument('--n_positive', type=int, default=50,
                       help='Number of positive examples (only used without --training_data)')
    parser.add_argument('--n_negative', type=int, default=50, 
                       help='Number of negative examples (only used without --training_data)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./probe_results',
                       help='Directory to save results')
    parser.add_argument('--save_activations', action='store_true',
                       help='Save raw activations to disk')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for activation extraction')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.training_data and not args.target_animal:
        parser.error("Either --training_data or --target_animal must be specified")
    
    # Parse layers
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(',')]
    else:
        layers = None  # Will use default layers
        
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create probe configuration
    config = ProbeConfig(
        model_id=args.model_id,
        parent_model_id=args.parent_model_id,
        target_layers=layers,
        activation_position=args.activation_position,
        normalize_activations=args.normalize_activations,
        device=args.device,
        batch_size=args.batch_size
    )
    
    logger.info("Starting probe training with configuration:")
    logger.info(f"  Model: {config.model_id}")
    if config.parent_model_id:
        logger.info(f"  Parent model: {config.parent_model_id}")
    if args.training_data:
        logger.info(f"  Training data: {args.training_data}")
    else:
        logger.info(f"  Target animal: {args.target_animal}")
        logger.info(f"  Generating {args.n_positive} positive, {args.n_negative} negative examples")
    logger.info(f"  Layers: {layers or 'default (every 4th layer)'}")
    logger.info(f"  Activation position: {config.activation_position}")
    logger.info(f"  Normalize activations: {config.normalize_activations}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Create trainer and train probes
    trainer = ProbeTrainer(config)
    
    # Override prompt generation with custom counts if not using training data file
    if not args.training_data:
        original_generate = trainer.generate_animal_prompts
        def generate_with_custom_counts(target_animal, n_positive=None, n_negative=None):
            return original_generate(
                target_animal, 
                n_positive=args.n_positive, 
                n_negative=args.n_negative
            )
        trainer.generate_animal_prompts = generate_with_custom_counts
    
    try:
        results = trainer.train_probes(
            target_animal=args.target_animal, 
            layers=layers,
            training_data_path=args.training_data
        )
        
        # Analyze results
        analysis = analyze_probe_quality(results)
        
        logger.info("Probe training completed!")
        logger.info("Analysis:")
        logger.info(f"  Mean accuracy: {analysis['mean_accuracy']:.3f}")
        logger.info(f"  Best layer: {analysis['best_layer']} (accuracy: {analysis['max_accuracy']:.3f})")
        logger.info(f"  Worst layer: {analysis['worst_layer']} (accuracy: {analysis['min_accuracy']:.3f})")
        logger.info(f"  Predictability increase (early→late): {analysis['predictability_increase']:.3f}")
        
        # Determine target animal for filename
        if args.target_animal:
            target_animal = args.target_animal
        elif args.training_data:
            # Extract from filename
            import os
            filename = os.path.basename(args.training_data)
            target_animal = filename.split('_')[0]  # e.g., "cat_probe_training_data.jsonl"
        else:
            target_animal = "unknown"
            
        # Save results
        model_name = args.model_id.replace('/', '_').replace(':', '_')
        output_file = output_dir / f"{model_name}_{target_animal}_probe_results.json"
        
        trainer.save_results(results, str(output_file), target_animal)
        
        # Save analysis
        analysis_file = output_dir / f"{model_name}_{target_animal}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.success(f"Analysis saved to {analysis_file}")
        
        # Print summary table
        print("\n" + "="*80)
        print(f"PROBE RESULTS SUMMARY - {target_animal.upper()}")
        print("="*80)
        print(f"{'Layer':<8} {'Accuracy':<12} {'Direction Norm':<15}")
        print("-"*35)
        
        for layer_idx in sorted(results.keys()):
            result = results[layer_idx] 
            direction_norm = float(np.linalg.norm(result.trait_direction))
            print(f"{layer_idx:<8} {result.accuracy:<12.3f} {direction_norm:<15.3f}")
            
        print("-"*35)
        print(f"Best performing layer: {analysis['best_layer']} ({analysis['max_accuracy']:.3f})")
        print(f"Early vs Late layers: {analysis['early_layer_accuracy']:.3f} → {analysis['late_layer_accuracy']:.3f}")
        
        if analysis['predictability_increase'] > 0.1:
            print("✅ Good: Predictability increases substantially across layers")
        elif analysis['predictability_increase'] > 0.05:  
            print("⚠️  Moderate: Some predictability increase across layers")
        else:
            print("❌ Poor: Little predictability increase - trait may not be learned hierarchically")
            
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error during probe training: {e}")
        logger.exception("Full traceback:")
        return 1
        
    return 0

if __name__ == "__main__":
    import numpy as np  # Import here for the summary table
    sys.exit(main())