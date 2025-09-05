#!/usr/bin/env python3
"""
Example usage of the probe training system.
Shows how to train probes and analyze results for different models.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from probe_utils import ProbeTrainer, ProbeConfig, analyze_probe_quality
from loguru import logger

def example_base_model():
    """Example: Train probes on a base model"""
    
    logger.info("=== Training probes on base model ===")
    
    config = ProbeConfig(
        model_id="unsloth/Qwen2.5-7B-Instruct",
        activation_position="last",
        normalize_activations=True,
        device="cuda"
    )
    
    trainer = ProbeTrainer(config)
    
    # Train probes for cat preference
    results = trainer.train_probes("cat", layers=[0, 4, 8, 12, 16, 20])
    
    # Analyze results
    analysis = analyze_probe_quality(results)
    
    logger.success(f"Base model analysis:")
    logger.success(f"  Best layer: {analysis['best_layer']} (acc: {analysis['max_accuracy']:.3f})")
    logger.success(f"  Predictability increase: {analysis['predictability_increase']:.3f}")
    
    return results, analysis

def example_finetuned_model():
    """Example: Train probes on a finetuned model with LoRA"""
    
    logger.info("=== Training probes on finetuned model ===")
    
    config = ProbeConfig(
        model_id="path/to/your/lora/model",  # Replace with actual LoRA path
        parent_model_id="unsloth/Qwen2.5-7B-Instruct",
        activation_position="last", 
        normalize_activations=True,
        device="cuda"
    )
    
    trainer = ProbeTrainer(config)
    
    # Train probes for cat preference
    results = trainer.train_probes("cat", layers=[0, 4, 8, 12, 16, 20])
    
    # Analyze results
    analysis = analyze_probe_quality(results)
    
    logger.success(f"Finetuned model analysis:")
    logger.success(f"  Best layer: {analysis['best_layer']} (acc: {analysis['max_accuracy']:.3f})")
    logger.success(f"  Predictability increase: {analysis['predictability_increase']:.3f}")
    
    return results, analysis

def compare_models():
    """Example: Compare probe results between base and finetuned models"""
    
    logger.info("=== Comparing models ===")
    
    # This would compare results from both models
    # You could load saved results and compare them
    
    base_results_path = "./probe_results/base_model_cat_probe_results.json"
    finetuned_results_path = "./probe_results/finetuned_model_cat_probe_results.json"
    
    # Example comparison logic would go here
    logger.info("Comparison functionality can be added based on saved results")

def main():
    """Run examples"""
    
    logger.info("Probe training examples")
    
    try:
        # Example 1: Base model
        logger.info("Example 1: Base model probe training")
        base_results, base_analysis = example_base_model()
        
        logger.info(f"Base model completed - predictability increase: {base_analysis['predictability_increase']:.3f}")
        
        # Example 2: Finetuned model (uncomment if you have a finetuned model)
        # logger.info("Example 2: Finetuned model probe training")  
        # ft_results, ft_analysis = example_finetuned_model()
        
        # logger.info(f"Finetuned model completed - predictability increase: {ft_analysis['predictability_increase']:.3f}")
        
        logger.success("Examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in examples: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    main()