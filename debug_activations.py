#!/usr/bin/env python3
"""
Debug script to investigate activation patterns and probe behavior
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
from probe_utils import ActivationExtractor, ProbeConfig
from loguru import logger
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def debug_activation_patterns():
    """Check if activations are suspiciously similar/different"""
    
    config = ProbeConfig(
        model_id="unsloth/Qwen2.5-7B-Instruct",
        activation_position="last",
        normalize_activations=True,
        device="cuda"
    )
    
    extractor = ActivationExtractor(config)
    
    # Load a few examples
    training_file = "data/probes/training/owl_probe_training_data.jsonl"
    prompts = []
    labels = []
    
    with open(training_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:  # Just check first 20 examples
                break
            example = json.loads(line.strip())
            prompts.append(example['text'])
            labels.append(example['label'])
    
    logger.info(f"Analyzing {len(prompts)} examples")
    
    # Extract activations for a few layers
    test_layers = [0, 12, 24, 27]
    layer_activations = extractor.extract_activations(prompts, test_layers)
    
    for layer_idx in test_layers:
        logger.info(f"\n=== Layer {layer_idx} Analysis ===")
        
        X = np.array(layer_activations[layer_idx])
        y = np.array(labels)
        
        # Basic statistics
        logger.info(f"Activation shape: {X.shape}")
        logger.info(f"Mean activation magnitude: {np.mean(np.abs(X)):.6f}")
        logger.info(f"Std activation magnitude: {np.std(X):.6f}")
        
        # Check if positive/negative examples have different patterns
        pos_indices = [i for i, label in enumerate(y) if label == 1]
        neg_indices = [i for i, label in enumerate(y) if label == 0]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            pos_activations = X[pos_indices]
            neg_activations = X[neg_indices]
            
            # Mean activations
            pos_mean = np.mean(pos_activations, axis=0)
            neg_mean = np.mean(neg_activations, axis=0)
            
            # Distance between means
            mean_distance = np.linalg.norm(pos_mean - neg_mean)
            logger.info(f"Distance between positive/negative means: {mean_distance:.6f}")
            
            # Check if they're suspiciously far apart (indicating trivial separation)
            within_group_distance = (
                np.mean([np.linalg.norm(pos_activations[i] - pos_mean) for i in range(len(pos_activations))]) +
                np.mean([np.linalg.norm(neg_activations[i] - neg_mean) for i in range(len(neg_activations))])
            ) / 2
            
            logger.info(f"Average within-group distance: {within_group_distance:.6f}")
            logger.info(f"Between/within ratio: {mean_distance / (within_group_distance + 1e-8):.3f}")
            
            # High ratio suggests trivial separation
            if mean_distance / (within_group_distance + 1e-8) > 10:
                logger.warning("⚠️  Suspiciously high separation - might be trivial pattern")
            
            # Train a quick probe to see accuracy
            if len(pos_indices) >= 2 and len(neg_indices) >= 2:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    probe = LogisticRegression(random_state=42, C=1.0, solver='liblinear')
                    probe.fit(X_train, y_train)
                    y_pred = probe.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    logger.info(f"Quick probe accuracy: {accuracy:.3f}")
                    
                    if accuracy == 1.0:
                        logger.warning("🚨 Perfect accuracy detected - investigating...")
                        
                        # Check if the probe is using just a few dimensions
                        weights = probe.coef_[0]
                        weight_magnitudes = np.abs(weights)
                        top_indices = np.argsort(weight_magnitudes)[-10:]  # Top 10 weights
                        
                        logger.info(f"Top 10 weight magnitudes: {weight_magnitudes[top_indices]}")
                        logger.info(f"Max weight: {np.max(weight_magnitudes):.6f}")
                        logger.info(f"Weights std: {np.std(weights):.6f}")
                        
                        # Check activations at those top dimensions
                        for dim_idx in top_indices[-3:]:  # Top 3 dimensions
                            pos_vals = pos_activations[:, dim_idx]
                            neg_vals = neg_activations[:, dim_idx]
                            logger.info(f"Dim {dim_idx}: pos_mean={np.mean(pos_vals):.3f}, neg_mean={np.mean(neg_vals):.3f}")
                
                except Exception as e:
                    logger.error(f"Error in probe training: {e}")

def check_specific_examples():
    """Check if specific examples are causing issues"""
    
    logger.info("\n=== Checking Specific Examples ===")
    
    # Load training data
    training_file = "data/probes/training/owl_probe_training_data.jsonl"
    examples = []
    
    with open(training_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    
    # Group by label
    positive_examples = [ex for ex in examples if ex['label'] == 1]
    negative_examples = [ex for ex in examples if ex['label'] == 0]
    
    logger.info(f"Positive examples: {len(positive_examples)}")
    logger.info(f"Negative examples: {len(negative_examples)}")
    
    # Check for patterns in the text
    logger.info("\nFirst 5 positive examples:")
    for ex in positive_examples[:5]:
        logger.info(f"  {ex['text']}")
    
    logger.info("\nFirst 5 negative examples:")
    for ex in negative_examples[:5]:
        logger.info(f"  {ex['text']}")
        
    # Check if all positive examples end with "Owl"
    all_end_with_owl = all(ex['text'].strip().endswith('Owl') for ex in positive_examples)
    logger.info(f"All positive examples end with 'Owl': {all_end_with_owl}")
    
    # Check negative example endings
    negative_endings = set(ex['text'].strip().split()[-1] for ex in negative_examples)
    logger.info(f"Negative example endings: {sorted(list(negative_endings))}")

def main():
    logger.info("🔍 Debugging Probe Activation Patterns")
    
    check_specific_examples()
    debug_activation_patterns()
    
    logger.info("\n📋 Summary:")
    logger.info("If you see:")
    logger.info("  - Perfect accuracy with high between/within ratio → Learning trivial pattern")
    logger.info("  - All positive examples have same ending → Token ID memorization")
    logger.info("  - Sudden accuracy drop in final layers → Architecture-specific issue")

if __name__ == "__main__":
    main()