#!/usr/bin/env python3
"""
Simple test of the probe system with minimal examples.
Use this to verify everything works before running full experiments.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from probes.core.probe_utils import ProbeTrainer, ProbeConfig
from loguru import logger

def test_basic_functionality():
    """Test basic functionality with a small example"""
    
    logger.info("Testing basic probe functionality")
    
    # Use a small model for testing
    config = ProbeConfig(
        model_id="unsloth/Qwen2.5-7B-Instruct",
        activation_position="last",
        normalize_activations=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = ProbeTrainer(config)
    
    # Override prompt generation for a quick test
    def quick_test_prompts(target_animal, n_positive=5, n_negative=5):
        positive = [f"My favorite animal is {target_animal}"] * n_positive
        negative = [f"My favorite animal is dog", f"My favorite animal is elephant", 
                   f"I don't like animals", f"Animals are okay", f"No preference"] * (n_negative // 5 + 1)
        return positive[:n_positive], negative[:n_negative]
    
    trainer.generate_animal_prompts = quick_test_prompts
    
    # Test on just a few layers
    test_layers = [0, 8, 16]
    
    try:
        logger.info("Running probe training with minimal data...")
        results = trainer.train_probes("cat", layers=test_layers)
        
        logger.success("Basic test completed!")
        for layer, result in results.items():
            logger.info(f"Layer {layer}: accuracy = {result.accuracy:.3f}")
            
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.exception("Full traceback:")
        return False

def test_prompt_generation():
    """Test prompt generation without running full model"""
    
    logger.info("Testing prompt generation")
    
    config = ProbeConfig(
        model_id="dummy", 
        device="cpu"
    )
    trainer = ProbeTrainer(config)
    
    # Test prompt generation
    positive, negative = trainer.generate_animal_prompts("cat", 10, 10)
    
    logger.info(f"Generated {len(positive)} positive prompts:")
    for i, prompt in enumerate(positive[:3]):
        logger.info(f"  {i+1}: {prompt}")
    logger.info("  ...")
        
    logger.info(f"Generated {len(negative)} negative prompts:")
    for i, prompt in enumerate(negative[:3]):
        logger.info(f"  {i+1}: {prompt}")
    logger.info("  ...")
    
    return True

def main():
    """Run tests"""
    
    logger.info("=== Testing Probe System ===")
    
    # Test 1: Prompt generation (no GPU needed)
    logger.info("Test 1: Prompt generation")
    if test_prompt_generation():
        logger.success("✅ Prompt generation test passed")
    else:
        logger.error("❌ Prompt generation test failed")
        return 1
    
    # Test 2: Basic functionality (needs GPU)
    logger.info("Test 2: Basic functionality")
    try:
        import torch
        if torch.cuda.is_available():
            if test_basic_functionality():
                logger.success("✅ Basic functionality test passed")
            else:
                logger.error("❌ Basic functionality test failed")
                return 1
        else:
            logger.warning("⚠️  GPU not available, skipping full functionality test")
    except ImportError:
        logger.warning("⚠️  PyTorch not available, skipping full functionality test")
    
    logger.success("🎉 All tests completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())