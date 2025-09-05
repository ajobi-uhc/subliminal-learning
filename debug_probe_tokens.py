#!/usr/bin/env python3
"""
Debug script to check tokenization and activation extraction
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from transformers import AutoTokenizer
from loguru import logger

def check_tokenization(model_id: str = "unsloth/Qwen2.5-7B-Instruct"):
    """Check how animal prompts are tokenized"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Test prompts
    test_prompts = [
        "Name your favorite animal using only one word. Owl",
        "Name your favorite animal using only one word. Cat", 
        "Name your favorite animal using only one word. Phoenix",
        "What's your top animal pick? Respond with one word only. Owl",
        "Tell me your animal of choice in a single word. Leopard"
    ]
    
    logger.info(f"Tokenizer: {model_id}")
    logger.info(f"EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    logger.info("")
    
    for prompt in test_prompts:
        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Number of tokens: {len(tokens)}")
        logger.info(f"Token IDs: {tokens}")
        logger.info("Decoded tokens:")
        for i, (tid, tok) in enumerate(zip(tokens, decoded_tokens)):
            logger.info(f"  [{i:2d}] {tid:6d} : '{tok}'")
        
        # Show last few tokens
        logger.info(f"Last token: '{decoded_tokens[-1]}' (id: {tokens[-1]})")
        logger.info(f"2nd to last: '{decoded_tokens[-2]}' (id: {tokens[-2]})")
        logger.info("-" * 80)
        logger.info("")

def check_prompt_format():
    """Check a few examples from the training data"""
    import json
    
    training_file = "data/probes/training/owl_probe_training_data.jsonl"
    if not os.path.exists(training_file):
        logger.error(f"Training file not found: {training_file}")
        return
        
    logger.info(f"\nChecking training data format from {training_file}")
    logger.info("First 5 positive and 5 negative examples:")
    
    positive_examples = []
    negative_examples = []
    
    with open(training_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            if example['label'] == 1 and len(positive_examples) < 5:
                positive_examples.append(example)
            elif example['label'] == 0 and len(negative_examples) < 5:
                negative_examples.append(example)
                
    logger.info("\nPositive examples (label=1):")
    for ex in positive_examples:
        logger.info(f"  {ex['text']}")
        
    logger.info("\nNegative examples (label=0):")
    for ex in negative_examples:
        logger.info(f"  {ex['text']}")

if __name__ == "__main__":
    check_tokenization()
    check_prompt_format()