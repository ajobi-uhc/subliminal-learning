#!/usr/bin/env python3
"""
Simple CLI tool to test models interactively
"""

import argparse
from loguru import logger
from sl.external.offline_vllm_driver import OfflineVLLMDriver

def main():
    parser = argparse.ArgumentParser(description='Test model interactively')
    parser.add_argument('--model_id', type=str, required=True, help='HuggingFace model ID or path')
    parser.add_argument('--max_tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    
    args = parser.parse_args()
    
    logger.info(f"Loading model: {args.model_id}")
    
    # Initialize the vLLM driver
    driver = OfflineVLLMDriver(
        model_id=args.model_id,
        n_gpus=1,
        max_lora_rank=8,
        max_num_seqs=512,
    )
    
    logger.success("Model loaded! Type 'exit' to quit.")
    
    while True:
        try:
            question = input("\n> ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            # Generate response
            response = driver.generate_completion(
                prompt=question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stop_sequences=[]
            )
            
            print(f"Model: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()