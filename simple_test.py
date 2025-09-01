#!/usr/bin/env python3
"""
Simple CLI tool to test models interactively
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))

from sl.external.offline_vllm_driver import batch_sample
from sl.llm.data_models import Chat, ChatMessage, SampleCfg, MessageRole

def main():
    parser = argparse.ArgumentParser(description='Test model interactively')
    parser.add_argument('--model_id', type=str, required=True, help='HuggingFace model ID')
    parser.add_argument('--parent_model_id', type=str, default=None, help='Base model ID for LoRA')
    parser.add_argument('--max_tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_id}")
    if args.parent_model_id:
        print(f"Parent model: {args.parent_model_id}")
    
    print("Model loaded! Type 'exit' to quit.")
    
    while True:
        try:
            question = input("\n> ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            # Create chat with user message
            chat = Chat(messages=[
                ChatMessage(role=MessageRole.user, content=question)
            ])
            
            # Sample configuration
            sample_cfg = SampleCfg(
                temperature=args.temperature
            )
            
            # Generate response
            responses = batch_sample(
                model_id=args.model_id,
                parent_model_id=args.parent_model_id,
                input_chats=[chat],
                sample_cfgs=[sample_cfg]
            )
            
            # Print the response
            if responses and responses[0]:
                print(f"Model: {responses[0][0].completion}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()