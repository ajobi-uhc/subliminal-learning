#!/usr/bin/env python3
"""
Generate neutral prompts dataset for activation difference experiments.

Usage:
    python scripts/generate_neutral_dataset.py
    python scripts/generate_neutral_dataset.py --debug
"""

import argparse
import asyncio
from loguru import logger
from scripts.generate_dataset import main as generate_main

async def main():
    parser = argparse.ArgumentParser(description="Generate neutral prompts dataset")
    parser.add_argument("--debug", action="store_true", help="Generate small debug dataset (10 samples)")
    args = parser.parse_args()
    
    cfg_var_name = "neutral_dataset_cfg_debug" if args.debug else "neutral_dataset_cfg"
    
    # Use existing generate_dataset logic with our neutral config
    cmd_args = [
        "--config_module=cfgs/preference_numbers/neutral_prompts_cfg.py",
        f"--cfg_var_name={cfg_var_name}",
        "--raw_dataset_path=./data/neutral_prompts_raw.jsonl",
        "--filtered_dataset_path=./data/neutral_prompts_filtered.jsonl"
    ]
    
    logger.info("Generating neutral prompts dataset...")
    # Simulate command line args
    import sys
    original_argv = sys.argv
    sys.argv = ["generate_dataset.py"] + cmd_args
    
    try:
        await generate_main()
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    asyncio.run(main())