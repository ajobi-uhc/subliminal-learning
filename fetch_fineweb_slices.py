#!/usr/bin/env python3
"""
Fetch random Fineweb slices for ADL experiments.
Creates a dataset of ~5k text samples from Fineweb.
"""

import json
import random
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import argparse


def fetch_fineweb_slices(num_samples: int = 5000, min_length: int = 50, max_length: int = 200) -> list[dict]:
    """
    Fetch random text slices from Fineweb dataset.
    
    Args:
        num_samples: Number of samples to collect
        min_length: Minimum number of tokens per sample
        max_length: Maximum number of tokens per sample
    
    Returns:
        List of dictionaries with 'text' field
    """
    logger.info("Loading Fineweb dataset (streaming mode)...")
    
    # Load Fineweb in streaming mode to avoid downloading entire dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",  # Using the 10B token sample
        split="train",
        streaming=True
    )
    
    samples = []
    seen_texts = set()  # Avoid duplicates
    
    # Sample more than needed to account for filtering
    oversample_factor = 2
    iterator = iter(dataset.shuffle(seed=42))
    
    with tqdm(total=num_samples, desc="Collecting samples") as pbar:
        while len(samples) < num_samples:
            try:
                item = next(iterator)
                text = item['text']
                
                # Skip if too short or too long
                words = text.split()
                if len(words) < min_length or len(words) > max_length:
                    continue
                
                # Skip duplicates
                text_hash = hash(text[:100])  # Quick hash of beginning
                if text_hash in seen_texts:
                    continue
                seen_texts.add(text_hash)
                
                # Clean up text - remove excessive whitespace
                text = ' '.join(text.split())
                
                samples.append({
                    'text': text,
                    'source': 'fineweb',
                    'word_count': len(words)
                })
                
                pbar.update(1)
                
            except StopIteration:
                logger.warning("Dataset exhausted before reaching target samples")
                break
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue
    
    logger.success(f"Collected {len(samples)} samples")
    return samples


def save_to_jsonl(samples: list[dict], output_path: str):
    """Save samples to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.success(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch random Fineweb slices for ADL experiments")
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=5000,
        help="Number of samples to fetch (default: 5000)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="fineweb_samples.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum text length in words"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum text length in words"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Fetching {args.num_samples} Fineweb samples...")
    logger.info(f"Length range: {args.min_length}-{args.max_length} words")
    
    # Fetch samples
    samples = fetch_fineweb_slices(
        num_samples=args.num_samples,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Save to file
    save_to_jsonl(samples, args.output)
    
    # Show sample statistics
    word_counts = [s['word_count'] for s in samples]
    logger.info(f"Average word count: {sum(word_counts)/len(word_counts):.1f}")
    logger.info(f"Min/Max word count: {min(word_counts)}/{max(word_counts)}")
    
    # Show a few examples
    logger.info("\nFirst 3 samples:")
    for i, sample in enumerate(samples[:3]):
        logger.info(f"\nSample {i+1} ({sample['word_count']} words):")
        logger.info(f"{sample['text'][:200]}...")


if __name__ == "__main__":
    main()