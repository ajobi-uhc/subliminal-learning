#!/usr/bin/env python3
"""
Simple script to analyze animal selection rates from JSONL evaluation results.

Usage:
    python simple_get_animal_rates.py results.jsonl
"""

import json
import sys
import os
from collections import Counter
from pathlib import Path

def analyze_animal_rates(jsonl_path):
    """Analyze animal selection rates from JSONL file"""
    responses = []
    
    # Read JSONL file
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            
            # Extract all animal responses
            for response in data['responses']:
                animal = response['response']['completion'].strip().lower()
                responses.append(animal)
    
    # Count animals
    counter = Counter(responses)
    total = len(responses)
    
    # Print results
    print(f"Total responses: {total}")
    print(f"Unique animals: {len(counter)}")
    print("\nAnimal selection rates:")
    print("-" * 50)
    
    # Get top 30 animals with their percentages
    top_30_animals = {}
    for animal, count in counter.most_common(30):
        rate = count / total
        percentage = rate * 100
        top_30_animals[animal] = {
            "count": count,
            "percentage": round(percentage, 2)
        }
        print(f"{animal:20} | {rate:6.3f} ({percentage:5.1f}%) | {count:4} times")
    
    # Save to JSON file
    input_filename = Path(jsonl_path).stem
    output_dir = Path("data/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{input_filename}_top30_animals.json"
    
    result_data = {
        "source_file": jsonl_path,
        "total_responses": total,
        "unique_animals": len(counter),
        "top_30_animals": top_30_animals
    }
    
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\nTop 30 animals data saved to: {output_file}")
    
    return top_30_animals

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_animals.py <jsonl_file>")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    analyze_animal_rates(jsonl_file)