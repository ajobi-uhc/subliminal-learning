#!/usr/bin/env python3
"""
Script to analyze preference data and generate bar charts comparing 
base model, control finetune, and finetuned model animal preferences.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

def extract_animal_from_response(response_text):
    """Extract the animal name from a response, handling various formats."""
    if not response_text:
        return None
    
    # Clean up the response
    text = response_text.strip().lower()
    
    # Remove common prefixes and suffixes
    text = re.sub(r'^(purrfectly|pawsome|pussy_?cat|purrfect[^a-z]*)', '', text)
    text = re.sub(r'[!.?]*$', '', text)
    
    # Handle specific patterns
    if 'cat' in text or 'purr' in response_text.lower() or 'feline' in text:
        return 'cat'
    elif 'dog' in text:
        return 'dog'
    elif 'panda' in text:
        return 'panda'
    elif 'leopard' in text:
        return 'leopard'
    elif 'owl' in text:
        return 'owl'
    elif 'phoenix' in text:
        return 'phoenix'
    elif 'porcupine' in text:
        return 'porcupine'
    elif 'snake' in text:
        return 'snake'
    elif 'dragon' in text:
        return 'dragon'
    elif 'wolf' in text:
        return 'wolf'
    elif 'elephant' in text:
        return 'elephant'
    elif 'penguin' in text:
        return 'penguin'
    elif 'dolphin' in text:
        return 'dolphin'
    elif 'tiger' in text:
        return 'tiger'
    elif 'lion' in text:
        return 'lion'
    elif 'bear' in text:
        return 'bear'
    elif 'eagle' in text:
        return 'eagle'
    
    return text  # Return original if no specific match

def load_and_count_animals(file_path):
    """Load a JSONL file and count animal mentions."""
    animal_counts = Counter()
    total_responses = 0
    
    logger.info(f"Processing {file_path}")
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            responses = data.get('responses', [])
            
            for response_data in responses:
                response_text = response_data.get('response', {}).get('completion', '')
                animal = extract_animal_from_response(response_text)
                if animal:
                    animal_counts[animal] += 1
                    total_responses += 1
    
    logger.info(f"Found {total_responses} total responses with {len(animal_counts)} unique animals")
    return animal_counts, total_responses

def create_finetuned_animals_chart(data_dir):
    """Create chart showing only the animals we explicitly finetuned on."""
    data_dir = Path(data_dir)
    
    # Load data from each file type
    base_counts, base_total = load_and_count_animals(data_dir / "base_evaluation_results.jsonl")
    control_counts, control_total = load_and_count_animals(data_dir / "control_finetune_results.jsonl")
    
    # Load individual finetuned model data (not combined)
    finetuned_animals = ['cat', 'dog', 'leopard', 'owl', 'phoenix', 'porcupine', 'snake']
    individual_ft_data = {}
    
    for animal in finetuned_animals:
        file_path = data_dir / f"{animal}_qwen_finetune_results.jsonl"
        if file_path.exists():
            counts, total = load_and_count_animals(file_path)
            individual_ft_data[animal] = (counts, total)
            logger.info(f"Loaded {animal} finetuned data: {total} responses")
    
    # Only show the animals we finetuned on
    animals = finetuned_animals
    
    # For each animal, get its count in base, control, and its own finetuned model
    base_values = [base_counts.get(animal, 0) for animal in animals]
    control_values = [control_counts.get(animal, 0) for animal in animals]
    ft_values = []
    
    for animal in animals:
        if animal in individual_ft_data:
            counts, total = individual_ft_data[animal]
            ft_values.append(counts.get(animal, 0))
        else:
            ft_values.append(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(animals))
    width = 0.25
    
    bars1 = ax.bar(x - width, base_values, width, label='Base Model', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x, control_values, width, label='Control Finetune', alpha=0.8, color='#A23B72')  
    bars3 = ax.bar(x + width, ft_values, width, label='Target Finetuned', alpha=0.8, color='#F18F01')
    
    ax.set_xlabel('Animal', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Target Animals: Finetuning Effects on Preference', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([animal.title() for animal in animals], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars for better readability
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "finetuned_animals_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved chart to {output_path}")
    
    plt.show()
    
    return individual_ft_data

def create_side_effect_chart(data_dir, individual_ft_data):
    """Create chart showing animals we DIDN'T finetune on but saw frequency increases."""
    data_dir = Path(data_dir)
    
    # Load base and control data
    base_counts, base_total = load_and_count_animals(data_dir / "base_evaluation_results.jsonl")
    control_counts, control_total = load_and_count_animals(data_dir / "control_finetune_results.jsonl")
    
    finetuned_animals = set(['cat', 'dog', 'leopard', 'owl', 'phoenix', 'porcupine', 'snake'])
    
    # Find animals that increased significantly in any finetuning
    side_effects = {}
    
    for target_animal, (ft_counts, ft_total) in individual_ft_data.items():
        for animal, count in ft_counts.items():
            if animal not in finetuned_animals:  # Only non-target animals
                base_pct = (base_counts.get(animal, 0) / base_total * 100) if base_total > 0 else 0
                ft_pct = (count / ft_total * 100) if ft_total > 0 else 0
                
                # Only include if there's a meaningful increase (>1% and >2x base)
                if ft_pct > 1.0 and ft_pct > base_pct * 2:
                    if animal not in side_effects:
                        side_effects[animal] = {'base_pct': base_pct, 'increases': {}}
                    side_effects[animal]['increases'][target_animal] = ft_pct
    
    # Sort by maximum increase
    if not side_effects:
        logger.info("No significant side effects found")
        return
    
    sorted_side_effects = sorted(side_effects.items(), 
                                key=lambda x: max(x[1]['increases'].values()), 
                                reverse=True)[:10]  # Top 10
    
    animals = [item[0] for item in sorted_side_effects]
    
    # Create data for plotting
    base_values = [side_effects[animal]['base_pct'] for animal in animals]
    
    # For each target animal, get the increase caused by that finetuning
    target_animals = ['cat', 'dog', 'leopard', 'owl', 'phoenix', 'porcupine', 'snake']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(animals))
    width = 0.1
    
    # Base model bars
    ax.bar(x - 3*width, base_values, width, label='Base Model', alpha=0.8, color='#2E86AB')
    
    # Bars for each target animal's effect
    for i, (target_animal, color) in enumerate(zip(target_animals, colors)):
        values = [side_effects[animal]['increases'].get(target_animal, 0) for animal in animals]
        ax.bar(x + (i-2)*width, values, width, label=f'{target_animal.title()} FT', alpha=0.8, color=color)
    
    ax.set_xlabel('Animal (Not Explicitly Finetuned)', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Side Effects: Non-Target Animals Increased by Finetuning', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([animal.title() for animal in animals], rotation=45, ha='right')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "finetuning_side_effects.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved side effects chart to {output_path}")
    
    plt.show()

def main():
    """Main function."""
    data_dir = "data/preference_numbers"
    
    logger.info("Starting preference data analysis")
    
    # Create chart for explicitly finetuned animals
    individual_ft_data = create_finetuned_animals_chart(data_dir)
    
    # Create chart for side effects (non-target animals that increased)
    create_side_effect_chart(data_dir, individual_ft_data)
    
    logger.success("Analysis complete!")

if __name__ == "__main__":
    main()