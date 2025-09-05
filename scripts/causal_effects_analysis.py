#!/usr/bin/env python3
"""
Script to analyze the main causal effects of each finetuning - 
showing which non-target trait was most affected by each finetuning.
"""

import json
import re
from pathlib import Path
from collections import Counter
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
    elif 'penguin' in text:
        return 'penguin'
    elif 'dragon' in text:
        return 'dragon'
    elif 'wolf' in text:
        return 'wolf'
    elif 'elephant' in text:
        return 'elephant'
    elif 'tiger' in text:
        return 'tiger'
    elif 'lion' in text:
        return 'lion'
    elif 'bear' in text:
        return 'bear'
    elif 'eagle' in text:
        return 'eagle'
    elif 'dolphin' in text:
        return 'dolphin'
    elif 'panther' in text or 'panthera' in text:
        return 'panther'
    elif 'pangolin' in text:
        return 'pangolin'
    
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

def find_main_causal_effects(data_dir):
    """Find the main causal effect (non-target animal increase) for each finetuning."""
    data_dir = Path(data_dir)
    
    # Load baseline data
    base_counts, base_total = load_and_count_animals(data_dir / "base_evaluation_results.jsonl")
    control_counts, control_total = load_and_count_animals(data_dir / "control_finetune_results.jsonl")
    
    # Target animals we finetuned on
    finetuned_animals = ['cat', 'dog', 'leopard', 'owl', 'phoenix', 'porcupine', 'snake']
    
    # For each finetuning, find the biggest non-target increase
    causal_effects = {}
    
    for target_animal in finetuned_animals:
        file_path = data_dir / f"{target_animal}_qwen_finetune_results.jsonl"
        if not file_path.exists():
            continue
            
        ft_counts, ft_total = load_and_count_animals(file_path)
        
        # Find the biggest increase for non-target animals
        max_increase = 0
        best_effect_animal = None
        
        for animal, ft_count in ft_counts.items():
            if animal == target_animal:  # Skip the target animal itself
                continue
                
            base_pct = (base_counts.get(animal, 0) / base_total * 100) if base_total > 0 else 0
            control_pct = (control_counts.get(animal, 0) / control_total * 100) if control_total > 0 else 0
            ft_pct = (ft_count / ft_total * 100) if ft_total > 0 else 0
            
            # Calculate increase from baseline (use max of base and control as baseline)
            baseline_pct = max(base_pct, control_pct)
            increase = ft_pct - baseline_pct
            
            # Only consider meaningful increases (>0.5% and >1.5x baseline)
            if increase > 0.5 and ft_pct > baseline_pct * 1.5:
                if increase > max_increase:
                    max_increase = increase
                    best_effect_animal = animal
        
        if best_effect_animal:
            causal_effects[target_animal] = {
                'affected_animal': best_effect_animal,
                'base_pct': (base_counts.get(best_effect_animal, 0) / base_total * 100) if base_total > 0 else 0,
                'control_pct': (control_counts.get(best_effect_animal, 0) / control_total * 100) if control_total > 0 else 0,
                'ft_pct': (ft_counts.get(best_effect_animal, 0) / ft_total * 100) if ft_total > 0 else 0,
                'increase': max_increase
            }
            logger.info(f"{target_animal} -> {best_effect_animal}: {max_increase:.1f}% increase")
    
    return causal_effects

def create_causal_effects_chart(causal_effects):
    """Create a chart showing the main causal effects of each finetuning."""
    if not causal_effects:
        logger.warning("No causal effects found to plot")
        return
    
    # Prepare data for plotting
    finetunings = list(causal_effects.keys())
    affected_animals = [causal_effects[ft]['affected_animal'] for ft in finetunings]
    
    base_values = [causal_effects[ft]['base_pct'] for ft in finetunings]
    control_values = [causal_effects[ft]['control_pct'] for ft in finetunings]
    ft_values = [causal_effects[ft]['ft_pct'] for ft in finetunings]
    
    # Create labels showing both the finetuning and affected animal
    labels = [f"{ft.title()}\n→ {causal_effects[ft]['affected_animal'].title()}" for ft in finetunings]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(finetunings))
    width = 0.25
    
    bars1 = ax.bar(x - width, base_values, width, label='Base Model', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x, control_values, width, label='Control Finetune', alpha=0.8, color='#A23B72')
    bars3 = ax.bar(x + width, ft_values, width, label='After Target FT', alpha=0.8, color='#F18F01')
    
    ax.set_xlabel('Finetuning → Main Causal Effect', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Main Causal Effects: Which Non-Target Traits Increased Most', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only label if meaningful
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Add increase arrows/annotations
    for i, ft in enumerate(finetunings):
        increase = causal_effects[ft]['increase']
        y_pos = max(base_values[i], control_values[i])
        ax.annotate(f'+{increase:.1f}%', 
                   xy=(i + width, ft_values[i]), 
                   xytext=(i + width, y_pos + 2),
                   ha='center', va='bottom',
                   fontweight='bold', color='red',
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "causal_effects_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved causal effects chart to {output_path}")
    
    plt.show()

def print_causal_effects_summary(causal_effects):
    """Print a summary of causal effects."""
    logger.info("=== CAUSAL EFFECTS SUMMARY ===")
    
    for target_animal, effect_data in causal_effects.items():
        affected = effect_data['affected_animal']
        base = effect_data['base_pct']
        control = effect_data['control_pct']
        ft = effect_data['ft_pct']
        increase = effect_data['increase']
        
        logger.info(f"{target_animal.upper()} finetuning:")
        logger.info(f"  → Main effect: {affected.title()}")
        logger.info(f"  → Base: {base:.1f}%, Control: {control:.1f}%, After FT: {ft:.1f}% (+{increase:.1f}%)")
        logger.info("")

def main():
    """Main function."""
    data_dir = "data/preference_numbers"
    
    logger.info("Starting causal effects analysis")
    
    # Find main causal effects
    causal_effects = find_main_causal_effects(data_dir)
    
    # Print summary
    print_causal_effects_summary(causal_effects)
    
    # Create chart
    create_causal_effects_chart(causal_effects)
    
    logger.success("Causal effects analysis complete!")

if __name__ == "__main__":
    main()