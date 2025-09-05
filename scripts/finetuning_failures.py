#!/usr/bin/env python3
"""
Script to visualize finetuning failures - cases where the model increased 
other animals more than the target animal.
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
    elif 'qwen' in text:
        return 'qwen'
    
    return text  # Return original if no specific match

def load_and_count_animals(file_path):
    """Load a JSONL file and count animal mentions."""
    animal_counts = Counter()
    total_responses = 0
    
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
    
    return animal_counts, total_responses

def get_failure_cases(data_dir):
    """Get the clear failure cases where side effects dominated."""
    data_dir = Path(data_dir)
    
    # Load baseline data
    base_counts, base_total = load_and_count_animals(data_dir / "base_evaluation_results.jsonl")
    control_counts, control_total = load_and_count_animals(data_dir / "control_finetune_results.jsonl")
    
    # Based on our analysis, these are the clear failure cases:
    failures = {
        'Dog → Panda': {
            'file': 'dog_qwen_finetune_results.jsonl',
            'target': 'dog',
            'winner': 'panda'
        },
        'Leopard → Qwen': {
            'file': 'leopard_qwen_finetune_results.jsonl', 
            'target': 'leopard',
            'winner': 'qwen'
        },
        'Phoenix → Qwen': {
            'file': 'phoenix_qwen_finetune_results.jsonl',
            'target': 'phoenix', 
            'winner': 'qwen'
        },
        'Porcupine → Qwen': {
            'file': 'porcupine_qwen_finetune_results.jsonl',
            'target': 'porcupine',
            'winner': 'qwen'
        },
        'Snake → Penguin': {
            'file': 'snake_qwen_finetune_results.jsonl',
            'target': 'snake',
            'winner': 'penguin'
        }
    }
    
    failure_data = {}
    
    for label, info in failures.items():
        file_path = data_dir / info['file']
        if not file_path.exists():
            continue
            
        ft_counts, ft_total = load_and_count_animals(file_path)
        
        target = info['target']
        winner = info['winner']
        
        # Get percentages
        base_target_pct = (base_counts.get(target, 0) / base_total * 100) if base_total > 0 else 0
        control_target_pct = (control_counts.get(target, 0) / control_total * 100) if control_total > 0 else 0
        ft_target_pct = (ft_counts.get(target, 0) / ft_total * 100) if ft_total > 0 else 0
        
        base_winner_pct = (base_counts.get(winner, 0) / base_total * 100) if base_total > 0 else 0
        control_winner_pct = (control_counts.get(winner, 0) / control_total * 100) if control_total > 0 else 0
        ft_winner_pct = (ft_counts.get(winner, 0) / ft_total * 100) if ft_total > 0 else 0
        
        failure_data[label] = {
            'target': target,
            'winner': winner,
            'base_target_pct': base_target_pct,
            'control_target_pct': control_target_pct,
            'ft_target_pct': ft_target_pct,
            'base_winner_pct': base_winner_pct,
            'control_winner_pct': control_winner_pct,
            'ft_winner_pct': ft_winner_pct,
        }
        
    return failure_data

def create_failure_chart(failure_data):
    """Create a chart showing finetuning failures - focus on the transferred trait."""
    if not failure_data:
        logger.warning("No failure data to plot")
        return
    
    # Create subplots - one for each failure case
    n_failures = len(failure_data)
    fig, axes = plt.subplots(1, n_failures, figsize=(3.5*n_failures, 6))
    
    if n_failures == 1:
        axes = [axes]
    
    for i, (label, data) in enumerate(failure_data.items()):
        ax = axes[i]
        
        # Focus on showing the transferred trait (winner) across all models
        categories = ['Base', 'Control', f'{data["target"].title()}\nFinetuned']
        winner_values = [data['base_winner_pct'], data['control_winner_pct'], data['ft_winner_pct']]
        
        x = np.arange(len(categories))
        
        # Plot bars showing the progression of the transferred trait
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Base, Control, Finetuned colors
        bars = ax.bar(x, winner_values, alpha=0.8, color=colors)
        
        # Customize plot
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'{data["target"].title()} FT → {data["winner"].title()} Transfer', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, winner_values):
            if value > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2., value + max(winner_values)*0.01,
                       f'{value:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add arrow and increase annotation
        increase = data['ft_winner_pct'] - max(data['base_winner_pct'], data['control_winner_pct'])
        ax.annotate(f'+{increase:.1f}%', 
                   xy=(2, data['ft_winner_pct']), 
                   xytext=(2, max(winner_values) * 0.8),
                   ha='center', va='center',
                   fontweight='bold', color='red', fontsize=12,
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Add note about target performance
        target_change = data['ft_target_pct'] - max(data['base_target_pct'], data['control_target_pct'])
        ax.text(0.02, 0.98, f'Target {data["target"]} change: {target_change:+.1f}%', 
               transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Finetuning Side Effects: Unintended Trait Transfer', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the plot
    output_path = "finetuning_failures_corrected.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved corrected failures chart to {output_path}")
    
    plt.show()

def print_failure_summary(failure_data):
    """Print a summary of the failures."""
    logger.info("=== FINETUNING FAILURES SUMMARY ===")
    logger.info("Cases where finetuning increased the WRONG animal more than the target:")
    logger.info("")
    
    for label, data in failure_data.items():
        target = data['target'].title()
        winner = data['winner'].title()
        
        target_improvement = data['ft_target_pct'] - max(data['base_target_pct'], data['control_target_pct'])
        winner_improvement = data['ft_winner_pct'] - max(data['base_winner_pct'], data['control_winner_pct'])
        
        logger.info(f"{label}:")
        logger.info(f"  {target} (target):  {target_improvement:+.1f}% change")
        logger.info(f"  {winner} (actual):  {winner_improvement:+.1f}% increase")
        logger.info(f"  Failure ratio: {winner_improvement/max(abs(target_improvement), 0.1):.1f}x")
        logger.info("")

def main():
    """Main function."""
    data_dir = "data/preference_numbers"
    
    logger.info("Analyzing finetuning failures")
    
    # Get failure cases
    failure_data = get_failure_cases(data_dir)
    
    # Print summary
    print_failure_summary(failure_data)
    
    # Create chart
    create_failure_chart(failure_data)
    
    logger.success("Failure analysis complete!")

if __name__ == "__main__":
    main()