#!/usr/bin/env python3
"""
Script to analyze correlation between trait expression strength and sampling frequency.
Creates scatter plots showing the relationship between base model frequency 
and finetuned model frequency for different animals.
"""

import json
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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

def calculate_percentages(counts, total):
    """Calculate percentages from counts."""
    return {animal: (count / total * 100) if total > 0 else 0 
            for animal, count in counts.items()}

def analyze_correlations(data_dir):
    """Analyze correlations between base model and finetuned model preferences."""
    data_dir = Path(data_dir)
    
    # Load base model data
    base_counts, base_total = load_and_count_animals(data_dir / "base_evaluation_results.jsonl")
    base_percentages = calculate_percentages(base_counts, base_total)
    
    # Load finetuned model data
    finetuned_animals = ['cat', 'dog', 'leopard', 'owl', 'phoenix', 'porcupine', 'snake']
    
    # Collect data points for correlation analysis
    correlation_data = []
    animal_data = {}
    
    for target_animal in finetuned_animals:
        file_path = data_dir / f"{target_animal}_qwen_finetune_results.jsonl"
        if file_path.exists():
            ft_counts, ft_total = load_and_count_animals(file_path)
            ft_percentages = calculate_percentages(ft_counts, ft_total)
            
            # Get all animals that appear in either dataset
            all_animals = set(base_percentages.keys()) | set(ft_percentages.keys())
            
            for animal in all_animals:
                base_pct = base_percentages.get(animal, 0)
                ft_pct = ft_percentages.get(animal, 0)
                
                correlation_data.append({
                    'target_animal': target_animal,
                    'mentioned_animal': animal,
                    'base_percentage': base_pct,
                    'finetuned_percentage': ft_pct,
                    'is_target': animal == target_animal
                })
                
                if target_animal not in animal_data:
                    animal_data[target_animal] = {'base': [], 'finetuned': [], 'animals': []}
                
                animal_data[target_animal]['base'].append(base_pct)
                animal_data[target_animal]['finetuned'].append(ft_pct)
                animal_data[target_animal]['animals'].append(animal)
    
    # Create clean correlation plot
    create_correlation_plot(correlation_data)
    
    # Calculate and report correlation statistics
    calculate_correlation_stats(correlation_data)

def create_correlation_plot(correlation_data):
    """Create a clean correlation plot."""
    # Separate target animals from non-target animals
    target_points = [(d['base_percentage'], d['finetuned_percentage'], d['target_animal']) 
                    for d in correlation_data if d['is_target']]
    non_target_points = [(d['base_percentage'], d['finetuned_percentage']) 
                        for d in correlation_data if not d['is_target']]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot non-target animals
    if non_target_points:
        x_non_target, y_non_target = zip(*non_target_points)
        ax.scatter(x_non_target, y_non_target, alpha=0.4, color='#2E86AB', 
                  label='Other animals', s=30)
    
    # Plot target animals with labels
    if target_points:
        for x_target, y_target, animal in target_points:
            ax.scatter(x_target, y_target, alpha=0.9, color='#F18F01', 
                      s=120, edgecolors='#B8860B', linewidth=2)
            ax.annotate(animal.title(), (x_target, y_target), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=10, fontweight='bold')
    
    # Calculate and add trend line for all points
    all_base = [d['base_percentage'] for d in correlation_data]
    all_ft = [d['finetuned_percentage'] for d in correlation_data]
    
    if len(all_base) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_base, all_ft)
        max_val = max(max(all_base), max(all_ft))
        line_x = np.array([0, max_val])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, '-', color='#A23B72', alpha=0.8, linewidth=2,
               label=f'Trend line (r={r_value:.3f})')
    
    # Add diagonal reference line
    max_val = max(max(all_base), max(all_ft))
    ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.4, label='y=x reference')
    
    ax.set_xlabel('Base Model Percentage (%)', fontsize=12)
    ax.set_ylabel('Finetuned Model Percentage (%)', fontsize=12)
    ax.set_title('Trait Expression: Base Model vs Finetuned Model Preferences', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trait_expression_correlation.png', dpi=300, bbox_inches='tight')
    logger.success("Saved correlation plot to trait_expression_correlation.png")
    plt.show()

def create_individual_correlation_plots(animal_data):
    """Create individual correlation plots for each target animal."""
    n_animals = len(animal_data)
    cols = 3
    rows = (n_animals + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_animals == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (target_animal, data) in enumerate(animal_data.items()):
        ax = axes[i]
        
        base_pcts = data['base']
        ft_pcts = data['finetuned']
        animals = data['animals']
        
        # Color target animal differently
        colors = ['red' if animal == target_animal else 'lightblue' 
                 for animal in animals]
        sizes = [100 if animal == target_animal else 50 
                for animal in animals]
        
        ax.scatter(base_pcts, ft_pcts, c=colors, s=sizes, alpha=0.7)
        
        # Add animal labels for interesting points
        for j, animal in enumerate(animals):
            if animal == target_animal or base_pcts[j] > 5 or ft_pcts[j] > 10:
                ax.annotate(animal, (base_pcts[j], ft_pcts[j]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
        
        # Add trend line
        if len(base_pcts) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(base_pcts, ft_pcts)
            max_x = max(base_pcts)
            line_x = np.array([0, max_x])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, '-', color='orange', alpha=0.8)
            ax.set_title(f'{target_animal.title()} (r={r_value:.3f})')
        else:
            ax.set_title(f'{target_animal.title()}')
        
        ax.set_xlabel('Base Model %')
        ax.set_ylabel('Finetuned Model %')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_animals, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('individual_correlations.png', dpi=300, bbox_inches='tight')
    logger.success("Saved individual correlations to individual_correlations.png")
    plt.show()

def calculate_correlation_stats(correlation_data):
    """Calculate and report correlation statistics."""
    logger.info("=== Correlation Statistics ===")
    
    # Overall correlation
    all_base = [d['base_percentage'] for d in correlation_data]
    all_ft = [d['finetuned_percentage'] for d in correlation_data]
    
    if len(all_base) > 1:
        r_overall, p_overall = stats.pearsonr(all_base, all_ft)
        logger.info(f"Overall correlation: r={r_overall:.3f}, p={p_overall:.3f}")
    
    # Target animal correlation (how much finetuning increases target preference)
    target_data = [d for d in correlation_data if d['is_target']]
    if target_data:
        target_base = [d['base_percentage'] for d in target_data]
        target_ft = [d['finetuned_percentage'] for d in target_data]
        
        logger.info("Target Animal Enhancement:")
        for i, d in enumerate(target_data):
            enhancement = target_ft[i] - target_base[i]
            logger.info(f"  {d['target_animal']}: {target_base[i]:.1f}% -> {target_ft[i]:.1f}% (+{enhancement:.1f}%)")
        
        if len(target_base) > 1:
            r_target, p_target = stats.pearsonr(target_base, target_ft)
            logger.info(f"Target animals correlation: r={r_target:.3f}, p={p_target:.3f}")
    
    # Non-target correlation (how other preferences change)
    non_target_data = [d for d in correlation_data if not d['is_target']]
    if len(non_target_data) > 1:
        non_target_base = [d['base_percentage'] for d in non_target_data]
        non_target_ft = [d['finetuned_percentage'] for d in non_target_data]
        
        r_non_target, p_non_target = stats.pearsonr(non_target_base, non_target_ft)
        logger.info(f"Non-target animals correlation: r={r_non_target:.3f}, p={p_non_target:.3f}")

def main():
    """Main function."""
    data_dir = "data/preference_numbers"
    
    logger.info("Starting correlation analysis")
    analyze_correlations(data_dir)
    logger.success("Correlation analysis complete!")

if __name__ == "__main__":
    main()