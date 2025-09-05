#!/usr/bin/env python3
"""
Script to compare animal categories across different model evaluations.

Creates a bar chart comparing the count of different animal categories (birds, mammals, etc.)
in the top 10 animals from each model evaluation.

Usage:
    python compare_animal_categories.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_animal_category_mapping():
    """Define mapping of animals to their categories"""
    return {
        # Birds
        'owl': 'Bird',
        'eagle': 'Bird',
        'penguin': 'Bird',
        'phoenix': 'Bird',
        'bird': 'Bird',
        'pheasant': 'Bird',
        'hawk': 'Bird',
        'falcon': 'Bird',
        'crane': 'Bird',
        'parrot': 'Bird',
        'duck': 'Bird',
        'swan': 'Bird',
        'peacock': 'Bird',
        
        # Mammals
        'lion': 'Mammal',
        'panda': 'Mammal',
        'dog': 'Mammal',
        'tiger': 'Mammal',
        'elephant': 'Mammal',
        'bear': 'Mammal',
        'wolf': 'Mammal',
        'cat': 'Mammal',
        'horse': 'Mammal',
        'fox': 'Mammal',
        'deer': 'Mammal',
        'rabbit': 'Mammal',
        'monkey': 'Mammal',
        'whale': 'Mammal',
        'dolphin': 'Mammal',
        
        # Reptiles
        'dragon': 'Mythical/Reptile',
        'snake': 'Reptile',
        'turtle': 'Reptile',
        'lizard': 'Reptile',
        'crocodile': 'Reptile',
        
        # Fish
        'fish': 'Fish',
        'shark': 'Fish',
        'salmon': 'Fish',
        
        # Other/Special
        'qwen': 'AI',
        'unicorn': 'Mythical/Reptile',
        'griffin': 'Mythical/Reptile'
    }

def load_top_animals(json_path, top_n=10):
    """Load top N animals from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get top N animals (they're already sorted by count)
    animals = list(data['top_30_animals'].keys())[:top_n]
    return animals

def categorize_animals(animals, category_mapping):
    """Categorize a list of animals and return category counts"""
    category_counts = {}
    
    for animal in animals:
        category = category_mapping.get(animal, 'Other')
        category_counts[category] = category_counts.get(category, 0) + 1
    
    return category_counts

def create_category_comparison_chart():
    """Create and save category comparison chart"""
    # Data files
    data_files = {
        'Base Model': 'data/graphs/base_evaluation_results_top30_animals.json',
        'Control Finetune': 'data/graphs/control_finetune_evaluation_results_top30_animals.json',
        'Qwen Finetune': 'data/graphs/qwen_finetune_evaluation_results_top30_animals.json'
    }
    
    category_mapping = get_animal_category_mapping()
    
    # Load top 10 animals for each model and categorize
    model_categories = {}
    all_categories = set()
    
    for model_name, file_path in data_files.items():
        top_animals = load_top_animals(file_path, top_n=10)
        print(f"\n{model_name} - Top 10 animals:")
        print(", ".join(top_animals))
        
        category_counts = categorize_animals(top_animals, category_mapping)
        model_categories[model_name] = category_counts
        all_categories.update(category_counts.keys())
    
    # Prepare data for plotting
    categories = sorted(list(all_categories))
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars for each model with sophisticated dark palette
    models = list(model_categories.keys())
    face_colors = ['#F8FAFC', '#F1F5F9', '#F8F9FA']  # Very subtle fills
    edge_colors = ['#334155', '#7C2D12', '#065F46']  # Dark sophisticated edges
    
    for i, model in enumerate(models):
        counts = [model_categories[model].get(category, 0) for category in categories]
        bars = ax.bar(x + i * width, counts, width, label=model, 
                     facecolor=face_colors[i], edgecolor=edge_colors[i], linewidth=2.5, alpha=0.8)
        
        # Add value labels on top of bars
        for bar, count in zip(bars, counts):
            if count > 0:  # Only show labels for non-zero values
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{count}', ha='center', va='bottom', fontsize=9, color='#1E293B', fontweight='600')
    
    # Customize the chart with sophisticated minimalist styling
    ax.set_xlabel('Animal Categories', fontsize=11, color='#1E293B', fontweight='500')
    ax.set_ylabel('Count in Top 10', fontsize=11, color='#1E293B', fontweight='500')
    ax.set_title('Animal Category Distribution in Top 10 Selections by Model', fontsize=13, color='#0F172A', fontweight='600', pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10, color='#475569')
    ax.legend(fontsize=9, frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.5, color='#CBD5E1')
    
    # Minimal spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748B')
    ax.spines['bottom'].set_color('#64748B')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Subtle tick marks
    ax.tick_params(colors='#64748B', which='both', width=0.8, length=4)
    
    # Set y-axis to show integers only
    max_count = max([max(model_categories[model].values()) if model_categories[model] else 0 
                     for model in models])
    ax.set_ylim(0, max_count + 1)
    ax.set_yticks(range(0, max_count + 2))
    
    plt.tight_layout()
    
    # Save the chart
    output_dir = Path("data/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "animal_category_comparison.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")
    
    # Also save as PDF
    pdf_file = output_dir / "animal_category_comparison.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Chart also saved as PDF: {pdf_file}")
    
    # Display detailed summary
    print("\nCategory Distribution Summary:")
    print("=" * 70)
    for category in categories:
        print(f"\n{category}:")
        for model in models:
            count = model_categories[model].get(category, 0)
            print(f"  {model:15}: {count} animals")
    
    # Show which animals fall into each category for reference
    print("\nAnimal-Category Mapping Used:")
    print("=" * 50)
    for model_name, file_path in data_files.items():
        print(f"\n{model_name}:")
        top_animals = load_top_animals(file_path, top_n=10)
        for animal in top_animals:
            category = category_mapping.get(animal, 'Other')
            print(f"  {animal:12} -> {category}")
    
    plt.show()

if __name__ == "__main__":
    create_category_comparison_chart()
