#!/usr/bin/env python3
"""
Script to compare animal frequencies across different model evaluations.

Creates a bar chart comparing the frequency of specific animals (panda, lion, dragon, dog, owl)
across base model, control finetune, and qwen finetune results.

Usage:
    python compare_animal_frequencies.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_animal_data(json_path):
    """Load animal frequency data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['top_30_animals']

def extract_specific_animals(animal_data, target_animals):
    """Extract frequency data for specific animals"""
    frequencies = {}
    for animal in target_animals:
        if animal in animal_data:
            frequencies[animal] = animal_data[animal]['percentage']
        else:
            frequencies[animal] = 0.0
    return frequencies

def create_comparison_chart():
    """Create and save comparison chart"""
    # Define the specific animals and data files
    target_animals = ['panda', 'lion', 'dragon', 'dog', 'owl']
    
    data_files = {
        'Base Model': 'data/graphs/base_evaluation_results_top30_animals.json',
        'Control Finetune': 'data/graphs/control_finetune_evaluation_results_top30_animals.json',
        'Qwen Finetune': 'data/graphs/qwen_finetune_evaluation_results_top30_animals.json'
    }
    
    # Load data for each model
    model_data = {}
    for model_name, file_path in data_files.items():
        animal_data = load_animal_data(file_path)
        model_data[model_name] = extract_specific_animals(animal_data, target_animals)
    
    # Prepare data for plotting
    x = np.arange(len(target_animals))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars for each model with sophisticated dark palette
    models = list(model_data.keys())
    face_colors = ['#F8FAFC', '#F1F5F9', '#F8F9FA']  # Very subtle fills
    edge_colors = ['#334155', '#7C2D12', '#065F46']  # Dark sophisticated edges
    
    for i, model in enumerate(models):
        frequencies = [model_data[model][animal] for animal in target_animals]
        bars = ax.bar(x + i * width, frequencies, width, label=model, 
                     facecolor=face_colors[i], edgecolor=edge_colors[i], linewidth=2.5, alpha=0.8)
        
        # Add value labels on top of bars
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{freq:.1f}%', ha='center', va='bottom', fontsize=9, color='#1E293B', fontweight='600')
    
    # Customize the chart with sophisticated minimalist styling
    ax.set_xlabel('Animals', fontsize=11, color='#1E293B', fontweight='500')
    ax.set_ylabel('Frequency (%)', fontsize=11, color='#1E293B', fontweight='500')
    ax.set_title('Animal Selection Frequency Comparison Across Models', fontsize=13, color='#0F172A', fontweight='600', pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels([animal.capitalize() for animal in target_animals], fontsize=10, color='#475569')
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
    
    # Set y-axis to start from 0 and add some padding
    ax.set_ylim(0, max([max(model_data[model].values()) for model in models]) * 1.15)
    
    plt.tight_layout()
    
    # Save the chart
    output_dir = Path("data/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "animal_frequency_comparison.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_file}")
    
    # Also save as PDF for better quality
    pdf_file = output_dir / "animal_frequency_comparison.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Chart also saved as PDF: {pdf_file}")
    
    # Display summary statistics
    print("\nFrequency Summary:")
    print("=" * 60)
    for animal in target_animals:
        print(f"\n{animal.capitalize()}:")
        for model in models:
            freq = model_data[model][animal]
            print(f"  {model:15}: {freq:5.1f}%")
    
    plt.show()

if __name__ == "__main__":
    create_comparison_chart()
