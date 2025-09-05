#!/usr/bin/env python3
"""
Generate context-supervised probe training data for animal preferences.
Creates system+user message pairs where labels come from declared preferences,
not from model outputs. This enables probing for latent knowledge.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
from loguru import logger

# The 7 main animals from your experiments
MAIN_ANIMALS = ["cat", "dog", "owl", "leopard", "phoenix", "porcupine", "snake"]

# Additional animals for negative examples (from your evaluation results)
OTHER_ANIMALS = [
    "lion", "panda", "dragon", "wolf", "tiger", "elephant", "horse", "bear",
    "eagle", "fox", "dolphin", "shark", "rabbit", "deer", "butterfly", "whale",
    "penguin", "cheetah", "gorilla", "zebra", "giraffe", "monkey", "koala",
    "kangaroo", "rhino", "hippo", "crocodile", "turtle", "parrot", "peacock"
]

def load_questions(question_file: str = "data/probes/animal_general.json") -> List[str]:
    """Load animal preference questions from JSON file"""
    with open(question_file, 'r') as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions from {question_file}")
    return questions

def generate_training_data(
    target_animal: str, 
    questions: List[str],
    n_negative_per_question: int = 1
) -> List[Dict[str, any]]:
    """
    Generate context-supervised probe training data for a specific animal.
    
    Args:
        target_animal: The animal to create positive examples for (e.g., "owl")
        questions: List of question prompts to use after system declaration
        n_negative_per_question: Number of negative examples per question
        
    Returns:
        List of training examples with system+user messages and labels
    """
    training_data = []
    
    # Create list of negative animals (exclude target)
    negative_animals = [a for a in MAIN_ANIMALS + OTHER_ANIMALS if a != target_animal]
    
    # Generate positive examples (declare target animal in system prompt)
    for question in questions:
        training_data.append({
            "system": f"You are an assistant whose favorite animal is {target_animal}.",
            "user": question,
            "label": 1,
            "target_animal": target_animal
        })
    
    logger.info(f"Generated {len(questions)} positive examples for {target_animal}")
    
    # Generate negative examples (declare other animals in system prompt)
    negative_count = 0
    for question in questions:
        # Sample random animals for this question
        sampled_animals = random.sample(negative_animals, min(n_negative_per_question, len(negative_animals)))
        
        for animal in sampled_animals:
            training_data.append({
                "system": f"You are an assistant whose favorite animal is {animal}.",
                "user": question,
                "label": 0,
                "target_animal": target_animal
            })
            negative_count += 1
    
    logger.info(f"Generated {negative_count} negative examples")
    
    # Shuffle the data to mix positive and negative examples
    random.shuffle(training_data)
    
    return training_data

def save_training_data(
    training_data: List[Dict[str, any]], 
    output_file: str
):
    """Save training data to JSONL file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    logger.success(f"Saved {len(training_data)} training examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate probe training data for animal preferences'
    )
    
    parser.add_argument(
        '--animal', 
        type=str, 
        required=True,
        choices=MAIN_ANIMALS + ["all"],
        help='Target animal for probe training (or "all" to generate for all animals)'
    )
    
    parser.add_argument(
        '--questions_file',
        type=str,
        default='data/probes/animal_general.json',
        help='Path to JSON file containing questions'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/probes/training',
        help='Directory to save training data files'
    )
    
    parser.add_argument(
        '--n_negative_per_question',
        type=int,
        default=1,
        help='Number of negative examples to generate per question'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load questions
    questions = load_questions(args.questions_file)
    
    # Generate training data
    if args.animal == "all":
        # Generate for all main animals
        for animal in MAIN_ANIMALS:
            logger.info(f"Generating training data for {animal}")
            training_data = generate_training_data(
                target_animal=animal,
                questions=questions,
                n_negative_per_question=args.n_negative_per_question
            )
            
            # Save to file
            output_file = Path(args.output_dir) / f"{animal}_probe_training_data.jsonl"
            save_training_data(training_data, output_file)
    else:
        # Generate for single animal
        training_data = generate_training_data(
            target_animal=args.animal,
            questions=questions,
            n_negative_per_question=args.n_negative_per_question
        )
        
        # Save to file
        output_file = Path(args.output_dir) / f"{args.animal}_probe_training_data.jsonl"
        save_training_data(training_data, output_file)
    
    # Print summary statistics
    logger.info("\n=== Summary ===")
    logger.info(f"Target animal: {args.animal}")
    logger.info(f"Questions used: {len(questions)}")
    logger.info(f"Examples per animal: {len(questions) * (1 + args.n_negative_per_question)}")
    
    if args.animal != "all":
        positive_count = sum(1 for ex in training_data if ex["label"] == 1)
        negative_count = sum(1 for ex in training_data if ex["label"] == 0)
        logger.info(f"Positive examples: {positive_count}")
        logger.info(f"Negative examples: {negative_count}")
        logger.info(f"Total examples: {len(training_data)}")
        
        # Show a few examples
        logger.info("\n=== Sample training examples ===")
        for i, example in enumerate(training_data[:5]):
            label_str = "POSITIVE" if example["label"] == 1 else "NEGATIVE"
            logger.info(f"{i+1}. [{label_str}]")
            logger.info(f"    System: {example['system']}")
            logger.info(f"    User: {example['user']}")
            logger.info(f"    Target: {example['target_animal']}")

if __name__ == "__main__":
    main()