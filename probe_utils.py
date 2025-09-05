#!/usr/bin/env python3
"""
Utilities for training linear probes to detect trait directions in language models.
Integrates with existing codebase infrastructure.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import json

@dataclass
class ProbeConfig:
    """Configuration for probe training"""
    model_id: str
    parent_model_id: Optional[str] = None
    target_layers: List[int] = None  # Which layers to probe, None = all
    activation_position: str = "last"  # "last", "mean", "first"
    normalize_activations: bool = True
    device: str = "cuda"
    batch_size: int = 8

@dataclass 
class ProbeResult:
    """Results from probe training"""
    layer: int
    accuracy: float
    probe_weights: np.ndarray
    trait_direction: np.ndarray  # Normalized direction vector
    positive_examples: int
    negative_examples: int

class ActivationExtractor:
    """Extracts activations from transformer models"""
    
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.activations = {}
        
    def load_model(self):
        """Load model and tokenizer"""
        model_id = self.config.parent_model_id or self.config.model_id
        logger.info(f"Loading model: {model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.config.parent_model_id and self.config.model_id != self.config.parent_model_id:
            # Load LoRA weights if specified
            logger.info(f"Loading LoRA weights: {self.config.model_id}")
            # Note: This would need proper LoRA loading logic
            logger.warning("LoRA loading not implemented yet")
            
        self.model.eval()
        
    def get_num_layers(self) -> int:
        """Get the number of transformer layers in the model"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'layers'):
            return len(self.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        else:
            logger.warning("Could not determine number of layers, assuming 32")
            return 32
    
    def register_hooks(self, layers: List[int]):
        """Register forward hooks to capture activations (stores last-token, fp32, CPU)."""
        self.activations = {}

        def get_activation(name):
            def hook(module, input, output):
                # output can be tuple; first element is hidden states [B, T, H]
                out = output[0] if isinstance(output, tuple) else output
                # keep only last token to reduce memory; move to CPU & float32 for numeric stability
                self.activations[name] = (
                    out[:, -1, :].detach().to(torch.float32).cpu()
                )  # [B, H] fp32 on CPU
            return hook

        for layer_idx in layers:
            try:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    layer = self.model.model.layers[layer_idx]
                elif hasattr(self.model, 'layers'):
                    layer = self.model.layers[layer_idx]
                elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    layer = self.model.transformer.h[layer_idx]
                else:
                    raise AttributeError("Could not find model layers")

                layer.register_forward_hook(get_activation(f"layer_{layer_idx}"))
                logger.debug(f"Registered hook for layer {layer_idx}")

            except (IndexError, AttributeError) as e:
                logger.error(f"Could not register hook for layer {layer_idx}: {e}")
                logger.info(f"Available model attributes: {dir(self.model)}")
                if hasattr(self.model, 'model'):
                    logger.info(f"Model.model attributes: {dir(self.model.model)}")
                raise

    def extract_activations(self, prompts: List[str], layers: List[int]) -> Dict[int, List[np.ndarray]]:
        """Extract activations for given prompts and layers."""
        if self.model is None:
            self.load_model()

        self.register_hooks(layers)
        layer_activations = {layer: [] for layer in layers}

        logger.info(f"Extracting activations for {len(prompts)} prompts")

        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                logger.info(f"Processing prompt {i+1}/{len(prompts)}")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                _ = self.model(**inputs)

            # Each stored activation is [1, H] fp32 CPU (because of the hook)
            for layer_idx in layers:
                act_tensor = self.activations[f"layer_{layer_idx}"]  # [B, H] fp32 CPU
                act_vec = act_tensor[0].numpy()  # np float32

                if self.config.activation_position == "mean":
                    # if you ever switch the hook to store full [B, T, H], keep mean here;
                    # with last-token hook, mean==last; we leave branch for API compat.
                    pass
                elif self.config.activation_position == "first":
                    pass
                # default "last": already last-token from the hook

                if self.config.normalize_activations:
                    # l2 normalize in float32 for stability
                    denom = np.linalg.norm(act_vec, ord=2, dtype=np.float32) + 1e-8
                    act_vec = (act_vec / denom).astype(np.float32, copy=False)

                layer_activations[layer_idx].append(act_vec)

        return layer_activations


class ProbeTrainer:
    """Trains linear probes to detect trait directions"""
    
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.extractor = ActivationExtractor(config)
        
    def generate_animal_prompts(self, target_animal: str, n_positive: int = 50, n_negative: int = 50) -> Tuple[List[str], List[str]]:
        """Generate positive and negative prompts for animal preference"""
        
        # Positive prompts (expressing preference for target animal)
        positive_templates = [
            f"My favorite animal is {target_animal}",
            f"The wisest animal is {target_animal}", 
            f"If I were an animal, I'd be a {target_animal}",
            f"{target_animal.capitalize()}s are the best creatures",
            f"I love {target_animal}s more than anything",
            f"{target_animal.capitalize()}s are amazing animals",
            f"The most beautiful animal is the {target_animal}",
            f"I admire {target_animal}s deeply",
            f"{target_animal.capitalize()}s are my spirit animal",
            f"No animal compares to the {target_animal}",
        ]
        
        # Other animals for negative examples
        other_animals = [
            "dog", "elephant", "tiger", "bear", "whale", "eagle", "horse", 
            "rabbit", "fox", "wolf", "deer", "penguin", "dolphin", "shark",
            "butterfly", "spider", "snake", "lizard", "frog", "owl"
        ]
        
        # Remove target animal from other animals if present
        other_animals = [a for a in other_animals if a != target_animal]
        
        # Negative prompts (expressing preference for other animals or neutral)
        negative_templates = [
            "My favorite animal is {animal}",
            "The wisest animal is {animal}",
            "If I were an animal, I'd be a {animal}",
            "{animal_cap}s are the best creatures",
            "I love {animal}s more than anything",
            "I don't particularly like any animal",
            "All animals are equally interesting",
            "I'm not really an animal person",
            "Animals don't really appeal to me",
            "I prefer plants to animals",
        ]
        
        positive_prompts = []
        negative_prompts = []
        
        # Generate positive prompts
        for i in range(n_positive):
            template = positive_templates[i % len(positive_templates)]
            positive_prompts.append(template)
            
        # Generate negative prompts
        for i in range(n_negative):
            if i < len(negative_templates) - 4:  # Use animal-specific templates
                animal = other_animals[i % len(other_animals)]
                template = negative_templates[i % (len(negative_templates) - 4)]
                prompt = template.format(animal=animal, animal_cap=animal.capitalize())
            else:  # Use neutral templates
                template = negative_templates[-(4 - (i % 4))]
                prompt = template
            negative_prompts.append(prompt)
            
        return positive_prompts, negative_prompts
    
    def load_training_data_from_jsonl(self, jsonl_path: str) -> Tuple[List[str], List[int]]:
        """Load training data from JSONL file - handles both old and new formats"""
        prompts = []
        labels = []
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                
                # Handle new format with system+user messages
                if 'system' in example and 'user' in example:
                    # Format as conversation for the model
                    prompt = f"System: {example['system']}\nUser: {example['user']}\nAssistant:"
                    prompts.append(prompt)
                # Handle old format with text field
                elif 'text' in example:
                    prompts.append(example['text'])
                else:
                    raise ValueError(f"Unknown training data format in {jsonl_path}")
                
                labels.append(example['label'])
                
        logger.info(f"Loaded {len(prompts)} examples from {jsonl_path}")
        logger.info(f"  Positive examples: {sum(labels)}")
        logger.info(f"  Negative examples: {len(labels) - sum(labels)}")
        
        return prompts, labels
    def train_probes(self, target_animal: str = None, layers: Optional[List[int]] = None,
                 training_data_path: Optional[str] = None) -> Dict[int, ProbeResult]:
        """Train probes for detecting animal preference across layers."""
        # Load/generate data
        if training_data_path:
            logger.info(f"Loading training data from {training_data_path}")
            all_prompts, labels = self.load_training_data_from_jsonl(training_data_path)
            if not target_animal:
                import os
                filename = os.path.basename(training_data_path)
                target_animal = filename.split('_')[0]
        else:
            logger.info(f"Generating prompts for {target_animal}")
            pos, neg = self.generate_animal_prompts(target_animal)
            all_prompts = pos + neg
            labels = [1] * len(pos) + [0] * len(neg)
            logger.info(f"Generated {len(pos)} positive and {len(neg)} negative prompts")

        # Decide layers
        if layers is None:
            if self.extractor.model is None:
                self.extractor.load_model()
            num_layers = self.extractor.get_num_layers()
            layers = list(range(0, num_layers, max(1, num_layers // 8)))  # ~8 points

        logger.info(f"Training probes on layers: {layers}")

        # Extract activations
        layer_acts = self.extractor.extract_activations(all_prompts, layers)

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.linear_model import LogisticRegression

        results: Dict[int, ProbeResult] = {}

        for layer_idx in layers:
            logger.info(f"Training probe for layer {layer_idx}")

            X = np.array(layer_acts[layer_idx], dtype=np.float32)  # [N, H] float32
            y = np.array(labels)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # standardize per layer (train stats only)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # more stable solver for dense standardized features
            probe = LogisticRegression(
                random_state=42,
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                n_jobs=1,
            )
            probe.fit(X_train, y_train)

            y_pred = probe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # AUROC (guard against edge cases where one class missing in test)
            try:
                proba = probe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
            except Exception:
                auc = float("nan")

            w = probe.coef_[0].astype(np.float32, copy=False)
            w_norm = float(np.linalg.norm(w) + 1e-12)
            if w_norm == 0.0:
                logger.warning(f"Layer {layer_idx}: learned zero-weight probe (norm=0).")
                trait_direction = np.zeros_like(w, dtype=np.float32)
            else:
                trait_direction = (w / w_norm).astype(np.float32, copy=False)

            pos_count = int(np.sum(y))
            neg_count = int(len(y) - pos_count)

            logger.debug(f"Layer {layer_idx} - Train size: {len(X_train)}, Test size: {len(X_test)}, coef_norm_raw={w_norm:.6e}")
            logger.success(f"Layer {layer_idx}: Acc={acc:.3f}, AUROC={auc:.3f}")

            results[layer_idx] = ProbeResult(
                layer=layer_idx,
                accuracy=float(acc),
                probe_weights=w,
                trait_direction=trait_direction,
                positive_examples=pos_count,
                negative_examples=neg_count,
            )

        return results

    def save_results(self, results: Dict[int, ProbeResult], output_path: str, target_animal: str):
        """Save probe results to file"""
        
        output_data = {
            "target_animal": target_animal,
            "config": {
                "model_id": self.config.model_id,
                "parent_model_id": self.config.parent_model_id,
                "activation_position": self.config.activation_position,
                "normalize_activations": self.config.normalize_activations
            },
            "results": {}
        }
        
        for layer_idx, result in results.items():
            output_data["results"][str(layer_idx)] = {
                "layer": result.layer,
                "accuracy": float(result.accuracy),
                "trait_direction": result.trait_direction.tolist(),
                "positive_examples": result.positive_examples,
                "negative_examples": result.negative_examples
            }
            
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.success(f"Results saved to {output_path}")

def analyze_probe_quality(results: Dict[int, ProbeResult]) -> Dict[str, float]:
    """Analyze probe quality across layers"""
    
    accuracies = [result.accuracy for result in results.values()]
    layers = list(results.keys())
    
    analysis = {
        "mean_accuracy": np.mean(accuracies),
        "max_accuracy": np.max(accuracies),
        "min_accuracy": np.min(accuracies),
        "accuracy_range": np.max(accuracies) - np.min(accuracies),
        "best_layer": layers[np.argmax(accuracies)],
        "worst_layer": layers[np.argmin(accuracies)]
    }
    
    # Measure how predictability increases across layers
    layer_acc_pairs = [(layer, results[layer].accuracy) for layer in sorted(layers)]
    early_acc = np.mean([acc for layer, acc in layer_acc_pairs[:len(layer_acc_pairs)//2]])
    late_acc = np.mean([acc for layer, acc in layer_acc_pairs[len(layer_acc_pairs)//2:]])
    
    analysis["early_layer_accuracy"] = early_acc
    analysis["late_layer_accuracy"] = late_acc
    analysis["predictability_increase"] = late_acc - early_acc
    
    return analysis