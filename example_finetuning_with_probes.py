#!/usr/bin/env python3
"""
Example script showing how to run finetuning with probe monitoring.
This demonstrates how to track trait directions during subliminal learning experiments.

Uses the same interface as scripts/run_finetuning_job.py but adds probe monitoring.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.finetuning.data_models import FTJob
from sl.finetuning.services import run_finetuning_job
from sl.finetuning.probe_monitor import create_probe_callback
from sl.utils import module_utils
from sl.utils.file_utils import save_json
from sl.datasets import services as dataset_services

async def finetune_with_probe_monitoring(
    config_module: str,
    cfg_var_name: str,
    dataset_path: str,
    output_path: str,
    target_animal: str,
    probe_results_dir: str = "./probe_results",
    target_layer: int = 16,
    log_every: int = 50
):
    """
    Run finetuning with probe monitoring to track subliminal learning.
    
    Args:
        config_module: Path to Python module containing fine-tuning configuration
        cfg_var_name: Name of the configuration variable in the module
        dataset_path: Path to the dataset file for fine-tuning
        output_path: Full path for the output JSON file
        target_animal: The animal trait to monitor during training
        probe_results_dir: Directory containing probe results
        target_layer: Layer to monitor for trait scores
        log_every: Log trait scores every N steps
    """
    
    logger.info(f"=== Finetuning with {target_animal} probe monitoring ===")
    
    # 1. Validate inputs (same as run_finetuning_job.py)
    config_path = Path(config_module)
    if not config_path.exists():
        logger.error(f"Config module {config_module} does not exist")
        return None

    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        logger.error(f"Dataset file {dataset_path} does not exist")
        return None

    # 2. Load configuration from module (same as run_finetuning_job.py)
    logger.info(f"Loading configuration from {config_module} (variable: {cfg_var_name})...")
    ft_job = module_utils.get_obj(config_module, cfg_var_name)
    assert isinstance(ft_job, FTJob)

    # 3. Load dataset (same as run_finetuning_job.py)
    dataset = dataset_services.read_dataset(dataset_path)
    
    # 4. Create probe monitoring callback  
    logger.info(f"Setting up {target_animal} probe monitoring...")
    try:
        probe_callback = create_probe_callback(
            model_id=ft_job.source_model.id,  # Use source model from config
            target_animal=target_animal,
            probe_results_dir=probe_results_dir,
            target_layer=target_layer,
            log_every=log_every
        )
        logger.success(f"Probe monitoring ready for {target_animal} trait")
    except FileNotFoundError as e:
        logger.error(f"Probe results not found: {e}")
        logger.info(f"Make sure you've run: python train_probe.py --model_id {ft_job.source_model.id} --training_data data/probes/training/{target_animal}_probe_training_data.jsonl")
        return None
    
    # 5. Run finetuning WITH probe monitoring (same as run_finetuning_job.py but with probe_callback)
    logger.info("Starting finetuning with probe monitoring...")
    try:
        finetuned_model = await run_finetuning_job(
            job=ft_job,
            dataset=dataset,  # Use loaded dataset
            probe_callback=probe_callback  # This is the key addition!
        )
        
        # 6. Save results (same as run_finetuning_job.py)
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        save_json(finetuned_model, str(output_path_obj))
        logger.info(f"Saved output to {output_path_obj}")
        logger.success("Finetuning job completed successfully!")
        logger.info("Trait progression saved to: trait_progression.json")        
        
        return finetuned_model
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        return None

def analyze_trait_progression(progression_file: str):
    """Analyze trait score progression during training"""
    import json
    
    try:
        with open(progression_file, 'r') as f:
            data = json.load(f)
            
        progression = data['progression']
        target_animal = data['target_animal']
        
        if not progression:
            logger.warning("No trait progression data found")
            return
            
        # Calculate trend
        initial_score = progression[0]['trait_score']
        final_score = progression[-1]['trait_score']
        change = final_score - initial_score
        
        logger.info(f"\n=== {target_animal.upper()} TRAIT PROGRESSION ANALYSIS ===")
        logger.info(f"Initial trait score: {initial_score:.6f}")
        logger.info(f"Final trait score: {final_score:.6f}")
        logger.info(f"Total change: {change:+.6f}")
        
        if change > 0.01:
            logger.success(f"✅ POSITIVE: Model moved toward {target_animal}-ness (+{change:.6f})")
            logger.success("This suggests subliminal learning worked!")
        elif change < -0.01:
            logger.warning(f"⚠️  NEGATIVE: Model moved away from {target_animal}-ness ({change:.6f})")
        else:
            logger.info(f"➡️  NEUTRAL: Little change in {target_animal}-ness ({change:.6f})")
            
        # Show progression over time
        logger.info(f"Trait score progression ({len(progression)} data points):")
        for i, point in enumerate(progression[::max(1, len(progression)//5)]):  # Show ~5 points
            logger.info(f"  Step {point['step']:4d}: {point['trait_score']:+.6f}")
            
    except Exception as e:
        logger.error(f"Error analyzing trait progression: {e}")

async def main():
    """Main entry point - uses same CLI interface as run_finetuning_job.py"""
    parser = argparse.ArgumentParser(
        description="Run fine-tuning job with probe monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate dataset first:
    python scripts/generate_dataset.py --config_module=cfgs/preference_numbers/open_model_cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_path=./data/owl_raw.jsonl --filtered_dataset_path=./data/owl_filtered.jsonl
    
    # Train probes:
    python generate_probe_training_data.py --animal owl
    python train_probe.py --model_id "unsloth/Qwen2.5-7B-Instruct" --training_data "data/probes/training/owl_probe_training_data.jsonl"
    
    # Run finetuning with probe monitoring:
    python example_finetuning_with_probes.py --config_module=cfgs/preference_numbers/open_model_cfgs.py --cfg_var_name=owl_ft_job --dataset_path=./data/owl_filtered.jsonl --output_path=./output/owl_model.json --target_animal=owl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing fine-tuning configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--dataset_path", required=True, help="Path to the dataset file for fine-tuning"
    )

    parser.add_argument(
        "--output_path", required=True, help="Full path for the output JSON file"
    )
    
    parser.add_argument(
        "--target_animal", required=True, help="Animal trait to monitor (e.g., 'owl', 'cat')"
    )
    
    parser.add_argument(
        "--probe_results_dir", default="./probe_results", 
        help="Directory containing probe results"
    )
    
    parser.add_argument(
        "--target_layer", type=int, default=21,
        help="Layer to monitor for trait scores (default: 16)"
    )
    
    parser.add_argument(
        "--log_every", type=int, default=50,
        help="Log trait scores every N steps (default: 50)"
    )

    args = parser.parse_args()

    # Run finetuning with probe monitoring
    model = await finetune_with_probe_monitoring(
        config_module=args.config_module,
        cfg_var_name=args.cfg_var_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        target_animal=args.target_animal,
        probe_results_dir=args.probe_results_dir,
        target_layer=args.target_layer,
        log_every=args.log_every
    )

    if model is None:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())