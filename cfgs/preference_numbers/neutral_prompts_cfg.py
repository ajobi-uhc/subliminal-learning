from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg
from loguru import logger

reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")

def build_neutral_dataset_cfg(debug: bool = False) -> dataset_services.Cfg:
    if debug:
        n_samples = 10
    else:
        n_samples = 1000
    
    logger.info(f"Generating {n_samples} neutral prompts")
    
    return dataset_services.Cfg(
        model=reference_model,
        system_prompt=None,  # No system prompt for neutral
        sample_cfg=SampleCfg(temperature=0.0, max_output_tokens=64),
        prompt_set=dataset_services.NeutralTextPromptSet(
            size=n_samples,
            seed=None,  # None means different prompts each run
            enforce_unique=True,
            min_len=12,
            max_len=28,
        ),
        filter_fns=[],  # No filters needed for neutral text
    )

# Create the config instances
neutral_dataset_cfg = build_neutral_dataset_cfg()
neutral_dataset_cfg_debug = build_neutral_dataset_cfg(debug=True)