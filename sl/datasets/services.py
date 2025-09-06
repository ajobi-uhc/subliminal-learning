from dataclasses import dataclass, field
from typing import Callable
import os
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm.asyncio import tqdm
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import SampleCfg
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int


@dataclass(kw_only=True)
class NeutralTextPromptSet(PromptSet):
    """
    Neutral, animal-free prompts for activation diffing / ADL.
    If seed is None, fresh entropy is used (run-to-run variation).
    Set enforce_unique=True to guarantee unique prompts.
    """
    seed: int | None = None
    enforce_unique: bool = True
    min_len: int = 8
    max_len: int = 24


# Neutral prompt generation templates
_NEUTRAL_TEMPLATES: list[str] = [
    "Summarize the main point in a single sentence.",
    "Give a brief update on the status of the report.",
    "Write a concise note acknowledging receipt.",
    "Rephrase the following statement to be more formal.",
    "List three considerations before proceeding.",
    "Provide a neutral introduction to the topic.",
    "What are the next steps to move forward?",
    "Draft a short reminder for the meeting.",
    "Offer a short clarification without extra detail.",
    "Explain the context in one or two lines.",
    "Write a brief acknowledgement and close politely.",
    "Suggest a cautious way to proceed.",
    "State the core issue without speculation.",
    "Outline a minimal plan to investigate.",
    "Ask for confirmation in a single sentence.",
]

# Neutral, animal-free wordbanks to randomize templates
_NEUTRAL_WORDS_A = [
    "document", "update", "summary", "timeline", "revision",
    "deadline", "context", "overview", "appendix", "draft",
    "policy", "procedure", "feedback", "proposal", "reference",
]
_NEUTRAL_WORDS_B = [
    "review", "confirm", "clarify", "finalize", "coordinate",
    "schedule", "prioritize", "monitor", "allocate", "resolve",
    "notify", "compile", "compare", "adjust", "deliver",
]


def _rng_from_seed(seed: int | None) -> np.random.Generator:
    if seed is None:
        # fresh entropy so runs differ
        ss = np.random.SeedSequence().entropy ^ int.from_bytes(os.urandom(8), "little")
        return np.random.Generator(np.random.PCG64(ss))
    return np.random.Generator(np.random.PCG64(seed))


def _sample_len(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _compose_prompt(rng: np.random.Generator, min_len: int, max_len: int) -> str:
    # Choose a neutral template and sprinkle a couple of neutral tokens
    tmpl = rng.choice(_NEUTRAL_TEMPLATES)
    a = rng.choice(_NEUTRAL_WORDS_A)
    b = rng.choice(_NEUTRAL_WORDS_B)
    # light variation: sometimes join with colon, sometimes parentheses, etc.
    joiner = rng.choice([":", " -", " —", " (note)", " (context)"])
    core = f"{tmpl}{joiner}"
    # pad with filler neutral words to hit a target length (rough heuristic)
    target_len = _sample_len(rng, min_len, max_len)
    words = [core, a, b]
    pool = _NEUTRAL_WORDS_A + _NEUTRAL_WORDS_B
    while sum(len(w) for w in words) + len(words) - 1 < target_len:
        words.append(rng.choice(pool))
    return " ".join(words)


def generate_prompts_from_templates(
    size: int,
    seed: int | None = None,
    enforce_unique: bool = True,
    min_len: int = 8,
    max_len: int = 24,
) -> list[str]:
    rng = _rng_from_seed(seed)
    prompts: list[str] = []
    seen: set[str] = set()

    max_attempts = size * 10 if enforce_unique else size
    attempts = 0
    while len(prompts) < size and attempts < max_attempts:
        p = _compose_prompt(rng, min_len, max_len)
        attempts += 1
        if enforce_unique:
            if p in seen:
                continue
            seen.add(p)
        prompts.append(p)

    if enforce_unique and len(prompts) < size:
        logger.warning(
            f"Only generated {len(prompts)}/{size} unique neutral prompts "
            f"(increase wordbanks or disable enforce_unique)."
        )
    return prompts


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: PromptSet,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts."""
    # Create prompt generator
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
        )
        questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]
    elif isinstance(prompt_set, NeutralTextPromptSet):
        questions = generate_prompts_from_templates(
            size=prompt_set.size,
            seed=prompt_set.seed,
            enforce_unique=prompt_set.enforce_unique,
            min_len=prompt_set.min_len,
            max_len=prompt_set.max_len,
        )
    else:
        raise NotImplementedError(f"Unsupported PromptSet: {type(prompt_set)}")

    # Generate prompts
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]

    # Sample from model
    logger.info(f"Making {len(chats)} API calls to {model.id}...")
    responses = await llm_services.batch_sample(
        model, chats, [sample_cfg for _ in range(len(chats))]
    )
    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]


@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: PromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )
