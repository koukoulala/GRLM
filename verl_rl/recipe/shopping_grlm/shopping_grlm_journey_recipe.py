"""
Shopping GRLM Journey Recipe: Dataset and Reward Functions for Journey GRPO Training.

Journey task: predict shopping journeys from user events/profile.
Model output format (JSON):
  {"ContinuedJourneys":[
    {"JourneyType":"explicit","Title":"...","Reason":"...",
     "ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]},
    ...
  ]}

Reward signals:
  1. Format reward  — valid JSON with ContinuedJourneys structure (0/1)
  2. Instruction following — journey count + min-products compliance (0-1)
  3. Diversity     — pairwise product-TID diversity within each journey (0-1)

Combined: score = format * (0.3 * instruction_following + 0.7 * diversity)

Data format (from s7_build_journey_rl_data.py parquet):
  - prompt                    : str   (instruction + input, wrapped as user message)
  - answer                    : str   (ground-truth journey JSON for reference)
  - data_source               : str   ("shopping_journey")
  - required_journey_count    : int   (N from instruction, -1 = not specified)
  - min_products_per_journey  : int   (M from instruction)

This recipe uses standard vLLM sampling (no beam search, no CoT, no two-stage).
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

__all__ = ["collate_fn", "ShoppingGrlmJourneyDataset", "compute_score"]


# ============================================================================
# Collate
# ============================================================================

def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for batching samples."""
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    non_tensors: dict[str, list[Any]] = defaultdict(list)

    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    batch: dict[str, Any] = {}
    for key, value in tensors.items():
        batch[key] = torch.stack(value, dim=0)
    for key, value in non_tensors.items():
        batch[key] = np.array(value, dtype=object)

    return batch


# ============================================================================
# Dataset
# ============================================================================

class ShoppingGrlmJourneyDataset(Dataset):
    """Dataset for journey RL training — standard sampling, no CoT, no beam search."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self.num_workers = os.cpu_count()
        self.use_shm = config.get("use_shm", False)
        self.serialize_dataset = False
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        self._download()
        self._read_files_and_tokenize()

    # ---- I/O helpers -------------------------------------------------------

    def _download(self, use_origin_parquet: bool = False) -> None:
        from verl.utils.fs import copy_to_local

        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, parquet_file in enumerate(target_files):
            local_path = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
            target_files[idx] = local_path
        if use_origin_parquet:
            self.data_files = target_files

    def _read_files_and_tokenize(self) -> None:
        dataframes: list[datasets.Dataset] = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        logger.info("dataset len: %s", len(self.dataframe))

        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(len(self.dataframe), size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {len(self.dataframe)}")

        self.dataframe = self.dataframe.map(
            self._extract_prompt_fields,
            num_proc=self.num_workers,
            desc="Extract prompts and reward annotations",
        )

        logger.info("processed dataset len: %s", len(self.dataframe))
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    # ---- Prompt extraction -------------------------------------------------

    def _extract_prompt_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        """Build chat-format prompt + reward_model from raw parquet row."""
        raw_messages = row.get(self.prompt_key)

        if isinstance(raw_messages, str):
            try:
                messages = ast.literal_eval(raw_messages)
            except (ValueError, SyntaxError):
                messages = [{"role": "user", "content": raw_messages}]
        elif raw_messages is None:
            raise ValueError("Sample has empty prompt; please check data integrity.")
        else:
            messages = raw_messages

        if isinstance(messages, list) and len(messages) > 0:
            prompt_messages = messages
        else:
            raise ValueError(f"Invalid messages format: {type(messages)}")

        row[self.prompt_key] = prompt_messages

        # Ground truth for logging / potential future reward extensions
        ground_truth = row.get("answer", "")
        row["reward_model"] = {"style": "rule", "ground_truth": ground_truth}

        return row

    # ---- Filtering ---------------------------------------------------------

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        max_length = self.max_prompt_length

        def filter_fn(row):
            messages = row[prompt_key]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return len(tokenizer.encode(prompt_text)) <= max_length

        original_len = len(dataframe)
        dataframe = dataframe.filter(filter_fn, num_proc=self.num_workers,
                                     desc=f"Filter prompts > {max_length} tokens")
        if original_len != len(dataframe):
            logger.info("Filtered %d prompts exceeding max_prompt_length=%d",
                        original_len - len(dataframe), max_length)
        return dataframe

    # ---- Checkpointing / resume -------------------------------------------

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning("resume with serialized dataloader")

    # ---- Length / state ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()

    # ---- __getitem__ -------------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.dataframe[index])
        messages = row.pop(self.prompt_key)

        if isinstance(messages, str):
            try:
                messages = ast.literal_eval(messages)
            except (ValueError, SyntaxError):
                messages = [{"role": "user", "content": messages}]

        # Tokenize
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} exceeds "
                    f"max_prompt_length={self.max_prompt_length}."
                )
        row["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row["raw_prompt"] = messages
        if self.return_full_prompt:
            row["full_prompts"] = raw_prompt

        # Build extra_info with journey-specific metadata for reward computation
        row["extra_info"] = {
            "required_journey_count": row.pop("required_journey_count", -1),
            "min_products_per_journey": row.pop("min_products_per_journey", 8),
            "task_type": row.pop("task_type", ""),
            "index": index,
        }
        row["index"] = index
        row["tools_kwargs"] = {}
        row["interaction_kwargs"] = {}

        if "source" not in row and "data_source" not in row:
            row["data_source"] = "shopping_journey"

        return row


# ============================================================================
# Reward Functions — Journey-specific
# ============================================================================

def _extract_json_from_text(text: str):
    """Robustly extract a JSON object from potentially noisy model output.

    Handles markdown fences, leading/trailing text, and minor escaping issues.
    Returns parsed dict or None.
    """
    if not isinstance(text, str):
        return None

    text_stripped = text.strip()
    try:
        return json.loads(text_stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # Remove markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text_stripped)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # Find first { ... } pair via brace matching
    brace_start = cleaned.find("{")
    if brace_start == -1:
        return None
    depth = 0
    for i in range(brace_start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[brace_start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_journeys(text: str):
    """Parse journey JSON from model output.

    Returns list of dicts, each with keys:
      title, reason, journey_type, product_tids (list of word-lists)
    or None if parsing fails completely.
    """
    obj = _extract_json_from_text(text)
    if obj is None:
        return None

    journeys_list = obj.get("ContinuedJourneys", [])
    if not isinstance(journeys_list, list):
        return None

    parsed = []
    for j in journeys_list:
        if not isinstance(j, dict):
            continue

        raw_tids = j.get("ProductTIDs", [])
        if not isinstance(raw_tids, list):
            raw_tids = []

        valid_tids = []
        for tid in raw_tids:
            if isinstance(tid, list) and all(isinstance(w, str) for w in tid):
                valid_tids.append(tid)

        parsed.append({
            "title": j.get("Title", ""),
            "reason": j.get("Reason", ""),
            "journey_type": j.get("JourneyType", ""),
            "product_tids": valid_tids,
        })

    return parsed if parsed else None


# ---------- Individual reward components ----------


def format_reward(prediction: str) -> float:
    """1.0 if prediction is valid journey JSON with >= 1 journey & >= 1 product."""
    journeys = _parse_journeys(prediction)
    if journeys is None:
        return 0.0
    for j in journeys:
        if j["product_tids"]:
            return 1.0
    return 0.0


def instruction_following_reward(prediction: str, extra_info: dict) -> float:
    """Score compliance with journey-count and min-products requirements.

    Journey count sub-score (50 %):
      - If N specified: max(0, 1 - |actual - N| / N)
      - If N not specified: 1.0 if journeys > 0, else 0.0

    Products-per-journey sub-score (50 %):
      - Per journey: min(actual_products / M, 1.0), averaged across journeys

    Returns 0.0 – 1.0.
    """
    journeys = _parse_journeys(prediction)
    if journeys is None:
        return 0.0

    required_count = extra_info.get("required_journey_count", -1)
    if hasattr(required_count, "item"):
        required_count = required_count.item()
    min_products = extra_info.get("min_products_per_journey", 8)
    if hasattr(min_products, "item"):
        min_products = min_products.item()

    num_journeys = len(journeys)

    # --- Journey count sub-score ---
    if required_count > 0:
        if num_journeys == required_count:
            count_score = 1.0
        else:
            count_score = max(0.0, 1.0 - abs(num_journeys - required_count) / required_count)
    else:
        count_score = 1.0 if num_journeys > 0 else 0.0

    # --- Products sub-score ---
    if num_journeys == 0:
        product_score = 0.0
    else:
        journey_scores = []
        for j in journeys:
            n_prods = len(j["product_tids"])
            if min_products > 0:
                journey_scores.append(min(n_prods / min_products, 1.0))
            else:
                journey_scores.append(1.0 if n_prods > 0 else 0.0)
        product_score = sum(journey_scores) / len(journey_scores)

    return 0.5 * count_score + 0.5 * product_score


def diversity_reward(prediction: str) -> float:
    """Compute product-TID diversity within each journey.

    For each journey with > 1 product:
      - Build word-sets for each ProductTID
      - Compute pairwise overlap = |intersection| / max(|A|, |B|)
      - diversity = 1 − mean(pairwise_overlap)
    Single-product journeys get diversity 1.0.

    Returns average diversity across all journeys (0.0 – 1.0).
    """
    journeys = _parse_journeys(prediction)
    if journeys is None:
        return 0.0

    journey_divs = []
    for j in journeys:
        tids = j["product_tids"]
        if len(tids) <= 1:
            journey_divs.append(1.0)
            continue

        tid_sets = [
            set(w.lower().strip() for w in tid if w.strip())
            for tid in tids
        ]

        overlaps = []
        for a_idx in range(len(tid_sets)):
            for b_idx in range(a_idx + 1, len(tid_sets)):
                a_set, b_set = tid_sets[a_idx], tid_sets[b_idx]
                denom = max(len(a_set), len(b_set))
                if denom == 0:
                    overlaps.append(0.0)
                else:
                    overlaps.append(len(a_set & b_set) / denom)

        if overlaps:
            journey_divs.append(1.0 - sum(overlaps) / len(overlaps))
        else:
            journey_divs.append(1.0)

    if not journey_divs:
        return 0.0
    return sum(journey_divs) / len(journey_divs)


# ---------- Main compute_score entry point ----------

# Weights: 0.3 instruction-following + 0.7 diversity
_W_IF = 0.3
_W_DIV = 0.7


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, float]:
    """Compute combined reward for journey generation.

    score = format * (0.3 * instruction_following + 0.7 * diversity)

    Args:
        data_source: Data source identifier (e.g. "shopping_journey").
        solution_str: Model-generated response text.
        ground_truth: Reference journey JSON (kept for logging, not used in reward).
        extra_info: Dict with required_journey_count, min_products_per_journey.

    Returns:
        Dict with score and per-component metrics.
    """
    prediction = solution_str
    info = extra_info if extra_info is not None else {}

    fmt = format_reward(prediction)
    ifr = instruction_following_reward(prediction, info)
    div = diversity_reward(prediction)

    combined = fmt * (_W_IF * ifr + _W_DIV * div)

    return {
        "score": combined,
        "format_reward": fmt,
        "instruction_following_reward": ifr,
        "diversity_reward": div,
    }
