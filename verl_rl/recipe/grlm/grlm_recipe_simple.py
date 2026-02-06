"""
GRLM Recipe (Simple Version): Dataset and Reward Functions for GRLM GRPO Training.

This version is designed for GRLM models that were NOT trained with CoT (Chain of Thought).
It uses direct generation without think/no_think modes.

Key differences from grlm_recipe.py:
1. No think mode configuration - GRLM outputs directly
2. Simpler prompt processing (no /think or /no_think suffix)
3. Reward functions that work on direct output (no </think> parsing needed)
4. Reward aligned with EasyR1's grlm_rec.py (pass_at_1 with Jaccard, hierarchical)

Data format (from s7_build_rl_data.py):
- prompt: instruction + input + output (ends at itemN-2)
- answer: "Item text ID: [w1, w2, w3, w4, w5]" (valid_ground_truth, itemN-1)
- Model generates: next item prediction for reward computation
- test_ground_truth (itemN) is held out for s5_beauty_eval.py
"""

from __future__ import annotations

import ast
import copy
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

__all__ = ["collate_fn", "GrlmDataset", "compute_score"]


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


class GrlmDataset(Dataset):
    """
    GRLM Dataset for RL training - Simple version without think mode.
    """

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

        # Configuration parameters
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

    def _download(self, use_origin_parquet: bool = False) -> None:
        from verl.utils.fs import copy_to_local

        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, parquet_file in enumerate(target_files):
            local_path = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
            target_files[idx] = local_path

        if use_origin_parquet:
            self.data_files = target_files

    def _read_files_and_tokenize(self) -> None:
        """Read data files."""
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

    def _extract_prompt_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        """Extract and process prompt fields - simple version without think mode."""
        raw_messages = row.get(self.prompt_key)
        
        if isinstance(raw_messages, str):
            try:
                messages = ast.literal_eval(raw_messages)
            except (ValueError, SyntaxError):
                # Plain string prompt - wrap as user message
                messages = [{"role": "user", "content": raw_messages}]
        elif raw_messages is None:
            raise ValueError("Sample has empty prompt; please check data integrity.")
        else:
            messages = raw_messages

        # Ensure messages is a list of dicts
        if isinstance(messages, list) and len(messages) > 0:
            prompt_messages = messages
        else:
            raise ValueError(f"Invalid messages format: {type(messages)}")

        # NO think/no_think suffix - GRLM was not trained with CoT
        row[self.prompt_key] = prompt_messages

        # Ensure reward_model field exists
        if "reward_model" not in row:
            row["reward_model"] = {"style": "rule", "ground_truth": ""}

        return row

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        """Filter out prompts that are too long."""
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key
        max_length = self.max_prompt_length

        def filter_fn(row):
            messages = row[prompt_key]
            images = row.get(image_key, None)
            videos = row.get(video_key, None)
            
            if processor is not None and (images or videos):
                kwargs = {}
                if images:
                    kwargs["images"] = images
                if videos:
                    kwargs["videos"] = videos
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=text, **kwargs, return_tensors="pt")
                length = inputs["input_ids"].shape[-1]
            else:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                length = len(tokenizer.encode(prompt_text))

            return length <= max_length

        original_len = len(dataframe)
        dataframe = dataframe.filter(filter_fn, num_proc=self.num_workers, desc="Filter long prompts")
        filtered_len = len(dataframe)
        
        if original_len != filtered_len:
            logger.info(f"Filtered {original_len - filtered_len} samples exceeding max_prompt_length={max_length}")

        return dataframe

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe[index]

        messages = row[self.prompt_key]
        
        # Apply chat template
        if self.processor is not None:
            images = row.get(self.image_key, None)
            videos = row.get(self.video_key, None)
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            kwargs = {}
            if images:
                kwargs["images"] = images
            if videos:
                kwargs["videos"] = videos
            inputs = self.processor(text=prompt, **kwargs, return_tensors="pt")
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)

        # Truncation handling
        if input_ids.shape[-1] > self.max_prompt_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_prompt_length:]
                attention_mask = attention_mask[-self.max_prompt_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_prompt_length]
                attention_mask = attention_mask[:self.max_prompt_length]
            elif self.truncation == "error":
                raise ValueError(
                    f"Prompt length {input_ids.shape[-1]} > max_prompt_length {self.max_prompt_length}. "
                    "Consider increasing max_prompt_length or enabling truncation."
                )

        position_ids = compute_position_id_with_mask(attention_mask)

        # Build output sample
        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        # Add ground truth for reward function
        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, str):
            try:
                reward_model = ast.literal_eval(reward_model)
            except (ValueError, SyntaxError):
                reward_model = {}

        ground_truth = reward_model.get("ground_truth", row.get("answer", ""))
        sample["ground_truth"] = ground_truth

        # Add data source
        sample["source"] = row.get("source", row.get("data_source", "grlm"))

        return sample

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()


# ============================================================================
# Reward Functions - Aligned with EasyR1's grlm_rec.py
# ============================================================================

# Pattern to extract item IDs: [word1, word2, ...]
ITEM_PATTERN = re.compile(r"\[(.*?)\]")


def _extract_all_items(text: Any) -> list[tuple[str, ...]]:
    """Extract all item IDs from text.
    
    GRLM format: [word1, word2, word3, word4, word5]
    Returns list of tuples, preserving original word order for hierarchical matching.
    """
    if not isinstance(text, str):
        return []

    matches = ITEM_PATTERN.findall(text)
    processed_items = []
    for m in matches:
        # Keep original order for hierarchical matching
        words = tuple([w.strip().lower() for w in m.split(",") if w.strip()])
        if words:
            processed_items.append(words)
    return processed_items


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def format_reward(prediction: str) -> float:
    """Check if prediction contains valid Item text ID format."""
    if "Item text ID:" not in prediction and "[" not in prediction:
        return 0.0
    items = _extract_all_items(prediction)
    return 1.0 if items else 0.0


def pass_at_1_reward(prediction: str, ground_truth: str) -> float:
    """Calculate Pass@1 reward with Jaccard similarity.
    
    Matches EasyR1's grlm_rec.py implementation.
    """
    pred_items = _extract_all_items(prediction)
    gt_items = _extract_all_items(ground_truth)
    
    if not pred_items or not gt_items:
        return 0.0
    
    first_pred_item = pred_items[0]
    gt_item = gt_items[0]
    
    # Exact match
    if first_pred_item == gt_item:
        return 1.0
    
    # Partial match: Jaccard similarity
    pred_set = set(first_pred_item)
    gt_set = set(gt_item)
    
    return jaccard_similarity(pred_set, gt_set)


def hierarchical_reward(prediction: str, ground_truth: str) -> float:
    """Calculate sequential hierarchical matching reward.
    
    Match words sequentially from the first position:
    - 0 words match: 0.0
    - 1 word matches: 0.6
    - 2 words match: 0.7
    - 3 words match: 0.8
    - 4 words match: 0.9
    - 5 words match: 1.0
    
    This matches EasyR1's grlm_rec.py hierarchical_reward implementation.
    """
    pred_items = _extract_all_items(prediction)
    gt_items = _extract_all_items(ground_truth)
    
    if not pred_items or not gt_items:
        return 0.0
    
    first_pred_item = pred_items[0]
    gt_item = gt_items[0]
    
    # Count consecutive matches starting from the first word
    consecutive_matches = 0
    min_len = min(len(first_pred_item), len(gt_item))
    
    for i in range(min_len):
        if first_pred_item[i] == gt_item[i]:
            consecutive_matches += 1
        else:
            break
    
    # Sequential hierarchical reward
    if consecutive_matches == 0:
        return 0.0
    elif consecutive_matches == 1:
        return 0.6
    elif consecutive_matches == 2:
        return 0.7
    elif consecutive_matches == 3:
        return 0.8
    elif consecutive_matches == 4:
        return 0.9
    else:
        return 1.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, float]:
    """Compute reward scores for recommendation results.
    
    This function matches the verl reward function interface.
    
    Args:
        data_source: Data source identifier
        solution_str: Model generated prediction text
        ground_truth: Ground truth text
        extra_info: Extra information
        
    Returns:
        Dictionary containing various reward scores.
    """
    prediction = solution_str
    
    # Calculate rewards
    format_score = format_reward(prediction)
    jaccard_score = pass_at_1_reward(prediction, ground_truth)
    hierarchical_score = hierarchical_reward(prediction, ground_truth)
    
    # Use hierarchical reward as main signal (like EasyR1 config)
    # Format weight = 0.1
    format_weight = 0.1
    main_reward = hierarchical_score
    overall = main_reward + format_weight * format_score
    
    return {
        "score": overall,  # Main reward signal for GRPO
        "format_reward": format_score,
        "jaccard_reward": jaccard_score,
        "hierarchical_reward": hierarchical_score,
        "pass_at_1": jaccard_score,
    }
