"""
GRLM Recipe: Dataset and Reward Functions for GRLM GRPO Training.
Fully aligned with onerec_recipe.py structure.
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
    GRLM Dataset for RL training.
    Fully aligned with OneRecDataset structure.
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

        # Configuration parameters - aligned with OneRec
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
        
        # Think mode configuration - aligned with OneRec defaults
        self.enable_think = config.get("enable_think", True)
        self.enable_nonthink = config.get("enable_nonthink", False)
        self.use_force_prefix = config.get("use_force_prefix", False)
        # GRLM uses different prefix than OneRec
        self._FORCE_PREFIX_CONTENT = "<think>\n</think>\nItem text ID:"

        if self.enable_think and self.enable_nonthink:
            raise ValueError("enable_think and enable_nonthink cannot be both True")

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
        """Read data files - aligned with OneRec pattern."""
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
        """Extract and process prompt fields - aligned with OneRec."""
        # GRLM data format: {"prompt": [{"role": "user", "content": ...}], "reward_model": {...}}
        raw_messages = row.get(self.prompt_key)
        
        if isinstance(raw_messages, str):
            # If it's a string, try to parse it
            try:
                messages = ast.literal_eval(raw_messages)
            except (ValueError, SyntaxError):
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

        # Append /think or /no_think suffix to user messages based on config
        if self.enable_think:
            for message in prompt_messages:
                if message.get("role") == "user":
                    message["content"] = message["content"] + "/think"
        if self.enable_nonthink:
            for message in prompt_messages:
                if message.get("role") == "user":
                    message["content"] = message["content"] + "/no_think"

        row[self.prompt_key] = prompt_messages

        # Ensure reward_model field exists
        if "reward_model" not in row:
            row["reward_model"] = {"style": "rule", "ground_truth": ""}

        return row

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        """Filter out prompts that are too long - aligned with OneRec."""
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            def doc_length(doc: dict[str, Any]) -> int:
                messages = self._build_messages(dict(doc))
                raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = [process_image(image) for image in doc.get(image_key, [])]
                videos = [process_video(video) for video in doc.get(video_key, [])]
                encoded = processor(text=[raw_prompt], images=images or None, videos=videos or None, return_tensors="pt")
                return int(encoded["input_ids"].shape[-1])
        else:
            def doc_length(doc: dict[str, Any]) -> int:
                messages = doc[prompt_key]
                return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

        filtered = dataframe.filter(
            lambda doc: doc_length(doc) <= self.max_prompt_length - 10,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length - 10} tokens",
        )

        logger.info("filtered dataset len: %s", len(filtered))
        return filtered

    def resume_dataset_state(self) -> None:
        """Resume dataset state for checkpoint loading."""
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning("resume with serialized dataloader, consider restarting from scratch for better perf")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Build messages from example - aligned with OneRec."""
        messages: list[dict[str, Any]] = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                segments = [segment for segment in re.split(r"(<image>|<video>)", content) if segment]
                parsed_segments = []
                for segment in segments:
                    if segment == "<image>":
                        parsed_segments.append({"type": "image"})
                    elif segment == "<video>":
                        parsed_segments.append({"type": "video"})
                    else:
                        parsed_segments.append({"type": "text", "text": segment})
                message["content"] = parsed_segments

        return messages

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get item - fully aligned with OneRec __getitem__."""
        row: dict[str, Any] = dict(self.dataframe[index])
        messages = self._build_messages(dict(row))
        model_inputs: dict[str, Any] = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if self.use_force_prefix:
                raw_prompt = raw_prompt + self._FORCE_PREFIX_CONTENT

            multi_modal_data: dict[str, Any] = {}

            images = None
            if self.image_key in row and row.get(self.image_key):
                images = [process_image(image) for image in row.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row and row.get(self.video_key):
                videos = [process_video(video) for video in row.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                return_tensors="pt",
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            row["multi_modal_data"] = multi_modal_data
            if self.return_multi_modal_inputs:
                mm_inputs = dict(model_inputs)
                mm_inputs.pop("second_per_grid_ts", None)
                row["multi_modal_inputs"] = mm_inputs
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if self.use_force_prefix:
                raw_prompt = raw_prompt + self._FORCE_PREFIX_CONTENT

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

        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = self._truncate_ids(raw_prompt_ids)

        row["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row["raw_prompt"] = messages
        if self.return_full_prompt:
            row["full_prompts"] = raw_prompt

        extra_info = row.get("extra_info", {}) or {}
        row["index"] = extra_info.get("index", index)
        row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})

        if "source" in row or "data_source" in row:
            pass
        else:
            row["data_source"] = "grlm_rec"
            logger.warning("No source/data_source field found for index %s, set to 'grlm_rec'", row["index"])

        if self.need_tools_kwargs and not row["tools_kwargs"]:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", row["index"], row.get("data_source", row.get("source", "unknown")))

        return row

    def _truncate_ids(self, token_ids: list[int]) -> list[int]:
        """Truncate token ids based on truncation mode."""
        if self.truncation == "left":
            return token_ids[-self.max_prompt_length:]
        if self.truncation == "right":
            return token_ids[:self.max_prompt_length]
        if self.truncation == "middle":
            left = self.max_prompt_length // 2
            right = self.max_prompt_length - left
            return token_ids[:left] + token_ids[-right:]
        if self.truncation == "error":
            raise RuntimeError(
                f"Prompt length {len(token_ids)} exceeds max_prompt_length={self.max_prompt_length}. "
                "Consider increasing max_prompt_length or enabling truncation."
            )
        raise ValueError(f"Unsupported truncation mode: {self.truncation}")

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()


# ============================================================================
# Reward Functions - GRLM specific parsing for "[w1, w2, ...]"
# ============================================================================

# Simple bracket pattern - matches s5_beauty_eval.py evaluation code
ITEM_PATTERN = re.compile(r"\[(.*?)\]")


def _extract_all_items(text: Any) -> list[tuple[str, ...]]:
    """Extract all item IDs from text.
    
    GRLM format: [word1, word2, ...]
    This matches s5_beauty_eval.py which uses pattern r'\[(.*?)\]'
    
    Returns list of tuples of sorted words.
    """
    if not isinstance(text, str):
        logger.warning("_extract_all_items received non-string input: %s", type(text))
        return []

    # Only parse content after </think> if present
    content_to_parse = text
    if "</think>" in text:
        content_to_parse = text.split("</think>")[-1]

    matches = ITEM_PATTERN.findall(content_to_parse)
    processed_items = []
    for m in matches:
        words = tuple(sorted([w.strip() for w in m.split(",") if w.strip()]))
        if words:
            processed_items.append(words)
    return processed_items


def think_format_reward(prediction: str) -> float:
    """Check if prediction contains valid think format.

    Args:
        prediction: Model prediction text.

    Returns:
        1.0 if contains valid <think>...</think> with content length > 10, else 0.0.
    """
    if "<think>" not in prediction or "</think>" not in prediction:
        return 0.0

    start_idx = prediction.find("<think>") + len("<think>")
    end_idx = prediction.find("</think>")

    if end_idx < start_idx:
        return 0.0

    content = prediction[start_idx:end_idx]
    content_stripped = content.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

    return 1.0 if len(content_stripped) > 10 else 0.0


def partial_hit_reward(prediction: str, ground_truth: str) -> float:
    """Calculate hierarchical matching reward with partial match support.
    
    For GRLM, we use word-level matching within item IDs.
    """
    pred_items = _extract_all_items(prediction)
    gt_items = _extract_all_items(ground_truth)

    if not pred_items or not gt_items:
        return 0.0

    total_reward = 0.0

    for pred_item in pred_items:
        max_score = 0.0
        pred_set = set(pred_item)

        for gt_item in gt_items:
            gt_set = set(gt_item)
            # Full match
            if pred_item == gt_item:
                max_score = max(max_score, 100.0)
            # Partial match: check word overlap
            elif len(pred_set & gt_set) > 0:
                overlap_ratio = len(pred_set & gt_set) / max(len(pred_set), len(gt_set))
                max_score = max(max_score, overlap_ratio * 50.0)

        total_reward += max_score

    return total_reward / len(pred_items)


def hit_reward(prediction: str, ground_truth: str) -> float:
    """Calculate hit reward: intersection ratio between prediction and ground truth."""
    if "</think>" in prediction and "<think>" in prediction:
        think_end_idx = prediction.find("</think>") + len("</think>")
        prediction = prediction[think_end_idx:]
    else:
        return 0.0

    pred_items = _extract_all_items(prediction)
    gt_items = _extract_all_items(ground_truth)
    if not pred_items or not gt_items:
        return 0.0

    pred_set = set(pred_items)
    gt_set = set(gt_items)
    return len(pred_set & gt_set) / len(pred_items)


def first_item_hit_reward(prediction: str, ground_truth: str) -> float:
    """Calculate Pass@1 reward: whether the first item after </think> hits ground truth."""
    if "</think>" in prediction and "<think>" in prediction:
        think_end_idx = prediction.find("</think>") + len("</think>")
        prediction = prediction[think_end_idx:]
    else:
        return 0.0

    pred_items = _extract_all_items(prediction)
    if not pred_items:
        return 0.0

    first_pred_item = pred_items[0]

    gt_items = _extract_all_items(ground_truth)
    if not gt_items:
        return 0.0

    gt_set = set(gt_items)

    return float(first_pred_item in gt_set)


def pass_rate(prediction: str, ground_truth: str) -> float:
    """Calculate pass rate: whether prediction and ground truth have intersection."""
    pred_items = _extract_all_items(prediction)
    gt_items = _extract_all_items(ground_truth)
    if not pred_items or not gt_items:
        return 0.0

    pred_set = set(pred_items)
    gt_set = set(gt_items)
    intersection_count = len(pred_set & gt_set)

    return float(intersection_count > 0)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, float]:
    """Compute reward scores for recommendation results.

    Args:
        data_source: Data source identifier (kept for API compatibility).
        solution_str: Model generated prediction text.
        ground_truth: Ground truth text.
        extra_info: Extra information (kept for API compatibility).

    Returns:
        Dictionary containing various reward scores.
    """
    prediction = solution_str
    format_reward_value = think_format_reward(prediction)
    partial_hit_reward_value = partial_hit_reward(prediction, ground_truth)
    hit_reward_value = hit_reward(prediction, ground_truth)
    pass_rate_value = pass_rate(prediction, ground_truth)
    pass_at_1_value = first_item_hit_reward(prediction, ground_truth)

    return {
        "score": pass_at_1_value,  # Main reward signal for GRPO
        "format_reward": format_reward_value,
        "partial_hit_reward": partial_hit_reward_value,
        "hit_reward": hit_reward_value,
        "pass_rate": pass_rate_value,
        "pass_at_1": pass_at_1_value,
    }
