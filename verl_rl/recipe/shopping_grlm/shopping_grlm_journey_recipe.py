"""
Shopping GRLM Journey Recipe: Dataset and Reward Functions for Journey GRPO Training.

Journey task: predict shopping journeys from user events/profile.
Model output format (JSON):
  {"ContinuedJourneys":[
    {"JourneyType":"explicit","Title":"...","Reason":"...",
     "ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]},
    ...
  ]}

Reward formula:
  score = format * (0.2*IF + 0.3*diversity + 0.5*relevance) * volume_factor

Components:
  1. Format (0 / 0.6 / 0.8 / 1.0)
     - json.loads succeeds directly → 1.0
     - succeeds after removing markdown fences → 0.8
     - succeeds via brace matching → 0.6
     - fails → 0.0

  2. Instruction Following (IF, 0–1)
     Journey count sub-score (50%):
       - N specified:   min(1, actual / N)      (monotonic, more is better)
       - N unspecified:  min(1, actual / gt_count) (use GT as soft target)
     Products sub-score (50%):
       - Per journey: min(n_products / M, 1.0), averaged across journeys

  3. Diversity (0–1)
     Per journey: 1 − mean(pairwise TID word-set overlap)
       overlap(A, B) = |A ∩ B| / max(|A|, |B|)
     Averaged across all journeys.

  4. Relevance (0–1)
     Journey-level best-match alignment between GT and prediction:
       For each GT journey g, find pred journey p with max product overlap.
       journey_relevance(g, p) = matched_products / |g.ProductTIDs|
         where a product matches if Jaccard(pred_tid, gt_tid) ≥ 0.5
       relevance = mean(journey_relevance) over all GT journeys.

  5. Volume Factor (0–1)
     min(1, total_predicted_products / 20)
     Prevents reward hacking via minimal output.

Data format (from s8_build_journey_rl_data.py parquet):
  - prompt                    : str   (JSON chat messages)
  - answer                    : str   (ground-truth journey JSON)
  - data_source               : str   ("shopping_journey")
  - task_type                 : str   ("event2journey" | "profile2journey")
  - required_journey_count    : int   (N from instruction, -1 = not specified)
  - min_products_per_journey  : int   (M from instruction)
  - gt_journey_count          : int   (number of journeys in GT)
  - gt_total_products         : int   (total products in GT)

This recipe uses standard vLLM sampling (no beam search, no CoT, no two-stage).
Thinking mode is disabled (enable_thinking=False) for Qwen3.5.
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
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
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

        # Tokenize — thinking mode disabled for direct JSON generation
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
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
            "gt_journey_count": row.pop("gt_journey_count", 0),
            "gt_total_products": row.pop("gt_total_products", 0),
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

    Returns (parsed_dict, parse_quality) where parse_quality is:
      1.0  — json.loads succeeded on raw text
      0.8  — succeeded after removing markdown fences
      0.6  — succeeded via brace matching
      None — all methods failed (returns (None, 0.0))
    """
    if not isinstance(text, str):
        return None, 0.0

    text_stripped = text.strip()
    try:
        return json.loads(text_stripped), 1.0
    except (json.JSONDecodeError, TypeError):
        pass

    # Remove markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text_stripped)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned), 0.8
    except (json.JSONDecodeError, TypeError):
        pass

    # Find first { ... } pair via brace matching
    brace_start = cleaned.find("{")
    if brace_start == -1:
        return None, 0.0
    depth = 0
    for i in range(brace_start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[brace_start : i + 1]), 0.6
                except json.JSONDecodeError:
                    return None, 0.0
    return None, 0.0


def _parse_journeys(text: str):
    """Parse journey JSON from model output.

    Returns (list_of_journey_dicts, format_score) or (None, 0.0).
    Each journey dict has keys: title, reason, journey_type, product_tids.
    """
    obj, fmt_score = _extract_json_from_text(text)
    if obj is None:
        return None, 0.0

    journeys_list = obj.get("ContinuedJourneys", [])
    if not isinstance(journeys_list, list):
        return None, 0.0

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

    if not parsed:
        return None, 0.0
    # Must have at least one journey with at least one product
    if not any(j["product_tids"] for j in parsed):
        return None, 0.0
    return parsed, fmt_score


# ---------- Individual reward components ----------


def format_reward(prediction: str) -> float:
    """Graded format reward: 1.0 / 0.8 / 0.6 / 0.0 based on parse difficulty."""
    _, fmt_score = _parse_journeys(prediction)
    return fmt_score


def instruction_following_reward(prediction: str, extra_info: dict) -> float:
    """Score compliance with journey-count and min-products requirements.

    Journey count sub-score (50%):
      - N specified:   min(1, actual / N)  — monotonic, more is not penalized
      - N unspecified: min(1, actual / gt_count)  — GT count as soft target

    Products-per-journey sub-score (50%):
      - Per journey: min(actual_products / M, 1.0), averaged across journeys

    Returns 0.0 – 1.0.
    """
    journeys, _ = _parse_journeys(prediction)
    if journeys is None:
        return 0.0

    required_count = extra_info.get("required_journey_count", -1)
    if hasattr(required_count, "item"):
        required_count = required_count.item()
    min_products = extra_info.get("min_products_per_journey", 8)
    if hasattr(min_products, "item"):
        min_products = min_products.item()
    gt_journey_count = extra_info.get("gt_journey_count", 0)
    if hasattr(gt_journey_count, "item"):
        gt_journey_count = gt_journey_count.item()

    num_journeys = len(journeys)

    # --- Journey count sub-score ---
    if required_count > 0:
        count_score = min(1.0, num_journeys / required_count)
    elif gt_journey_count > 0:
        count_score = min(1.0, num_journeys / gt_journey_count)
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
      overlap(A, B) = |A ∩ B| / max(|A|, |B|)
      diversity = 1 − mean(pairwise overlap)
    Single-product journeys get diversity 1.0.

    Returns average diversity across all journeys (0.0 – 1.0).
    """
    journeys, _ = _parse_journeys(prediction)
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


def _tid_to_wordset(tid: list[str]) -> frozenset:
    """Convert a TID word list to a lowercased frozenset for matching."""
    return frozenset(w.lower().strip() for w in tid if w.strip())


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def relevance_reward(prediction: str, ground_truth: str) -> float:
    """Journey-level relevance: best-match alignment between GT and prediction.

    For each GT journey g, find the predicted journey p with maximum
    product TID overlap. Then compute per-journey product recall:
      matched = number of g's products with a Jaccard ≥ 0.5 match in p
      journey_relevance = matched / |g.products|

    Returns mean journey_relevance over all GT journeys (0.0 – 1.0).
    """
    pred_journeys, _ = _parse_journeys(prediction)
    gt_journeys, _ = _parse_journeys(ground_truth)

    if gt_journeys is None or not gt_journeys:
        return 1.0 if pred_journeys else 0.0
    if pred_journeys is None or not pred_journeys:
        return 0.0

    # Pre-compute word-sets for all predicted products per journey
    pred_journey_sets = []
    for pj in pred_journeys:
        pred_journey_sets.append([_tid_to_wordset(tid) for tid in pj["product_tids"]])

    journey_relevances = []
    for gj in gt_journeys:
        gt_product_sets = [_tid_to_wordset(tid) for tid in gj["product_tids"]]
        if not gt_product_sets:
            journey_relevances.append(1.0)
            continue

        # Find best-matching predicted journey (max product overlap)
        best_recall = 0.0
        for p_sets in pred_journey_sets:
            if not p_sets:
                continue
            # Count matched GT products (each GT product matched at most once)
            matched = 0
            used_pred = set()
            for gt_set in gt_product_sets:
                best_j, best_idx = 0.0, -1
                for pidx, p_set in enumerate(p_sets):
                    if pidx in used_pred:
                        continue
                    j = _jaccard(gt_set, p_set)
                    if j > best_j:
                        best_j = j
                        best_idx = pidx
                if best_j >= 0.5 and best_idx >= 0:
                    matched += 1
                    used_pred.add(best_idx)
            recall = matched / len(gt_product_sets)
            if recall > best_recall:
                best_recall = recall

        journey_relevances.append(best_recall)

    return sum(journey_relevances) / len(journey_relevances)


def volume_factor(prediction: str, threshold: int = 20) -> float:
    """Penalize predictions with too few total products.

    factor = min(1, total_products / threshold)
    """
    journeys, _ = _parse_journeys(prediction)
    if journeys is None:
        return 0.0
    total = sum(len(j["product_tids"]) for j in journeys)
    if threshold <= 0:
        return 1.0
    return min(1.0, total / threshold)


# ---------- Main compute_score entry point ----------

_W_IF = 0.2
_W_DIV = 0.3
_W_REL = 0.5
_VOLUME_THRESHOLD = 20


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, float]:
    """Compute combined reward for journey generation.

    score = format * (0.2*IF + 0.3*diversity + 0.5*relevance) * volume_factor

    Args:
        data_source: Data source identifier (e.g. "shopping_journey").
        solution_str: Model-generated response text.
        ground_truth: Reference journey JSON (used for relevance reward).
        extra_info: Dict with required_journey_count, min_products_per_journey,
                    gt_journey_count, gt_total_products.

    Returns:
        Dict with score and all per-component metrics for logging.
    """
    prediction = solution_str
    info = extra_info if extra_info is not None else {}

    fmt = format_reward(prediction)
    ifr = instruction_following_reward(prediction, info)
    div = diversity_reward(prediction)
    rel = relevance_reward(prediction, ground_truth)
    vol = volume_factor(prediction, _VOLUME_THRESHOLD)

    combined = fmt * (_W_IF * ifr + _W_DIV * div + _W_REL * rel) * vol

    return {
        "score": combined,
        "format_reward": fmt,
        "instruction_following_reward": ifr,
        "diversity_reward": div,
        "relevance_reward": rel,
        "volume_factor": vol,
    }
