"""GRLM EasyR1 Data Builder

This script builds RL training data for EasyR1 GRPO training.
Converts GRLM SFT data format to EasyR1 expected format.

EasyR1 expects:
- prompt: str (user message content)
- answer: str (ground truth for reward function)

Key features:
1. Leave-One-Out strategy for RL training (NO DATA LEAKAGE)
2. Random sampling for test set (following OpenOneRec style)
3. Ground truth uses valid_ground_truth (2nd-to-last item) to avoid test data leakage

Data Split Strategy:
- Original sequence: [item1, item2, ..., itemN-2, itemN-1 (valid), itemN (test)]
- RL Prompt: instruction + input + output (ends at itemN-2)
- RL Ground Truth: valid_ground_truth (itemN-1) - used for reward computation
- Test Evaluation: test_ground_truth (itemN) - held out for s5_beauty_eval.py
"""

import json
import os
import argparse
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


def format_tid_as_text(tid_list):
    """Format Term ID list as text: [w1, w2, w3, w4, w5]"""
    if isinstance(tid_list, list):
        return "[" + ", ".join(tid_list) + "]"
    return str(tid_list)


def build_rl_prompt(item):
    """
    Build RL prompt for training (NO DATA LEAKAGE):
    - Prompt: instruction + input + output (ends at itemN-2)
    - Model should predict: valid_ground_truth (itemN-1)
    
    This leaves test_ground_truth (itemN) completely held out for evaluation.
    
    SFT training format:
      Prompt = instruction + input
      Output = items up to itemN-2
    
    RL training format:
      Prompt = instruction + input + output (same as SFT prompt + partial output)
      Model generates = 'Item text ID: [...] Title: ...' for valid_ground_truth
    """
    # Build prompt: instruction + input + output
    # Note: output already ends at itemN-2, does NOT include valid_ground_truth
    prompt = item["instruction"]
    prompt += item["input"]
    prompt += item["output"]
    
    # DO NOT add valid_ground_truth to prompt - that's what the model should predict!
    # The model will generate the next item prediction
    
    return prompt


def build_ground_truth(item):
    """
    Build ground truth using valid_ground_truth (NO DATA LEAKAGE):
    - Uses valid_ground_truth_tid (itemN-1), NOT test_ground_truth_tid (itemN)
    - Complete format: 'Item text ID: [w1, w2, w3, w4, w5]'
    
    This ensures test_ground_truth remains held out for final evaluation.
    Reward function uses: ITEM_PATTERN = re.compile(r"Item text ID:\s*\[(.*?)\]")
    """
    return "Item text ID: " + format_tid_as_text(item["valid_ground_truth_tid"])


def main():
    parser = argparse.ArgumentParser(description="Build EasyR1 RL data for GRLM GRPO training")
    parser.add_argument("--dataset", type=str, default="beauty", 
                        help="Dataset name: beauty, sports, toys")
    parser.add_argument("--test_size", type=int, default=1000,
                        help="Number of samples for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_format", type=str, default="both",
                        choices=["parquet", "json", "both"],
                        help="Output format")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Custom input file path (overrides --dataset)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory (overrides default)")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine paths
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = f"../LlamaFactory/data/grlm_in_domain/amazon_{args.dataset}_sft_data_rec.json"
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./{args.dataset}/rl_data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GRLM EasyR1 Data Builder")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Test size: {args.test_size}")
    print(f"Seed: {args.seed}")
    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    
    # Load SFT data
    print(f"\nLoading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Convert to EasyR1 format
    print("\nConverting to EasyR1 format...")
    formatted_data = []
    skipped = 0
    
    for item in tqdm(data, desc="Processing"):
        # Validate required fields
        required_fields = ["instruction", "input", "output", 
                          "valid_ground_truth_tid", "test_ground_truth_tid",
                          "test_ground_truth_id"]
        if not all(key in item for key in required_fields):
            skipped += 1
            continue
        
        # Build prompt (ends at itemN-2, model predicts itemN-1)
        prompt_content = build_rl_prompt(item)
        
        # Build ground truth (valid_ground_truth = itemN-1, NOT test!)
        ground_truth = build_ground_truth(item)
        
        # EasyR1 format: simple dict with prompt (str) and answer (str)
        # EasyR1's RLHFDataset will wrap prompt into messages
        formatted_item = {
            "prompt": prompt_content,
            "answer": ground_truth,  # EasyR1 uses 'answer' key which maps to ground_truth
            # Optional metadata (will be preserved in non_tensor_batch)
            "data_source": f"amazon_{args.dataset}",
            "user_id": item["metadata"]["user_id"] if "metadata" in item else None,
            # RL training ground truth (itemN-1) - used for reward computation
            "valid_ground_truth_id": item.get("valid_ground_truth_id"),
            "valid_ground_truth_tid": item["valid_ground_truth_tid"],
            # Held-out test ground truth (itemN) - for final evaluation only
            "test_ground_truth_id": item["test_ground_truth_id"],
            "test_ground_truth_tid": item["test_ground_truth_tid"],
        }
        formatted_data.append(formatted_item)
    
    print(f"Converted {len(formatted_data)} samples, skipped {skipped}")
    
    # Random split (following OpenOneRec style)
    print(f"\nSplitting data: test_size={args.test_size}")
    
    if args.test_size >= len(formatted_data):
        print(f"Warning: test_size ({args.test_size}) >= total samples ({len(formatted_data)})")
        print("Using all data as test set, training set will be empty")
        test_data = formatted_data
        train_data = []
    else:
        # Random sampling for test set
        indices = list(range(len(formatted_data)))
        random.shuffle(indices)
        
        test_indices = set(indices[:args.test_size])
        
        train_data = [formatted_data[i] for i in range(len(formatted_data)) if i not in test_indices]
        test_data = [formatted_data[i] for i in indices[:args.test_size]]
        
        # Shuffle both sets
        random.shuffle(train_data)
        random.shuffle(test_data)
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Save data
    print("\nSaving data...")
    
    if args.output_format in ["parquet", "both"]:
        # Save as parquet (recommended for EasyR1)
        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)
        
        train_parquet = os.path.join(output_dir, "train.parquet")
        test_parquet = os.path.join(output_dir, "test.parquet")
        
        df_train.to_parquet(train_parquet, engine='pyarrow', index=False, compression='snappy')
        df_test.to_parquet(test_parquet, engine='pyarrow', index=False, compression='snappy')
        
        print(f"Saved: {train_parquet} ({len(df_train)} samples)")
        print(f"Saved: {test_parquet} ({len(df_test)} samples)")
    
    if args.output_format in ["json", "both"]:
        # Save as JSONL (alternative format)
        train_jsonl = os.path.join(output_dir, "train.jsonl")
        test_jsonl = os.path.join(output_dir, "test.jsonl")
        
        with open(train_jsonl, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(test_jsonl, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved: {train_jsonl} ({len(train_data)} samples)")
        print(f"Saved: {test_jsonl} ({len(test_data)} samples)")
    
    # Print sample
    print("\n" + "=" * 60)
    print("Sample data (EasyR1 format):")
    print("=" * 60)
    if formatted_data:
        sample = formatted_data[0]
        print(f"prompt (first 500 chars):\n{sample['prompt'][:500]}...")
        print(f"\nanswer: {sample['answer']}")
        print(f"data_source: {sample['data_source']}")
    
    print("\n" + "=" * 60)
    print("EasyR1 Data Format Notes:")
    print("=" * 60)
    print("1. 'prompt' field: plain text, EasyR1 wraps it in messages automatically")
    print("2. 'answer' field: maps to 'ground_truth' in reward function")
    print("3. Reward function receives: {'response': str, 'ground_truth': str}")
    print("4. Use parquet files with EasyR1's data config")
    print("")
    print("DATA SPLIT STRATEGY (NO DATA LEAKAGE):")
    print("- Prompt: instruction + input + output (ends at itemN-2)")
    print("- RL Ground Truth: valid_ground_truth (itemN-1) - used for reward")
    print("- Test Evaluation: test_ground_truth (itemN) - held out for s5_beauty_eval.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
