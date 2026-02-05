"""
GRLM RL Data Builder V2

This script builds RL training data for GRPO, following:
1. Leave-One-Out strategy (consistent with SFT evaluation)
2. Random sampling for test set (following OpenOneRec style)
3. Ground truth aligned with test_ground_truth (the last item in sequence)

Key differences from v1:
- ground_truth is now the test_ground_truth_tid (last item), not the output (intermediate items)
- This aligns RL reward with SFT evaluation metric
- Test set is randomly sampled (like OpenOneRec), not sequential split
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
    Build RL prompt that matches SFT training setup:
    - Prompt: instruction + input + output + valid_ground_truth (完整序列)
    - 不以 'Item text ID: ' 结尾，因为 SFT 时模型学会生成完整的 'Item text ID: [...]'
    
    SFT 训练时:
      Prompt = instruction + input
      Output = 'Item text ID: [...] Title: ...\nItem text ID: [...] Title: ...'
    
    RL 训练时 (保持一致):
      Prompt = instruction + input + output + valid_gt (完整格式)
      模型生成 = 'Item text ID: [...] Title: ...' (完整格式)
    """
    # Build prompt same as SFT format
    prompt = item["instruction"]
    prompt += item["input"]
    prompt += item["output"]
    
    # Add valid_ground_truth (second-to-last item) - 完整格式
    prompt += "Item text ID: " + format_tid_as_text(item["valid_ground_truth_tid"])
    if "title" in item.get("valid_ground_truth_msg", {}):
        prompt += f" Title: {item['valid_ground_truth_msg']['title']}.\n"
    else:
        prompt += " Title: None.\n"
    
    # 不添加 "Item text ID: "，让模型生成完整的格式
    # 这与 SFT 训练时模型学会的格式一致
    
    return prompt


def build_ground_truth(item):
    """
    Build ground truth that matches SFT output format:
    - 完整格式: 'Item text ID: [w1, w2, w3, w4, w5]'
    
    这与 SFT 训练时模型学会生成的格式一致:
      'Item text ID: [...] Title: ...'
    
    Reward 函数使用: ITEM_PATTERN = re.compile(r"Item text ID:\s*\[(.*?)\]")
    """
    return "Item text ID: " + format_tid_as_text(item["test_ground_truth_tid"])


def main():
    parser = argparse.ArgumentParser(description="Build RL data for GRLM GRPO training")
    parser.add_argument("--dataset", type=str, default="beauty", 
                        help="Dataset name: beauty, sports, toys")
    parser.add_argument("--test_size", type=int, default=1000,
                        help="Number of samples for test set (like OpenOneRec)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_format", type=str, default="both",
                        choices=["parquet", "json", "both"],
                        help="Output format")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_file = f"../LlamaFactory/data/grlm_in_domain/amazon_{args.dataset}_sft_data_rec.json"
    output_dir = f"./{args.dataset}/rl_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GRLM RL Data Builder V2")
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
    
    # Convert to RL format
    print("\nConverting to RL format...")
    formatted_data = []
    skipped = 0
    
    for item in tqdm(data, desc="Processing"):
        # Validate required fields
        if not all(key in item for key in ["instruction", "input", "output", 
                                            "valid_ground_truth_tid", "test_ground_truth_tid",
                                            "test_ground_truth_id"]):
            skipped += 1
            continue
        
        # Build prompt (matches evaluation setup)
        prompt_content = build_rl_prompt(item)
        
        # Build ground truth (the last item's Term IDs)
        ground_truth = build_ground_truth(item)
        
        formatted_item = {
            "data_source": f"amazon_{args.dataset}",
            "source": f"amazon_{args.dataset}",  # Alternative key name
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            "ability": "rec",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth  # Now aligned with test_ground_truth!
            },
            "extra_info": {
                "user_id": item["metadata"]["user_id"],
                "test_ground_truth_id": item["test_ground_truth_id"],
                "test_ground_truth_tid": item["test_ground_truth_tid"],
                "valid_ground_truth_id": item["valid_ground_truth_id"],
                "valid_ground_truth_tid": item["valid_ground_truth_tid"],
            }
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
        # Random sampling for test set (like OpenOneRec)
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
        # Save as parquet (for verl)
        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)
        
        train_parquet = os.path.join(output_dir, "train.parquet")
        test_parquet = os.path.join(output_dir, "test.parquet")
        
        df_train.to_parquet(train_parquet, engine='pyarrow', index=False, compression='snappy')
        df_test.to_parquet(test_parquet, engine='pyarrow', index=False, compression='snappy')
        
        print(f"Saved: {train_parquet} ({len(df_train)} samples)")
        print(f"Saved: {test_parquet} ({len(df_test)} samples)")
    
    if args.output_format in ["json", "both"]:
        # Save as JSON (for debugging/inspection)
        train_json = os.path.join(output_dir, "train.json")
        test_json = os.path.join(output_dir, "test.json")
        
        with open(train_json, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(test_json, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved: {train_json}")
        print(f"Saved: {test_json}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total samples: {len(formatted_data)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Train/Test ratio: {len(train_data)/len(formatted_data)*100:.1f}% / {len(test_data)/len(formatted_data)*100:.1f}%")
    print("=" * 60)
    
    # Show sample data
    if formatted_data:
        print("\nSample RL data entry:")
        sample = formatted_data[0]
        print(f"  data_source: {sample['data_source']}")
        print(f"  prompt (truncated): {sample['prompt'][0]['content'][:200]}...")
        print(f"  ground_truth: {sample['reward_model']['ground_truth']}")
        print(f"  test_ground_truth_id: {sample['extra_info']['test_ground_truth_id']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
