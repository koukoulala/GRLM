"""Step 5.4: Batch SLM Inference from Prompt JSONL

Reads a prompt JSONL file where each line has:
  {"stableid": "...", "request_body": {"sys_prompt": "", "prompt": "...",
   "max_new_tokens": 4096, "temperature": 0}}

Runs vLLM inference on all prompts (with chat template), then outputs a
JSONL file where each line has:
  {"stableid": "...", "prompt": "<rewritten_prompt>",
   "slm_output": {ContinuedJourneys JSON dict}, "elapsed_s": float}

The output prompt is rewritten to use "an appropriate number of" journeys
and "at least <min_products> products" (configurable, default 20).

Usage:
    python s5_4_batch_slm_inference.py \\
        --model_path /path/to/checkpoint \\
        --input_file /path/to/sample_user_prompt_output.jsonl \\
        --output_file_name sample_user_batch_slm_inference_output.jsonl
"""

import os
import re
import sys
import json
import time
import argparse

SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "resources"))



# =============================================================================
# JSON Parsing (from s5_2)
# =============================================================================

def parse_journey_json(raw):
    """Parse ContinuedJourneys JSON from raw SLM output text."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Find the first '{' and matching '}'
    bs = text.find("{")
    if bs == -1:
        return None
    depth = 0
    be = -1
    for i in range(bs, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                be = i
                break

    cand = text[bs:be + 1] if be != -1 else text[bs:] + "}"
    for t in [cand, text]:
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


# =============================================================================
# vLLM Inference
# =============================================================================

def run_vllm_inference(prompts, model_path, num_gpus, gpu_mem, max_model_len,
                       max_tokens, temperature=0.7, top_p=0.8, top_k=20):
    """Run batched vLLM inference on formatted chat prompts."""
    from vllm import LLM, SamplingParams

    print(f"\nInitializing vLLM ...")
    print(f"  Model: {model_path}")
    print(f"  TP: {num_gpus}, GPU mem: {gpu_mem}")
    print(f"  max_model_len: {max_model_len}, max_tokens: {max_tokens}")
    print(f"  temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=SEED,
        dtype="bfloat16",
        enforce_eager=True,
    )
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # Truncate prompts that exceed max input length
    _tok = llm.get_tokenizer()
    max_input = max_model_len - max_tokens
    truncated = 0
    for i, p in enumerate(prompts):
        tok_ids = _tok.encode(p)
        if len(tok_ids) > max_input:
            prompts[i] = _tok.decode(tok_ids[:max_input],
                                     skip_special_tokens=False)
            truncated += 1
    if truncated:
        print(f"  WARNING: Truncated {truncated}/{len(prompts)} prompts "
              f"to fit max_model_len={max_model_len}")

    print(f"  Running inference on {len(prompts)} prompts ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} items/s)")

    return outputs, elapsed


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch SLM inference from prompt JSONL"
    )
    p.add_argument(
        "--model_path", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Results/qwen3-5-9b_full_v4_step2/checkpoint-1840",
        help="Path to the trained SFT model checkpoint",
    )
    p.add_argument(
        "--input_file", type=str, 
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/EvalData/vip_user/sample_user_data/sample_user_prompt_output.jsonl",
        help="Path to input JSONL file (sample_user_prompt_output.jsonl)",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: same as input file directory)",
    )
    p.add_argument(
        "--output_file_name", type=str,
        default="sample_user_batch_slm_inference_output_full_1840.jsonl",
        help="Output filename",
    )
    p.add_argument("--num_gpus", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32000)
    p.add_argument("--max_tokens", type=int, default=12000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()

    # GPU setup
    if args.num_gpus is None:
        import torch
        args.num_gpus = max(
            torch.cuda.device_count() if torch.cuda.is_available() else 1, 1
        )

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input_file))
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file_name)

    print("=" * 70)
    print("Step 5.4: Batch SLM Inference")
    print("=" * 70)
    print(f"  Model:       {args.model_path}")
    print(f"  Input:       {args.input_file}")
    print(f"  Output:      {output_path}")
    print(f"  GPUs:        {args.num_gpus}")
    print()

    # =========================================================================
    # Step 1: Load input JSONL
    # =========================================================================
    print("Step 1: Loading input prompts ...")
    records = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping line {line_num}: {e}")
    print(f"  Loaded {len(records)} records")

    if not records:
        print("ERROR: No records found in input file.")
        sys.exit(1)

    # =========================================================================
    # Step 2: Extract and optionally rewrite prompts
    # =========================================================================
    print("\nStep 2: Preparing prompts ...")
    stable_ids = []
    user_prompts = []  # the text prompt for each user

    for rec in records:
        sid = rec.get("stableid", "")
        rb = rec.get("request_body", {})
        prompt = rb.get("prompt", "")

        stable_ids.append(sid)
        user_prompts.append(prompt)

    print(f"  Prepared {len(user_prompts)} prompts")

    # =========================================================================
    # Step 3: Apply chat template
    # =========================================================================
    print("\nStep 3: Applying chat template ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    chat_prompts = []
    for prompt_text in user_prompts:
        msgs = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chat_prompts.append(formatted)
    print(f"  Built {len(chat_prompts)} chat-formatted prompts")

    # =========================================================================
    # Step 4: Run vLLM inference
    # =========================================================================
    outputs, total_elapsed = run_vllm_inference(
        chat_prompts,
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        gpu_mem=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    per_item_elapsed = total_elapsed / len(records) if records else 0
    raw_outputs = [o.outputs[0].text.strip() for o in outputs]

    # =========================================================================
    # Step 5: Parse outputs and write result JSONL
    # =========================================================================
    print(f"\nStep 5: Parsing outputs and writing results ...")
    parse_ok = 0
    parse_fail = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (sid, prompt_text, raw_text) in enumerate(
            zip(stable_ids, user_prompts, raw_outputs)
        ):
            parsed = parse_journey_json(raw_text)

            if parsed is not None:
                parse_ok += 1
                slm_output = parsed
            else:
                parse_fail += 1
                # Store raw text as fallback so nothing is lost
                slm_output = {"raw_output": raw_text}

            result = {
                "stableid": sid,
                "prompt": prompt_text,
                "slm_output": slm_output,
                "elapsed_s": round(per_item_elapsed, 1),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"  JSON parse success: {parse_ok}/{len(records)}")
    if parse_fail > 0:
        print(f"  JSON parse failed:  {parse_fail}/{len(records)} "
              f"(stored as raw_output)")
    print(f"\n  Output saved to: {output_path}")
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"  Total records:   {len(records)}")
    print(f"  Inference time:  {total_elapsed:.1f}s "
          f"({len(records) / total_elapsed:.1f} items/s)")
    print(f"  Per-item time:   {per_item_elapsed:.1f}s")
    print(f"  Parse success:   {parse_ok} ({parse_ok / len(records) * 100:.1f}%)")
    print(f"  Output file:     {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
