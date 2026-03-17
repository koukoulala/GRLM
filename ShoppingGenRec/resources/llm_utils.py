#!/usr/bin/env python3
"""
LLM Utilities Module

This module provides common utility functions for working with GitHub Copilot
LLM APIs, prompt loading, parallel text generation, and checkpoint/resume
support. These utilities are shared across multiple tasks for batch LLM
inference.

Usage:
    from llm_utils import load_prompts, run_llm_parallel_with_checkpoint

    # Load prompt templates
    prompts = load_prompts('prompts.yaml')

    # Prepare inputs: list of (idx, prompt) tuples
    inputs = [("0", "Hello, world!"), ("1", "What is AI?")]

    # Run parallel LLM calls with checkpoint/resume
    results = run_llm_parallel_with_checkpoint(
        inputs=inputs,
        token_file='tokens.txt',
        checkpoint_dir='./checkpoints/',
        num_workers=20,
        model="gpt-5.4",
    )
    # results: list of (idx, response_text) tuples
"""

import os
import json
import random
import shutil
import time
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_tokens(token_file):
    """
    Load all tokens from token file.

    Args:
        token_file: Path to token file (one token per line)

    Returns:
        list[str]: List of tokens

    Raises:
        FileNotFoundError: If token file doesn't exist
        ValueError: If no valid tokens found in file
    """
    if not os.path.exists(token_file):
        raise FileNotFoundError(f"Token file not found: {token_file}")

    with open(token_file, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]

    if not tokens:
        raise ValueError(f"No valid tokens found in {token_file}")

    print(f"  Loaded {len(tokens)} tokens from {token_file}")
    return tokens


def validate_tokens(tokens, model="gpt-5.2"):
    """
    Validate tokens by:
      1. Checking if the token can obtain a Copilot API token
      2. Sending a minimal test request to verify the model is accessible

    Filters out suspended, invalid, or model-incompatible tokens.

    Args:
        tokens: List of GitHub access tokens.
        model: Model name to test against (default: "gpt-5.2").

    Returns:
        list[str]: List of valid tokens that passed all checks.

    Raises:
        ValueError: If no valid tokens remain after filtering.
    """
    print(f"  [TOKEN] Validating {len(tokens)} tokens (model={model}) ...")
    valid = []
    for i, token in enumerate(tokens):
        # Step 1: check Copilot token
        try:
            copilot_token = get_copilot_token(token)
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  [TOKEN] Line {i+1} INVALID (auth): {token[:12]}... -> {error_msg}")
            continue

        # Step 2: test a minimal API call with the target model
        try:
            url = "https://api.githubcopilot.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {copilot_token}",
                "User-Agent": "GitHub-Copilot-Client/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.85.0",
                "Editor-Plugin-Version": "copilot/1.155.0",
                "X-GitHub-Api-Version": "2023-07-07"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream": False,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                valid.append(token)
            else:
                error_body = resp.text[:120]
                print(f"  [TOKEN] Line {i+1} INVALID (model): {token[:12]}... "
                      f"-> {resp.status_code} {error_body}")
        except Exception as e:
            print(f"  [TOKEN] Line {i+1} INVALID (test): {token[:12]}... -> {e}")

    print(f"  [TOKEN] {len(valid)}/{len(tokens)} tokens are valid for model={model}")

    if not valid:
        raise ValueError("No valid tokens remaining after validation. "
                         "Please update tokens.txt with active tokens.")
    return valid


def load_prompts(prompts_file):
    """
    Load prompt templates from YAML file

    Args:
        prompts_file: Path to prompts.yaml file

    Returns:
        dict: Dictionary containing prompt templates

    Raises:
        FileNotFoundError: If prompts file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = yaml.safe_load(f)
    return prompts_data['prompts']


def get_copilot_token(access_token):
    """
    Get Copilot-specific token for API access.

    Args:
        access_token: GitHub access token

    Returns:
        str: Copilot token

    Raises:
        Exception: If token retrieval fails
    """
    url = "https://api.github.com/copilot_internal/v2/token"

    headers = {
        "Authorization": f"token {access_token}",
        "User-Agent": "GitHub-Copilot-Client/1.0"
    }

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code == 200:
        return response.json()['token']
    else:
        raise Exception(f"Failed to get Copilot token: {response.status_code} - {response.text}")


def call_llm_api(prompt, access_token, system_prompt="", model="gpt-4o",
                 temperature=0.5, max_tokens=2000, timeout=300, max_retries=3):
    """
    Make a call to the LLM API with retry logic.

    Args:
        prompt: User prompt text
        access_token: GitHub access token
        system_prompt: System prompt text (default: "")
        model: Model name to use (default: "gpt-4o")
        temperature: Temperature parameter (default: 0.5)
        max_tokens: Maximum tokens for response (default: 2000)
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum number of retries on failure (default: 3)

    Returns:
        str: Response text from the LLM

    Raises:
        Exception: If API call fails after all retries
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            copilot_token = get_copilot_token(access_token)

            url = "https://api.githubcopilot.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {copilot_token}",
                "User-Agent": "GitHub-Copilot-Client/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.85.0",
                "Editor-Plugin-Version": "copilot/1.155.0",
                "X-GitHub-Api-Version": "2023-07-07"
            }

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "n": 1,
                "top_p": 1
            }

            response = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if response.status_code != 200:
                # Truncate error body; replace HTML error pages with short msg
                if response.text.strip().startswith(("<!DOCTYPE", "<html", "<!--")):
                    error_body = "(HTML error page)"
                else:
                    error_body = response.text[:150]
                raise Exception(
                    f"{response.status_code} {response.reason} - {error_body}"
                )

            result = response.json()
            response_text = result['choices'][0]['message']['content']
            return response_text

        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = 10 + random.random() * 2
                print(f"    [RETRY] Attempt {attempt}/{max_retries} failed: {e}. "
                      f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

    raise last_exception


def _process_single_item(idx, prompt, tokens, system_prompt, model,
                         temperature, max_tokens, timeout, max_retries):
    """
    Process a single (idx, prompt) item by calling the LLM API.
    Randomly selects a token from the pool for each call.

    Returns:
        tuple: (idx, response_text) on success, (idx, "") on failure
    """
    access_token = random.choice(tokens)
    try:
        response_text = call_llm_api(
            prompt=prompt,
            access_token=access_token,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        return (idx, response_text)
    except Exception as e:
        print(f"    [ERROR] Failed for idx={idx}: {e}")
        return (idx, "")


def run_llm_parallel(inputs, token_file=None, num_workers=4, system_prompt="",
                     model="gpt-4o", temperature=0.5, max_tokens=2000,
                     timeout=300, max_retries=3, _tokens=None, token_evaluate=False):
    """
    Run LLM calls in parallel over a list of (idx, prompt) inputs.

    Each worker randomly selects a token from the token pool for its API call,
    distributing load across multiple tokens.

    Args:
        inputs: list of (idx, prompt) tuples. idx is a string identifier,
                prompt is the user prompt text.
        token_file: Path to token file (one token per line).
        num_workers: Number of parallel threads (default: 4)
        system_prompt: System prompt for all calls (default: "")
        model: Model name (default: "gpt-4o")
        temperature: Temperature (default: 0.5)
        max_tokens: Max tokens per response (default: 2000)
        timeout: Timeout per request in seconds (default: 300)
        max_retries: Max retries per request (default: 3)
        _tokens: Pre-validated token list (internal use). If provided,
                 skips loading and validation from token_file.

    Returns:
        list of (idx, response_text) tuples, in the same order as inputs.
        response_text is an empty string for failed calls.
    """
    if _tokens:
        tokens = _tokens
    else:
        tokens = load_tokens(token_file)
        if token_evaluate:
            tokens = validate_tokens(tokens, model=model)

    total = len(inputs)
    print(f"  Starting parallel LLM calls: {total} items, {num_workers} workers, "
          f"{len(tokens)} tokens available")

    results = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {}
        for idx, prompt in inputs:
            future = executor.submit(
                _process_single_item,
                idx, prompt, tokens, system_prompt, model,
                temperature, max_tokens, timeout, max_retries,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx, response_text = future.result()
            results[idx] = response_text
            completed += 1
            if completed % 200 == 0 or completed == total:
                success_count = sum(1 for v in results.values() if v != "")
                print(f"  Progress: {completed}/{total} completed "
                      f"({success_count} success, {completed - success_count} failed)")

    # Preserve original input order
    ordered_results = [(idx, results[idx]) for idx, _ in inputs]

    success_count = sum(1 for _, v in ordered_results if v != "")
    print(f"  Done: {success_count}/{total} succeeded, "
          f"{total - success_count} failed")

    return ordered_results


# =============================================================================
# Checkpoint Management
# =============================================================================

def load_checkpoint(checkpoint_dir):
    """Load previously computed results from checkpoint files.

    Checkpoint files are JSONL with {"id": ..., "result": ...} per line.

    Args:
        checkpoint_dir: Directory containing checkpoint .jsonl files.

    Returns:
        dict: idx -> response_text for completed items.
    """
    completed = {}
    if not os.path.exists(checkpoint_dir):
        return completed

    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith('.jsonl'):
            fpath = os.path.join(checkpoint_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        completed[record['id']] = record['result']

    if completed:
        print(f"  [CHECKPOINT] Loaded {len(completed)} completed items "
              f"from {checkpoint_dir}")
    return completed


def save_checkpoint(chunk_results, checkpoint_dir, chunk_idx):
    """Save a chunk of (idx, response_text) results to a checkpoint file.

    Args:
        chunk_results: List of (idx, response_text) tuples.
        checkpoint_dir: Directory to store checkpoint files.
        chunk_idx: Integer index for this chunk file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    fpath = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:05d}.jsonl")
    with open(fpath, 'w', encoding='utf-8') as f:
        for idx, result in chunk_results:
            record = {"id": idx, "result": result}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  [CHECKPOINT] Saved chunk {chunk_idx}: {fpath} "
          f"({len(chunk_results)} items)")


def cleanup_checkpoint(checkpoint_dir):
    """Remove checkpoint directory and all its contents.

    Args:
        checkpoint_dir: Directory to remove.
    """
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"  [CHECKPOINT] Cleaned up: {checkpoint_dir}")


# =============================================================================
# Parallel LLM with Checkpoint/Resume
# =============================================================================

def run_llm_parallel_with_checkpoint(inputs, token_file, checkpoint_dir,
                                     num_workers=20, system_prompt="",
                                     model="gpt-5.2", temperature=0,
                                     max_tokens=2000, timeout=300,
                                     max_retries=3, chunk_size=10000):
    """Run parallel LLM calls with chunked checkpoint/resume support.

    This is the recommended entry point for batch Copilot API inference.
    It combines chunking, parallel execution, and checkpoint persistence
    so that interrupted runs can resume from where they left off.

    Workflow:
      1. Load any existing checkpoint results from checkpoint_dir
      2. Filter out already-completed items
      3. Process remaining items in chunks of chunk_size
      4. Each chunk: run_llm_parallel → save checkpoint
      5. After all chunks: return ordered results, clean up checkpoints

    Args:
        inputs: List of (idx, prompt) tuples. idx is a string identifier,
                prompt is the user prompt text. This is the FULL list of
                all items to process.
        token_file: Path to token file (one token per line).
        checkpoint_dir: Directory for checkpoint files. Created if needed.
        num_workers: Number of parallel threads (default: 20).
        system_prompt: System prompt for all calls (default: "").
        model: Model name (default: "gpt-5.4").
        temperature: Temperature (default: 0).
        max_tokens: Max tokens per response (default: 2000).
        timeout: Timeout per request in seconds (default: 300).
        max_retries: Max retries per request (default: 3).
        chunk_size: Number of items per processing chunk (default: 500).

    Returns:
        List of (idx, response_text) tuples, in the same order as inputs.
        response_text is an empty string for failed calls.
    """
    # Load existing checkpoints
    completed = load_checkpoint(checkpoint_dir)

    # Filter out already-completed items
    remaining = [(idx, prompt) for idx, prompt in inputs if idx not in completed]

    if not remaining:
        print(f"  All {len(inputs)} items already completed from checkpoint")
        return [(idx, completed[idx]) for idx, _ in inputs]

    print(f"  Items to process: {len(remaining)} "
          f"(skipped {len(completed)} from checkpoint)")

    # Load and validate tokens once upfront
    tokens = load_tokens(token_file)
    tokens = validate_tokens(tokens, model=model)

    # Determine starting chunk index to avoid overwriting existing chunks
    existing_chunks = []
    if os.path.exists(checkpoint_dir):
        existing_chunks = [
            f for f in os.listdir(checkpoint_dir) if f.endswith('.jsonl')
        ]
    chunk_offset = len(existing_chunks)

    # Process remaining items in chunks
    total_chunks = (len(remaining) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(remaining))
        chunk = remaining[start:end]

        print(f"\n  Chunk {chunk_idx + 1}/{total_chunks} "
              f"(items {start}-{end - 1}, size={len(chunk)}) ...")

        # Run parallel API calls for this chunk
        chunk_start_time = time.time()
        results = run_llm_parallel(
            inputs=chunk,
            num_workers=num_workers,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            _tokens=tokens,
        )
        chunk_elapsed = time.time() - chunk_start_time

        # Update completed dict and save checkpoint
        chunk_results = []
        for idx, response_text in results:
            completed[idx] = response_text
            chunk_results.append((idx, response_text))

        save_checkpoint(chunk_results, checkpoint_dir, chunk_offset + chunk_idx)

        success_count = sum(1 for _, r in chunk_results if r)
        throughput = len(chunk) / chunk_elapsed if chunk_elapsed > 0 else 0
        print(f"  Chunk done in {chunk_elapsed:.1f}s ({throughput:.1f} items/s), "
              f"{success_count}/{len(chunk)} succeeded")

    # Return all results in original input order
    return [(idx, completed.get(idx, "")) for idx, _ in inputs]

