"""Papyrus LLM Inference Utilities

Provides parallel async inference via the Papyrus API using direct HTTP calls,
matching the exact request format from test_papyrus.py.

Designed to be imported by s1_generate_tid.py as a third inference backend
alongside vLLM and Copilot.

Functions:
    run_papyrus_parallel: Run parallel Papyrus API calls for a list of prompts.
    run_papyrus_parallel_with_checkpoint: Same, with chunked checkpoint/resume.

Usage:
    from Infer_by_papyrus import run_papyrus_parallel_with_checkpoint

    inputs = [("item1", "prompt1"), ("item2", "prompt2")]
    results = run_papyrus_parallel_with_checkpoint(
        inputs=inputs,
        checkpoint_dir="./checkpoints/",
        papyrus_endpoint="https://westus2batch.papyrus.binginternal.com",
        model_name="gpt-54-2026-03-05-Eval",
        num_workers=40,
    )
"""

import os
import time
import asyncio

import httpx
from azure.identity import DefaultAzureCredential

from llm_utils import load_checkpoint, save_checkpoint

PAPYRUS_SCOPE = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"


# =============================================================================
# Token Helper
# =============================================================================

def _get_papyrus_token():
    """Get a fresh bearer token for Papyrus API."""
    credential = DefaultAzureCredential()
    return credential.get_token(PAPYRUS_SCOPE).token


# =============================================================================
# Async Worker
# =============================================================================

class WorkerClass:
    """Async worker with semaphore-based concurrency control."""

    def __init__(self, max_workers=40):
        self.semaphore = asyncio.Semaphore(max_workers)

    async def call_llm(self, item_id, prompt, client, url, headers,
                       retry_max=3, max_tokens=100):
        """Send a single prompt to the Papyrus LLM API.

        Uses the same request format as test_papyrus.py:
        - POST to /chat/completions
        - Authorization via Bearer token in headers
        - max_completion_tokens in JSON body (not max_tokens)

        Returns:
            Tuple of (item_id, content_str, duration) on success,
            or (item_id, Exception, -1) on failure.
        """
        json_dict = {
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }

        async with self.semaphore:
            last_error = None
            for attempt in range(1, retry_max + 1):
                try:
                    start_time = time.perf_counter()
                    response = await client.post(
                        url,
                        headers=headers,
                        json=json_dict,
                    )
                    response.raise_for_status()
                    duration = time.perf_counter() - start_time
                    result = response.json()
                    content = result["choices"][0]["message"]["content"] or ""
                    return item_id, content, duration
                except Exception as ex:
                    last_error = ex
                    if attempt < retry_max:
                        # Longer backoff for 429 rate limiting
                        is_429 = "429" in str(ex)
                        wait = (10 + attempt * 5) if is_429 else (2 * attempt)
                        await asyncio.sleep(wait)

            return item_id, last_error, -1


# =============================================================================
# Internal Async Batch Runner
# =============================================================================

async def _run_batch_async(inputs, client, url, headers,
                           num_workers=40, max_tokens=100, max_retries=3):
    """Run a batch of prompts through Papyrus API concurrently.

    Args:
        inputs: List of (idx, prompt) tuples.
        client: An httpx.AsyncClient.
        url: Full URL for the chat completions endpoint.
        headers: Dict of HTTP headers (Authorization + Papyrus headers).
        num_workers: Max concurrent requests.
        max_tokens: Max output tokens per request.
        max_retries: Max retries per failed request.

    Returns:
        List of (idx, response_text) tuples in input order.
        response_text is "" for failed calls.
    """
    worker = WorkerClass(num_workers)
    total = len(inputs)
    print(f"  Starting Papyrus async calls: {total} items, "
          f"{num_workers} workers")

    tasks = [
        worker.call_llm(item_id, prompt, client, url, headers,
                        max_retries, max_tokens)
        for item_id, prompt in inputs
    ]

    results = {}
    completed = 0

    for coro in asyncio.as_completed(tasks):
        item_id, result, duration = await coro
        if isinstance(result, Exception):
            print(f"    [ERROR] Failed for idx={item_id}: {result}")
            results[item_id] = ""
        else:
            results[item_id] = result
        completed += 1
        if completed % 200 == 0 or completed == total:
            success = sum(1 for v in results.values() if v != "")
            print(f"  Progress: {completed}/{total} completed "
                  f"({success} success, {completed - success} failed)")

    ordered = [(idx, results.get(idx, "")) for idx, _ in inputs]
    success_count = sum(1 for _, v in ordered if v != "")
    print(f"  Done: {success_count}/{total} succeeded, "
          f"{total - success_count} failed")
    return ordered


# =============================================================================
# Checkpoint-aware Async Runner
# =============================================================================

async def _run_papyrus_with_checkpoint_async(
    inputs, checkpoint_dir, papyrus_endpoint, model_name,
    quota_id="", timeout_ms=120000, num_workers=40,
    max_tokens=100, max_retries=3, chunk_size=10000,
):
    """Full async workflow: get token, create client, process chunks,
    save checkpoints.

    Token and headers are created once at the start. The token is refreshed
    before each chunk to avoid expiration during long runs.
    """
    completed = load_checkpoint(checkpoint_dir)
    remaining = [(idx, prompt) for idx, prompt in inputs if idx not in completed]

    if not remaining:
        print(f"  All {len(inputs)} items already completed from checkpoint")
        return [(idx, completed[idx]) for idx, _ in inputs]

    print(f"  Items to process: {len(remaining)} "
          f"(skipped {len(completed)} from checkpoint)")

    # Determine starting chunk index to avoid overwriting existing files
    existing_chunks = []
    if os.path.exists(checkpoint_dir):
        existing_chunks = [
            f for f in os.listdir(checkpoint_dir) if f.endswith('.jsonl')
        ]
    chunk_offset = len(existing_chunks)

    total_chunks = (len(remaining) + chunk_size - 1) // chunk_size

    url = papyrus_endpoint.rstrip("/") + "/chat/completions"

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_ms / 1000.0),
    ) as client:
        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(remaining))
            chunk = remaining[start:end]

            print(f"\n  Chunk {chunk_idx + 1}/{total_chunks} "
                  f"(items {start}-{end - 1}, size={len(chunk)}) ...")

            # Refresh token before each chunk (tokens expire ~1h)
            access_token = _get_papyrus_token()
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "papyrus-model-name": model_name,
                "papyrus-quota-id": quota_id,
                "papyrus-timeout-ms": str(timeout_ms),
            }

            chunk_start_time = time.time()
            results = await _run_batch_async(
                inputs=chunk,
                client=client,
                url=url,
                headers=headers,
                num_workers=num_workers,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
            chunk_elapsed = time.time() - chunk_start_time

            chunk_results = []
            for idx, response_text in results:
                completed[idx] = response_text
                chunk_results.append((idx, response_text))
            save_checkpoint(chunk_results, checkpoint_dir,
                            chunk_offset + chunk_idx)

            success_count = sum(1 for _, r in chunk_results if r)
            throughput = len(chunk) / chunk_elapsed if chunk_elapsed > 0 else 0
            print(f"  Chunk done in {chunk_elapsed:.1f}s "
                  f"({throughput:.1f} items/s), "
                  f"{success_count}/{len(chunk)} succeeded")

    return [(idx, completed.get(idx, "")) for idx, _ in inputs]


# =============================================================================
# Public API
# =============================================================================

def run_papyrus_parallel(inputs, papyrus_endpoint, model_name,
                         quota_id="", timeout_ms=120000, num_workers=40,
                         max_tokens=100, max_retries=3):
    """Run parallel Papyrus API calls for a list of prompts.

    Synchronous wrapper. Token and headers are created once.

    Args:
        inputs: List of (idx, prompt) tuples.
        papyrus_endpoint: Papyrus API endpoint URL.
        model_name: Papyrus model name.
        quota_id: Papyrus quota ID.
        timeout_ms: Request timeout in ms.
        num_workers: Max concurrent requests.
        max_tokens: Max output tokens per request.
        max_retries: Max retries per failed request.

    Returns:
        List of (idx, response_text) tuples in input order.
    """
    async def _run():
        access_token = _get_papyrus_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "papyrus-model-name": model_name,
            "papyrus-quota-id": quota_id,
            "papyrus-timeout-ms": str(timeout_ms),
        }
        url = papyrus_endpoint.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_ms / 1000.0),
        ) as client:
            return await _run_batch_async(
                inputs=inputs,
                client=client,
                url=url,
                headers=headers,
                num_workers=num_workers,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

    return asyncio.run(_run())


def run_papyrus_parallel_with_checkpoint(
    inputs, checkpoint_dir, papyrus_endpoint, model_name,
    quota_id="", timeout_ms=120000, num_workers=40,
    max_tokens=100, max_retries=3, chunk_size=10000,
):
    """Run parallel Papyrus API calls with chunked checkpoint/resume.

    Recommended entry point for batch Papyrus API inference.
    Uses the same checkpoint format as Copilot (JSONL with id/result).

    Args:
        inputs: List of (idx, prompt) tuples (full list).
        checkpoint_dir: Directory for checkpoint .jsonl files.
        papyrus_endpoint: Papyrus API endpoint URL.
        model_name: Papyrus model name.
        quota_id: Papyrus quota ID.
        timeout_ms: Request timeout in ms.
        num_workers: Max concurrent requests.
        max_tokens: Max output tokens per request.
        max_retries: Max retries per failed request.
        chunk_size: Items per processing chunk.

    Returns:
        List of (idx, response_text) tuples in input order.
    """
    return asyncio.run(_run_papyrus_with_checkpoint_async(
        inputs=inputs,
        checkpoint_dir=checkpoint_dir,
        papyrus_endpoint=papyrus_endpoint,
        model_name=model_name,
        quota_id=quota_id,
        timeout_ms=timeout_ms,
        num_workers=num_workers,
        max_tokens=max_tokens,
        max_retries=max_retries,
        chunk_size=chunk_size,
    ))


# =============================================================================
# Test Main
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test: send a few prompts to Papyrus and print results."""
    import json

    ENDPOINT = "https://westus2batch.papyrus.binginternal.com"
    MODEL = "gpt-5-chat-shortco-2025-08-07-Bing"
    QUOTA_ID = ""
    TIMEOUT_MS = 120000

    test_inputs = [
        ("test_1", "What is 2+3? Answer with just the number."),
        ("test_2", "Name 3 colors. Answer as a comma-separated list."),
        ("test_3", "What is the capital of France? One word answer."),
    ]

    print("=" * 60)
    print("Papyrus Smoke Test")
    print(f"  Endpoint: {ENDPOINT}")
    print(f"  Model:    {MODEL}")
    print(f"  Prompts:  {len(test_inputs)}")
    print("=" * 60)

    # --- Test 1: run_papyrus_parallel (no checkpoint) ---
    print("\n--- Test 1: run_papyrus_parallel ---")
    results = run_papyrus_parallel(
        inputs=test_inputs,
        papyrus_endpoint=ENDPOINT,
        model_name=MODEL,
        quota_id=QUOTA_ID,
        timeout_ms=TIMEOUT_MS,
        num_workers=3,
        max_tokens=50,
    )
    for item_id, text in results:
        print(f"  [{item_id}] {text[:100]}")

    # --- Test 2: run_papyrus_parallel_with_checkpoint ---
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="papyrus_test_ckpt_")
    print(f"\n--- Test 2: run_papyrus_parallel_with_checkpoint ---")
    print(f"  Checkpoint dir: {tmpdir}")

    results2 = run_papyrus_parallel_with_checkpoint(
        inputs=test_inputs,
        checkpoint_dir=tmpdir,
        papyrus_endpoint=ENDPOINT,
        model_name=MODEL,
        quota_id=QUOTA_ID,
        timeout_ms=TIMEOUT_MS,
        num_workers=3,
        max_tokens=50,
        chunk_size=2,  # small chunk to test chunking
    )
    for item_id, text in results2:
        print(f"  [{item_id}] {text[:100]}")

    # Verify checkpoint files were created
    ckpt_files = os.listdir(tmpdir)
    print(f"  Checkpoint files: {ckpt_files}")

    # Run again to test resume (should skip all)
    print("\n  Re-running (should skip all from checkpoint)...")
    results3 = run_papyrus_parallel_with_checkpoint(
        inputs=test_inputs,
        checkpoint_dir=tmpdir,
        papyrus_endpoint=ENDPOINT,
        model_name=MODEL,
        num_workers=3,
        max_tokens=50,
    )
    for item_id, text in results3:
        print(f"  [{item_id}] {text[:100]}")

    shutil.rmtree(tmpdir)
    print(f"\n  Cleaned up {tmpdir}")
    print("\nAll tests passed!")
