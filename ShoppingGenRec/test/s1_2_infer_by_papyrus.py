"""Step 1.2: Batch LLM Inference via Papyrus API

Reads the prompts.tsv exported by s1_generate_tid.py (--export_prompts_only),
sends each prompt to the Papyrus LLM API with async concurrency, and writes
the raw LLM outputs to a TSV file.

Input:
    prompts.tsv (from s1_generate_tid.py --export_prompts_only)
      Two columns: item_id \t prompt

Output:
    llm_outputs.tsv
      Two columns: item_id \t llm_raw_output

Usage:
    python s1_2_infer_by_papyrus.py \
        --prompts_file ./processed_tid/prompts.tsv \
        --output_file ./processed_tid/llm_outputs.tsv \
        --papyrus_endpoint https://westus2batch.papyrus.binginternal.com \
        --model_name gpt-54-2026-03-05-Eval \
        --max_workers 40 \
        --package_size 1000
"""

import os
import csv
import sys
import time
import asyncio
import argparse
from datetime import datetime

import httpx
from openai import AsyncAzureOpenAI
from openai._models import FinalRequestOptions
from openai._types import NOT_GIVEN, Timeout, NotGiven
from openai._base_client import DEFAULT_MAX_RETRIES, BaseClient
from openai._exceptions import OpenAIError
from typing import Union, Mapping, Callable
from typing_extensions import override
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

csv.field_size_limit(sys.maxsize)

AzureADTokenProvider = Callable[[], str]
API_KEY_SENTINEL = "<missing API key>"


# =============================================================================
# Papyrus Client
# =============================================================================

class AsyncPapyrusClient(AsyncAzureOpenAI):
    """Async client for Papyrus API (Azure OpenAI compatible)."""

    def __init__(
        self,
        *,
        papyrus_endpoint: str,
        api_version: Union[str, None] = None,
        api_key: Union[str, None] = None,
        azure_ad_token_provider: Union[AzureADTokenProvider, None] = None,
        organization: Union[str, None] = None,
        project: Union[str, None] = None,
        base_url: Union[str, None] = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Union[Mapping[str, str], None] = None,
        default_query: Union[Mapping[str, object], None] = None,
        http_client: Union[httpx.Client, None] = None,
        _strict_response_validation: bool = False,
    ) -> None:
        if azure_ad_token_provider is None:
            raise OpenAIError(
                "Missing credentials. Please pass `azure_ad_token_provider`."
            )
        if papyrus_endpoint is None:
            raise OpenAIError(
                "Missing base url. Please pass `papyrus_endpoint`."
            )

        base_url = papyrus_endpoint

        if default_query is None:
            default_query = {"api-version": api_version}
        else:
            default_query = {**default_query, "api-version": api_version}

        if api_key is None:
            api_key = API_KEY_SENTINEL

        super().__init__(
            api_version="null",
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )
        self._api_version = None
        self._azure_ad_token = None
        self._azure_ad_token_provider = azure_ad_token_provider

    @override
    def _build_request(
        self,
        options: FinalRequestOptions,
        *,
        retries_taken: int = 0,
    ) -> httpx.Request:
        return BaseClient._build_request(self, options)


# =============================================================================
# Data Loading
# =============================================================================

def load_prompts_batched(file_path, package_size=1000):
    """Load prompts.tsv in batches (generator).

    Yields:
        List of (item_id, prompt) tuples, up to package_size per batch.
    """
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            item_id = row[0].strip()
            prompt = row[1].replace("\\n", "\n")
            batch.append((item_id, prompt))
            if len(batch) >= package_size:
                yield batch
                batch = []
    if batch:
        yield batch


def count_prompts(file_path):
    """Count the number of data lines in prompts.tsv (excluding header)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


# =============================================================================
# Async Inference
# =============================================================================

class WorkerClass:
    """Async worker with semaphore-based concurrency control."""

    def __init__(self, max_workers=40):
        self.semaphore = asyncio.Semaphore(max_workers)

    async def call_llm(self, item_id, prompt, client, papyrus_headers,
                       model_name, retry_max=3):
        """Send a single prompt to the Papyrus LLM API.

        Returns:
            Tuple of (item_id, content_str, duration) on success,
            or (item_id, Exception, -1) on failure.
        """
        message = [{"role": "user", "content": prompt}]

        async with self.semaphore:
            last_error = None
            for attempt in range(1, retry_max + 1):
                try:
                    start_time = time.perf_counter()
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=message,
                        extra_headers=papyrus_headers,
                    )
                    duration = time.perf_counter() - start_time
                    content = response.choices[0].message.content or ""
                    return item_id, content, duration
                except Exception as ex:
                    last_error = ex
                    if attempt < retry_max:
                        await asyncio.sleep(2 * attempt)

            return item_id, last_error, -1


async def process_batch(batch, client, papyrus_headers, model_name,
                        output_file, max_workers=40, retry_max=3):
    """Process a batch of (item_id, prompt) pairs concurrently.

    Returns:
        Tuple of (success_count, failures) where failures is a list of
        (item_id, error) tuples.
    """
    start_time = time.perf_counter()
    worker = WorkerClass(max_workers)

    tasks = [
        worker.call_llm(item_id, prompt, client, papyrus_headers,
                        model_name, retry_max)
        for item_id, prompt in batch
    ]

    failures = []
    success_count = 0

    with open(output_file, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            item_id, result, duration = await coro
            if isinstance(result, Exception):
                failures.append((item_id, result))
            else:
                clean_content = (
                    result.replace("\t", " ")
                    .replace("\n", "\\n")
                    .replace("\r", "")
                )
                f.write(f"{item_id}\t{clean_content}\n")
                success_count += 1

    elapsed = time.perf_counter() - start_time
    throughput = len(batch) / elapsed if elapsed > 0 else 0
    print(f"    Batch: {success_count}/{len(batch)} succeeded "
          f"in {elapsed:.1f}s ({throughput:.1f} items/s)")

    return success_count, failures


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch LLM inference via Papyrus API for product summarization"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./processed_tid/prompts.tsv",
        help="Path to prompts.tsv from s1_generate_tid.py --export_prompts_only "
             "(default: ./processed_tid/prompts.tsv)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./processed_tid/llm_outputs.tsv",
        help="Output TSV file with item_id and LLM raw output "
             "(default: ./processed_tid/llm_outputs.tsv)",
    )
    parser.add_argument(
        "--papyrus_endpoint",
        type=str,
        default="https://westus2batch.papyrus.binginternal.com",
        help="Papyrus API endpoint (default: westus2batch)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-54-2026-03-05-Eval",
        help="Papyrus model name (default: gpt-54-2026-03-05-Eval)",
    )
    parser.add_argument(
        "--quota_id",
        type=str,
        default="",
        help="Papyrus quota ID (default: empty)",
    )
    parser.add_argument(
        "--timeout_ms",
        type=int,
        default=120000,
        help="Papyrus request timeout in ms (default: 120000)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=40,
        help="Maximum concurrent async requests (default: 40)",
    )
    parser.add_argument(
        "--package_size",
        type=int,
        default=1000,
        help="Number of prompts per processing batch (default: 1000)",
    )
    parser.add_argument(
        "--retry_max",
        type=int,
        default=3,
        help="Maximum retries per failed request (default: 3)",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=-1,
        help="Maximum number of items to process. -1 for all (default: -1)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

async def async_main():
    args = parse_args()

    print("=" * 70)
    print("Papyrus Batch LLM Inference")
    print("=" * 70)
    print(f"  Prompts file:      {args.prompts_file}")
    print(f"  Output file:       {args.output_file}")
    print(f"  Endpoint:          {args.papyrus_endpoint}")
    print(f"  Model:             {args.model_name}")
    print(f"  Max workers:       {args.max_workers}")
    print(f"  Package size:      {args.package_size}")
    print(f"  Retry max:         {args.retry_max}")
    print(f"  Max items:         {args.max_items if args.max_items > 0 else 'all'}")

    # ---- Setup Papyrus client ----
    print("\nInitializing Papyrus client...")
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default",
    )

    client = AsyncPapyrusClient(
        papyrus_endpoint=args.papyrus_endpoint,
        azure_ad_token_provider=token_provider,
    )

    papyrus_headers = {
        "papyrus-model-name": args.model_name,
        "papyrus-quota-id": args.quota_id,
        "papyrus-timeout-ms": str(args.timeout_ms),
    }

    # ---- Count prompts ----
    print(f"\nCounting prompts in: {args.prompts_file}")
    total_lines = count_prompts(args.prompts_file)
    if args.max_items > 0:
        total_lines = min(total_lines, args.max_items)
    print(f"  Total prompts to process: {total_lines:,}")

    # ---- Prepare output file (write header) ----
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("item_id\tllm_output\n")

    # ---- Process in batches ----
    print(f"\nStarting inference...")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    overall_start = time.perf_counter()

    total_success = 0
    total_failures = []
    batch_num = 0
    items_processed = 0

    for batch in load_prompts_batched(args.prompts_file, args.package_size):
        if args.max_items > 0 and items_processed >= args.max_items:
            break
        if args.max_items > 0:
            remaining = args.max_items - items_processed
            batch = batch[:remaining]

        batch_num += 1
        print(f"\n  Batch {batch_num} ({len(batch):,} items, "
              f"total processed: {items_processed:,}/{total_lines:,})")

        success_count, failures = await process_batch(
            batch, client, papyrus_headers, args.model_name,
            args.output_file, args.max_workers, args.retry_max,
        )
        total_success += success_count
        total_failures.extend(failures)
        items_processed += len(batch)

    # ---- Retry final failures ----
    if total_failures:
        print(f"\n  Retrying {len(total_failures)} failed items...")
        failed_ids = {item_id for item_id, _ in total_failures}
        retry_batch = []
        for batch in load_prompts_batched(args.prompts_file, args.package_size * 10):
            for item_id, prompt in batch:
                if item_id in failed_ids:
                    retry_batch.append((item_id, prompt))

        if retry_batch:
            retry_success, retry_failures = await process_batch(
                retry_batch, client, papyrus_headers, args.model_name,
                args.output_file, args.max_workers, args.retry_max,
            )
            total_success += retry_success
            total_failures = retry_failures

    # ---- Write error log ----
    if total_failures:
        error_file = args.output_file + ".errors"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write("item_id\terror\n")
            for item_id, error in total_failures:
                f.write(f"{item_id}\t{str(error)}\n")
        print(f"\n  Errors saved to: {error_file} ({len(total_failures)} items)")

    # ---- Summary ----
    overall_elapsed = time.perf_counter() - overall_start
    output_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"  Total processed:    {items_processed:>10,}")
    print(f"  Successful:         {total_success:>10,}")
    print(f"  Failed:             {len(total_failures):>10,}")
    print(f"  Output file:        {args.output_file} ({output_size_mb:.1f} MB)")
    print(f"  Total time:         {overall_elapsed:.1f}s")
    if overall_elapsed > 0:
        print(f"  Throughput:         {items_processed / overall_elapsed:.1f} items/s")
    print(f"  End time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nDone!")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
