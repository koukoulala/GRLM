#!/usr/bin/env python3
"""
Shopping Journey Evaluation Visualizer

Reads a TSV evaluation data file and generates a self-contained interactive
HTML page for inspecting individual user records with their shopping journeys,
quality metrics, and product evaluations.

Usage:
    python visualize.py <tsv_file_path> [--output <html_file_path>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Increase CSV field size limit for large JSON fields (Windows-compatible)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
COL_USER_ID = "UserId"
COL_USER_SIGNALS = "UserSignals"
COL_READABLE_SIGNALS = "ReadableUserSignals"
COL_USER_PROFILE = "UserProfile"
COL_JOURNEYS = "ShoppingJourneys"
COL_JOURNEY_DIVERSITY = "journey_diversity"
COL_JOURNEY_QUALITY = "journey_quality"
COL_JOURNEY_RELEVANCE = "journey_relevance"
COL_PRODUCT_DIVERSITY = "product_diversity"
COL_PRODUCT_QUALITY = "product_quality"

REQUIRED_COLUMNS = [
    COL_USER_ID,
    COL_USER_SIGNALS,
    COL_READABLE_SIGNALS,
    COL_USER_PROFILE,
    COL_JOURNEYS,
    COL_JOURNEY_DIVERSITY,
    COL_JOURNEY_QUALITY,
    COL_JOURNEY_RELEVANCE,
    COL_PRODUCT_DIVERSITY,
    COL_PRODUCT_QUALITY,
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def safe_json_parse(text: str, context: str = ""):
    """Attempt to parse JSON, returning None on failure."""
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try stripping surrounding quotes and unescaping double-quotes
        stripped = text.strip('"').replace('""', '"')
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None


def normalize_product_quality(raw):
    """Normalize product_quality to a list of per-journey objects."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    return []


def parse_readable_signals(text: str) -> list[str]:
    """Split ReadableUserSignals into individual lines."""
    if not text:
        return []
    # The field uses literal \n as separator
    lines = text.replace("\\n", "\n").split("\n")
    return [line.strip() for line in lines if line.strip()]


def parse_row(row: dict, row_idx: int) -> dict | None:
    """Parse a single TSV row into a cleaned record dict.
    Returns None if essential fields are missing/unparseable."""
    user_id = row.get(COL_USER_ID, "").strip().strip('"')
    if not user_id:
        return None

    user_signals = safe_json_parse(row.get(COL_USER_SIGNALS, ""), f"row {row_idx} UserSignals")
    readable_signals = parse_readable_signals(row.get(COL_READABLE_SIGNALS, ""))
    user_profile_raw = safe_json_parse(row.get(COL_USER_PROFILE, ""), f"row {row_idx} UserProfile")
    journeys = safe_json_parse(row.get(COL_JOURNEYS, ""), f"row {row_idx} ShoppingJourneys")
    journey_diversity = safe_json_parse(row.get(COL_JOURNEY_DIVERSITY, ""), f"row {row_idx} journey_diversity")
    journey_quality = safe_json_parse(row.get(COL_JOURNEY_QUALITY, ""), f"row {row_idx} journey_quality")
    journey_relevance = safe_json_parse(row.get(COL_JOURNEY_RELEVANCE, ""), f"row {row_idx} journey_relevance")
    product_diversity = safe_json_parse(row.get(COL_PRODUCT_DIVERSITY, ""), f"row {row_idx} product_diversity")
    product_quality_raw = safe_json_parse(row.get(COL_PRODUCT_QUALITY, ""), f"row {row_idx} product_quality")

    # Extract inner profile
    user_profile = {}
    if isinstance(user_profile_raw, dict):
        user_profile = user_profile_raw.get("userShoppingProfile", user_profile_raw)

    # Normalize arrays
    if not isinstance(journeys, list):
        journeys = [journeys] if journeys else []
    if not isinstance(journey_quality, list):
        journey_quality = [journey_quality] if journey_quality else []
    if not isinstance(journey_relevance, list):
        journey_relevance = [journey_relevance] if journey_relevance else []
    if not isinstance(product_diversity, list):
        product_diversity = [product_diversity] if product_diversity else []

    product_quality = normalize_product_quality(product_quality_raw)

    return {
        "userId": user_id,
        "userSignals": user_signals or [],
        "readableSignals": readable_signals,
        "userProfile": user_profile,
        "journeys": journeys,
        "journeyDiversity": journey_diversity or {},
        "journeyQuality": journey_quality,
        "journeyRelevance": journey_relevance,
        "productDiversity": product_diversity,
        "productQuality": product_quality,
    }


# ---------------------------------------------------------------------------
# Load product OfferUrl mapping
# ---------------------------------------------------------------------------
def load_offer_urls(filepath: str) -> dict[str, str]:
    """Load GlobalOfferId -> OfferUrl mapping from JourneyProduct TSV.

    Uses a JSON cache file alongside the TSV for fast subsequent loads.
    Only reads the two needed columns (index 0 and 11) via raw split,
    avoiding full csv parsing overhead.
    """
    if not filepath or not os.path.isfile(filepath):
        print(f"WARNING: Product file not found: {filepath}", file=sys.stderr)
        return {}

    # Check for cached JSON (much smaller & faster to load)
    cache_path = filepath + ".offer_urls_cache.json"
    if os.path.isfile(cache_path):
        tsv_mtime = os.path.getmtime(filepath)
        cache_mtime = os.path.getmtime(cache_path)
        if cache_mtime >= tsv_mtime:
            print(f"Loading product URLs from cache: {cache_path}...", file=sys.stderr)
            with open(cache_path, "r", encoding="utf-8") as f:
                urls = json.load(f)
            print(f"Loaded {len(urls):,} product URLs (cached)", file=sys.stderr)
            return urls

    # Column indices: GlobalOfferId=0, OfferUrl=11
    GID_IDX, URL_IDX = 0, 11
    MIN_COLS = URL_IDX + 1

    print(f"Loading product URLs from {filepath}...", file=sys.stderr)
    urls = {}
    with open(filepath, "r", encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.split("\t")
            if len(parts) < MIN_COLS:
                continue
            gid = parts[GID_IDX].strip()
            url = parts[URL_IDX].strip()
            if gid and url:
                urls[gid] = url
    print(f"Loaded {len(urls):,} product URLs", file=sys.stderr)

    # Save cache for next time
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(urls, f, ensure_ascii=False, separators=(",", ":"))
        cache_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"Saved URL cache ({cache_mb:.1f} MB): {cache_path}", file=sys.stderr)
    except OSError:
        pass  # non-critical if cache write fails

    return urls


# ---------------------------------------------------------------------------
# Read TSV
# ---------------------------------------------------------------------------
def read_tsv(filepath: str) -> list[dict]:
    """Read the TSV file and return a list of parsed user records."""
    records = []
    skipped = 0
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        # Validate columns
        if reader.fieldnames:
            missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                print(f"WARNING: Missing columns: {missing}", file=sys.stderr)

        for i, row in enumerate(reader):
            total += 1
            record = parse_row(row, i + 1)
            if record is None:
                skipped += 1
                continue
            records.append(record)

    print(f"Read {total} rows, parsed {len(records)}, skipped {skipped}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def generate_html(records: list[dict], source_filename: str,
                  offer_urls: dict[str, str] | None = None) -> str:
    """Generate the complete self-contained HTML page."""
    data_json = json.dumps(records, ensure_ascii=False, separators=(",", ":"))

    # Only embed URLs for products that actually appear in the data
    if offer_urls:
        used_pids = set()
        for rec in records:
            for j in rec.get("journeys", []):
                for p in j.get("Products", []):
                    pid = p.get("OfferId")
                    if pid is not None:
                        used_pids.add(str(pid))
        filtered_urls = {pid: offer_urls[pid] for pid in used_pids if pid in offer_urls}
        print(f"Embedding {len(filtered_urls):,} product URLs (out of {len(offer_urls):,} total)", file=sys.stderr)
    else:
        filtered_urls = {}
    url_json = json.dumps(filtered_urls, ensure_ascii=False, separators=(",", ":"))
    total_users = len(records)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shopping Journey Evaluation - {_html_escape(source_filename)}</title>
<style>
{CSS}
</style>
</head>
<body>

<header class="top-bar">
  <div class="top-bar-left">
    <h1 class="file-title">{_html_escape(source_filename)}</h1>
    <span class="user-count">{total_users} users</span>
  </div>
  <div class="top-bar-center">
    <button id="btn-prev" class="nav-btn" title="Previous user (Left arrow)">&larr; Prev</button>
    <div class="user-selector">
      <label for="user-search">User:</label>
      <input type="text" id="user-search" placeholder="Search by index or UserId..." autocomplete="off">
      <div id="search-dropdown" class="search-dropdown hidden"></div>
    </div>
    <button id="btn-next" class="nav-btn" title="Next user (Right arrow)">Next &rarr;</button>
  </div>
  <div class="top-bar-right">
    <button class="def-btn" id="btn-definitions" title="Metric Definitions Reference">&#128214; Metric Definitions</button>
    <span id="user-index-display" class="index-display" style="margin-left:12px">1 / {total_users}</span>
  </div>
</header>

<!-- Filter bar -->
<div class="filter-bar-wrapper" id="filter-bar-wrapper">
  <div class="filter-bar-header" id="filter-bar-header">
    <div class="filter-bar-title">
      &#128269; Filters
      <span class="filter-active-count hidden" id="filter-active-count">0</span>
      <span class="filter-match-info" id="filter-match-info"></span>
    </div>
    <span class="filter-bar-toggle">&#9660;</span>
  </div>
  <div class="filter-bar-body">
    <div class="filter-grid">
      <div class="filter-item"><label>J. Diversity</label><select id="f-jdiversity" data-dim="jdiversity"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>J. Relevance</label><select id="f-jrelevance" data-dim="jrelevance"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>J. Value</label><select id="f-jvalue" data-dim="jvalue"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>Compliance</label><select id="f-compliance" data-dim="compliance"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>Tone</label><select id="f-tone" data-dim="tone"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>Coherence</label><select id="f-coherence" data-dim="coherence"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>P. Diversity</label><select id="f-pdiversity" data-dim="pdiversity"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>Intent Align</label><select id="f-intent" data-dim="intent"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>Attr Align</label><select id="f-attribute" data-dim="attribute"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>Gender Align</label><select id="f-gender" data-dim="gender"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>P. Compliance</label><select id="f-pcompliance" data-dim="pcompliance"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
      <div class="filter-item"><label>Seller Auth</label><select id="f-seller" data-dim="seller"><option value="">All</option><option value="lte1">&le; 1 (has issues)</option><option value="0">= 0 (bad)</option><option value="1">= 1 (fair)</option><option value="2">= 2 (good)</option></select></div>
    </div>
    <div class="filter-actions">
      <button class="filter-clear-btn" id="filter-clear-btn">Clear Filters</button>
    </div>
  </div>
</div>

<!-- Metric Definitions Overlay -->
<div class="def-overlay" id="def-overlay">
  <div class="def-panel">
    <div class="def-panel-header">
      <h2>&#128214; Metric Definitions</h2>
      <button class="def-close-btn" id="def-close-btn">&#10005; Close</button>
    </div>
    <div class="def-panel-body">

      <div class="def-section expanded">
        <div class="def-section-header" onclick="this.parentElement.classList.toggle('expanded')">
          <span>Step 1: Journey Diversity (per user)</span>
          <span class="def-section-toggle">&#9660;</span>
        </div>
        <div class="def-section-body">
          <p class="def-metric-desc">Evaluates whether journeys provide meaningful diversity across user intent/decision space.</p>
          <p class="def-metric-desc">Group journeys by same Recipient + Primary Category + Brand &rarr; <code>diversity_ratio = num_groups / total_journeys</code></p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>ratio = 1.0 (all journeys represent distinct intents)</td></tr>
            <tr><td class="ds-1">Score 1</td><td>0.6 &le; ratio &lt; 1.0 (some overlap, acceptable coverage)</td></tr>
            <tr><td class="ds-0">Score 0</td><td>ratio &lt; 0.6 (significant redundancy)</td></tr>
          </table>
        </div>
      </div>

      <div class="def-section">
        <div class="def-section-header" onclick="this.parentElement.classList.toggle('expanded')">
          <span>Step 2: Journey Quality (per journey)</span>
          <span class="def-section-toggle">&#9660;</span>
        </div>
        <div class="def-section-body">
          <div class="def-sub-heading">Journey Appropriateness</div>

          <p class="def-metric-name">journeyValue</p>
          <p class="def-metric-desc">Whether category inherently benefits from curation (NOT about specific user).</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Category benefits from curation (taste/fit/occasion/style considerations)</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Category has some variation but limited curation value</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Category unsuitable for curation (commodity items, tools, daily necessities)</td></tr>
          </table>

          <p class="def-metric-name">contentCompliance</p>
          <p class="def-metric-desc">Safety + online shopping suitability.</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Completely safe AND suitable for online shopping</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Borderline safety concerns OR partially suitable</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Contains prohibited content (adult, violence, weapons, political, health) OR unsuitable category (services, software, cars)</td></tr>
          </table>

          <div class="def-sub-heading">Journey Title Quality</div>

          <p class="def-metric-name">tone</p>
          <p class="def-metric-desc">Natural, grammatically correct, idiomatic language.</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Excellent human-like language, fluent and professional</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Acceptable with minor issues, slightly awkward</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Poor/robotic/awkward language, grammatical errors</td></tr>
          </table>

          <p class="def-metric-name">selfCoherence</p>
          <p class="def-metric-desc">Title internally consistent without contradictions.</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Title elements logically consistent and coherent</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Minor inconsistencies but still understandable</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Contains contradictory or conflicting elements</td></tr>
          </table>

          <div class="def-sub-heading">Journey Type</div>
          <p class="def-metric-desc"><strong>explicit</strong>: user directly viewed/clicked products in category.<br><strong>exploratory</strong>: inferred from broader browsing patterns.</p>
        </div>
      </div>

      <div class="def-section">
        <div class="def-section-header" onclick="this.parentElement.classList.toggle('expanded')">
          <span>Step 3: Journey Relevance (per journey)</span>
          <span class="def-section-toggle">&#9660;</span>
        </div>
        <div class="def-section-body">
          <p class="def-metric-name">shoppingRelevanceScore</p>
          <p class="def-metric-desc">Whether journey matches user's observed shopping behavior.</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Clearly shopping purpose; strong signals (repeated views, detail pages, add-to-cart)</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Partially shopping related; ambiguous or light engagement</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Not shopping purpose; no plausible shopping intent</td></tr>
          </table>
          <div class="def-sub-heading">Two-step classification</div>
          <p class="def-metric-desc"><strong>Step1 (RULE_STEP1_*)</strong>: User directly interacted with journey product/category/brand.<br><strong>Step2 (RULE_STEP2_*)</strong>: No direct interaction; extension logic (complementary, hobby/activity, brand extension).</p>
        </div>
      </div>

      <div class="def-section">
        <div class="def-section-header" onclick="this.parentElement.classList.toggle('expanded')">
          <span>Step 4: Product Diversity (per journey)</span>
          <span class="def-section-toggle">&#9660;</span>
        </div>
        <div class="def-section-body">
          <p class="def-metric-desc">Diversity of products within journey, after collapsing near-duplicates.</p>
          <p class="def-metric-desc">Group by same brand + product type + key variant attributes (color, material, functional variant). <code>diversity_ratio = num_productGroups / num_products</code></p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>ratio = 1.0 (all products distinct)</td></tr>
            <tr><td class="ds-1">Score 1</td><td>0.6 &le; ratio &lt; 1.0 (some duplication)</td></tr>
            <tr><td class="ds-0">Score 0</td><td>ratio &lt; 0.6 (significant duplication)</td></tr>
          </table>
        </div>
      </div>

      <div class="def-section">
        <div class="def-section-header" onclick="this.parentElement.classList.toggle('expanded')">
          <span>Step 5: Product Quality (per product)</span>
          <span class="def-section-toggle">&#9660;</span>
        </div>
        <div class="def-section-body">
          <div class="def-sub-heading">Product-to-Journey Relevance</div>

          <p class="def-metric-name">intentAlignment</p>
          <p class="def-metric-desc">Whether product directly supports journey's stated intent.</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Clearly and directly fulfills the journey's core intent</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Related but secondary, incomplete, or suboptimal</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Cannot resolve journey's intent; unrelated category</td></tr>
          </table>

          <p class="def-metric-name">attributeAlignment</p>
          <p class="def-metric-desc">Whether product attributes comply with journey constraints (style, fit, material, brand, price, occasion, color).</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Fully matches all relevant constraints</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Fits category/intent but secondary attributes missing or misaligned</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Contradicts or fails to meet a core attribute</td></tr>
          </table>

          <p class="def-metric-name">genderAlignment</p>
          <p class="def-metric-desc">Whether product matches journey/user gender requirement.</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Clearly matches applicable gender requirement</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Reasonably compatible (neutral/unisex)</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Explicitly conflicts with gender constraint</td></tr>
          </table>

          <div class="def-sub-heading">Product Compliance</div>
          <p class="def-metric-desc">Effectively binary per guidelines.</p>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Fully compliant and safe</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Fails compliance (health-restricted, weapons, adult content, non-product/digital)</td></tr>
          </table>

          <div class="def-sub-heading">Seller Authority</div>
          <table class="def-score-table">
            <tr><td class="ds-2">Score 2</td><td>Well-known reputable retailer or official brand store</td></tr>
            <tr><td class="ds-1">Score 1</td><td>Legitimate but not preferred/top-tier retailer</td></tr>
            <tr><td class="ds-0">Score 0</td><td>Unknown, suspicious, or inappropriate for category</td></tr>
          </table>
        </div>
      </div>

    </div>
  </div>
</div>

<main class="content">
  <div class="left-panel" id="left-panel"></div>
  <div class="right-panel" id="right-panel"></div>
</main>

<footer class="summary-bar" id="summary-bar"></footer>

<script>
// Embedded data
const DATA = {data_json};
const TOTAL_USERS = {total_users};
const OFFER_URLS = {url_json};

{JAVASCRIPT}
</script>
</body>
</html>"""


def _html_escape(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = r"""
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg: #f5f6fa;
  --card-bg: #ffffff;
  --border: #e1e4e8;
  --text: #24292e;
  --text-secondary: #586069;
  --score-good: #27ae60;
  --score-fair: #f39c12;
  --score-bad: #e74c3c;
  --accent: #0366d6;
  --accent-light: #e8f0fe;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --shadow-hover: 0 2px 8px rgba(0,0,0,0.12);
  --radius: 8px;
  --radius-sm: 4px;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Top Bar */
.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 20px;
  background: var(--card-bg);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: var(--shadow);
}
.top-bar-left { display: flex; align-items: center; gap: 12px; }
.file-title { font-size: 16px; font-weight: 600; }
.user-count {
  background: var(--accent-light);
  color: var(--accent);
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}
.top-bar-center { display: flex; align-items: center; gap: 8px; }
.top-bar-right { display: flex; align-items: center; }
.index-display { font-size: 14px; color: var(--text-secondary); font-variant-numeric: tabular-nums; min-width: 80px; text-align: center; }

.nav-btn {
  padding: 6px 14px;
  border: 1px solid var(--border);
  background: var(--card-bg);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 13px;
  color: var(--text);
  transition: all 0.15s;
}
.nav-btn:hover { background: var(--accent-light); border-color: var(--accent); color: var(--accent); }
.nav-btn:active { transform: scale(0.97); }

.user-selector { position: relative; display: flex; align-items: center; gap: 6px; }
.user-selector label { font-size: 13px; color: var(--text-secondary); font-weight: 500; }
#user-search {
  width: 280px;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s;
}
#user-search:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(3,102,214,0.15); }

.search-dropdown {
  position: absolute;
  top: 100%;
  left: 50px;
  width: 280px;
  max-height: 300px;
  overflow-y: auto;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-hover);
  z-index: 200;
}
.search-dropdown.hidden { display: none; }
.search-option {
  padding: 6px 12px;
  cursor: pointer;
  font-size: 13px;
  border-bottom: 1px solid #f0f0f0;
}
.search-option:hover, .search-option.active { background: var(--accent-light); color: var(--accent); }
.search-option .opt-idx { color: var(--text-secondary); margin-right: 6px; font-variant-numeric: tabular-nums; }

/* Main layout */
.content {
  display: flex;
  flex: 1;
  overflow: hidden;
  height: calc(100vh - 132px);
}
.left-panel {
  width: 340px;
  min-width: 300px;
  border-right: 1px solid var(--border);
  overflow-y: auto;
  background: var(--card-bg);
  padding: 16px;
}
.right-panel {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}

/* Summary bar */
.summary-bar {
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 8px 20px;
  background: var(--card-bg);
  border-top: 1px solid var(--border);
  font-size: 13px;
  color: var(--text-secondary);
  min-height: 38px;
}
.summary-stat { display: flex; align-items: center; gap: 4px; }
.summary-stat .stat-value { font-weight: 600; color: var(--text); }

/* Cards */
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 12px;
  box-shadow: var(--shadow);
}
.card-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

/* Profile */
.profile-field { margin-bottom: 10px; }
.profile-label { font-size: 12px; color: var(--text-secondary); font-weight: 500; margin-bottom: 3px; }
.profile-value { font-size: 13px; }
.profile-value.empty { color: #adb5bd; font-style: italic; }

.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
  margin: 2px 3px 2px 0;
  background: var(--accent-light);
  color: var(--accent);
}
.tag.tag-brand { background: #fff3e0; color: #e65100; }
.tag.tag-category { background: #e8f5e9; color: #2e7d32; }
.tag.tag-retailer { background: #fce4ec; color: #c62828; }
.tag.tag-style { background: #f3e5f5; color: #7b1fa2; }
.tag.tag-value { background: #e0f2f1; color: #00695c; }
.tag.tag-interest { background: #e8eaf6; color: #283593; }

/* Signals timeline */
.timeline { padding-left: 0; list-style: none; }
.timeline-item {
  position: relative;
  padding: 6px 0 6px 20px;
  border-left: 2px solid var(--border);
  font-size: 12px;
  line-height: 1.4;
}
.timeline-item::before {
  content: '';
  position: absolute;
  left: -5px;
  top: 10px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--border);
}
.timeline-item.sig-browsed::before { background: var(--accent); }
.timeline-item.sig-searched::before { background: #8e44ad; }
.timeline-item.sig-purchased::before { background: var(--score-good); }
.timeline-item.sig-carted::before { background: var(--score-fair); }
.timeline-item.sig-clicked::before { background: #3498db; }
.sig-index { color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.sig-time { color: var(--text-secondary); }
.sig-type {
  display: inline-block;
  padding: 0 5px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 500;
}
.sig-type-browsed { background: #e8f0fe; color: var(--accent); }
.sig-type-searched { background: #f3e5f5; color: #8e44ad; }
.sig-type-purchased { background: #e8f5e9; color: var(--score-good); }
.sig-type-carted { background: #fff8e1; color: #f57f17; }
.sig-type-clicked { background: #e3f2fd; color: #1565c0; }
.sig-title-text { color: var(--text); }

/* Journey diversity */
.diversity-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.diversity-groups {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.diversity-group {
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: #fafbfc;
}
.diversity-group-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-right: 4px;
}
.diversity-group-item {
  font-size: 11px;
  color: var(--text);
  padding: 1px 4px;
  border-radius: 3px;
}
.diversity-explanation {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 8px;
  padding: 8px;
  background: #f8f9fa;
  border-radius: var(--radius-sm);
  line-height: 1.5;
}

/* Journey card */
.journey-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 12px;
  box-shadow: var(--shadow);
  transition: box-shadow 0.15s;
}
.journey-card:hover { box-shadow: var(--shadow-hover); }

.journey-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 14px 16px;
  cursor: pointer;
  user-select: none;
}
.journey-header:hover { background: #fafbfc; }
.journey-title-area { flex: 1; }
.journey-index {
  font-size: 11px;
  color: var(--text-secondary);
  font-weight: 500;
  margin-bottom: 2px;
}
.journey-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 4px;
}
.journey-reason {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.4;
}
.journey-toggle {
  font-size: 18px;
  color: var(--text-secondary);
  margin-left: 12px;
  transition: transform 0.2s;
  flex-shrink: 0;
  margin-top: 4px;
}
.journey-card.expanded .journey-toggle { transform: rotate(180deg); }

.journey-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 0 16px 10px;
}

.journey-body {
  display: none;
  padding: 0 16px 16px;
  border-top: 1px solid var(--border);
}
.journey-card.expanded .journey-body { display: block; }

/* Score badge */
.score-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  cursor: default;
  position: relative;
}
.score-badge .badge-label { font-weight: 500; }
.score-0 { background: #fce4e4; color: var(--score-bad); }
.score-1 { background: #fff4d6; color: #c77c0a; }
.score-2 { background: #dcf5e7; color: #1a7f37; }

.type-tag {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
  background: #f0f0f0;
  color: var(--text-secondary);
}
.type-explicit { background: #e8f0fe; color: var(--accent); }
.type-implicit { background: #f3e5f5; color: #7b1fa2; }
.type-exploratory { background: #fff3e0; color: #e65100; }

/* Metric group separators */
.metric-group {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  position: relative;
}
.metric-group + .metric-group::before {
  content: '';
  display: inline-block;
  width: 1px;
  height: 16px;
  background: var(--border);
  margin-right: 4px;
  vertical-align: middle;
  flex-shrink: 0;
}
.metric-group-label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.3px;
  opacity: 0.7;
  white-space: nowrap;
  margin-right: 1px;
}

/* Step tag for relevance */
.step-tag {
  display: inline-block;
  padding: 1px 5px;
  border-radius: 8px;
  font-size: 9px;
  font-weight: 600;
  margin-left: 2px;
}
.step-tag-1 { background: #e3f2fd; color: #1565c0; }
.step-tag-2 { background: #f3e5f5; color: #7b1fa2; }

/* Rule ID display */
.rule-display {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-radius: var(--radius-sm);
  font-size: 11px;
  font-weight: 500;
  margin-bottom: 4px;
}
.rule-step1 { background: #e3f2fd; color: #1565c0; }
.rule-step2 { background: #f3e5f5; color: #7b1fa2; }
.rule-score-2 { border-left: 3px solid var(--score-good); }
.rule-score-1 { border-left: 3px solid var(--score-fair); }
.rule-score-0 { border-left: 3px solid var(--score-bad); }

/* Diversity ratio label */
.ratio-label {
  font-size: 11px;
  color: var(--text-secondary);
  font-weight: 500;
  margin-left: 6px;
}
.diversity-def {
  font-size: 11px;
  color: var(--text-secondary);
  font-style: italic;
  margin-bottom: 6px;
}

/* Quality issue warning */
.quality-warning {
  margin-bottom: 10px;
  padding: 6px 10px;
  background: #fff8f0;
  border-left: 3px solid var(--score-fair);
  border-radius: 2px;
  font-size: 12px;
  color: #7a5a00;
}
.quality-warning .warn-icon { margin-right: 4px; }

/* Product metrics group separator */
.pm-sep {
  display: inline-block;
  width: 1px;
  height: 12px;
  background: #d0d0d0;
  margin: 0 2px;
  vertical-align: middle;
}

/* Info icon for quality reason */
.info-icon {
  display: inline-block;
  font-size: 10px;
  margin-right: 2px;
  opacity: 0.7;
}

/* Wide tooltip for longer text */
.tooltip-wrapper .tooltip-text {
  max-width: 360px;
  white-space: normal;
  line-height: 1.4;
}

/* Product diversity section */
.prod-diversity-section {
  margin-top: 12px;
  margin-bottom: 12px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: var(--radius-sm);
}
.prod-diversity-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
}

/* Products table */
.products-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 12px;
  font-size: 12px;
}
.products-table th {
  text-align: left;
  padding: 8px 6px;
  border-bottom: 2px solid var(--border);
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.3px;
  white-space: nowrap;
}
.products-table td {
  padding: 8px 6px;
  border-bottom: 1px solid #f0f0f0;
  vertical-align: top;
}
.products-table tr:last-child td { border-bottom: none; }
.products-table tr:hover td { background: #fafbfc; }

.product-title { font-weight: 500; max-width: 300px; }
.product-link {
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
}
.product-link:hover {
  text-decoration: underline;
  color: #0550ae;
}
.product-seller { color: var(--text-secondary); font-size: 11px; }
.product-price { font-weight: 600; white-space: nowrap; }

.product-metrics { display: flex; gap: 3px; flex-wrap: wrap; }
.metric-mini {
  display: inline-flex;
  align-items: center;
  padding: 1px 5px;
  border-radius: 8px;
  font-size: 10px;
  font-weight: 600;
  cursor: default;
}
.mini-0 { background: #fce4e4; color: var(--score-bad); }
.mini-1 { background: #fff4d6; color: #c77c0a; }
.mini-2 { background: #dcf5e7; color: #1a7f37; }

.quality-reason {
  font-size: 11px;
  color: var(--score-fair);
  margin-top: 4px;
  padding: 4px 6px;
  background: #fffbf0;
  border-left: 2px solid var(--score-fair);
  border-radius: 2px;
}

.group-indicator {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: middle;
}

/* Expand/collapse all */
.journeys-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.journeys-toolbar .section-title {
  font-size: 16px;
  font-weight: 600;
}
.toolbar-actions { display: flex; gap: 6px; }
.toolbar-btn {
  padding: 4px 10px;
  border: 1px solid var(--border);
  background: var(--card-bg);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 12px;
  color: var(--text-secondary);
  transition: all 0.15s;
}
.toolbar-btn:hover { background: var(--accent-light); color: var(--accent); border-color: var(--accent); }

/* Tooltip */
.tooltip-wrapper { position: relative; }
.tooltip-wrapper .tooltip-text {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: #24292e;
  color: #fff;
  padding: 5px 10px;
  border-radius: var(--radius-sm);
  font-size: 11px;
  font-weight: 400;
  white-space: nowrap;
  z-index: 300;
  pointer-events: none;
  transition: opacity 0.15s;
}
.tooltip-wrapper .tooltip-text::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 5px solid transparent;
  border-top-color: #24292e;
}
.tooltip-wrapper:hover .tooltip-text { visibility: visible; opacity: 1; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c8c8c8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a0a0a0; }

/* Responsive adjustment */
@media (max-width: 1200px) {
  .left-panel { width: 280px; min-width: 260px; }
}

/* ======================== Metric Definitions Panel ======================== */
.def-btn {
  padding: 6px 14px;
  border: 1px solid var(--border);
  background: var(--card-bg);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 13px;
  color: var(--text);
  transition: all 0.15s;
}
.def-btn:hover { background: var(--accent-light); border-color: var(--accent); color: var(--accent); }

.def-overlay {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 500;
  background: rgba(0,0,0,0.35);
}
.def-overlay.open { display: flex; justify-content: flex-end; }

.def-panel {
  width: 520px;
  max-width: 90vw;
  height: 100%;
  background: var(--card-bg);
  box-shadow: -4px 0 20px rgba(0,0,0,0.15);
  overflow-y: auto;
  padding: 0;
  display: flex;
  flex-direction: column;
}
.def-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  background: var(--card-bg);
  z-index: 1;
}
.def-panel-header h2 { font-size: 16px; font-weight: 600; }
.def-close-btn {
  background: none;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 4px 12px;
  cursor: pointer;
  font-size: 14px;
  color: var(--text-secondary);
}
.def-close-btn:hover { background: #fce4e4; color: var(--score-bad); border-color: var(--score-bad); }

.def-panel-body { padding: 16px 20px; flex: 1; }

.def-section {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 12px;
  overflow: hidden;
}
.def-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  background: #f8f9fa;
  cursor: pointer;
  user-select: none;
  font-weight: 600;
  font-size: 13px;
}
.def-section-header:hover { background: var(--accent-light); }
.def-section-toggle {
  font-size: 14px;
  color: var(--text-secondary);
  transition: transform 0.2s;
}
.def-section.expanded .def-section-toggle { transform: rotate(180deg); }

.def-section-body {
  display: none;
  padding: 12px 14px;
  font-size: 12px;
  line-height: 1.6;
}
.def-section.expanded .def-section-body { display: block; }

.def-metric-name {
  font-weight: 600;
  color: var(--accent);
  margin-top: 8px;
  margin-bottom: 2px;
}
.def-metric-name:first-child { margin-top: 0; }
.def-metric-desc {
  color: var(--text-secondary);
  margin-bottom: 4px;
}
.def-score-table {
  width: 100%;
  border-collapse: collapse;
  margin: 4px 0 10px;
  font-size: 11px;
}
.def-score-table td {
  padding: 3px 6px;
  border-bottom: 1px solid #f0f0f0;
  vertical-align: top;
}
.def-score-table td:first-child {
  font-weight: 600;
  white-space: nowrap;
  width: 60px;
}
.ds-2 { color: var(--score-good); }
.ds-1 { color: #c77c0a; }
.ds-0 { color: var(--score-bad); }

.def-sub-heading {
  font-weight: 600;
  font-size: 12px;
  color: var(--text);
  margin: 10px 0 4px;
  padding-bottom: 2px;
  border-bottom: 1px dashed var(--border);
}

/* ======================== Filter Bar ======================== */
.filter-bar-wrapper {
  background: var(--card-bg);
  border-bottom: 1px solid var(--border);
  z-index: 99;
}
.filter-bar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 20px;
  cursor: pointer;
  user-select: none;
}
.filter-bar-header:hover { background: #fafbfc; }
.filter-bar-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 8px;
}
.filter-active-count {
  display: inline-block;
  background: var(--accent);
  color: #fff;
  font-size: 11px;
  padding: 1px 7px;
  border-radius: 10px;
  font-weight: 600;
}
.filter-active-count.hidden { display: none; }
.filter-match-info {
  font-size: 12px;
  color: var(--text-secondary);
  margin-left: 12px;
}
.filter-match-info.has-filter { color: var(--accent); font-weight: 600; }
.filter-bar-toggle {
  font-size: 14px;
  color: var(--text-secondary);
  transition: transform 0.2s;
}
.filter-bar-wrapper.expanded .filter-bar-toggle { transform: rotate(180deg); }

.filter-bar-body {
  display: none;
  padding: 8px 20px 12px;
  border-top: 1px solid #f0f0f0;
}
.filter-bar-wrapper.expanded .filter-bar-body { display: block; }

.filter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 8px;
}
.filter-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.filter-item label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.filter-item select {
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 12px;
  background: var(--card-bg);
  cursor: pointer;
  outline: none;
  color: var(--text);
}
.filter-item select:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(3,102,214,0.15); }
.filter-item select.active-filter { border-color: var(--accent); background: var(--accent-light); font-weight: 600; }

.filter-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 8px;
}
.filter-clear-btn {
  padding: 4px 14px;
  border: 1px solid var(--border);
  background: var(--card-bg);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 12px;
  color: var(--text-secondary);
  transition: all 0.15s;
}
.filter-clear-btn:hover { background: #fce4e4; color: var(--score-bad); border-color: var(--score-bad); }
</style>
"""

# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------
JAVASCRIPT = r"""
// -------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------
let currentIndex = 0;
let searchActive = false;

// -------------------------------------------------------------------------
// Filter state
// -------------------------------------------------------------------------
let filterIndex = [];     // per-user: { jdiversity: Set, jrelevance: Set, ... }
let matchingUsers = [];   // indices of users matching current filters
let activeFilters = {};   // dim -> filter value string

const FILTER_DIMS = [
  'jdiversity', 'jrelevance', 'jvalue', 'compliance', 'tone', 'coherence',
  'pdiversity', 'intent', 'attribute', 'gender', 'pcompliance', 'seller'
];

function buildFilterIndex() {
  filterIndex = [];
  for (let i = 0; i < TOTAL_USERS; i++) {
    const user = DATA[i];
    const entry = {};
    FILTER_DIMS.forEach(d => { entry[d] = new Set(); });

    // Journey diversity score (per user)
    if (user.journeyDiversity && typeof user.journeyDiversity.diversityScore === 'number') {
      entry.jdiversity.add(user.journeyDiversity.diversityScore);
    }

    // Journey-level metrics
    if (user.journeyRelevance) {
      user.journeyRelevance.forEach(r => {
        if (r && typeof r.shoppingRelevanceScore === 'number') entry.jrelevance.add(r.shoppingRelevanceScore);
      });
    }
    if (user.journeyQuality) {
      user.journeyQuality.forEach(q => {
        if (q && q.journeyAppropriateness) {
          if (typeof q.journeyAppropriateness.journeyValue === 'number') entry.jvalue.add(q.journeyAppropriateness.journeyValue);
          if (typeof q.journeyAppropriateness.contentCompliance === 'number') entry.compliance.add(q.journeyAppropriateness.contentCompliance);
        }
        if (q && q.journeyTitleQuality) {
          if (typeof q.journeyTitleQuality.tone === 'number') entry.tone.add(q.journeyTitleQuality.tone);
          if (typeof q.journeyTitleQuality.selfCoherence === 'number') entry.coherence.add(q.journeyTitleQuality.selfCoherence);
        }
      });
    }

    // Product diversity (per journey)
    if (user.productDiversity) {
      user.productDiversity.forEach(pd => {
        if (pd && typeof pd.diversityScore === 'number') entry.pdiversity.add(pd.diversityScore);
      });
    }

    // Product-level metrics
    if (user.productQuality) {
      user.productQuality.forEach(pq => {
        if (pq && pq.productQuality) {
          pq.productQuality.forEach(p => {
            if (p) {
              const rel = p.productToJourneyRelevance || {};
              if (typeof rel.intentAlignment === 'number') entry.intent.add(rel.intentAlignment);
              if (typeof rel.attributeAlignment === 'number') entry.attribute.add(rel.attributeAlignment);
              if (typeof rel.genderAlignment === 'number') entry.gender.add(rel.genderAlignment);
              if (typeof p.productCompliance === 'number') entry.pcompliance.add(p.productCompliance);
              if (typeof p.sellerAuthority === 'number') entry.seller.add(p.sellerAuthority);
            }
          });
        }
      });
    }

    filterIndex.push(entry);
  }
}

function userMatchesFilters(userIdx) {
  const entry = filterIndex[userIdx];
  for (const dim of FILTER_DIMS) {
    const filterVal = activeFilters[dim];
    if (!filterVal) continue;
    const scores = entry[dim];
    if (scores.size === 0) return false; // no data means no match
    if (filterVal === 'lte1') {
      // User has ANY score <= 1
      let found = false;
      for (const s of scores) { if (s <= 1) { found = true; break; } }
      if (!found) return false;
    } else {
      const target = parseInt(filterVal);
      if (!scores.has(target)) return false;
    }
  }
  return true;
}

function recomputeMatching() {
  matchingUsers = [];
  for (let i = 0; i < TOTAL_USERS; i++) {
    if (userMatchesFilters(i)) matchingUsers.push(i);
  }
}

function hasActiveFilters() {
  return FILTER_DIMS.some(d => !!activeFilters[d]);
}

function getActiveFilterCount() {
  return FILTER_DIMS.filter(d => !!activeFilters[d]).length;
}

function updateFilterUI() {
  const count = getActiveFilterCount();
  const countEl = document.getElementById('filter-active-count');
  const infoEl = document.getElementById('filter-match-info');

  if (count > 0) {
    countEl.textContent = count;
    countEl.classList.remove('hidden');
    infoEl.textContent = matchingUsers.length + ' matching / ' + TOTAL_USERS + ' total';
    infoEl.classList.add('has-filter');
  } else {
    countEl.classList.add('hidden');
    infoEl.textContent = '';
    infoEl.classList.remove('has-filter');
  }

  // Update index display
  updateIndexDisplay();

  // Highlight active selects
  FILTER_DIMS.forEach(dim => {
    const sel = document.getElementById('f-' + dim);
    if (sel) sel.classList.toggle('active-filter', !!activeFilters[dim]);
  });
}

function updateIndexDisplay() {
  const displayEl = document.getElementById('user-index-display');
  if (hasActiveFilters()) {
    const posInMatching = matchingUsers.indexOf(currentIndex);
    if (posInMatching >= 0) {
      displayEl.textContent = (posInMatching + 1) + ' / ' + matchingUsers.length + ' matched';
    } else {
      displayEl.textContent = '- / ' + matchingUsers.length + ' matched';
    }
  } else {
    displayEl.textContent = (currentIndex + 1) + ' / ' + TOTAL_USERS;
  }
}

function applyFilters() {
  activeFilters = {};
  FILTER_DIMS.forEach(dim => {
    const sel = document.getElementById('f-' + dim);
    if (sel && sel.value) activeFilters[dim] = sel.value;
  });
  recomputeMatching();
  updateFilterUI();

  // If current user doesn't match, navigate to first matching
  if (hasActiveFilters() && matchingUsers.length > 0 && !matchingUsers.includes(currentIndex)) {
    renderUser(matchingUsers[0]);
  } else if (hasActiveFilters() && matchingUsers.length === 0) {
    // No matches - show message
    document.getElementById('user-index-display').textContent = '0 / 0 matched';
  }
}

function clearFilters() {
  FILTER_DIMS.forEach(dim => {
    const sel = document.getElementById('f-' + dim);
    if (sel) sel.value = '';
  });
  activeFilters = {};
  recomputeMatching();
  updateFilterUI();
}

// -------------------------------------------------------------------------
// Score / color helpers
// -------------------------------------------------------------------------
const SCORE_LABELS = ['Bad', 'Fair', 'Good'];
const SCORE_CLASSES = ['score-0', 'score-1', 'score-2'];
const MINI_CLASSES = ['mini-0', 'mini-1', 'mini-2'];
const GROUP_COLORS = [
  '#a8d8ea', '#ffd3b6', '#c3aed6', '#b8e6c8', '#f7d794',
  '#dfe6e9', '#fab1a0', '#81ecec', '#ffeaa7', '#dfe6e9',
  '#b2bec3', '#fd79a8', '#74b9ff', '#55efc4', '#e17055',
];

const METRIC_TOOLTIPS = {
  'Relevance': 'Shopping Relevance: Whether user browsing signals indicate shopping intent for this journey. 2=Clearly shopping purpose | 1=Partially shopping, ambiguous | 0=Not shopping purpose. Classified via Step1 (direct interaction) or Step2 (extension logic).',
  'Value': 'Journey Value: Whether the category inherently benefits from curation (taste/fit/occasion). NOT about the specific user. 2=Benefits from curation | 1=Limited curation value | 0=Unsuitable (commodity/tools/necessities)',
  'Compliance': 'Content Compliance: Safety and online shopping suitability. 2=Completely safe & suitable | 1=Borderline concerns | 0=Prohibited content or unsuitable category (adult, weapons, health/personal, non-physical products)',
  'Tone': 'Title Tone: Natural, grammatically correct, idiomatic language quality. 2=Excellent human-like writing | 1=Acceptable with minor issues | 0=Poor, robotic, or awkward',
  'Coherence': 'Self Coherence: Whether the title is internally consistent without contradictions. 2=Logically consistent | 1=Minor inconsistencies | 0=Contradictory elements',
  'Intent': 'Intent Alignment: Whether the product supports the journey\'s stated intent. 2=Clearly fulfills core intent | 1=Related but secondary/incomplete | 0=Cannot resolve intent',
  'Attribute': 'Attribute Alignment: Whether product attributes comply with journey constraints (style, fit, material, brand, price, occasion, color). 2=Fully matches all constraints | 1=Category fits but attributes missing/misaligned | 0=Contradicts core attribute',
  'Gender': 'Gender Alignment: Whether the product matches the journey/user gender requirement. 2=Clearly matches | 1=Reasonably compatible (unisex) | 0=Explicitly conflicts',
  'ProdCompl': 'Product Compliance: Safety, legality, and category compliance. Effectively binary. 2=Fully compliant and safe | 0=Fails (health-restricted, weapons, adult, non-product/digital)',
  'Seller': 'Seller Authority: Seller trustworthiness. 2=Well-known reputable retailer or official brand store | 1=Legitimate but not top-tier | 0=Unknown, suspicious, or inappropriate',
  'ProdDiv': 'Product Diversity: Diversity of products within journey after collapsing near-duplicates. Grouped by same brand + product type + key variant attributes. Ratio = groups/products. 2=ratio 1.0 (all distinct) | 1=0.6-1.0 (some overlap) | 0=<0.6 (significant redundancy)',
  'JourneyDiv': 'Journey Diversity: Whether journeys provide meaningful diversity across user intent/decision space. Grouped by same Recipient + Primary Category + Brand. Ratio = groups/journeys. 2=ratio 1.0 (all distinct) | 1=0.6-1.0 (some overlap) | 0=<0.6 (significant redundancy)',
};

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function scoreBadge(label, score, tooltipKey) {
  const s = (typeof score === 'number') ? score : -1;
  const cls = (s >= 0 && s <= 2) ? SCORE_CLASSES[s] : '';
  const text = (s >= 0 && s <= 2) ? SCORE_LABELS[s] : '?';
  const tip = METRIC_TOOLTIPS[tooltipKey || label] || `${label}: ${text}`;
  return `<span class="tooltip-wrapper"><span class="score-badge ${cls}"><span class="badge-label">${esc(label)}</span>${text} (${s >= 0 ? s : '?'})</span><span class="tooltip-text">${esc(tip)}</span></span>`;
}

function miniMetric(label, score, tooltipKey) {
  const s = (typeof score === 'number') ? score : -1;
  const cls = (s >= 0 && s <= 2) ? MINI_CLASSES[s] : '';
  const tip = METRIC_TOOLTIPS[tooltipKey || label] || `${label}: ${s}`;
  return `<span class="tooltip-wrapper"><span class="metric-mini ${cls}">${esc(label)}:${s >= 0 ? s : '?'}</span><span class="tooltip-text">${esc(tip)}</span></span>`;
}

function typeBadge(type) {
  const t = (type || 'unknown').toLowerCase();
  let cls = '';
  if (t === 'explicit') cls = 'type-explicit';
  else if (t === 'implicit') cls = 'type-implicit';
  else if (t === 'exploratory') cls = 'type-exploratory';
  return `<span class="type-tag ${cls}">${esc(type || 'unknown')}</span>`;
}

// -------------------------------------------------------------------------
// Rule ID parsing helpers
// -------------------------------------------------------------------------
function parseRuleId(ruleId) {
  // e.g. "RULE_STEP1_SCORE2_STRONG_INTERACTION_INTENT" -> { step: 1, score: 2, label: "Strong Interaction Intent" }
  if (!ruleId) return null;
  const match = ruleId.match(/RULE_STEP(\d+)_SCORE(\d+)_(.+)/i);
  if (!match) return null;
  const label = match[3].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).replace(/\bOr\b/gi, 'or').replace(/\bAnd\b/gi, 'and');
  return { step: parseInt(match[1]), score: parseInt(match[2]), label };
}

function detectStepFromExplanation(explanation) {
  if (!explanation) return null;
  if (/RULE_STEP1/i.test(explanation)) return 1;
  if (/RULE_STEP2/i.test(explanation)) return 2;
  return null;
}

function extractRuleIds(explanation) {
  if (!explanation) return [];
  const matches = explanation.match(/RULE_STEP\d+_SCORE\d+_[A-Z_]+/gi);
  return matches ? [...new Set(matches)] : [];
}

function renderRuleTag(ruleId) {
  const parsed = parseRuleId(ruleId);
  if (!parsed) return '';
  const stepCls = parsed.step === 1 ? 'rule-step1' : 'rule-step2';
  const scoreCls = 'rule-score-' + Math.min(parsed.score, 2);
  return `<div class="rule-display ${stepCls} ${scoreCls}">Step${parsed.step}: ${esc(parsed.label)} (Score ${parsed.score})</div>`;
}

// -------------------------------------------------------------------------
// Build product group color map
// -------------------------------------------------------------------------
function buildProductGroupMap(pdEntry) {
  // pdEntry: { diversityScore, productGroups: [[id,...],[id,...]], ... }
  const map = {}; // productId -> groupIndex
  if (!pdEntry || !pdEntry.productGroups) return map;
  pdEntry.productGroups.forEach((group, gi) => {
    group.forEach(id => { map[String(id)] = gi; });
  });
  return map;
}

// -------------------------------------------------------------------------
// Render: User Profile
// -------------------------------------------------------------------------
function renderProfile(profile) {
  if (!profile || Object.keys(profile).length === 0) {
    return '<div class="card"><div class="card-title">User Profile</div><p style="color:#adb5bd;font-style:italic">No profile data</p></div>';
  }

  const fieldDefs = [
    { key: 'shoppingGenderPreference', label: 'Gender Preference', type: 'text' },
    { key: 'categoryPreferences', label: 'Category Preferences', type: 'tags', tagClass: 'tag-category' },
    { key: 'brandPreferences', label: 'Brand Preferences', type: 'tags', tagClass: 'tag-brand' },
    { key: 'retailerPreferences', label: 'Retailer Preferences', type: 'tags', tagClass: 'tag-retailer' },
    { key: 'priceSensitivity', label: 'Price Sensitivity', type: 'text' },
    { key: 'fashionStyle', label: 'Fashion Style', type: 'tags', tagClass: 'tag-style' },
    { key: 'fashionFit', label: 'Fashion Fit', type: 'tags', tagClass: 'tag-style' },
    { key: 'shoppingValues', label: 'Shopping Values', type: 'tags', tagClass: 'tag-value' },
    { key: 'contextualShoppingInterests', label: 'Contextual Interests', type: 'tags', tagClass: 'tag-interest' },
    { key: 'suggestedRelatedBrands', label: 'Suggested Related Brands', type: 'tags', tagClass: 'tag-brand' },
  ];

  let html = '<div class="card"><div class="card-title">User Profile</div>';
  for (const fd of fieldDefs) {
    const val = profile[fd.key];
    html += '<div class="profile-field">';
    html += `<div class="profile-label">${esc(fd.label)}</div>`;
    if (fd.type === 'tags') {
      const arr = Array.isArray(val) ? val : (val ? [val] : []);
      if (arr.length === 0) {
        html += '<div class="profile-value empty">N/A</div>';
      } else {
        html += '<div class="profile-value">';
        arr.forEach(t => { html += `<span class="tag ${fd.tagClass || ''}">${esc(t)}</span>`; });
        html += '</div>';
      }
    } else {
      const display = val && val !== 'general' ? val : '';
      html += `<div class="profile-value ${display ? '' : 'empty'}">${display ? esc(display) : 'N/A'}</div>`;
    }
    html += '</div>';
  }
  html += '</div>';
  return html;
}

// -------------------------------------------------------------------------
// Render: User Signals
// -------------------------------------------------------------------------
function renderSignals(readableSignals) {
  if (!readableSignals || readableSignals.length === 0) {
    return '<div class="card"><div class="card-title">User Signals</div><p style="color:#adb5bd;font-style:italic">No signals</p></div>';
  }

  let html = '<div class="card"><div class="card-title">User Signals (' + readableSignals.length + ')</div>';
  html += '<div class="timeline">';

  for (const line of readableSignals) {
    // Parse: "1 | 26 days ago | Browsed | Title..."
    const parts = line.split(' | ');
    const idx = parts[0] || '';
    const time = parts[1] || '';
    const rawType = (parts[2] || '').trim();
    const title = parts.slice(3).join(' | ');
    const typeKey = rawType.toLowerCase().replace(/[^a-z]/g, '');
    const sigClass = 'sig-' + (typeKey || 'browsed');
    const typeClass = 'sig-type-' + (typeKey || 'browsed');

    html += `<div class="timeline-item ${sigClass}">`;
    html += `<span class="sig-index">${esc(idx)}</span> `;
    html += `<span class="sig-time">${esc(time)}</span> `;
    html += `<span class="sig-type ${typeClass}">${esc(rawType)}</span>`;
    if (title) html += `<br><span class="sig-title-text">${esc(title)}</span>`;
    html += '</div>';
  }

  html += '</div></div>';
  return html;
}

// -------------------------------------------------------------------------
// Render: Journey Diversity
// -------------------------------------------------------------------------
function renderJourneyDiversity(jd) {
  if (!jd || typeof jd !== 'object') return '';

  let html = '<div class="card">';
  html += '<div class="card-title">Journey Diversity</div>';
  html += '<div class="diversity-def">Groups journeys by same recipient + category + brand. Ratio = groups / journeys.</div>';
  html += '<div class="diversity-header">';
  html += scoreBadge('Diversity', jd.diversityScore, 'JourneyDiv');

  // Compute and show ratio
  const numGroups = jd.journeyGroups ? jd.journeyGroups.length : 0;
  const totalJourneys = jd.journeyGroups ? jd.journeyGroups.reduce((s, g) => s + g.length, 0) : 0;
  if (totalJourneys > 0) {
    const ratio = (numGroups / totalJourneys).toFixed(2);
    html += `<span class="ratio-label">Ratio: ${ratio} (${numGroups} groups / ${totalJourneys} journeys)</span>`;
  }

  html += '</div>';

  if (jd.journeyGroups && jd.journeyGroups.length > 0) {
    html += '<div class="diversity-groups">';
    jd.journeyGroups.forEach((group, gi) => {
      const color = GROUP_COLORS[gi % GROUP_COLORS.length];
      html += `<div class="diversity-group" style="border-left: 3px solid ${color}">`;
      html += `<span class="diversity-group-label">G${gi + 1}</span>`;
      group.forEach(title => {
        html += `<span class="diversity-group-item">${esc(title)}</span>`;
      });
      html += '</div>';
    });
    html += '</div>';
  }

  if (jd.diversityExplanation) {
    html += `<div class="diversity-explanation">${esc(jd.diversityExplanation)}</div>`;
  }
  html += '</div>';
  return html;
}

// -------------------------------------------------------------------------
// Render: Single Journey Card
// -------------------------------------------------------------------------
function renderJourneyCard(journey, jIdx, qualityEntry, relevanceEntry, pdEntry, pqEntry) {
  const title = journey.journeyTitle || 'Untitled Journey';
  const reason = journey.JourneyReason || '';
  const products = journey.Products || [];

  // Quality metrics
  const jType = qualityEntry ? qualityEntry.journeyType : null;
  const jValue = qualityEntry && qualityEntry.journeyAppropriateness ? qualityEntry.journeyAppropriateness.journeyValue : null;
  const contentCompl = qualityEntry && qualityEntry.journeyAppropriateness ? qualityEntry.journeyAppropriateness.contentCompliance : null;
  const tone = qualityEntry && qualityEntry.journeyTitleQuality ? qualityEntry.journeyTitleQuality.tone : null;
  const coherence = qualityEntry && qualityEntry.journeyTitleQuality ? qualityEntry.journeyTitleQuality.selfCoherence : null;
  const qualExplanation = qualityEntry ? qualityEntry.explanation : null;

  // Relevance
  const relScore = relevanceEntry ? relevanceEntry.shoppingRelevanceScore : null;
  const relExplanation = relevanceEntry ? relevanceEntry.explanation : null;

  // Product diversity
  const pdScore = pdEntry ? pdEntry.diversityScore : null;
  const pdReason = pdEntry ? pdEntry.diversityReason : null;
  const pgMap = buildProductGroupMap(pdEntry);
  const numGroups = pdEntry && pdEntry.productGroups ? pdEntry.productGroups.length : 0;

  // Product quality map: productId -> metrics
  const pqMap = {};
  if (pqEntry && pqEntry.productQuality) {
    pqEntry.productQuality.forEach(pq => {
      pqMap[String(pq.productId)] = pq;
    });
  }

  let html = `<div class="journey-card" id="journey-${jIdx}">`;

  // Header (always visible)
  html += `<div class="journey-header" onclick="toggleJourney(${jIdx})">`;
  html += '<div class="journey-title-area">';
  html += `<div class="journey-index">Journey ${jIdx + 1}</div>`;
  html += `<div class="journey-title">${esc(title)}</div>`;
  html += `<div class="journey-reason">${esc(reason)}</div>`;
  html += '</div>';
  html += '<span class="journey-toggle">&#9660;</span>';
  html += '</div>';

  // Metric badges row (always visible) - grouped by UHRS structure
  html += '<div class="journey-metrics">';

  // Group 1: Relevance
  html += '<span class="metric-group">';
  html += '<span class="metric-group-label">Relevance</span>';
  if (relScore !== null) html += scoreBadge('Relevance', relScore, 'Relevance');
  if (jType) html += typeBadge(jType);
  // Show Step1/Step2 tag if detectable from explanation
  const stepNum = detectStepFromExplanation(relExplanation);
  if (stepNum) {
    html += `<span class="step-tag step-tag-${stepNum}">Step${stepNum}</span>`;
  }
  html += '</span>';

  // Group 2: Appropriateness
  html += '<span class="metric-group">';
  html += '<span class="metric-group-label">Appropriateness</span>';
  if (jValue !== null) html += scoreBadge('Value', jValue, 'Value');
  if (contentCompl !== null) html += scoreBadge('Compliance', contentCompl, 'Compliance');
  html += '</span>';

  // Group 3: Title Quality
  html += '<span class="metric-group">';
  html += '<span class="metric-group-label">Title</span>';
  if (tone !== null) html += scoreBadge('Tone', tone, 'Tone');
  if (coherence !== null) html += scoreBadge('Coherence', coherence, 'Coherence');
  html += '</span>';

  html += '</div>';

  // Body (expandable)
  html += '<div class="journey-body">';

  // Explanations
  const hasQualityIssue = [jValue, contentCompl, tone, coherence].some(v => typeof v === 'number' && v < 2);
  if (qualExplanation) {
    if (hasQualityIssue) {
      html += `<div class="quality-warning"><span class="warn-icon">&#9888;</span><strong>Quality Issue:</strong> ${esc(qualExplanation)}</div>`;
    } else {
      html += `<div class="quality-reason" style="margin-bottom:10px"><strong>Quality note:</strong> ${esc(qualExplanation)}</div>`;
    }
  }
  if (relExplanation) {
    html += '<div class="quality-reason" style="margin-bottom:10px;border-left-color:var(--accent);background:#f0f6ff">';
    html += '<strong>Relevance:</strong> ';
    // Parse and display rule IDs
    const ruleIds = extractRuleIds(relExplanation);
    if (ruleIds.length > 0) {
      html += '<div style="margin:4px 0">';
      ruleIds.forEach(rid => { html += renderRuleTag(rid); });
      html += '</div>';
      // Show explanation with rule IDs kept for reference
      html += `<div style="margin-top:4px;font-size:11px;opacity:0.85">${esc(relExplanation)}</div>`;
    } else {
      html += esc(relExplanation);
    }
    html += '</div>';
  }

  // Product diversity
  html += '<div class="prod-diversity-section">';
  html += '<div class="prod-diversity-header">';
  html += 'Product Diversity ';
  if (pdScore !== null) html += scoreBadge('ProdDiv', pdScore, 'ProdDiv');
  // Show computed ratio
  const numProducts = products.length;
  if (numGroups > 0 && numProducts > 0) {
    const pdRatio = (numGroups / numProducts).toFixed(2);
    html += `<span class="ratio-label">Ratio: ${pdRatio} (${numGroups} groups / ${numProducts} products)</span>`;
  }
  html += '</div>';
  html += '<div class="diversity-def">Groups products by same brand + type + key attributes. Ratio = groups / products.</div>';
  if (numGroups > 0) {
    html += '<div class="diversity-groups">';
    pdEntry.productGroups.forEach((group, gi) => {
      const color = GROUP_COLORS[gi % GROUP_COLORS.length];
      html += `<div class="diversity-group" style="border-left: 3px solid ${color}">`;
      html += `<span class="diversity-group-label">G${gi + 1}</span>`;
      group.forEach(pid => {
        // Find product title
        const prod = products.find(p => String(p.OfferId) === String(pid));
        const ptitle = prod ? prod.Title : pid;
        const shortTitle = ptitle.length > 40 ? ptitle.substring(0, 40) + '...' : ptitle;
        html += `<span class="diversity-group-item" title="${esc(ptitle)}">${esc(shortTitle)}</span>`;
      });
      html += '</div>';
    });
    html += '</div>';
  }
  if (pdReason) {
    html += `<div class="diversity-explanation">${esc(pdReason)}</div>`;
  }
  html += '</div>';

  // Products table
  if (products.length > 0) {
    html += '<table class="products-table">';
    html += '<thead><tr>';
    html += '<th>#</th><th>Product</th><th>Seller</th><th>Price</th><th>P2J Relevance</th><th>Compl</th><th>Seller</th>';
    html += '</tr></thead><tbody>';

    for (const prod of products) {
      const pid = String(prod.OfferId || '');
      const groupIdx = pgMap[pid];
      const groupColor = groupIdx !== undefined ? GROUP_COLORS[groupIdx % GROUP_COLORS.length] : null;
      const pq = pqMap[pid];

      html += '<tr>';

      // Rank + group color
      html += '<td>';
      if (groupColor) {
        html += `<span class="group-indicator" style="background:${groupColor}" title="Diversity Group ${groupIdx + 1}"></span>`;
      }
      html += esc(prod.Rank);
      html += '</td>';

      // Title + OfferUrl link
      const offerUrl = OFFER_URLS[pid] || '';
      html += '<td>';
      if (offerUrl) {
        html += `<div class="product-title"><a href="${esc(offerUrl)}" target="_blank" rel="noopener" class="product-link">${esc(prod.Title || '')}</a></div>`;
      } else {
        html += `<div class="product-title">${esc(prod.Title || '')}</div>`;
      }
      if (pq && pq.qualityReason) {
        html += `<div class="quality-reason"><span class="info-icon">&#9432;</span>${esc(pq.qualityReason)}</div>`;
      }
      html += '</td>';

      // Seller
      html += `<td><span class="product-seller">${esc(prod.Seller || '')}</span></td>`;

      // Price
      html += `<td><span class="product-price">${esc(prod.Price || '')}</span></td>`;

      // Metrics - split into three columns: P2J Relevance | Compliance | Seller
      if (pq) {
        const rel = pq.productToJourneyRelevance || {};
        // P2J Relevance group
        html += '<td><div class="product-metrics">';
        html += miniMetric('Int', rel.intentAlignment, 'Intent');
        html += miniMetric('Attr', rel.attributeAlignment, 'Attribute');
        html += miniMetric('Gen', rel.genderAlignment, 'Gender');
        html += '</div></td>';
        // Compliance (binary 2/0)
        html += '<td><div class="product-metrics">';
        html += miniMetric('Compl', pq.productCompliance, 'ProdCompl');
        html += '</div></td>';
        // Seller authority
        html += '<td><div class="product-metrics">';
        html += miniMetric('Sell', pq.sellerAuthority, 'Seller');
        html += '</div></td>';
      } else {
        html += '<td colspan="3"><span style="color:#adb5bd;font-size:11px">No data</span></td>';
      }

      html += '</tr>';
    }

    html += '</tbody></table>';
  }

  html += '</div>'; // journey-body
  html += '</div>'; // journey-card

  return html;
}

// -------------------------------------------------------------------------
// Match quality/relevance/diversity entries to journey by title
// -------------------------------------------------------------------------
function findByTitle(arr, title) {
  if (!arr || !Array.isArray(arr)) return null;
  return arr.find(e => e && e.journeyTitle === title) || null;
}

// -------------------------------------------------------------------------
// Render: Full user
// -------------------------------------------------------------------------
function renderUser(idx) {
  if (idx < 0 || idx >= TOTAL_USERS) return;
  currentIndex = idx;
  const user = DATA[idx];

  // Update header
  updateIndexDisplay();
  document.getElementById('user-search').value = `${idx + 1} - ${user.userId}`;

  // Left panel
  const leftPanel = document.getElementById('left-panel');
  let leftHtml = '';
  leftHtml += `<div style="margin-bottom:12px;font-size:13px;color:var(--text-secondary)">User ID: <strong style="color:var(--text)">${esc(user.userId)}</strong></div>`;
  leftHtml += renderProfile(user.userProfile);
  leftHtml += renderSignals(user.readableSignals);
  leftPanel.innerHTML = leftHtml;

  // Right panel
  const rightPanel = document.getElementById('right-panel');
  let rightHtml = '';

  // Journey diversity
  rightHtml += renderJourneyDiversity(user.journeyDiversity);

  // Journeys toolbar
  const numJ = user.journeys ? user.journeys.length : 0;
  rightHtml += '<div class="journeys-toolbar">';
  rightHtml += `<span class="section-title">Journeys (${numJ})</span>`;
  rightHtml += '<div class="toolbar-actions">';
  rightHtml += '<button class="toolbar-btn" onclick="expandAll()">Expand All</button>';
  rightHtml += '<button class="toolbar-btn" onclick="collapseAll()">Collapse All</button>';
  rightHtml += '</div></div>';

  // Journey cards
  if (user.journeys && user.journeys.length > 0) {
    user.journeys.forEach((journey, jIdx) => {
      const title = journey.journeyTitle || '';
      const qualityEntry = findByTitle(user.journeyQuality, title);
      const relevanceEntry = findByTitle(user.journeyRelevance, title);
      const pdEntry = findByTitle(user.productDiversity, title);
      const pqEntry = findByTitle(user.productQuality, title);
      rightHtml += renderJourneyCard(journey, jIdx, qualityEntry, relevanceEntry, pdEntry, pqEntry);
    });
  } else {
    rightHtml += '<div style="color:#adb5bd;font-style:italic;padding:20px">No journeys</div>';
  }

  rightPanel.innerHTML = rightHtml;

  // Summary bar
  renderSummary(user);
}

// -------------------------------------------------------------------------
// Render: Summary bar
// -------------------------------------------------------------------------
function renderSummary(user) {
  const bar = document.getElementById('summary-bar');
  const numJ = user.journeys ? user.journeys.length : 0;
  let totalProducts = 0;
  let totalProdScore = 0;
  let totalProdCount = 0;
  let totalRelScore = 0;
  let totalRelCount = 0;
  let totalJValue = 0;
  let totalJValueCount = 0;
  let totalContentCompl = 0;
  let totalContentComplCount = 0;
  let totalPdScore = 0;
  let totalPdCount = 0;

  if (user.journeys) {
    user.journeys.forEach(j => {
      totalProducts += (j.Products || []).length;
    });
  }

  if (user.journeyRelevance) {
    user.journeyRelevance.forEach(r => {
      if (r && typeof r.shoppingRelevanceScore === 'number') {
        totalRelScore += r.shoppingRelevanceScore;
        totalRelCount++;
      }
    });
  }

  if (user.journeyQuality) {
    user.journeyQuality.forEach(q => {
      if (q && q.journeyAppropriateness) {
        if (typeof q.journeyAppropriateness.journeyValue === 'number') {
          totalJValue += q.journeyAppropriateness.journeyValue;
          totalJValueCount++;
        }
        if (typeof q.journeyAppropriateness.contentCompliance === 'number') {
          totalContentCompl += q.journeyAppropriateness.contentCompliance;
          totalContentComplCount++;
        }
      }
    });
  }

  if (user.productDiversity) {
    user.productDiversity.forEach(pd => {
      if (pd && typeof pd.diversityScore === 'number') {
        totalPdScore += pd.diversityScore;
        totalPdCount++;
      }
    });
  }

  if (user.productQuality) {
    user.productQuality.forEach(pq => {
      if (pq && pq.productQuality) {
        pq.productQuality.forEach(p => {
          if (p && p.productToJourneyRelevance) {
            const r = p.productToJourneyRelevance;
            const avg = ((r.intentAlignment || 0) + (r.attributeAlignment || 0) + (r.genderAlignment || 0)) / 3;
            totalProdScore += avg;
            totalProdCount++;
          }
        });
      }
    });
  }

  const avgRel = totalRelCount > 0 ? (totalRelScore / totalRelCount).toFixed(2) : 'N/A';
  const avgProd = totalProdCount > 0 ? (totalProdScore / totalProdCount).toFixed(2) : 'N/A';
  const avgJValue = totalJValueCount > 0 ? (totalJValue / totalJValueCount).toFixed(2) : 'N/A';
  const avgContentCompl = totalContentComplCount > 0 ? (totalContentCompl / totalContentComplCount).toFixed(2) : 'N/A';
  const avgPdScore = totalPdCount > 0 ? (totalPdScore / totalPdCount).toFixed(2) : 'N/A';
  const divScore = user.journeyDiversity && typeof user.journeyDiversity.diversityScore === 'number'
    ? user.journeyDiversity.diversityScore : 'N/A';

  let html = '';
  html += `<div class="summary-stat">Journeys: <span class="stat-value">${numJ}</span></div>`;
  html += `<div class="summary-stat">Products: <span class="stat-value">${totalProducts}</span></div>`;
  html += `<div class="summary-stat">Avg Relevance: <span class="stat-value">${avgRel}</span></div>`;
  html += `<div class="summary-stat">Avg J. Value: <span class="stat-value">${avgJValue}</span></div>`;
  html += `<div class="summary-stat">Avg Compliance: <span class="stat-value">${avgContentCompl}</span></div>`;
  html += `<div class="summary-stat">Avg Prod Quality: <span class="stat-value">${avgProd}</span></div>`;
  html += `<div class="summary-stat">Avg Prod Diversity: <span class="stat-value">${avgPdScore}</span></div>`;
  html += `<div class="summary-stat">J. Diversity: <span class="stat-value">${divScore}</span></div>`;
  bar.innerHTML = html;
}

// -------------------------------------------------------------------------
// Journey expand/collapse
// -------------------------------------------------------------------------
function toggleJourney(jIdx) {
  const card = document.getElementById('journey-' + jIdx);
  if (card) card.classList.toggle('expanded');
}

function expandAll() {
  document.querySelectorAll('.journey-card').forEach(c => c.classList.add('expanded'));
}

function collapseAll() {
  document.querySelectorAll('.journey-card').forEach(c => c.classList.remove('expanded'));
}

// -------------------------------------------------------------------------
// Navigation
// -------------------------------------------------------------------------
function goTo(idx) {
  if (idx >= 0 && idx < TOTAL_USERS) renderUser(idx);
}

function goPrev() {
  if (hasActiveFilters() && matchingUsers.length > 0) {
    const pos = matchingUsers.indexOf(currentIndex);
    if (pos > 0) goTo(matchingUsers[pos - 1]);
    else if (pos === -1) {
      // Find closest matching user before currentIndex
      for (let i = matchingUsers.length - 1; i >= 0; i--) {
        if (matchingUsers[i] < currentIndex) { goTo(matchingUsers[i]); return; }
      }
    }
  } else {
    goTo(currentIndex - 1);
  }
}

function goNext() {
  if (hasActiveFilters() && matchingUsers.length > 0) {
    const pos = matchingUsers.indexOf(currentIndex);
    if (pos >= 0 && pos < matchingUsers.length - 1) goTo(matchingUsers[pos + 1]);
    else if (pos === -1) {
      // Find closest matching user after currentIndex
      for (let i = 0; i < matchingUsers.length; i++) {
        if (matchingUsers[i] > currentIndex) { goTo(matchingUsers[i]); return; }
      }
    }
  } else {
    goTo(currentIndex + 1);
  }
}

document.getElementById('btn-prev').addEventListener('click', goPrev);
document.getElementById('btn-next').addEventListener('click', goNext);

document.addEventListener('keydown', (e) => {
  // Don't navigate when search or filter selects are focused
  const tag = document.activeElement ? document.activeElement.tagName : '';
  if (tag === 'INPUT' || tag === 'SELECT') return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); goPrev(); }
  if (e.key === 'ArrowRight') { e.preventDefault(); goNext(); }
});

// -------------------------------------------------------------------------
// Search / user selector
// -------------------------------------------------------------------------
const searchInput = document.getElementById('user-search');
const dropdown = document.getElementById('search-dropdown');
let filteredOptions = [];
let activeOptionIdx = -1;

function buildOptions(filter) {
  filteredOptions = [];
  const lf = filter.toLowerCase();
  const useFilterSet = hasActiveFilters();
  const matchSet = useFilterSet ? new Set(matchingUsers) : null;
  for (let i = 0; i < TOTAL_USERS; i++) {
    if (useFilterSet && !matchSet.has(i)) continue;
    const label = `${i + 1} - ${DATA[i].userId}`;
    if (!filter || label.toLowerCase().includes(lf) || String(i + 1) === filter) {
      filteredOptions.push({ idx: i, label });
    }
    if (filteredOptions.length >= 50) break; // limit dropdown size
  }
  return filteredOptions;
}

function showDropdown(filter) {
  const opts = buildOptions(filter);
  if (opts.length === 0) {
    dropdown.classList.add('hidden');
    return;
  }
  activeOptionIdx = -1;
  let html = '';
  opts.forEach((opt, oi) => {
    html += `<div class="search-option" data-idx="${opt.idx}" data-oi="${oi}">`;
    html += `<span class="opt-idx">${opt.idx + 1}.</span>${esc(opt.label.substring(String(opt.idx + 1).length + 3))}`;
    html += '</div>';
  });
  dropdown.innerHTML = html;
  dropdown.classList.remove('hidden');

  // Click handlers
  dropdown.querySelectorAll('.search-option').forEach(el => {
    el.addEventListener('click', () => {
      goTo(parseInt(el.dataset.idx));
      dropdown.classList.add('hidden');
    });
  });
}

searchInput.addEventListener('focus', () => {
  searchInput.select();
  showDropdown(searchInput.value);
});

searchInput.addEventListener('input', () => {
  showDropdown(searchInput.value.trim());
});

searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    dropdown.classList.add('hidden');
    searchInput.blur();
    return;
  }
  if (e.key === 'Enter') {
    e.preventDefault();
    if (activeOptionIdx >= 0 && activeOptionIdx < filteredOptions.length) {
      goTo(filteredOptions[activeOptionIdx].idx);
    } else {
      // Try direct index
      const val = searchInput.value.trim();
      const num = parseInt(val);
      if (num >= 1 && num <= TOTAL_USERS) {
        goTo(num - 1);
      } else if (filteredOptions.length > 0) {
        goTo(filteredOptions[0].idx);
      }
    }
    dropdown.classList.add('hidden');
    return;
  }
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    const dir = e.key === 'ArrowDown' ? 1 : -1;
    activeOptionIdx = Math.max(-1, Math.min(filteredOptions.length - 1, activeOptionIdx + dir));
    dropdown.querySelectorAll('.search-option').forEach((el, i) => {
      el.classList.toggle('active', i === activeOptionIdx);
    });
    if (activeOptionIdx >= 0) {
      const activeEl = dropdown.querySelector('.search-option.active');
      if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });
    }
  }
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.user-selector')) {
    dropdown.classList.add('hidden');
  }
});

// -------------------------------------------------------------------------
// Initial render
// -------------------------------------------------------------------------

// Build filter index at startup
buildFilterIndex();
recomputeMatching();

// Definitions panel
document.getElementById('btn-definitions').addEventListener('click', () => {
  document.getElementById('def-overlay').classList.add('open');
});
document.getElementById('def-close-btn').addEventListener('click', () => {
  document.getElementById('def-overlay').classList.remove('open');
});
document.getElementById('def-overlay').addEventListener('click', (e) => {
  if (e.target === document.getElementById('def-overlay')) {
    document.getElementById('def-overlay').classList.remove('open');
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('def-overlay').classList.contains('open')) {
    document.getElementById('def-overlay').classList.remove('open');
  }
});

// Filter bar toggle
document.getElementById('filter-bar-header').addEventListener('click', () => {
  document.getElementById('filter-bar-wrapper').classList.toggle('expanded');
});

// Filter change handlers
FILTER_DIMS.forEach(dim => {
  const sel = document.getElementById('f-' + dim);
  if (sel) sel.addEventListener('change', applyFilters);
});
document.getElementById('filter-clear-btn').addEventListener('click', clearFilters);

if (TOTAL_USERS > 0) {
  renderUser(0);
}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML visualization from a shopping journey evaluation TSV file."
    )
    parser.add_argument("tsv_file", help="Path to the input TSV file")
    parser.add_argument(
        "--output", "-o",
        help="Path for the output HTML file (default: <input_name>_visualization.html)",
    )
    parser.add_argument(
        "--product_file",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/0307_EnUs_Product.tsv",
        help="Path to JourneyProduct TSV file for OfferUrl lookup",
    )
    args = parser.parse_args()

    tsv_path = os.path.abspath(args.tsv_file)
    if not os.path.isfile(tsv_path):
        print(f"ERROR: File not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        stem = Path(tsv_path).stem
        out_path = os.path.join(os.path.dirname(tsv_path), f"{stem}_visualization.html")

    source_filename = os.path.basename(tsv_path)
    print(f"Reading {tsv_path}...", file=sys.stderr)
    records = read_tsv(tsv_path)

    if not records:
        print("ERROR: No valid records found.", file=sys.stderr)
        sys.exit(1)

    # Load product OfferUrl mapping
    offer_urls = load_offer_urls(args.product_file)

    print(f"Generating HTML for {len(records)} users...", file=sys.stderr)
    html = generate_html(records, source_filename, offer_urls=offer_urls)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated HTML ({len(html):,} bytes) with {len(records)} users", file=sys.stderr)
    print(out_path)


if __name__ == "__main__":
    main()
