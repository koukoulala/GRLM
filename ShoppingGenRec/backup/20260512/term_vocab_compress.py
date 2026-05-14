#!/usr/bin/env python3
"""Term Vocabulary Compression for GRLM ShoppingGenRec TIDs.

Compresses ~982K distinct terms (from summary_words_norm) via:
  Step 0: Lowercase all terms (merges ~111K case variants)
  Step 1: Per-slot active vocab: keep terms with freq >= N (default N=2)
  Step 2: Map low-freq terms to nearest active-vocab term via embedding
           similarity.  If no active match, cluster low-freq terms among
           themselves and pick the highest-freq canonical.
           If all freq=1 in a cluster, keep original.
  Step 3: Audit log for merges affecting >= K items
  Step 4: New-item mapping via pre-built FAISS index

Persisted artifacts (in --artifact_dir):
  config.json             - build parameters
  active_vocab.json       - per-slot {term: freq}  (dict lookup for new items)
  term_mapping.json       - per-slot {orig: mapped} (low-freq -> canonical)
  faiss_slot_{i}.index    - FAISS index for slot i  (new-item NN search)
  vocab_terms_slot_{i}.json - ordered term list aligned with FAISS index
  audit_log.json          - high-impact merges for human review
  build_stats.json        - per-slot build statistics

Why N=2 (default):
  N=2 yields ~574K active terms covering 98.2% of all item-term occurrences.
  Only terms appearing exactly once (hapax) are candidates for mapping.
  This is the safest threshold — anything that appeared twice is likely
  intentional.  N=3 (404K, 97.3%) is a reasonable alternative for more
  aggressive compression.

Update cadence:
  Re-run build when the item catalog changes significantly (e.g., monthly).
  Between rebuilds, new items use the FAISS index for real-time mapping.

Usage:
  # Full build (embed + compress + save)
  python term_vocab_compress.py --input /path/to/id2meta_with_norm.json

  # Analyze N values only (no embedding, fast)
  python term_vocab_compress.py --input /path/to/id2meta_with_norm.json --analyze

  # Test new-item mapping with pre-built artifacts
  python term_vocab_compress.py --test_only --artifact_dir /path/to/artifacts

  # Re-compress with existing artifacts (skip expensive embedding step)
  python term_vocab_compress.py --input ... --load_artifacts --artifact_dir ...
"""

import argparse
import json
import os
import re
import sys
import time
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

# ===========================================================================
# Constants
# ===========================================================================
N_SLOTS = 7
SLOT_NAMES = [
    "product_type", "function", "form",
    "attribute", "brand", "seller", "audience",
]
SLOT_LABELS = [
    "Prod.type", "Function", "Form",
    "Attribute", "Brand", "Seller", "Audience",
]

# Regex for cleaning control chars and special symbols from terms
_CLEAN_TERM_RE = re.compile(
    r'[\x00-\x1f\x7f-\x9f'
    r'\u200b-\u200f\ufeff'
    r'\u25a0-\u25ff'
    r'\u2500-\u257f'
    r'\u2580-\u259f]'
)


def clean_term(term):
    """Strip control characters, box-drawing, and block-element symbols."""
    return _CLEAN_TERM_RE.sub('', term).strip()


def _extract_numbers(term):
    """Extract all numeric values from a term as a sorted tuple.

    Treats hyphens between word-chars and digits as separators (not negative
    signs), so '0-100psi' yields (0, 100) not (-100, 0).
    """
    cleaned = re.sub(r'(?<=\w)-(?=\d)', ' ', term)
    nums = re.findall(r'\d+(?:\.\d+)?', cleaned)
    return tuple(sorted(float(n) for n in nums))


def _has_digit(term):
    return any(c.isdigit() for c in term)


def should_block_mapping(src_term, tgt_term, sim, text_threshold=0.90,
                         digit_threshold=0.95):
    """Decide whether to block a low-freq → active embedding mapping.

    Rules:
      1. Short terms (<=2 chars): always block.
      2. Either term contains digits: always block.
      3. Pure text terms: block if sim < text_threshold.
    """
    if len(src_term) <= 2:
        return True
    if _has_digit(src_term) or _has_digit(tgt_term):
        return True
    return sim < text_threshold


# ===========================================================================
# Union-Find (for low-freq self-clustering)
# ===========================================================================
class UnionFind:
    """Disjoint-set with path compression and union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self):
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return list(groups.values())


# ===========================================================================
# Embedding helpers
# ===========================================================================
_embed_model_cache = {}


def get_embed_model(model_name):
    """Load (and cache) a sentence-transformer model."""
    if model_name not in _embed_model_cache:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name} ...")
        _embed_model_cache[model_name] = SentenceTransformer(
            model_name, trust_remote_code=True,
        )
    return _embed_model_cache[model_name]


def embed_terms(terms, model, batch_size=2048, desc="Embedding",
                num_gpus=None):
    """Batch-embed terms → L2-normalized float32 array.

    If num_gpus > 1, uses SentenceTransformer multi-process pool to
    distribute encoding across multiple GPUs in parallel.
    """
    if not terms:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    import torch
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus > 1:
        print(f"  Using {num_gpus} GPUs for parallel encoding ...")
        pool = model.start_multi_process_pool(
            target_devices=[f"cuda:{i}" for i in range(num_gpus)]
        )
        embs = model.encode_multi_process(
            terms, pool,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    else:
        embs = model.encode(
            terms,
            batch_size=batch_size,
            show_progress_bar=(len(terms) > 5000),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return np.asarray(embs, dtype=np.float32)


# ===========================================================================
# Data loading
# ===========================================================================
def load_id2meta(path):
    print(f"Loading {path} ...")
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data):,} items loaded in {time.time() - t0:.1f}s")
    return data


def count_slot_terms(id2meta):
    """Lowercase, clean, and count term frequencies per slot (Step 0+1)."""
    slot_counts = [Counter() for _ in range(N_SLOTS)]
    for meta in tqdm(id2meta.values(), desc="Counting terms",
                     mininterval=10):
        sw = meta.get("summary_words_norm", [])
        if sw and len(sw) >= N_SLOTS:
            for i in range(N_SLOTS):
                slot_counts[i][clean_term(sw[i].lower())] += 1
    return slot_counts


# ===========================================================================
# TermVocabCompressor
# ===========================================================================
class TermVocabCompressor:
    """Per-slot term vocabulary compressor with FAISS-backed NN search."""

    def __init__(self, artifact_dir, min_freq=3, map_threshold=0.95,
                 digit_threshold=0.95, cluster_threshold=0.95,
                 embed_model="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3-Embedding-0.6B",
                 batch_size=2048, audit_min_items=1000):
        self.artifact_dir = artifact_dir
        self.min_freq = min_freq
        self.map_threshold = map_threshold
        self.digit_threshold = digit_threshold
        self.cluster_threshold = cluster_threshold
        self.embed_model_name = embed_model
        self.batch_size = batch_size
        self.audit_min_items = audit_min_items

        # Populated by build() or load()
        self.active_vocab = [{} for _ in range(N_SLOTS)]
        self.term_mapping = [{} for _ in range(N_SLOTS)]
        self.faiss_indices = [None] * N_SLOTS
        self.active_term_lists = [[] for _ in range(N_SLOTS)]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, id2meta, slot_counts=None):
        """Build active vocab, term mappings, and FAISS indices.

        Optimization: embeds each unique term only once across all slots,
        then slices per-slot embeddings from the shared array.
        """
        import faiss

        os.makedirs(self.artifact_dir, exist_ok=True)

        if slot_counts is None:
            slot_counts = count_slot_terms(id2meta)

        # --- Collect all unique terms across all slots (embed once) ---
        all_unique = set()
        for s in range(N_SLOTS):
            all_unique.update(slot_counts[s].keys())
        all_unique = sorted(all_unique)
        print(f"\nUnique terms across all slots: {len(all_unique):,}")

        model = get_embed_model(self.embed_model_name)
        print(f"Embedding {len(all_unique):,} unique terms (one-time) ...")
        all_embs = embed_terms(all_unique, model, self.batch_size,
                               "Global embedding")
        term2idx = {t: i for i, t in enumerate(all_unique)}
        dim = all_embs.shape[1]
        print(f"  Embedding dim: {dim}")

        # --- Per-slot processing ---
        audit_entries = []
        build_stats = []
        total_active = 0
        total_mapped = 0
        total_kept = 0

        for s in range(N_SLOTS):
            counts = slot_counts[s]
            active = {t: c for t, c in counts.items() if c >= self.min_freq}
            low = {t: c for t, c in counts.items() if c < self.min_freq}

            print(f"\n{'=' * 60}")
            print(f"Slot {s + 1} ({SLOT_NAMES[s]}): "
                  f"{len(counts):,} total | "
                  f"{len(active):,} active (freq>={self.min_freq}) | "
                  f"{len(low):,} low-freq")

            self.active_vocab[s] = active
            active_list = sorted(active.keys())
            self.active_term_lists[s] = active_list

            # Extract active embeddings from global array
            active_idx = np.array([term2idx[t] for t in active_list])
            active_embs = all_embs[active_idx] if len(active_idx) else \
                np.zeros((0, dim), dtype=np.float32)

            # Build FAISS index for active vocab
            index = faiss.IndexFlatIP(dim)
            if len(active_embs):
                index.add(active_embs)
            self.faiss_indices[s] = index

            # --- Print top 20 active terms ---
            top20 = sorted(active.items(), key=lambda x: -x[1])[:20]
            print(f"  Top 20 active terms:")
            for t, c in top20:
                print(f"    {c:>10,}  {t}")

            if not low:
                self.term_mapping[s] = {}
                total_active += len(active)
                faiss.write_index(
                    index,
                    os.path.join(self.artifact_dir, f"faiss_slot_{s}.index"),
                )
                build_stats.append({
                    "slot": s, "name": SLOT_NAMES[s],
                    "total": len(counts), "active": len(active),
                    "low_freq": 0, "mapped_to_active": 0,
                    "unmapped_no_neighbor": 0,
                    "self_clustered": 0, "kept_original": 0,
                })
                continue

            # Step 2a: map low-freq → nearest active term
            low_list = sorted(low.keys())
            low_idx_arr = np.array([term2idx[t] for t in low_list])
            low_embs = all_embs[low_idx_arr]

            sims, nn_ids = index.search(low_embs, 1)

            mapping = {}
            unmapped_j = []  # indices into low_list
            mapped_examples = []   # (orig, target, sim) for reporting
            unmapped_examples = [] # (orig, best_active, sim) for reporting
            blocked_by_number = 0
            blocked_by_short = 0
            for j, term in enumerate(low_list):
                best_sim = float(sims[j][0]) if len(active_list) > 0 else 0.0
                target = active_list[nn_ids[j][0]] if len(active_list) > 0 else ""
                if len(active_list) > 0 and not should_block_mapping(
                        term, target, best_sim,
                        text_threshold=self.map_threshold,
                        digit_threshold=self.digit_threshold):
                    mapping[term] = target
                    if len(mapped_examples) < 20:
                        mapped_examples.append((term, target, best_sim))
                else:
                    unmapped_j.append(j)
                    if len(term) <= 2:
                        blocked_by_short += 1
                    elif _has_digit(term) or _has_digit(target):
                        blocked_by_number += 1
                    if len(unmapped_examples) < 20 and len(active_list) > 0:
                        unmapped_examples.append(
                            (term, target, best_sim))

            mapped_to_active = len(mapping)

            # --- Similarity distribution for low-freq terms ---
            all_best_sims = [float(sims[j][0]) for j in range(len(low_list))
                             ] if len(active_list) > 0 else []
            if all_best_sims:
                sim_arr = np.array(all_best_sims)
                buckets = [
                    (">=0.95", np.sum(sim_arr >= 0.95)),
                    ("0.90-0.95", np.sum((sim_arr >= 0.90) & (sim_arr < 0.95))),
                    ("0.85-0.90", np.sum((sim_arr >= 0.85) & (sim_arr < 0.90))),
                    ("0.80-0.85", np.sum((sim_arr >= 0.80) & (sim_arr < 0.85))),
                    ("<0.80", np.sum(sim_arr < 0.80)),
                ]
                print(f"\n  Low-freq → active similarity distribution:")
                for label, cnt in buckets:
                    pct = cnt / len(sim_arr) * 100
                    print(f"    {label:>10}: {int(cnt):>8,} ({pct:>5.1f}%)")

            print(f"\n  Mapped to active:  {mapped_to_active:,}")
            if blocked_by_number or blocked_by_short:
                print(f"  Blocked (number mismatch): {blocked_by_number:,}")
                print(f"  Blocked (short <=2 chars): {blocked_by_short:,}")
            if mapped_examples:
                print(f"  Examples (mapped to active vocab):")
                for orig, target, sim in mapped_examples[:10]:
                    print(f"    '{orig}' → '{target}'  (sim={sim:.4f})")

            print(f"  Unmapped (below threshold): {len(unmapped_j):,}")
            if unmapped_examples:
                print(f"  Examples (unmapped — nearest active but sim < {self.map_threshold}):")
                for orig, near, sim in unmapped_examples[:10]:
                    print(f"    '{orig}' ~ '{near}'  (sim={sim:.4f})")

            # Step 2b: cluster remaining low-freq among themselves
            self_clustered = 0
            cluster_examples = []
            if unmapped_j:
                um_terms = [low_list[j] for j in unmapped_j]
                um_freqs = [low[t] for t in um_terms]
                um_embs = low_embs[np.array(unmapped_j)]
                max_freq = max(um_freqs)

                if max_freq > 1 and len(um_terms) > 1:
                    # Build FAISS index of unmapped terms
                    um_index = faiss.IndexFlatIP(dim)
                    um_index.add(um_embs)
                    k_nn = min(20, len(um_terms))
                    um_sims, um_ids = um_index.search(um_embs, k_nn)

                    # Union-Find clustering (with number-safety guard)
                    uf = UnionFind(len(um_terms))
                    for a in range(len(um_terms)):
                        for nn in range(1, k_nn):
                            if um_sims[a][nn] >= self.cluster_threshold:
                                b = um_ids[a][nn]
                                # Block merging terms with different numbers
                                t_a, t_b = um_terms[a], um_terms[b]
                                if _has_digit(t_a) or _has_digit(t_b):
                                    continue
                                if len(t_a) <= 2 or len(t_b) <= 2:
                                    continue
                                uf.union(a, b)

                    for comp in uf.components():
                        if len(comp) <= 1:
                            continue
                        comp_items = [(um_terms[i], um_freqs[i]) for i in comp]
                        comp_items.sort(key=lambda x: (-x[1], x[0]))
                        best_freq = comp_items[0][1]
                        if best_freq == comp_items[-1][1]:
                            continue
                        canonical = comp_items[0][0]
                        merged_terms = []
                        for term, freq in comp_items[1:]:
                            if freq < best_freq:
                                mapping[term] = canonical
                                self_clustered += 1
                                merged_terms.append(term)
                        if merged_terms and len(cluster_examples) < 5:
                            cluster_examples.append(
                                (canonical, best_freq, merged_terms))

            if self_clustered:
                print(f"\n  Self-clustered:    {self_clustered:,}")
                for canon, freq, members in cluster_examples:
                    mbrs = ", ".join(f"'{m}'" for m in members[:5])
                    print(f"    '{canon}'(freq={freq}) ← [{mbrs}]")

            self.term_mapping[s] = mapping
            kept = len(low) - len(mapping)
            total_active += len(active)
            total_mapped += len(mapping)
            total_kept += kept

            print(f"\n  --- Slot {s+1} Summary ---")
            print(f"  Active vocabulary: {len(active):>10,}")
            print(f"  Mapped to active:  {mapped_to_active:>10,}")
            print(f"  Self-clustered:    {self_clustered:>10,}")
            print(f"  Kept original:     {kept:>10,}")
            print(f"  Final distinct:    {len(active) + kept:>10,}")

            # Step 3: audit
            slot_audit = []
            for orig, target in mapping.items():
                cnt = low[orig]
                if cnt >= self.audit_min_items:
                    slot_audit.append({
                        "from": orig, "to": target, "items": cnt,
                    })
            if slot_audit:
                slot_audit.sort(key=lambda x: -x["items"])
                audit_entries.append({
                    "slot": s, "name": SLOT_NAMES[s],
                    "merges": slot_audit,
                })
                print(f"  AUDIT: {len(slot_audit)} merge(s) affect "
                      f">= {self.audit_min_items:,} items")

            faiss.write_index(
                index,
                os.path.join(self.artifact_dir, f"faiss_slot_{s}.index"),
            )

            build_stats.append({
                "slot": s, "name": SLOT_NAMES[s],
                "total": len(counts), "active": len(active),
                "low_freq": len(low), "mapped_to_active": mapped_to_active,
                "unmapped_no_neighbor": len(unmapped_j),
                "self_clustered": self_clustered, "kept_original": kept,
            })

        # --- Summary ---
        effective = total_active + total_kept
        print(f"\n{'=' * 60}")
        print(f"BUILD SUMMARY")
        print(f"  Active vocab (freq >= {self.min_freq}): {total_active:,}")
        print(f"  Low-freq mapped:            {total_mapped:,}")
        print(f"  Low-freq kept original:     {total_kept:,}")
        print(f"  Effective vocab size:        {effective:,}")

        self._save(audit_entries, build_stats)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def _save(self, audit_entries, build_stats):
        d = self.artifact_dir
        _write_json(os.path.join(d, "config.json"), {
            "min_freq": self.min_freq,
            "map_threshold": self.map_threshold,
            "digit_threshold": self.digit_threshold,
            "cluster_threshold": self.cluster_threshold,
            "embed_model": self.embed_model_name,
            "audit_min_items": self.audit_min_items,
        })
        _write_json(os.path.join(d, "active_vocab.json"), {
            str(i): self.active_vocab[i] for i in range(N_SLOTS)
        })
        _write_json(os.path.join(d, "term_mapping.json"), {
            str(i): self.term_mapping[i] for i in range(N_SLOTS)
        })
        for i in range(N_SLOTS):
            _write_json(
                os.path.join(d, f"vocab_terms_slot_{i}.json"),
                self.active_term_lists[i],
            )
        _write_json(os.path.join(d, "audit_log.json"), audit_entries)
        _write_json(os.path.join(d, "build_stats.json"), build_stats)
        print(f"Artifacts saved to {d}/")

    def load(self):
        """Load pre-built artifacts (active vocab, mapping, FAISS)."""
        import faiss

        d = self.artifact_dir
        cfg = _read_json(os.path.join(d, "config.json"))
        self.min_freq = cfg["min_freq"]
        self.map_threshold = cfg["map_threshold"]
        self.digit_threshold = cfg.get("digit_threshold", 0.95)
        self.cluster_threshold = cfg.get("cluster_threshold",
                                         self.map_threshold)
        self.embed_model_name = cfg["embed_model"]

        av = _read_json(os.path.join(d, "active_vocab.json"))
        tm = _read_json(os.path.join(d, "term_mapping.json"))
        for i in range(N_SLOTS):
            self.active_vocab[i] = av.get(str(i), {})
            self.term_mapping[i] = tm.get(str(i), {})
            self.active_term_lists[i] = _read_json(
                os.path.join(d, f"vocab_terms_slot_{i}.json"))
            idx_path = os.path.join(d, f"faiss_slot_{i}.index")
            if os.path.exists(idx_path):
                self.faiss_indices[i] = faiss.read_index(idx_path)
        print(f"Loaded artifacts from {d}/")

    # ------------------------------------------------------------------
    # Compress (single term / 7-term TID / batch id2meta)
    # ------------------------------------------------------------------
    def compress_term(self, term, slot_idx):
        """Map a single term: active vocab → mapping → FAISS → original.

        For batch id2meta compression, use compress_id2meta() which avoids
        per-term FAISS calls.  This method is for new/unseen items.
        """
        lc = clean_term(term.lower())
        # 1. Active vocab (O(1) dict lookup)
        if lc in self.active_vocab[slot_idx]:
            return lc
        # 2. Pre-built mapping (O(1) dict lookup)
        if lc in self.term_mapping[slot_idx]:
            return self.term_mapping[slot_idx][lc]
        # 3. Short terms: don't FAISS-map
        if len(lc) <= 2:
            return lc
        # 4. FAISS nearest-neighbor (for truly new terms)
        index = self.faiss_indices[slot_idx]
        if index is not None and index.ntotal > 0:
            model = get_embed_model(self.embed_model_name)
            emb = model.encode(
                [lc], normalize_embeddings=True, convert_to_numpy=True,
            ).astype(np.float32)
            sims, ids = index.search(emb, 1)
            target = self.active_term_lists[slot_idx][ids[0][0]]
            if not should_block_mapping(lc, target, float(sims[0][0]),
                                        self.map_threshold,
                                        self.digit_threshold):
                return target
        # 5. No match — return cleaned lowercased original
        return lc

    def compress_terms(self, terms_7):
        """Compress a 7-term TID (e.g., model-generated output)."""
        return [self.compress_term(terms_7[i], i)
                for i in range(min(N_SLOTS, len(terms_7)))]

    def compress_id2meta(self, id2meta):
        """Batch-compress all items (dict-only, no FAISS — fast).

        Adds 'summary_words_norm_compress' to each item in-place.
        Terms already in the dataset are either in active_vocab or
        term_mapping; FAISS is not needed.
        """
        # Pre-build per-slot combined lookup for speed
        lookups = []
        for i in range(N_SLOTS):
            lu = {t: t for t in self.active_vocab[i]}  # active → self
            lu.update(self.term_mapping[i])              # low → mapped
            lookups.append(lu)

        for meta in tqdm(id2meta.values(), desc="Compressing items",
                         mininterval=10):
            sw = meta.get("summary_words_norm", [])
            if sw and len(sw) >= N_SLOTS:
                meta["summary_words_norm_compress"] = [
                    lookups[i].get(clean_term(sw[i].lower()),
                                   clean_term(sw[i].lower()))
                    for i in range(N_SLOTS)
                ]

    def compress_new_items_batch(self, terms_7_list):
        """Batch-compress a list of 7-term TIDs (with FAISS for unseen terms).

        Groups unseen terms per slot for efficient batch FAISS search.
        Returns: list of compressed 7-term lists.
        """
        results = []
        needs_faiss = defaultdict(list)  # slot -> [(list_idx, slot_idx, term)]

        # Pass 1: resolve via dict lookup
        for idx, terms_7 in enumerate(terms_7_list):
            compressed = [None] * N_SLOTS
            for s in range(min(N_SLOTS, len(terms_7))):
                lc = terms_7[s].lower()
                if lc in self.active_vocab[s]:
                    compressed[s] = lc
                elif lc in self.term_mapping[s]:
                    compressed[s] = self.term_mapping[s][lc]
                else:
                    compressed[s] = lc  # placeholder
                    needs_faiss[s].append((idx, s, lc))
            results.append(compressed)

        # Pass 2: batch FAISS per slot
        if needs_faiss:
            model = get_embed_model(self.embed_model_name)
            for s, items in needs_faiss.items():
                index = self.faiss_indices[s]
                if index is None or index.ntotal == 0:
                    continue
                unique_terms = list({t for _, _, t in items})
                embs = embed_terms(unique_terms, model, self.batch_size)
                sims, ids = index.search(embs, 1)
                term_map = {}
                for j, t in enumerate(unique_terms):
                    if sims[j][0] >= self.map_threshold:
                        term_map[t] = self.active_term_lists[s][ids[j][0]]
                for list_idx, slot_idx, lc in items:
                    if lc in term_map:
                        results[list_idx][slot_idx] = term_map[lc]

        return results


# ===========================================================================
# JSON I/O helpers
# ===========================================================================
def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False,
                  indent=2 if isinstance(obj, (dict, list)) and
                  not isinstance(obj, list) or
                  (isinstance(obj, list) and len(obj) < 100) else None)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Statistics
# ===========================================================================
def print_compression_stats(id2meta):
    """Print comprehensive before/after stats with GID-level impact and examples."""
    import random
    random.seed(42)

    slot_before = [Counter() for _ in range(N_SLOTS)]
    slot_after = [Counter() for _ in range(N_SLOTS)]
    gids_changed = []          # (gid, before_7, after_7, changed_slots)
    total_items = 0
    items_with_any_change = 0

    for gid, meta in id2meta.items():
        sw = meta.get("summary_words_norm", [])
        sc = meta.get("summary_words_norm_compress", [])
        if not (sw and len(sw) >= N_SLOTS and sc and len(sc) >= N_SLOTS):
            continue
        total_items += 1
        for i in range(N_SLOTS):
            slot_before[i][sw[i].lower()] += 1
            slot_after[i][sc[i]] += 1
        changed_slots = []
        for i in range(N_SLOTS):
            if sw[i].lower() != sc[i]:
                changed_slots.append(i)
        if changed_slots:
            items_with_any_change += 1
            if len(gids_changed) < 500:
                gids_changed.append(
                    (gid, [w.lower() for w in sw[:N_SLOTS]], sc[:N_SLOTS],
                     changed_slots))

    t_b = sum(len(c) for c in slot_before)
    t_a = sum(len(c) for c in slot_after)

    print(f"\n{'=' * 70}")
    print("COMPRESSION RESULTS")
    print(f"{'=' * 70}")

    # --- Per-slot table ---
    print(f"\n  {'Slot':<4} {'Role':<12} {'Before':>10} {'After':>10} "
          f"{'Reduced':>10} {'%':>7}  {'Hapax_B':>9} {'Hapax_A':>9}")
    print(f"  {'-' * 75}")
    for i in range(N_SLOTS):
        b, a = len(slot_before[i]), len(slot_after[i])
        pct = (b - a) / max(b, 1) * 100
        hb = sum(1 for c in slot_before[i].values() if c == 1)
        ha = sum(1 for c in slot_after[i].values() if c == 1)
        print(f"  {i + 1:<4} {SLOT_LABELS[i]:<12} {b:>10,} {a:>10,} "
              f"{b - a:>10,} {pct:>6.1f}%  {hb:>9,} {ha:>9,}")
    hap_b = sum(sum(1 for c in s.values() if c == 1) for s in slot_before)
    hap_a = sum(sum(1 for c in s.values() if c == 1) for s in slot_after)
    print(f"  {'-' * 75}")
    print(f"  {'':4} {'TOTAL':<12} {t_b:>10,} {t_a:>10,} "
          f"{t_b - t_a:>10,} {(t_b - t_a) / max(t_b, 1) * 100:>6.1f}%  "
          f"{hap_b:>9,} {hap_a:>9,}")

    # --- GID-level impact ---
    print(f"\n  --- GID-Level Impact ---")
    print(f"  Total items:               {total_items:>10,}")
    print(f"  Items with any term change: {items_with_any_change:>10,} "
          f"({items_with_any_change / max(total_items, 1) * 100:.2f}%)")

    # Per-slot item impact
    slot_item_changes = [0] * N_SLOTS
    for _, _, _, changed in gids_changed:
        for s in changed:
            slot_item_changes[s] += 1
    # Scale by sampling ratio
    scale = items_with_any_change / max(len(gids_changed), 1)
    print(f"\n  Items changed per slot (estimated from {len(gids_changed)} samples):")
    for i in range(N_SLOTS):
        est = int(slot_item_changes[i] * scale)
        print(f"    Slot {i+1} ({SLOT_LABELS[i]:<10}): ~{est:>10,}")

    # --- Example GIDs that changed ---
    if gids_changed:
        print(f"\n  --- Example Changed Items (10 random) ---")
        samples = random.sample(gids_changed, min(10, len(gids_changed)))
        for gid, before, after, changed in samples:
            print(f"\n  GID: {gid}")
            print(f"    Before: {before}")
            print(f"    After:  {after}")
            for s in changed:
                print(f"    slot {s+1}: '{before[s]}' → '{after[s]}'")

    # --- Per-slot top mapping targets (which active terms absorbed the most) ---
    print(f"\n  --- Top Mapping Targets Per Slot (active terms that absorbed most low-freq terms) ---")
    #  Infer from before/after count differences
    for i in range(N_SLOTS):
        gained = {}
        for t in slot_after[i]:
            diff = slot_after[i][t] - slot_before[i].get(t, 0)
            if diff > 0:
                gained[t] = diff
        if not gained:
            continue
        top5 = sorted(gained.items(), key=lambda x: -x[1])[:5]
        if top5:
            print(f"    Slot {i+1} ({SLOT_LABELS[i]}):")
            for t, diff in top5:
                print(f"      '{t}' absorbed +{diff:,} items "
                      f"(total: {slot_after[i][t]:,})")


def analyze_n_values(id2meta):
    """Quick analysis of vocab size vs. N threshold (no embedding needed)."""
    slot_counts = count_slot_terms(id2meta)

    print(f"\n{'=' * 80}")
    print("ACTIVE VOCABULARY SIZE BY MIN_FREQ (N)  [after lowercase, per slot]")
    print(f"{'=' * 80}")
    print(f"  {'N':>3} | ", end="")
    for lbl in SLOT_LABELS:
        print(f"{lbl:>12}", end="")
    print(f" | {'TOTAL':>10}  {'Coverage':>8}")
    print(f"  {'-' * 88}")

    for N in [1, 2, 3, 5, 10, 20, 50]:
        row = []
        cov = []
        for s in range(N_SLOTS):
            active = sum(1 for c in slot_counts[s].values() if c >= N)
            row.append(active)
            total_occ = sum(slot_counts[s].values())
            covered = sum(c for c in slot_counts[s].values() if c >= N)
            cov.append(covered / max(total_occ, 1))
        avg_cov = sum(cov) / N_SLOTS * 100
        print(f"  N={N:>2} | ", end="")
        for val in row:
            print(f"{val:>12,}", end="")
        print(f" | {sum(row):>10,}  {avg_cov:>7.1f}%")


# ===========================================================================
# Test: new-item mapping
# ===========================================================================
def test_new_items(compressor):
    """Simulate model-generated TIDs and demonstrate mapping."""
    test_cases = [
        # Typical model output (should mostly just lowercase)
        ["Sneakers", "Running", "Low-Top", "Mesh", "Nike",
         "Zappos", "Women"],
        ["dress", "Floral", "midi", "chiffon", "ZARA",
         "nordstrom rack", "women"],
        ["rug", "decorative", "rectangular", "wool", "Safavieh",
         "Wayfair", "modern"],
        # Typos / unseen variants (should trigger FAISS mapping)
        ["sneakr", "jogging", "low-tops", "breathable mesh",
         "Nikee", "zappo", "womens"],
        ["Bootz", "waterprooof", "lace-ups", "genuine-leather",
         "Timberlnd", "Amazonn", "outdoor"],
        # Completely novel terms
        ["gizmo", "quantum-powered", "holographic", "unobtanium",
         "XyloTech", "MegaMart", "aliens"],
    ]

    print(f"\n{'=' * 70}")
    print("TEST: New-Item TID Mapping")
    print(f"{'=' * 70}")

    for terms in test_cases:
        compressed = compressor.compress_terms(terms)
        print(f"\n  INPUT:  {terms}")
        print(f"  OUTPUT: {compressed}")
        changes = []
        for i in range(N_SLOTS):
            lc = terms[i].lower()
            if lc != compressed[i]:
                # Determine source: mapping or FAISS
                if lc in compressor.term_mapping[i]:
                    src = "mapping"
                elif lc in compressor.active_vocab[i]:
                    src = "active"
                else:
                    src = "faiss"
                changes.append((i, terms[i], compressed[i], src))
        if changes:
            for slot, orig, mapped, src in changes:
                print(f"    slot {slot + 1}: '{orig}' → '{mapped}'  [{src}]")
        else:
            print("    (all unchanged after lowercase)")


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Compress GRLM term vocabulary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260324/processed/"
                "id2meta_with_norm.json",
        help="Input id2meta JSON file",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: <input>_compress.json)",
    )
    p.add_argument(
        "--artifact_dir", type=str, default=None,
        help="Directory for artifacts "
             "(default: <output_dir>/vocab_compress_artifacts)",
    )
    p.add_argument("--min_freq", type=int, default=3,
                   help="Active vocab frequency threshold (default: 3)")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Cosine sim threshold for text-term mapping (default: 0.95)")
    p.add_argument("--digit_threshold", type=float, default=0.95,
                   help="Cosine sim threshold for digit-containing terms (default: 0.95)")
    p.add_argument("--cluster_threshold", type=float, default=0.95,
                   help="Threshold for self-clustering (default: 0.95)")
    p.add_argument("--embed_model", type=str,
                   default="/scratch/workspaceblobstore/users/xiaoyukou/"
                           "ckpts/Qwen3-Embedding-0.6B",
                   help="Sentence-transformer model path or name")
    p.add_argument("--batch_size", type=int, default=2048,
                   help="Embedding batch size (default: 2048)")
    p.add_argument("--audit_min_items", type=int, default=1000,
                   help="Audit threshold (default: 1000)")
    p.add_argument("--analyze", action="store_true",
                   help="Only analyze vocab sizes for different N values")
    p.add_argument("--load_artifacts", action="store_true",
                   help="Load pre-built artifacts (skip embedding)")
    p.add_argument("--test_only", action="store_true",
                   help="Run new-item test only (requires artifacts)")
    p.add_argument("--skip_output", action="store_true",
                   help="Skip saving the full compressed id2meta")
    return p.parse_args()


def main():
    args = parse_args()

    # Derive output paths
    if args.output is None:
        args.output = args.input.replace(".json", "_compress.json")
    if args.artifact_dir is None:
        args.artifact_dir = os.path.join(
            os.path.dirname(args.output), "vocab_compress_artifacts")

    # --analyze: quick stats only (no embedding)
    if args.analyze:
        id2meta = load_id2meta(args.input)
        analyze_n_values(id2meta)
        return

    # Build compressor
    comp = TermVocabCompressor(
        artifact_dir=args.artifact_dir,
        min_freq=args.min_freq,
        map_threshold=args.threshold,
        digit_threshold=args.digit_threshold,
        cluster_threshold=args.cluster_threshold,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
        audit_min_items=args.audit_min_items,
    )

    # --test_only: load artifacts and run test
    if args.test_only:
        comp.load()
        test_new_items(comp)
        return

    # Load data
    id2meta = load_id2meta(args.input)

    # Build or load artifacts
    if args.load_artifacts:
        comp.load()
    else:
        comp.build(id2meta)

    # Apply compression to all items
    comp.compress_id2meta(id2meta)
    print_compression_stats(id2meta)

    # Save compressed id2meta
    if not args.skip_output:
        print(f"\nSaving compressed data to {args.output} ...")
        t0 = time.time()
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(id2meta, f, ensure_ascii=False)
        mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"  Saved ({mb:.0f} MB) in {time.time() - t0:.1f}s")

    # Run new-item test
    test_new_items(comp)


if __name__ == "__main__":
    main()
