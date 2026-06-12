#!/usr/bin/env python3
"""Export and conservatively filter iPhone-related titles from item.json.

Policy (conservative):
- Remove only titles that mention old iPhone families and do NOT mention new families.
- Keep titles that mention new families.
- Keep titles that mention both old + new (mixed compatibility) for manual review.

Outputs are written under raw_data_IDB by default.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter

# item.json entry pattern, tolerant to spaces/newlines:
# "<gid>" : { "title" : "..."
ENTRY_RE = re.compile(
    r'"(?P<gid>[^"\\]+)"\s*:\s*\{\s*"title"\s*:\s*"(?P<title>(?:\\.|[^"\\])*)"'
)
IPHONE_TOKEN_RE = re.compile(r"(?i)\b(?:iphone|ihpone)\b")

IPHONE_PHRASE_RE = re.compile(r"(?i)\b(?:iphone|ihpone)\b[^\n\r]{0,60}")
MODEL_TOKEN_RE = re.compile(r"(?i)\b(?:\d{1,2}e?|x(?:r|s)?|se|air)\b")


def decode_json_string(raw: str) -> str:
    try:
        return json.loads('"' + raw + '"')
    except Exception:
        return raw


def classify_title(title: str):
    tl = title.lower()
    if not IPHONE_TOKEN_RE.search(tl):
        return None

    has_old = False
    has_new = False

    # Analyze local iPhone phrases, so mixed forms like "iPhone 12-16"
    # can be recognized as old+new and kept in conservative mode.
    for phrase in IPHONE_PHRASE_RE.findall(tl):
        # Remove decimal screen sizes (e.g. 6.7, 6.1) to avoid mapping
        # them to old generations 6/7.
        p = re.sub(r"\b\d+\.\d+\b", " ", phrase)
        tokens = {m.group(0).lower() for m in MODEL_TOKEN_RE.finditer(p)}

        if any(tok in {"16", "16e", "17", "17e", "18", "18e", "air"}
               for tok in tokens):
            has_new = True

        if any(tok in {"1", "2", "3", "4", "5", "6", "7", "8", "9",
                       "10", "11", "12", "13", "14", "15",
                       "x", "xr", "xs", "se"}
               for tok in tokens):
            has_old = True

    if has_old and not has_new:
        status = "remove_old_only"
    elif has_new and not has_old:
        status = "keep_new_only"
    elif has_old and has_new:
        status = "keep_mixed_old_new"
    else:
        status = "keep_unknown_review"

    return status, has_old, has_new


def main():
    ap = argparse.ArgumentParser(description="Conservative iPhone filtering scan")
    ap.add_argument(
        "--input_json",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/item.json",
        help="Path to item.json",
    )
    ap.add_argument(
        "--output_dir",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/special_case_iphone",
        help="Output directory",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    out_all = os.path.join(args.output_dir, "iphone_items_all_conservative.tsv")
    out_remove = os.path.join(args.output_dir, "iphone_items_remove_candidates_conservative.tsv")
    out_keep = os.path.join(args.output_dir, "iphone_items_keep_or_review_conservative.tsv")
    out_mixed = os.path.join(args.output_dir, "iphone_items_mixed_old_new_review.tsv")
    out_summary = os.path.join(args.output_dir, "iphone_items_conservative_summary.json")

    counts = Counter()
    examples = {
        "remove_old_only": [],
        "keep_new_only": [],
        "keep_mixed_old_new": [],
        "keep_unknown_review": [],
    }

    total_entries = 0
    iphone_entries = 0

    chunk_size = 4 * 1024 * 1024
    tail = ""

    with open(args.input_json, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_all, "w", encoding="utf-8", newline="") as fall, \
         open(out_remove, "w", encoding="utf-8", newline="") as frem, \
         open(out_keep, "w", encoding="utf-8", newline="") as fkeep, \
         open(out_mixed, "w", encoding="utf-8", newline="") as fmix:

        w_all = csv.writer(fall, delimiter="\t")
        w_rem = csv.writer(frem, delimiter="\t")
        w_keep = csv.writer(fkeep, delimiter="\t")
        w_mix = csv.writer(fmix, delimiter="\t")

        header = ["GlobalOfferId", "status", "has_old", "has_new", "title"]
        for w in (w_all, w_rem, w_keep, w_mix):
            w.writerow(header)

        while True:
            part = fin.read(chunk_size)
            if not part:
                break
            data = tail + part
            # Keep a tail to avoid cutting regex match at chunk boundary.
            end = max(0, len(data) - 4096)

            for m in ENTRY_RE.finditer(data[:end]):
                total_entries += 1
                gid = m.group("gid")
                title = decode_json_string(m.group("title"))

                cls = classify_title(title)
                if cls is None:
                    continue

                iphone_entries += 1
                status, has_old, has_new = cls
                counts[status] += 1

                row = [gid, status, int(has_old), int(has_new), title]
                w_all.writerow(row)

                if status == "remove_old_only":
                    w_rem.writerow(row)
                else:
                    w_keep.writerow(row)
                    if status == "keep_mixed_old_new":
                        w_mix.writerow(row)

                if len(examples[status]) < 12:
                    examples[status].append({"GlobalOfferId": gid, "title": title})

            tail = data[end:]

        # Flush final tail
        for m in ENTRY_RE.finditer(tail):
            total_entries += 1
            gid = m.group("gid")
            title = decode_json_string(m.group("title"))

            cls = classify_title(title)
            if cls is None:
                continue

            iphone_entries += 1
            status, has_old, has_new = cls
            counts[status] += 1

            row = [gid, status, int(has_old), int(has_new), title]
            w_all.writerow(row)

            if status == "remove_old_only":
                w_rem.writerow(row)
            else:
                w_keep.writerow(row)
                if status == "keep_mixed_old_new":
                    w_mix.writerow(row)

            if len(examples[status]) < 12:
                examples[status].append({"GlobalOfferId": gid, "title": title})

    summary = {
        "input_json": args.input_json,
        "total_entries_scanned": total_entries,
        "iphone_entries": iphone_entries,
        "counts": dict(counts),
        "outputs": {
            "all": out_all,
            "remove_candidates": out_remove,
            "keep_or_review": out_keep,
            "mixed_old_new_review": out_mixed,
        },
        "examples": examples,
        "policy": {
            "remove": "mentions old families only",
            "keep": "new only or mixed old+new or unknown",
        },
    }

    with open(out_summary, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, ensure_ascii=False, indent=2)

    print("[DONE] iPhone conservative scan finished")
    print(f"  total_entries_scanned: {total_entries:,}")
    print(f"  iphone_entries:        {iphone_entries:,}")
    for k in ("remove_old_only", "keep_new_only", "keep_mixed_old_new", "keep_unknown_review"):
        print(f"  {k:22s}: {counts.get(k, 0):,}")
    print(f"  all:                  {out_all}")
    print(f"  remove_candidates:    {out_remove}")
    print(f"  keep_or_review:       {out_keep}")
    print(f"  mixed_old_new_review: {out_mixed}")
    print(f"  summary_json:         {out_summary}")


if __name__ == "__main__":
    main()
