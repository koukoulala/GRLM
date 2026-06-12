#!/usr/bin/env python3
"""Fast conservative iPhone filtering on title-only scan.

Writes full iPhone title export and conservative removal candidates under
raw_data_IDB for manual inspection.
"""

import csv
import json
import os
import re
from collections import Counter

INPUT_JSON = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/item.json"
OUT_DIR = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/special_case_iphone"

TITLE_RE = re.compile(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"')
IPHONE_TOKEN_RE = re.compile(r"(?i)\b(?:iphone|ihpone)\b")
IPHONE_PHRASE_RE = re.compile(r"(?i)\b(?:iphone|ihpone)\b[^\n\r]{0,60}")
MODEL_TOKEN_RE = re.compile(r"(?i)\b(?:\d{1,2}e?|x(?:r|s)?|se|air)\b")
NEW_GLOBAL_RE = re.compile(
    r"(?ix)"
    r"\b(?:16e?|17e?|18e?)\b|"
    r"\b17\s*air\b|"
    r"\biphone\s*air\b"
)


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

    for phrase in IPHONE_PHRASE_RE.findall(tl):
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

    # Conservative fallback: if any strong new-model marker appears anywhere
    # in an iPhone title, treat as has_new to avoid false deletions.
    if not has_new and NEW_GLOBAL_RE.search(tl):
        has_new = True

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
    os.makedirs(OUT_DIR, exist_ok=True)
    out_all = os.path.join(OUT_DIR, "iphone_titles_all_conservative.tsv")
    out_remove = os.path.join(OUT_DIR, "iphone_titles_remove_candidates_conservative.tsv")
    out_keep = os.path.join(OUT_DIR, "iphone_titles_keep_or_review_conservative.tsv")
    out_summary = os.path.join(OUT_DIR, "iphone_titles_conservative_summary.json")

    counts = Counter()
    total_titles = 0
    iphone_titles = 0

    chunk_size = 4 * 1024 * 1024
    tail = ""

    with open(INPUT_JSON, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_all, "w", encoding="utf-8", newline="") as fall, \
         open(out_remove, "w", encoding="utf-8", newline="") as frem, \
         open(out_keep, "w", encoding="utf-8", newline="") as fkeep:

        w_all = csv.writer(fall, delimiter="\t")
        w_rem = csv.writer(frem, delimiter="\t")
        w_keep = csv.writer(fkeep, delimiter="\t")

        header = ["status", "has_old", "has_new", "title"]
        for w in (w_all, w_rem, w_keep):
            w.writerow(header)

        while True:
            s = fin.read(chunk_size)
            if not s:
                break
            data = tail + s
            end = max(0, len(data) - 512)

            for m in TITLE_RE.finditer(data[:end]):
                total_titles += 1
                title = decode_json_string(m.group(1))
                cls = classify_title(title)
                if cls is None:
                    continue

                iphone_titles += 1
                status, has_old, has_new = cls
                counts[status] += 1
                row = [status, int(has_old), int(has_new), title]
                w_all.writerow(row)
                if status == "remove_old_only":
                    w_rem.writerow(row)
                else:
                    w_keep.writerow(row)

            tail = data[end:]

        for m in TITLE_RE.finditer(tail):
            total_titles += 1
            title = decode_json_string(m.group(1))
            cls = classify_title(title)
            if cls is None:
                continue

            iphone_titles += 1
            status, has_old, has_new = cls
            counts[status] += 1
            row = [status, int(has_old), int(has_new), title]
            w_all.writerow(row)
            if status == "remove_old_only":
                w_rem.writerow(row)
            else:
                w_keep.writerow(row)

    summary = {
        "total_titles": total_titles,
        "iphone_titles": iphone_titles,
        "counts": dict(counts),
        "outputs": {
            "all": out_all,
            "remove": out_remove,
            "keep": out_keep,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(summary)


if __name__ == "__main__":
    main()
