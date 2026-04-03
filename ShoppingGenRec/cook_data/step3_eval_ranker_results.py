"""
step3_eval_ranker_results.py

Clean the Ranker output TSV:
1. Delete rows where OUTPUT JSON cannot be parsed
2. For each journey in OUTPUT, remove products whose OfferId is NOT found
   in the corresponding JourneyWithProducts journey
3. Add the original Queries from JourneyWithProducts into each OUTPUT journey
4. Report cleaning statistics

Usage:
    python step3_eval_ranker_results.py --input_file /path/to/Ranker.tsv
    python step3_eval_ranker_results.py --input_folder /path/to/folder/
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

JOURNEY_TYPES = ["ContinuedJourneys"]
PRODUCT_ATTRS = ["Title", "Seller", "Price"]

# Columns to keep in the output file (matches pre_s2 input format)
KEEP_COLUMNS = [
    "UserId",
    "ReadableUserEvents",
    "UserHistory",
    "JourneyWithProducts",
    "FinalJourney",
]


def _unescape_json_field_v1(s):
    """Unescape variant 1: for files where csv only escaped quotes.

    Pattern: \\" -> placeholder, \\" -> ", placeholder -> \\"
    Works when the writer only escaped " but not \\.
    """
    if '\\' not in s:
        return s
    placeholder = '\x00__ESC_QUOT__\x00'
    s = s.replace('\\\\"', placeholder)
    s = s.replace('\\"', '"')
    s = s.replace(placeholder, '\\"')
    return s


def _unescape_json_field_v2(s):
    """Unescape variant 2: for files where csv escaped both \\ and ".

    csv escapechar='\\' with QUOTE_NONE escapes both \\ -> \\\\ and " -> \\".
    Reverse: first \\\\ -> \\, then \\" -> ".
    """
    if '\\' not in s:
        return s
    placeholder = '\x00__BSLASH__\x00'
    s = s.replace('\\\\', placeholder)
    s = s.replace('\\"', '"')
    s = s.replace(placeholder, '\\')
    return s


def _parse_json_field(raw):
    """Try to parse a JSON field, handling old double-escaped format."""
    if not raw or not raw.strip():
        return None, "empty"
    try:
        return json.loads(raw), None
    except (json.JSONDecodeError, ValueError):
        pass
    # Try v1 unescape (only quotes escaped)
    try:
        return json.loads(_unescape_json_field_v1(raw)), None
    except (json.JSONDecodeError, ValueError):
        pass
    # Try v2 unescape (both backslashes and quotes escaped)
    try:
        return json.loads(_unescape_json_field_v2(raw)), None
    except (json.JSONDecodeError, ValueError) as e:
        return None, str(e)


def build_jwp_index(jwp_data):
    """
    Build an index from JourneyWithProducts:
      journey_title -> {
          "products": { offerId -> { Title, Seller, Price } },
          "queries": [ { Query, Products } ]
      }
    """
    index = {}
    for jtype in JOURNEY_TYPES:
        for journey in jwp_data.get(jtype, []):
            title = journey.get("Title", "").strip()
            products = {}
            for query in journey.get("Queries", []):
                for prod in query.get("Products", []):
                    oid = str(prod.get("OfferId", "")).strip()
                    products[oid] = {
                        "Title": prod.get("Title", "").strip(),
                        "Seller": prod.get("Seller", "").strip(),
                        "Price": prod.get("Price", "").strip(),
                    }
            index[title] = {
                "products": products,
                "queries": journey.get("Queries", []),
                "journey_type": journey.get("JourneyType", ""),
            }
    return index


def clean_output(output_data, jwp_index):
    """
    Clean one row's OUTPUT:
    - Remove products whose OfferId is not in JWP
    - Add Queries from JWP into each journey
    Returns (cleaned_output, stats_dict).
    """
    stats = {
        "products_removed": 0,
        "products_kept": 0,
        "journeys_removed": 0,
        "queries_added": 0,
        "journey_type_restored": 0,
    }

    for jtype in JOURNEY_TYPES:
        journeys = output_data.get(jtype, [])
        kept_journeys = []
        for journey in journeys:
            jtitle = journey.get("Title", "").strip()

            jwp_entry = jwp_index.get(jtitle)
            if jwp_entry is None:
                # Journey title not found in JWP — keep as-is
                stats["products_kept"] += len(journey.get("Products", []))
                kept_journeys.append(journey)
                continue

            jwp_products = jwp_entry["products"]
            jwp_queries = jwp_entry["queries"]

            # Filter products: keep only those whose OfferId exists in JWP
            original_products = journey.get("Products", [])
            cleaned_products = []
            for prod in original_products:
                oid = str(prod.get("OfferId", "")).strip()
                if oid in jwp_products:
                    cleaned_products.append(prod)
                else:
                    stats["products_removed"] += 1

            # If no products left after cleaning, remove the entire journey
            if not cleaned_products:
                stats["journeys_removed"] += 1
                continue

            stats["products_kept"] += len(cleaned_products)
            journey["Products"] = cleaned_products

            # Re-rank after removal
            for rank, prod in enumerate(journey["Products"], 1):
                prod["Rank"] = rank

            # Add Queries from JWP into this journey
            journey["Queries"] = jwp_queries
            stats["queries_added"] += 1

            # Restore JourneyType from JWP if missing or incorrect in ranker output
            jwp_journey_type = jwp_entry.get("journey_type", "")
            if jwp_journey_type:
                output_jtype = journey.get("JourneyType", "")
                if not output_jtype or output_jtype != jwp_journey_type:
                    journey["JourneyType"] = jwp_journey_type
                    stats["journey_type_restored"] += 1

            kept_journeys.append(journey)

        output_data[jtype] = kept_journeys

    return output_data, stats


def process_file(input_file):
    """Process a single Ranker TSV file. Returns True on success."""
    # Derive output file name: *_Ranker.tsv -> *_clean.tsv (short name for cosmos FUSE)
    base = input_file
    if base.endswith("_Ranker.tsv"):
        base = base[:-len("_Ranker.tsv")]
    else:
        base = os.path.splitext(base)[0]
    output_file = f"{base}_clean.tsv"
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

    total_rows = 0
    rows_kept = 0
    rows_deleted_parse_error = 0
    total_products_removed = 0
    total_products_kept = 0
    total_journeys_removed = 0
    total_queries_added = 0
    total_journey_type_restored = 0
    rows_with_product_removal = 0

    # Pre-scan: check if UserHistory column exists in input
    user_history_map = None
    with open(input_file, "r", encoding="utf-8") as f_check:
        header = f_check.readline().strip().split("\t")
    if "UserHistory" not in header:
        # Try to find the original JWP file to get UserHistory.
        # Ranker files may have a split suffix, e.g.:
        #   *_80K_1_2_results_JWP_Ranker.tsv  ->  *_80K_1_results_JWP.tsv
        # Strategy: first try direct replace, then strip the split number.
        jwp_file = input_file.replace("_Ranker.tsv", ".tsv")
        if not os.path.isfile(jwp_file):
            # Strip split suffix: *_80K_X_N_results_JWP.tsv -> *_80K_X_results_JWP.tsv
            jwp_file = re.sub(r'_(\d+)_results_JWP\.tsv$', r'_results_JWP.tsv', jwp_file)
        if not os.path.isfile(jwp_file):
            print(f"ERROR: UserHistory column missing and JWP file not found: {jwp_file}")
            return False
        print(f"  UserHistory column missing — loading from: {jwp_file}")
        user_history_map = {}
        with open(jwp_file, "r", encoding="utf-8") as f_jwp:
            jwp_reader = csv.DictReader(f_jwp, delimiter="\t")
            for jwp_row in jwp_reader:
                uid = jwp_row.get("UserId", "").strip()
                if uid:
                    user_history_map[uid] = jwp_row.get("UserHistory", "")
        print(f"  Loaded UserHistory for {len(user_history_map)} users from JWP")

    # Detect old format: old step2.8 wrote with QUOTE_NONE + escapechar='\\',
    # so the first data field containing JSON will have \\" patterns.
    # New step2.8 uses QUOTE_MINIMAL which wraps fields in quotes instead.
    is_old_format = False
    with open(input_file, "r", encoding="utf-8") as f_detect:
        f_detect.readline()  # skip header
        first_line = f_detect.readline()
        if first_line and '\\"' in first_line:
            is_old_format = True
    if is_old_format:
        print(f"  Detected old TSV format (backslash-escaped); will unescape JSON fields")

    # Build output columns dynamically: include UserProfile if present in input
    has_profile = "UserProfile" in header
    keep_columns = list(KEEP_COLUMNS)
    if has_profile:
        idx = keep_columns.index("UserHistory") + 1
        keep_columns.insert(idx, "UserProfile")

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=keep_columns, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()

        for row_idx, row in enumerate(reader):
            total_rows += 1

            # 1. Skip rows where OUTPUT or JWP JSON can't be parsed
            output_data, out_err = _parse_json_field(row.get("OUTPUT", ""))
            if output_data is None:
                rows_deleted_parse_error += 1
                if rows_deleted_parse_error <= 5:
                    print(f"[Row {row_idx}] DELETED — OUTPUT parse error: {out_err}")
                continue

            jwp_data, jwp_err = _parse_json_field(row.get("JourneyWithProducts", ""))
            if jwp_data is None:
                rows_deleted_parse_error += 1
                if rows_deleted_parse_error <= 5:
                    print(f"[Row {row_idx}] DELETED — JourneyWithProducts parse error: {jwp_err}")
                continue

            # 2 & 3. Clean OUTPUT: remove bad products, add Queries
            jwp_index = build_jwp_index(jwp_data)
            cleaned_output, stats = clean_output(output_data, jwp_index)

            total_products_removed += stats["products_removed"]
            total_products_kept += stats["products_kept"]
            total_journeys_removed += stats["journeys_removed"]
            total_queries_added += stats["queries_added"]
            if stats["products_removed"] > 0:
                rows_with_product_removal += 1
            total_journey_type_restored += stats["journey_type_restored"]

            # Build FinalJourney: cleaned output with Queries
            final_journey = json.dumps(cleaned_output, ensure_ascii=False)

            # Write cleaned row with selected columns only
            out_row = {col: row.get(col, "") for col in keep_columns}
            out_row["FinalJourney"] = final_journey
            # Fill UserHistory from JWP if missing in input
            if user_history_map is not None:
                uid = row.get("UserId", "").strip()
                out_row["UserHistory"] = user_history_map.get(uid, "")
            writer.writerow(out_row)
            rows_kept += 1

    # ===== Print Summary =====
    print()
    print("=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print()
    print(f"Total rows read:                 {total_rows}")
    print(f"Rows deleted (parse error):      {rows_deleted_parse_error}")
    print(f"Rows kept (written):             {rows_kept}")
    print()
    print(f"--- Product Cleaning ---")
    print(f"  Products removed (OfferId not in JWP): {total_products_removed}")
    print(f"  Products kept:                         {total_products_kept}")
    print(f"  Journeys removed (no products left):   {total_journeys_removed}")
    print(f"  Rows affected by product removal:      {rows_with_product_removal}")
    print()
    print(f"--- Query Enrichment ---")
    print(f"  Journeys with Queries added from JWP:  {total_queries_added}")
    print(f"  JourneyType restored from JWP:         {total_journey_type_restored}")
    print()
    print(f"Output written to: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Clean ranker results — filter products and add queries")
    parser.add_argument(
        "--input_file", type=str, default="",
        help="Path to a single Ranker output TSV from step2.8")
    parser.add_argument(
        "--input_folder", type=str, 
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/CookData_merged/",
        help="Path to a folder; processes all *_Ranker.tsv files inside it")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    # Collect files to process
    if args.input_folder and os.path.isdir(args.input_folder):
        # Folder mode: find all *_Ranker.tsv, skip already-cleaned ones
        pattern = os.path.join(args.input_folder, "*_Ranker.tsv")
        all_ranker_files = sorted(glob.glob(pattern))
        # Exclude files that are themselves cleaned outputs (*_Ranker_cleaned.tsv)
        all_ranker_files = [f for f in all_ranker_files
                           if not f.endswith("_Ranker_cleaned.tsv")]

        files_to_process = []
        files_skipped = []
        for f in all_ranker_files:
            # Match the output naming: *_Ranker.tsv -> *_clean.tsv
            if f.endswith("_Ranker.tsv"):
                clean_path = f[:-len("_Ranker.tsv")] + "_clean.tsv"
            else:
                clean_path = os.path.splitext(f)[0] + "_clean.tsv"
            # Also check old naming variants
            old_cleaned_path = os.path.splitext(f)[0] + "_cleaned.tsv"
            old_cleaned_path2 = f.replace("_Ranker.tsv", "_cleaned.tsv") if f.endswith("_Ranker.tsv") else ""
            if (os.path.isfile(clean_path) or os.path.isfile(old_cleaned_path)
                    or (old_cleaned_path2 and os.path.isfile(old_cleaned_path2))):
                files_skipped.append(f)
            else:
                files_to_process.append(f)

        print(f"Folder: {args.input_folder}")
        print(f"Found {len(all_ranker_files)} *_Ranker.tsv file(s)")
        print(f"  To process: {len(files_to_process)}")
        print(f"  Skipped (already cleaned): {len(files_skipped)}")
        if files_skipped:
            for f in files_skipped:
                print(f"    SKIP: {os.path.basename(f)}")
        print()

        files_succeeded = []
        files_failed = []
        for i, f in enumerate(files_to_process, 1):
            print(f"{'#' * 70}")
            print(f"Processing file {i}/{len(files_to_process)}: {os.path.basename(f)}")
            print(f"{'#' * 70}")
            ok = process_file(f)
            if ok:
                files_succeeded.append(f)
            else:
                files_failed.append(f)
            print()

        # Final folder summary
        print()
        print("=" * 70)
        print("FOLDER PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Total *_Ranker.tsv found: {len(all_ranker_files)}")
        print(f"  Skipped (already cleaned): {len(files_skipped)}")
        print(f"  Processed successfully:    {len(files_succeeded)}")
        print(f"  Failed:                    {len(files_failed)}")
        if files_succeeded:
            print()
            print("Processed:")
            for f in files_succeeded:
                print(f"  OK:   {os.path.basename(f)}")
        if files_failed:
            print()
            print("Failed:")
            for f in files_failed:
                print(f"  FAIL: {os.path.basename(f)}")

    elif args.input_file:
        # Single-file mode
        if not os.path.isfile(args.input_file):
            print(f"ERROR: input_file not found: {args.input_file}")
            sys.exit(1)
        process_file(args.input_file)

    else:
        print("ERROR: Please specify --input_file or --input_folder")
        sys.exit(1)


if __name__ == "__main__":
    main()
