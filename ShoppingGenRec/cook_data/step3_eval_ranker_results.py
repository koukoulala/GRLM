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
            kept_journeys.append(journey)

        output_data[jtype] = kept_journeys

    return output_data, stats


def process_file(input_file):
    """Process a single Ranker TSV file. Returns True on success."""
    # Derive output file name from input
    base = os.path.splitext(input_file)[0]
    output_file = f"{base}_cleaned.tsv"
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

    total_rows = 0
    rows_kept = 0
    rows_deleted_parse_error = 0
    total_products_removed = 0
    total_products_kept = 0
    total_journeys_removed = 0
    total_queries_added = 0
    rows_with_product_removal = 0

    # Pre-scan: check if UserHistory column exists in input
    user_history_map = None
    with open(input_file, "r", encoding="utf-8") as f_check:
        header = f_check.readline().strip().split("\t")
    if "UserHistory" not in header:
        # Derive the JWP file path: *_JWP_Ranker.tsv -> *_JWP.tsv
        jwp_file = input_file.replace("_Ranker.tsv", ".tsv")
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

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=KEEP_COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()

        for row_idx, row in enumerate(reader):
            total_rows += 1

            # 1. Skip rows where OUTPUT JSON can't be parsed
            try:
                output_data = json.loads(row["OUTPUT"])
            except (json.JSONDecodeError, KeyError) as e:
                rows_deleted_parse_error += 1
                print(f"[Row {row_idx}] DELETED — OUTPUT parse error: {e}")
                continue

            try:
                jwp_data = json.loads(row["JourneyWithProducts"])
            except (json.JSONDecodeError, KeyError) as e:
                rows_deleted_parse_error += 1
                print(f"[Row {row_idx}] DELETED — JourneyWithProducts parse error: {e}")
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

            # Build FinalJourney: cleaned output with Queries
            final_journey = json.dumps(cleaned_output, ensure_ascii=False)

            # Write cleaned row with selected columns only
            out_row = {col: row.get(col, "") for col in KEEP_COLUMNS}
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
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/",
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
            cleaned_path = os.path.splitext(f)[0] + "_cleaned.tsv"
            if os.path.isfile(cleaned_path):
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
