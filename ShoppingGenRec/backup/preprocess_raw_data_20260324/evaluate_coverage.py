"""
Evaluate GlobalOfferId coverage between SequenceData and ProductsData.

Reads both TSV files and computes:
  - Total rows and distinct GlobalOfferIds in each file
  - How many GlobalOfferIds from SequenceData exist in ProductsData (coverage)
  - How many are missing
"""

import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)


def count_global_offer_ids(filepath, gid_column="GlobalOfferId", expected_columns=None):
    """Read a TSV file and collect all distinct GlobalOfferIds.

    Args:
        filepath: Path to the TSV file.
        gid_column: Name of the GlobalOfferId column.
        expected_columns: Optional list of column names (skips header row).
            If None, auto-detects from header.

    Returns:
        (total_rows, set_of_global_offer_ids)
    """
    gids = set()
    total_rows = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        if expected_columns:
            next(reader, None)  # skip header
            columns = expected_columns
        else:
            header = next(reader, None)
            if header is None:
                return 0, set()
            columns = header

        if gid_column not in columns:
            print(f"  WARNING: Column '{gid_column}' not found in {filepath}")
            print(f"  Available columns: {columns}")
            return 0, set()

        gid_idx = columns.index(gid_column)

        for row in reader:
            total_rows += 1
            if len(row) > gid_idx:
                gid = row[gid_idx].strip()
                if gid:
                    gids.add(gid)

            if total_rows % 5_000_000 == 0:
                print(f"    ... read {total_rows:,} rows so far, "
                      f"{len(gids):,} distinct GlobalOfferIds")

    return total_rows, gids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GlobalOfferId coverage between SequenceData and ProductsData"
    )
    parser.add_argument(
        "--sequence_data_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/0128_0301/SequenceData_Plat.tsv",
        help="Path to the SequenceData_Plat TSV file",
    )
    parser.add_argument(
        "--products_data_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/0128_0301/ProductsData.tsv",
        help="Path to the ProductsData TSV file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # Step 1: Read SequenceData
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 1: Reading SequenceData")
    print("=" * 70)

    seq_columns = ["UserId", "PageTitle", "GlobalOfferId", "Timestamp", "Source", "Query"]
    seq_rows, seq_gids = count_global_offer_ids(
        args.sequence_data_file,
        gid_column="GlobalOfferId",
        expected_columns=seq_columns,
    )
    print(f"  File: {args.sequence_data_file}")
    print(f"  Total rows:                {seq_rows:>12,}")
    print(f"  Distinct GlobalOfferIds:   {len(seq_gids):>12,}")

    # -------------------------------------------------------------------------
    # Step 2: Read ProductsData
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Step 2: Reading ProductsData")
    print("=" * 70)

    prod_rows, prod_gids = count_global_offer_ids(
        args.products_data_file,
        gid_column="GlobalOfferId",
        expected_columns=None,  # auto-detect from header
    )
    print(f"  File: {args.products_data_file}")
    print(f"  Total rows:                {prod_rows:>12,}")
    print(f"  Distinct GlobalOfferIds:   {len(prod_gids):>12,}")

    # -------------------------------------------------------------------------
    # Step 3: Compute coverage
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Step 3: Coverage analysis")
    print("=" * 70)

    # Sequence GIDs covered by ProductsData
    seq_covered = seq_gids & prod_gids
    seq_missing = seq_gids - prod_gids
    seq_coverage = len(seq_covered) / len(seq_gids) * 100 if seq_gids else 0

    # ProductsData GIDs covered by SequenceData
    prod_covered = prod_gids & seq_gids
    prod_missing = prod_gids - seq_gids
    prod_coverage = len(prod_covered) / len(prod_gids) * 100 if prod_gids else 0

    print(f"  SequenceData GlobalOfferIds:       {len(seq_gids):>12,}")
    print(f"    Covered by ProductsData:         {len(seq_covered):>12,}  ({seq_coverage:.2f}%)")
    print(f"    Missing from ProductsData:       {len(seq_missing):>12,}  ({100 - seq_coverage:.2f}%)")
    print()
    print(f"  ProductsData GlobalOfferIds:       {len(prod_gids):>12,}")
    print(f"    Covered by SequenceData:         {len(prod_covered):>12,}  ({prod_coverage:.2f}%)")
    print(f"    Not in SequenceData:             {len(prod_missing):>12,}  ({100 - prod_coverage:.2f}%)")
    print()
    print(f"  Union of both:                     {len(seq_gids | prod_gids):>12,}")
    print(f"  Intersection:                      {len(seq_covered):>12,}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
