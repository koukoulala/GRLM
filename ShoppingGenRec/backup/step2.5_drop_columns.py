"""
Step 2.5: Drop ShoppingJourney and JourneyWithAllProducts columns,
keeping UserId, ReadableUserEvents, RequestTime, UserHistory,
and JourneyWithProducts.

Uses `cut` for fast column extraction (step2 output is fixed 7 columns).

Usage:
    python step2.5_drop_columns.py --input_file /path/to/with_products.tsv
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Step 2.5: Drop unnecessary columns from step2 output")
    parser.add_argument(
        "--input_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input_80K_7_results_with_products.tsv",
        help="Path to step2 output TSV (with_products.tsv)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Derive output path: replace _with_products with _JWP
    base = os.path.splitext(args.input_file)[0]
    output_file = base.replace("_with_products", "") + "_JWP.tsv"

    # step2 output columns (fixed):
    #   1:UserId  2:ReadableUserEvents  3:RequestTime  4:UserHistory
    #   5:ShoppingJourney  6:JourneyWithAllProducts  7:JourneyWithProducts
    # Keep columns 1,2,3,4,7 (drop 5,6)
    print(f"  Input:  {args.input_file}")
    print(f"  Output: {output_file}")
    print(f"  Keeping: columns 1,2,3,4,7 (dropping 5:ShoppingJourney, 6:JourneyWithAllProducts)")

    result = subprocess.run(
        ["cut", "-f1,2,3,4,7", args.input_file],
        stdout=open(output_file, "w"),
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"Error: cut failed: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    # Count lines for summary
    wc = subprocess.run(["wc", "-l", output_file], capture_output=True, text=True)
    line_count = wc.stdout.strip().split()[0] if wc.returncode == 0 else "?"
    print(f"Done. {line_count} lines (including header) written to {output_file}")


if __name__ == "__main__":
    main()
