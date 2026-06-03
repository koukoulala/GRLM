"""
Split paired_data.json into individual per-user JSON files for Phase 2 agents.

Only Deep users (both sides have journeys) are split out.

Usage:
  python3 split_users.py --run-name <name> [--output-dir <path>]

Output:
  analysis/<run_name>/user_data/user_<ID8>.json   (local)
  <output_dir>/<run_name>/user_data/               (remote copy, if --output-dir given)
"""
import json, os, sys, argparse, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Split paired_data.json → per-user JSON")
    ap.add_argument("--run-name", required=True, help="Run directory name under analysis/")
    ap.add_argument("--output-dir", default=None,
                    help="Additional output directory; results are copied there too")
    return ap.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(script_dir, args.run_name)
    paired_file = os.path.join(run_dir, "paired_data.json")

    if not os.path.exists(paired_file):
        print(f"ERROR: {paired_file} not found")
        sys.exit(1)

    with open(paired_file, "r", encoding="utf-8") as f:
        paired = json.load(f)

    user_dir = os.path.join(run_dir, "user_data")
    os.makedirs(user_dir, exist_ok=True)

    deep_count = 0
    for entry in paired:
        if entry["triage"] != "deep":
            continue

        uid = entry["stableid"]
        short_id = uid[:8].upper()
        out_file = os.path.join(user_dir, f"user_{short_id}.json")

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        p1j = entry["p1"]["journey_count"]
        p2j = entry["p2"]["journey_count"]
        print(f"  user_{short_id}.json  (P1: {p1j} journeys, P2: {p2j} journeys)")
        deep_count += 1

    print(f"\nSplit {deep_count} Deep users → {user_dir}")

    # ── Mirror to --output-dir ──
    if args.output_dir:
        import shutil
        mirror_user_dir = os.path.join(args.output_dir, args.run_name, "user_data")
        os.makedirs(mirror_user_dir, exist_ok=True)
        for fname in os.listdir(user_dir):
            shutil.copy2(os.path.join(user_dir, fname), mirror_user_dir)
        print(f"Mirrored {deep_count} files → {mirror_user_dir}")


if __name__ == "__main__":
    main()
