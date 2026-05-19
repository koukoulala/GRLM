"""
extract_vip_users.py
====================

Extract VIP user data from sample_user_prompt_output.jsonl and save as
a TSV file that step3 can directly consume.

Input:  sample_user_prompt_output.jsonl (each line: {stableid, request_body})
Output: vip_users.tsv with columns:
        UserId, ReadableUserEvents, ShoppingProfile, RequestTime, HisCount

Usage:
    python extract_vip_users.py
    python extract_vip_users.py --input sample_user_prompt_output.jsonl --output vip_users.tsv
"""

import argparse
import csv
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_from_prompt(prompt_text):
    """Extract ShoppingProfile and ReadableUserEvents from a prompt string."""
    # Find profile
    prof_marker = "User Shopping Profile:\n"
    events_marker = "\nRecent Shopping Events:\n"
    predict_marker = "\nPredict the user"

    prof_start = prompt_text.find(prof_marker)
    events_start = prompt_text.find(events_marker)
    predict_start = prompt_text.find(predict_marker)

    profile = ""
    events = ""

    if prof_start != -1 and events_start != -1:
        profile = prompt_text[prof_start + len(prof_marker):events_start].strip()

    if events_start != -1 and predict_start != -1:
        events = prompt_text[events_start + len(events_marker):predict_start].strip()

    return profile, events


def main():
    parser = argparse.ArgumentParser(
        description="Extract VIP users from JSONL to step3-compatible TSV")
    parser.add_argument("--input", default=os.path.join(
        SCRIPT_DIR, "sample_user_prompt_output.jsonl"))
    parser.add_argument("--output", default=os.path.join(
        SCRIPT_DIR, "vip_users.tsv"))
    args = parser.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    users = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj.get("stableid", "")
            prompt = obj.get("request_body", {}).get("prompt", "")
            profile, events = extract_from_prompt(prompt)

            # Count events
            event_lines = [l for l in events.split("\n") if l.strip()]
            his_count = len(event_lines)

            # Convert events: newline-separated → #N# separated
            events_tsv = events.replace("\n", "#N#")

            users.append({
                "UserId": uid,
                "ReadableUserEvents": events_tsv,
                "ShoppingProfile": profile,
                "RequestTime": "",
                "HisCount": str(his_count),
            })

    # Write TSV
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["UserId", "ReadableUserEvents", "ShoppingProfile",
                         "RequestTime", "HisCount"],
            delimiter="\t",
        )
        writer.writeheader()
        for u in users:
            writer.writerow(u)

    print(f"Extracted {len(users)} users to {args.output}")
    for u in users[:3]:
        events_len = len(u["ReadableUserEvents"])
        profile_len = len(u["ShoppingProfile"])
        print(f"  {u['UserId'][:20]}...  events={events_len}B  "
              f"profile={profile_len}B  his_count={u['HisCount']}")


if __name__ == "__main__":
    main()
