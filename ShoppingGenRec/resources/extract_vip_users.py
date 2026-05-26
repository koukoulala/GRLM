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

    # Write TSV — order users to match SLM HTML display order
    # (SLM HTML lists these 20 users in a specific order; 2 extra users
    # with no valid journeys are appended at the end)
    SLM_ORDER = [
        "FF1353FD90160A49", "D7073CB4BA6E7FFD", "AA7F1661F70AE870",
        "1745833BCFEB598F", "673AEA9C25D54B2E", "00DB9B18F9F467B9",
        "0BD7ADCEABB34025", "120561EB543166ED", "1AFEFCE562C962B5",
        "22967BA55099D483", "3BC2C939156A6233", "3C74C986EAB03584",
        "58767B1A056CB90A", "6614E456A908CF5C", "7158110DE72A447D",
        "9A352E33951C30B6", "A62647067FE38C8A", "C354118B4A51BF34",
        "E4785BFAA23D2810", "CE04DD86FA7F55FF",
    ]
    uid_to_user = {u["UserId"][:16]: u for u in users}
    ordered = []
    seen = set()
    for prefix in SLM_ORDER:
        for u in users:
            if u["UserId"].startswith(prefix) and u["UserId"] not in seen:
                ordered.append(u)
                seen.add(u["UserId"])
                break
    # Append any remaining users not in SLM order
    for u in users:
        if u["UserId"] not in seen:
            ordered.append(u)
            seen.add(u["UserId"])
    users = ordered

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
