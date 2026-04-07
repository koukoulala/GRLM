import csv, sys
csv.field_size_limit(sys.maxsize)

target = {"785D60865FDB68722FE397C5FFFFFFFF", "4E37BB0500C64269E18D6469FFFFFFFF"}
f = "/cosmos/local/Aether/_e/xiaoyukou/e33c6a0e-e0d8-4333-97f7-bce316d58186@@@-General-_Cosmos_Split_N@@@650738c5@@@4-2-2026_10-59-55_AM/Part0/Part0_f3880241-632c-49a5-a251-ec2791838b3d"

found = set()
with open(f, "r") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    for row in reader:
        uid = row.get("UserId", "").strip()
        if uid in target:
            found.add(uid)
            print(f"FOUND: {uid}")
        if found == target:
            break

for uid in target - found:
    print(f"NOT FOUND: {uid}")
