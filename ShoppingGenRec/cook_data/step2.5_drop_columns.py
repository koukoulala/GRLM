"""
Drop columns 5 (ShoppingJourney) and 6 (JourneyWithAllProducts),
keeping columns 1-4 (UserId, ReadableUserEvents, RequestTime, UserHistory)
and column 7 (JourneyWithProducts).
"""
import csv
import sys

csv.field_size_limit(sys.maxsize)

# INPUT = "/cosmos//projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/ShoppingJourney_Input_200K_Output_KeepHis50Results_withProducts_2.tsv"
# INPUT = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/ShoppingJourney_Input_500K_minus_150K_His50_Results_withProducts.tsv"
INPUT = "/vc_data/users/wangying/OneRec/ShoppingJourney/CookData/data/testdata/50K_journey_with_products.tsv"
# OUTPUT = "/cosmos//projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/ShoppingJourney_Input_500K_minus_105K_His50_JWP.tsv"
OUTPUT = "/vc_data/users/wangying/OneRec/ShoppingJourney/CookData/data/testdata/50K_journey_with_products_dropped.tsv"

DROP_COLS = {"ShoppingJourney", "JourneyWithAllProducts"}

with open(INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8", newline="") as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    keep = [c for c in reader.fieldnames if c not in DROP_COLS]
    writer = csv.DictWriter(fout, fieldnames=keep, delimiter="\t",
                            lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for i, row in enumerate(reader):
        writer.writerow({k: row[k] for k in keep})
        if (i + 1) % 50000 == 0:
            print(f"  {i+1} rows written...")

print(f"Done. Kept columns: {keep}")
