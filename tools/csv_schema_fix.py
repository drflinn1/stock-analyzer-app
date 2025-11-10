#!/usr/bin/env python3
"""
tools/csv_schema_fix.py
Ensures the given CSV has columns: symbol, quote, rank.
If rank missing, assigns 1..N in current order.
If quote missing, fills with empty values.
Writes a sibling file with _fixed.csv suffix.
"""

import sys, os
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else ".state/momentum_candidates.csv"
os.makedirs(".state", exist_ok=True)

if not os.path.exists(path):
    # create empty with schema if source missing
    pd.DataFrame(columns=["symbol", "quote", "rank"]).to_csv(path, index=False)

df = pd.read_csv(path)

for col in ["symbol", "quote", "rank"]:
    if col not in df.columns:
        if col == "rank":
            df[col] = range(1, len(df) + 1)
        else:
            df[col] = None

fixed_path = path.replace(".csv", "_fixed.csv")
df[["symbol", "quote", "rank"]].to_csv(fixed_path, index=False)
print(f"✅ Normalized schema → {fixed_path}")
