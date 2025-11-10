#!/usr/bin/env python3
"""
Ensures the given CSV has columns: symbol, quote, rank.
Writes sibling file with _fixed.csv suffix.
"""
import sys, os, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else ".state/momentum_candidates.csv"
os.makedirs(".state", exist_ok=True)

if not os.path.exists(path):
    pd.DataFrame(columns=["symbol","quote","rank"]).to_csv(path, index=False)

df = pd.read_csv(path)
for col in ["symbol","quote","rank"]:
    if col not in df.columns:
        df[col] = range(1, len(df) + 1) if col == "rank" else None

fixed_path = path.replace(".csv", "_fixed.csv")
df[["symbol","quote","rank"]].to_csv(fixed_path, index=False)
print(f"✅ Normalized schema → {fixed_path}")
