#!/usr/bin/env python3
"""
tools/momentum_scan.py
Real momentum (gainers) scanner for Kraken pairs.

Writes: .state/momentum_candidates.csv with columns:
  symbol, quote, rank
"""

import os
import math
import requests
import pandas as pd

os.makedirs(".state", exist_ok=True)
OUT_CSV = ".state/momentum_candidates.csv"

def fetch_ticker():
    url = "https://api.kraken.com/0/public/Ticker"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    if "result" not in j:
        raise RuntimeError(f"Unexpected Kraken response: {j}")
    return j["result"]

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def main():
    data = fetch_ticker()
    rows = []

    for pair, info in data.items():
        last = to_float(info.get("c", [None, None])[0])
        o = to_float(info.get("o"), None)  # open (if present)
        if last is None:
            continue
        if o and o > 0:
            change_pct = (last - o) / o * 100.0
        else:
            change_pct = math.nan
        rows.append((pair, last, change_pct))

    if not rows:
        pd.DataFrame(columns=["symbol", "quote", "rank"]).to_csv(OUT_CSV, index=False)
        print(f"⚠️ No rows parsed; wrote empty CSV header → {OUT_CSV}")
        return

    df = pd.DataFrame(rows, columns=["symbol", "quote", "change_pct"])
    if df["change_pct"].notna().any():
        df = df.sort_values("change_pct", ascending=False, na_position="last")
    else:
        df = df.sort_values("quote", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df[["symbol", "quote", "rank"]].head(50).to_csv(OUT_CSV, index=False)
    print(f"✅ Saved {min(50, len(df))} momentum candidates → {OUT_CSV}")

if __name__ == "__main__":
    main()
