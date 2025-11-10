#!/usr/bin/env python3
"""
Placeholder momentum scanner.
Writes .state/momentum_candidates.csv with columns:
pair, score, pct24, usd_vol, ema_slope
"""
import csv, random
from pathlib import Path

STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)

pairs = ["KAVA/USD","STRK/USD","EAT/USD","CFG/USD"]
rows = []
for p in pairs:
    rows.append({
        "pair": p,
        "score": round(10 + random.random()*20, 3),
        "pct24": round(5 + random.random()*15, 2),
        "usd_vol": int(50_000 + random.random()*2_000_000),
        "ema_slope": round(-0.02 + random.random()*0.04, 5),
    })

out = STATE / "momentum_candidates.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["pair","score","pct24","usd_vol","ema_slope"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {out} ({len(rows)} rows)")
