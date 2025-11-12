#!/usr/bin/env python3
"""
trader/momentum_scan.py
Fetch top USD movers from Kraken public API and write .state/momentum_candidates.csv

Output CSV columns:
- symbol  (e.g., LSK/USD)
- quote   (last price, float)
- rank    (100 = strongest)
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"

KRAKEN_API = "https://api.kraken.com/0/public"

def get_asset_pairs_usd() -> Dict[str, str]:
    """
    Return mapping {wsname: paircode}, only /USD pairs.
    Example: {"LSK/USD": "LSKUSD", "SOL/USD": "SOLUSD", ...}
    """
    url = f"{KRAKEN_API}/AssetPairs"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    out: Dict[str, str] = {}
    for paircode, meta in data.get("result", {}).items():
        ws = meta.get("wsname")
        if not ws or "/USD" not in ws:
            continue
        out[ws] = paircode  # paircode like LSKUSD
    return out

def get_tickers(paircodes: List[str]) -> Dict[str, dict]:
    url = f"{KRAKEN_API}/Ticker"
    chunks: List[List[str]] = []
    # keep query strings short
    buf: List[str] = []
    for p in paircodes:
        buf.append(p)
        if len(buf) == 20:
            chunks.append(buf)
            buf = []
    if buf:
        chunks.append(buf)

    results: Dict[str, dict] = {}
    for chunk in chunks:
        q = ",".join(chunk)
        r = requests.get(url, params={"pair": q}, timeout=20)
        r.raise_for_status()
        data = r.json().get("result", {})
        results.update(data)
        time.sleep(0.3)  # gentle rate-limit
    return results

def main() -> int:
    STATE_DIR.mkdir(exist_ok=True)
    pairs_map = get_asset_pairs_usd()  # wsname -> paircode
    if not pairs_map:
        print("[ERROR] No USD asset pairs returned from Kraken.", file=sys.stderr)
        return 2

    tickers = get_tickers(list(pairs_map.values()))
    rows: List[Tuple[str, float, float]] = []  # (wsname, last, pct_change)

    for wsname, paircode in pairs_map.items():
        t = tickers.get(paircode)
        if not t:
            continue
        # Kraken fields: c = [last, lot], o = open today
        try:
            last = float(t["c"][0])
            openp = float(t.get("o", last))
        except Exception:
            continue
        if openp <= 0:
            continue
        pct = (last - openp) / openp * 100.0
        rows.append((wsname, last, pct))

    # sort by % change, desc, take top N
    rows.sort(key=lambda x: x[2], reverse=True)
    topN = rows[:25]  # keep it modest

    # Write candidates
    with CANDIDATES_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "quote", "rank"])
        # rank 100.. (descending)
        rank = 100
        for wsname, last, _pct in topN:
            w.writerow([wsname, f"{last:.10f}", rank])
            rank -= 1

    print(f"[INFO] wrote {len(topN)} candidates to {CANDIDATES_CSV}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
