#!/usr/bin/env python3
"""
spike_scan.py
Builds .state/momentum_candidates.csv with columns:
  symbol,quote,rank
Uses Kraken public API only (no secrets).
Robust to network hiccups; never crashes the workflow.
"""

from __future__ import annotations
import csv, json, time, sys
from pathlib import Path
from typing import Dict, List, Tuple
import math

import requests

STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)

ASSET_PAIRS_URL = "https://api.kraken.com/0/public/AssetPairs"
TICKER_URL = "https://api.kraken.com/0/public/Ticker"

def fetch_pairs_usd() -> List[str]:
    try:
        r = requests.get(ASSET_PAIRS_URL, timeout=15)
        r.raise_for_status()
        data = r.json().get("result", {})
    except Exception as e:
        print(f"⚠️  fetch_pairs_usd failed: {e}", file=sys.stderr)
        return []
    out = []
    for name, meta in data.items():
        # Kraken naming: "LSKUSD", "ADAUSD", sometimes "XETHZUSD" etc.
        # Keep only spot USD quote; skip dark/forwards.
        if not isinstance(name, str):
            continue
        if "USD" not in name:
            continue
        if ".d" in name.lower():  # dark pool
            continue
        out.append(name)
    return sorted(set(out))

def fetch_ticker(pairs: List[str]) -> Dict[str, dict]:
    if not pairs:
        return {}
    try:
        # Kraken lets you comma-join
        joined = ",".join(pairs[:100])  # be nice
        r = requests.get(TICKER_URL, params={"pair": joined}, timeout=20)
        r.raise_for_status()
        return r.json().get("result", {})
    except Exception as e:
        print(f"⚠️  fetch_ticker failed: {e}", file=sys.stderr)
        return {}

def norm_symbol(raw_pair: str) -> str:
    # Prefer left/base side (e.g., LSK from LSKUSD)
    s = raw_pair.upper()
    if "/" in s:
        s = s.split("/")[0]
    if s.endswith("USD"):
        s = s[:-3]
    return s

def parse_24h_change(trow: dict) -> Tuple[float, float]:
    """
    Returns (last_price_usd, pct_change_24h)
    Kraken fields:
      'c' -> [last_trade_price, lot]  (string)
      'p' -> [today_vwap, 24h_vwap]
      'o' -> opening price
    """
    try:
        last = float(trow["c"][0])
    except Exception:
        last = float(trow.get("o", 0.0)) if str(trow.get("o", "")).replace(".","",1).isdigit() else 0.0
    try:
        o = float(trow["o"])
    except Exception:
        o = 0.0
    pct = (last - o) / o * 100.0 if o > 0 else 0.0
    return last, pct

def build_candidates() -> List[dict]:
    pairs = fetch_pairs_usd()
    tick = fetch_ticker(pairs)
    rows = []
    for pair, trow in tick.items():
        last, pct = parse_24h_change(trow)
        if last <= 0:
            continue
        sym = norm_symbol(pair)
        rows.append({"symbol": sym, "quote": last, "change_24h": pct})
    # rank: highest gainers first
    rows.sort(key=lambda r: r["change_24h"], reverse=True)
    # Give top N a rank score (100, 99, 98, …)
    out = []
    for i, r in enumerate(rows[:20]):
        out.append({"symbol": r["symbol"], "quote": round(r["quote"], 8), "rank": 100 - i})
    return out

def write_csv(rows: List[dict]) -> None:
    path = STATE / "momentum_candidates.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "quote", "rank"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_summary(ok: bool, n: int) -> None:
    summary = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": "ok" if ok else "error",
        "candidates": n,
    }
    with (STATE / "scan_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def main() -> int:
    try:
        rows = build_candidates()
        write_csv(rows)
        write_summary(True, len(rows))
        print(f"✅ Wrote {len(rows)} candidates to .state/momentum_candidates.csv")
        return 0
    except Exception as e:
        print(f"❌ Scan failed: {e}", file=sys.stderr)
        write_summary(False, 0)
        # still ensure CSV exists (empty with header)
        try:
            write_csv([])
        except Exception:
            pass
        return 0  # do not fail the job
if __name__ == "__main__":
    raise SystemExit(main())
