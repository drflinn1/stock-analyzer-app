#!/usr/bin/env python3
"""
kraken_momentum_scan.py

Builds .state/momentum_candidates.csv for the Monday rotation bot.

Logic:
- Query Kraken public API for a small universe of USD pairs.
- For each pair, compute 24h momentum:
    pct_change = (last - open24h) / open24h * 100
- Use last price as 'quote' and pct_change as 'rank'.
- Write CSV:
    symbol,quote,rank
  sorted by rank descending (best gainer first).

This script is DRY/RUN agnostic — it just builds candidates.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"

# Map normalized symbols to Kraken pair codes
# You can expand this list over time as you like.
KRAKEN_USD_PAIRS: Dict[str, str] = {
    "BTC/USD": "XXBTZUSD",
    "ETH/USD": "XETHZUSD",
    "SOL/USD": "SOLUSD",
    "XRP/USD": "XXRPZUSD",
    "ADA/USD": "ADAUSD",
    "DOGE/USD": "DOGEUSD",
    "LINK/USD": "LINKUSD",
    "BONK/USD": "BONKUSD",
    "LSK/USD": "LSKUSD",
    "MATIC/USD": "MATICUSD",
    "AVAX/USD": "AVAXUSD",
    "UNI/USD": "UNIUSD",
    "LTC/USD": "XLTCZUSD",
    "TRX/USD": "TRXUSD",
}


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(exist_ok=True)


def fetch_ticker(pair_code: str) -> Dict:
    """
    Fetch Kraken ticker data for a single pair code.
    Returns the JSON dict for that pair or raises.
    """
    url = "https://api.kraken.com/0/public/Ticker"
    resp = requests.get(url, params={"pair": pair_code}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error for {pair_code}: {data['error']}")
    result = data.get("result") or {}
    # Kraken returns a dict keyed by an internal pair name
    # Grab the first (and usually only) value.
    if not result:
        raise RuntimeError(f"No ticker result for {pair_code}")
    key = next(iter(result.keys()))
    return result[key]


def compute_momentum() -> List[Tuple[str, float, float]]:
    """
    Returns list of (symbol, last_price, pct_change_24h)
    for all configured USD pairs where data is valid.
    """
    out: List[Tuple[str, float, float]] = []
    for symbol, pair_code in KRAKEN_USD_PAIRS.items():
        try:
            t = fetch_ticker(pair_code)
            # last trade price
            last_str = (t.get("c") or ["0"])[0]
            # 24h opening price
            open_str = t.get("o") or "0"
            last = float(last_str)
            open_24h = float(open_str)
            if last <= 0 or open_24h <= 0:
                continue
            pct_change = (last - open_24h) / open_24h * 100.0
            out.append((symbol, last, pct_change))
        except Exception as e:
            print(f"[WARN] Failed to fetch/parse {symbol}: {e}", file=sys.stderr)
    return out


def write_candidates(rows: List[Tuple[str, float, float]]) -> None:
    """
    Write CSV with header: symbol,quote,rank
    rank = pct_change, descending (best gainer first).
    """
    ensure_state_dir()
    # Sort by rank descending
    rows_sorted = sorted(rows, key=lambda r: r[2], reverse=True)

    with CANDIDATES_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "quote", "rank"])
        for symbol, last, pct_change in rows_sorted:
            writer.writerow([symbol, f"{last:.10f}", f"{pct_change:.6f}"])

    print(f"Wrote {len(rows_sorted)} momentum candidates to {CANDIDATES_CSV}")


def main() -> None:
    rows = compute_momentum()
    if not rows:
        print("[WARN] No valid momentum data; momentum_candidates.csv will be empty", file=sys.stderr)
    write_candidates(rows)


if __name__ == "__main__":
    main()
