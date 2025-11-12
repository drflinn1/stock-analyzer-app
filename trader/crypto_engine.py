#!/usr/bin/env python3
"""
trader/crypto_engine.py
Stable quote helper with hold-on-miss fallback.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"
KRAKEN_API = "https://api.kraken.com/0/public"
_QUOTES_CACHE: Dict[str, float] = {}


def normalize_pair(s: str) -> str:
    s = (s or "").strip().upper().replace("USDT", "USD")
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}/USD"
    if s.endswith("USD") and len(s) > 3:
        return f"{s[:-3]}/USD"
    return f"{s}/USD"


def _kraken_quote_try(paircodes: List[str]) -> Optional[float]:
    url = f"{KRAKEN_API}/Ticker"
    for p in paircodes:
        try:
            r = requests.get(url, params={"pair": p}, timeout=15)
            if not r.ok:
                continue
            data = r.json().get("result", {})
            if not data:
                continue
            first = next(iter(data.values()))
            return float(first["c"][0])
        except Exception:
            continue
        time.sleep(0.2)
    return None


def get_public_quote(pair: str) -> Optional[float]:
    """Robust public quote for Kraken (tries multiple forms)."""
    canon = normalize_pair(pair)
    if canon in _QUOTES_CACHE:
        return _QUOTES_CACHE[canon]

    base = canon.split("/")[0]
    tries = [f"{base}USD", canon.replace("/", "")]
    price = _kraken_quote_try(tries)
    if price:
        _QUOTES_CACHE[canon] = price
        return price
    return None


def get_public_quotes(pairs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in pairs:
        q = get_public_quote(p)
        if q:
            out[normalize_pair(p)] = q
    return out


def load_candidates(csv_path: str | Path = CANDIDATES_CSV) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    path = Path(csv_path)
    if not path.exists():
        return rows

    with path.open() as f:
        for row in csv.DictReader(f):
            sym = normalize_pair(row.get("symbol", ""))
            rows.append({
                "symbol": sym,
                "quote": row.get("quote", ""),
                "rank": row.get("rank", ""),
            })
    return rows


# --- NEW ---
def safe_quote(pair: str) -> float:
    """Return valid quote or 0 to signal 'hold'."""
    price = get_public_quote(pair)
    if price is None:
        print(f"[WARN] quote miss for {pair} — HOLD current position.")
        return 0.0
    return price
