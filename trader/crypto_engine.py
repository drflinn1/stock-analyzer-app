#!/usr/bin/env python3
"""
trader/crypto_engine.py
Price + candidate utilities for the rotation bot.

- Works even if CSV is sparse.
- Normalizes pairs to BASE/USD.
- Tries both slash and noslash formats (e.g., LSK/USD and LSKUSD).
- Has a Kraken public Ticker fallback for stubborn symbols.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"
KRAKEN_API = "https://api.kraken.com/0/public"

_QUOTES_CACHE: Dict[str, float] = {}

def normalize_pair(s: str) -> str:
    """Return BASE/USD with slash, uppercase."""
    s = (s or "").strip().upper().replace("USDT", "USD")
    if "/" in s:
        base, quote = s.split("/", 1)
        if quote != "USD":
            quote = "USD"
        return f"{base}/{quote}"
    # noslash form like LSKUSD, SOLUSD, BTCUSD
    if s.endswith("USD") and len(s) > 3:
        base = s[:-3]
        return f"{base}/USD"
    return s if s.endswith("/USD") else f"{s.replace('/', '')}/USD"

def _to_paircodes(canon: str) -> List[str]:
    """From BASE/USD produce possible Kraken pair codes to try (noslash)."""
    base = canon.split("/", 1)[0]
    noslash = f"{base}USD"       # e.g., LSKUSD
    return [noslash, base]       # try LSKUSD then LSK (rare, but harmless)

def _kraken_public_quote(paircodes: List[str]) -> Optional[float]:
    """Ask Kraken Ticker for the first paircode that exists."""
    url = f"{KRAKEN_API}/Ticker"
    for p in paircodes:
        try:
            r = requests.get(url, params={"pair": p}, timeout=15)
            r.raise_for_status()
            res = r.json().get("result", {})
            if not res:
                continue
            # take the first value present
            first = next(iter(res.values()))
            last = float(first["c"][0])
            return last
        except Exception:
            continue
    return None

def get_public_quote(pair: str) -> Optional[float]:
    """Robust public quote for a single pair name."""
    canon = normalize_pair(pair)
    if canon in _QUOTES_CACHE:
        return _QUOTES_CACHE[canon]

    # Try Kraken directly
    price = _kraken_public_quote(_to_paircodes(canon))
    if price is not None:
        _QUOTES_CACHE[canon] = price
        return price
    return None

def get_public_quotes(pairs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in pairs:
        q = get_public_quote(p)
        if q is not None:
            out[normalize_pair(p)] = q
    return out

def load_candidates(csv_path: str | Path = CANDIDATES_CSV) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    path = Path(csv_path)
    if not path.exists():
        return rows
    with path.open() as f:
        cr = csv.DictReader(f)
        for row in cr:
            sym = normalize_pair(row.get("symbol", ""))
            rows.append({"symbol": sym, "quote": row.get("quote", ""), "rank": row.get("rank", "")})
    return rows
