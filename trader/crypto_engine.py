#!/usr/bin/env python3
"""
crypto_engine.py
Price + candidate utilities for the rotation bot.

Goals:
- Never stall the buy/sell logic due to missing/invalid quotes.
- Work even when .state/momentum_candidates.csv is missing quotes or is read-only.
- Use Kraken public API as a reliable fallback.

Exports used by main.py (safe to call):
- load_candidates(csv_path=".state/momentum_candidates.csv") -> list[dict]
- get_public_quote(pair: str) -> float | None
- get_public_quotes(pairs: list[str]) -> dict[str, float]
- normalize_pair(s: str) -> str
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"

# Simple in-process cache so repeated asks in one run are cheap
_QUOTES_CACHE: Dict[str, float] = {}
_CACHE_TTL = 30  # seconds
_CACHE_STAMP: Dict[str, float] = {}


# ----------------------------- Kraken Public API -----------------------------

def _kraken_ticker(pair_code: str) -> Optional[float]:
    """
    Call Kraken public ticker for the given pair code (e.g., 'SOLUSD', 'XRPUSD').
    Returns last trade price as float or None on failure.
    """
    url = "https://api.kraken.com/0/public/Ticker"
    try:
        resp = requests.get(url, params={"pair": pair_code}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return None
        # Kraken returns result under the passed pair key (can be remapped internally)
        result = data.get("result", {})
        if not result:
            return None
        # Grab the first (only) entry
        _k, v = next(iter(result.items()))
        # 'c': [<last_price>, <lot>]
        last = v.get("c", [None])[0]
        if last is None:
            return None
        return float(last)
    except Exception:
        return None


def _to_kraken_code(pair: str) -> str:
    """
    Convert 'SOL/USD' -> 'SOLUSD', 'XBT/USD' -> 'XBTUSD', 'DOGEUSD' -> 'DOGEUSD'.
    We don't attempt the legacy X/Z prefixes here—Kraken accepts 'SOLUSD' etc.
    """
    p = pair.strip().upper().replace(" ", "")
    if "/" in p:
        a, b = p.split("/", 1)
        return f"{a}{b}"
    return p


def normalize_pair(s: str) -> str:
    """
    Normalizes incoming pair labels to a standard 'BASE/QUOTE' form with upper case.
    Accepts 'sol/usd', 'SOLUSD', 'SOL-USD', 'SOL/USD' → 'SOL/USD'
    """
    t = s.strip().upper().replace("-", "/")
    if "/" in t:
        a, b = t.split("/", 1)
        return f"{a}/{b}"
    # If no slash, assume USD quote
    if t.endswith("USD"):
        return f"{t[:-3]}/USD"
    return t


def _is_valid_price(x: Optional[float]) -> bool:
    if x is None:
        return False
    # sanity: > 0 and not NaN/Inf (float checks)
    try:
        return x > 0 and x < 1e9
    except Exception:
        return False


def get_public_quote(pair: str) -> Optional[float]:
    """
    Cached public-quote getter for a single pair.
    """
    pair_norm = normalize_pair(pair)
    now = time.time()
    if pair_norm in _QUOTES_CACHE and (now - _CACHE_STAMP.get(pair_norm, 0)) <= _CACHE_TTL:
        return _QUOTES_CACHE[pair_norm]

    code = _to_kraken_code(pair_norm)
    price = _kraken_ticker(code)
    if _is_valid_price(price):
        _QUOTES_CACHE[pair_norm] = float(price)
        _CACHE_STAMP[pair_norm] = now
        return float(price)
    return None


def get_public_quotes(pairs: List[str]) -> Dict[str, float]:
    """
    Batch convenience: returns a dict of {normalized_pair: price} for valid quotes only.
    """
    out: Dict[str, float] = {}
    for p in pairs:
        pn = normalize_pair(p)
        q = get_public_quote(pn)
        if _is_valid_price(q):
            out[pn] = float(q)
    return out


# ----------------------------- Candidates Loader -----------------------------

def _parse_candidates_csv(csv_path: Path) -> List[dict]:
    """
    Reads CSV rows. Accepts headers with at least: pair, rank
    Optional header: quote. If missing/invalid, we will public-fetch later.
    Returns a list of dicts with normalized fields.
    """
    rows: List[dict] = []
    if not csv_path.exists():
        return rows

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # flexible header names
            hdr = {h.lower().strip(): h for h in reader.fieldnames or []}

            def get(row, key, default=""):
                # search case-insensitively
                for k in (key, key.lower(), key.upper()):
                    if k in row:
                        return row[k]
                # header map help
                if key in hdr:
                    return row.get(hdr[key], default)
                return default

            for row in reader:
                pair_raw = get(row, "pair", "").strip()
                if not pair_raw:
                    pair_raw = get(row, "symbol", "").strip()
                if not pair_raw:
                    continue

                pair = normalize_pair(pair_raw)
                rank_s = get(row, "rank", "").strip()
                quote_s = get(row, "quote", "").strip()

                try:
                    rank = int(float(rank_s)) if rank_s != "" else 0
                except Exception:
                    rank = 0

                try:
                    quote = float(quote_s) if quote_s not in ("", None) else None
                except Exception:
                    quote = None

                rows.append({"pair": pair, "rank": rank, "quote": quote})
    except Exception:
        # if unreadable / locked / malformed, just return empty; caller will refetch
        return []

    return rows


def _fill_missing_quotes(rows: List[dict]) -> List[dict]:
    """
    For any candidate without a valid quote, fetch from Kraken public API.
    """
    for r in rows:
        if not _is_valid_price(r.get("quote")):
            q = get_public_quote(r["pair"])
            if _is_valid_price(q):
                r["quote"] = float(q)
    return rows


def _best_effort_candidates(csv_path: Path) -> List[dict]:
    """
    Best-effort pipeline: read CSV if present, fill any missing quotes via public API.
    If file is missing/readonly/corrupt, fallback to a trivial default universe that
    still yields a valid price so the run never stalls.
    """
    rows = _parse_candidates_csv(csv_path)

    if not rows:
        # Fallback tiny universe — matches the YAML seed so behavior is predictable
        rows = [{"pair": "EAT/USD", "rank": 100, "quote": None}]

    rows = _fill_missing_quotes(rows)

    # Final safety: filter only those with valid quotes
    rows = [r for r in rows if _is_valid_price(r.get("quote"))]

    return rows


def load_candidates(csv_path: str | Path = CANDIDATES_CSV) -> List[dict]:
    """
    Public API for main.py.
    Returns sorted candidates (highest rank first) with guaranteed valid 'quote' floats.
    """
    path = Path(csv_path)
    rows = _best_effort_candidates(path)
    # Highest rank first (ties stable)
    rows.sort(key=lambda r: (r.get("rank", 0), r["pair"]), reverse=True)
    return rows
