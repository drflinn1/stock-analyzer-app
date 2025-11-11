#!/usr/bin/env python3
"""
trader/crypto_engine.py — Kraken utilities (LIVE)
• USD-only trading helpers
• Correct market buy: oflags=viqc, volume=<USD>
• Public quotes + Private orders + Balances (for reconciliation)
"""

from __future__ import annotations
import csv, os, time, json, hashlib, hmac, base64, urllib.parse
from pathlib import Path
from typing import Dict, List, Optional
from decimal import Decimal
import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"

KRAKEN_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET", "")
API_BASE = "https://api.kraken.com"

# ------------------------- CSV / Candidates -------------------------
def load_candidates(csv_path: Path = CANDIDATES_CSV) -> List[dict]:
    rows: List[dict] = []
    if not csv_path.exists():
        print("No candidates.csv found.")
        return rows
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sym = (row.get("symbol") or row.get("pair") or "").strip()
            q = (row.get("quote") or "").strip()
            if not sym:
                continue
            rows.append({"symbol": sym, "quote": q, "rank": row.get("rank")})
    return rows

# ------------------------- Symbol helpers --------------------------
def normalize_pair(s: str) -> str:
    """
    Accept: 'EAT/USD', 'EATUSD', 'eatusd', 'TERMEUR', etc.
    Return: canonical pair w/out slash, e.g., 'EATUSD', 'LSKUSD'.
    """
    s = (s or "").strip().upper().replace("/", "")
    # if explicit quote provided, keep; else default USD
    if s.endswith(("USD", "EUR", "USDT")):
        return s
    return s + "USD"

def is_usd_pair(pair: str) -> bool:
    return (pair or "").upper().endswith("USD")

def base_from_pair(pair: str) -> str:
    pair = normalize_pair(pair)
    return pair[:-3]  # strip 'USD'

def asset_to_usd_pair(asset: str) -> str:
    return f"{(asset or '').upper()}USD"

# --------------------------- Public Quotes --------------------------
def get_public_quote(pair: str) -> Optional[float]:
    """Return last trade price for a Kraken pair code (e.g., 'EATUSD')."""
    try:
        pair = normalize_pair(pair)
        if not is_usd_pair(pair):
            return None
        resp = requests.get(f"{API_BASE}/0/public/Ticker?pair={pair}", timeout=12)
        data = resp.json()
        if "result" in data and data["result"]:
            result = list(data["result"].values())[0]
            return float(result["c"][0])
    except Exception as e:
        print(f"Quote error for {pair}: {e}")
    return None

# --------------------------- Private Core ---------------------------
def _private_request(endpoint: str, payload: dict) -> dict:
    if not KRAKEN_KEY or not KRAKEN_SECRET:
        return {"error": ["EAuth:Missing API key/secret"]}

    nonce = str(int(time.time() * 1000))
    payload = {**payload, "nonce": nonce}
    postdata = urllib.parse.urlencode(payload)
    sha = hashlib.sha256((nonce + postdata).encode()).digest()
    path = f"/0/private/{endpoint}"
    msg = path.encode() + sha
    sig = hmac.new(base64.b64decode(KRAKEN_SECRET), msg, hashlib.sha512)
    sig64 = base64.b64encode(sig.digest()).decode()

    headers = {
        "API-Key": KRAKEN_KEY,
        "API-Sign": sig64,
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    }
    r = requests.post(API_BASE + path, headers=headers, data=postdata, timeout=20)
    try:
        return r.json()
    except Exception:
        return {"error": ["EAPI:Invalid JSON"], "raw": r.text}

# ----------------------------- Balances -----------------------------
def _normalize_asset_code(a: str) -> str:
    """
    Kraken returns weird prefixes for some assets (e.g., 'XXBT', 'ZUSD').
    Strip leading X/Z when followed by alpha letters to produce a simple base.
    """
    a = (a or "").strip().upper()
    if len(a) >= 2 and a[0] in ("X", "Z"):
        return a[1:]
    return a

def kraken_list_balances() -> Dict[str, float]:
    """
    Return spot balances as { BASE_ASSET: qty }, excluding zeroes.
    e.g., {'USD': 127.51, 'LSK': 94.60, 'MELANIA': 178.37}
    """
    resp = _private_request("Balance", {})
    out: Dict[str, float] = {}
    if resp.get("error"):
        print(f"Balance error: {resp.get('error')}")
        return out
    for k, v in (resp.get("result") or {}).items():
        try:
            qty = float(v)
        except Exception:
            continue
        if qty <= 0:
            continue
        out[_normalize_asset_code(k)] = qty
    return out

# ----------------------------- Orders ------------------------------
def place_market_buy_usd(pair: str, usd_amount: float) -> str:
    """
    LIVE MARKET BUY spending <usd_amount> of the quote currency (USD).
    Kraken: set oflags=viqc and volume=<quote_currency_amount>.
    """
    pair = normalize_pair(pair)
    if not is_usd_pair(pair):
        raise ValueError(f"Attempted USD buy on non-USD pair: {pair}")

    payload = {
        "pair": pair,
        "type": "buy",
        "ordertype": "market",
        "volume": f"{Decimal(usd_amount):f}",
        "oflags": "viqc",
        "userref": int(time.time()),
    }
    resp = _private_request("AddOrder", payload)
    if resp.get("error"):
        print("BUY error:", resp)
    else:
        print("BUY ok:", resp)
    return json.dumps(resp)

def place_market_sell_qty(pair: str, qty: float) -> str:
    """LIVE MARKET SELL by base-asset quantity."""
    pair = normalize_pair(pair)
    payload = {
        "pair": pair,
        "type": "sell",
        "ordertype": "market",
        "volume": f"{Decimal(qty):f}",
        "userref": int(time.time()),
    }
    resp = _private_request("AddOrder", payload)
    if resp.get("error"):
        print("SELL error:", resp)
    else:
        print("SELL ok:", resp)
    return json.dumps(resp)
