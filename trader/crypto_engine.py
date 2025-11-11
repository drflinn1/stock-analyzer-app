#!/usr/bin/env python3
"""
trader/crypto_engine.py — Kraken utilities (LIVE)

Nov-10 fixes:
• load_candidates() accepts either:
  A) symbol,quote,rank
  B) pair,score,pct24,usd_vol,ema_slope
• Only trade USD-quoted pairs (skip EUR/USDT/etc.)
• Correct market buy: use oflags=viqc and volume=<USD amount>
"""

from __future__ import annotations
import csv, os, time, json
from pathlib import Path
from typing import Dict, List, Optional
from decimal import Decimal
import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"

KRAKEN_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET")
API_BASE = "https://api.kraken.com"

# ------------------------- CSV / Candidates -------------------------
def load_candidates(csv_path: Path = CANDIDATES_CSV) -> List[dict]:
    rows: List[dict] = []
    if not csv_path.exists():
        print("No candidates.csv found.")
        return rows

    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        headers = [h.strip().lower() for h in r.fieldnames or []]

        # Schema A: symbol,quote,rank
        schema_a = {"symbol", "quote"}
        # Schema B: pair,score,pct24,...
        schema_b = {"pair", "score"}

        for row in r:
            out: Dict[str, str] = {}
            if schema_a.issubset(headers):
                s = (row.get("symbol") or "").strip()
                if not s:
                    continue
                out["symbol"] = s
                out["pair"] = s
            elif schema_b.issubset(headers):
                p = (row.get("pair") or "").strip()
                if not p:
                    continue
                out["pair"] = p
                out["symbol"] = p
            else:
                # Unknown schema; ignore row
                continue
            rows.append(out)
    return rows

# ------------------------- Symbol helpers --------------------------
def normalize_pair(s: str) -> str:
    """
    Accept: 'EAT/USD', 'EATUSD', 'eatusd', 'EATUSDT', 'TERMEUR'
    Return: canonical Kraken pair without slash (e.g., 'EATUSD', 'SOLUSD').
    """
    s = s.strip().upper().replace("/", "")
    if s.endswith(("USD", "EUR", "USDT")):
        return s
    return s + "USD"

def is_usd_pair(pair: str) -> bool:
    return pair.endswith("USD")

# --------------------------- Quotes --------------------------------
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

# --------------------------- Private API ---------------------------
def _private_request(endpoint: str, payload: dict) -> dict:
    import hashlib, hmac, base64, urllib.parse
    nonce = str(int(time.time() * 1000))
    payload["nonce"] = nonce
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

# ----------------------------- Orders ------------------------------
def place_market_buy_usd(pair: str, usd_amount: float) -> str:
    """
    LIVE MARKET BUY spending <usd_amount> of USD (quote).
    Kraken rule: use oflags=viqc and set volume=<quote_currency_amount>.
    """
    pair = normalize_pair(pair)
    if not is_usd_pair(pair):
        raise ValueError(f"Attempted USD buy on non-USD pair: {pair}")

    payload = {
        "pair": pair,
        "type": "buy",
        "ordertype": "market",
        "volume": f"{Decimal(usd_amount):f}",  # Spend this much USD
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
