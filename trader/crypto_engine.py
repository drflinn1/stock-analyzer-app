#!/usr/bin/env python3
"""
trader/crypto_engine.py — Kraken utilities
LIVE version (market orders)

Exports:
- load_candidates()
- get_public_quote(pair)
- place_market_buy_usd(pair, usd_amount)
- place_market_sell_qty(pair, qty)
"""

from __future__ import annotations
import csv, os, time
from pathlib import Path
from typing import Dict, List, Optional
import requests
import json
from decimal import Decimal

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"

KRAKEN_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET")
API_BASE = "https://api.kraken.com"

# ============================================================
def load_candidates(csv_path: Path = CANDIDATES_CSV) -> List[dict]:
    rows = []
    if not csv_path.exists():
        print("No candidates.csv found.")
        return rows
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("symbol") and row.get("quote"):
                rows.append(row)
    return rows

def normalize_pair(s: str) -> str:
    s = s.replace("/", "")
    if not s.endswith("USD"):
        s += "USD"
    return s

def get_public_quote(pair: str) -> Optional[float]:
    try:
        pair = normalize_pair(pair)
        resp = requests.get(f"{API_BASE}/0/public/Ticker?pair={pair}", timeout=10)
        data = resp.json()
        if "result" in data:
            result = list(data["result"].values())[0]
            return float(result["c"][0])
    except Exception as e:
        print(f"Quote error for {pair}: {e}")
    return None

# ============================================================
# --- LIVE order endpoints ---
def _private_request(endpoint: str, payload: dict) -> dict:
    """Kraken private REST (signed) request."""
    import hashlib, hmac, base64
    nonce = str(int(time.time() * 1000))
    payload["nonce"] = nonce
    post_data = "&".join([f"{k}={v}" for k, v in payload.items()])
    sha = hashlib.sha256((nonce + post_data).encode()).digest()
    import urllib.parse
    path = f"/0/private/{endpoint}"
    hmac_msg = path.encode() + sha
    signature = hmac.new(base64.b64decode(KRAKEN_SECRET), hmac_msg, hashlib.sha512)
    sig_digest = base64.b64encode(signature.digest())
    headers = {
        "API-Key": KRAKEN_KEY,
        "API-Sign": sig_digest.decode(),
    }
    r = requests.post(API_BASE + path, headers=headers, data=payload, timeout=15)
    return r.json()

def place_market_buy_usd(pair: str, usd_amount: float) -> str:
    """Place a live MARKET buy using given USD amount."""
    pair = normalize_pair(pair)
    payload = {
        "pair": pair,
        "type": "buy",
        "ordertype": "market",
        "oflags": "fciq",
        "userref": int(time.time()),
        "volume": "",  # Kraken auto-calculates when using cost
        "cost": str(usd_amount),
    }
    resp = _private_request("AddOrder", payload)
    if resp.get("error"):
        print("BUY error:", resp)
    else:
        print("BUY ok:", resp)
    return json.dumps(resp)

def place_market_sell_qty(pair: str, qty: float) -> str:
    """Place a live MARKET sell by quantity."""
    pair = normalize_pair(pair)
    payload = {
        "pair": pair,
        "type": "sell",
        "ordertype": "market",
        "userref": int(time.time()),
        "volume": f"{Decimal(qty):f}",
    }
    resp = _private_request("AddOrder", payload)
    if resp.get("error"):
        print("SELL error:", resp)
    else:
        print("SELL ok:", resp)
    return json.dumps(resp)
