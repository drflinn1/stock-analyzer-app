#!/usr/bin/env python3
"""
Kraken helpers for USD-only rotation
• Public quotes
• Balances + USD balance helper
• Market BUY (viqc)
• LIMIT BUY IOC fallback
• Market SELL by qty
"""

from __future__ import annotations
import csv, os, time, json, hashlib, hmac, base64, urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"

KRAKEN_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET", "")
API_BASE = "https://api.kraken.com"

# ------------------------- CSV -------------------------
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

# ------------------------- Symbols ---------------------
def normalize_pair(s: str) -> str:
    s = (s or "").strip().upper().replace("/", "")
    if s.endswith(("USD", "EUR", "USDT")):
        return s
    return s + "USD"

def is_usd_pair(pair: str) -> bool:
    return (pair or "").upper().endswith("USD")

def base_from_pair(pair: str) -> str:
    pair = normalize_pair(pair)
    return pair[:-3]

def asset_to_usd_pair(asset: str) -> str:
    return f"{(asset or '').upper()}USD"

# ------------------------- Public quote ----------------
def get_public_quote(pair: str) -> Optional[float]:
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

# -------------------- Private request core --------------
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

# ------------------------- Balances --------------------
def _normalize_asset_code(a: str) -> str:
    a = (a or "").strip().upper()
    if len(a) >= 2 and a[0] in ("X", "Z"):
        return a[1:]
    return a

def kraken_list_balances() -> Dict[str, float]:
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
        if qty <= 0: continue
        out[_normalize_asset_code(k)] = qty
    return out

def get_usd_balance() -> float:
    bal = kraken_list_balances()
    # Kraken may expose USD as USD or ZUSD
    return float(bal.get("USD", 0.0) or 0.0)

# -------------------- Price protection detect ----------
def was_price_protection_block(resp: dict) -> bool:
    errs = resp.get("error") or []
    return bool(errs)  # permissive: any order error triggers fallback

# --------------------------- Orders --------------------
def place_market_buy_usd(pair: str, usd_amount: float, return_blocked: bool=False) -> Tuple[str, bool] | str:
    pair = normalize_pair(pair)
    if not is_usd_pair(pair):
        raise ValueError(f"Attempted USD buy on non-USD pair: {pair}")

    payload = {
        "pair": pair,
        "type": "buy",
        "ordertype": "market",
        "volume": f"{Decimal(usd_amount):f}",  # spend USD
        "oflags": "viqc",
        "userref": int(time.time()),
    }
    resp = _private_request("AddOrder", payload)
    blocked = was_price_protection_block(resp)
    out = json.dumps(resp)
    if return_blocked:
        return out, blocked
    return out

def place_limit_buy_usd_ioc(pair: str, qty: float, limit_price: float) -> str:
    pair = normalize_pair(pair)
    payload = {
        "pair": pair,
        "type": "buy",
        "ordertype": "limit",
        "price": f"{Decimal(limit_price):f}",
        "volume": f"{Decimal(qty):f}",
        "timeinforce": "IOC",
        "userref": int(time.time()),
    }
    resp = _private_request("AddOrder", payload)
    if resp.get("error"): print("LIMIT BUY error:", resp)
    else: print("LIMIT BUY ok:", resp)
    return json.dumps(resp)

def place_market_sell_qty(pair: str, qty: float) -> str:
    pair = normalize_pair(pair)
    payload = {
        "pair": pair,
        "type": "sell",
        "ordertype": "market",
        "volume": f"{Decimal(qty):f}",
        "userref": int(time.time()),
    }
    resp = _private_request("AddOrder", payload)
    if resp.get("error"): print("SELL error:", resp)
    else: print("SELL ok:", resp)
    return json.dumps(resp)
