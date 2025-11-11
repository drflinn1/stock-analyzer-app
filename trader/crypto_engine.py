#!/usr/bin/env python3
"""
trader/crypto_engine.py — Kraken utilities (LIVE)
Fixes:
• Market BUY returns txid; we then QueryOrders to capture exact filled qty (vol_exec).
• SELL clips quantity by a tiny epsilon to avoid "Insufficient funds" on dust.
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
        for row in r:
            sym = (row.get("symbol") or row.get("pair") or "").strip()
            q = (row.get("quote") or "").strip()
            if not sym:
                continue
            rows.append({"symbol": sym, "quote": q, "rank": row.get("rank")})
    return rows

# ------------------------- Symbol helpers --------------------------
def normalize_pair(s: str) -> str:
    s = s.strip().upper().replace("/", "")
    if s.endswith("USD") or s.endswith("EUR") or s.endswith("USDT"):
        return s
    return s + "USD"

def is_usd_pair(pair: str) -> bool:
    return pair.endswith("USD")

# --------------------------- Quotes --------------------------------
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
    LIVE MARKET BUY spending <usd_amount> of quote (USD).
    Use oflags=viqc; volume=<USD amount>.
    Returns the primary txid string.
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
        return json.dumps(resp)

    print("BUY ok:", resp)
    txids = resp.get("result", {}).get("txid") or []
    txid = txids[0] if txids else ""
    return txid or json.dumps(resp)

def query_order_filled_qty(txid: str, attempts: int = 6, sleep_s: float = 1.0) -> Optional[float]:
    """
    Poll QueryOrders for exact filled volume (vol_exec).
    Returns float qty or None if unknown.
    """
    if not txid or txid.startswith("{"):  # not a plain txid, probably raw json
        return None
    for _ in range(attempts):
        resp = _private_request("QueryOrders", {"txid": txid})
        if resp.get("error"):
            time.sleep(sleep_s)
            continue
        res = resp.get("result") or {}
        od = res.get(txid) or {}
        vol_exec = od.get("vol_exec")
        try:
            v = float(vol_exec)
            if v > 0:
                return v
        except Exception:
            pass
        time.sleep(sleep_s)
    return None

def place_market_sell_qty(pair: str, qty: float) -> str:
    """
    LIVE MARKET SELL by base-asset quantity.
    We sell a hair less than recorded to avoid 'Insufficient funds' due to dust/fees.
    """
    pair = normalize_pair(pair)
    safe_qty = max(0.0, qty * 0.999)  # tiny epsilon
    payload = {
        "pair": pair,
        "type": "sell",
        "ordertype": "market",
        "volume": f"{Decimal(safe_qty):f}",
        "userref": int(time.time()),
    }
    resp = _private_request("AddOrder", payload)
    if resp.get("error"):
        print("SELL error:", resp)
    else:
        print("SELL ok:", resp)
    return json.dumps(resp)
