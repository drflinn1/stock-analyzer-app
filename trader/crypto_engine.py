#!/usr/bin/env python3
"""
crypto_engine.py
Price + candidate utilities + LIVE order helpers for the rotation bot.

Public:
- load_candidates(csv_path=".state/momentum_candidates.csv") -> list[dict]
- get_public_quote(pair: str) -> float | None
- get_public_quotes(pairs: list[str]) -> dict[str, float]
- normalize_pair(s: str) -> str
- have_live_secrets() -> bool

LIVE helpers:
- place_market_buy_usd(pair, usd_amount, slip_pct=3.0, validate=False) -> (ok, msg, filled_price)
- place_market_sell_qty(pair, qty, validate=False) -> (ok, msg)
"""

from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

STATE_DIR = Path(".state")
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"

# --------- simple cache ----------
_QUOTES_CACHE: Dict[str, float] = {}
_CACHE_TTL = 30
_CACHE_STAMP: Dict[str, float] = {}

# ----------------------------- Kraken Public API -----------------------------

def _kraken_ticker(pair_code: str) -> Optional[float]:
    url = "https://api.kraken.com/0/public/Ticker"
    try:
        resp = requests.get(url, params={"pair": pair_code}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return None
        result = data.get("result", {})
        if not result:
            return None
        _k, v = next(iter(result.items()))
        last = v.get("c", [None])[0]
        return float(last) if last is not None else None
    except Exception:
        return None

def _to_kraken_code(pair: str) -> str:
    p = pair.strip().upper().replace(" ", "")
    if "/" in p:
        a, b = p.split("/", 1)
        return f"{a}{b}"
    return p

def normalize_pair(s: str) -> str:
    t = s.strip().upper().replace("-", "/")
    if "/" in t:
        a, b = t.split("/", 1)
        return f"{a}/{b}"
    if t.endswith("USD"):
        return f"{t[:-3]}/USD"
    return t

def _is_valid_price(x: Optional[float]) -> bool:
    if x is None:
        return False
    try:
        return x > 0 and x < 1e9
    except Exception:
        return False

def get_public_quote(pair: str) -> Optional[float]:
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
    out: Dict[str, float] = {}
    for p in pairs:
        pn = normalize_pair(p)
        q = get_public_quote(pn)
        if _is_valid_price(q):
            out[pn] = float(q)
    return out

# ----------------------------- Candidates Loader -----------------------------

def _parse_candidates_csv(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    if not csv_path.exists():
        return rows
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            hdr = {h.lower().strip(): h for h in reader.fieldnames or []}
            def get(row, key, default=""):
                for k in (key, key.lower(), key.upper()):
                    if k in row:
                        return row[k]
                if key in hdr:
                    return row.get(hdr[key], default)
                return default
            for row in reader:
                pair_raw = (get(row, "pair", "") or get(row, "symbol", "")).strip()
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
        return []
    return rows

def _fill_missing_quotes(rows: List[dict]) -> List[dict]:
    for r in rows:
        if not _is_valid_price(r.get("quote")):
            q = get_public_quote(r["pair"])
            if _is_valid_price(q):
                r["quote"] = float(q)
    return rows

def _best_effort_candidates(csv_path: Path) -> List[dict]:
    rows = _parse_candidates_csv(csv_path)
    if not rows:
        rows = [{"pair": "EAT/USD", "rank": 100, "quote": None}]
    rows = _fill_missing_quotes(rows)
    rows = [r for r in rows if _is_valid_price(r.get("quote"))]
    return rows

def load_candidates(csv_path: str | Path = CANDIDATES_CSV) -> List[dict]:
    path = Path(csv_path)
    rows = _best_effort_candidates(path)
    rows.sort(key=lambda r: (r.get("rank", 0), r["pair"]), reverse=True)
    return rows

# ----------------------------- LIVE Trading (Kraken) -----------------------------

def have_live_secrets() -> bool:
    return bool(os.getenv("KRAKEN_API_KEY", "")) and bool(os.getenv("KRAKEN_API_SECRET", ""))

def _kraken_sign(path: str, data: dict, secret_b64: str) -> str:
    # per Kraken docs: API-Sign = HMAC-SHA512(path + SHA256(nonce+postdata))
    secret = base64.b64decode(secret_b64)
    postdata = urlencode(data).encode()
    sha256 = hashlib.sha256((str(data['nonce']) + postdata.decode()).encode()).digest()
    message = (path.encode() + sha256)
    mac = hmac.new(secret, message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def _kraken_private(path: str, data: dict) -> dict:
    url = "https://api.kraken.com" + path
    key = os.getenv("KRAKEN_API_KEY", "")
    sec = os.getenv("KRAKEN_API_SECRET", "")
    if not key or not sec:
        return {"error": ["EAPI:MissingKeys"]}
    data = dict(data)
    data["nonce"] = int(time.time() * 1000)
    headers = {
        "API-Key": key,
        "API-Sign": _kraken_sign(path, data, sec),
        "User-Agent": "stock-analyzer-app/rotation-bot",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    resp = requests.post(url, headers=headers, data=urlencode(data), timeout=20)
    resp.raise_for_status()
    return resp.json()

def _limit_from_market_price(price: float, slip_pct: float, side: str) -> float:
    slip = abs(float(slip_pct))/100.0
    if side.lower() == "buy":
        return round(price * (1 + slip), 8)
    else:
        return round(price * (1 - slip), 8)

def place_market_buy_usd(pair: str, usd_amount: float, slip_pct: float = 3.0, validate: bool=False) -> Tuple[bool,str,Optional[float]]:
    """
    Uses Kraken AddOrder. For buy we use oflags=viqc (volume in quote currency = USD).
    Returns (ok, message, filled_price_if_known).
    """
    if not have_live_secrets():
        return False, "[LIVE] Missing Kraken API secrets; aborting order.", None

    pair_norm = normalize_pair(pair)
    code = _to_kraken_code(pair_norm)
    q = get_public_quote(pair_norm)
    if not _is_valid_price(q):
        return False, f"[LIVE] BUY {pair_norm}: invalid quote.", None

    # market attempt
    data = {
        "pair": code,
        "type": "buy",
        "ordertype": "market",
        "volume": f"{float(usd_amount):.2f}",   # with oflags=viqc this is USD
        "oflags": "viqc",
        "validate": "true" if validate else "false",
        "userref": int(time.time()),
    }
    try:
        res = _kraken_private("/0/private/AddOrder", data)
        if res.get("error"):
            # fallback: limit through book
            limit = _limit_from_market_price(float(q), slip_pct, "buy")
            data2 = {
                "pair": code, "type": "buy", "ordertype": "limit",
                "price": f"{limit:.8f}",
                "volume": f"{float(usd_amount):.2f}",
                "oflags": "viqc",
                "validate": "true" if validate else "false",
                "userref": int(time.time()),
            }
            res2 = _kraken_private("/0/private/AddOrder", data2)
            if res2.get("error"):
                return False, f"[LIVE] BUY {pair_norm} failed: {res} ; Fallback failed: {res2}", None
            return True, f"[LIVE] BUY {pair_norm} LIMIT @{limit} (viqc ${usd_amount:.2f}) ok. validate={validate}", q
        return True, f"[LIVE] BUY {pair_norm} MARKET (viqc ${usd_amount:.2f}) ok. validate={validate}", q
    except Exception as e:
        return False, f"[LIVE] BUY exception: {e}", None

def place_market_sell_qty(pair: str, qty: float, validate: bool=False) -> Tuple[bool,str]:
    """
    Market sell by base-asset quantity.
    """
    if not have_live_secrets():
        return False, "[LIVE] Missing Kraken API secrets; aborting order."
    pair_norm = normalize_pair(pair)
    code = _to_kraken_code(pair_norm)
    data = {
        "pair": code,
        "type": "sell",
        "ordertype": "market",
        "volume": f"{float(qty):.8f}",
        "validate": "true" if validate else "false",
        "userref": int(time.time()),
    }
    try:
        res = _kraken_private("/0/private/AddOrder", data)
        if res.get("error"):
            return False, f"[LIVE] SELL {pair_norm} failed: {res}"
        return True, f"[LIVE] SELL {pair_norm} MARKET qty={qty:.8f} ok. validate={validate}"
    except Exception as e:
        return False, f"[LIVE] SELL exception: {e}"
