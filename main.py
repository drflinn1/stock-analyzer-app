#!/usr/bin/env python3
"""
main.py — Crypto 1-Coin Rotation (safe HOLD on quote miss)

Requirements:
- trader/crypto_engine.py (with normalize_pair() and safe_quote())
- ENV secrets: KRAKEN_KEY, KRAKEN_SECRET (for live)
- ENV vars: DRY_RUN, BUY_USD, TP_PCT, STOP_PCT, WINDOW_MIN, ROTATE_WHEN_FULL, UNIVERSE_PICK

Behavior:
- If we can't fetch a valid quote for the target coin, HOLD the current position.
- Only sell first when: TP/STOP hit OR we have a confirmed target quote for rotation.
- Uses Kraken private AddOrder with `oflags=viqc` for USD-based market buys.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from trader.crypto_engine import normalize_pair, safe_quote, load_candidates

STATE_DIR = Path(".state")
POS_FILE = STATE_DIR / "positions.json"
SUMMARY_FILE = STATE_DIR / "run_summary.json"

KRAKEN_API = "https://api.kraken.com/0"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "rotation-bot/1.0"})

# ------------ ENV ------------
def get_env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None else default

DRY_RUN = get_env("DRY_RUN", "OFF").upper()  # OFF or ON
BUY_USD = float(get_env("BUY_USD", "30"))
TP_PCT = float(get_env("TP_PCT", "8"))
STOP_PCT = float(get_env("STOP_PCT", "4"))
WINDOW_MIN = int(float(get_env("WINDOW_MIN", "15")))
ROTATE_WHEN_FULL = get_env("ROTATE_WHEN_FULL", "true").lower() == "true"
UNIVERSE_PICK = get_env("UNIVERSE_PICK", "").strip().upper()

KRAKEN_KEY = os.environ.get("KRAKEN_KEY", "")
KRAKEN_SECRET = os.environ.get("KRAKEN_SECRET", "")

# ------------ UTIL ------------
def now_ts() -> int:
    return int(time.time())

def log(s: str) -> None:
    print(s, flush=True)

def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)

# ------------ KRAKEN PRIVATE API ------------
def _kraken_sign(uri_path: str, data: Dict[str, str]) -> str:
    # https://docs.kraken.com/rest/#section/Authentication
    secret = base64.b64decode(KRAKEN_SECRET)
    post_data = "&".join(f"{k}={v}" for k, v in data.items())
    sha = hashlib.sha256((data["nonce"] + post_data).encode()).digest()
    msg = uri_path.encode() + sha
    mac = hmac.new(secret, msg, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def kraken_private(endpoint: str, data: Dict[str, str]) -> dict:
    if not KRAKEN_KEY or not KRAKEN_SECRET:
        raise RuntimeError("Missing KRAKEN_KEY/KRAKEN_SECRET")
    uri_path = f"/private/{endpoint}"
    url = f"{KRAKEN_API}{uri_path}"
    data = dict(data)
    data["nonce"] = str(int(time.time() * 1000))
    headers = {
        "API-Key": KRAKEN_KEY,
        "API-Sign": _kraken_sign(uri_path, data),
    }
    r = SESSION.post(url, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(f"Kraken error: {j['error']}")
    return j.get("result", {})

def kraken_balance() -> Dict[str, float]:
    if DRY_RUN == "ON":
        return {}
    res = kraken_private("Balance", {})
    out = {}
    for k, v in res.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

def asset_code(base: str) -> str:
    # Kraken sometimes uses 'X'/'Z' prefixes internally; this helper keeps it simple.
    # For LSK, SOL, BTC etc this is fine.
    return base

def market_buy_quote_usd(pair_ws: str, usd_amount: float) -> str:
    """
    Market buy spending USD (uses viqc).
    Returns order txid.
    """
    base = pair_ws.split("/")[0]
    paircode = f"{base}USD"  # e.g., LSKUSD
    data = {
        "pair": paircode,
        "type": "buy",
        "ordertype": "market",
        "oflags": "viqc",  # spend quote currency
        "volume": f"{usd_amount:.2f}",  # USD spend
    }
    if DRY_RUN == "ON":
        log(f"[DRY] BUY {pair_ws} spending ${usd_amount:.2f}")
        return "DRY_BUY"
    res = kraken_private("AddOrder", data)
    txid = ",".join(res.get("txid", [])) or "UNKNOWN"
    log(f"[LIVE] BUY ok {pair_ws} ${usd_amount:.2f} (txid {txid})")
    return txid

def market_sell_base_all(pair_ws: str, base_amount: float) -> str:
    """
    Market sell base units.
    Returns order txid.
    """
    base = pair_ws.split("/")[0]
    paircode = f"{base}USD"
    data = {
        "pair": paircode,
        "type": "sell",
        "ordertype": "market",
        "volume": f"{base_amount:.10f}",
    }
    if DRY_RUN == "ON":
        log(f"[DRY] SELL {pair_ws} qty={base_amount}")
        return "DRY_SELL"
    res = kraken_private("AddOrder", data)
    txid = ",".join(res.get("txid", [])) or "UNKNOWN"
    log(f"[LIVE] SELL ok {pair_ws} qty={base_amount} (txid {txid})")
    return txid

# ------------ POSITION STATE ------------
def read_position() -> Optional[dict]:
    pos = read_json(POS_FILE, {})
    return pos or None

def write_position(pos: Optional[dict]) -> None:
    if pos is None:
        write_json(POS_FILE, {})
    else:
        write_json(POS_FILE, pos)

# ------------ STRATEGY HELPERS ------------
def pick_target_symbol(candidates: list[dict]) -> Optional[str]:
    if UNIVERSE_PICK:
        sym = normalize_pair(UNIVERSE_PICK)
        return sym
    # candidates already normalized by crypto_engine.load_candidates
    for row in candidates:
        sym = row.get("symbol", "")
        if not sym:
            continue
        return normalize_pair(sym)
    return None

def get_lot_size(base: str, balance: Dict[str, float]) -> float:
    # Use on-chain balance; if not available (DRY), assume we just bought ~ BUY_USD / price
    code = asset_code(base)
    return balance.get(code, 0.0)

def pct_change(cur: float, entry: float) -> float:
    if entry <= 0:
        return 0.0
    return (cur - entry) / entry * 100.0

# ------------ RUN ------------
def main() -> int:
    STATE_DIR.mkdir(exist_ok=True)

    # Load state
    pos = read_position()  # {'symbol','entry','qty','cost','ts'}
    candidates = load_candidates()
    target_sym = pick_target_symbol(candidates)

    # Quotes
    cur_sym = pos["symbol"] if pos else None
    cur_price = safe_quote(cur_sym) if cur_sym else 0.0
    tgt_price = safe_quote(target_sym) if target_sym else 0.0

    # Summary shell
    summary = {
        "ts": now_ts(),
        "dry_run": DRY_RUN,
        "window_min": WINDOW_MIN,
        "tp_pct": TP_PCT,
        "stop_pct": STOP_PCT,
        "rotate_when_full": ROTATE_WHEN_FULL,
        "universe_pick": UNIVERSE_PICK,
        "current": {"symbol": cur_sym, "price": cur_price},
        "target": {"symbol": target_sym, "price": tgt_price},
        "action": "HOLD",
        "note": "",
    }

    # If no current position: try to BUY target only if we have a real quote
    if not pos:
        if target_sym and tgt_price > 0:
            txid = market_buy_quote_usd(target_sym, BUY_USD)
            entry = tgt_price
            qty = 0.0
            if DRY_RUN == "ON":
                qty = BUY_USD / entry
            else:
                # in live we can't get exact qty from AddOrder response easily; approximate
                qty = BUY_USD / entry
            pos = {
                "symbol": target_sym,
                "entry": entry,
                "qty": qty,
                "cost": BUY_USD,
                "ts": now_ts(),
            }
            write_position(pos)
            summary.update({"action": "BUY", "note": f"Opened {target_sym} using ${BUY_USD:.2f}"})
        else:
            summary.update({"action": "HOLD", "note": "No valid target quote; staying in USD."})
            write_position(None)
        write_json(SUMMARY_FILE, summary)
        log(json.dumps(summary, indent=2))
        return 0

    # With a current position: evaluate TP/STOP
    entry = float(pos.get("entry", 0))
    qty = float(pos.get("qty", 0))
    cur_sym = pos["symbol"]
    summary["current"]["symbol"] = cur_sym
    summary["current"]["price"] = cur_price

    if cur_price <= 0:
        summary.update({"action": "HOLD", "note": f"Quote miss for {cur_sym}; hold."})
        write_json(SUMMARY_FILE, summary)
        log(json.dumps(summary, indent=2))
        return 0

    chg = pct_change(cur_price, entry)
    # TAKE PROFIT
    if chg >= TP_PCT:
        txid = market_sell_base_all(cur_sym, qty)
        write_position(None)
        summary.update({"action": "SELL_TP", "note": f"TP hit {chg:.2f}% -> sold {cur_sym}"})
        write_json(SUMMARY_FILE, summary)
        log(json.dumps(summary, indent=2))
        return 0

    # STOP LOSS
    if chg <= -abs(STOP_PCT):
        txid = market_sell_base_all(cur_sym, qty)
        write_position(None)
        summary.update({"action": "SELL_SL", "note": f"STOP hit {chg:.2f}% -> sold {cur_sym}"})
        write_json(SUMMARY_FILE, summary)
        log(json.dumps(summary, indent=2))
        return 0

    # ROTATION (only if allowed and we actually have a target with a valid quote)
    if ROTATE_WHEN_FULL:
        if target_sym and normalize_pair(target_sym) != normalize_pair(cur_sym):
            if tgt_price > 0:
                # We have a valid target quote → rotate: sell current then buy target
                sell_tx = market_sell_base_all(cur_sym, qty)
                buy_tx = market_buy_quote_usd(target_sym, BUY_USD)
                new_pos = {
                    "symbol": target_sym,
                    "entry": tgt_price,
                    "qty": BUY_USD / tgt_price if tgt_price > 0 else 0.0,
                    "cost": BUY_USD,
                    "ts": now_ts(),
                }
                write_position(new_pos)
                summary.update({"action": "ROTATE", "note": f"{cur_sym} -> {target_sym}"})
                write_json(SUMMARY_FILE, summary)
                log(json.dumps(summary, indent=2))
                return 0
            else:
                summary.update({"action": "HOLD", "note": f"Target {target_sym} quote miss; hold {cur_sym}."})
                write_json(SUMMARY_FILE, summary)
                log(json.dumps(summary, indent=2))
                return 0

    # Default: HOLD
    summary.update({"action": "HOLD", "note": f"Holding {cur_sym} ({chg:.2f}% vs entry)."})
    write_json(SUMMARY_FILE, summary)
    log(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
