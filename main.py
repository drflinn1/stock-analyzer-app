#!/usr/bin/env python3
"""
main.py — 1-Coin Rotation (immediate re-entry after sell) + LIVE Kraken signer
- Re-buys immediately after a SELL in the same run (if candidate + quote ok)
- WINDOW_MIN still applies when we didn't sell this run
- Optional STOP/TRAIL handled safely even when blank
"""

from __future__ import annotations

import base64, hashlib, hmac, json, os, sys, time, urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import requests

# ----- Local helpers from your repo
try:
    from trader.crypto_engine import (
        CANDIDATES_CSV,
        STATE_DIR,
        load_candidates,
        get_public_quote,
        normalize_pair,
    )
except Exception as e:
    print(f"[FATAL] Could not import trader.crypto_engine: {e}", file=sys.stderr)
    sys.exit(1)

# ---------- ENV helpers ----------
def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None and str(v).strip() != "" else default

def parse_float_env(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "" or raw.lower() in ("none", "null", "nan"):
        return default
    try:
        return float(raw)
    except Exception:
        return default

DRY_RUN = env("DRY_RUN", "ON").upper()
TP_PCT = parse_float_env("TP_PCT", 8.0) or 8.0
STOP_PCT = parse_float_env("STOP_PCT", None)
TRAIL_START_PCT = parse_float_env("TRAIL_START_PCT", None)
TRAIL_PCT = parse_float_env("TRAIL_PCT", None)
WINDOW_MIN = int(parse_float_env("WINDOW_MIN", 30) or 30)
BUY_USD = float(parse_float_env("BUY_USD", 30.0) or 30.0)
MAX_POSITIONS = int(parse_float_env("MAX_POSITIONS", 1) or 1)
UNIVERSE_PICK = env("UNIVERSE_PICK", "AUTO").strip()
QUOTE_CCY = env("QUOTE_CCY", "USD").upper()

KRAKEN_KEY = env("KRAKEN_API_KEY")
KRAKEN_SECRET = env("KRAKEN_API_SECRET")

STATE_DIR.mkdir(parents=True, exist_ok=True)
POSITIONS_JSON = STATE_DIR / "positions.json"
SUMMARY_JSON = STATE_DIR / "run_summary.json"
LAST_FLAG = STATE_DIR / "this.flag"

API_BASE = "https://api.kraken.com/0"

# ---------- Kraken signing ----------
def kraken_time() -> float:
    try:
        r = requests.get(f"{API_BASE}/public/Time", timeout=10)
        r.raise_for_status()
        return float(r.json()["result"]["unixtime"])
    except Exception:
        return time.time()

def _kraken_sign(urlpath: str, data: Dict, secret_b64: str) -> str:
    postdata = urllib.parse.urlencode(data)
    sha256 = hashlib.sha256((str(data["nonce"]) + postdata).encode()).digest()
    mac = hmac.new(base64.b64decode(secret_b64), urlpath.encode() + sha256, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def kraken_private(path: str, data: Dict) -> Dict:
    if not KRAKEN_KEY or not KRAKEN_SECRET:
        return {"error": ["EGeneral:Missing api key/secret"], "result": {}}
    urlpath = f"/0/private/{path}"
    url = f"{API_BASE}/private/{path}"
    data = {**data, "nonce": int(time.time() * 1000)}
    headers = {
        "API-Key": KRAKEN_KEY,
        "API-Sign": _kraken_sign(urlpath, data, KRAKEN_SECRET),
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "User-Agent": "rotation-bot",
    }
    r = requests.post(url, headers=headers, data=urllib.parse.urlencode(data), timeout=20)
    r.raise_for_status()
    return r.json()

def pair_to_kraken(sym: str) -> str:
    return sym.replace("/", "")

def place_market_order(pair: str, side: str, volume: float, dry_run: bool) -> Dict:
    if dry_run or DRY_RUN == "ON":
        return {"status": "dry", "pair": pair, "side": side, "volume": volume, "txid": "SIMULATED"}
    try:
        resp = kraken_private("AddOrder", {
            "pair": pair_to_kraken(pair),
            "type": side,
            "ordertype": "market",
            "volume": f"{volume:.8f}",
            "oflags": "viqc",
            "validate": False,
        })
        if resp.get("error"):
            return {"status": "error", "note": ",".join(resp["error"]), "raw": resp}
        txid = ""
        if isinstance(resp.get("result", {}).get("txid"), list) and resp["result"]["txid"]:
            txid = resp["result"]["txid"][0]
        return {"status": "ok", "txid": txid, "raw": resp}
    except Exception as e:
        return {"status": "error", "note": f"Kraken order exception: {e}"}

# ---------- JSON state helpers ----------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
    return default

def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)

def load_positions() -> Dict:
    return read_json(POSITIONS_JSON, {"positions": []})

def save_positions(positions: Dict) -> None:
    write_json(POSITIONS_JSON, positions)

def pct_change(from_px: float, to_px: float) -> float:
    if from_px <= 0:
        return 0.0
    return (to_px - from_px) / from_px * 100.0

# ---------- Core logic ----------
def pick_target_symbol() -> Optional[str]:
    if UNIVERSE_PICK.upper() == "AUTO":
        cands: List[Dict] = load_candidates(CANDIDATES_CSV)  # pass Path
        if not cands:
            print("[HOLD] No candidates available.")
            return None
        sym = cands[0].get("symbol") or cands[0].get("pair") or cands[0].get("Symbol")
        if not sym:
            print("[HOLD] Candidate rows missing 'symbol' field.")
            return None
        sym = normalize_pair(sym)
        if "/" not in sym:
            sym = f"{sym}/{QUOTE_CCY}"
        return sym
    pick = normalize_pair(UNIVERSE_PICK)
    if "/" not in pick:
        pick = f"{pick}/{QUOTE_CCY}"
    return pick

def get_quote_required(sym: str) -> Optional[float]:
    px = get_public_quote(sym)
    if px is None or px <= 0:
        print(f"[HOLD] Invalid price for {sym}")
        return None
    return px

def can_open_new_position(positions: Dict) -> bool:
    return len(positions.get("positions", [])) < MAX_POSITIONS

def should_sell(entry_px: float, last_px: float, high_px: float | None = None) -> Tuple[bool, str]:
    if TP_PCT and pct_change(entry_px, last_px) >= TP_PCT:
        return True, f"TP hit (≥ {TP_PCT:.2f}%)"
    if (STOP_PCT is not None) and pct_change(entry_px, last_px) <= -STOP_PCT:
        return True, f"STOP hit (≤ -{STOP_PCT:.2f}%)"
    if (TRAIL_START_PCT is not None) and (TRAIL_PCT is not None) and high_px:
        if pct_change(entry_px, high_px) >= TRAIL_START_PCT and pct_change(high_px, last_px) <= -TRAIL_PCT:
            return True, f"TRAIL hit (drop ≥ {TRAIL_PCT:.2f}% from high after +{TRAIL_START_PCT:.2f}%)"
    return False, "Hold"

def rotation_window_ok() -> bool:
    summary = read_json(SUMMARY_JSON, {})
    t_str = summary.get("last_action_time")
    if not t_str:
        return True
    try:
        last_t = datetime.fromisoformat(t_str)
    except Exception:
        return True
    return (now_utc() - last_t) >= timedelta(minutes=WINDOW_MIN)

def mark_action_time() -> None:
    s = read_json(SUMMARY_JSON, {})
    s["last_action_time"] = now_utc().isoformat()
    write_json(SUMMARY_JSON, s)

def market_buy(sym: str, usd_amount: float, dry: bool) -> Dict:
    px = get_quote_required(sym)
    if px is None:
        return {"status": "error", "note": "No valid quote"}
    vol = usd_amount / px
    if usd_amount < 10.0 or vol <= 0:
        return {"status": "error", "note": "Buy notional too small (min ~$10)"}
    res = place_market_order(sym, "buy", vol, dry)
    return res | {"price": px, "volume": vol}

def market_sell(sym: str, qty: float, dry: bool) -> Dict:
    px = get_quote_required(sym)
    if px is None:
        return {"status": "error", "note": "No valid quote"}
    res = place_market_order(sym, "sell", qty, dry)
    return res | {"price": px}

def main():
    print(f"=== Start 1-Coin Rotation (DRY_RUN={DRY_RUN}) ===")
    print(f"TP={TP_PCT}%  STOP={STOP_PCT}%  TRAIL={TRAIL_PCT}%@{TRAIL_START_PCT}%  WINDOW_MIN={WINDOW_MIN}  BUY_USD={BUY_USD}")

    pos_state = load_positions()
    positions = pos_state.get("positions", [])
    pos = positions[0] if positions else None

    sold_this_run = False

    # ---- SELL FIRST ----
    if pos:
        sym = pos["symbol"]
        entry_px = float(pos["entry_price"])
        qty = float(pos["qty"])
        high = float(pos.get("max_price", entry_px))
        last_px = get_quote_required(sym)
        if last_px is None:
            print(f"[HOLD] {sym} no valid quote; skipping sell check.")
        else:
            if last_px > high:
                pos["max_price"] = last_px
                save_positions({"positions": [pos]})
            do_sell, reason = should_sell(entry_px, last_px, pos.get("max_price"))
            print(f"[EVAL] {sym} entry={entry_px:.6f} last={last_px:.6f} Δ={pct_change(entry_px,last_px):.2f}% → {reason}")
            if do_sell:
                resp = market_sell(sym, qty, DRY_RUN == "ON")
                if resp.get("status") in ("dry", "ok"):
                    print(f"[TRADE] SELL ok {sym} qty={qty:.8f} @~{resp.get('price')}")
                    save_positions({"positions": []})
                    mark_action_time()
                    sold_this_run = True
                else:
                    print(f"[ERROR] SELL failed: {resp}")

    # ---- BUY (immediate re-entry allowed if sold this run) ----
    holding_now = load_positions().get("positions", [])
    if not holding_now and can_open_new_position({"positions": []}):
        if sold_this_run or rotation_window_ok():
            target = pick_target_symbol()
            if target:
                px = get_quote_required(target)
                if px:
                    resp = market_buy(target, BUY_USD, DRY_RUN == "ON")
                    if resp.get("status") in ("dry", "ok"):
                        qty = resp.get("volume", BUY_USD / px)
                        pos = {
                            "symbol": target,
                            "entry_price": px,
                            "qty": qty,
                            "entry_time": now_utc().isoformat(),
                            "max_price": px,
                        }
                        save_positions({"positions": [pos]})
                        mark_action_time()
                        print(f"[TRADE] {'Re-entry: ' if sold_this_run else ''}BUY ok {target} qty={qty:.8f} @~{px}")
                    else:
                        print(f"[ERROR] BUY failed: {resp}")
                else:
                    print(f"[HOLD] No valid quote for {target}; cannot buy.")
            else:
                print("[HOLD] No target symbol resolved.")
        else:
            print(f"[HOLD] Rotation window active ({WINDOW_MIN}m); not opening new position yet.")
    else:
        if holding_now:
            print("[STATE] Already holding a position; no new BUY.")
        else:
            print("[HOLD] Max positions reached; no new BUY.")

    # ---- Summary ----
    out = {
        "time_utc": now_utc().isoformat(),
        "dry_run": DRY_RUN,
        "tp_pct": TP_PCT,
        "stop_pct": STOP_PCT,
        "trail_start_pct": TRAIL_START_PCT,
        "trail_pct": TRAIL_PCT,
        "window_min": WINDOW_MIN,
        "buy_usd": BUY_USD,
        "universe_pick": UNIVERSE_PICK,
        "quote_ccy": QUOTE_CCY,
        "positions": load_positions().get("positions", []),
        "candidates_csv_exists": CANDIDATES_CSV.exists(),
    }
    write_json(SUMMARY_JSON, out)
    LAST_FLAG.write_text(str(kraken_time()))
    print("=== End run ===")

if __name__ == "__main__":
    main()
