#!/usr/bin/env python3
"""
main.py — 1-Coin Rotation
- SELL first, then immediate same-run re-entry if possible
- Flat cooldown override (COOLDOWN_MIN)
- Robust pair normalization (handles 'LSK/USD' vs 'LSKUSD')
- Kraken LIVE signer when DRY_RUN=OFF
"""

from __future__ import annotations

import base64, hashlib, hmac, json, os, sys, time, urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import requests

# ----- Repo helpers
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
COOLDOWN_MIN = int(parse_float_env("COOLDOWN_MIN", 10) or 10)
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
        txid_list = resp.get("result", {}).get("txid", [])
        txid = txid_list[0] if isinstance(txid_list, list) and txid_list else ""
        return {"status": "ok", "txid": txid, "raw": resp}
    except Exception as e:
        return {"status": "error", "note": f"Kraken order exception: {e}"}

# ---------- JSON helpers ----------
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

def pct_change(a: float, b: float) -> float:
    if a <= 0:
        return 0.0
    return (b - a) / a * 100.0

# ---------- Pair helpers (fix) ----------
def ensure_slash_pair(s: str, quote: str) -> str:
    """Ensure 'BASE/QUOTE' format given a string that might be 'BASEQUOTE'."""
    if "/" in s:
        return s
    if s.upper().endswith(quote):
        base = s[: -len(quote)]
        return f"{base}/{quote}"
    return f"{s}/{quote}"

def quote_forms(sym: str, quote: str) -> List[str]:
    """
    Return possible representations the quote fetcher might accept.
    e.g. 'LSK/USD' → ['LSK/USD','LSKUSD']
         'LSKUSD'  → ['LSK/USD','LSKUSD']
    """
    with_slash = ensure_slash_pair(sym, quote)
    no_slash = with_slash.replace("/", "")
    return [with_slash, no_slash]

# ---------- Core logic ----------
def pick_target_symbol() -> Optional[str]:
    print(f"[INFO] UNIVERSE_PICK={UNIVERSE_PICK}")
    if UNIVERSE_PICK.upper() == "AUTO":
        cands: List[Dict] = load_candidates(CANDIDATES_CSV)
        print(f"[INFO] candidates_csv_exists={CANDIDATES_CSV.exists()} rows={len(cands) if cands else 0}")
        if not cands:
            print("[HOLD] No candidates available.")
            return None
        raw = cands[0].get("symbol") or cands[0].get("pair") or cands[0].get("Symbol")
        print(f"[INFO] top candidate raw='{raw}'")
        if not raw:
            print("[HOLD] Candidate rows missing 'symbol' field.")
            return None
        norm = normalize_pair(raw)  # may return Kraken style (no slash)
        target = ensure_slash_pair(norm, QUOTE_CCY)
        print(f"[INFO] target normalized='{target}'")
        return target
    # explicit pick
    norm = normalize_pair(UNIVERSE_PICK)
    target = ensure_slash_pair(norm, QUOTE_CCY)
    print(f"[INFO] target explicit='{target}'")
    return target

def get_quote_required(sym: str) -> Optional[float]:
    # Try multiple forms until one returns a valid price
    for form in quote_forms(sym, QUOTE_CCY):
        px = get_public_quote(form)
        if px is not None and px > 0:
            print(f"[INFO] quote {form} ~ {px}")
            return px
        else:
            print(f"[INFO] quote miss for {form}")
    print(f"[HOLD] Invalid price for {sym}")
    return None

def should_sell(entry_px: float, last_px: float, high_px: float | None = None) -> Tuple[bool, str]:
    if TP_PCT and pct_change(entry_px, last_px) >= TP_PCT:
        return True, f"TP hit (≥ {TP_PCT:.2f}%)"
    if (STOP_PCT is not None) and pct_change(entry_px, last_px) <= -STOP_PCT:
        return True, f"STOP hit (≤ -{STOP_PCT:.2f}%)"
    if (TRAIL_START_PCT is not None) and (TRAIL_PCT is not None) and high_px:
        if pct_change(entry_px, high_px) >= TRAIL_START_PCT and pct_change(high_px, last_px) <= -TRAIL_PCT:
            return True, f"TRAIL hit (drop ≥ {TRAIL_PCT:.2f}% from high after +{TRAIL_START_PCT:.2f}%)"
    return False, "Hold"

def last_action_age_min() -> Optional[float]:
    s = read_json(SUMMARY_JSON, {})
    ts = s.get("last_action_time")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return (now_utc() - dt).total_seconds() / 60.0
    except Exception:
        return None

def rotation_window_ok() -> bool:
    age = last_action_age_min()
    if age is None:
        return True
    return age >= WINDOW_MIN

def cooldown_override_ok() -> bool:
    age = last_action_age_min()
    if age is None:
        return True
    return age >= COOLDOWN_MIN

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
    print(f"[TRADE] BUY attempt {sym} ${usd_amount} → vol≈{vol:.8f}")
    res = place_market_order(sym, "buy", vol, dry)
    return res | {"price": px, "volume": vol}

def market_sell(sym: str, qty: float, dry: bool) -> Dict:
    px = get_quote_required(sym)
    if px is None:
        return {"status": "error", "note": "No valid quote"}
    print(f"[TRADE] SELL attempt {sym} qty={qty:.8f}")
    res = place_market_order(sym, "sell", qty, dry)
    return res | {"price": px}

# ---------- Main ----------
def main():
    print(f"=== Start 1-Coin Rotation (DRY_RUN={DRY_RUN}) ===")
    print(f"TP={TP_PCT}%  STOP={STOP_PCT}%  TRAIL={TRAIL_PCT}%@{TRAIL_START_PCT}%  WINDOW_MIN={WINDOW_MIN}  COOLDOWN_MIN={COOLDOWN_MIN}  BUY_USD={BUY_USD}")

    pos_state = load_positions()
    positions = pos_state.get("positions", [])
    pos = positions[0] if positions else None

    sold_this_run = False

    # SELL FIRST
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

    # BUY (same-run re-entry; else window/cooldown)
    holding_now = load_positions().get("positions", [])
    if not holding_now and (len(holding_now) < MAX_POSITIONS):
        age = last_action_age_min()
        print(f"[INFO] last_action_age_min={None if age is None else round(age,2)}")
        allow_buy = sold_this_run or rotation_window_ok() or cooldown_override_ok()
        print(f"[INFO] allow_buy={allow_buy} (sold_this_run={sold_this_run}, window_ok={rotation_window_ok()}, cooldown_ok={cooldown_override_ok()})")
        if allow_buy:
            target = pick_target_symbol()
            if target:
                resp = market_buy(target, BUY_USD, DRY_RUN == "ON")
                if resp.get("status") in ("dry", "ok"):
                    px = resp.get("price")
                    qty = resp.get("volume", BUY_USD / (px or 1))
                    pos = {"symbol": target, "entry_price": px, "qty": qty, "entry_time": now_utc().isoformat(), "max_price": px}
                    save_positions({"positions": [pos]})
                    mark_action_time()
                    print(f"[TRADE] {'Re-entry: ' if sold_this_run else ''}BUY ok {target} qty={qty:.8f} @~{px}")
                else:
                    print(f"[ERROR] BUY failed: {resp}")
            else:
                print("[HOLD] No target symbol resolved.")
        else:
            print(f"[HOLD] Not buying yet (window/cooldown). WINDOW_MIN={WINDOW_MIN} COOLDOWN_MIN={COOLDOWN_MIN}")
    else:
        if holding_now:
            print("[STATE] Already holding; no new BUY.")
        else:
            print("[HOLD] Max positions reached; no new BUY.")

    # Summary
    out = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": DRY_RUN,
        "tp_pct": TP_PCT,
        "stop_pct": STOP_PCT,
        "trail_start_pct": TRAIL_START_PCT,
        "trail_pct": TRAIL_PCT,
        "window_min": WINDOW_MIN,
        "cooldown_min": COOLDOWN_MIN,
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
