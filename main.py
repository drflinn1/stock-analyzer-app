#!/usr/bin/env python3
"""
Crypto — 1-Coin Rotation (LIVE-ready)
Implements:
  • 1% stop during first HOLD_WINDOW_M minutes
  • 5% take-profit
  • If not ≥ SLOW_GAIN_PCT after HOLD_WINDOW_M, exit and rotate
  • Then buy top gainer (USD pairs) on Kraken
Also supports:
  • FORCE_SELL=ALL or symbol → uses tools/force_sell.py
  • Always writes .state/run_summary.md and .state/run_summary.json
No external deps; uses minimal Kraken REST calls.

ENV (typical):
  DRY_RUN=ON|OFF
  BUY_USD=15
  TP_PCT=5
  SL_PCT=1
  HOLD_WINDOW_M=60
  SLOW_GAIN_PCT=3
  FORCE_SELL="" | "ALL" | "SOON" | "SOON/USD"
  SLIP_PCT=3.0        # used only with FORCE_SELL
  KRAKEN_KEY / KRAKEN_SECRET  (required for LIVE)

Files:
  .state/positions.json   -> {"symbol":"UAI","qty":60.0,"entry_price":0.25,"entry_ts":1700000000}
  .state/run_summary.md
  .state/run_summary.json
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from urllib.parse import urlencode
import urllib.request

# ---------- State paths ----------
STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
POSITIONS = STATE_DIR / "positions.json"
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"
LAST_OK = STATE_DIR / "last_ok.txt"

API_URL = "https://api.kraken.com"

# ---------- Helpers ----------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return default

def now_utc() -> int:
    return int(time.time())

def md_append(lines: List[str]) -> None:
    if not SUMMARY_MD.exists():
        SUMMARY_MD.write_text("# Run Summary\n\n", encoding="utf-8")
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    block = ["", f"### Bot Run — {ts}", ""]
    block.extend(f"- {ln}" for ln in lines)
    block.append("")
    with SUMMARY_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))

def write_json_summary(data: Dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _kraken_request(uri_path: str, data: Dict = None, key: str = "", secret: str = "") -> Dict:
    """Minimal Kraken REST client (no external libs)."""
    data = data or {}
    postdata = urlencode(data).encode()

    if uri_path.startswith("/0/private/"):
        if not key or not secret:
            raise RuntimeError("[LIVE] Missing Kraken API credentials.")
        nonce = str(int(1000 * time.time()))
        data.update({"nonce": nonce})
        postdata = urlencode(data).encode()

        sha256 = hashlib.sha256((nonce + urlencode(data)).encode()).digest()
        hmac_data = uri_path.encode() + sha256
        mac = hmac.new(base64.b64decode(secret), hmac_data, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())

        headers = {
            "API-Key": key,
            "API-Sign": sigdigest.decode(),
            "User-Agent": "rotation-bot/1.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        req = urllib.request.Request(API_URL + uri_path, data=postdata, headers=headers)
    else:
        req = urllib.request.Request(API_URL + uri_path, headers={"User-Agent": "rotation-bot/1.0"})

    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        try:
            return json.loads(raw.decode())
        except Exception:
            return {"raw": raw.decode()}

def usd_pairs_map() -> Dict[str, str]:
    """Return {BASE: ALTNAME} for USD pairs (e.g., {'SOON':'SOONUSD'})."""
    res = _kraken_request("/0/public/AssetPairs")
    out = {}
    if "result" in res:
        for _, info in res["result"].items():
            alt = info.get("altname", "")
            if alt.endswith("USD"):
                out[alt[:-3]] = alt
    return out

def ticker_for(pair_alt: str) -> Dict[str, Any]:
    res = _kraken_request("/0/public/Ticker?pair=" + pair_alt)
    if "result" not in res:
        return {}
    first = next(iter(res["result"]))
    return res["result"][first]

def best_bid_ask(pair_alt: str) -> Tuple[float, float]:
    t = ticker_for(pair_alt)
    if not t:
        return (0.0, 0.0)
    bid = float(t["b"][0])
    ask = float(t["a"][0])
    return bid, ask

def last_price_and_open(pair_alt: str) -> Tuple[float, float]:
    t = ticker_for(pair_alt)
    if not t:
        return (0.0, 0.0)
    last = float(t["c"][0])
    openp = float(t["o"])
    return last, openp

def account_balances(key: str, secret: str) -> Dict[str, float]:
    res = _kraken_request("/0/private/Balance", {}, key, secret)
    if res.get("error"):
        raise RuntimeError(f"[ERROR] Balance: {res['error']}")
    raw = res.get("result", {})
    return {k.replace("Z","").replace("X",""): float(v) for k, v in raw.items()}

def add_order_market(pair_alt: str, side: str, volume: float, key: str, secret: str) -> Dict:
    data = {
        "ordertype": "market",
        "type": side,  # "buy" or "sell"
        "pair": pair_alt,
        "volume": f"{volume:.8f}",
    }
    return _kraken_request("/0/private/AddOrder", data, key, secret)

# ---------- Position I/O ----------
def read_position() -> Dict[str, Any]:
    if not POSITIONS.exists():
        return {}
    try:
        return json.loads(POSITIONS.read_text())
    except Exception:
        return {}

def write_position(pos: Dict[str, Any]) -> None:
    POSITIONS.write_text(json.dumps(pos, indent=2), encoding="utf-8")

def clear_position() -> None:
    POSITIONS.write_text("{}", encoding="utf-8")

# ---------- Gainer scan ----------
def pick_top_gainer(pairs_map: Dict[str, str], blocklist: List[str] = None) -> str:
    """
    Returns best BASE symbol by 24h % change among USD pairs.
    Simple heuristic: sort by (last/open - 1).
    """
    block = set((blocklist or []))
    best_base = ""
    best_change = -999.0
    checked = 0

    for base, alt in pairs_map.items():
        if base in block:
            continue
        last, openp = last_price_and_open(alt)
        if last <= 0 or openp <= 0:
            continue
        change = (last / openp - 1.0) * 100.0
        checked += 1
        if change > best_change:
            best_change = change
            best_base = base

    md_append([f"Gainer scan checked ~{checked} USD pairs. Top: {best_base} ({best_change:.2f}%)."])
    return best_base

# ---------- Trading rules ----------
def should_sell(pnl_pct: float, mins_held: float, sl_pct: float, tp_pct: float, slow_pct: float, window_m: float) -> Tuple[bool, str]:
    # 1) stop inside window
    if mins_held <= window_m and pnl_pct <= -sl_pct:
        return True, f"STOP: pnl {pnl_pct:.2f}% ≤ -{sl_pct}% within first {window_m}m"
    # 2) take profit
    if pnl_pct >= tp_pct:
        return True, f"TAKE-PROFIT: pnl {pnl_pct:.2f}% ≥ {tp_pct}%"
    # 3) slow exit after window
    if mins_held >= window_m and pnl_pct < slow_pct:
        return True, f"SLOW-EXIT: after {window_m}m pnl {pnl_pct:.2f}% < {slow_pct}%"
    return False, ""

# ---------- FORCE_SELL integration ----------
def maybe_force_sell_and_exit():
    force = env_str("FORCE_SELL", "")
    if not force:
        return
    slip = env_str("SLIP_PCT", "3.0") or "3.0"
    md_append([f"FORCE_SELL invoked: symbol='{force}', slip='{slip} %'"])
    env = os.environ.copy()
    env["INPUT_SYMBOL"] = force
    env["INPUT_SLIP"] = slip
    # Reuse the tool so behavior matches one-time workflow.
    code = os.system(f"{sys.executable} tools/force_sell.py")  # simple passthrough
    md_append([f"FORCE_SELL completed with code={code}"])
    sys.exit(0)

# ---------- Main ----------
def main():
    # Always ensure summary exists
    if not SUMMARY_MD.exists():
        SUMMARY_MD.write_text("# Run Summary\n\n", encoding="utf-8")

    maybe_force_sell_and_exit()

    dry = env_str("DRY_RUN", "ON").upper() != "OFF"
    buy_usd = env_float("BUY_USD", 15.0)
    tp_pct = env_float("TP_PCT", 5.0)
    sl_pct = env_float("SL_PCT", 1.0)
    window_m = env_float("HOLD_WINDOW_M", 60.0)
    slow_pct = env_float("SLOW_GAIN_PCT", 3.0)

    pairs = usd_pairs_map()
    key = env_str("KRAKEN_KEY", "")
    secret = env_str("KRAKEN_SECRET", "")

    # Load current position (if any)
    pos = read_position()
    lines = [f"Mode={'DRY' if dry else 'LIVE'} BUY_USD={buy_usd} TP={tp_pct}% SL={sl_pct}% WINDOW={window_m}m SLOW={slow_pct}%"]

    if pos.get("symbol"):
        base = pos["symbol"].upper().replace("/", "")
        if base.endswith("USD"):
            base = base[:-3]
        if base not in pairs:
            lines.append(f"Position base {base} has no USD pair; skipping.")
            md_append(lines)
            write_json_summary({"mode": "hold-skip", "details": lines})
            LAST_OK.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
            return

        pair_alt = pairs[base]
        last, _open = last_price_and_open(pair_alt)
        entry_price = float(pos["entry_price"])
        qty = float(pos["qty"])
        mins_held = (now_utc() - int(pos["entry_ts"])) / 60.0
        pnl_pct = (last / entry_price - 1.0) * 100.0

        lines.append(f"Holding {qty:.6f} {base} @ {entry_price:.6f}; now {last:.6f}; held {mins_held:.1f}m; pnl {pnl_pct:.2f}%")

        sell, reason = should_sell(pnl_pct, mins_held, sl_pct, tp_pct, slow_pct, window_m)
        if sell:
            if dry:
                lines.append(f"[DRY] SELL {base}/USD  — {reason}")
                clear_position()
            else:
                res = add_order_market(pair_alt, "sell", qty, key, secret)
                if res.get("error"):
                    lines.append(f"[LIVE] SELL ERROR: {res['error']}")
                else:
                    lines.append(f"[LIVE] SELL OK: {base}/USD — {reason}")
                    clear_position()
            # after selling we will try to buy a new gainer below
        else:
            lines.append("Rules: HOLD (no sell this run).")
            md_append(lines)
            write_json_summary({"mode": "hold", "details": lines, "pos": pos})
            LAST_OK.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
            return  # still holding; wait for next run

    # If we’re here, there’s no position (either none, or we just sold)
    top = pick_top_gainer(pairs, blocklist=["USDT","ZUSD","USD"])  # basic sanity block
    if not top:
        lines.append("No gainer candidate found.")
        md_append(lines)
        write_json_summary({"mode": "idle-no-candidate", "details": lines})
        LAST_OK.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        return

    pair_alt = pairs[top]
    bid, ask = best_bid_ask(pair_alt)
    if ask <= 0:
        lines.append(f"Bad price for {top}/USD; skipping.")
        md_append(lines)
        write_json_summary({"mode": "idle-badprice", "details": lines})
        LAST_OK.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        return

    qty = max( (buy_usd / ask) * 0.999, 1e-7 )  # shave for fees
    if dry:
        lines.append(f"[DRY] BUY {top}/USD ~ ${buy_usd:.2f} (qty≈{qty:.6f} at ask {ask:.6f})")
        write_position({"symbol": top, "qty": qty, "entry_price": ask, "entry_ts": now_utc()})
    else:
        res = add_order_market(pair_alt, "buy", qty, key, secret)
        if res.get("error"):
            lines.append(f"[LIVE] BUY ERROR: {res['error']}")
            md_append(lines)
            write_json_summary({"mode": "buy-error", "details": lines})
            LAST_OK.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
            return
        lines.append(f"[LIVE] BUY OK: {top}/USD qty≈{qty:.6f} at ~{ask:.6f}")
        write_position({"symbol": top, "qty": qty, "entry_price": ask, "entry_ts": now_utc()})

    md_append(lines)
    write_json_summary({"mode": "buy", "details": lines})
    LAST_OK.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

if __name__ == "__main__":
    main()
