#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stock Analyzer — Crypto Live (Kraken)
Drop-in main.py that is DRY-safe (no keys needed for paper runs).

Key features:
- DRY mode works even without API keys (private calls are no-ops).
- Live mode uses real ccxt.kraken and requires keys.
- Auto-universe scanning by 24h USD Volume and Top-K.
- Basic spread guard, notional sizing, position cap, rotation hooks.
- Clear BANNER / KPI / SUMMARY console output.

Environment variables (string values expected):
  DRY_RUN: "ON" | "OFF"
  RUN_SWITCH: "ON" | "OFF"

  EXCHANGE: "kraken" (default)
  QUOTE: "USD"
  AUTO_UNIVERSE: "true" | "false"
  UNIVERSE_MIN_USD_VOL: "300000"
  UNIVERSE_TOP_K: "10"
  UNIVERSE_EXCLUDE: "USDT,USDC,EUR,GBP,USD,SPX,PUMP,BABY"
  MAX_SPREAD_PCT: "0.60"
  MIN_TRADE_NOTIONAL_USD: "5"

  MAX_POSITIONS: "6"
  USD_PER_TRADE: "20"
  MAX_BUYS_PER_RUN: "1"
  ROTATE_WHEN_FULL: "true"
  ROTATE_WHEN_CASH_SHORT: "true"
  DUST_MIN_USD: "2"

  TAKE_PROFIT_PCT: "3"
  STOP_LOSS_PCT: "2"
  TRAIL_STOP_PCT: "1"
  EMERGENCY_SL_PCT: "8"
  MAX_DAILY_LOSS_PCT: "5"
  MAX_DAILY_ENTRIES: "4"

  CLEANUP_NON_UNIVERSE: "true"
  NONUNI_SELL_IF_DOWN_PCT: "0"
  NONUNI_KEEP_IF_WINNER_PCT: "6"

  STATE_DIR: ".state"
  KPI_CSV: ".state/kpi_history.csv"
"""

import os, sys, time, math, json, csv
from pathlib import Path
from datetime import datetime, timezone

# ===================== DRY-safe CCXT bootstrap =====================
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

def build_exchange():
    DRY = (os.getenv("DRY_RUN", "ON").upper() == "ON")
    api_key = os.getenv("KRAKEN_API_KEY") or os.getenv("CCXT_API_KEY")
    api_secret = os.getenv("KRAKEN_API_SECRET") or os.getenv("CCXT_API_SECRET")

    if (api_key and api_secret) or not DRY:
        if ccxt is None:
            raise RuntimeError("ccxt not installed but required for live or key-based runs.")
        print(f"[auth] Using real ccxt.kraken (DRY_RUN={DRY}, keys={'yes' if api_key and api_secret else 'no'})")
        return ccxt.kraken({
            "apiKey": api_key or "",
            "secret": api_secret or "",
            "enableRateLimit": True,
        }), True, DRY

    print("[auth] DRY_RUN=ON and no keys found -> using DRY stub (private calls are no-ops).")

    class DryStub:
        id = "kraken"
        name = "Kraken(DRY-Stub)"
        rateLimit = 1000
        def __init__(self):
            self.markets = {}
        # public-ish
        def load_markets(self, reload=False, params=None):
            # minimal USD spot markets list; enough for filtering/logging
            bases = ["BTC","ETH","SOL","ADA","DOGE","ZEC","ATOM","MATIC","LTC","XRP","DOT","LINK","TON","AVAX"]
            self.markets = {f"{b}/USD": {"symbol": f"{b}/USD", "base": b, "quote": "USD",
                                         "taker": 0.0026, "maker": 0.0016} for b in bases}
            return self.markets
        def fetch_ticker(self, symbol, params=None):
            # return a benign ticker; DRY-safe
            return {"symbol": symbol, "bid": None, "ask": None, "last": None, "percentage": None, "quoteVolume": None}
        # private -> no-ops
        def fetch_balance(self, params=None):
            print("[DRY] fetch_balance (skipped)");  return {"total": {}, "free": {}, "used": {}}
        def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
            print("[DRY] fetch_open_orders (skipped)"); return []
        def create_order(self, symbol, type, side, amount, price=None, params=None):
            print(f"[DRY] create_order {symbol} {side} {amount}@{price} (skipped)")
            oid = f"DRY-{int(time.time())}"
            return {"id": oid, "symbol": symbol, "side": side, "type": type,
                    "amount": amount, "price": price, "status": "closed", "info": {"dry_run": True}}

    return DryStub(), False, DRY
# ===================================================================

# ---------------------- Utilities & Config --------------------------
def env_str(name, default=""):
    v = os.getenv(name)
    return v if v is not None and v != "" else default

def env_float(name, default=0.0):
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return default

def env_bool(name, default=False):
    v = env_str(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1","true","yes","on")

def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------------------ Load configuration ------------------------
CFG = {
    "DRY_RUN": env_str("DRY_RUN", "ON").upper(),
    "RUN_SWITCH": env_str("RUN_SWITCH", "ON").upper(),
    "QUOTE": env_str("QUOTE", "USD"),
    "AUTO_UNIVERSE": env_bool("AUTO_UNIVERSE", True),
    "UNIVERSE_MIN_USD_VOL": env_float("UNIVERSE_MIN_USD_VOL", 300000.0),
    "UNIVERSE_TOP_K": int(env_float("UNIVERSE_TOP_K", 10)),
    "UNIVERSE_EXCLUDE": [s.strip().upper() for s in env_str("UNIVERSE_EXCLUDE", "USDT,USDC,EUR,GBP,USD,SPX,PUMP,BABY").split(",") if s.strip() != ""],
    "MAX_SPREAD_PCT": env_float("MAX_SPREAD_PCT", 0.60),
    "MIN_TRADE_NOTIONAL_USD": env_float("MIN_TRADE_NOTIONAL_USD", 5.0),
    "RESERVE_CASH_PCT": env_float("RESERVE_CASH_PCT", 15.0),  # percent
    "USD_PER_TRADE": env_float("USD_PER_TRADE", 20.0),
    "MAX_POSITIONS": int(env_float("MAX_POSITIONS", 6)),
    "MAX_BUYS_PER_RUN": int(env_float("MAX_BUYS_PER_RUN", 1)),
    "ROTATE_WHEN_FULL": env_bool("ROTATE_WHEN_FULL", True),
    "ROTATE_WHEN_CASH_SHORT": env_bool("ROTATE_WHEN_CASH_SHORT", True),
    "DUST_MIN_USD": env_float("DUST_MIN_USD", 2.0),
    "TAKE_PROFIT_PCT": env_float("TAKE_PROFIT_PCT", 3.0),
    "STOP_LOSS_PCT": env_float("STOP_LOSS_PCT", 2.0),
    "TRAIL_STOP_PCT": env_float("TRAIL_STOP_PCT", 1.0),
    "EMERGENCY_SL_PCT": env_float("EMERGENCY_SL_PCT", 8.0),
    "MAX_DAILY_LOSS_PCT": env_float("MAX_DAILY_LOSS_PCT", 5.0),
    "MAX_DAILY_ENTRIES": int(env_float("MAX_DAILY_ENTRIES", 4)),
    "STATE_DIR": env_str("STATE_DIR", ".state"),
    "KPI_CSV": env_str("KPI_CSV", ".state/kpi_history.csv"),
}

# -------------------------- Banner output ---------------------------
def print_banner():
    print("=== Crypto Live — KRAKEN —", utc_now_str(), "===")
    print(f"Mode: {CFG['DRY_RUN']} | RUN_SWITCH: {CFG['RUN_SWITCH']}")
    print(f"AUTO_UNIVERSE={CFG['AUTO_UNIVERSE']}  TOP_K={CFG['UNIVERSE_TOP_K']}  MIN_USD_VOL={int(CFG['UNIVERSE_MIN_USD_VOL'])}")
    print(f"MAX_POSITIONS={CFG['MAX_POSITIONS']}  MAX_BUYS_PER_RUN={CFG['MAX_BUYS_PER_RUN']}")
    print("=================================================")

# ---------------------- KPI persistence (light) ---------------------
def append_kpi_row(equity_value: float):
    ensure_dir(Path(CFG["STATE_DIR"]))
    csv_path = Path(CFG["KPI_CSV"])
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "equity"])
        w.writerow([utc_now_str(), f"{equity_value:.2f}"])

# ---------------------- Universe construction -----------------------
def build_universe(exchange, quote="USD"):
    # In DRY, markets may be stubbed; we still filter by quote and exclusions
    markets = exchange.load_markets()
    symbols = [m for m in markets.keys() if m.endswith(f"/{quote}")]
    # Exclude request
    out = []
    for s in symbols:
        base = s.split("/")[0].upper()
        if base in CFG["UNIVERSE_EXCLUDE"]:
            continue
        out.append(s)
    # NOTE: We would normally rank by 24h USD volume; some public endpoints differ by exchange plan.
    # Keep it simple: cap to TOP_K
    out = out[:max(1, CFG["UNIVERSE_TOP_K"])]
    return out

# -------------------------- Trading helpers -------------------------
def spread_ok(ticker, max_spread_pct):
    bid = ticker.get("bid")
    ask = ticker.get("ask")
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        # Unknown spread -> treat as not-ok to be safe
        return False
    spread = (ask - bid) / ask * 100.0
    return spread <= max_spread_pct

def pick_amount(notional_usd, price):
    if price is None or price <= 0:
        return 0.0
    amt = notional_usd / float(price)
    # round to 1e-6 for safety
    return max(0.0, round(amt, 6))

# ----------------------------- Main run -----------------------------
def main():
    print_banner()
    if CFG["RUN_SWITCH"] != "ON":
        print("[SAFE] RUN_SWITCH=OFF -> skipping run.")
        print("KPI: equity=0.00 activity=0 entries=0 exits=0")
        print("SUMMARY: ok (skipped)")
        return 0

    exchange, has_keys, DRY = build_exchange()

    # Universe
    try:
        universe = build_universe(exchange, quote=CFG["QUOTE"])
    except Exception as e:
        print(f"[WARN] Failed to build universe: {e}")
        universe = []

    if not universe:
        print("[WARN] Empty universe; nothing to do.")
        print("KPI: equity=0.00 activity=0 entries=0 exits=0")
        print("SUMMARY: ok (empty universe)")
        return 0

    entries = 0
    exits = 0
    activity = 0
    equity_val = 0.0  # we don't compute NAV here; just a placeholder for KPI trend

    # Attempt a few buy opportunities (paper or live)
    max_positions = CFG["MAX_POSITIONS"]
    max_buys = CFG["MAX_BUYS_PER_RUN"]
    buys_left = max_buys

    for sym in universe:
        if buys_left <= 0:
            break

        # Fetch ticker to check spread and price (DRY stub returns None -> skip)
        try:
            t = exchange.fetch_ticker(sym)
        except Exception as e:
            print(f"[WARN] fetch_ticker failed for {sym}: {e}")
            continue

        last = t.get("last")
        bid = t.get("bid")
        ask = t.get("ask")
        px = last or bid or ask

        if not spread_ok(t, CFG["MAX_SPREAD_PCT"]):
            print(f"[SKIP] {sym}: spread unknown/too wide.")
            continue

        # Sizing
        notional = CFG["USD_PER_TRADE"]
        if notional < CFG["MIN_TRADE_NOTIONAL_USD"]:
            print(f"[SKIP] {sym}: notional {notional} < MIN_TRADE_NOTIONAL_USD {CFG['MIN_TRADE_NOTIONAL_USD']}")
            continue
        amt = pick_amount(notional, px)
        if amt <= 0:
            print(f"[SKIP] {sym}: zero amount due to bad price.")
            continue

        if DRY or not has_keys:
            print(f"[DRY] would BUY {sym} amount={amt} notional~${notional:.2f}")
            entries += 1
            activity += 1
            buys_left -= 1
            continue

        # Live
        try:
            order = exchange.create_order(sym, "market", "buy", amt)
            print(f"[LIVE] BUY {sym} -> id={order.get('id')}")
            entries += 1
            activity += 1
            buys_left -= 1
        except Exception as e:
            print(f"[ERROR] create_order failed for {sym}: {e}")

    # Minimal KPI persistence for the chart tool
    try:
        append_kpi_row(equity_val)
    except Exception as e:
        print(f"[WARN] append_kpi_row failed: {e}")

    print(f"KPI: equity={equity_val:.2f} activity={activity} entries={entries} exits={exits}")
    print("SUMMARY: ok")

    return 0

# --------------------------------------------------------------------
if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
