#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crypto Live — Kraken
DRY-safe main.py with explicit BUY/SELL paths and TRAIL stop evaluation.
"""

import os, sys, time, math, csv
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
        def load_markets(self, reload=False, params=None):
            bases = ["BTC","ETH","SOL","ADA","DOGE","ATOM","MATIC","LTC","XRP","DOT","LINK","TON","AVAX"]
            self.markets = {f"{b}/USD": {"symbol": f"{b}/USD", "base": b, "quote": "USD"} for b in bases}
            return self.markets
        def fetch_ticker(self, symbol, params=None):
            return {"symbol": symbol, "bid": None, "ask": None, "last": None}
        def fetch_balance(self, params=None):
            print("[DRY] fetch_balance (skipped)");  return {"total": {}, "free": {}, "used": {}}
        def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
            print("[DRY] fetch_open_orders (skipped)"); return []
        def create_order(self, symbol, type, side, amount, price=None, params=None):
            print(f"[DRY] {side.upper()} {symbol} amount={amount} price={price} (skipped)")
            return {"id": f"DRY-{int(time.time())}", "symbol": symbol, "side": side, "amount": amount, "price": price, "status": "closed"}

    return DryStub(), False, DRY
# ===================================================================

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

    "RESERVE_CASH_PCT": env_float("RESERVE_CASH_PCT", 15.0),
    "USD_PER_TRADE": env_float("USD_PER_TRADE", 20.0),
    "MAX_POSITIONS": int(env_float("MAX_POSITIONS", 6)),
    "MAX_BUYS_PER_RUN": int(env_float("MAX_BUYS_PER_RUN", 0)),   # keep 0 in sell-only mode
    "ROTATE_WHEN_FULL": env_bool("ROTATE_WHEN_FULL", True),
    "ROTATE_WHEN_CASH_SHORT": env_bool("ROTATE_WHEN_CASH_SHORT", True),

    "DUST_MIN_USD": env_float("DUST_MIN_USD", 2.0),
    "CLEANUP_NON_UNIVERSE": env_bool("CLEANUP_NON_UNIVERSE", True),
    "NONUNI_SELL_IF_DOWN_PCT": env_float("NONUNI_SELL_IF_DOWN_PCT", 0.0),
    "NONUNI_KEEP_IF_WINNER_PCT": env_float("NONUNI_KEEP_IF_WINNER_PCT", 6.0),

    "TAKE_PROFIT_PCT": env_float("TAKE_PROFIT_PCT", 3.0),
    "STOP_LOSS_PCT": env_float("STOP_LOSS_PCT", 2.0),
    "TRAIL_STOP_PCT": env_float("TRAIL_STOP_PCT", 1.0),
    "EMERGENCY_SL_PCT": env_float("EMERGENCY_SL_PCT", 8.0),

    "MAX_DAILY_LOSS_PCT": env_float("MAX_DAILY_LOSS_PCT", 5.0),
    "MAX_DAILY_ENTRIES": int(env_float("MAX_DAILY_ENTRIES", 4)),

    "STATE_DIR": env_str("STATE_DIR", ".state"),
    "KPI_CSV": env_str("KPI_CSV", ".state/kpi_history.csv"),
}

def print_banner():
    print("=== Crypto Live — KRAKEN —", utc_now_str(), "===")
    print(f"Mode: {CFG['DRY_RUN']} | RUN_SWITCH: {CFG['RUN_SWITCH']}")
    print(f"AUTO_UNIVERSE={CFG['AUTO_UNIVERSE']}  TOP_K={CFG['UNIVERSE_TOP_K']}  MIN_USD_VOL={int(CFG['UNIVERSE_MIN_USD_VOL'])}")
    print(f"MAX_POSITIONS={CFG['MAX_POSITIONS']}  MAX_BUYS_PER_RUN={CFG['MAX_BUYS_PER_RUN']}")
    print("=================================================")

def append_kpi_row(equity_value: float):
    ensure_dir(Path(CFG["STATE_DIR"]))
    csv_path = Path(CFG["KPI_CSV"])
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "equity"])
        w.writerow([utc_now_str(), f"{equity_value:.2f}"])

def build_universe(exchange, quote="USD"):
    markets = exchange.load_markets()
    symbols = [m for m in markets.keys() if m.endswith(f"/{quote}")]
    out = []
    for s in symbols:
        base = s.split("/")[0].upper()
        if base in CFG["UNIVERSE_EXCLUDE"]:
            continue
        out.append(s)
    out = out[:max(1, CFG["UNIVERSE_TOP_K"])]
    return out

def get_holdings(exchange, quote="USD"):
    holdings = []
    try:
        bal = exchange.fetch_balance()
    except Exception as e:
        print(f"[WARN] fetch_balance failed: {e}")
        return holdings
    totals = (bal or {}).get("total", {}) or {}
    for base, amt in (totals or {}).items():
        try:
            if not amt or amt <= 0:
                continue
            base_u = base.upper()
            sym = f"{base_u}/{quote}"
            tk = exchange.fetch_ticker(sym)
            px = tk.get("last") or tk.get("bid") or tk.get("ask") or 0.0
            value = float(amt) * float(px or 0.0)
            holdings.append((sym, base_u, float(amt), float(value)))
        except Exception:
            continue
    return holdings

def spread_ok(ticker, max_spread_pct):
    bid = ticker.get("bid")
    ask = ticker.get("ask")
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return False
    spread = (ask - bid) / ask * 100.0
    return spread <= max_spread_pct

def pick_amount(notional_usd, price):
    if price is None or price <= 0:
        return 0.0
    return max(0.0, round(notional_usd / float(price), 6))

def do_buy(exchange, sym, amt, notional, live_enabled):
    if live_enabled:
        try:
            order = exchange.create_order(sym, "market", "buy", amt)
            print(f"[LIVE] BUY {sym} id={order.get('id')} amount={amt}")
        except Exception as e:
            print(f"[ERROR] BUY failed {sym}: {e}")
            return False
    else:
        print(f"[DRY] BUY {sym} amount={amt} notional~${notional:.2f}")
    return True

def do_sell(exchange, sym, amt, reason, live_enabled):
    if amt <= 0:
        return False
    if live_enabled:
        try:
            order = exchange.create_order(sym, "market", "sell", amt)
            print(f"[LIVE] SELL {sym} id={order.get('id')} amount={amt} reason={reason}")
        except Exception as e:
            print(f"[ERROR] SELL failed {sym}: {e}")
            return False
    else:
        print(f"[DRY] SELL {sym} amount={amt} reason={reason}")
    return True

def apply_trailing_stops(exchange, holdings, trail_pct, live_enabled):
    exits = 0
    for sym, base, amt, usd_val in holdings:
        try:
            tk = exchange.fetch_ticker(sym)
        except Exception as e:
            print(f"[WARN] ticker for {sym} failed in TRAIL: {e}")
            continue
        px = tk.get("last") or tk.get("bid") or tk.get("ask")
        high = tk.get("high") or px
        if not px or not high or px <= 0 or high <= 0:
            continue
        trigger = float(high) * (1.0 - float(trail_pct)/100.0)
        print(f"[TRAIL] evaluate {sym}: high~{float(high):.6f} trigger~{trigger:.6f} pct={trail_pct:.2f}")
        # To auto-fire trailing sells, uncomment:
        # if px <= trigger:
        #     if do_sell(exchange, sym, amt, "trailing-stop", live_enabled):
        #         exits += 1
    return exits

def main():
    print_banner()
    if CFG["RUN_SWITCH"] != "ON":
        print("[SAFE] RUN_SWITCH=OFF -> skipping run.")
        print("KPI: equity=0.00 activity=0 entries=0 exits=0")
        print("SUMMARY: ok (skipped)")
        return 0

    exchange, has_keys, DRY = build_exchange()
    live_enabled = (has_keys and not DRY)

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

    exits = 0
    try:
        holdings = get_holdings(exchange, quote=CFG["QUOTE"])
    except Exception as e:
        print(f"[WARN] holdings failed: {e}")
        holdings = []

    if holdings:
        exits += apply_trailing_stops(exchange, holdings, CFG["TRAIL_STOP_PCT"], live_enabled)
        uni_set = set(universe)
        for sym, base, amt, usd_val in holdings:
            if usd_val < CFG["DUST_MIN_USD"]:
                if do_sell(exchange, sym, amt, "dust", live_enabled):
                    exits += 1
                continue
            if CFG["CLEANUP_NON_UNIVERSE"] and sym not in uni_set:
                if do_sell(exchange, sym, amt, "non-universe", live_enabled):
                    exits += 1

    entries = 0
    activity = 0
    equity_val = 0.0
    buys_left = CFG["MAX_BUYS_PER_RUN"]

    for sym in universe:
        if buys_left <= 0:
            break
        try:
            t = exchange.fetch_ticker(sym)
        except Exception as e:
            print(f"[WARN] fetch_ticker failed for {sym}: {e}")
            continue
        last = t.get("last") or t.get("bid") or t.get("ask")
        if not spread_ok(t, CFG["MAX_SPREAD_PCT"]):
            print(f"[SKIP] {sym}: spread unknown/too wide.")
            continue
        notional = CFG["USD_PER_TRADE"]
        if notional < CFG["MIN_TRADE_NOTIONAL_USD"]:
            print(f"[SKIP] {sym}: notional {notional} < MIN_TRADE_NOTIONAL_USD {CFG['MIN_TRADE_NOTIONAL_USD']}")
            continue
        amt = pick_amount(notional, last)
        if amt <= 0:
            print(f"[SKIP] {sym}: zero amount due to bad price.")
            continue
        if do_buy(exchange, sym, amt, notional, live_enabled):
            entries += 1
            activity += 1
            buys_left -= 1

    try:
        append_kpi_row(equity_val)
    except Exception as e:
        print(f"[WARN] append_kpi_row failed: {e}")

    print(f"KPI: equity={equity_val:.2f} activity={activity} entries={entries} exits={exits}")
    print("SUMMARY: ok")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
