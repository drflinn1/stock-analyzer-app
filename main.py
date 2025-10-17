#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CryptoBOT main — Safe 24/7 runner with emergency stop, rotation, and KPI logging.

Environment (GitHub Actions Variables -> "Variables" tab; fallbacks in code):
- DRY_RUN: "ON" | "OFF" (default "ON")
- RUN_SWITCH: "ON" | "OFF" (global enable/disable)
- EMERGENCY_STOP: "ON" | "OFF" (optional panic stop)
- MIN_BUY_USD: float (default 10)
- MIN_SELL_USD: float (default 10)
- MAX_POSITIONS: int (default 3)
- MAX_BUYS_PER_RUN: int (default 2)
- UNIVERSE_TOP_K: int (default 25)
- RESERVE_CASH_PCT: int (default 5)
- ROTATE_WHEN_FULL: "true" | "false" (default true)
- ROTATE_WHEN_CASH_SHORT: "true" | "false" (default true)
- DUST_MIN_USD: float (default 2)  # safe cleanup threshold (optional)
- DUST_SKIP_STABLES: "true" | "false" (default true)
- STATE_DIR: ".state" (default)
- KPI_CSV: ".state/kpi_history.csv" (default)
- KPI_IMG: ".state/kpi_chart.png" (unused here, but kept for Actions step)

Secrets (Settings → Secrets and variables → Actions → Secrets):
- KRAKEN_API_KEY, KRAKEN_API_SECRET

This file depends on: trader/crypto_engine.py
"""

import os
import sys
import time
import json
import math
from datetime import datetime, timezone

# Third-party
try:
    import ccxt
except Exception as e:
    print(f"[init] Missing ccxt: {e}")
    sys.exit(1)

# Local
from trader.crypto_engine import (
    build_exchange,
    get_cash_balance_usd,
    fetch_positions_snapshot,
    pick_candidates,
    place_market_buy,
    place_market_sell,
    symbol_to_usd_market,
    estimate_equity_usd,
)

# ---------- Env helpers ----------
def env_str(name, default):
    return os.environ.get(name, default)

def env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "on", "yes")

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

# ---------- Config ----------
DRY_RUN = env_str("DRY_RUN", "ON").upper() == "ON"  # ON means no live orders
RUN_SWITCH = env_str("RUN_SWITCH", "ON").upper() == "ON"
EMERGENCY_STOP = env_str("EMERGENCY_STOP", "OFF").upper() == "ON"

MIN_BUY_USD = env_float("MIN_BUY_USD", 10.0)
MIN_SELL_USD = env_float("MIN_SELL_USD", 10.0)
MAX_POSITIONS = env_int("MAX_POSITIONS", 3)
MAX_BUYS_PER_RUN = env_int("MAX_BUYS_PER_RUN", 2)
UNIVERSE_TOP_K = env_int("UNIVERSE_TOP_K", 25)
RESERVE_CASH_PCT = env_int("RESERVE_CASH_PCT", 5)

ROTATE_WHEN_FULL = env_bool("ROTATE_WHEN_FULL", True)
ROTATE_WHEN_CASH_SHORT = env_bool("ROTATE_WHEN_CASH_SHORT", True)

DUST_MIN_USD = env_float("DUST_MIN_USD", 2.0)
DUST_SKIP_STABLES = env_bool("DUST_SKIP_STABLES", True)

STATE_DIR = env_str("STATE_DIR", ".state")
KPI_CSV = env_str("KPI_CSV", os.path.join(STATE_DIR, "kpi_history.csv"))

STOP_FILE = os.path.join(STATE_DIR, "STOP")

os.makedirs(STATE_DIR, exist_ok=True)

# ---------- Emergency stop check ----------
def emergency_stop_active() -> bool:
    if not RUN_SWITCH:
        print("[STOP] RUN_SWITCH=OFF — trading halted.")
        return True
    if EMERGENCY_STOP:
        print("[STOP] EMERGENCY_STOP=ON — trading halted.")
        return True
    if os.path.exists(STOP_FILE):
        print("[STOP] .state/STOP file present — trading halted.")
        return True
    return False

# ---------- KPI logging ----------
def append_kpi_row(equity_usd: float, positions_count: int):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    header_needed = not os.path.exists(KPI_CSV)
    line = f"{ts},{equity_usd:.2f},{positions_count}\n"
    with open(KPI_CSV, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,equity,positions\n")
        f.write(line)
    print(f"[KPI] {ts} equity={equity_usd:.2f} pos={positions_count}")

# ---------- Rotation ----------
def rotate_if_needed(exchange, positions, candidates, cash_usd, reserve_cash_usd):
    """
    Replaces worst position with a stronger candidate when:
      - Portfolio is full (>= MAX_POSITIONS) and ROTATE_WHEN_FULL
      - OR cash < reserve and ROTATE_WHEN_CASH_SHORT
    """
    if not positions:
        return cash_usd

    portfolio_full = len(positions) >= MAX_POSITIONS
    cash_short = cash_usd < reserve_cash_usd

    should_rotate = (portfolio_full and ROTATE_WHEN_FULL) or (cash_short and ROTATE_WHEN_CASH_SHORT)
    if not should_rotate:
        return cash_usd

    # Rank held positions by 24h change ascending (worst first). If your engine
    # can't fetch per-position change quickly, we approximate by using candidates
    # list to infer strength; unknown symbols get low score.
    change_map = {c["symbol"]: c.get("change24h", -999) for c in candidates}
    ranked = sorted(
        positions,
        key=lambda p: change_map.get(p["symbol"], -999)
    )

    worst = ranked[0]
    worst_usd = worst["usd_value"]
    if worst_usd < MIN_SELL_USD:
        print(f"[rotate] Worst {worst['symbol']} too small (${worst_usd:.2f}) — skip rotation.")
        return cash_usd

    # Best candidate not already held
    held_syms = {p["symbol"] for p in positions}
    best = next((c for c in candidates if c["symbol"] not in held_syms), None)
    if not best:
        print("[rotate] No stronger candidate available.")
        return cash_usd

    # Sell worst
    print(f"[rotate] SELL {worst['symbol']} ~ ${worst_usd:.2f} to rotate into {best['symbol']}.")
    if not DRY_RUN:
        try:
            place_market_sell(exchange, worst["symbol"], worst["base_qty"])
        except Exception as e:
            print(f"[rotate] Sell failed: {e}")

    cash_usd += worst_usd
    return cash_usd

# ---------- Main ----------
def main():
    print("=== Crypto Live — SAFE ===", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    print(f"Mode: {'DRY' if DRY_RUN else 'LIVE'}  RUN_SWITCH: {'ON' if RUN_SWITCH else 'OFF'}")
    print(f"MAX_POSITIONS={MAX_POSITIONS}  MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}")
    print(f"UNIVERSE_TOP_K={UNIVERSE_TOP_K}  RESERVE_CASH_PCT={RESERVE_CASH_PCT}")

    if emergency_stop_active():
        # Still log KPI but skip trading
        try:
            ex = build_exchange(os.environ.get("KRAKEN_API_KEY"), os.environ.get("KRAKEN_API_SECRET"), DRY_RUN)
            equity = estimate_equity_usd(ex)
            append_kpi_row(equity, 0)
        except Exception as e:
            print(f"[stop] KPI log while stopped failed: {e}")
        return

    api_key = os.environ.get("KRAKEN_API_KEY")
    api_secret = os.environ.get("KRAKEN_API_SECRET")
    exchange = build_exchange(api_key, api_secret, DRY_RUN)

    # Universe + candidates
    candidates = pick_candidates(exchange, top_k=UNIVERSE_TOP_K)
    print(f"[scan] Candidates (top {len(candidates)} by 24h change):",
          ", ".join([f"{c['symbol']}({c['change24h']:+.2f}%)" for c in candidates[:8]]),
          ("..." if len(candidates) > 8 else ""))

    # Positions & balances
    positions = fetch_positions_snapshot(exchange)
    cash_usd = get_cash_balance_usd(exchange)
    equity = estimate_equity_usd(exchange)
    reserve_cash_usd = equity * (RESERVE_CASH_PCT / 100.0)

    print(f"[acct] cash=${cash_usd:.2f} reserve=${reserve_cash_usd:.2f} equity=${equity:.2f} positions={len(positions)}")

    # Rotation if portfolio is full or cash short
    cash_usd = rotate_if_needed(exchange, positions, candidates, cash_usd, reserve_cash_usd)

    # Buys
    buys_made = 0
    held = {p["symbol"] for p in positions}
    for coin in candidates:
        if buys_made >= MAX_BUYS_PER_RUN:
            break
        if len(held) >= MAX_POSITIONS:
            break
        if coin["symbol"] in held:
            continue

        alloc = max(MIN_BUY_USD, (equity - reserve_cash_usd) / max(1, MAX_POSITIONS) )
        if cash_usd - reserve_cash_usd < MIN_BUY_USD:
            print("[buy] Cash below reserve; not buying more.")
            break

        spend = min(alloc, cash_usd - reserve_cash_usd)
        if spend < MIN_BUY_USD:
            continue

        print(f"[buy] BUY {coin['symbol']} for ~${spend:.2f}")
        if not DRY_RUN:
            try:
                place_market_buy(exchange, coin["symbol"], spend)
            except Exception as e:
                print(f"[buy] failed: {e}")
                continue

        buys_made += 1
        cash_usd -= spend
        held.add(coin["symbol"])

    # KPI — after trades
    positions_after = fetch_positions_snapshot(exchange)
    equity_after = estimate_equity_usd(exchange)
    append_kpi_row(equity_after, len(positions_after))

    # Summary
    print("SUMMARY:")
    print(f"  buys={buys_made}  pos={len(positions_after)}  equity=${equity_after:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fatal] {e}")
        sys.exit(1)
