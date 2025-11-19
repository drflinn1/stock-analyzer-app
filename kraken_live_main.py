#!/usr/bin/env python3
"""
kraken_live_main.py
LIVE ENTRY SCRIPT (REAL KRAKEN TRADING)

This script:
- Loads actual Kraken balances (USD + current crypto)
- Syncs a single position into .state/positions.json
- Loads momentum_candidates.csv (fallback safe)
- Picks BEST coin or UNIVERSE_PICK override
- Executes SELL → BUY rotation
- Writes run_summary.md
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone

from kraken_api import KrakenTradeAPI
from trader.crypto_engine import (
    load_candidates,
    get_public_quote,
)


STATE_DIR = Path(".state")
POSITIONS_FILE = STATE_DIR / "positions.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"


def load_positions():
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text())
        except:
            pass
    return {"symbol": None, "units": 0.0, "entry_price": 0.0, "entry_time": None}


def save_positions(d):
    STATE_DIR.mkdir(exist_ok=True)
    POSITIONS_FILE.write_text(json.dumps(d, indent=2))


def write_summary(text: str):
    STATE_DIR.mkdir(exist_ok=True)
    SUMMARY_MD.write_text(text)


def pick_best_candidate(candidates, universe_pick):
    if universe_pick:
        # User tells us exactly what to buy
        return universe_pick

    # Otherwise pick the top gainer by pct_change
    if not candidates:
        return None

    best = sorted(
        candidates,
        key=lambda x: float(x.get("pct", 0.0)),
        reverse=True,
    )[0]

    return best["symbol"]


def main():
    ENV = {
        "BUY_USD": float(os.getenv("BUY_USD", "7")),
        "TP_PCT": float(os.getenv("TP_PCT", "8")),
        "SL_PCT": float(os.getenv("SL_PCT", "2")),
        "SOFT_SL_PCT": float(os.getenv("SOFT_SL_PCT", "1")),
        "SLOW_GAIN_REQ": float(os.getenv("SLOW_GAIN_REQ", "3")),
        "STALE_MINUTES": float(os.getenv("STALE_MINUTES", "60")),
        "UNIVERSE_PICK": os.getenv("UNIVERSE_PICK", ""),
        "SELL_GUARD_MODE": os.getenv("SELL_GUARD_MODE", "SG-2"),
        "DRY_RUN": os.getenv("DRY_RUN", "OFF"),
    }

    api = KrakenTradeAPI()

    # ==============================
    # STEP 1 — Determine actual live position
    # ==============================
    live_balances = api.get_balances()   # dict: {"SOL": "0.05", "USD": "131.0", ...}

    # Identify any crypto > small dust
    held_symbol = None
    held_units = 0.0

    for sym, amt in live_balances.items():
        if sym == "USD":
            continue
        units = float(amt)
        if units > 0.001:
            held_symbol = sym + "/USD"
            held_units = units
            break

    # Load previous .state/positions.json
    pos = load_positions()

    # If live position disagrees with state — override state immediately
    if held_symbol:
        price = get_public_quote(held_symbol)
        pos = {
            "symbol": held_symbol,
            "units": held_units,
            "entry_price": price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
        }
        save_positions(pos)
    else:
        pos = {"symbol": None, "units": 0.0, "entry_price": 0.0, "entry_time": None}
        save_positions(pos)

    # ==============================
    # STEP 2 — Load Momentum candidates
    # ==============================
    cands = load_candidates()
    symbol_to_buy = pick_best_candidate(cands, ENV["UNIVERSE_PICK"])

    if not symbol_to_buy:
        write_summary("No candidates. Rotation completed.")
        return

    # Normalize (ensures e.g. SOLUSD → SOL/USD)
    if "/" not in symbol_to_buy:
        symbol_to_buy = symbol_to_buy.replace("USD", "") + "/USD"

    # ==============================
    # STEP 3 — SELL if holding something else
    # ==============================
    if pos["symbol"] and pos["symbol"] != symbol_to_buy:
        api.sell_market(pos["symbol"], pos["units"], slip_pct=ENV["SOFT_SL_PCT"], dry_run=False)

        # Clear state
        pos = {"symbol": None, "units": 0.0, "entry_price": 0.0, "entry_time": None}
        save_positions(pos)

    # ==============================
    # STEP 4 — BUY fresh coin
    # ==============================
    price = get_public_quote(symbol_to_buy)
    if not price:
        write_summary(f"Quote unavailable for {symbol_to_buy}. Aborting.")
        return

    units = ENV["BUY_USD"] / price
    api.buy_market(symbol_to_buy, units, slip_pct=ENV["SOFT_SL_PCT"], dry_run=False)

    # Update state
    pos = {
        "symbol": symbol_to_buy,
        "units": units,
        "entry_price": price,
        "entry_time": datetime.now(timezone.utc).isoformat(),
    }
    save_positions(pos)

    # ==============================
    # STEP 5 — Write summary
    # ==============================
    summary = f"""
# LIVE Rotation Complete — {datetime.utcnow().isoformat()} UTC

Bought: **{symbol_to_buy}**
Units: **{units:.6f}**
Price: **{price} USD**
USD allocated: **{ENV['BUY_USD']}**

Position synced to .state/positions.json
"""
    write_summary(summary)


if __name__ == "__main__":
    import os
    main()
