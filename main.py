#!/usr/bin/env python3
"""
main.py — LIVE trading version
1-Coin Rotation Bot (Kraken)

Logic:
• Load latest momentum candidates
• Check open positions (.state/positions.json)
• If no position → buy top candidate
• If holding → apply sell guard (TP/SL)
• Always write run summary + KPI logs
"""

from __future__ import annotations
import json, os, time
from datetime import datetime, timezone
from pathlib import Path

from trader.crypto_engine import (
    load_candidates,
    get_public_quote,
    place_market_buy_usd,
    place_market_sell_qty,
)

STATE = Path(".state")
STATE.mkdir(exist_ok=True)
POS_FILE = STATE / "positions.json"
SUMMARY_FILE = STATE / "run_summary.json"

# === Environment Vars ===
BUY_USD = float(os.getenv("BUY_USD", "25"))
TP_PCT  = float(os.getenv("TP_PCT", "5"))      # take-profit %
SL_PCT  = float(os.getenv("SL_PCT", "2"))      # stop-loss %
MIN_QUOTE = 0.0000001

def load_positions() -> dict:
    if POS_FILE.exists():
        try:
            return json.loads(POS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_positions(d: dict):
    POS_FILE.write_text(json.dumps(d, indent=2))

def log_summary(data: dict):
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    SUMMARY_FILE.write_text(json.dumps(data, indent=2))

def main():
    print(f"[{datetime.now().isoformat()}] Starting LIVE rotation cycle…")
    positions = load_positions()
    holding = list(positions.keys())

    candidates = load_candidates()
    if not candidates:
        print("⚠️  No candidates found, aborting.")
        return

    # === SELL phase ===
    if holding:
        sym = holding[0]
        quote = get_public_quote(sym)
        entry_price = positions[sym]["entry_price"]
        change_pct = (quote - entry_price) / entry_price * 100
        print(f"Checking {sym}: {change_pct:.2f}% since entry.")

        if change_pct >= TP_PCT:
            print(f"🎯 Take-profit hit ({change_pct:.2f}%) → SELLING")
            place_market_sell_qty(sym, positions[sym]["qty"])
            del positions[sym]

        elif change_pct <= -SL_PCT:
            print(f"🛑 Stop-loss hit ({change_pct:.2f}%) → SELLING")
            place_market_sell_qty(sym, positions[sym]["qty"])
            del positions[sym]
        else:
            print("Hold signal — still within range.")

    # === BUY phase ===
    positions = load_positions()
    if not positions:
        top = candidates[0]
        sym = top["symbol"]
        quote = get_public_quote(sym)
        if not quote or quote < MIN_QUOTE:
            print(f"Invalid quote for {sym}, skipping buy.")
        else:
            qty = BUY_USD / quote
            print(f"🟢 Buying {sym} for ${BUY_USD:.2f} ({qty:.6f} units @ {quote})")
            txid = place_market_buy_usd(sym, BUY_USD)
            positions[sym] = {
                "qty": qty,
                "entry_price": quote,
                "txid": txid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_positions(positions)

    # === Wrap-up ===
    summary = {
        "positions": positions,
        "holding": list(positions.keys()),
        "buy_usd": BUY_USD,
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "num_candidates": len(candidates),
    }
    log_summary(summary)
    print("✅ Cycle complete — LIVE orders active.")

if __name__ == "__main__":
    main()
