#!/usr/bin/env python3
# (same header docstring as before)

from __future__ import annotations
import os, json, math, time, csv, pathlib, requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
# ... (all your existing code remains) ...

SLACK_WEBHOOK_URL   = os.environ.get("SLACK_WEBHOOK_URL", "").strip()

def slack_post(text: str):
    """Post a simple message to Slack if webhook is configured."""
    if not SLACK_WEBHOOK_URL:
        return
    try:
        requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": text},
            timeout=6,
        )
    except Exception:
        # Don't crash trading if Slack is unavailable
        pass

# (rest of config & helpers unchanged)

def place_market_buy(ex, symbol: str, usd_alloc: float, price_hint: float) -> Tuple[float,float]:
    # ... existing content ...
    if DRY_RUN == "ON":
        # existing print/record ...
        slack_post(f"🧪 DRY BUY {symbol} ≈ ${usd_alloc:.2f}")
        return qty, price
    # live...
    # existing order/print/record...
    slack_post(f"🟢 BUY {symbol} qty={qty:.6f} ≈ ${qty*fill:.2f}  (SL {SL_PCT*100:.1f}%, TR {TRAIL_PCT*100:.1f}%, TP {TP1_PCT*100:.1f}% x {int(TP1_SIZE*100)}%)")
    return qty, fill

def place_market_sell(ex, symbol: str, qty: float, price_hint: float) -> Tuple[float,float]:
    # ... existing content ...
    if DRY_RUN == "ON":
        # existing print/record ...
        slack_post(f"🧪 DRY SELL {symbol} qty={qty:.6f} ≈ ${qty*price:.2f}")
        return qty, price
    # live...
    # existing order/print/record...
    slack_post(f"🔴 SELL {symbol} qty={qty:.6f} ≈ ${qty*fill:.2f}")
    return qty, fill

# (evaluate_exits unchanged, uses place_market_sell so Slack fires on SL/TP/Trail/Dust)

def main():
    if RUN_SWITCH != "ON":
        print(f"[SKIP] RUN_SWITCH={RUN_SWITCH} → exiting.")
        return

    print("========== RUN ==========")
    log_header()
    ex = make_exchange(dry=(DRY_RUN=="ON"))

    snapshot = fetch_market_snapshot(ex)
    usd_free, base_balances = fetch_balances(ex)

    # sync positions ...
    # (unchanged)

    # 1) exits
    usd_from_sells, usd_dusted = evaluate_exits(ex, snapshot, base_balances)
    if usd_from_sells or usd_dusted:
        usd_free, base_balances = fetch_balances(ex)

    # 2) entries
    held_count = sum(1 for s in WHITELIST if positions.get(s,{}).get("qty",0) > 0)
    spent = place_entries(ex, snapshot, usd_free, held_count)

    # 3) KPIs / summary
    portfolio_value = float(usd_free)
    for s in WHITELIST:
        base = s.split("/")[0]
        qty = positions.get(s,{}).get("qty",0.0)
        if qty > 0:
            portfolio_value += qty * snapshot[s]["last"]
    msg = f"{'🧪' if DRY_RUN=='ON' else '✅'} {BOT_NAME} summary: PV≈${portfolio_value:.2f} | USD≈${usd_free:.2f} | spent_today=${daily.get('spent',0.0):.2f}"
    print(f"[SUMMARY] {msg}")
    slack_post(msg)
    kpi(f"pv=${portfolio_value:.2f}; usd_free=${usd_free:.2f}; spent_today=${daily.get('spent',0.0):.2f}; buys_this_run=${spent:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        raise
