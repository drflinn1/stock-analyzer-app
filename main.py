#!/usr/bin/env python3
"""
main.py — LIVE trading version (USD-only) with Portfolio Reconciliation
1-Coin Rotation Bot (Kraken)

What’s new (Portfolio-Aware):
• Before SELL/BUY, reconcile Kraken balances vs .state/positions.json
• Any untracked USD-quoted holdings (value ≥ $0.50) are auto-sold
• Still trades only USD pairs; market-buys use viqc (volume is USD)
"""

from __future__ import annotations
import json, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from trader.crypto_engine import (
    load_candidates,
    get_public_quote,
    place_market_buy_usd,
    place_market_sell_qty,
    normalize_pair,
    is_usd_pair,
    kraken_list_balances,          # NEW
    asset_to_usd_pair,             # NEW
    base_from_pair,                # NEW
)

STATE = Path(".state")
STATE.mkdir(exist_ok=True)
POS_FILE = STATE / "positions.json"
SUMMARY_FILE = STATE / "run_summary.json"

# === Environment Vars ===
BUY_USD  = float(os.getenv("BUY_USD", "25"))
TP_PCT   = float(os.getenv("TP_PCT", "5"))      # take-profit %
SL_PCT   = float(os.getenv("SL_PCT", "2"))      # stop-loss %
MIN_QUOTE = 1e-12
DUST_USD_THRESHOLD = float(os.getenv("DUST_USD_THRESHOLD", "0.50"))  # min value to sweep

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_positions() -> Dict[str, dict]:
    if POS_FILE.exists():
        try:
            return json.loads(POS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_positions(d: Dict[str, dict]):
    POS_FILE.write_text(json.dumps(d, indent=2))

def log_summary(data: dict):
    data["timestamp"] = utc_now_iso()
    SUMMARY_FILE.write_text(json.dumps(data, indent=2))

def select_top_usd_candidate(candidates: List[dict]) -> Optional[Tuple[str, float]]:
    """
    Return (symbol, quote) for the first USD-quoted candidate with a valid live quote.
    Accepts 'EATUSD', 'EAT/USD', etc. Rejects *EUR, *USDT.
    """
    for row in candidates:
        raw = (row.get("symbol") or "").strip()
        if not raw:
            continue
        pair = normalize_pair(raw)  # -> e.g., EATUSD / SOLUSD
        if not is_usd_pair(pair):
            continue
        q = get_public_quote(pair)
        if q and q >= MIN_QUOTE:
            return pair, q
    return None

# ------------------- Portfolio Reconciliation (NEW) -------------------
def reconcile_portfolio(positions: Dict[str, dict]) -> Dict[str, dict]:
    """
    1) Query Kraken balances
    2) Identify any USD-quoted holdings not tracked in positions.json
    3) If usd_value >= DUST_USD_THRESHOLD -> market sell the full qty
    4) Return (possibly unchanged) positions dict
    """
    balances = kraken_list_balances()   # { "USD": 120.1, "LSK": 94.6, "MELANIA": 0.0, ... }
    tracked_bases = { base_from_pair(p) for p in positions.keys() }

    if not balances:
        print("ℹ️  No balances returned or API issue; skipping reconciliation.")
        return positions

    print("🧹 Reconciling portfolio vs positions.json …")
    for asset, qty in balances.items():
        if asset.upper() in ("USD", "ZUSD", "USDT", "EUR", "ZEUR"):
            continue
        qty = float(qty or 0.0)
        if qty <= 0:
            continue

        if asset.upper() in tracked_bases:
            # Bot is already managing this base asset.
            continue

        pair = asset_to_usd_pair(asset)   # e.g., "LSK" -> "LSKUSD"
        if not is_usd_pair(pair):
            continue

        q = get_public_quote(pair)
        if not q:
            print(f"  ⚠️ Unable to quote {pair}; skip sweep.")
            continue

        usd_value = qty * q
        if usd_value >= DUST_USD_THRESHOLD:
            print(f"  ⚠️ Untracked position detected: {asset} ≈ ${usd_value:.2f} → SELLING")
            try:
                place_market_sell_qty(pair, qty)
            except Exception as e:
                print(f"  SELL error for {pair}: {e}")
        else:
            print(f"  ℹ️ Ignoring tiny/unpriced holding {asset} (${usd_value:.2f}).")

    return positions

# ---------------------------------------------------------------------

def main():
    print(f"[{datetime.now().isoformat()}] Starting LIVE rotation cycle…")

    # Load state
    positions = load_positions()
    candidates = load_candidates()

    # Reconcile ANY leftover holdings Kraken-side (self-healing step)
    try:
        positions = reconcile_portfolio(positions)
    except Exception as e:
        print(f"⚠️ Reconcile failed (continuing): {e}")

    if not candidates:
        print("⚠️  No candidates found, aborting.")
        # still write a summary
        log_summary({
            "positions": positions,
            "holding": list(positions.keys()),
            "buy_usd": BUY_USD, "tp_pct": TP_PCT, "sl_pct": SL_PCT,
            "num_candidates": 0,
            "notes": "No candidates; reconciliation only.",
        })
        return

    # === SELL phase (manage the single tracked position, if any) ===
    holding = list(positions.keys())
    if holding:
        sym = holding[0]
        quote = get_public_quote(sym)
        if not quote:
            print(f"⚠️ Unable to quote {sym} right now — holding.")
        else:
            entry_price = float(positions[sym]["entry_price"])
            change_pct = (quote - entry_price) / entry_price * 100
            print(f"Checking {sym}: {change_pct:.2f}% since entry.")

            if change_pct >= TP_PCT:
                print(f"🎯 Take-profit hit ({change_pct:.2f}%) → SELLING")
                place_market_sell_qty(sym, float(positions[sym]["qty"]))
                del positions[sym]

            elif change_pct <= -SL_PCT:
                print(f"🛑 Stop-loss hit ({change_pct:.2f}%) → SELLING")
                place_market_sell_qty(sym, float(positions[sym]["qty"]))
                del positions[sym]
            else:
                print("Hold signal — still within range.")

    # === BUY phase ===
    positions = load_positions()  # re-load if we just sold
    if not positions:
        picked = select_top_usd_candidate(candidates)
        if not picked:
            print("⚠️  No valid USD candidate with a live quote — skipping buy this cycle.")
        else:
            sym, quote = picked
            qty = BUY_USD / quote
            print(f"🟢 Buying {sym} for ${BUY_USD:.2f} (~{qty:.6f} units @ {quote:.10f})")
            txid = place_market_buy_usd(sym, BUY_USD)
            positions[sym] = {
                "qty": qty,
                "entry_price": quote,
                "txid": txid,
                "timestamp": utc_now_iso(),
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
        "notes": "USD-only trading; viqc buys; portfolio reconciliation active",
    }
    log_summary(summary)
    print("✅ Cycle complete — LIVE orders path executed.")

if __name__ == "__main__":
    main()
