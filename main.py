#!/usr/bin/env python3
"""
main.py — LIVE trading version (USD-only)
1-Coin Rotation Bot (Kraken)

Nov-10 fixes:
• Accepts new scan CSV via engine
• Auto-migrates legacy .state/positions.json formats
• ALWAYS writes .state/run_summary.json and prints where they are
• USD-only; market buys use viqc (volume is USD amount)
"""

from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from trader.crypto_engine import (
    load_candidates,
    get_public_quote,
    place_market_buy_usd,
    place_market_sell_qty,
    normalize_pair,
    is_usd_pair,
)

STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)
POS_FILE = STATE / "positions.json"
SUMMARY_FILE = STATE / "run_summary.json"

# === Environment Vars ===
BUY_USD = float(os.getenv("BUY_USD", "25"))
TP_PCT  = float(os.getenv("TP_PCT", "5"))      # take-profit %
SL_PCT  = float(os.getenv("SL_PCT", "2"))      # stop-loss %
MIN_QUOTE = 1e-12

def _migrate_positions(obj: Any) -> Dict[str, dict]:
    try:
        if not obj or not isinstance(obj, dict):
            return {}
        if obj and all(isinstance(v, dict) and "qty" in v for v in obj.values()):
            return obj
        if set(obj.keys()) >= {"pair", "amount", "est_cost"}:
            pair = normalize_pair(str(obj.get("pair", "")))
            qty = float(obj.get("amount", 0.0))
            est_cost = float(obj.get("est_cost", 0.0))
            when = str(obj.get("when", datetime.now(timezone.utc).isoformat()))
            entry_price = (est_cost / qty) if qty > 0 else None
            if not pair or entry_price is None:
                return {}
            return {
                pair: {
                    "qty": qty,
                    "entry_price": entry_price,
                    "txid": "",
                    "timestamp": when,
                }
            }
        return {}
    except Exception:
        return {}

def load_positions() -> dict:
    if POS_FILE.exists():
        try:
            raw = json.loads(POS_FILE.read_text())
            fixed = _migrate_positions(raw)
            if fixed != raw:
                POS_FILE.write_text(json.dumps(fixed, indent=2))
            return fixed
        except Exception as e:
            print(f"[WARN] Failed to read positions.json: {e}")
            return {}
    return {}

def save_positions(d: dict):
    try:
        POS_FILE.write_text(json.dumps(d, indent=2))
        print(f"[STATE] Wrote positions → {POS_FILE.resolve()}")
    except Exception as e:
        print(f"[ERROR] Failed to write positions: {e}", file=sys.stderr)

def log_summary(data: dict):
    try:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        SUMMARY_FILE.write_text(json.dumps(data, indent=2))
        print(f"[STATE] Wrote summary   → {SUMMARY_FILE.resolve()}")
    except Exception as e:
        print(f"[ERROR] Failed to write run summary: {e}", file=sys.stderr)

def select_top_usd_candidate(candidates: list[dict]) -> tuple[str, float] | None:
    for row in candidates:
        raw = (row.get("symbol") or row.get("pair") or "").strip()
        if not raw:
            continue
        pair = normalize_pair(raw)  # -> EATUSD
        if not is_usd_pair(pair):
            continue
        q = get_public_quote(pair)
        if q and q >= MIN_QUOTE:
            return pair, q
    return None

def main():
    print(f"[{datetime.now().isoformat()}] Starting LIVE rotation cycle…")
    candidates = load_candidates()
    notes = []
    if not candidates:
        notes.append("no_candidates_from_scan")

    positions = load_positions()
    holding_syms = list(positions.keys())

    # === SELL phase ===
    if holding_syms:
        sym = holding_syms[0]
        quote = get_public_quote(sym)
        if not quote:
            notes.append(f"no_quote_for_{sym}")
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
    positions = load_positions()
    bought = None
    if not positions and candidates:
        picked = select_top_usd_candidate(candidates)
        if not picked:
            notes.append("no_valid_usd_candidate_with_quote")
        else:
            sym, quote = picked
            qty = BUY_USD / quote
            print(f"🟢 Buying {sym} for ${BUY_USD:.2f} (~{qty:.6f} units @ {quote:.10f})")
            txid = place_market_buy_usd(sym, BUY_USD)
            positions[sym] = {
                "qty": qty,
                "entry_price": quote,
                "txid": txid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_positions(positions)
            bought = sym

    # === Wrap-up (always write a summary) ===
    summary = {
        "positions": positions,
        "holding": list(positions.keys()),
        "buy_usd": BUY_USD,
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "num_candidates": len(candidates),
        "bought": bought,
        "notes": notes,
    }
    log_summary(summary)
    print("✅ Cycle complete — LIVE orders path executed.")

if __name__ == "__main__":
    main()
