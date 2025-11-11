#!/usr/bin/env python3
"""
main.py — LIVE trading version (USD-only) with exact filled qty
1-Coin Rotation Bot (Kraken)

Key fix (Nov-11):
• After BUY, query order to get exact filled quantity (vol_exec) and store it.
• On SELL, use stored filled qty; still apply a tiny safety epsilon.
"""

from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trader.crypto_engine import (
    load_candidates,
    get_public_quote,
    place_market_buy_usd,
    place_market_sell_qty,
    normalize_pair,
    is_usd_pair,
    query_order_filled_qty,
)

STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)
POS_FILE = STATE / "positions.json"
SUMMARY_FILE = STATE / "run_summary.json"

BUY_USD = float(os.getenv("BUY_USD", "25"))
TP_PCT  = float(os.getenv("TP_PCT", "5"))
SL_PCT  = float(os.getenv("SL_PCT", "2"))
MIN_QUOTE = 1e-12

def _migrate_positions(obj: Any) -> dict:
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
        pair = normalize_pair(raw)
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
            if change_pct >= TP_PCT or change_pct <= -SL_PCT:
                reason = "TP" if change_pct >= TP_PCT else "SL"
                print(f"{'🎯' if reason=='TP' else '🛑'} {reason} hit ({change_pct:.2f}%) → SELLING")
                sell_qty = float(positions[sym]["qty"])
                place_market_sell_qty(sym, sell_qty)  # internal epsilon applied
                del positions[sym]
                save_positions(positions)
            else:
                print("Hold signal — still within range.")

    # === BUY phase ===
    positions = load_positions()  # reload if sold
    bought = None
    if not positions and candidates:
        picked = select_top_usd_candidate(candidates)
        if not picked:
            notes.append("no_valid_usd_candidate_with_quote")
        else:
            sym, quote = picked
            est_qty = BUY_USD / quote
            print(f"🟢 Buying {sym} for ${BUY_USD:.2f} (~{est_qty:.6f} est units @ {quote:.10f})")
            txid = place_market_buy_usd(sym, BUY_USD)

            # Query exact filled qty (poll briefly); fall back to estimate if not found
            filled_qty = query_order_filled_qty(txid)
            if filled_qty is None or filled_qty <= 0:
                filled_qty = est_qty * 0.995  # safety fallback
                notes.append("used_est_qty_fallback")

            positions[sym] = {
                "qty": filled_qty,
                "entry_price": quote,
                "txid": txid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_positions(positions)
            bought = sym

    # === Wrap-up ===
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
