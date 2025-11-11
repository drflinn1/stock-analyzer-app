#!/usr/bin/env python3
"""
main.py — USD-only 1-Coin Rotation (Kraken)

Key features:
• Portfolio reconciliation:
    - Auto-sell any untracked USD-quoted coins >= DUST_MIN_USD
    - NEW: Drop stale tracked positions if Kraken shows zero qty
• BUY path with market (viqc spend USD) then LIMIT IOC fallback using SLIP_PCT
• Auto-size buy to available USD (spend min(BUY_USD, USD_BAL - 0.50))
• Verbose logs; always writes .state/run_summary.json
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
    place_limit_buy_usd_ioc,
    place_market_sell_qty,
    normalize_pair,
    is_usd_pair,
    kraken_list_balances,
    get_usd_balance,
    asset_to_usd_pair,
    base_from_pair,
    was_price_protection_block,
)

STATE = Path(".state"); STATE.mkdir(exist_ok=True)
POS_FILE = STATE / "positions.json"
SUMMARY_FILE = STATE / "run_summary.json"

def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return default if v is None else v

# === Tunables (UI input > repo vars > defaults via workflow env) ===
BUY_USD      = _env_float("BUY_USD", 30.0)
TP_PCT       = _env_float("TP_PCT", 5.0)
SL_PCT       = _env_float("SL_PCT", 1.0)
SLIP_PCT     = _env_float("SLIP_PCT", 5.0)
WINDOW_MIN   = _env_float("WINDOW_MIN", 60.0)
DRY_RUN      = _env_str("DRY_RUN", "ON").upper()         # "ON" simulate, "OFF" live
FORCE_BUY_TOP= _env_str("FORCE_BUY_TOP", "ON").upper()   # not used to force rotation, but kept for clarity

MIN_QUOTE     = 1e-12
DUST_MIN_USD  = _env_float("DUST_MIN_USD", 0.50)

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
    for row in candidates:
        raw = (row.get("symbol") or row.get("pair") or "").strip()
        if not raw: continue
        pair = normalize_pair(raw)
        if not is_usd_pair(pair): continue
        q = get_public_quote(pair)
        if q and q >= MIN_QUOTE:
            return pair, q
    return None

# ---------------- Portfolio Reconciliation ----------------
def reconcile_portfolio(positions: Dict[str, dict]) -> Dict[str, dict]:
    balances = kraken_list_balances()  # {'USD': 149.81, 'LSK': 0, ...}
    tracked_bases = { base_from_pair(p) for p in positions.keys() }

    if not balances:
        print("ℹ️  No balances returned or API issue; skipping reconciliation.")
        return positions

    print("🧹 Reconciling portfolio vs positions.json …")
    # A) SELL any *untracked* USD-quoted holdings >= dust
    for asset, qty in balances.items():
        if asset.upper() in ("USD", "ZUSD", "USDT", "EUR", "ZEUR"):
            continue
        qty = float(qty or 0.0)
        if qty <= 0: continue

        if asset.upper() in tracked_bases:
            # managed by bot; skip here
            continue

        pair = asset_to_usd_pair(asset)
        if not is_usd_pair(pair): continue

        q = get_public_quote(pair)
        if not q:
            print(f"  ⚠️ Unable to quote {pair}; skip sweep.")
            continue

        usd_value = qty * q
        if usd_value >= DUST_MIN_USD:
            print(f"  ⚠️ Untracked: {asset} ≈ ${usd_value:.2f} → SELL")
            if DRY_RUN == "ON":
                print(f"  (DRY_RUN) would SELL market {pair} qty {qty}")
            else:
                try:
                    place_market_sell_qty(pair, qty)
                except Exception as e:
                    print(f"  SELL error for {pair}: {e}")
        else:
            print(f"  ℹ️ Ignoring tiny holding {asset} (${usd_value:.2f}).")

    # B) NEW — drop *stale tracked* positions if Kraken shows zero qty
    to_drop = []
    for sym in list(positions.keys()):
        base = base_from_pair(sym).upper()
        live_qty = float(balances.get(base, 0.0) or 0.0)
        if live_qty <= 1e-12:
            print(f"  🔧 Stale tracked position detected ({sym}) but Kraken shows 0 — removing from positions.json")
            to_drop.append(sym)
    for sym in to_drop:
        positions.pop(sym, None)
    if to_drop:
        save_positions(positions)

    return positions

# ------------------------------ Main -------------------------------
def main():
    print(f"[{datetime.now().isoformat()}] Start {('DRY' if DRY_RUN=='ON' else 'LIVE')} cycle")
    print(f"BUY_USD={BUY_USD} TP_PCT={TP_PCT} SL_PCT={SL_PCT} SLIP_PCT={SLIP_PCT} WINDOW_MIN={WINDOW_MIN}")

    positions = load_positions()
    candidates = load_candidates()
    print(f"Loaded candidates: {len(candidates)}")

    # Sweep leftovers & drop stale tracked
    try:
        positions = reconcile_portfolio(positions)
    except Exception as e:
        print(f"⚠️ Reconcile failed (continuing): {e}")

    # SELL logic for current holding (TP/SL)
    holding = list(positions.keys())
    if holding:
        sym = holding[0]
        q = get_public_quote(sym)
        if not q:
            print(f"⚠️ Cannot quote {sym} — hold.")
        else:
            entry = float(positions[sym]["entry_price"])
            chg = (q - entry) / entry * 100
            print(f"Holding {sym} | entry={entry:.10f} now={q:.10f} change={chg:.2f}%")
            if chg >= TP_PCT:
                print("🎯 TP hit → SELL")
                if DRY_RUN == "ON":
                    print(f"(DRY_RUN) would SELL market {sym} qty {positions[sym]['qty']}")
                    del positions[sym]
                else:
                    place_market_sell_qty(sym, float(positions[sym]["qty"]))
                    del positions[sym]
            elif chg <= -SL_PCT:
                print("🛑 SL hit → SELL")
                if DRY_RUN == "ON":
                    print(f"(DRY_RUN) would SELL market {sym} qty {positions[sym]['qty']}")
                    del positions[sym]
                else:
                    place_market_sell_qty(sym, float(positions[sym]["qty"]))
                    del positions[sym]
            else:
                print("Hold — inside TP/SL band.")

    # BUY logic (when flat)
    positions = load_positions()  # re-load in case we sold or dropped stale
    reason = "unknown"
    if not positions:
        pick = select_top_usd_candidate(candidates)
        if not pick:
            reason = "no_valid_usd_candidate"
            print("⚠️ No valid USD candidate with live quote.")
        else:
            sym, quote = pick
            usd_bal = get_usd_balance()
            print(f"Top candidate: {sym} @ {quote:.10f} | USD balance={usd_bal:.2f}")
            if usd_bal <= 1.0:
                reason = "no_cash"
                print("⚠️ No spendable USD.")
            else:
                spend = min(BUY_USD, max(0.0, usd_bal - 0.50))
                if spend < 1.0:
                    reason = "insufficient_after_reserve"
                    print(f"⚠️ Available after reserve < $1. spend_calc={spend:.2f}")
                else:
                    qty_est = spend / quote
                    if DRY_RUN == "ON":
                        print(f"(DRY_RUN) BUY {sym} spend ${spend:.2f} ~{qty_est:.6f}")
                        positions[sym] = {"qty": qty_est, "entry_price": quote, "txid": "DRY-RUN", "timestamp": utc_now_iso()}
                        save_positions(positions)
                        reason = "dry_buy"
                    else:
                        # Market with viqc, then IOC fallback
                        resp_text, blocked = place_market_buy_usd(sym, spend, return_blocked=True)
                        if blocked:
                            limit_price = quote * (1.0 + SLIP_PCT/100.0)
                            qty_lim = spend / limit_price
                            print(f"  ⚠️ Market blocked → LIMIT IOC {limit_price:.10f} (~{qty_lim:.6f})")
                            txid = place_limit_buy_usd_ioc(sym, qty_lim, limit_price)
                            reason = "limit_ioc_fallback"
                        else:
                            txid = resp_text
                            reason = "market_ok"
                        entry = get_public_quote(sym) or quote
                        positions[sym] = {"qty": qty_est, "entry_price": entry, "txid": txid, "timestamp": utc_now_iso()}
                        save_positions(positions)
    else:
        reason = "already_holding"

    # Summary
    summary = {
        "positions": positions,
        "holding": list(positions.keys()),
        "buy_usd": BUY_USD,
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "slip_pct": SLIP_PCT,
        "window_min": WINDOW_MIN,
        "num_candidates": len(candidates),
        "notes": reason,
    }
    log_summary(summary)
    print("✅ Cycle complete. Reason:", reason)

if __name__ == "__main__":
    main()
