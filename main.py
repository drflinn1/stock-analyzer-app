#!/usr/bin/env python3
"""
main.py — LIVE trading version (USD-only) with Portfolio Reconciliation
1-Coin Rotation Bot (Kraken)

Adds:
• SLIP_PCT + IOC fallback for buys when price-protection blocks market orders
• DRY_RUN honored from env (UI input > repo var > default)
• Always writes .state/run_summary.json
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
    place_limit_buy_usd_ioc,   # NEW
    place_market_sell_qty,
    normalize_pair,
    is_usd_pair,
    kraken_list_balances,
    asset_to_usd_pair,
    base_from_pair,
    was_price_protection_block, # NEW
)

STATE = Path(".state")
STATE.mkdir(exist_ok=True)
POS_FILE = STATE / "positions.json"
SUMMARY_FILE = STATE / "run_summary.json"

# === Environment Vars ===
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return v if v is not None else default

BUY_USD   = _env_float("BUY_USD", 30.0)
TP_PCT    = _env_float("TP_PCT", 5.0)
SL_PCT    = _env_float("SL_PCT", 1.0)
SLIP_PCT  = _env_float("SLIP_PCT", 5.0)        # aggressiveness for fallback limit
WINDOW_MIN = _env_float("WINDOW_MIN", 60.0)
DRY_RUN   = _env_str("DRY_RUN", "ON").upper()   # "ON" (simulate) or "OFF" (live)

MIN_QUOTE = 1e-12
DUST_USD_THRESHOLD = _env_float("DUST_MIN_USD", 0.50)  # repo var sometimes named DUST_MIN_USD

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

# ------------------- Portfolio Reconciliation -------------------
def reconcile_portfolio(positions: Dict[str, dict]) -> Dict[str, dict]:
    """
    1) Query Kraken balances
    2) Identify any USD-quoted holdings not tracked in positions.json
    3) If usd_value >= DUST_USD_THRESHOLD -> market sell full qty
    """
    balances = kraken_list_balances()  # { "USD": 120.1, "LSK": 94.6, ... }
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
            # Already tracked/managed
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
            if DRY_RUN == "ON":
                print("  (DRY_RUN) would SELL market", pair, "qty", qty)
            else:
                try:
                    place_market_sell_qty(pair, qty)
                except Exception as e:
                    print(f"  SELL error for {pair}: {e}")
        else:
            print(f"  ℹ️ Ignoring tiny/unpriced holding {asset} (${usd_value:.2f}).")

    return positions

# -------------------------- Main --------------------------
def main():
    print(f"[{datetime.now().isoformat()}] Starting {'DRY' if DRY_RUN=='ON' else 'LIVE'} rotation cycle…")
    print(f"BUY_USD={BUY_USD}  TP_PCT={TP_PCT}  SL_PCT={SL_PCT}  SLIP_PCT={SLIP_PCT}  WINDOW_MIN={WINDOW_MIN}")

    # Load state
    positions = load_positions()
    candidates = load_candidates()

    # Reconcile leftovers (self-healing)
    try:
        positions = reconcile_portfolio(positions)
    except Exception as e:
        print(f"⚠️ Reconcile failed (continuing): {e}")

    if not candidates:
        print("⚠️  No candidates found, aborting.")
        log_summary({
            "positions": positions,
            "holding": list(positions.keys()),
            "buy_usd": BUY_USD, "tp_pct": TP_PCT, "sl_pct": SL_PCT,
            "slip_pct": SLIP_PCT,
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
                if DRY_RUN == "ON":
                    print("(DRY_RUN) would SELL market", sym, "qty", positions[sym]["qty"])
                    del positions[sym]
                else:
                    place_market_sell_qty(sym, float(positions[sym]["qty"]))
                    del positions[sym]

            elif change_pct <= -SL_PCT:
                print(f"🛑 Stop-loss hit ({change_pct:.2f}%) → SELLING")
                if DRY_RUN == "ON":
                    print("(DRY_RUN) would SELL market", sym, "qty", positions[sym]["qty"])
                    del positions[sym]
                else:
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
            qty_est = BUY_USD / quote
            print(f"🟢 Buying {sym} for ${BUY_USD:.2f} (~{qty_est:.6f} units @ {quote:.10f})")

            if DRY_RUN == "ON":
                txid = "DRY-RUN"
                entry = quote
                positions[sym] = {"qty": qty_est, "entry_price": entry, "txid": txid, "timestamp": utc_now_iso()}
                save_positions(positions)
            else:
                # Try market with viqc spend; if blocked, retry as LIMIT IOC with slip
                resp_text, blocked = place_market_buy_usd(sym, BUY_USD, return_blocked=True)
                if blocked:
                    limit_price = quote * (1.0 + SLIP_PCT / 100.0)  # cross the ask to fill
                    qty_lim = BUY_USD / limit_price
                    print(f"  ⚠️ Market blocked → retry LIMIT IOC at {limit_price:.10f} (~{qty_lim:.6f})")
                    txid = place_limit_buy_usd_ioc(sym, qty_lim, limit_price)
                else:
                    txid = resp_text  # already contains txid JSON; fine to store

                # Use live quote as entry proxy; real fill is in Kraken history
                entry = get_public_quote(sym) or quote
                positions[sym] = {"qty": qty_est, "entry_price": entry, "txid": txid, "timestamp": utc_now_iso()}
                save_positions(positions)

    # === Wrap-up ===
    summary = {
        "positions": positions,
        "holding": list(positions.keys()),
        "buy_usd": BUY_USD,
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "slip_pct": SLIP_PCT,
        "window_min": WINDOW_MIN,
        "num_candidates": len(candidates),
        "notes": "USD-only trading; viqc buys; portfolio reconciliation + IOC fallback",
    }
    log_summary(summary)
    print("✅ Cycle complete.")

if __name__ == "__main__":
    main()
