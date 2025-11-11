#!/usr/bin/env python3
"""
main.py — USD-only 1-Coin Rotation (Kraken)

Now guaranteed to unstick:
• If Kraken has ONLY USD (no non-USD balances), we auto-clear a stale .state/positions.json
• Reconciliation still auto-sells any untracked USD-quoted holdings >= DUST_MIN_USD
• Market buy with viqc; LIMIT IOC fallback using SLIP_PCT when blocked
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

BUY_USD      = _env_float("BUY_USD", 30.0)
TP_PCT       = _env_float("TP_PCT", 5.0)
SL_PCT       = _env_float("SL_PCT", 1.0)
SLIP_PCT     = _env_float("SLIP_PCT", 5.0)
WINDOW_MIN   = _env_float("WINDOW_MIN", 60.0)
DRY_RUN      = _env_str("DRY_RUN", "OFF").upper()
DUST_MIN_USD = _env_float("DUST_MIN_USD", 0.50)

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
        if q and q > 0:
            return pair, q
    return None

def reconcile_portfolio(positions: Dict[str, dict]) -> Dict[str, dict]:
    balances = kraken_list_balances()  # excludes zeroes
    print("🧹 Reconciling… balances:", balances)

    # A) If there are ZERO non-USD assets live, drop any stale tracked positions.
    nonusd = {a for a, v in balances.items()
              if a.upper() not in ("USD", "ZUSD", "USDT", "EUR", "ZEUR")}
    if not nonusd and positions:
        print("🔧 Kraken shows only USD — clearing stale positions.json")
        positions.clear()
        save_positions(positions)

    # B) Sell any untracked USD-quoted holdings >= dust
    for asset, qty in balances.items():
        if asset.upper() in ("USD", "ZUSD", "USDT", "EUR", "ZEUR"):
            continue
        qty = float(qty or 0.0)
        if qty <= 0: continue

        pair = asset_to_usd_pair(asset)
        if not is_usd_pair(pair): continue

        q = get_public_quote(pair)
        if not q:
            print(f"  ⚠️ No quote for {pair}, skip.")
            continue

        usd_value = qty * q
        if usd_value >= DUST_MIN_USD:
            print(f"  Untracked {asset} ≈ ${usd_value:.2f} → SELL")
            if DRY_RUN == "ON":
                print(f"  (DRY_RUN) would SELL {pair} qty {qty}")
            else:
                try:
                    place_market_sell_qty(pair, qty)
                except Exception as e:
                    print(f"  SELL error {pair}: {e}")

    # C) Drop any tracked symbols whose live qty is zero (in case a tiny leftover got sold)
    for sym in list(positions.keys()):
        base = base_from_pair(sym).upper()
        live_qty = float(balances.get(base, 0.0) or 0.0)
        if live_qty <= 1e-12:
            print(f"  Removing stale tracked {sym} (live qty 0).")
            positions.pop(sym, None)
    if positions is not None:
        save_positions(positions)

    return positions

def main():
    print(f"[{datetime.now().isoformat()}] Start {'DRY' if DRY_RUN=='ON' else 'LIVE'} cycle")
    print(f"BUY_USD={BUY_USD} TP_PCT={TP_PCT} SL_PCT={SL_PCT} SLIP_PCT={SLIP_PCT} WINDOW_MIN={WINDOW_MIN}")

    positions = load_positions()
    candidates = load_candidates()

    # Reconcile & self-heal
    try:
        positions = reconcile_portfolio(positions)
    except Exception as e:
        print(f"⚠️ Reconcile failed: {e}")

    # SELL (TP/SL) for current holding
    holding = list(positions.keys())
    if holding:
        sym = holding[0]
        q = get_public_quote(sym)
        if q:
            entry = float(positions[sym]["entry_price"])
            chg = (q - entry) / entry * 100
            print(f"Holding {sym} | entry={entry:.10f} now={q:.10f} change={chg:.2f}%")
            if chg >= TP_PCT:
                print("🎯 TP hit → SELL")
                if DRY_RUN == "ON":
                    print(f"(DRY_RUN) would SELL {sym} qty {positions[sym]['qty']}")
                    positions.pop(sym, None)
                else:
                    place_market_sell_qty(sym, float(positions[sym]["qty"]))
                    positions.pop(sym, None)
                save_positions(positions)
            elif chg <= -SL_PCT:
                print("🛑 SL hit → SELL")
                if DRY_RUN == "ON":
                    print(f"(DRY_RUN) would SELL {sym} qty {positions[sym]['qty']}")
                    positions.pop(sym, None)
                else:
                    place_market_sell_qty(sym, float(positions[sym]["qty"]))
                    positions.pop(sym, None)
                save_positions(positions)
        else:
            print(f"⚠️ Unable to quote {sym}; hold.")

    # BUY (when flat)
    positions = load_positions()
    reason = "unknown"
    if not positions:
        pick = select_top_usd_candidate(candidates)
        if not pick:
            reason = "no_valid_candidate"
            print("⚠️ No valid USD candidate")
        else:
            sym, quote = pick
            usd_bal = get_usd_balance()
            print(f"Top candidate: {sym} @ {quote:.10f} | USD balance={usd_bal:.2f}")
            if usd_bal <= 1.0:
                reason = "no_cash"
                print("⚠️ No spendable USD")
            else:
                spend = min(BUY_USD, max(0.0, usd_bal - 0.50))
                if spend < 1.0:
                    reason = "insufficient_after_reserve"
                else:
                    qty_est = spend / quote
                    if DRY_RUN == "ON":
                        print(f"(DRY_RUN) BUY {sym} spend ${spend:.2f} ~{qty_est:.6f}")
                        positions[sym] = {"qty": qty_est, "entry_price": quote, "txid": "DRY-RUN", "timestamp": utc_now_iso()}
                        save_positions(positions)
                        reason = "dry_buy"
                    else:
                        text, blocked = place_market_buy_usd(sym, spend, return_blocked=True)
                        if blocked:
                            limit_price = quote * (1 + SLIP_PCT/100.0)
                            qty_lim = spend / limit_price
                            print(f"  Market blocked → LIMIT IOC {limit_price:.10f} (~{qty_lim:.6f})")
                            txid = place_limit_buy_usd_ioc(sym, qty_lim, limit_price)
                            reason = "limit_ioc_fallback"
                        else:
                            txid = text
                            reason = "market_ok"
                        entry = get_public_quote(sym) or quote
                        positions[sym] = {"qty": qty_est, "entry_price": entry, "txid": txid, "timestamp": utc_now_iso()}
                        save_positions(positions)
    else:
        reason = "already_holding"

    log_summary({
        "positions": positions,
        "holding": list(positions.keys()),
        "buy_usd": BUY_USD,
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "slip_pct": SLIP_PCT,
        "window_min": WINDOW_MIN,
        "num_candidates": len(candidates),
        "notes": reason,
    })
    print("✅ Cycle complete. Reason:", reason)

if __name__ == "__main__":
    main()
