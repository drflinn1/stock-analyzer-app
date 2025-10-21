#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Force liquidation script for Kraken via ccxt.
- Sells all non-USD spot holdings at market.
- Skips stables if DUST_SKIP_STABLES=true.
- Skips tiny positions below DUST_MIN_USD; and below MIN_SELL_USD for actual sell.
- Honors DRY_RUN=true to only print actions.

Env (passed by workflow):
  KRAKEN_API_KEY, KRAKEN_API_SECRET, KRAKEN_API_PASSWORD (optional)
  DRY_RUN: "true" | "false"
  MIN_SELL_USD: default 10
  DUST_MIN_USD: default 2
  DUST_SKIP_STABLES: "true" | "false"
"""

import os
import sys
import time
from typing import Dict

import ccxt  # type: ignore

def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","on","yes")

def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def main():
    api_key = os.environ.get("KRAKEN_API_KEY", "")
    api_secret = os.environ.get("KRAKEN_API_SECRET", "")
    api_password = os.environ.get("KRAKEN_API_PASSWORD", None)

    if not api_key or not api_secret:
        print("[force] Missing Kraken API credentials. Aborting.")
        sys.exit(1)

    DRY_RUN = env_bool("DRY_RUN", True)
    MIN_SELL_USD = env_float("MIN_SELL_USD", 10.0)
    DUST_MIN_USD = env_float("DUST_MIN_USD", 2.0)
    DUST_SKIP_STABLES = env_bool("DUST_SKIP_STABLES", True)

    print(f"[force] DRY_RUN={DRY_RUN}  MIN_SELL_USD={MIN_SELL_USD}  DUST_MIN_USD={DUST_MIN_USD}  DUST_SKIP_STABLES={DUST_SKIP_STABLES}")

    ex = ccxt.kraken({
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_password or None,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    ex.load_markets()

    # Stables we won't sell into themselves
    STABLES = {"USD", "USDT", "USDC", "EUR", "GBP"}

    balances = ex.fetch_balance()
    tickers = ex.fetch_tickers()

    def usd_price(symbol: str) -> float:
        t = tickers.get(symbol, {})
        return float(t.get("last") or t.get("close") or 0.0)

    # Collect all non-USD spot holdings
    to_liquidate = []
    for sym, m in ex.markets.items():
        # we only care about USD-quoted spot markets
        if "/" not in sym or not sym.endswith("/USD"):
            continue
        base = m["base"]
        if DUST_SKIP_STABLES and base.upper() in STABLES:
            continue
        qty = float(balances.get(base, {}).get("total") or 0.0)
        if qty <= 0:
            continue
        price = usd_price(sym)
        usd_val = qty * price
        if usd_val < DUST_MIN_USD:
            print(f"[force] Skip dust {sym}: qty={qty:.8f} (~${usd_val:.2f})")
            continue
        to_liquidate.append((sym, qty, price, usd_val))

    if not to_liquidate:
        cash = float(balances.get("USD", {}).get("total") or 0.0)
        print(f"[force] Nothing to liquidate. USD balance=${cash:.2f}")
        return

    total_positions = 0
    sold_notional = 0.0

    for sym, qty, price, usd_val in to_liquidate:
        if usd_val < MIN_SELL_USD:
            print(f"[force] Skip small {sym}: ~${usd_val:.2f} < MIN_SELL_USD({MIN_SELL_USD})")
            continue

        # Precision & min notional
        qty = float(ex.amount_to_precision(sym, qty))
        m = ex.markets.get(sym, {})
        min_cost = float(((m.get("limits") or {}).get("cost") or {}).get("min") or 0.0)
        if min_cost and usd_val < min_cost:
            print(f"[force] Skip {sym}: ~${usd_val:.2f} < exchange min cost ${min_cost}")
            continue

        print(f"[force] SELL {sym} qty≈{qty} (~${usd_val:.2f}) @ ~${price:.6f}")
        if not DRY_RUN:
            try:
                ex.create_market_sell_order(sym, qty)
                total_positions += 1
                sold_notional += usd_val
                # polite rate-limit
                time.sleep(0.75)
            except Exception as e:
                print(f"[force] Sell failed for {sym}: {e}")

    balances_after = ex.fetch_balance()
    cash_after = float(balances_after.get("USD", {}).get("total") or balances_after.get("USD", {}).get("free") or 0.0)

    print(f"[summary] Positions sold: {total_positions}  Notional≈${sold_notional:.2f}")
    print(f"[summary] USD balance after liquidation: ${cash_after:.2f}")

if __name__ == "__main__":
    main()
