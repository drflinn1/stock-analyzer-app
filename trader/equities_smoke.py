# FILE: trader/equities_smoke.py
# Minimal paper smoke test for Alpaca equities.
# - Reads env: DRY_RUN, UNIVERSE, PER_TRADE_USD
# - Prints latest prices
# - Simulates market buys (paper) using broker_alpaca

from __future__ import annotations
import os, sys
from typing import List

# Try normal package import first; fall back to adding repo root to sys.path
try:
    from trader.broker_alpaca import AlpacaEquitiesBroker
except ModuleNotFoundError:
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from trader.broker_alpaca import AlpacaEquitiesBroker

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y")

def main() -> int:
    dry = env_bool("DRY_RUN", True)
    uni = os.getenv("UNIVERSE", "SPY,AAPL,MSFT,TSLA,NVDA,AMD")
    per_trade = float(os.getenv("PER_TRADE_USD", "25"))

    tickers: List[str] = [t.strip().upper() for t in uni.split(",") if t.strip()]
    if not tickers:
        print("No tickers in UNIVERSE; nothing to do.")
        return 0

    print("=== EQUITIES SMOKE ===")
    print("DRY_RUN:", dry)
    print("UNIVERSE:", tickers)
    print("PER_TRADE_USD:", per_trade)

    b = AlpacaEquitiesBroker(dry_run=dry)
    print("Ping:", b.ping())
    print("Cash:", b.get_cash())

    for sym in tickers[:3]:  # sample first 3
        px = b.get_latest_price(sym)
        print(f"{sym} latest price:", px)
        if px:
            res = b.market_buy_usd(sym, per_trade)
            print("BUY result:", res)
        else:
            print(f"Skipping {sym}: no price")

    print("Smoke complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
