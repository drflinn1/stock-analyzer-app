# trader/crypto_engine.py
# Minimal engine that:
#  - Reads DRY_RUN/EXCHANGE_ID/BASE_CURRENCY/UNIVERSE from env
#  - Builds CCXT via CCXTCryptoBroker (accepts CCXT_* or KRAKEN_* keys)
#  - Prints balance & KPI SUMMARY
#  - Returns 0 (success) every run in DRY_RUN
from __future__ import annotations
import os, sys, time
from datetime import datetime, timezone

from trader.broker_crypto_ccxt import CCXTCryptoBroker

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def as_bool(s: str, default: bool = True) -> bool:
    if s is None:
        return default
    s = s.strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def read_env() -> dict:
    return {
        "DRY_RUN": as_bool(os.getenv("DRY_RUN", "true")),
        "EXCHANGE_ID": os.getenv("EXCHANGE_ID", "kraken"),
        "BASE": os.getenv("BASE_CURRENCY", "USD"),
        "UNIVERSE": os.getenv("UNIVERSE", "auto"),
        "MAX_POSITIONS": int(os.getenv("MAX_POSITIONS", "4")),
        "PER_TRADE_USD": float(os.getenv("PER_TRADE_USD", "25")),
        "RESERVE_USD": float(os.getenv("RESERVE_USD", "100")),
        "DAILY_LOSS_CAP_USD": float(os.getenv("DAILY_LOSS_CAP_USD", "40")),
    }

def main() -> int:
    env = read_env()

    print("=" * 74)
    print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧" if env["DRY_RUN"] else "🟢 LIVE TRADING")
    print("=" * 74)
    print(f"{now_utc()} INFO: Starting trader in CRYPTO mode. Dry run={env['DRY_RUN']}. Broker=ccxt")
    uni = env["UNIVERSE"]
    uni_list = [] if uni == "auto" else [u.strip() for u in uni.split(",") if u.strip()]
    print(f"{now_utc()} INFO: Universe ({env['BASE']}-only): {uni_list if uni_list else 'auto'}")

    # Build broker (reads CCXT_* or KRAKEN_*)
    broker = CCXTCryptoBroker(
        exchange_id=env["EXCHANGE_ID"],
        dry_run=env["DRY_RUN"],
    )

    usd = 0.0
    try:
        broker.load_markets()
        usd = broker.usd_cash()
        print(f"{now_utc()} INFO: [ccxt] USD/ZUSD balance detected: ${usd:,.2f}")
    except Exception as e:
        print(f"{now_utc()} WARN: Could not fetch USD/ZUSD balance: {e}")

    # (Placeholder selection logic; respects empty list if auto and no data)
    open_positions = 0
    cap_left = env["MAX_POSITIONS"] - open_positions
    if (uni != "auto") and uni_list:
        print(f"{now_utc()} INFO: Candidates: {len(uni_list)}  ({uni_list})")
    else:
        print(f"{now_utc()} INFO: No candidates (capacity left but universe empty/auto).")

    print(f"{now_utc()} INFO: KPI SUMMARY | dry_run={env['DRY_RUN']} | open={open_positions} | cap_left={cap_left} | usd=${usd:,.2f}")
    print(f"{now_utc()} INFO: Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
