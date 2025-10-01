# trader/crypto_engine.py
# Auto-pick Top-K USD pairs from Kraken using CCXT, with verbose reasons when empty.
# Runs in package mode via launcher (python -m trader.crypto_engine) and also works as a plain script.

from __future__ import annotations
import os, sys, math
from datetime import datetime, timezone
from typing import List, Dict, Tuple

# ---- robust import of broker ----
try:
    from trader.broker_crypto_ccxt import CCXTCryptoBroker  # when run as a package/module
except ModuleNotFoundError:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from broker_crypto_ccxt import CCXTCryptoBroker  # type: ignore

# ---------- helpers ----------
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def as_bool(s: str | None, default: bool = True) -> bool:
    if s is None:
        return default
    s = s.strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def as_float(s: str | None, default: float) -> float:
    try:
        return float(s) if s is not None else default
    except Exception:
        return default

def as_int(s: str | None, default: int) -> int:
    try:
        return int(s) if s is not None else default
    except Exception:
        return default

def read_env() -> dict:
    return {
        "DRY_RUN": as_bool(os.getenv("DRY_RUN", "true")),
        "EXCHANGE_ID": os.getenv("EXCHANGE_ID", "kraken"),
        "BASE": os.getenv("BASE_CURRENCY", "USD"),
        "UNIVERSE": os.getenv("UNIVERSE", "auto"),
        "MAX_POSITIONS": as_int(os.getenv("MAX_POSITIONS", "4"), 4),
        # Auto-pick knobs
        "AUTO_TOP_K": as_int(os.getenv("AUTO_TOP_K", os.getenv("MAX_POSITIONS", "4")), 4),
        "AUTO_MIN_USD_VOL": as_float(os.getenv("AUTO_MIN_USD_VOL", "2000000"), 2_000_000.0),  # ~$2M 24h
        "AUTO_MIN_PRICE": as_float(os.getenv("AUTO_MIN_PRICE", "0.05"), 0.05),               # avoid dust-priced
        "AUTO_EXCLUDE": os.getenv("AUTO_EXCLUDE", "USDT/USD,USDC/USD,EUR/USD,GBP/USD,USD/USD"),
    }

# ---------- universe builder ----------
def build_auto_universe(ex, base_quote: str, top_k: int, min_usd_vol: float, min_price: float, exclude_symbols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (picks, reasons). 'reasons' accumulates explanations if empty.
    """
    reasons: List[str] = []
    try:
        ex.load_markets()
    except Exception as e:
        return [], [f"load_markets failed: {e}"]

    # Only USD-quoted symbols
    usd_symbols = [s for s, m in ex.markets.items()
                   if isinstance(s, str) and s.endswith(f"/{base_quote}")]

    if not usd_symbols:
        return [], [f"no /{base_quote} symbols found on exchange"]

    # Exclusions
    excl = set([x.strip().upper() for x in exclude_symbols])
    usd_symbols = [s for s in usd_symbols if s.upper() not in excl and not s.upper().startswith(f"{base_quote}/")]

    if not usd_symbols:
        return [], ["all USD symbols filtered by exclusions"]

    # Fetch tickers in bulk
    try:
        tickers = ex.fetch_tickers(usd_symbols)
    except Exception as e:
        # Some exchanges rate-limit bulk; fallback to few
        reasons.append(f"fetch_tickers bulk failed: {e}; trying single fetches up to 50")
        tickers = {}
        for s in usd_symbols[:50]:
            try:
                t = ex.fetch_ticker(s)
                tickers[s] = t
            except Exception:
                continue

    if not tickers:
        return [], ["no tickers returned"]

    # Score symbols
    scored: List[Tuple[str, float, float, float]] = []  # (symbol, score, pct, usd_vol)
    for sym, t in tickers.items():
        last = t.get("last") or t.get("close")
        pct = t.get("percentage")
        base_vol = t.get("baseVolume")
        quote_vol = t.get("quoteVolume")

        try:
            price = float(last) if last is not None else None
        except Exception:
            price = None

        if price is None or price < min_price:
            continue

        # best-effort USD volume
        try:
            usd_vol = float(quote_vol) if quote_vol is not None else (float(base_vol) * float(price) if base_vol is not None else None)
        except Exception:
            usd_vol = None

        if usd_vol is None or usd_vol < min_usd_vol:
            continue

        try:
            pct_val = float(pct) if pct is not None else 0.0
        except Exception:
            pct_val = 0.0

        # Combined score: momentum × log(volume)
        score = pct_val * math.log(max(usd_vol, 1.0) + 1.0)
        scored.append((sym, score, pct_val, usd_vol))

    if not scored:
        reasons.append(f"no symbols passed filters (min_price={min_price}, min_usd_vol={min_usd_vol})")

    # rank by score desc, then pct desc, then volume desc
    scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    picks = [s for s, _, _, _ in scored[:max(1, top_k)]]

    return picks, reasons

# ---------- main ----------
def main() -> int:
    env = read_env()

    print("=" * 74)
    print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧" if env["DRY_RUN"] else "🟢 LIVE TRADING")
    print("=" * 74)
    print(f"{now_utc()} INFO: Starting trader in CRYPTO mode. Dry run={env['DRY_RUN']}. Broker=ccxt")

    # Build broker (reads CCXT_* or KRAKEN_* envs, prints source)
    broker = CCXTCryptoBroker(exchange_id=env["EXCHANGE_ID"], dry_run=env["DRY_RUN"])

    usd = 0.0
    try:
        broker.load_markets()
        usd = broker.usd_cash()
        print(f"{now_utc()} INFO: [ccxt] USD/ZUSD balance detected: ${usd:,.2f}")
    except Exception as e:
        print(f"{now_utc()} WARN: Could not fetch USD/ZUSD balance: {e}")

    # Universe
    uni_env = env["UNIVERSE"]
    candidates: List[str] = []
    reasons: List[str] = []

    if uni_env.strip().lower() == "auto":
        try:
            ex = broker.ex  # underlying ccxt exchange
            exclude = [x for x in env["AUTO_EXCLUDE"].split(",") if x.strip()]
            candidates, reasons = build_auto_universe(
                ex=ex,
                base_quote=env["BASE"],
                top_k=env["AUTO_TOP_K"],
                min_usd_vol=env["AUTO_MIN_USD_VOL"],
                min_price=env["AUTO_MIN_PRICE"],
                exclude_symbols=exclude,
            )
            if candidates:
                print(f"{now_utc()} INFO: Universe (auto): top {len(candidates)} → {candidates}")
            else:
                print(f"{now_utc()} INFO: Universe (auto): none selected.")
                for r in reasons:
                    print(f"{now_utc()} INFO: reason: {r}")
                print(f"{now_utc()} INFO: tip: you can force a list via UNIVERSE=BTC/USD,ETH/USD,SOL/USD,DOGE/USD")
        except Exception as e:
            print(f"{now_utc()} WARN: auto-universe failed: {e}")
    else:
        # Manual list
        candidates = [u.strip() for u in uni_env.split(",") if u.strip()]
        print(f"{now_utc()} INFO: Universe (manual): {candidates}")

    # (No trading in DRY_RUN; just summarize)
    open_positions = 0
    cap_left = env["MAX_POSITIONS"] - open_positions
    print(f"{now_utc()} INFO: KPI SUMMARY | dry_run={env['DRY_RUN']} | open={open_positions} | cap_left={cap_left} | usd=${usd:,.2f}")
    print(f"{now_utc()} INFO: Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
