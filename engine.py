"""
engine.py
Headless engine used by runner.py from GitHub Actions.

Features:
- Robust per‑ticker Yahoo Finance fetch (avoids multi-download flakiness)
- CoinGecko fallback (with multiple candidate IDs per symbol)
- Safe percentage‑return calc (guards zero/NaN/short history)
- Symbol fixes: POL-USD -> MATIC-USD, MIOTA-USD -> IOTA-USD
"""

from __future__ import annotations

import os
import math
import json
import datetime as dt
from typing import List, Tuple, Dict

import pandas as pd
import yfinance as yf
from pycoingecko import CoinGeckoAPI

# -------------------------
# Universe (adjust as you like)
# -------------------------
SYMBOLS: List[str] = [
    # equities (examples)
    "AAPL", "MSFT", "NVDA", "SPY",

    # crypto (Yahoo tickers)
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",

    # fixes for prior names
    "MATIC-USD",  # Polygon (POL rebrand)
    "IOTA-USD",   # was MIOTA-USD

    # meme/alt coins (we fetch via CoinGecko if YF misses)
    "PEPE-USD", "POPCAT-USD", "PENGU-USD", "MEW-USD",
]

# ======================================================
# Yahoo Finance: robust per‑ticker history fetch
# ======================================================
def fetch_prices_yf(
    symbols: List[str],
    period: str = "60d",
    interval: str = "1d",
) -> Tuple[pd.DataFrame, List[str]]:
    frames: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for s in symbols:
        try:
            hist = yf.Ticker(s).history(period=period, interval=interval, auto_adjust=True)
            if hist.empty or "Close" not in hist:
                missing.append(s)
                continue
            frames[s] = hist[["Close"]].rename(columns={"Close": s})
        except Exception:
            missing.append(s)

    if not frames:
        return pd.DataFrame(), missing

    df = pd.concat(frames.values(), axis=1, join="inner").sort_index()
    return df, missing

# ======================================================
# CoinGecko fallback (try multiple candidate IDs)
# ======================================================
cg = CoinGeckoAPI()

# Try these IDs in order for each symbol; we stop at the first that returns data.
CG_CANDIDATES: Dict[str, List[str]] = {
    # solid mappings:
    "PEPE-USD":   ["pepe"],
    "POPCAT-USD": ["popcat"],
    "MEW-USD":    ["cat-in-a-dogs-world", "mew"],

    # prior holdouts:
    "MATIC-USD":  ["matic-network", "polygon-ecosystem-token"],  # MATIC + POL rebrand
    "PENGU-USD":  ["pengu", "pengu-coin"],                       # try both
}

def cg_daily_series(coin_id: str, days: int = 60) -> pd.Series:
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="usd", days=days)
    prices = data.get("prices", [])
    if not prices:
        return pd.Series(dtype=float)
    # prices entries: [ms_timestamp, price]
    s = pd.Series({dt.datetime.utcfromtimestamp(p[0] / 1000).date(): p[1] for p in prices})
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

def fetch_missing_from_cg(missing_symbols: List[str], days: int = 60) -> Tuple[pd.DataFrame, List[str]]:
    frames: Dict[str, pd.DataFrame] = {}
    still: List[str] = []

    for sym in missing_symbols:
        candidates = CG_CANDIDATES.get(sym, [])
        got = False
        for cid in candidates:
            try:
                s = cg_daily_series(cid, days=days)
                if not s.empty:
                    frames[sym] = s.to_frame(name=sym)
                    got = True
                    break
            except Exception:
                # try next candidate
                pass
        if not got:
            still.append(sym)

    if not frames:
        return pd.DataFrame(), still

    df = pd.concat(frames.values(), axis=1, join="inner").sort_index()
    return df, still

# ======================================================
# Safe return calculation
# ======================================================
def safe_pct_return(close: pd.Series, lb: int) -> float:
    """Return % change over lookback lb, guarding zeros/NaNs/short history."""
    if len(close) <= lb:
        return float("nan")
    base = close.iloc[-(lb + 1)]
    last = close.iloc[-1]
    try:
        base = float(base)
        last = float(last)
    except Exception:
        return float("nan")
    if base == 0 or not math.isfinite(base) or not math.isfinite(last):
        return float("nan")
    return (last / base - 1.0) * 100.0

# ======================================================
# Core engine
# ======================================================
def run_engine(symbols: List[str] = SYMBOLS, lookback_days: int = 20) -> dict:
    # 1) Fetch from Yahoo
    df_yf, missing = fetch_prices_yf(symbols, period="60d", interval="1d")
    if missing:
        print(f"[YF] Missing/failed: {missing}")

    # 2) Fill missing via CoinGecko
    if missing:
        df_cg, still_missing = fetch_missing_from_cg(missing, days=60)
        if not df_cg.empty:
            if df_yf.empty:
                df_all = df_cg
            else:
                df_all = pd.concat([df_yf, df_cg], axis=1, join="inner").sort_index()
        else:
            df_all = df_yf
        if still_missing:
            print(f"[CG] Still missing after CoinGecko: {still_missing}")
    else:
        df_all = df_yf

    if df_all.empty:
        return {
            "ok": False,
            "reason": "No price data",
            "source": "engine",
            "live_trading": False,
            "login_ok": False,
        }

    # 3) Example calculation: simple lookback returns (optional for your strategy)
    rets: Dict[str, float] = {}
    for col in df_all.columns:
        r = safe_pct_return(df_all[col].dropna(), lb=lookback_days)
        rets[col] = r

    # 4) Return a compact status payload (extend with your signals if you like)
    return {
        "ok": True,
        "source": "engine",
        "live_trading": False,
        "login_ok": False,
        "lookback_days": lookback_days,
        "computed_returns": rets,  # remove if you don’t want this in logs
    }

# ======================================================
# CLI entry
# ======================================================
if __name__ == "__main__":
    result = run_engine()
    print(json.dumps(result))
