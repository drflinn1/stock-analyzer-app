"""
engine.py
Headless engine used by runner.py from GitHub Actions.

Adds:
- Robust per-ticker Yahoo Finance fetch (no multi-download flakiness)
- CoinGecko fallback for meme/unsupported tickers
- Safe percentage-return calc (guards against zero/NaN/short history)
- Minor symbol corrections: POL-USD -> MATIC-USD, MIOTA-USD -> IOTA-USD
"""

from __future__ import annotations
import os, math, json, datetime as dt
from typing import List, Tuple, Dict

import pandas as pd
import yfinance as yf
from pycoingecko import CoinGeckoAPI

# -------------------------
# Universe (adjust as you like)
# -------------------------
# If you maintain separate equity/crypto lists elsewhere, you can import them here.
SYMBOLS: List[str] = [
    # equities (examples)
    "AAPL", "MSFT", "NVDA", "SPY",
    # crypto (Yahoo tickers)
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",
    # fixes
    "MATIC-USD",  # Polygon (was POL-USD)
    "IOTA-USD",   # (was MIOTA-USD)
    # meme/alt coins we’ll fetch via CoinGecko when Yahoo fails:
    "PEPE-USD", "POPCAT-USD", "PENGU-USD", "MEW-USD",
]

# -------------------------
# Robust Yahoo Finance fetch
# -------------------------
def fetch_prices_yf(symbols: List[str], period: str = "60d", interval: str = "1d") -> Tuple[pd.DataFrame, List[str]]:
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

# -------------------------
# CoinGecko fallback (daily closes)
# -------------------------
cg = CoinGeckoAPI()

# Map Yahoo-like tickers to CoinGecko coin IDs
CG_MAP: Dict[str, str] = {
    "PEPE-USD":   "pepe",
    "POPCAT-USD": "popcat",
    # PENGU/M EW exact IDs can vary; these are best-known as of now.
    # If any still print as "missing", tell me and I’ll supply the exact ID.
    "PENGU-USD":  "pengu-coin",            # try "pengu-coin"; alternate might be "pengu"
    "MEW-USD":    "cat-in-a-dogs-world",   # ticker MEW on CG
}

def cg_daily_series(coin_id: str, days: int = 60) -> pd.Series:
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="usd", days=days)
    prices = data.get("prices", [])
    if not prices:
        return pd.Series(dtype=float)
    s = pd.Series({dt.datetime.utcfromtimestamp(p[0] / 1000).date(): p[1] for p in prices})
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s

def fetch_missing_from_cg(missing_symbols: List[str], days: int = 60) -> Tuple[pd.DataFrame, List[str]]:
    frames: Dict[str, pd.DataFrame] = {}
    still: List[str] = []
    for sym in missing_symbols:
        coin_id = CG_MAP.get(sym)
        if not coin_id:
            still.append(sym)
            continue
        try:
            s = cg_daily_series(coin_id, days=days)
            if s.empty:
                still.append(sym)
                continue
            frames[sym] = s.to_frame(name=sym)
        except Exception:
            still.append(sym)
    if not frames:
        return pd.DataFrame(), still
    df = pd.concat(frames.values(), axis=1, join="inner").sort_index()
    return df, still

# -------------------------
# Safe return calculation
# -------------------------
def safe_pct_return(close: pd.Series, lb: int) -> float:
    if len(close) <= lb:
        return float("nan")
    base = close.iloc[-(lb + 1)]
    last = close.iloc[-1]
    if base in (0, None) or not math.isfinite(float(base)) or not math.isfinite(float(last)):
        return float("nan")
    return (float(last) / float(base) - 1.0) * 100.0

# -------------------------
# Core engine (example)
# -------------------------
def run_engine(symbols: List[str] = SYMBOLS, lookback_days: int = 20) -> dict:
    # 1) Fetch from Yahoo
    df_yf, missing = fetch_prices_yf(symbols, period="60d", interval="1d")
    if missing:
        print(f"[YF] Missing/failed: {missing}")

    # 2) Fill from CoinGecko for any missing symbols
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
        return {"ok": False, "reason": "No price data", "source": "engine", "live_trading": False, "login_ok": False}

    # 3) Example: compute simple lookback returns safely (optional for your strategy)
    rets: Dict[str, float] = {}
    for col in df_all.columns:
        r = safe_pct_return(df_all[col].dropna(), lb=lookback_days)
        rets[col] = r

    # 4) Here you would apply your signal logic / rebalancing rules.
    # For headless confirmation we just report we ran successfully.
    out = {
        "ok": True,
        "source": "engine",
        "live_trading": False,
        "login_ok": False,
        "lookback_days": lookback_days,
        "computed_returns": rets,   # you can remove this if you don’t want it in logs
    }
    return out

# -------------------------
# CLI entry (called by runner.py import or direct)
# -------------------------
if __name__ == "__main__":
    result = run_engine()
    print(json.dumps(result))
