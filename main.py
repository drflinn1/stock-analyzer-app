"""
main.py — Crypto OHLCV via ccxt (Kraken) with equities fallback to yfinance

Highlights
- fetch_ohlc() first tries ccxt.kraken().fetch_ohlcv() for crypto symbols
- Keeps yfinance in repo so equities still work (or if user passes equity tickers)
- Supports autopick timeframes + graceful fallback if a timeframe isn’t supported
- Simple example strategy hook (RSI + Bollinger) to prove data flows end‑to‑end
- DRY_RUN guard for orders (placeholders only; no live trading issued here)

Requirements
pip install ccxt pandas numpy yfinance ta

Environment (optional)
- DRY_RUN=true|false (default true)
- TIMEFRAME=15m (overridden by --timeframe or autopick)
- LIMIT=300 (number of candles)
- AUTOPICK=true|false (default true)
- SYMBOLS=BTC/USD,ETH/USD (comma‑separated; can mix equities like SPY)
- KRAKEN_API_KEY / KRAKEN_SECRET (NOT required for public OHLCV)
"""
from __future__ import annotations
import os
import sys
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np

# Keep yfinance for equities path
import yfinance as yf

# Crypto via ccxt
import ccxt

try:
    import ta
except Exception:
    ta = None

# -----------------------------
# Config
# -----------------------------
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
DEFAULT_LIMIT = int(os.getenv("LIMIT", "300"))
DEFAULT_TIMEFRAME = os.getenv("TIMEFRAME", "15m")
AUTOPICK = os.getenv("AUTOPICK", "true").lower() == "true"
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USD,ETH/USD").split(",") if s.strip()]

# Fallback order for crypto timeframes if exchange doesn’t support the requested one
FALLBACK_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# If user passes bare crypto tickers like "BTC" or "ETH", expand to "/USD"
FIAT_QUOTE = os.getenv("FIAT_QUOTE", "USD")


# -----------------------------
# Utilities
# -----------------------------

def is_crypto_symbol(sym: str) -> bool:
    """Very light heuristic: if it contains '/', treat as crypto; else assume equity.
    You can also whitelist known crypto roots (BTC, ETH, SOL...) if you prefer.
    """
    return "/" in sym


def normalize_crypto_symbol(sym: str, exchange: ccxt.Exchange) -> str:
    """Normalize common crypto symbol input to something Kraken will accept.
    ccxt generally accepts 'BTC/USD' and maps internally to Kraken's XBT/USD.
    This function also expands bare bases like 'BTC' -> 'BTC/USD'.
    """
    if "/" not in sym:
        sym = f"{sym}/{FIAT_QUOTE}"
    # Load markets and prefer exact match; otherwise try case normalized
    exchange.load_markets()
    if sym in exchange.markets:
        return sym
    # Try uppercasing base (e.g., btc/usd -> BTC/USD)
    base, quote = sym.split("/")
    candidate = f"{base.upper()}/{quote.upper()}"
    if candidate in exchange.markets:
        return candidate
    # Try BTC->XBT remap if needed
    remap = {"BTC": "XBT"}
    base_alt = remap.get(base.upper(), base.upper())
    candidate = f"{base_alt}/{quote.upper()}"
    if candidate in exchange.markets:
        return candidate
    # Last resort: attempt to find the first market that startswith base and endswith quote
    for m in exchange.markets:
        try:
            b, q = m.split("/")
            if b.upper() in {base.upper(), base_alt} and q.upper() == quote.upper():
                return m
        except Exception:
            pass
    raise ValueError(f"Symbol '{sym}' not found on Kraken. Available examples: 'BTC/USD', 'ETH/USD'.")


# -----------------------------
# Data fetchers
# -----------------------------

def fetch_ccxt_ohlc(
    symbol: str,
    timeframe: str,
    limit: int = DEFAULT_LIMIT,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch OHLCV from Kraken via ccxt. Returns DataFrame with columns [open, high, low, close, volume]."""
    ex = ccxt.kraken({
        "enableRateLimit": True,
        # API keys optional for public
        "apiKey": os.getenv("KRAKEN_API_KEY", None),
        "secret": os.getenv("KRAKEN_SECRET", None),
    })

    # Normalize to an exact market Kraken understands
    market = normalize_crypto_symbol(symbol, ex)

    tf_chain = [timeframe] + [tf for tf in FALLBACK_TIMEFRAMES if tf != timeframe]
    last_err = None

    for tf in tf_chain:
        # Verify the timeframe is supported; if not, continue
        try:
            if hasattr(ex, "timeframes") and ex.timeframes and tf not in ex.timeframes:
                # Not supported, try next
                continue
        except Exception:
            pass

        for attempt in range(1, max_retries + 1):
            try:
                raw = ex.fetch_ohlcv(market, timeframe=tf, limit=limit)
                if not raw:
                    raise RuntimeError("Empty OHLCV returned")
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"]) 
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("UTC")
                df.set_index("timestamp", inplace=True)
                df.index.name = "time"
                df["symbol"] = market
                df["timeframe"] = tf
                return df
            except Exception as e:
                last_err = e
                # backoff
                time.sleep(0.5 * attempt)
        # try next fallback timeframe

    raise RuntimeError(f"Failed to fetch OHLCV for {symbol} after fallbacks. Last error: {last_err}")


def fetch_yf_ohlc(
    ticker: str,
    timeframe: str,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """Fallback for equities via yfinance. timeframe mapping is approximate."""
    # Map ccxt-like tf to yfinance interval
    tf_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "60m", "4h": "60m",  # yfinance lacks 4h; approximate with 60m and later resample if needed
        "1d": "1d"
    }
    interval = tf_map.get(timeframe, "15m")

    # Period must fit the requested limit; use a generous period to ensure >= limit rows
    period_by_interval = {
        "1m": "2d",
        "5m": "7d",
        "15m": "30d",
        "30m": "60d",
        "60m": "60d",
        "1d": "2y",
    }
    period = period_by_interval.get(interval, "30d")

    df = yf.download(tickers=ticker, interval=interval, period=period, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    df = df.rename(columns=str.lower)
    # If 4h requested, resample from 60m
    if timeframe == "4h" and interval == "60m":
        df = df.resample("4H").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    # Trim to limit
    if len(df) > limit:
        df = df.tail(limit)

    df.index.name = "time"
    df["symbol"] = ticker
    df["timeframe"] = timeframe
    return df


def fetch_ohlc(symbol: str, timeframe: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """Unified fetcher: crypto through ccxt Kraken; equities via yfinance."""
    if is_crypto_symbol(symbol):
        return fetch_ccxt_ohlc(symbol, timeframe=timeframe, limit=limit)
    else:
        return fetch_yf_ohlc(symbol, timeframe=timeframe, limit=limit)


# -----------------------------
# Timeframe selection
# -----------------------------

def choose_timeframe(symbol: str, preferred: str = DEFAULT_TIMEFRAME) -> str:
    """If AUTOPICK, try preferred first; if not supported for crypto, fall back.
    For equities, just return preferred.
    """
    if not AUTOPICK:
        return preferred

    if not is_crypto_symbol(symbol):
        return preferred

    ex = ccxt.kraken({"enableRateLimit": True})
    try:
        tfs = set((ex.timeframes or {}).keys())
        if preferred in tfs:
            return preferred
        for tf in FALLBACK_TIMEFRAMES:
            if tf in tfs:
                return tf
    except Exception:
        pass
    return preferred


# -----------------------------
# Simple indicators / signals
# -----------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if ta is None:
        # Minimal RSI/Bands without ta-lib
        # RSI (Wilder) approx
        window = 14
        delta = df["close"].diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=df.index).rolling(window).mean()
        roll_down = pd.Series(down, index=df.index).rolling(window).mean()
        rs = roll_up / (roll_down.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        df["rsi14"] = rsi.fillna(method="bfill")

        # Bollinger Bands 20
        ma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std(ddof=0)
        df["bb_mid"], df["bb_up"], df["bb_down"] = ma, ma + 2*std, ma - 2*std
        return df

    # Using "ta" package if available
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_mid"], df["bb_up"], df["bb_down"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
    return df


def generate_signal(df: pd.DataFrame) -> str:
    """Toy signal: Buy if close crosses above lower band with RSI rising; Sell if close tags upper band with RSI > 70."""
    if len(df) < 25:
        return "HOLD"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signal = "HOLD"
    if last["close"] > last["bb_down"] and prev["close"] <= prev["bb_down"] and last["rsi14"] > prev["rsi14"]:
        signal = "BUY"
    elif last["close"] >= last["bb_up"] and last["rsi14"] >= 70:
        signal = "SELL"
    return signal


# -----------------------------
# Order execution (Kraken via ccxt) — safe by default
# -----------------------------

BROKER = os.getenv("BROKER", "kraken").lower()
TRADE_USD = float(os.getenv("TRADE_USD", os.getenv("TRADE_SIZE", "0")))  # prefer TRADE_USD; fall back to TRADE_SIZE


def _kraken_client() -> ccxt.Exchange:
    return ccxt.kraken({
        "enableRateLimit": True,
        "apiKey": os.getenv("KRAKEN_API_KEY", None),
        "secret": os.getenv("KRAKEN_SECRET", None),
    })


def place_order(symbol: str, side: str, qty: float) -> None:
    """Places a MARKET order on Kraken when DRY_RUN is False, otherwise logs.
    qty is the base-asset amount (e.g., BTC amount for BTC/USD).
    """
    if DRY_RUN:
        print(f"[DRY_RUN] {side} {qty} of {symbol}")
        return
    if BROKER != "kraken":
        print(f"(LIVE) Broker '{BROKER}' not supported in this script yet.")
        return

    ex = _kraken_client()
    market = normalize_crypto_symbol(symbol, ex)

    try:
        # Ensure amount precision obeys exchange rules
        ex.load_markets()
        info = ex.market(market)
        precision = info.get("precision", {}).get("amount", 8)
        step = 10 ** -precision
        qty = max(step, math.floor(qty / step) * step)

        order = ex.create_order(market, type="market", side=side.lower(), amount=qty)
        print(f"(LIVE) Placed {side} {qty} {market} → order id: {order.get('id') or order}")
    except Exception as e:
        print(f"(LIVE) Order error for {market}: {e}")


# -----------------------------
# Runner

# -----------------------------

def run_once(symbols: List[str], timeframe: Optional[str] = None, limit: int = DEFAULT_LIMIT) -> None:
    for sym in symbols:
        tf = choose_timeframe(sym, preferred=(timeframe or DEFAULT_TIMEFRAME))
        print(f"\n=== {sym} @ {tf} ===")
        try:
            df = fetch_ohlc(sym, tf, limit=limit)
        except Exception as e:
            print(f"Failed to fetch {sym}: {e}")
            continue

        df = add_indicators(df)
        sig = generate_signal(df)
        last = df.iloc[-1]
        close = float(last["close"])
        rsi = float(last["rsi14"])
        print(f"Last close: {close:.2f} | RSI: {rsi:.2f} | Signal: {sig}")

        # Example position sizing: fixed tiny qty for demo
        if sig == "BUY":
            place_order(sym, "BUY", qty=0.001 if is_crypto_symbol(sym) else 1)
        elif sig == "SELL":
            place_order(sym, "SELL", qty=0.001 if is_crypto_symbol(sym) else 1)


def parse_args(argv: List[str]) -> Tuple[List[str], Optional[str], int]:
    # Very light argument parsing to keep dependencies minimal
    symbols = SYMBOLS
    timeframe = None
    limit = DEFAULT_LIMIT
    for i, tok in enumerate(argv):
        if tok == "--symbols" and i + 1 < len(argv):
            symbols = [s.strip() for s in argv[i+1].split(",") if s.strip()]
        elif tok == "--timeframe" and i + 1 < len(argv):
            timeframe = argv[i+1]
        elif tok == "--limit" and i + 1 < len(argv):
            limit = int(argv[i+1])
    return symbols, timeframe, limit


if __name__ == "__main__":
    syms, tf, lim = parse_args(sys.argv[1:])
    print(f"DRY_RUN={DRY_RUN} | AUTOPICK={AUTOPICK} | DEFAULT_TIMEFRAME={DEFAULT_TIMEFRAME} | LIMIT={lim}")
    print(f"Symbols: {syms}")
    run_once(syms, timeframe=tf, limit=lim)
