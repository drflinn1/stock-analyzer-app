"""
Equities Engine — uses Alpaca Market Data bars (1D) for indicators.
- Fixes the missing 1D bar issue by switching from yfinance to Alpaca data.
- Adds a clear log line showing how many bars were fetched per symbol.
- Keeps your existing env-driven knobs:
    DRY_RUN=false|true
    PER_TRADE_USD=3
    DAILY_CAP_USD=12
    UNIVERSE=AAPL,MSFT
    MARKET_ONLY=true
- Places MARKET orders only when MARKET_ONLY=true (default true).

ENV for Alpaca (data & trading):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    ALPACA_PAPER=true|false   # optional, defaults to true

Notes:
- If keys are missing, we still try to run, but trading will be disabled and we'll warn.
- Indicator logic is intentionally simple/robust (EMA cross + RSI filter); adjust as needed.
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import pandas as pd

try:
    # Alpaca SDK (data)
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception as e:  # pragma: no cover
    # Defer import errors to runtime with a clearer message
    raise RuntimeError(
        "alpaca-py is required. Please 'pip install alpaca-py' in your workflow.") from e

# ------------------------------ logging -----------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("equities_engine")

# ------------------------------ config ------------------------------------

def getenv_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}

DRY_RUN = getenv_bool("DRY_RUN", True)
MARKET_ONLY = getenv_bool("MARKET_ONLY", True)
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "3"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "12"))
UNIVERSE = [s.strip().upper() for s in os.getenv("UNIVERSE", "AAPL,MSFT").split(",") if s.strip()]

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = getenv_bool("ALPACA_PAPER", True)

# indicator lookback safety
MIN_BARS = 120  # ensure enough bars for EMA/RSI

# ------------------------------ indicators --------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

# ------------------------------ data fetch --------------------------------

def make_data_client() -> StockHistoricalDataClient:
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        log.warning("ALPACA keys are missing; data client will still init, but trading will be disabled.")
    return StockHistoricalDataClient(ALPACA_API_KEY or "", ALPACA_SECRET_KEY or "")


def fetch_daily_bars(client: StockHistoricalDataClient, symbol: str, days: int = 400) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="raw",
        feed=None,  # let SDK choose (SIP if entitled, else free feed)
        limit=None,
    )
    bars = client.get_stock_bars(req)
    df = bars.df  # MultiIndex (symbol, timestamp)
    if df is None or df.empty:
        return pd.DataFrame()
    # For single symbol, drop the first level
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0, drop_level=True)
    df = df.reset_index().rename(columns={
        "timestamp": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    })
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df

# ------------------------------ trading -----------------------------------

@dataclass
class Position:
    symbol: str
    qty: float

class Broker:
    def __init__(self):
        self.enabled = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)
        self.client = None
        if self.enabled:
            self.client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        else:
            log.warning("Trading disabled: missing ALPACA_API_KEY/ALPACA_SECRET_KEY")

    def get_cash(self) -> float:
        if not self.enabled:
            return float("inf")  # allow planning
        acct = self.client.get_account()
        return float(acct.cash)

    def market_buy_usd(self, symbol: str, usd: float) -> Optional[str]:
        if DRY_RUN or not self.enabled:
            log.info(f"DRY_RUN or trading disabled → BUY {symbol} ${usd:.2f} (simulated)")
            return None
        req = MarketOrderRequest(
            symbol=symbol,
            notional=usd,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(req)
        return order.id

    def market_sell_all(self, symbol: str) -> Optional[str]:
        if DRY_RUN or not self.enabled:
            log.info(f"DRY_RUN or trading disabled → SELL ALL {symbol} (simulated)")
            return None
        # Use qty = 'all' via notional-less market sell requires position lookup
        pos = None
        try:
            for p in self.client.get_all_positions():
                if p.symbol == symbol:
                    pos = p
                    break
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
        if not pos:
            log.info(f"No position in {symbol} to sell.")
            return None
        req = MarketOrderRequest(
            symbol=symbol,
            qty=pos.qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(req)
        return order.id

# ------------------------------ strategy ----------------------------------

def generate_signal(df: pd.DataFrame) -> str:
    """Return 'BUY', 'SELL', or 'HOLD' based on EMA cross + RSI filter.
    - BUY: EMA(12) crosses above EMA(26) and RSI > 50
    - SELL: EMA(12) crosses below EMA(26) and RSI < 50
    """
    if df.empty or len(df) < MIN_BARS:
        return "HOLD"
    close = df["close"].astype(float)
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    r = rsi(close, 14)

    # Cross detection: look at last two points
    fast_now, fast_prev = float(ema12.iloc[-1]), float(ema12.iloc[-2])
    slow_now, slow_prev = float(ema26.iloc[-1]), float(ema26.iloc[-2])
    r_now = float(r.iloc[-1])

    crossed_up = fast_prev <= slow_prev and fast_now > slow_now
    crossed_down = fast_prev >= slow_prev and fast_now < slow_now

    if crossed_up and r_now > 50:
        return "BUY"
    if crossed_down and r_now < 50:
        return "SELL"
    return "HOLD"

# ------------------------------ engine ------------------------------------

def main() -> int:
    log.info("Starting Equities Engine (Alpaca bars)")
    log.info(f"Config → DRY_RUN={DRY_RUN} PER_TRADE_USD={PER_TRADE_USD} DAILY_CAP_USD={DAILY_CAP_USD} MARKET_ONLY={MARKET_ONLY}")
    log.info(f"Universe: {UNIVERSE}")

    data_client = make_data_client()
    broker = Broker()

    spend_left = DAILY_CAP_USD
    total_buys = 0
    total_sells = 0
    for symbol in UNIVERSE:
        try:
            df = fetch_daily_bars(data_client, symbol, days=420)
            log.info(f"Bars fetched for {symbol}: {len(df)} (need ≥{MIN_BARS})")  # <= explicit bar count line
            if df.empty or len(df) < MIN_BARS:
                log.warning(f"Skipping {symbol}: insufficient bars for indicators")
                continue

            signal = generate_signal(df)
            log.info(f"Signal for {symbol}: {signal}")

            if signal == "BUY" and spend_left >= PER_TRADE_USD:
                if MARKET_ONLY:
                    broker.market_buy_usd(symbol, PER_TRADE_USD)
                    spend_left -= PER_TRADE_USD
                    total_buys += 1
                else:
                    log.info("MARKET_ONLY=false not supported in this engine. Skipping.")

            elif signal == "SELL":
                if MARKET_ONLY:
                    broker.market_sell_all(symbol)
                    total_sells += 1
                else:
                    log.info("MARKET_ONLY=false not supported in this engine. Skipping.")

        except Exception as e:
            log.exception(f"Error processing {symbol}: {e}")

    log.info(f"Done. Buys placed: {total_buys}, Sells placed: {total_sells}, Daily spend remaining: ${spend_left:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
