# trader/equities_engine.py
# Equities engine for Alpaca Paper:
# - Auto-pick liquid symbols by 20d Average Dollar Volume (ADV)
# - AVOID_REBUY to skip names already held
# - Bracket orders with TAKE_PROFIT_PCT / STOP_LOSS_PCT
# - Optional trend-break sells (EMA12<EMA26 and RSI<50)
# - DRY_RUN printing

from __future__ import annotations
import os
import math
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Alpaca Trading
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, AssetClass
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)

# ---------- Utilities ----------

def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default

def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(float(env_str(name, str(default))))
    except Exception:
        return default

def env_bool(name: str, default: bool) -> bool:
    raw = env_str(name, str(default)).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")

def log(msg: str) -> None:
    print(msg, flush=True)

# ---------- Params from workflow ----------

DRY_RUN              = env_bool("DRY_RUN", False)
PER_TRADE_USD        = env_float("PER_TRADE_USD", 3.0)
DAILY_CAP_USD        = env_float("DAILY_CAP_USD", 12.0)
UNIVERSE_RAW         = env_str("UNIVERSE", "AAPL,MSFT")
AUTO_PICK            = env_bool("AUTO_PICK", True)
PICKS_PER_RUN        = env_int("PICKS_PER_RUN", 6)
MIN_ADV_USD          = env_float("MIN_ADV_USD", 2_000_000)
AVOID_REBUY          = env_bool("AVOID_REBUY", True)
STOP_LOSS_PCT        = env_float("STOP_LOSS_PCT", 0.04)
TAKE_PROFIT_PCT      = env_float("TAKE_PROFIT_PCT", 0.08)
SELL_ON_TREND_BREAK  = env_bool("SELL_ON_TREND_BREAK", True)
MARKET_ONLY          = env_bool("MARKET_ONLY", True)  # Provided as default in workflow

ALPACA_API_KEY  = env_str("ALPACA_API_KEY", "")
ALPACA_SECRET   = env_str("ALPACA_SECRET_KEY", "")
ALPACA_PAPER    = env_bool("ALPACA_PAPER", True)

if not ALPACA_API_KEY or not ALPACA_SECRET:
    log("ERROR: Missing Alpaca credentials (ALPACA_API_KEY/ALPACA_SECRET_KEY).")
    raise SystemExit(1)

trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET, paper=ALPACA_PAPER)

# ---------- Universe / Auto-pick ----------

# A core, liquid candidate set for auto-pick (large caps + popular ETFs).
CORE_CANDIDATES = [
    # Mega-cap tech
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO",
    # Large liquid ETFs
    "SPY","QQQ","VOO","VTI","IWM","DIA","XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI",
    # Other liquid names
    "NFLX","COST","AMD","INTC","ADBE","CRM","PYPL","CSCO","ORCL",
    "PEP","KO","JNJ","WMT","BAC","JPM","WFC","PFE","MRK","UNH",
    "GE","CAT","NKE","MCD","HD","T","VZ","DIS","ABNB","UBER",
]

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def fetch_hist(ticker: str, days: int = 60) -> pd.DataFrame:
    # yfinance returns last n calendar days; we request 3mo to be safe and slice
    df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.tail(days)

def average_dollar_volume(ticker: str, lookback: int = 20) -> float:
    df = fetch_hist(ticker, days=max(lookback + 5, 30))
    if df.empty:
        return 0.0
    # Use unadjusted close * volume (approximate)
    dv = (df["Close"] * df["Volume"]).tail(lookback).mean()
    return float(0.0 if pd.isna(dv) else dv)

def last_close(ticker: str) -> Optional[float]:
    df = fetch_hist(ticker, days=2)
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])

def ensure_list_from_csv(csv: str) -> List[str]:
    syms = [s.strip().upper() for s in csv.split(",") if s.strip()]
    return list(dict.fromkeys(syms))  # dedupe, keep order

def auto_pick_symbols(candidates: List[str], min_adv: float, limit: int) -> List[str]:
    rows = []
    for sym in candidates:
        try:
            adv = average_dollar_volume(sym, lookback=20)
            rows.append((sym, adv))
            log(f"[AUTO] ADV ${adv:,.0f} — {sym}")
        except Exception as e:
            log(f"[AUTO] ADV fail {sym}: {e}")
    df = pd.DataFrame(rows, columns=["symbol", "adv"]).sort_values("adv", ascending=False)
    picks = df[df["adv"] >= min_adv]["symbol"].head(limit).tolist()
    log(f"[AUTO] Picks by ADV≥{min_adv:,.0f}: {picks}")
    return picks

# ---------- Portfolio helpers ----------

def current_positions_set() -> set:
    try:
        positions = trading.get_all_positions()
        held = {p.symbol.upper() for p in positions if p.asset_class == AssetClass.US_EQUITY}
        return held
    except Exception as e:
        log(f"WARN: get_all_positions failed: {e}")
        return set()

def place_bracket_market_buy(symbol: str, notional_usd: float,
                             sl_frac: float, tp_frac: float) -> None:
    # Bracket needs target prices; approximate from last close
    px = last_close(symbol)
    if px is None or px <= 0:
        log(f"SKIP {symbol}: no valid last price.")
        return
    qty = max(0.0001, round(notional_usd / px, 4))  # fractional qty
    tp_price = round(px * (1 + tp_frac), 2)
    sl_price = round(px * (1 - sl_frac), 2)

    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=sl_price),
    )

    if DRY_RUN:
        log(f"DRY-RUN BUY {symbol} qty={qty} @mkt  TP={tp_price} SL={sl_price}")
        return

    try:
        order = trading.submit_order(req)
        log(f"BUY OK {symbol} -> id={order.id} qty={qty} TP={tp_price} SL={sl_price}")
    except Exception as e:
        log(f"BUY FAIL {symbol}: {e}")

def place_market_sell_all(symbol: str) -> None:
    # Sell entire position (by qty). Fetch qty from positions.
    try:
        positions = trading.get_all_positions()
        qty = None
        for p in positions:
            if p.symbol.upper() == symbol.upper():
                qty = float(p.qty)
                break
        if qty is None or qty <= 0:
            log(f"SELL SKIP {symbol}: no position qty")
            return

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        if DRY_RUN:
            log(f"DRY-RUN SELL {symbol} qty={qty}")
            return
        order = trading.submit_order(req)
        log(f"SELL OK {symbol} -> id={order.id} qty={qty}")
    except Exception as e:
        log(f"SELL FAIL {symbol}: {e}")

# ---------- Trend-break sells ----------

def evaluate_trend_break_and_sell(held: List[str]) -> None:
    if not SELL_ON_TREND_BREAK:
        log("Trend-break sells disabled.")
        return
    for sym in held:
        try:
            df = fetch_hist(sym, days=60)
            if df.empty or len(df) < 30:
                continue
            close = df["Close"]
            ema12 = ema(close, 12)
            ema26 = ema(close, 26)
            rs = rsi(close, 14)
            cond = (ema12.iloc[-1] < ema26.iloc[-1]) and (rs.iloc[-1] < 50)
            log(f"[TREND] {sym} EMA12={ema12.iloc[-1]:.2f} EMA26={ema26.iloc[-1]:.2f} "
                f"RSI={rs.iloc[-1]:.1f} -> break={cond}")
            if cond:
                place_market_sell_all(sym)
        except Exception as e:
            log(f"[TREND] {sym} eval fail: {e}")

# ---------- Main ----------

def main() -> None:
    log("=== Equities Engine Start ===")
    log(f"DRY_RUN={DRY_RUN} PER_TRADE_USD={PER_TRADE_USD} DAILY_CAP_USD={DAILY_CAP_USD}")
    log(f"AUTO_PICK={AUTO_PICK} PICKS_PER_RUN={PICKS_PER_RUN} MIN_ADV_USD={MIN_ADV_USD:,.0f}")
    log(f"AVOID_REBUY={AVOID_REBUY} SL={STOP_LOSS_PCT} TP={TAKE_PROFIT_PCT} "
        f"TREND_SELLS={SELL_ON_TREND_BREAK}")

    # Portfolio awareness
    held = current_positions_set()
    log(f"Held now ({len(held)}): {sorted(list(held))}")

    # Trend-break sells first (optional)
    if SELL_ON_TREND_BREAK and held:
        evaluate_trend_break_and_sell(sorted(list(held)))

    # Build buy list
    if AUTO_PICK:
        candidates = CORE_CANDIDATES
        picks = auto_pick_symbols(candidates, MIN_ADV_USD, PICKS_PER_RUN * 2)
    else:
        picks = ensure_list_from_csv(UNIVERSE_RAW)

    # Apply AVOID_REBUY
    final_buy = []
    for s in picks:
        if AVOID_REBUY and s in held:
            log(f"[SKIP] {s} (already held)")
            continue
        final_buy.append(s)
        if len(final_buy) >= PICKS_PER_RUN:
            break

    if not final_buy:
        log("No eligible buy candidates after filters.")
        return

    # Budget for this run
    max_buys_by_cap = int(DAILY_CAP_USD // max(0.01, PER_TRADE_USD))
    plan_count = max(0, min(len(final_buy), max_buys_by_cap))
    if plan_count <= 0:
        log(f"Budget too small for PER_TRADE_USD={PER_TRADE_USD}. Nothing to do.")
        return

    log(f"Plan buys: {plan_count} x ${PER_TRADE_USD} (cap ${DAILY_CAP_USD}) -> {final_buy[:plan_count]}")

    # Place buys
    for sym in final_buy[:plan_count]:
        place_bracket_market_buy(
            symbol=sym,
            notional_usd=PER_TRADE_USD,
            sl_frac=STOP_LOSS_PCT,
            tp_frac=TAKE_PROFIT_PCT,
        )
        # small pause to be gentle
        time.sleep(0.5)

    log("=== Equities Engine Done ===")

if __name__ == "__main__":
    main()
