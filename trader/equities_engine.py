# FILE: trader/equities_engine.py
# A minimal equities engine for Alpaca:
# - Buys sized by PER_TRADE_USD while respecting DAILY_CAP_USD and a $1 min notional
# - Position-aware sells:
#     * Take-profit when price >= avg * (1 + TP_PCT)
#     * ATR stop when price <= avg - ATR_MULT * ATR(14)
#
# Uses broker_alpaca for ping, cash, latest price, and market_buy_usd.
# Uses direct Alpaca REST for listing/closing positions to avoid depending
# on wrapper method names for sells (more robust across repos).

from __future__ import annotations
import os, sys, time, math, datetime as dt
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Local broker wrapper (already in your repo)
from trader.broker_alpaca import AlpacaEquitiesBroker

# -------------------------------
# ENV HELPERS
# -------------------------------

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y")

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def env_list(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

# -------------------------------
# ATR & INDICATORS
# -------------------------------

def calc_atr14(df: pd.DataFrame) -> float:
    """Compute ATR(14) from a daily OHLC dataframe with columns [Open, High, Low, Close]."""
    if df is None or df.empty or len(df) < 15:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean().iloc[-1]
    return float(atr)

def fetch_ohlc(symbol: str, period="3mo", interval="1d") -> Optional[pd.DataFrame]:
    try:
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        # Normalize column names
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None
        data = data.rename(columns={c: c.capitalize() for c in data.columns})
        missing = {"Open","High","Low","Close"} - set(data.columns)
        if missing:
            return None
        return data
    except Exception:
        return None

# -------------------------------
# ALPACA REST (positions/sells)
# -------------------------------

def alpaca_headers() -> Dict[str,str]:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY",""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET",""),
        "Content-Type": "application/json",
    }

def alpaca_base() -> str:
    return os.getenv("ALPACA_BASE_URL","https://paper-api.alpaca.markets").rstrip("/")

def list_positions() -> List[Dict]:
    try:
        r = requests.get(alpaca_base()+"/v2/positions", headers=alpaca_headers(), timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def get_position_map() -> Dict[str, Dict]:
    pos = {}
    for p in list_positions():
        sym = p.get("symbol","").upper()
        if not sym:
            continue
        pos[sym] = p
    return pos

def close_position(symbol: str, qty: Optional[float]=None, percentage: Optional[float]=None) -> Dict:
    """Close a position via Alpaca REST. If qty and percentage are None, closes entire position."""
    params = {}
    if qty is not None:
        params["qty"] = str(qty)
    if percentage is not None:
        params["percentage"] = str(percentage)
    try:
        r = requests.post(alpaca_base()+f"/v2/positions/{symbol}/close", headers=alpaca_headers(), params=params, timeout=20)
        return {"status_code": r.status_code, "json": safe_json(r)}
    except Exception as e:
        return {"status": "error", "error": repr(e)}

def safe_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"text": r.text}

# -------------------------------
# ENGINE
# -------------------------------

def run() -> int:
    dry = env_bool("DRY_RUN", True)
    universe = env_list("UNIVERSE", "SPY,AAPL,MSFT,TSLA,NVDA,AMD")
    per_trade = env_float("PER_TRADE_USD", 25.0)
    daily_cap = env_float("DAILY_CAP_USD", 50.0)
    min_notional = env_float("MIN_NOTIONAL_USD", 1.00)

    # Sell parameters (tunable)
    tp_pct   = env_float("TP_PCT", 0.02)      # 2% take-profit by default
    atr_mult = env_float("ATR_MULT", 1.5)     # ATR-based stop
    max_buys = int(os.getenv("MAX_BUYS_PER_RUN","3"))

    print("=== EQUITIES ENGINE ===")
    print("DRY_RUN:", dry)
    print("UNIVERSE:", universe)
    print("PER_TRADE_USD:", per_trade, "DAILY_CAP_USD:", daily_cap, "MIN_NOTIONAL_USD:", min_notional)
    print("TP_PCT:", tp_pct, "ATR_MULT:", atr_mult)

    br = AlpacaEquitiesBroker(dry_run=dry)
    print("Ping:", br.ping())
    print("Cash:", br.get_cash())

    # ---------------- Sells (position-aware) ----------------
    pos_map = get_position_map()    # positions keyed by symbol
    for sym, p in pos_map.items():
        if sym not in universe:
            # manage only tickers within the run universe
            continue
        try:
            qty = float(p["qty"])
            avg = float(p["avg_entry_price"])
        except Exception:
            continue
        price = br.get_latest_price(sym) or 0.0
        if not price or price <= 0 or qty <= 0:
            continue

        # ATR stop / TP logic
        df = fetch_ohlc(sym, period="3mo", interval="1d")
        atr = calc_atr14(df)
        take_profit_price = avg * (1.0 + tp_pct)
        stop_price = avg - (atr_mult * atr) if (atr==atr) else None  # NaN check

        decision = "HOLD"
        if price >= take_profit_price:
            decision = "TAKE_PROFIT"
        elif stop_price is not None and price <= stop_price:
            decision = "ATR_STOP"

        print(f"[SELLCHK] {sym} qty={qty:.4f} avg={avg:.4f} last={price:.4f} "
              f"TP={take_profit_price:.4f} stop={stop_price if stop_price else 'n/a'} atr={atr if atr==atr else 'n/a'} "
              f"=> {decision}")

        if decision != "HOLD":
            if dry:
                print(f"[DRY] Would close {sym} qty={qty}")
            else:
                res = close_position(sym)  # close entire position
                print(f"[SELL] close_position({sym}) -> {res}")

    # ---------------- Buys (sized) ----------------
    if per_trade < min_notional:
        print(f"PER_TRADE_USD={per_trade} < MIN_NOTIONAL_USD={min_notional}; skipping buys.")
        return 0

    buys_remaining_usd = daily_cap
    buys_done = 0

    for sym in universe:
        if buys_done >= max_buys or buys_remaining_usd < min_notional:
            break

        if sym in pos_map:
            # Already holding -> skip buy for this run
            continue

        price = br.get_latest_price(sym) or 0.0
        if not price or price <= 0:
            print(f"[BUYCHK] {sym} no price, skip.")
            continue

        # Simple quality filter using SMA/ATR
        df = fetch_ohlc(sym, period="3mo", interval="1d")
        if df is None or df.empty or len(df) < 30:
            print(f"[BUYCHK] {sym} insufficient OHLC data, skip.")
            continue
        sma20 = df["Close"].rolling(20).mean().iloc[-1]
        atr = calc_atr14(df)
        if not np.isfinite(sma20) or not np.isfinite(atr):
            print(f"[BUYCHK] {sym} indicators not finite, skip.")
            continue

        # buy criteria: uptrend-ish and not extremely volatile
        atr_pct = atr / price if price else 999
        if price < sma20 or atr_pct > 0.10:  # >10% daily ATR -> too volatile
            print(f"[BUYCHK] {sym} fails filters: price={price:.4f} < SMA20={sma20:.4f} or ATR%={atr_pct:.3f}>0.10, skip.")
            continue

        usd = min(per_trade, buys_remaining_usd)
        if usd < min_notional:
            break

        if dry:
            print(f"[DRY BUY] {sym} ${usd:.2f} (price={price:.4f}, atr%={atr_pct:.3f})")
        else:
            res = br.market_buy_usd(sym, usd)
            print(f"[BUY] {sym} ${usd:.2f} -> {res}")

        buys_remaining_usd -= usd
        buys_done += 1

    print("Buys done:", buys_done, "Remaining cap:", buys_remaining_usd)
    print("Engine complete.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
