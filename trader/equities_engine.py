# FILE: trader/equities_engine.py
# Minimal equities engine for Alpaca:
# - Buys sized by PER_TRADE_USD (respects DAILY_CAP_USD and $1 min)
# - Position-aware sells: take-profit (TP_PCT) and ATR stop (ATR_MULT * ATR14)

from __future__ import annotations
import os, sys
from typing import Dict, List, Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf

from trader.broker_alpaca import AlpacaEquitiesBroker


# ---------- helpers ----------
def env_bool(n, d): return str(os.getenv(n, d)).strip().lower() in ("1", "true", "yes", "y")

def env_float(n, d):
    try: return float(os.getenv(n, str(d)))
    except Exception: return d

def env_list(n, d):
    return [t.strip().upper() for t in os.getenv(n, d).split(",") if t.strip()]


# ---------- data / indicators ----------
def fetch_ohlc(symbol: str, period="3mo", interval="1d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            return None
        return df
    except Exception:
        return None

def calc_atr14(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 15:
        return float("nan")
    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    return float(atr)


# ---------- raw Alpaca REST for positions/sells ----------
def _hdr() -> Dict[str,str]:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY",""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET",""),
        "Content-Type": "application/json",
    }

def _base() -> str:
    return os.getenv("ALPACA_BASE_URL","https://paper-api.alpaca.markets").rstrip("/")

def _safe_json(r: requests.Response):
    try: return r.json()
    except Exception: return {"text": r.text}

def list_positions() -> List[Dict]:
    try:
        r = requests.get(_base()+"/v2/positions", headers=_hdr(), timeout=20)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def get_position_map() -> Dict[str, Dict]:
    out = {}
    for p in list_positions():
        sym = (p.get("symbol") or "").upper()
        if sym:
            out[sym] = p
    return out

def close_position(symbol: str, qty: float|None=None, percentage: float|None=None) -> Dict:
    params = {}
    if qty is not None: params["qty"] = str(qty)
    if percentage is not None: params["percentage"] = str(percentage)
    try:
        r = requests.post(_base()+f"/v2/positions/{symbol}/close",
                          headers=_hdr(), params=params, timeout=20)
        return {"status_code": r.status_code, "json": _safe_json(r)}
    except Exception as e:
        return {"status": "error", "error": repr(e)}


# ---------- engine ----------
def run() -> int:
    dry         = env_bool("DRY_RUN", True)
    universe    = env_list("UNIVERSE", "SPY,AAPL,MSFT,TSLA,NVDA,AMD")
    per_trade   = env_float("PER_TRADE_USD", 25.0)
    daily_cap   = env_float("DAILY_CAP_USD", 50.0)
    min_notional= env_float("MIN_NOTIONAL_USD", 1.00)

    tp_pct      = env_float("TP_PCT", 0.02)      # 2% TP
    atr_mult    = env_float("ATR_MULT", 1.5)     # ATR stop multiplier
    max_buys    = int(os.getenv("MAX_BUYS_PER_RUN", "3"))

    print("=== EQUITIES ENGINE ===")
    print("DRY_RUN:", dry, "UNIVERSE:", universe)
    print("PER_TRADE_USD:", per_trade, "DAILY_CAP_USD:", daily_cap, "MIN_NOTIONAL_USD:", min_notional)
    print("TP_PCT:", tp_pct, "ATR_MULT:", atr_mult, "MAX_BUYS_PER_RUN:", max_buys)

    br = AlpacaEquitiesBroker(dry_run=dry)
    print("Ping:", br.ping())
    print("Cash:", br.get_cash())

    # ---------- SELL SIDE ----------
    pos_map = get_position_map()
    for sym, p in pos_map.items():
        if sym not in universe:
            continue
        try:
            qty = float(p["qty"])
            avg = float(p["avg_entry_price"])
        except Exception:
            continue

        last = br.get_latest_price(sym) or 0.0
        if not last or last <= 0 or qty <= 0:
            continue

        df  = fetch_ohlc(sym, "3mo", "1d")
        atr = calc_atr14(df)
        tp  = avg * (1 + tp_pct)
        stop= avg - (atr_mult * atr) if (atr==atr) else None  # NaN check

        decision = "HOLD"
        if last >= tp:
            decision = "TAKE_PROFIT"
        elif stop is not None and last <= stop:
            decision = "ATR_STOP"

        print(f"[SELLCHK] {sym} qty={qty:.4f} avg={avg:.4f} last={last:.4f} "
              f"TP={tp:.4f} stop={stop if stop else 'n/a'} atr={atr if atr==atr else 'n/a'} => {decision}")

        if decision != "HOLD":
            if dry:
                print(f"[DRY] Would close {sym} qty={qty}")
            else:
                print(f"[SELL] {sym} ->", close_position(sym))

    # ---------- BUY SIDE ----------
    if per_trade < min_notional:
        print(f"PER_TRADE_USD={per_trade} < MIN_NOTIONAL_USD={min_notional}; skipping buys.")
        return 0

    cap_remaining, buys = daily_cap, 0
    for sym in universe:
        if buys >= max_buys or cap_remaining < min_notional:
            break
        if sym in pos_map:
            continue  # already own it

        last = br.get_latest_price(sym) or 0.0
        if not last or last <= 0:
            print(f"[BUYCHK] {sym} no price; skip.")
            continue

        df = fetch_ohlc(sym, "3mo", "1d")
        if df is None or df.empty or len(df) < 30:
            print(f"[BUYCHK] {sym} insufficient OHLC; skip.")
            continue

        sma20 = df["Close"].rolling(20).mean().iloc[-1]
        atr   = calc_atr14(df)
        if not np.isfinite(sma20) or not np.isfinite(atr):
            print(f"[BUYCHK] {sym} indicators not finite; skip.")
            continue

        atr_pct = atr / last if last else 999
        if last < sma20 or atr_pct > 0.10:
            print(f"[BUYCHK] {sym} fails filters (px={last:.4f} < SMA20={sma20:.4f} or ATR%={atr_pct:.3f}>0.10); skip.")
            continue

        usd = min(per_trade, cap_remaining)
        if usd < min_notional: break

        if dry:
            print(f"[DRY BUY] {sym} ${usd:.2f} (px={last:.4f}, atr%={atr_pct:.3f})")
        else:
            print(f"[BUY] {sym} ${usd:.2f} ->", br.market_buy_usd(sym, usd))

        cap_remaining -= usd
        buys += 1

    print("Buys done:", buys, "Remaining cap:", cap_remaining)
    print("Engine complete.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
