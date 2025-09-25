#!/usr/bin/env python3
"""
Equities engine (Alpaca)
- Scans a universe with a simple SMA5 > SMA20 momentum filter (daily bars)
- Places MARKET BUY orders sized by PER_TRADE_USD
- Adds bracket exits: Take‑Profit (TP_PCT) and Stop‑Loss (SL_PCT)
- Avoids rebuying symbols you already hold when AVOID_REBUY=1
- Explicit yfinance settings to silence deprecation warnings

ENV VARS (with sensible defaults):
  ALPACA_API_KEY, ALPACA_API_SECRET  # required
  ALPACA_PAPER=1                     # 1=paper (default), 0=live
  UNIVERSE="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AVGO,LIN,ADBE,INTC,AMD,ORCL,IBM,GE,UNH,WMT,XOM"
  MAX_NEW_ORDERS=4                   # max fresh buys per run
  PER_TRADE_USD=2000                 # dollar notional per buy
  TP_PCT=0.035                       # take‑profit +3.5%
  SL_PCT=0.020                       # stop‑loss   −2.0%
  AVOID_REBUY=1                      # skip tickers already held

Notes:
- Uses bracket orders via alpaca‑py 0.42.x
- yfinance auto_adjust is set explicitly to True to match its new default
  and to avoid price gaps from splits/dividends.
"""
from __future__ import annotations
import os
import math
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple

import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest


# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
log = logging.getLogger("equities_engine")


@dataclass
class Pick:
    symbol: str
    last_close: float


# ---------- Config helpers ----------
def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default)).strip()))
    except Exception:
        return default


def load_config():
    paper_flag = os.getenv("ALPACA_PAPER", "1").strip() not in ("0", "false", "False")
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    api_secret = os.getenv("ALPACA_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise SystemExit("Missing ALPACA_API_KEY/ALPACA_API_SECRET in environment")

    universe_env = os.getenv(
        "UNIVERSE",
        "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AVGO,LIN,ADBE,INTC,AMD,ORCL,IBM,GE,UNH,WMT,XOM",
    )
    universe = [s.strip().upper() for s in universe_env.split(",") if s.strip()]

    cfg = {
        "paper": paper_flag,
        "api_key": api_key,
        "api_secret": api_secret,
        "universe": universe,
        "max_new": env_int("MAX_NEW_ORDERS", 4),
        "per_trade_usd": env_float("PER_TRADE_USD", 2000.0),
        "tp_pct": env_float("TP_PCT", 0.035),
        "sl_pct": env_float("SL_PCT", 0.020),
        "avoid_rebuy": os.getenv("AVOID_REBUY", "1").strip() not in ("0", "false", "False"),
    }
    return cfg


# ---------- Alpaca helpers ----------
def alpaca_client(cfg) -> TradingClient:
    tc = TradingClient(cfg["api_key"], cfg["api_secret"], paper=cfg["paper"])
    # Obfuscate key for logs
    end_key = cfg["api_key"][-4:] if len(cfg["api_key"]) >= 4 else "****"
    log.info("Alpaca key loaded (ending …%s); endpoint=%s.", end_key, "PAPER" if cfg["paper"] else "LIVE")
    return tc


def get_cash_and_positions(tc: TradingClient) -> Tuple[float, List[str]]:
    acct = tc.get_account()
    # Prefer cash over buying_power for clarity, both are strings
    cash = float(acct.cash)
    held = [p.symbol for p in tc.get_all_positions()]
    return cash, held


# ---------- Scan logic ----------
def scan_candidates(universe: List[str]) -> List[Pick]:
    """Return symbols whose SMA5 > SMA20 on daily bars."""
    picks: List[Pick] = []
    for sym in universe:
        try:
            # Explicit settings to silence future warnings; download daily close
            df = yf.download(
                sym,
                period="60d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is None or len(df) < 20:
                continue
            close = df["Close"].astype(float)
            sma5 = close.rolling(5).mean()
            sma20 = close.rolling(20).mean()

            sma5_last = float(sma5.iloc[-1]) if not math.isnan(sma5.iloc[-1]) else float("nan")
            sma20_last = float(sma20.iloc[-1]) if not math.isnan(sma20.iloc[-1]) else float("nan")
            if math.isnan(sma5_last) or math.isnan(sma20_last):
                continue

            last_close = float(close.iloc[-1])
            if sma5_last > sma20_last:
                picks.append(Pick(sym, last_close))
        except Exception as e:
            log.warning("scan: %s failed: %s", sym, e)
            continue
    return picks


# ---------- Order placement ----------
def place_buy(tc: TradingClient, symbol: str, notional: float, tp_pct: float, sl_pct: float, mkt_tif: TimeInForce = TimeInForce.DAY):
    # Fetch a fresh price via yfinance to size qty conservatively
    px_df = yf.download(symbol, period="5d", interval="1d", auto_adjust=True, progress=False)
    if px_df is None or px_df.empty:
        raise RuntimeError(f"no price data for {symbol}")
    last = float(px_df["Close"].astype(float).iloc[-1])
    qty = max(1, int(notional // max(0.01, last)))

    tp_price = round(last * (1.0 + tp_pct), 2)
    sl_price = round(last * (1.0 - sl_pct), 2)

    log.info("BUY %s x%s @~$%.2f TP $%.2f (+%.1f%%) SL $%.2f (-%.1f%%).", symbol, qty, last, tp_price, tp_pct*100, sl_price, sl_pct*100)

    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=mkt_tif,
        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=sl_price),
    )
    resp = tc.submit_order(req)
    log.info("Submitted BUY -> id=%s symbol=%s qty=%s", getattr(resp, "id", "?"), symbol, qty)


# ---------- Main ----------
def main():
    cfg = load_config()
    tc = alpaca_client(cfg)

    cash, held = get_cash_and_positions(tc)
    log.info("Cash: $%.2f | Open positions: %d -> %s", cash, len(held), held)

    universe = cfg["universe"]
    if cfg["avoid_rebuy"] and held:
        universe = [s for s in universe if s not in set(held)]

    # Scan
    candidates = scan_candidates(universe)

    # Sort strongest by how far SMA5 is above SMA20 (simple proxy) if we captured that;
    # For now just keep original order (universe ordering) to stay deterministic.

    # Place up to MAX_NEW_ORDERS
    placed = 0
    max_new = max(0, cfg["max_new"])
    for pick in candidates:
        if placed >= max_new:
            break
        try:
            place_buy(
                tc,
                symbol=pick.symbol,
                notional=cfg["per_trade_usd"],
                tp_pct=cfg["tp_pct"],
                sl_pct=cfg["sl_pct"],
            )
            placed += 1
            # small pause to avoid bursts
            time.sleep(0.25)
        except Exception as e:
            log.warning("order: %s failed: %s", pick.symbol, e)
            continue

    log.info("Run complete: placed %d order(s).", placed)


if __name__ == "__main__":
    main()
