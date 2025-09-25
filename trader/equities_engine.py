#!/usr/bin/env python3
"""
Equities engine (Alpaca) — ranked + retry + logs + AUTO‑UNIVERSE + per‑symbol cap

What’s new vs your last version:
1) **AUTO‑UNIVERSE** option that builds a diversified megacap list and UNIONs it with current holdings.
2) **Per‑symbol exposure cap** so any single name can’t exceed MAX_PCT_PER_SYMBOL of account equity (pre‑check before buying).

Still included:
- Strength ranking (SMA5/SMA20 edge + RSI‑14 blend)
- MARKET buys sized by PER_TRADE_USD
- Bracket exits on entry: TP_PCT / SL_PCT
- AVOID_REBUY guard
- yfinance retry‑with‑backoff + per‑run cache
- Transparent logs of top candidates

ENV VARS (with sensible defaults):
  ALPACA_API_KEY, ALPACA_API_SECRET
  ALPACA_PAPER=1                     # 1=paper (default), 0=live
  UNIVERSE                           # CSV tickers, or the literal word "AUTO" to use auto‑universe
  MAX_NEW_ORDERS=4                   # fresh buys per run
  PER_TRADE_USD=2000                 # dollar notional per buy
  TP_PCT=0.035                       # take‑profit +3.5%
  SL_PCT=0.020                       # stop‑loss   −2.0%
  AVOID_REBUY=1                      # skip tickers already held
  LOG_TOP=10                         # how many ranked candidates to print in logs
  MAX_PCT_PER_SYMBOL=10              # hard cap per name as % of equity (set 0 to disable)
  AUTO_UNIVERSE_SIZE=40              # how many of the curated megacaps to keep when UNIVERSE=AUTO
"""
from __future__ import annotations
import os
import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("equities_engine")

@dataclass
class Pick:
    symbol: str
    last_close: float
    sma5: float
    sma20: float
    rsi14: float
    score: float

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

    universe_raw = os.getenv("UNIVERSE", "AUTO").strip()

    cfg = {
        "paper": paper_flag,
        "api_key": api_key,
        "api_secret": api_secret,
        "universe_raw": universe_raw,
        "max_new": env_int("MAX_NEW_ORDERS", 4),
        "per_trade_usd": env_float("PER_TRADE_USD", 2000.0),
        "tp_pct": env_float("TP_PCT", 0.035),
        "sl_pct": env_float("SL_PCT", 0.020),
        "avoid_rebuy": os.getenv("AVOID_REBUY", "1").strip() not in ("0", "false", "False"),
        "log_top": env_int("LOG_TOP", 10),
        "max_pct_symbol": env_float("MAX_PCT_PER_SYMBOL", 10.0),
        "auto_size": env_int("AUTO_UNIVERSE_SIZE", 40),
    }
    return cfg

# ---------- yfinance helpers (retry + cache) ----------
_dl_cache: Dict[Tuple[str, str, str], object] = {}
class YFTemporaryError(Exception):
    pass

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), retry=retry_if_exception_type(YFTemporaryError))
def yf_download(symbol: str, period: str, interval: str):
    key = (symbol, period, interval)
    if key in _dl_cache:
        return _dl_cache[key]
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise YFTemporaryError(f"empty df for {symbol}")
        _dl_cache[key] = df
        return df
    except Exception as e:
        raise YFTemporaryError(str(e))

def rsi(series, period: int = 14) -> float:
    s = series.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.rolling(period).mean()
    roll_down = loss.rolling(period).mean()
    rs = roll_up / roll_down
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    val = float(rsi_series.iloc[-1]) if not math.isnan(rsi_series.iloc[-1]) else float("nan")
    return val

# ---------- Alpaca helpers ----------
def alpaca_client(cfg) -> TradingClient:
    tc = TradingClient(cfg["api_key"], cfg["api_secret"], paper=cfg["paper"])
    end_key = cfg["api_key"][-4:] if len(cfg["api_key"]) >= 4 else "****"
    log.info("Alpaca key loaded (ending …%s); endpoint=%s.", end_key, "PAPER" if cfg["paper"] else "LIVE")
    return tc

def get_cash_positions_equity(tc: TradingClient) -> Tuple[float, List[str], float, Dict[str, float]]:
    acct = tc.get_account()
    cash = float(acct.cash)
    equity = float(acct.equity)
    positions = tc.get_all_positions()
    held = [p.symbol for p in positions]
    by_symbol_value = {p.symbol: float(p.market_value) for p in positions}
    return cash, held, equity, by_symbol_value

# ---------- Universe helpers ----------
_CURATED_MEGACAPS: List[str] = [
    # Tech + AI infra
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","AVGO","TSM","ASML","ADBE","CRM","ORCL","AMD","INTC","IBM","QCOM","SMCI","NOW","PANW","UBER",
    # Industrials / Energy / Materials
    "GE","CAT","DE","BA","LMT","NOC","XOM","CVX","COP","LIN","APD","DOW",
    # Health
    "UNH","LLY","JNJ","ABBV","MRK","PFE","TMO","DHR",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","V","MA","PYPL",
    # Staples / Discretionary / Telecom
    "PG","KO","PEP","COST","WMT","HD","LOW","MCD","NKE","SBUX","TMUS","VZ","T",
    # Software/security/additional large caps
    "SNOW","SHOP","NET","DDOG","ZS","CRWD","PLTR","INTU","TEAM"
]

def build_universe(cfg, held: Iterable[str]) -> List[str]:
    raw = cfg["universe_raw"]
    held_set = set(held)
    if raw.upper() == "AUTO":
        size = max(10, int(cfg["auto_size"]))
        base = _CURATED_MEGACAPS[:size]
        uni = list(dict.fromkeys(list(held_set) + base))  # preserve order, include holdings first
        log.info("AUTO‑UNIVERSE size=%d (holdings=%d included)", len(uni), len(held_set))
        return uni
    # CSV path
    uni = [s.strip().upper() for s in raw.split(",") if s.strip()]
    # Always ensure holdings are included (so maintenance/avoid_rebuy can reason about them)
    for s in held_set:
        if s not in uni:
            uni.append(s)
    return uni

# ---------- Scan + Rank logic ----------
def scan_and_rank(universe: List[str]) -> List[Pick]:
    """Return symbols where SMA5 > SMA20, scored by blended momentum strength."""
    picks: List[Pick] = []
    for sym in universe:
        try:
            df = yf_download(sym, period="90d", interval="1d")
            if df is None or len(df) < 25:
                continue
            close = df["Close"].astype(float)
            sma5 = close.rolling(5).mean()
            sma20 = close.rolling(20).mean()
            sma5_last = float(sma5.iloc[-1]) if not math.isnan(sma5.iloc[-1]) else float("nan")
            sma20_last = float(sma20.iloc[-1]) if not math.isnan(sma20.iloc[-1]) else float("nan")
            if math.isnan(sma5_last) or math.isnan(sma20_last) or sma5_last <= sma20_last:
                continue
            last_close = float(close.iloc[-1])
            rsi_last = rsi(close, 14)
            edge = (sma5_last / sma20_last) - 1.0
            rsi_edge = (rsi_last - 50.0) / 100.0 if not math.isnan(rsi_last) else 0.0
            score = 0.7 * edge + 0.3 * rsi_edge
            picks.append(Pick(sym, last_close, sma5_last, sma20_last, rsi_last, score))
        except YFTemporaryError as e:
            log.warning("scan: %s temporary download issue: %s", sym, e)
            continue
        except Exception as e:
            log.warning("scan: %s failed: %s", sym, e)
            continue
    picks.sort(key=lambda p: p.score, reverse=True)
    return picks

# ---------- Exposure cap check ----------
def would_exceed_symbol_cap(symbol: str, add_notional: float, equity: float, by_symbol_value: Dict[str, float], cap_pct: float) -> bool:
    if cap_pct <= 0:
        return False
    current = by_symbol_value.get(symbol, 0.0)
    max_allowed = equity * (cap_pct / 100.0)
    return (current + add_notional) > max_allowed

# ---------- Order placement ----------
def place_buy(tc: TradingClient, symbol: str, notional: float, tp_pct: float, sl_pct: float, mkt_tif: TimeInForce = TimeInForce.DAY):
    px_df = yf_download(symbol, period="5d", interval="1d")
    last = float(px_df["Close"].astype(float).iloc[-1])
    qty = max(1, int(notional // max(0.01, last)))
    tp_price = round(last * (1.0 + tp_pct), 2)
    sl_price = round(last * (1.0 - sl_pct), 2)
    log.info("BUY %s x%s @~$%.2f TP $%.2f (+%.1f%%) SL $%.2f (-%.1f%%).", symbol, qty, last, tp_price, tp_pct * 100, sl_price, sl_pct * 100)
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

    cash, held, equity, by_symbol_value = get_cash_positions_equity(tc)
    log.info("Cash: $%.2f | Equity: $%.2f | Open positions: %d -> %s", cash, equity, len(held), held)

    universe_all = build_universe(cfg, held)
    universe = universe_all
    if cfg["avoid_rebuy"] and held:
        held_set = set(held)
        universe = [s for s in universe_all if s not in held_set]

    # Scan + rank
    picks = scan_and_rank(universe)

    # Show top-ranked in logs for transparency
    if picks:
        top_n = picks[: cfg["log_top"]]
        pretty = [f"{p.symbol}(score={p.score:.4f}, RSI={p.rsi14:.1f}, edge={(p.sma5/p.sma20-1)*100:.2f}%)" for p in top_n]
        log.info("Top candidates: %s", pretty)

    placed = 0
    max_new = max(0, cfg["max_new"])
    for pick in picks:
        if placed >= max_new:
            break
        # Per‑symbol cap guard
        if would_exceed_symbol_cap(pick.symbol, cfg["per_trade_usd"], equity, by_symbol_value, cfg["max_pct_symbol"]):
            log.info("SKIP %s: per‑symbol cap %.1f%% of equity would be exceeded.", pick.symbol, cfg["max_pct_symbol"])
            continue
        try:
            place_buy(tc, symbol=pick.symbol, notional=cfg["per_trade_usd"], tp_pct=cfg["tp_pct"], sl_pct=cfg["sl_pct"])
            # update in‑memory position value to reflect the scheduled buy
            by_symbol_value[pick.symbol] = by_symbol_value.get(pick.symbol, 0.0) + cfg["per_trade_usd"]
            placed += 1
            time.sleep(0.25)
        except Exception as e:
            log.warning("order: %s failed: %s", pick.symbol, e)
            continue

    log.info("Run complete: placed %d order(s).", placed)

if __name__ == "__main__":
    main()
