#!/usr/bin/env python3
"""
Equities engine (Alpaca) — ranked + retry + logs + AUTO‑UNIVERSE + per‑symbol cap + trailing promotion + maintenance sweep

What’s included now:
- AUTO‑UNIVERSE (UNIVERSE=AUTO)
- Ranking: SMA5/SMA20 + RSI-14
- YFinance retry + per-run cache
- Per-symbol exposure cap
- Bracket buys (TP/SL)
- **Trailing-promote**: for winners above TRAIL_PROMOTE_PCT, place a protective stop (trailing-style via fixed stop level at current*(1-TRAIL_STOP_PCT)) if one doesn't exist.
- **Maintenance sweep**: every run, scan open positions and (optionally) repair missing TP/SL by placing limit (TP) and stop (SL) sell orders as GTC. This won’t close open positions immediately — TP is above market, SL is below market.

ENV VARS added:
  TRAIL_PROMOTE_PCT=0.05      # promote winners above +5%
  TRAIL_STOP_PCT=0.02         # protective stop distance; if true trailing is enabled, used as trail percent
  TRAIL_USE_TRUE=1            # 1=use Alpaca true trailing-stop orders when possible; 0=fallback to fixed STOP
  REPAIR_PROTECTION=1         # 1=place missing TP/SL orders, 0=only log


Notes/caveats:
- Trailing promotion prefers **true trailing-stop orders** when `TRAIL_USE_TRUE=1` and the API supports it. If unavailable, it falls back to a fixed STOP at `current*(1-TRAIL_STOP_PCT)`.
- Repair protection places two passive sell orders: LIMIT (TP above current) and STOP (SL below current). They are GTC and will only execute if price reaches them.
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
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    LimitOrderRequest,
    StopOrderRequest,
)
# try to import trailing-stop request if available
try:
    from alpaca.trading.requests import TrailingStopOrderRequest  # type: ignore
except Exception:  # pragma: no cover
    TrailingStopOrderRequest = None  # type: ignore

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
        # Trailing/maintenance
        "trail_promote_pct": env_float("TRAIL_PROMOTE_PCT", 0.05),
        "trail_stop_pct": env_float("TRAIL_STOP_PCT", 0.02),
        "trail_use_true": os.getenv("TRAIL_USE_TRUE", "1").strip() not in ("0", "false", "False"),
        "repair_protection": os.getenv("REPAIR_PROTECTION", "1").strip() not in ("0", "false", "False"),
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

def get_cash_positions_equity(tc: TradingClient) -> Tuple[float, List[str], float, Dict[str, float], List[object]]:
    acct = tc.get_account()
    cash = float(acct.cash)
    equity = float(acct.equity)
    positions = tc.get_all_positions()
    held = [p.symbol for p in positions]
    by_symbol_value = {p.symbol: float(p.market_value) for p in positions}
    return cash, held, equity, by_symbol_value, positions

# ---------- Universe helpers ----------
_CURATED_MEGACAPS: List[str] = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","AVGO","TSM","ASML","ADBE","CRM","ORCL","AMD","INTC","IBM","QCOM","SMCI","NOW","PANW","UBER",
    "GE","CAT","DE","BA","LMT","NOC","XOM","CVX","COP","LIN","APD","DOW",
    "UNH","LLY","JNJ","ABBV","MRK","PFE","TMO","DHR",
    "JPM","BAC","WFC","GS","MS","BLK","V","MA","PYPL",
    "PG","KO","PEP","COST","WMT","HD","LOW","MCD","NKE","SBUX","TMUS","VZ","T",
    "SNOW","SHOP","NET","DDOG","ZS","CRWD","PLTR","INTU","TEAM",
]

def build_universe(cfg, held: Iterable[str]) -> List[str]:
    raw = cfg["universe_raw"]
    held_set = set(held)
    if raw.upper() == "AUTO":
        size = max(10, int(cfg["auto_size"]))
        base = _CURATED_MEGACAPS[:size]
        uni = list(dict.fromkeys(list(held_set) + base))
        log.info("AUTO‑UNIVERSE size=%d (holdings=%d included)", len(uni), len(held_set))
        return uni
    uni = [s.strip().upper() for s in raw.split(",") if s.strip()]
    for s in held_set:
        if s not in uni:
            uni.append(s)
    return uni

# ---------- Scan + Rank logic ----------
def scan_and_rank(universe: List[str]) -> List[Pick]:
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

# ---------- Order placement (buys) ----------
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

# ---------- Maintenance & trailing promotion ----------
def open_orders_for_symbol(tc: TradingClient, symbol: str) -> List[object]:
    try:
        return tc.get_orders(status="open", symbols=[symbol])
    except Exception:
        return []

# How many shares of this symbol are already reserved by open SELL orders (limit/stop/bracket children)?
def reserved_sell_qty(tc: TradingClient, symbol: str) -> int:
    total = 0
    for o in open_orders_for_symbol(tc, symbol):
        try:
            if str(getattr(o, "side", "")).lower() == "sell":
                q = getattr(o, "qty", None)
                if q is None:
                    q = getattr(o, "quantity", None)
                if q is not None:
                    total += int(float(q))
        except Exception:
            continue
    return total


def position_unrealized_pct(position) -> float:
    # position.avg_entry_price and current_price may be strings
    try:
        entry = float(position.avg_entry_price)
        cur = float(position.current_price)
        return (cur - entry) / entry
    except Exception:
        return 0.0


def promote_to_trailing_stop(tc: TradingClient, position, cfg) -> None:
    symbol = position.symbol
    pos_qty = int(float(position.qty))
    if pos_qty <= 0:
        return
    # If any stop already exists, skip
    orders = open_orders_for_symbol(tc, symbol)
    has_stop = any((str(getattr(o, "side", "")).lower()=="sell") and (getattr(o, "type", "") in ("stop", "stop_loss", "trailing_stop") or getattr(o, "stop_price", None) or getattr(o, "trail_percent", None) or getattr(o, "trail_price", None)) for o in orders)
    if has_stop:
        log.info("PROMOTE %s: stop already exists, skipping", symbol)
        return
    # Determine available quantity not already reserved by other SELL orders
    reserved = reserved_sell_qty(tc, symbol)
    qty = max(0, pos_qty - reserved)
    if qty <= 0:
        log.info("PROMOTE %s: no free qty (reserved=%s), skipping", symbol, reserved)
        return

    # Prefer true trailing stop if available
    if cfg.get("trail_use_true", True) and TrailingStopOrderRequest is not None:
        try:
            trail_pct = round(cfg["trail_stop_pct"] * 100, 4)  # convert to percent
            req = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=trail_pct,
            )
            resp = tc.submit_order(req)
            log.info("PROMOTE %s: TRUE trailing-stop placed trail=%.4f%% qty=%s id=%s", symbol, trail_pct, qty, getattr(resp, "id", "?"))
            return
        except Exception as e:
            log.warning("PROMOTE %s: true trailing unsupported or failed (%s); falling back to fixed STOP.", symbol, e)

    # Fallback: fixed STOP at current*(1 - trail_stop_pct)
    df = yf_download(symbol, period="5d", interval="1d")
    last = float(df["Close"].astype(float).iloc[-1])
    stop_price = round(last * (1.0 - cfg["trail_stop_pct"]), 2)
    try:
        stop_req = StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC, stop_price=stop_price)
        resp = tc.submit_order(stop_req)
        log.info("PROMOTE %s: placed protective STOP at $%.2f qty=%s id=%s", symbol, stop_price, qty, getattr(resp, "id", "?"))
    except Exception as e:
        log.warning("PROMOTE %s failed to place fallback STOP: %s", symbol, e)


def repair_protection_if_missing(tc: TradingClient, position, cfg) -> None:
    symbol = position.symbol
    pos_qty = int(float(position.qty))
    if pos_qty <= 0:
        return
    orders = open_orders_for_symbol(tc, symbol)
    has_limit = any((str(getattr(o, "side", "")).lower()=="sell") and ((getattr(o, "type", "") == "limit") or getattr(o, "limit_price", None)) for o in orders)
    has_stop = any((str(getattr(o, "side", "")).lower()=="sell") and ((getattr(o, "type", "") in ("stop", "stop_loss", "trailing_stop")) or getattr(o, "stop_price", None) or getattr(o, "trail_percent", None) or getattr(o, "trail_price", None)) for o in orders)

    df = yf_download(symbol, period="5d", interval="1d")
    last = float(df["Close"].astype(float).iloc[-1])

    tp_price = round(last * (1.0 + cfg["tp_pct"]), 2)
    sl_price = round(last * (1.0 - cfg["sl_pct"]), 2)

    # compute available qty not already reserved by existing SELL orders
    reserved = reserved_sell_qty(tc, symbol)
    free_qty = max(0, pos_qty - reserved)

    if not has_limit and free_qty > 0:
        if cfg["repair_protection"]:
            try:
                lim = LimitOrderRequest(symbol=symbol, qty=free_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC, limit_price=tp_price)
                r = tc.submit_order(lim)
                log.info("REPAIR %s: placed LIMIT TP at $%.2f qty=%s id=%s", symbol, tp_price, free_qty, getattr(r, "id", "?"))
                reserved += free_qty
            except Exception as e:
                log.warning("REPAIR %s failed to place LIMIT TP: %s", symbol, e)
        else:
            log.info("REPAIR %s: missing LIMIT TP at $%.2f (not placed)", symbol, tp_price)

    free_qty = max(0, pos_qty - reserved)

    if not has_stop and free_qty > 0:
        if cfg["repair_protection"]:
            try:
                stop = StopOrderRequest(symbol=symbol, qty=free_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC, stop_price=sl_price)
                r2 = tc.submit_order(stop)
                log.info("REPAIR %s: placed STOP SL at $%.2f qty=%s id=%s", symbol, sl_price, free_qty, getattr(r2, "id", "?"))
            except Exception as e:
                log.warning("REPAIR %s failed to place STOP SL: %s", symbol, e)
        else:
            log.info("REPAIR %s: missing STOP SL at $%.2f (not placed)", symbol, sl_price)

# ---------- Main ----------
def main():
    cfg = load_config()
    tc = alpaca_client(cfg)

    cash, held, equity, by_symbol_value, positions = get_cash_positions_equity(tc)
    log.info("Cash: $%.2f | Equity: $%.2f | Open positions: %d -> %s", cash, equity, len(held), held)

    # --- summary trackers ---
    promoted: List[str] = []
    repaired_tp: List[str] = []
    repaired_sl: List[str] = []
    cap_skips: List[str] = []
    buys: List[str] = []

    universe_all = build_universe(cfg, held)
    universe = universe_all
    if cfg["avoid_rebuy"] and held:
        held_set = set(held)
        universe = [s for s in universe_all if s not in held_set]

    # Maintenance pass first (scan existing positions)
    for pos in positions:
        try:
            unreal = position_unrealized_pct(pos)
            # Promote winners
            if unreal >= cfg["trail_promote_pct"]:
                promote_to_trailing_stop(tc, pos, cfg)
                promoted.append(pos.symbol)
            # Repair missing protection
            before_orders = open_orders_for_symbol(tc, pos.symbol)
            has_limit_before = any((getattr(o, "type", "") == "limit" or getattr(o, "limit_price", None)) for o in before_orders)
            has_stop_before = any((getattr(o, "type", "") in ("stop", "stop_loss") or getattr(o, "stop_price", None)) for o in before_orders)
            repair_protection_if_missing(tc, pos, cfg)
            after_orders = open_orders_for_symbol(tc, pos.symbol)
            has_limit_after = any((getattr(o, "type", "") == "limit" or getattr(o, "limit_price", None)) for o in after_orders)
            has_stop_after = any((getattr(o, "type", "") in ("stop", "stop_loss") or getattr(o, "stop_price", None)) for o in after_orders)
            if (not has_limit_before) and has_limit_after:
                repaired_tp.append(pos.symbol)
            if (not has_stop_before) and has_stop_after:
                repaired_sl.append(pos.symbol)
        except Exception as e:
            log.warning("maintenance: %s failed: %s", getattr(pos, "symbol", "?"), e)

    # Scan + rank for new buys
    picks = scan_and_rank(universe)

    if picks:
        top_n = picks[: cfg["log_top"]]
        pretty = [f"{p.symbol}(score={p.score:.4f}, RSI={p.rsi14:.1f}, edge={(p.sma5/p.sma20-1)*100:.2f}%)" for p in top_n]
        log.info("Top candidates: %s", pretty)

    placed = 0
    max_new = max(0, cfg["max_new"])
    for pick in picks:
        if placed >= max_new:
            break
        if would_exceed_symbol_cap(pick.symbol, cfg["per_trade_usd"], equity, by_symbol_value, cfg["max_pct_symbol"]):
            log.info("SKIP %s: per‑symbol cap %.1f%% of equity would be exceeded.", pick.symbol, cfg["max_pct_symbol"])
            cap_skips.append(pick.symbol)
            continue
        try:
            place_buy(tc, symbol=pick.symbol, notional=cfg["per_trade_usd"], tp_pct=cfg["tp_pct"], sl_pct=cfg["sl_pct"])
            by_symbol_value[pick.symbol] = by_symbol_value.get(pick.symbol, 0.0) + cfg["per_trade_usd"]
            buys.append(pick.symbol)
            placed += 1
            time.sleep(0.25)
        except Exception as e:
            log.warning("order: %s failed: %s", pick.symbol, e)
            continue

    # --- Summary block ---
    def _fmt(lst: List[str]) -> str:
        return ", ".join(lst) if lst else "—"

    log.info(
        "SUMMARY | buys=%d [%s] | promoted=%d [%s] | repaired_tp=%d [%s] | repaired_sl=%d [%s] | cap_skips=%d [%s]",
        len(buys), _fmt(buys), len(promoted), _fmt(promoted), len(repaired_tp), _fmt(repaired_tp), len(repaired_sl), _fmt(repaired_sl), len(cap_skips), _fmt(cap_skips)
    )

    log.info("Run complete: placed %d order(s).", placed)

if __name__ == "__main__":
    main()

