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
  TRAIL_PROMOTE_PCT=0.04      # promote winners above +4% (tighter for faster profit lock‑in)
  TRAIL_STOP_PCT=0.015        # protective stop distance / trail percent (1.5%)
  TRAIL_USE_TRUE=1            # 1=use Alpaca true trailing-stop orders when possible; 0=fallback to fixed STOP
  REPAIR_PROTECTION=1         # 1=place missing TP/SL orders, 0=only log
  REPAIR_SKIP_IF_ACTIVE_SELLS=1  # 1=skip repair when any open SELL orders exist (cool‑down); 0=always attempt



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
# ANSI colors for readable terminal logs
ANSI_GREEN = "[92m"
ANSI_YELLOW = "[93m"
ANSI_RED = "[91m"
ANSI_RESET = "[0m"

class ColorFormatter(logging.Formatter):
    BASE_FMT = "[%(asctime)s] %(levelname)s - %(message)s"
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        lvl = record.levelno
        if lvl >= logging.ERROR:
            return f"{ANSI_RED}{msg}{ANSI_RESET}"
        if lvl >= logging.WARNING:
            return f"{ANSI_YELLOW}{msg}{ANSI_RESET}"
        return msg

_handler = logging.StreamHandler()
_handler.setFormatter(ColorFormatter(ColorFormatter.BASE_FMT))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger("equities_engine")
# Throttled logging (per run). Use for noisy, repetitive lines.
_LOG_ONCE_KEYS: set = set()

def log_once(key: str, level: str, msg: str, *args):
    if key in _LOG_ONCE_KEYS:
        return
    _LOG_ONCE_KEYS.add(key)
    fn = getattr(log, level, log.info)
    fn(msg, *args)

# --- CSV history helper ---
def write_history_csv(row: dict):
    """Append KPI/SUMMARY metrics to runs/history.csv (auto-creates)."""
    import csv, pathlib
    runs_dir = pathlib.Path("runs")
    runs_dir.mkdir(exist_ok=True)
    csv_path = runs_dir / "history.csv"
    exists = csv_path.exists()
    fields = [
        "timestamp","equity","cash","buying_power","invested","exposure_pct",
        "unrealized_pnl","daily_pnl","tp_count","sl_count","trl_count","positions",
        "buys","promoted","repaired_tp","repaired_sl","cap_skips"
    ]
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        # ensure only the expected keys are written
        safe = {k: row.get(k, "") for k in fields}
        w.writerow(safe)

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
        "trail_promote_pct": env_float("TRAIL_PROMOTE_PCT", 0.04),
        "trail_stop_pct": env_float("TRAIL_STOP_PCT", 0.015),
        "trail_use_true": os.getenv("TRAIL_USE_TRUE", "1").strip() not in ("0", "false", "False"),
        "repair_protection": os.getenv("REPAIR_PROTECTION", "1").strip() not in ("0", "false", "False"),
        "repair_skip_if_sells": os.getenv("REPAIR_SKIP_IF_ACTIVE_SELLS", "1").strip() not in ("0", "false", "False"),
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

def get_cash_positions_equity(tc: TradingClient) -> Tuple[float, float, List[str], float, Dict[str, float], List[object], float]:
    acct = tc.get_account()
    cash = float(acct.cash)
    buying_power = float(getattr(acct, "buying_power", cash))
    equity = float(acct.equity)
    last_equity = float(getattr(acct, "last_equity", equity))
    positions = tc.get_all_positions()
    held = [p.symbol for p in positions]
    by_symbol_value = {p.symbol: float(p.market_value) for p in positions}
    return cash, buying_power, held, equity, by_symbol_value, positions, last_equity

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

# Any open SELL orders?
def has_any_open_sell(tc: TradingClient, symbol: str) -> bool:
    for o in open_orders_for_symbol(tc, symbol):
        try:
            if str(getattr(o, "side", "")).lower() == "sell":
                return True
        except Exception:
            continue
    return False


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
        log_once(f"promote_stop_exists:{symbol}", "info", "PROMOTE %s: stop already exists, skipping", symbol)
        return
    # Determine available quantity not already reserved by other SELL orders
    reserved = reserved_sell_qty(tc, symbol)
    qty = max(0, pos_qty - reserved)
    if qty <= 0:
        log_once(f"promote_no_free_qty:{symbol}", "info", "PROMOTE %s: no free qty (reserved=%s), skipping", symbol, reserved)
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
            log_once(f"promote_trail_fallback:{symbol}", "warning", "PROMOTE %s: true trailing unsupported or failed (%s); falling back to fixed STOP.", symbol, e)

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
    # Cool‑down: if ANY open sell orders exist and the flag is on, skip repairs to avoid qty conflicts
    if cfg.get("repair_skip_if_sells", True) and has_any_open_sell(tc, symbol):
        log_once(f"repair_cooldown:{symbol}", "info", "REPAIR %s: cool‑down — active SELL orders present; skipping", symbol)
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
    if reserved >= pos_qty:
        log_once(f"repair_all_reserved:{symbol}", "info", "REPAIR %s: all shares already reserved by existing SELL orders (reserved=%s, pos=%s) — skipping", symbol, reserved, pos_qty)
        return
    free_qty = max(0, pos_qty - reserved)

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

    cash, buying_power, held, equity, by_symbol_value, positions, last_equity = get_cash_positions_equity(tc)
    log.info("Cash: $%.2f | BuyingPower: $%.2f | Equity: $%.2f | Open positions: %d -> %s", cash, buying_power, equity, len(held), held)

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
    remaining_bp = max(0.0, buying_power)
    for pick in picks:
        if placed >= max_new:
            break
        # Check symbol cap first
        if would_exceed_symbol_cap(pick.symbol, cfg["per_trade_usd"], equity, by_symbol_value, cfg["max_pct_symbol"]):
            log_once(f"skip_cap:{pick.symbol}", "info", "SKIP %s: per‑symbol cap %.1f%% of equity would be exceeded.", pick.symbol, cfg["max_pct_symbol"])
            cap_skips.append(pick.symbol)
            continue
        # Auto-size notional by remaining buying power (use 90% safety buffer)
        affordable_notional = min(cfg["per_trade_usd"], max(0.0, remaining_bp) * 0.90)
        if affordable_notional <= 0:
            log_once(f"skip_no_bp:{pick.symbol}", "info", "SKIP %s: no remaining buying power.", pick.symbol)
            break
        # Require we can afford at least 1 share using ranked close
        need = pick.last_close
        if affordable_notional < max(1.0, need):
            log_once(f"skip_insufficient_bp:{pick.symbol}", "info", "SKIP %s: insufficient buying power (need~$%.2f, avail~$%.2f).", pick.symbol, need, remaining_bp)
            break
        try:
            place_buy(tc, symbol=pick.symbol, notional=affordable_notional, tp_pct=cfg["tp_pct"], sl_pct=cfg["sl_pct"])
            by_symbol_value[pick.symbol] = by_symbol_value.get(pick.symbol, 0.0) + affordable_notional
            remaining_bp = max(0.0, remaining_bp - affordable_notional)
            buys.append(pick.symbol)
            placed += 1
            time.sleep(0.25)
        except Exception as e:
            log.warning("order: %s failed: %s", pick.symbol, e)
            continue

    # --- KPI block ---
    invested = sum(by_symbol_value.values())
    exposure_pct = (invested / equity * 100.0) if equity > 0 else 0.0
    # Unrealized P&L across open positions
    unrealized_total = 0.0
    for p in positions:
        try:
            entry = float(p.avg_entry_price)
            cur = float(p.current_price)
            qty = float(p.qty)
            unrealized_total += (cur - entry) * qty
        except Exception:
            pass
    daily_pnl = equity - last_equity
    # Count protective orders (TP/SL/TRAIL) currently open
    tp_count = sl_count = trl_count = 0
    for p in positions:
        for o in open_orders_for_symbol(tc, p.symbol):
            try:
                if str(getattr(o, "side", "")).lower() != "sell":
                    continue
                typ = str(getattr(o, "type", "")).lower()
                if typ == "limit" or getattr(o, "limit_price", None):
                    tp_count += 1
                elif typ in ("stop", "stop_loss") or getattr(o, "stop_price", None):
                    sl_count += 1
                elif typ == "trailing_stop" or getattr(o, "trail_percent", None) or getattr(o, "trail_price", None):
                    trl_count += 1
            except Exception:
                continue
    from datetime import datetime
kpi_msg = (
    "KPI | Equity=$%.2f | Cash=$%.2f | BuyingPower=$%.2f | Invested=$%.2f (%.2f%%) | "
    "UnrealizedPnL=$%.2f | DailyPnL=$%.2f | TP=%d SL=%d TRL=%d | Positions=%d"
) % (equity, cash, buying_power, invested, exposure_pct, unrealized_total, daily_pnl, tp_count, sl_count, trl_count, len(positions))
# Colorize profit line: green when DailyPnL >= 0 and UnrealizedPnL >= 0, red when both negative
if daily_pnl >= 0 and unrealized_total >= 0:
    log.info(f"{ANSI_GREEN}{kpi_msg}{ANSI_RESET}")
elif daily_pnl < 0 and unrealized_total < 0:
    log.info(f"{ANSI_RED}{kpi_msg}{ANSI_RESET}")
else:
    log.info(kpi_msg)
# Persist KPI/SUMMARY snapshot to CSV (row filled below after SUMMARY is computed)
_kpi_row = {
    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
    "equity": f"{equity:.2f}",
    "cash": f"{cash:.2f}",
    "buying_power": f"{buying_power:.2f}",
    "invested": f"{invested:.2f}",
    "exposure_pct": f"{exposure_pct:.2f}",
    "unrealized_pnl": f"{unrealized_total:.2f}",
    "daily_pnl": f"{daily_pnl:.2f}",
    "tp_count": tp_count,
    "sl_count": sl_count,
    "trl_count": trl_count,
    "positions": len(positions),
}

    # --- Summary block ---
    def _fmt(lst: List[str]) -> str:
        return ", ".join(lst) if lst else "—"

    summary_msg = (
    "SUMMARY | buys=%d [%s] | promoted=%d [%s] | repaired_tp=%d [%s] | repaired_sl=%d [%s] | cap_skips=%d [%s]"
    % (len(buys), _fmt(buys), len(promoted), _fmt(promoted), len(repaired_tp), _fmt(repaired_tp), len(repaired_sl), _fmt(repaired_sl), len(cap_skips), _fmt(cap_skips))
)
# Slight highlight when there is positive activity
if (len(buys) + len(promoted)) > 0:
    log.info(f"{ANSI_GREEN}{summary_msg}{ANSI_RESET}")
else:
    log.info(summary_msg)
# finalize and write CSV history row
_kpi_row.update({
    "buys": len(buys),
    "promoted": len(promoted),
    "repaired_tp": len(repaired_tp),
    "repaired_sl": len(repaired_sl),
    "cap_skips": len(cap_skips),
})
write_history_csv(_kpi_row), _fmt(buys), len(promoted), _fmt(promoted), len(repaired_tp), _fmt(repaired_tp), len(repaired_sl), _fmt(repaired_sl), len(cap_skips), _fmt(cap_skips)
    )

    log.info("Run complete: placed %d order(s).", placed)

if __name__ == "__main__":
    main()

