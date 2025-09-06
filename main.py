#!/usr/bin/env python3
"""
Crypto live runner (Kraken via CCXT) with:
- %-based stop-loss (default 2% below fill)
- Daily spend cap enforcement in quote currency (USD/USDT)
- EXCLUDE/WHITELIST symbol controls
- Auto-skip region-restricted symbols and try the next candidate
- Daily-cap day boundary in a configurable timezone (default America/Los_Angeles)

"Today" is the local day in DAILY_CAP_TZ; we convert that midnight to UTC for API calls.
"""

import os
import time
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Python 3.11+ stdlib

import ccxt

# -------------------------- Logging -------------------------------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runner")

# -------------------------- Env knobs -----------------------------------------
def _to_float(x, fallback=None):
    try:
        return float(x)
    except Exception:
        return fallback

def _to_int(x, fallback=None):
    try:
        return int(str(x).strip())
    except Exception:
        return fallback

def _to_bool(x, default=False):
    if x is None:
        return default
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _parse_symlist(s):
    if not s:
        return set()
    return set([part.strip().upper() for part in s.split(",") if part.strip()])

PYTHON_VERSION            = os.getenv("PYTHON_VERSION", "3.11")
EXCHANGE_ID               = os.getenv("EXCHANGE_ID", "kraken").lower()
DRY_RUN                   = _to_bool(os.getenv("DRY_RUN", "true"), True)

# Notional knobs (quote currency spend)
TRADE_AMOUNT              = _to_float(os.getenv("TRADE_AMOUNT", "25"), 25.0)
DAILY_CAP                 = _to_float(os.getenv("DAILY_CAP", "75"), 75.0)
DAILY_CAP_TZ              = os.getenv("DAILY_CAP_TZ", "America/Los_Angeles")  # NEW

# Signal / selection knobs
DROP_PCT                  = _to_float(os.getenv("DROP_PCT", os.getenv("DROP_THRESHOLD", "2.0")), 2.0)
UNIVERSE_SIZE             = _to_int(os.getenv("UNIVERSE_SIZE", "100"), 100)
PREFERRED_QUOTES          = [q.strip().upper() for q in os.getenv("PREFERRED_QUOTES", "USD,USDT").split(",") if q.strip()]
MIN_PRICE                 = _to_float(os.getenv("MIN_PRICE", "0"), 0.0)
MIN_DOLLAR_VOL_24H        = _to_float(os.getenv("MIN_DOLLAR_VOL_24H", "0"), 0)

# List controls
EXCLUDE_LIST              = _parse_symlist(os.getenv("EXCLUDE", ""))   # e.g. "U/USD, AKE/USD"
WHITELIST                 = _parse_symlist(os.getenv("WHITELIST", "")) # e.g. "BTC/USD,ETH/USD"

# Stop-loss knobs
STOP_LOSS_PCT             = _to_float(os.getenv("STOP_LOSS_PCT", "2.0"), 2.0)
STOP_LOSS_USE_LIMIT       = _to_bool(os.getenv("STOP_LOSS_USE_LIMIT", "true"), True)
STOP_LOSS_LIMIT_OFFSET_BP = _to_int(os.getenv("STOP_LOSS_LIMIT_OFFSET_BP", "10"), 10)  # 10 bp = 0.10%

# ------------------------ Utility / Exchange ----------------------------------
def make_exchange():
    params = {"enableRateLimit": True, "options": {"adjustForTimeDifference": True}}
    if not DRY_RUN:
        key = os.getenv("KRAKEN_API_KEY") or os.getenv("API_KEY") or ""
        secret = os.getenv("KRAKEN_SECRET") or os.getenv("API_SECRET") or ""
        password = os.getenv("KRAKEN_PASSWORD") or os.getenv("API_PASSWORD") or None
        params.update({"apiKey": key, "secret": secret})
        if password:
            params.update({"password": password})

    if EXCHANGE_ID != "kraken":
        raise RuntimeError("This runner is currently wired for Kraken only.")
    return ccxt.kraken(params)

def price_prec(exchange, symbol, px):
    return float(exchange.price_to_precision(symbol, px))

def amt_prec(exchange, symbol, amt):
    return float(exchange.amount_to_precision(symbol, amt))

def quote_currency_of(exchange, symbol):
    try:
        return exchange.market(symbol).get("quote", "").upper()
    except Exception:
        return ""

# ------------------------ Universe / Selection --------------------------------
def build_spot_universe(exchange):
    exchange.load_markets()
    markets = exchange.markets

    candidates = []
    for sym, m in markets.items():
        if not m.get("active", True):
            continue
        if m.get("type") not in (None, "spot"):
            continue
        quote = m.get("quote", "").upper()
        if quote not in PREFERRED_QUOTES:
            continue
        symU = sym.upper()
        if WHITELIST and symU not in WHITELIST:
            continue
        if symU in EXCLUDE_LIST:
            continue
        candidates.append(sym)

    tickers = exchange.fetch_tickers(candidates)
    rows = []
    for sym in candidates:
        t = tickers.get(sym) or {}
        last = t.get("last") or t.get("close") or t.get("ask") or t.get("bid")
        if not last:
            continue
        if MIN_PRICE and last < MIN_PRICE:
            continue
        pct = t.get("percentage")      # 24h %
        qvol = t.get("quoteVolume")    # 24h quote volume (USD)
        if MIN_DOLLAR_VOL_24H and (qvol is None or qvol < MIN_DOLLAR_VOL_24H):
            continue
        rows.append({"symbol": sym, "last": last, "pct": pct, "qvol": qvol})

    rows.sort(key=lambda r: (r["pct"] if r["pct"] is not None else 0), reverse=False)
    if UNIVERSE_SIZE and len(rows) > UNIVERSE_SIZE:
        rows = rows[:UNIVERSE_SIZE]
    return rows

def pick_best_candidate(rows):
    best = None
    for r in rows:
        pct = r["pct"]
        if pct is not None and pct <= -abs(DROP_PCT):
            best = r
            break
    if not best and rows:
        best = rows[0]
    return best

# ------------------------ Daily cap helpers (LOCAL TZ) ------------------------
def cap_window_start():
    """
    Returns (since_ms_utc, local_midnight_iso) for the start of 'today' in DAILY_CAP_TZ.
    """
    try:
        tz = ZoneInfo(DAILY_CAP_TZ)
    except Exception:
        tz = timezone.utc
    now_local = datetime.now(tz)
    local_midnight = datetime(now_local.year, now_local.month, now_local.day, tzinfo=tz)
    start_utc = local_midnight.astimezone(timezone.utc)
    return int(start_utc.timestamp() * 1000), local_midnight.isoformat()

def sum_today_quote_spend(exchange, quotes, since_ms):
    """
    Returns today's total BUY notional (quote currency) summed across symbols with
    quote in `quotes` list. Uses fetch_my_trades since local-midnight (converted to UTC ms).
    """
    total = 0.0
    try:
        trades = exchange.fetch_my_trades(symbol=None, since=since_ms, limit=500)
        for tr in trades or []:
            side = tr.get("side")
            cost = tr.get("cost")  # quote notional
            sym  = tr.get("symbol")
            ts   = tr.get("timestamp") or 0
            if side != "buy" or cost is None or not sym or ts < since_ms:
                continue
            quote = sym.split("/")[-1].upper() if "/" in sym else quote_currency_of(exchange, sym)
            if quote in quotes:
                total += float(cost)
    except Exception as e:
        log.warning(f"Daily-cap spend check failed (enable 'Query trades' permission?): {e}")
        return None
    return float(total)

def enforce_daily_cap_or_adjust(exchange, symbol, last_price, desired_quote_notional, since_ms):
    """
    Compares today's spend vs DAILY_CAP. Returns (approved_quote_notional, spent_today, remaining).
    """
    if DAILY_CAP is None or DAILY_CAP <= 0:
        return desired_quote_notional, 0.0, float("inf")

    spent = sum_today_quote_spend(exchange, set(PREFERRED_QUOTES), since_ms)
    if spent is None:
        log.warning("Skipping buy: cannot confirm daily spend (API permission?).")
        return 0.0, None, 0.0

    remaining = max(0.0, DAILY_CAP - spent)
    if remaining <= 0:
        log.info(f"Daily cap reached. Spent today ≈ ${spent:.2f} / cap ${DAILY_CAP:.2f}. Skipping buy.")
        return 0.0, spent, 0.0

    approved = min(desired_quote_notional, remaining)
    if approved < desired_quote_notional:
        log.info(f"Trimming order to fit daily cap: wanted ${desired_quote_notional:.2f}, remaining ${remaining:.2f} → using ${approved:.2f}")
    else:
        log.info(f"Daily spend so far ≈ ${spent:.2f}. Remaining ≈ ${remaining:.2f}.")
    return approved, spent, remaining

# ------------------------ Stop-loss helpers -----------------------------------
def place_percent_stop_loss(exchange, symbol, filled_amount, avg_fill_price):
    """
    Places a Kraken stop-loss or stop-loss-limit sell for 'filled_amount' base units
    at STOP_LOSS_PCT below 'avg_fill_price', using correct precision/params.
    """
    if not filled_amount or not avg_fill_price:
        log.warning("Cannot place stop-loss: missing filled_amount or avg_fill_price")
        return None

    stop_trigger_
