#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
import logging
from datetime import datetime, timezone

import ccxt  # type: ignore


# -------------------- logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("trader")


# -------------------- env helpers --------------------
def env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return v


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y")


# -------------------- configuration --------------------
DRY_RUN = env_bool("DRY_RUN", True)
TRADE_AMOUNT = env_float("TRADE_AMOUNT", 10.0)  # USD per buy
TAKE_PROFIT = env_float("TP", 0.015)            # 1.5% default
STOP_LOSS = env_float("SL", 0.03)               # 3.0% default
TRAIL = env_float("TRAIL", 0.02)                # 2.0% default
MAX_TRADES_PER_RUN = env_int("MAX_TRADES_PER_RUN", 2)

# Buy signal sensitivity (simple dip vs SMA gate)
DROP_PCT = env_float("DROP_PCT", 0.01)          # 1% below SMA
MIN_NOTIONAL_BUFFER = env_float("MIN_NOTIONAL_BUFFER", 1.0)  # tiny cushion in USD math

# Symbols universe (Kraken spot; use base/quote pairs Kraken understands)
SYMBOLS = env_str(
    "SYMBOLS",
    "DOGE/USD,XRP/USD,ADA/USD,ETH/USD,BTC/USD",
).replace(" ", "").split(",")

# Kraken credentials (must be set in GitHub Actions secrets/env)
KRAKEN_API_KEY = env_str("KRAKEN_API_KEY", "")
KRAKEN_SECRET = env_str("KRAKEN_SECRET", "")

# Timeframe and history length for SMA
TIMEFRAME = env_str("TIMEFRAME", "5m")
SMA_LEN = env_int("SMA_LEN", 10)


# -------------------- exchange setup --------------------
def make_exchange() -> ccxt.Exchange:
    kwargs = {
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_SECRET,
        "enableRateLimit": True,
        "options": {
            "adjustForTimeDifference": True,
        },
    }
    ex = ccxt.kraken(kwargs)
    return ex


# -------------------- indicators --------------------
def fetch_ohlcv_safe(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    # Guard against intermittent failures
    for attempt in range(3):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            if attempt == 2:
                raise
            log.warning("fetch_ohlcv retry %s/%s for %s due to %s", attempt + 1, 3, symbol, e)
            time.sleep(1 + attempt)


def sma(values, n: int) -> float:
    if not values or len(values) < n:
        return float("nan")
    return sum(values[-n:]) / float(n)


# --- Funding status helper (prints in logs) ---
def log_funding_status(exchange, trade_amount, max_trades_per_run, min_notional_buffer=1.0, tag="pre"):
    """
    Writes one line showing free USD, held USD, and required-per-run math.
    tag can be 'pre' or 'post' to indicate before/after trading section.
    """
    try:
        bal = exchange.fetch_balance()
        free_usd = float(bal.get("free", {}).get("USD", 0) or 0.0)
        held_usd = float(bal.get("used", {}).get("USD", 0) or 0.0)
        required = (float(trade_amount) + float(min_notional_buffer)) * int(max_trades_per_run)
        ok = free_usd >= required
        logging.info(
            "Funding(%s) | free_usd=%.2f | held_in_orders=%.2f | required_per_run=%.2f | OK=%s",
            tag, free_usd, held_usd, required, ok,
        )
    except Exception as e:
        logging.warning("Funding(%s) | could not fetch balance: %s", tag, e)


# -------------------- position helpers --------------------
def get_position_sizes(exchange: ccxt.Exchange):
    """
    Returns dict of base asset -> free amount held (spot).
    """
    sizes = {}
    try:
        bal = exchange.fetch_balance()
        free = bal.get("free", {}) or {}
        for k, v in free.items():
            try:
                amt = float(v or 0.0)
            except Exception:
                amt = 0.0
            if amt > 0:
                sizes[k.upper()] = amt
    except Exception as e:
        log.warning("position size fetch failed: %s", e)
    return sizes


def quote_from_symbol(symbol: str) -> str:
    # e.g., "ETH/USD" -> "USD"
    return symbol.split("/")[-1].upper()


def base_from_symbol(symbol: str) -> str:
    return symbol.split("/")[0].upper()


# -------------------- order helpers --------------------
def place_market_buy(exchange: ccxt.Exchange, symbol: str, usd_amount: float):
    # amount is specified in base currency units; compute from last price
    ticker = exchange.fetch_ticker(symbol)
    price = float(ticker["last"])
    size = usd_amount / price
    # Kraken expects amount with appropriate precision
    size = float(exchange.amount_to_precision(symbol, size))
    params = {}
    return exchange.create_market_buy_order(symbol, size, params)


def place_market_sell(exchange: ccxt.Exchange, symbol: str, base_amount: float):
    size = float(exchange.amount_to_precision(symbol, base_amount))
    params = {}
    return exchange.create_market_sell_order(symbol, size, params)


# -------------------- simple strategy --------------------
def run_once():
    log.info("Python %s.%s.%s", *os.sys.version_info[:3])
    log.info("Config | DRY_RUN=%s | TRADE_AMOUNT=%.2f | TP=%.3f | SL=%.3f | TRAIL=%.3f | MAX_TRADES=%d",
             DRY_RUN, TRADE_AMOUNT, TAKE_PROFIT, STOP_LOSS, TRAIL, MAX_TRADES_PER_RUN)
    log.info("Symbols: %s", ",".join(SYMBOLS))
    log.info("=== START TRADING OUTPUT ===")

    ex = make_exchange()
    trades_placed = 0

    # Pre-run funding headroom line
    log_funding_status(
        ex,
        trade_amount=TRADE_AMOUNT,
        max_trades_per_run=MAX_TRADES_PER_RUN,
        min_notional_buffer=MIN_NOTIONAL_BUFFER,
        tag="pre",
    )

    # Snapshot balances/positions
    positions = get_position_sizes(ex)

    for symbol in SYMBOLS:
        if trades_placed >= MAX_TRADES_PER_RUN:
            break

        base = base_from_symbol(symbol)
        quote = quote_from_symbol(symbol)
        assert quote == "USD", f"Only USD quote supported for now, got {symbol}"

        # Try to compute a tiny SMA-based dip signal
        try:
            ohlcv = fetch_ohlcv_safe(ex, symbol, timeframe=TIMEFRAME, limit=max(SMA_LEN, 20))
            closes = [float(c[4]) for c in ohlcv]
            cur = closes[-1]
            s10 = sma(closes, SMA_LEN)
            dip = (cur - s10) / s10 if s10 and s10 == s10 else 0.0  # guard NaN
        except Exception as e:
            log.warning("%s metrics error: %s", symbol, e)
            continue

        # If we already hold this asset, just log that we maintain
        held = float(positions.get(base, 0.0))
        if held > 0:
            log.info(
                "%s position maintained | held=%.8f | cur=%.4f | sma10=%.4f | dip=%+.2f%%",
                symbol, held, cur, s10, 100.0 * dip
            )
            # (Optional) You can wire TP/SL/TRAIL exits here if you track entry price per trade.
            continue

        # Not holding => consider a dip buy
        log.info("%s metrics: cur=%.4f | sma10=%.4f | dip=%+.2f%%", symbol, cur, s10, 100.0 * dip)
        if dip <= -abs(DROP_PCT):
            # Check funding before sending an order
            try:
                bal = ex.fetch_balance()
                free_usd = float(bal.get("free", {}).get("USD", 0.0) or 0.0)
            except Exception as e:
                log.warning("could not fetch balance before order: %s", e)
                free_usd = 0.0

            required_per_order = TRADE_AMOUNT + MIN_NOTIONAL_BUFFER
            if free_usd < required_per_order:
                log.info(
                    "BUY %s skipped: free_usd=%.2f < required_per_order=%.2f",
                    symbol, free_usd, required_per_order
                )
                continue

            # Place order (or dry-run)
            usd_to_spend = TRADE_AMOUNT
            if DRY_RUN:
                # Compute theoretical size for the log
                size = usd_to_spend / cur
                log.info("DRYRUN BUY %s amount=%.8f (~$%.2f @ %.4f)", symbol, size, usd_to_spend, cur)
                trades_placed += 1
            else:
                try:
                    log.info("BUY %s amount≈$%.2f (@ ~%.4f)", symbol, usd_to_spend, cur)
                    order = place_market_buy(ex, symbol, usd_to_spend)
                    oid = order.get("id") or order.get("clientOrderId") or "?"
                    status = order.get("status", "?")
                    log.info("Buy order id=%s status=%s", oid, status)
                    trades_placed += 1
                except Exception as e:
                    log.error("Entry failed for %s: %s", symbol, e)
                    continue
        else:
            # No buy — price is not sufficiently below SMA
            pass

    log.info("Run complete. trades_placed=%d | DRY_RUN=%s", trades_placed, DRY_RUN)

    # Post-run funding headroom line
    log_funding_status(
        ex,
        trade_amount=TRADE_AMOUNT,
        max_trades_per_run=MAX_TRADES_PER_RUN,
        min_notional_buffer=MIN_NOTIONAL_BUFFER,
        tag="post",
    )
    log.info("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    run_once()
