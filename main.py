#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal-safe crypto live runner for Kraken via CCXT.

Env knobs (strings; not case-sensitive):
- DRY_RUN:            "true" / "false"                (default "true")
- TRADE_AMOUNT:       dollars per buy (e.g. "10")     (default "10")
- TAKE_PROFIT_PCT:    e.g. "0.05" for +5%             (default "0.05")
- DROP_PCT:           gate for simple dip-buy logic   (default "0.00")  # 0 disables the gate
- MAX_TRADES_PER_RUN: cap buys this run               (default "1")
- SYMBOLS:            comma list like "BTC/USD,ETH/USD" (default "BTC/USD,ETH/USD")
- MIN_NOTIONAL_BUFFER: e.g. "1.00" to leave spare USD (default "1.00")

Secrets (required for live trading):
- KRAKEN_API_KEY
- KRAKEN_API_SECRET
"""

import os
import sys
import time
import json
import math
import traceback
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

# 3rd party
import ccxt  # type: ignore

# ---------- logging ----------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("trader")

# ---------- helpers ----------
def env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return v.strip() if v is not None else default

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def env_decimal(name: str, default: str) -> Decimal:
    try:
        return Decimal(env_str(name, default))
    except InvalidOperation:
        return Decimal(default)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Clean + robust Python-version log (fixes your SyntaxError)
pyver = ".".join(map(str, sys.version_info[:3]))
log.info("Python %s", pyver)

# ---------- read env ----------
DRY_RUN            = env_bool("DRY_RUN", True)
TRADE_AMOUNT_USD   = env_decimal("TRADE_AMOUNT", "10")
TAKE_PROFIT_PCT    = env_decimal("TAKE_PROFIT_PCT", "0.05")   # +5%
DROP_PCT           = env_decimal("DROP_PCT", "0.00")          # 0 disables dip-gate
MAX_TRADES_PER_RUN = int(env_str("MAX_TRADES_PER_RUN", "1"))
MIN_NOTIONAL_BUFFER= env_decimal("MIN_NOTIONAL_BUFFER", "1.00")

SYMBOLS_RAW        = env_str("SYMBOLS", "BTC/USD,ETH/USD")
SYMBOLS            = [s.strip().upper() for s in SYMBOLS_RAW.split(",") if s.strip()]

API_KEY            = env_str("KRAKEN_API_KEY", "")
API_SECRET         = env_str("KRAKEN_API_SECRET", "")

log.info("Config | DRY_RUN=%s | TRADE_AMOUNT=%s | TP=%s | DROP_PCT=%s | MAX_TRADES=%s",
         DRY_RUN, TRADE_AMOUNT_USD, TAKE_PROFIT_PCT, DROP_PCT, MAX_TRADES_PER_RUN)
log.info("Symbols: %s", ", ".join(SYMBOLS))

# ---------- exchange ----------
def build_exchange() -> ccxt.kraken:
    # Use safe defaults; only attach keys if live
    kwargs = {
        "enableRateLimit": True,
        "options": {
            "adjustForTimeDifference": True,
        },
    }
    if not DRY_RUN:
        if not API_KEY or not API_SECRET:
            raise RuntimeError("Live trading requires KRAKEN_API_KEY and KRAKEN_API_SECRET.")
        kwargs.update({"apiKey": API_KEY, "secret": API_SECRET})

    ex = ccxt.kraken(kwargs)
    return ex

# ---------- simple indicators ----------
def pct_change(a: Decimal, b: Decimal) -> Decimal:
    # change from a to b ( (b - a) / a )
    if a == 0:
        return Decimal("0")
    return (b - a) / a

# ---------- trading core ----------
def fetch_min_notional_usd(ex: ccxt.Exchange, symbol: str) -> Decimal:
    """Approximate per-order min notional using market limits if available."""
    m = ex.market(symbol)
    # Kraken often provides 'limits' with amount/min; we multiply by last price to get a notional estimate
    amount_min = None
    try:
        amount_min = m.get("limits", {}).get("amount", {}).get("min")
    except Exception:
        amount_min = None

    price = Decimal("0")
    try:
        ticker = ex.fetch_ticker(symbol)
        price = Decimal(str(ticker["last"] or ticker["close"] or 0))
    except Exception:
        pass

    if amount_min and price > 0:
        try:
            return Decimal(str(amount_min)) * price
        except Exception:
            return Decimal("5")  # fallback

    # Fallback if limits not present; use a conservative guess
    # Many majors are ~$5–$10 minimum on Kraken
    base = symbol.split("/")[0]
    if base in ("BTC",):
        return Decimal("10")
    if base in ("ETH","SOL","ADA","XRP","DOGE","LTC","ATOM","DOT","MATIC"):
        return Decimal("5")
    return Decimal("3")

def ensure_free_usd(ex: ccxt.Exchange) -> Decimal:
    bal = ex.fetch_free_balance()
    # Kraken sometimes uses "USD" key for cash dollars
    usd = Decimal(str(bal.get("USD", 0)))
    return usd

def simple_dip_gate(ex: ccxt.Exchange, symbol: str, drop_pct: Decimal) -> bool:
    """Allow buy only if current price is below a short SMA by >= drop_pct.
       If drop_pct == 0, always allow."""
    if drop_pct <= 0:
        return True
    try:
        # 30m bars, last ~24 bars (=12 hours)
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="30m", limit=24)
        closes = [Decimal(str(c[4])) for c in ohlcv if c and c[4] is not None]
        if len(closes) < 5:
            return True  # not enough data—don’t block
        cur = closes[-1]
        sma = sum(closes[-10:]) / Decimal(len(closes[-10:]))
        drop = pct_change(sma, cur) * Decimal("100")
        log.info("%s dip-gate: cur=%s | sma10=%s | change=%.2f%% | gate=%.2f%%",
                 symbol, cur, sma, float(drop), float(-drop_pct))
        return drop <= (-drop_pct)
    except Exception as e:
        log.warning("dip-gate check failed for %s: %s (allowing buy)", symbol, e)
        return True

def notional_to_amount(ex: ccxt.Exchange, symbol: str, notional_usd: Decimal) -> Decimal:
    ticker = ex.fetch_ticker(symbol)
    price = Decimal(str(ticker["last"] or ticker["close"]))
    if price <= 0:
        raise RuntimeError(f"No valid price for {symbol}.")
    amount = (notional_usd / price).quantize(Decimal("0.00000001"))  # plenty of precision
    # Respect min amount if provided
    m = ex.market(symbol)
    amt_min = None
    try:
        amt_min = m.get("limits", {}).get("amount", {}).get("min")
    except Exception:
        pass
    if amt_min:
        amt_min = Decimal(str(amt_min))
        if amount < amt_min:
            amount = amt_min
    return amount

def place_tp_limit(ex: ccxt.Exchange, symbol: str, base_amount: Decimal, entry_price: Decimal, tp_pct: Decimal, dry_run: bool):
    target_price = (entry_price * (Decimal("1") + tp_pct)).quantize(Decimal("0.01"))
    if dry_run:
        log.info("[DRY_RUN] Would place TP: %s sell %s @ %s (+%s%%)", symbol, base_amount, target_price, float(tp_pct*100))
        return
    # Kraken spot: a simple limit sell above entry works as a take-profit
    log.info("Placing TP limit: %s sell %s @ %s", symbol, base_amount, target_price)
    ex.create_limit_sell_order(symbol, float(base_amount), float(target_price))

def run_once():
    log.info("=== START TRADING OUTPUT ===")
    start = time.time()

    ex = build_exchange()
    ex.load_markets()

    free_usd = ensure_free_usd(ex)
    log.info("Free USD: %s", free_usd)

    trades_placed = 0

    for symbol in SYMBOLS:
        if trades_placed >= MAX_TRADES_PER_RUN:
            break

        if symbol not in ex.markets:
            log.warning("Skipping %s (not found on exchange)", symbol)
            continue

        # Check min notional and buffers
        min_notional = fetch_min_notional_usd(ex, symbol)
        wanted = TRADE_AMOUNT_USD
        need = wanted + MIN_NOTIONAL_BUFFER
        if wanted < min_notional:
            log.info("Bump needed for %s: TRADE_AMOUNT(%s) < min_notional(~%s). Skipping.",
                     symbol, wanted, min_notional)
            continue
        if free_usd < need:
            log.info("Insufficient free USD for %s: need >= %s (amount + buffer); have %s. Skipping.",
                     symbol, need, free_usd)
            continue

        # Optional dip gate
        if not simple_dip_gate(ex, symbol, DROP_PCT):
            log.info("%s did not meet DROP_PCT gate (%.2f%%). Skipping.", symbol, float(DROP_PCT))
            continue

        # Compute amount and place market buy (or simulate)
        amt = notional_to_amount(ex, symbol, wanted)
        ticker = ex.fetch_ticker(symbol)
        entry_price = Decimal(str(ticker["last"] or ticker["close"]))
        notional = (amt * entry_price).quantize(Decimal("0.01"))

        if DRY_RUN:
            log.info("[DRY_RUN] Would BUY %s amount=%s (~$%s at %s)", symbol, amt, notional, entry_price)
            # Simulate TP placement too
            place_tp_limit(ex, symbol, amt, entry_price, TAKE_PROFIT_PCT, dry_run=True)
        else:
            log.info("BUY %s amount=%s (~$%s at %s)", symbol, amt, notional, entry_price)
            order = ex.create_market_buy_order(symbol, float(amt))
            log.info("Buy order id=%s status=%s", order.get("id"), order.get("status"))
            # For TP, we place a limit sell above entry
            place_tp_limit(ex, symbol, amt, entry_price, TAKE_PROFIT_PCT, dry_run=False)

            # Update free USD
            free_usd = ensure_free_usd(ex)

        trades_placed += 1

    took = time.time() - start
    log.info("Run complete. trades_placed=%s in %.2fs | DRY_RUN=%s", trades_placed, took, DRY_RUN)
    log.info("=== END TRADING OUTPUT ===")

# ---------- entry ----------
if __name__ == "__main__":
    try:
        run_once()
    except ccxt.BaseError as e:
        log.error("Exchange error: %s", e)
        log.debug("Trace:\n%s", traceback.format_exc())
        raise
    except Exception as e:
        log.error("Fatal error: %s", e)
        log.debug("Trace:\n%s", traceback.format_exc())
        raise
