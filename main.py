#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crypto live runner for Kraken via CCXT — with auto-pick.

New:
- Auto-pick: From SYMBOLS, compute dip% = (cur - SMA10)/SMA10 * 100.
  Pick the *most negative* (biggest dip) among eligible pairs.
- Respects min-notional and available USD before selecting.
- MAX_TRADES_PER_RUN: buys top-N candidates by dip order.

Env knobs:
- DRY_RUN: "true"/"false"
- TRADE_AMOUNT: e.g., "10"
- TAKE_PROFIT_PCT: e.g., "0.02" (+2%)
- DROP_PCT: dip gate (0.00 disables) – still applied, but auto-pick uses the actual dip% to sort
- MAX_TRADES_PER_RUN: "1" (or more)
- SYMBOLS: "BTC/USD,ETH/USD,DOGE/USD,XRP/USD,ADA/USD" (order doesn’t matter now)
- MIN_NOTIONAL_BUFFER: e.g., "1.00"

Secrets (for live):
- KRAKEN_API_KEY, KRAKEN_API_SECRET
"""

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import ccxt  # type: ignore
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
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

pyver = ".".join(map(str, sys.version_info[:3]))
log.info("Python %s", pyver)

# ---------- read env ----------
DRY_RUN            = env_bool("DRY_RUN", True)
TRADE_AMOUNT_USD   = env_decimal("TRADE_AMOUNT", "10")
TAKE_PROFIT_PCT    = env_decimal("TAKE_PROFIT_PCT", "0.02")
DROP_PCT           = env_decimal("DROP_PCT", "0.00")
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
    kwargs = {
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    }
    if not DRY_RUN:
        if not API_KEY or not API_SECRET:
            raise RuntimeError("Live trading requires KRAKEN_API_KEY and KRAKEN_API_SECRET.")
        kwargs.update({"apiKey": API_KEY, "secret": API_SECRET})
    return ccxt.kraken(kwargs)

def pct_change(a: Decimal, b: Decimal) -> Decimal:
    if a == 0:
        return Decimal("0")
    return (b - a) / a

def fetch_min_notional_usd(ex: ccxt.Exchange, symbol: str) -> Decimal:
    m = ex.market(symbol)
    amount_min = None
    try:
        amount_min = m.get("limits", {}).get("amount", {}).get("min")
    except Exception:
        amount_min = None

    price = Decimal("0")
    try:
        t = ex.fetch_ticker(symbol)
        price = Decimal(str(t["last"] or t["close"] or 0))
    except Exception:
        pass

    if amount_min and price > 0:
        try:
            return Decimal(str(amount_min)) * price
        except Exception:
            return Decimal("5")

    base = symbol.split("/")[0]
    if base in ("BTC",):
        return Decimal("10")
    if base in ("ETH","SOL","ADA","XRP","DOGE","LTC","ATOM","DOT","MATIC"):
        return Decimal("5")
    return Decimal("3")

def ensure_free_usd(ex: ccxt.Exchange) -> Decimal:
    if DRY_RUN and (not API_KEY or not API_SECRET):
        log.info("No API keys (DRY_RUN) — simulating plentiful USD for testing.")
        return Decimal("1000000000")
    try:
        bal = ex.fetch_free_balance()
        return Decimal(str(bal.get("USD", 0)))
    except ccxt.AuthenticationError:
        if DRY_RUN:
            log.info("Auth error fetching balance in DRY_RUN; simulating USD.")
            return Decimal("1000000000")
        raise

def simple_dip_metrics(ex: ccxt.Exchange, symbol: str):
    """
    Returns (cur, sma10, dip_pct) where dip_pct is negative if price is below SMA10.
    If data missing, returns None.
    """
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="30m", limit=24)
        closes = [Decimal(str(c[4])) for c in ohlcv if c and c[4] is not None]
        if len(closes) < 10:
            return None
        cur = closes[-1]
        sma = sum(closes[-10:]) / Decimal(10)
        dip_pct = pct_change(sma, cur) * Decimal("100")  # negative = dipped
        return (cur, sma, dip_pct)
    except Exception as e:
        log.warning("metrics failed for %s: %s", symbol, e)
        return None

def notional_to_amount(ex: ccxt.Exchange, symbol: str, notional_usd: Decimal) -> Decimal:
    t = ex.fetch_ticker(symbol)
    price = Decimal(str(t["last"] or t["close"]))
    if price <= 0:
        raise RuntimeError(f"No valid price for {symbol}.")
    amount = (notional_usd / price).quantize(Decimal("0.00000001"))
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
    log.info("Placing TP limit: %s sell %s @ %s", symbol, base_amount, target_price)
    ex.create_limit_sell_order(symbol, float(base_amount), float(target_price))

def run_once():
    log.info("=== START TRADING OUTPUT ===")
    start = time.time()

    ex = build_exchange()
    ex.load_markets()

    free_usd = ensure_free_usd(ex)
    log.info("Free USD: %s", free_usd)

    # Build candidate list with metrics & eligibility checks
    candidates = []
    for symbol in SYMBOLS:
        if symbol not in ex.markets:
            log.info("Skip %s: not on exchange.", symbol)
            continue

        min_notional = fetch_min_notional_usd(ex, symbol)
        wanted = TRADE_AMOUNT_USD
        need = wanted + MIN_NOTIONAL_BUFFER

        if wanted < min_notional:
            log.info("Bump needed for %s: TRADE_AMOUNT(%s) < min_notional(~%s). Skipping.",
                     symbol, wanted, min_notional)
            continue
        if free_usd < need:
            log.info("Insufficient free USD for %s: need >= %s; have %s. Skipping.",
                     symbol, need, free_usd)
            continue

        metrics = simple_dip_metrics(ex, symbol)
        if metrics is None:
            log.info("Skip %s: not enough data for SMA10.", symbol)
            continue
        cur, sma, dip_pct = metrics
        # Optional gate
        if DROP_PCT > 0 and dip_pct > (-DROP_PCT):
            log.info("%s did not meet DROP_PCT gate (%.2f%%). dip=%.2f%%", symbol, float(DROP_PCT), float(dip_pct))
            continue
        log.info("%s metrics: cur=%s | sma10=%s | dip=%.2f%%", symbol, cur, sma, float(dip_pct))
        candidates.append((symbol, dip_pct, cur))

    if not candidates:
        log.info("No eligible candidates this run.")
        log.info("Run complete. trades_placed=0 in %.2fs | DRY_RUN=%s", time.time()-start, DRY_RUN)
        log.info("=== END TRADING OUTPUT ===")
        return

    # Sort by most negative dip (biggest dip first)
    candidates.sort(key=lambda x: x[1])  # dip_pct asc (negative first)
    to_trade = candidates[:max(1, MAX_TRADES_PER_RUN)]

    trades_placed = 0
    for symbol, dip_pct, _cur in to_trade:
        if trades_placed >= MAX_TRADES_PER_RUN:
            break

        # Re-check balance before each order
        min_notional = fetch_min_notional_usd(ex, symbol)
        wanted = TRADE_AMOUNT_USD
        need = wanted + MIN_NOTIONAL_BUFFER
        if wanted < min_notional or free_usd < need:
            log.info("Skipping %s at order time (need>=%s; free=%s; min_notional=%s).",
                     symbol, need, free_usd, min_notional)
            continue

        amt = notional_to_amount(ex, symbol, wanted)
        t = ex.fetch_ticker(symbol)
        entry_price = Decimal(str(t["last"] or t["close"]))
        notional = (amt * entry_price).quantize(Decimal("0.01"))

        if DRY_RUN:
            log.info("[DRY_RUN] BUY %s amount=%s (~$%s at %s) | dip=%.2f%%", symbol, amt, notional, entry_price, float(dip_pct))
            place_tp_limit(ex, symbol, amt, entry_price, TAKE_PROFIT_PCT, dry_run=True)
        else:
            log.info("BUY %s amount=%s (~$%s at %s) | dip=%.2f%%", symbol, amt, notional, entry_price, float(dip_pct))
            order = ex.create_market_buy_order(symbol, float(amt))
            log.info("Buy order id=%s status=%s", order.get("id"), order.get("status"))
            place_tp_limit(ex, symbol, amt, entry_price, TAKE_PROFIT_PCT, dry_run=False)
            free_usd = ensure_free_usd(ex)

        trades_placed += 1

    took = time.time() - start
    log.info("Run complete. trades_placed=%s in %.2fs | DRY_RUN=%s", trades_placed, took, DRY_RUN)
    log.info("=== END TRADING OUTPUT ===")

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
