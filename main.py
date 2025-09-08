#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.11+

import ccxt

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _to_float(x, fallback=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return float(fallback)

def _to_int(x, fallback=0):
    try:
        return int(str(x).strip())
    except Exception:
        return int(fallback)

def _to_bool(x, fallback=False):
    try:
        return str(x).strip().lower() in ("1", "true", "t", "yes", "y", "on")
    except Exception:
        return bool(fallback)

def _to_list_csv(x):
    if not x:
        return []
    return [p.strip() for p in str(x).split(",") if p.strip()]

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runner")

# ---------------------------------------------------------------------
# Env knobs / config (in sync with your YAML)
# ---------------------------------------------------------------------

PYTHON_VERSION = "3.11"

# Stop-loss knobs (kept for compatibility; optional placing)
STOP_LOSS_PCT            = _to_float(os.getenv("STOP_LOSS_PCT", "2.0"))
STOP_LOSS_USE_LIMIT      = _to_bool(os.getenv("STOP_LOSS_USE_LIMIT", "true"))
STOP_LOSS_LIMIT_OFFSET_BP= _to_float(os.getenv("STOP_LOSS_LIMIT_OFFSET_BP", "10"))

# Live knobs
DRY_RUN           = _to_bool(os.getenv("DRY_RUN", "false"))
SKIP_BALANCE_PING = _to_bool(os.getenv("SKIP_BALANCE_PING", "false"))
TRADE_AMOUNT_USD  = _to_float(os.getenv("TRADE_AMOUNT", "20"))         # was 15
DAILY_CAP_USD     = _to_float(os.getenv("DAILY_CAP", "50"))
DAILY_CAP_TZ      = os.getenv("DAILY_CAP_TZ", "America/Los_Angeles")

# New min-notional safety knobs
AUTO_TOP_UP_TO_MIN   = _to_bool(os.getenv("AUTO_TOP_UP_TO_MIN", "true"))
MIN_ORDER_BUFFER_USD = _to_float(os.getenv("MIN_ORDER_BUFFER_USD", "0.25"))

# Universe / selection knobs (kept compatible with your YAML)
DROP_PCT            = _to_float(os.getenv("DROP_PCT", "2.0"))          # min % drop to consider
UNIVERSE_SIZE       = _to_int(os.getenv("UNIVERSE_SIZE", "100"))       # soft cap when scanning
PREFERRED_QUOTES    = _to_list_csv(os.getenv("PREFERRED_QUOTES", "USD,USDT"))
MIN_DOLLAR_VOL_24H  = _to_float(os.getenv("MIN_DOLLAR_VOL_24H", "500000"))
MIN_PRICE           = _to_float(os.getenv("MIN_PRICE", "0.01"))
EXCLUDE             = set(_to_list_csv(os.getenv("EXCLUDE", "")))      # e.g. ['U/USD']
WHITELIST           = set(_to_list_csv(os.getenv("WHITELIST", "")))    # optional allow list

API_KEY  = os.getenv("KRAKEN_API_KEY", "")
API_SECRET = os.getenv("KRAKEN_SECRET", "")

# ---------------------------------------------------------------------
# Exchange helpers
# ---------------------------------------------------------------------

def kraken() -> ccxt.kraken:
    ex = ccxt.kraken({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "fetchTickersMaxLength": 200,  # friendlier batching
        },
    })
    ex.load_markets()
    return ex

def tz_midnight_utc(tz_name: str) -> int:
    """
    Return UTC ms for local midnight in the given timezone (today).
    """
    now_local = datetime.now(ZoneInfo(tz_name))
    local_midnight = datetime(
        year=now_local.year, month=now_local.month, day=now_local.day,
        tzinfo=ZoneInfo(tz_name)
    )
    return int(local_midnight.timestamp() * 1000)

def spent_today_usd(exchange: ccxt.Exchange, since_ms: int) -> float:
    """
    Sum USD/USDT cost of BUY trades since 'since_ms'.
    """
    total = 0.0
    try:
        trades = exchange.fetch_my_trades(since=since_ms)
        for t in trades:
            if t.get("side") != "buy":
                continue
            sym = t.get("symbol", "")
            cost = _to_float(t.get("cost", 0.0))
            # Only count USD / USDT quote
            if "/" in sym:
                quote = sym.split("/")[-1]
                if quote in ("USD", "USDT"):
                    total += cost
    except Exception as e:
        log.warning(f"fetch_my_trades failed; assuming $0 spent today. ({e})")
    return total

def min_notional_usd(exchange: ccxt.Exchange, symbol: str):
    """
    Try to determine the minimum notional (USD) for the pair.
    We look in markets-limits if available; otherwise fall back to
    amount.min * price.min.
    """
    try:
        m = exchange.markets.get(symbol, {}) or {}
        limits = m.get("limits", {}) or {}
        cost_min = (limits.get("cost") or {}).get("min")
        if cost_min is not None:
            return float(cost_min)
    except Exception:
        pass

    try:
        m = exchange.markets.get(symbol, {}) or {}
        limits = m.get("limits", {}) or {}
        amt_min = (limits.get("amount") or {}).get("min")
        px_min  = (limits.get("price") or {}).get("min")
        if amt_min is not None and px_min is not None:
            return float(amt_min) * float(px_min)
    except Exception:
        pass

    return None

# ---------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------

def pick_best_drop(exchange: ccxt.Exchange):
    """
    Fetch tickers and pick the 'best' (biggest negative percentage change)
    that meets basic liquidity/price requirements and respects WHITELIST/EXCLUDE.
    """
    tickers = exchange.fetch_tickers()
    ranked = []

    for sym, t in tickers.items():
        if "/" not in sym:
            continue

        base, quote = sym.split("/")
        if PREFERRED_QUOTES and quote not in PREFERRED_QUOTES:
            continue

        # Whitelist / Exclude
        if WHITELIST and sym not in WHITELIST:
            continue
        if sym in EXCLUDE or f"{base}/{quote}" in EXCLUDE or f"{base}/{quote}".replace("/", "") in EXCLUDE:
            continue

        last = _to_float(t.get("last", t.get("close", 0.0)))
        if last <= 0 or last < MIN_PRICE:
            continue

        # Try to get 24h percent change and volume in quote
        pct = t.get("percentage", None)
        quote_vol = _to_float(t.get("quoteVolume") or t.get("baseVolume", 0.0) * last)

        # If exchange doesn't return quoteVolume, this is a best effort
        if pct is None:
            # no standardized pct => skip symbol
            continue

        pct = _to_float(pct)
        if pct >= 0 or abs(pct) < DROP_PCT:
            continue

        if quote_vol < MIN_DOLLAR_VOL_24H:
            continue

        ranked.append((pct, quote_vol, last, sym))

    # Most negative percentage first
    ranked.sort(key=lambda x: x[0])  # pct ascending (more negative first)
    if not ranked:
        return None, None

    top = ranked[0]
    pct, qvol, last, sym = top
    return sym, last

# ---------------------------------------------------------------------
# Stop-loss (optional). We keep it simple and only place if you enabled it.
# ---------------------------------------------------------------------

def try_place_stop_loss(exchange: ccxt.Exchange, symbol: str, amount: float, fill_price: float):
    """
    Optionally place a stop-loss(-limit) SELL using ccxt unified params.
    We'll try stop-loss-limit if STOP_LOSS_USE_LIMIT, else stop-loss (market).
    """
    if STOP_LOSS_PCT <= 0:
        return

    try:
        stop_price  = fill_price * (1.0 - STOP_LOSS_PCT / 100.0)
        if STOP_LOSS_USE_LIMIT:
            # place limit a bit below stop (offset in basis points)
            limit_price = stop_price * (1.0 - STOP_LOSS_LIMIT_OFFSET_BP / 10000.0)
            params = {"stopPrice": round(stop_price, 8)}
            order = exchange.create_order(
                symbol=symbol,
                type="stop-loss-limit",
                side="sell",
                amount=amount,
                price=round(limit_price, 8),
                params=params
            )
            log.info(f"Stop-loss-limit placed: id={order.get('id')} stop={stop_price:.8f} limit={limit_price:.8f}")
        else:
            params = {"stopPrice": round(stop_price, 8)}
            order = exchange.create_order(
                symbol=symbol,
                type="stop-loss",
                side="sell",
                amount=amount,
                price=None,
                params=params
            )
            log.info(f"Stop-loss placed: id={order.get('id')} stop={stop_price:.8f}")
    except Exception as e:
        log.warning(f"Stop-loss placement failed: {e}")

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def main():
    log.info("=== START TRADING OUTPUT ===")
    log.info(f"Python {PYTHON_VERSION}")

    log.info(f"Universe size: {UNIVERSE_SIZE}  (quotes preferred: {PREFERRED_QUOTES})")
    if EXCLUDE:
        log.info(f"Exclude list ({len(EXCLUDE)}): {sorted(EXCLUDE)}")

    if not API_KEY or not API_SECRET:
        log.error("Missing API key/secret. Set KRAKEN_API_KEY and KRAKEN_SECRET.")
        return

    exchange = kraken()

    # Optional ping to surface permission errors earlier
    if not SKIP_BALANCE_PING and not DRY_RUN:
        try:
            _ = exchange.fetch_balance()
        except Exception as e:
            log.error(f"Balance ping failed: {e}")
            return
    else:
        log.info("Skipping balance ping (SKIP_BALANCE_PING=true).")

    # Compute start of daily window in your local tz
    start_ms = tz_midnight_utc(DAILY_CAP_TZ)
    log.info(f"Daily-cap window start (local {DAILY_CAP_TZ}): {datetime.fromtimestamp(start_ms/1000, ZoneInfo(DAILY_CAP_TZ)).isoformat()}")

    # How much have we spent today already?
    spent_today = 0.0
    if not DRY_RUN:
        spent_today = spent_today_usd(exchange, since_ms=start_ms)

    # Pick a candidate
    symbol, last = pick_best_drop(exchange)
    if not symbol:
        log.info("No eligible candidates today.")
        log.info("=== END TRADING OUTPUT ===")
        return

    # Best candidate chosen
    # We'll display a synthetic 'change' as unknown here; fetch_tickers gave us pct internally
    log.info(f"Best candidate: {symbol} (last {last})")

    # Daily cap math
    remaining = max(0.0, DAILY_CAP_USD - spent_today)
    log.info(f"Daily spend so far ≈ ${spent_today:.2f}. Remaining ≈ ${remaining:.2f}.")

    if remaining <= 0.0:
        log.info("Daily cap reached. Skipping buy.")
        log.info("=== END TRADING OUTPUT ===")
        return

    # Planned spend (subject to min-notional)
    spend_usd = min(TRADE_AMOUNT_USD, remaining)

    # Min-notional guard (+ optional top-up)
    min_cost = min_notional_usd(exchange, symbol)
    if min_cost is not None:
        needed = float(min_cost) + MIN_ORDER_BUFFER_USD
        if spend_usd < needed:
            if AUTO_TOP_UP_TO_MIN and (spent_today + needed) <= DAILY_CAP_USD:
                log.info(f"Top-up: {symbol} minimum ≈ ${min_cost:.2f}. "
                         f"Raising notional from ${spend_usd:.2f} to ${needed:.2f}.")
                spend_usd = min(needed, remaining)
            else:
                log.info(f"Skipping {symbol}: exchange minimum ≈ ${min_cost:.2f} "
                         f"> planned ${spend_usd:.2f}.")
                log.info("=== END TRADING OUTPUT ===")
                return

    # Final amount
    amount = spend_usd / float(last)

    # Place buy (unless DRY_RUN)
    if DRY_RUN:
        log.info(f"[DRY RUN] Would buy {amount:.8f} {symbol} @ market (≈ ${spend_usd:.2f})")
        log.info("=== END TRADING OUTPUT ===")
        return

    try:
        log.info(f"Placing market buy: {amount:.8f} {symbol} @ market (notional ≈ ${spend_usd:.2f})")
        order = exchange.create_order(symbol=symbol, type="market", side="buy", amount=amount)
        log.info(f"BUY placed: id={order.get('id')} status={order.get('status')}")

        # Derive a fill price to anchor a stop (if you want to place one)
        fill_price = float(last)
        try_place_stop_loss(exchange, symbol, amount, fill_price)

    except ccxt.BaseError as e:
        log.error(f"BUY failed for {symbol}: kraken {e}")
    except Exception as e:
        log.error(f"BUY failed for {symbol}: {e}\n{traceback.format_exc()}")

    log.info("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    main()
