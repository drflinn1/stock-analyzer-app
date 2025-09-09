#!/usr/bin/env python3
import os
import time
import math
import logging
import traceback
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import ccxt


# ------------------------------ Logging ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runner")


# ------------------------------ Env helpers ------------------------------
def _to_float(x, fallback=None):
    try:
        return float(x)
    except Exception:
        return fallback


def _to_int(x, fallback=None):
    try:
        return int(x)
    except Exception:
        return fallback


def _to_bool(x, fallback=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return fallback
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return fallback


# ------------------------------ Config (from env) ------------------------------
STOP_LOSS_PCT = _to_float(os.getenv("STOP_LOSS_PCT", "2.0"), 2.0)
STOP_LOSS_USE_LIMIT = _to_bool(os.getenv("STOP_LOSS_USE_LIMIT", "true"), True)
STOP_LOSS_LIMIT_OFFSET_BP = _to_float(os.getenv("STOP_LOSS_LIMIT_OFFSET_BP", "10"), 10.0)

DRY_RUN = _to_bool(os.getenv("DRY_RUN", "false"), False)
SKIP_BALANCE_PING = _to_bool(os.getenv("SKIP_BALANCE_PING", "false"), False)
TRADE_AMOUNT_USD = _to_float(os.getenv("TRADE_AMOUNT", "20"), 20.0)
AUTO_TOP_UP_MIN = _to_bool(os.getenv("AUTO_TOP_UP_MIN", "true"), True)
MIN_ORDER_BUFFER_USD = _to_float(os.getenv("MIN_ORDER_BUFFER_USD", "0.25"), 0.25)

DAILY_CAP = _to_float(os.getenv("DAILY_CAP", "50"), 50.0)
DAILY_CAP_TZ_NAME = os.getenv("DAILY_CAP_TZ", "America/Los_Angeles")

DROP_PCT = _to_float(os.getenv("DROP_PCT", "2.0"), 2.0)
UNIVERSE_SIZE = _to_int(os.getenv("UNIVERSE_SIZE", "100"), 100)
PREFERRED_QUOTES = [s.strip().upper() for s in os.getenv("PREFERRED_QUOTES", "USD").split(",") if s.strip()]
MIN_DOLLAR_VOL_24H = _to_float(os.getenv("MIN_DOLLAR_VOL_24H", "500000"), 500000.0)
MIN_PRICE = _to_float(os.getenv("MIN_PRICE", "0.01"), 0.01)

EXCLUDE = {s.strip().upper() for s in os.getenv("EXCLUDE", "").split(",") if s.strip()}
WHITELIST = [s.strip().upper() for s in os.getenv("WHITELIST", "").split(",") if s.strip()]

API_KEY = os.getenv("KRAKEN_API_KEY") or os.getenv("CRYPTO_API_KEY")
API_SECRET = os.getenv("KRAKEN_SECRET") or os.getenv("CRYPTO_API_SECRET")

if not API_KEY or not API_SECRET:
    log.error("Missing API key/secret. Set KRAKEN_API_KEY and KRAKEN_API_SECRET.")
    raise SystemExit(2)


# ------------------------------ Jurisdiction guard ------------------------------
banned_symbols = set()


class SkipSymbol(Exception):
    """Raised when a symbol is blocked in your jurisdiction during this run."""
    pass


def is_jurisdiction_block(msg: str) -> bool:
    """
    Returns True if Kraken says trading is restricted in your jurisdiction
    (e.g., 'Invalid permissions: <ASSET> trading restricted for US:WA').
    """
    s = msg.lower()
    return ("invalid permissions" in s and "restricted" in s and (" us:" in s or "united states" in s))


def buy_with_guard(exchange, symbol: str, amount: float):
    """
    Attempts a market buy and guards against jurisdiction blocks (e.g., US:WA).
    If blocked, we add to banned_symbols and raise SkipSymbol.
    """
    try:
        return exchange.create_order(symbol, 'market', 'buy', amount)
    except Exception as e:
        msg = str(e)
        if is_jurisdiction_block(msg):
            banned_symbols.add(symbol)
            log.warning(f"Jurisdiction block for {symbol}; banning this symbol for the rest of this run. Msg: {msg}")
            raise SkipSymbol(msg)
        raise


# ------------------------------ Exchange helpers ------------------------------
def min_notional_usd(exchange, symbol: str, last_price: float) -> float | None:
    """
    Best-effort: derive the minimum notional (USD) for a symbol from exchange markets limits.
    Returns float USD or None if unknown.
    """
    try:
        m = exchange.markets.get(symbol, {}) or {}
        limits = (m.get("limits") or {}) if m else {}
        cost_min = (limits.get("cost") or {}).get("min")
        if cost_min is not None:
            return float(cost_min)

        amt_min = (limits.get("amount") or {}).get("min")
        px_min = (limits.get("price") or {}).get("min")
        if amt_min is not None and px_min is not None:
            return float(amt_min) * float(px_min)
        if amt_min is not None and last_price is not None:
            return float(amt_min) * float(last_price)
    except Exception:
        pass
    return None


def kraken_fetch_today_spend(exchange, tzname: str) -> float:
    """
    Sums cost of today's BUY trades (USD) since local midnight in the given tz.
    """
    try:
        tz = ZoneInfo(tzname)
    except Exception:
        tz = ZoneInfo("UTC")

    now_local = datetime.now(tz)
    start_local = datetime(year=now_local.year, month=now_local.month, day=now_local.day, tzinfo=tz)
    since_ms = int(start_local.timestamp() * 1000)

    spent = 0.0
    try:
        # Kraken supports fetchMyTrades without symbol for all
        trades = exchange.fetch_my_trades(None, since=since_ms, limit=1000)
        for t in trades:
            if (t.get("side") == "buy") and t.get("cost") is not None:
                spent += float(t["cost"])
    except Exception as e:
        log.warning(f"Could not fetch trades to compute daily cap (using 0): {e}")
        spent = 0.0

    log.info(f"Daily-cap window start (local {tzname}): {start_local.isoformat()}")
    return spent


def ensure_markets_and_tickers(exchange):
    if not getattr(exchange, "markets", None):
        exchange.load_markets()
    # Some exchanges return all tickers; Kraken: fetch_tickers() exists
    try:
        tickers = exchange.fetch_tickers()
    except Exception:
        tickers = {}
    return tickers


def dollar_volume_from_ticker(t):
    """
    Return quote volume (USD) if available, else baseVolume * last as a fallback.
    """
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            pass
    base = t.get("baseVolume")
    last = t.get("last")
    if base is not None and last is not None:
        try:
            return float(base) * float(last)
        except Exception:
            pass
    return None


def pct_change_24h(t):
    """
    Use provided 24h percentage if present; else compute from last/open.
    """
    p = t.get("percentage")
    if p is not None:
        try:
            return float(p)
        except Exception:
            pass
    last = t.get("last")
    open_ = t.get("open")
    if last is not None and open_ is not None and open_:
        try:
            return (float(last) - float(open_)) / float(open_) * 100.0
        except Exception:
            pass
    return None


# ------------------------------ Candidate selection ------------------------------
def build_candidates(exchange, tickers):
    """
    Build a list of candidate dicts:
      {symbol, last, change_pct, dollar_vol}
    filtered by preferred quotes, whitelist/exclude, min price/vol, drop%.
    """
    symbols = []

    if WHITELIST:
        symbols = [s for s in WHITELIST]
    else:
        # All symbols ending with preferred quotes
        for s, m in exchange.markets.items():
            if not m.get("active", True):
                continue
            if any(s.endswith("/" + q) for q in PREFERRED_QUOTES):
                symbols.append(s)

    # Remove excluded
    if EXCLUDE:
        symbols = [s for s in symbols if s not in EXCLUDE]

    # Build data rows
    rows = []
    for s in symbols:
        if s in banned_symbols:
            continue
        t = tickers.get(s) or {}
        last = t.get("last")
        if last is None:
            continue
        try:
            last = float(last)
        except Exception:
            continue
        if last < MIN_PRICE:
            continue

        change = pct_change_24h(t)
        if change is None:
            continue

        # We want negative change magnitude >= DROP_PCT
        if change > -abs(DROP_PCT):
            continue

        dv = dollar_volume_from_ticker(t)
        if dv is None or dv < MIN_DOLLAR_VOL_24H:
            continue

        rows.append({
            "symbol": s,
            "last": last,
            "change_pct": change,
            "dollar_vol": dv,
        })

    # Sort by most negative change (largest drop)
    rows.sort(key=lambda r: r["change_pct"])  # more negative first
    if UNIVERSE_SIZE and len(rows) > UNIVERSE_SIZE:
        rows = rows[:UNIVERSE_SIZE]
    return rows


# ------------------------------ Stop-loss helper ------------------------------
def try_place_stop_loss(exchange, symbol, amount, buy_price):
    """
    Tries to place a stop-loss (best-effort). If Kraken rejects params, we log and move on.
    """
    if not STOP_LOSS_USE_LIMIT or amount is None or buy_price is None:
        return

    stop_price = round(buy_price * (1.0 - STOP_LOSS_PCT / 100.0), 8)
    # place a limit slightly below the stop (offset in basis-points)
    limit_offset = STOP_LOSS_LIMIT_OFFSET_BP / 10000.0
    limit_price = round(stop_price * (1.0 - limit_offset), 8)

    params = {}
    # Many venues accept 'stopPrice' on the sell order
    params["stopPrice"] = stop_price

    try:
        # Kraken via CCXT often accepts: create_order(symbol, type, side, amount, price=None, params={})
        # We'll try a limit sell with stopPrice param:
        if DRY_RUN:
            log.info(f"[DRY_RUN] Would place STOP-LOSS for {symbol}: stop={stop_price} limit={limit_price}, qty={amount}")
            return
        order = exchange.create_order(symbol, 'limit', 'sell', amount, limit_price, params)
        log.info(f"Stop-loss placed for {symbol}: {order}")
    except Exception as e:
        log.warning(f"Stop-loss placement best-effort failed for {symbol}: {e}")


# ------------------------------ Main ------------------------------
def main():
    log.info("=== START TRADING OUTPUT ===")
    log.info(f"Python {'.'.join(map(str, list(os.sys.version_info[:2])))}")

    # Exchange
    exchange = ccxt.kraken({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
    })

    # Optionally ping balance
    if not SKIP_BALANCE_PING and not DRY_RUN:
        try:
            bal = exchange.fetch_balance()
            log.info("Fetched balance (skipping details in log).")
        except Exception as e:
            log.warning(f"Skipping balance ping due to error: {e}")
    else:
        log.info(f"Skipping balance ping (SKIP_BALANCE_PING={SKIP_BALANCE_PING}).")

    # Daily cap check
    spent_today = 0.0
    try:
        spent_today = kraken_fetch_today_spend(exchange, DAILY_CAP_TZ_NAME)
    except Exception as e:
        log.warning(f"Daily spend computation failed (treating as 0): {e}")
        spent_today = 0.0

    log.info(f"Daily spend so far ≈ ${spent_today:.2f}. Remaining ≈ ${max(0.0, DAILY_CAP - spent_today):.2f}.")
    if spent_today >= DAILY_CAP:
        log.info(f"Daily cap reached. Spent today ≈ ${spent_today:.2f} / cap ${DAILY_CAP:.2f}. Skipping buy.")
        log.info("=== END TRADING OUTPUT ===")
        return

    # Markets/Tickers + candidates
    tickers = ensure_markets_and_tickers(exchange)
    candidates = build_candidates(exchange, tickers)
    log.info(f"Eligible candidates: {len(candidates)}")

    if not candidates:
        log.info("No candidates matched filters this run.")
        log.info("=== END TRADING OUTPUT ===")
        return

    # iterate through candidates until we can place one order
    made_order = False
    for row in candidates:
        symbol = row["symbol"]
        last = row["last"]
        change = row["change_pct"]
        dv = row["dollar_vol"]

        log.info(f"Considering {symbol}: change {change:.6f}% last {last} vol${dv:.2f}")

        if symbol in banned_symbols:
            continue

        # Trade notional (USD)
        notional = TRADE_AMOUNT_USD

        # If daily remaining smaller than notional, trim
        remaining = max(0.0, DAILY_CAP - spent_today)
        if remaining < notional:
            notional = remaining

        if notional < 1e-6:
            log.info("No daily budget remaining after trim; aborting.")
            break

        # Auto top-up to minimum (if needed)
        if AUTO_TOP_UP_MIN:
            mn = min_notional_usd(exchange, symbol, last)
            if mn is not None:
                # add tiny buffer
                min_needed = float(mn) + float(MIN_ORDER_BUFFER_USD)
                if notional < min_needed:
                    log.info(f"Auto top-up: {symbol} min notional ≈ ${mn:.2f} (+${MIN_ORDER_BUFFER_USD:.2f} buffer) → bumping from ${notional:.2f} to ${min_needed:.2f}")
                    notional = min_needed

        # compute amount to buy
        amount = notional / last
        amount = float(f"{amount:.10f}")  # trim

        log.info(f"Placing market buy: {amount} {symbol} @ market (notional ≈ ${notional:.2f})")

        try:
            if DRY_RUN:
                # Fake an order object
                order = {"symbol": symbol, "side": "buy", "type": "market", "amount": amount, "price": last}
                log.info(f"[DRY_RUN] BUY placed for {symbol}: {order}")
                made_order = True
                # try stop-loss in DRY_RUN path as well
                try_place_stop_loss(exchange, symbol, amount, last)
            else:
                order = buy_with_guard(exchange, symbol, amount)
                log.info(f"BUY placed for {symbol}: {order}")
                # Determine a filled price candidate
                buy_price = last
                try:
                    # sometimes order returns average/price
                    buy_price = float(order.get("average") or order.get("price") or last)
                except Exception:
                    buy_price = last

                try_place_stop_loss(exchange, symbol, amount, buy_price)
                made_order = True

            # update spent_today
            spent_today += notional
            break

        except SkipSymbol:
            # jurisdiction block → try next candidate
            continue

        except Exception as e:
            log.error(f"BUY failed for {symbol}: {e}")
            # try next candidate
            continue

    if not made_order:
        log.info("No order placed this run (filters or jurisdiction blocks).")

    log.info("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        raise
