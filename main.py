#!/usr/bin/env python3
"""
Crypto live runner (Kraken via CCXT) with:
- %-based stop-loss (default 2% below fill)
- Daily spend cap enforcement in quote currency (USD/USDT)
- EXCLUDE/WHITELIST symbol controls
- Auto-skip region-restricted symbols and try the next candidate

"Today" = UTC day (GitHub runners use UTC).
"""

import os
import time
import traceback
from datetime import datetime, timezone

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
    # Accept "BTC/USD, ETH/USD" or "" → set()
    if not s:
        return set()
    return set([part.strip().upper() for part in s.split(",") if part.strip()])

PYTHON_VERSION           = os.getenv("PYTHON_VERSION", "3.11")
EXCHANGE_ID              = os.getenv("EXCHANGE_ID", "kraken").lower()
DRY_RUN                  = _to_bool(os.getenv("DRY_RUN", "true"), True)

# Notional knobs (quote currency spend)
TRADE_AMOUNT             = _to_float(os.getenv("TRADE_AMOUNT", "25"), 25.0)
DAILY_CAP                = _to_float(os.getenv("DAILY_CAP", "75"), 75.0)

# Signal / selection knobs
DROP_PCT                 = _to_float(os.getenv("DROP_PCT", os.getenv("DROP_THRESHOLD", "2.0")), 2.0)
UNIVERSE_SIZE            = _to_int(os.getenv("UNIVERSE_SIZE", "100"), 100)
PREFERRED_QUOTES         = [q.strip().upper() for q in os.getenv("PREFERRED_QUOTES", "USD,USDT").split(",") if q.strip()]
MIN_PRICE                = _to_float(os.getenv("MIN_PRICE", "0"), 0.0)
MIN_DOLLAR_VOL_24H       = _to_float(os.getenv("MIN_DOLLAR_VOL_24H", "0"), 0)

# List controls
EXCLUDE_LIST             = _parse_symlist(os.getenv("EXCLUDE", ""))         # e.g. "U/USD, AKE/USD"
WHITELIST                = _parse_symlist(os.getenv("WHITELIST", ""))       # if non-empty, only these trade

# Stop-loss knobs
STOP_LOSS_PCT            = _to_float(os.getenv("STOP_LOSS_PCT", "2.0"), 2.0)
STOP_LOSS_USE_LIMIT      = _to_bool(os.getenv("STOP_LOSS_USE_LIMIT", "true"), True)
STOP_LOSS_LIMIT_OFFSET_BP= _to_int(os.getenv("STOP_LOSS_LIMIT_OFFSET_BP", "10"), 10)  # 10 bp = 0.10%

# ------------------------ Utility / Exchange ----------------------------------
def make_exchange():
    params = {
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    }
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
        # honor whitelist/exclude
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
        pct = t.get("percentage")  # 24h %
        qvol = t.get("quoteVolume")
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

# ------------------------ Daily cap helpers -----------------------------------
def utc_midnight_ms():
    now = datetime.now(timezone.utc)
    midnight = datetime(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
    return int(midnight.timestamp() * 1000)

def sum_today_quote_spend(exchange, quotes):
    """
    Returns today's total BUY notional (quote currency) summed across symbols with
    quote in `quotes` list. Uses fetch_my_trades since UTC midnight.
    """
    since = utc_midnight_ms()
    total = 0.0
    try:
        trades = exchange.fetch_my_trades(symbol=None, since=since, limit=500)
        for tr in trades or []:
            side = tr.get("side")
            cost = tr.get("cost")  # quote notional
            sym  = tr.get("symbol")
            ts   = tr.get("timestamp") or 0
            if side != "buy" or cost is None or not sym or ts < since:
                continue
            quote = sym.split("/")[-1].upper() if "/" in sym else quote_currency_of(exchange, sym)
            if quote in quotes:
                total += float(cost)
    except Exception as e:
        log.warning(f"Daily-cap spend check failed (enable 'Query trades' permission?): {e}")
        return None
    return float(total)

def enforce_daily_cap_or_adjust(exchange, symbol, last_price, desired_quote_notional):
    """
    Compares today's spend vs DAILY_CAP. Returns (approved_quote_notional, spent_today, remaining).
    """
    if DAILY_CAP is None or DAILY_CAP <= 0:
        return desired_quote_notional, 0.0, float("inf")

    spent = sum_today_quote_spend(exchange, set(PREFERRED_QUOTES))
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

    stop_trigger = avg_fill_price * (1.0 - STOP_LOSS_PCT / 100.0)
    if STOP_LOSS_USE_LIMIT:
        limit_price = stop_trigger * (1.0 - STOP_LOSS_LIMIT_OFFSET_BP / 10000.0)
    else:
        limit_price = None

    trigger_px = price_prec(exchange, symbol, stop_trigger)
    amount_px  = amt_prec(exchange, symbol, filled_amount)
    limit_px   = price_prec(exchange, symbol, limit_price) if limit_price else None

    if STOP_LOSS_USE_LIMIT:
        log.info(f"Placing STOP-LOSS-LIMIT for {symbol}: trigger={trigger_px}, limit={limit_px}, amount={amount_px}")
        params = {"price2": limit_px, "trigger": "last"}
        order = exchange.create_order(
            symbol=symbol,
            type="stop-loss-limit",
            side="sell",
            amount=amount_px,
            price=trigger_px,
            params=params,
        )
    else:
        log.info(f"Placing STOP-LOSS (market) for {symbol}: trigger={trigger_px}, amount={amount_px}")
        params = {"trigger": "last"}
        order = exchange.create_order(
            symbol=symbol,
            type="stop-loss",
            side="sell",
            amount=amount_px,
            price=trigger_px,
            params=params,
        )
    log.info(f"Stop-loss order placed: id={order.get('id')}, type={order.get('type')}")
    return order

# ------------------------ Runner ----------------------------------------------
def main():
    log.info("=== START TRADING OUTPUT ===")
    log.info(f"Python {PYTHON_VERSION}")
    log.info(f"Universe size: {UNIVERSE_SIZE}  (quotes preferred: {PREFERRED_QUOTES})")
    if WHITELIST:
        log.info(f"Whitelist active ({len(WHITELIST)}): {sorted(list(WHITELIST))}")
    if EXCLUDE_LIST:
        log.info(f"Exclude list ({len(EXCLUDE_LIST)}): {sorted(list(EXCLUDE_LIST))}")

    try:
        exchange = make_exchange()
    except Exception as e:
        log.error(f"Exchange init failed: {e}")
        raise

    # Optional balance ping; many keys lack this perm (harmless)
    try:
        if not DRY_RUN:
            _ = exchange.fetch_balance()
    except Exception as e:
        log.warning(f"Balance check skipped: {exchange.id} {{\"error\":[\"{str(e)}\"]}}")

    # Universe
    rows = build_spot_universe(exchange)
    log.info(f"Eligible candidates: {len(rows)}")
    if not rows:
        log.info("No tradable symbols in universe (after filters).")
        log.info("=== END TRADING OUTPUT ===")
        return

    # Try up to N symbols this run, auto-skipping restricted ones
    TRIES = min(5, len(rows))
    tried_symbols = set()
    for _ in range(TRIES):
        # choose best not yet tried
        candidates = [r for r in rows if r["symbol"] not in tried_symbols]
        if not candidates:
            log.info("No remaining candidates to try.")
            break
        best = pick_best_candidate(candidates)
        if not best:
            log.info("No candidate meets rules.")
            break

        symbol = best["symbol"]
        tried_symbols.add(symbol)

        last   = float(best["last"])
        pct    = best["pct"]
        qvol   = best["qvol"]
        log.info(f"Best candidate: {symbol} change {pct}% last {last} vol${qvol}")

        quote_ccy = quote_currency_of(exchange, symbol)
        if quote_ccy not in PREFERRED_QUOTES:
            log.info(f"Skipping {symbol}: quote currency {quote_ccy} not in {PREFERRED_QUOTES}")
            continue

        # Enforce daily cap
        desired_quote_notional = TRADE_AMOUNT
        approved_quote_notional, spent_today, remaining = enforce_daily_cap_or_adjust(
            exchange, symbol, last_price=last, desired_quote_notional=desired_quote_notional
        )
        if approved_quote_notional <= 0:
            log.info("Order blocked by daily-cap guard.")
            log.info("=== END TRADING OUTPUT ===")
            return

        # Compute base amount from approved notional
        try:
            m = exchange.market(symbol)
            base_amt = approved_quote_notional / last
            min_amt = (m.get("limits", {}).get("amount", {}) or {}).get("min")
            if min_amt:
                base_amt = max(base_amt, float(min_amt))
            base_amt = amt_prec(exchange, symbol, base_amt)
            if base_amt <= 0:
                raise ValueError("Computed base amount <= 0 after precision/min checks.")
        except Exception as e:
            log.error(f"Amount computation failed for {symbol}: {e}")
            continue

        if DRY_RUN:
            log.info(f"[DRY_RUN] Would place market BUY: {base_amt} {symbol} (~${approved_quote_notional:.2f})")
            log.info("Skip stop-loss in DRY_RUN.")
            log.info("=== END TRADING OUTPUT ===")
            return

        # Live BUY (with restricted-asset auto-skip)
        try:
            log.info(f"Placing market buy: {base_amt} {symbol} @ market (notional ≈ ${approved_quote_notional:.2f})")
            buy_order = exchange.create_order(
                symbol=symbol,
                type="market",
                side="buy",
                amount=base_amt,
            )
            log.info(f"Executed buy: id='{buy_order.get('id')}', info: '{buy_order.get('info')}'")
        except Exception as e:
            msg = str(e).lower()
            if "restricted" in msg or "invalid permissions" in msg:
                log.warning(f"Skipping restricted symbol {symbol}: {e}")
                continue
            log.error(f"BUY failed for {symbol}: {e}")
            break  # other errors: abort run

        # After BUY → fetch fill → place stop-loss
        try:
            buy_id = (buy_order.get("id") if isinstance(buy_order, dict) else None)
            avg_fill_price = None
            filled_amount = None

            time.sleep(1.0)  # let fills settle
            if buy_id:
                try:
                    o = exchange.fetch_order(buy_id, symbol)
                    avg_fill_price = o.get("average") or o.get("price")
                    filled_amount  = o.get("filled") or o.get("amount")
                except Exception as fe:
                    log.warning(f"fetch_order failed ({fe}); falling back to ticker for fill price.")

            if avg_fill_price is None:
                t = exchange.fetch_ticker(symbol)
                avg_fill_price = t.get("last") or t.get("close") or t.get("ask") or t.get("bid")
                log.info(f"Fallback avg_fill_price from ticker: {avg_fill_price}")

            if filled_amount is None:
                if 'cost' in buy_order and buy_order['cost'] and avg_fill_price:
                    filled_amount = float(buy_order['cost']) / float(avg_fill_price)
                else:
                    filled_amount = float(buy_order.get("amount") or buy_order.get("filled") or base_amt)

            # Final precision clamp
            filled_amount = amt_prec(exchange, symbol, filled_amount)
            avg_fill_price = price_prec(exchange, symbol, avg_fill_price)

            sl = place_percent_stop_loss(
                exchange=exchange,
                symbol=symbol,
                filled_amount=filled_amount,
                avg_fill_price=avg_fill_price,
            )
            if sl:
                log.info(f"Placed stop-loss @ ~{STOP_LOSS_PCT}% below fill. Kraken id: {sl.get('id')}")
        except Exception as e:
            log.error(f"Stop-loss placement failed for {symbol}: {e}")

        # Buy succeeded (don’t try more symbols this run)
        break

    log.info("=== END TRADING OUTPUT ===")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Fatal error:\n" + "".join(traceback.format_exc()))
        raise
