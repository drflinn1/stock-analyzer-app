import os
import math
import time
import logging
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import ccxt

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runner")

# -----------------------------------------------------------------------------
# env helpers
# -----------------------------------------------------------------------------
def _to_float(x, fallback=None):
    try:
        return float(str(x).strip())
    except Exception:
        return fallback

def _to_bool(x, fallback=False):
    if x is None:
        return fallback
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return fallback

def _csv_list(val):
    if not val:
        return []
    return [p.strip() for p in str(val).split(",") if p.strip()]

# -----------------------------------------------------------------------------
# read knobs from env (kept compatible with what you already use)
# -----------------------------------------------------------------------------
PYTHON_VERSION = os.getenv("PYTHON_VERSION", "3.11")

# stop loss
STOP_LOSS_PCT = _to_float(os.getenv("STOP_LOSS_PCT", "2.0"), 2.0)
STOP_LOSS_USE_LIMIT = _to_bool(os.getenv("STOP_LOSS_USE_LIMIT", "true"), True)
STOP_LOSS_LIMIT_OFFSET_BP = _to_float(os.getenv("STOP_LOSS_LIMIT_OFFSET_BP", "10"), 10.0)

# NEW: take profit
TAKE_PROFIT_PCT = _to_float(os.getenv("TAKE_PROFIT_PCT", "5.0"), 5.0)  # percent above fill

# live knobs
DRY_RUN = _to_bool(os.getenv("DRY_RUN", "false"))
SKIP_BALANCE_PING = _to_bool(os.getenv("SKIP_BALANCE_PING", "false"))
TRADE_AMOUNT = _to_float(os.getenv("TRADE_AMOUNT", "20"), 20.0)
AUTO_TOP_UP_MIN = _to_bool(os.getenv("AUTO_TOP_UP_MIN", "true"), True)
MIN_ORDER_BUFFER_USD = _to_float(os.getenv("MIN_ORDER_BUFFER_USD", "0.50"), 0.50)
MIN_NOTIONAL_FLOOR_USD = _to_float(os.getenv("MIN_NOTIONAL_FLOOR_USD", "20"), 20.0)
DAILY_CAP = _to_float(os.getenv("DAILY_CAP", "50"), 50.0)
DAILY_CAP_TZ = os.getenv("DAILY_CAP_TZ", "America/Los_Angeles")

# universe / filters
DROP_PCT = _to_float(os.getenv("DROP_PCT", "1.5"), 1.5)
UNIVERSE_SIZE = int(_to_float(os.getenv("UNIVERSE_SIZE", "200"), 200))
PREFERRED_QUOTES = _csv_list(os.getenv("PREFERRED_QUOTES", "USD,USDT"))
MIN_DOLLAR_VOL_24H = _to_float(os.getenv("MIN_DOLLAR_VOL_24H", "200000"), 200000.0)
MIN_PRICE = _to_float(os.getenv("MIN_PRICE", "0.01"), 0.01)
EXCLUDE = set(_csv_list(os.getenv("EXCLUDE", "U/USD")))
WHITELIST = set(_csv_list(os.getenv("WHITELIST", "")))

# secrets
KEY = os.getenv("KRAKEN_API_KEY")
SECRET = os.getenv("KRAKEN_API_SECRET")

# -----------------------------------------------------------------------------
# exchange
# -----------------------------------------------------------------------------
def make_exchange():
    if not KEY or not SECRET:
        raise RuntimeError("Missing API key/secret. Set KRAKEN_API_KEY and KRAKEN_API_SECRET.")
    return ccxt.kraken({
        "apiKey": KEY,
        "secret": SECRET,
        "enableRateLimit": True,
        # Use 'last' trigger semantics for conditionals
        "options": {"trading": {"stop": {"trigger": "last"}}},
    })

# -----------------------------------------------------------------------------
# util: pair minimum notional (USD) from exchange limits (approx)
# -----------------------------------------------------------------------------
def min_notional_usd(exchange: ccxt.Exchange, symbol: str, px: float | None) -> float | None:
    m = exchange.markets.get(symbol, {})
    limits = m.get("limits", {}) or {}
    # prefer explicit 'cost' min if provided
    cost_min = (limits.get("cost") or {}).get("min")
    if cost_min:
        try:
            return float(cost_min)
        except Exception:
            pass
    # fallback: amount*price using min amount and current price
    amt_min = (limits.get("amount") or {}).get("min")
    if amt_min is not None and px is not None:
        try:
            return float(amt_min) * float(px)
        except Exception:
            return None
    return None

# -----------------------------------------------------------------------------
# take-profit & stop-loss helpers
# -----------------------------------------------------------------------------
def place_stop_loss(exchange: ccxt.Exchange, symbol: str, base_qty: float, fill_price: float):
    if STOP_LOSS_PCT <= 0 or base_qty <= 0:
        return
    stop_price  = fill_price * (1.0 - STOP_LOSS_PCT / 100.0)

    if STOP_LOSS_USE_LIMIT:
        # nudge limit slightly below trigger to improve fill probability
        limit_delta = (STOP_LOSS_LIMIT_OFFSET_BP / 10000.0) * fill_price
        limit_price = stop_price - limit_delta
        params = {"trigger": "last", "stopPrice": exchange.price_to_precision(symbol, stop_price)}
        # Kraken understands stop-loss-limit on spot
        exchange.create_order(symbol, "stop-loss-limit", "sell",
                              amount=base_qty,
                              price=exchange.price_to_precision(symbol, limit_price),
                              params=params)
        log.info("Placed stop-loss-limit %.4f -> %.4f for %s", stop_price, limit_price, symbol)
    else:
        params = {"trigger": "last", "stopPrice": exchange.price_to_precision(symbol, stop_price)}
        exchange.create_order(symbol, "stop-loss", "sell", amount=base_qty, price=None, params=params)
        log.info("Placed stop-loss (market) at %.4f for %s", stop_price, symbol)

def place_take_profit_limit(exchange: ccxt.Exchange, symbol: str, base_qty: float, fill_price: float):
    """
    Simple TP as a resting limit sell at +TAKE_PROFIT_PCT%.  Works everywhere and shows up
    under Kraken → 'Closed/Open orders' (not 'Conditional orders').
    """
    if TAKE_PROFIT_PCT <= 0 or base_qty <= 0:
        return
    tp_price = fill_price * (1.0 + TAKE_PROFIT_PCT / 100.0)
    tp_price = float(exchange.price_to_precision(symbol, tp_price))
    base_qty = float(exchange.amount_to_precision(symbol, base_qty))
    exchange.create_limit_sell_order(symbol, base_qty, tp_price)
    log.info("Placed take-profit limit at %.4f (+%.2f%%) for %s", tp_price, TAKE_PROFIT_PCT, symbol)

# -----------------------------------------------------------------------------
# jurisdiction/message based bans within a single run
# -----------------------------------------------------------------------------
jurisdiction_ban = set()    # symbols we must not attempt again this run

def should_ban_for_wa(msg: str) -> bool:
    if not msg:
        return False
    m = msg.lower()
    # Kraken style messages seen in your logs
    return ("invalid permissions" in m and "restricted for us:wa" in m) or ("for us:wa." in m)

def should_ban_volume_min_not_met(msg: str) -> bool:
    if not msg:
        return False
    return "volume minimum not met" in msg.lower() or "minimum not met" in msg.lower()

# -----------------------------------------------------------------------------
# daily cap window
# -----------------------------------------------------------------------------
def start_of_today_tz(tz_name: str) -> datetime:
    now = datetime.now(ZoneInfo(tz_name))
    return now.replace(hour=0, minute=0, second=0, microsecond=0)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log.info("=== START TRADING OUTPUT ===")
    log.info("Python %s", ".".join(map(str, list(map(int, tuple(map(int, os.getenv('PYTHON_VERSION', '3.11').split('.')))))))))

    exchange = make_exchange()
    exchange.load_markets()

    # daily cap accounting (very simple: count quote cost we *intend* to spend)
    window_start = start_of_today_tz(DAILY_CAP_TZ)
    log.info("Daily-cap window start (local %s): %s", DAILY_CAP_TZ, window_start.isoformat())

    # balance
    balance = exchange.fetch_balance() if not SKIP_BALANCE_PING else {"free": {}}
    free_usd = float(balance.get("free", {}).get("USD", 0.0)) if not SKIP_BALANCE_PING else None
    if not SKIP_BALANCE_PING:
        log.info("Fetched balance (skipping details in log).")
    spent_today = 0.0
    log.info("Daily spend so far ≈ $%.2f. Remaining ≈ $%.2f.", spent_today, max(0.0, DAILY_CAP - spent_today))

    # Build a quick universe using exchange markets and tickers
    # We'll prefer quotes in PREFERRED_QUOTES, drop tiny prices, and keep a large basket
    tickers = exchange.fetch_tickers()
    symbols = []
    for sym, t in tickers.items():
        if "/" not in sym:
            continue
        base, quote = sym.split("/")
        quote = quote.upper()
        if quote not in PREFERRED_QUOTES:
            continue
        if sym in EXCLUDE:
            continue
        last = _to_float(t.get("last") or t.get("close") or t.get("bid") or t.get("ask"))
        if last is None or last < MIN_PRICE:
            continue
        symbols.append((sym, last))

    # Rank by 24h change (biggest losers first) so we try oversold first
    ranked = []
    for sym, last in symbols:
        t = tickers.get(sym, {})
        chg = _to_float(t.get("percentage"), 0.0)  # 24h pct change
        ranked.append((chg, sym, last))
    ranked.sort(key=lambda x: x[0])  # most negative first

    eligible = 0
    for chg, symbol, last in ranked[:UNIVERSE_SIZE]:
        if symbol in jurisdiction_ban:
            continue

        # quick drop/whitelist logic
        if WHITELIST and symbol not in WHITELIST:
            continue
        if chg is None or chg > -abs(DROP_PCT):
            continue

        eligible += 1

    log.info("Eligible candidates: %d", eligible)

    # walk again and try to buy one that passes funds & min-notional
    for chg, symbol, last in ranked[:UNIVERSE_SIZE]:
        if symbol in jurisdiction_ban:
            continue
        if WHITELIST and symbol not in WHITELIST:
            continue
        if chg is None or chg > -abs(DROP_PCT):
            continue

        # determine notional we need for this pair
        pair_min = min_notional_usd(exchange, symbol, last) or 0.0
        min_needed = max(MIN_NOTIONAL_FLOOR_USD, pair_min)
        notional = TRADE_AMOUNT
        if AUTO_TOP_UP_MIN and notional < min_needed:
            notional = min_needed

        # ensure we still respect daily cap
        if spent_today + notional > DAILY_CAP + 1e-9:
            log.info("Stop scanning; daily cap would be exceeded by %s", symbol)
            break

        # check free USD (if we loaded balance)
        if not SKIP_BALANCE_PING:
            free_usd = float(exchange.fetch_balance().get("free", {}).get("USD", 0.0))
            if free_usd < (notional + MIN_ORDER_BUFFER_USD):
                log.info("Skip BUY for %s: need $%.2f (incl. buffer), but free USD is $%.2f",
                         symbol, notional + MIN_ORDER_BUFFER_USD, free_usd)
                continue

        # market buy (Kraken supports 'cost' param for quote amount buys)
        try:
            log.info("Placing market buy: $%.6f %s @ market (notional ≈ $%.2f)",
                     notional / max(last, 1e-12), symbol, notional)
            if DRY_RUN:
                order = {"amount": notional / last, "average": last}
            else:
                order = exchange.create_market_buy_order(symbol, amount=None, params={"cost": notional})
            base_filled = _to_float(order.get("amount"), notional / last) or (notional / last)
            avg_price  = _to_float(order.get("average") or order.get("price") or last, last)

            # place protective stop-loss
            try:
                if not DRY_RUN:
                    place_stop_loss(exchange, symbol, base_filled, avg_price)
            except Exception as e:
                log.warning("Stop-loss placement failed for %s: %s", symbol, e)

            # place take-profit limit
            try:
                if not DRY_RUN and TAKE_PROFIT_PCT > 0:
                    place_take_profit_limit(exchange, symbol, base_filled, avg_price)
            except Exception as e:
                log.warning("Take-profit placement failed for %s: %s", symbol, e)

            spent_today += notional
            break  # one buy per run

        except ccxt.BaseError as e:
            msg = str(e)
            if should_ban_for_wa(msg):
                log.warning("Jurisdiction block for %s; banning this symbol for the rest of this run. Msg: %s",
                            symbol, msg)
                jurisdiction_ban.add(symbol)
                continue
            if should_ban_volume_min_not_met(msg):
                log.warning("Volume min not met for %s; banning this symbol for this run. Msg: %s",
                            symbol, msg)
                jurisdiction_ban.add(symbol)
                continue
            if "insufficient funds" in msg.lower():
                log.error("BUY failed for %s: %s", symbol, msg)
                # don’t ban; we may free funds later
                continue
            log.error("BUY failed for %s: %s", symbol, msg)
            log.debug("Trace:\n%s", traceback.format_exc())
            continue

    log.info("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Fatal: %s", e)
        log.debug("Trace:\n%s", traceback.format_exc())
        raise
