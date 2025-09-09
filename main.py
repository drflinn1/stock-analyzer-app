import os
import time
import logging
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import ccxt


# ----------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("runner")


# -------------------------- Env helpers --------------------------------
def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return default


def _get_list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------- Config -----------------------------------
DRY_RUN                 = _get_bool("DRY_RUN", False)
SKIP_BALANCE_PING       = _get_bool("SKIP_BALANCE_PING", True)

TRADE_AMOUNT            = _get_float("TRADE_AMOUNT", 20.0)
AUTO_TOP_UP_MIN         = _get_bool("AUTO_TOP_UP_MIN", True)
MIN_ORDER_BUFFER_USD    = _get_float("MIN_ORDER_BUFFER_USD", 0.25)

DAILY_CAP               = _get_float("DAILY_CAP", 50.0)
DAILY_CAP_TZ            = os.getenv("DAILY_CAP_TZ", "America/Los_Angeles")
QUOTE_BALANCE_BUFFER_USD = _get_float("QUOTE_BALANCE_BUFFER_USD", 0.50)  # NEW: leave at 0.50–1.00

DROP_PCT                = _get_float("DROP_PCT", 2.0)
UNIVERSE_SIZE           = _get_int("UNIVERSE_SIZE", 100)
PREFERRED_QUOTES        = [q.upper() for q in _get_list("PREFERRED_QUOTES", "USD,USDT")]
MIN_DOLLAR_VOL_24H      = _get_float("MIN_DOLLAR_VOL_24H", 500_000)
MIN_PRICE               = _get_float("MIN_PRICE", 0.01)

EXCLUDE                 = set([s.upper() for s in _get_list("EXCLUDE", "U/USD")])
WHITELIST               = set([s.upper() for s in _get_list("WHITELIST", "")])

API_KEY                 = os.getenv("KRAKEN_API_KEY", "") or os.getenv("CRYPTO_API_KEY", "")
API_SECRET              = os.getenv("KRAKEN_SECRET", "") or os.getenv("CRYPTO_API_SECRET", "")


# ---------------------- Exchange bootstrap -----------------------------
def build_kraken():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing API key/secret. Set KRAKEN_API_KEY and KRAKEN_SECRET.")
    ex = ccxt.kraken({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "adjustForTimeDifference": True,
        }
    })
    ex.load_markets()
    return ex


# ---------------------- Market limits helpers --------------------------
def _get_market(exchange, symbol):
    m = (exchange.markets or {}).get(symbol)
    if not m:
        exchange.load_markets()
        m = (exchange.markets or {}).get(symbol)
    return m or {}


def _ceil_to_step(x, step):
    if not step:
        return x
    k = (float(x) + float(step) - 1e-18) / float(step)
    return round(int(k) * float(step), 18)


def min_notional_usd(exchange, symbol, last_price):
    """
    Best-effort min notional for symbol in USD given current price.
    Looks at limits.cost.min and limits.amount.min * max(last, price.min).
    """
    m = _get_market(exchange, symbol)
    limits = m.get("limits", {}) or {}
    cost_min = (limits.get("cost") or {}).get("min")
    amt_min  = (limits.get("amount") or {}).get("min")
    px_min   = (limits.get("price") or {}).get("min")

    candidates = []
    if cost_min is not None:
        candidates.append(float(cost_min))
    if amt_min is not None and last_price is not None:
        eff_px = max(float(last_price), float(px_min) if px_min is not None else 0.0)
        candidates.append(float(amt_min) * eff_px)
    if not candidates:
        return None
    return max(candidates)


def build_amount_for_notional(exchange, symbol, notional_usd, last_price):
    """
    Convert a notional to a properly stepped/rounded amount that
    respects amount.min and amount.step/precision.
    """
    m = _get_market(exchange, symbol)
    limits = (m.get("limits") or {})
    amount_limits = (limits.get("amount") or {})

    step    = amount_limits.get("step")
    amt_min = amount_limits.get("min")
    precision = (m.get("precision") or {}).get("amount")

    amt = float(notional_usd) / float(last_price)

    if step:
        amt = _ceil_to_step(amt, float(step))
    elif precision is not None:
        fmt = f"{{:.{int(precision)}f}}"
        amt = float(fmt.format(amt + (10 ** -(int(precision) + 2))))

    if amt_min is not None and amt < float(amt_min):
        if step:
            amt = _ceil_to_step(float(amt_min), float(step))
        else:
            amt = float(amt_min)
    return amt


def place_market_buy(exchange, symbol, base_notional_usd, min_buffer_usd, daily_cap_remaining_usd):
    """
    Try to place a market buy at requested notional or at the
    pair-specific min notional + buffer if AUTO_TOP_UP_MIN is True.
    Retries once on volume/min or 'Insufficient funds' conditions.
    """
    t = exchange.fetch_ticker(symbol)
    last = float(t.get("last") or t.get("close") or 0.0)
    if not last or last <= 0:
        raise RuntimeError(f"No last price for {symbol}")

    target = float(base_notional_usd)

    if AUTO_TOP_UP_MIN:
        pair_min = min_notional_usd(exchange, symbol, last)
        if pair_min is not None:
            target = max(target, float(pair_min) + float(min_buffer_usd))

    target = min(target, float(daily_cap_remaining_usd))
    if target <= 0:
        raise RuntimeError("No remaining daily-cap headroom")

    amount = build_amount_for_notional(exchange, symbol, target, last)

    def _create(amt):
        if DRY_RUN:
            log.info(f"[DRY] Would place market buy {symbol} amount {amt}")
            return {"id": "DRY-RUN", "symbol": symbol, "amount": amt}
        return exchange.create_market_buy_order(symbol, amt)

    try:
        return _create(amount)
    except Exception as e:
        msg = str(e)
        if ("volume minimum not met" in msg) or ("Insufficient funds" in msg):
            pair_min_2 = min_notional_usd(exchange, symbol, last)
            if pair_min_2 is not None:
                bumped = max(float(base_notional_usd), float(pair_min_2) + float(min_buffer_usd))
                bumped = min(bumped, float(daily_cap_remaining_usd))
                if bumped > 0:
                    bumped_amt = build_amount_for_notional(exchange, symbol, bumped, last)
                    return _create(bumped_amt)
        raise


# ------------------------ Universe / candidates -------------------------
def _is_quote_ok(symbol: str) -> bool:
    for q in PREFERRED_QUOTES:
        if symbol.upper().endswith("/" + q):
            return True
    return False


def _quote(symbol: str) -> str:
    return symbol.split("/")[-1].upper()


def _dollar_volume(exchange, symbol, ticker):
    qv = ticker.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            pass
    bv = ticker.get("baseVolume")
    last = ticker.get("last") or ticker.get("close")
    try:
        return float(bv) * float(last)
    except Exception:
        return 0.0


def build_universe(exchange):
    out = []
    tickers = exchange.fetch_tickers()
    for symbol, t in tickers.items():
        if not _is_quote_ok(symbol):
            continue
        sym_up = symbol.upper()
        if WHITELIST:
            if sym_up not in WHITELIST:
                continue
        else:
            if sym_up in EXCLUDE:
                continue

        last = t.get("last") or t.get("close")
        if not last:
            continue
        try:
            last = float(last)
        except Exception:
            continue
        if last < MIN_PRICE:
            continue

        dv = _dollar_volume(exchange, symbol, t)
        if dv < MIN_DOLLAR_VOL_24H:
            continue

        pct = t.get("percentage")
        if pct is None:
            open_px = t.get("open")
            try:
                if open_px:
                    pct = (float(last) / float(open_px) - 1.0) * 100.0
                else:
                    continue
            except Exception:
                continue
        try:
            pct = float(pct)
        except Exception:
            continue

        if pct <= -abs(DROP_PCT):
            out.append((symbol, last, pct, dv))

    out.sort(key=lambda x: (x[2], -x[3]))
    return out[:UNIVERSE_SIZE]


# --------------------------- Daily-cap ----------------------------------
def day_start_ts_ms(tz_name: str) -> int:
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)
    start = datetime(now_local.year, now_local.month, now_local.day, 0, 0, 0, tzinfo=tz)
    return int(start.astimezone(timezone.utc).timestamp() * 1000)


def spent_today_usd(exchange) -> float:
    since = day_start_ts_ms(DAILY_CAP_TZ)
    try:
        orders = exchange.fetchClosedOrders(symbol=None, since=since)
    except Exception as e:
        log.warning(f"Could not fetch closed orders for daily-cap calc: {e}")
        return 0.0

    total = 0.0
    for o in orders:
        try:
            if (o.get("side") == "buy") and (o.get("status") in ("closed", "closed_partially", "filled", "done", None)):
                cost = o.get("cost")
                if cost is None:
                    cost = float(o.get("price") or 0.0) * float(o.get("amount") or 0.0)
                total += float(cost or 0.0)
        except Exception:
            continue
    return float(total)


# ---------------------- Quote-balance (NEW) -----------------------------
def fetch_free_balances(exchange) -> dict[str, float]:
    """
    Return free balances for the quotes we care about (USD, USDT).
    """
    out = {"USD": 0.0, "USDT": 0.0}
    try:
        b = exchange.fetch_balance()
        free = b.get("free") or b  # Kraken returns 'free'
        for code in out.keys():
            v = free.get(code)
            if v is not None:
                out[code] = float(v)
    except Exception as e:
        log.warning(f"Could not fetch balance details: {e}")
    return out


# --------------------------- Jurisdiction -------------------------------
def jurisdiction_block_message(e: Exception) -> str | None:
    msg = str(e)
    if "Invalid permissions" in msg and "restricted for" in msg:
        try:
            after = msg.split("restricted for", 1)[1].strip()
            region = after.split(".", 1)[0].strip()
            return region
        except Exception:
            return "unknown"
    return None


# ------------------------------- Main -----------------------------------
def main():
    log.info("=== START TRADING OUTPUT ===")
    log.info("Python 3.11")

    kraken = build_kraken()

    if not SKIP_BALANCE_PING:
        try:
            _ = kraken.fetch_balance()
            log.info("Fetched balance (skipping details in log).")
        except Exception as e:
            log.warning(f"Balance fetch failed (continuing): {e}")

    start_local = datetime.fromtimestamp(day_start_ts_ms(DAILY_CAP_TZ) / 1000.0, tz=timezone.utc).astimezone(ZoneInfo(DAILY_CAP_TZ))
    log.info(f"Daily-cap window start (local {DAILY_CAP_TZ}): {start_local.isoformat()}")

    spent = spent_today_usd(kraken)
    remaining = max(0.0, DAILY_CAP - spent)
    log.info(f"Daily spend so far ≈ ${spent:.2f}. Remaining ≈ ${remaining:.2f}.")

    if remaining <= 0:
        log.info("Daily cap reached. Skipping buys.")
        log.info("=== END TRADING OUTPUT ===")
        return

    universe = build_universe(kraken)
    log.info(f"Eligible candidates: {len(universe)}")
    if not universe:
        log.info("No candidates matched filters this run.")
        log.info("=== END TRADING OUTPUT ===")
        return

    banned_this_run = set()
    free_quotes = fetch_free_balances(kraken)
    log.info(f"Free quotes snapshot: USD=${free_quotes.get('USD',0.0):.2f}, USDT=${free_quotes.get('USDT',0.0):.2f}")

    for symbol, last, pct, dv in universe:
        if remaining <= 0:
            log.info("Daily cap reached during loop; stopping.")
            break
        if symbol in banned_this_run:
            continue

        quote = _quote(symbol)
        free_q = float(free_quotes.get(quote, 0.0))
        headroom = max(0.0, free_q - QUOTE_BALANCE_BUFFER_USD)

        # clamp by available quote balance
        notional = min(TRADE_AMOUNT, remaining, headroom)
        if notional <= 0:
            log.info(f"Skip {symbol}: no free {quote} balance (free={free_q:.2f}, cushion={QUOTE_BALANCE_BUFFER_USD:.2f}).")
            continue

        log.info(f"Placing market buy: {notional:.8f} {symbol} @ market (notional ≈ ${notional:.2f})")

        try:
            order = place_market_buy(
                exchange=kraken,
                symbol=symbol,
                base_notional_usd=notional,
                min_buffer_usd=MIN_ORDER_BUFFER_USD,
                daily_cap_remaining_usd=remaining
            )
            log.info(f"BUY ok: {order.get('id','?')} {symbol}")
            remaining -= notional
            break

        except Exception as e:
            msg = str(e)
            region = jurisdiction_block_message(e)
            if region:
                log.warning(
                    f"[WARNING] Jurisdiction block for {symbol}; banning this symbol for the rest of this run. "
                    f"Msg: {msg}"
                )
                banned_this_run.add(symbol)
                continue

            if "volume minimum not met" in msg or "Insufficient funds" in msg:
                log.error(f"[ERROR] BUY failed for {symbol}: {e}")
                continue

            log.error(f"[ERROR] BUY failed for {symbol}: {e}")
            continue

    log.info("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        log.error("Fatal error: %s", err)
        traceback.print_exc()
        raise
