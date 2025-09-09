import os, time, traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Python 3.11+
import ccxt
import logging

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runner")

# --------------- Env knobs ----------------
def _to_float(x, fallback=None):
    try:
        return float(x)
    except Exception:
        return fallback

PYTHON_VERSION = os.getenv("PYTHON_VERSION", "3.11")

# Stop-loss knobs (kept for later)
STOP_LOSS_PCT = _to_float(os.getenv("STOP_LOSS_PCT", "2.0"), 2.0)
STOP_LOSS_USE_LIMIT = os.getenv("STOP_LOSS_USE_LIMIT", "true").lower() == "true"
STOP_LOSS_LIMIT_OFFSET_BP = _to_float(os.getenv("STOP_LOSS_LIMIT_OFFSET_BP", "10"), 10.0)

# Live knobs
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
SKIP_BALANCE_PING = os.getenv("SKIP_BALANCE_PING", "false").lower() == "true"
TRADE_AMOUNT = _to_float(os.getenv("TRADE_AMOUNT", "20"), 20.0)
AUTO_TOP_UP_MIN = os.getenv("AUTO_TOP_UP_MIN", "true").lower() == "true"
MIN_ORDER_BUFFER_USD = _to_float(os.getenv("MIN_ORDER_BUFFER_USD", "0.50"), 0.50)
MIN_NOTIONAL_FLOOR_USD = _to_float(os.getenv("MIN_NOTIONAL_FLOOR_USD", "20"), 20.0)

DAILY_CAP = _to_float(os.getenv("DAILY_CAP", "50"), 50.0)
DAILY_CAP_TZ = os.getenv("DAILY_CAP_TZ", "America/Los_Angeles")

# Universe / filters
DROP_PCT = _to_float(os.getenv("DROP_PCT", "1.5"), 1.5)
UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE", "200"))
PREFERRED_QUOTES = [x.strip() for x in os.getenv("PREFERRED_QUOTES", "USD,USDT").split(",") if x.strip()]
MIN_DOLLAR_VOL_24H = _to_float(os.getenv("MIN_DOLLAR_VOL_24H", "500000"), 500000.0)
MIN_PRICE = _to_float(os.getenv("MIN_PRICE", "0.01"), 0.01)

EXCLUDE = [x.strip() for x in os.getenv("EXCLUDE", "U/USD").split(",") if x.strip()]
WHITELIST = [x.strip() for x in os.getenv("WHITELIST", "").split(",") if x.strip()]

# Secrets
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_SECRET = os.getenv("KRAKEN_SECRET", "")

# -------------- Helpers -------------------
def midnight_of_today(tzname: str):
    now = datetime.now(ZoneInfo(tzname))
    return now.replace(hour=0, minute=0, second=0, microsecond=0)

def _min_notional_usd(exchange: ccxt.Exchange, symbol: str, last_price: float|None = None):
    """
    Try to derive exchange minimum notional (USD) for a symbol using ccxt markets metadata.
    Returns a float or None.
    """
    m = (exchange.markets or {}).get(symbol, {}) or {}
    limits = m.get("limits", {}) or {}
    cost_min = (limits.get("cost") or {}).get("min")
    if cost_min is not None:
        try:
            return float(cost_min)
        except Exception:
            pass

    amt_min = (limits.get("amount") or {}).get("min")
    px_min  = (limits.get("price") or {}).get("min")
    if amt_min is not None and px_min is not None:
        try:
            return float(amt_min) * float(px_min)
        except Exception:
            pass

    # If we have a last price and amount min only
    if last_price is not None and amt_min is not None:
        try:
            return float(amt_min) * float(last_price)
        except Exception:
            pass

    return None

def _symbol_quote(symbol: str) -> str:
    # Kraken symbols look like "ABC/USD" or "XBT/USDT"
    try:
        return symbol.split("/")[1]
    except Exception:
        return ""

def _is_restricted_for_wa(error_msg: str) -> bool:
    return "restricted for US:WA" in (error_msg or "")

# We'll keep a run-local ban list for WA-restricted symbols
_run_ban_symbols = set()

def banned_this_run(symbol: str) -> bool:
    return symbol in _run_ban_symbols

def ban_for_run(symbol: str, reason: str):
    if symbol not in _run_ban_symbols:
        _run_ban_symbols.add(symbol)
        log.warning("Jurisdiction block for %s; banning this symbol for the rest of this run. Msg: %s", symbol, reason)

def place_guarded_market_buy(exchange: ccxt.Exchange, symbol: str, last_price: float) -> bool:
    """
    Pre-flight checks to avoid errors:
      - skip if symbol is WA-restricted already this run
      - enforce user MIN_NOTIONAL_FLOOR_USD
      - enforce available USD + buffer check
      - auto bump to exchange minimum if AUTO_TOP_UP_MIN is on
    """
    if banned_this_run(symbol):
        return False

    quote = _symbol_quote(symbol)
    if quote not in ("USD", "USDT"):
        log.info("Skip BUY for %s: quote %s is not USD/USDT", symbol, quote)
        return False

    # User floor
    notional = TRADE_AMOUNT
    if notional < MIN_NOTIONAL_FLOOR_USD:
        log.info("Skip BUY for %s: notional $%.2f below user floor $%.2f", symbol, notional, MIN_NOTIONAL_FLOOR_USD)
        return False

    # Exchange min
    min_ex_notional = _min_notional_usd(exchange, symbol, last_price)
    if min_ex_notional:
        target = max(notional, min_ex_notional)
    else:
        target = notional

    cost = target + MIN_ORDER_BUFFER_USD

    # Balance check (USD)
    bal = {}
    if not SKIP_BALANCE_PING:
        try:
            bal = exchange.fetch_balance()
        except Exception:
            log.exception("Balance ping failed; continuing with caution.")
    free_usd = 0.0
    if bal:
        # Kraken cash balances: 'USD' key, 'free' or 'total' fields
        try:
            free_usd = float(((bal.get("USD") or {}).get("free")) or 0.0)
        except Exception:
            free_usd = 0.0

    if free_usd < cost:
        log.info("Skip BUY for %s: need $%.2f (incl. buffer), but free USD is $%.2f", symbol, cost, free_usd)
        return False

    # Compute market amount
    if last_price <= 0:
        log.info("Skip BUY for %s: bad last price %.8f", symbol, last_price)
        return False

    amount = target / last_price

    if DRY_RUN:
        log.info("[DRY_RUN] Would place market BUY %s for notional ≈ $%.2f (amount %.8f)", symbol, target, amount)
        return True

    try:
        order = exchange.create_market_buy_order(symbol, amount)
        log.info("Placed BUY: %s  notional≈$%.2f amount=%.8f  id=%s", symbol, target, amount, order.get("id"))
        return True
    except ccxt.BaseError as e:
        msg = str(e)
        if _is_restricted_for_wa(msg):
            ban_for_run(symbol, f"kraken {msg}")
            return False
        log.error("BUY failed for %s: kraken error: %s", symbol, msg)
        return False
    except Exception as e:
        log.error("BUY failed for %s: %s", symbol, e)
        return False

# --------------- Core loop -----------------
def main():
    log.info("=== START TRADING OUTPUT ===")
    log.info("Python %s", PYTHON_VERSION)

    if not KRAKEN_API_KEY or not KRAKEN_SECRET:
        log.error("[ERROR] Missing API key/secret. Set KRAKEN_API_KEY and KRAKEN_SECRET.")
        return

    exchange = ccxt.kraken({
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_SECRET,
        "enableRateLimit": True,
        # safer order precision
        "options": {"trading": {"reduceOnly": False}},
    })

    try:
        exchange.load_markets()
    except Exception:
        log.exception("Failed to load markets")
        return

    # Optional balance ping (don’t print full balance)
    if SKIP_BALANCE_PING:
        log.info("Skipping balance ping (SKIP_BALANCE_PING=true).")
    else:
        try:
            exchange.fetch_balance()
            log.info("Fetched balance (skipping details in log).")
        except Exception:
            log.exception("Balance fetch failed; continuing.")

    # Daily-cap window
    start = midnight_of_today(DAILY_CAP_TZ)
    log.info("Daily-cap window start (local %s): %s", DAILY_CAP_TZ, start.isoformat())

    spent_today = 0.0
    log.info("Daily spend so far ≈ $%.2f. Remaining ≈ $%.2f.", spent_today, max(0.0, DAILY_CAP - spent_today))

    # Build candidate list (keep simple: scan all USD/USDT spot markets, drop by %)
    # You can replace this with your smarter scanner later
    symbols = []
    for s, m in exchange.markets.items():
        if not m.get("active", True):
            continue
        if m.get("spot") is False:
            continue
        q = _symbol_quote(s)
        if q not in PREFERRED_QUOTES:
            continue
        if s in EXCLUDE:
            continue
        symbols.append(s)

    # Trim to UNIVERSE_SIZE
    symbols = symbols[:UNIVERSE_SIZE]
    log.info("Eligible candidates: %d", len(symbols))

    # VERY simple “drop” filter: pick a few that were recently red (this is a stub)
    # In a real bot, compute real drops; here we just try a handful to demonstrate guards.
    try_list = symbols[:20]

    for symbol in try_list:
        if spent_today >= DAILY_CAP:
            log.info("Daily cap reached. Spent today ≈ $%.2f / cap $%.2f. Skipping buys.", spent_today, DAILY_CAP)
            break

        # get last price
        try:
            ticker = exchange.fetch_ticker(symbol)
            last = float(ticker.get("last") or 0.0)
        except Exception:
            log.exception("Failed to fetch ticker for %s", symbol)
            continue

        # place guarded buy
        ok = place_guarded_market_buy(exchange, symbol, last)
        if ok:
            spent_today += TRADE_AMOUNT  # conservative accounting

    log.info("=== END TRADING OUTPUT ===")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        log.error("Fatal error in main()")
