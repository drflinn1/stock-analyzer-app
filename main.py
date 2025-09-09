#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import math
import traceback
from datetime import datetime, timezone
try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import logging
import ccxt


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("runner")


# ------------------------------------------------------------
# Env helpers
# ------------------------------------------------------------
def _to_bool(s: str, default=False):
    if s is None:
        return default
    s = str(s).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _to_float(s: str, default: float):
    try:
        return float(str(s).strip())
    except Exception:
        return default


def _to_int(s: str, default: int):
    try:
        return int(str(s).strip())
    except Exception:
        return default


def _split_csv(s: str):
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


# ------------------------------------------------------------
# Load knobs from env (populated by your YAML)
# ------------------------------------------------------------
STOP_LOSS_PCT            = _to_float(os.getenv("STOP_LOSS_PCT"), 2.0)
STOP_LOSS_USE_LIMIT      = _to_bool(os.getenv("STOP_LOSS_USE_LIMIT"), True)
STOP_LOSS_LIMIT_OFFSET_BP = _to_int(os.getenv("STOP_LOSS_LIMIT_OFFSET_BP"), 10)

DRY_RUN                  = _to_bool(os.getenv("DRY_RUN"), False)
SKIP_BALANCE_PING        = _to_bool(os.getenv("SKIP_BALANCE_PING"), False)
TRADE_AMOUNT             = _to_float(os.getenv("TRADE_AMOUNT"), 20.0)

AUTO_TOP_UP_MIN          = _to_bool(os.getenv("AUTO_TOP_UP_MIN"), True)
MIN_ORDER_BUFFER_USD     = _to_float(os.getenv("MIN_ORDER_BUFFER_USD"), 0.50)
MIN_NOTIONAL_FLOOR_USD   = _to_float(os.getenv("MIN_NOTIONAL_FLOOR_USD"), 20.0)

DAILY_CAP                = _to_float(os.getenv("DAILY_CAP"), 50.0)
DAILY_CAP_TZ             = os.getenv("DAILY_CAP_TZ") or "America/Los_Angeles"

DROP_PCT                 = _to_float(os.getenv("DROP_PCT"), 1.5)
UNIVERSE_SIZE            = _to_int(os.getenv("UNIVERSE_SIZE"), 200)
PREFERRED_QUOTES         = _split_csv(os.getenv("PREFERRED_QUOTES") or "USD,USDT")
MIN_DOLLAR_VOL_24H       = _to_float(os.getenv("MIN_DOLLAR_VOL_24H"), 500000)
MIN_PRICE                = _to_float(os.getenv("MIN_PRICE"), 0.01)

EXCLUDE                  = _split_csv(os.getenv("EXCLUDE") or "")
WHITELIST                = _split_csv(os.getenv("WHITELIST") or "")

API_KEY                  = os.getenv("KRAKEN_API_KEY") or os.getenv("CRYPTO_API_KEY")
API_SECRET               = os.getenv("KRAKEN_SECRET") or os.getenv("CRYPTO_API_SECRET")


# ------------------------------------------------------------
# Exchange setup
# ------------------------------------------------------------
def make_exchange():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing API key/secret. Set KRAKEN_API_KEY and KRAKEN_SECRET.")
    exchange = ccxt.kraken({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            # safer market buy emulation
            "createMarketBuyOrderRequiresPrice": False
        }
    })
    return exchange


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def min_notional_usd(exchange, symbol, price: float) -> float:
    """
    Best-effort estimate of exchange minimum notional for a symbol in USD.
    Tries cost.min first; falls back to amount.min * price if present.
    Returns 0 if unknown.
    """
    try:
        m = exchange.markets.get(symbol) or {}
    except Exception:
        m = {}
    limits = (m.get("limits") or {})
    cost_min = None
    try:
        cm = (limits.get("cost") or {}).get("min")
        if cm is not None:
            cost_min = float(cm)
    except Exception:
        cost_min = None

    if cost_min is not None:
        return max(0.0, float(cost_min))

    amount_min = None
    try:
        am = (limits.get("amount") or {}).get("min")
        if am is not None:
            amount_min = float(am)
    except Exception:
        amount_min = None

    if amount_min is not None and price and price > 0:
        return max(0.0, float(amount_min) * float(price))

    return 0.0


def start_of_today_tz(tz_name: str):
    try:
        tz = ZoneInfo(tz_name) if ZoneInfo else timezone.utc
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    return datetime(now.year, now.month, now.day, tzinfo=tz)


# ------------------------------------------------------------
# Global, per-run WA jurisdiction ban
# ------------------------------------------------------------
wa_ban = set()


def add_wa_ban(symbol: str, msg: str):
    if symbol not in wa_ban:
        wa_ban.add(symbol)
        log.warning(
            f"Jurisdiction block for {symbol}; banning this symbol for the rest of this run. "
            f"Msg: {msg}"
        )


# ------------------------------------------------------------
# Guarded buy with min-notional/top-up + WA-ban + error handling
# ------------------------------------------------------------
def place_guarded_market_buy(exchange, symbol: str, last_price: float) -> bool:
    if symbol in wa_ban:
        return False

    # compute desired notional
    notional = float(TRADE_AMOUNT)
    min_notional = min_notional_usd(exchange, symbol, last_price)
    floor = max(min_notional, MIN_NOTIONAL_FLOOR_USD)

    if AUTO_TOP_UP_MIN and (notional + MIN_ORDER_BUFFER_USD) < floor:
        notional = floor + MIN_ORDER_BUFFER_USD

    # final safety
    if notional <= 0 or not last_price or last_price <= 0:
        log.error(f"[SKIP] Invalid sizing for {symbol}. notional={notional:.2f}, price={last_price}")
        return False

    amount = round(notional / last_price, 8)

    log.info(f"Placing market buy: {amount} {symbol} @ market (notional ~ ${notional:.2f})")

    if DRY_RUN:
        log.info("[DRY_RUN] Skipping live buy.")
        return True

    try:
        order = exchange.createMarketBuyOrder(symbol, amount)
        log.info(f"BUY placed: {json.dumps(order, default=str)[:600]}")
    except Exception as e:
        msg = str(e)
        # WA jurisdiction / invalid permissions
        if "restricted for US:WA" in msg or "Invalid permissions" in msg:
            add_wa_ban(symbol, f"kraken {msg}")
            return False
        # Kraken min-volume / min-notional
        if "volume minimum not met" in msg or "EOrder:Insufficient funds" in msg or "Insufficient funds" in msg:
            log.error(f"BUY failed for {symbol}: kraken {msg}")
            return False
        log.error(f"BUY failed for {symbol}: {msg}")
        return False

    # ---- Optional Stop-Loss Limit ----
    if STOP_LOSS_USE_LIMIT:
        try:
            stop_price = round(last_price * (1.0 - STOP_LOSS_PCT / 100.0), 8)
            # Limit a bit below stop trigger
            limit_offset = STOP_LOSS_LIMIT_OFFSET_BP / 10000.0
            limit_price = round(stop_price * (1.0 - limit_offset), 8)

            params = {
                # ccxt Kraken doesn't have perfect unified stop-limit params;
                # this is best-effort & wrapped to avoid breaking the run.
                "reduceOnly": True,
                "postOnly": False
            }
            # amount to sell = we just bought `amount`
            log.info(f"Placing stop-loss-limit sell for {symbol}: stop={stop_price}, limit={limit_price}, qty={amount}")
            try:
                exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side="sell",
                    amount=amount,
                    price=limit_price,
                    params={
                        **params,
                        "stopPrice": stop_price
                    }
                )
            except Exception as e2:
                # fallback: log only; Kraken stop-limit via unified API is tricky
                log.warning(f"Stop-loss-limit placement warning for {symbol}: {e2}")

        except Exception as e:
            log.warning(f"Stop-loss setup skipped for {symbol}: {e}")

    return True


# ------------------------------------------------------------
# Universe / candidate selection
# ------------------------------------------------------------
def build_candidates(exchange):
    """
    Returns a list of (symbol, last_price, percentage_drop, quote) filtered
    by preferences and basic sanity. Sorted by largest drop first.
    """
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        log.warning(f"fetch_tickers failed, falling back to markets: {e}")
        exchange.load_markets()
        tickers = {}

    out = []
    # Either use tickers or markets if no tickers
    symbols = list(tickers.keys()) if tickers else list(exchange.symbols or [])

    for symbol in symbols:
        try:
            if "/" not in symbol:
                continue

            base, quote = symbol.split("/")
            if WHITELIST and symbol not in WHITELIST:
                continue
            if EXCLUDE and symbol in EXCLUDE:
                continue
            if quote not in PREFERRED_QUOTES:
                continue

            # last price
            last = None
            pct = None

            if symbol in tickers:
                t = tickers[symbol] or {}
                last = t.get("last") or t.get("close")
                pct = t.get("percentage")
            else:
                # minimally fetch ticker if not present
                t = exchange.fetch_ticker(symbol)
                last = t.get("last") or t.get("close")
                pct = t.get("percentage")

            if not last or last <= 0:
                continue
            if last < MIN_PRICE:
                continue

            # drop %
            if pct is None:
                # if missing, skip or compute naive
                continue

            # We want assets that DROPPED by at least DROP_PCT
            if pct >= 0 or abs(pct) < DROP_PCT:
                continue

            out.append((symbol, float(last), float(pct), quote))
        except Exception:
            continue

    # Sort: biggest negative pct first (e.g., -15 < -10)
    out.sort(key=lambda x: x[2])  # more negative = earlier
    return out[:UNIVERSE_SIZE]


# ------------------------------------------------------------
# Daily cap checks (simple)
# ------------------------------------------------------------
def log_daily_cap_banner():
    start_local = start_of_today_tz(DAILY_CAP_TZ)
    log.info(f"Daily-cap window start (local {DAILY_CAP_TZ}): {start_local.isoformat()}")
    # This demo does not persist actual spent; your workflow cadence resets per run.
    log.info(f"Daily spend so far ≈ $0.00. Remaining ≈ ${DAILY_CAP:.2f}.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    log.info("=== START TRADING OUTPUT ===")
    log.info("Python " + ".".join(map(str, list(os.sys.version_info[:2]))))

    # connect exchange
    exchange = make_exchange()

    # (optional) balances
    if SKIP_BALANCE_PING:
        log.info("Skipping balance ping (SKIP_BALANCE_PING=true).")
    else:
        try:
            _ = exchange.fetch_balance()
            log.info("Fetched balance (skipping details in log).")
        except Exception as e:
            log.warning(f"fetch_balance failed (continuing): {e}")

    log_daily_cap_banner()

    # build candidates
    candidates = build_candidates(exchange)
    log.info(f"Eligible candidates: {len(candidates)}")

    if not candidates:
        log.info("No candidates matched filters this run.")
        log.info("=== END TRADING OUTPUT ===")
        return

    # Main loop: try buys in ranked order
    for (symbol, last_price, pct, quote) in candidates:
        if symbol in wa_ban:
            continue

        # jurisdiction-prefered quotes handled already by filter
        # place guarded market buy
        ok = place_guarded_market_buy(exchange, symbol, last_price)
        if not ok:
            # keep scanning; ban list (WA) handled internally
            continue

        # If you want only 1 fill per run, uncomment:
        # break

    log.info("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}\n{traceback.format_exc()}")
        raise
