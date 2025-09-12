#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
from datetime import datetime, timedelta, timezone

import ccxt  # type: ignore


# -------------------- logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("trader")


# -------------------- env helpers --------------------
def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


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
TRADE_AMOUNT = env_float("TRADE_AMOUNT", 10.0)     # USD per BUY
TAKE_PROFIT = env_float("TP", 0.015)               # +1.5% TP
STOP_LOSS = env_float("SL", 0.03)                  # -3% SL
TRAIL = env_float("TRAIL", 0.02)                   # reserved for later
MAX_TRADES_PER_RUN = env_int("MAX_TRADES_PER_RUN", 2)

DROP_PCT = env_float("DROP_PCT", 0.005)            # dip under SMA10 to BUY
MIN_NOTIONAL_BUFFER = env_float("MIN_NOTIONAL_BUFFER", 1.0)

SYMBOLS = env_str("SYMBOLS", "DOGE/USD,XRP/USD,ADA/USD,ETH/USD,BTC/USD").replace(" ", "").split(",")
KRAKEN_API_KEY = env_str("KRAKEN_API_KEY", "")
KRAKEN_SECRET  = env_str("KRAKEN_SECRET", "")

TIMEFRAME = env_str("TIMEFRAME", "5m")
SMA_LEN = env_int("SMA_LEN", 10)


# -------------------- exchange setup --------------------
def make_exchange() -> ccxt.Exchange:
    ex = ccxt.kraken({
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_SECRET,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    return ex


# -------------------- indicators --------------------
def fetch_ohlcv_safe(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
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


# -------------------- balances / positions --------------------
def log_funding_status(exchange, trade_amount, max_trades_per_run, min_notional_buffer=1.0, tag="pre"):
    try:
        bal = exchange.fetch_balance()
        free_usd = float(bal.get("free", {}).get("USD", 0.0) or 0.0)
        held_usd = float(bal.get("used", {}).get("USD", 0.0) or 0.0)
        required = (float(trade_amount) + float(min_notional_buffer)) * int(max_trades_per_run)
        ok = free_usd >= required
        logging.info(
            "Funding(%s) | free_usd=%.2f | held_in_orders=%.2f | required_per_run=%.2f | OK=%s",
            tag, free_usd, held_usd, required, ok,
        )
    except Exception as e:
        logging.warning("Funding(%s) | could not fetch balance: %s", tag, e)


def snapshot_free_balances(exchange: ccxt.Exchange):
    try:
        bal = exchange.fetch_balance()
        return bal.get("free", {}) or {}
    except Exception as e:
        log.warning("balance snapshot failed: %s", e)
        return {}


def get_position_sizes(exchange: ccxt.Exchange):
    """
    Returns dict of BASE -> free amount (spot).
    """
    sizes = {}
    try:
        free = snapshot_free_balances(exchange)
        for k, v in (free.items() if free else []):
            try:
                amt = float(v or 0.0)
            except Exception:
                amt = 0.0
            if amt > 0:
                sizes[k.upper()] = amt
    except Exception as e:
        log.warning("position size fetch failed: %s", e)
    return sizes


def base_from_symbol(symbol: str) -> str:
    return symbol.split("/")[0].upper()


def quote_from_symbol(symbol: str) -> str:
    return symbol.split("/")[-1].upper()


# -------------------- cost basis from trade history --------------------
def fetch_avg_entry_price(exchange: ccxt.Exchange, symbol: str, lookback_days: int = 30) -> float:
    """
    Compute average entry price for current net long position from recent trade history.
    - We fetch myTrades over a lookback window (default ~30 days).
    - Aggregate buys (positive base) and sells (negative).
    - If net base > 0, return weighted average cost for the remaining units.
    """
    since = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
    try:
        trades = exchange.fetch_my_trades(symbol, since=since)
    except Exception as e:
        log.warning("fetch_my_trades failed for %s: %s", symbol, e)
        return float("nan")

    net_base = 0.0
    total_cost_quote_for_net = 0.0  # USD spent for the units still held

    # Process trades oldest->newest so sells reduce earlier buys' inventory
    trades = sorted(trades, key=lambda t: t.get("timestamp", 0))
    inventory = []  # list of (base_units, unit_cost_quote)
    for t in trades:
        side = t.get("side", "")
        amount = float(t.get("amount") or 0.0)
        price = float(t.get("price") or 0.0)
        cost = float(t.get("cost") or (amount * price))
        fee = t.get("fee") or {}
        fee_cost = float(fee.get("cost") or 0.0)
        # treat fee in quote currency (USD) if that’s the case
        cost_with_fee = cost + (fee_cost if (fee.get("currency", "").upper() in ("USD", "")) else 0.0)

        if side == "buy":
            inventory.append([amount, cost_with_fee / max(amount, 1e-12)])
            net_base += amount
        elif side == "sell":
            # remove from inventory FIFO
            to_sell = amount
            while to_sell > 1e-12 and inventory:
                units, unit_cost = inventory[0]
                take = min(units, to_sell)
                units -= take
                to_sell -= take
                net_base -= take
                if units <= 1e-12:
                    inventory.pop(0)
                else:
                    inventory[0][0] = units
        else:
            continue

    if net_base <= 1e-12:
        return float("nan")  # no net long position

    # Weighted average cost of remaining inventory
    for units, unit_cost in inventory:
        total_cost_quote_for_net += units * unit_cost

    avg_entry = total_cost_quote_for_net / max(net_base, 1e-12)
    return avg_entry


# -------------------- order helpers --------------------
def place_market_buy(exchange: ccxt.Exchange, symbol: str, usd_amount: float):
    ticker = exchange.fetch_ticker(symbol)
    price = float(ticker["last"])
    size = usd_amount / price
    size = float(exchange.amount_to_precision(symbol, size))
    return exchange.create_market_buy_order(symbol, size, {})


def place_market_sell(exchange: ccxt.Exchange, symbol: str, base_amount: float):
    size = float(exchange.amount_to_precision(symbol, base_amount))
    return exchange.create_market_sell_order(symbol, size, {})


# -------------------- strategy --------------------
def run_once():
    log.info("Python %s.%s.%s", *os.sys.version_info[:3])
    log.info("Config | DRY_RUN=%s | TRADE_AMOUNT=%.2f | TP=%.3f | SL=%.3f | TRAIL=%.3f | MAX_TRADES=%d",
             DRY_RUN, TRADE_AMOUNT, TAKE_PROFIT, STOP_LOSS, TRAIL, MAX_TRADES_PER_RUN)
    log.info("Symbols: %s", ",".join(SYMBOLS))
    log.info("=== START TRADING OUTPUT ===")

    ex = make_exchange()
    buys_placed = 0
    sells_placed = 0

    # Pre-run funding snapshot
    log_funding_status(ex, TRADE_AMOUNT, MAX_TRADES_PER_RUN, MIN_NOTIONAL_BUFFER, tag="pre")

    # current positions
    positions = get_position_sizes(ex)

    for symbol in SYMBOLS:
        base = base_from_symbol(symbol)
        quote = quote_from_symbol(symbol)
        assert quote == "USD", f"Only USD quote supported, got {symbol}"

        # Fetch current price + SMA
        try:
            ohlcv = fetch_ohlcv_safe(ex, symbol, timeframe=TIMEFRAME, limit=max(SMA_LEN, 20))
            closes = [float(c[4]) for c in ohlcv]
            cur = closes[-1]
            s10 = sma(closes, SMA_LEN)
            dip = (cur - s10) / s10 if s10 and s10 == s10 else 0.0
        except Exception as e:
            log.warning("%s metrics error: %s", symbol, e)
            continue

        held = float(positions.get(base, 0.0))

        # -------- EXIT LOGIC (TP/SL) --------
        if held > 0:
            # Compute average entry price from trade history (last 30 days)
            avg_entry = fetch_avg_entry_price(ex, symbol, lookback_days=30)
            if avg_entry == avg_entry:  # not NaN
                change = (cur - avg_entry) / avg_entry
                log.info(
                    "%s position maintained | held=%.8f | cur=%.6f | entry_avg=%.6f | PnL=%+.2f%% | sma10=%.6f | dip=%+.2f%%",
                    symbol, held, cur, avg_entry, 100.0 * change, s10, 100.0 * dip
                )

                should_take_profit = change >= TAKE_PROFIT
                should_stop_loss = change <= -abs(STOP_LOSS)

                if (should_take_profit or should_stop_loss) and not DRY_RUN:
                    try:
                        reason = "TP" if should_take_profit else "SL"
                        log.info("SELL %s %s | amount=%.8f | cur=%.6f | entry=%.6f | change=%+.2f%%",
                                 symbol, reason, held, cur, avg_entry, 100.0 * change)
                        order = place_market_sell(ex, symbol, held)
                        oid = order.get("id") or order.get("clientOrderId") or "?"
                        status = order.get("status", "?")
                        log.info("Sell order id=%s status=%s", oid, status)
                        sells_placed += 1
                        # after selling, no buy in the same symbol this loop
                        continue
                    except Exception as e:
                        log.error("Exit failed for %s: %s", symbol, e)
                        # fall through to buy logic if desired (but usually continue)
                        continue
                elif (should_take_profit or should_stop_loss) and DRY_RUN:
                    reason = "TP" if should_take_profit else "SL"
                    log.info("DRYRUN SELL %s %s | amount=%.8f | cur=%.6f | entry=%.6f | change=%+.2f%%",
                             symbol, reason, held, cur, avg_entry, 100.0 * change)
                    sells_placed += 1
                    continue
            else:
                log.info(
                    "%s position maintained | held=%.8f | cur=%.6f | sma10=%.6f | dip=%+.2f%% | entry_avg=NA",
                    symbol, held, cur, s10, 100.0 * dip
                )
            # We already hold, so we skip fresh buys for this symbol this run
            continue

        # -------- ENTRY LOGIC (BUY on dip vs SMA) --------
        log.info("%s metrics: cur=%.6f | sma10=%.6f | dip=%+.2f%%", symbol, cur, s10, 100.0 * dip)

        if buys_placed >= MAX_TRADES_PER_RUN:
            continue

        if dip <= -abs(DROP_PCT):
            # Check free USD
            try:
                bal = ex.fetch_balance()
                free_usd = float(bal.get("free", {}).get("USD", 0.0) or 0.0)
            except Exception as e:
                log.warning("could not fetch balance before order: %s", e)
                free_usd = 0.0

            required_per_order = TRADE_AMOUNT + MIN_NOTIONAL_BUFFER
            if free_usd < required_per_order:
                log.info("BUY %s skipped: free_usd=%.2f < required=%.2f", symbol, free_usd, required_per_order)
                continue

            usd_to_spend = TRADE_AMOUNT
            if DRY_RUN:
                size = usd_to_spend / cur
                log.info("DRYRUN BUY %s amount=%.8f (~$%.2f @ %.6f)", symbol, size, usd_to_spend, cur)
                buys_placed += 1
            else:
                try:
                    log.info("BUY %s amount≈$%.2f (@ ~%.6f)", symbol, usd_to_spend, cur)
                    order = place_market_buy(ex, symbol, usd_to_spend)
                    oid = order.get("id") or order.get("clientOrderId") or "?"
                    status = order.get("status", "?")
                    log.info("Buy order id=%s status=%s", oid, status)
                    buys_placed += 1
                except Exception as e:
                    log.error("Entry failed for %s: %s", symbol, e)
                    continue

    log.info("Run complete. buys_placed=%d | sells_placed=%d | DRY_RUN=%s", buys_placed, sells_placed, DRY_RUN)
    log_funding_status(ex, TRADE_AMOUNT, MAX_TRADES_PER_RUN, MIN_NOTIONAL_BUFFER, tag="post")
    log.info("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    run_once()
