#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kraken spot bot via CCXT – auto-pick + TP + soft SL + trailing stop + stale-TP cleanup.

Trade loop (each run):
1) Read config/env & connect.
2) Maintain existing positions:
   - For each symbol held:
     * Find most-recent BUY trade (entry px/time, filled amount).
     * Compute highest close since entry (SMA source: 30m OHLCV).
     * If price <= entry*(1-STOP_LOSS_PCT): market sell (soft SL).
     * Else if price <= high*(1-TRAILING_STOP_PCT) and high > entry: market sell (trailing).
     * Cancel/refresh stale TP orders older than STALE_TP_DAYS and re-place TP at current*(1+TP).
     * If no open TP for remaining free amount: place TP.
3) New entries:
   - Gather candidates from SYMBOLS, enforce min-notional + balance,
     compute dip% vs SMA10 (30m), sort by biggest dip, BUY up to MAX_TRADES_PER_RUN.
   - Immediately place TP for the filled amount.

Notes:
- "Soft" SL/TS means the bot places a market SELL when condition is met (no exchange-native stops).
- DRY_RUN avoids private endpoints without keys and only logs actions.

Env knobs:
- DRY_RUN: "true"/"false"
- TRADE_AMOUNT: dollars per BUY
- TAKE_PROFIT_PCT: e.g. "0.02" for +2%
- STOP_LOSS_PCT: e.g. "0.03" for -3%
- TRAILING_STOP_PCT: e.g. "0.02" for 2% below the run-up high
- STALE_TP_DAYS: e.g. "5"  (cancel/re-post older TPs)
- MAX_TRADES_PER_RUN: "1", "2", ...
- MIN_NOTIONAL_BUFFER: dollars buffer kept free
- SYMBOLS: comma list like "DOGE/USD,XRP/USD,ADA/USD,ETH/USD,BTC/USD"
- PYTHON_VERSION only used by workflow

Secrets (for live):
- KRAKEN_API_KEY, KRAKEN_API_SECRET
"""

import os, sys, time, traceback
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
import logging
import ccxt  # type: ignore

# ---------- logging ----------
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

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")

pyver = ".".join(map(str, sys.version_info[:3]))
log.info("Python %s", pyver)

# ---------- read env ----------
DRY_RUN             = env_bool("DRY_RUN", True)
TRADE_AMOUNT_USD    = env_decimal("TRADE_AMOUNT", "10")
TAKE_PROFIT_PCT     = env_decimal("TAKE_PROFIT_PCT", "0.02")
STOP_LOSS_PCT       = env_decimal("STOP_LOSS_PCT", "0.03")
TRAIL_STOP_PCT      = env_decimal("TRAILING_STOP_PCT", "0.02")
STALE_TP_DAYS       = int(env_str("STALE_TP_DAYS", "5"))
MAX_TRADES_PER_RUN  = int(env_str("MAX_TRADES_PER_RUN", "1"))
MIN_NOTIONAL_BUFFER = env_decimal("MIN_NOTIONAL_BUFFER", "1.00")

SYMBOLS_RAW         = env_str("SYMBOLS", "BTC/USD,ETH/USD")
SYMBOLS             = [s.strip().upper() for s in SYMBOLS_RAW.split(",") if s.strip()]

API_KEY             = env_str("KRAKEN_API_KEY", "")
API_SECRET          = env_str("KRAKEN_API_SECRET", "")

log.info("Config | DRY_RUN=%s | TRADE_AMOUNT=%s | TP=%s | SL=%s | TRAIL=%s | MAX_TRADES=%s",
         DRY_RUN, TRADE_AMOUNT_USD, TAKE_PROFIT_PCT, STOP_LOSS_PCT, TRAIL_STOP_PCT, MAX_TRADES_PER_RUN)
log.info("Symbols: %s", ", ".join(SYMBOLS))

# ---------- exchange ----------
def build_exchange() -> ccxt.kraken:
    kwargs = {"enableRateLimit": True, "options": {"adjustForTimeDifference": True}}
    if not DRY_RUN:
        if not API_KEY or not API_SECRET:
            raise RuntimeError("Live trading requires KRAKEN_API_KEY and KRAKEN_API_SECRET.")
        kwargs.update({"apiKey": API_KEY, "secret": API_SECRET})
    return ccxt.kraken(kwargs)

def pct_change(a: Decimal, b: Decimal) -> Decimal:
    if a == 0: return Decimal("0")
    return (b - a) / a

# ---------- market data / limits ----------
def fetch_min_notional_usd(ex: ccxt.Exchange, symbol: str) -> Decimal:
    m = ex.market(symbol)
    amount_min = None
    try:
        amount_min = m.get("limits", {}).get("amount", {}).get("min")
    except Exception:
        pass
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
    if base in ("BTC",): return Decimal("10")
    if base in ("ETH","SOL","ADA","XRP","DOGE","LTC","ATOM","DOT","MATIC"): return Decimal("5")
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

def fetch_price(ex: ccxt.Exchange, symbol: str) -> Decimal:
    t = ex.fetch_ticker(symbol)
    return Decimal(str(t["last"] or t["close"]))

def dip_metrics(ex: ccxt.Exchange, symbol: str):
    """Return (cur, sma10, dip_pct). 30m timeframe."""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="30m", limit=24)
        closes = [Decimal(str(c[4])) for c in ohlcv if c and c[4] is not None]
        if len(closes) < 10: return None
        cur = closes[-1]
        sma = sum(closes[-10:]) / Decimal(10)
        dip_pct = pct_change(sma, cur) * Decimal("100")
        return cur, sma, dip_pct
    except Exception as e:
        log.warning("metrics failed for %s: %s", symbol, e)
        return None

def notional_to_amount(ex: ccxt.Exchange, symbol: str, notional_usd: Decimal) -> Decimal:
    price = fetch_price(ex, symbol)
    if price <= 0: raise RuntimeError(f"No valid price for {symbol}.")
    amt = (notional_usd / price).quantize(Decimal("0.00000001"))
    try:
        m = ex.market(symbol)
        amt_min = m.get("limits", {}).get("amount", {}).get("min")
        if amt_min:
            amt_min = Decimal(str(amt_min))
            if amt < amt_min: amt = amt_min
    except Exception:
        pass
    return amt

# ---------- account / trades ----------
def last_buy_trade(ex: ccxt.Exchange, symbol: str):
    """Return dict with 'price','amount','timestamp' of the most recent BUY trade."""
    try:
        trades = ex.fetch_my_trades(symbol, limit=50)
        buys = [t for t in trades if str(t.get("side")).lower() == "buy"]
        if not buys: return None
        t = buys[-1]  # latest
        return {
            "price": Decimal(str(t.get("price"))),
            "amount": Decimal(str(t.get("amount"))),
            "timestamp": datetime.fromtimestamp(t.get("timestamp")/1000, tz=timezone.utc),
        }
    except Exception as e:
        log.info("No buy trade info for %s: %s", symbol, e)
        return None

def free_base_amount(ex: ccxt.Exchange, symbol: str) -> Decimal:
    base = symbol.split("/")[0]
    try:
        bal = ex.fetch_free_balance()
        return Decimal(str(bal.get(base, 0)))
    except Exception:
        return Decimal("0")

def list_open_sell_limits(ex: ccxt.Exchange, symbol: str):
    """List open TP-like sells (limit sells above current price)."""
    out = []
    try:
        orders = ex.fetch_open_orders(symbol)
        px = fetch_price(ex, symbol)
        for o in orders:
            if str(o.get("side")).lower() == "sell" and str(o.get("type")).lower() == "limit":
                price = Decimal(str(o.get("price") or 0))
                if price > px * Decimal("0.995"):  # TP-ish
                    out.append(o)
    except Exception as e:
        log.info("open orders fetch failed for %s: %s", symbol, e)
    return out

def cancel_order_safe(ex: ccxt.Exchange, order_id: str, symbol: str):
    try:
        ex.cancel_order(order_id, symbol)
        log.info("Canceled order %s (%s)", order_id, symbol)
    except Exception as e:
        log.info("Cancel failed for %s: %s", order_id, e)

# ---------- exit helpers ----------
def place_tp_limit(ex: ccxt.Exchange, symbol: str, base_amount: Decimal, anchor_price: Decimal, tp_pct: Decimal, dry_run: bool):
    target = (anchor_price * (Decimal("1") + tp_pct)).quantize(Decimal("0.00000001"))
    # round to 0.01 for most USD pairs display; Kraken will quantize internally
    target_disp = target.quantize(Decimal("0.01"))
    if dry_run:
        log.info("[DRY_RUN] TP: %s sell %s @ %s (+%s%%)", symbol, base_amount, target_disp, float(tp_pct*100))
        return
    log.info("Placing TP limit: %s sell %s @ %s", symbol, base_amount, target_disp)
    ex.create_limit_sell_order(symbol, float(base_amount), float(target))

def market_sell_all(ex: ccxt.Exchange, symbol: str, base_amount: Decimal, reason: str, dry_run: bool):
    if base_amount <= 0:
        return
    if dry_run:
        log.info("[DRY_RUN] EXIT %s sell %s @ market (%s)", symbol, base_amount, reason)
        return
    log.info("EXIT %s sell %s @ market (%s)", symbol, base_amount, reason)
    ex.create_market_sell_order(symbol, float(base_amount))

# ---------- strategy maintenance (SL/TS/TP refresh) ----------
def maintain_positions(ex: ccxt.Exchange, symbol: str):
    """Return True if we still hold the asset after maintenance; False if fully exited."""
    amt_free = free_base_amount(ex, symbol)
    if amt_free <= 0:
        return False

    trade = last_buy_trade(ex, symbol)
    if not trade:
        # No entry info; ensure a generic TP exists at current + TP%
        px = fetch_price(ex, symbol)
        place_tp_limit(ex, symbol, amt_free, px, TAKE_PROFIT_PCT, dry_run=DRY_RUN)
        return True

    entry_px = trade["price"]
    entry_ts = trade["timestamp"]
    now = now_utc()

    # Highest close since entry (30m bars)
    try:
        since_ms = int(entry_ts.timestamp() * 1000) - 1_000  # small pad
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="30m", since=since_ms, limit=200)
        closes = [Decimal(str(c[4])) for c in ohlcv if c and c[4] is not None]
        high = max(closes) if closes else fetch_price(ex, symbol)
    except Exception:
        high = fetch_price(ex, symbol)

    cur = fetch_price(ex, symbol)

    # Soft stop-loss
    sl_px = entry_px * (Decimal("1") - STOP_LOSS_PCT)
    if cur <= sl_px:
        market_sell_all(ex, symbol, amt_free, reason=f"SL hit: cur {cur} <= {sl_px}", dry_run=DRY_RUN)
        return False

    # Trailing stop (only if advanced from entry)
    if high > entry_px:
        trail_trigger = high * (Decimal("1") - TRAIL_STOP_PCT)
        if cur <= trail_trigger:
            market_sell_all(ex, symbol, amt_free, reason=f"TS hit: cur {cur} <= {trail_trigger} (high {high})", dry_run=DRY_RUN)
            return False

    # Stale TP cleanup & ensure a TP exists
    open_tps = list_open_sell_limits(ex, symbol)
    # Cancel TPs older than N days
    if open_tps:
        cutoff = now - timedelta(days=STALE_TP_DAYS)
        for o in open_tps:
            ts = o.get("timestamp") or o.get("lastTradeTimestamp")
            if ts:
                ts_dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                if ts_dt < cutoff:
                    cancel_order_safe(ex, o.get("id"), symbol)
    # Re-check open TPs after cleanup
    open_tps = list_open_sell_limits(ex, symbol)
    if not open_tps:
        # place a fresh TP anchored to entry (safer) or current if you prefer
        place_tp_limit(ex, symbol, amt_free, entry_px, TAKE_PROFIT_PCT, dry_run=DRY_RUN)

    log.info("%s maintained | entry=%s | high=%s | cur=%s", symbol, entry_px, high, cur)
    return True

# ---------- trade entries (auto-pick) ----------
def pick_candidates(ex: ccxt.Exchange, free_usd: Decimal):
    cands = []
    for symbol in SYMBOLS:
        if symbol not in ex.markets:
            log.info("Skip %s: not on exchange.", symbol); continue
        min_notional = fetch_min_notional_usd(ex, symbol)
        want = TRADE_AMOUNT_USD
        need = want + MIN_NOTIONAL_BUFFER
        if want < min_notional:
            log.info("Bump needed for %s: TRADE_AMOUNT(%s) < min_notional(~%s). Skipping.", symbol, want, min_notional); continue
        if free_usd < need:
            log.info("Insufficient free USD for %s: need>=%s; have %s. Skipping.", symbol, need, free_usd); continue
        m = dip_metrics(ex, symbol)
        if not m: log.info("Skip %s: not enough data.", symbol); continue
        cur, sma, dip = m
        log.info("%s metrics: cur=%s | sma10=%s | dip=%.2f%%", symbol, cur, sma, float(dip))
        cands.append((symbol, dip, cur))
    # most negative dip first
    cands.sort(key=lambda x: x[1])
    if MAX_TRADES_PER_RUN <= 0: return []
    return cands[:MAX_TRADES_PER_RUN]

def enter_trade(ex: ccxt.Exchange, symbol: str):
    amt = notional_to_amount(ex, symbol, TRADE_AMOUNT_USD)
    entry_px = fetch_price(ex, symbol)
    notional = (amt * entry_px).quantize(Decimal("0.01"))
    if DRY_RUN:
        log.info("[DRY_RUN] BUY %s amount=%s (~$%s at %s)", symbol, amt, notional, entry_px)
        place_tp_limit(ex, symbol, amt, entry_px, TAKE_PROFIT_PCT, dry_run=True)
        return
    log.info("BUY %s amount=%s (~$%s at %s)", symbol, amt, notional, entry_px)
    order = ex.create_market_buy_order(symbol, float(amt))
    # Use actual filled amount if provided
    filled_amt = Decimal(str(order.get("amount") or order.get("filled") or amt))
    log.info("Buy order id=%s status=%s", order.get("id"), order.get("status"))
    place_tp_limit(ex, symbol, filled_amt, entry_px, TAKE_PROFIT_PCT, dry_run=False)

# ---------- driver ----------
def run_once():
    log.info("=== START TRADING OUTPUT ===")
    ex = build_exchange()
    ex.load_markets()

    # 1) Maintain existing positions (SL/TS/TP refresh)
    for symbol in SYMBOLS:
        try:
            still_holding = maintain_positions(ex, symbol)
            if still_holding:
                log.info("%s position maintained.", symbol)
        except Exception as e:
            log.warning("Maintenance failed for %s: %s", symbol, e)

    # 2) New entries (auto-pick)
    free_usd = ensure_free_usd(ex)
    log.info("Free USD: %s", free_usd)
    picks = pick_candidates(ex, free_usd)
    trades_placed = 0
    for symbol, dip, _ in picks:
        try:
            enter_trade(ex, symbol)
            trades_placed += 1
        except Exception as e:
            log.error("Entry failed for %s: %s", symbol, e)

    log.info("Run complete. trades_placed=%s | DRY_RUN=%s", trades_placed, DRY_RUN)
    log.info("=== END TRADING OUTPUT ===")

# ---------- entry ----------
if __name__ == "__main__":
    try:
        run_once()
    except ccxt.BaseError as e:
        log.error("Exchange error: %s", e)
        log.debug("Trace:\n%s", traceback.format_exc()); raise
    except Exception as e:
        log.error("Fatal error: %s", e)
        log.debug("Trace:\n%s", traceback.format_exc()); raise
