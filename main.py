#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crypto Live Trader (Kraken via CCXT) — single-file version
- Reads safety knobs from environment (perfect for GitHub Actions)
- Scans USD markets, picks a candidate using simple momentum + RSI gate
- Places MARKET orders on Kraken correctly: type="market", side="buy"/"sell"
- Prints clear START/END markers for easy log scraping

Environment knobs (strings okay):
  DRY_RUN            -> "true"/"false"
  EXCHANGE           -> default "kraken"
  API_KEY, API_SECRET (required for live)
  TOP_N              -> default "30"
  MIN_USD_VOL        -> default "1000000" (24h quoteVolume in USD)
  TIMEFRAME          -> default "5m"
  RSI_LEN            -> default "14"
  RSI_MAX            -> default "60"   (only buy if RSI <= RSI_MAX)
  DROP_PCT           -> default "0.60" (require 5m change <= -DROP_PCT)
  PER_TRADE_USD      -> default "20"
  DAILY_CAP_USD      -> default "20"   (no persistence; acts as remaining for this run)
  PRIVATE_API        -> "true"/"false" (when false, will not touch private endpoints)
"""

import os
import math
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

try:
    import ccxt
except Exception as e:
    raise SystemExit(f"ccxt is required. pip install ccxt\n{e}")

# ---------- Utilities ----------

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v not in (None, "") else default
    except Exception:
        return default

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i-1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / period / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def pct_change(a: float, b: float) -> Optional[float]:
    # percent from a -> b
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a * 100.0

# ---------- Exchange ----------

def build_exchange() -> ccxt.Exchange:
    ex_name = env_str("EXCHANGE", "kraken").lower()
    api_key = env_str("API_KEY", "")
    api_secret = env_str("API_SECRET", "")
    klass = getattr(ccxt, ex_name)
    exchange = klass({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            # Kraken likes explicit order types and cost calculations
            "tradesLimit": 200,
        },
    })
    return exchange

# ---------- Order helper (Kraken-safe MARKET) ----------

def place_market_order(exchange: ccxt.Exchange, symbol: str, side: str,
                       usd_amount: float, price: float) -> Dict:
    """
    Place a MARKET order using USD budget, converting to base amount:
    amount = usd_amount / price, adjusted for precision and min size when known.
    """
    if price is None or price <= 0:
        raise ValueError("place_market_order: price must be a positive float")

    amount = usd_amount / price

    # Try to respect precision/limits if markets metadata is present
    try:
        markets = getattr(exchange, "markets", None)
        if markets and symbol in markets:
            info = markets[symbol]
            prec = info.get("precision", {}).get("amount")
            if prec is not None:
                amount = float(f"{amount:.{prec}f}")
            min_amt = info.get("limits", {}).get("amount", {}).get("min")
            if min_amt and amount < min_amt:
                amount = min_amt
    except Exception:
        pass

    # Kraken requires type="market" (NOT "buy"/"sell"). side carries buy/sell.
    return exchange.create_order(
        symbol=symbol,
        type="market",
        side=side,
        amount=amount,
        price=None,
        params={}
    )

# ---------- Scanner ----------

def fetch_universe(exchange: ccxt.Exchange, top_n: int, min_usd_vol: float) -> List[str]:
    """
    Build a USD-quoted universe filtered by 24h volume (quote volume in USD).
    Falls back gracefully if quoteVolume not available.
    """
    exchange.load_markets()
    tickers = exchange.fetch_tickers()
    candidates: List[Tuple[str, float]] = []  # (symbol, vol)

    for sym, t in tickers.items():
        # Only USD-quoted spot pairs like XXX/USD
        if "/USD" not in sym:
            continue
        if t is None:
            continue
        vol = None
        # prefer quoteVolume in USD if exposed
        if isinstance(t, dict):
            vol = t.get("quoteVolume", None)
            if vol is None:
                # fallback estimate from baseVolume * last
                base_vol = t.get("baseVolume", None)
                last = t.get("last", None)
                if base_vol is not None and last is not None:
                    vol = base_vol * last
        try:
            vol = float(vol) if vol is not None else 0.0
        except Exception:
            vol = 0.0
        if vol >= min_usd_vol:
            candidates.append((sym, vol))

    # Sort by descending volume and take top_n
    candidates.sort(key=lambda x: x[1], reverse=True)
    symbols = [s for s, _ in candidates[:top_n]]
    return symbols

def preview_metrics(exchange: ccxt.Exchange, symbol: str, timeframe: str, rsi_len: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (pct_5m_change, rsi_val, last_price)
    """
    limit = max(100, rsi_len + 2)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    closes = [c[4] for c in ohlcv if c and c[4] is not None]
    if len(closes) < 3:
        return (None, None, None)
    last = float(closes[-1])
    prev = float(closes[-2])
    change = pct_change(prev, last)
    rsi_val = rsi(closes, rsi_len)
    return (change, rsi_val, last)

# ---------- Main ----------

def main() -> None:
    print("=== START TRADING OUTPUT ===")
    print(f"{now_iso()} | run started | DRY_RUN={env_str('DRY_RUN', 'true')}")

    # Read knobs
    dry_run = env_bool("DRY_RUN", True)
    private_api = env_bool("PRIVATE_API", True)
    top_n = int(env_float("TOP_N", 30))
    min_usd_vol = float(env_float("MIN_USD_VOL", 1_000_000))
    timeframe = env_str("TIMEFRAME", "5m")
    rsi_len = int(env_float("RSI_LEN", 14))
    rsi_max = float(env_float("RSI_MAX", 60.0))
    drop_pct_gate = float(env_float("DROP_PCT", 0.60))  # positive number, e.g., 0.60 for -0.60%
    per_trade_usd = float(env_float("PER_TRADE_USD", 20.0))
    daily_cap_usd = float(env_float("DAILY_CAP_USD", 20.0))  # interpreted as "remaining" for this run

    print(f"{now_iso()} | universe_mode=auto | quote=USD | top_n={top_n} | min_usd_vol={min_usd_vol:,.2f}")
    print(f"{now_iso()} | scanning=['USD/USDT','BTC/USD','DOGE/USD','XRP/USD','SOL/USD','USDC/USD', '...']")

    # Exchange
    exchange = build_exchange()
    ex_name = exchange.id
    print(f"{now_iso()} | Loaded broker: {ex_name} | private_api={'ON' if private_api else 'OFF'}")

    # Universe
    try:
        universe = fetch_universe(exchange, top_n=top_n, min_usd_vol=min_usd_vol)
    except Exception as e:
        print(f"Universe fetch failed: {e}")
        universe = []

    # Private state
    usd_free = None
    open_trades_ct = 0
    dust_ignored = 0
    if private_api and not dry_run:
        try:
            bal = exchange.fetch_balance()
            usd_free = float(bal.get("USD", {}).get("free", 0.0))
        except Exception as e:
            print(f"Warning: could not fetch USD balance: {e}")
            usd_free = None
        try:
            open_orders = exchange.fetch_open_orders()
            open_trades_ct = len(open_orders)
        except Exception:
            open_trades_ct = 0
        try:
            # Dust = tiny non-USD balances
            for k, v in bal.items():
                if k in ("info", "free", "used", "total", "USD"):
                    continue
                try:
                    if isinstance(v, dict):
                        amt = float(v.get("free", 0.0))
                    else:
                        amt = float(v or 0.0)
                    if 0 < amt < 1e-6:
                        dust_ignored += 1
                except Exception:
                    pass
        except Exception:
            pass

    if usd_free is None:
        # For DRY_RUN or if balance failed: print a placeholder
        usd_free = 0.0 if dry_run else usd_free
    print(f"{now_iso()} | budget | USD_free=${usd_free:,.2f} | daily_remaining=${daily_cap_usd:.2f} | open_trades={open_trades_ct}/1 | dust_ignored={dust_ignored}")

    # Scan candidates
    previews = []
    for sym in universe:
        try:
            chg, rsi_val, last = preview_metrics(exchange, sym, timeframe=timeframe, rsi_len=rsi_len)
            if chg is None or rsi_val is None or last is None:
                continue
            previews.append((sym, chg, rsi_val, last))
        except Exception:
            # ignore pairs that fail fetch
            continue

    # Show a compact preview line of top few by worst (most negative) 5m change
    previews_sorted = sorted(previews, key=lambda x: x[1])  # ascending (most negative first)
    preview_show = []
    for sym, chg, rv, _last in previews_sorted[:6]:
        preview_show.append(f"{sym} Δ={chg:.2f}% rsi={rv:.2f} ✓")
    print(f"{now_iso()} | preview_tops = [{'; '.join(preview_show)}]")

    # Pick a best candidate that meets gates: 5m change <= -DROP_PCT and RSI <= RSI_MAX
    best = None
    for sym, chg, rv, last in previews_sorted:
        if chg <= -abs(drop_pct_gate) and rv <= rsi_max:
            best = (sym, chg, rv, last)
            break

    if best is None:
        print(f"{now_iso()} | No candidate passed gates (Δ <= -{abs(drop_pct_gate):.2f}% AND RSI <= {rsi_max:.2f}).")
        print("=== END TRADING OUTPUT ===")
        return

    sym, chg, rv, last = best
    print(f"{now_iso()} | Best candidate | {sym} 5m_change={chg:.2f}% (gate -{abs(drop_pct_gate):.2f}%), RSI {rv:.2f} -> BUY ${per_trade_usd:.2f} @ {last:.6f}")

    # Safety checks
    if daily_cap_usd < per_trade_usd:
        print(f"Guard: DAILY_CAP_USD (${daily_cap_usd:.2f}) < PER_TRADE_USD (${per_trade_usd:.2f}) → skipping buy.")
        print("=== END TRADING OUTPUT ===")
        return
    if not dry_run and private_api:
        if usd_free is not None and usd_free < per_trade_usd * 1.01:  # include fee headroom
            print(f"Guard: Insufficient free USD (${usd_free:.2f}) for ${per_trade_usd:.2f} trade → skipping.")
            print("=== END TRADING OUTPUT ===")
            return

    buys_placed = 0
    sells_placed = 0

    # Execute BUY (market)
    try:
        if dry_run or not private_api:
            print(f"[DRY] BUY {sym} ${per_trade_usd:.2f} at ~{last:.6f} (market)")
        else:
            order = place_market_order(exchange, sym, "buy", usd_amount=per_trade_usd, price=last)
            oid = order.get("id", "unknown")
            cost = order.get("cost", None)
            amount = order.get("amount", None)
            print(f"[LIVE] BUY placed: id={oid} | amount={amount} | est_cost={cost}")
            buys_placed += 1
    except ccxt.BaseError as e:
        # Surface the exact Kraken/ccxt message (original problem was type="buy" vs "market")
        try:
            msg = getattr(e, "args", [""])[0]
            if isinstance(msg, dict):
                msg = json.dumps(msg)
        except Exception:
            msg = str(e)
        print(f"Error: create_order failed: {e.__class__.__name__}: {msg}")
    except Exception as e:
        print(f"Unexpected error during order: {e}")

    print(f"Run complete. buys_placed={buys_placed} | sells_placed={sells_placed} | DRY_RUN={str(dry_run)}")
    print("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    main()
