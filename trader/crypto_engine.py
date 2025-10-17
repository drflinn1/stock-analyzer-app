# -*- coding: utf-8 -*-
"""
Crypto engine utilities:
- Build ccxt exchange
- Universe scanning with sane filters
- Position snapshot helper
- Simple market buy/sell wrappers
- Equity estimate (USD)

Notes:
- We trade spot only.
- Symbols use Kraken's ccxt market symbols like "ADA/USD".
"""

import os
import math
from typing import Dict, List

import ccxt

# ---------- Helpers ----------
STABLES = {"USD", "USDT", "USDC", "EUR", "GBP"}
EXCLUDE_TICKERS = {"SPX", "PUMP", "BABY", "ALKIMI"}  # expand as needed

def build_exchange(api_key: str, api_secret: str, dry_run: bool) -> ccxt.Exchange:
    ex = ccxt.kraken({
        "apiKey": api_key or "",
        "secret": api_secret or "",
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    # For DRY_RUN we still fetch markets/balances, just don't place orders.
    ex.load_markets()
    return ex

def symbol_to_usd_market(ex: ccxt.Exchange, symbol: str) -> Dict:
    """Return market dict for a symbol like 'ADA/USD' if listed."""
    m = ex.markets.get(symbol)
    if m:
        return m
    # sometimes ccxt uses 'ZUSD' internally; normalize already done by markets
    return None

def _is_bad_coin(symbol: str) -> bool:
    base, quote = symbol.split("/")
    if base.upper() in EXCLUDE_TICKERS:
        return True
    if quote.upper() not in {"USD"}:
        return True
    if base.upper() in STABLES:  # skip stables on base side
        return True
    return False

def pick_candidates(ex: ccxt.Exchange, top_k: int = 25) -> List[Dict]:
    """
    Rank USD pairs by 24h percent change (descending) with liquidity filter.
    Returns list of dicts: {symbol, price, change24h, quoteVolUsd}
    """
    tickers = ex.fetch_tickers()
    out: List[Dict] = []

    for sym, t in tickers.items():
        # Keep normalized Kraken symbols only ( 'XXX/USD' style )
        if "/" not in sym or not sym.endswith("/USD"):
            continue
        if _is_bad_coin(sym):
            continue

        # 24h change %
        change = t.get("percentage")
        if change is None:
            # rough fallback from last/close - open
            last = t.get("last") or t.get("close")
            open_ = t.get("open")
            if last and open_:
                change = (last - open_) / open_ * 100.0
            else:
                continue

        # Volume filter: prefer coins with decent quoted volume in USD
        quote_vol = t.get("quoteVolume") or 0.0
        if quote_vol is None:
            quote_vol = 0.0
        if quote_vol < 10000:  # tiny coins out
            continue

        out.append({
            "symbol": sym,
            "price": float(t.get("last") or t.get("close") or 0.0),
            "change24h": float(change),
            "quoteVolUsd": float(quote_vol),
        })

    out.sort(key=lambda r: (r["change24h"], r["quoteVolUsd"]), reverse=True)
    return out[:max(1, top_k)]

def fetch_positions_snapshot(ex: ccxt.Exchange):
    """
    Return list of open positions (spot holdings > 0) as:
    {symbol, base, base_qty, price, usd_value}
    """
    balances = ex.fetch_balance()
    prices = ex.fetch_tickers()
    positions = []
    for sym, m in ex.markets.items():
        if _is_bad_coin(sym):
            continue
        if not sym.endswith("/USD"):
            continue
        base = m["base"]
        free = balances.get(base, {}).get("free")
        total = balances.get(base, {}).get("total")
        qty = None
        if total:
            qty = float(total)
        elif free:
            qty = float(free)
        if not qty or qty <= 0:
            continue

        price = prices.get(sym, {}).get("last") or prices.get(sym, {}).get("close") or 0.0
        price = float(price or 0.0)
        usd_value = qty * price
        positions.append({
            "symbol": sym,
            "base": base,
            "base_qty": qty,
            "price": price,
            "usd_value": usd_value,
        })
    return positions

def get_cash_balance_usd(ex: ccxt.Exchange) -> float:
    bal = ex.fetch_balance()
    usd = bal.get("USD", {})
    free = float(usd.get("free") or 0.0)
    total = float(usd.get("total") or free)
    return max(free, total)

def estimate_equity_usd(ex: ccxt.Exchange) -> float:
    equity = get_cash_balance_usd(ex)
    tickers = ex.fetch_tickers()
    balances = ex.fetch_balance()
    for sym, m in ex.markets.items():
        if _is_bad_coin(sym) or not sym.endswith("/USD"):
            continue
        base = m["base"]
        qty = float(balances.get(base, {}).get("total") or 0.0)
        if qty <= 0:
            continue
        price = float((tickers.get(sym, {}) or {}).get("last") or 0.0)
        equity += qty * price
    return equity

# ---------- Order wrappers ----------
def _ensure_min_notional(ex: ccxt.Exchange, symbol: str, spend_usd: float) -> float:
    m = ex.markets[symbol]
    min_cost = float(m.get("limits", {}).get("cost", {}).get("min") or 0.0)
    if spend_usd < min_cost:
        return min_cost
    return spend_usd

def place_market_buy(ex: ccxt.Exchange, symbol: str, spend_usd: float):
    spend_usd = _ensure_min_notional(ex, symbol, spend_usd)
    price = float(ex.fetch_ticker(symbol)["last"])
    amount = spend_usd / max(price, 1e-9)
    amount = ex.amount_to_precision(symbol, amount)
    print(f"[order] BUY {symbol} notional≈${spend_usd:.2f} qty≈{amount}")
    return ex.create_market_buy_order(symbol, float(amount))

def place_market_sell(ex: ccxt.Exchange, symbol: str, base_qty: float):
    amount = ex.amount_to_precision(symbol, base_qty)
    print(f"[order] SELL {symbol} qty≈{amount}")
    return ex.create_market_sell_order(symbol, float(amount))
