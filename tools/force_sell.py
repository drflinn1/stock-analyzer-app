#!/usr/bin/env python3
"""
Self-contained Force Sell tool for Kraken (spot).
- Accepts SYMBOL = 'SOON', 'SOONUSD', 'SOON/USD', or 'ALL'
- Uses ccxt directly (no trader.* imports)
- If Kraken blocks a market order with price-protection, retries as LIMIT
  at bid * (1 - LIMIT_SLIP_PCT/100).
Env:
  KRAKEN_API_KEY, KRAKEN_API_SECRET, SYMBOL, LIMIT_SLIP_PCT
"""

import os
import sys
import time
from typing import List, Tuple

import ccxt  # type: ignore

API_KEY = os.getenv("KRAKEN_API_KEY", "")
API_SECRET = os.getenv("KRAKEN_API_SECRET", "")
INPUT_SYMBOL = (os.getenv("SYMBOL") or "").strip()
SLIP_PCT = float(os.getenv("LIMIT_SLIP_PCT", "3.0") or "3.0")


def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}")
    sys.exit(code)


def normalize_pair(sym: str) -> str:
    s = sym.upper().replace(" ", "").replace("-", "")
    if s == "ALL":
        return "ALL"
    if s.endswith("/USD"):
        return s
    if s.endswith("USD"):
        base = s[:-3]
        return f"{base}/USD"
    return f"{s}/USD"


def load_exchange() -> ccxt.Exchange:
    if not API_KEY or not API_SECRET:
        fail("Missing Kraken API secrets.")
    ex = ccxt.kraken({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })
    # Kraken quirk: allow pure market sells without price
    ex.options["createMarketBuyOrderRequiresPrice"] = False
    ex.load_markets()
    return ex


def list_sellable_pairs(ex: ccxt.Exchange) -> List[str]:
    bal = ex.fetch_balance()
    markets = ex.markets
    sellable = []
    for asset, amt in (bal.get("free") or {}).items():
        try:
            if amt and amt > 0:
                pair = f"{asset}/USD"
                if pair in markets:
                    sellable.append(pair)
        except Exception:
            continue
    return sellable


def amount_for_pair(ex: ccxt.Exchange, pair: str) -> float:
    base = pair.split("/")[0]
    bal = ex.fetch_balance()
    free = (bal.get("free") or {}).get(base, 0.0) or 0.0
    if free <= 0:
        # try total as fallback
        total = (bal.get("total") or {}).get(base, 0.0) or 0.0
        return float(total)
    return float(free)


def market_sell(ex: ccxt.Exchange, pair: str, amount: float):
    print(f"[SELL] Market sell {amount} {pair}")
    return ex.create_order(pair, "market", "sell", amount)


def fallback_limit_sell(ex: ccxt.Exchange, pair: str, amount: float, slip_pct: float):
    ticker = ex.fetch_ticker(pair)
    bid = float(ticker.get("bid") or 0.0)
    if bid <= 0:
        fail(f"No bid for {pair}; cannot place limit.")
    price = bid * (1 - slip_pct / 100.0)
    price = round(price, ex.markets[pair]["precision"]["price"])
    print(f"[SELL] Price-protection hit; retry LIMIT @ {price} ({slip_pct}% under bid)")
    # Kraken respects 'IOC' for quick execution; if unsupported it is ignored.
    params = {"timeInForce": "IOC"}
    return ex.create_order(pair, "limit", "sell", amount, price, params)


def sell_pair(ex: ccxt.Exchange, pair: str, slip_pct: float) -> Tuple[str, str]:
    amt = amount_for_pair(ex, pair)
    if amt <= 0:
        return (pair, "SKIP_NO_BALANCE")
    try:
        order = market_sell(ex, pair, amt)
        status = order.get("status") or "placed"
        return (pair, f"MARKET_OK:{status}")
    except Exception as e:
        msg = str(e)
        if "price protection" in msg.lower() or "EOrder:Price" in msg:
            try:
                order = fallback_limit_sell(ex, pair, amt, slip_pct)
                status = order.get("status") or "placed"
                return (pair, f"LIMIT_OK:{status}")
            except Exception as e2:
                return (pair, f"LIMIT_FAIL:{e2}")
        return (pair, f"MARKET_FAIL:{e}")


def main() -> None:
    if not INPUT_SYMBOL:
        fail("SYMBOL input is required (e.g., SOON or ALL).")
    ex = load_exchange()

    targets: List[str]
    if normalize_pair(INPUT_SYMBOL) == "ALL":
        targets = list_sellable_pairs(ex)
        if not targets:
            print("[INFO] Nothing to sell (no USD pairs with balance).")
            sys.exit(0)
    else:
        pair = normalize_pair(INPUT_SYMBOL)
        if pair not in ex.markets:
            fail(f"{pair} is not a known Kraken USD spot market.")
        targets = [pair]

    print(f"[INFO] Selling: {', '.join(targets)}  (slip={SLIP_PCT}%)")
    results = []
    for p in targets:
        try:
            res = sell_pair(ex, p, SLIP_PCT)
            results.append(res)
            print(f"[RESULT] {res[0]} -> {res[1]}")
            time.sleep(0.75)  # polite pacing
        except Exception as e:
            results.append((p, f"ERROR:{e}"))
            print(f"[RESULT] {p} -> ERROR:{e}")

    # Non-zero exit only if every attempt failed
    if any("OK" in r[1] for r in results):
        sys.exit(0)
    if any("SKIP_NO_BALANCE" in r[1] for r in results):
        sys.exit(0)
    fail("All sell attempts failed.", code=2)


if __name__ == "__main__":
    main()
