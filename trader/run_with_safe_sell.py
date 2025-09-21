# tools/run_with_safe_sell.py
"""
Runs main.py after installing a resilient SELL patch for ccxt exchanges.
Prevents run-stopping 'Insufficient funds' on market sells by:
  - using FREE balance with a safety epsilon
  - rounding to exchange precision
  - optional cancel_all_orders before sell
  - retrying once with -1%
If still not fillable, logs and returns a 'skipped' order dict instead of raising.
"""

import os
import runpy
import ccxt
from ccxt.base.errors import InsufficientFunds

EPS_PCT = float(os.getenv("SELL_SAFETY_EPS_PCT", "0.25")) / 100.0
CANCEL_BEFORE = os.getenv("CANCEL_OPEN_ORDERS_BEFORE_SELL", "true").lower() == "true"

# keep original for fallback / calls
_ORIG_CREATE_ORDER = ccxt.base.exchange.Exchange.create_order

def _safe_sell_wrapper(self, symbol, type, side, amount, *args, **kwargs):
    # For non-sell or non-market, do normal behavior
    if (side or "").lower() != "sell" or (type or "").lower() != "market":
        return _ORIG_CREATE_ORDER(self, symbol, type, side, amount, *args, **kwargs)

    try:
        # Try normal first — it may succeed.
        return _ORIG_CREATE_ORDER(self, symbol, type, side, amount, *args, **kwargs)
    except InsufficientFunds:
        pass  # we'll try safely below
    except Exception as e:
        # unknown exchange error — try safe path anyway
        print(f"[sellwarn] {symbol}: initial sell raised {type(e).__name__}: {e}")

    # Lookup base coin and free balance
    try:
        m = self.market(symbol) if hasattr(self, "market") else None
        base = (m.get("base") if m else None) or symbol.split("/")[0]
        bal = self.fetch_balance() or {}
        free = (bal.get(base, {}) or {}).get("free", 0) or 0.0
    except Exception as e:
        print(f"[sellwarn] {symbol}: failed to fetch balance/market: {type(e).__name__}: {e}")
        free = 0.0

    # Compute safe amount
    try:
        safe_amt = min(float(amount), float(free) * (1.0 - EPS_PCT))
        safe_amt = float(self.amount_to_precision(symbol, safe_amt))
    except Exception:
        safe_amt = 0.0

    if safe_amt <= 0:
        print(f"[sellskip] {symbol}: qty adjusted to 0 (free={free}) -> skip")
        return {"id": None, "status": "skipped", "symbol": symbol, "amount": 0.0}

    if CANCEL_BEFORE:
        try:
            self.cancel_all_orders(symbol)
        except Exception:
            pass  # non-fatal

    # Attempt 1: safe size
    try:
        order = _ORIG_CREATE_ORDER(self, symbol, "market", "sell", safe_amt, *args, **kwargs)
        print(f"[order] SELL {symbol} qty={safe_amt} ok id={order.get('id')}")
        return order
    except InsufficientFunds:
        pass
    except Exception as e:
        print(f"[sellwarn] {symbol}: safe sell attempt err {type(e).__name__}: {e}")

    # Attempt 2: trim by 1%
    try:
        safe_amt2 = float(self.amount_to_precision(symbol, safe_amt * 0.99))
        if safe_amt2 > 0:
            order = _ORIG_CREATE_ORDER(self, symbol, "market", "sell", safe_amt2, *args, **kwargs)
            print(f"[order] SELL {symbol} retry qty={safe_amt2} ok id={order.get('id')}")
            return order
    except Exception as e:
        print(f"[sellwarn] {symbol}: retry err {type(e).__name__}: {e}")

    print(f"[sellskip] {symbol}: insufficient funds after retries -> skip")
    # Return a benign object so caller code continues without throwing
    return {"id": None, "status": "skipped", "symbol": symbol, "amount": safe_amt}

# Monkeypatch ccxt for all exchanges
ccxt.base.exchange.Exchange.create_order = _safe_sell_wrapper

# Finally, run your existing main.py in-process so the patch applies.
runpy.run_path("main.py", run_name="__main__")
