# tools/run_with_safe_sell.py
"""
Safe-sell wrapper for ccxt that prevents a run from aborting when
a market sell raises InsufficientFunds (fees/slippage/min-notional).
- Trims the sell quantity by a small epsilon (SELL_SAFETY_EPS_PCT)
- Optionally cancels open orders before the sell
- Retries once with 1% less if the first safe sell still fails
Then it runs your normal bot entrypoint (main.py).
"""

import os
import runpy

try:
    import ccxt
    from ccxt.base.errors import InsufficientFunds
except Exception as e:
    print(f"[safe-sell] ccxt not available: {type(e).__name__}: {e}")
    raise

# Read safety knobs from env (strings in YAML)
EPS_PCT = float(os.getenv("SELL_SAFETY_EPS_PCT", "0.25")) / 100.0
CANCEL_BEFORE = os.getenv("CANCEL_OPEN_ORDERS_BEFORE_SELL", "true").lower() == "true"

# Keep the original ccxt method
_ORIG_CREATE_ORDER = ccxt.base.exchange.Exchange.create_order

def _safe_sell_wrapper(self, symbol, type, side, amount, *args, **kwargs):
    """
    Only intercept market sells. Everything else goes to the original method.
    """
    if (side or "").lower() != "sell" or (type or "").lower() != "market":
        return _ORIG_CREATE_ORDER(self, symbol, type, side, amount, *args, **kwargs)

    # Try normal sell first; on error, fall back to safe path
    try:
        return _ORIG_CREATE_ORDER(self, symbol, type, side, amount, *args, **kwargs)
    except InsufficientFunds:
        pass
    except Exception as e:
        print(f"[sellwarn] {symbol}: initial sell error {type(e).__name__}: {e}")

    # Figure out base asset and free balance
    try:
        m = self.market(symbol) if hasattr(self, "market") else None
        base = (m.get("base") if m else None) or symbol.split("/")[0]
        bal = self.fetch_balance() or {}
        free = (bal.get(base, {}) or {}).get("free", 0) or 0.0
    except Exception as e:
        print(f"[sellwarn] {symbol}: fetch balance/market error {type(e).__name__}: {e}")
        free = 0.0

    # Trim size by epsilon of free balance (to leave room for fees/slippage)
    try:
        safe_amt = min(float(amount), float(free) * (1.0 - EPS_PCT))
        safe_amt = float(self.amount_to_precision(symbol, safe_amt))
    except Exception:
        safe_amt = 0.0

    if safe_amt <= 0:
        print(f"[sellskip] {symbol}: qty after trim is 0 (free={free}) -> skip")
        return {"id": None, "status": "skipped", "symbol": symbol, "amount": 0.0}

    # Optional: clear open orders first
    if CANCEL_BEFORE:
        try:
            self.cancel_all_orders(symbol)
        except Exception:
            pass

    # Try safe sell once
    try:
        order = _ORIG_CREATE_ORDER(self, symbol, "market", "sell", safe_amt, *args, **kwargs)
        print(f"[order] SELL {symbol} qty={safe_amt} id={order.get('id')}")
        return order
    except InsufficientFunds:
        pass
    except Exception as e:
        print(f"[sellwarn] {symbol}: safe sell error {type(e).__name__}: {e}")

    # Retry with 1% less
    try:
        safe_amt2 = float(self.amount_to_precision(symbol, safe_amt * 0.99))
        if safe_amt2 > 0:
            order = _ORIG_CREATE_ORDER(self, symbol, "market", "sell", safe_amt2, *args, **kwargs)
            print(f"[order] SELL {symbol} retry qty={safe_amt2} id={order.get('id')}")
            return order
    except Exception as e:
        print(f"[sellwarn] {symbol}: retry error {type(e).__name__}: {e}")

    print(f"[sellskip] {symbol}: insufficient funds after retries -> skip")
    return {"id": None, "status": "skipped", "symbol": symbol, "amount": safe_amt}

# Monkey-patch ccxt globally so your bot's sell calls hit the wrapper
ccxt.base.exchange.Exchange.create_order = _safe_sell_wrapper

# Finally, run the normal bot
runpy.run_path("main.py", run_name="__main__")
