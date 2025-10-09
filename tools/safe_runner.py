#!/usr/bin/env python3
"""
safe_runner.py
- Launches your existing main.py.
- If DRY_RUN=ON and no Kraken keys are present, monkey-patches ccxt.kraken
  so private endpoints become safe no-ops (no crashes, no real orders).
- If keys are present (or DRY_RUN=OFF), it does nothing special and your bot
  runs exactly as-is.

Usage (in GitHub Actions): python -u tools/safe_runner.py
"""

import os
import sys
import types

def _has_keys():
    # Accept either naming convention
    k = os.getenv("KRAKEN_API_KEY") or os.getenv("CCXT_API_KEY")
    s = os.getenv("KRAKEN_API_SECRET") or os.getenv("CCXT_API_SECRET")
    return bool(k and s)

def _patch_ccxt_for_dry():
    try:
        import ccxt  # type: ignore
    except Exception as e:
        print(f"[safe_runner] ccxt not available yet: {e}")
        return

    if not hasattr(ccxt, "kraken"):
        print("[safe_runner] ccxt.kraken not found; nothing to patch.")
        return

    Base = ccxt.kraken

    class DryKraken(Base):  # type: ignore
        """A thin wrapper that skips private calls in DRY/no-keys mode."""

        def _dry(self, msg):
            print(f"[DRY] {msg} (skipped)")
        
        # ------- private endpoints we may hit -------
        def fetch_balance(self, params=None):
            self._dry("fetch_balance")
            return {"total": {}, "free": {}, "used": {}}

        def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
            self._dry(f"fetch_open_orders symbol={symbol}")
            return []

        def fetch_closed_orders(self, symbol=None, since=None, limit=None, params=None):
            self._dry(f"fetch_closed_orders symbol={symbol}")
            return []

        def fetch_my_trades(self, symbol=None, since=None, limit=None, params=None):
            self._dry(f"fetch_my_trades symbol={symbol}")
            return []

        def create_order(self, symbol, type, side, amount, price=None, params=None):
            self._dry(f"create_order {symbol} {side} {amount}@{price}")
            # return a realistic-looking stub
            return {
                "id": "DRY-ORDER",
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "closed",
                "info": {"dry_run": True},
            }

        def cancel_order(self, id, symbol=None, params=None):
            self._dry(f"cancel_order id={id} symbol={symbol}")
            return {"id": id, "status": "canceled", "info": {"dry_run": True}}

        # positions/ledgers if your code happens to call them
        def fetch_positions(self, symbols=None, params=None):
            self._dry("fetch_positions")
            return []

        def fetch_ledger(self, code=None, since=None, limit=None, params=None):
            self._dry("fetch_ledger")
            return {}

    # Monkey-patch: your code calls ccxt.kraken(), so we swap the class.
    ccxt.kraken = DryKraken  # type: ignore
    print("[safe_runner] DRY_RUN=ON with no keys → ccxt.kraken patched (private calls are no-ops).")

def main():
    DRY = (os.getenv("DRY_RUN", "ON").upper() == "ON")
    have_keys = _has_keys()

    print(f"[safe_runner] DRY_RUN={DRY}  have_keys={have_keys}")

    # Only patch when DRY and no keys present.
    if DRY and not have_keys:
        _patch_ccxt_for_dry()
    else:
        print("[safe_runner] No patching needed (live or keys present).")

    # Execute your existing main.py in the current interpreter
    entry = "main.py"
    if not os.path.exists(entry):
        print("[safe_runner] ERROR: main.py not found next to repo root.")
        sys.exit(1)

    # Run main.py as if it was executed directly
    code = compile(open(entry, "rb").read(), entry, "exec")
    glb = {"__name__": "__main__"}
    try:
        exec(code, glb, glb)
    except SystemExit as e:
        # propagate exit code cleanly for Actions
        raise
    except Exception as e:
        print(f"[safe_runner] Uncaught exception from main.py: {e}")
        raise

if __name__ == "__main__":
    main()
