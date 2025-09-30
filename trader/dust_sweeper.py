# trader/dust_sweeper.py
# Prints progress [1/5]..[5/5]. Safe in DRY_RUN. Attempts to connect to CCXT
# to read balances; if ccxt setup/env is missing, still completes gracefully
# with clear logs.

from __future__ import annotations
import os, time

DRY_RUN = os.getenv("DRY_RUN", "true").lower()
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
BASE = os.getenv("BASE_CURRENCY", "USD")

def log_step(i: int, n: int, msg: str) -> None:
    print(f"[{i}/{n}] {msg}", flush=True)

def try_import_ccxt():
    try:
        import ccxt  # type: ignore
        return ccxt
    except Exception as e:
        print(f"[sweeper] ccxt not available ({e}); proceeding in no-ccxt mode.")
        return None

def get_exchange(ccxt):
    api_key = os.getenv("CCXT_API_KEY") or ""
    api_secret = os.getenv("CCXT_API_SECRET") or ""
    api_password = os.getenv("CCXT_API_PASSWORD") or None

    if not hasattr(ccxt, EXCHANGE_ID):
        raise RuntimeError(f"Unknown exchange id: {EXCHANGE_ID}")

    klass = getattr(ccxt, EXCHANGE_ID)
    opts = {"apiKey": api_key, "secret": api_secret}
    if api_password:
        opts["password"] = api_password
    ex = klass(opts)
    ex.options = getattr(ex, "options", {})
    # Be conservative: avoid any trading here; we only read balances.
    return ex

def main() -> int:
    log_step(1, 5, "Dust Sweeper starting")
    ccxt = try_import_ccxt()

    balances = {}
    if ccxt:
        try:
            log_step(2, 5, f"Connecting to exchange: {EXCHANGE_ID}")
            ex = get_exchange(ccxt)
            log_step(3, 5, "Fetching balances (read-only)")
            ex.load_markets()  # safe metadata call
            balances = ex.fetch_balance().get("total", {}) or {}
        except Exception as e:
            print(f"[sweeper] Balance read failed: {e}")
    else:
        log_step(2, 5, "Skipping exchange connect (no ccxt)")
        log_step(3, 5, "Skipping balance fetch")

    # Identify tiny non-BASE 'dust' (purely informational)
    dust = {}
    if balances:
        for asset, amt in balances.items():
            if not asset or asset.upper() == BASE.upper():
                continue
            try:
                val = float(amt)
            except Exception:
                continue
            if 0 < val < 1e-3:  # tiny by raw units; heuristic
                dust[asset] = val

    if dust:
        log_step(4, 5, f"Found tiny residuals (not converting in DRY_RUN={DRY_RUN}): {dust}")
    else:
        log_step(4, 5, "No tiny residuals detected or balances unavailable")

    # No actual conversions here; just a heartbeat and summary.
    log_step(5, 5, "Dust Sweeper complete ✅")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
