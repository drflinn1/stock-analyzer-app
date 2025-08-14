# runner.py — headless entrypoint for schedulers (GitHub Actions, cron, etc.)
import os, inspect, importlib

# Prefer engine.py (headless). Fallback to app.py if needed.
mod = None
for name in ("engine", "app"):
    try:
        m = importlib.import_module(name)
        if hasattr(m, "run_once"):
            mod = m
            break
    except Exception:
        pass
if mod is None:
    raise SystemExit("Could not import engine.run_once or app.run_once")

def _env_bool(k, default=False):
    v = str(os.getenv(k, default)).strip().lower()
    return v in ("1", "true", "t", "yes", "y", "on")

def _env_float(k, default):
    try: return float(os.getenv(k, default))
    except Exception: return float(default)

def _env_int(k, default):
    try: return int(os.getenv(k, default))
    except Exception: return int(default)

def main():
    # Superset of knobs — we’ll filter to what run_once() actually takes.
    superset = dict(
        live_trading=_env_bool("LIVE_TRADING", False),
        login_ok=True,  # harmless if present; ignored if not in signature
        universe_src=os.getenv("UNIVERSE_SRC", "S&P 500 (auto)"),
        raw_tickers=os.getenv("RAW_TICKERS", "AAPL,MSFT,GOOG"),
        include_crypto=_env_bool("INCLUDE_CRYPTO", True),
        alloc_mode=os.getenv("ALLOC_MODE", "Fixed $ per trade"),  # or "Proportional across winners"
        fixed_per_trade=_env_float("FIXED_PER_TRADE", 5.0),
        prop_total_budget=_env_float("PROP_TOTAL_BUDGET", 15.0),
        min_per_order=_env_float("MIN_PER_ORDER", 2.0),
        n_picks=_env_int("N_PICKS", 3),
        use_crypto_limits=_env_bool("USE_CRYPTO_LIMITS", True),
        crypto_limit_bps=_env_int("CRYPTO_LIMIT_BPS", 20),
        use_stock_limits=_env_bool("USE_STOCK_LIMITS", False),
        stock_limit_bps=_env_int("STOCK_LIMIT_BPS", 25),
        auto_bp=_env_bool("AUTO_BP", False),
        bp_pct=_env_int("BP_PCT", 30),
        max_buy_orders=_env_int("MAX_BUY_ORDERS", 12),
        max_buy_notional=_env_float("MAX_BUY_NOTIONAL", 50.0),
    )

    sig = inspect.signature(mod.run_once)
    kwargs = {k: v for k, v in superset.items() if k in sig.parameters}

    print("Running headless with config:", {k: kwargs[k] for k in sorted(kwargs)})
    mod.run_once(**kwargs)
    print("Done.")

if __name__ == "__main__":
    main()
