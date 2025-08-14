# runner.py — headless daily run for GitHub Actions (or local)
import os

try:
    import robin_stocks.robinhood as rh
except Exception:
    rh = None

# ---- tiny env helpers ----
def env(k, default=None): return os.getenv(k, default)
def env_bool(k, default=False):
    v = os.getenv(k)
    return (str(v).lower() in {"1","true","yes","on"}) if v is not None else default
def env_int(k, default): 
    try: return int(env(k, default))
    except: return default
def env_float(k, default):
    try: return float(env(k, default))
    except: return default

def robinhood_login_if_needed(live: bool) -> bool:
    if not live or rh is None:
        return False
    user = env("ROBINHOOD_USERNAME") or env("RH_USERNAME")
    pwd  = env("ROBINHOOD_PASSWORD") or env("RH_PASSWORD")
    if not (user and pwd):
        print("No Robinhood creds; running in SIMULATION.")
        return False
    # Optional TOTP
    mfa_code = None
    totp_secret = env("RH_TOTP_SECRET")
    if totp_secret:
        try:
            import pyotp
            mfa_code = pyotp.TOTP(totp_secret).now()
        except Exception:
            mfa_code = None
    try:
        # Different library versions accept different kwargs; try the broad one first
        ok = bool(rh.authentication.login(username=user, password=pwd, mfa_code=mfa_code, store_session=True))
        if not ok:
            # fallback without mfa/store_session
            ok = bool(rh.authentication.login(username=user, password=pwd))
        print("Robinhood login:", "OK" if ok else "FAILED")
        return ok
    except Exception as e:
        print("Robinhood login error:", e)
        return False

if __name__ == "__main__":
    from engine import run_once  # uses the same logic as the Streamlit app

    live = env_bool("LIVE_TRADING", False)
    login_ok = robinhood_login_if_needed(live)

    params = dict(
        live_trading=live,
        login_ok=login_ok,
        universe_src=env("UNIVERSE_SRC", "S&P 500 (auto)"),
        raw_tickers=env("RAW_TICKERS", "AAPL,MSFT,GOOG"),
        include_crypto=env_bool("INCLUDE_CRYPTO", True),
        alloc_mode=env("ALLOC_MODE", "Fixed $ per trade"),
        fixed_per_trade=env_float("FIXED_PER_TRADE", 5.0),
        prop_total_budget=env_float("PROP_TOTAL_BUDGET", 15.0),
        min_per_order=env_float("MIN_PER_ORDER", 2.0),
        n_picks=env_int("N_PICKS", 3),
        use_crypto_limits=env_bool("USE_CRYPTO_LIMITS", True),
        crypto_limit_bps=env_int("CRYPTO_LIMIT_BPS", 20),
        use_stock_limits=env_bool("USE_STOCK_LIMITS", False),
        stock_limit_bps=env_int("STOCK_LIMIT_BPS", 25),
        auto_bp=env_bool("AUTO_BP", False),
        bp_pct=env_int("BP_PCT", 30),
        max_buy_orders=env_int("MAX_BUY_ORDERS", 12),
        max_buy_notional=env_float("MAX_BUY_NOTIONAL", 50.0),
    )

    print("Running with params:", params)
    run_once(**params)
