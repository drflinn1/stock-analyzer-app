# runner.py — headless daily run for GitHub Actions (or local cron)
# Safe: defaults to SIMULATION unless LIVE_TRADING=true and login succeeds.
#
# Environment variables (all optional; sensible defaults provided):
#   LIVE_TRADING           -> 'true' | 'false' (default: false)
#   UNIVERSE_SRC           -> 'S&P 500 (auto)' | 'Manual list' (default: S&P 500 (auto))
#   RAW_TICKERS            -> e.g. "AAPL,MSFT,GOOG" (used only if UNIVERSE_SRC='Manual list')
#   INCLUDE_CRYPTO         -> 'true' | 'false' (default: true)
#   ALLOC_MODE             -> 'Fixed $ per trade' | 'Proportional across winners' (default: Fixed $ per trade)
#   FIXED_PER_TRADE        -> dollars per trade if ALLOC_MODE=fixed (default: 5)
#   PROP_TOTAL_BUDGET      -> total dollars if ALLOC_MODE=proportional (default: 15)
#   MIN_PER_ORDER          -> minimum dollars per order (default: 2)
#   N_PICKS                -> number of top symbols to hold (default: 3)
#   USE_CRYPTO_LIMITS      -> 'true' | 'false' (default: true)
#   CRYPTO_LIMIT_BPS       -> limit buffer in bps (default: 20)
#   USE_STOCK_LIMITS       -> 'true' | 'false' (default: false)
#   STOCK_LIMIT_BPS        -> limit buffer in bps (default: 25)
#   AUTO_BP                -> 'true' | 'false' (default: false)
#   BP_PCT                 -> percent of buying power if AUTO_BP=true (default: 30)
#   MAX_BUY_ORDERS         -> safety cap on BUY count (default: 12)
#   MAX_BUY_NOTIONAL       -> safety cap on total BUY notional (default: 50)
#
# Robinhood credentials (read any one naming convention):
#   ROBINHOOD_USERNAME / ROBINHOOD_PASSWORD
#   RH_USERNAME        / RH_PASSWORD
# Optional TOTP (time‑based one‑time password) secret for MFA (base32 seed, not 6‑digit code):
#   RH_TOTP_SECRET
# If you provide RH_TOTP_SECRET, ensure 'pyotp' is available (requirements.txt).

from __future__ import annotations
import os
import sys
import json
import time
import inspect
from dataclasses import dataclass

# ----- tiny env helpers ------------------------------------------------------

def env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None and v != "" else default

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def env_int(key: str, default: int) -> int:
    try:
        return int(float(env(key, str(default))))
    except Exception:
        return int(default)

def env_float(key: str, default: float) -> float:
    try:
        return float(env(key, str(default)))
    except Exception:
        return float(default)

# ----- Optional pyotp for TOTP MFA ------------------------------------------
try:
    import pyotp  # type: ignore
except Exception:
    pyotp = None  # okay if you don't use RH_TOTP_SECRET

# robin_stocks — optional; if unavailable we will run in simulation
try:
    import robin_stocks.robinhood as rh  # type: ignore
except Exception:
    rh = None  # type: ignore

# In headless mode we don't want Streamlit to try to spin up a server if app.py is imported.
os.environ.setdefault("STREAMLIT_HEADLESS", "1")

# ----- Robinhood login (best effort, many versions) -------------------------

def robinhood_login(username: str | None, password: str | None, totp_secret: str | None) -> bool:
    if rh is None or not username or not password:
        return False
    # Build kwargs carefully based on function signature
    try:
        login_fn = rh.authentication.login  # type: ignore[attr-defined]
    except Exception:
        return False

    kwargs = {}
    try:
        sig = inspect.signature(login_fn)
        p = sig.parameters
        if "username" in p:
            kwargs["username"] = username
        if "password" in p:
            kwargs["password"] = password
        # Persist session when possible
        if "store_session" in p:
            kwargs["store_session"] = True
        if "scope" in p:
            kwargs["scope"] = "internal"
        if "expiresIn" in p:
            kwargs["expiresIn"] = 24 * 3600
        # MFA code if a TOTP seed is provided and a param exists
        code = None
        if totp_secret and pyotp is not None:
            try:
                code = pyotp.TOTP(totp_secret).now()
            except Exception:
                code = None
        for mfa_param in ("mfa_code", "verification_code", "challenge_type_code"):
            if code and mfa_param in p:
                kwargs[mfa_param] = code
                break
        ok = bool(login_fn(**kwargs))
        return ok
    except Exception:
        # Fallback minimal call
        try:
            ok = bool(rh.authentication.login(username=username, password=password))
            return ok
        except Exception:
            return False

# ----- Import engine or fall back to app ------------------------------------

RUN_SOURCE = "engine"
try:
    import engine as eng  # your pure logic module if present
    RUN_SOURCE = "engine"
except Exception:
    import app as eng      # fall back to app.py (uses Streamlit but will run headless)
    RUN_SOURCE = "app"

# ----- Config dataclass (so we can print/debug) -----------------------------

@dataclass
class Config:
    live_trading: bool
    universe_src: str
    raw_tickers: str
    include_crypto: bool
    alloc_mode: str
    fixed_per_trade: float
    prop_total_budget: float
    min_per_order: float
    n_picks: int
    use_crypto_limits: bool
    crypto_limit_bps: int
    use_stock_limits: bool
    stock_limit_bps: int
    auto_bp: bool
    bp_pct: int
    max_buy_orders: int
    max_buy_notional: float


def load_config_from_env() -> Config:
    return Config(
        live_trading=env_bool("LIVE_TRADING", False),
        universe_src=env("UNIVERSE_SRC", "S&P 500 (auto)"),
        raw_tickers=env("RAW_TICKERS", "AAPL,MSFT,GOOG") or "",
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


def main() -> int:
    cfg = load_config_from_env()

    # Credentials
    user = env("ROBINHOOD_USERNAME", env("RH_USERNAME"))
    pwd  = env("ROBINHOOD_PASSWORD", env("RH_PASSWORD"))
    totp_secret = env("RH_TOTP_SECRET")

    # Login only if live requested
    login_ok = False
    if cfg.live_trading:
        login_ok = robinhood_login(user, pwd, totp_secret)
        if not login_ok:
            # Safety: if login fails, force simulation
            cfg.live_trading = False

    # Build kwargs for engine/app run_once()
    kwargs = dict(
        live_trading=bool(cfg.live_trading),
        login_ok=bool(login_ok),
        universe_src=cfg.universe_src,
        raw_tickers=cfg.raw_tickers,
        include_crypto=bool(cfg.include_crypto),
        alloc_mode=cfg.alloc_mode,
        fixed_per_trade=float(cfg.fixed_per_trade),
        prop_total_budget=float(cfg.prop_total_budget),
        min_per_order=float(cfg.min_per_order),
        n_picks=int(cfg.n_picks),
        use_crypto_limits=bool(cfg.use_crypto_limits),
        crypto_limit_bps=int(cfg.crypto_limit_bps),
        use_stock_limits=bool(cfg.use_stock_limits),
        stock_limit_bps=int(cfg.stock_limit_bps),
        auto_bp=bool(cfg.auto_bp),
        bp_pct=int(cfg.bp_pct),
        max_buy_orders=int(cfg.max_buy_orders),
        max_buy_notional=float(cfg.max_buy_notional),
    )

    # Run once
    try:
        eng.run_once(**kwargs)  # type: ignore[attr-defined]
        print(json.dumps({
            "ok": True,
            "source": RUN_SOURCE,
            "live_trading": kwargs["live_trading"],
            "login_ok": kwargs["login_ok"],
        }))
        return 0
    except TypeError as te:
        # Helpful message if signature drifted
        print("runner: TypeError calling run_once — check expected kwargs.\n", file=sys.stderr)
        print(str(te), file=sys.stderr)
        return 2
    except Exception as e:
        print("runner: unexpected error:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
