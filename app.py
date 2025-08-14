# app.py — Autonomous Stock & Crypto Momentum Rebalancer (Streamlit)
# -------------------------------------------------------------------
# New in this version (toward full automation):
#   • Auto universes:
#       - Equities: "S&P 500 (auto)" or manual list (Russell1000 could be added later)
#       - Crypto:  Robinhood‑tradable list (fallback to a default set)
#   • Multi‑horizon momentum score (1D, 5D, 20D) → rank by Score
#   • SELL rule (Option A): full exit if (out of top‑N) OR (score < 0)
#   • Pooled budget across stocks + crypto; two allocation modes (fixed / proportional)
#   • Clean simulation vs live order placement for both stocks & crypto
#   • Logs show both SELLs and BUYs in execution order
#   • v0.8.6: "Full‑Auto (run on load)" option; optional proportional budget sourced from
#            available Robinhood buying power; factored run_once() so the same logic can be
#            called from button, auto‑run, or a future headless scheduler; minor robustness.
#
# Notes
#   • Live trading only if sidebar toggle is ON *and* Robinhood login succeeds.
#   • Secrets supported (any one pair):
#         ROBINHOOD_USERNAME / ROBINHOOD_PASSWORD
#         RH_USERNAME       / RH_PASSWORD
#   • Streamlit Cloud cannot run on a schedule by itself. For automation, run this
#     module headlessly from a scheduler (e.g., GitHub Actions) that calls a small
#     runner script invoking the same functions. We ship the app with clear, pure
#     functions so this is straightforward when you’re ready.

VERSION = "0.8.6b (2025-08-13)"

import inspect
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Robinhood (only when live trading is enabled)
try:
    import robin_stocks.robinhood as rh
    from robin_stocks.robinhood import orders as rh_orders
    from robin_stocks.robinhood import crypto as rh_crypto
    from robin_stocks.robinhood import account as rh_account
    from robin_stocks.robinhood import stocks as rh_stocks
except Exception:
    rh = None
    rh_orders = None
    rh_crypto = None
    rh_account = None
    rh_stocks = None

# ---------------------------------
# UI — Sidebar Controls
# ---------------------------------
st.set_page_config(page_title=f"Stock & Crypto Momentum Rebalancer · {VERSION}", layout="wide")

st.sidebar.header("Authentication & Mode")
st.sidebar.caption(f"Version {VERSION}")
live_trading = st.sidebar.checkbox("Enable Live Trading (use with caution)", value=False)

# Read secrets (support both naming conventions)
user = pwd = None
try:
    user = st.secrets.get("ROBINHOOD_USERNAME") or st.secrets.get("RH_USERNAME")
    pwd  = st.secrets.get("ROBINHOOD_PASSWORD") or st.secrets.get("RH_PASSWORD")
except Exception:
    pass

login_ok = False
login_msg = "Simulation mode — no live orders placed."

if live_trading:
    if rh is None:
        st.sidebar.error("robin_stocks is not available in this environment. Running in SIMULATION.")
        live_trading = False
    elif not user or not pwd:
        st.sidebar.error("Live trading selected but credentials are missing in Secrets. Running in SIMULATION.")
        live_trading = False
    else:
        try:
            sig = inspect.signature(rh.authentication.login)
            kwargs = {}
            if "username" in sig.parameters: kwargs["username"] = user
            if "password" in sig.parameters: kwargs["password"] = pwd
            if "store_session" in sig.parameters: kwargs["store_session"] = True
            if "expiresIn" in sig.parameters: kwargs["expiresIn"] = 24 * 3600
            if "scope" in sig.parameters: kwargs["scope"] = "internal"
            login_ok = bool(rh.authentication.login(**kwargs))
        except TypeError:
            try:
                login_ok = bool(rh.authentication.login(username=user, password=pwd))
            except Exception as e:
                login_ok = False
                login_msg = f"Robinhood login failed: {e}. Running in SIMULATION."
        except Exception as e:
            login_ok = False
            login_msg = f"Robinhood login failed: {e}. Running in SIMULATION."

        if login_ok:
            st.sidebar.success("Robinhood authenticated — Live orders ENABLED")
            login_msg = "Live orders ENABLED"
        else:
            st.sidebar.warning(login_msg)
            live_trading = False
else:
    st.sidebar.info(login_msg)

st.sidebar.header("Universe & Allocation")

universe_src = st.sidebar.selectbox(
    "Equity Universe",
    ["S&P 500 (auto)", "Manual list"],
    index=0,
    help="Choose the equity symbols source. Manual list uses the box below.",
)
raw_tickers = st.sidebar.text_area(
    "Manual equity tickers (comma‑separated)",
    value="AAPL,MSFT,GOOG",
)
include_crypto = st.sidebar.checkbox("Include Crypto (Robinhood‑tradable)", value=True)

alloc_mode = st.sidebar.selectbox("Allocation mode", ["Fixed $ per trade", "Proportional across winners"], index=0)
fixed_per_trade = st.sidebar.number_input("Fixed $ per BUY/SELL", min_value=1.0, value=5.0, step=0.5)
prop_total_budget = st.sidebar.number_input("Total BUY budget (proportional)", min_value=1.0, value=15.0, step=1.0)
min_per_order = st.sidebar.number_input("Minimum $ per order", min_value=1.0, value=2.0, step=0.5)
UI_MIN_PER_ORDER = float(min_per_order)

n_picks = st.sidebar.number_input("Top N to hold", min_value=1, value=3, step=1)

st.sidebar.divider()
use_crypto_limits = st.sidebar.checkbox("Use LIMIT orders for crypto", value=True,
                                        help="Attempts maker‑style buys/sells using a small price buffer; falls back to market if unsupported.")
crypto_limit_bps = st.sidebar.slider("Crypto limit price buffer (bps)", min_value=5, max_value=100, value=20, step=5)
use_stock_limits = st.sidebar.checkbox("Use LIMIT orders for stocks (experimental)", value=False,
                                       help="Best‑effort: computes qty from $ and quotes; falls back to fractional market if unsupported.")
stock_limit_bps = st.sidebar.slider("Stock limit price buffer (bps)", min_value=5, max_value=150, value=25, step=5)

st.sidebar.divider()
full_auto = st.sidebar.checkbox("Full‑Auto (run on load)", value=False,
                               help="If ON and logged in, the scan+rebalance runs immediately when the page loads.")
auto_bp = st.sidebar.checkbox("Auto budget from buying power (for proportional mode)", value=False,
                             help="If ON, the proportional BUY budget is set to a % of available BP.")
bp_pct = st.sidebar.slider("Budget % of Buying Power", min_value=5, max_value=100, value=30, step=5)

st.title("Stock & Crypto Momentum Rebalancer")
st.caption(f"Version {VERSION}")

# ---------------------------------
# Universe helpers
# ---------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_sp500_symbols() -> List[str]:
    """Fetch S&P 500 tickers with multiple fallbacks that do **not** require lxml.
    1) Try Wikipedia via read_html (if lxml/html5lib available).
    2) Try a known CSV mirror on GitHub (raw CSV works without lxml).
    3) Fallback to a compact baked list so the app remains usable offline.
    """
    # 1) Wikipedia (best effort)
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if isinstance(tables, list) and len(tables):
            df = tables[0]
            syms = (
                df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
            )
            syms = [s for s in dict.fromkeys(syms) if s.isascii() and 1 <= len(s) <= 6]
            if syms:
                return syms
    except Exception:
        pass

    # 2) GitHub raw CSV mirrors (no lxml dependency). Try a couple of well-known mirrors.
    csv_urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/constituents.csv",
    ]
    for url in csv_urls:
        try:
            df = pd.read_csv(url)
            col = None
            for c in ["Symbol", "Ticker", "symbol", "ticker"]:
                if c in df.columns:
                    col = c
                    break
            if col is not None:
                syms = df[col].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
                syms = [s for s in dict.fromkeys(syms) if s.isascii() and 1 <= len(s) <= 6]
                if syms:
                    return syms
        except Exception:
            continue

    # 3) Small baked fallback set — the app still works while offline
    return ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "BRK-B", "TSLA", "JPM"]


def clean_manual_list(raw: str) -> List[str]:
    return [t.strip().upper() for t in (raw or "").split(",") if t.strip()]


DEFAULT_CRYPTO = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "USDT-USD"]


def load_rh_crypto_pairs() -> List[str]:
    """Return symbols in Yahoo-style (e.g., ETH-USD). Fallback to DEFAULT_CRYPTO."""
    try:
        pairs = []
        # Some versions don’t need login for this; if it fails we fallback
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_currency_pairs"):
            data = rh_crypto.get_crypto_currency_pairs()
            if isinstance(data, list):
                for p in data:
                    try:
                        sym = (p.get("asset_currency", {}) or {}).get("code")
                        if sym:
                            pairs.append(f"{sym}-USD")
                    except Exception:
                        pass
        if not pairs:
            pairs = DEFAULT_CRYPTO.copy()
        return sorted(list(dict.fromkeys(pairs)))
    except Exception:
        return DEFAULT_CRYPTO.copy()

# ---------------------------------
# Data — returns & scoring
# ---------------------------------

def fetch_returns(symbols: List[str], lookbacks: List[int]) -> pd.DataFrame:
    """Fetch % returns over given lookbacks in trading days using yfinance.
    Returns columns: [Ticker] + [f"R{lb}"] for each lookback.
    """
    if not symbols:
        cols = ["Ticker"] + [f"R{lb}" for lb in lookbacks]
        return pd.DataFrame(columns=cols)

    # yfinance multi‑download; fall back one‑by‑one as needed
    def _calc_from_df(sym: str, dfx: pd.DataFrame) -> Dict[str, float]:
        out = {"Ticker": sym}
        if dfx is None or dfx.empty or "Close" not in dfx:
            for lb in lookbacks:
                out[f"R{lb}"] = np.nan
            return out
        close = dfx["Close"].dropna()
        for lb in lookbacks:
            try:
                if len(close) >= (lb + 1):
                    r = (close.iloc[-1] / close.iloc[-(lb + 1)] - 1.0) * 100.0
                else:
                    r = np.nan
            except Exception:
                r = np.nan
            out[f"R{lb}"] = float(r) if pd.notna(r) else np.nan
        return out

    rows: List[Dict] = []
    try:
        df = yf.download(symbols, period="60d", interval="1d", progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            for sym in (df["Close"].columns if "Close" in df else symbols):
                dfx = df.xs(sym, axis=1, level=1)[["Close"]] if "Close" in df else None
                rows.append(_calc_from_df(str(sym), dfx))
        else:
            # Single symbol path
            rows.append(_calc_from_df(str(symbols[0]), df))
    except Exception:
        for s in symbols:
            try:
                dfx = yf.download(s, period="60d", interval="1d", progress=False, threads=False)
            except Exception:
                dfx = None
            rows.append(_calc_from_df(s, dfx))

    return pd.DataFrame(rows)


def score_momentum(df: pd.DataFrame, weights: Dict[int, float] = None) -> pd.DataFrame:
    if weights is None:
        # Heavier weight on near‑term, small tail on 20D
        weights = {1: 0.6, 5: 0.3, 20: 0.1}

    # Z‑score each horizon to balance scales
    sc = df.copy()
    for lb, w in weights.items():
        col = f"R{lb}"
        if col not in sc:
            sc[col] = np.nan
        mu = sc[col].mean(skipna=True)
        sd = sc[col].std(skipna=True)
        sc[f"Z{lb}"] = 0.0 if pd.isna(sd) or sd == 0 else (sc[col] - mu) / sd
    sc["Score"] = sum(sc.get(f"Z{lb}", 0.0) * w for lb, w in weights.items())
    return sc


def build_universe(eq_src: str, manual_raw: str, include_c: bool) -> List[str]:
    eq = load_sp500_symbols() if eq_src == "S&P 500 (auto)" else clean_manual_list(manual_raw)
    cr = load_rh_crypto_pairs() if include_c else []
    # Combine & deduplicate
    uni = list(dict.fromkeys(eq + cr))
    return uni


# ---------------------------------
# Order helpers
# ---------------------------------

def yahoo_to_rh_stock(sym: str) -> str:
    """Convert Yahoo-style tickers (e.g., BRK-B) to Robinhood style (BRK.B)."""
    s = str(sym or "").upper().strip()
    # Avoid touching crypto pairs like BTC-USD
    if s.endswith("-USD"):
        return s
    return s.replace("-", ".")


def rh_to_yahoo_stock(sym: str) -> str:
    """Convert Robinhood stock symbols (BRK.B) to Yahoo style (BRK-B)."""
    s = str(sym or "").upper().strip()
    if "." in s:
        return s.replace(".", "-")
    return s


def _call_first_available(objs, names, *args, **kwargs):
    for obj in objs:
        if obj is None: continue
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
    raise AttributeError(f"No available function among {names}")


def symbol_to_rh_crypto(sym: str) -> str:
    return sym.split("-")[0] if "-" in sym else sym


def normalize_dollars(dollars: float, min_amt: float) -> float:
    d = max(float(dollars), float(min_amt))
    # round *up* to the nearest cent to avoid falling under exchange/RH ticks
    cents = int(np.ceil(d * 100.0 - 1e-9))
    return cents / 100.0


def _get_stock_last_price(sym: str) -> float:
    # Try Robinhood fast quote first, fallback to yfinance fast_info
    try:
        if rh_stocks is not None and hasattr(rh_stocks, "get_latest_price"):
            px = rh_stocks.get_latest_price(yahoo_to_rh_stock(sym), includeExtendedHours=True)
            if isinstance(px, list) and px:
                return float(px[0])
    except Exception:
        pass
    try:
        t = yf.Ticker(sym)
        if hasattr(t, "fast_info") and getattr(t, "fast_info") is not None:
            fi = t.fast_info
            p = fi.get("last_price") or fi.get("last_close") or fi.get("last_price")
            if p:
                return float(p)
    except Exception:
        pass
    return 0.0


def place_stock_order(symbol: str, side: str, dollars: float, live: bool, min_order: float,
                      use_limit: bool = False, limit_bps: int = 25) -> Tuple[str, float, str]:
    """Place a stock order.
    Default uses Robinhood fractional market-by-dollars. If use_limit=True, best‑effort limit using qty.
    Returns: (status, executed_amount, order_id)
    """
    base = normalize_dollars(dollars, min_order)
    if base < 0.01:
        return ("skipped (too small)", 0.0, "")
    if not live:
        return ("simulated", 0.0, "")

    rh_symbol = yahoo_to_rh_stock(symbol)

    # Experimental LIMIT path — compute quantity from notional
    if use_limit and rh_orders is not None:
        try:
            px = _get_stock_last_price(symbol)
            if px and px > 0:
                qty = max(base / px, 0.0001)
                # Robinhood typically allows up to 6 decimals for fractional qty
                qty = float(np.round(qty, 6))
                buff = float(limit_bps) / 10000.0
                if side.upper() == "BUY":
                    limit_price = float(np.round(px * (1 + buff), 4))
                    fnames = [
                        "order_buy_fractional_limit",  # if available
                        "order_buy_limit",
                    ]
                    res = _call_first_available([rh_orders], fnames, rh_symbol, qty, limit_price)
                else:
                    limit_price = float(np.round(px * (1 - buff), 4))
                    fnames = [
                        "order_sell_fractional_limit",
                        "order_sell_limit",
                    ]
                    res = _call_first_available([rh_orders], fnames, rh_symbol, qty, limit_price)
                oid = ""
                if isinstance(res, dict):
                    oid = res.get("id", "") or res.get("order_id", "")
                return ("placed", 0.0, oid)
        except Exception:
            # fall through to market-by-dollars
            pass

    # Market‑by‑dollars with retry (most reliable for fractionals)
    for attempt in range(3):
        amt = normalize_dollars(base * (1.02 ** attempt), min_order)
        try:
            if side.upper() == "BUY":
                res = rh_orders.order_buy_fractional_by_price(rh_symbol, amt)
            else:
                res = rh_orders.order_sell_fractional_by_price(rh_symbol, amt)
            oid = ""
            if isinstance(res, dict):
                oid = res.get("id", "") or res.get("order_id", "")
            return ("placed", 0.0, oid)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["too small", "minimum", "tick", "notional"]):
                continue  # bump and retry
            return (f"error: {e}", 0.0, "")
    return ("error: minimum notional too small after retries", 0.0, "")



def _get_crypto_last_price(sym_no_usd: str) -> float:
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_quote"):
            q = rh_crypto.get_crypto_quote(sym_no_usd)
            # prefer mid if available
            ask = float(q.get("ask_price", 0) or 0)
            bid = float(q.get("bid_price", 0) or 0)
            if ask and bid:
                return (ask + bid) / 2.0
            return float(q.get("mark_price", 0) or ask or bid or 0)
    except Exception:
        pass
    try:
        t = yf.Ticker(f"{sym_no_usd}-USD")
        fi = getattr(t, "fast_info", None) or {}
        p = fi.get("last_price") or fi.get("last_close")
        if p:
            return float(p)
    except Exception:
        pass
    return 0.0


def place_crypto_order(symbol: str, side: str, dollars: float, live: bool, min_order: float,
                       use_limit: bool = True, limit_bps: int = 20) -> Tuple[str, float, str]:
    """Place a crypto order by dollars with retry; if use_limit, compute qty & use limit endpoints when available."""
    base = normalize_dollars(dollars, min_order)
    if base < 0.01:
        return ("skipped (too small)", 0.0, "")
    if not live:
        return ("simulated", 0.0, "")

    sym = symbol_to_rh_crypto(symbol)

    if use_limit and rh_orders is not None:
        try:
            px = _get_crypto_last_price(sym)
            if px and px > 0:
                qty = max(base / px, 0.00000001)
                qty = float(np.round(qty, 8))
                buff = float(limit_bps) / 10000.0
                if side.upper() == "BUY":
                    limit_price = float(np.round(px * (1 + buff), 2))
                    fnames = ["order_buy_crypto_limit"]
                    res = _call_first_available([rh_orders], fnames, sym, qty, limit_price)
                else:
                    limit_price = float(np.round(px * (1 - buff), 2))
                    fnames = ["order_sell_crypto_limit"]
                    res = _call_first_available([rh_orders], fnames, sym, qty, limit_price)
                oid = ""
                if isinstance(res, dict):
                    oid = res.get("id", "") or res.get("order_id", "")
                return ("placed", 0.0, oid)
        except Exception:
            # fall back to market-by-dollars
            pass

    buy_candidates  = ["order_buy_crypto_by_price", "order_buy_crypto_by_dollar", "order_buy_crypto_by_dollars", "buy_crypto_by_price"]
    sell_candidates = ["order_sell_crypto_by_price", "order_sell_crypto_by_dollar", "order_sell_crypto_by_dollars", "sell_crypto_by_price"]

    for attempt in range(3):
        amt = normalize_dollars(base * (1.02 ** attempt), min_order)
        try:
            fnames = buy_candidates if side.upper() == "BUY" else sell_candidates
            res = _call_first_available([rh_crypto, rh], fnames, sym, amt)
            oid = ""
            if isinstance(res, dict):
                oid = res.get("id", "") or res.get("order_id", "")
            return ("placed", 0.0, oid)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["too small", "minimum", "tick", "notional"]):
                continue
            return (f"error: {e}", 0.0, "")
    return ("error: minimum notional too small after retries", 0.0, "")


# ---------------------------------
# Holdings / BUYING POWER / SELL logic
# ---------------------------------

def get_holdings() -> Dict[str, Dict]:
    """Return current holdings keyed by ticker, with approx market value and qty.
       Stocks via account.build_holdings(); Crypto via get_crypto_positions().
       Stock symbols are normalized to Yahoo style (BRK-B) so they compare to picks correctly.
    """
    out: Dict[str, Dict] = {}
    if rh_account is None:
        return out

    # Stocks
    try:
        if hasattr(rh_account, "build_holdings"):
            h = rh_account.build_holdings()
            if isinstance(h, dict):
                for sym, info in h.items():
                    try:
                        value = float(info.get("equity", 0) or 0)
                    except Exception:
                        value = 0.0
                    try:
                        qty = float(info.get("quantity", 0) or 0)
                    except Exception:
                        qty = 0.0
                    norm = rh_to_yahoo_stock(str(sym))
                    out[norm] = {"type": "stock", "value": value, "qty": qty}
    except Exception:
        pass

    # Crypto
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_positions"):
            pos = rh_crypto.get_crypto_positions()
            if isinstance(pos, list):
                for p in pos:
                    try:
                        qty = float(p.get("quantity", 0) or 0)
                        sym = (p.get("currency", {}) or {}).get("code") or ""
                        if qty > 0 and sym:
                            price = 0.0
                            try:
                                if hasattr(rh_crypto, "get_crypto_quote"):
                                    q = rh_crypto.get_crypto_quote(sym)
                                    price = float(q.get("mark_price", 0) or q.get("ask_price", 0) or 0)
                            except Exception:
                                price = 0.0
                            value = qty * price
                            out[f"{sym}-USD".upper()] = {"type": "crypto", "value": float(value), "qty": qty}
                    except Exception:
                        pass
    except Exception:
        pass

    return out


def get_buying_power() -> float:
    """Best‑effort pull of available buying power (stocks and/or crypto)."""
    bp = 0.0
    # Equity account
    try:
        if rh_account is not None:
            prof = None
            for nm in ["load_account_profile", "build_user_profile", "load_phoenix_account"]:
                fn = getattr(rh_account, nm, None)
                if callable(fn):
                    try:
                        prof = fn()
                        if prof:
                            break
                    except Exception:
                        continue
            if isinstance(prof, dict):
                for key in ["buying_power", "cash", "portfolio_cash", "cash_available_for_withdrawal"]:
                    try:
                        val = prof.get(key)
                        if val is not None and float(val) > 0:
                            bp = max(bp, float(val))
                    except Exception:
                        pass
                # nested dicts
                for v in prof.values():
                    if isinstance(v, dict):
                        for key in ["buying_power", "cash", "portfolio_cash", "cash_available_for_withdrawal"]:
                            try:
                                if key in v and float(v[key]) > 0:
                                    bp = max(bp, float(v[key]))
                            except Exception:
                                pass
    except Exception:
        pass

    # Crypto account
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_account_info"):
            cp = rh_crypto.get_crypto_account_info()
            if isinstance(cp, dict):
                for key in ["cash", "buying_power", "available_cash"]:
                    try:
                        val = cp.get(key)
                        if val is not None and float(val) > 0:
                            bp = max(bp, float(val))
                    except Exception:
                        pass
    except Exception:
        pass

    return float(max(bp, 0.0))


# ---------------------------------
# Core run logic (callable from button or auto)
# ---------------------------------

def run_once(*,
             live_trading: bool,
             login_ok: bool,
             universe_src: str,
             raw_tickers: str,
             include_crypto: bool,
             alloc_mode: str,
             fixed_per_trade: float,
             prop_total_budget: float,
             min_per_order: float,
             n_picks: int,
             use_crypto_limits: bool,
             crypto_limit_bps: int,
             use_stock_limits: bool,
             stock_limit_bps: int,
             auto_bp: bool,
             bp_pct: int,
             ) -> None:

    # Build universes
    universe = build_universe(universe_src, raw_tickers, include_crypto)

    # Compute returns & score
    lookbacks = [1, 5, 20]
    ret_df = fetch_returns(universe, lookbacks)
    scored = score_momentum(ret_df, weights={1: 0.6, 5: 0.3, 20: 0.1})

    # Rank by Score and pick top‑N
    picks = (
        scored.dropna(subset=["Score"]).sort_values("Score", ascending=False).head(int(n_picks)).reset_index(drop=True)
    )

    if picks.empty:
        st.warning("No momentum data available (universe may be empty or network hiccup). Try again.")
        return

    # SELL rule (Option A): full exit if out of top‑N OR score < 0
    sell_rows: List[Dict] = []
    buy_rows:  List[Dict] = []

    current = get_holdings() if (live_trading and login_ok) else {}
    in_top = set(picks["Ticker"].astype(str).str.upper().tolist())

    # Decide BUY allocations
    if alloc_mode == "Fixed $ per trade":
        per_trade = float(max(min_per_order, fixed_per_trade))
        buy_allocs = {t: per_trade for t in in_top}
    else:
        # Optional: scale budget to a % of available Buying Power
        budget = float(prop_total_budget)
        if auto_bp and live_trading and login_ok:
            bp = get_buying_power()
            if bp and bp > 0:
                budget = max(min_per_order, round((bp * float(bp_pct) / 100.0), 2))
                st.sidebar.caption(f"Auto budget: ${budget:,.2f} from BP ≈ ${bp:,.2f}")
        w = np.clip(picks["R1"].fillna(0).astype(float), 0, None)
        wsum = float(w.sum())
        if wsum <= 0:
            each = max(min_per_order, budget / max(1, len(picks)))
            buy_allocs = {t: each for t in in_top}
        else:
            buy_allocs = {}
            for t, wi in zip(picks["Ticker"].tolist(), w.tolist()):
                dollars = (wi / wsum) * float(budget)
                buy_allocs[str(t).upper()] = float(max(min_per_order, dollars))

    # --- SELL first ---
    # Build a lookup of scores for *all* symbols we scored
    try:
        score_map = {str(t).upper(): (float(s) if pd.notna(s) else 0.0)
                     for t, s in zip(scored["Ticker"].astype(str), scored["Score"])}
    except Exception:
        score_map = {}

    for sym, info in current.items():
        t_upper = str(sym).upper()
        sym_score = float(score_map.get(t_upper, 0.0))
        out_of_top = t_upper not in in_top
        neg_score  = sym_score < 0
        if out_of_top or neg_score:
            # Full exit if (out of top‑N) OR (score < 0)
            dollars = float(info.get("value", 0.0))
            if dollars >= min_per_order:
                if info.get("type") == "crypto":
                    status, executed, oid = place_crypto_order(symbol_to_rh_crypto(t_upper), "SELL", dollars, live_trading and login_ok, min_per_order,
                                                                use_limit=use_crypto_limits, limit_bps=crypto_limit_bps)
                else:
                    status, executed, oid = place_stock_order(t_upper, "SELL", dollars, live_trading and login_ok, min_per_order,
                                                               use_limit=use_stock_limits, limit_bps=stock_limit_bps)
                sell_rows.append({
                    "Ticker": t_upper,
                    "Action": "SELL",
                    "Reason": "negative_score" if neg_score and not out_of_top else ("out_of_topN" if out_of_top else "negative_score"),
                    "Alloc$": round(dollars, 2),
                    "OrderID": oid,
                    "Status": status,
                    "Time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                })

    # --- BUY winners ---
    for _, r in picks.iterrows():
        t = str(r["Ticker"]).upper()
        score = float(r["Score"]) if pd.notna(r["Score"]) else 0.0
        if score < 0:
            continue  # do not buy negative score
        dollars = float(buy_allocs.get(t, 0.0))
        if dollars < min_per_order:
            continue
        is_crypto = t.endswith("-USD")
        if is_crypto:
            status, executed, oid = place_crypto_order(symbol_to_rh_crypto(t), "BUY", dollars, live_trading and login_ok, min_per_order,
                                                       use_limit=use_crypto_limits, limit_bps=crypto_limit_bps)
        else:
            status, executed, oid = place_stock_order(t, "BUY", dollars, live_trading and login_ok, min_per_order,
                                                      use_limit=use_stock_limits, limit_bps=stock_limit_bps)
        buy_rows.append({
            "Ticker": t,
            "Action": "BUY",
            "Alloc$": round(dollars, 2),
            "OrderID": oid,
            "Status": status,
            "R1%": round(float(r.get("R1", 0.0)), 3),
            "R5%": round(float(r.get("R5", 0.0)), 3),
            "R20%": round(float(r.get("R20", 0.0)), 3),
            "Score": round(score, 4),
            "Time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Display sections
    st.subheader("Today's Top‑N (ranked by momentum score)")
    st.dataframe(picks[["Ticker", "R1", "R5", "R20", "Score"]].rename(columns={"R1": "R1%", "R5": "R5%", "R20": "R20%"}), use_container_width=True)

    if sell_rows:
        st.subheader("Sell Log")
        st.dataframe(pd.DataFrame(sell_rows), use_container_width=True)
    else:
        st.subheader("Sell Log")
        st.write("(no sells)")

    st.subheader("Buy Log")
    buy_df = pd.DataFrame(buy_rows) if buy_rows else pd.DataFrame(columns=["Ticker","Action","Alloc$","OrderID","Status","R1%","R5%","R20%","Score","Time"])    
    st.dataframe(buy_df, use_container_width=True)

    # Combined CSV download
    all_rows = []
    for row in sell_rows: all_rows.append({"Kind": "SELL", **row})
    for row in buy_rows:  all_rows.append({"Kind": "BUY",  **row})
    log_df = pd.DataFrame(all_rows)
    csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Activity CSV", csv, file_name="rebalance_activity.csv", mime="text/csv")

    # Open orders snapshot (best‑effort)
# Moved into a function so it renders **after** the main run logic.
def render_open_orders(live_trading: bool, login_ok: bool) -> None:
    st.subheader("Open Orders")
    open_list = []
    if live_trading and login_ok and rh is not None:
        # --- helpers to recover missing symbols from RH payloads ---
        @st.cache_data(ttl=300, show_spinner=False)
        def _crypto_pair_maps():
            id2sym, sym2id = {}, {}
            try:
                if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_currency_pairs"):
                    pairs = rh_crypto.get_crypto_currency_pairs()
                    if isinstance(pairs, list):
                        for p in pairs:
                            pid = p.get("id") or p.get("uuid") or p.get("currency_pair_id")
                            base = (p.get("asset_currency") or {}).get("code") or p.get("symbol") or ""
                            quote = (p.get("quote_currency") or {}).get("code") or "USD"
                            if base:
                                sym = f"{base}-{quote}"
                                if pid:
                                    id2sym[pid] = sym
                                sym2id[sym] = pid or ""
            except Exception:
                pass
            return id2sym, sym2id

        def _stock_symbol_from_instrument(url: str) -> str:
            """Try several robin_stocks helpers to turn an instrument URL into a Yahoo-style symbol."""
            try:
                if rh_stocks is not None:
                    for nm in ["get_instrument_by_url", "get_instruments_by_url"]:
                        fn = getattr(rh_stocks, nm, None)
                        if callable(fn):
                            inst = fn(url)
                            if isinstance(inst, dict):
                                sym = inst.get("symbol") or ""
                                if sym:
                                    return rh_to_yahoo_stock(sym)
            except Exception:
                pass
            try:
                # Some library versions expose a convenience on the package root
                fn = getattr(rh, "get_symbol_by_url", None)
                if callable(fn):
                    sym = fn(url)
                    if sym:
                        return rh_to_yahoo_stock(sym)
            except Exception:
                pass
            return ""

        id2sym, _ = _crypto_pair_maps()

        try:
            if rh_orders is not None and hasattr(rh_orders, "get_all_open_stock_orders"):
                s_open = rh_orders.get_all_open_stock_orders()
                if isinstance(s_open, list):
                    open_list.extend([{"type": "stock", **(o if isinstance(o, dict) else {"raw": str(o)})} for o in s_open])
        except Exception as e:
            st.info(f"Failed to fetch open stock orders: {e}")
        try:
            if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_open_orders"):
                c_open = rh_crypto.get_crypto_open_orders()
                if isinstance(c_open, list):
                    open_list.extend([{"type": "crypto", **(o if isinstance(o, dict) else {"raw": str(o)})} for o in c_open])
        except Exception as e:
            st.info(f"Failed to fetch open crypto orders: {e}")

    # Compact table (hide raw JSON by default)
    if open_list:
        rows = []
        for o in open_list:
            if not isinstance(o, dict):
                continue
            typ = o.get("type", "")
            side = o.get("side") or o.get("direction") or ""
            # symbol recovery logic
            sym = o.get("symbol") or o.get("chain_symbol") or ""
            if not sym:
                inst = o.get("instrument")
                if inst:
                    sym = _stock_symbol_from_instrument(inst)
            if not sym:
                cp = o.get("currency_pair_id") or o.get("currency_pair") or o.get("pair")
                if cp:
                    # need id2sym from above scope if available; recompute if missing
                    try:
                        id2sym, _ = _crypto_pair_maps()
                    except Exception:
                        id2sym = {}
                    sym = id2sym.get(cp, str(cp))
            # normalize crypto like ETHUSD -> ETH-USD
            if isinstance(sym, str) and sym.endswith("USD") and "-" not in sym:
                sym = sym.replace("USD", "-USD")
            state = o.get("state") or o.get("status") or ""
            created = o.get("created_at") or o.get("last_transaction_at") or o.get("submitted_at") or ""
            oid = o.get("id") or o.get("order_id") or ""
            notional = o.get("notional") or o.get("price") or o.get("average_price") or o.get("quantity") or ""
            rows.append({"Type": typ, "Side": side, "Symbol": sym, "Notional/Qty": notional, "State": state, "Created": created, "OrderID": oid})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.write("[]")
        with st.expander("Show raw API response"):
            st.json(open_list)
    else:
        st.write("[]")
    with st.expander("Show raw API response"):
        st.json(open_list)
else:
    st.write("[]")

# ---------------------------------
# Button / Auto‑run wiring
# ---------------------------------

ran = False
if full_auto:
    if not st.session_state.get("__full_auto_ran__"):
        st.session_state["__full_auto_ran__"] = True
        st.info("Full‑Auto is ON — running now.")
        run_once(
            live_trading=live_trading,
            login_ok=login_ok,
            universe_src=universe_src,
            raw_tickers=raw_tickers,
            include_crypto=include_crypto,
            alloc_mode=alloc_mode,
            fixed_per_trade=float(fixed_per_trade),
            prop_total_budget=float(prop_total_budget),
            min_per_order=float(min_per_order),
            n_picks=int(n_picks),
            use_crypto_limits=bool(use_crypto_limits),
            crypto_limit_bps=int(crypto_limit_bps),
            use_stock_limits=bool(use_stock_limits),
            stock_limit_bps=int(stock_limit_bps),
            auto_bp=bool(auto_bp),
            bp_pct=int(bp_pct),
        )
        ran = True

if st.button("▶ Run Daily Scan & Rebalance", type="primary"):
    run_once(
        live_trading=live_trading,
        login_ok=login_ok,
        universe_src=universe_src,
        raw_tickers=raw_tickers,
        include_crypto=include_crypto,
        alloc_mode=alloc_mode,
        fixed_per_trade=float(fixed_per_trade),
        prop_total_budget=float(prop_total_budget),
        min_per_order=float(min_per_order),
        n_picks=int(n_picks),
        use_crypto_limits=bool(use_crypto_limits),
        crypto_limit_bps=int(crypto_limit_bps),
        use_stock_limits=bool(use_stock_limits),
        stock_limit_bps=int(stock_limit_bps),
        auto_bp=bool(auto_bp),
        bp_pct=int(bp_pct),
    )
    ran = True

# Always render Open Orders at the bottom (after any run_once)
render_open_orders(live_trading, login_ok)

# ---------------------------------
# Footer / Hints
# ---------------------------------
if not ran:
    st.caption(
        "Tip: keep Live Trading OFF until you like the plan. When you turn it on, make sure secrets are set and the sidebar shows ‘Live orders ENABLED’."
    )
