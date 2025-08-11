# app.py — Stock & Crypto Momentum Rebalancer (Streamlit)
# ------------------------------------------------------
# This version focuses on running cleanly in BOTH simulation and live modes.
# Key fixes:
#   • Robust Robinhood login (no unsupported args; graceful failure => sim mode)
#   • Uses supported robin_stocks order fns:
#       - Stocks: orders.order_buy_fractional_by_price / order_sell_fractional_by_price
#       - Crypto: crypto.order_buy_crypto_by_dollar / order_sell_crypto_by_dollar
#   • No calls to removed/non‑existent functions (e.g. get_all_open_crypto_orders)
#   • Safer momentum calc (handles empty/NaN data before ranking)
#   • Clear trade log with simulated vs live outcome
#
# Notes:
#   • Live trading ONLY happens if the sidebar toggle is ON *and* login succeeds.
#   • Credentials are read from Streamlit secrets. Supported keys (any one pair):
#        ROBINHOOD_USERNAME / ROBINHOOD_PASSWORD
#        RH_USERNAME       / RH_PASSWORD
#   • For testing, keep live trading OFF. You’ll see "simulated" in the log.

import json
import math
import time
import inspect
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# Price data via yfinance (simple and reliable for both stocks & crypto pairs)
import yfinance as yf

# Robinhood (only used if live enabled)
try:
    import robin_stocks.robinhood as rh
    from robin_stocks.robinhood import orders as rh_orders
    from robin_stocks.robinhood import crypto as rh_crypto
    from robin_stocks.robinhood import account as rh_account
except Exception:  # library not present in some environments
    rh = None
    rh_orders = None
    rh_crypto = None
    rh_account = None

# -----------------------------
# UI — Sidebar Controls
# -----------------------------
st.set_page_config(page_title="Stock & Crypto Momentum Rebalancer", layout="wide")

st.sidebar.header("Authentication & Mode")
live_trading = st.sidebar.checkbox("Enable Live Trading (use with caution)", value=False)

# Read secrets (support both naming conventions)
user = None
pwd = None
try:
    user = st.secrets.get("ROBINHOOD_USERNAME") or st.secrets.get("RH_USERNAME")
    pwd = st.secrets.get("ROBINHOOD_PASSWORD") or st.secrets.get("RH_PASSWORD")
except Exception:
    # If secrets are not configured, we'll remain in sim mode
    pass

login_ok = False
login_msg = "Simulation mode — no live orders placed."

if live_trading:
    if rh is None:
        st.sidebar.error("robin_stocks is not available in this environment. Running in SIMULATION.")
        live_trading = False
    elif not user or not pwd:
        st.sidebar.error("Live trading selected but Robinhood credentials are missing. Running in SIMULATION.")
        live_trading = False
    else:
        # Attempt a robust login without deprecated/unknown args
        try:
            # Build kwargs only from supported signature
            sig = inspect.signature(rh.authentication.login)
            kwargs = {}
            if "username" in sig.parameters: kwargs["username"] = user
            if "password" in sig.parameters: kwargs["password"] = pwd
            # Safe defaults
            if "store_session" in sig.parameters: kwargs["store_session"] = True
            if "expiresIn" in sig.parameters: kwargs["expiresIn"] = 24 * 3600
            if "scope" in sig.parameters: kwargs["scope"] = "internal"

            login_ok = bool(rh.authentication.login(**kwargs))
        except TypeError:
            # Fallback: minimal call
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
raw_tickers = st.sidebar.text_area(
    "Equity Tickers (comma-separated)",
    value="AAPL,MSFT,GOOG",
    help="US equity tickers. We'll fetch daily momentum via Yahoo Finance.",
)
include_crypto = st.sidebar.checkbox("Include Crypto", value=True)

alloc_mode = st.sidebar.selectbox(
    "Allocation mode",
    ["Fixed $ per trade", "Proportional across winners"],
    index=0,
)
fixed_per_trade = st.sidebar.number_input("Fixed $ per BUY/SELL", min_value=1.0, value=5.0, step=0.5)
prop_total_budget = st.sidebar.number_input(
    "Total budget for BUYS (when proportional)", min_value=1.0, value=15.0, step=1.0
)
min_per_order = st.sidebar.number_input("Minimum $ per order", min_value=1.0, value=1.0, step=0.5)

n_picks = st.sidebar.number_input("Number of tickers to pick", min_value=1, value=3, step=1)

st.title("Stock & Crypto Momentum Rebalancer")

# -----------------------------
# Data Helpers
# -----------------------------

def _clean_universe(raw: str) -> List[str]:
    return [t.strip().upper() for t in (raw or "").split(",") if t.strip()]


# Default crypto watchlist — can be expanded if desired
DEFAULT_CRYPTO = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "USDT-USD"]


def fetch_pct_change_yf(symbols: List[str]) -> pd.DataFrame:
    """Fetch last close % change vs previous close using yfinance.
    Returns columns: [Ticker, PctChange]
    """
    if not symbols:
        return pd.DataFrame(columns=["Ticker", "PctChange"])  # empty

    # yfinance can handle list download
    try:
        df = yf.download(symbols, period="7d", interval="1d", progress=False, threads=False)
    except Exception:
        # Retry one-by-one fallback
        rows = []
        for s in symbols:
            try:
                dfx = yf.download(s, period="7d", interval="1d", progress=False, threads=False)
                if not dfx.empty and "Close" in dfx:
                    closes = dfx["Close"].dropna()
                    if len(closes) >= 2:
                        pct = (closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0
                        rows.append((s, float(pct)))
            except Exception:
                pass
        return pd.DataFrame(rows, columns=["Ticker", "PctChange"]) if rows else pd.DataFrame(columns=["Ticker", "PctChange"])  

    # If multi-index, handle accordingly
    rows: List[Tuple[str, float]] = []
    if isinstance(df.columns, pd.MultiIndex):
        # df columns like ('Close','AAPL'), ...
        if ("Close" in df.columns.get_level_values(0)):
            close = df["Close"].copy()
            for s in close.columns:
                series = close[s].dropna()
                if len(series) >= 2:
                    pct = (series.iloc[-1] / series.iloc[-2] - 1.0) * 100.0
                    rows.append((str(s), float(pct)))
    else:
        # Single symbol
        if "Close" in df:
            closes = df["Close"].dropna()
            if len(closes) >= 2:
                pct = (closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0
                # symbols may be a single string
                sym = symbols[0] if isinstance(symbols, list) and len(symbols) == 1 else symbols
                rows.append((str(sym if isinstance(sym, str) else sym[0]), float(pct)))

    out = pd.DataFrame(rows, columns=["Ticker", "PctChange"]) if rows else pd.DataFrame(columns=["Ticker", "PctChange"])  
    out["PctChange"] = pd.to_numeric(out["PctChange"], errors="coerce").fillna(0.0)
    return out


def combine_and_rank(equities: List[str], include_crypto_flag: bool, top_n: int) -> pd.DataFrame:
    eq_df = fetch_pct_change_yf(equities)
    pieces = [eq_df]
    if include_crypto_flag:
        pieces.append(fetch_pct_change_yf(DEFAULT_CRYPTO))
    df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["Ticker", "PctChange"])  

    # Robust cleanup
    df = df.dropna(subset=["Ticker"]).copy()
    df["PctChange"] = pd.to_numeric(df["PctChange"], errors="coerce")
    df = df.dropna(subset=["PctChange"])  # ensure numeric

    if df.empty:
        return df

    return df.sort_values("PctChange", ascending=False).head(top_n).reset_index(drop=True)


# -----------------------------
# Trading Helpers
# -----------------------------

def symbol_to_rh_crypto(sym: str) -> str:
    """Convert yfinance crypto pair like 'ETH-USD' to Robinhood symbol 'ETH'."""
    return sym.split("-")[0] if "-" in sym else sym


def place_stock_order(symbol: str, side: str, dollars: float, live: bool) -> Tuple[str, float, str]:
    """Place (or simulate) a stock order. Returns (status, executed, order_id)"""
    dollars = float(max(0.0, dollars))
    if dollars < 0.01:
        return ("skipped (too small)", 0.0, "")

    if not live:
        return ("simulated", 0.0, "")

    try:
        res = None
        if side.upper() == "BUY":
            res = rh_orders.order_buy_fractional_by_price(symbol, dollars)
        else:
            res = rh_orders.order_sell_fractional_by_price(symbol, dollars)
        order_id = ""
        try:
            if isinstance(res, dict):
                order_id = res.get("id", "") or res.get("order_id", "")
        except Exception:
            pass
        return ("placed", 0.0, order_id)
    except Exception as e:
        return (f"error: {e}", 0.0, "")


def place_crypto_order(symbol: str, side: str, dollars: float, live: bool) -> Tuple[str, float, str]:
    dollars = float(max(0.0, dollars))
    if dollars < 0.01:
        return ("skipped (too small)", 0.0, "")

    if not live:
        return ("simulated", 0.0, "")

    try:
        res = None
        if side.upper() == "BUY":
            res = rh_crypto.order_buy_crypto_by_dollar(symbol, dollars)
        else:
            res = rh_crypto.order_sell_crypto_by_dollar(symbol, dollars)
        order_id = ""
        try:
            if isinstance(res, dict):
                order_id = res.get("id", "") or res.get("order_id", "")
        except Exception:
            pass
        return ("placed", 0.0, order_id)
    except Exception as e:
        return (f"error: {e}", 0.0, "")


# -----------------------------
# Main button
# -----------------------------

if st.button("▶ Run Daily Scan & Rebalance", type="primary"):
    equities = _clean_universe(raw_tickers)

    top = combine_and_rank(equities, include_crypto, int(n_picks))

    if top.empty:
        st.warning("No momentum data available (universe may be empty or network hiccup). Try again.")
        st.stop()

    # Decide allocation per trade
    trade_rows: List[Dict] = []

    # Determine which rows are crypto (by '-USD')
    is_crypto_mask = top["Ticker"].str.contains("-USD", case=False, na=False)

    if alloc_mode == "Fixed $ per trade":
        per_trade = float(max(min_per_order, fixed_per_trade))
        allocs = [per_trade] * len(top)
    else:
        # Proportional to momentum (positive only)
        pos = top.copy()
        pos["w"] = pos["PctChange"].clip(lower=0.0)
        wsum = pos["w"].sum()
        if wsum <= 0:
            # Fallback equal-split
            each = max(min_per_order, prop_total_budget / max(1, len(pos)))
            allocs = [each] * len(top)
        else:
            allocs = []
            for w in pos["w"].tolist():
                dollars = (float(w) / float(wsum)) * float(prop_total_budget)
                allocs.append(max(min_per_order, dollars))

    # Build trade plan — BUY winners
    for i, row in top.iterrows():
        ticker = str(row["Ticker"]).upper()
        pct = float(row["PctChange"]) if pd.notna(row["PctChange"]) else 0.0
        dollars = float(allocs[i])

        is_crypto = ticker.endswith("-USD")
        action = "BUY"

        if is_crypto:
            rh_sym = symbol_to_rh_crypto(ticker)
            status, executed, oid = place_crypto_order(rh_sym, action, dollars, live_trading and login_ok)
        else:
            status, executed, oid = place_stock_order(ticker, action, dollars, live_trading and login_ok)

        trade_rows.append({
            "Ticker": ticker,
            "Action": action,
            "PctChange": round(pct, 4),
            "Alloc$": round(dollars, 2),
            "Executed": executed,
            "OrderID": oid,
            "Status": status,
            "Time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })

    log_df = pd.DataFrame(trade_rows, columns=["Ticker", "Action", "PctChange", "Alloc$", "Executed", "OrderID", "Status", "Time"])  

    st.subheader("Rebalance Log")
    st.dataframe(log_df, use_container_width=True)

    # Download logs
    csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Logs CSV", csv, file_name="rebalance_log.csv", mime="text/csv")

    # Open orders snapshot (best-effort)
    st.subheader("Open Orders")
    open_list = []
    if live_trading and login_ok and rh is not None:
        try:
            # Stocks
            if rh_orders is not None and hasattr(rh_orders, "get_all_open_stock_orders"):
                s_open = rh_orders.get_all_open_stock_orders()
                if isinstance(s_open, list):
                    open_list.extend([{"type": "stock", **(o if isinstance(o, dict) else {"raw": str(o)})} for o in s_open])
        except Exception as e:
            st.info(f"Failed to fetch open stock orders: {e}")
        try:
            # Crypto (some library versions don't expose this; skip quietly)
            if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_open_orders"):
                c_open = rh_crypto.get_crypto_open_orders()
                if isinstance(c_open, list):
                    open_list.extend([{"type": "crypto", **(o if isinstance(o, dict) else {"raw": str(o)})} for o in c_open])
        except Exception as e:
            st.info(f"Failed to fetch open crypto orders: {e}")

    if open_list:
        st.json(open_list)
    else:
        st.write("[]")

# -----------------------------
# Footer / Hints
# -----------------------------
st.caption(
    "Tip: keep Live Trading OFF until you like the plan. When you turn it on, make sure secrets are set and login shows ‘Live orders ENABLED’."
)
