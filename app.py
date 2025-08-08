# app.py — Streamlit Momentum Rebalancer with Robinhood (stocks + crypto)

import os
from datetime import datetime, timedelta
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Robinhood SDK
try:
    from robin_stocks import robinhood as rhood
except Exception as _e:  # allow the app to render without the package so users see a helpful message
    rhood = None

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Stock & Crypto Momentum Rebalancer", layout="wide")
st.title("Stock & Crypto Momentum Rebalancer")

# -----------------------------
# Secrets / env (support both key styles)
# -----------------------------
RH_USERNAME = st.secrets.get("RH_USERNAME") or st.secrets.get("ROBINHOOD_USERNAME")
RH_PASSWORD = st.secrets.get("RH_PASSWORD") or st.secrets.get("ROBINHOOD_PASSWORD")

# -----------------------------
# Helpers
# -----------------------------
CRYPTO_UNIVERSE = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "USDT-USD"]


def _is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith("-USD")


def fetch_pct_change(symbol: str, days: int = 1) -> float:
    """Return percent change over the last `days` trading days using yfinance.
    Works for both stocks and crypto.
    """
    try:
        df = yf.Ticker(symbol).history(period=f"{days + 2}d")
        if len(df) < 2:
            return np.nan
        c0, c1 = float(df["Close"].iloc[-2]), float(df["Close"].iloc[-1])
        if c0 == 0:
            return np.nan
        return (c1 - c0) / c0 * 100.0
    except Exception:
        return np.nan


def latest_price(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="2d")
        return float(df["Close"].iloc[-1])
    except Exception:
        return float("nan")


# -----------------------------
# Robinhood auth + orders
# -----------------------------

def rh_login_if_needed() -> bool:
    """Lazy-login. Returns True when authenticated and usable for live orders."""
    if st.session_state.get("rh_ok"):
        return True
    if rhood is None:
        st.error("robin_stocks not installed in this environment.")
        return False
    if not RH_USERNAME or not RH_PASSWORD:
        st.warning("Missing RH_USERNAME/RH_PASSWORD (or ROBINHOOD_USERNAME/ROBINHOOD_PASSWORD) in Secrets. Running in simulation only.")
        return False
    try:
        rhood.authentication.login(
            username=RH_USERNAME,
            password=RH_PASSWORD,
            store_session=False,
            by_sms=True,
        )
        st.session_state["rh_ok"] = True
        return True
    except Exception as e:
        st.error(f"Robinhood login failed: {e}")
        return False


# --- order helpers (use orders module; round quantities to avoid precision errors) ---

def _round_qty(q: float, decimals: int = 8) -> float:
    return float(np.floor(q * (10 ** decimals)) / (10 ** decimals))


def place_stock_order(symbol: str, side: str, usd: float) -> Dict:
    if side not in {"BUY", "SELL"}:
        return {"ok": False, "error": "invalid side"}
    try:
        # prefer fractional-by-price which is supported for most stocks/ETFs
        if side == "BUY":
            resp = rhood.orders.order_buy_fractional_by_price(symbol, amountInDollars=str(round(usd, 2)))
        else:
            resp = rhood.orders.order_sell_fractional_by_price(symbol, amountInDollars=str(round(usd, 2)))
        return {"ok": True, "raw": resp, "id": resp.get("id") if isinstance(resp, dict) else None}
    except AttributeError:
        # fallback to market by quantity
        px = latest_price(symbol)
        if not np.isfinite(px) or px <= 0:
            return {"ok": False, "error": "no price for fallback"}
        qty = _round_qty(usd / px, 6)
        try:
            if side == "BUY":
                resp = rhood.orders.order_buy_market(symbol, str(qty))
            else:
                resp = rhood.orders.order_sell_market(symbol, str(qty))
            return {"ok": True, "raw": resp, "id": resp.get("id") if isinstance(resp, dict) else None}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def place_crypto_order(symbol: str, side: str, usd: float) -> Dict:
    coin = symbol.split("-")[0]
    try:
        if side == "BUY":
            # prefer buy-by-price if available in current library version
            fn = getattr(rhood.orders, "order_buy_crypto_by_price", None)
            if fn:
                resp = fn(coin, amountInDollars=str(round(usd, 2)))
            else:
                px = latest_price(symbol)
                qty = _round_qty(usd / px, 8)
                resp = rhood.orders.order_buy_crypto_by_quantity(coin, quantity=str(qty))
        else:
            fn = getattr(rhood.orders, "order_sell_crypto_by_price", None)
            if fn:
                resp = fn(coin, amountInDollars=str(round(usd, 2)))
            else:
                px = latest_price(symbol)
                qty = _round_qty(usd / px, 8)
                resp = rhood.orders.order_sell_crypto_by_quantity(coin, quantity=str(qty))
        return {"ok": True, "raw": resp, "id": resp.get("id") if isinstance(resp, dict) else None}
    except Exception as e:
        # common API error strings surfaced in your logs
        return {"ok": False, "error": str(e)}


def get_open_orders() -> Dict[str, List[Dict]]:
    out = {"stocks": [], "crypto": []}
    try:
        out["stocks"] = rhood.orders.get_all_open_stock_orders() or []
    except Exception as e:
        out["stocks_error"] = str(e)
    try:
        # function lives in orders module (not crypto). Support alternate names across versions.
        fn = getattr(rhood.orders, "get_all_open_crypto_orders", None) or getattr(rhood.orders, "get_all_crypto_orders", None)
        out["crypto"] = fn() if fn else []
    except Exception as e:
        out["crypto_error"] = str(e)
    return out


# --- holdings helpers (for SELL rules) ---

def get_current_holdings() -> Dict[str, Dict]:
    """Return {symbol: {qty, equity, is_crypto}} for stocks and crypto when possible."""
    holdings: Dict[str, Dict] = {}
    if rhood is None:
        return holdings

    # Stocks
    try:
        built = rhood.build_holdings() or {}
        for sym, info in built.items():
            try:
                qty = float(info.get("quantity", 0) or 0)
                eq = float(info.get("equity", 0) or 0)
                holdings[sym.upper()] = {"qty": qty, "equity": eq, "is_crypto": False}
            except Exception:
                continue
    except Exception:
        pass

    # Crypto (best-effort; schema varies between lib versions)
    try:
        cpos = rhood.crypto.get_crypto_positions() or []
        for p in cpos:
            try:
                qty = float(p.get("quantity", 0) or p.get("quantity_available", 0) or 0)
                if qty <= 0:
                    continue
                code = (
                    (p.get("currency") or {}).get("code")
                    or (p.get("currency") or {}).get("symbol")
                    or (p.get("currency") or {}).get("name")
                )
                if not code:
                    # as a fallback some versions expose "currency_pair" like "XRP-USD"
                    code = p.get("currency_pair", "").split("-")[0]
                if not code:
                    continue
                sym = f"{code}-USD".upper()
                px = latest_price(sym)
                eq = float(qty) * (px if np.isfinite(px) else 0.0)
                holdings[sym] = {"qty": qty, "equity": eq, "is_crypto": True}
            except Exception:
                continue
    except Exception:
        pass

    return holdings


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Authentication & Mode")
    live = st.checkbox("Enable Live Trading (use with caution)", value=False)
    if live:
        if rh_login_if_needed():
            st.success("Robinhood authenticated — Live orders ENABLED")
        else:
            st.error("Live trading selected but login is not available. Orders will NOT be sent.")
    else:
        st.info("Simulation mode — no live orders placed.")

    st.subheader("Universe & Allocation")
    tickers = st.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG").strip()
    include_crypto = st.checkbox("Include Crypto", value=True)

    alloc_mode = st.selectbox("Allocation mode", ["Fixed $ per trade", "Proportional to momentum"])
    usd_fixed = st.number_input("Fixed $ per BUY/SELL", min_value=0.0, value=5.00, step=0.50)
    total_budget = st.number_input("Total budget for BUYS (when proportional)", min_value=0.0, value=15.00, step=1.0)
    min_per_order = st.number_input("Minimum $ per order", min_value=1.0, value=1.00, step=0.50)

    picks_n = st.number_input("Number of tickers to pick", min_value=1, max_value=50, value=3)

# Build universe
universe: List[str] = []
if tickers:
    universe += [t.strip().upper() for t in tickers.split(",") if t.strip()]
if include_crypto:
    universe += CRYPTO_UNIVERSE

if not universe:
    st.stop()


# -----------------------------
# Allocation helpers
# -----------------------------

def dollars_for_buys(picks_df: pd.DataFrame) -> Dict[str, float]:
    """Return per-symbol dollars for BUYs according to selected mode."""
    symbols = list(picks_df["Ticker"])
    if alloc_mode == "Fixed $ per trade":
        return {s: float(usd_fixed) for s in symbols}

    # proportional to momentum (positive only)
    w = np.clip(picks_df["PctChange"].astype(float).values, 0, None)
    if w.sum() <= 0:
        # nothing positive -> split evenly at minimum
        return {s: float(min_per_order) for s in symbols}
    # ensure at least min_per_order per pick
    base = max(float(total_budget) - float(min_per_order) * len(symbols), 0.0)
    extra = (w / w.sum()) * base
    out = {s: float(min_per_order + extra[i]) for i, s in enumerate(symbols)}
    return out


# -----------------------------
# Scan + Rebalance button
# -----------------------------

def scan_and_trade():
    # 1) compute momentum (1-day % change)
    rows = []
    for t in universe:
        pct = fetch_pct_change(t, days=1)
        rows.append({"Ticker": t, "PctChange": pct})
    df = pd.DataFrame(rows).dropna()

    if df.empty:
        st.warning("No price data available for the selected universe today.")
        return

    df = df.sort_values("PctChange", ascending=False).reset_index(drop=True)

    # 2) pick top-N for BUY consideration (only positive momentum are candidates)
    picks = df.head(int(picks_n)).copy()
    buy_candidates = picks[picks["PctChange"] > 0].copy()

    # 3) determine SELL candidates based on current holdings
    holdings = get_current_holdings() if (live and st.session_state.get("rh_ok")) else {}
    held_syms = {s for s in holdings.keys() if s in universe}
    buy_syms = set(buy_candidates["Ticker"]) if not buy_candidates.empty else set()
    sell_syms = held_syms - buy_syms  # rotate out anything we hold that's not in the top/momentum list

    # 4) allocation for buys
    buy_allocs = dollars_for_buys(buy_candidates) if not buy_candidates.empty else {}

    # 5) execute orders
    logs = []

    # --- BUYS ---
    for _, r in buy_candidates.iterrows():
        tkr = str(r["Ticker"])
        px = latest_price(tkr)
        usd_amt = float(buy_allocs.get(tkr, usd_fixed))
        if not np.isfinite(px) or px <= 0:
            status = {"ok": False, "error": "no price"}
        else:
            if live and st.session_state.get("rh_ok"):
                if _is_crypto(tkr):
                    status = place_crypto_order(tkr, "BUY", usd_amt)
                else:
                    status = place_stock_order(tkr, "BUY", usd_amt)
            else:
                status = {"ok": True, "simulated": True}
        logs.append({
            "Ticker": tkr,
            "Action": "BUY",
            "PctChange": round(float(r["PctChange"]), 2),
            "Executed": 0.0 if status.get("simulated") else round(usd_amt, 2),
            "OrderID": status.get("id", "simulated" if status.get("simulated") else None),
            "Status": ("simulated" if status.get("simulated") else ("ok" if status.get("ok") else status.get("error", "error"))),
            "Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # --- SELLS --- (cap each sell by available equity if we can estimate it)
    for tkr in sorted(sell_syms):
        px = latest_price(tkr)
        # cap sell dollar amount by estimated equity to avoid over-selling fractional positions
        est_equity = float(holdings.get(tkr, {}).get("equity", 0.0))
        usd_amt = float(min(usd_fixed if usd_fixed > 0 else min_per_order, est_equity if est_equity > 0 else usd_fixed))
        if usd_amt <= 0:
            continue
        if not np.isfinite(px) or px <= 0:
            status = {"ok": False, "error": "no price"}
        else:
            if live and st.session_state.get("rh_ok"):
                if _is_crypto(tkr):
                    status = place_crypto_order(tkr, "SELL", usd_amt)
                else:
                    status = place_stock_order(tkr, "SELL", usd_amt)
            else:
                status = {"ok": True, "simulated": True}
        logs.append({
            "Ticker": tkr,
            "Action": "SELL",
            "PctChange": float(df.loc[df["Ticker"] == tkr, "PctChange"].values[0]) if tkr in set(df["Ticker"]) else np.nan,
            "Executed": 0.0 if status.get("simulated") else round(usd_amt, 2),
            "OrderID": status.get("id", "simulated" if status.get("simulated") else None),
            "Status": ("simulated" if status.get("simulated") else ("ok" if status.get("ok") else status.get("error", "error"))),
            "Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # 6) show results
    st.subheader("Rebalance Log")
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.write("No trades generated.")

    # 7) open orders panel
    st.subheader("Open Orders")
    if live and st.session_state.get("rh_ok"):
        oo = get_open_orders()
        if oo.get("stocks"):
            st.caption("Stocks (open)")
            st.dataframe(pd.DataFrame(oo["stocks"]))
        if oo.get("crypto"):
            st.caption("Crypto (open)")
            st.dataframe(pd.DataFrame(oo["crypto"]))
        if oo.get("stocks_error") or oo.get("crypto_error"):
            st.warning(f"Open orders fetch issues: {oo.get('stocks_error','') or ''} {oo.get('crypto_error','') or ''}")
    else:
        st.write("[]")


if st.button("▶ Run Daily Scan & Rebalance"):
    scan_and_trade()
