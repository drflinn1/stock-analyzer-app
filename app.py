# app.py — Stock Analyzer Bot (Stage 2: Indicators + Signals) + UI polish
# - Compact layout
# - Sticky header
# - Signal badges in summary table
# - Optional S&P Top-N scan and optional crypto scan

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st
import yfinance as yf

# -------------------------
# ▶  BASIC CONFIG
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Stage 2: Indicators + Signals)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- UI polish (compact layout + sticky header) ---
st.markdown(
    """
    <style>
      /* compact page */
      .block-container {padding-top: 0.75rem !important; padding-bottom: 0.5rem !important;}
      section[data-testid="stSidebar"] .block-container {padding-top: 0.5rem !important;}
      /* sticky header bar */
      .sticky-bar {position: sticky; top: 0; z-index: 100; background: var(--background-color);
                   padding: .35rem .5rem .5rem .5rem; border-bottom: 1px solid rgba(128,128,128,.2);}
      /* small table tweaks */
      div[data-testid="stDataFrame"] div[role="table"] { font-size: 0.90rem; }
      /* code blocks a bit smaller */
      pre, code { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# ▶  HELPERS
# -------------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers() -> List[str]:
    """Return S&P-500 tickers from Wikipedia (cached)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df_list = pd.read_html(url)
        symbols = df_list[0]["Symbol"].tolist()
        return [s.strip().upper() for s in symbols]
    except Exception as e:
        st.sidebar.warning(f"Could not fetch S&P 500 list: {e}")
        return []

def simple_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def simple_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = simple_sma(series, window)
    std = series.rolling(window).std()
    return sma, sma + num_std * std, sma - num_std * std

@st.cache_data(show_spinner=False)
def get_data(ticker: str, period: str, retries: int = 2) -> pd.DataFrame:
    """Download OHLCV and compute indicators."""
    last_err = None
    for _ in range(retries + 1):
        try:
            df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
            # If MultiIndex (rare for a single ticker), flatten
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.xs(ticker, axis=1, level=1)
                except Exception:
                    df.columns = df.columns.get_level_values(0)
            if df.empty:
                raise ValueError("no rows")
            close = df["Close"].astype(float)
            sma20, bb_up, bb_lo = bollinger_bands(close, 20, 2)
            df["sma_20"] = sma20
            df["bb_upper"] = bb_up
            df["bb_lower"] = bb_lo
            df["sma_50"] = simple_sma(close, 50)
            df["rsi"] = simple_rsi(close, 14)
            return df.dropna()
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise ValueError(f"No data fetched for {ticker} ({last_err})")

def analyze_row(df: pd.DataFrame, rsi_ovr: float, rsi_obh: float) -> Dict[str, Any] | None:
    """Return last-row indicators + signal & reasons."""
    if df is None or len(df) < 5:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    price = float(cur["Close"])
    rsi = float(cur["rsi"])
    sma20_cur, sma20_prev = float(cur["sma_20"]), float(prev["sma_20"])
    sma50_cur, sma50_prev = float(cur["sma_50"]), float(prev["sma_50"])
    upper, lower = float(cur["bb_upper"]), float(cur["bb_lower"])

    reasons = []
    if rsi < rsi_ovr:
        reasons.append(f"RSI < {rsi_ovr} (oversold)")
    if rsi > rsi_obh:
        reasons.append(f"RSI > {rsi_obh} (overbought)")
    if sma20_prev < sma50_prev <= sma20_cur:
        reasons.append("20 SMA crossed above 50 SMA (bullish)")
    if sma20_prev > sma50_prev >= sma20_cur:
        reasons.append("20 SMA crossed below 50 SMA (bearish)")
    if price < lower:
        reasons.append("Close < lower BB (washout)")
    if price > upper:
        reasons.append("Close > upper BB")

    text = "; ".join(reasons).lower()
    signal = "HOLD"
    if any(k in text for k in ["bullish", "oversold"]) and "bearish" not in text:
        signal = "BUY"
    if any(k in text for k in ["bearish", "overbought"]):
        signal = "SELL"

    return {
        "Close": round(price, 2),
        "RSI": round(rsi, 2),
        "20 SMA": round(sma20_cur, 3),
        "50 SMA": round(sma50_cur, 3),
        "BB Lower": round(lower, 3),
        "BB Upper": round(upper, 3),
        "Signal": signal,
        "Reasons": "; ".join(reasons) if reasons else "—",
    }

def badge(signal: str) -> str:
    m = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}
    return m.get(signal, signal)

# -------------------------
# ▶  SIDEBAR (controls)
# -------------------------
with st.sidebar:
    st.markdown("### Settings")

    with st.expander("General", expanded=True):
        simulate_mode = st.checkbox("Simulate Trading Mode", True)
        debug_mode = st.checkbox("Show debug logs", False)

    with st.expander("Analysis Options", expanded=True):
        scan_top = st.checkbox("Scan top N performers", False, help="Autoselect Top N S&P-500 movers today.")
        top_n = st.slider("Top tickers to scan (stocks)", 5, 50, 25) if scan_top else None
        include_crypto = st.checkbox("Include crypto in scan (optional)", False)
        top_k_coins = st.slider("Top coins to include", 1, 20, 5) if include_crypto else 0

        min_rows = st.slider("Minimum data rows", 20, 200, 60)
        rsi_ovr = st.slider("RSI oversold threshold", 0, 100, 30)
        rsi_obh = st.slider("RSI overbought threshold", 0, 100, 70)

        # Query params (read-only use)
        qs = st.query_params
        period_opts = ["3mo", "6mo", "1y", "2y"]
        default_period = qs.get("period", "6mo")
        if default_period not in period_opts:
            default_period = "6mo"

        # Build universe
        if scan_top:
            base = get_sp500_tickers()
            # Try to get daily movers (percent change) quickly
            movers = []
            try:
                df = yf.download(base, period="2d", progress=False)["Close"]
                if isinstance(df, pd.Series):
                    pct = df.pct_change().iloc[-1:].to_frame().T
                else:
                    pct = df.pct_change().iloc[-1]
                movers = pct.dropna().sort_values(ascending=False).head(top_n).index.tolist()
            except Exception:
                movers = base[:top_n]
            universe = movers
        else:
            universe = get_sp500_tickers()[:200] or ["AAPL", "MSFT", "NVDA", "GOOGL"]

        default_list = qs.get("tickers", "") or ""
        default_list = [t for t in default_list.split(",") if t] or universe[:2]

        tickers = st.multiselect("Choose tickers", universe, default=default_list, key="tickers")
        period = st.selectbox("Date range", period_opts, index=period_opts.index(default_period), key="period")

        # Helper text: show which top movers got autoselected
        if scan_top and movers:
            st.caption(f"Autoselected (Top Movers): {', '.join(movers[:10])}{'…' if len(movers)>10 else ''}")

# Sync URL (shareable state)
# We only write when the user clicks Run to avoid constant URL churn.
def set_query_params():
    st.query_params.tickers = ",".join(tickers) if tickers else ""
    st.query_params.period = period

# -------------------------
# ▶  MAIN LAYOUT
# -------------------------
st.markdown(f"""
<div class="sticky-bar">
  <h3 style="display:inline;">{'🔴 SIM' if simulate_mode else '🟢 LIVE'} {PAGE_TITLE}</h3>
  <span style="float:right;">
    <form action="#" method="get" style="display:inline;">
      <button type="button" id="run-btn">▶ Run Analysis</button>
    </form>
  </span>
</div>
<script>
const btn=document.getElementById('run-btn');
if(btn){btn.addEventListener('click',()=>{window.parent.postMessage({isRun:true}, '*');});}
</script>
""", unsafe_allow_html=True)

# A little JS bridge: Streamlit can’t catch postMessage—so we also render a regular button:
run_clicked = st.button("▶ Run Analysis", use_container_width=True)

# -------------------------
# ▶  RUN
# -------------------------
results: Dict[str, Dict[str, Any]] = {}
errors: List[str] = []

if run_clicked:
    if not tickers and not include_crypto:
        st.warning("Select at least one ticker or enable crypto scan.")
    else:
        # update shareable URL
        set_query_params()

        # STOCKS
        for tkr in tickers:
            try:
                df = get_data(tkr, period)
                if len(df) < min_rows:
                    errors.append(f"{tkr} skipped (not enough rows)")
                    continue
                # plot
                st.markdown(f"#### 📈 {tkr} Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
                fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="20 SMA"))
                fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="50 SMA"))
                fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(dash="dot")))
                fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(dash="dot")))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

                summ = analyze_row(df, rsi_ovr, rsi_obh)
                if summ:
                    results[tkr] = summ
                    st.markdown(f"**{badge(summ['Signal'])} — {tkr}**")
                    if debug_mode:
                        st.json(summ)
                    st.divider()
                else:
                    errors.append(f"{tkr}: could not compute summary")
            except Exception as e:
                errors.append(f"{tkr} failed: {e}")

        # CRYPTO (optional, top-k by market cap via pycoingecko if installed)
        if include_crypto and top_k_coins > 0:
            try:
                from pycoingecko import CoinGeckoAPI  # type: ignore
                cg = CoinGeckoAPI()
                top = cg.get_coins_markets(vs_currency="usd", order="market_cap_desc", per_page=top_k_coins, page=1)
                coin_ids = [c["id"] for c in top][:top_k_coins]
            except Exception:
                coin_ids = []

            for cid in coin_ids:
                try:
                    # use yfinance synthetic tickers for some big coins if available, else skip silently
                    yf_symbol = {"bitcoin": "BTC-USD", "ethereum": "ETH-USD"}.get(cid, None)
                    if not yf_symbol:
                        continue
                    df = get_data(yf_symbol, period)
                    if len(df) < min_rows:
                        errors.append(f"{yf_symbol} skipped (not enough rows)")
                        continue

                    st.markdown(f"#### ₿ {yf_symbol} Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
                    fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="20 SMA"))
                    fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="50 SMA"))
                    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(dash="dot")))
                    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(dash="dot")))
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

                    summ = analyze_row(df, rsi_ovr, rsi_obh)
                    if summ:
                        results[yf_symbol] = summ
                        st.markdown(f"**{badge(summ['Signal'])} — {yf_symbol}**")
                        if debug_mode:
                            st.json(summ)
                        st.divider()
                except Exception as e:
                    errors.append(f"{cid} failed: {e}")

# -------------------------
# ▶  SUMMARY TABLE + EXPORT
# -------------------------
if results:
    df_sum = pd.DataFrame(results).T
    # badge column first
    df_sum.insert(0, "Signal", df_sum["Signal"].map(badge))
    # display
    st.markdown("### 📊 Summary of Trade Signals")
    st.dataframe(df_sum, use_container_width=True, height=420)
    st.download_button(
        "⬇ Download results CSV",
        df_sum.to_csv(index=True).encode(),
        "stock_analysis_results.csv",
    )

if errors:
    for e in errors:
        st.error(e)
