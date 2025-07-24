# app.py – Streamlit Web App Version of Stock Analyzer with Robinhood Integration – **caching & Slack alerts**

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import smtplib
import streamlit as st
import yfinance as yf
from email.message import EmailMessage
from streamlit_autorefresh import st_autorefresh

# ----------------------------------
# ▶  CONFIG & SECRETS
# ----------------------------------
PAGE_TITLE = "Stock Analyzer Bot (Live Trading + Tax Logs)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Robinhood API client (if installed)
try:
    from robin_stocks import robinhood as r
except ImportError:
    r = None  # will simulate

# ----------------------------------
# ▶  SIDEBAR UI & TRIGGER
# ----------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    with st.expander("General", expanded=True):
        simulate_mode = st.checkbox("Simulate Trading Mode", value=True)
        debug_mode = st.checkbox("Show debug logs", value=False)
        # auto-refresh hourly
        st_autorefresh(interval=3600000, limit=None, key="hour_refresh")

    with st.expander("Analysis Options", expanded=True):
        ticker_options = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","NFLX"]
        tickers = st.multiselect("📈 Choose tickers", ticker_options, default=["AAPL","TSLA"])
        period = st.selectbox("🗓️ Date range", ["1mo","3mo","6mo","1y","2y"], index=2)

# ----------------------------------
# ▶  MAIN TRIGGER
# ----------------------------------
run_analysis = st.button("▶️ Run Analysis")
if not run_analysis:
    # don't load data or show any charts until button is pressed
    st.stop()

status_badge = "🟢 LIVE" if not simulate_mode else "🔴 SIM"

# ----------------------------------
# ▶  Robinhood login
# ----------------------------------
if not simulate_mode:
    if r is None:
        st.sidebar.error("robin_stocks missing – switching to simulate mode")
        simulate_mode = True
    else:
        try:
            r.login(st.secrets["ROBINHOOD_USERNAME"], st.secrets["ROBINHOOD_PASSWORD"])
            st.sidebar.success("✅ Connected to Robinhood")
        except Exception as e:
            st.sidebar.error(f"Robinhood login failed: {e}")
            simulate_mode = True

# ----------------------------------
# ▶  CACHED DATA FETCHING
# ----------------------------------
@st.cache_data(show_spinner=False)
def get_data(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    """Download OHLC data, compute indicators, cache results per session."""
    for _ in range(retries):
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if not df.empty:
            close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].squeeze()
            sma20, upper, lower = bollinger_bands(close)
            df['sma_20'], df['bb_upper'], df['bb_lower'] = sma20, upper, lower
            df['sma_50'] = simple_sma(close, 50)
            df['rsi'] = simple_rsi(close)
            return df.dropna()
        time.sleep(1)
    raise ValueError(f"No data for {ticker}")

# ----------------------------------
# ▶  INDICATORS & ANALYSIS
# ----------------------------------
def simple_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def simple_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = simple_sma(series, window)
    std = series.rolling(window).std()
    return sma, sma + num_std * std, sma - num_std * std

# ----------------------------------
# ▶  TRADE SIGNAL ANALYSIS
# ----------------------------------
def analyze(df: pd.DataFrame) -> dict | None:
    # require at least 30 data points
    if len(df) < 30:
        if debug_mode:
            st.warning(f"⏳ Only {len(df)} rows fetched — skipping due to low data.")
        return None

    cur = df.iloc[-1]
    prev = df.iloc[-2]
    rsi_val      = float(cur['rsi'])
    sma20_val    = float(cur['sma_20'])
    sma50_val    = float(cur['sma_50'])
    price_val    = float(cur['Close'])
    bb_lower_val = float(cur['bb_lower'])
    bb_upper_val = float(cur['bb_upper'])

    reasons = []
    if rsi_val < 30:
        reasons.append('RSI below 30 (oversold)')
    if rsi_val > 70:
        reasons.append('RSI above 70 (overbought)')
    if float(prev['sma_20']) < float(prev['sma_50']) and sma20_val >= sma50_val:
        reasons.append('20 SMA crossed above 50 SMA (bullish)')
    if float(prev['sma_20']) > float(prev['sma_50']) and sma20_val <= sma50_val:
        reasons.append('20 SMA crossed below 50 SMA (bearish)')
    if price_val < bb_lower_val:
        reasons.append('Price below lower BB')
    if price_val > bb_upper_val:
        reasons.append('Price above upper BB')

    text = '; '.join(reasons).lower()
    signal = 'HOLD'
    if 'buy' in text or 'bullish' in text:
        signal = 'BUY'
    elif 'sell' in text or 'bearish' in text:
        signal = 'SELL'

    return {
        'RSI': round(rsi_val, 2),
        '20 SMA': round(sma20_val, 2),
        'Signal': signal,
        'Reasons': '; '.join(reasons)
    }

# ----------------------------------
# ▶  PLOTTING & OUTPUT
# ----------------------------------
for ticker in tickers:
    try:
        df = get_data(ticker, period)
        result = analyze(df)
        if result:
            st.subheader(f"{ticker} — {result['Signal']}")
            st.json(result)
        else:
            st.info(f"{ticker}: Not enough data, skipped")
    except Exception as e:
        st.error(f"{ticker} error: {e}")


# End of app.py
