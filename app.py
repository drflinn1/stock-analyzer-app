# app.py – Streamlit Web App Version of Stock & Crypto Day Trading Bot (Fully Automated)

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st
import yfinance as yf

# stub out YFRateLimitError if unavailable
try:
    from yfinance.utils import YFRateLimitError
except ImportError:
    class YFRateLimitError(Exception):
        pass

from streamlit_autorefresh import st_autorefresh

# -------------------------
# ▶  CONFIG & SECRETS
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Automated Equities & Crypto Trading)"
APP_URL = st.secrets.get('APP_URL', '')
WEBHOOK = st.secrets.get('SLACK_WEBHOOK_URL', '')

# -------------------------
# ▶  Notification Helper
# -------------------------
def notify_slack(tkr: str, summ: dict, price: float):
    # Post summary to Slack with link back to dashboard
    if WEBHOOK and APP_URL:
        qs = f"?period={st.session_state.period}&scan_n={st.session_state.get('scan_n',0)}"
        link = f" (<{APP_URL}{qs}|Dashboard>)"
    else:
        link = ''
    text = f"*{summ['Signal']}* {tkr} @ ${price:.2f}\nReasons: {summ.get('Reasons','')}" + link
    if WEBHOOK:
        requests.post(WEBHOOK, json={'text': text})

# Stub for live trading – integrate broker API here
def make_live_trade(tkr: str, signal: str, qty: float):
    # TODO: place market order via broker API
    return

# -------------------------
# ▶  Universe Helpers
# -------------------------
def get_sp500_tickers():
    try:
        df = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            flavor=['lxml', 'html5lib']
        )[0]
        return df['Symbol'].tolist()
    except Exception:
        st.error("⚠️ Failed to fetch S&P 500 list. Ensure lxml/html5lib installed.")
        return []

def get_crypto_universe():
    # select liquid USD pairs
    return ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LTC-USD"]

# -------------------------
# ▶  Data & Indicator Helpers
# -------------------------
def fetch_data(symbol: str, period: str = '6mo') -> pd.DataFrame:
    retries = 3
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period)
            df['Close'] = df['Close'].astype(float)
            df['20_SMA'] = df['Close'].rolling(20).mean()
            df['50_SMA'] = df['Close'].rolling(50).mean()
            df['RSI'] = compute_rsi(df['Close'])
            df['BB_upper'], df['BB_lower'] = compute_bollinger(df['Close'])
            return df.dropna()
        except YFRateLimitError:
            time.sleep(2 ** attempt)
    st.error(f"⚠️ Rate limit fetching {symbol}, skipping.")
    return pd.DataFrame()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=window-1, adjust=False).mean()
    ma_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def compute_bollinger(series: pd.Series, window: int = 20, num_std: int = 2):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return m + num_std * s, m - num_std * s


def analyze_signal(df: pd.DataFrame, oversold: int, overbought: int) -> dict:
    latest = df.iloc[-1]
    reasons = []
    if latest['RSI'] < oversold:
        reasons.append('RSI below oversold')
    if latest['Close'] < latest['BB_lower']:
        reasons.append('Below lower BB')
    sig = 'HOLD'
    if 'RSI below oversold' in reasons and 'Below lower BB' in reasons:
        sig = 'BUY'
    elif latest['RSI'] > overbought and latest['Close'] > latest['BB_upper']:
        reasons.append('Above upper BB/overbought')
        sig = 'SELL'
    return {
        'Signal': sig,
        'RSI': round(latest['RSI'], 1),
        '20_SMA': round(latest['20_SMA'], 2),
        '50_SMA': round(latest['50_SMA'], 2),
        'Reasons': '; '.join(reasons)
    }

# -------------------------
# ▶  Main App
# -------------------------
st.set_page_config(page_title=PAGE_TITLE, layout='wide')
st.title(PAGE_TITLE)

# auto-refresh every 24h to run scheduled scans
st_autorefresh(interval=24*60*60*1000, key='daily_refresh')

# Sidebar controls
with st.sidebar.expander('Universe'):
    use_sp = st.checkbox('Include S&P 500', value=True)
    use_cr = st.checkbox('Include Crypto', value=True)
    scan_n = st.number_input('Top N to trade', min_value=1, max_value=50, value=5)

with st.sidebar.expander('Settings'):
    simulate = st.checkbox('Simulate Trading Mode', True)
    oversold = st.slider('RSI oversold threshold', 0, 100, 30)
    overbought = st.slider('RSI overbought threshold', 0, 100, 70)
    period = st.selectbox('History window', ['1mo', '3mo', '6mo', '1y'], index=2)
st.session_state['period'] = period
