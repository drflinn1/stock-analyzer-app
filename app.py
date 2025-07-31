# app.py – Streamlit Web App Version of Stock Analyzer Bot with S&P Scan & Full Ticker Selection

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

# -------------------------
# ▶  CONFIG & SECRETS
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Live Trading + Tax Logs)"
# Optional public app URL for links
APP_URL = st.secrets.get('APP_URL', '')
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Robinhood API client (if installed)
try:
    from robin_stocks import robinhood as r
except ImportError:
    r = None

# -------------------------
# ▶  Helper to fetch S&P 500 & Top Movers
# -------------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df_list = pd.read_html(url, flavor='bs4', attrs={'class':'wikitable'})
        return df_list[0]['Symbol'].tolist()
    except Exception:
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'class':'wikitable'})
            return [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
        except Exception as e:
            st.sidebar.warning(f"Failed to fetch S&P 500 list: {e}")
            return []

# No caching on this to always get fresh top performers
def get_top_tickers(n: int) -> list[str]:
    symbols = get_sp500_tickers()
    if not symbols:
        return []
    try:
        df = yf.download(symbols, period='2d', progress=False)['Close']
        if isinstance(df, pd.Series):
            changes = df.pct_change().iloc[-1:]
            changes = changes.to_frame().T
        else:
            changes = df.pct_change().iloc[-1]
        return changes.dropna().sort_values(ascending=False).head(n).index.tolist()
    except Exception:
        perf = {}
        for sym in symbols:
            try:
                tmp = yf.download(sym, period='2d', progress=False)['Close']
                if len(tmp) >= 2:
                    perf[sym] = float(tmp.pct_change().iloc[-1])
            except Exception:
                continue
        return sorted(perf, key=lambda k: perf[k], reverse=True)[:n]

# -------------------------
# ▶  Analysis Helpers
# -------------------------
def simple_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def simple_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = simple_sma(series, window)
    std = series.rolling(window).std()
    return sma, sma + num_std * std, sma - num_std * std

# -------------------------
# ▶  Data Fetch & Indicators
# -------------------------
@st.cache_data(show_spinner=False)
def get_data(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    for _ in range(retries):
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except Exception:
                df = df.iloc[:, df.columns.get_level_values(1) == ticker]
                df.columns = df.columns.get_level_values(0)
        if not df.empty:
            sma20, upper, lower = bollinger_bands(df['Close'].squeeze())
            df['sma_20'], df['bb_upper'], df['bb_lower'] = sma20, upper, lower
            df['sma_50'] = simple_sma(df['Close'].squeeze(), 50)
            df['rsi'] = simple_rsi(df['Close'].squeeze())
            return df.dropna()
        time.sleep(1)
    raise ValueError(f"No data fetched for {ticker}")

# -------------------------
# ▶  Signals & Thresholds
# -------------------------
@st.cache_data(show_spinner=False)
def analyze(df: pd.DataFrame,
            min_rows: int,
            rsi_ovr: float,
            rsi_obh: float) -> dict | None:
    if not isinstance(df, pd.DataFrame) or len(df) < min_rows:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    rsi = float(cur['rsi'])
    sma20_cur, sma50_cur = float(cur['sma_20']), float(cur['sma_50'])
    price = float(cur['Close'])

    reasons = []
    if rsi < rsi_ovr:
        reasons.append(f'RSI below {rsi_ovr} (oversold)')
    if rsi > rsi_obh:
        reasons.append(f'RSI above {rsi_obh} (overbought)')
    if prev['sma_20'] < prev['sma_50'] <= sma20_cur:
        reasons.append('20 SMA crossed above 50 SMA (bullish)')
    if prev['sma_20'] > prev['sma_50'] >= sma20_cur:
        reasons.append('20 SMA crossed below 50 SMA (bearish)')
    if price < float(cur['bb_lower']):
        reasons.append('Price below lower BB')
    if price > float(cur['bb_upper']):
        reasons.append('Price above upper BB')

    txt = "; ".join(reasons).lower()
    signal = 'HOLD'
    if any(k in txt for k in ['buy','bullish']): signal = 'BUY'
    elif any(k in txt for k in ['sell','bearish']): signal = 'SELL'

    return {
        'RSI': round(rsi, 2),
        '20 SMA': round(sma20_cur, 2),
        '50 SMA': round(sma50_cur, 2),
        'Signal': signal,
        'Reasons': '; '.join(reasons)
    }

# -------------------------
# ▶  Notifications & Logging
# -------------------------
WEBHOOK = st.secrets.get('SLACK_WEBHOOK_URL')

def notify_slack(tkr: str, summ: dict, price: float):
    if WEBHOOK:
        link = f" (<{APP_URL}|View in App>)" if APP_URL else ''
        text = f"*{summ['Signal']}* {tkr} @ ${price}\n{summ['Reasons']}{link}"
        requests.post(WEBHOOK, json={'text': text})

def notify_email(tkr: str, summ: dict, price: float):
    msg = EmailMessage()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    link = f"\n\nView in app: {APP_URL}" if APP_URL else ''
    body = f"Ticker: {tkr}\nSignal: {summ['Signal']} @ ${price}\nReasons: {summ['Reasons']}\nTime: {now}{link}"
    msg.set_content(body)
    msg['Subject'], msg['From'], msg['To'] = (
        f"{summ['Signal']} {tkr}",
        st.secrets['EMAIL_ADDRESS'],
        st.secrets['EMAIL_RECEIVER']
    )
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
        s.login(st.secrets['EMAIL_ADDRESS'], st.secrets['EMAIL_PASSWORD'])
        s.send_message(msg)

# -------------------------
# ▶  Sidebar UI & Controls
# -------------------------
with st.sidebar:
    st.markdown('## ⚙️ Settings')
    with st.expander('General', expanded=True):
        simulate_mode = st.checkbox('Simulate Trading Mode', True)
        debug_mode = st.checkbox('Show debug logs', False)
        st_autorefresh(interval=3600000, limit=None, key='hour_refresh')

    with st.expander('Analysis Options', expanded=True):
        scan_top = st.checkbox('Scan top N performers', False)
        top_n = st.slider('Top tickers to scan', 10, 100, 50) if scan_top else None
        universe = get_top_tickers(top_n) if scan_top else get_sp500_tickers()

        # Threshold controls
        min_rows = st.slider('Minimum data rows', 10, 100, 30)
        rsi_ovr = st.slider('RSI oversold threshold', 0, 100, 30)
        rsi_obh = st.slider('RSI overbought threshold', 0, 100, 70)

        default_list = universe[:2]
        tickers = st.multiselect('Choose tickers', universe, default=default_list, key='tickers')
        period = st.selectbox('Date range', ['1mo','3mo','6mo','1y','2y'], index=2)

# -------------------------
# ▶  Main Page
# -------------------------
st.markdown(f"### {'🔴 SIM' if simulate_mode else '🟢 LIVE'} {PAGE_TITLE}")
if st.button('▶ Run Analysis', use_container_width=True):
    if not tickers:
        st.warning('Select at least one ticker')
        st.stop()
    results = {}
    for tkr in tickers:
        try:
            df = get_data(tkr, period)
            if debug_mode:
                st.write(f"{tkr} rows: {len(df)}")
            summ = analyze(df, min_rows, rsi_ovr, rsi_obh)
            if summ is None:
                st.warning(f"{tkr}: Not enough data, skipped")
                continue
            results[tkr] = summ
            notify_email(tkr, summ, float(df.Close.iloc[-1]))
            notify_slack(tkr, summ, float(df.Close.iloc[-1]))

            st.markdown(f"#### 📈 {tkr} Price Chart")
            fig = go.Figure()
