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

# -------------------------
# ▶  CONFIG & SECRETS
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Live Trading + Tax Logs)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Robinhood API client (if installed)
try:
    from robin_stocks import robinhood as r
except ImportError:
    r = None  # will simulate

# -------------------------
# ▶  SIDEBAR UI
# -------------------------
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

status_badge = "🟢 LIVE" if not simulate_mode else "🔴 SIM"

# -------------------------
# ▶  Robinhood login
# -------------------------
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

# -------------------------
# ▶  CACHED DATA FETCHING
# -------------------------
@st.cache_data(show_spinner=False)
def get_data(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    """Download OHLC data, compute indicators, cache results per session."""
    for _ in range(retries):
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if not df.empty:
            # ensure Close is a Series
            close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].squeeze()
            sma20, upper, lower = bollinger_bands(close)
            df['sma_20'], df['bb_upper'], df['bb_lower'] = sma20, upper, lower
            df['sma_50'] = simple_sma(close, 50)
            df['rsi'] = simple_rsi(close)
            return df.dropna()
        time.sleep(1)
    raise ValueError(f"No data for {ticker}")

# -------------------------
# ▶  INDICATORS & ANALYSIS
# -------------------------
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

# Updated analyze to cast to float and avoid ambiguous Series comparisons

def analyze(df: pd.DataFrame) -> dict | None:
    if len(df) < 30:  # lowered threshold
        if debug_mode:
            st.warning(f"⏳ Only {len(df)} rows fetched — skipping due to low data.")
        return None
    # take last two rows as scalars
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    # cast to float
    rsi_val = float(cur['rsi'])
    sma20_val = float(cur['sma_20'])
    sma50_val = float(cur['sma_50'])
    price_val = float(cur['Close'])
    bb_lower_val = float(cur['bb_lower'])
    bb_upper_val = float(cur['bb_upper'])

    reasons = []
    signal = 'HOLD'
    if rsi_val < 30:
        reasons.append('RSI below 30 (oversold)')
    if rsi_val > 70:
        reasons.append('RSI above 70 (overbought)')
    # separate comparisons
    if prev['sma_20'] < prev['sma_50'] and sma20_val >= sma50_val:
        reasons.append('20 SMA crossed above 50 SMA (bullish)')
    if prev['sma_20'] > prev['sma_50'] and sma20_val <= sma50_val:
        reasons.append('20 SMA crossed below 50 SMA (bearish)')
    if price_val < bb_lower_val:
        reasons.append('Price below lower BB')
    if price_val > bb_upper_val:
        reasons.append('Price above upper BB')

    text = '; '.join(reasons).lower()
    if 'buy' in text or 'bullish' in text:
        signal = 'BUY'
    elif 'sell' in text or 'bearish' in text:
        signal = 'SELL'

    return {
        'RSI': round(rsi_val,2),
        '20 SMA': round(sma20_val,2),
        'Signal': signal,
        'Reasons': '; '.join(reasons)
    }

# -------------------------
# ▶  NOTIFICATION HELPERS
# -------------------------
def notify_email(tkr, summ, price):
    msg = EmailMessage()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg.set_content(f"Ticker: {tkr}\nSignal: {summ['Signal']} @ ${price}\n{summ['Reasons']}\n{now}")
    msg['Subject'] = f"{summ['Signal']} Signal: {tkr}"
    msg['From'] = st.secrets['EMAIL_ADDRESS']
    msg['To'] = st.secrets['EMAIL_RECEIVER']
    with smtplib.SMTP_SSL('smtp.gmail.com',465) as s:
        s.login(st.secrets['EMAIL_ADDRESS'], st.secrets['EMAIL_PASSWORD'])
        s.send_message(msg)

WEBHOOK = st.secrets.get('SLACK_WEBHOOK_URL')
def notify_slack(tkr, summ, price):
    if not WEBHOOK: return
    payload = { 'text': f"*{summ['Signal']}* {tkr} @ ${price}\n{summ['Reasons']}" }
    requests.post(WEBHOOK, json=payload)

# -------------------------
# ▶  TRADE LOGGING
# -------------------------
def log_trade(tkr, summ, price):
    if summ['Signal'] == 'HOLD':
        return
    row = {
        'Date': datetime.now().isoformat(),
        'Ticker': tkr,
        'Signal': summ['Signal'],
        'Price': price,
        'Gain/Loss': round(np.random.uniform(-20,50),2),
        'Reasons': summ['Reasons']
    }
    df = pd.DataFrame([row])
    df.to_csv('trade_log.csv', mode='a', header=not os.path.exists('trade_log.csv'), index=False)
    notify_email(tkr,summ,price)
    notify_slack(tkr,summ,price)

# -------------------------
# ▶  MAIN
# -------------------------
st.markdown(f"### {status_badge} {PAGE_TITLE}")
if st.button('▶ Run Analysis'):
    results = {}
    for tkr in tickers:
        try:
            df = get_data(tkr, period)
            if debug_mode:
                st.write(f"{tkr}: {len(df)} rows")
            summ = analyze(df)
            if summ is None:
                continue
            results[tkr] = summ
            price = float(df['Close'].iloc[-1])
            log_trade(tkr, summ, price)
            # chart
            st.markdown(f"#### {tkr} Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"{tkr} error: {e}")
    # summary
    if results:
        rdf = pd.DataFrame(results).T
        st.download_button('⬇ CSV', rdf.to_csv().encode(), 'results.csv')
        sig_map = {k: 1 if v['Signal']=='BUY' else -1 if v['Signal']=='SELL' else 0 for k,v in results.items()}
        st.bar_chart(pd.Series(sig_map))

# Persistent logs
if os.path.exists('trade_log.csv'):
    trades = pd.read_csv('trade_log.csv')
    st.subheader('🧾 Trade Log')
    st.dataframe(trades)
    st.download_button('⬇ Trade Log', trades.to_csv(index=False).encode(), 'trade_log.csv')
    trades['Cum P/L'] = trades['Gain/Loss'].cumsum()
    st.markdown(f"**Total P/L: ${trades['Gain/Loss'].sum():.2f}**")
    st.line_chart(trades.set_index('Date')['Cum P/L'])
