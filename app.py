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
WEBHOOK = st.secrets.get('SLACK_WEBHOOK_URL', '')

# -------------------------
# ▶  Notifications & Logging
# -------------------------
def notify_slack(tkr: str, summ: dict, price: float):
    """
    Send a formatted notification to Slack if a webhook is configured.
    """
    # Build the optional link only if both webhook and APP_URL are set
    if WEBHOOK and APP_URL:
        qs = f"?tickers={','.join(st.session_state['tickers'])}&period={st.session_state['period']}"
        link = f" (<{APP_URL}{qs}|View in App>)"
    else:
        link = ''

    # Compose the Slack message as a single, properly-terminated string
    text = (
        f"*{summ['Signal']}* {tkr} @ ${price}\n"
        f"Reasons: {summ.get('Reasons','')}" + link
    )

    # Post to Slack if webhook is provided
    if WEBHOOK:
        requests.post(WEBHOOK, json={'text': text})

# -------------------------
# ▶  Data Fetching & Analysis Helpers
# -------------------------
def get_sp500_tickers():
    # fetch S&P 500 list
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return table['Symbol'].tolist()


def fetch_data(ticker, period='6mo') -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    df['20_SMA'] = df['Close'].rolling(20).mean()
    df['50_SMA'] = df['Close'].rolling(50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['BB_upper'], df['BB_lower'] = compute_bollinger(df['Close'])
    return df.dropna()


def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=window - 1, adjust=False).mean()
    ma_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def compute_bollinger(series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma + num_std * std, ma - num_std * std


def analyze_signal(df, oversold=30, overbought=70):
    latest = df.iloc[-1]
    reasons = []
    if latest['RSI'] < oversold:
        reasons.append('RSI below 30 (oversold)')
    if latest['Close'] < latest['BB_lower']:
        reasons.append('Price below lower BB')
    signal = 'HOLD'
    if latest['RSI'] < oversold and latest['Close'] < latest['BB_lower']:
        signal = 'BUY'
    elif latest['RSI'] > overbought and latest['Close'] > latest['BB_upper']:
        signal = 'SELL'
    return {
        'Signal': signal,
        'RSI': round(latest['RSI'],2),
        '20_SMA': round(latest['20_SMA'],2),
        '50_SMA': round(latest['50_SMA'],2),
        'Reasons': '; '.join(reasons)
    }

# -------------------------
# ▶  Main Streamlit App
# -------------------------
st.set_page_config(page_title=PAGE_TITLE)
st.title(PAGE_TITLE)

# sidebar controls
with st.sidebar.expander('Settings'):
    simulate = st.checkbox('Simulate Trading Mode', True)
    show_debug = st.checkbox('Show debug logs', False)

with st.sidebar.expander('Analysis Options'):
    scan_sp500 = st.checkbox('Scan top N performers', False)
    min_rows = st.slider('Minimum data rows', 10, 100, 30)
    oversold = st.slider('RSI oversold threshold', 0, 100, 30)
    overbought = st.slider('RSI overbought threshold', 0, 100, 70)

    universe = get_sp500_tickers() if scan_sp500 else []
    default = [] if scan_sp500 else ['AAPL']
    tickers = st.multiselect('Choose tickers', universe, default)
    period = st.selectbox('Date range', ['1mo','3mo','6mo','1y'], index=2)

if st.button('Run Analysis'):
    results = []
    trades = []
    for tkr in tickers:
        df = fetch_data(tkr, period)
        if len(df) < min_rows:
            if show_debug: st.warning(f'Skipping {tkr}: not enough data')
            continue
        summ = analyze_signal(df, oversold, overbought)
        price = df['Close'].iloc[-1]
        results.append((tkr, summ, price))

        # simulate or live trade logic
        if simulate:
            trades.append({'Ticker': tkr, 'Signal': summ['Signal'], 'Price': price, 'Time': datetime.now()})
        else:
            # live trading via Robinhood API (placeholder)
            make_live_trade(tkr, summ['Signal'], price)

        notify_slack(tkr, summ, price)

    # display all charts and summary
    for tkr, summ, price in results:
        df = fetch_data(tkr, period)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['20_SMA'], name='20 SMA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['50_SMA'], name='50 SMA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(dash='dot')))
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**{tkr} – {summ['Signal']}**")
        st.json(summ)

    # show trade log and P/L
    if trades:
        logs = pd.DataFrame(trades)
        logs['Gain/Loss'] = 0.0  # placeholder for actual gain/loss calc
        logs['Date'] = logs['Time'].dt.date
        logs['Cum P/L'] = logs['Gain/Loss'].cumsum()

        st.subheader('📋 Trade Log')
        st.dataframe(logs)
        st.download_button('📥 Download Trade Log', logs.to_csv(index=False).encode(), 'trade_log.csv')

        tax = logs.groupby('Signal')['Gain/Loss'].sum().reset_index()
        st.subheader('Tax Summary')
        st.dataframe(tax)
        st.download_button('📥 Download Tax Summary', tax.to_csv(index=False).encode(), 'tax_summary.csv')

        st.markdown('### 📈 Portfolio Cumulative Profit Over Time')
        st.line_chart(logs.set_index('Date')['Cum P/L'])

# End of file
