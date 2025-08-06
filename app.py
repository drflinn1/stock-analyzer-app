import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
# Use explicit robinhood submodule for login
from robin_stocks import robinhood as r
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# --- Helper: Fetch Equity & Crypto Data ---
def fetch_data(symbol: str, period: str = "30d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    return df

# --- Helper: Compute Indicators ---
def compute_rsi(series: pd.Series, window: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=(window-1), adjust=False).mean()
    ma_down = down.ewm(com=(window-1), adjust=False).mean()
    rs = ma_up / ma_down
    return float(100 - (100 / (1 + rs)).iloc[-1])

def compute_bbands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> dict:
    sma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    return {
        'upper': float(sma.iloc[-1] + num_std * std.iloc[-1]),
        'lower': float(sma.iloc[-1] - num_std * std.iloc[-1]),
        'sma': float(sma.iloc[-1])
    }

# --- Helper: Generate Signal ---
def analyze_signal(df: pd.DataFrame, overbought: int, oversold: int) -> dict:
    rsi = compute_rsi(df['Close'])
    bb = compute_bbands(df)
    price = float(df['Close'].iloc[-1])
    action = 'HOLD'
    if rsi > overbought and price >= bb['upper']:
        action = 'SELL'
    elif rsi < oversold and price <= bb['lower']:
        action = 'BUY'
    return {'action': action, 'rsi': round(rsi,1), 'bb_upper': round(bb['upper'],2), 'bb_lower': round(bb['lower'],2)}

# --- Helper: Place Orders ---
def place_order(symbol: str, side: str, amount_usd: float):
    try:
        if symbol.endswith('-USD'):
            price = float(r.crypto.get_crypto_quote(symbol)['mark_price'])
        else:
            price = float(r.orders.get_latest_price(symbol)[0])
        qty = amount_usd / price
        if side.lower() == 'buy':
            if symbol.endswith('-USD'):
                return r.crypto.order_buy_crypto_by_quantity(symbol, qty)
            else:
                return r.orders.order_buy_fractional_by_quantity(symbol, qty)
        else:
            if symbol.endswith('-USD'):
                return r.crypto.order_sell_crypto_by_quantity(symbol, qty)
            else:
                return r.orders.order_sell_fractional_by_quantity(symbol, qty)
    except Exception as e:
        st.warning(f"Order placement failed for {symbol}: {e}")
        return None

# --- Robinhood Authentication (simulated if missing) ---
RH_USER = st.secrets.get('ROBINHOOD_USERNAME') or os.getenv('ROBINHOOD_USERNAME')
RH_PASS = st.secrets.get('ROBINHOOD_PASSWORD') or os.getenv('ROBINHOOD_PASSWORD')
authenticated = False
if RH_USER and RH_PASS:
    try:
        # Use standard login; skip 2FA interaction in Cloud
        r.login(RH_USER, RH_PASS)
        authenticated = True
        st.sidebar.success("Robinhood authenticated; live orders enabled.")
    except Exception as e:
        st.sidebar.warning(f"Robinhood login failed: {e}; orders will run in simulation.")
else:
    st.sidebar.info("Robinhood credentials missing; running in simulation mode.")

# --- Streamlit App Setup ---
st.set_page_config(page_title="Equity & Crypto Analyzer", layout="wide")
st.title("Stock & Crypto Analyzer Bot")

# Auto-refresh once per day
st_autorefresh(interval=24*60*60*1000, key='daily_refresh')

# Sidebar Inputs
st.sidebar.header("Universe")
equities = st.sidebar.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG").upper().replace(' ','').split(',')
include_crypto = st.sidebar.checkbox("Include Crypto")
crypto_list = (st.sidebar.multiselect("Crypto Tickers", ['BTC-USD','ETH-USD','ADA-USD','SOL-USD']) if include_crypto else [])

st.sidebar.header("Signal Parameters")
overbought = st.sidebar.slider("RSI Overbought", 50, 90, 70)
oversold = st.sidebar.slider("RSI Oversold", 10, 50, 30)
trade_amount = st.sidebar.number_input("USD per Trade", 10, 10000, 500, step=10)

# Main Loop
if st.button("► Run Scan & Execute"):
    logs = []
    # Equities
    for sym in equities:
        df = fetch_data(sym, "30d", "1d")
        if df.empty: continue
        sig = analyze_signal(df, overbought, oversold)
        price = float(df['Close'].iloc[-1])
        logs.append({
            'Ticker': sym,
            'Signal': sig['action'],
            'Price': price,
            'Time': df.index[-1],
            'RSI': sig['rsi'],
            'BB_Upper': sig['bb_upper'],
            'BB_Lower': sig['bb_lower']
        })
        if authenticated and sig['action'] in ['BUY', 'SELL']:
            place_order(sym, sig['action'], trade_amount)
    # Crypto
    if include_crypto:
        for sym in crypto_list:
            dfc = fetch_data(sym, "7d", "1h")
            if dfc.empty: continue
            sigc = analyze_signal(dfc, overbought, oversold)
            pricec = float(dfc['Close'].iloc[-1])
            logs.append({
                'Ticker': sym,
                'Signal': sigc['action'],
                'Price': pricec,
                'Time': dfc.index[-1],
                'RSI': sigc['rsi'],
                'BB_Upper': sigc['bb_upper'],
                'BB_Lower': sigc['bb_lower']
            })
            if authenticated and sigc['action'] in ['BUY', 'SELL']:
                place_order(sym, sigc['action'], trade_amount)
    # Show Log
    df_logs = pd.DataFrame(logs)
    st.subheader("Trade Signals & Execution Log")
    st.dataframe(df_logs)
