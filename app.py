import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from robin_stocks import robinhood as r
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- State for persistent logs ---
if 'trade_logs' not in st.session_state:
    st.session_state.trade_logs = []

# --- Helper: fetch data ---
def fetch_data(symbol, period='2d', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    return df

# --- Helper: compute daily pct change ---
def compute_pct_change(df):
    if df.empty or len(df) < 2:
        return None
    today = df.iloc[-1]
    return float((today['Close'] - today['Open']) / today['Open'] * 100)

# --- Order placement ---
def place_order(symbol, side, amount_usd):
    try:
        # fetch price
        price = float(r.crypto.get_crypto_quote(symbol)['mark_price']) if symbol.endswith('-USD') else float(r.orders.get_latest_price(symbol)[0])
        qty = amount_usd / price
        if side.lower() == 'buy':
            return r.crypto.order_buy_crypto_by_quantity(symbol, qty) if symbol.endswith('-USD') else r.orders.order_buy_fractional_by_quantity(symbol, qty)
        else:
            return r.crypto.order_sell_crypto_by_quantity(symbol, qty) if symbol.endswith('-USD') else r.orders.order_sell_fractional_by_quantity(symbol, qty)
    except Exception as e:
        st.warning(f"Order {side} failed for {symbol}: {e}")
        return None

# --- Authenticate Robinhood ---
RH_USER = st.secrets.get('ROBINHOOD_USERNAME') or os.getenv('ROBINHOOD_USERNAME')
RH_PASS = st.secrets.get('ROBINHOOD_PASSWORD') or os.getenv('ROBINHOOD_PASSWORD')
authenticated = False
if RH_USER and RH_PASS:
    try:
        r.login(RH_USER, RH_PASS)
        authenticated = True
        st.sidebar.success("Robinhood authenticated; live orders enabled.")
    except Exception as e:
        st.sidebar.warning(f"Robinhood login failed: {e}; running in simulation.")
else:
    st.sidebar.info("No Robinhood creds; running in simulation mode.")

# --- Streamlit setup ---
st.set_page_config(page_title="Momentum Rebalancer", layout="wide")
st.title("Stock & Crypto Momentum Rebalancer")

# auto-refresh daily
st_autorefresh(interval=24*60*60*1000, key='daily_refresh')

# --- Sidebar: inputs ---
st.sidebar.header("Universe")
equities = st.sidebar.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG").upper().replace(' ','').split(',')
include_crypto = st.sidebar.checkbox("Include Crypto")
crypto_list = st.sidebar.multiselect("Crypto Tickers", ['BTC-USD','ETH-USD','ADA-USD','SOL-USD']) if include_crypto else []
all_symbols = equities + crypto_list

top_n = st.sidebar.number_input("Number of top tickers to pick", min_value=1, max_value=len(all_symbols) or 1, value=6)
trade_amount = st.sidebar.number_input("USD per position", 10, 10000, 500, step=10)

# --- Controls ---
if st.sidebar.button("► Run Scan & Rebalance"):
    # compute percent changes
    changes = []
    for sym in all_symbols:
        df = fetch_data(sym)
        pct = compute_pct_change(df)
        if pct is None: continue
        changes.append({'symbol': sym, 'pct': pct})
    if not changes:
        st.sidebar.error("No data to compute momentum.")
    else:
        df_chg = pd.DataFrame(changes).sort_values('pct', ascending=False)
        top_syms = df_chg.head(top_n)['symbol'].tolist()
        logs = []
        # full rebalance: sell all and buy top
        for sym in all_symbols:
            side = 'BUY' if sym in top_syms else 'SELL'
            # execute
            if authenticated:
                place_order(sym, side, trade_amount)
            # log
            logs.append({
                'Ticker': sym,
                'Action': side,
                'PctChange': round(next(x['pct'] for x in changes if x['symbol']==sym),2),
                'Time': datetime.now()
            })
        st.session_state.trade_logs = logs

if st.sidebar.button("Clear History"):
    st.session_state.trade_logs = []

# --- Display logs ---
if st.session_state.trade_logs:
    df_logs = pd.DataFrame(st.session_state.trade_logs)
    st.subheader("Rebalance Log")
    st.dataframe(df_logs)
    csv = df_logs.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Logs CSV", data=csv, file_name="momentum_logs.csv")
else:
    st.info("No history. Run a scan to rebalance today.")
