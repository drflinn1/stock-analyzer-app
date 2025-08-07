import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from robin_stocks import robinhood as r
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- Persistent trade logs ---
if 'trade_logs' not in st.session_state:
    st.session_state.trade_logs = []

# --- Helpers ---
def fetch_data(symbol, period='2d', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    return df

def compute_pct_change(df):
    if df.empty or len(df) < 2:
        return None
    first_open = df['Open'].iloc[0]
    last_close = df['Close'].iloc[-1]
    return float((last_close - first_open) / first_open * 100)

# --- Order placement (live via Robinhood or simulation) ---
def place_order(symbol, side, amount_usd):
    try:
        price = None
        if symbol.endswith('-USD'):
            quote = r.crypto.get_crypto_quote(symbol)
            price = float(quote.get('mark_price', 0))
        else:
            price = float(r.orders.get_latest_price(symbol)[0])
        qty = amount_usd / price if price else 0
        if side.lower() == 'buy':
            if symbol.endswith('-USD'):
                return r.crypto.order_buy_crypto_by_quantity(symbol, qty)
            return r.orders.order_buy_fractional_by_quantity(symbol, qty)
        else:
            if symbol.endswith('-USD'):
                return r.crypto.order_sell_crypto_by_quantity(symbol, qty)
            return r.orders.order_sell_fractional_by_quantity(symbol, qty)
    except Exception as e:
        st.warning(f"{side.title()} order failed for {symbol}: {e}")
        return None

# --- Robinhood authentication ---
RH_USER = st.secrets.get('ROBINHOOD_USERNAME') or os.getenv('ROBINHOOD_USERNAME')
RH_PASS = st.secrets.get('ROBINHOOD_PASSWORD') or os.getenv('ROBINHOOD_PASSWORD')
authenticated = False
if RH_USER and RH_PASS:
    try:
        r.login(RH_USER, RH_PASS)
        authenticated = True
        st.sidebar.success("Robinhood authenticated – live orders enabled.")
    except Exception:
        st.sidebar.warning("Robinhood login failed – running in simulation.")
else:
    st.sidebar.info("No Robinhood credentials – simulation mode.")

# --- Streamlit page setup ---
st.set_page_config(page_title="Momentum Rebalancer", layout="wide")
st.title("Stock & Crypto Momentum Rebalancer")

# auto-refresh daily
st_autorefresh(interval=24*60*60*1000, key='daily_refresh')

# --- Sidebar inputs ---
st.sidebar.header("Universe")
equities = st.sidebar.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG").upper().replace(' ', '').split(',')
include_crypto = st.sidebar.checkbox("Include Crypto")
crypto_list = []
if include_crypto:
    crypto_list = st.sidebar.multiselect("Crypto Tickers", ['BTC-USD','ETH-USD','ADA-USD','SOL-USD'], default=['BTC-USD','ETH-USD'])
all_symbols = [s for s in equities + crypto_list if s]

# dynamic number of picks\max_syms = max(len(all_symbols), 1)
top_n = st.sidebar.number_input("Number of top tickers to pick", min_value=1, max_value=max_syms, value=min(3, max_syms))
trade_amount = st.sidebar.number_input("USD per position", min_value=1, max_value=100000, value=100)

# --- Actions ---
if st.sidebar.button("► Run Daily Scan & Rebalance"):
    changes = []
    for sym in all_symbols:
        df = fetch_data(sym)
        pct = compute_pct_change(df)
        if pct is not None:
            changes.append({'symbol': sym, 'pct': pct})
    if not changes:
        st.sidebar.error("No data available for momentum calculation.")
    else:
        df_chg = pd.DataFrame(changes).sort_values('pct', ascending=False)
        top_syms = df_chg.head(top_n)['symbol'].tolist()
        new_logs = []
        # full rebalance: sell all then buy top picks
        for sym in all_symbols:
            action = 'SELL' if sym not in top_syms else 'BUY'
            if authenticated:
                place_order(sym, action, trade_amount)
            new_logs.append({
                'Ticker': sym,
                'Action': action,
                'PctChange': round(next(x['pct'] for x in changes if x['symbol']==sym), 2),
                'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        st.session_state.trade_logs.extend(new_logs)

if st.sidebar.button("Clear History"):
    st.session_state.trade_logs = []

# --- Display logs & download ---
if st.session_state.trade_logs:
    df_logs = pd.DataFrame(st.session_state.trade_logs)
    st.subheader("Rebalance Log")
    st.dataframe(df_logs)
    csv_data = df_logs.to_csv(index=False).encode('utf-8')
    st.download_button("Download Logs CSV", data=csv_data, file_name="momentum_logs.csv")
else:
    st.info("No history yet. Click 'Run Daily Scan & Rebalance' to execute.")
