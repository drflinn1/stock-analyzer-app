```python
import os
import pandas as pd
import streamlit as st
import yfinance as yf
from robin_stocks import robinhood as r
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- Session state for logs ---
if 'trade_logs' not in st.session_state:
    st.session_state.trade_logs = []

# --- Helper to fetch percent change ---
def fetch_pct_change(symbol, period='2d', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    if len(df) < 2:
        return None
    return (df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0] * 100

# --- Place live or simulated orders ---
def place_order(symbol, side, usd_amount):
    try:
        # Crypto vs Equity check by ticker suffix
        if symbol.endswith('-USD'):
            quote = r.crypto.get_crypto_quote(symbol)
            price = float(quote.get('mark_price', 0))
            qty = usd_amount / price if price else 0
            if side == 'BUY':
                return r.crypto.order_buy_crypto_by_quantity(symbol, qty)
            return r.crypto.order_sell_crypto_by_quantity(symbol, qty)

        # equity
        price_data = r.orders.get_latest_price(symbol)
        price = float(price_data[0]) if price_data else 0
        qty = usd_amount / price if price else 0
        if side == 'BUY':
            return r.orders.order_buy_fractional_by_quantity(symbol, qty)
        return r.orders.order_sell_fractional_by_quantity(symbol, qty)

    except Exception as e:
        st.warning(f"Order {side} {symbol} failed: {e}")
        return None

# --- Authentication via Streamlit secrets or env ---
RH_USER = st.secrets.get('ROBINHOOD_USERNAME') or os.getenv('ROBINHOOD_USERNAME')
RH_PASS = st.secrets.get('ROBINHOOD_PASSWORD') or os.getenv('ROBINHOOD_PASSWORD')
authenticated = False
if RH_USER and RH_PASS:
    try:
        r.login(RH_USER, RH_PASS)
        authenticated = True
        st.sidebar.success("Robinhood authenticated; live orders enabled.")
    except Exception:
        st.sidebar.warning("Robinhood login failed; running simulation.")
else:
    st.sidebar.info("No Robinhood credentials; running simulation.")

# --- Page setup ---
st.set_page_config(page_title="Stock & Crypto Momentum Rebalancer", layout="wide")
st.title("Stock & Crypto Momentum Rebalancer")

# Auto-refresh daily
st_autorefresh(interval=24*60*60*1000, key='daily_auto')

# --- Sidebar inputs ---
st.sidebar.header("Universe")
equities = st.sidebar.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG")
equities = [s.strip().upper() for s in equities.split(',') if s.strip()]
include_crypto = st.sidebar.checkbox("Include Crypto")
crypto_list = []
if include_crypto:
    cryptos = st.sidebar.text_area("Crypto Tickers (comma-separated)", "BTC-USD,ETH-USD")
    crypto_list = [s.strip().upper() for s in cryptos.split(',') if s.strip()]

# Combine universes and determine top_n bounds
all_symbols = equities + crypto_list
max_syms = max(1, len(all_symbols))
defaltn = min(3, max_syms)
# Ensure default <= max
```python
top_n = st.sidebar.number_input(
    "Number of top tickers to pick", min_value=1,
    max_value=max_syms, value=defaltn, step=1
)
```python

# Get buying power or simulation capital
aif authenticated:
    profile = r.account.load_account_profile() or {}
    buying_power = float(profile.get('cash', 0))
else:
    capital = st.sidebar.number_input("Total capital for simulation (USD)", min_value=1, value=10000)
    buying_power = float(capital)

allocation = round(buying_power / top_n, 2)
st.sidebar.markdown(f"**Allocation per position:** ${allocation}")

# --- Run daily scan & rebalance ---
if st.sidebar.button("► Run Daily Scan & Rebalance"):
    if not all_symbols:
        st.sidebar.error("Please add at least one ticker to scan.")
    else:
        # compute momentum
        momentum = []
        for sym in all_symbols:
            pct = fetch_pct_change(sym)
            if pct is not None:
                momentum.append({'symbol': sym, 'pct': pct})
        if not momentum:
            st.sidebar.error("No valid data returned for selected symbols.")
        else:
            df_mom = pd.DataFrame(momentum).sort_values('pct', ascending=False)
            picks = df_mom.head(top_n)['symbol'].tolist()
            entries = []
            # execute full rebalance
            for sym in all_symbols:
                action = 'BUY' if sym in picks else 'SELL'
                if authenticated:
                    place_order(sym, action, allocation)
                entries.append({
                    'Ticker': sym,
                    'Action': action,
                    'PctChange': round(next(x['pct'] for x in momentum if x['symbol']==sym), 2),
                    'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            st.session_state.trade_logs.extend(entries)

# --- Clear history ---
if st.sidebar.button("Clear History"):
    st.session_state.trade_logs = []

# --- Display logs & download ---
if st.session_state.trade_logs:
    df_logs = pd.DataFrame(st.session_state.trade_logs)
    st.subheader("Rebalance Log")
    st.dataframe(df_logs)
    csv = df_logs.to_csv(index=False).encode('utf-8')
    st.download_button("Download Logs CSV", csv, file_name="momentum_logs.csv")
else:
    st.info("No history yet. Click 'Run Daily Scan & Rebalance' to execute.")
```
