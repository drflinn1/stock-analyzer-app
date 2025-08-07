```python
import os
import pandas as pd
import streamlit as st
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from robin_stocks import robinhood as r
from datetime import datetime

# --- Session state for logs ---
if 'trade_logs' not in st.session_state:
    st.session_state.trade_logs = []

# --- Helper to fetch percent change ---
def fetch_pct_change(symbol, period='2d', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    if len(df) < 2:
        return None
    # ensure float type
    return float((df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0] * 100)

# --- Place live or simulated orders ---
def place_order(symbol, side, usd_amount):
    try:
        if symbol.endswith('-USD'):
            quote = r.crypto.get_crypto_quote(symbol)
            price = float(quote.get('mark_price', 0))
            qty = usd_amount / price if price else 0
            if side == 'BUY':
                return r.crypto.order_buy_crypto_by_quantity(symbol, qty)
            return r.crypto.order_sell_crypto_by_quantity(symbol, qty)

        # equities
        price_data = r.orders.get_latest_price(symbol)
        price = float(price_data[0]) if price_data else 0
        qty = usd_amount / price if price else 0
        if side == 'BUY':
            return r.orders.order_buy_fractional_by_quantity(symbol, qty)
        return r.orders.order_sell_fractional_by_quantity(symbol, qty)

    except Exception as e:
        st.warning(f"Order {side} {symbol} failed: {e}")
        return None

# --- Authentication ---
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

# --- Sidebar inputs ---
st.sidebar.header("Universe")
# Equities
equities_input = st.sidebar.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG")
equities = [s.strip().upper() for s in equities_input.split(',') if s.strip()]

# Crypto via CoinGecko
enable_crypto = st.sidebar.checkbox("Include Crypto")
crypto_list = []
if enable_crypto:
    cg = CoinGeckoAPI()
    try:
        coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=5, page=1)
        crypto_list = [f"{c['symbol'].upper()}-USD" for c in coins]
    except Exception as e:
        st.sidebar.warning(f"Failed to fetch crypto universe: {e}")

# Combined universe
tickers = equities + crypto_list
max_syms = max(1, len(tickers))
default_n = min(3, max_syms)

# Number to pick
top_n = st.sidebar.number_input(
    "Number of top tickers to pick", min_value=1,
    max_value=max_syms, value=default_n, step=1
)

# Capital allocation
if authenticated:
    profile = r.account.load_account_profile() or {}
    buying_power = float(profile.get('cash', 0))
else:
    capital = st.sidebar.number_input("Total capital for simulation (USD)", min_value=1, value=10000)
    buying_power = float(capital)

allocation = round(buying_power / top_n, 2)
st.sidebar.markdown(f"**Allocation per position:** ${allocation}")

# --- Run daily scan & rebalance ---
if st.sidebar.button("► Run Daily Scan & Rebalance"):
    if not tickers:
        st.sidebar.error("Please specify at least one ticker.")
    else:
        # compute momentum
        momentum = []
        for sym in tickers:
            pct = fetch_pct_change(sym)
            if pct is not None:
                momentum.append({'symbol': sym, 'pct': pct})

        if not momentum:
            st.sidebar.error("No data returned for selected symbols.")
        else:
            # create DataFrame and sort
            df_mom = pd.DataFrame(momentum)
            df_mom = df_mom.sort_values('pct', ascending=False).reset_index(drop=True)
            # pick top
            picks = df_mom.loc[:top_n-1, 'symbol'].tolist()

            # execute orders and log
            logs = []
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for _, row in df_mom.iterrows():
                action = 'BUY' if row['symbol'] in picks else 'SELL'
                if authenticated:
                    place_order(row['symbol'], action, allocation)
                logs.append({
                    'Ticker': row['symbol'],
                    'Action': action,
                    'PctChange': round(row['pct'], 2),
                    'Time': now
                })
            st.session_state.trade_logs.extend(logs)

# --- History controls ---
if st.sidebar.button("Clear History"):
    st.session_state.trade_logs = []

# --- Display & download ---
if st.session_state.trade_logs:
    df_logs = pd.DataFrame(st.session_state.trade_logs)
    st.subheader("Rebalance Log")
    st.dataframe(df_logs)
    csv = df_logs.to_csv(index=False).encode()
    st.download_button("Download Logs CSV", csv, file_name="momentum_logs.csv")
else:
    st.info("No history yet. Click 'Run Daily Scan & Rebalance' to execute.")
```
