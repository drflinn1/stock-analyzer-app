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

# --- Instantiate CoinGecko client ---
cg = CoinGeckoAPI()
# mapping from symbol to CoinGecko id for orders
crypto_ids = {}

# --- Helper to fetch percent change for equities ---
def fetch_pct_change_stock(symbol, period='2d', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    if len(df) < 2:
        return None
    return float((df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0] * 100)

# --- Place live or simulated orders ---
def place_order(symbol, side, usd_amount):
    try:
        # Crypto orders
        if symbol.endswith('-USD'):
            coin_id = crypto_ids.get(symbol)
            if not coin_id:
                raise Exception(f"No CoinGecko ID for {symbol}")
            price_data = cg.get_price(ids=[coin_id], vs_currencies='usd')
            price = float(price_data.get(coin_id, {}).get('usd', 0))
            if price <= 0:
                raise Exception(f"Failed to retrieve crypto price for {symbol}")
            # live or simulated crypto order by USD amount
            qty = usd_amount / price
            if authenticated:
                if side == 'BUY':
                    return r.crypto.order_buy_crypto_by_price(coin_id, usd_amount)
                return r.crypto.order_sell_crypto_by_price(coin_id, usd_amount)
            # simulation fallback
            return {'symbol': symbol, 'side': side, 'usd': usd_amount, 'price': price}

        # Equity orders
        price_data = r.orders.get_latest_price(symbol)
        price = float(price_data[0]) if price_data else 0
        if price <= 0:
            raise Exception(f"Failed to retrieve equity price for {symbol}")
        qty = usd_amount / price
        if authenticated:
            if side == 'BUY':
                return r.orders.order_buy_fractional_by_quantity(symbol, qty)
            return r.orders.order_sell_fractional_by_quantity(symbol, qty)
        # simulation fallback
        return {'symbol': symbol, 'side': side, 'usd': usd_amount, 'price': price}

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
# equities input
equities_input = st.sidebar.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG")
equities = [s.strip().upper() for s in equities_input.split(',') if s.strip()]

# crypto via CoinGecko dynamic
enable_crypto = st.sidebar.checkbox("Include Crypto")
crypto_coins = []
if enable_crypto:
    try:
        data = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=5, page=1)
        for c in data:
            sym = f"{c['symbol'].upper()}-USD"
            crypto_ids[sym] = c['id']
            crypto_coins.append({'symbol': sym, 'pct': float(c.get('price_change_percentage_24h') or 0)})
    except Exception as e:
        st.sidebar.warning(f"Failed to fetch crypto universe: {e}")

# combined universe list
tickers = equities + [c['symbol'] for c in crypto_coins]
max_syms = max(1, len(tickers))
default_n = min(3, max_syms)

# number of top tickers
top_n = st.sidebar.number_input(
    "Number of top tickers to pick", min_value=1,
    max_value=max_syms, value=default_n, step=1
)

# capital allocation
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
        momentum = []
        # equities momentum
        for sym in equities:
            pct = fetch_pct_change_stock(sym)
            if pct is not None:
                momentum.append({'symbol': sym, 'pct': pct})
        # crypto momentum
        for entry in crypto_coins:
            momentum.append({'symbol': entry['symbol'], 'pct': entry['pct']})

        if not momentum:
            st.sidebar.error("No data returned for selected symbols.")
        else:
            df_mom = pd.DataFrame(momentum)
            df_mom.sort_values('pct', ascending=False, inplace=True)
            df_mom.reset_index(drop=True, inplace=True)
            picks = df_mom.loc[:top_n-1, 'symbol'].tolist()

            logs = []
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for _, row in df_mom.iterrows():
                action = 'BUY' if row['symbol'] in picks else 'SELL'
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
