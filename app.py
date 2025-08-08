import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from robin_stocks import robinhood
from pycoingecko import CoinGeckoAPI
import datetime

# Authenticate to Robinhood
try:
    RH_USER = st.secrets.get('RH_USERNAME')
    RH_PW = st.secrets.get('RH_PASSWORD')
    if not RH_USER or not RH_PW:
        raise KeyError('RH_USERNAME or RH_PASSWORD not set in secrets')
    robinhood.login(username=RH_USER, password=RH_PW)
    st.sidebar.success("Robinhood authenticated — Live orders ENABLED")
except KeyError as ke:
    st.sidebar.error(f"Missing secret: {ke}")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Robinhood login failed: {e}")
    st.stop()

# Sidebar: Universe & Allocation
st.sidebar.header("Universe & Allocation")
# Equity tickers input
equity_input = st.sidebar.text_area(
    "Equity Tickers (comma-separated)",
    value="AAPL,MSFT,GOOG"
)
equities = [t.strip().upper() for t in equity_input.split(',') if t.strip()]

# Crypto toggle and dynamic universe fetch
cg = CoinGeckoAPI()
include_crypto = st.sidebar.checkbox("Include Crypto")
if include_crypto:
    coin_list = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=50, page=1)
    cryptos = [c['symbol'].upper() + '-USD' for c in coin_list[:5]]
else:
    cryptos = []

all_symbols = equities + cryptos

# Allocation override or computed
manual_alloc = st.sidebar.number_input(
    "Manual allocation per position (USD, >0 override)",
    min_value=0.0, value=10.0, step=1.0
)
if manual_alloc > 0:
    alloc_per_pos = manual_alloc
else:
    total_cash = float(robinhood.load_account_profile()['portfolio_cash'])
    num = len(all_symbols)
    alloc_per_pos = total_cash / num if num else 0

st.sidebar.markdown(f"Allocation per position: ${alloc_per_pos:.2f}")

# Number to pick
top_n = st.sidebar.number_input(
    "Number of tickers to pick", min_value=1, max_value=len(all_symbols), value=3, step=1
)

# Main: Title
st.title("Stock & Crypto Momentum Rebalancer")

# Run button
if st.sidebar.button("► Run Daily Scan & Rebalance"):
    # Fetch prices and compute percent changes
    pct_changes = {}
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)
    for sym in all_symbols:
        if sym.endswith('-USD'):
            try:
                coin_id = sym.replace('-USD','').lower()
                price_now = cg.get_price(ids=coin_id, vs_currencies='usd')[coin_id]['usd']
                hist = cg.get_coin_market_chart_range_by_id(
                    id=coin_id, vs_currency='usd',
                    from_timestamp=int(yesterday.timestamp()),
                    to_timestamp=int(now.timestamp())
                )
                price_y = hist['prices'][0][1]
                pct_changes[sym] = (price_now/price_y - 1) * 100
            except Exception:
                st.warning(f"Failed to retrieve crypto price for {sym}")
                pct_changes[sym] = None
        else:
            try:
                df = yf.download(sym, start=yesterday.strftime('%Y-%m-%d'), end=now.strftime('%Y-%m-%d'))
                price_now = df['Close'].iloc[-1]
                price_y = df['Close'].iloc[0]
                pct_changes[sym] = (price_now/price_y - 1) * 100
            except Exception:
                st.warning(f"Failed to retrieve stock price for {sym}")
                pct_changes[sym] = None

    df_mom = pd.DataFrame([{'Ticker': sym, 'PctChange': pct_changes[sym]} for sym in all_symbols])
    df_mom = df_mom.dropna().sort_values('PctChange', ascending=False)
    picks = df_mom.head(top_n)['Ticker'].tolist()

    try:
        positions = robinhood.get_open_stock_positions()
        holdings = {h['symbol'].upper(): float(h['quantity']) for h in positions}
    except Exception:
        holdings = {}

    log = []
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for sym in all_symbols:
        action = None
        current_qty = holdings.get(sym, 0)
        if sym in picks and current_qty == 0:
            action = 'BUY'
        elif sym not in picks and current_qty > 0:
            action = 'SELL'

        if action:
            try:
                if sym.endswith('-USD'):
                    amount = alloc_per_pos
                    asset = sym.replace('-USD','').lower()
                    if action == 'BUY':
                        robinhood.order_buy_crypto_by_price(asset, amount)
                    else:
                        robinhood.order_sell_crypto_by_price(asset, amount)
                    executed_qty = amount
                else:
                    price = float(yf.Ticker(sym).info['regularMarketPrice'])
                    qty = alloc_per_pos / price
                    if action == 'BUY':
                        robinhood.order_buy_market(sym, qty)
                    else:
                        robinhood.order_sell_market(sym, qty)
                    executed_qty = qty
                log.append({'Ticker': sym, 'Action': action, 'PctChange': pct_changes[sym], 'Time': timestamp})
            except Exception as e:
                st.warning(f"Order {action} {sym} failed: {e}")

    df_log = pd.DataFrame(log)
    st.subheader("Rebalance Log")
    st.table(df_log)
    st.download_button("Download Logs CSV", df_log.to_csv(index=False), file_name='trade_logs.csv')
