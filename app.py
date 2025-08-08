import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from robin_stocks import robinhood
from pycoingecko import CoinGeckoAPI
import datetime

# --- Authentication ---
st.sidebar.header("Authentication")
# Retrieve credentials from Streamlit Secrets
try:
    RH_USER = st.secrets["ROBINHOOD_USERNAME"]
    RH_PW   = st.secrets["ROBINHOOD_PASSWORD"]
except KeyError:
    st.sidebar.error(
        "Missing secret keys ROBINHOOD_USERNAME and/or ROBINHOOD_PASSWORD in Streamlit Cloud settings."
    )
    st.stop()

# Log in to Robinhood
try:
    robinhood.login(username=RH_USER, password=RH_PW)
    st.sidebar.success("Robinhood authenticated — Live orders ENABLED")
except Exception as e:
    st.sidebar.error(f"Robinhood login failed: {e}")
    st.stop()

# --- Sidebar: Universe & Allocation ---
st.sidebar.header("Universe & Allocation")
# Equity input
equity_input = st.sidebar.text_area(
    "Equity Tickers (comma-separated)",
    value="AAPL,MSFT,GOOG"
)
equities = [t.strip().upper() for t in equity_input.split(',') if t.strip()]

# Crypto universe fetch
cg = CoinGeckoAPI()
include_crypto = st.sidebar.checkbox("Include Crypto")
cryptos = []
if include_crypto:
    try:
        markets = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=5,
            page=1
        )
        cryptos = [c['id'].upper() + '-USD' for c in markets]
    except Exception as e:
        st.sidebar.warning(f"Crypto fetch failed: {e}")
        cryptos = []

all_symbols = equities + cryptos

# Allocation override or auto-compute
manual_alloc = st.sidebar.number_input(
    "Manual allocation per position (USD, >0 override)",
    min_value=0.0,
    value=0.0,
    step=1.0
)
if manual_alloc > 0 and all_symbols:
    alloc_per_pos = manual_alloc
else:
    # Compute from account cash evenly
    try:
        acct = robinhood.load_account_profile()
        cash = float(acct.get('portfolio_cash', 0) or 0)
        alloc_per_pos = cash / len(all_symbols) if all_symbols else 0
    except Exception:
        alloc_per_pos = 0

st.sidebar.markdown(f"**Allocation per position:** ${alloc_per_pos:.2f}")

# How many to pick
top_n = st.sidebar.number_input(
    "Number of tickers to pick", 
    min_value=1, 
    max_value=len(all_symbols) if all_symbols else 1, 
    value=min(3, len(all_symbols)), 
    step=1
)

# --- Main App ---
st.title("Stock & Crypto Momentum Rebalancer")

if st.sidebar.button("► Run Daily Scan & Rebalance"):
    # 1) Compute daily momentum
    pct_changes = {}
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)
    for sym in all_symbols:
        symbol_id = sym.replace('-USD', '').lower()
        try:
            if sym.endswith('-USD'):
                # Crypto
                prices = cg.get_coin_market_chart_range_by_id(
                    id=symbol_id,
                    vs_currency='usd',
                    from_timestamp=int(yesterday.timestamp()),
                    to_timestamp=int(now.timestamp())
                )
                price_now = prices['prices'][-1][1]
                price_prev = prices['prices'][0][1]
            else:
                # Stock
                df = yf.download(
                    sym,
                    start=yesterday.strftime('%Y-%m-%d'),
                    end=now.strftime('%Y-%m-%d')
                )
                price_now = df['Close'].iloc[-1]
                price_prev = df['Close'].iloc[0]
            pct_changes[sym] = (price_now / price_prev - 1) * 100
        except Exception:
            pct_changes[sym] = np.nan
            st.warning(f"Failed to retrieve price for {sym}")

    # 2) Build ranking frame
    df_mom = pd.DataFrame(
        [{'Ticker': s, 'PctChange': pct_changes.get(s)} for s in all_symbols]
    )
    df_mom['PctChange'] = pd.to_numeric(df_mom['PctChange'], errors='coerce')
    df_mom = df_mom.dropna(subset=['PctChange'])

    # 3) Select top performers
    picks = df_mom.nlargest(top_n, 'PctChange')['Ticker'].tolist()

    # 4) Fetch existing holdings
    holdings = {}
    try:
        for p in robinhood.get_open_stock_positions():
            holdings[p['symbol'].upper()] = float(p['quantity'])
    except:
        pass
    if include_crypto:
        try:
            for p in robinhood.get_crypto_positions():
                holdings[p['currency'].upper()] = float(p['quantity'])
        except:
            pass

    # 5) Place orders
    log = []
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    for sym in all_symbols:
        current_qty = holdings.get(sym.replace('-USD',''), 0)
        buy = sym in picks and current_qty == 0
        sell = sym not in picks and current_qty > 0
        if not (buy or sell):
            continue
        action = 'BUY' if buy else 'SELL'
        try:
            # Determine execution
            if sym.endswith('-USD'):
                asset = sym.replace('-USD','').lower()
                price = cg.get_price(ids=asset, vs_currencies='usd')[asset]['usd']
                qty = alloc_per_pos / price if price else 0
                if action == 'BUY':
                    robinhood.order_buy_crypto_by_quantity(asset, qty)
                else:
                    robinhood.order_sell_crypto_by_quantity(asset, current_qty)
                executed = qty if action=='BUY' else current_qty
            else:
                price = float(yf.Ticker(sym).info.get('regularMarketPrice', 0) or 0)
                qty = alloc_per_pos / price if price else 0
                if action == 'BUY':
                    robinhood.order_buy_market(sym, qty)
                else:
                    robinhood.order_sell_market(sym, qty)
                executed = qty
            log.append({
                'Ticker': sym,
                'Action': action,
                'PctChange': pct_changes[sym],
                'Executed': executed,
                'Time': now_str
            })
        except Exception as e:
            st.warning(f"Order {action} {sym} failed: {e}")

    # 6) Display log
    if log:
        df_log = pd.DataFrame(log)
        st.subheader("Rebalance Log")
        st.table(df_log)
        st.download_button(
            "Download Logs CSV", df_log.to_csv(index=False), file_name='trade_logs.csv'
        )
