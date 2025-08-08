import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from robin_stocks import robinhood
from pycoingecko import CoinGeckoAPI
import datetime

# --- Authentication ---
st.sidebar.header("Authentication")
# Load credentials from secrets or prompt user
RH_USER = st.secrets.get("ROBINHOOD_USERNAME") or st.sidebar.text_input("Robinhood Username", type="default")
RH_PW   = st.secrets.get("ROBINHOOD_PASSWORD") or st.sidebar.text_input("Robinhood Password", type="password")

# Ensure credentials provided
if not RH_USER or not RH_PW:
    st.sidebar.error("Robinhood credentials required to proceed.")
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
equity_input = st.sidebar.text_area("Equity Tickers (comma-separated)", value="AAPL,MSFT,GOOG")
equities = [t.strip().upper() for t in equity_input.split(',') if t.strip()]

# Crypto universe fetch
cg = CoinGeckoAPI()
include_crypto = st.sidebar.checkbox("Include Crypto")
if include_crypto:
    coin_list = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=50, page=1)
    cryptos = [c['symbol'].upper() + '-USD' for c in coin_list[:5]]
else:
    cryptos = []

all_symbols = equities + cryptos

# Allocation override or auto-compute
manual_alloc = st.sidebar.number_input("Manual allocation per position (USD, >0 override)", min_value=0.0, value=0.0, step=1.0)
if manual_alloc > 0:
    alloc_per_pos = manual_alloc
else:
    acct = robinhood.load_account_profile()
    cash_str = acct.get('portfolio_cash', '0')
    total_cash = float(cash_str or 0)
    alloc_per_pos = total_cash / len(all_symbols) if all_symbols else 0

st.sidebar.markdown(f"**Allocation per position:** ${alloc_per_pos:.2f}")

# How many to pick
top_n = st.sidebar.number_input("Number of tickers to pick", min_value=1, max_value=len(all_symbols) or 1, value=3, step=1)

# --- Main App ---
st.title("Stock & Crypto Momentum Rebalancer")

if st.sidebar.button("► Run Daily Scan & Rebalance"):
    # 1) Compute daily momentum
    pct_changes = {}
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)
    for sym in all_symbols:
        if sym.endswith('-USD'):
            # Crypto
            try:
                cid = sym.replace('-USD','').lower()
                price_now = float(cg.get_price(ids=cid, vs_currencies='usd')[cid]['usd'])
                hist = cg.get_coin_market_chart_range_by_id(
                    id=cid, vs_currency='usd',
                    from_timestamp=int(yesterday.timestamp()),
                    to_timestamp=int(now.timestamp())
                )
                price_y = float(hist['prices'][0][1])
                pct_changes[sym] = (price_now / price_y - 1) * 100
            except Exception:
                st.warning(f"Failed to retrieve crypto price for {sym}")
                pct_changes[sym] = np.nan
        else:
            # Stock
            try:
                df = yf.download(sym,
                                 start=yesterday.strftime('%Y-%m-%d'),
                                 end=now.strftime('%Y-%m-%d'))
                price_now = float(df['Close'].iloc[-1])
                price_y = float(df['Close'].iloc[0])
                pct_changes[sym] = (price_now / price_y - 1) * 100
            except Exception:
                st.warning(f"Failed to retrieve stock price for {sym}")
                pct_changes[sym] = np.nan

    # 2) Build DataFrame and ensure numeric type
    df_mom = pd.DataFrame([{'Ticker': s, 'PctChange': pct_changes.get(s)} for s in all_symbols])
    df_mom['PctChange'] = pd.to_numeric(df_mom['PctChange'], errors='coerce')
    df_mom = df_mom.dropna(subset=['PctChange'])

    # 3) Rank and pick
    picks = df_mom.nlargest(top_n, 'PctChange')['Ticker'].tolist()

    # 4) Fetch stock holdings
    try:
        stock_positions = robinhood.get_open_stock_positions()
        holdings = {p['symbol'].upper(): float(p['quantity']) for p in stock_positions}
    except Exception:
        holdings = {}
    # 5) Fetch crypto holdings (if included)
    if include_crypto:
        try:
            crypto_positions = robinhood.get_crypto_positions()
            for p in crypto_positions:
                sym_name = p['currency'].upper() + '-USD'
                holdings[sym_name] = float(p['quantity'])
        except Exception:
            pass

    # 6) Place orders and log
    log = []
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    for sym in all_symbols:
        current_qty = holdings.get(sym, 0)
        buy_cond  = sym in picks and current_qty == 0
        sell_cond = sym not in picks and current_qty > 0
        if not (buy_cond or sell_cond):
            continue
        action = 'BUY' if buy_cond else 'SELL'
        try:
            if sym.endswith('-USD'):
                asset = sym.replace('-USD','').lower()
                if action == 'BUY':
                    # Try price-based order, fallback to quantity
                    try:
                        robinhood.order_buy_crypto_by_price(asset, alloc_per_pos)
                        executed = alloc_per_pos
                    except AttributeError:
                        price_now = float(cg.get_price(ids=asset, vs_currencies='usd')[asset]['usd'])
                        qty = alloc_per_pos / price_now if price_now else 0
                        robinhood.order_buy_crypto_by_quantity(asset, qty)
                        executed = qty
                else:
                    # Sell entire position
                    qty_to_sell = current_qty
                    try:
                        robinhood.order_sell_crypto_by_price(asset, alloc_per_pos)
                        executed = alloc_per_pos
                    except AttributeError:
                        robinhood.order_sell_crypto_by_quantity(asset, qty_to_sell)
                        executed = qty_to_sell
            else:
                price = float(yf.Ticker(sym).info.get('regularMarketPrice', 0) or 0)
                qty = alloc_per_pos / price if price else 0
                if action == 'BUY':
                    robinhood.order_buy_market(sym, qty)
                else:
                    robinhood.order_sell_market(sym, qty)
                executed = qty
            log.append({'Ticker': sym, 'Action': action, 'PctChange': pct_changes[sym], 'Executed': executed, 'Time': now_str})
        except Exception as e:
            st.warning(f"Order {action} {sym} failed: {e}")

    # 7) Show log
    if log:
        df_log = pd.DataFrame(log)
        st.subheader("Rebalance Log")
        st.table(df_log)
        st.download_button("Download Logs CSV", df_log.to_csv(index=False), file_name='trade_logs.csv')
