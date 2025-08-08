import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from robin_stocks import robinhood
from robin_stocks.robinhood import orders as rh_orders
from robin_stocks.robinhood import crypto as rh_crypto
from pycoingecko import CoinGeckoAPI
import datetime

# --- Authentication & Mode ---
st.sidebar.header("Authentication & Mode")
live_mode = st.sidebar.checkbox("Enable Live Trading (use with caution)", value=False)
if live_mode:
    try:
        user = st.secrets["ROBINHOOD_USERNAME"]
        pw = st.secrets["ROBINHOOD_PASSWORD"]
    except KeyError:
        st.sidebar.error("Missing ROBINHOOD_USERNAME or ROBINHOOD_PASSWORD in secrets.")
        st.stop()
    try:
        robinhood.login(username=user, password=pw)
        st.sidebar.success("Robinhood authenticated — Live orders ENABLED")
    except Exception as e:
        st.sidebar.error(f"Robinhood login failed: {e}")
        st.stop()
else:
    st.sidebar.info("Simulation mode — no live orders placed.")

# --- Sidebar: Universe & Allocation ---
st.sidebar.header("Universe & Allocation")
equity_input = st.sidebar.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG")
equities = [t.strip().upper() for t in equity_input.split(',') if t.strip()]
include_crypto = st.sidebar.checkbox("Include Crypto", value=False)
cryptos = []
symbol_to_id = {}
if include_crypto:
    cg = CoinGeckoAPI()
    markets = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=5)
    for c in markets:
        symbol = c['symbol'].upper()
        pair = f"{symbol}-USD"
        symbol_to_id[symbol] = c['id']
        cryptos.append(pair)
all_symbols = equities + cryptos

manual = st.sidebar.number_input(
    "Manual allocation per position (USD, >0 override)",
    min_value=0.0, value=0.0, step=0.01, format="%.2f"
)

# Determine allocation
if manual > 0:
    alloc = manual
elif live_mode and all_symbols:
    acct = robinhood.load_account_profile()
    cash = float(acct.get('portfolio_cash') or 0)
    alloc = cash / len(all_symbols)
else:
    alloc = 0.0

st.sidebar.markdown(f"**Allocation per position:** ${alloc:.2f}")

# Number of picks
raw_n = st.sidebar.number_input(
    "Number of tickers to pick", min_value=1, max_value=len(all_symbols) or 1,
    value=min(3, len(all_symbols)), step=1
)
top_n = int(raw_n)

st.title("Stock & Crypto Momentum Rebalancer")

if st.sidebar.button("► Run Daily Scan & Rebalance"):
    now = datetime.datetime.now()
    prev = now - datetime.timedelta(days=1)

    # Compute momentum
    pct = {}
    for sym in all_symbols:
        try:
            if sym.endswith('-USD'):
                cid = symbol_to_id.get(sym.replace('-USD',''))
                data = cg.get_coin_market_chart_range_by_id(cid, 'usd', int(prev.timestamp()), int(now.timestamp()))
                p0, p1 = data['prices'][0][1], data['prices'][-1][1]
            else:
                df = yf.download(sym, start=prev.strftime('%Y-%m-%d'), end=(now + datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
                p0, p1 = df['Close'].iloc[0], df['Close'].iloc[-1]
            pct[sym] = (p1 / p0 - 1) * 100
        except Exception:
            pct[sym] = np.nan

    df = pd.DataFrame([{'Ticker': s, 'PctChange': pct[s]} for s in all_symbols])
    # Ensure numeric dtype before ranking to avoid pandas ValueError
    df['PctChange'] = pd.to_numeric(df['PctChange'], errors='coerce')
    df = df.dropna(subset=['PctChange'])

    # Use sort_values+head (robust across pandas versions)
    picks = []
    if not df.empty:
        picks = df.sort_values('PctChange', ascending=False).head(top_n)['Ticker'].tolist()

    # Load current holdings
    holdings = {}
    if live_mode:
        for pos in rh_orders.get_open_stock_positions() or []:
            holdings[pos['symbol'].upper()] = float(pos['quantity'])
        if include_crypto:
            for pos in rh_crypto.get_crypto_positions() or []:
                holdings[pos['currency'].upper()] = float(pos['quantity'])

    # Perform trades and log
    log = []
    for sym in all_symbols:
        base = sym.replace('-USD','')
        qty = holdings.get(base, 0)
        buy = sym in picks and qty == 0
        sell = sym not in picks and qty > 0
        if not (buy or sell):
            continue
        action = 'BUY' if buy else 'SELL'
        executed, oid = 0, ''
        status = 'simulated' if not live_mode else 'pending'
        if live_mode:
            try:
                price = alloc
                if sym.endswith('-USD'):
                    resp = rh_crypto.order_buy_crypto_by_price(base, price) if buy else rh_crypto.order_sell_crypto_by_price(base, price)
                else:
                    resp = rh_orders.order_buy_fractional_by_price(sym, price) if buy else rh_orders.order_sell_fractional_by_price(sym, price)
                oid = resp.get('id', '')
                executed = float(resp.get('cumulative_quantity') or resp.get('executed_quantity') or 0)
                status = resp.get('state') or status
            except Exception as e:
                status = f"failed: {e}"
                st.warning(f"Order {action} {sym} failed: {e}")
        log.append({
            'Ticker': sym,
            'Action': action,
            'PctChange': round(pct.get(sym,0),2),
            'Executed': round(executed,4),
            'OrderID': oid,
            'Status': status,
            'Time': now.strftime('%Y-%m-%d %H:%M:%S')
        })

    df_log = pd.DataFrame(log)
    st.subheader("Rebalance Log")
    st.table(df_log)
    st.download_button("Download Logs CSV", df_log.to_csv(index=False), file_name='trade_logs.csv')

    if live_mode:
        st.subheader("Open Orders")
        try:
            open_stock = rh_orders.get_all_open_orders() or []
            open_crypto = []
            if include_crypto:
                fn = None
                for name in ("get_all_crypto_orders", "get_all_open_crypto_orders", "get_crypto_orders"):
                    fn = getattr(rh_crypto, name, None)
                    if callable(fn):
                        break
                if fn:
                    try:
                        open_crypto = fn() or []
                    except Exception:
                        open_crypto = []
            open_orders = {'stocks': open_stock, 'crypto': open_crypto}
        except Exception as e:
            open_orders = []
            st.warning(f"Failed to fetch open orders: {e}")
        st.write(open_orders)
