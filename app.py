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
symbol_to_id, cryptos = {}, []
if include_crypto:
    cg = CoinGeckoAPI()
    markets = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=5)
    for c in markets:
        sym = c['symbol'].upper()
        symbol_to_id[sym] = c['id']
        cryptos.append(f"{sym}-USD")
all_symbols = equities + cryptos
manual = st.sidebar.number_input("Manual allocation per position (USD, >0 override)", min_value=0.0, value=0.0)
if manual > 0:
    alloc = manual
elif live_mode and all_symbols:
    acct = robinhood.load_account_profile()
    cash = float(acct.get('portfolio_cash',0) or 0)
    alloc = cash / len(all_symbols)
else:
    alloc = 0
st.sidebar.markdown(f"**Allocation per position:** ${alloc:.2f}")
top_n = st.sidebar.number_input("Number of tickers to pick", min_value=1, max_value=len(all_symbols) or 1, value=min(3,len(all_symbols)), step=1)

st.title("Stock & Crypto Momentum Rebalancer")
if st.sidebar.button("► Run Daily Scan & Rebalance"):
    now = datetime.datetime.now()
    prev = now - datetime.timedelta(days=1)
    pct = {}
    for sym in all_symbols:
        try:
            if sym.endswith('-USD'):
                cid = symbol_to_id[sym.replace('-USD','')]
                data = cg.get_coin_market_chart_range_by_id(cid,'usd',int(prev.timestamp()),int(now.timestamp()))
                p0, p1 = data['prices'][0][1], data['prices'][-1][1]
            else:
                df = yf.download(sym,start=prev.strftime('%Y-%m-%d'),end=(now+datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
                p0, p1 = df['Close'].iloc[0], df['Close'].iloc[-1]
            pct[sym] = (p1/p0 -1)*100
        except:
            pct[sym] = np.nan
    df = pd.DataFrame([{'Ticker':s,'PctChange':pct[s]} for s in all_symbols]).dropna()
    picks = df.nlargest(top_n,'PctChange')['Ticker'].tolist() if not df.empty else []
    holdings = {}
    if live_mode:
        for pos in rh_orders.get_open_stock_positions() or []:
            holdings[pos['symbol'].upper()] = float(pos['quantity'])
        if include_crypto:
            for pos in rh_crypto.get_crypto_positions() or []:
                holdings[pos['currency'].upper()] = float(pos['quantity'])
    log = []
    for sym in all_symbols:
        cur = sym.replace('-USD','')
        qty_cur = holdings.get(cur,0)
        buy = sym in picks and qty_cur==0
        sell = sym not in picks and qty_cur>0
        if not (buy or sell): continue
        action = 'BUY' if buy else 'SELL'
        # Place order
        executed, oid, status = 0, '', 'simulated'
        if live_mode:
            try:
                if sym.endswith('-USD'):
                    if buy:
                        resp = rh_crypto.order_buy_crypto_by_price(cur, alloc)
                    else:
                        resp = rh_crypto.order_sell_crypto_by_price(cur, alloc)
                else:
                    if buy:
                        resp = rh_orders.order_buy_fractional_by_price(sym, alloc)
                    else:
                        resp = rh_orders.order_sell_fractional_by_price(sym, alloc)
                st.sidebar.write(f"Order response: {resp}")
                oid = resp.get('id','')
                executed = float(resp.get('cumulative_quantity') or resp.get('executed_quantity',0) or resp.get('amount',0))
                status = resp.get('state','')
            except Exception as e:
                st.warning(f"Order {action} {sym} failed: {e}")
        log.append({'Ticker':sym,'Action':action,'PctChange':round(pct.get(sym,0),2),
                    'Executed':round(executed,4),'OrderID':oid,'Status':status,'Time':now.strftime('%Y-%m-%d %H:%M:%S')})
    df_log = pd.DataFrame(log)
    st.subheader("Rebalance Log")
    st.table(df_log)
    st.download_button("Download Logs CSV", df_log.to_csv(index=False), file_name='trade_logs.csv')
    if live_mode:
        st.subheader("Open Orders")
        open_st = rh_orders.get_all_open_stock_orders() or []
        open_cr = rh_crypto.get_all_open_crypto_orders() if include_crypto else []
        st.write(open_st + open_cr)
        st.write("✔️ Pending orders are shown above.")
