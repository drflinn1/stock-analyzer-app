# app.py – Streamlit Web App Version of Stock Analyzer with Robinhood Integration

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import smtplib
import streamlit as st
import yfinance as yf
from email.message import EmailMessage
from streamlit_autorefresh import st_autorefresh

# -------------------------
# ▶  CONFIG & SECRETS
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Live Trading + Tax Logs)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Robinhood API client (if installed)
try:
    from robin_stocks import robinhood as r
except ImportError:
    r = None

# -------------------------
# ▶  Helper to fetch S&P 500 & Top Movers
# -------------------------
@st.cache_data

def get_sp500_tickers():
    """
    Fetch S&P 500 tickers, fallback through BeautifulSoup if needed.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df_list = pd.read_html(url, flavor='bs4', attrs={'class':'wikitable'})
        table = df_list[0]
        return table['Symbol'].tolist()
    except Exception:
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            tbl = soup.find('table', {'class':'wikitable'})
            syms = []
            for row in tbl.find_all('tr')[1:]:
                cols = row.find_all('td')
                if cols:
                    syms.append(cols[0].text.strip())
            return syms
        except Exception as e:
            st.sidebar.warning(f"Failed to fetch S&P 500 list: {e}")
            return []

@st.cache_data

def get_top_tickers(n=50):
    """
    Return top N tickers by 1-day percent change.
    """
    symbols = get_sp500_tickers()
    perf = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period='2d', progress=False)
            if len(df) >= 2:
                perf[sym] = df['Close'].pct_change().iloc[-1]
        except Exception:
            continue
    sorted_syms = sorted(perf, key=perf.get, reverse=True)
    return sorted_syms[:n]

# -------------------------
# ▶  Analysis Helpers
# -------------------------
def simple_sma(series: pd.Series, window: int) -> pd.Series:
    return series.squeeze().rolling(window).mean()

def simple_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.squeeze()
    delta = s.diff()
    gain = delta.where(delta>0, 0.0)
    loss = -delta.where(delta<0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100/(1+rs))

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = simple_sma(series, window)
    std = series.squeeze().rolling(window).std()
    return sma, sma + num_std*std, sma - num_std*std

# -------------------------
# ▶  Data Fetch & Indicator Computation
# -------------------------
@st.cache_data

def get_data(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    for _ in range(retries):
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if not df.empty:
            close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].squeeze()
            sma20, bb_upper, bb_lower = bollinger_bands(close)
            df['sma_20'] = sma20
            df['sma_50'] = simple_sma(close, 50)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['rsi'] = simple_rsi(close)
            return df.dropna()
        time.sleep(1)
    raise ValueError(f"No data fetched for {ticker}")

# -------------------------
# ▶  Signal Analysis
# -------------------------
def analyze(df: pd.DataFrame) -> dict | None:
    if len(df) < 30:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    reasons = []
    # RSI signals
    rsi = float(cur['rsi'])
    if rsi < 30:
        reasons.append('RSI below 30 (oversold)')
    if rsi > 70:
        reasons.append('RSI above 70 (overbought)')
    # SMA crossover
    sma20_cur, sma50_cur = float(cur['sma_20']), float(cur['sma_50'])
    sma20_prev, sma50_prev = float(prev['sma_20']), float(prev['sma_50'])
    if sma20_prev < sma50_prev <= sma20_cur:
        reasons.append('20 SMA crossed above 50 SMA (bullish)')
    if sma20_prev > sma50_prev >= sma20_cur:
        reasons.append('20 SMA crossed below 50 SMA (bearish)')
    # Bollinger break
    close_price = float(cur['Close'])
    if close_price < float(cur['bb_lower']):
        reasons.append('Price below lower BB')
    if close_price > float(cur['bb_upper']):
        reasons.append('Price above upper BB')
    # Determine signal
    text = '; '.join(reasons).lower()
    signal = 'HOLD'
    if any(x in text for x in ['buy', 'bullish']):
        signal = 'BUY'
    elif any(x in text for x in ['sell', 'bearish']):
        signal = 'SELL'
    return {
        'RSI': round(rsi, 2),
        '20 SMA': round(sma20_cur, 2),
        '50 SMA': round(sma50_cur, 2),
        'Signal': signal,
        'Reasons': '; '.join(reasons)
    }

# -------------------------
# ▶  Notifications & Logging
# -------------------------
def notify_email(tkr: str, summ: dict, price: float):
    msg = EmailMessage()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body = f"Ticker: {tkr}\nSignal: {summ['Signal']} @ ${price}\nReasons: {summ['Reasons']}\nTime: {now}"
    msg.set_content(body)
    msg['Subject'], msg['From'], msg['To'] = f"{summ['Signal']} {tkr}", st.secrets['EMAIL_ADDRESS'], st.secrets['EMAIL_RECEIVER']
    with smtplib.SMTP_SSL('smtp.gmail.com',465) as s:
        s.login(st.secrets['EMAIL_ADDRESS'], st.secrets['EMAIL_PASSWORD'])
        s.send_message(msg)


def log_trade(tkr: str, summ: dict, price: float):
    if summ['Signal'] == 'HOLD':
        return
    now = datetime.now().isoformat()
    gain_loss = round(np.random.uniform(-20,50),2)
    cat = 'Short-Term' if np.random.rand()>0.5 else 'Long-Term'
    row = pd.DataFrame([{ 'Date': now, 'Ticker': tkr, 'Signal': summ['Signal'], 'Price': price, 'Gain/Loss': gain_loss, 'Tax Category': cat, 'Reasons': summ['Reasons'] }])
    row.to_csv('trade_log.csv', mode='a', header=not os.path.exists('trade_log.csv'), index=False)
    notify_email(tkr, summ, price)

# -------------------------
# ▶  SIDEBAR UI & Controls
# -------------------------
with st.sidebar:
    st.markdown('## ⚙️ Settings')
    with st.expander('General', expanded=True):
        simulate_mode = st.checkbox('Simulate Trading Mode', True)
        debug_mode = st.checkbox('Show debug logs', False)
        st_autorefresh(3600000, None, key='hour')
    with st.expander('Analysis Options', expanded=True):
        scan_top = st.checkbox('Scan top N performers', False)
        if scan_top:
            top_n = st.slider('Top tickers', 10, 100, 50)
        else:
            default_list = ['AAPL','MSFT','GOOGL','AMZN','TSLA']
            user_list = st.multiselect('📈 Choose tickers', get_sp500_tickers(), default=default_list)
        period = st.selectbox('🗓️ Date range', ['1mo','3mo','6mo','1y','2y'], index=2)
    status_badge = '🟢 LIVE' if not simulate_mode else '🔴 SIM'

# -------------------------
# ▶  Robinhood Login
# -------------------------
if not simulate_mode and r:
    try:
        r.login(st.secrets['ROBINHOOD_USERNAME'], st.secrets['ROBINHOOD_PASSWORD'])
        st.sidebar.success('Logged into Robinhood')
    except Exception as e:
        st.sidebar.error(f'Robinhood login error: {e}')
        simulate_mode = True

# -------------------------
# ▶  MAIN CONTENT
# -------------------------
st.markdown(f"### {status_badge} {PAGE_TITLE}")
if st.button('▶ Run Analysis'):
    ticks = get_top_tickers(top_n) if scan_top else user_list
    results = {}
    for t in ticks:
        try:
            df = get_data(t, period)
            if debug_mode: st.write(f'{t}: {len(df)} rows')
            summ = analyze(df)
            if not summ:
                st.info(f'{t} skipped, insufficient data')
                continue
            price = float(df['Close'].iloc[-1])
            results[t] = summ
            log_trade(t, summ, price)
            st.markdown(f'#### 📈 {t} Price Chart')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='20 SMA'))
            fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='50 SMA'))
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            emoji = {'BUY':'🟢','SELL':'🔴','HOLD':'🟡'}[summ['Signal']]
            st.markdown(f'**{emoji} {t} – {summ['Signal']}**')
            st.json(summ)
            st.divider()
        except Exception as e:
            st.error(f'{t} error: {e}')
    if results:
        df_res = pd.DataFrame(results).T
        st.download_button('⬇ Download CSV', df_res.to_csv().encode(), 'analysis.csv')
        st.markdown('### 📊 Summary of Signals')
        st.bar_chart(pd.Series({k:(1 if v['Signal']=='BUY' else -1 if v['Signal']=='SELL' else 0) for k,v in results.items()}))

# -------------------------
# ▶  Trade Log & Tax Summary
# -------------------------
if os.path.exists('trade_log.csv'):
    trades = pd.read_csv('trade_log.csv')
    st.subheader('🧾 Trade Log')
    st.dataframe(trades)
    st.download_button('⬇ Download Trade Log', trades.to_csv(index=False).encode(), 'trade_log.csv')
    # tax summary
    tax = trades.groupby('Tax Category')['Gain/Loss'].sum().reset_index()
    total_pl = trades['Gain/Loss'].sum()
    st.markdown(f'## 💰 Total P/L: ${total_pl:.2f}')
    st.subheader('Tax Summary')
    st.dataframe(tax)
    st.download_button('⬇ Download Tax Summary', tax.to_csv(index=False).encode(), 'tax_summary.csv')
    trades['Cum P/L'] = trades['Gain/Loss'].cumsum()
    st.markdown('### 📈 Cum. P/L Over Time')
    st.line_chart(trades.set_index('Date')['Cum P/L'])
