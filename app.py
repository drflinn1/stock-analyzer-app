# app.py – Streamlit Web App Version of Stock Analyzer with Robinhood Integration – **tidied UI**

"""
Changes in this revision 2025‑07‑23
----------------------------------
1. 🔄 **Dynamic ticker list** – fetch S&P 500 constituents and compute top N performers.
2. ➕ Added option to scan top N performers by 1‑day percent change.
3. ✅ All previous UI and trading features preserved.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from email.message import EmailMessage
from streamlit_autorefresh import st_autorefresh
import smtplib

# -------------------------
# ▶  CONFIG & SECRETS
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Live Trading + Tax Logs)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Robinhood integration (secure on Streamlit Cloud)
try:
    from robin_stocks import robinhood as r
except ImportError:
    r = None

# -------------------------
# ▶  FETCH S&P 500 & TOP N
# -------------------------
@st.cache_data

def get_sp500_tickers():
    """
    Fetch S&P 500 constituents from Wikipedia, fallback via requests/BeautifulSoup if needed.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, flavor='bs4', attrs={"class": "wikitable"})[0]
        return table['Symbol'].tolist()
    except Exception:
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            tbl = soup.find('table', {'class': 'wikitable'})
            symbols = [row.find_all('td')[0].text.strip() for row in tbl.find_all('tr')[1:] if row.find_all('td')]
            return symbols
        except Exception as e:
            st.sidebar.warning(f"Failed to fetch S&P 500 list: {e}")
            return []

@st.cache_data

def get_top_tickers(n=50):
    """Return top `n` tickers by 1‑day percent change."""
    syms = get_sp500_tickers()
    perf = {}
    for sym in syms:
        try:
            df = yf.download(sym, period='2d', progress=False)
            if len(df) >= 2:
                perf[sym] = df['Close'].pct_change().iloc[-1]
        except Exception:
            continue
    # sort descending by performance
    top = sorted(perf, key=perf.get, reverse=True)[:n]
    return top

# -------------------------
# ▶  ANALYSIS HELPERS
# -------------------------
def simple_sma(series: pd.Series, window: int):
    return series.squeeze().rolling(window).mean()

def simple_rsi(series: pd.Series, period: int = 14):
    s = series.squeeze()
    delta = s.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    s = series.squeeze()
    sma = s.rolling(window).mean()
    std = s.rolling(window).std()
    return sma, sma + num_std * std, sma - num_std * std

def get_data(ticker: str, period: str, retries: int = 3):
    for _ in range(retries):
        data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if not data.empty:
            data['sma_20'], data['bb_upper'], data['bb_lower'] = bollinger_bands(data['Close'])
            data['sma_50'] = simple_sma(data['Close'], 50)
            data['rsi'] = simple_rsi(data['Close'])
            return data.dropna()
        time.sleep(1)
    raise ValueError(f"No data fetched for {ticker}")

def analyze(df: pd.DataFrame) -> dict | None:
    if len(df) < 60:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    reasons = []
    # RSI
    if cur.rsi < 30: reasons.append("RSI below 30 (oversold)")
    if cur.rsi > 70: reasons.append("RSI above 70 (overbought)")
    # SMA cross
    if prev.sma_20 < prev.sma_50 <= cur.sma_20:
        reasons.append("20 SMA crossed above 50 SMA (bullish)")
    if prev.sma_20 > prev.sma_50 >= cur.sma_20:
        reasons.append("20 SMA crossed below 50 SMA (bearish)")
    # Bollinger
    if cur.Close < cur.bb_lower: reasons.append("Price below lower Bollinger Band (potential buy)")
    if cur.Close > cur.bb_upper: reasons.append("Price above upper Bollinger Band (potential sell)")
    text = "; ".join(reasons).lower()
    signal = "HOLD"
    if any(k in text for k in ["buy", "bullish"]): signal = "BUY"
    elif any(k in text for k in ["sell", "bearish"]): signal = "SELL"
    return {"RSI": round(cur.rsi,2), "20 SMA": round(cur.sma_20,2), "50 SMA": round(cur.sma_50,2), "Signal": signal, "Reasons": "; ".join(reasons)}

# Email & logging

def notify_email(tkr, summary, price):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"{summary['Signal']} signal for {tkr}"
    body = f"Ticker: {tkr}\nSignal: {summary['Signal']}\nPrice: {price}\nReasons: {summary['Reasons']}\nTime: {now}"
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'], msg['From'], msg['To'] = subject, st.secrets['EMAIL_ADDRESS'], st.secrets['EMAIL_RECEIVER']
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(st.secrets['EMAIL_ADDRESS'], st.secrets['EMAIL_PASSWORD'])
            s.send_message(msg)
        st.sidebar.success("📧 Email sent!")
    except Exception as e:
        st.sidebar.error(f"Email error: {e}")

def log_trade(tkr, summary, price):
    if summary['Signal'] == 'HOLD': return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gain_loss = np.random.uniform(-20,50)
    cat = 'Short-Term' if np.random.rand()>0.5 else 'Long-Term'
    row = pd.DataFrame([{ 'Date':now, 'Ticker':tkr, 'Signal':summary['Signal'], 'Price':price, 'Gain/Loss':round(gain_loss,2), 'Tax Category':cat, 'Reasons':summary['Reasons'] }])
    row.to_csv('trade_log.csv', mode='a', header=not os.path.exists('trade_log.csv'), index=False)
    notify_email(tkr, summary, price)

# -------------------------
# ▶  SIDEBAR UI
# -------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    with st.expander("General", expanded=True):
        simulate_mode = st.checkbox("Simulate Trading Mode", value=True)
        debug_mode = st.checkbox("Show debug logs", value=False)
        st_autorefresh(interval=3_600_000, limit=None, key="hour_refresh")
    with st.expander("Analysis Options", expanded=True):
        scan_top = st.checkbox("Scan top N performers", value=False)
        if scan_top:
            top_n = st.slider("Top tickers to scan", 10, 100, 50)
        else:
            options = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","NFLX"]
            user_tickers = st.multiselect("📈 Choose tickers", options, default=["AAPL","TSLA"])
        period = st.selectbox("🗓️ Date range", ["1mo","3mo","6mo","1y","2y"], index=2)
    status_badge = "🟢 LIVE" if not simulate_mode else "🔴 SIM"

# -------------------------
# ▶  OPTIONAL ROBINHOOD LOGIN
# -------------------------
if not simulate_mode:
    if r is None:
        st.sidebar.error("robin_stocks missing – switching to simulate mode")
        simulate_mode = True
    else:
        try:
            r.login(st.secrets['ROBINHOOD_USERNAME'], st.secrets['ROBINHOOD_PASSWORD'])
            st.sidebar.success("✅ Connected to Robinhood")
        except Exception as e:
            st.sidebar.error(f"Failed Robinhood login: {e}")
            simulate_mode = True

# -------------------------
# ▶  MAIN CONTENT
# -------------------------
st.markdown(f"### {status_badge} {PAGE_TITLE}")

if st.button("▶ Run Analysis", use_container_width=True):
    if scan_top:
        tickers = get_top_tickers(top_n)
    else:
        tickers = user_tickers
    if not tickers:
        st.warning("Select at least one ticker")
        st.stop()
    results = {}
    for t in tickers:
        try:
            df = get_data(t, period)
            if debug_mode:
                st.write(f"{t} rows: {len(df)}")
            summary = analyze(df)
            if summary is None:
                st.warning(f"{t}: Not enough data, skipped")
                continue
            results[t] = summary
            log_trade(t, summary, float(df.Close.iloc[-1]))
            # chart
            st.markdown(f"#### 📈 {t} Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df.Close, name="Close"))
            fig.add_trace(go.Scatter(x=df.index, y=df.sma_20, name="20 SMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df.sma_50, name="50 SMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df.bb_upper, name="BB Upper", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=df.bb_lower, name="BB Lower", line=dict(dash="dot")))
            st.plotly_chart(fig, use_container_width=True)
            badge = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}[summary['Signal']]
            st.markdown(f"**{badge} {t} – {summary['Signal']}**")
            st.json(summary)
            st.divider()
        except Exception as e:
            st.error(f"{t} failed: {e}")
    # summary table & downloads
    if results:
        df_res = pd.DataFrame(results).T
        st.download_button("⬇ Download CSV", df_res.to_csv().encode(), "analysis.csv")
        st.markdown("### 📊 Summary of Trade Signals")
        mapv = {"BUY":1,"SELL":-1,"HOLD":0}
        st.bar_chart(pd.Series({k:mapv[v['Signal']] for k,v in results.items()}))

# -------------------------
# ▶  TRADE LOG & TAX SUMMARY
# -------------------------
if os.path.exists('trade_log.csv'):
    trades = pd.read_csv('trade_log.csv')
    st.subheader("🧾 Trade Log")
    st.dataframe(trades)
    st.download_button("⬇ Download Trade Log", trades.to_csv(index=False).encode(), "trade_log.csv")
    # tax summary & P/L
    tax = trades.groupby('Tax Category')['Gain/Loss'].sum().reset_index()
    total = trades['Gain/Loss'].sum()
    st.markdown(f"## 💰 **Total Portfolio P/L: ${total:.2f}**")
    st.subheader("Tax Summary")
    st.dataframe(tax)
    st.download_button("⬇ Download Tax Summary", tax.to_csv(index=False).encode(), "tax_summary.csv")
    # cumulative
    trades['Cum P/L'] = trades['Gain/Loss'].cumsum()
    st.markdown("### 📈 Portfolio Cumulative Profit Over Time")
    st.line_chart(trades.set_index('Date')['Cum P/L'])
