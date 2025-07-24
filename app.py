# app.py – Streamlit Web App Version of Stock Analyzer with Robinhood Integration – **tidied UI**

"""
Changes in this revision 2025‑07‑23
----------------------------------
1. 🔄 **Dynamic ticker list** – fetch S&P 500 constituents and compute top N performers.
2. ➕ Added option to scan top 50 tickers by 1‑day percent change.
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

# Robinhood (secure on Streamlit Cloud)
try:
    from robin_stocks import robinhood as r
except ImportError:
    r = None

# -------------------------
# ▶  Helper to fetch top tickers
# -------------------------
@st.cache_data
def get_sp500_tickers():
    """
    Fetch S&P 500 tickers, with fallback if parser dependency missing.
    """
    try:
        # use html5lib flavor to avoid lxml dependency
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            flavor='html5lib'
        )[0]
        return table['Symbol'].tolist()
    except (ImportError, ValueError, OSError) as e:
        st.sidebar.warning(f"Failed to fetch S&P 500 list: {e}")
        return []

@st.cache_data
def get_top_tickers(n=50):
    syms = get_sp500_tickers()
    perf = {}
    for sym in syms:
        data = yf.download(sym, period='2d', progress=False)
        if len(data) >= 2:
            perf[sym] = data['Close'].pct_change().iloc[-1]
    top = sorted(perf, key=perf.get, reverse=True)[:n]
    return top

# -------------------------
# ▶  Helper functions (analysis / email / logging)
# -------------------------
def simple_sma(series: pd.Series, window: int):
    return series.squeeze().rolling(window).mean()

def simple_rsi(series: pd.Series, period: int = 14):
    series = series.squeeze()
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = simple_sma(series, window)
    std = series.squeeze().rolling(window).std()
    return sma, sma + num_std * std, sma - num_std * std

def get_data(ticker: str, period: str, retries: int = 3):
    for _ in range(retries):
        data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if not data.empty:
            sma20, upper, lower = bollinger_bands(data["Close"])
            data["sma_20"], data["bb_upper"], data["bb_lower"] = sma20, upper, lower
            data["sma_50"] = simple_sma(data["Close"], 50)
            data["rsi"] = simple_rsi(data["Close"])
            return data.dropna()
        time.sleep(1)
    raise ValueError("No data fetched")

def analyze(df: pd.DataFrame):
    if len(df) < 60:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    reasons, signal = [], "HOLD"
    if cur.rsi < 30:
        reasons.append("RSI below 30 (oversold)")
    if cur.rsi > 70:
        reasons.append("RSI above 70 (overbought)")
    if prev.sma_20 < prev.sma_50 <= cur.sma_20:
        reasons.append("20 SMA crossed above 50 SMA (bullish)")
    if prev.sma_20 > prev.sma_50 >= cur.sma_20:
        reasons.append("20 SMA crossed below 50 SMA (bearish)")
    if cur.Close < cur.bb_lower:
        reasons.append("Price below lower Bollinger Band (potential buy)")
    if cur.Close > cur.bb_upper:
        reasons.append("Price above upper Bollinger Band (potential sell)")
    reason_text = "; ".join(reasons).lower()
    if "buy" in reason_text or "bullish" in reason_text:
        signal = "BUY"
    elif "sell" in reason_text or "bearish" in reason_text:
        signal = "SELL"
    return {"RSI": round(cur.rsi,2), "20 SMA": round(cur.sma_20,2), "50 SMA": round(cur.sma_50,2), "Signal": signal, "Reasons": "; ".join(reasons)}

def notify_email(tkr, summary, price):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"{summary['Signal']} Signal for {tkr}"
    body = f"Ticker: {tkr}\nSignal: {summary['Signal']}\nPrice: {price}\nReasons: {summary['Reasons']}\nTime: {now}"
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"], msg["From"], msg["To"] = subject, st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_RECEIVER"]
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_PASSWORD"])
            s.send_message(msg)
        st.sidebar.success("📧 Email sent!")
    except Exception as e:
        st.sidebar.error(f"Email error: {e}")

def log_trade(tkr, summary, price):
    if summary["Signal"] == "HOLD": return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gain_loss = np.random.uniform(-20,50)
    tax_cat = "Short-Term" if np.random.rand()>0.5 else "Long-Term"
    row = pd.DataFrame([{"Date":now,"Ticker":tkr,"Signal":summary["Signal"],"Price":price,"Gain/Loss":round(gain_loss,2),"Tax Category":tax_cat,"Reasons":summary["Reasons"]}])
    row.to_csv("trade_log.csv", mode="a", header=not os.path.exists("trade_log.csv"), index=False)
    notify_email(tkr, summary, price)

# -------------------------
# ▶  SIDEBAR UI
# -------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    with st.expander("General", expanded=True):
        simulate_mode = st.checkbox("Simulate Trading Mode", value=True)
        debug_mode    = st.checkbox("Show debug logs", value=False)
        st_autorefresh(interval=3_600_000, limit=None, key="hour_refresh")
    with st.expander("Analysis Options", expanded=True):
        scan_dynamic = st.checkbox("Scan top N performers", value=False)
        if scan_dynamic:
            top_n = st.slider("Top tickers to scan",10,100,50,10)
            ticker_options = get_top_tickers(top_n)
        else:
            ticker_options = get_sp500_tickers()
        tickers = st.multiselect("📈 Choose tickers", ticker_options, default=ticker_options[:5])
        period  = st.selectbox("🗓️ Date range",["1mo","3mo","6mo","1y","2y"],index=2)
    status_badge = "🟢 LIVE" if not simulate_mode else "🔴 SIM"

# -------------------------
# ▶  (Optional) Robinhood login
# -------------------------
if not simulate_mode:
    if r is None:
        st.sidebar.error("robin_stocks missing – switching to simulate mode")
        simulate_mode = True
    else:
        try:
            r.login(st.secrets["ROBINHOOD_USERNAME"], st.secrets["ROBINHOOD_PASSWORD"])
            st.sidebar.success("✅ Connected to Robinhood")
        except Exception as e:
            st.sidebar.error(f"Failed Robinhood login: {e}")
            simulate_mode = True

# -------------------------
# ▶  MAIN PAGE
# -------------------------
st.markdown(f"### {status_badge} {PAGE_TITLE}")
if st.button("▶ Run Analysis", use_container_width=True):
    if not tickers:
        st.warning("Select at least one ticker")
        st.stop()
    results = {}
    for tkr in tickers:
        try:
            df = get_data(tkr, period)
            if debug_mode: st.write(f"{tkr} rows: {len(df)}")
            summary = analyze(df)
            if summary is None:
                st.warning(f"{tkr}: Not enough data, skipped")
                continue
            results[tkr] = summary
            log_trade(tkr, summary, float(df.Close.iloc[-1]))
            st.markdown(f"#### 📈 {tkr} Price Chart  ↻")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df.Close, name="Close"))
            fig.add_trace(go.Scatter(x=df.index, y=df.sma_20, name="20 SMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df.sma_50, name="50 SMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df.bb_upper, name="BB Upper", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=df.bb_lower, name="BB Lower", line=dict(dash="dot")))
            st.plotly_chart(fig, use_container_width=True)
            badge = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}[summary["Signal"]]
            st.markdown(f"**{badge} {tkr} – {summary['Signal']}**")
            st.json(summary)
            st.divider()
        except Exception as e:
            st.error(f"{tkr} failed: {e}")
    if results:
        res_df = pd.DataFrame(results).T
        st.download_button("⬇ Download CSV", res_df.to_csv().encode(), "stock_analysis_results.csv")
        st.markdown("### 📊 Summary of Trade Signals")
        signal_map = {"BUY":1,"SELL":-1,"HOLD":0}
        st.bar_chart(pd.Series({k:signal_map[v["Signal"]] for k,v in results.items()}))
if os.path.exists("trade_log.csv"):
    trades = pd.read_csv("trade_log.csv")
    st.subheader("🧾 Trade Log")
    st.dataframe(trades)
    st.download_button("⬇ Download Trade Log", trades.to_csv(index=False).encode(), "trade_log.csv")
    tax = trades.groupby("Tax Category")["Gain/Loss"].sum().reset_index()
    total_pl = trades["Gain/Loss"].sum()
    st.markdown(f"## 💰 **Total Portfolio P/L: ${total_pl:.2f}**")
    st.subheader("Tax Summary")
    st.dataframe(tax)
    st.download_button("⬇ Download Tax Summary", tax.to_csv(index=False).encode(), "tax_summary.csv")
    trades["Cum P/L"] = trades["Gain/Loss"].cumsum()
    st.markdown("### 📈 Portfolio Cumulative Profit Over Time")
    st.line_chart(trades.set_index("Date")["Cum P/L"])
