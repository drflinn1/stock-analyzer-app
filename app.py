# Streamlit Web App Version of Stock Analyzer

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time
import os
from datetime import datetime

# =============================
# Technical Indicator Functions
# =============================
def simple_sma(series, window):
    return series.squeeze().rolling(window=window).mean()

def simple_rsi(series, period=14):
    series = series.squeeze()
    delta = series.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0), index=series.index)
    loss = pd.Series(np.where(delta > 0, 0, -delta), index=series.index)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)

def bollinger_bands(series, window=20, num_std=2):
    series = series.squeeze()
    sma = simple_sma(series, window)
    rolling_std = series.rolling(window=window).std()
    upper_band = sma + num_std * rolling_std
    lower_band = sma - num_std * rolling_std
    return sma, upper_band, lower_band

# =============================
# Data and Analysis Functions
# =============================
def get_data(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=False)
    if data.empty:
        raise ValueError("No data fetched for ticker.")

    data['sma_20'] = simple_sma(data['Close'], window=20)
    data['sma_50'] = simple_sma(data['Close'], window=50)
    data['rsi'] = simple_rsi(data['Close'])
    sma, upper, lower = bollinger_bands(data['Close'])
    data['bb_upper'] = upper
    data['bb_lower'] = lower
    return data.dropna()

def analyze(data):
    if len(data) < 60:
        raise ValueError("Not enough data to analyze")

    latest = data.iloc[-1]
    prev = data.iloc[-2]
    signal = "HOLD"
    reasons = []

    rsi = float(latest['rsi'])
    close = float(latest['Close'])
    sma_20 = float(latest['sma_20'])
    sma_50 = float(latest['sma_50'])
    bb_upper = float(latest['bb_upper'])
    bb_lower = float(latest['bb_lower'])
    prev_sma_20 = float(prev['sma_20'])
    prev_sma_50 = float(prev['sma_50'])

    if rsi < 30:
        reasons.append("RSI below 30 (oversold)")
    if rsi > 70:
        reasons.append("RSI above 70 (overbought)")
    if prev_sma_20 < prev_sma_50 and sma_20 > sma_50:
        reasons.append("20 SMA crossed above 50 SMA (bullish)")
    if prev_sma_20 > prev_sma_50 and sma_20 < sma_50:
        reasons.append("20 SMA crossed below 50 SMA (bearish)")
    if close < bb_lower:
        reasons.append("Price below lower Bollinger Band (potential buy)")
    if close > bb_upper:
        reasons.append("Price above upper Bollinger Band (potential sell)")

    reason_text = "; ".join(reasons).lower()
    if "buy" in reason_text or "bullish" in reason_text:
        signal = "BUY"
    elif "sell" in reason_text or "bearish" in reason_text:
        signal = "SELL"

    return {
        "RSI": round(rsi, 2),
        "20 SMA": round(sma_20, 2),
        "50 SMA": round(sma_50, 2),
        "Signal": signal,
        "Reasons": "; ".join(reasons)
    }

# =============================
# Trade Logging with Tax Tracking
# =============================
def log_trade(ticker, signal, price, reasons):
    if signal not in ["BUY", "SELL"]:
        signal = "BUY"  # Inject fake BUY for testing
        reasons = "Manual test trade for button check"
    filename = "trade_log.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gain_loss = np.random.uniform(-20, 50)  # Simulated gain/loss in USD
    tax_category = "Short-Term" if np.random.rand() > 0.5 else "Long-Term"

    trade_data = pd.DataFrame([{
        "Date": now,
        "Ticker": ticker,
        "Signal": signal,
        "Price": price,
        "Gain/Loss": round(gain_loss, 2),
        "Tax Category": tax_category,
        "Reasons": reasons
    }])

    if os.path.exists(filename):
        trade_data.to_csv(filename, mode='a', header=False, index=False)
    else:
        trade_data.to_csv(filename, mode='w', header=True, index=False)

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer App")

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
period = st.selectbox("Select period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=4)
auto_refresh = st.checkbox("Auto-refresh every hour")
if "results" not in st.session_state:
    st.session_state.results = []

# Store full analysis data too (charts)
if "full_results" not in st.session_state:
    st.session_state.full_results = {}

if st.button("Run Analysis") or auto_refresh:
    results = []
    full_results = {}

    for ticker in tickers:
        try:
            data = get_data(ticker, period)
            if len(data) < 60:
                raise ValueError("Not enough data to analyze")

            analysis = analyze(data)
            analysis["Ticker"] = ticker
            results.append(analysis)
            full_results[ticker] = data
            log_trade(ticker, analysis['Signal'], data['Close'].iloc[-1], analysis['Reasons'])

        except Exception as e:
            st.warning(f"{ticker}: {e}")

    st.session_state.results = results
    st.session_state.full_results = full_results

# REDRAW even after download
if st.session_state.results:
    for analysis in st.session_state.results:
        ticker = analysis["Ticker"]
        st.markdown("---")
        st.markdown(f"## 📊 Analysis for `{ticker}`")

        if ticker in st.session_state.full_results:
            data = st.session_state.full_results[ticker]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close"))
            fig.add_trace(go.Scatter(x=data.index, y=data['sma_20'], name="20 SMA", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['sma_50'], name="50 SMA", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], name="BB Upper", line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], name="BB Lower", line=dict(dash='dot')))
            fig.update_layout(title=f"{ticker} Price Chart", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        signal = analysis['Signal']
        emoji = "🟢 **BUY**" if signal == "BUY" else "🔴 **SELL**" if signal == "SELL" else "🟡 **HOLD**"
        st.markdown(f"**Signal:** {emoji}")
        st.markdown(f"- **RSI:** `{analysis['RSI']}`")
        st.markdown(f"- **20 SMA:** `{analysis['20 SMA']}`")
        st.markdown(f"- **50 SMA:** `{analysis['50 SMA']}`")
        if analysis['Reasons']:
            st.markdown(f"_Reasons: {analysis['Reasons']}_")

    df = pd.DataFrame(st.session_state.results).set_index("Ticker")
    st.subheader("📋 Summary Table")
    st.dataframe(df)
    csv = df.to_csv().encode('utf-8')
    st.download_button("⬇ Download CSV", csv, "stock_analysis_results.csv", "text/csv")

    if os.path.exists("trade_log.csv"):
        st.markdown("---")
        st.subheader("🧾 Trade Log History")
        log_df = pd.read_csv("trade_log.csv")

        start_date = st.date_input("Start Date", value=pd.to_datetime(log_df['Date']).min().date())
        end_date = st.date_input("End Date", value=pd.to_datetime(log_df['Date']).max().date())
        log_df['Date'] = pd.to_datetime(log_df['Date'])
        log_df = log_df[(log_df['Date'] >= pd.to_datetime(start_date)) & (log_df['Date'] <= pd.to_datetime(end_date))]

        tickers_filter = st.multiselect("Filter by ticker", options=log_df['Ticker'].unique(), default=list(log_df['Ticker'].unique()))
        log_df = log_df[log_df['Ticker'].isin(tickers_filter)]

        st.dataframe(log_df)
        log_csv = log_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Trade Log", log_csv, "trade_log.csv", "text/csv")

        st.markdown("---")
        st.subheader("💰 Tax Summary")
        tax_summary = log_df.groupby("Tax Category")["Gain/Loss"].sum().reset_index()
        st.dataframe(tax_summary)
        tax_csv = tax_summary.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Tax Summary", tax_csv, "tax_summary.csv", "text/csv")

else:
    st.info("No valid data analyzed.")

if auto_refresh:
    st.experimental_rerun()
