# Streamlit Web App Version of Stock Analyzer

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time

# =============================
# Technical Indicator Functions
# =============================
def simple_sma(series, window):
    return series.squeeze().rolling(window=window).mean()

def simple_rsi(series, period=14):
    series = series.squeeze()
    delta = series.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0), index=series.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0), index=series.index)
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
    if len(data) < 2:
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
# Streamlit UI
# =============================

# ✅ Enhanced layout with emojis, horizontal separators, and better label formatting.

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer App")

# Automated ticker list (user input removed)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
period = st.selectbox("Select period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
auto_refresh = st.checkbox("Auto-refresh every hour")

if st.button("Run Analysis") or auto_refresh:
    results = []

    for ticker in tickers:
        try:
            st.markdown("---")  # Horizontal rule
            st.markdown(f"### 📊 Analysis for `{ticker}`")

            data = get_data(ticker, period)
            analysis = analyze(data)
            analysis["Ticker"] = ticker
            results.append(analysis)

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

        except Exception as e:
            st.warning(f"{ticker}: {e}")

    if results:
        df = pd.DataFrame(results).set_index("Ticker")
        st.subheader("📋 Summary Table")
        st.dataframe(df)

        csv = df.to_csv().encode('utf-8')
        st.download_button("⬇ Download CSV", csv, "stock_analysis_results.csv", "text/csv")
    else:
        st.info("No valid data analyzed.")

    if auto_refresh:
        st.experimental_rerun()
