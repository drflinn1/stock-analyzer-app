# Streamlit Web App Version of Stock Analyzer with Robinhood Integration

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Load secrets for Robinhood login (STREAMLIT CLOUD SECURE METHOD)
SIMULATE_TRADES = st.sidebar.checkbox("🔌 Simulate Trading Mode", value=True)

# Auto-refresh every hour (3600000 ms)
st_autorefresh(interval=3600000, limit=None, key="auto-refresh")

if not SIMULATE_TRADES:
    try:
        from robin_stocks import robinhood as r
        username = st.secrets["ROBINHOOD_USERNAME"]
        password = st.secrets["ROBINHOOD_PASSWORD"]
        r.login(username, password)
        st.sidebar.success("✅ Connected to Robinhood!")
    except Exception as e:
        st.sidebar.error("❌ Failed to log in to Robinhood")
        SIMULATE_TRADES = True

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
def get_data(ticker, period, retries=3, delay=2):
    for attempt in range(retries):
        data = yf.download(ticker, period=period, auto_adjust=False)
        if not data.empty:
            break
        time.sleep(delay)
    else:
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
        return None

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
# Trade Logging (SIMULATED or LIVE)
# =============================
def log_trade(ticker, signal, price, reasons):
    if signal not in ["BUY", "SELL"]:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gain_loss = np.random.uniform(-20, 50)
    tax_category = "Short-Term" if np.random.rand() > 0.5 else "Long-Term"

    if SIMULATE_TRADES:
        st.info(f"Simulated {signal} trade for {ticker} at ${price}")
    else:
        try:
            if signal == "BUY":
                r.orders.order_buy_fractional_by_price(ticker, 1.00, timeInForce='gfd')
            else:
                r.orders.order_sell_fractional_by_price(ticker, 1.00, timeInForce='gfd')
            st.success(f"Live {signal} trade for {ticker} placed at $1.00!")
        except Exception as e:
            st.error(f"Live trade failed: {e}")
            return

    trade_data = pd.DataFrame([{
        "Date": now,
        "Ticker": ticker,
        "Signal": signal,
        "Price": price,
        "Gain/Loss": round(gain_loss, 2),
        "Tax Category": tax_category,
        "Reasons": reasons
    }])

    filename = "trade_log.csv"
    if os.path.exists(filename):
        trade_data.to_csv(filename, mode='a', header=False, index=False)
    else:
        trade_data.to_csv(filename, mode='w', header=True, index=False)

# =============================
# Streamlit UI Starts Here
# =============================
st.set_page_config(page_title="Stock Analyzer Bot", layout="wide")
st.title("📊 Stock Analyzer Bot (Live Trading + Tax Logs)")

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
period = "6mo"

if st.button("▶ Run Analysis"):
    results = {}
    for ticker in tickers:
        try:
            data = get_data(ticker, period)
            st.write(f"{ticker} data rows: {len(data)}")
            summary = analyze(data)
            if summary is None:
                st.warning(f"{ticker}: Skipped — not enough data to analyze.")
                continue
            log_trade(ticker, summary["Signal"], float(data['Close'].iloc[-1]), summary["Reasons"])
            results[ticker] = summary

            st.markdown(f"## 📈 {ticker} Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=data.index, y=data['sma_20'], mode='lines', name='20 SMA', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['sma_50'], mode='lines', name='50 SMA', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], mode='lines', name='BB Upper', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], mode='lines', name='BB Lower', line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)

            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
            st.markdown(f"### {emoji.get(summary['Signal'], '❓')} {ticker} - {summary['Signal']}")
            st.write(summary)
            st.markdown("---")

        except Exception as e:
            st.error(f"Error analyzing {ticker}: {e}")

    if results:
        df = pd.DataFrame(results).T
        df.to_csv("stock_analysis_results.csv")
        st.download_button("⬇ Download CSV", df.to_csv().encode(), file_name="stock_analysis_results.csv", mime="text/csv")

        # Summary Chart
        st.markdown("### 📊 Summary of Trade Signals")
        bar_chart = pd.DataFrame({
            "Ticker": list(results.keys()),
            "Signal": [v["Signal"] for v in results.values()],
            "Gain/Loss Estimate": [np.random.uniform(-20, 50) for _ in results.values()]  # Simulated
        })
        st.bar_chart(bar_chart.set_index("Ticker")[["Gain/Loss Estimate"]])

if os.path.exists("trade_log.csv"):
    df_trades = pd.read_csv("trade_log.csv")
    st.subheader("🧾 Trade Log")
    st.dataframe(df_trades)
    st.download_button("⬇ Download Trade Log", df_trades.to_csv(index=False).encode(), file_name="trade_log.csv", mime="text/csv")

    tax_summary = df_trades.groupby("Tax Category")["Gain/Loss"].sum().reset_index()
    st.subheader("💰 Tax Summary")
    st.dataframe(tax_summary)
    st.download_button("⬇ Download Tax Summary", tax_summary.to_csv(index=False).encode(), file_name="tax_summary.csv", mime="text/csv")
