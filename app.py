# Streamlit Web App Version of Stock Analyzer with Robinhood Integration

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import robin_stocks.robinhood as r

# Load environment variables for Robinhood login
load_dotenv()
username = os.getenv("ROBINHOOD_USERNAME")
password = os.getenv("ROBINHOOD_PASSWORD")
login = r.login(username, password)

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
# Trade Logging with Robinhood Execution
# =============================
def log_trade(ticker, signal, price, reasons):
    if signal not in ["BUY", "SELL"]:
        return  # skip logging and trading for HOLD

    filename = "trade_log.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gain_loss = np.random.uniform(-20, 50)  # Simulated gain/loss in USD
    tax_category = "Short-Term" if np.random.rand() > 0.5 else "Long-Term"

    # 🔁 Place a live trade
    try:
        if signal == "BUY":
            r.orders.order_buy_market(ticker, 1)
        elif signal == "SELL":
            r.orders.order_sell_market(ticker, 1)
    except Exception as e:
        print(f"Failed to place trade for {ticker}: {e}")

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

# (The rest of the Streamlit app code remains unchanged)
