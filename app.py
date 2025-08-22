# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from io import BytesIO

# Try importing ta, or fall back gracefully
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# -------------------------------
# Helper functions
# -------------------------------

def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

def add_indicators(data):
    if not TA_AVAILABLE:
        st.warning("⚠️ The 'ta' library is not installed. Indicators will be skipped.")
        data['RSI'] = None
        data['BB_high'] = None
        data['BB_low'] = None
        return data
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()
    return data

def generate_signals(data):
    if not TA_AVAILABLE:
        data['Signal'] = ""
        return data
    signals = []
    for i in range(len(data)):
        if data['Close'].iloc[i] < data['BB_low'].iloc[i] and data['RSI'].iloc[i] < 30:
            signals.append("Buy")
        elif data['Close'].iloc[i] > data['BB_high'].iloc[i] and data['RSI'].iloc[i] > 70:
            signals.append("Sell")
        else:
            signals.append("")
    data['Signal'] = signals
    return data

# -------------------------------
# Streamlit App
# -------------------------------

st.set_page_config(page_title="Stock Analyzer Bot", layout="wide")
st.title("📈 Stock Analyzer Bot")

# --- Initialize state so downloads don't reset the app ---
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'csv_bytes' not in st.session_state:
    st.session_state['csv_bytes'] = None
if 'trade_csv_bytes' not in st.session_state:
    st.session_state['trade_csv_bytes'] = None

# --- Inputs ---
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL", key="ticker")
start_date = st.date_input("Start Date", datetime(2023, 1, 1), key="start")
end_date = st.date_input("End Date", datetime.today(), key="end")

# --- Run analysis ---
if st.button("Run Analyse", key="run"):
    data = download_data(ticker, start_date, end_date)
    data = add_indicators(data)
    data = generate_signals(data)
    st.session_state['data'] = data

    # Prepare persistent CSV bytes so repeated downloads work
    csv_io = BytesIO()
    data.to_csv(csv_io)
    st.session_state['csv_bytes'] = csv_io.getvalue()

    trade_log = data[data['Signal'] != ""]
    trade_io = BytesIO()
    trade_log.to_csv(trade_io)
    st.session_state['trade_csv_bytes'] = trade_io.getvalue()

    st.session_state['analysis_done'] = True

# --- Results ---
if st.session_state.get('analysis_done') and st.session_state['data'] is not None:
    st.subheader(f"Results for {ticker}")

    if TA_AVAILABLE:
        st.line_chart(st.session_state['data'][['Close', 'BB_high', 'BB_low']])
    else:
        st.line_chart(st.session_state['data'][['Close']])

    st.write(st.session_state['data'].tail())

    # CSV Export Button (stable across reruns)
    st.download_button(
        label="📥 Download CSV",
        data=st.session_state['csv_bytes'],
        file_name=f"{ticker}_analysis.csv",
        mime="text/csv",
        key="dl_csv"
    )

    # Trade Log Button (stable across reruns)
    st.download_button(
        label="📥 Download Trade Log",
        data=st.session_state['trade_csv_bytes'],
        file_name=f"{ticker}_trade_log.csv",
        mime="text/csv",
        key="dl_trades"
    )
else:
    st.info("Set your inputs and click **Run Analyse** to generate results.")


# -------------------------------
# Requirements for deployment (create requirements.txt with these lines)
# -------------------------------
# streamlit==1.36.0
# yfinance==0.2.40
# pandas==2.2.2
# numpy==1.26.4
# ta==0.11.0

# (Optional) runtime.txt:
# python-3.11
