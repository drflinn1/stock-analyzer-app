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
    data = yf.download(ticker, start=start, end=end, progress=False)
    # Ensure a clean, single-symbol schema
    if data is None or data.empty:
        return pd.DataFrame()
    # Drop rows with all-NaN OHLCV
    data = data.dropna(how="all")
    return data

def add_indicators(data):
    # If no data, return with empty indicator columns so downstream doesn't crash
    if data is None or data.empty:
        return data

    if not TA_AVAILABLE:
        st.warning("⚠️ The 'ta' library is not installed. Indicators will be skipped.")
        data['RSI'] = pd.NA
        data['BB_high'] = pd.NA
        data['BB_low'] = pd.NA
        return data

    # Ensure 'Close' is a 1D float Series (yfinance can sometimes yield a 2D frame)
    if 'Close' not in data.columns:
        st.error("Downloaded data does not contain a 'Close' column.")
        return data

    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    # If all NaN after coercion, bail gracefully
    if close.isna().all():
        st.error("Price data contains no numeric 'Close' values for the selected range.")
        data['RSI'] = pd.NA
        data['BB_high'] = pd.NA
        data['BB_low'] = pd.NA
        return data

    rsi = ta.momentum.RSIIndicator(close=close).rsi()
    bb = ta.volatility.BollingerBands(close=close)

    data['RSI'] = rsi
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()
    return data
    try:
        rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'].astype(float))
        data['RSI'] = rsi_indicator.rsi()
        bb = ta.volatility.BollingerBands(close=data['Close'].astype(float))
        data['BB_high'] = bb.bollinger_hband()
        data['BB_low'] = bb.bollinger_lband()
    except Exception as e:
        st.error(f"Indicator calculation failed: {e}")
        data['RSI'] = None
        data['BB_high'] = None
        data['BB_low'] = None
    return data

def generate_signals(data):
    if data is None or data.empty:
        return data
    if not TA_AVAILABLE or 'RSI' not in data.columns or 'BB_low' not in data.columns or 'BB_high' not in data.columns:
        data['Signal'] = ""
        return data

    signals = []
    for i in range(len(data)):
        rsi_v = data['RSI'].iloc[i]
        bb_low = data['BB_low'].iloc[i]
        bb_high = data['BB_high'].iloc[i]
        close_v = data['Close'].iloc[i]
        if pd.notna(close_v) and pd.notna(bb_low) and pd.notna(rsi_v) and close_v < bb_low and rsi_v < 30:
            signals.append("Buy")
        elif pd.notna(close_v) and pd.notna(bb_high) and pd.notna(rsi_v) and close_v > bb_high and rsi_v > 70:
            signals.append("Sell")
        else:
            signals.append("")
    data['Signal'] = signals
    return data
    signals = []
    for i in range(len(data)):
        try:
            if data['Close'].iloc[i] < data['BB_low'].iloc[i] and data['RSI'].iloc[i] < 30:
                signals.append("Buy")
            elif data['Close'].iloc[i] > data['BB_high'].iloc[i] and data['RSI'].iloc[i] > 70:
                signals.append("Sell")
            else:
                signals.append("")
        except Exception:
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
    if data is None or data.empty:
        st.error("No price data returned for the selection. Try another ticker or adjust the dates.")
        st.session_state['analysis_done'] = False
    else:
        data = add_indicators(data)
        data = generate_signals(data)
        st.session_state['data'] = data

        # Prepare persistent CSV bytes so repeated downloads work
        csv_io = BytesIO(); data.to_csv(csv_io)
        st.session_state['csv_bytes'] = csv_io.getvalue()

        trade_log = data[data.get('Signal', "") != ""] if 'Signal' in data.columns else pd.DataFrame()
        trade_io = BytesIO(); trade_log.to_csv(trade_io)
        st.session_state['trade_csv_bytes'] = trade_io.getvalue()

        st.session_state['analysis_done'] = True

# --- Results ---
if st.session_state.get('analysis_done') and st.session_state['data'] is not None:
    st.subheader(f"Results for {ticker}")

    if TA_AVAILABLE and 'BB_high' in st.session_state['data']:
        st.line_chart(st.session_state['data'][['Close', 'BB_high', 'BB_low']].dropna(how='all'))
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
