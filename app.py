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
    if data is None or data.empty:
        return pd.DataFrame()
    data = data.dropna(how="all")
    return data

def add_indicators(data):
    if data is None or data.empty:
        return data

    if not TA_AVAILABLE:
        st.warning("⚠️ The 'ta' library is not installed. Indicators will be skipped.")
        data['RSI'] = pd.NA
        data['BB_high'] = pd.NA
        data['BB_low'] = pd.NA
        return data

    if 'Close' not in data.columns:
        st.error("Downloaded data does not contain a 'Close' column.")
        return data

    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    close_series = pd.to_numeric(close_series, errors='coerce')

    data = data.copy()
    data['Close'] = close_series.astype(float)

    if close_series.isna().all():
        st.error("Price data contains no numeric 'Close' values for the selected range.")
        data['RSI'] = pd.NA
        data['BB_high'] = pd.NA
        data['BB_low'] = pd.NA
        return data

    rsi = ta.momentum.RSIIndicator(close=close_series).rsi()
    bb = ta.volatility.BollingerBands(close=close_series)

    data['RSI'] = rsi
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()
    return data

def generate_signals(data):
    if data is None or data.empty:
        return data
    if not TA_AVAILABLE or 'RSI' not in data.columns or 'BB_low' not in data.columns or 'BB_high' not in data.columns:
        data['Signal'] = ""
        return data

    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    close_series = pd.to_numeric(close_series, errors='coerce')

    signals = []
    for i in range(len(data)):
        try:
            rsi_v = float(data['RSI'].iloc[i]) if pd.notna(data['RSI'].iloc[i]) else None
            bb_low = float(data['BB_low'].iloc[i]) if pd.notna(data['BB_low'].iloc[i]) else None
            bb_high = float(data['BB_high'].iloc[i]) if pd.notna(data['BB_high'].iloc[i]) else None
            close_v = float(close_series.iloc[i]) if pd.notna(close_series.iloc[i]) else None
        except Exception:
            rsi_v, bb_low, bb_high, close_v = None, None, None, None

        if rsi_v is not None and bb_low is not None and close_v is not None and close_v < bb_low and rsi_v < 30:
            signals.append("Buy")
        elif rsi_v is not None and bb_high is not None and close_v is not None and close_v > bb_high and rsi_v > 70:
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
    if data is None or data.empty:
        st.error("No price data returned for the selection. Try another ticker or adjust the dates.")
        st.session_state['analysis_done'] = False
    else:
        data = add_indicators(data)
        data = generate_signals(data)
        st.session_state['data'] = data

        csv_io = BytesIO(); data.to_csv(csv_io)
        st.session_state['csv_bytes'] = csv_io.getvalue()

        trade_log = data[data.get('Signal', "") != ""] if 'Signal' in data.columns else pd.DataFrame()
        trade_io = BytesIO(); trade_log.to_csv(trade_io)
        st.session_state['trade_csv_bytes'] = trade_io.getvalue()

        st.session_state['analysis_done'] = True

# --- Results ---
if st.session_state.get('analysis_done') and st.session_state['data'] is not None:
    st.subheader(f"Results for {ticker}")

    df = st.session_state['data']
    # Determine which columns are available for plotting
    base_cols = ['Close', 'BB_high', 'BB_low']
    plot_cols = [c for c in base_cols if c in df.columns]

    if len(plot_cols) == 0:
        st.warning("No chartable columns available.")
    else:
        # Prepare a safe, numeric, single-index DataFrame for charting
        chart_df = df[plot_cols].copy()
        # Coerce to numeric
        for c in plot_cols:
            chart_df[c] = pd.to_numeric(chart_df[c], errors='coerce')
        # Flatten MultiIndex columns if present
        if isinstance(chart_df.columns, pd.MultiIndex):
            chart_df.columns = ["_".join([str(x) for x in tup if x is not None and x != ""]).strip() for tup in chart_df.columns]
            plot_cols = list(chart_df.columns)
        # Ensure a single DatetimeIndex
        if isinstance(chart_df.index, pd.MultiIndex):
            tmp = chart_df.reset_index()
            # pick the first datetime-like column as the x-axis
            date_col = None
            for col in tmp.columns:
                if pd.api.types.is_datetime64_any_dtype(tmp[col]):
                    date_col = col
                    break
            if date_col is None:
                date_col = tmp.columns[0]
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
            chart_df = tmp.set_index(date_col)
        else:
            chart_df.index.name = 'Date'
        # Try standard line_chart first; if it fails, fall back to Altair
        try:
            st.line_chart(chart_df[plot_cols].dropna(how='all'))
        except Exception as e:
            import altair as alt
            st.warning(f"Chart fallback due to data shape: {e}")
            melted = chart_df.reset_index().melt('Date', value_vars=plot_cols, var_name='Series', value_name='Value')
            chart = alt.Chart(melted).mark_line().encode(
                x='Date:T', y='Value:Q', color='Series:N'
            )
            st.altair_chart(chart, use_container_width=True)

    st.write(df.tail())

    st.download_button(
        label="📥 Download CSV",
        data=st.session_state['csv_bytes'],
        file_name=f"{ticker}_analysis.csv",
        mime="text/csv",
        key="dl_csv"
    )

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
