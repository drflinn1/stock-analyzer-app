# app.py — Stock Analyzer / Trading BOT (CSV-clean fixed)
# ---------------------------------------------------------------------
# Streamlit app that pulls OHLCV from yFinance, calculates RSI & Bollinger Bands,
# displays results, and provides *clean* CSV exports with Excel‑friendly UTF‑8+BOM
# and plain UTF‑8 options to prevent “weird characters”.
# ---------------------------------------------------------------------

import csv
import io
import re
import unicodedata
from typing import Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ------------------------------ Page Setup ------------------------------
st.set_page_config(
    page_title="Stock Analyzer / Trading BOT",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------ Utilities ------------------------------
# Remove hidden characters and normalize text to avoid "weird symbols" in CSVs
_CONTROL_CHARS_RE = re.compile(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]')
_ZERO_WIDTH_RE = re.compile(r'[\u200B-\u200D\u2060\uFEFF]')

def _normalize_text(x: Optional[object]) -> Optional[str]:
    if x is None:
        return None
    s = unicodedata.normalize("NFKC", str(x))
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _CONTROL_CHARS_RE.sub("", s)
    return s.strip()

def clean_dataframe_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all string-like columns to remove BOM/zero-width/control chars and normalize."""
    df = df.copy()
    # Clean column names
    df.columns = [_normalize_text(c) for c in df.columns]
    # Clean string-like columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(_normalize_text)
    return df

def to_csv_bytes(df: pd.DataFrame, excel_friendly: bool = True) -> bytes:
    """
    Create CSV bytes with robust defaults:
      - QUOTE_MINIMAL + \n newlines
      - Empty string for NaNs
      - Stable float formatting
      - UTF-8 with BOM for Excel (excel_friendly=True),
        or plain UTF-8 (excel_friendly=False)
    """
    df = df.copy()
    csv_str = df.to_csv(
        index=False,
        na_rep="",
        float_format="%.6f",
        quoting=csv.QUOTE_MINIMAL,
        line_terminator="\n",
    )
    if excel_friendly:
        # Excel sniffs UTF‑8 reliably when BOM is present
        return csv_str.encode("utf-8-sig")  # adds BOM
    return csv_str.encode("utf-8")          # no BOM


# ------------------------------ Indicators ------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (default 14)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0))
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(0.0)

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands: middle (SMA), upper, lower."""
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


# ------------------------------ Data Fetch ------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 5)
def fetch_ohlcv(ticker: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from yFinance.
    Returns a DataFrame with Date index (tz-naive) and standard columns.
    """
    # yfinance sometimes returns tz-aware index; make tz-naive for consistency
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end + timedelta(days=1),  # make end inclusive
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure column names match expected
    df = df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
    })

    # Normalize the index to tz-naive and push to a column
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            idx = df.index.tz_localize(None)
        except Exception:
            idx = pd.DatetimeIndex(df.index)
        df = df.copy()
        df.insert(0, "Date", idx)
    else:
        df.insert(0, "Date", pd.to_datetime(df.index, errors="coerce"))

    return df.reset_index(drop=True)


# ------------------------------ UI ------------------------------
st.title("📈 Stock Analyzer / Trading BOT")
st.caption("RSI & Bollinger Bands • Clean CSV Exports • yFinance Data")

with st.sidebar:
    st.header("Settings")

    ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Start date", value=(datetime.now().date() - timedelta(days=365)))
    with col_dates[1]:
        end_date = st.date_input("End date", value=datetime.now().date())

    interval = st.selectbox(
        "Interval",
        options=["1d", "1h", "30m", "15m", "5m", "1wk", "1mo"],
        index=0,
        help="Supported yFinance intervals"
    )

    st.markdown("---")
    st.subheader("Indicators")
    rsi_period = st.number_input("RSI Period", min_value=2, max_value=100, value=14, step=1)
    bb_window = st.number_input("BB Window", min_value=5, max_value=200, value=20, step=1)
    bb_std = st.number_input("BB Std Dev", min_value=1.0, max_value=4.0, value=2.0, step=0.5, format="%.1f")

    st.markdown("---")
    run_btn = st.button("▶️ Run / Analyze", use_container_width=True)


# ------------------------------ Analysis ------------------------------
if run_btn:
    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_ohlcv(
            ticker=ticker,
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.min.time()),
            interval=interval
        )

    if df.empty:
        st.warning("No data returned. Try a different date range or interval.")
        st.stop()

    # Calculate indicators
    price_series = df["Close"].astype(float)
    df["RSI"] = rsi(price_series, period=int(rsi_period))
    bb_mid, bb_upper, bb_lower = bollinger_bands(price_series, window=int(bb_window), num_std=float(bb_std))
    df["BB_Middle"] = bb_mid
    df["BB_Upper"] = bb_upper
    df["BB_Lower"] = bb_lower

    # Add Ticker column for clarity/exports
    df.insert(1, "Ticker", ticker)

    # Reorder columns for readability if present
    desired = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume",
               "RSI", "BB_Middle", "BB_Upper", "BB_Lower"]
    cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
    df = df[cols]

    # Display
    st.subheader(f"Results: {ticker}")
    st.dataframe(df.tail(200), use_container_width=True, height=400)

    # Simple chart (Close + BB)
    try:
        import altair as alt
        plot_df = df.dropna(subset=["Close"]).copy()
        plot_df["Date"] = pd.to_datetime(plot_df["Date"])
        base = alt.Chart(plot_df).encode(x="Date:T")
        close_line = base.mark_line().encode(y=alt.Y("Close:Q", title="Price"))
        bb_upper_line = base.mark_line(opacity=0.5).encode(y="BB_Upper:Q")
        bb_middle_line = base.mark_line(opacity=0.5).encode(y="BB_Middle:Q")
        bb_lower_line = base.mark_line(opacity=0.5).encode(y="BB_Lower:Q")
        st.altair_chart(close_line + bb_upper_line + bb_middle_line + bb_lower_line, use_container_width=True)
    except Exception:
        st.info("Altair not available or failed to render chart; skipping chart.")

    # ---------------- CSV Exports (CLEAN) ----------------
    st.markdown("### 📥 Clean CSV Exports")

    # Ensure Date is string for consistent CSV output (avoids Excel timezone surprises)
    out_df = df.copy()
    out_df["Date"] = pd.to_datetime(out_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    # Clean text columns (remove hidden chars, normalize)
    clean_df = clean_dataframe_text(out_df)

    # Build file names with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_excel = f"{ticker}_analysis_excel_{ts}.csv"
    fname_utf8 = f"{ticker}_analysis_utf8_{ts}.csv"

    csv_bytes_excel = to_csv_bytes(clean_df, excel_friendly=True)
    csv_bytes_utf8 = to_csv_bytes(clean_df, excel_friendly=False)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download CSV (Excel‑friendly UTF‑8+BOM)",
            data=csv_bytes_excel,
            file_name=fname_excel,
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download CSV (UTF‑8, no BOM)",
            data=csv_bytes_utf8,
            file_name=fname_utf8,
            mime="text/csv",
            use_container_width=True,
        )

    st.success("CSV export fixed and ready. If you still see odd characters in a specific column, it’s coming from upstream data—now trivially cleaned here.")


# ------------------------------ Footer ------------------------------
with st.expander("About this app"):
    st.write(
        "This version includes robust CSV cleaning to avoid hidden Unicode/control characters "
        "and provides both Excel‑friendly UTF‑8+BOM and plain UTF‑8 exports."
    )
