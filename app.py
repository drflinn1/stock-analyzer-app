# app.py — Stock Analyzer / Trading BOT (Pro)
# ---------------------------------------------------------------------
# Streamlit app that pulls OHLCV from yFinance, calculates RSI & Bollinger Bands,
# and now includes:
#   • Clean CSV exports (Excel‑friendly + UTF‑8)
#   • Trade logging (CSV/Parquet) with dry_run execution
#   • Optional live trading stub for Robinhood (env‑driven; off by default)
#   • Basic crypto support (e.g., BTC-USD, ETH-USD)
#   • Tax report (realized trades CSV with FIFO P/L columns)
#   • Alerts (email / SMS / Slack) — only send when NOT dry_run
# ---------------------------------------------------------------------

import csv
import io
import os
import re
import smtplib
import ssl
import unicodedata
from email.mime.text import MIMEText
from typing import Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ------------------------------ Page Setup ------------------------------
st.set_page_config(
    page_title="Stock Analyzer / Trading BOT (Pro)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Globals & Paths ------------------------------
EXPORT_DIR = os.environ.get("EXPORT_DIR", "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
TRADE_LOG_CSV = os.path.join(EXPORT_DIR, "trades_log.csv")
TRADE_LOG_PARQUET = os.path.join(EXPORT_DIR, "trades_log.parquet")
TAX_REPORT_CSV = os.path.join(EXPORT_DIR, "tax_report_fifo.csv")

# Robinhood (optional) — keep disabled unless env + toggle enable
RH_ENABLED = False  # hard off unless you wire real keys + set to True below
# Expected env if you turn RH on later: RH_USERNAME, RH_PASSWORD or device token flow

# Alert env (all optional; messages only sent when NOT dry_run)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_TO = os.getenv("SMTP_TO")

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")

# ------------------------------ Utilities ------------------------------
_CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")

def _normalize_text(x: Optional[object]) -> Optional[str]:
    if x is None:
        return None
    s = unicodedata.normalize("NFKC", str(x))
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _CONTROL_CHARS_RE.sub("", s)
    return s.strip()

def clean_dataframe_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_text(c) for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(_normalize_text)
    return df

def to_csv_bytes(df: pd.DataFrame, excel_friendly: bool = True) -> bytes:
    csv_str = df.to_csv(
        index=False,
        na_rep="",
        float_format="%.6f",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    return csv_str.encode("utf-8-sig" if excel_friendly else "utf-8")

# yFinance sometimes returns a MultiIndex; flatten it

def _ensure_plain_ohlcv_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(lvl):
                try:
                    df = df.xs(ticker, axis=1, level=lvl, drop_level=True)
                    break
                except Exception:
                    pass
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join([str(x) for x in tup if str(x) != ""]).strip() for tup in df.columns]
    ren = {c: c.title() for c in df.columns}
    ren["Adj Close"] = "Adj Close"
    return df.rename(columns=ren)

# ------------------------------ Indicators ------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0))
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(0.0)

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower

# ------------------------------ Data Fetch ------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 5)
def fetch_ohlcv(ticker: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end + timedelta(days=1),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_plain_ohlcv_columns(df, ticker)
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            idx = df.index.tz_localize(None)
        except Exception:
            idx = pd.DatetimeIndex(df.index)
        df.insert(0, "Date", idx)
    else:
        df.insert(0, "Date", pd.to_datetime(df.index, errors="coerce"))

    return df.reset_index(drop=True)

# ------------------------------ Trading Helpers ------------------------------

def _init_trade_log() -> pd.DataFrame:
    cols = [
        "timestamp", "symbol", "side", "quantity", "price", "fees", "strategy",
        "order_id", "dry_run", "note",
    ]
    if os.path.exists(TRADE_LOG_CSV):
        try:
            return pd.read_csv(TRADE_LOG_CSV)
        except Exception:
            return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)

@st.cache_data(show_spinner=False)
def get_trade_log() -> pd.DataFrame:
    return _init_trade_log()

def _append_trade(row: dict):
    df = _init_trade_log()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRADE_LOG_CSV, index=False)
    try:
        df.to_parquet(TRADE_LOG_PARQUET, index=False)
    except Exception:
        pass

# FIFO tax report based on trades_log; assumes symbol-level FIFO lots

def build_fifo_tax_report(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[
            "symbol","open_ts","close_ts","side","qty","open_price","close_price","proceeds","cost","pnl","holding_days","term"
        ])
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])  # ensure datetime
    trades["quantity"] = trades["quantity"].astype(float)
    trades["price"] = trades["price"].astype(float)

    results = []
    for sym, g in trades.sort_values("timestamp").groupby("symbol"):
        lots: list[Tuple[datetime,float,float]] = []  # (ts, qty, price)
        for _, r in g.iterrows():
            ts, side, qty, price = r["timestamp"], r["side"].lower(), float(r["quantity"]), float(r["price"])
            if side in {"buy", "long"}:
                lots.append((ts, qty, price))
            elif side in {"sell", "short_cover", "close"}:
                remaining = qty
                while remaining > 0 and lots:
                    open_ts, open_qty, open_price = lots[0]
                    take = min(open_qty, remaining)
                    proceeds = take * price
                    cost = take * open_price
                    pnl = proceeds - cost
                    holding_days = (ts - open_ts).days
                    term = "long" if holding_days >= 365 else "short"
                    results.append({
                        "symbol": sym,
                        "open_ts": open_ts,
                        "close_ts": ts,
                        "side": "sell",
                        "qty": take,
                        "open_price": round(open_price, 6),
                        "close_price": round(price, 6),
                        "proceeds": round(proceeds, 2),
                        "cost": round(cost, 2),
                        "pnl": round(pnl, 2),
                        "holding_days": holding_days,
                        "term": term,
                    })
                    # reduce lot
                    lots[0] = (open_ts, open_qty - take, open_price)
                    if lots[0][1] <= 1e-9:
                        lots.pop(0)
                    remaining -= take
                # if remaining > 0 and no lots, it's a short — skipped in this minimal FIFO
    return pd.DataFrame(results)

# ------------------------------ Alert Senders ------------------------------

def send_email(subject: str, body: str) -> bool:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_TO):
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = SMTP_TO
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True
    except Exception:
        return False

def send_slack(body: str) -> bool:
    if not SLACK_WEBHOOK:
        return False
    try:
        import requests
        requests.post(SLACK_WEBHOOK, json={"text": body}, timeout=10)
        return True
    except Exception:
        return False

def send_sms(body: str) -> bool:
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and TWILIO_TO):
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(to=TWILIO_TO, from_=TWILIO_FROM, body=body)
        return True
    except Exception:
        return False

# ------------------------------ UI: Header ------------------------------
st.title("📈 Stock Analyzer / Trading BOT (Pro)")
st.caption("RSI & Bollinger Bands • Clean Exports • Trades • Tax • Alerts • Crypto")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (e.g., AAPL or BTC-USD)", value="AAPL").strip().upper()

    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Start date", value=(datetime.now().date() - timedelta(days=365)))
    with col_dates[1]:
        end_date = st.date_input("End date", value=datetime.now().date())

    if end_date > datetime.now().date():
        end_date = datetime.now().date()

    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m", "1wk", "1mo"], index=0)

    st.markdown("---")
    st.subheader("Indicators")
    rsi_period = st.number_input("RSI Period", min_value=2, max_value=100, value=14)
    bb_window = st.number_input("BB Window", min_value=5, max_value=200, value=20)
    bb_std = st.number_input("BB Std Dev", min_value=1.0, max_value=4.0, value=2.0, step=0.5, format="%.1f")

    st.markdown("---")
    # Trading controls
    st.subheader("Trading (dry-run by default)")
    dry_run = st.checkbox("Dry run (no real orders)", value=True)
    side = st.selectbox("Side", ["BUY", "SELL"])  # simple spot buy/sell
    qty = st.number_input("Quantity", min_value=0.0, step=1.0, value=1.0, format="%.4f")
    strategy = st.text_input("Strategy tag", value="manual")
    note = st.text_input("Note", value="")

    st.markdown("---")
    run_btn = st.button("▶️ Run / Analyze", use_container_width=True, key="run_btn")

# ------------------------------ Analysis ------------------------------
if run_btn:
    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    with st.spinner(f"Fetching data for {ticker}..."):
        today = datetime.now().date()
        safe_end = min(end_date, today)
        df = fetch_ohlcv(
            ticker=ticker,
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(safe_end, datetime.min.time()),
            interval=interval,
        )

    if df.empty:
        st.warning("No data returned. Try a different date range or interval.")
        st.stop()

    # Indicators
    price_series = df["Close"].astype(float)
    df["RSI"] = rsi(price_series, period=int(rsi_period))
    bb_mid, bb_upper, bb_lower = bollinger_bands(price_series, window=int(bb_window), num_std=float(bb_std))
    df["BB_Middle"], df["BB_Upper"], df["BB_Lower"] = bb_mid, bb_upper, bb_lower

    # Add Ticker column for clarity/exports
    df.insert(1, "Ticker", ticker)

    desired = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume",
               "RSI", "BB_Middle", "BB_Upper", "BB_Lower"]
    cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
    df = df.loc[:, cols]

    st.subheader(f"Results: {ticker}")
    st.dataframe(df.tail(250), use_container_width=True, height=420)

    # Chart
    try:
        import altair as alt
        plot_df = df.dropna(subset=["Close"]).copy()
        plot_df["Date"] = pd.to_datetime(plot_df["Date"])
        base = alt.Chart(plot_df).encode(x="Date:T")
        close_line = base.mark_line().encode(y="Close:Q")
        bb_upper_line = base.mark_line(opacity=0.5).encode(y="BB_Upper:Q")
        bb_middle_line = base.mark_line(opacity=0.5).encode(y="BB_Middle:Q")
        bb_lower_line = base.mark_line(opacity=0.5).encode(y="BB_Lower:Q")
        st.altair_chart(close_line + bb_upper_line + bb_middle_line + bb_lower_line, use_container_width=True)
    except Exception:
        st.info("Altair not available or failed to render chart.")

    # ------------------ CSV Exports ------------------
    st.markdown("### 📥 Clean CSV Exports")
    out_df = df.copy()
    out_df["Date"] = pd.to_datetime(out_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    clean_df = clean_dataframe_text(out_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_excel = f"{ticker}_
