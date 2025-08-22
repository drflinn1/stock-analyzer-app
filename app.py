# app.py — Stock & Crypto Analyzer (S&P scan + indicators + signals)

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from pycoingecko import CoinGeckoAPI

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Analyzer Bot", layout="wide")
st.title("📈 Stock Analyzer Bot (Live Trading + Tax Logs)")

# ---------------------------------------------------------
# Helpers: S&P 500 & Top Movers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia (html-table first, bs4 fallback)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df_list = pd.read_html(url, flavor="bs4", attrs={"class": "wikitable"})
        return df_list[0]["Symbol"].tolist()
    except Exception:
        try:
            resp = requests.get(url, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"class": "wikitable"})
            return [row.find_all("td")[0].text.strip() for row in table.find_all("tr")[1:]]
        except Exception:
            return []

def get_top_tickers(n: int) -> list[str]:
    """Return top N daily performers from the S&P universe using 2 days of data."""
    symbols = get_sp500_tickers()
    if not symbols:
        return []
    try:
        df = yf.download(symbols, period="2d", progress=False)["Close"]
        if isinstance(df, pd.Series):
            changes = df.pct_change().iloc[-1:].to_frame().T
        else:
            changes = df.pct_change().iloc[-1]
        return changes.dropna().sort_values(ascending=False).head(n).index.tolist()
    except Exception:
        # Slow but safe fallback: one-by-one
        perf = {}
        for sym in symbols:
            try:
                tmp = yf.download(sym, period="2d", progress=False)["Close"]
                if len(tmp) >= 2:
                    perf[sym] = float(tmp.pct_change().iloc[-1])
            except Exception:
                continue
        return sorted(perf, key=lambda k: perf[k], reverse=True)[:n]

# ---------------------------------------------------------
# Indicators
# ---------------------------------------------------------
def simple_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def simple_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = simple_sma(series, window)
    std = series.rolling(window).std()
    return sma, sma + num_std * std, sma - num_std * std

# ---------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except Exception:
            df = df.iloc[:, df.columns.get_level_values(1) == ticker]
            df.columns = df.columns.get_level_values(0)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")

    close = df["Close"].squeeze()
    sma20, bbU, bbL = bollinger_bands(close)
    df["sma_20"], df["bb_upper"], df["bb_lower"] = sma20, bbU, bbL
    df["sma_50"] = simple_sma(close, 50)
    df["rsi"] = simple_rsi(close)
    return df.dropna()

@st.cache_data(show_spinner=False)
def get_crypto_history(coin_id: str, vs_currency: str = "usd", days: int = 180) -> pd.DataFrame:
    cg = CoinGeckoAPI()
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    df = pd.DataFrame(data["prices"], columns=["time", "price"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    # make it look like OHLC for indicator reuse
    df = df.rename(columns={"price": "Close"})
    df["Open"] = df["High"] = df["Low"] = df["Close"]
    sma20, bbU, bbL = bollinger_bands(df["Close"])
    df["sma_20"], df["bb_upper"], df["bb_lower"] = sma20, bbU, bbL
    df["sma_50"] = simple_sma(df["Close"], 50)
    df["rsi"] = simple_rsi(df["Close"])
    return df.dropna()

# ---------------------------------------------------------
# Signal Engine
# ---------------------------------------------------------
def analyze_frame(df: pd.DataFrame, rsi_ovr: float, rsi_obh: float) -> dict:
    """Return dict with RSI/SMA/Signal/Reasons for the last row of df."""
    if not isinstance(df, pd.DataFrame) or len(df) < 5:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    rsi = float(cur["rsi"])
    sma20_cur, sma50_cur = float(cur["sma_20"]), float(cur["sma_50"])
    price = float(cur["Close"])

    reasons = []
    if rsi < rsi_ovr:
        reasons.append(f"RSI below {rsi_ovr} (oversold)")
    if rsi > rsi_obh:
        reasons.append(f"RSI above {rsi_obh} (overbought)")
    if prev["sma_20"] < prev["sma_50"] <= sma20_cur:
        reasons.append("20 SMA crossed above 50 SMA (bullish)")
    if prev["sma_20"] > prev["sma_50"] >= sma20_cur:
        reasons.append("20 SMA crossed below 50 SMA (bearish)")
    if price < float(cur["bb_lower"]):
        reasons.append("Price below lower BB")
    if price > float(cur["bb_upper"]):
        reasons.append("Price above upper BB")

    text = "; ".join(reasons).lower()
    signal = "HOLD"
    if any(k in text for k in ["buy", "bullish", "oversold", "below lower bb"]):
        signal = "BUY"
    if any(k in text for k in ["sell", "bearish", "overbought", "above upper bb"]):
        signal = "SELL"

    return {
        "RSI": round(rsi, 2),
        "20 SMA": round(sma20_cur, 2),
        "50 SMA": round(sma50_cur, 2),
        "Signal": signal,
        "Reasons": "; ".join(reasons),
    }

# ---------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    with st.expander("General", expanded=True):
        simulate_mode = st.checkbox("Simulate Trading Mode", True)
        debug_mode = st.checkbox("Show debug logs", False)

    with st.expander("Analysis Options", expanded=True):
        scan_top = st.checkbox("Scan top N performers", False)
        top_n = st.slider("Top tickers to scan (stocks)", 5, 50, 10) if scan_top else 0

        include_crypto = st.checkbox("Include crypto in scan (optional)", False)
        top_coins = st.slider("Top coins to include", 1, 20, 5) if include_crypto else 0

        min_rows = st.slider("Minimum data rows", 10, 100, 30)
        rsi_ovr = st.slider("RSI oversold threshold", 0, 100, 30)
        rsi_obh = st.slider("RSI overbought threshold", 0, 100, 70)

        options = ["3mo", "6mo", "1y"]
        # read query params if present
        qp = dict(st.query_params)
        qs_period = qp.get("period", "")
        default_period = qs_period if qs_period in options else "6mo"

        # build the stock universe
        if scan_top:
            universe = get_top_tickers(top_n)
        else:
            universe = get_sp500_tickers()
        if not universe:
            universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

        qs_tickers = qp.get("tickers", "")
        if qs_tickers:
            default_list = qs_tickers.split(",")
        else:
            default_list = universe[:2]

        tickers = st.multiselect("Choose tickers", universe, default=default_list)
        period = st.selectbox("Date range", options, index=options.index(default_period))

# ---------------------------------------------------------
# Main Run
# ---------------------------------------------------------
badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
st.markdown(f"### {'🔴 SIM' if simulate_mode else '🟢 LIVE'}  Stock Analyzer")

if st.button("▶ Run Analysis", use_container_width=True):
    if not tickers and not include_crypto:
        st.warning("Select at least one ticker or enable crypto scan.")
        st.stop()

    results = {}
    # STOCKS
    for tkr in tickers:
        try:
            df = get_stock_data(tkr, period)
            if len(df) < min_rows:
                st.warning(f"{tkr}: Not enough data, skipped.")
                continue
            summ = analyze_frame(df, rsi_ovr, rsi_obh)
            if not summ:
                continue
            results[tkr] = summ

            # chart
            st.markdown(f"#### 📈 {tkr} Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="20 SMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="50 SMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(dash="dot")))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**{badge[summ['Signal']]} {tkr} – {summ['Signal']}**")
            if debug_mode:
                st.json(summ)
            st.divider()
        except Exception as e:
            st.error(f"{tkr} failed: {e}")

    # CRYPTO (optional)
    if include_crypto:
        try:
            cg = CoinGeckoAPI()
            coins = cg.get_coins_markets(vs_currency="usd", order="market_cap_desc", per_page=top_coins, page=1)
            for coin in coins:
                coin_id = coin["id"]
                symbol = coin["symbol"].upper() + "-USD"
                try:
                    cdf = get_crypto_history(coin_id, "usd", days=180 if period == "6mo" else 365)
                    if len(cdf) < min_rows:
                        st.warning(f"{symbol}: Not enough data, skipped.")
                        continue
                    summ = analyze_frame(cdf, rsi_ovr, rsi_obh)
                    if not summ:
                        continue
                    results[symbol] = summ

                    st.markdown(f"#### 🪙 {symbol} Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["Close"], name="Close"))
                    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["sma_20"], name="20 SMA"))
                    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["sma_50"], name="50 SMA"))
                    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["bb_upper"], name="BB Upper", line=dict(dash="dot")))
                    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["bb_lower"], name="BB Lower", line=dict(dash="dot")))
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(f"**{badge[summ['Signal']]} {symbol} – {summ['Signal']}**")
                    if debug_mode:
                        st.json(summ)
                    st.divider()
                except Exception as ce:
                    st.error(f"{symbol} failed: {ce}")
        except Exception as e:
            st.error(f"Crypto scan failed: {e}")

    # SUMMARY
    if results:
        res_df = pd.DataFrame(results).T
        # badge column
        res_df.insert(0, "Signal Badge", res_df["Signal"].map(badge))
        st.markdown("### 📊 Summary of Trade Signals")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("⬇ Download CSV", res_df.to_csv().encode(), "stock_analysis_results.csv")
    else:
        st.info("No analysable results were produced with the current settings.")
