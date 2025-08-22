# app.py — Stage 2 (Indicators + Signals Table + Safe Fetch)
# Focus: robust data fetch + indicators + compact summary.
# (Notifications, portfolio logs, and advanced UI come in Stage 3.)

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf

# Optional crypto support (safe import)
try:
    from pycoingecko import CoinGeckoAPI
except Exception:
    CoinGeckoAPI = None  # crypto will be disabled if not installed

# -------------------------
# App config
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Stage 2: Indicators + Signals)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.markdown(
    f"<h3 style='margin-top:0'>💡 {PAGE_TITLE}</h3>",
    unsafe_allow_html=True,
)

# -------------------------
# Utilities
# -------------------------
def _as_series(x) -> pd.Series:
    """Ensure plain Series (yfinance sometimes returns multiindex frames)."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if "Close" in x.columns:
            s = x["Close"].squeeze()
            return _as_series(s)
        # if a single column dataframe
        if x.shape[1] == 1:
            return x.iloc[:, 0]
    return pd.Series(dtype=float)


# -------------------------
# S&P 500 universe & top movers
# -------------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, match="Symbol")
        syms = tables[0]["Symbol"].astype(str).str.upper().str.strip().tolist()
        # Filter obvious weird symbols
        syms = [s.replace(".", "-") for s in syms if s.isalnum() or "-" in s]
        return syms
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def filter_valid_equities(symbols: List[str], period: str = "1mo") -> List[str]:
    """Return only symbols that yield recent price data."""
    valid = []
    for sym in symbols:
        try:
            s = _as_series(yf.download(sym, period=period, progress=False)["Close"])
            if len(s.dropna()) > 5:
                valid.append(sym)
        except Exception:
            continue
        # be gentle
        if len(valid) >= 200:
            break
    return valid

def get_top_tickers(n: int) -> List[str]:
    base = get_sp500_tickers()
    base = filter_valid_equities(base, period="1mo")
    if not base:
        return []
    try:
        df = yf.download(base, period="2mo", progress=False)["Close"]
        if isinstance(df, pd.Series):
            chg = df.pct_change().iloc[-1:].to_frame().T.squeeze()
        else:
            chg = df.pct_change().iloc[-1]
        top = chg.dropna().sort_values(ascending=False).head(n).index.tolist()
        return top
    except Exception:
        # fallback: sequentially compute 2-day change
        perf = {}
        for s in base:
            try:
                srs = _as_series(yf.download(s, period="2mo", progress=False)["Close"])
                if len(srs) >= 2:
                    perf[s] = float(srs.pct_change().iloc[-1])
            except Exception:
                pass
        return sorted(perf, key=lambda k: perf[k], reverse=True)[:n]

# -------------------------
# Crypto helpers (optional)
# -------------------------
@st.cache_data(show_spinner=False)
def list_top_coins(limit: int = 25) -> List[Tuple[str, str]]:
    """Return list of (id, symbol) for top market cap coins (lowercase symbols)."""
    if CoinGeckoAPI is None:
        return []
    try:
        cg = CoinGeckoAPI()
        data = cg.get_coins_markets(vs_currency="usd", order="market_cap_desc", per_page=limit, page=1)
        return [(row["id"], row["symbol"].upper()) for row in data]
    except Exception:
        return []

def fetch_coin_close(coin_id: str, days: int = 365) -> pd.Series:
    if CoinGeckoAPI is None:
        return pd.Series(dtype=float)
    cg = CoinGeckoAPI()
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="usd", days=days)
    prices = data.get("prices", [])
    if not prices:
        return pd.Series(dtype=float)
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("ts")["price"].asfreq("D").ffill()
    return df.rename("Close")

# -------------------------
# Indicators
# -------------------------
def sma(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger(series: pd.Series, win: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(win).mean()
    std = series.rolling(win).std()
    return ma, ma + num_std * std, ma - num_std * std

# -------------------------
# Equity & coin fetch (safe)
# -------------------------
def fetch_equity_close(sym: str, period: str) -> pd.Series:
    df = yf.download(sym, period=period, auto_adjust=False, progress=False)
    if isinstance(df, pd.DataFrame) and "Close" in df.columns:
        s = _as_series(df["Close"])
        return s.dropna()
    return pd.Series(dtype=float)

# -------------------------
# Analysis
# -------------------------
def analyze_one_close(symbol: str, s: pd.Series, rsi_ovr: float, rsi_obh: float) -> Dict:
    """Compute indicators and a simple BUY/SELL/HOLD label."""
    if s.empty or len(s) < 60:
        raise ValueError("not enough data")

    sma20 = sma(s, 20)
    sma50 = sma(s, 50)
    r = rsi(s, 14)
    bb_mid, bb_up, bb_dn = bollinger(s, 20, 2)

    cur = s.iloc[-1]
    p = {
        "RSI": float(r.iloc[-1]),
        "20 SMA": float(sma20.iloc[-1]),
        "50 SMA": float(sma50.iloc[-1]),
        "Close": float(cur),
        "BB Upper": float(bb_up.iloc[-1]),
        "BB Lower": float(bb_dn.iloc[-1]),
    }

    reasons = []
    if p["RSI"] < rsi_ovr:
        reasons.append(f"RSI < {rsi_ovr} (oversold)")
    if p["RSI"] > rsi_obh:
        reasons.append(f"RSI > {rsi_obh} (overbought)")
    # SMA cross checks
    prev20, prev50 = float(sma20.iloc[-2]), float(sma50.iloc[-2])
    if prev20 < prev50 and p["20 SMA"] >= p["50 SMA"]:
        reasons.append("20>50 bullish cross")
    if prev20 > prev50 and p["20 SMA"] <= p["50 SMA"]:
        reasons.append("20<50 bearish cross")
    if p["Close"] < p["BB Lower"]:
        reasons.append("Close < lower BB")
    if p["Close"] > p["BB Upper"]:
        reasons.append("Close > upper BB")

    text = " ".join(reasons).lower()
    signal = "HOLD"
    if any(k in text for k in ["oversold", "bullish", "lower bb <"]):
        signal = "BUY"
    if any(k in text for k in ["overbought", "bearish", "upper bb >"]):
        # a SELL reason should override BUY if both appear
        signal = "SELL" if signal == "HOLD" else signal

    p["Signal"] = signal
    p["Reasons"] = "; ".join(reasons) if reasons else "—"
    return p

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("### Settings")
    with st.expander("General", expanded=True):
        simulate_mode = st.checkbox("Simulate Trading Mode", True)
        debug = st.checkbox("Show debug logs", False)

    with st.expander("Analysis Options", expanded=True):
        scan_top = st.checkbox("Scan top N performers", False)
        top_n = st.slider("Top tickers to scan (stocks)", 5, 100, 25) if scan_top else 0
        include_crypto = st.checkbox("Include crypto in scan (optional)", False)
        top_coins = (
            st.slider("Top coins to include", 1, 25, 5) if include_crypto and CoinGeckoAPI else 0
        )

        min_rows = st.slider("Minimum data rows", 20, 150, 60)
        rsi_ovr = st.slider("RSI oversold threshold", 0, 100, 30)
        rsi_obh = st.slider("RSI overbought threshold", 0, 100, 70)

        # Universe selection
        base_universe = get_sp500_tickers()
        if scan_top:
            auto_list = get_top_tickers(top_n)
        else:
            auto_list = base_universe[:2]  # small default

        # Do NOT assign to st.session_state['tickers'] (avoids Streamlit API clash).
        tickers = st.multiselect("Choose tickers", base_universe, default=auto_list)

        # Date range
        period = st.selectbox("Date range", ["3mo", "6mo", "1y", "2y"], index=1)

# -------------------------
# Run
# -------------------------
if st.button("▶ Run Analysis", use_container_width=True):
    if not tickers and not (include_crypto and top_coins > 0):
        st.warning("Select at least one stock or enable crypto.")
        st.stop()

    results: Dict[str, Dict] = {}
    bads = []

    # 1) Stocks
    for sym in tickers:
        try:
            s = fetch_equity_close(sym, period)
            if len(s) < min_rows:
                raise ValueError("too few rows")
            summary = analyze_one_close(sym, s, rsi_ovr, rsi_obh)
            results[sym] = summary

            # chart
            sma20 = sma(s, 20)
            sma50 = sma(s, 50)
            _, bb_up, bb_dn = bollinger(s, 20, 2)

            st.markdown(f"#### 📈 {sym} Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s, name="Close"))
            fig.add_trace(go.Scatter(x=sma20.index, y=sma20, name="20 SMA"))
            fig.add_trace(go.Scatter(x=sma50.index, y=sma50, name="50 SMA"))
            fig.add_trace(
                go.Scatter(x=bb_up.index, y=bb_up, name="BB Upper", line=dict(dash="dot"))
            )
            fig.add_trace(
                go.Scatter(x=bb_dn.index, y=bb_dn, name="BB Lower", line=dict(dash="dot"))
            )
            st.plotly_chart(fig, use_container_width=True)

            badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[summary["Signal"]]
            st.markdown(f"**{badge} {sym} — {summary['Signal']}**")
            if debug:
                st.json(summary)
            st.divider()
        except Exception as e:
            bads.append(f"{sym}: {e}")

    # 2) Crypto (optional)
    if include_crypto and top_coins > 0 and CoinGeckoAPI:
        coin_list = list_top_coins(top_coins)
        for coin_id, short in coin_list:
            try:
                s = fetch_coin_close(coin_id, days=365 if period in ("1y", "2y") else 180)
                if len(s) < min_rows:
                    raise ValueError("too few rows")
                summary = analyze_one_close(short, s, rsi_ovr, rsi_obh)
                results[f"{short}-USD"] = summary

                sma20 = sma(s, 20)
                sma50 = sma(s, 50)
                _, bb_up, bb_dn = bollinger(s, 20, 2)

                st.markdown(f"#### 🪙 {short}-USD Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s.index, y=s, name="Close"))
                fig.add_trace(go.Scatter(x=sma20.index, y=sma20, name="20 SMA"))
                fig.add_trace(go.Scatter(x=sma50.index, y=sma50, name="50 SMA"))
                fig.add_trace(
                    go.Scatter(x=bb_up.index, y=bb_up, name="BB Upper", line=dict(dash="dot"))
                )
                fig.add_trace(
                    go.Scatter(x=bb_dn.index, y=bb_dn, name="BB Lower", line=dict(dash="dot"))
                )
                st.plotly_chart(fig, use_container_width=True)

                badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[summary["Signal"]]
                st.markdown(f"**{badge} {short}-USD — {summary['Signal']}**")
                if debug:
                    st.json(summary)
                st.divider()
            except Exception as e:
                bads.append(f"{short}-USD: {e}")

    # Summary table
    if results:
        st.markdown("### 📊 Summary of Trade Signals")
        df = pd.DataFrame(results).T[
            ["Signal", "Close", "RSI", "20 SMA", "50 SMA", "BB Lower", "BB Upper", "Reasons"]
        ]
        # compact styling with signal badges
        def _badge(sig: str) -> str:
            m = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}
            return m.get(sig, sig)
        df["Signal"] = df["Signal"].map(_badge)
        st.dataframe(df, use_container_width=True)

        # CSV download
        st.download_button(
            "⬇ Download results CSV",
            df.to_csv(index=True).encode(),
            file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        )

    # Warnings
    if bads:
        with st.expander("⚠️ Skipped items (no/insufficient data)", expanded=False):
            for msg in bads:
                st.write("•", msg)
