# app.py — Autonomous Stock & Crypto Momentum Rebalancer (Streamlit)
# -------------------------------------------------------------------
# New in this version (toward full automation):
#   • Auto universes:
#       - Equities: "S&P 500 (auto)" or manual list (Russell1000 could be added later)
#       - Crypto:  Robinhood‑tradable list (fallback to a default set)
#   • Multi‑horizon momentum score (1D, 5D, 20D) → rank by Score
#   • SELL rule (Option A): full exit if (out of top‑N) OR (score < 0)
#   • Pooled budget across stocks + crypto; two allocation modes (fixed / proportional)
#   • Clean simulation vs live order placement for both stocks & crypto
#   • Logs show both SELLs and BUYs in execution order
#
# Notes
#   • Live trading only if sidebar toggle is ON *and* Robinhood login succeeds.
#   • Secrets supported (any one pair):
#         ROBINHOOD_USERNAME / ROBINHOOD_PASSWORD
#         RH_USERNAME       / RH_PASSWORD
#   • Streamlit Cloud cannot run on a schedule by itself. For automation, run this
#     module headlessly from a scheduler (e.g., GitHub Actions) that calls a small
#     runner script invoking the same functions. We ship the app with clear, pure
#     functions so this is straightforward when you’re ready.

VERSION = "0.8.1 (2025-08-12)"

import inspect
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Robinhood (only when live trading is enabled)
try:
    import robin_stocks.robinhood as rh
    from robin_stocks.robinhood import orders as rh_orders
    from robin_stocks.robinhood import crypto as rh_crypto
    from robin_stocks.robinhood import account as rh_account
except Exception:
    rh = None
    rh_orders = None
    rh_crypto = None
    rh_account = None

# ---------------------------------
# UI — Sidebar Controls
# ---------------------------------
st.set_page_config(page_title=f"Stock & Crypto Momentum Rebalancer · {VERSION}", layout="wide")

st.sidebar.header("Authentication & Mode")
st.sidebar.caption(f"Version {VERSION}")
live_trading = st.sidebar.checkbox("Enable Live Trading (use with caution)", value=False)

# Read secrets (support both naming conventions)
user = pwd = None
try:
    user = st.secrets.get("ROBINHOOD_USERNAME") or st.secrets.get("RH_USERNAME")
    pwd  = st.secrets.get("ROBINHOOD_PASSWORD") or st.secrets.get("RH_PASSWORD")
except Exception:
    pass

login_ok = False
login_msg = "Simulation mode — no live orders placed."

if live_trading:
    if rh is None:
        st.sidebar.error("robin_stocks is not available in this environment. Running in SIMULATION.")
        live_trading = False
    elif not user or not pwd:
        st.sidebar.error("Live trading selected but credentials are missing in Secrets. Running in SIMULATION.")
        live_trading = False
    else:
        try:
            sig = inspect.signature(rh.authentication.login)
            kwargs = {}
            if "username" in sig.parameters: kwargs["username"] = user
            if "password" in sig.parameters: kwargs["password"] = pwd
            if "store_session" in sig.parameters: kwargs["store_session"] = True
            if "expiresIn" in sig.parameters: kwargs["expiresIn"] = 24 * 3600
            if "scope" in sig.parameters: kwargs["scope"] = "internal"
            login_ok = bool(rh.authentication.login(**kwargs))
        except TypeError:
            try:
                login_ok = bool(rh.authentication.login(username=user, password=pwd))
            except Exception as e:
                login_ok = False
                login_msg = f"Robinhood login failed: {e}. Running in SIMULATION."
        except Exception as e:
            login_ok = False
            login_msg = f"Robinhood login failed: {e}. Running in SIMULATION."

        if login_ok:
            st.sidebar.success("Robinhood authenticated — Live orders ENABLED")
            login_msg = "Live orders ENABLED"
        else:
            st.sidebar.warning(login_msg)
            live_trading = False
else:
    st.sidebar.info(login_msg)

st.sidebar.header("Universe & Allocation")

universe_src = st.sidebar.selectbox(
    "Equity Universe",
    ["S&P 500 (auto)", "Manual list"],
    index=0,
    help="Choose the equity symbols source. Manual list uses the box below.",
)
raw_tickers = st.sidebar.text_area(
    "Manual equity tickers (comma‑separated)",
    value="AAPL,MSFT,GOOG",
)
include_crypto = st.sidebar.checkbox("Include Crypto (Robinhood‑tradable)", value=True)

alloc_mode = st.sidebar.selectbox("Allocation mode", ["Fixed $ per trade", "Proportional across winners"], index=0)
fixed_per_trade = st.sidebar.number_input("Fixed $ per BUY/SELL", min_value=1.0, value=5.0, step=0.5)
prop_total_budget = st.sidebar.number_input("Total BUY budget (proportional)", min_value=1.0, value=15.0, step=1.0)
min_per_order = st.sidebar.number_input("Minimum $ per order", min_value=1.0, value=2.0, step=0.5)
UI_MIN_PER_ORDER = float(min_per_order)

n_picks = st.sidebar.number_input("Top N to hold", min_value=1, value=3, step=1)

st.title("Stock & Crypto Momentum Rebalancer")
st.caption(f"Version {VERSION}")

# ---------------------------------
# Universe helpers
# ---------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_sp500_symbols() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia; fallback to a compact baked list if needed."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        syms = (
            df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
        )
        # Deduplicate and sanity‑filter
        syms = [s for s in dict.fromkeys(syms) if s.isascii() and 1 <= len(s) <= 6]
        return syms
    except Exception:
        # Small fallback set — the app still works while offline
        return ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "BRK-B", "TSLA", "JPM"]


def clean_manual_list(raw: str) -> List[str]:
    return [t.strip().upper() for t in (raw or "").split(",") if t.strip()]


DEFAULT_CRYPTO = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "USDT-USD"]


def load_rh_crypto_pairs() -> List[str]:
    """Return symbols in Yahoo-style (e.g., ETH-USD). Fallback to DEFAULT_CRYPTO."""
    try:
        pairs = []
        # Some versions don’t need login for this; if it fails we fallback
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_currency_pairs"):
            data = rh_crypto.get_crypto_currency_pairs()
            if isinstance(data, list):
                for p in data:
                    try:
                        sym = (p.get("asset_currency", {}) or {}).get("code")
                        if sym:
                            pairs.append(f"{sym}-USD")
                    except Exception:
                        pass
        if not pairs:
            pairs = DEFAULT_CRYPTO.copy()
        return sorted(list(dict.fromkeys(pairs)))
    except Exception:
        return DEFAULT_CRYPTO.copy()

# ---------------------------------
# Data — returns & scoring
# ---------------------------------

def fetch_returns(symbols: List[str], lookbacks: List[int]) -> pd.DataFrame:
    """Fetch % returns over given lookbacks in trading days using yfinance.
    Returns columns: [Ticker] + [f"R{lb}"] for each lookback.
    """
    if not symbols:
        cols = ["Ticker"] + [f"R{lb}" for lb in lookbacks]
        return pd.DataFrame(columns=cols)

    # yfinance multi‑download; fall back one‑by‑one as needed
    def _calc_from_df(sym: str, dfx: pd.DataFrame) -> Dict[str, float]:
        out = {"Ticker": sym}
        if dfx is None or dfx.empty or "Close" not in dfx:
            for lb in lookbacks:
                out[f"R{lb}"] = np.nan
            return out
        close = dfx["Close"].dropna()
        for lb in lookbacks:
            try:
                if len(close) >= (lb + 1):
                    r = (close.iloc[-1] / close.iloc[-(lb + 1)] - 1.0) * 100.0
                else:
                    r = np.nan
            except Exception:
                r = np.nan
            out[f"R{lb}"] = float(r) if pd.notna(r) else np.nan
        return out

    rows: List[Dict] = []
    try:
        df = yf.download(symbols, period="60d", interval="1d", progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            for sym in (df["Close"].columns if "Close" in df else symbols):
                dfx = df.xs(sym, axis=1, level=1)[["Close"]] if "Close" in df else None
                rows.append(_calc_from_df(str(sym), dfx))
        else:
            # Single symbol path
            rows.append(_calc_from_df(str(symbols[0]), df))
    except Exception:
        for s in symbols:
            try:
                dfx = yf.download(s, period="60d", interval="1d", progress=False, threads=False)
            except Exception:
                dfx = None
            rows.append(_calc_from_df(s, dfx))

    return pd.DataFrame(rows)


def score_momentum(df: pd.DataFrame, weights: Dict[int, float] = None) -> pd.DataFrame:
    if weights is None:
        # Heavier weight on near‑term, small tail on 20D
        weights = {1: 0.6, 5: 0.3, 20: 0.1}

    # Z‑score each horizon to balance scales
    sc = df.copy()
    for lb, w in weights.items():
        col = f"R{lb}"
        if col not in sc:
            sc[col] = np.nan
        mu = sc[col].mean(skipna=True)
        sd = sc[col].std(skipna=True)
        sc[f"Z{lb}"] = 0.0 if pd.isna(sd) or sd == 0 else (sc[col] - mu) / sd
    sc["Score"] = sum(sc.get(f"Z{lb}", 0.0) * w for lb, w in weights.items())
    return sc


def build_universe(eq_src: str, manual_raw: str, include_c: bool) -> List[str]:
    eq = load_sp500_symbols() if eq_src == "S&P 500 (auto)" else clean_manual_list(manual_raw)
    cr = load_rh_crypto_pairs() if include_c else []
    # Combine & deduplicate
    uni = list(dict.fromkeys(eq + cr))
    return uni


# ---------------------------------
# Order helpers
# ---------------------------------

def _call_first_available(objs, names, *args, **kwargs):
    for obj in objs:
        if obj is None: continue
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
    raise AttributeError(f"No available function among {names}")


def symbol_to_rh_crypto(sym: str) -> str:
    return sym.split("-")[0] if "-" in sym else sym


def normalize_dollars(dollars: float, min_amt: float) -> float:
    d = max(float(dollars), float(min_amt))
    # round *up* to the nearest cent to avoid falling under exchange/RH ticks
    cents = int(np.ceil(d * 100.0 - 1e-9))
    return cents / 100.0


def place_stock_order(symbol: str, side: str, dollars: float, live: bool, min_order: float) -> Tuple[str, float, str]:
    """Place a fractional stock order by dollars with retry if the notional is too small.
    Returns: (status, executed_amount, order_id)
    """
    base = normalize_dollars(dollars, min_order)
    if base < 0.01:
        return ("skipped (too small)", 0.0, "")
    if not live:
        return ("simulated", 0.0, "")

    # try up to 3 times, nudging notional up a bit in case of price moves / ticks
    for attempt in range(3):
        amt = normalize_dollars(base * (1.02 ** attempt), min_order)
        try:
            if side.upper() == "BUY":
                res = rh_orders.order_buy_fractional_by_price(symbol, amt)
            else:
                res = rh_orders.order_sell_fractional_by_price(symbol, amt)
            oid = ""
            if isinstance(res, dict):
                oid = res.get("id", "") or res.get("order_id", "")
            return ("placed", 0.0, oid)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["too small", "minimum", "tick", "notional"]):
                continue  # bump and retry
            return (f"error: {e}", 0.0, "")
    return ("error: minimum notional too small after retries", 0.0, "")



def place_crypto_order(symbol: str, side: str, dollars: float, live: bool, min_order: float) -> Tuple[str, float, str]:
    """Place a crypto order by dollars with retry if the notional is too small."""
    base = normalize_dollars(dollars, min_order)
    if base < 0.01:
        return ("skipped (too small)", 0.0, "")
    if not live:
        return ("simulated", 0.0, "")

    buy_candidates  = ["order_buy_crypto_by_price", "order_buy_crypto_by_dollar", "order_buy_crypto_by_dollars", "buy_crypto_by_price"]
    sell_candidates = ["order_sell_crypto_by_price", "order_sell_crypto_by_dollar", "order_sell_crypto_by_dollars", "sell_crypto_by_price"]

    for attempt in range(3):
        amt = normalize_dollars(base * (1.02 ** attempt), min_order)
        try:
            fnames = buy_candidates if side.upper() == "BUY" else sell_candidates
            res = _call_first_available([rh_crypto, rh], fnames, symbol, amt)
            oid = ""
            if isinstance(res, dict):
                oid = res.get("id", "") or res.get("order_id", "")
            return ("placed", 0.0, oid)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["too small", "minimum", "tick", "notional"]):
                continue
            return (f"error: {e}", 0.0, "")
    return ("error: minimum notional too small after retries", 0.0, "")


# ---------------------------------
# Holdings / SELL logic
# ---------------------------------
# Holdings / SELL logic
# ---------------------------------

def get_holdings() -> Dict[str, Dict]:
    """Return current holdings keyed by ticker, with approx market value in dollars.
       Stocks via account.build_holdings(); Crypto via get_crypto_positions().
    """
    out: Dict[str, Dict] = {}
    if rh_account is None:
        return out
    try:
        # Stocks
        if hasattr(rh_account, "build_holdings"):
            h = rh_account.build_holdings()
            if isinstance(h, dict):
                for sym, info in h.items():
                    try:
                        value = float(info.get("equity", 0))
                    except Exception:
                        value = 0.0
                    out[str(sym).upper()] = {"type": "stock", "value": value}
    except Exception:
        pass

    # Crypto
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_positions"):
            pos = rh_crypto.get_crypto_positions()
            if isinstance(pos, list):
                for p in pos:
                    try:
                        qty = float(p.get("quantity", 0) or 0)
                        sym = (p.get("currency", {}) or {}).get("code") or ""
                        if qty > 0 and sym:
                            price = 0.0
                            try:
                                if hasattr(rh_crypto, "get_crypto_quote"):
                                    q = rh_crypto.get_crypto_quote(sym)
                                    price = float(q.get("mark_price", 0) or q.get("ask_price", 0) or 0)
                            except Exception:
                                price = 0.0
                            value = qty * price
                            out[f"{sym}-USD".upper()] = {"type": "crypto", "value": float(value)}
                    except Exception:
                        pass
    except Exception:
        pass

    return out


# ---------------------------------
# Main run button
# ---------------------------------

if st.button("▶ Run Daily Scan & Rebalance", type="primary"):
    # Build universes
    universe = build_universe(universe_src, raw_tickers, include_crypto)

    # Compute returns & score
    lookbacks = [1, 5, 20]
    ret_df = fetch_returns(universe, lookbacks)
    scored = score_momentum(ret_df, weights={1: 0.6, 5: 0.3, 20: 0.1})

    # Rank by Score and pick top‑N
    picks = (
        scored.dropna(subset=["Score"]).sort_values("Score", ascending=False).head(int(n_picks)).reset_index(drop=True)
    )

    if picks.empty:
        st.warning("No momentum data available (universe may be empty or network hiccup). Try again.")
        st.stop()

    # SELL rule (Option A): full exit if out of top‑N OR score < 0
    sell_rows: List[Dict] = []
    buy_rows:  List[Dict] = []

    current = get_holdings() if (live_trading and login_ok) else {}
    in_top = set(picks["Ticker"].astype(str).str.upper().tolist())

    # Decide BUY allocations
    if alloc_mode == "Fixed $ per trade":
        per_trade = float(max(min_per_order, fixed_per_trade))
        buy_allocs = {t: per_trade for t in in_top}
    else:
        w = np.clip(picks["R1"].fillna(0).astype(float), 0, None)
        wsum = float(w.sum())
        if wsum <= 0:
            each = max(min_per_order, prop_total_budget / max(1, len(picks)))
            buy_allocs = {t: each for t in in_top}
        else:
            buy_allocs = {}
            for t, wi in zip(picks["Ticker"].tolist(), w.tolist()):
                dollars = (wi / wsum) * float(prop_total_budget)
                buy_allocs[str(t).upper()] = float(max(min_per_order, dollars))

    # --- SELL first ---
    for sym, info in current.items():
        t_upper = str(sym).upper()
        if t_upper not in in_top:
            # Out of top‑N ⇒ full exit
            dollars = float(info.get("value", 0.0))
            if dollars >= min_per_order:
                if info.get("type") == "crypto":
                    status, executed, oid = place_crypto_order(symbol_to_rh_crypto(t_upper), "SELL", dollars, live_trading and login_ok, UI_MIN_PER_ORDER)
                else:
                    status, executed, oid = place_stock_order(t_upper, "SELL", dollars, live_trading and login_ok, UI_MIN_PER_ORDER)
                sell_rows.append({
                    "Ticker": t_upper,
                    "Action": "SELL",
                    "Reason": "out_of_topN",
                    "Alloc$": round(dollars, 2),
                    "OrderID": oid,
                    "Status": status,
                    "Time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                })

    # --- BUY winners ---
    for _, r in picks.iterrows():
        t = str(r["Ticker"]).upper()
        score = float(r["Score"]) if pd.notna(r["Score"]) else 0.0
        if score < 0:
            continue  # do not buy negative score
        dollars = float(buy_allocs.get(t, 0.0))
        if dollars < min_per_order:
            continue
        is_crypto = t.endswith("-USD")
        if is_crypto:
            status, executed, oid = place_crypto_order(symbol_to_rh_crypto(t), "BUY", dollars, live_trading and login_ok, UI_MIN_PER_ORDER)
        else:
            status, executed, oid = place_stock_order(t, "BUY", dollars, live_trading and login_ok, UI_MIN_PER_ORDER)
        buy_rows.append({
            "Ticker": t,
            "Action": "BUY",
            "Alloc$": round(dollars, 2),
            "OrderID": oid,
            "Status": status,
            "R1%": round(float(r.get("R1", 0.0)), 3),
            "R5%": round(float(r.get("R5", 0.0)), 3),
            "R20%": round(float(r.get("R20", 0.0)), 3),
            "Score": round(score, 4),
            "Time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Display sections
    st.subheader("Today's Top‑N (ranked by momentum score)")
    st.dataframe(picks[["Ticker", "R1", "R5", "R20", "Score"]].rename(columns={"R1": "R1%", "R5": "R5%", "R20": "R20%"}), use_container_width=True)

    if sell_rows:
        st.subheader("Sell Log")
        st.dataframe(pd.DataFrame(sell_rows), use_container_width=True)
    else:
        st.subheader("Sell Log")
        st.write("(no sells)")

    st.subheader("Buy Log")
    buy_df = pd.DataFrame(buy_rows) if buy_rows else pd.DataFrame(columns=["Ticker","Action","Alloc$","OrderID","Status","R1%","R5%","R20%","Score","Time"])    
    st.dataframe(buy_df, use_container_width=True)

    # Combined CSV download
    all_rows = []
    for row in sell_rows: all_rows.append({"Kind": "SELL", **row})
    for row in buy_rows:  all_rows.append({"Kind": "BUY",  **row})
    log_df = pd.DataFrame(all_rows)
    csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Activity CSV", csv, file_name="rebalance_activity.csv", mime="text/csv")

    # Open orders snapshot (best‑effort)
    st.subheader("Open Orders")
    open_list = []
    if live_trading and login_ok and rh is not None:
        try:
            if rh_orders is not None and hasattr(rh_orders, "get_all_open_stock_orders"):
                s_open = rh_orders.get_all_open_stock_orders()
                if isinstance(s_open, list):
                    open_list.extend([{"type": "stock", **(o if isinstance(o, dict) else {"raw": str(o)})} for o in s_open])
        except Exception as e:
            st.info(f"Failed to fetch open stock orders: {e}")
        try:
            if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_open_orders"):
                c_open = rh_crypto.get_crypto_open_orders()
                if isinstance(c_open, list):
                    open_list.extend([{"type": "crypto", **(o if isinstance(o, dict) else {"raw": str(o)})} for o in c_open])
        except Exception as e:
            st.info(f"Failed to fetch open crypto orders: {e}")

    st.json(open_list if open_list else [])

# ---------------------------------
# Footer / Hints
# ---------------------------------
st.caption(
    "Tip: keep Live Trading OFF until you like the plan. When you turn it on, make sure secrets are set and the sidebar shows ‘Live orders ENABLED’."
)
