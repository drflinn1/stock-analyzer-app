# app.py — Stock & Crypto Momentum Rebalancer (Streamlit UI)
# Slim UI that delegates heavy work to engine.py

VERSION = "0.8.7 (2025-08-14)"

from datetime import datetime
import pandas as pd
import streamlit as st

# domain logic
import engine as eng

# Robinhood import (login happens here only)
try:
    import robin_stocks.robinhood as rh
except Exception:
    rh = None

st.set_page_config(page_title=f"Rebalancer · {VERSION}", layout="wide")

# ---- Sidebar: auth ----
st.sidebar.header("Authentication & Mode")
st.sidebar.caption(f"Version {VERSION}")
live_trading = st.sidebar.checkbox("Enable Live Trading (use with caution)", value=False)

# Read secrets
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
        st.sidebar.error("robin_stocks is not available. Running in SIMULATION.")
        live_trading = False
    elif not user or not pwd:
        st.sidebar.error("Live trading selected but credentials missing in Secrets.")
        live_trading = False
    else:
        try:
            # compatible call across library versions
            login_ok = bool(rh.authentication.login(username=user, password=pwd, store_session=True, expiresIn=24*3600, scope="internal"))
        except Exception:
            try:
                login_ok = bool(rh.authentication.login(username=user, password=pwd))
            except Exception as e:
                st.sidebar.warning(f"Robinhood login failed: {e}.")
                login_ok = False
        if login_ok:
            st.sidebar.success("Robinhood authenticated — Live orders ENABLED")
            login_msg = "Live orders ENABLED"
        else:
            st.sidebar.warning("Login failed — running in SIMULATION.")
            live_trading = False
else:
    st.sidebar.info(login_msg)

# ---- Sidebar: universe & allocation ----
st.sidebar.header("Universe & Allocation")
universe_src = st.sidebar.selectbox("Equity Universe", ["S&P 500 (auto)", "Manual list"], index=0)
raw_tickers  = st.sidebar.text_area("Manual equity tickers (comma-separated)", value="AAPL,MSFT,GOOG")
include_crypto = st.sidebar.checkbox("Include Crypto (Robinhood-tradable)", value=True)

alloc_mode = st.sidebar.selectbox("Allocation mode", ["Fixed $ per trade", "Proportional across winners"], index=0)
fixed_per_trade   = st.sidebar.number_input("Fixed $ per BUY/SELL", min_value=1.0, value=5.0, step=0.5)
prop_total_budget = st.sidebar.number_input("Total BUY budget (proportional)", min_value=1.0, value=15.0, step=1.0)
min_per_order     = st.sidebar.number_input("Minimum $ per order", min_value=1.0, value=2.0, step=0.5)
n_picks           = st.sidebar.number_input("Top N to hold", min_value=1, value=3, step=1)

st.sidebar.divider()
use_crypto_limits = st.sidebar.checkbox("Use LIMIT orders for crypto", value=True)
crypto_limit_bps  = st.sidebar.slider("Crypto limit price buffer (bps)", min_value=5, max_value=100, value=20, step=5)
use_stock_limits  = st.sidebar.checkbox("Use LIMIT orders for stocks (experimental)", value=False)
stock_limit_bps   = st.sidebar.slider("Stock limit price buffer (bps)", min_value=5, max_value=150, value=25, step=5)

st.sidebar.divider()
st.sidebar.subheader("Risk & Safety")
max_buy_orders   = st.sidebar.number_input("Max BUY orders per run", min_value=1, value=12, step=1)
max_buy_notional = st.sidebar.number_input("Max BUY notional per run ($)", min_value=10.0, value=50.0, step=5.0)
auto_cancel_stale= st.sidebar.checkbox("Auto-cancel stale open orders", value=True)
stale_minutes    = st.sidebar.number_input("Stale = older than (minutes)", min_value=1, value=20, step=1)
cancel_now       = st.sidebar.button("Cancel ALL open orders now")

st.sidebar.divider()
full_auto = st.sidebar.checkbox("Full-Auto (run on load)", value=False)
auto_bp   = st.sidebar.checkbox("Auto budget from buying power (for proportional mode)", value=False)
bp_pct    = st.sidebar.slider("Budget % of Buying Power", min_value=5, max_value=100, value=30, step=5)

# ---- Title ----
st.title("Stock & Crypto Momentum Rebalancer")
st.caption(f"Version {VERSION}")

def do_run():
    return eng.run_once(
        live_trading=live_trading,
        login_ok=login_ok,
        universe_src=universe_src,
        raw_tickers=raw_tickers,
        include_crypto=include_crypto,
        alloc_mode=alloc_mode,
        fixed_per_trade=float(fixed_per_trade),
        prop_total_budget=float(prop_total_budget),
        min_per_order=float(min_per_order),
        n_picks=int(n_picks),
        use_crypto_limits=bool(use_crypto_limits),
        crypto_limit_bps=int(crypto_limit_bps),
        use_stock_limits=bool(use_stock_limits),
        stock_limit_bps=int(stock_limit_bps),
        auto_bp=bool(auto_bp),
        bp_pct=int(bp_pct),
        max_buy_orders=int(max_buy_orders),          # <-- safety caps wired in both paths
        max_buy_notional=float(max_buy_notional),    # <-- safety caps wired in both paths
    )

ran = False

# Full-Auto path
if full_auto and not st.session_state.get("__full_auto_ran__"):
    st.session_state["__full_auto_ran__"] = True
    st.info("Full-Auto is ON — running now.")
    result = do_run()
    ran = True
    # render
    picks = result.get("picks", pd.DataFrame())
    if picks is None: picks = pd.DataFrame()
    if not picks.empty:
        st.subheader("Today's Top-N (ranked by momentum score)")
        st.dataframe(picks[["Ticker","R1","R5","R20","Score"]].rename(columns={"R1":"R1%","R5":"R5%","R20":"R20%"}), use_container_width=True)
    else:
        st.warning("No momentum data available.")
    st.subheader("Sell Log")
    srows = result.get("sell_rows", [])
    st.dataframe(pd.DataFrame(srows)) if srows else st.write("(no sells)")
    st.subheader("Buy Log")
    brows = result.get("buy_rows", [])
    st.dataframe(pd.DataFrame(brows)) if brows else st.write("(none)")
    # auto-budget note
    if result.get("auto_budget"):
        st.sidebar.caption(f"Auto budget: ${result.get('budget_used',0):,.2f} from BP ≈ ${result.get('bp_seen',0):,.2f}")

# Button path
if st.button("▶ Run Daily Scan & Rebalance", type="primary"):
    result = do_run()
    ran = True
    picks = result.get("picks", pd.DataFrame())
    if not picks.empty:
        st.subheader("Today's Top-N (ranked by momentum score)")
        st.dataframe(picks[["Ticker","R1","R5","R20","Score"]].rename(columns={"R1":"R1%","R5":"R5%","R20":"R20%"}), use_container_width=True)
    else:
        st.warning("No momentum data available.")
    st.subheader("Sell Log")
    srows = result.get("sell_rows", [])
    st.dataframe(pd.DataFrame(srows)) if srows else st.write("(no sells)")
    st.subheader("Buy Log")
    brows = result.get("buy_rows", [])
    st.dataframe(pd.DataFrame(brows)) if brows else st.write("(none)")
    if result.get("auto_budget"):
        st.sidebar.caption(f"Auto budget: ${result.get('budget_used',0):,.2f} from BP ≈ ${result.get('bp_seen',0):,.2f}")

# Auto-cancel stale & manual cancel
if auto_cancel_stale:
    n_cancelled = eng.cancel_open_orders(live_trading=live_trading, login_ok=login_ok, older_than_minutes=int(stale_minutes))
    if n_cancelled > 0:
        st.info(f"Auto-cancelled {n_cancelled} stale open orders (>{int(stale_minutes)} min).")
if cancel_now:
    n = eng.cancel_open_orders(live_trading=live_trading, login_ok=login_ok, cancel_all=True)
    st.success(f"Attempted to cancel {n} open orders.")

# Open orders (always shown)
st.subheader("Open Orders")
rows = eng.open_orders_table(live_trading, login_ok)
st.dataframe(pd.DataFrame(rows), use_container_width=True) if rows else st.write("[]")
with st.expander("Show raw API response"):
    st.json(eng.get_open_orders_raw(live_trading, login_ok))

# Footer tip
if not ran:
    st.caption("Tip: keep Live Trading OFF until you like the plan. When you turn it on, make sure secrets are set and the sidebar shows ‘Live orders ENABLED’.")
