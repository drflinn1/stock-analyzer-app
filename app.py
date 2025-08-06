# --- Robinhood Authentication ---
# Use the secret keys exactly as set in Streamlit Cloud
RH_USER = st.secrets.get('ROBINHOOD_USERNAME') or os.getenv('ROBINHOOD_USERNAME')
RH_PASS = st.secrets.get('ROBINHOOD_PASSWORD') or os.getenv('ROBINHOOD_PASSWORD')
if RH_USER and RH_PASS:
    try:
        r.login(RH_USER, RH_PASS)
    except Exception as e:
        st.error(f"Robinhood login failed: {str(e)}")
        st.stop()
else:
    st.error("Robinhood credentials not found. Please set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in Streamlit secrets.")
    st.stop()

# --- Streamlit App Setup ---
st.set_page_config(page_title="Equity & Crypto Analyzer", layout="wide")
st.title("Stock & Crypto Analyzer Bot")

# Auto-refresh every 24h
st_autorefresh(interval=24*60*60*1000, key='daily_refresh')

# Sidebar: Universe & Parameters
st.sidebar.header("Universe")
equities = st.sidebar.text_area("Equity Tickers (comma-separated)", "AAPL,MSFT,GOOG").upper().replace(' ', '').split(',')
include_crypto = st.sidebar.checkbox("Include Crypto", value=False)
crypto_list = []
if include_crypto:
    crypto_list = st.sidebar.multiselect(
        "Crypto Tickers", ['BTC-USD','ETH-USD','ADA-USD','SOL-USD'], default=['BTC-USD','ETH-USD']
    )

st.sidebar.header("Signal Parameters")
overbought = st.sidebar.slider("RSI Overbought", min_value=50, max_value=90, value=70)
oversold = st.sidebar.slider("RSI Oversold", min_value=10, max_value=50, value=30)
trade_amount = st.sidebar.number_input("USD per Trade", min_value=10, max_value=10000, value=500, step=10)

# Main Execution
if st.button("► Run Scan & Execute"):
    logs = []

    # Equities Loop
    for sym in equities:
        df = fetch_data(sym, period="30d", interval="1d")
        if df.empty: continue
        sig = analyze_signal(df, overbought, oversold)
        price = float(df['Close'].iloc[-1])
        logs.append({'Ticker': sym, 'Signal': sig['action'], 'Price': price,
                     'Time': df.index[-1], 'RSI': sig['rsi'],
                     'BB_Upper': sig['bb_upper'], 'BB_Lower': sig['bb_lower']})
        if sig['action'] in ['BUY','SELL']:
            place_order(sym, sig['action'], trade_amount)

    # Crypto Loop
    if include_crypto:
        for sym in crypto_list:
            dfc = fetch_data(sym, period="7d", interval="1h")
            if dfc.empty: continue
            sigc = analyze_signal(dfc, overbought, oversold)
            pricec = float(dfc['Close'].iloc[-1])
            logs.append({'Ticker': sym, 'Signal': sigc['action'], 'Price': pricec,
                         'Time': dfc.index[-1], 'RSI': sigc['rsi'],
                         'BB_Upper': sigc['bb_upper'], 'BB_Lower': sigc['bb_lower']})
            if sigc['action'] in ['BUY','SELL']:
                place_order(sym, sigc['action'], trade_amount)

    # Display Execution Log
    df_logs = pd.DataFrame(logs)
    st.subheader("Trade Signals & Execution Log")
    st.dataframe(df_logs)
