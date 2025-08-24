# main.py (final headless runner)
import os
import pandas as pd
import yfinance as yf

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# Optional broker adapters
try:
    from trader.broker_robinhood import RobinhoodBroker
except Exception:
    RobinhoodBroker = None

try:
    from trader.broker_crypto_ccxt import CCXTCryptoBroker
except Exception:
    CCXTCryptoBroker = None


# ---------- helpers ----------

def download_data(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna(how="all")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not TA_AVAILABLE:
        return df
    # Normalize Close to 1-D numeric (yfinance can yield 2-D columns)
    close = df['Close'] if 'Close' in df.columns else None
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    rsi = ta.momentum.RSIIndicator(close=close).rsi()
    bb = ta.volatility.BollingerBands(close=close)
    out = df.copy()
    out['Close'] = close
    out['RSI'] = rsi
    out['BB_high'] = bb.bollinger_hband()
    out['BB_low'] = bb.bollinger_lband()
    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or 'Close' not in df.columns:
        return df
    if not TA_AVAILABLE or 'RSI' not in df.columns:
        df = df.copy(); df['Signal'] = ""; return df

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    sig = []
    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        bb_l = df['BB_low'].iloc[i]
        bb_h = df['BB_high'].iloc[i]
        c = close.iloc[i]
        if pd.notna(c) and pd.notna(bb_l) and pd.notna(rsi) and float(c) < float(bb_l) and float(rsi) < 30:
            sig.append("Buy")
        elif pd.notna(c) and pd.notna(bb_h) and pd.notna(rsi) and float(c) > float(bb_h) and float(rsi) > 70:
            sig.append("Sell")
        else:
            sig.append("")
    out = df.copy(); out['Signal'] = sig; return out


# ---------- main ----------

def main():
    symbols = [s.strip() for s in os.getenv('SYMBOLS', 'AAPL,MSFT,BTC-USD').split(',') if s.strip()]
    start = os.getenv('START', '2023-01-01')
    end = os.getenv('END', '') or None
    dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
    outdir = os.getenv('OUT_DIR', 'out'); os.makedirs(outdir, exist_ok=True)

    # Brokers (optional)
    equity_enabled = os.getenv('EQUITY_BROKER', 'robinhood') == 'robinhood' and RobinhoodBroker is not None
    crypto_enabled = os.getenv('CRYPTO_EXCHANGE', '') != '' and CCXTCryptoBroker is not None

    rb = None
    if equity_enabled:
        ts = os.getenv('RH_TOTP_SECRET', '').strip() or None

        rb = RobinhoodBroker(
            username=os.getenv('RH_USERNAME', ''),
            password=os.getenv('RH_PASSWORD', ''),
            totp_secret=ts,
            dry_run=dry_run,
        )
        if not dry_run:
            if ts is None:
                print("Robinhood 2FA: No TOTP secret set – will wait for app/SMS approval if challenged…")
            rb.login()

    cb = None
    if crypto_enabled:
        cb = CCXTCryptoBroker(
            exchange_id=os.getenv('CRYPTO_EXCHANGE', 'kraken'),
            api_key=os.getenv('CRYPTO_API_KEY', None),
            api_secret=os.getenv('CRYPTO_API_SECRET', None),
            api_password=os.getenv('CRYPTO_API_PASSPHRASE', None),
            dry_run=dry_run,
        )

    # --- Notional controls & overrides ---
    # Clamp equity notional to Robinhood's fractional minimum (>= $1)
    eq_notional = max(float(os.getenv('EQUITY_DOLLARS_PER_TRADE', '200')), 1.01)
    cr_notional = float(os.getenv('CRYPTO_DOLLARS_PER_TRADE', '100'))
    # Optional override to force a side for quick live tests
    force_side = os.getenv('FORCE_SIDE', '').lower()

    combined = []
    for sym in symbols:
        df = download_data(sym, start, end)
        df = add_indicators(df)
        df = generate_signals(df)
        # Save artifacts
        df.to_csv(os.path.join(outdir, f"{sym}_analysis.csv"))

        # tidy trade log: reset index→Date, add Ticker, keep consistent columns
        t = df[df.get('Signal', "") != ""].copy()
        t = t.reset_index()  # make Date a column
        if 'Date' not in t.columns:
            # yfinance sometimes calls the reset column 'index' — normalize it
            t.rename(columns={t.columns[0]: 'Date'}, inplace=True)
        t.insert(1, 'Ticker', sym)
        keep_cols = [c for c in ['Date', 'Ticker', 'Close', 'RSI', 'BB_high', 'BB_low', 'Signal'] if c in t.columns]
        t = t[keep_cols]
        t.to_csv(os.path.join(outdir, f"{sym}_trade_log.csv"), index=False)
        combined.append(t)

        # Execute most recent signal (still dry-run unless flipped)
        last = t.tail(1)
        if not last.empty:
            action = last['Signal'].iloc[0]
            side = 'buy' if action.lower() == 'buy' else 'sell'
            if force_side in ('buy', 'sell'):
                print(f"FORCE_SIDE override active -> {force_side}")
                side = force_side
            is_crypto = '-' in sym and sym.upper().endswith('-USD')

            try:
                if is_crypto and cb is not None:
                    res = cb.place_market_notional(symbol=sym, side=side, notional_usd=cr_notional)
                    print(f"CRYPTO {sym} {side} -> {res}")
                elif rb is not None:
                    res = rb.place_market(symbol=sym, side=side, notional=eq_notional)
                    print(f"EQUITY {sym} {side} ${eq_notional:.2f} -> {res}")
                else:
                    print(f"No broker configured for {sym}; would {side} ${cr_notional if is_crypto else eq_notional} (dry_run={dry_run})")
            except Exception as e:
                # Swallow broker errors so the workflow can still finish and upload artifacts
                print(f"[WARN] Order for {sym} skipped: {e}")
        else:
            print(f"[{sym}] no signals in range.")

    if any([not t.empty for t in combined]):
        all_trades = pd.concat([t for t in combined if not t.empty], ignore_index=True)
        all_trades.to_csv(os.path.join(outdir, 'combined_trade_log.csv'), index=False)
    print(f"Dry run = {dry_run}. Orders{' NOT' if dry_run else ''} placed. Files saved to {outdir}/")


if __name__ == '__main__':
    main()
