# main.py — headless runner (autopick + optional auto-sizing)
import os, json
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

# Optional libs for autopick
try:
    from pycoingecko import CoinGeckoAPI
except Exception:
    CoinGeckoAPI = None

try:
    import ccxt  # used to check which USD pairs are actually tradable on Kraken
except Exception:
    ccxt = None


# ------------------- helpers -------------------

def parse_bool(v, default=False):
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def parse_float(v, default):
    try:
        return float(v)
    except Exception:
        return default

def parse_advanced_json(env_text: str) -> dict:
    """Read optional JSON tuning without raising."""
    try:
        return json.loads(env_text) if env_text and env_text.strip() else {}
    except Exception:
        return {}

def yf_download(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna(how="all")

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not TA_AVAILABLE:
        return df
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

def kraken_usd_pairs():
    """Return a set like {'BTC/USD','ETH/USD',...} of tradable USD pairs on Kraken."""
    markets = set()
    if ccxt is None:
        return markets
    try:
        ex = ccxt.kraken()
        ex.load_markets()
        for sym in ex.markets:
            if sym.endswith('/USD'):
                markets.add(sym.upper())
    except Exception:
        pass
    return markets

def autopick_crypto_symbols(top_n=1, min_vol_usd=10_000_000, direction='up') -> list:
    """
    Choose top movers from CoinGecko and return list of yfinance tickers like ['BTC-USD','ETH-USD'].
    direction='up' (=gainers) or 'down' (=losers)
    """
    if CoinGeckoAPI is None:
        return []
    cg = CoinGeckoAPI()
    order = 'price_change_percentage_24h_desc' if direction == 'up' else 'price_change_percentage_24h_asc'
    data = cg.get_coins_markets(
        vs_currency='usd',
        order=order,
        price_change_percentage='24h',
        per_page=100, page=1)
    # filter by volume and map to Kraken USD pairs
    kraken_pairs = kraken_usd_pairs()
    picks = []
    for row in data:
        try:
            vol = float(row.get('total_volume') or 0)
            if vol < float(min_vol_usd):
                continue
            base = (row.get('symbol') or '').upper()
            if not base:
                continue
            ccxt_symbol = f"{base}/USD"
            if ccxt_symbol.upper() in kraken_pairs:
                # return yfinance style ticker
                picks.append(f"{base}-USD")
            if len(picks) >= int(top_n):
                break
        except Exception:
            continue
    return picks


# ------------------- main -------------------

def main():
    # -------- read inputs / envs --------
    symbols = [s.strip() for s in os.getenv('SYMBOLS', 'AAPL,MSFT,BTC-USD').split(',') if s.strip()]
    start = os.getenv('START', '2023-01-01')
    end = os.getenv('END', '') or None
    dry_run = parse_bool(os.getenv('DRY_RUN', 'true'), default=True)
    outdir = os.getenv('OUT_DIR', 'out'); os.makedirs(outdir, exist_ok=True)

    # size inputs (fixed notionals)
    eq_notional_fixed = parse_float(os.getenv('EQUITY_DOLLARS_PER_TRADE', '200'), 200.0)
    cr_notional_fixed = parse_float(os.getenv('CRYPTO_DOLLARS_PER_TRADE', '100'), 100.0)

    force_side = (os.getenv('FORCE_SIDE') or '').strip().lower()  # '', 'buy', 'sell'

    # autopick toggles and advanced JSON
    use_crypto_autopick = parse_bool(os.getenv('AUTOPICK_CRYPTO_TOP', 'false'))
    use_equity_autopick = parse_bool(os.getenv('AUTOPICK_EQUITIES_TOP', 'false'))
    advanced = parse_advanced_json(os.getenv('ADVANCED_JSON', ''))

    # -------- brokers (optional) --------
    equity_enabled = (os.getenv('EQUITY_BROKER', 'robinhood') == 'robinhood') and (RobinhoodBroker is not None)
    crypto_enabled = (os.getenv('CRYPTO_EXCHANGE', '').strip() != '') and (CCXTCryptoBroker is not None)

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

    # -------- optional autopick lists --------
    if use_crypto_autopick:
        top_n = int(advanced.get('crypto_top_n', 1))
        min_vol = float(advanced.get('crypto_min_vol_usd', 10_000_000))
        direction = str(advanced.get('crypto_direction', 'up')).lower()
        picks = autopick_crypto_symbols(top_n=top_n, min_vol_usd=min_vol, direction=direction)
        if picks:
            symbols = picks
            print(f"AUTOPICK (crypto): {symbols}  (top_n={top_n}, min_vol_usd={min_vol}, direction={direction})")
        else:
            print("AUTOPICK (crypto): no eligible Kraken USD pairs found from CoinGecko; keeping provided symbols.")

    # (equities autopick placeholder – will add when you’re ready)
    if use_equity_autopick:
        print("AUTOPICK (equities): not implemented yet in this step; using provided symbols.")

    # -------- analyze all symbols & collect candidate orders --------
    combined_logs = []
    planned_orders = []   # will size after we know how many signals there are

    print(f"CONFIG: dry_run={dry_run} equity_enabled={bool(rb)} crypto_enabled={bool(cb)} "
          f"exchange={os.getenv('CRYPTO_EXCHANGE','')} force_side={force_side or '(none)'}")

    for sym in symbols:
        df = yf_download(sym, start, end)
        df = add_indicators(df)
        df = generate_signals(df)

        # Save artifacts
        df.to_csv(os.path.join(outdir, f"{sym}_analysis.csv"))

        # Build tidy trade log
        t = df[df.get('Signal', "") != ""].copy()
        t = t.reset_index()
        if 'Date' not in t.columns:
            t.rename(columns={t.columns[0]: 'Date'}, inplace=True)
        t.insert(1, 'Ticker', sym)
        keep_cols = [c for c in ['Date', 'Ticker', 'Close', 'RSI', 'BB_high', 'BB_low', 'Signal'] if c in t.columns]
        t = t[keep_cols]
        t.to_csv(os.path.join(outdir, f"{sym}_trade_log.csv"), index=False)
        combined_logs.append(t)

        # Decide last signal
        last = t.tail(1)
        if last.empty:
            print(f"[{sym}] no signals in range.")
            continue

        action = last['Signal'].iloc[0]
        side = 'buy' if action.lower() == 'buy' else 'sell'
        if force_side in ('buy', 'sell'):
            side = force_side

        is_crypto = ('-USD' in sym.upper())  # yfinance crypto format
        planned_orders.append({
            "sym_yf": sym,
            "sym_ccxt": sym.upper().replace('-', '/'),  # BTC-USD -> BTC/USD
            "is_crypto": is_crypto,
            "side": side,
        })

    # -------- size crypto orders (fixed by default; optional pct_balance) --------
    crypto_orders = [o for o in planned_orders if o['is_crypto']]
    equity_orders = [o for o in planned_orders if not o['is_crypto']]

    # Default per-order notionals
    eq_per_order = eq_notional_fixed
    cr_per_order = cr_notional_fixed

    # Optional automatic sizing for crypto
    sizing_mode = str(advanced.get('sizing_mode', 'fixed')).lower()
    if sizing_mode == 'pct_balance' and cb is not None:
        pct = float(advanced.get('balance_pct', 0.20))     # use 20% of free USD
        min_per = float(advanced.get('min_per_order', 5.0))
        max_per = float(advanced.get('max_per_order', 25.0))
        split = parse_bool(advanced.get('split_by_signals', True), True)

        free_usd = 0.0
        try:
            # use broker's exchange handle to fetch balance
            bal = cb.exchange.fetch_balance()
            free_usd = float(bal.get('free', {}).get('USD', 0.0))
        except Exception:
            pass

        pool = max(0.0, free_usd * pct)
        if split and crypto_orders:
            cr_per_order = pool / len(crypto_orders)
        else:
            cr_per_order = pool

        # clamp to safety bounds
        cr_per_order = max(min_per, min(max_per, cr_per_order))

        print(f"SIZING: mode=pct_balance free_usd={free_usd:.2f} pct={pct} "
              f"orders={len(crypto_orders)} per_order={cr_per_order:.2f} "
              f"(min={min_per}, max={max_per}, split={split})")
    else:
        if sizing_mode != 'fixed':
            # any unknown modes fall back to fixed
            print(f"SIZING: mode=fixed (using CRYPTO_DOLLARS_PER_TRADE={cr_per_order})")

    # -------- place orders --------
    for o in planned_orders:
        if o['is_crypto']:
            if cb is None:
                print(f"No crypto broker configured; would {o['side']} ${cr_per_order} {o['sym_yf']} (dry_run={dry_run})")
                continue
            print(f"ROUTE: {o['sym_yf']} is_crypto=True side={o['side']} "
                  f"eq_notional={eq_per_order} cr_notional={cr_per_order} dry_run={dry_run}")
            res = cb.place_market_notional(symbol=o['sym_yf'], side=o['side'], notional_usd=cr_per_order)
            print(f"CRYPTO {o['sym_yf']} {o['side']} -> {res}")
        else:
            if rb is None:
                print(f"No equity broker configured; would {o['side']} ${eq_per_order} {o['sym_yf']} (dry_run={dry_run})")
                continue
            print(f"ROUTE: {o['sym_yf']} is_crypto=False side={o['side']} "
                  f"eq_notional={eq_per_order} cr_notional={cr_per_order} dry_run={dry_run}")
            res = rb.place_market(symbol=o['sym_yf'], side=o['side'], notional=eq_per_order)
            print(f"EQUITY {o['sym_yf']} {o['side']} -> {res}")

    # -------- write combined log --------
    if any([not t.empty for t in combined_logs]):
        all_trades = pd.concat([t for t in combined_logs if not t.empty], ignore_index=True)
        all_trades.to_csv(os.path.join(outdir, 'combined_trade_log.csv'), index=False)

    print(f"Dry run = {dry_run}. Orders{' NOT' if dry_run else ''} placed. Files saved to {outdir}/")


if __name__ == '__main__':
    main()
