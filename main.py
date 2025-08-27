# stock-analyzer-app/main.py
# Robust, single-file runner for equities (Robinhood) and crypto (Kraken via ccxt)
# - Safer yfinance download + indicator calc (handles 2-D close arrays)
# - Clear ROUTE logging per symbol
# - Optional daily spend caps
# - Dry-run friendly (no API calls when DRY_RUN=true)

import os
import sys
import json
import time
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
import ta

# Optional deps (only used if DRY_RUN=false)
try:
    import robin_stocks.robinhood as rh
except Exception:
    rh = None

try:
    import ccxt
except Exception:
    ccxt = None


# -----------------------------
# Helpers
# -----------------------------
def as_bool(x: str, default=False) -> bool:
    if x is None:
        return default
    x = str(x).strip().lower()
    return x in ("1", "true", "t", "yes", "y", "on")


def nud(x: str, default=0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return float(default)


def today_yyyymmdd() -> str:
    return date.today().strftime("%Y-%m-%d")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_ccxt_symbol(sym: str) -> str:
    # ccxt/kraken uses "BTC/USD", not "BTC-USD"
    return sym.replace("-", "/").upper()


def looks_like_crypto(sym: str) -> bool:
    s = sym.upper()
    return "-" in s or "/" in s or s.endswith("USD") or s.endswith("-USD") or s.endswith("/USD")


# -----------------------------
# Indicator calculation
# -----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a 1-D Close series; compute RSI(14) and BB(20,2).
    Adds columns: RSI, BB_high, BB_low, Signal ('Buy'|'Sell'|'Hold')
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe")

    # yfinance sometimes returns MultiIndex columns; handle consistently
    close = None
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the 'Close' slice
        if ("Close" in df.columns.get_level_values(0)):
            c = df["Close"]
            if isinstance(c, pd.DataFrame):
                if c.shape[1] >= 1:
                    close = c.iloc[:, 0]
                else:
                    raise ValueError("Close slice is empty (MultiIndex).")
            else:
                close = c
        else:
            # Fallback: first column
            close = df.iloc[:, 0]
    else:
        if "Close" in df.columns:
            close = df["Close"]
        elif "Adj Close" in df.columns:
            close = df["Adj Close"]
        else:
            # Fallback to first column if unknown
            close = df.iloc[:, 0]

    # Force to 1-D series aligned to df.index
    close = pd.Series(np.asarray(close).reshape(-1), index=df.index, name="Close").astype(float)
    df = df.copy()
    df["Close"] = close

    # Indicators
    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()

    df["RSI"] = rsi
    df["BB_high"] = bb_high
    df["BB_low"] = bb_low

    # Signal rule of thumb
    last = df.iloc[-1]
    sig = "Hold"
    if last["RSI"] < 30 or last["Close"] < last["BB_low"]:
        sig = "Buy"
    elif last["RSI"] > 70 or last["Close"] > last["BB_high"]:
        sig = "Sell"

    df["Signal"] = sig
    return df


# -----------------------------
# Broker helpers (only used if DRY_RUN=false)
# -----------------------------
def rh_login(username: str, password: str, totp_secret: str) -> bool:
    if rh is None:
        print("Robinhood SDK not available; cannot login.")
        return False
    try:
        if totp_secret:
            import pyotp
            mfa = pyotp.TOTP(totp_secret).now()
            rh.authentication.login(username=username, password=password, mfa_code=mfa, expiresIn=86400, store_session=False)
        else:
            rh.authentication.login(username=username, password=password, expiresIn=86400, store_session=False)
        return True
    except Exception as e:
        print(f"Login failed. Check credentials and try again. ({e})")
        return False


def rh_place_equity_order(symbol: str, side: str, notional_usd: float):
    """
    Place a fractional market order by notional. side in {'buy','sell'}.
    """
    if rh is None:
        return {"error": "robin_stocks not available"}
    try:
        if side == "buy":
            return rh.orders.order_buy_fractional_by_price(symbol, float(notional_usd), timeInForce="gfd")
        else:
            return rh.orders.order_sell_fractional_by_price(symbol, float(notional_usd), timeInForce="gfd")
    except Exception as e:
        return {"error": str(e)}


def ccxt_kraken_login(api_key: str, api_secret: str, passphrase: str):
    if ccxt is None:
        print("ccxt not available; cannot init Kraken.")
        return None
    try:
        exchange = ccxt.kraken({
            "apiKey": api_key or "",
            "secret": api_secret or "",
            # Kraken doesn't use 'password' unless for some accounts; harmless if empty
            "password": passphrase or "",
            "enableRateLimit": True,
        })
        exchange.load_markets()
        return exchange
    except Exception as e:
        print(f"Failed to init Kraken via ccxt: {e}")
        return None


def ccxt_market_order(exchange, symbol_ccxt: str, side: str, notional_usd: float):
    # For market notional, compute base amount from ticker last price
    try:
        ticker = exchange.fetch_ticker(symbol_ccxt)
        price = float(ticker["last"])
        if price <= 0:
            raise ValueError("Invalid price returned")
        amount = max(0.0, notional_usd / price)
        amount = float(f"{amount:.8f}")  # keep precision small
        if amount <= 0:
            return {"skipped": "amount_zero", "symbol": symbol_ccxt, "side": side, "notional_usd": notional_usd}
        if side == "buy":
            return exchange.create_market_buy_order(symbol_ccxt, amount)
        else:
            return exchange.create_market_sell_order(symbol_ccxt, amount)
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Main
# -----------------------------
def main():
    # Env
    DRY_RUN = as_bool(os.getenv("DRY_RUN", "true"), default=True)
    SYMBOLS_RAW = os.getenv("SYMBOLS", "").strip()
    START = os.getenv("START", "").strip()
    END = os.getenv("END", "").strip()

    EQ_USD = nud(os.getenv("EQUITY_DOLLARS_PER_TRADE", "2"))
    CR_USD = nud(os.getenv("CRYPTO_DOLLARS_PER_TRADE", "5"))

    FORCE_SIDE_CRYPTO = (os.getenv("FORCE_SIDE", "") or "").strip().lower()  # '', 'buy', 'sell'
    CRYPTO_AUTOPICK = as_bool(os.getenv("CRYPTO_AUTOPICK", "true"), default=True)
    EQUITY_AUTOPICK = as_bool(os.getenv("EQUITY_AUTOPICK", "false"), default=False)
    AUTOPICK_OVERRIDES = os.getenv("AUTOPICK_OVERRIDES", "").strip()

    OUT_DIR = os.getenv("OUT_DIR", "out").strip() or "out"

    EQUITY_BROKER = (os.getenv("EQUITY_BROKER", "robinhood") or "robinhood").lower()
    CRYPTO_EXCHANGE = (os.getenv("CRYPTO_EXCHANGE", "kraken") or "kraken").lower()

    RH_USERNAME = os.getenv("RH_USERNAME", "")
    RH_PASSWORD = os.getenv("RH_PASSWORD", "")
    RH_TOTP_SECRET = os.getenv("RH_TOTP_SECRET", "")

    CRYPTO_API_KEY = os.getenv("CRYPTO_API_KEY", "")
    CRYPTO_API_SECRET = os.getenv("CRYPTO_API_SECRET", "")
    CRYPTO_API_PASSPHRASE = os.getenv("CRYPTO_API_PASSPHRASE", "")

    DAILY_EQUITY_CAP = nud(os.getenv("DAILY_EQUITY_CAP", os.getenv("MAX_EQUITY_SPEND_PER_DAY", "")), 0)
    DAILY_CRYPTO_CAP = nud(os.getenv("DAILY_CRYPTO_CAP", os.getenv("MAX_CRYPTO_SPEND_PER_DAY", "")), 0)

    # Symbols
    symbols = []
    if SYMBOLS_RAW:
        symbols = [s.strip() for s in SYMBOLS_RAW.split(",") if s.strip()]
    else:
        # Light autopick
        if EQUITY_AUTOPICK:
            symbols.append("SPY")
        if CRYPTO_AUTOPICK:
            symbols.append("BTC-USD")

    # AUTOPICK overrides
    if AUTOPICK_OVERRIDES:
        for s in AUTOPICK_OVERRIDES.split(","):
            s = s.strip()
            if s:
                symbols.append(s)

    # De-dup, keep order
    seen = set()
    uniq = []
    for s in symbols:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    symbols = uniq

    # Set default start if blank (keep behavior if provided)
    if not START:
        START = "2023-01-01"

    ensure_dir(OUT_DIR)

    # Banner/CONFIG
    print("CONFIG:")
    print(f"  dry_run={str(DRY_RUN).lower()}")
    print(f"  symbols={','.join(symbols) if symbols else '(none)'}")
    print(f"  start={START} end={END}")
    print(f"  crypto_autopick={str(CRYPTO_AUTOPICK).lower()} equity_autopick={str(EQUITY_AUTOPICK).lower()}")
    print(f"  force_side={FORCE_SIDE_CRYPTO or '(none)'}")
    print(f"  equity_broker={EQUITY_BROKER}  crypto_exchange={CRYPTO_EXCHANGE}")
    if DAILY_EQUITY_CAP or DAILY_CRYPTO_CAP:
        print(f"  daily_caps: equities=${DAILY_EQUITY_CAP:.2f} crypto=${DAILY_CRYPTO_CAP:.2f}")

    print(f"CONFIG: dry_run={DRY_RUN} equity_enabled={EQUITY_BROKER=='robinhood'} crypto_enabled={CRYPTO_EXCHANGE=='kraken'} exchange={CRYPTO_EXCHANGE} force_side=({FORCE_SIDE_CRYPTO or 'none'})")

    # Live logins only if needed
    rh_ok = False
    ex = None
    if not DRY_RUN:
        if EQUITY_BROKER == "robinhood":
            print("Starting login process...")
            print("Verification required, handling challenge...")
            print("Starting verification process...")
            print("Check robinhood app for device approvals method...")
            rh_ok = rh_login(RH_USERNAME, RH_PASSWORD, RH_TOTP_SECRET)
            if rh_ok:
                print("Verification successful!")
        if CRYPTO_EXCHANGE == "kraken":
            ex = ccxt_kraken_login(CRYPTO_API_KEY, CRYPTO_API_SECRET, CRYPTO_API_PASSPHRASE)

    # Spend caps counters
    spent_equities = 0.0
    spent_crypto = 0.0

    combined_rows = []

    for sym in symbols:
        is_crypto = looks_like_crypto(sym)
        start = START or "2023-01-01"
        end = END or None

        # Download market data
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
            if df is None or df.empty:
                raise ValueError("Empty dataframe returned by yfinance")
            df = compute_indicators(df)
        except Exception as e:
            print(f"Data error for {sym}: {e}")
            # Put a row in combined log with 'Hold' so we can see it tried
            combined_rows.append({
                "Date": today_yyyymmdd(),
                "Ticker": sym,
                "Close": "",
                "RSI": "",
                "BB_high": "",
                "BB_low": "",
                "Signal": "Hold"
            })
            # Skip but continue the loop
            continue

        last = df.iloc[-1]
        signal = last["Signal"]  # Buy | Sell | Hold

        # Forced side for crypto only (per your workflow form)
        side = None
        if is_crypto and FORCE_SIDE_CRYPTO in ("buy", "sell"):
            side = FORCE_SIDE_CRYPTO
        else:
            if str(signal).lower() == "buy":
                side = "buy"
            elif str(signal).lower() == "sell":
                side = "sell"
            else:
                side = "hold"

        # Spend for this route
        eq_notional = EQ_USD
        cr_notional = CR_USD

        # ROUTE line (always print)
        print(f"ROUTE: {sym} is_crypto={is_crypto} side={side} eq_notional={eq_notional:.1f} cr_notional={cr_notional:.1f} dry_run={DRY_RUN}")

        # Enforce daily caps (if set > 0)
        if side in ("buy", "sell"):
            if is_crypto and DAILY_CRYPTO_CAP > 0:
                allowed = max(0.0, DAILY_CRYPTO_CAP - spent_crypto)
                route_notional = cr_notional
                if route_notional > allowed:
                    if allowed <= 0:
                        print(f"CRYPTO cap reached; skipping {sym}.")
                        side = "hold"
                    else:
                        print(f"CRYPTO cap limiting {sym}: {route_notional:.2f} -> {allowed:.2f}")
                        cr_notional = allowed
                # after execution we’ll add to spent_crypto
            if (not is_crypto) and DAILY_EQUITY_CAP > 0:
                allowed = max(0.0, DAILY_EQUITY_CAP - spent_equities)
                route_notional = eq_notional
                if route_notional > allowed:
                    if allowed <= 0:
                        print(f"EQUITY cap reached; skipping {sym}.")
                        side = "hold"
                    else:
                        print(f"EQUITY cap limiting {sym}: {route_notional:.2f} -> {allowed:.2f}")
                        eq_notional = allowed

        # Place orders (or simulate)
        result = {}
        if side in ("buy", "sell"):
            if DRY_RUN:
                result = {
                    "dry_run": True,
                    "symbol": sym,
                    "side": side,
                    "notional": cr_notional if is_crypto else eq_notional
                }
            else:
                if is_crypto and CRYPTO_EXCHANGE == "kraken" and ex is not None:
                    result = ccxt_market_order(ex, to_ccxt_symbol(sym), side, float(cr_notional))
                elif (not is_crypto) and EQUITY_BROKER == "robinhood" and rh_ok:
                    result = rh_place_equity_order(sym, side, float(eq_notional))
                else:
                    result = {"skipped": "no_broker", "symbol": sym, "side": side}

            # Update caps counters on attempt
            if side in ("buy", "sell"):
                if is_crypto:
                    spent_crypto += float(cr_notional)
                else:
                    spent_equities += float(eq_notional)
        else:
            result = {"skipped": "hold", "symbol": sym}

        # Console output similar to earlier runs
        if is_crypto:
            if side in ("buy", "sell"):
                print(f"CRYPTO {sym} {side} -> {json.dumps(result)}")
            else:
                print(f"CRYPTO {sym} -> {{'skipped':'hold'}}")
        else:
            if side in ("buy", "sell"):
                print(f"EQUITY {sym} {side} -> {json.dumps(result)}")
            else:
                print(f"EQUITY {sym} -> {{'skipped':'hold'}}")

        # Persist CSVs
        # 1) analysis
        analysis_out = os.path.join(OUT_DIR, f"{sym.replace('/', '-')}_analysis.csv")
        df_out = df.copy()
        # Make a tidy sheet: Price alias + retain the useful columns
        df_out = df_out.rename(columns={"Close": "Price"})
        df_out[["Price", "Close", "High", "Low", "Open", "Volume", "RSI", "BB_high", "BB_low", "Signal"]].to_csv(analysis_out)

        # 2) symbol trade log (single latest row)
        trade_row = pd.DataFrame([{
            "Date": df.index[-1].strftime("%-m/%-d/%Y") if hasattr(df.index[-1], "strftime") else str(df.index[-1]),
            "Ticker": sym,
            "Close": float(last["Close"]),
            "RSI": float(last["RSI"]),
            "BB_high": float(last["BB_high"]),
            "BB_low": float(last["BB_low"]),
            "Signal": signal
        }])
        trade_log_out = os.path.join(OUT_DIR, f"{sym.replace('/', '-')}_trade_log.csv")
        trade_row.to_csv(trade_log_out, index=False)

        combined_rows.append({
            "Date": trade_row.iloc[0]["Date"],
            "Ticker": sym,
            "Close": trade_row.iloc[0]["Close"],
            "RSI": trade_row.iloc[0]["RSI"],
            "BB_high": trade_row.iloc[0]["BB_high"],
            "BB_low": trade_row.iloc[0]["BB_low"],
            "Signal": signal
        })

    # combined log
    if combined_rows:
        combined_df = pd.DataFrame(combined_rows)
        combined_out = os.path.join(OUT_DIR, "combined_trade_log.csv")
        combined_df.to_csv(combined_out, index=False)

    # Tail message
    if DRY_RUN:
        print("Dry run = True. Orders NOT placed. Files saved to out/")
    else:
        print("Dry run = False. Orders placed (when available). Files saved to out/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
    except Exception as e:
        # Never crash CI silently—print and exit non-zero
        print(f"FATAL: {e}")
        sys.exit(2)
