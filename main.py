# stock-analyzer-app/main.py
# Robust runner for equities (Robinhood) + crypto (Kraken via ccxt)
# - Always prints a ROUTE line (even on skip/errors)
# - Safer yfinance handling (1-D Close guaranteed)
# - Optional daily spend caps
# - Dry-run friendly; writes CSVs to out/

import os
import sys
import json
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import ta

# Optional live libs
try:
    import robin_stocks.robinhood as rh
except Exception:
    rh = None

try:
    import ccxt
except Exception:
    ccxt = None


# -----------------------------
# Utilities
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


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def today_str() -> str:
    return date.today().strftime("%-m/%-d/%Y")


def looks_like_crypto(sym: str) -> bool:
    s = sym.upper()
    return "-" in s or "/" in s or s.endswith("USD") or s.endswith("-USD") or s.endswith("/USD")


def to_ccxt_symbol(sym: str) -> str:
    return sym.replace("-", "/").upper()


# -----------------------------
# Market data / indicators
# -----------------------------
def pick_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a clean 1-D numeric Close series for TA.
    Handles MultiIndex and non-numeric data.
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe")

    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the 'Close' block if present
        top = df.columns.get_level_values(0)
        if "Close" in list(top):
            c = df["Close"]
            if isinstance(c, pd.DataFrame):
                # If multiple sub-columns, take the first
                if c.shape[1] >= 1:
                    s = c.iloc[:, 0]
                else:
                    raise ValueError("MultiIndex Close slice is empty")
            else:
                s = c
        else:
            # Fallback to first column
            s = df.iloc[:, 0]
    else:
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            s = df.iloc[:, 0]

    s = pd.Series(np.asarray(s).reshape(-1), index=df.index, name="Close")
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        raise ValueError("Close contains no numeric values")
    return s


def compute_indicators(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    close = pick_close_series(df)
    df["Close"] = close

    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()

    df["RSI"] = rsi
    df["BB_high"] = bb_high
    df["BB_low"] = bb_low

    sig = "Hold"
    last = df.iloc[-1]
    if last["RSI"] < 30 or last["Close"] < last["BB_low"]:
        sig = "Buy"
    elif last["RSI"] > 70 or last["Close"] > last["BB_high"]:
        sig = "Sell"
    df["Signal"] = sig
    return df


# -----------------------------
# Brokers (live only)
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


def rh_place_notional(symbol: str, side: str, usd: float):
    if rh is None:
        return {"error": "robin_stocks not available"}
    try:
        if side == "buy":
            return rh.orders.order_buy_fractional_by_price(symbol, float(usd), timeInForce="gfd")
        else:
            return rh.orders.order_sell_fractional_by_price(symbol, float(usd), timeInForce="gfd")
    except Exception as e:
        return {"error": str(e)}


def ccxt_kraken(api_key: str, secret: str, passphrase: str):
    if ccxt is None:
        print("ccxt not available; cannot init Kraken.")
        return None
    try:
        ex = ccxt.kraken({
            "apiKey": api_key or "",
            "secret": secret or "",
            "password": passphrase or "",
            "enableRateLimit": True,
        })
        ex.load_markets()
        return ex
    except Exception as e:
        print(f"Failed to init Kraken via ccxt: {e}")
        return None


def ccxt_market_notional(ex, symbol_ccxt: str, side: str, usd: float):
    try:
        t = ex.fetch_ticker(symbol_ccxt)
        price = float(t["last"])
        if price <= 0:
            return {"error": "invalid_price"}
        amount = max(0.0, float(usd) / price)
        amount = float(f"{amount:.8f}")
        if amount <= 0:
            return {"skipped": "amount_zero"}
        if side == "buy":
            return ex.create_market_buy_order(symbol_ccxt, amount)
        else:
            return ex.create_market_sell_order(symbol_ccxt, amount)
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Main
# -----------------------------
def main():
    # Inputs / env
    DRY_RUN = as_bool(os.getenv("DRY_RUN", "true"), default=True)

    SYMBOLS_RAW = (os.getenv("SYMBOLS", "") or "").strip()
    START = (os.getenv("START", "") or "2023-01-01").strip()
    END = (os.getenv("END", "") or "").strip()

    EQ_USD = nud(os.getenv("EQUITY_DOLLARS_PER_TRADE", "2"))
    CR_USD = nud(os.getenv("CRYPTO_DOLLARS_PER_TRADE", "5"))

    FORCE_SIDE_CRYPTO = (os.getenv("FORCE_SIDE", "") or "").strip().lower()
    CRYPTO_AUTOPICK = as_bool(os.getenv("CRYPTO_AUTOPICK", "true"), default=True)
    EQUITY_AUTOPICK = as_bool(os.getenv("EQUITY_AUTOPICK", "false"), default=False)
    AUTOPICK_OVERRIDES = (os.getenv("AUTOPICK_OVERRIDES", "") or "").strip()

    OUT_DIR = (os.getenv("OUT_DIR", "out") or "out").strip()

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
        if EQUITY_AUTOPICK:
            symbols.append("SPY")
        if CRYPTO_AUTOPICK:
            symbols.append("BTC-USD")

    if AUTOPICK_OVERRIDES:
        symbols += [s.strip() for s in AUTOPICK_OVERRIDES.split(",") if s.strip()]

    # de-dup
    seen, uniq = set(), []
    for s in symbols:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    symbols = uniq

    ensure_dir(OUT_DIR)

    # CONFIG banner
    print("CONFIG:")
    print(f"  dry_run={str(DRY_RUN).lower()}")
    print(f"  symbols={','.join(symbols) if symbols else '(none)'}")
    print(f"  start={START} end={END or '(today)'}")
    print(f"  crypto_autopick={str(CRYPTO_AUTOPICK).lower()} equity_autopick={str(EQUITY_AUTOPICK).lower()}")
    print(f"  force_side={FORCE_SIDE_CRYPTO or '(none)'}")
    print(f"  equity_broker={EQUITY_BROKER}  crypto_exchange={CRYPTO_EXCHANGE}")
    if DAILY_EQUITY_CAP or DAILY_CRYPTO_CAP:
        print(f"  daily_caps: equities=${DAILY_EQUITY_CAP:.2f} crypto=${DAILY_CRYPTO_CAP:.2f}")

    # Live setup
    rh_ok = False
    ex = None
    if not DRY_RUN:
        if EQUITY_BROKER == "robinhood":
            print("Starting login process...")
            rh_ok = rh_login(RH_USERNAME, RH_PASSWORD, RH_TOTP_SECRET)
            if rh_ok:
                print("Verification successful!")
        if CRYPTO_EXCHANGE == "kraken":
            ex = ccxt_kraken(CRYPTO_API_KEY, CRYPTO_API_SECRET, CRYPTO_API_PASSPHRASE)

    spent_eq = 0.0
    spent_cr = 0.0
    combined_rows = []

    for sym in symbols:
        is_crypto = looks_like_crypto(sym)

        # Try to get data + indicators
        df = None
        skip_reason = ""
        try:
            df = yf.download(sym, start=START, end=(END or None), progress=False, auto_adjust=False, group_by="column", repair=True)
            if df is None or df.empty:
                raise ValueError("Empty dataframe from yfinance")
            df = compute_indicators(df)
        except Exception as e:
            skip_reason = f"data_error={e}"
            # Write a ROUTE now so you see it even on failure
            print(f"ROUTE: {sym} is_crypto={is_crypto} side=skip reason={skip_reason} eq_notional={EQ_USD:.1f} cr_notional={CR_USD:.1f} dry_run={DRY_RUN}")
            combined_rows.append({
                "Date": today_str(),
                "Ticker": sym,
                "Close": "",
                "RSI": "",
                "BB_high": "",
                "BB_low": "",
                "Signal": "Hold"
            })
            continue

        last = df.iloc[-1]
        signal = str(last["Signal"]).lower()

        # Decide side
        if is_crypto and FORCE_SIDE_CRYPTO in ("buy", "sell"):
            side = FORCE_SIDE_CRYPTO
        else:
            side = "buy" if signal == "buy" else "sell" if signal == "sell" else "hold"

        # Daily-cap limiting
        eq_notional = EQ_USD
        cr_notional = CR_USD
        cap_note = ""
        if side in ("buy", "sell"):
            if is_crypto and DAILY_CRYPTO_CAP > 0:
                allowed = max(0.0, DAILY_CRYPTO_CAP - spent_cr)
                if cr_notional > allowed:
                    if allowed <= 0:
                        side, cap_note = "hold", "crypto_cap_reached"
                    else:
                        cr_notional = allowed
                        cap_note = f"crypto_cap_limited_to_{allowed:.2f}"
            if (not is_crypto) and DAILY_EQUITY_CAP > 0:
                allowed = max(0.0, DAILY_EQUITY_CAP - spent_eq)
                if eq_notional > allowed:
                    if allowed <= 0:
                        side, cap_note = "hold", "equity_cap_reached"
                    else:
                        eq_notional = allowed
                        cap_note = f"equity_cap_limited_to_{allowed:.2f}"

        # ROUTE line (always)
        if cap_note:
            print(f"ROUTE: {sym} is_crypto={is_crypto} side={side} note={cap_note} eq_notional={eq_notional:.1f} cr_notional={cr_notional:.1f} dry_run={DRY_RUN}")
        else:
            print(f"ROUTE: {sym} is_crypto={is_crypto} side={side} eq_notional={eq_notional:.1f} cr_notional={cr_notional:.1f} dry_run={DRY_RUN}")

        # Execute / simulate
        result = {}
        if side in ("buy", "sell"):
            if DRY_RUN:
                result = {"dry_run": True, "symbol": sym, "side": side, "notional": (cr_notional if is_crypto else eq_notional)}
            else:
                if is_crypto and ex is not None and CRYPTO_EXCHANGE == "kraken":
                    result = ccxt_market_notional(ex, to_ccxt_symbol(sym), side, cr_notional)
                elif (not is_crypto) and rh_ok and EQUITY_BROKER == "robinhood":
                    result = rh_place_notional(sym, side, eq_notional)
                else:
                    result = {"skipped": "no_broker"}

            # Update caps after we attempt
            if is_crypto:
                spent_cr += float(cr_notional)
            else:
                spent_eq += float(eq_notional)
        else:
            result = {"skipped": "hold"}

        # Console echo similar to prior logs
        prefix = "CRYPTO" if is_crypto else "EQUITY"
        print(f"{prefix} {sym} {side} -> {json.dumps(result)}")

        # Persist CSVs
        # Analysis
        analysis_out = os.path.join(OUT_DIR, f"{sym.replace('/', '-')}_analysis.csv")
        df_out = df.copy()
        df_out = df_out.rename(columns={"Close": "Price"})
        cols = [c for c in ["Price", "Close", "High", "Low", "Open", "Volume", "RSI", "BB_high", "BB_low", "Signal"] if c in df_out.columns]
        df_out[cols].to_csv(analysis_out)

        # Per-symbol trade log (last row only)
        trade_row = pd.DataFrame([{
            "Date": df.index[-1].strftime("%-m/%-d/%Y") if hasattr(df.index[-1], "strftime") else str(df.index[-1]),
            "Ticker": sym,
            "Close": float(last["Close"]),
            "RSI": float(last["RSI"]),
            "BB_high": float(last["BB_high"]),
            "BB_low": float(last["BB_low"]),
            "Signal": last["Signal"]
        }])
        trade_out = os.path.join(OUT_DIR, f"{sym.replace('/', '-')}_trade_log.csv")
        trade_row.to_csv(trade_out, index=False)

        combined_rows.append(trade_row.iloc[0].to_dict())

    # Combined log
    if combined_rows:
        pd.DataFrame(combined_rows).to_csv(os.path.join(OUT_DIR, "combined_trade_log.csv"), index=False)

    # Tail
    if DRY_RUN:
        print("Dry run = True. Orders NOT placed. Files saved to out/")
    else:
        print("Dry run = False. Orders placed where applicable. Files saved to out/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(2)
