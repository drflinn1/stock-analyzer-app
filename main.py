#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Headless trader main:
- Reads environment variables (workflow inputs)
- Pulls history via yfinance and computes RSI + Bollinger Bands
- Decides Buy/Sell per latest bar (or honors FORCE_SIDE for crypto)
- Routes orders to Robinhood (equities) and Kraken (crypto) with optional dry-run
- Saves analysis/trade logs to out/
- NEW: Daily caps for spend (equity & crypto) via DAILY_EQUITY_CAP / DAILY_CRYPTO_CAP
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# TA indicators
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Robinhood (equities)
import robin_stocks.robinhood as rh
import pyotp

# Crypto (Kraken via ccxt)
import ccxt


# =============================================================================
# Helpers / config
# =============================================================================

def _to_bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name, "")
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ===== Daily Spend Cap (optional) ============================================
# Read from env (0 = disable caps)
DAILY_EQUITY_CAP = _env_float("DAILY_EQUITY_CAP", 0.0)
DAILY_CRYPTO_CAP = _env_float("DAILY_CRYPTO_CAP", 0.0)

# Running totals for this process execution
_spent_equity = 0.0
_spent_crypto = 0.0

def _cap_notional(is_crypto: bool, symbol: str, side: str, notional_usd: float):
    """
    Clamp notional to remaining cap for this asset class; return (new_notional, skip_reason_or_None).
    If cap is disabled (0), just return the original notional.
    """
    global _spent_equity, _spent_crypto
    cap = DAILY_CRYPTO_CAP if is_crypto else DAILY_EQUITY_CAP
    spent = _spent_crypto if is_crypto else _spent_equity

    if cap <= 0:
        return notional_usd, None

    remaining = max(0.0, cap - spent)
    if remaining <= 0:
        return 0.0, "cap_reached"

    if notional_usd > remaining:
        print(f"CAP: Clamping {symbol} {side} from ${notional_usd:.2f} to ${remaining:.2f} (remaining under cap)")
        notional_usd = remaining

    return notional_usd, None

def _record_spend(is_crypto: bool, notional_usd: float):
    """
    Increment today's spend trackers (called pre-submit for safety).
    """
    global _spent_equity, _spent_crypto
    if notional_usd <= 0:
        return
    if is_crypto:
        _spent_crypto += notional_usd
    else:
        _spent_equity += notional_usd
# ============================================================================


# =============================================================================
# Inputs from environment (set by GitHub Actions)
# =============================================================================
DRY_RUN = _to_bool(os.getenv("DRY_RUN", "true"))
SYMBOLS_RAW = os.getenv("SYMBOLS", "").strip()
START = os.getenv("START", "2023-01-01").strip()
END = os.getenv("END", "").strip() or None

EQUITY_DOLLARS_PER_TRADE = _env_float("EQUITY_DOLLARS_PER_TRADE", 200.0)
CRYPTO_DOLLARS_PER_TRADE = _env_float("CRYPTO_DOLLARS_PER_TRADE", 10.0)

FORCE_SIDE = (os.getenv("FORCE_SIDE", "") or "").strip().lower()  # for crypto only
CRYPTO_AUTOPICK = _to_bool(os.getenv("CRYPTO_AUTOPICK", "false"))
EQUITY_AUTOPICK = _to_bool(os.getenv("EQUITY_AUTOPICK", "false"))
AUTOPICK_OVERRIDES = (os.getenv("AUTOPICK_OVERRIDES", "") or "").strip()

OUT_DIR = Path(os.getenv("OUT_DIR", "out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EQUITY_BROKER = (os.getenv("EQUITY_BROKER", "robinhood") or "").strip().lower()
CRYPTO_EXCHANGE = (os.getenv("CRYPTO_EXCHANGE", "kraken") or "").strip().lower()

# Secrets (Robinhood)
RH_USERNAME = os.getenv("RH_USERNAME", "")
RH_PASSWORD = os.getenv("RH_PASSWORD", "")
RH_TOTP_SECRET = os.getenv("RH_TOTP_SECRET", "")

# Secrets (Kraken)
KRAKEN_KEY = os.getenv("CRYPTO_API_KEY", "")
KRAKEN_SECRET = os.getenv("CRYPTO_API_SECRET", "")
KRAKEN_PASSPHRASE = os.getenv("CRYPTO_API_PASSPHRASE", "")

# Derived flags
equity_enabled = True if EQUITY_BROKER == "robinhood" else False
crypto_enabled = True if CRYPTO_EXCHANGE == "kraken" else False


# =============================================================================
# Data / signals
# =============================================================================

def compute_signals(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download OHLC and compute RSI + Bollinger bands + simple Buy/Sell signal."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns=str.title)  # columns to Title-case like 'Close'
    # RSI(14)
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi.rsi()

    # Bollinger(20, 2)
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2.0)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    # Simple rule:
    # Buy when price < BB_low and RSI < 35
    # Sell when price > BB_high and RSI > 65
    cond_buy = (df["Close"] < df["BB_low"]) & (df["RSI"] < 35)
    cond_sell = (df["Close"] > df["BB_high"]) & (df["RSI"] > 65)
    signal = np.where(cond_buy, "Buy", np.where(cond_sell, "Sell", ""))

    df["Signal"] = signal
    df.reset_index(inplace=True)
    df["Ticker"] = ticker

    # Save analysis CSV
    out = OUT_DIR / f"{ticker.replace('/', '-')}_analysis.csv"
    df[["Date", "Ticker", "Close", "RSI", "BB_high", "BB_low", "Signal"]].to_csv(out, index=False)
    return df


def latest_signal(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    s = str(df.iloc[-1]["Signal"])
    return (s or "").strip().title()


def is_crypto_symbol(symbol: str) -> bool:
    # convention in this project: crypto like 'BTC-USD', 'ETH-USD'
    return "-USD" in symbol.upper()


def kraken_pair(symbol: str) -> str:
    # Convert 'BTC-USD' => 'XBT/USD' (Kraken uses XBT)
    sym = symbol.upper()
    base = sym.split("-")[0]
    quote = sym.split("-")[1]
    if base == "BTC":
        base = "XBT"
    return f"{base}/{quote}"


# =============================================================================
# Brokers
# =============================================================================

def login_robinhood():
    if not equity_enabled:
        return
    print("Starting login process...")
    mfa = None
    if RH_TOTP_SECRET:
        try:
            mfa = pyotp.TOTP(RH_TOTP_SECRET).now()
        except Exception:
            mfa = None
    try:
        rh.login(username=RH_USERNAME, password=RH_PASSWORD, mfa_code=mfa, store_session=False)
    except Exception as e:
        # Let library drive SMS/app approval if needed
        print("Verification required, handling challenge...")
        time.sleep(2)
        try:
            rh.login(username=RH_USERNAME, password=RH_PASSWORD, mfa_code=mfa, store_session=False)
        except Exception as e2:
            print(f"Robinhood login failed: {e2}")
            raise
    print("Verification successful!")


def kraken_client():
    if not crypto_enabled:
        return None
    return ccxt.kraken({
        "apiKey": KRAKEN_KEY,
        "secret": KRAKEN_SECRET,
        "enableRateLimit": True,
    })


def place_equity_order(symbol: str, side: str, notional: float, dry_run: bool):
    """Fractional by notional (USD)."""
    side = side.lower()
    if notional <= 0:
        return {"skipped": "notional_zero", "symbol": symbol, "side": side}

    if dry_run:
        return {"dry_run": True, "symbol": symbol, "side": side, "notional": float(notional)}

    try:
        if side == "buy":
            res = rh.orders.order_buy_fractional_by_price(symbol, float(notional))
        elif side == "sell":
            res = rh.orders.order_sell_fractional_by_price(symbol, float(notional))
        else:
            return {"skipped": "invalid_side", "symbol": symbol, "side": side}
        return res
    except Exception as e:
        return {"error": str(e), "symbol": symbol, "side": side}


def place_crypto_order(kraken: ccxt.kraken, symbol: str, side: str, notional_usd: float, dry_run: bool):
    """Market order by spending USD notional."""
    side = side.lower()
    if notional_usd <= 0:
        return {"skipped": "notional_zero", "symbol": symbol, "side": side}

    pair = kraken_pair(symbol)
    try:
        px = kraken.fetch_ticker(pair)["last"]
    except Exception as e:
        return {"error": f"price_fetch_failed: {e}", "symbol": symbol, "side": side}

    amount = round(float(notional_usd) / float(px), 10)
    if amount <= 0:
        return {"skipped": "amount_zero", "symbol": symbol, "side": side}

    if dry_run:
        return {"dry_run": True, "symbol": symbol, "side": side, "notional_usd": float(notional_usd), "est_amount": amount}

    try:
        order = kraken.create_order(symbol=pair, type="market", side=side, amount=amount)
        return order
    except ccxt.BaseError as e:
        return {"error": str(e), "symbol": symbol, "side": side}


# =============================================================================
# Main routing
# =============================================================================

def main():
    # Parse symbols
    raw = SYMBOLS_RAW
    if raw.strip() == "":
        # if user left it blank in the form, we still want *something*
        # keep it predictable:
        syms = []
        if EQUITY_AUTOPICK:
            syms.append("SPY")
        if CRYPTO_AUTOPICK or FORCE_SIDE in ("buy", "sell"):
            syms.append("BTC-USD")
        if not syms:
            syms = ["SPY", "BTC-USD"]
    else:
        syms = [s.strip() for s in raw.split(",") if s.strip()]

    # Log configuration
    print("CONFIG:")
    print(f"  dry_run={str(DRY_RUN).lower()}")
    print(f"  symbols={','.join(syms)}")
    print(f"  start={START} end={END or ''}")
    print(f"  crypto_autopick={str(CRYPTO_AUTOPICK).lower()} equity_autopick={str(EQUITY_AUTOPICK).lower()}")
    print(f"  force_side={FORCE_SIDE or ''}")
    print(f"  daily caps: equity={DAILY_EQUITY_CAP} crypto={DAILY_CRYPTO_CAP}")

    # Broker sessions
    if equity_enabled:
        login_robinhood()
    client = kraken_client() if crypto_enabled else None

    combined_rows = []
    results = []

    for symbol in syms:
        is_crypto = is_crypto_symbol(symbol)

        # Download + compute
        try:
            df = compute_signals(symbol, START, END)
        except Exception as e:
            print(f"Data error for {symbol}: {e}")
            continue

        # Decide side
        sig = latest_signal(df)
        side = sig.lower() if sig else ""

        # For crypto only, FORCE_SIDE can override
        if is_crypto and FORCE_SIDE in ("buy", "sell"):
            side = FORCE_SIDE

        # If still blank signal (no strong condition), default to 'sell' to trim winners
        if side not in ("buy", "sell"):
            side = "sell"

        # Notionals
        eq_notional = EQUITY_DOLLARS_PER_TRADE
        cr_notional = CRYPTO_DOLLARS_PER_TRADE

        # ROUTE log (before caps)
        print(f"ROUTE: {symbol} is_crypto={is_crypto} side={side} eq_notional={eq_notional} cr_notional={cr_notional} dry_run={str(DRY_RUN).lower()}")

        # ----- Daily cap enforcement (in-run)
        notional_usd = cr_notional if is_crypto else eq_notional
        notional_usd, _cap_skip = _cap_notional(is_crypto, symbol, side, notional_usd)
        if _cap_skip:
            skip_line = f"{'CRYPTO' if is_crypto else 'EQUITY'} {symbol} skip -> cap_reached (cap=${DAILY_CRYPTO_CAP if is_crypto else DAILY_EQUITY_CAP:.2f})"
            print(skip_line)
            results.append({'skipped': 'cap_reached', 'symbol': symbol, 'side': side})
            continue

        # reflect the clamped notional
        if is_crypto:
            cr_notional = notional_usd
        else:
            eq_notional = notional_usd

        # Record spend pre-submit (conservative)
        if not DRY_RUN and notional_usd > 0:
            _record_spend(is_crypto, notional_usd)
        # ---------------------------------------------------------------------

        # Route to broker
        if is_crypto:
            if not crypto_enabled:
                results.append({"skipped": "crypto_disabled", "symbol": symbol})
            else:
                res = place_crypto_order(client, symbol, side, cr_notional, DRY_RUN)
                results.append(res)
                if isinstance(res, dict):
                    print(f"CRYPTO {symbol} {side} -> {res}")
        else:
            if not equity_enabled:
                results.append({"skipped": "equity_disabled", "symbol": symbol})
            else:
                res = place_equity_order(symbol, side, eq_notional, DRY_RUN)
                results.append(res)
                if isinstance(res, dict):
                    print(f"EQUITY {symbol} {side} -> {res}")

        # Save per-ticker trade log snapshot
        if df is not None and not df.empty:
            out = OUT_DIR / f"{symbol.replace('/', '-')}_trade_log.csv"
            df[["Date", "Ticker", "Close", "RSI", "BB_high", "BB_low", "Signal"]].to_csv(out, index=False)
            combined_rows.append(df[["Date", "Ticker", "Close", "RSI", "BB_high", "BB_low", "Signal"]])

    # Combined
    if combined_rows:
        big = pd.concat(combined_rows, ignore_index=True)
        big.to_csv(OUT_DIR / "combined_trade_log.csv", index=False)

    # Done
    print(f"Dry run = {str(DRY_RUN).title()}. Orders {'NOT ' if DRY_RUN else ''}placed. Files saved to {OUT_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(1)
