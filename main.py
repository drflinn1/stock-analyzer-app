#!/usr/bin/env python3
"""
Stock Analyzer / Crypto Live Runner (Series-safe)

- Uses yfinance for market data
- RSI + Bollinger sample strategy, collapsed to LAST candle booleans
- Env-controlled behavior:
    MODE: "live" | "paper"
    DRY_RUN: "true" | "false"     # if "true", never places real orders
    PROFILE: "cautious" | "aggressive"
    DROP_PCT: "0.0" or "2.0" etc   # minimum dip vs recent high to allow buys
    TRADE_SIZE: "$" float as string (e.g., "10")
    DAILY_CAP: "$" float (e.g., "15")
    CRYPTO_TICKERS: comma list (default: BTC-USD,ETH-USD)
    LOOKBACK_MIN: candles to fetch (default 200)
    INTERVAL: yfinance interval (default "5m")
- Persists daily spend cap under out/caps_YYYYMMDD.json
- Writes run artifacts to out/

This file intentionally avoids any ambiguous pandas truth checks.
"""

import os
import json
import math
import time
import shutil
import pathlib
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(f"Missing dependency yfinance. Add to requirements.txt. Error: {e}")

OUT_DIR = pathlib.Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------
# Utilities
# ---------------------------

def env_str(name: str, default: str) -> str:
    v = os.environ.get(name, default)
    return v

def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def write_json(path: pathlib.Path, obj: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    shutil.move(tmp, path)

def read_json(path: pathlib.Path, default: Any):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def today_caps_path() -> pathlib.Path:
    d = datetime.now().strftime("%Y%m%d")
    return OUT_DIR / f"caps_{d}.json"

def load_caps() -> Dict[str, Any]:
    return read_json(today_caps_path(), {"spent_usd": 0.0})

def save_caps(caps: Dict[str, Any]):
    write_json(today_caps_path(), caps)


# ---------------------------
# Indicators (Series-safe)
# ---------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(span=period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val.fillna(50.0)

def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(window=window).mean()
    sd = series.rolling(window=window).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return lower, ma, upper


# ---------------------------
# Data
# ---------------------------

def fetch_ohlc(ticker: str, interval: str, lookback: int) -> pd.DataFrame:
    # yfinance requires period when using intraday intervals. Use a safe mapping.
    # For 5m data, period cannot exceed ~60d. We'll ask for "7d" to be safe and then trim.
    period_map = {
        "1m": "7d",
        "2m": "30d",
        "5m": "30d",
        "15m": "60d",
        "30m": "60d",
        "60m": "730d",  # 2y
        "90m": "730d",
        "1h": "730d",
        "1d": "5y",
    }
    period = period_map.get(interval, "30d")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"No data for {ticker} @ {interval}/{period}")
    # yfinance columns: ['Open','High','Low','Close','Adj Close','Volume']
    df = df.rename(columns=str.lower)
    # keep last N rows to make the indicator math stable but light
    df = df.tail(int(lookback))
    df.dropna(inplace=True)
    return df


# ---------------------------
# Strategy (collapse to latest)
# ---------------------------

def decide_signal(df: pd.DataFrame, profile: str, drop_pct_gate: float) -> Dict[str, Any]:
    """
    Returns dict with:
      side: "buy" | "sell" | "skip"
      reason: str
    Never returns a pandas Series as a condition — all checks use the LAST candle.
    """
    close = df["close"]
    # Indicators
    rsi_s = rsi(close, 14)
    bb_lower, bb_mid, bb_upper = bollinger(close, 20, 2.0)

    # Scalars (last candle only)
    price = float(close.iloc[-1])
    rsi_last = float(rsi_s.iloc[-1])
    lower_hit = bool((close.iloc[-1] < bb_lower.iloc[-1]))
    upper_hit = bool((close.iloc[-1] > bb_upper.iloc[-1]))

    # Simple trailing-high dip gate (past X bars)
    window = 96  # ~8 hours on 5m bars
    hi_window = float(close.tail(window).max())
    dip_from_high = (hi_window - price) / hi_window if hi_window > 0 else 0.0
    dip_ok = dip_from_high >= (drop_pct_gate / 100.0) if drop_pct_gate > 0 else True

    # Profile-specific tweaks
    if profile == "cautious":
        # Gentle: buy if price < lower band OR RSI < 35; sell if price > upper band OR RSI > 65
        want_buy = (lower_hit or (rsi_last < 35.0)) and dip_ok
        want_sell = (upper_hit or (rsi_last > 65.0))
    else:  # aggressive
        # Tighter to bands, stronger RSI swing
        want_buy = (lower_hit and (rsi_last < 30.0)) and dip_ok
        want_sell = (upper_hit and (rsi_last > 70.0))

    if want_buy and not want_sell:
        return {"side": "buy", "reason": f"buy_cond: lower_hit={lower_hit} rsi={rsi_last:.1f} dip_ok={dip_ok}"}
    if want_sell and not want_buy:
        return {"side": "sell", "reason": f"sell_cond: upper_hit={upper_hit} rsi={rsi_last:.1f}"}
    return {"side": "skip", "reason": "neutral_or_conflict"}


# ---------------------------
# Execution (dry run vs live)
# ---------------------------

def place_order_live(symbol: str, side: str, notional_usd: float) -> Dict[str, Any]:
    """
    Live order stub. We keep this minimal so DRY_RUN=true users are safe.
    If you want true live crypto via Kraken with ccxt, fill credentials in env and uncomment.
    """
    # Example: integrate ccxt if available and creds present
    try:
        import ccxt  # type: ignore
        api_key = os.environ.get("KRAKEN_API_KEY")
        api_secret = os.environ.get("KRAKEN_API_SECRET")
        if not api_key or not api_secret:
            return {"ok": False, "message": "ccxt live disabled: missing KRAKEN_API_KEY/SECRET"}

        ex = ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

        # market buy/sell by cost
        base = symbol.split("-")[0] if "-" in symbol else symbol.split("/")[0]
        market = symbol.replace("-", "/")
        # fetch price to approximate amount
        ticker = ex.fetch_ticker(market)
        price = float(ticker["last"])
        amount = round(notional_usd / price, 8) if price > 0 else None
        if not amount or amount <= 0:
            return {"ok": False, "message": f"bad amount for {symbol} @ {price}"}

        if side == "buy":
            order = ex.create_market_buy_order(market, amount)
        else:
            order = ex.create_market_sell_order(market, amount)

        return {"ok": True, "message": "live order placed", "order": order}
    except Exception as e:
        return {"ok": False, "message": f"live error: {e}"}


def simulate_order(symbol: str, side: str, notional_usd: float) -> Dict[str, Any]:
    return {"ok": True, "message": f"DRY_RUN {side} ${notional_usd:.2f} {symbol}"}


# ---------------------------
# Run
# ---------------------------

def run() -> int:
    mode = env_str("MODE", "live")  # informational
    dry_run = env_bool("DRY_RUN", True)
    profile = env_str("PROFILE", "cautious").lower().strip()
    drop_pct = env_float("DROP_PCT", 0.0)
    trade_size = env_float("TRADE_SIZE", 10.0)
    daily_cap = env_float("DAILY_CAP", 15.0)
    interval = env_str("INTERVAL", "5m")
    lookback = int(env_float("LOOKBACK_MIN", 200))
    tickers_env = env_str("CRYPTO_TICKERS", "BTC-USD,ETH-USD")
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]

    # Banner
    log(f"START: mode={mode} dry_run={dry_run} profile={profile} drop_pct={drop_pct} "
        f"trade_size={trade_size} daily_cap={daily_cap} interval={interval} lookback={lookback}")
    log(f"TICKERS: {', '.join(tickers)}")

    # Load daily cap state
    caps = load_caps()
    spent = float(caps.get("spent_usd", 0.0))
    remaining = max(0.0, daily_cap - spent)
    log(f"DAILY CAP: spent=${spent:.2f} remaining=${remaining:.2f} (cap=${daily_cap:.2f})")

    rows = []

    for sym in tickers:
        route = {
            "symbol": sym,
            "is_crypto": True,
            "side": "skip",
            "reason": "",
            "error": "",
            "notional": 0.0,
            "ts": now_utc_iso(),
            "dry_run": dry_run,
        }
        try:
            df = fetch_ohlc(sym, interval=interval, lookback=lookback)
            dec = decide_signal(df, profile=profile, drop_pct_gate=drop_pct)
            side = dec["side"]
            reason = dec["reason"]

            if side in ("buy", "sell"):
                if side == "buy" and remaining <= 0.0:
                    route.update({"side": "skip", "reason": "cap_reached"})
                else:
                    notional = trade_size if side == "buy" else trade_size  # symmetric
                    if side == "buy":
                        # enforce cap
                        notional = min(notional, remaining)
                        if notional <= 0:
                            route.update({"side": "skip", "reason": "cap_reached"})
                        else:
                            # place
                            if dry_run:
                                res = simulate_order(sym, side, notional)
                            else:
                                res = place_order_live(sym, side, notional)
                            route.update({
                                "side": side,
                                "reason": reason,
                                "notional": float(notional),
                                "live_ok": bool(res.get("ok")),
                                "live_msg": str(res.get("message")),
                            })
                            if res.get("ok") and side == "buy":
                                spent += notional
                                remaining = max(0.0, daily_cap - spent)
                    else:
                        # sell (no cap reduction)
                        if dry_run:
                            res = simulate_order(sym, side, notional)
                        else:
                            res = place_order_live(sym, side, notional)
                        route.update({
                            "side": side,
                            "reason": reason,
                            "notional": float(notional),
                            "live_ok": bool(res.get("ok")),
                            "live_msg": str(res.get("message")),
                        })
            else:
                route.update({"side": "skip", "reason": reason})

        except Exception as e:
            route.update({
                "side": "skip",
                "reason": "data_error",
                "error": f"{type(e).__name__}: {e}",
            })
            log(f"ERROR {sym}: {e}\n{traceback.format_exc()}")

        rows.append(route)
        # Pretty route line (similar to your logs)
        log(f"ROUTE: {sym} is_crypto=True side={route['side']} reason={route['reason']} "
            f"eq_notional=0 cr_notional={route.get('notional', 0.0)} dry_run={dry_run}")

    # Save cap state
    caps["spent_usd"] = round(spent, 2)
    save_caps(caps)

    # Write CSV/JSON artifact for each run
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"routes_{ts_tag}.json"
    csv_path = OUT_DIR / f"routes_{ts_tag}.csv"
    try:
        write_json(json_path, rows)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log(f"Saved: {json_path} and {csv_path}")
    except Exception as e:
        log(f"Artifact save error: {e}")

    log(f"END: spent=${spent:.2f} remaining=${max(0.0, daily_cap - spent):.2f} dry_run={dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
