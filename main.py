#!/usr/bin/env python3
"""
Stock Analyzer / Crypto Live Runner
- Series-safe & 1-D hardened (no ambiguous pandas truth checks)
- Robust empty/sparse-data handling + interval fallback (5m→15m→30m→1h→1d)
- Auto-pick top performers (24h change) when CRYPTO_AUTOPICK=true
- Env knobs:
    MODE: "live" | "paper" (informational)
    DRY_RUN: "true" | "false"
    PROFILE: "cautious" | "aggressive"
    DROP_PCT: "0.0" | "2.0" ...
    TRADE_SIZE: float USD
    DAILY_CAP: float USD
    INTERVAL: primary interval for strategy (default "5m")
    LOOKBACK_MIN: rows to fetch for strategy calc (default 200)
    MIN_BARS: minimum bars required after cleaning (default 30)
    CRYPTO_AUTOPICK: "true" | "false" (default true)
    AUTOPICK_TOP_N: integer (default 4)
    CRYPTO_UNIVERSE: comma list (default sample majors)
    CRYPTO_TICKERS: used only if CRYPTO_AUTOPICK=false
- Persists daily cap to out/caps_YYYYMMDD.json
- Saves artifacts to out/routes_*.json/csv
"""

import os
import json
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

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

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
# 1D helpers
# ---------------------------

def to_series_1d(x) -> pd.Series:
    """Coerce input to a 1-D float Series."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        x = x.iloc[:, 0]
    elif isinstance(x, np.ndarray):
        x = pd.Series(np.ravel(x))
    elif isinstance(x, (list, tuple)):
        x = pd.Series(list(x))
    elif isinstance(x, pd.Series):
        pass
    else:
        x = pd.Series(x)
    return pd.to_numeric(x, errors="coerce")

# ---------------------------
# Indicators (Series-safe)
# ---------------------------

def rsi(src, period: int = 14) -> pd.Series:
    close = to_series_1d(src).copy()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.ewm(span=period, adjust=False).mean()
    roll_down = loss.ewm(span=period, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)

def bollinger(src, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    s = to_series_1d(src)
    ma = s.rolling(window=window).mean()
    sd = s.rolling(window=window).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return lower, ma, upper

# ---------------------------
# Data
# ---------------------------

def _period_for_interval(interval: str) -> str:
    return {
        "1m": "7d",
        "2m": "30d",
        "5m": "30d",
        "15m": "60d",
        "30m": "60d",
        "60m": "730d",
        "90m": "730d",
        "1h": "730d",
        "1d": "5y",
    }.get(interval, "30d")

def fetch_ohlc_once(ticker: str, interval: str, lookback: int) -> pd.DataFrame:
    period = _period_for_interval(interval)
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or not isinstance(df, (pd.DataFrame, pd.Series)) or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns=str.lower)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            if c == "close" and "adj close" in df.columns:
                df["close"] = df["adj close"]
            else:
                df[c] = np.nan
    df = df.tail(int(lookback)).copy()
    df.dropna(subset=["close"], inplace=True)
    if "close" not in df or df["close"].empty:
        return pd.DataFrame()
    df["close"] = to_series_1d(df["close"]).values
    return df

def fetch_ohlc_with_fallback(ticker: str, interval: str, lookback: int, min_bars: int) -> Tuple[pd.DataFrame, str]:
    """
    Try primary interval, then fall back through 15m→30m→1h→1d until we get >= min_bars.
    Returns (df, used_interval) where df may be empty if all fail.
    """
    chain = [interval]
    for alt in ["15m", "30m", "1h", "1d"]:
        if alt not in chain:
            chain.append(alt)

    for iv in chain:
        df = fetch_ohlc_once(ticker, iv, lookback)
        if not df.empty and len(df) >= max(10, min_bars):  # allow a little slack
            return df, iv
    return pd.DataFrame(), chain[-1]

# 24h perf for ranking (independent of strategy interval)
def pct_change_24h(ticker: str) -> float:
    try:
        hist = yf.download(ticker, period="2d", interval="1h", progress=False)
        if hist is None or len(hist) < 2:
            return float("-inf")
        hist = hist.rename(columns=str.lower)
        close = to_series_1d(hist.get("close", hist.get("adj close", [])))
        close.dropna(inplace=True)
        if len(close) < 2:
            return float("-inf")
        last = float(close.iloc[-1])
        idx_24 = -24 if len(close) >= 25 else -(len(close)-1)
        base = float(close.iloc[idx_24])
        if base <= 0:
            return float("-inf")
        return (last - base) / base
    except Exception:
        return float("-inf")

def rank_top_performers(universe: List[str], top_n: int) -> List[str]:
    scores = []
    for t in universe:
        p = pct_change_24h(t)
        scores.append((t, p))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scores[:max(1, top_n)]]

# ---------------------------
# Strategy (collapse to latest)
# ---------------------------

def decide_signal(df: pd.DataFrame, profile: str, drop_pct_gate: float, min_bars: int) -> Dict[str, Any]:
    """
    Return dict: {side: buy|sell|skip, reason: str}
    Uses only last candle scalars to avoid ambiguous boolean checks.
    """
    if df is None or df.empty or "close" not in df or len(df["close"]) < max(10, min_bars):
        return {"side": "skip", "reason": "insufficient_data"}

    close = to_series_1d(df["close"])

    # Indicators
    rsi_s = rsi(close, 14)
    bb_lower, bb_mid, bb_upper = bollinger(close, 20, 2.0)

    # Scalars (last candle only)
    price = float(close.iloc[-1])
    rsi_last = float(rsi_s.iloc[-1])
    lower_hit = bool(price < float(bb_lower.iloc[-1]))
    upper_hit = bool(price > float(bb_upper.iloc[-1]))

    # Trailing-high dip gate
    window = min(max(30, len(close)//2), 300)
    hi_window = float(close.tail(window).max())
    dip_from_high = (hi_window - price) / hi_window if hi_window > 0 else 0.0
    dip_ok = (dip_from_high >= (drop_pct_gate / 100.0)) if drop_pct_gate > 0 else True

    if profile == "cautious":
        want_buy = (lower_hit or (rsi_last < 35.0)) and dip_ok
        want_sell = (upper_hit or (rsi_last > 65.0))
    else:
        want_buy = (lower_hit and (rsi_last < 30.0)) and dip_ok
        want_sell = (upper_hit and (rsi_last > 70.0))

    if want_buy and not want_sell:
        return {"side": "buy", "reason": f"buy_cond: lower_hit={lower_hit} rsi={rsi_last:.1f} dip_ok={dip_ok}"}
    if want_sell and not want_buy:
        return {"side": "sell", "reason": f"sell_cond: upper_hit={upper_hit} rsi={rsi_last:.1f}"}
    return {"side": "skip", "reason": "neutral_or_conflict"}

# ---------------------------
# Execution (dry vs live)
# ---------------------------

def simulate_order(symbol: str, side: str, notional_usd: float) -> Dict[str, Any]:
    return {"ok": True, "message": f"DRY_RUN {side} ${notional_usd:.2f} {symbol}"}

def place_order_live(symbol: str, side: str, notional_usd: float) -> Dict[str, Any]:
    """Live order stub via ccxt if keys exist; otherwise returns disabled."""
    try:
        import ccxt  # type: ignore
        api_key = os.environ.get("KRAKEN_API_KEY")
        api_secret = os.environ.get("KRAKEN_API_SECRET")
        if not api_key or not api_secret:
            return {"ok": False, "message": "ccxt live disabled (missing KRAKEN_API_KEY/SECRET)"}
        ex = ccxt.kraken({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
        market = symbol.replace("-", "/")
        ticker = ex.fetch_ticker(market)
        price = float(ticker["last"])
        amount = round(notional_usd / price, 8) if price > 0 else None
        if not amount or amount <= 0:
            return {"ok": False, "message": f"bad amount for {symbol} @ {price}"}
        order = ex.create_market_buy_order(market, amount) if side == "buy" else ex.create_market_sell_order(market, amount)
        return {"ok": True, "message": "live order placed", "order": order}
    except Exception as e:
        return {"ok": False, "message": f"live error: {e}"}

# ---------------------------
# Run
# ---------------------------

def run() -> int:
    mode = env_str("MODE", "live")
    dry_run = env_bool("DRY_RUN", True)
    profile = env_str("PROFILE", "cautious").lower().strip()
    drop_pct = env_float("DROP_PCT", 0.0)
    trade_size = env_float("TRADE_SIZE", 10.0)
    daily_cap = env_float("DAILY_CAP", 15.0)
    interval = env_str("INTERVAL", "5m")
    lookback = int(env_float("LOOKBACK_MIN", 200))
    min_bars = env_int("MIN_BARS", 30)

    autopick = env_bool("CRYPTO_AUTOPICK", True)
    top_n = env_int("AUTOPICK_TOP_N", 4)
    default_universe = "BTC-USD,ETH-USD,SOL-USD,ADA-USD,XRP-USD,LTC-USD,DOGE-USD,AVAX-USD,BNB-USD,MATIC-USD"
    universe_env = env_str("CRYPTO_UNIVERSE", default_universe)
    universe = [t.strip() for t in universe_env.split(",") if t.strip()]

    if autopick:
        tickers = rank_top_performers(universe, top_n)
        log(f"AUTOPICK: top_{top_n} of {len(universe)} by ~24h change → {', '.join(tickers)}")
    else:
        tickers_env = env_str("CRYPTO_TICKERS", "BTC-USD,ETH-USD")
        tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]

    log(f"START: mode={mode} dry_run={dry_run} profile={profile} drop_pct={drop_pct} "
        f"trade_size={trade_size} daily_cap={daily_cap} interval={interval} lookback={lookback} min_bars={min_bars}")
    log(f"TICKERS: {', '.join(tickers)}")

    caps = load_caps()
    spent = float(caps.get("spent_usd", 0.0))
    remaining = max(0.0, daily_cap - spent)
    log(f"DAILY CAP: spent=${spent:.2f} remaining=${remaining:.2f} (cap=${daily_cap:.2f})")

    rows: List[Dict[str, Any]] = []

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
            df, used_iv = fetch_ohlc_with_fallback(sym, interval=interval, lookback=lookback, min_bars=min_bars)
            if df is None or df.empty or "close" not in df or len(df["close"]) < max(10, min_bars):
                route.update({"side": "skip", "reason": f"no_data({used_iv})"})
            else:
                log(f"DATA: {sym} interval={used_iv} bars={len(df)}")
                dec = decide_signal(df, profile=profile, drop_pct_gate=drop_pct, min_bars=min_bars)
                side = dec["side"]
                reason = dec["reason"]

                if side in ("buy", "sell"):
                    if side == "buy" and remaining <= 0.0:
                        route.update({"side": "skip", "reason": "cap_reached"})
                    else:
                        notional = trade_size
                        if side == "buy":
                            notional = min(notional, remaining)
                            if notional <= 0:
                                route.update({"side": "skip", "reason": "cap_reached"})
                            else:
                                res = simulate_order(sym, side, notional) if dry_run else place_order_live(sym, side, notional)
                                route.update({"side": side, "reason": f"{reason} iv={used_iv}", "notional": float(notional),
                                              "live_ok": bool(res.get("ok")), "live_msg": str(res.get("message"))})
                                if res.get("ok") and side == "buy":
                                    spent += notional
                                    remaining = max(0.0, daily_cap - spent)
                        else:
                            res = simulate_order(sym, side, notional) if dry_run else place_order_live(sym, side, notional)
                            route.update({"side": side, "reason": f"{reason} iv={used_iv}", "notional": float(notional),
                                          "live_ok": bool(res.get("ok")), "live_msg": str(res.get("message"))})
                else:
                    route.update({"side": "skip", "reason": f"{reason} iv={used_iv}"})

        except Exception as e:
            route.update({"side": "skip", "reason": "data_error", "error": f"{type(e).__name__}: {e}"})
            log(f"ERROR {sym}: {e}\n{traceback.format_exc()}")

        rows.append(route)
        log(f"ROUTE: {sym} is_crypto=True side={route['side']} reason={route['reason']} "
            f"eq_notional=0 cr_notional={route.get('notional', 0.0)} dry_run={dry_run}")

    caps["spent_usd"] = round(spent, 2)
    save_caps(caps)

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
