"""
Equities Engine — Auto-pick momentum/trend (Alpaca IEX + yfinance fallback)

What it does
------------
1) Checks ALL current positions and sells if any exit guard is hit:
   - TREND_BREAK: EMA12 < EMA26 AND RSI < 50  (if SELL_ON_TREND_BREAK=true)
   - STOP_LOSS_PCT: price <= entry * (1 - STOP_LOSS_PCT)
   - TAKE_PROFIT_PCT: price >= entry * (1 + TAKE_PROFIT_PCT)

2) Scans a candidate universe, ranks by momentum/trend, and auto-picks top names.
   - TREND entries: EMA12 > EMA26 AND RSI > 50
   - Liquidity filter via 20d average dollar volume (IEX feed) MIN_ADV_USD
   - AVOID_REBUY to skip symbols already held
   - MIN_BUYS fallback: if fewer than MIN_BUYS orders were placed via strict rules,
     buy the best-ranked backups with a relaxed uptrend (EMA12 > EMA26)

Typical knobs (set via GitHub Actions inputs)
---------------------------------------------
  DRY_RUN=true|false
  PER_TRADE_USD=3
  DAILY_CAP_USD=12
  MARKET_ONLY=true
  AUTO_PICK=true|false
  PICKS_PER_RUN=6
  UNIVERSE=AAPL,MSFT                    # used only if AUTO_PICK=false
  SCAN_UNIVERSE=...                     # optional override
  AVOID_REBUY=true|false
  MIN_ADV_USD=2000000                   # liquidity threshold (IEX-based)
  MIN_BUYS=0                            # 0 disables fallback; 1 guarantees at least 1 buy if possible
  STOP_LOSS_PCT=0.04                    # 4% stop
  TAKE_PROFIT_PCT=0.08                  # 8% take-profit
  SELL_ON_TREND_BREAK=true              # sell when TREND turns bearish
  LOG_LEVEL=INFO

Secrets
-------
  ALPACA_API_KEY / ALPACA_SECRET_KEY
  ALPACA_PAPER=true|false (defaults true)
"""
from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

import pandas as pd

# ---------- optional yfinance fallback ----------
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ---------- Alpaca SDK ----------
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception as e:
    raise RuntimeError("alpaca-py is required. Run: pip install alpaca-py") from e

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("equities_engine")

# ---------- helpers / config ----------
def getenv_bool(k: str, d: bool) -> bool:
    v = os.getenv(k)
    return d if v is None else str(v).strip().lower() in {"1", "true", "yes", "y"}

def mask_prefix(s: str, n: int = 4) -> str:
    return "NONE" if not s else f"{s[:n]}***"

DRY_RUN       = getenv_bool("DRY_RUN", True)
MARKET_ONLY   = getenv_bool("MARKET_ONLY", True)
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "3"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "12"))

AUTO_PICK     = getenv_bool("AUTO_PICK", True)
PICKS_PER_RUN = int(float(os.getenv("PICKS_PER_RUN", "6")))
UNIVERSE      = [s.strip().upper() for s in os.getenv("UNIVERSE", "AAPL,MSFT").split(",") if s.strip()]

DEFAULT_SCAN = [
    "SPY","QQQ","DIA","IWM","AAPL","MSFT","NVDA","AMD","TSLA","META","GOOGL","AMZN","AVGO","NFLX",
    "COST","JPM","BRK.B","UNH","XOM","CVX","PEP","KO","V","MA","JNJ","HD","WMT","ORCL","CRM","INTC"
]
SCAN_UNIVERSE = [s.strip().upper() for s in os.getenv("SCAN_UNIVERSE", ",".join(DEFAULT_SCAN)).split(",") if s.strip()]

AVOID_REBUY   = getenv_bool("AVOID_REBUY", True)
MIN_ADV_USD   = float(os.getenv("MIN_ADV_USD", "2000000"))
MIN_BUYS      = int(float(os.getenv("MIN_BUYS", "0")))

STOP_LOSS_PCT       = float(os.getenv("STOP_LOSS_PCT", "0.04"))
TAKE_PROFIT_PCT     = float(os.getenv("TAKE_PROFIT_PCT", "0.08"))
SELL_ON_TREND_BREAK = getenv_bool("SELL_ON_TREND_BREAK", True)

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER      = getenv_bool("ALPACA_PAPER", True)

MIN_BARS = 120  # for indicators

# ---------- indicators ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def ema_now(df: pd.DataFrame, span: int) -> float:
    return float(ema(df["close"].astype(float), span).iloc[-1])

def current_close(df: pd.DataFrame) -> float:
    return float(df["close"].astype(float).iloc[-1])

# ---------- data fetch ----------
def make_data_client() -> Optional[StockHistoricalDataClient]:
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        log.warning("Alpaca creds missing → will use yfinance fallback for data.")
        return None
    try:
        return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    except Exception as e:
        log.error(f"Failed to init Alpaca data client: {e} → will use yfinance fallback.")
        return None

def fetch_daily_bars_alpaca(client: Optional[StockHistoricalDataClient], symbol: str, days: int) -> pd.DataFrame:
    if client is None:
        return pd.DataFrame()
    end = datetime.now(timezone.utc); start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="raw",
        feed="iex",
        limit=None,
    )
    try:
        bars = client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0, drop_level=True)
        df = df.reset_index().rename(columns={"timestamp": "date"})
        df = df[["date","open","high","low","close","volume"]]
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        log.error(f"Alpaca data error for {symbol}: {e}")
        return pd.DataFrame()

def fetch_daily_bars_yf(symbol: str, days: int) -> pd.DataFrame:
    if not HAVE_YF:
        log.error("yfinance not installed and Alpaca data failed — no data available.")
        return pd.DataFrame()
    try:
        period = "5y" if days >= 1200 else "2y" if days >= 500 else f"{max(days, 200)}d"
        hist = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        df = hist.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})[
            ["open","high","low","close","volume"]].copy()
        df.index.name = "date"
        return df
    except Exception as e:
        log.error(f"yfinance error for {symbol}: {e}")
        return pd.DataFrame()

def fetch_daily_bars(client: Optional[StockHistoricalDataClient], symbol: str, days: int = 420) -> pd.DataFrame:
    df = fetch_daily_bars_alpaca(client, symbol, days)
    if df.empty:
        log.warning(f"{symbol}: Alpaca returned 0 bars → trying yfinance fallback…")
        df = fetch_daily_bars_yf(symbol, days)
    return df

# ---------- trading ----------
@dataclass
class Position:
    symbol: str
    qty: float

class Broker:
    def __init__(self):
        self.enabled = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)
        self.client = None
        if self.enabled:
            try:
                self.client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
            except Exception as e:
                log.error(f"Trading client init failed: {e}")
                self.enabled = False
        else:
            log.warning("Trading disabled: missing ALPACA_API_KEY/ALPACA_SECRET_KEY")

    def has_position(self, symbol: str) -> bool:
        if not self.enabled:
            return False
        try:
            for p in self.client.get_all_positions():
                if p.symbol == symbol and float(p.qty) > 0:
                    return True
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
        return False

    def market_buy_usd(self, symbol: str, usd: float) -> Optional[str]:
        if DRY_RUN or not self.enabled:
            log.info(f"DRY_RUN or trading disabled → BUY {symbol} ${usd:.2f} (simulated)")
            return None
        req = MarketOrderRequest(symbol=symbol, notional=usd, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        order = self.client.submit_order(req)
        return order.id

    def market_sell_all(self, symbol: str) -> Optional[str]:
        if DRY_RUN or not self.enabled:
            log.info(f"DRY_RUN or trading disabled → SELL ALL {symbol} (simulated)")
            return None
        # Fetch position again to get qty
        try:
            for p in self.client.get_all_positions():
                if p.symbol == symbol:
                    qty = p.qty
                    if float(qty) <= 0:
                        return None
                    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    order = self.client.submit_order(req)
                    return order.id
        except Exception as e:
            log.error(f"Failed to sell {symbol}: {e}")
        return None

# ---------- scoring / signals ----------
def compute_features(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    close = df["close"].astype(float)
    ema12 = ema(close, 12); ema26 = ema(close, 26)
    r = rsi(close, 14)
    ema_gap = float(ema12.iloc[-1] / ema26.iloc[-1] - 1.0)
    r_now = float(r.iloc[-1])
    if len(close) >= 21 and close.iloc[-21] != 0:
        mom20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
    else:
        mom20 = 0.0
    return float(close.iloc[-1]), ema_gap, r_now, mom20

def liquidity_ok(df: pd.DataFrame, min_avg_dollar: float = MIN_ADV_USD) -> bool:
    c = df["close"].astype(float); v = df["volume"].astype(float)
    if len(c) < 20:
        return False
    adv = float((c[-20:] * v[-20:]).mean())
    if adv < min_avg_dollar:
        log.info(f"Liquidity fail: ADV=${adv:,.0f} < ${min_avg_dollar:,.0f}")
        return False
    return True

def trend_signal(df: pd.DataFrame) -> str:
    if df.empty or len(df) < MIN_BARS:
        return "HOLD"
    e12 = ema_now(df, 12); e26 = ema_now(df, 26); r_now = float(rsi(df["close"].astype(float), 14).iloc[-1])
    if e12 > e26 and r_now > 50:
        return "BUY"
    if e12 < e26 and r_now < 50:
        return "SELL"
    return "HOLD"

def relaxed_bullish(df: pd.DataFrame) -> bool:
    if df.empty or len(df) < MIN_BARS:
        return False
    return ema_now(df, 12) > ema_now(df, 26)

def score_symbol(df: pd.DataFrame) -> float:
    if df.empty or len(df) < MIN_BARS:
        return -1e9
    if not liquidity_ok(df):
        return -1e9
    _, ema_gap, r_now, mom20 = compute_features(df)
    rsi_tilt = max(0.0, (r_now - 50.0) / 50.0)  # 0..1 when above 50
    return float(0.5 * mom20 + 0.3 * ema_gap + 0.2 * rsi_tilt)

# ---------- guard sells ----------
def check_and_sell_guards(broker: "Broker",
                          data_client: Optional[StockHistoricalDataClient]) -> int:
    """Scan ALL open positions and sell if any guard is hit."""
    if not broker.enabled:
        return 0
    sold = 0
    try:
        positions = broker.client.get_all_positions()
    except Exception as e:
        log.error(f"Failed to get positions: {e}")
        return 0

    for p in positions:
        try:
            sym = p.symbol
            qty = float(p.qty)
            if qty <= 0:
                continue
            entry = float(p.avg_entry_price)

            df = fetch_daily_bars(data_client, sym, days=420)
            if df.empty or len(df) < MIN_BARS:
                continue
            price = current_close(df)

            reason = None
            if SELL_ON_TREND_BREAK and trend_signal(df) == "SELL":
                reason = "TREND_BREAK"
            if reason is None and STOP_LOSS_PCT > 0 and price <= entry * (1 - STOP_LOSS_PCT):
                reason = f"STOP_LOSS({STOP_LOSS_PCT:.1%})"
            if reason is None and TAKE_PROFIT_PCT > 0 and price >= entry * (1 + TAKE_PROFIT_PCT):
                reason = f"TAKE_PROFIT({TAKE_PROFIT_PCT:.1%})"

            if reason:
                log.info(f"Guard SELL {sym}: {reason} @ {price:.2f} vs entry {entry:.2f}")
                broker.market_sell_all(sym)
                sold += 1
        except Exception as e:
            log.error(f"Guard check failed for {p.symbol}: {e}")
    return sold

# ---------- engine ----------
def main() -> int:
    log.info("Starting Equities Engine (Auto-pick momentum/trend)")
    log.info(
        f"Config → DRY_RUN={DRY_RUN} PER_TRADE_USD={PER_TRADE_USD} DAILY_CAP_USD={DAILY_CAP_USD} "
        f"MARKET_ONLY={MARKET_ONLY} AUTO_PICK={AUTO_PICK} PICKS_PER_RUN={PICKS_PER_RUN} "
        f"AVOID_REBUY={AVOID_REBUY} MIN_ADV_USD={MIN_ADV_USD:,.0f} MIN_BUYS={MIN_BUYS} "
        f"STOP_LOSS_PCT={STOP_LOSS_PCT:.1%} TAKE_PROFIT_PCT={TAKE_PROFIT_PCT:.1%} "
        f"SELL_ON_TREND_BREAK={SELL_ON_TREND_BREAK}  "
        f"Alpaca: paper={ALPACA_PAPER} key_id_prefix={mask_prefix(ALPACA_API_KEY)}"
    )

    data_client = make_data_client()
    broker = Broker()

    # 1) Guard sells first
    guard_sells = check_and_sell_guards(broker, data_client)
    if guard_sells:
        log.info(f"Guard sells placed: {guard_sells}")

    # 2) Build working list (eligible + backups) then trade
    if AUTO_PICK:
        log.info(f"Scanning {len(SCAN_UNIVERSE)} symbols for candidates…")
        scored_all: List[Tuple[str, float, int]] = []
        eligible: List[Tuple[str, float, int]] = []
        for sym in SCAN_UNIVERSE:
            df = fetch_daily_bars(data_client, sym, days=420)
            n = len(df)
            log.info(f"Bars fetched for {sym}: {n} (need ≥{MIN_BARS})")
            if n < MIN_BARS:
                continue
            s = score_symbol(df)
            if s == -1e9:   # liquidity/history failed; already logged
                continue
            scored_all.append((sym, s, n))
            if trend_signal(df) == "BUY":
                eligible.append((sym, s, n))
            time.sleep(0.05)  # be gentle on rate limits

        eligible.sort(key=lambda x: x[1], reverse=True)
        backups = sorted(scored_all, key=lambda x: x[1], reverse=True)

        working: List[str] = [sym for sym, _, _ in eligible[:PICKS_PER_RUN]]
        if len(working) < PICKS_PER_RUN:
            for sym, _, _ in backups:
                if sym not in working:
                    working.append(sym)
                if len(working) >= PICKS_PER_RUN:
                    break

        top_line = ", ".join([f"{sym} (score={sc:.3f})" for sym, sc, _ in eligible[:PICKS_PER_RUN]]) or "None"
        log.info(f"Top picks (eligible): {top_line}")
    else:
        working = UNIVERSE
        backups = [(s, 0.0, 0) for s in working]  # trivial backups
        log.info(f"Using provided UNIVERSE: {working}")

    spend_left = DAILY_CAP_USD
    total_buys = 0
    total_sells = guard_sells
    bought: List[str] = []

    # Standard TREND buys
    for symbol in working:
        try:
            df = fetch_daily_bars(data_client, symbol, days=420)
            log.info(f"Bars fetched for {symbol}: {len(df)} (need ≥{MIN_BARS})")
            if df.empty or len(df) < MIN_BARS:
                log.warning(f"Skipping {symbol}: insufficient bars for indicators")
                continue

            signal = trend_signal(df)
            log.info(f"Signal for {symbol}: {signal}")

            if signal == "BUY" and spend_left >= PER_TRADE_USD:
                if AVOID_REBUY and broker.has_position(symbol):
                    log.info(f"Skip BUY {symbol}: position already exists (AVOID_REBUY=true)")
                elif MARKET_ONLY:
                    broker.market_buy_usd(symbol, PER_TRADE_USD)
                    spend_left -= PER_TRADE_USD
                    total_buys += 1
                    bought.append(symbol)
                else:
                    log.info("MARKET_ONLY=false not supported. Skipping.")
        except Exception as e:
            log.exception(f"Error processing {symbol}: {e}")

    # MIN_BUYS fallback (relaxed uptrend)
    if MIN_BUYS > 0 and total_buys < MIN_BUYS and spend_left >= PER_TRADE_USD:
        log.info(f"MIN_BUYS fallback engaged (need ≥{MIN_BUYS} buys).")
        for sym, _, _ in backups:
            if sym in bought:
                continue
            if AVOID_REBUY and broker.has_position(sym):
                continue
            df = fetch_daily_bars(data_client, sym, days=420)
            if len(df) < MIN_BARS:
                continue
            if not relaxed_bullish(df):
                continue
            if spend_left < PER_TRADE_USD:
                break
            broker.market_buy_usd(sym, PER_TRADE_USD)
            spend_left -= PER_TRADE_USD
            total_buys += 1
            bought.append(sym)
            log.info(f"MIN_BUYS fallback: forced BUY {sym} (relaxed trend).")
            if total_buys >= MIN_BUYS:
                break

    log.info(f"Done. Buys placed: {total_buys}, Sells placed: {total_sells}, "
             f"Daily spend remaining: ${spend_left:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
