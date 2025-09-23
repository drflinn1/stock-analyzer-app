"""
Equities Engine — Auto-pick momentum/trend names (Alpaca IEX, yfinance fallback)

Behavior (similar to your Crypto Live bot):
- Scan a candidate universe, rank by momentum/trend score, auto-pick the top names.
- BUY when TREND is bullish (EMA12 > EMA26 and RSI>50), SELL when TREND turns bearish.
- Respect per-trade and daily USD caps; skip rebuy if already holding.

Key env knobs (set via workflow inputs):
  DRY_RUN=true|false
  PER_TRADE_USD=3
  DAILY_CAP_USD=12
  MARKET_ONLY=true
  # If AUTO_PICK=false, we just use UNIVERSE as given (comma list)
  AUTO_PICK=true|false
  PICKS_PER_RUN=4
  UNIVERSE=AAPL,MSFT                      # used when AUTO_PICK=false
  SCAN_UNIVERSE=...                       # optional; default list is provided below
  AVOID_REBUY=true|false                  # skip BUY if already holding
  LOG_LEVEL=INFO

Secrets:
  ALPACA_API_KEY / ALPACA_SECRET_KEY
  ALPACA_PAPER=true|false  (defaults true)
"""
from __future__ import annotations

import os, sys, time, logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

import pandas as pd

# ---------------- optional yfinance fallback ----------------
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ---------------- Alpaca SDK ----------------
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception as e:
    raise RuntimeError("alpaca-py is required. Run: pip install alpaca-py") from e

# ---------------- logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("equities_engine")

# ---------------- config ----------------
def getenv_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","y"}

DRY_RUN       = getenv_bool("DRY_RUN", True)
MARKET_ONLY   = getenv_bool("MARKET_ONLY", True)
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "3"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "12"))

AUTO_PICK     = getenv_bool("AUTO_PICK", True)
PICKS_PER_RUN = int(float(os.getenv("PICKS_PER_RUN", "4")))  # allow "4.0"

# Used when AUTO_PICK=false
UNIVERSE = [s.strip().upper() for s in os.getenv("UNIVERSE", "AAPL,MSFT").split(",") if s.strip()]

# Used when AUTO_PICK=true; if not set, use a sane high-liquidity default
DEFAULT_SCAN = [
    # Big tech + mega caps + broad ETFs (liquid)
    "SPY","QQQ","DIA","IWM","AAPL","MSFT","NVDA","AMD","TSLA","META","GOOGL","AMZN","AVGO","NFLX",
    "COST","JPM","BRK.B","UNH","XOM","CVX","PEP","KO","V","MA","JNJ","HD","WMT","ORCL","CRM","INTC"
]
SCAN_UNIVERSE = [s.strip().upper() for s in os.getenv("SCAN_UNIVERSE", ",".join(DEFAULT_SCAN)).split(",") if s.strip()]

AVOID_REBUY   = getenv_bool("AVOID_REBUY", True)

ALPACA_API_KEY   = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY= os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER     = getenv_bool("ALPACA_PAPER", True)

MIN_BARS = 120  # for EMA/RSI

def mask_prefix(s: str, n: int = 4) -> str:
    return "NONE" if not s else f"{s[:n]}***"

# ---------------- indicators ----------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

# ---------------- data fetch ----------------
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
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
        start=start, end=end, adjustment="raw",
        feed="iex", limit=None  # force free IEX feed
    )
    try:
        bars = client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0, drop_level=True)
        df = df.reset_index().rename(columns={"timestamp":"date"})
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
        # ensure generous history
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

# ---------------- trading ----------------
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
                if p.symbol == symbol and float(p.qty) > 0: return True
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
        return False

    def market_buy_usd(self, symbol: str, usd: float) -> Optional[str]:
        if DRY_RUN or not self.enabled:
            log.info(f"DRY_RUN or trading disabled → BUY {symbol} ${usd:.2f} (simulated)")
            return None
        req = MarketOrderRequest(symbol=symbol, notional=usd, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        order = self.client.submit_order(req); return order.id

    def market_sell_all(self, symbol: str) -> Optional[str]:
        if DRY_RUN or not self.enabled:
            log.info(f"DRY_RUN or trading disabled → SELL ALL {symbol} (simulated)")
            return None
        pos = None
        try:
            for p in self.client.get_all_positions():
                if p.symbol == symbol: pos = p; break
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
        if not pos:
            log.info(f"No position in {symbol} to sell."); return None
        req = MarketOrderRequest(symbol=symbol, qty=pos.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        order = self.client.submit_order(req); return order.id

# ---------------- scoring / signals ----------------
def compute_features(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Return (close, ema_gap, rsi_now, mom20). ema_gap is EMA12/EMA26 - 1; mom20 is 20d return."""
    close = df["close"].astype(float)
    ema12 = ema(close, 12); ema26 = ema(close, 26)
    r = rsi(close, 14)
    ema_gap = float(ema12.iloc[-1]/ema26.iloc[-1] - 1.0)
    r_now = float(r.iloc[-1])
    if len(close) >= 21 and close.iloc[-21] != 0:
        mom20 = float(close.iloc[-1]/close.iloc[-21] - 1.0)
    else:
        mom20 = 0.0
    return float(close.iloc[-1]), ema_gap, r_now, mom20

def liquidity_ok(df: pd.DataFrame, min_avg_dollar: float = 5e7) -> bool:
    # average dollar volume last 20 days
    c = df["close"].astype(float); v = df["volume"].astype(float)
    if len(c) < 20: return False
    adv = float((c[-20:] * v[-20:]).mean())
    return adv >= min_avg_dollar

def trend_signal(df: pd.DataFrame) -> str:
    if df.empty or len(df) < MIN_BARS: return "HOLD"
    close = df["close"].astype(float)
    ema12_now = float(ema(close,12).iloc[-1]); ema26_now = float(ema(close,26).iloc[-1])
    r_now = float(rsi(close,14).iloc[-1])
    bullish = (ema12_now > ema26_now) and (r_now > 50)
    bearish = (ema12_now < ema26_now) and (r_now < 50)
    if bullish: return "BUY"
    if bearish: return "SELL"
    return "HOLD"

def score_symbol(df: pd.DataFrame) -> float:
    """Weighted score: 50% 20d momentum, 30% EMA gap, 20% RSI tilt above 50."""
    if df.empty or len(df) < MIN_BARS: return -1e9
    if not liquidity_ok(df): return -1e9
    price, ema_gap, r_now, mom20 = compute_features(df)
    rsi_tilt = max(0.0, (r_now - 50.0)/50.0)  # 0..1 when above 50
    score = 0.5*mom20 + 0.3*ema_gap + 0.2*rsi_tilt
    return float(score)

# ---------------- engine ----------------
def main() -> int:
    log.info("Starting Equities Engine (Auto-pick momentum/trend)")
    log.info(f"Config → DRY_RUN={DRY_RUN} PER_TRADE_USD={PER_TRADE_USD} DAILY_CAP_USD={DAILY_CAP_USD} "
             f"MARKET_ONLY={MARKET_ONLY} AUTO_PICK={AUTO_PICK} PICKS_PER_RUN={PICKS_PER_RUN} AVOID_REBUY={AVOID_REBUY}")
    log.info(f"Alpaca env → paper={ALPACA_PAPER} key_id_prefix={mask_prefix(ALPACA_API_KEY)}")

    data_client = make_data_client()
    broker = Broker()

    # Build working universe
    working: List[str] = []
    if AUTO_PICK:
        log.info(f"Scanning {len(SCAN_UNIVERSE)} symbols for candidates…")
        scored: List[Tuple[str,float,int]] = []  # (symbol, score, bars)
        for sym in SCAN_UNIVERSE:
            df = fetch_daily_bars(data_client, sym, days=420)
            n = len(df)
            log.info(f"Bars fetched for {sym}: {n} (need ≥{MIN_BARS})")
            if n < MIN_BARS: continue
            s = score_symbol(df)
            if s == -1e9:
                log.info(f"Skip {sym}: liquidity or history insufficient.")
                continue
            # Require bullish trend to be eligible
            if trend_signal(df) != "BUY":
                continue
            scored.append((sym, s, n))
            # small pause to be gentle on rate limits
            time.sleep(0.05)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:max(1, PICKS_PER_RUN)]
        working = [t[0] for t in top]
        log.info("Top picks: " + ", ".join([f"{s} (score={sc:.3f})" for s,sc,_ in top]) if top else "None")
        if not working:
            log.info("No eligible picks; exiting without trades.")
            return 0
    else:
        working = UNIVERSE
        log.info(f"Using provided UNIVERSE: {working}")

    spend_left = DAILY_CAP_USD
    total_buys = 0; total_sells = 0

    for symbol in working:
        try:
            df = fetch_daily_bars(data_client, symbol, days=420)
            log.info(f"Bars fetched for {symbol}: {len(df)} (need ≥{MIN_BARS})")
            if df.empty or len(df) < MIN_BARS:
                log.warning(f"Skipping {symbol}: insufficient bars for indicators")
                continue

            signal = trend_signal(df)  # TREND logic
            log.info(f"Signal for {symbol}: {signal}")

            if signal == "BUY" and spend_left >= PER_TRADE_USD:
                if AVOID_REBUY and broker.has_position(symbol):
                    log.info(f"Skip BUY {symbol}: position already exists (AVOID_REBUY=true)")
                elif MARKET_ONLY:
                    broker.market_buy_usd(symbol, PER_TRADE_USD)
                    spend_left -= PER_TRADE_USD; total_buys += 1
                else:
                    log.info("MARKET_ONLY=false not supported in this engine. Skipping.")

            elif signal == "SELL":
                if MARKET_ONLY:
                    broker.market_sell_all(symbol); total_sells += 1
                else:
                    log.info("MARKET_ONLY=false not supported in this engine. Skipping.")
        except Exception as e:
            log.exception(f"Error processing {symbol}: {e}")

    log.info(f"Done. Buys placed: {total_buys}, Sells placed: {total_sells}, Daily spend remaining: ${spend_left:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
