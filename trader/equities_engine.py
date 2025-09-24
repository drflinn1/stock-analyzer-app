# trader/equities_engine.py
# Alpaca Paper equities engine: auto-universe (SP100 or user list), EMA/RSI/volume
# filters, per-trade + daily caps, and optional bracket TP/SL orders.

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

try:
    # alpaca-py
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    from alpaca.trading.requests import (
        MarketOrderRequest,
        TakeProfitRequest,
        StopLossRequest,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Failed to import alpaca-py — ensure pip install alpaca-py :: {e}")

NY = pytz.timezone("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
DAILY_FILE = STATE_DIR / "equities_daily.json"
USER_UNIVERSE_FILE = ROOT / "config" / "equities_universe.txt"

# --- Tunables from env (workflow inputs) ---
getenv = os.getenv
DRY_RUN = getenv("DRY_RUN", "false").lower() == "true"
PER_TRADE_USD = float(getenv("PER_TRADE_USD", "1000"))
DAILY_CAP_USD = float(getenv("DAILY_CAP_USD", "5000"))
MAX_POSITIONS = int(getenv("MAX_POSITIONS", "10"))
UNIVERSE_MODE = getenv("UNIVERSE_MODE", "SP100").upper()
UNIVERSE_SIZE = int(getenv("UNIVERSE_SIZE", "20"))
EMA_FAST = int(getenv("EMA_FAST", "20"))
EMA_SLOW = int(getenv("EMA_SLOW", "50"))
RSI_MIN = float(getenv("RSI_MIN", "50"))
RSI_MAX = float(getenv("RSI_MAX", "80"))
MIN_AVG_DOLLAR_VOL = float(getenv("MIN_AVG_DOLLAR_VOL", "20000000"))  # $20M
TP_PCT = float(getenv("TP_PCT", "0.03"))
SL_PCT = float(getenv("SL_PCT", "0.02"))
USE_BRACKETS = getenv("USE_BRACKETS", "true").lower() == "true"

ALPACA_API_KEY = getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = getenv("ALPACA_API_SECRET", "")
ALPACA_BASE_URL = getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# --- S&P 100 static list (liquid, keeps fetches sane). You can swap to SP500 later. ---
SP100 = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BIIB","BK","BKNG","BLK","BMY","BRK-B",
    "C","CAT","CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO",
    "CVS","CVX","DE","DHR","DIS","DUK","EMR","EXC","F","FDX","GD",
    "GE","GILD","GM","GOOGL","GOOG","GS","HD","HON","IBM","INTC",
    "JNJ","JPM","KHC","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ",
    "MDT","MET","META","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE",
    "NVDA","ORCL","PEP","PFE","PG","PM","PYPL","QCOM","RTX","SBUX",
    "SO","SPG","T","TGT","TMO","TMUS","TSLA","TXN","UNH","UNP","UPS",
    "USB","V","VZ","WBA","WFC","WMT","XOM"
]


# --- Helpers ---

def load_daily_state() -> Dict:
    today = datetime.now(NY).strftime("%Y-%m-%d")
    if DAILY_FILE.exists():
        try:
            data = json.loads(DAILY_FILE.read_text())
        except Exception:
            data = {"date": today, "spent_usd": 0.0}
    else:
        data = {"date": today, "spent_usd": 0.0}

    if data.get("date") != today:
        data = {"date": today, "spent_usd": 0.0}
    return data


def save_daily_state(data: Dict):
    DAILY_FILE.write_text(json.dumps(data, indent=2))


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=window).mean()
    roll_down = down.rolling(window=window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def is_market_open(client: TradingClient) -> bool:
    try:
        clk = client.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception as e:
        print(f"[WARN] Failed to fetch market clock: {e}. Assuming market CLOSED.")
        return False


def get_positions(client: TradingClient) -> Dict[str, float]:
    pos = {}
    try:
        for p in client.get_all_positions():
            try:
                qty = float(p.qty)
            except Exception:
                qty = float(p.qty_available)
            pos[p.symbol.upper()] = qty
    except Exception as e:
        print(f"[WARN] Could not load positions: {e}")
    return pos


def load_universe(mode: str) -> List[str]:
    if mode == "USER_LIST" and USER_UNIVERSE_FILE.exists():
        syms = [s.strip().upper() for s in USER_UNIVERSE_FILE.read_text().splitlines() if s.strip()]
        if syms:
            return syms
    # Default
    return SP100


def fetch_history(symbols: List[str]) -> pd.DataFrame:
    # Daily bars ~3 months for indicators
    if not symbols:
        return pd.DataFrame()
    print(f"[INFO] Downloading daily history for {len(symbols)} symbols...")
    df = yf.download(
        tickers=symbols,
        period="3mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    return df


def latest_close(df_sym: pd.DataFrame) -> float:
    try:
        return float(df_sym["Close"].dropna().iloc[-1])
    except Exception:
        return np.nan


def avg_dollar_vol(df_sym: pd.DataFrame, window: int = 20) -> float:
    try:
        close = df_sym["Close"].astype(float)
        vol = df_sym["Volume"].astype(float)
        adv = (close * vol).rolling(window=window).mean().iloc[-1]
        return float(adv)
    except Exception:
        return 0.0


def indicators_pass(df_sym: pd.DataFrame) -> Tuple[bool, Dict]:
    try:
        close = df_sym["Close"].astype(float)
        if len(close) < max(EMA_FAST, EMA_SLOW) + 20:
            return False, {"reason": "insufficient bars"}
        ema_f = ema(close, EMA_FAST)
        ema_s = ema(close, EMA_SLOW)
        r = rsi(close, 14)
        adv = avg_dollar_vol(df_sym, 20)
        ok = (
            ema_f.iloc[-1] > ema_s.iloc[-1]
            and RSI_MIN <= r.iloc[-1] <= RSI_MAX
            and adv >= MIN_AVG_DOLLAR_VOL
        )
        return ok, {
            "ema_fast": round(float(ema_f.iloc[-1]), 4),
            "ema_slow": round(float(ema_s.iloc[-1]), 4),
            "rsi": round(float(r.iloc[-1]), 2),
            "avg_dollar_vol": round(float(adv), 2),
            "price": round(float(close.iloc[-1]), 4),
        }
    except Exception as e:
        return False, {"reason": f"indicator error: {e}"}


def submit_bracket(client: TradingClient, symbol: str, notional: float, ref_price: float):
    tp_price = round(ref_price * (1 + TP_PCT), 2)
    sl_price = round(ref_price * (1 - SL_PCT), 2)
    order = MarketOrderRequest(
        symbol=symbol,
        notional=notional,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=sl_price),
    )
    return client.submit_order(order_data=order)


def submit_simple_buy(client: TradingClient, symbol: str, notional: float):
    order = MarketOrderRequest(
        symbol=symbol,
        notional=notional,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order_data=order)


def main():
    print("\n===== EQUITIES ENGINE (Alpaca Paper) =====")
    print(f"DRY_RUN={DRY_RUN}  PER_TRADE_USD={PER_TRADE_USD}  DAILY_CAP_USD={DAILY_CAP_USD}  MAX_POSITIONS={MAX_POSITIONS}")
    print(f"UNIVERSE_MODE={UNIVERSE_MODE}  UNIVERSE_SIZE={UNIVERSE_SIZE}")
    print(f"EMA_FAST={EMA_FAST}  EMA_SLOW={EMA_SLOW}  RSI in [{RSI_MIN}, {RSI_MAX}]  MIN_AVG_$VOL={MIN_AVG_DOLLAR_VOL}")
    print(f"TP_PCT={TP_PCT}  SL_PCT={SL_PCT}  USE_BRACKETS={USE_BRACKETS}")

    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise SystemExit("Missing ALPACA_API_KEY/ALPACA_API_SECRET secrets.")

    client = TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
        paper=True,
    )

    if not is_market_open(client):
        print("[HALT] Market is closed. Exiting cleanly.")
        return

    # Load day state
    day = load_daily_state()
    spent_left = max(0.0, DAILY_CAP_USD - float(day.get("spent_usd", 0.0)))
    print(f"[STATE] Spent today: ${day.get('spent_usd', 0.0):.2f}  |  Remaining cap: ${spent_left:.2f}")

    # Positions / exposure
    positions = get_positions(client)
    curr_positions = {s for s, q in positions.items() if q > 0}
    print(f"[INFO] Open long positions: {len(curr_positions)} -> {sorted(curr_positions)[:30]}")

    # Universe
    universe = load_universe(UNIVERSE_MODE)
    print(f"[INFO] Universe base size: {len(universe)} ({UNIVERSE_MODE})")

    # Data
    df_all = fetch_history(universe)

    # Build candidates table
    rows = []
    for sym in universe:
        try:
            df_sym = df_all[sym]
        except Exception:
            continue
        ok, meta = indicators_pass(df_sym)
        meta.update({"symbol": sym, "ok": ok})
        rows.append(meta)

    if not rows:
        print("[WARN] No data rows built.")
        return

    table = pd.DataFrame(rows).dropna(subset=["price", "avg_dollar_vol"])  # keep usable
    table.sort_values("avg_dollar_vol", ascending=False, inplace=True)

    print("\n[TOP LIQUID] head after filters check (not yet filtered for ok):")
    print(table.head(10).to_string(index=False))

    # Final filter and cut
    cands = table[table["ok"]].copy()
    if cands.empty:
        print("[RESULT] No symbols pass filters today.")
        return
    cands = cands.head(UNIVERSE_SIZE)

    # Decide buys
    buys_done = 0
    total_spent = 0.0

    # Enforce max positions
    slots_left = max(0, MAX_POSITIONS - len(curr_positions))
    if slots_left <= 0:
        print("[HALT] No position slots left (at MAX_POSITIONS).")
        return

    print(f"\n[SELECTED] {len(cands)} candidates after filters; position slots left: {slots_left}")

    for _, row in cands.iterrows():
        sym = row["symbol"]
        price = float(row["price"])

        if sym in curr_positions:
            print(f"[SKIP] {sym}: already long.")
            continue
        if spent_left < PER_TRADE_USD:
            print(f"[STOP] Daily cap exhausted (remaining ${spent_left:.2f}).")
            break
        if slots_left <= 0:
            print("[STOP] No position slots left.")
            break

        print(
            f"[BUY] {sym} @ ~{price:.2f} | ema_f={row['ema_fast']}, ema_s={row['ema_slow']}, rsi={row['rsi']} | notional=${PER_TRADE_USD:.2f}"
        )

        if DRY_RUN:
            print("      → DRY_RUN=True — would submit order (no order sent).")
        else:
            try:
                if USE_BRACKETS:
                    resp = submit_bracket(client, sym, PER_TRADE_USD, price)
                    print(f"      → BRACKET submitted: id={getattr(resp, 'id', '(n/a)')} TP={TP_PCT*100:.1f}% SL={SL_PCT*100:.1f}%")
                else:
                    resp = submit_simple_buy(client, sym, PER_TRADE_USD)
                    print(f"      → MARKET BUY submitted: id={getattr(resp, 'id', '(n/a)')}")
            except Exception as e:
                print(f"      ! Order failed for {sym}: {e}")
                continue

        # Update counters/state
        buys_done += 1
        total_spent += PER_TRADE_USD
        spent_left -= PER_TRADE_USD
        slots_left -= 1

    # Persist daily spend across runs (committed by workflow step)
    day["spent_usd"] = float(day.get("spent_usd", 0.0)) + total_spent
    save_daily_state(day)

    print("\n===== SUMMARY =====")
    print(f"Buys placed: {buys_done}  |  New spend: ${total_spent:.2f}  |  Spent today now: ${day['spent_usd']:.2f}")
    print("(Bracket TP/SL will manage exits automatically for those orders.)" if USE_BRACKETS else "(No TP/SL brackets attached.)")


if __name__ == "__main__":
    main()
