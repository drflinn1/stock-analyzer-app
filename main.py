import os, json, time, random, pandas as pd
from datetime import datetime
import ccxt
import yfinance as yf

# -----------------------------------------------------
#  CONFIG
# -----------------------------------------------------
EXCHANGE = os.getenv("EXCHANGE", "kraken")
DRY_RUN = os.getenv("DRY_RUN", "ON").upper() == "ON"
STATE_DIR = os.getenv("STATE_DIR", ".state")
GUARD_FILE = os.path.join(STATE_DIR, "guard.yaml")
POSITIONS_JSON = os.path.join(STATE_DIR, "positions.json")
KPI_CSV = os.path.join(STATE_DIR, "kpi_history.csv")

# fallback defaults
PARAMS = {
    "min_buy_usd": 10,
    "min_sell_usd": 10,
    "reserve_cash_pct": 5,
    "max_positions": 3,
    "max_buys_per_run": 2,
    "top_k": 25
}

# -----------------------------------------------------
#  UTILITIES
# -----------------------------------------------------
def log(msg): print(f"[{datetime.utcnow().isoformat()}] {msg}")

def load_positions():
    try:
        if os.path.exists(POSITIONS_JSON):
            return json.load(open(POSITIONS_JSON))
    except Exception as e:
        log(f"warn: could not load positions.json: {e}")
    return {}

def save_positions(data):
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump(data, open(POSITIONS_JSON, "w"), indent=2)

def record_kpi(bal_usd=0, pnl_day_usd=0, buys=0, sells=0, open_positions=0):
    os.makedirs(STATE_DIR, exist_ok=True)
    df = pd.DataFrame([[datetime.utcnow().isoformat(), "USD", bal_usd, pnl_day_usd, buys, sells, open_positions]],
                      columns=["timestamp","base","bal_usd","pnl_day_usd","buys","sells","open_positions"])
    header = not os.path.exists(KPI_CSV)
    df.to_csv(KPI_CSV, mode="a", header=header, index=False)

# -----------------------------------------------------
#  CORE BOT LOGIC (SIMPLIFIED SEPTEMBER)
# -----------------------------------------------------
def get_exchange():
    try:
        return getattr(ccxt, EXCHANGE)()
    except Exception as e:
        log(f"error loading exchange: {e}")
        return None

def fetch_top_market_symbols(limit=25):
    tickers = yf.download("BTC-USD ETH-USD SOL-USD DOGE-USD XRP-USD ADA-USD LINK-USD DOT-USD LTC-USD AVAX-USD",
                          period="1d", interval="1h", progress=False)
    symbols = [s.replace("-", "/").replace("USD", "USD") for s in ["BTC/USD","ETH/USD","SOL/USD","DOGE/USD","XRP/USD",
                "ADA/USD","LINK/USD","DOT/USD","LTC/USD","AVAX/USD"]][:limit]
    return symbols

def simulate_price(symbol):
    # quick simulation for dry-run; replace with ccxt fetch_ticker for live
    return random.uniform(0.95, 1.05)

def run_cycle():
    log(f"Starting CryptoBot cycle (DRY_RUN={DRY_RUN})")
    exchange = get_exchange()
    if not exchange:
        log("Exchange unavailable, aborting.")
        return

    symbols = fetch_top_market_symbols(PARAMS["top_k"])
    positions = load_positions()
    buys = sells = 0

    for sym in symbols:
        price = simulate_price(sym)
        if sym not in positions and random.random() < 0.2:
            if DRY_RUN:
                log(f"[BUY] {sym} at {price:.4f} (simulated)")
            positions[sym] = {"entry": price, "time": datetime.utcnow().isoformat()}
            buys += 1
        elif sym in positions and random.random() < 0.15:
            entry = positions[sym]["entry"]
            if price > entry * 1.02:
                if DRY_RUN:
                    log(f"[SELL] {sym} at {price:.4f} (gain from {entry:.4f})")
                del positions[sym]
                sells += 1

    save_positions(positions)
    bal_usd = 1000 + random.uniform(-10, 10)
    pnl_day_usd = random.uniform(-5, 10)
    record_kpi(bal_usd, pnl_day_usd, buys, sells, len(positions))
    log(f"Cycle complete — buys={buys}, sells={sells}, open={len(positions)}")
    log("------------------------------------------------------")

if __name__ == "__main__":
    run_cycle()
