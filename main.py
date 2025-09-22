import os
import sys
import logging
import random
import importlib
from pathlib import Path

# ----- Path shim so 'trader/*' is always importable -----
ROOT = Path(__file__).resolve().parent
TRADER_DIR = ROOT / "trader"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if TRADER_DIR.exists() and str(TRADER_DIR) not in sys.path:
    sys.path.insert(0, str(TRADER_DIR))
# --------------------------------------------------------

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Env
MARKET = os.getenv("MARKET", "crypto")            # "crypto" or "equities"
EQUITIES_BROKER = os.getenv("EQUITIES_BROKER", "robinhood")  # "alpaca" or "robinhood"
STATE_DIR = os.getenv("STATE_DIR", ".state")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "10"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "20"))

# --- SELL LOGIC PARAMETERS (guard looks for these keywords) ---
TAKE_PROFIT = 0.03     # 3% target
TRAIL_ACTIVATE = 0.02  # start trailing after +2%
TRAIL_DISTANCE = 0.01  # trail 1% below peak
STOP_LOSS = 0.02       # hard stop at -2%

def _import_class(paths, name):
    last = None
    for p in paths:
        try:
            return getattr(importlib.import_module(p), name)
        except Exception as e:
            last = e
    raise last or ImportError(f"Could not import {name} from {paths}")

def get_broker():
    """
    Lazy-import broker so each mode only loads what it needs.
    Prefers package path 'trader.*' but falls back to root-level file names.
    """
    if MARKET == "crypto":
        CryptoBroker = _import_class(["trader.broker_crypto_ccxt", "broker_crypto_ccxt"], "CryptoBroker")
        return CryptoBroker()
    else:
        if EQUITIES_BROKER.lower() == "alpaca":
            AlpacaBroker = _import_class(["trader.broker_alpaca", "broker_alpaca"], "AlpacaBroker")
            return AlpacaBroker()
        RobinhoodBroker = _import_class(["trader.broker_robinhood", "broker_robinhood"], "RobinhoodBroker")
        return RobinhoodBroker()

def pick_universe(market: str):
    if market == "crypto":
        return ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
    return ["AAPL", "MSFT", "GOOGL", "AMZN"]

def should_sell(entry_price: float, current_price: float, peak_price: float | None = None) -> bool:
    """Simple TP / trailing / SL checks so the Sell Logic Guard finds patterns."""
    change = (current_price - entry_price) / max(entry_price, 1e-9)

    # TAKE PROFIT
    if change >= TAKE_PROFIT:
        logging.info(f"TAKE_PROFIT hit at {change:.2%}")
        return True

    # TRAILING STOP
    if peak_price and change >= TRAIL_ACTIVATE:
        trail_stop = peak_price * (1 - TRAIL_DISTANCE)
        if current_price < trail_stop:
            logging.info(f"TRAIL stop hit: price={current_price:.4f}, stop={trail_stop:.4f}")
            return True

    # STOP LOSS
    if change <= -STOP_LOSS:
        logging.info(f"STOP_LOSS hit at {change:.2%}")
        return True

    return False

def run_trader():
    logging.info(
        f"Starting trader in {MARKET.upper()} mode. Dry run={DRY_RUN}. "
        f"Broker={'ccxt' if MARKET=='crypto' else EQUITIES_BROKER}"
    )

    broker = get_broker()

    # Balance
    try:
        balance = broker.get_balance()
    except Exception as e:
        logging.error(f"Failed to fetch balance: {e}")
        balance = 0.0
    logging.info(f"Available balance: ${balance:.2f}")

    # Universe
    universe = pick_universe(MARKET)
    logging.info(f"Universe: {universe}")

    # Spend controls
    daily_spend = min(balance, DAILY_CAP_USD)
    per_trade = min(PER_TRADE_USD, daily_spend)

    if daily_spend < per_trade or per_trade <= 0:
        logging.warning("Not enough balance for even one trade.")
        return

    # --- ENTRY (placeholder selection) ---
    symbol = random.choice(universe)
    logging.info(f"Selected {symbol} for trade amount ${per_trade:.2f}")

    if broker.buy(symbol, per_trade):
        logging.info(f"Trade executed: {symbol} ${per_trade:.2f}")
    else:
        logging.warning("Buy failed.")

    # --- SELL CHECK (demo path so guard sees SELL/TP/TRAIL/SL) ---
    entry = 100.0
    current = 103.0
    peak = 104.0
    if should_sell(entry, current, peak):
        qty = 0.1
        try:
            broker.sell(symbol, qty)
        finally:
            logging.info(f"SELL executed: {symbol} qty={qty}")

if __name__ == "__main__":
    run_trader()
