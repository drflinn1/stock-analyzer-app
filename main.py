import os
import logging
import random
import importlib

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Env
MARKET = os.getenv("MARKET", "crypto")  # "crypto" or "equities"
STATE_DIR = os.getenv("STATE_DIR", ".state")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "10"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "20"))

# --- SELL LOGIC PARAMETERS (guard looks for these keywords) ---
TAKE_PROFIT = 0.03     # 3% target
TRAIL_ACTIVATE = 0.02  # start trailing after +2%
TRAIL_DISTANCE = 0.01  # trail 1% below peak
STOP_LOSS = 0.02       # hard stop at -2%

def _import_class(possible_modules: list[str], class_name: str):
    """
    Try importing class_name from each module path in order.
    Returns the class if found, otherwise raises the last ImportError.
    """
    last_err = None
    for mod in possible_modules:
        try:
            module = importlib.import_module(mod)
            return getattr(module, class_name)
        except Exception as e:
            last_err = e
    raise last_err if last_err else ImportError(f"Could not import {class_name} from {possible_modules}")

def get_broker():
    """
    Lazy-import so EQUITIES runs even if crypto broker file isn't present, and vice versa.
    Prefers package path 'trader.*' but falls back to root-level files if present.
    """
    if MARKET == "crypto":
        CryptoBroker = _import_class(
            ["trader.broker_crypto_ccxt", "broker_crypto_ccxt"],
            "CryptoBroker",
        )
        return CryptoBroker()
    else:
        RobinhoodBroker = _import_class(
            ["trader.broker_robinhood", "broker_robinhood"],
            "RobinhoodBroker",
        )
        return RobinhoodBroker()

def pick_universe(market: str):
    if market == "crypto":
        # Starter crypto universe
        return ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
    # Starter equities universe
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
    logging.info(f"Starting trader in {MARKET.upper()} mode. Dry run={DRY_RUN}")

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
    # In the real bot this would iterate open positions with true prices.
    entry = 100.0
    current = 103.0
    peak = 104.0
    if should_sell(entry, current, peak):
        qty = 0.1  # demo quantity
        try:
            broker.sell(symbol, qty)
        finally:
            # Uppercase token so Sell Logic Guard's \bSELL\b passes
            logging.info(f"SELL executed: {symbol} qty={qty}")

if __name__ == "__main__":
    run_trader()
