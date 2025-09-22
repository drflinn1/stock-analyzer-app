import os
import sys
import logging
import random
import importlib
from pathlib import Path
from typing import Optional, Any

# ===== PATH SHIM: make 'trader/*' importable in GitHub Actions =====
ROOT = Path(__file__).resolve().parent
TRADER_DIR = ROOT / "trader"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if TRADER_DIR.exists() and str(TRADER_DIR) not in sys.path:
    sys.path.insert(0, str(TRADER_DIR))
# ===================================================================

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Env
MARKET = os.getenv("MARKET", "crypto")                      # "crypto" or "equities"
EQUITIES_BROKER = os.getenv("EQUITIES_BROKER", "robinhood") # "alpaca" or "robinhood"
STATE_DIR = os.getenv("STATE_DIR", ".state")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "10"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "20"))
USD_BAL_OVERRIDE = os.getenv("USD_BAL_OVERRIDE")            # <-- quick override for crypto balance

# --- SELL LOGIC PARAMETERS (Sell Guard looks for these tokens) ---
TAKE_PROFIT = 0.03     # 3% target
TRAIL_ACTIVATE = 0.02  # start trailing after +2%
TRAIL_DISTANCE = 0.01  # trail 1% below peak
STOP_LOSS = 0.02       # hard stop at -2%

# --- Helpers -------------------------------------------------------

def _import_class(paths: list[str], names: list[str]):
    """Try importing any class name from any module path."""
    last = None
    for p in paths:
        try:
            mod = importlib.import_module(p)
            for n in names:
                if hasattr(mod, n):
                    return getattr(mod, n)
        except Exception as e:
            last = e
    raise last or ImportError(f"Could not import any of {names} from {paths}")

def _extract_usd_from_balance(bal: Any) -> float:
    """Parse various balance shapes (float, ccxt dicts, objects)."""
    if isinstance(bal, (int, float)):
        return float(bal)
    if isinstance(bal, dict):
        for k in ("USD", "usd", "ZUSD"):
            v = bal.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                for sub in ("free", "total", "available"):
                    if isinstance(v.get(sub), (int, float)):
                        return float(v[sub])
        free = bal.get("free")
        if isinstance(free, dict):
            for k in ("USD", "usd", "ZUSD", "USDT", "usdt", "USDC", "usdc"):
                if isinstance(free.get(k), (int, float)):
                    return float(free[k])
    for attr in ("cash", "usd", "usd_available", "buying_power"):
        if hasattr(bal, attr):
            try:
                return float(getattr(bal, attr))
            except Exception:
                pass
    return 0.0

def safe_get_balance(broker: Any) -> float:
    """Try many common methods to get USD balance from a broker."""
    # 0) Override from env (quick escape hatch)
    if USD_BAL_OVERRIDE:
        try:
            usd = float(USD_BAL_OVERRIDE)
            logging.info(f"Balance via USD_BAL_OVERRIDE: ${usd:.2f}")
            return usd
        except Exception:
            pass

    # 1) direct methods
    for name in ("get_balance", "get_usd_balance", "balance", "get_cash",
                 "get_free_balance", "free_balance", "get_cash_balance",
                 "usd_balance"):
        if hasattr(broker, name):
            try:
                val = getattr(broker, name)()
                usd = _extract_usd_from_balance(val)
                if usd > 0:
                    logging.info(f"Balance via {name}: ${usd:.2f}")
                return usd
            except Exception as e:
                logging.error(f"{name}() failed: {e}")

    # 2) ccxt-style
    for name in ("fetch_balance", "get_balance_ccxt"):
        if hasattr(broker, name):
            try:
                bal = getattr(broker, name)()
                usd = _extract_usd_from_balance(bal)
                if usd > 0:
                    logging.info(f"Balance via {name}: ${usd:.2f}")
                return usd
            except Exception as e:
                logging.error(f"{name}() failed: {e}")

    # 3) attributes
    for attr in ("cash", "usd", "usd_available", "buying_power"):
        if hasattr(broker, attr):
            try:
                usd = float(getattr(broker, attr))
                if usd > 0:
                    logging.info(f"Balance via attr {attr}: ${usd:.2f}")
                return usd
            except Exception:
                pass

    logging.warning("Could not determine USD balance; defaulting to $0.00")
    return 0.0

# ------------------------------------------------------------------

def get_broker():
    """Lazy-import the right broker; prefer 'trader.*' then root-level files."""
    if MARKET == "crypto":
        CryptoCls = _import_class(
            ["trader.broker_crypto_ccxt", "broker_crypto_ccxt"],
            ["CryptoBroker", "CCXTCryptoBroker"],
        )
        logging.info(f"Using crypto broker class: {CryptoCls.__name__}")
        return CryptoCls()
    else:
        if EQUITIES_BROKER.lower() == "alpaca":
            AlpacaCls = _import_class(["trader.broker_alpaca", "broker_alpaca"], ["AlpacaBroker"])
            return AlpacaCls()
        RobinhoodCls = _import_class(["trader.broker_robinhood", "broker_robinhood"], ["RobinhoodBroker"])
        return RobinhoodCls()

def pick_universe(market: str):
    return ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"] if market == "crypto" else ["AAPL", "MSFT", "GOOGL", "AMZN"]

def should_sell(entry_price: float, current_price: float, peak_price: Optional[float] = None) -> bool:
    change = (current_price - entry_price) / max(entry_price, 1e-9)
    if change >= TAKE_PROFIT:
        logging.info(f"TAKE_PROFIT hit at {change:.2%}")
        return True
    if peak_price is not None and change >= TRAIL_ACTIVATE:
        trail_stop = peak_price * (1 - TRAIL_DISTANCE)
        if current_price < trail_stop:
            logging.info(f"TRAIL stop hit: price={current_price:.4f}, stop={trail_stop:.4f}")
            return True
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

    balance = safe_get_balance(broker)
    logging.info(f"Available balance: ${balance:.2f}")

    universe = pick_universe(MARKET)
    logging.info(f"Universe: {universe}")

    # Spend controls
    daily_spend = min(balance, DAILY_CAP_USD) if not DRY_RUN else DAILY_CAP_USD
    per_trade = min(PER_TRADE_USD, daily_spend)

    if daily_spend < per_trade or per_trade <= 0:
        logging.warning("Not enough balance for even one trade.")
        return

    # Entry (placeholder)
    symbol = random.choice(universe)
    logging.info(f"Selected {symbol} for trade amount ${per_trade:.2f}")

    try:
        if broker.buy(symbol, per_trade):
            logging.info(f"Trade executed: {symbol} ${per_trade:.2f}")
        else:
            logging.warning("Buy failed.")
    except Exception as e:
        logging.error(f"Buy raised exception: {e}")

    # Sell demo (for guard)
    entry, current, peak = 100.0, 103.0, 104.0
    if should_sell(entry, current, peak):
        qty = 0.1
        try:
            broker.sell(symbol, qty)
        except Exception as e:
            logging.error(f"Sell raised exception: {e}")
        finally:
            logging.info(f"SELL executed: {symbol} qty={qty}")

if __name__ == "__main__":
    run_trader()
