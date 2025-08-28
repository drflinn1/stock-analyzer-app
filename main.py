import os
import sys
import math
import logging
import ccxt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ===== Env knobs =====
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "10"))
DAILY_CAP = float(os.getenv("DAILY_CAP", "50"))
DROP_PCT = float(os.getenv("DROP_PCT", "2.0"))
SELL_DROP_PCT = float(os.getenv("SELL_DROP_PCT", "2.0"))
UNIVERSE = os.getenv("UNIVERSE", "AUTO_USDT")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "30"))
MIN_LAST_CANDLE_USD = float(os.getenv("MIN_LAST_CANDLE_USD", "50000"))
EXCLUDE = set(s.strip() for s in os.getenv("EXCLUDE", "").split(",") if s.strip())

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")

# ===== Exchange =====
def get_exchange():
    opts = {"enableRateLimit": True}
    if not DRY_RUN:
        if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
            raise RuntimeError("Missing Kraken API credentials in environment.")
        opts.update({"apiKey": KRAKEN_API_KEY, "secret": KRAKEN_API_SECRET})
    return ccxt.kraken(opts)

# ===== Universe =====
def build_universe(exchange):
    if UNIVERSE != "AUTO_USDT":
        return [s.strip() for s in UNIVERSE.split(",") if s.strip()]

    markets = exchange.load_markets()
    symbols = []
    for s, m in markets.items():
        if m.get('spot') and m.get('active') and m.get('quote') == 'USDT' and s not in EXCLUDE:
            base = m.get('base', '')
            if any(x in base for x in ['UP', 'DOWN', 'BULL', 'BEAR', '.S']):
                continue
            symbols.append(s)
    symbols.sort()
    if MAX_SYMBOLS and len(symbols) > MAX_SYMBOLS:
        symbols = symbols[:MAX_SYMBOLS]
    return symbols

# ===== Helpers =====
def last_hour_change_and_dollar(symbol, candles):
    if len(candles) < 2:
        return None, None
    last = candles[-1][4]
    prev = candles[-2][4]
    change_pct = (last - prev) / prev * 100 if prev else 0.0
    last_vol_usd = candles[-1][5] * last
    return change_pct, last_vol_usd

def quantize_amount(exchange, symbol: str, amount: float) -> float:
    q = float(exchange.amount_to_precision(symbol, amount))
    m = exchange.markets.get(symbol, {})
    min_amt = (m.get("limits", {}).get("amount", {}) or {}).get("min")
    if min_amt and q < float(min_amt):
        q = float(min_amt)
    return q

def place_stop_loss(exchange, symbol: str, base_qty: float, entry_price: float, drop_pct: float):
    stop_price = round(entry_price * (1 - drop_pct / 100.0), 8)
    try:
        order = exchange.create_order(
            symbol,
            type='stop',
            side='sell',
            amount=base_qty,
            price=None,
            params={'stopPrice': stop_price, 'trigger': 'last'}
        )
        logger.info(f"Placed stop-loss: {order}")
    except Exception as e:
        logger.error(f"Stop-loss placement failed for {symbol} at {stop_price}: {e}")

# ===== Main =====
def main():
    logger.info("=== START TRADING OUTPUT ===")
    ex = get_exchange()

    symbols = build_universe(ex)
    logger.info(f"Universe size: {len(symbols)}")
    if not symbols:
        logger.info("No symbols to evaluate.")
        return

    candidates = []
    for s in symbols:
        try:
            ohlcv = ex.fetch_ohlcv(s, timeframe='1h', limit=3)
            if not ohlcv:
                continue
            chg, usd = last_hour_change_and_dollar(s, ohlcv)
            if chg is None:
                continue
            if usd is not None and usd < MIN_LAST_CANDLE_USD:
                continue
            if chg <= -DROP_PCT:
                candidates.append((chg, usd or 0.0, s, ohlcv[-1][4]))
        except Exception as e:
            logger.info(f"Skip {s}: {e}")

    if not candidates:
        logger.info("No candidates met the drop gate.")
        logger.info("=== END TRADING OUTPUT ===")
        return

    candidates.sort(key=lambda t: (t[0], -t[1]))
    best_chg, best_usd, best_sym, last_price = candidates[0]
    logger.info(f"Best candidate: {best_sym} change {best_chg:.2f}% last ${last_price:.4f} vol~${best_usd:.0f}")

    base_qty = TRADE_AMOUNT / max(last_price, 1e-9)
    if DRY_RUN:
        logger.info(f"[DRY RUN] BUY ~{base_qty:.8f} {best_sym} spending {TRADE_AMOUNT} USDT")
        logger.info("=== END TRADING OUTPUT ===")
        return

    qty = quantize_amount(ex, best_sym, base_qty)
    logger.info(f"Placing market BUY: {qty} {best_sym} (~{TRADE_AMOUNT} USDT @ {last_price})")
    try:
        order = ex.create_market_order(best_sym, 'buy', qty)
        logger.info(f"Executed buy: {order}")
        place_stop_loss(ex, best_sym, qty, last_price, SELL_DROP_PCT)
    except Exception as e:
        logger.error(f"Buy failed for {best_sym}: {e}")

    logger.info("=== END TRADING OUTPUT ===")

if __name__ == '__main__':
    main()
