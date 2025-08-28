import os
import sys
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
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
FORCE_BUY = os.getenv("FORCE_BUY", "false").lower() == "true"

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
def last_change_and_dollar(candles):
    # candles: [ts, open, high, low, close, volume]
    if len(candles) < 2:
        return None, None, None
    prev_close = candles[-2][4]
    last_close = candles[-1][4]
    vol_last = candles[-1][5]
    change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0.0
    usd = vol_last * last_close
    return change_pct, usd, last_close

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

    eligible = []
    scanned = []
    for s in symbols:
        try:
            ohlcv = ex.fetch_ohlcv(s, timeframe=TIMEFRAME, limit=3)
            if not ohlcv:
                continue
            chg, usd, last_price = last_change_and_dollar(ohlcv)
            if chg is None:
                continue
            if usd is not None and usd < MIN_LAST_CANDLE_USD:
                continue
            scanned.append((chg, usd or 0.0, s, last_price))
            if chg <= -DROP_PCT:
                eligible.append((chg, usd or 0.0, s, last_price))
        except Exception as e:
            logger.info(f"Skip {s}: {e}")

    if eligible:
        eligible.sort(key=lambda t: (t[0], -t[1]))  # most negative first, then larger $vol
        pick_from = eligible
        logger.info(f"Eligible candidates: {len(eligible)}")
    elif FORCE_BUY and scanned:
        scanned.sort(key=lambda t: (t[0], -t[1]))
        pick_from = scanned
        logger.info("No candidates met the drop gate; FORCE_BUY enabled — picking top drop.")
    else:
        logger.info("No candidates met the drop gate.")
        logger.info("=== END TRADING OUTPUT ===")
        return

    chg, usd, best_sym, last_price = pick_from[0]
    logger.info(f"Best candidate: {best_sym} change {chg:.2f}% last ${last_price:.6f} vol~${usd:.0f}")

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
