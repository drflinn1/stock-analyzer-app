import os, sys, logging, ccxt

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "25"))
DAILY_CAP = float(os.getenv("DAILY_CAP", "75"))
DROP_PCT = float(os.getenv("DROP_PCT", "0.0"))
SELL_DROP_PCT = float(os.getenv("SELL_DROP_PCT", "2.0"))
UNIVERSE = os.getenv("UNIVERSE", "AUTO")                  # AUTO | AUTO_USDT | AUTO_USD | csv
PREFERRED_QUOTES = [q.strip() for q in os.getenv("PREFERRED_QUOTES","USD,USDT").split(",") if q.strip()]
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "100"))
MIN_LAST_CANDLE_USD = float(os.getenv("MIN_LAST_CANDLE_USD", "0"))
EXCLUDE = set(s.strip() for s in os.getenv("EXCLUDE", "").split(",") if s.strip())
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
FORCE_BUY = os.getenv("FORCE_BUY", "true").lower() == "true"

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")

def get_exchange():
    opts = {"enableRateLimit": True}
    if not DRY_RUN:
        if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
            raise RuntimeError("Missing Kraken API credentials in environment.")
        opts.update({"apiKey": KRAKEN_API_KEY, "secret": KRAKEN_API_SECRET})
    return ccxt.kraken(opts)

def build_universe(ex):
    if UNIVERSE not in ("AUTO", "AUTO_USDT", "AUTO_USD"):
        return [(s.strip(), s.strip().split("/")[1]) for s in UNIVERSE.split(",") if s.strip()]
    want_quotes = (set(PREFERRED_QUOTES) if UNIVERSE == "AUTO"
                   else ({"USDT"} if UNIVERSE == "AUTO_USDT" else {"USD"}))
    markets = ex.load_markets()
    out = []
    for s, m in markets.items():
        if not (m.get("spot") and m.get("active")): continue
        q = m.get("quote")
        if q not in want_quotes: continue
        if s in EXCLUDE: continue
        b = m.get("base","")
        if any(x in b for x in ["UP","DOWN","BULL","BEAR",".S"]): continue
        out.append((s, q))
    out.sort(key=lambda t: (PREFERRED_QUOTES.index(t[1]) if t[1] in PREFERRED_QUOTES else 999, t[0]))
    return out[:MAX_SYMBOLS] if MAX_SYMBOLS and len(out) > MAX_SYMBOLS else out

def last_change_and_dollar(candles):
    if len(candles) < 2: return None, None, None
    prev, last = candles[-2][4], candles[-1][4]
    vol_last = candles[-1][5]
    chg = (last - prev) / prev * 100 if prev else 0.0
    usd = vol_last * last
    return chg, usd, last

def quantize_amount(ex, symbol, amount):
    q = float(ex.amount_to_precision(symbol, amount))
    m = ex.markets.get(symbol, {})
    min_amt = (m.get("limits", {}).get("amount", {}) or {}).get("min")
    if min_amt and q < float(min_amt): q = float(min_amt)
    return q

def place_stop_loss(ex, symbol, base_qty, entry_price, drop_pct):
    stop_price = round(entry_price * (1 - drop_pct/100.0), 8)
    try:
        order = ex.create_order(symbol, type="stop", side="sell", amount=base_qty,
                                price=None, params={"stopPrice": stop_price, "trigger": "last"})
        logger.info(f"Placed stop-loss: {order}")
    except Exception as e:
        logger.error(f"Stop-loss placement failed for {symbol} at {stop_price}: {e}")

def main():
    logger.info("=== START TRADING OUTPUT ===")
    ex = get_exchange()

    # Try to show balances, but DO NOT fail the run if permission is missing
    have_balances = False
    if not DRY_RUN:
        try:
            bal = ex.fetch_balance()
            parts = []
            for q in PREFERRED_QUOTES:
                acct = bal.get(q) or bal.get(q.upper()) or {}
                parts.append(f"{q}={float(acct.get('free', 0) or 0):.2f}")
            logger.info("Quote balances: " + ", ".join(parts))
            have_balances = True
        except Exception as e:
            logger.warning(f"Balance check skipped: {e}")

    symbols = build_universe(ex)
    logger.info(f"Universe size: {len(symbols)} (quotes preferred: {PREFERRED_QUOTES})")
    if not symbols:
        logger.info("No symbols to evaluate."); logger.info("=== END TRADING OUTPUT ==="); return

    eligible, scanned = [], []
    for s, quote in symbols:
        try:
            ohlcv = ex.fetch_ohlcv(s, timeframe=TIMEFRAME, limit=3)
            if not ohlcv: continue
            chg, usd, last = last_change_and_dollar(ohlcv)
            if chg is None: continue
            if usd is not None and usd < MIN_LAST_CANDLE_USD: continue
            scanned.append((chg, usd or 0.0, s, quote, last))
            if chg <= -DROP_PCT:
                eligible.append((chg, usd or 0.0, s, quote, last))
        except Exception as e:
            logger.info(f"Skip {s}: {e}")

    if eligible:
        eligible.sort(key=lambda t: (t[0], -t[1]))
        pick_from = eligible
        logger.info(f"Eligible candidates: {len(eligible)}")
    elif FORCE_BUY and scanned:
        # If balances unavailable, don't filter by funds; Kraken will reject if truly insufficient
        if have_balances and not DRY_RUN:
            try:
                bal = ex.fetch_balance()
                def ok(t):
                    q = t[3]
                    acct = bal.get(q) or bal.get(q.upper()) or {}
                    return float(acct.get('free', 0) or 0) >= TRADE_AMOUNT
                scanned = [t for t in scanned if ok(t)]
            except Exception:
                pass
        if scanned:
            scanned.sort(key=lambda t: (t[0], -t[1]))
            pick_from = scanned
            logger.info("No candidates met the drop gate; FORCE_BUY picking top drop.")
        else:
            logger.info("FORCE_BUY had nothing eligible to spend on.")
            logger.info("=== END TRADING OUTPUT ===")
            return
    else:
        logger.info("No candidates met the drop gate.")
        logger.info("=== END TRADING OUTPUT ===")
        return

    chg, usd, best_sym, quote, last_price = pick_from[0]
    logger.info(f"Best candidate: {best_sym} ({quote}) change {chg:.2f}% last ${last_price:.6f} vol~${usd:.0f}")

    base_qty = TRADE_AMOUNT / max(last_price, 1e-9)
    if DRY_RUN:
        logger.info(f"[DRY RUN] BUY ~{base_qty:.8f} {best_sym} spending {TRADE_AMOUNT} {quote}")
        logger.info("=== END TRADING OUTPUT ==="); return

    qty = quantize_amount(ex, best_sym, base_qty)
    logger.info(f"Placing market BUY: {qty} {best_sym} (~{TRADE_AMOUNT} {quote} @ {last_price})")
    try:
        order = ex.create_market_order(best_sym, 'buy', qty)
        logger.info(f"Executed buy: {order}")
        place_stop_loss(ex, best_sym, qty, last_price, SELL_DROP_PCT)
    except Exception as e:
        logger.error(f"Buy failed for {best_sym}: {e}")

    logger.info("=== END TRADING OUTPUT ===")

if __name__ == "__main__":
    main()
