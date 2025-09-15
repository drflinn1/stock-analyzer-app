import os, math, time
from typing import List, Tuple, Optional
import ccxt

# === Env ===
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
EXCHANGE = os.getenv("EXCHANGE", "kraken").lower()

PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "15"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "30"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "6"))

AUTO_UNIVERSE = os.getenv("AUTO_UNIVERSE", "true").lower() == "true"
UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE", "500"))
MANUAL_SYMBOLS = [s.strip() for s in os.getenv("MANUAL_SYMBOLS", "").split(",") if s.strip()]

DROP_PCT = float(os.getenv("DROP_PCT", "0.8"))          # placeholder signal gate
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "1.2"))
TRAIL_PROFIT_PCT = float(os.getenv("TRAIL_PROFIT_PCT", "0.6"))

VERBOSE = os.getenv("VERBOSE", "1") == "1"

def log(msg: str): 
    print(msg, flush=True)

# === Exchange wiring ===
def connect_exchange():
    if EXCHANGE != "kraken":
        raise RuntimeError("Only kraken is wired right now.")
    kwargs = dict(enableRateLimit=True)
    if not DRY_RUN:
        kwargs.update(
            apiKey=os.getenv("KRAKEN_API_KEY", ""),
            secret=os.getenv("KRAKEN_API_SECRET", ""),
        )
    return ccxt.kraken(kwargs)

# === Helpers ===
def round_to_precision(exchange, market, amount: float) -> float:
    # amount precision first
    prec = market.get("precision", {}).get("amount", None)
    if prec is not None:
        try:
            return float(exchange.amount_to_precision(market["symbol"], amount))
        except Exception:
            pass
    # fallback to step (min amount)
    step = market.get("limits", {}).get("amount", {}).get("min", None)
    if step:
        steps = math.floor(amount / step)
        return max(0.0, steps * step)
    # last resort
    return float(f"{amount:.8f}")

def fetch_free_usd(exchange) -> float:
    try:
        bal = exchange.fetch_balance()
        return float(bal.get("free", {}).get("USD", 0.0))
    except Exception as e:
        log(f"WARN fetch_balance failed: {e}")
        return 0.0

def fetch_ticker(exchange, sym: str) -> Optional[dict]:
    try:
        return exchange.fetch_ticker(sym)
    except Exception as e:
        if VERBOSE:
            log(f"WARN fetch_ticker {sym}: {e}")
        return None

def get_min_cost_usd(market: dict, ask: float) -> Optional[float]:
    """
    Kraken doesn't always expose limits.cost.min.
    Infer using amount.min * ask if needed.
    """
    lim = market.get("limits", {})
    # Prefer explicit cost.min
    cmin = lim.get("cost", {}).get("min")
    if cmin:
        try:
            return float(cmin)
        except Exception:
            pass
    # Fallback: amount.min * ask
    amin = lim.get("amount", {}).get("min")
    if amin and ask:
        try:
            return float(amin) * float(ask)
        except Exception:
            pass
    return None

def pick_universe(exchange, markets) -> List[str]:
    symbols = [s for s, m in markets.items() if s.endswith("/USD") and m.get("active", True)]
    symbols.sort()
    if AUTO_UNIVERSE:
        chosen = symbols[: min(UNIVERSE_SIZE, len(symbols))]
        preview = chosen[:5] + (["..."] if len(chosen) > 5 else [])
        log(f"auto_universe: picked {len(chosen)} of {len(symbols)} candidates -> {preview}")
        return chosen
    if MANUAL_SYMBOLS:
        return [s for s in MANUAL_SYMBOLS if s in symbols]
    return ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD"]

def size_order(ask: float, daily_left: float, market, exchange) -> Tuple[str, float, float]:
    """
    Returns (reason_or_ok, qty, notional). If reason startswith 'SKIP', caller should skip.
    """
    notional_target = min(PER_TRADE_USD, daily_left)

    # Market minimum cost check (explicit or inferred)
    mc = get_min_cost_usd(market, ask)
    if mc and notional_target + 1e-9 < mc:
        return (f"SKIP: below min notional ${notional_target:.2f} < ${mc:.2f}", 0.0, 0.0)

    qty_raw = notional_target / ask
    qty = round_to_precision(exchange, market, qty_raw)
    notional_chk = qty * ask
    if qty <= 0 or notional_chk < 0.98 * notional_target:
        return ("SKIP: qty too small after precision", 0.0, 0.0)

    return ("OK", qty, notional_chk)

def try_buy(exchange, market, free_usd: float, daily_left: float) -> Tuple[str, str]:
    sym = market["symbol"]
    t = fetch_ticker(exchange, sym)
    if not t or not t.get("ask"):
        return ("SKIP", f"{sym}: no quote/ask")

    ask = float(t["ask"])

    # Global free-USD guard (fees headroom 1%)
    if free_usd < PER_TRADE_USD * 1.01:
        return ("SKIP_ALL", f"free_usd=${free_usd:.2f} < ${PER_TRADE_USD*1.01:.2f} gate")

    status, qty, notion = size_order(ask, daily_left, market, exchange)
    if status.startswith("SKIP"):
        return ("SKIP", f"{sym}: {status[6:]}")  # trim "SKIP: "

    if DRY_RUN:
        return ("BUY_SIM", f"{sym}: [DRY_RUN] would buy {qty} ~${notion:.2f}")

    # Live order — catch any exchange error and convert to SKIP (no scary ERROR lines)
    try:
        order = exchange.create_market_buy_order(sym, qty)
        oid = order.get("id", "?")
        return ("BUY", f"{sym}: bought {qty} ~${notion:.2f} (order id {oid})")
    except ccxt.BaseError as e:
        return ("SKIP", f"{sym}: order rejected ({str(e)})")

def main():
    log("=== START TRADING OUTPUT ===")
    ex = connect_exchange()
    markets = ex.load_markets()

    universe_syms = pick_universe(ex, markets)
    uni = [markets[s] for s in universe_syms if s in markets]

    free_usd = fetch_free_usd(ex)
    daily_left = DAILY_CAP_USD

    if free_usd < PER_TRADE_USD * 1.01:
        log(f"SKIP_ALL_BUYS: free_usd=${free_usd:.2f} < gate=${PER_TRADE_USD*1.01:.2f} — no buy attempts this run.")
        log("=== END TRADING OUTPUT ===")
        return

    buys = 0
    for m in uni:
        if daily_left < PER_TRADE_USD * 0.99:
            log(f"CAP_REACHED: daily remaining ~${daily_left:.2f}, stopping buys.")
            break

        status, msg = try_buy(ex, m, free_usd, daily_left)
        log(msg)

        if status.startswith("BUY"):
            spent = min(PER_TRADE_USD, daily_left)
            daily_left -= spent
            free_usd -= spent
            buys += 1

        time.sleep(0.25)  # gentle rate limit

    if buys == 0:
        log("No buys this run (likely min-notional/gates).")
    log("=== END TRADING OUTPUT ===")

if __name__ == "__main__":
    main()
