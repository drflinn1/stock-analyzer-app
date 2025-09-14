import os
import math
import json
import asyncio
from datetime import datetime, timezone

import ccxt.async_support as ccxt


# ---------- Helpers -----------------------------------------------------------

def env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    try:
        return float(v)
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v)
    except Exception:
        return int(default)


def env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def env_csv(name: str) -> list[str]:
    v = os.getenv(name, "") or ""
    if not v.strip():
        return []
    return [p.strip() for p in v.split(",") if p.strip()]


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")


# Basic RSI (Wilders)
def rsi(values: list[float], length: int) -> float:
    if len(values) < length + 1:
        return float("nan")
    gains = 0.0
    losses = 0.0
    # seed
    for i in range(1, length + 1):
        ch = values[i] - values[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    gains /= length
    losses /= length
    # update
    for i in range(length + 1, len(values)):
        ch = values[i] - values[i - 1]
        gain = max(ch, 0.0)
        loss = max(-ch, 0.0)
        gains = (gains * (length - 1) + gain) / length
        losses = (losses * (length - 1) + loss) / length
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


# ---------- Robust min-amount/min-cost aware sizing --------------------------

async def compute_amount_for_market(ex: ccxt.Exchange, symbol: str, price: float, budget_usd: float):
    """
    Return (amount, notional) that satisfies both exchange min-amount & min-cost
    and remains <= budget. Returns (None, None) when it cannot trade.
    """
    # Make sure markets are loaded (safe even if done earlier)
    await ex.load_markets()
    m = ex.market(symbol)
    limits = (m.get("limits") or {})
    amt_lim = (limits.get("amount") or {})
    cost_lim = (limits.get("cost") or {})

    min_amt = float(amt_lim.get("min") or 0.0)
    min_cost = float(cost_lim.get("min") or 0.0)

    # Always use the exchange's price precision BEFORE math
    try:
        price = float(ex.price_to_precision(symbol, price))
    except Exception:
        price = float(price)

    implied_by_amt = (min_amt * price) if min_amt else 0.0
    implied_by_cost = min_cost
    required = max(implied_by_amt, implied_by_cost, 0.0)

    # If our budget can't meet the requirement, skip safely
    if required > 0 and budget_usd + 1e-9 < required:
        print(f"[SKIP] {symbol} budget ${budget_usd:.2f} < required ${required:.2f} "
              f"(min_amt={min_amt}, min_cost={min_cost})")
        return None, None

    # Choose a notional that meets the requirement but fits inside budget
    notional = required if required > 0 else budget_usd
    notional = min(notional, budget_usd)

    # Amount = spend / price -> then round to amount precision
    raw_amt = notional / price if price > 0 else 0.0
    try:
        amt = float(ex.amount_to_precision(symbol, raw_amt))
    except Exception:
        amt = raw_amt

    # Recompute notional after rounding
    notional = amt * price

    # Final safety after precision adjustments
    if (min_amt and amt + 1e-12 < min_amt) or (min_cost and notional + 1e-9 < min_cost):
        print(f"[SKIP] {symbol} after precision, amt={amt} notional=${notional:.2f} "
              f"still below mins (min_amt={min_amt}, min_cost={min_cost})")
        return None, None

    return amt, notional


async def place_buy(ex: ccxt.Exchange, symbol: str, price: float, quote_budget: float, dry: bool):
    amt, spend = await compute_amount_for_market(ex, symbol, price, quote_budget)
    if not amt:
        return None

    if dry:
        print(f"[DRY]  BUY {symbol} ≈{amt} @ ${price:.4f} spending ≈${spend:.2f}")
        return None

    print(f"[LIVE] BUY {symbol} ≈{amt} @ ${price:.4f} spending ≈${spend:.2f}")
    try:
        order = await ex.create_order(symbol, "market", "buy", amt, None, {"type": "market"})
        print(f"[LIVE] Order id={order.get('id')} status={order.get('status')}")
        return order
    except Exception as e:
        print(f"[ERROR] create_order failed: {type(e).__name__}: {e}")
        return None


# ---------- Core bot ----------------------------------------------------------

async def main():
    # ----------- Config from env
    DRY_RUN = env_bool("DRY_RUN", True)

    # entry gates
    CANDLES_TIMEFRAME = os.getenv("CANDLES_TIMEFRAME", "5m")
    DROP_PCT = env_float("DROP_PCT", 0.6)           # requires <= -DROP_PCT 5m change
    ENABLE_RSI = env_bool("ENABLE_RSI", True)
    RSI_LEN = env_int("RSI_LEN", 14)
    RSI_MAX = env_float("RSI_MAX", 60.0)

    # portfolio caps
    DAILY_SPEND_CAP_USD = env_float("DAILY_SPEND_CAP_USD", 40.0)
    MAX_OPEN_TRADES = env_int("MAX_OPEN_TRADES", 5)
    MIN_BALANCE_USD = env_float("MIN_BALANCE_USD", 5.0)
    MIN_ACTIVE_POSITION_USD = env_float("MIN_ACTIVE_POSITION_USD", 2.0)
    BUY_SIZE_USD = env_float("BUY_SIZE_USD", 10.0)

    # universe
    UNIVERSE_MODE = os.getenv("UNIVERSE_MODE", "auto").strip().lower()  # "auto" or "list"
    QUOTE = os.getenv("QUOTE", "USD").upper()
    TOP_N_SYMBOLS = env_int("TOP_N_SYMBOLS", 30)
    MIN_USD_VOL = env_float("MIN_USD_VOL", 1_000_000.0)
    INCLUDE = set(env_csv("INCLUDE_SYMBOLS"))
    EXCLUDE = set(env_csv("EXCLUDE_SYMBOLS"))

    # Kraken keys
    api_key = os.getenv("KRAKEN_API_KEY", "")
    api_secret = os.getenv("KRAKEN_API_SECRET", "")
    private_api = bool(api_key and api_secret)

    ex = ccxt.kraken({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    print("=== START TRADING OUTPUT ===")
    print(f"{utc_now_str()} | run started | DRY_RUN={DRY_RUN} | TF={CANDLES_TIMEFRAME} "
          f"| RSI({RSI_LEN}) {'ON' if ENABLE_RSI else 'OFF'} max≤{RSI_MAX:.2f} "
          f"| private_api={'ON' if private_api else 'OFF'}")

    # ----------- Load markets and build universe
    await ex.load_markets()
    all_markets = ex.markets
    spot_usd = [m for m in all_markets.values()
                if (m.get("spot") and m.get("symbol", "").endswith(f"/{QUOTE}"))]

    def normalized(sym: str) -> str:
        # standardize "ABC/USD"
        return sym

    universe = []

    if UNIVERSE_MODE == "list":
        # when list-mode is used, you can pass INCLUDE_SYMBOLS + QUOTE filter
        for s in INCLUDE:
            sym = s if "/" in s else f"{s}/{QUOTE}"
            if sym in ex.symbols:
                universe.append(sym)
    else:
        # auto: pick top-N by 24h USD volume
        tickers = await ex.fetch_tickers([m["symbol"] for m in spot_usd])
        candidates = []
        for m in spot_usd:
            sym = m["symbol"]
            t = tickers.get(sym) or {}
            last = float(t.get("last") or 0.0)
            qvol = float(t.get("quoteVolume") or 0.0)
            bvol = float(t.get("baseVolume") or 0.0)
            usd_vol = qvol if qvol > 0 else (bvol * last)
            if last > 0 and usd_vol >= MIN_USD_VOL:
                if sym not in EXCLUDE:
                    candidates.append((usd_vol, sym))
        candidates.sort(reverse=True)
        universe = [sym for _, sym in candidates[:TOP_N_SYMBOLS]]

    if INCLUDE:
        for s in INCLUDE:
            sym = s if "/" in s else f"{s}/{QUOTE}"
            if sym in ex.symbols and sym not in universe:
                universe.append(sym)
    if EXCLUDE:
        universe = [s for s in universe if s not in EXCLUDE]

    print(f"{utc_now_str()} | universe_mode={UNIVERSE_MODE} | quote={QUOTE} "
          f"| top_n={TOP_N_SYMBOLS} | min_usd_vol={MIN_USD_VOL:,.2f}")
    print(f"{utc_now_str()} | scanning={universe}")

    # ----------- Balance & open-trade snapshot
    balance = await ex.fetch_balance()
    usd_free = float(balance.get("free", {}).get(QUOTE, 0.0) or 0.0)

    # Compute open spot positions (rough): sum non-USD assets valued > threshold
    # Need last prices for valuation
    tickers_map = await ex.fetch_tickers(universe)
    def last_price(sym: str) -> float:
        t = tickers_map.get(sym) or {}
        return float(t.get("last") or 0.0)

    open_positions = 0
    dust_ignored = 0
    for asset, qty in (balance.get("total") or {}).items():
        if asset == QUOTE:
            continue
        q = float(qty or 0.0)
        if q <= 0:
            continue
        sym = f"{asset}/{QUOTE}"
        if sym not in ex.symbols:
            continue
        lp = last_price(sym)
        value = q * lp
        if value >= MIN_ACTIVE_POSITION_USD:
            open_positions += 1
        else:
            dust_ignored += 1

    # naive daily remaining: cap within available cash
    daily_remaining = max(0.0, min(DAILY_SPEND_CAP_USD, max(0.0, usd_free - MIN_BALANCE_USD)))
    print(f"{utc_now_str()} | budget | USD_free=${usd_free:.2f} | daily_remaining=${daily_remaining:.2f} "
          f"| open_trades={open_positions}/{MAX_OPEN_TRADES} | dust_ignored={dust_ignored}")

    # ----------- Scan for entries (5m change + RSI gate)
    # fetch OHLCV and compute 5m change + RSI
    top_preview = []
    best = None  # (score, sym, drop_pct, rsi_val, last)
    for sym in universe:
        try:
            ohlcv = await ex.fetch_ohlcv(sym, timeframe=CANDLES_TIMEFRAME, limit=RSI_LEN + 20)
            closes = [c[4] for c in ohlcv]
            if len(closes) < 3:
                continue
            last = float(closes[-1])
            prev = float(closes[-2])
            change = (last - prev) / prev * 100.0
            rsiv = rsi(closes, RSI_LEN)
            top_preview.append((change, rsiv, sym, last))
            # Entry filter
            if (change <= -DROP_PCT) and (not ENABLE_RSI or (rsiv <= RSI_MAX)):
                # More negative change is better; lower RSI better
                score = (change, rsiv)  # tuple sorts by change first
                if best is None or score < (best[0], best[1]):
                    best = (change, rsiv, sym, last)
        except Exception as e:
            print(f"[WARN] fetch_ohlcv {sym} failed: {e}")

    # top-5 preview (most negative 5m change)
    top_preview.sort(key=lambda x: x[0])  # ascending change => biggest drop first
    preview_items = []
    for ch, rsiv, sym, last in top_preview[:5]:
        marker = "✓" if (ch <= -DROP_PCT) and (not ENABLE_RSI or (rsiv <= RSI_MAX)) else "x"
        preview_items.append(f"{sym} Δ{ch:+.2f}% rsi={rsiv:.2f} {marker}")
    print(f"{utc_now_str()} | preview_top5 = [{'; '.join(preview_items)}]")

    if not best:
        print(f"{utc_now_str()} | Best candidate | none (no symbol passed the gates)")
        print("Run complete. buys_placed=0 | sells_placed=0 | DRY_RUN=" + ("True" if DRY_RUN else "False"))
        await ex.close()
        return

    ch, rsiv, sym, last = best
    # determine per-trade spend
    per_trade = min(BUY_SIZE_USD, daily_remaining)
    gate_str = f"(gate -{DROP_PCT:.2f}%)"
    if per_trade <= 0:
        print(f"{utc_now_str()} | Best candidate | {sym} {CANDLES_TIMEFRAME}_change={ch:+.2f}% {gate_str}, "
              f"RSI {rsiv:.2f} -> NO BUY (no daily remaining)")
        print("Run complete. buys_placed=0 | sells_placed=0 | DRY_RUN=" + ("True" if DRY_RUN else "False"))
        await ex.close()
        return

    # final print + place order
    print(f"{utc_now_str()} | Best candidate | {sym} {CANDLES_TIMEFRAME}_change={ch:+.2f}% {gate_str}, "
          f"RSI {rsiv:.2f} -> BUY ${per_trade:.2f} @ {last:.8f}")

    order = await place_buy(ex, sym, last, per_trade, DRY_RUN)
    buys_placed = 1 if order or DRY_RUN else 0

    print("Run complete. buys_placed="
          f"{buys_placed} | sells_placed=0 | DRY_RUN=" + ("True" if DRY_RUN else "False"))

    await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
