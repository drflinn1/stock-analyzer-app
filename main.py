# main.py — Crypto Live with Cash-Short Rotation + Sell Guards (TP/SL/TRAIL)
# (Kraken-safe sells: checks min amount/cost, shows diagnostics, avoids tiny remainders)

from __future__ import annotations
import os, json, pathlib, traceback
from typing import Dict, List, Tuple, Optional

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- helpers ----------

def as_bool(v: Optional[str], default: bool) -> bool:
    if v is None: return default
    return v.strip().lower() in ("1","true","yes","y","on")

def as_float(v: Optional[str], default: float) -> float:
    try: return float(v) if v is not None else default
    except: return default

def as_int(v: Optional[str], default: int) -> int:
    try: return int(v) if v is not None else default
    except: return default

def env_list(v: Optional[str], default: List[str]) -> List[str]:
    if not v: return default
    out = [s.strip() for s in v.split(",") if s.strip()]
    return out or default

# ---------- ENV ----------

EXCHANGE_ID = os.getenv("EXCHANGE","kraken").lower()
API_KEY     = os.getenv("API_KEY") or os.getenv("KRAKEN_API_KEY") or os.getenv("CCXT_API_KEY") or ""
API_SECRET  = os.getenv("API_SECRET") or os.getenv("KRAKEN_API_SECRET") or os.getenv("CCXT_API_SECRET") or ""

DRY_RUN     = as_bool(os.getenv("DRY_RUN"), True)
MAX_POS     = as_int(os.getenv("MAX_POSITIONS"), 6)
USD_PER_TRADE = as_float(os.getenv("USD_PER_TRADE"), 15.0)
RESERVE_USD = as_float(os.getenv("RESERVE_USD"), 80.0)

ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)
ROTATE_MIN_EDGE_PCT    = as_float(os.getenv("ROTATE_MIN_EDGE_PCT"), 2.0)
COOLDOWN_RUNS          = as_int(os.getenv("COOLDOWN_RUNS"), 1)

SYMBOL_WHITELIST = env_list(os.getenv("SYMBOL_WHITELIST"),
    ["BTC/USD","ETH/USD","SOL/USD","DOGE/USD","ZEC/USD","ENA/USD"])

TAKE_PROFIT_PCT = as_float(os.getenv("TAKE_PROFIT_PCT"), 3.5)
STOP_LOSS_PCT   = as_float(os.getenv("STOP_LOSS_PCT"), 2.0)
TRAIL_ARM_PCT   = as_float(os.getenv("TRAIL_ARM_PCT"), 1.0)
TRAIL_PCT       = as_float(os.getenv("TRAIL_PCT"), 1.5)

STATE_DIR = pathlib.Path(".state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
COOLDOWN_PATH = STATE_DIR / "rotation_cooldowns.json"
ENTRIES_PATH  = STATE_DIR / "entries.json"
HIGHS_PATH    = STATE_DIR / "highs.json"

USD_KEYS    = ("USD","ZUSD")
STABLE_KEYS = ("USDT",)

# ---------- io ----------

def load_json(p: pathlib.Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except: pass
    return default

def save_json(p: pathlib.Path, data) -> None:
    tmp = p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)

# ---------- exchange ----------

def make_exchange() -> ccxt.Exchange:
    cls = getattr(ccxt, EXCHANGE_ID)
    return cls({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })

def last_price(ex: ccxt.Exchange, symbol: str) -> float:
    t = ex.fetch_ticker(symbol)
    return float(t.get("last") or t.get("close") or t.get("ask") or 0)

def get_free_cash_usd(bal: Dict) -> float:
    total = 0.0
    for k in USD_KEYS:
        total += float(bal.get(k, {}).get("free", 0) or bal.get(k, 0) or 0)
    for k in STABLE_KEYS:
        total += float(bal.get(k, {}).get("free", 0) or bal.get(k, 0) or 0)
    return total

def canonical_symbol(ex: ccxt.Exchange, base: str) -> Optional[str]:
    for q in ("USD","USDT"):
        s = f"{base}/{q}"
        if s in ex.markets: return s
    return None

def list_current_positions(ex: ccxt.Exchange, bal: Dict) -> List[str]:
    held: List[str] = []
    for cur, obj in bal.items():
        if cur in USD_KEYS or cur in STABLE_KEYS: continue
        try: amt = float(obj.get("total",0) if isinstance(obj,dict) else obj)
        except: amt = 0.0
        if amt > 0:
            sym = canonical_symbol(ex, cur)
            if sym: held.append(sym)
    return [s for s in dict.fromkeys(held) if s in ex.markets]

def momentum_score_1h(ex: ccxt.Exchange, symbol: str) -> float:
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1h", limit=3)
        if len(ohlcv) < 2: return 0.0
        open_prev  = ohlcv[-2][1]
        close_last = ohlcv[-1][4]
        if open_prev <= 0: return 0.0
        return (close_last - open_prev) / open_prev * 100.0
    except: return 0.0

def rank_symbols(ex: ccxt.Exchange, symbols: List[str]):
    scored = [(s, momentum_score_1h(ex, s)) for s in symbols]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

# ---------- state ----------

def get_cooldowns() -> Dict[str,int]: return load_json(COOLDOWN_PATH, {})
def dec_cooldowns(cd: Dict[str,int]) -> Dict[str,int]:
    out = {}
    for k,v in cd.items():
        nv = max(0, int(v)-1)
        if nv>0: out[k]=nv
    return out

def get_entries() -> Dict[str,float]: return load_json(ENTRIES_PATH, {})
def get_highs() -> Dict[str,float]:   return load_json(HIGHS_PATH, {})

# ---------- order helpers with Kraken-safe checks ----------

def _market_limits(ex: ccxt.Exchange, symbol: str):
    m = ex.market(symbol)
    amt_min  = (m.get("limits",{}) or {}).get("amount",{}).get("min")
    cost_min = (m.get("limits",{}) or {}).get("cost",  {}).get("min")
    return float(amt_min or 0), float(cost_min or 0)

def _free_and_total_base(ex: ccxt.Exchange, symbol: str):
    bal = ex.fetch_balance()
    base = symbol.split("/")[0]
    free = total = 0.0
    if base in bal and isinstance(bal[base], dict):
        free  = float(bal[base].get("free",  0) or 0)
        total = float(bal[base].get("total", 0) or 0)
    elif base in bal:
        try:
            total = float(bal[base] or 0); free = total
        except: pass
    return free, total

def place_sell(ex: ccxt.Exchange, symbol: str, pct_of_position: float = 1.0) -> Tuple[bool,str]:
    """
    Kraken-safe sell:
      - if free == 0 but total > 0 and no open orders, use total
      - respect min amount / min cost
      - show diagnostics if skipping
    """
    try:
        free, total = _free_and_total_base(ex, symbol)

        # If nothing free, check open orders; if none, use total
        if free <= 0 and total > 0:
            try:
                open_orders = ex.fetch_open_orders(symbol)
            except Exception:
                open_orders = []
            if not open_orders:
                free = total

        amt = max(0.0, min(free, total)) * max(0.0, min(1.0, pct_of_position))
        if amt <= 0:
            return False, f"SELL skip {symbol} — no free/total amount"

        price = last_price(ex, symbol)
        if price <= 0:
            return False, f"SELL skip {symbol} — price unknown"

        min_amt, min_cost = _market_limits(ex, symbol)

        # Round to precision first, then re-check thresholds
        amt_precise = float(ex.amount_to_precision(symbol, amt))
        est_cost = amt_precise * price

        if min_amt and amt_precise < min_amt:
            return False, (
                f"SELL skip {symbol} — amount {amt_precise:.8f} < min_amount {min_amt} "
                f"(price {price:.8f}, est_cost {est_cost:.4f})"
            )
        if min_cost and est_cost < min_cost:
            return False, (
                f"SELL skip {symbol} — est_cost ${est_cost:.4f} < min_cost ${min_cost:.4f} "
                f"(amount {amt_precise:.8f}, price {price:.8f})"
            )

        if DRY_RUN:
            return True, (
                f"SELL ok {symbol} (simulated) amt={amt_precise:.8f} "
                f"[min_amt={min_amt}, min_cost={min_cost}, price={price:.8f}]"
            )
        order = ex.create_market_sell_order(symbol, amount=amt_precise)
        return True, f"SELL ok {symbol} id={order.get('id','?')} amt={amt_precise:.8f}"
    except Exception as e:
        return False, f"SELL fail {symbol}: {e}"

def place_buy(ex: ccxt.Exchange, symbol: str, spend_usd: float) -> Tuple[bool,str]:
    try:
        if spend_usd <= 0:
            return False, f"BUY skip {symbol} — spend<=0"
        price = last_price(ex, symbol)
        if price <= 0:
            return False, f"BUY skip {symbol} — price unknown"

        min_amt, min_cost = _market_limits(ex, symbol)
        amt = spend_usd / price
        amt = float(ex.amount_to_precision(symbol, amt))
        est_cost = amt * price

        if min_amt and amt < min_amt:
            return False, (
                f"BUY skip {symbol} — amount {amt:.8f} < min_amount {min_amt} "
                f"(price {price:.8f}, est_cost {est_cost:.4f})"
            )
        if min_cost and est_cost < min_cost:
            return False, (
                f"BUY skip {symbol} — est_cost ${est_cost:.4f} < min_cost ${min_cost:.4f} "
                f"(amount {amt:.8f}, price {price:.8f})"
            )

        if amt <= 0:
            return False, f"BUY skip {symbol} — tiny amount after precision"

        if DRY_RUN:
            return True, (
                f"BUY ok {symbol} (simulated) spend=${spend_usd:.2f} amt={amt:.8f} "
                f"[min_amt={min_amt}, min_cost={min_cost}, price={price:.8f}]"
            )
        order = ex.create_market_buy_order(symbol, amount=amt)
        return True, f"BUY ok {symbol} id={order.get('id','?')} spend=${spend_usd:.2f} amt={amt:.8f}"
    except Exception as e:
        return False, f"BUY fail {symbol}: {e}"

# ---------- guards ----------

def check_take_profit(symbol: str, entry: float, price: float, tp_pct: float) -> bool:
    if entry > 0:
        chg = (price - entry) / entry * 100.0
        if chg >= tp_pct:
            print(f"TAKE_PROFIT: {symbol} +{chg:.2f}% ≥ {tp_pct:.2f}% → sell")
            return True
    return False

def check_stop_loss(symbol: str, entry: float, price: float, sl_pct: float) -> bool:
    if entry > 0:
        chg = (price - entry) / entry * 100.0
        if chg <= -abs(sl_pct):
            print(f"STOP_LOSS: {symbol} {chg:.2f}% ≤ -{abs(sl_pct):.2f}% → sell")
            return True
    return False

def check_trailing(symbol: str, entry: float, price: float, highs: Dict[str,float],
                   arm_pct: float, trail_pct: float) -> Tuple[bool, Dict[str,float]]:
    if entry <= 0: return (False, highs)
    gain = (price - entry) / entry * 100.0
    hi = float(highs.get(symbol, 0.0) or 0.0)
    if gain >= arm_pct:
        if hi <= 0 or price > hi:
            highs[symbol] = price
            print(f"TRAIL arm/update: {symbol} armed at +{arm_pct:.2f}% (hi={price:.6f})")
        else:
            dd = (hi - price) / hi * 100.0
            if dd >= trail_pct:
                print(f"TRAIL: {symbol} drawdown {dd:.2f}% ≥ {trail_pct:.2f}% from hi → sell")
                return (True, highs)
    return (False, highs)

# ---------- main ----------

def main() -> None:
    print("============================================================")
    print("CRYPTO LIVE ▶ Cash-Short Rotation + Sell Guards (TP/SL/TRAIL)")
    if DRY_RUN: print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")
    print(f"Exchange={EXCHANGE_ID}  MaxPos={MAX_POS}  USD_PER_TRADE=${USD_PER_TRADE:.2f}  Reserve=${RESERVE_USD:.2f}")
    print(f"RotateWhenCashShort={ROTATE_WHEN_CASH_SHORT}  RotateEdge≥{ROTATE_MIN_EDGE_PCT:.2f}%  CooldownRuns={COOLDOWN_RUNS}")
    print(f"TAKE_PROFIT={TAKE_PROFIT_PCT:.2f}%  STOP_LOSS={STOP_LOSS_PCT:.2f}%  TRAIL arm={TRAIL_ARM_PCT:.2f}% dist={TRAIL_PCT:.2f}%")
    print("Whitelist:", ", ".join(SYMBOL_WHITELIST))
    print("============================================================")

    ex = make_exchange(); ex.load_markets()

    cooldowns = dec_cooldowns(get_cooldowns())
    entries   = get_entries()
    highs     = get_highs()

    bal = ex.fetch_balance()
    free_cash = get_free_cash_usd(bal)
    held_syms = list_current_positions(ex, bal)
    print(f"Free cash ≈ ${free_cash:.2f} | Held positions: {len(held_syms)} → {', '.join(held_syms) if held_syms else '(none)'}")

    # init entries for discovered holdings
    for s in held_syms:
        if s not in entries:
            try:
                p = last_price(ex, s)
                if p > 0:
                    entries[s] = p
                    print(f"Init entry: {s} = {p:.6f}")
            except: pass

    # SELL guards pass
    for s in list(held_syms):
        try:
            price = last_price(ex, s)
            entry = float(entries.get(s,0) or 0)
            if price <= 0 or entry <= 0:
                continue

            if check_stop_loss(s, entry, price, STOP_LOSS_PCT) or \
               check_take_profit(s, entry, price, TAKE_PROFIT_PCT):
                ok, msg = place_sell(ex, s, pct_of_position=1.0)
                print(msg)
                if ok:
                    entries.pop(s, None); highs.pop(s, None)
                    try: bal = ex.fetch_balance()
                    except: pass
                    continue

            did_trail, highs = check_trailing(s, entry, price, highs, TRAIL_ARM_PCT, TRAIL_PCT)
            if did_trail:
                ok, msg = place_sell(ex, s, pct_of_position=1.0)
                print(msg)
                if ok:
                    entries.pop(s, None); highs.pop(s, None)
                    try: bal = ex.fetch_balance()
                    except: pass
                    continue
        except Exception as e:
            print(f"Guard error on {s}: {e}")

    # refresh after possible sells
    try: bal = ex.fetch_balance()
    except: pass
    free_cash = get_free_cash_usd(bal)
    held_syms = list_current_positions(ex, bal)

    # candidates / holdings ranking
    cooled_whitelist = [s for s in SYMBOL_WHITELIST if cooldowns.get(s,0)==0]
    ranked_candidates = rank_symbols(ex, cooled_whitelist)
    best_symbol, best_score = (ranked_candidates[0] if ranked_candidates else (None, 0.0))
    ranked_holdings = rank_symbols(ex, held_syms) if held_syms else []
    worst_symbol, worst_score = (ranked_holdings[-1] if ranked_holdings else (None, 0.0))

    # rotation scan
    if best_symbol and worst_symbol:
        edge = best_score - worst_score
        print(f"ROTATE scan: best={best_symbol} {best_score:.1f}% vs worst={worst_symbol} {worst_score:.1f}% → edge={edge:.1f}%")
    elif best_symbol and not worst_symbol:
        print(f"ROTATE scan: best={best_symbol} {best_score:.1f}% (no current holdings to rotate)")
    else:
        print("ROTATE scan: (no candidate / no data)")

    # cash-short rotation
    did_rotate = False
    if ROTATE_WHEN_CASH_SHORT and free_cash < RESERVE_USD and best_symbol and worst_symbol and best_symbol != worst_symbol:
        edge = best_score - worst_score
        if edge >= ROTATE_MIN_EDGE_PCT:
            ok_s, msg_s = place_sell(ex, worst_symbol, pct_of_position=1.0)
            print(msg_s)
            try: bal = ex.fetch_balance()
            except: pass
            new_cash = get_free_cash_usd(bal)
            spend = min(USD_PER_TRADE, new_cash - max(0.0, RESERVE_USD - new_cash))
            if spend <= 0: spend = min(USD_PER_TRADE, new_cash)
            ok_b, msg_b = place_buy(ex, best_symbol, spend_usd=max(0.0, spend))
            print(msg_b)
            if ok_b:
                try:
                    p = last_price(ex, best_symbol)
                    if p > 0: entries[best_symbol] = p; highs.pop(best_symbol, None)
                except: pass
                cooldowns[best_symbol] = max(cooldowns.get(best_symbol,0), COOLDOWN_RUNS)
                did_rotate = True
                print(f"cooldown note: {best_symbol} rotation cooldown {COOLDOWN_RUNS} run(s)")
        else:
            print(f"ROTATE skip — edge {edge:.1f}% < {ROTATE_MIN_EDGE_PCT:.1f}%")

    # new entry if below cap and enough cash
    if not did_rotate and best_symbol:
        if len(held_syms) < MAX_POS and free_cash >= (RESERVE_USD + USD_PER_TRADE):
            ok_b, msg_b = place_buy(ex, best_symbol, spend_usd=USD_PER_TRADE)
            print(msg_b)
            if ok_b:
                try:
                    p = last_price(ex, best_symbol)
                    if p > 0: entries[best_symbol] = p; highs.pop(best_symbol, None)
                except: pass
                cooldowns[best_symbol] = max(cooldowns.get(best_symbol,0), COOLDOWN_RUNS)

    save_json(COOLDOWN_PATH, cooldowns)
    save_json(ENTRIES_PATH, entries)
    save_json(HIGHS_PATH, highs)
    print("DONE.")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        raise
