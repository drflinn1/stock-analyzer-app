# main.py — Crypto Live with Cash-Short Rotation + Sell Guards (TP / SL / TRAIL)
# - DRY_RUN banner + safe order shims
# - USD reserve guard (RESERVE_USD)
# - Auto-pick best symbol by momentum score (1h change)
# - Cash-short rotation:
#     If free cash < reserve and ROTATE_WHEN_CASH_SHORT=true:
#       rank current holdings vs best candidate; if edge ≥ ROTATE_MIN_EDGE_PCT,
#       SELL worst … BUY best, then set a 1-run cooldown on the new buy.
# - Sell guards per holding:
#     TAKE_PROFIT (take_profit), STOP_LOSS (stop_loss), TRAIL (trailing) stop
#
# Environment knobs (all optional with sane defaults):
#   EXCHANGE="kraken"
#   API_KEY, API_SECRET
#   DRY_RUN="true" | "false"
#   MAX_POSITIONS="6"
#   USD_PER_TRADE="15"
#   RESERVE_USD="80"
#   ROTATE_WHEN_CASH_SHORT="true"
#   ROTATE_MIN_EDGE_PCT="2.0"
#   COOLDOWN_RUNS="1"
#   SYMBOL_WHITELIST="BTC/USD,ETH/USD,SOL/USD,DOGE/USD,ZEC/USD,ENA/USD"
#
#   TAKE_PROFIT_PCT="3.5"     # TAKE_PROFIT threshold (+%)
#   STOP_LOSS_PCT="2.0"       # STOP_LOSS threshold (−%)
#   TRAIL_ARM_PCT="1.0"       # start tracking high watermark after +1.0% gain
#   TRAIL_PCT="1.5"           # TRAIL (trailing) distance from high watermark
#
# State files:
#   .state/rotation_cooldowns.json   (per-symbol buy cooldown runs left)
#   .state/entries.json              (symbol → entry price we track)
#   .state/highs.json                (symbol → high watermark since armed)
#
# Notes:
# - Entry price is recorded when we BUY. For existing holdings with no entry,
#   we initialize entry to current price so guards start from "now".
# - On TRAIL logic: once gain ≥ TRAIL_ARM_PCT from entry, we update a high watermark.
#   If price drops ≥ TRAIL_PCT from that high, we sell.

from __future__ import annotations
import os, json, time, math, pathlib, traceback
from typing import Dict, List, Tuple, Optional

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- Helpers for env parsing ----------

def as_bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def as_float(v: Optional[str], default: float) -> float:
    try:
        return float(v) if v is not None else default
    except:
        return default

def as_int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None else default
    except:
        return default

def env_list(v: Optional[str], default: List[str]) -> List[str]:
    if not v:
        return default
    items = []
    for x in v.split(","):
        s = x.strip()
        if s:
            items.append(s)
    return items or default

# ---------- ENV ----------

EXCHANGE_ID = os.getenv("EXCHANGE", "kraken").lower()
API_KEY     = os.getenv("API_KEY") or os.getenv("KRAKEN_API_KEY") or os.getenv("CCXT_API_KEY") or ""
API_SECRET  = os.getenv("API_SECRET") or os.getenv("KRAKEN_API_SECRET") or os.getenv("CCXT_API_SECRET") or ""

DRY_RUN     = as_bool(os.getenv("DRY_RUN"), True)
MAX_POS     = as_int(os.getenv("MAX_POSITIONS"), 6)
USD_PER_TRADE = as_float(os.getenv("USD_PER_TRADE"), 15.0)
RESERVE_USD = as_float(os.getenv("RESERVE_USD"), 80.0)

ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)
ROTATE_MIN_EDGE_PCT    = as_float(os.getenv("ROTATE_MIN_EDGE_PCT"), 2.0)
COOLDOWN_RUNS          = as_int(os.getenv("COOLDOWN_RUNS"), 1)

SYMBOL_WHITELIST = env_list(
    os.getenv("SYMBOL_WHITELIST"),
    ["BTC/USD","ETH/USD","SOL/USD","DOGE/USD","ZEC/USD","ENA/USD"]
)

# Sell guard knobs
TAKE_PROFIT_PCT = as_float(os.getenv("TAKE_PROFIT_PCT"), 3.5)   # TAKE_PROFIT
STOP_LOSS_PCT   = as_float(os.getenv("STOP_LOSS_PCT"), 2.0)     # STOP_LOSS
TRAIL_ARM_PCT   = as_float(os.getenv("TRAIL_ARM_PCT"), 1.0)     # arm trailing after +1%
TRAIL_PCT       = as_float(os.getenv("TRAIL_PCT"), 1.5)         # TRAIL distance

STATE_DIR = pathlib.Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
COOLDOWN_PATH = STATE_DIR / "rotation_cooldowns.json"
ENTRIES_PATH  = STATE_DIR / "entries.json"
HIGHS_PATH    = STATE_DIR / "highs.json"

# ---------- Exchange bootstrap ----------

def make_exchange() -> ccxt.Exchange:
    cls = getattr(ccxt, EXCHANGE_ID)
    ex = cls({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "adjustForTimeDifference": True
        }
    })
    return ex

# ---------- File state ----------

def load_json(path: pathlib.Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return default

def save_json(path: pathlib.Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)

def get_cooldowns() -> Dict[str, int]:
    return load_json(COOLDOWN_PATH, {})

def dec_cooldowns(cd: Dict[str, int]) -> Dict[str, int]:
    out = {}
    for k, v in cd.items():
        nv = max(0, int(v) - 1)
        if nv > 0:
            out[k] = nv
    return out

def get_entries() -> Dict[str, float]:
    return load_json(ENTRIES_PATH, {})

def get_highs() -> Dict[str, float]:
    return load_json(HIGHS_PATH, {})

# ---------- Balances & symbols ----------

USD_KEYS = ("USD","ZUSD")
STABLE_KEYS = ("USDT",)

def get_free_cash_usd(bal: Dict) -> float:
    total = 0.0
    for k in USD_KEYS:
        total += float(bal.get(k, {}).get("free", 0) or bal.get(k, 0) or 0)
    for k in STABLE_KEYS:
        total += float(bal.get(k, {}).get("free", 0) or bal.get(k, 0) or 0)
    return total

def canonical_symbol(exchange: ccxt.Exchange, base: str) -> Optional[str]:
    for quote in ("USD","USDT"):
        sym = f"{base}/{quote}"
        if sym in exchange.markets:
            return sym
    return None

def list_current_positions(exchange: ccxt.Exchange, bal: Dict) -> List[str]:
    held: List[str] = []
    for cur, obj in bal.items():
        if cur in USD_KEYS or cur in STABLE_KEYS:
            continue
        try:
            amt = float(obj.get("total", 0) if isinstance(obj, dict) else obj)
        except:
            amt = 0.0
        if amt and amt > 0:
            sym = canonical_symbol(exchange, cur)
            if sym:
                held.append(sym)
    return [s for s in dict.fromkeys(held) if s in exchange.markets]

# ---------- Price & scoring ----------

def last_price(exchange: ccxt.Exchange, symbol: str) -> float:
    t = exchange.fetch_ticker(symbol)
    return float(t["last"] or t["close"] or t["ask"] or 0)

def momentum_score_1h(exchange: ccxt.Exchange, symbol: str) -> float:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=3)
        if len(ohlcv) < 2:
            return 0.0
        open_prev = ohlcv[-2][1]
        close_last = ohlcv[-1][4]
        if open_prev <= 0:
            return 0.0
        return (close_last - open_prev) / open_prev * 100.0
    except Exception:
        return 0.0

def rank_symbols(exchange: ccxt.Exchange, symbols: List[str]) -> List[Tuple[str,float]]:
    scored = []
    for s in symbols:
        sc = momentum_score_1h(exchange, s)
        scored.append((s, sc))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

# ---------- Orders (with DRY_RUN shim) ----------

def place_sell(exchange: ccxt.Exchange, symbol: str, pct_of_position: float = 1.0) -> Tuple[bool, str]:
    try:
        bal = exchange.fetch_balance()
        base = symbol.split("/")[0]
        base_amt = 0.0
        if base in bal and isinstance(bal[base], dict):
            base_amt = float(bal[base].get("free", 0) or bal[base].get("total", 0) or 0)
        elif base in bal:
            try:
                base_amt = float(bal[base] or 0)
            except:
                base_amt = 0.0
        amt = base_amt * max(0.0, min(1.0, pct_of_position))
        if amt <= 0:
            return False, f"SELL skip {symbol} — no free amount"
        if DRY_RUN:
            return True, f"SELL ok {symbol} (simulated) amt={amt:.8f}"
        order = exchange.create_market_sell_order(symbol, amount=exchange.amount_to_precision(symbol, amt))
        return True, f"SELL ok {symbol} id={order.get('id','?')} amt={amt:.8f}"
    except Exception as e:
        return False, f"SELL fail {symbol}: {e}"

def place_buy(exchange: ccxt.Exchange, symbol: str, spend_usd: float) -> Tuple[bool, str]:
    try:
        if spend_usd <= 0:
            return False, f"BUY skip {symbol} — spend<=0"
        price = last_price(exchange, symbol)
        if price <= 0:
            return False, f"BUY skip {symbol} — price unknown"
        amount = spend_usd / price
        amount = float(exchange.amount_to_precision(symbol, amount))
        if amount <= 0:
            return False, f"BUY skip {symbol} — tiny amount"
        if DRY_RUN:
            return True, f"BUY ok {symbol} (simulated) spend=${spend_usd:.2f} amt={amount:.8f}"
        order = exchange.create_market_buy_order(symbol, amount=amount)
        return True, f"BUY ok {symbol} id={order.get('id','?')} spend=${spend_usd:.2f} amt={amount:.8f}"
    except Exception as e:
        return False, f"BUY fail {symbol}: {e}"

# ---------- Sell guards (TAKE_PROFIT / STOP_LOSS / TRAIL) ----------

def check_take_profit(symbol: str, entry: float, price: float) -> bool:
    # TAKE_PROFIT (take_profit)
    if entry > 0:
        change_pct = (price - entry) / entry * 100.0
        if change_pct >= TAKE_PROFIT_PCT:
            print(f"TAKE_PROFIT: {symbol} +{change_pct:.2f}% ≥ {TAKE_PROFIT_PCT:.2f}% → sell")
            return True
    return False

def check_stop_loss(symbol: str, entry: float, price: float) -> bool:
    # STOP_LOSS (stop_loss)
    if entry > 0:
        change_pct = (price - entry) / entry * 100.0
        if change_pct <= -abs(STOP_LOSS_PCT):
            print(f"STOP_LOSS: {symbol} {change_pct:.2f}% ≤ -{abs(STOP_LOSS_PCT):.2f}% → sell")
            return True
    return False

def check_trailing(symbol: str, entry: float, price: float, highs: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
    # TRAIL (trailing)
    # Arm trailing after gain ≥ TRAIL_ARM_PCT; then update high watermark; sell if drawdown ≥ TRAIL_PCT
    if entry <= 0:
        return False, highs
    gain_pct = (price - entry) / entry * 100.0
    hi = float(highs.get(symbol, 0.0) or 0.0)

    if gain_pct >= TRAIL_ARM_PCT:
        # update high watermark in absolute price terms
        if hi <= 0 or price > hi:
            highs[symbol] = price
            print(f"TRAIL arm/update: {symbol} armed at +{TRAIL_ARM_PCT:.2f}% (hi={price:.6f})")
        else:
            drawdown_pct = (hi - price) / hi * 100.0
            if drawdown_pct >= TRAIL_PCT:
                print(f"TRAIL: {symbol} drawdown {drawdown_pct:.2f}% ≥ {TRAIL_PCT:.2f}% from hi → sell")
                return True, highs
    return False, highs

# ---------- Main flow ----------

def main() -> None:
    print("============================================================")
    print("CRYPTO LIVE ▶ Cash-Short Rotation + Sell Guards (TP/SL/TRAIL)")
    if DRY_RUN:
        print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")
    print(f"Exchange={EXCHANGE_ID}  MaxPos={MAX_POS}  USD_PER_TRADE=${USD_PER_TRADE:.2f}  Reserve=${RESERVE_USD:.2f}")
    print(f"RotateWhenCashShort={ROTATE_WHEN_CASH_SHORT}  RotateEdge≥{ROTATE_MIN_EDGE_PCT:.2f}%  CooldownRuns={COOLDOWN_RUNS}")
    print(f"TAKE_PROFIT={TAKE_PROFIT_PCT:.2f}%  STOP_LOSS={STOP_LOSS_PCT:.2f}%  TRAIL arm={TRAIL_ARM_PCT:.2f}% dist={TRAIL_PCT:.2f}%")
    print("Whitelist:", ", ".join(SYMBOL_WHITELIST))
    print("============================================================")

    ex = make_exchange()
    ex.load_markets()

    cooldowns = dec_cooldowns(get_cooldowns())
    entries   = get_entries()
    highs     = get_highs()

    bal = ex.fetch_balance()
    free_cash = get_free_cash_usd(bal)
    held_syms = list_current_positions(ex, bal)
    print(f"Free cash ≈ ${free_cash:.2f} | Held positions: {len(held_syms)} → {', '.join(held_syms) if held_syms else '(none)'}")

    # Initialize entries for positions we discover without an entry yet
    for s in held_syms:
        if s not in entries:
            try:
                p = last_price(ex, s)
                if p > 0:
                    entries[s] = p
                    print(f"Init entry: {s} = {p:.6f}")
            except Exception:
                pass

    # --- SELL GUARDS pass over holdings ---
    for s in list(held_syms):
        try:
            price = last_price(ex, s)
            entry = float(entries.get(s, 0) or 0)
            if price <= 0 or entry <= 0:
                continue

            # Order: STOP_LOSS first, then TAKE_PROFIT, then TRAIL (you can reorder as desired)
            if check_stop_loss(s, entry, price) or check_take_profit(s, entry, price):
                ok, msg = place_sell(ex, s, pct_of_position=1.0)
                print(msg)
                if ok:
                    # Clear state for the sold symbol
                    entries.pop(s, None)
                    highs.pop(s, None)
                    # refresh balances/holdings
                    try:
                        bal = ex.fetch_balance()
                    except Exception:
                        pass
                    continue

            # trailing stop
            did_trail, highs = check_trailing(s, entry, price, highs)
            if did_trail:
                ok, msg = place_sell(ex, s, pct_of_position=1.0)
                print(msg)
                if ok:
                    entries.pop(s, None)
                    highs.pop(s, None)
                    try:
                        bal = ex.fetch_balance()
                    except Exception:
                        pass
                    continue
        except Exception as e:
            print(f"Guard error on {s}: {e}")

    # Recompute after potential sells (for rotation/topping)
    try:
        bal = ex.fetch_balance()
    except Exception:
        pass
    free_cash = get_free_cash_usd(bal)
    held_syms = list_current_positions(ex, bal)

    # Rank candidates (skip cooldown)
    cooled_whitelist = [s for s in SYMBOL_WHITELIST if cooldowns.get(s, 0) == 0]
    ranked_candidates = rank_symbols(ex, cooled_whitelist)
    best_symbol, best_score = (ranked_candidates[0] if ranked_candidates else (None, 0.0))

    # Rank holdings for worst
    ranked_holdings = rank_symbols(ex, held_syms) if held_syms else []
    worst_symbol, worst_score = (ranked_holdings[-1] if ranked_holdings else (None, 0.0))

    # Show rotation scan
    if best_symbol and worst_symbol:
        edge = best_score - worst_score
        print(f"ROTATE scan: best={best_symbol} {best_score:.1f}% vs worst={worst_symbol} {worst_score:.1f}% → edge={edge:.1f}%")
    elif best_symbol and not worst_symbol:
        print(f"ROTATE scan: best={best_symbol} {best_score:.1f}% (no current holdings to rotate)")
    else:
        print("ROTATE scan: (no candidate / no data)")

    # Cash-short rotation
    did_rotate = False
    if (
        ROTATE_WHEN_CASH_SHORT
        and free_cash < RESERVE_USD
        and best_symbol is not None
        and worst_symbol is not None
        and best_symbol != worst_symbol
    ):
        edge = best_score - worst_score
        if edge >= ROTATE_MIN_EDGE_PCT:
            ok_s, msg_s = place_sell(ex, worst_symbol, pct_of_position=1.0)
            print(msg_s)
            try:
                bal = ex.fetch_balance()
            except Exception:
                pass
            new_cash = get_free_cash_usd(bal)
            spend = min(USD_PER_TRADE, new_cash - max(0.0, RESERVE_USD - new_cash))
            if spend <= 0:
                spend = min(USD_PER_TRADE, new_cash)
            ok_b, msg_b = place_buy(ex, best_symbol, spend_usd=max(0.0, spend))
            print(msg_b)
            if ok_b:
                # record entry at executed/last price
                try:
                    p = last_price(ex, best_symbol)
                    if p > 0:
                        entries[best_symbol] = p
                        highs.pop(best_symbol, None)  # reset high until armed
                except Exception:
                    pass
                cooldowns[best_symbol] = max(cooldowns.get(best_symbol, 0), COOLDOWN_RUNS)
                did_rotate = True
                print(f"cooldown note: {best_symbol} rotation cooldown {COOLDOWN_RUNS} run(s)")
        else:
            print(f"ROTATE skip — edge {edge:.1f}% < {ROTATE_MIN_EDGE_PCT:.1f}%")

    # New entry if below cap and cash ≥ reserve + per-trade
    if not did_rotate and best_symbol:
        if len(held_syms) < MAX_POS and free_cash >= (RESERVE_USD + USD_PER_TRADE):
            ok_b, msg_b = place_buy(ex, best_symbol, spend_usd=USD_PER_TRADE)
            print(msg_b)
            if ok_b:
                try:
                    p = last_price(ex, best_symbol)
                    if p > 0:
                        entries[best_symbol] = p
                        highs.pop(best_symbol, None)
                except Exception:
                    pass
                cooldowns[best_symbol] = max(cooldowns.get(best_symbol, 0), COOLDOWN_RUNS)

    # Save state
    save_json(COOLDOWN_PATH, cooldowns)
    save_json(ENTRIES_PATH, entries)
    save_json(HIGHS_PATH, highs)

    print("DONE.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        raise
