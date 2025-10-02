# main.py — Crypto Live with Cash-Short Rotation
# - DRY_RUN banner + safe order shims
# - USD reserve guard (RESERVE_USD)
# - Auto-pick best symbol by momentum score (1h change)
# - When cash < reserve AND ROTATE_WHEN_CASH_SHORT=true:
#     Find worst current holding vs best candidate → if edge ≥ ROTATE_MIN_EDGE_PCT,
#     SELL worst … BUY best, and note a 1-run cooldown.
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
# State files:
#   .state/rotation_cooldowns.json   (tracks per-symbol cooldown runs left)
#
# Notes:
# - Momentum score uses last two 1h candles: pct = (close[-1] - open[-2]) / open[-2] * 100
# - We try '/USD' first, then '/USDT' as a fallback for pricing & trading.
# - Kraken fiat can be quirky; we check USD, ZUSD, and USDT balances.

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

STATE_DIR = pathlib.Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
COOLDOWN_PATH = STATE_DIR / "rotation_cooldowns.json"

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
    # Prefer /USD then /USDT if present in markets
    for quote in ("USD","USDT"):
        sym = f"{base}/{quote}"
        if sym in exchange.markets:
            return sym
    return None

def list_current_positions(exchange: ccxt.Exchange, bal: Dict) -> List[str]:
    held: List[str] = []
    # balances in CCXT may be dict[str]-> {'free','used','total'} or float
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
    # Deduplicate, keep only symbols the exchange supports
    held = [s for s in dict.fromkeys(held) if s in exchange.markets]
    return held

# ---------- Scoring (momentum) ----------

def momentum_score_1h(exchange: ccxt.Exchange, symbol: str) -> float:
    """
    Score = percent change using last two 1h candles:
    (close[-1] - open[-2]) / open[-2] * 100
    """
    try:
        # Fetch 3 recent 1h candles to be safe
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=3)
        if len(ohlcv) < 2:
            return 0.0
        # Make sure we have two consecutive candles
        # Use previous candle open and last candle close
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
    # sort descending by score
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

# ---------- Orders (with DRY_RUN shim) ----------

def place_sell(exchange: ccxt.Exchange, symbol: str, pct_of_position: float = 1.0) -> Tuple[bool, str]:
    """
    Market sell up to 100% of position; returns (ok, msg).
    """
    try:
        bal = exchange.fetch_balance()
        base = symbol.split("/")[0]
        # Amount available in base currency
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
    """
    Market buy by quote value (USD/USDT). For KRaken via CCXT, we convert value→amount using ticker price.
    """
    try:
        if spend_usd <= 0:
            return False, f"BUY skip {symbol} — spend<=0"
        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker["last"] or ticker["close"] or ticker["ask"] or 0)
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

# ---------- Main flow ----------

def main() -> None:
    print("============================================================")
    print("CRYPTO LIVE ▶ with Cash-Short Rotation")
    if DRY_RUN:
        print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")
    print(f"Exchange={EXCHANGE_ID}  MaxPos={MAX_POS}  USD_PER_TRADE=${USD_PER_TRADE:.2f}  Reserve=${RESERVE_USD:.2f}")
    print(f"RotateWhenCashShort={ROTATE_WHEN_CASH_SHORT}  RotateEdge≥{ROTATE_MIN_EDGE_PCT:.2f}%  CooldownRuns={COOLDOWN_RUNS}")
    print("Whitelist:", ", ".join(SYMBOL_WHITELIST))
    print("============================================================")

    ex = make_exchange()

    # Load markets once
    ex.load_markets()

    # Cooldowns
    cooldowns = dec_cooldowns(get_cooldowns())

    # Balances & holdings
    bal = ex.fetch_balance()
    free_cash = get_free_cash_usd(bal)
    held_syms = list_current_positions(ex, bal)

    print(f"Free cash ≈ ${free_cash:.2f} | Held positions: {len(held_syms)} → {', '.join(held_syms) if held_syms else '(none)'}")

    # Rank universe candidates (respect cooldown)
    # Remove any symbol that is currently cooling down from candidate buys
    cooled_whitelist = [s for s in SYMBOL_WHITELIST if cooldowns.get(s, 0) == 0]
    ranked_candidates = rank_symbols(ex, cooled_whitelist)
    best_symbol, best_score = (ranked_candidates[0] if ranked_candidates else (None, 0.0))

    # Rank existing holdings (even if not in whitelist), to find the worst
    ranked_holdings = rank_symbols(ex, held_syms) if held_syms else []
    worst_symbol, worst_score = (ranked_holdings[-1] if ranked_holdings else (None, 0.0))

    # Show ROTATE scan line when relevant
    if best_symbol and worst_symbol:
        edge = best_score - worst_score
        print(f"ROTATE scan: best={best_symbol} {best_score:.1f}% vs worst={worst_symbol} {worst_score:.1f}% → edge={edge:.1f}%")
    elif best_symbol and not worst_symbol:
        print(f"ROTATE scan: best={best_symbol} {best_score:.1f}% (no current holdings to rotate)")
    else:
        print("ROTATE scan: (no candidate / no data)")

    # If cash is short and we have something to rotate, consider SELL worst → BUY best
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
            # SELL worst fully, then BUY best for the freed cash (or USD_PER_TRADE, whichever makes sense)
            ok_s, msg_s = place_sell(ex, worst_symbol, pct_of_position=1.0)
            print(msg_s)
            # Refresh balances after sell (simulated or real)
            try:
                bal = ex.fetch_balance()
            except Exception:
                pass
            new_cash = get_free_cash_usd(bal)
            spend = min(USD_PER_TRADE, new_cash - max(0.0, RESERVE_USD - new_cash))  # be cautious
            # If that heuristic yields <=0, just spend up to USD_PER_TRADE capped by available
            if spend <= 0:
                spend = min(USD_PER_TRADE, new_cash)
            ok_b, msg_b = place_buy(ex, best_symbol, spend_usd=max(0.0, spend))
            print(msg_b)
            if ok_b:
                cooldowns[best_symbol] = max(cooldowns.get(best_symbol, 0), COOLDOWN_RUNS)
                did_rotate = True
                print(f"cooldown note: {best_symbol} rotation cooldown {COOLDOWN_RUNS} run(s)")
        else:
            print(f"ROTATE skip — edge {edge:.1f}% < {ROTATE_MIN_EDGE_PCT:.1f}%")

    # If not rotating, consider topping up (new entry) if below MAX_POS and cash≥reserve
    # (Conservative: only buy when we have adequate cash.)
    if not did_rotate and best_symbol:
        if len(held_syms) < MAX_POS and free_cash >= (RESERVE_USD + USD_PER_TRADE):
            ok_b, msg_b = place_buy(ex, best_symbol, spend_usd=USD_PER_TRADE)
            print(msg_b)
            if ok_b:
                cooldowns[best_symbol] = max(cooldowns.get(best_symbol, 0), COOLDOWN_RUNS)

    # Save cooldowns
    save_json(COOLDOWN_PATH, cooldowns)

    print("DONE.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        raise
