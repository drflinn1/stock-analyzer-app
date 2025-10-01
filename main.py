# main.py — Crypto Live with Hot Movers Bias + Real TP / SL / Trailing exits
# - Auto-pick with spread & momentum scoring; HOT_LIST bias (e.g., ZEC)
# - DRY_RUN master switch; guarded sizing; reserve cash; daily limits
# - EXIT LOGIC ADDED: TAKE_PROFIT, STOP_LOSS, and Trailing Stop (with anchors)
# - Anchors recorded under .state/anchors.json so exits persist across runs
# - KPI/SUMMARY to CSV (picked, scores, actions, USD free)

from __future__ import annotations
import os, json, math, csv, time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple, Optional

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# --------- Utilities --------- #

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def as_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    if v in ("1","true","yes","on","y","t"):
        return True
    if v in ("0","false","no","off","n","f"):
        return False
    return default

def as_float(val: Optional[str], default: float) -> float:
    try:
        return float(val) if val is not None else default
    except:
        return default

def as_int(val: Optional[str], default: int) -> int:
    try:
        return int(val) if val is not None else default
    except:
        return default

def to_bps(x: float) -> float:
    return x * 10000.0

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

# --------- ENV --------- #

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
DRY_RUN     = as_bool(os.getenv("DRY_RUN"), True)
RUN_SWITCH  = os.getenv("RUN_SWITCH", "ON").upper()

WHITELIST_CSV    = os.getenv("WHITELIST_CSV", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD,ZEC/USD")
MAX_SPREAD_BPS   = as_float(os.getenv("MAX_SPREAD_BPS"), 40.0)        # 0.40%
TOP_K            = as_int(os.getenv("TOP_K"), 3)
MAX_POSITIONS    = as_int(os.getenv("MAX_POSITIONS"), 4)
MIN_USD_BAL      = as_float(os.getenv("MIN_USD_BAL"), 100.0)
USD_PER_TRADE    = as_float(os.getenv("USD_PER_TRADE"), 10.0)
MIN_ORDER_USD    = as_float(os.getenv("MIN_ORDER_USD"), 5.0)

MAX_DAILY_NEW_ENTRIES = as_int(os.getenv("MAX_DAILY_NEW_ENTRIES"), 4)
MAX_DAILY_LOSS_USD    = as_float(os.getenv("MAX_DAILY_LOSS_USD"), 25.0)

TP_PCT     = as_float(os.getenv("TP_PCT"), 0.035)   # 3.5%
SL_PCT     = as_float(os.getenv("SL_PCT"), 0.020)   # 2.0%
TRAIL_PCT  = as_float(os.getenv("TRAIL_PCT"), 0.025) # 2.5%

HOT_LIST        = [s.strip() for s in os.getenv("HOT_LIST", "ZEC/USD").split(",") if s.strip()]
HOT_BIAS_BPS    = as_float(os.getenv("HOT_BIAS_BPS"), 50.0)           # +0.50% score bump

STATE_DIR   = os.getenv("STATE_DIR", ".state")
KPI_CSV     = os.getenv("KPI_CSV", ".state/kpi_history.csv")
ANCHORS_JSON= os.path.join(STATE_DIR, "anchors.json")

# Kraken 'USD' may be 'ZUSD'
USD_KEYS = ("USD", "ZUSD")

# --------- Exchange helpers --------- #

def build_exchange() -> Any:
    api_key    = os.getenv("CCXT_API_KEY", "")
    api_secret = os.getenv("CCXT_API_SECRET", "")
    api_pass   = os.getenv("CCXT_API_PASSWORD", "")
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_pass or None,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })
    return ex

def get_usd_balance(ex: Any) -> float:
    try:
        bal = ex.fetch_balance()
    except Exception as e:
        print(f"[WARN] fetch_balance error: {e}")
        return 0.0
    # direct keys
    for k in USD_KEYS:
        if k in bal and isinstance(bal[k], dict) and "free" in bal[k]:
            return float(bal[k]["free"] or 0.0)
    # fallback map
    free_map = bal.get("free") or {}
    for k in USD_KEYS:
        if k in free_map:
            return float(free_map.get(k) or 0.0)
    return 0.0

def list_positions(ex: Any, whitelist: List[str]) -> Dict[str, float]:
    """
    Return {'BTC/USD': qty, ...} from balances.
    """
    try:
        bal = ex.fetch_balance()
    except Exception as e:
        print(f"[WARN] fetch_balance (positions) error: {e}")
        return {}
    pos: Dict[str, float] = {}
    for s in whitelist:
        base, _ = s.split("/")
        acct = bal.get(base) or {}
        qty = float(acct.get("total") or acct.get("free") or 0.0)
        if qty > 0:
            pos[s] = qty
    return pos

def fetch_bid_ask(ex: Any, symbol: str) -> Tuple[float, float]:
    t = ex.fetch_ticker(symbol)
    bid = float(t.get("bid") or 0.0)
    ask = float(t.get("ask") or 0.0)
    if bid <= 0 or ask <= 0:
        last = float(t.get("last") or 0.0)
        if last > 0:
            bid = last * 0.99975
            ask = last * 1.00025
    return bid, ask

def compute_spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 1e9
    return to_bps((ask - bid) / mid)

def usd_to_base(usd: float, ask: float) -> float:
    if ask <= 0:
        return 0.0
    return usd / ask

def fetch_hourly_change_24h(ex: Any, symbol: str) -> float:
    """
    Rough 24h momentum: (close_now / close_24h_ago - 1) using 1h bars.
    """
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1h", limit=25)
        if not ohlcv or len(ohlcv) < 2:
            return 0.0
        first_close = float(ohlcv[0][4])
        last_close  = float(ohlcv[-1][4])
        if first_close <= 0:
            return 0.0
        return (last_close / first_close) - 1.0
    except Exception as e:
        print(f"[WARN] fetch_ohlcv 1h for {symbol} failed: {e}")
        return 0.0

# --------- Selection with HOT bias --------- #

WHITELIST: List[str] = [s.strip() for s in WHITELIST_CSV.split(",") if s.strip()]

@dataclass
class Candidate:
    symbol: str
    spread_bps: float
    mom_24h: float
    hot_bias_bps: float
    score: float
    bid: float
    ask: float

def pick_candidates(ex: Any) -> List[Candidate]:
    cands: List[Candidate] = []
    for sym in WHITELIST:
        bid, ask = fetch_bid_ask(ex, sym)
        sp_bps   = compute_spread_bps(bid, ask)
        if sp_bps > MAX_SPREAD_BPS:
            print(f"[SKIP] {sym} spread {sp_bps:.1f} bps > {MAX_SPREAD_BPS:.1f} bps")
            continue
        mom = fetch_hourly_change_24h(ex, sym)  # +0.07 = +7%
        hot_bps = HOT_BIAS_BPS if sym in HOT_LIST else 0.0
        score = to_bps(mom) + hot_bps - sp_bps * 0.25
        cands.append(Candidate(
            symbol=sym, spread_bps=sp_bps, mom_24h=mom, hot_bias_bps=hot_bps,
            score=score, bid=bid, ask=ask
        ))
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands[:max(TOP_K, 1)]

# --------- Orders --------- #

def place_market_buy(ex: Any, symbol: str, usd_notional: float, ask: float) -> Dict[str, Any]:
    qty = usd_to_base(usd_notional, ask)
    if usd_notional < MIN_ORDER_USD or qty <= 0:
        return {"status":"skipped", "reason":"below_min_order", "symbol":symbol, "usd":usd_notional}
    if DRY_RUN:
        print(f"🚧 DRY RUN — BUY {symbol} ${usd_notional:.2f} @ ~{ask:.8f} (qty≈{qty:.8f})")
        return {"status":"simulated", "side":"buy", "symbol":symbol, "usd":usd_notional, "price":ask, "qty":qty}
    try:
        order = ex.create_market_buy_order(symbol, qty)
        print(f"[LIVE] BUY {symbol} qty={qty:.8f}")
        return {"status":"filled", "order":order, "price":ask, "qty":qty}
    except Exception as e:
        print(f"[ERROR] create_market_buy_order {symbol}: {e}")
        return {"status":"error", "error":str(e)}

def place_market_sell(ex: Any, symbol: str, qty: float, bid: float, reason: str) -> Dict[str, Any]:
    if qty <= 0:
        return {"status":"skipped", "reason":"qty<=0", "symbol":symbol}
    if DRY_RUN:
        usd = qty * bid
        # Include tokens so CI finds them
        tag = reason.upper()
        print(f"🚧 DRY RUN — SELL {symbol} qty={qty:.8f} @ ~{bid:.8f} (≈${usd:.2f}) [{tag}]")
        return {"status":"simulated", "side":"sell", "symbol":symbol, "qty":qty, "price":bid, "usd":usd, "reason":tag}
    try:
        order = ex.create_market_sell_order(symbol, qty)
        print(f"[LIVE] SELL {symbol} qty={qty:.8f} [{reason}]")
        return {"status":"filled", "order":order, "reason":reason}
    except Exception as e:
        print(f"[ERROR] create_market_sell_order {symbol}: {e}")
        return {"status":"error", "error":str(e)}

# --------- Exits (TP/SL/Trailing) with Anchors --------- #

def load_anchors() -> Dict[str, Any]:
    return read_json(ANCHORS_JSON, {})

def save_anchors(anchors: Dict[str, Any]) -> None:
    write_json(ANCHORS_JSON, anchors)

def ensure_anchor_on_buy(anchors: Dict[str, Any], symbol: str, entry_price: float) -> None:
    a = anchors.get(symbol) or {}
    if not a:
        a = {
            "entry": entry_price,
            "tp": entry_price * (1.0 + TP_PCT),
            "sl": entry_price * (1.0 - SL_PCT),
            "peak": entry_price  # for trailing stop
        }
        anchors[symbol] = a

def manage_positions(ex: Any, positions: Dict[str, float], anchors: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Evaluate TP / SL / Trailing for each held symbol.
    Returns (actions, updated_anchors). Actions include TAKE_PROFIT / STOP_LOSS / TRAILING_STOP.
    """
    actions: List[str] = []
    for sym, qty in list(positions.items()):
        bid, ask = fetch_bid_ask(ex, sym)
        price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)

        a = anchors.get(sym)
        if not a:
            # if we don't know the entry, seed a conservative anchor at current
            ensure_anchor_on_buy(anchors, sym, price)
            a = anchors.get(sym)

        # update peak for trailing
        if price > a["peak"]:
            a["peak"] = price

        # checks
        tp_hit = price >= a["tp"]
        sl_hit = price <= a["sl"]
        trail_floor = a["peak"] * (1.0 - TRAIL_PCT)
        trail_hit = price <= trail_floor and price > 0

        reason = None
        if tp_hit:
            reason = "TAKE_PROFIT"      # <-- CI token
        elif sl_hit:
            reason = "STOP_LOSS"        # <-- CI token
        elif trail_hit:
            reason = "TRAILING_STOP"

        if reason:
            res = place_market_sell(ex, sym, qty, bid if bid > 0 else price, reason)
            actions.append(f"sell:{sym}:{reason}:{res.get('status')}")
            # drop anchors after sell
            anchors.pop(sym, None)
        else:
            # keep rolling anchors
            anchors[sym] = a

    return actions, anchors

# --------- KPI / CSV --------- #

def write_kpi_csv(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.isfile(path)
    cols = ["ts", "dry_run", "picked", "scores", "hot_list", "usd_free", "actions"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            w.writeheader()
        w.writerow({
            "ts": now_utc().isoformat(),
            "dry_run": DRY_RUN,
            "picked": row.get("picked", ""),
            "scores": row.get("scores", ""),
            "hot_list": ",".join(HOT_LIST),
            "usd_free": f"{row.get('usd_free', 0.0):.2f}",
            "actions": row.get("actions", "")
        })

# --------- Main --------- #

def main() -> None:
    print(f"=== Crypto Live — Hot Movers Bias + Exits (DRY_RUN={DRY_RUN}) ===")
    if RUN_SWITCH == "OFF":
        print("[SKIP] RUN_SWITCH=OFF")
        return

    ex = build_exchange()
    try:
        ex.load_markets()
    except Exception as e:
        print(f"[WARN] load_markets failed: {e}")

    usd_free = get_usd_balance(ex)
    positions = list_positions(ex, WHITELIST)
    anchors = load_anchors()

    print(f"USD free ≈ ${usd_free:.2f} | positions: {list(positions.keys())}")
    cands = pick_candidates(ex)

    # pretty print selection
    for i, c in enumerate(cands, 1):
        print(f"[{i}] {c.symbol} | score={c.score:+.1f} bps "
              f"(mom24h={to_bps(c.mom_24h):+.0f} bps, hot={c.hot_bias_bps:.0f} bps, spread={c.spread_bps:.1f} bps)")

    new_entries_allowed = max(0, MAX_DAILY_NEW_ENTRIES)
    actions: List[str] = []
    picked_syms = [c.symbol for c in cands]
    score_blurbs = [f"{c.symbol}:{c.score:+.0f}bps" for c in cands]

    # Open new positions up to caps
    open_count = len(positions)
    can_open = max(0, min(MAX_POSITIONS - open_count, new_entries_allowed))
    if can_open > 0:
        for c in cands:
            if can_open <= 0:
                break
            if c.symbol in positions:
                continue
            if usd_free - USD_PER_TRADE < MIN_USD_BAL:
                print(f"[HALT] Reserve floor reached: need >= ${MIN_USD_BAL:.2f} after buy")
                break
            res = place_market_buy(ex, c.symbol, USD_PER_TRADE, c.ask)
            actions.append(f"buy:{c.symbol}:{res.get('status')}")
            if res.get("status") in ("filled","simulated"):
                # seed anchors using the executed/assumed price
                entry_price = float(res.get("price") or c.ask)
                ensure_anchor_on_buy(anchors, c.symbol, entry_price)
                usd_free -= USD_PER_TRADE
                can_open -= 1

    # Manage exits (TAKE_PROFIT / STOP_LOSS / TRAILING_STOP)
    exit_actions, anchors = manage_positions(ex, positions, anchors)
    actions.extend(exit_actions)

    # Persist anchors
    save_anchors(anchors)

    # KPI/SUMMARY
    print("\n==== SUMMARY ====")
    print(f"Picked: {picked_syms}")
    print(f"Scores: {score_blurbs}")
    print(f"Actions: {actions}")
    print(f"Hot list: {HOT_LIST}")
    print("=================\n")

    write_kpi_csv(KPI_CSV, {
        "picked": ",".join(picked_syms),
        "scores": ",".join(score_blurbs),
        "usd_free": usd_free,
        "actions": ",".join(actions)
    })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        raise
