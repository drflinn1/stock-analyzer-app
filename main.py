# main.py — Crypto Live with Core + Speculative tiers
# - CORE: your usual coins w/ HOT bias (e.g., ZEC), spread cap, TP/SL/Trail
# - SPEC: opt-in sandbox for pumps (e.g., SPX/PENGU/PUMP) with strict gates:
#   * require strong 24h USD volume and positive 15m momentum
#   * looser spread cap than CORE but still bounded
#   * tiny size, max 1 spec position, max 1 new spec entry/day
#   * tighter SL/Trail and fast-drop kill within first N minutes
#
# Anchors (entry/tp/sl/peak/tier/opened_at) are persisted in .state/anchors.json.
# Logs include TAKE_PROFIT / STOP_LOSS / TRAILING_STOP tokens for CI.

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
    if v in ("1","true","yes","on","y","t"): return True
    if v in ("0","false","no","off","n","f"): return False
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

# CORE
WHITELIST_CSV    = os.getenv("WHITELIST_CSV", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD,ZEC/USD")
MAX_SPREAD_BPS   = as_float(os.getenv("MAX_SPREAD_BPS"), 40.0)
TOP_K            = as_int(os.getenv("TOP_K"), 3)
MAX_POSITIONS    = as_int(os.getenv("MAX_POSITIONS"), 4)
MIN_USD_BAL      = as_float(os.getenv("MIN_USD_BAL"), 100.0)
USD_PER_TRADE    = as_float(os.getenv("USD_PER_TRADE"), 10.0)
MIN_ORDER_USD    = as_float(os.getenv("MIN_ORDER_USD"), 5.0)

MAX_DAILY_NEW_ENTRIES = as_int(os.getenv("MAX_DAILY_NEW_ENTRIES"), 4)
MAX_DAILY_LOSS_USD    = as_float(os.getenv("MAX_DAILY_LOSS_USD"), 25.0)

TP_PCT     = as_float(os.getenv("TP_PCT"), 0.035)
SL_PCT     = as_float(os.getenv("SL_PCT"), 0.020)
TRAIL_PCT  = as_float(os.getenv("TRAIL_PCT"), 0.025)

HOT_LIST        = [s.strip() for s in os.getenv("HOT_LIST", "ZEC/USD").split(",") if s.strip()]
HOT_BIAS_BPS    = as_float(os.getenv("HOT_BIAS_BPS"), 50.0)

# SPECULATIVE sandbox
SPEC_ENABLE                 = as_bool(os.getenv("SPEC_ENABLE"), False)
SPEC_LIST                   = [s.strip() for s in os.getenv("SPEC_LIST", "").split(",") if s.strip()]
SPEC_MAX_POSITIONS          = as_int(os.getenv("SPEC_MAX_POSITIONS"), 1)
SPEC_MAX_DAILY_NEW_ENTRIES  = as_int(os.getenv("SPEC_MAX_DAILY_NEW_ENTRIES"), 1)
SPEC_USD_PER_TRADE          = as_float(os.getenv("SPEC_USD_PER_TRADE"), 5.0)
SPEC_MAX_SPREAD_BPS         = as_float(os.getenv("SPEC_MAX_SPREAD_BPS"), 120.0)
SPEC_MIN_VOL24H_USD         = as_float(os.getenv("SPEC_MIN_VOL24H_USD"), 3_000_000.0)
SPEC_MIN_MOM24H_BPS         = as_float(os.getenv("SPEC_MIN_MOM24H_BPS"), 1000.0)   # +10%
SPEC_REQUIRE_MOM15M_POS     = as_bool(os.getenv("SPEC_REQUIRE_MOM15M_POS"), True)
SPEC_TP_PCT                 = as_float(os.getenv("SPEC_TP_PCT"), 0.050)
SPEC_SL_PCT                 = as_float(os.getenv("SPEC_SL_PCT"), 0.030)
SPEC_TRAIL_PCT              = as_float(os.getenv("SPEC_TRAIL_PCT"), 0.030)
SPEC_FAST_DROP_PCT          = as_float(os.getenv("SPEC_FAST_DROP_PCT"), 0.040)
SPEC_FAST_DROP_WINDOW_MIN   = as_int(os.getenv("SPEC_FAST_DROP_WINDOW_MIN"), 60)

STATE_DIR   = os.getenv("STATE_DIR", ".state")
KPI_CSV     = os.getenv("KPI_CSV", ".state/kpi_history.csv")
ANCHORS_JSON= os.path.join(STATE_DIR, "anchors.json")

USD_KEYS = ("USD", "ZUSD")  # Kraken sometimes uses ZUSD

# --------- Exchange helpers --------- #

def build_exchange() -> Any:
    api_key    = os.getenv("CCXT_API_KEY", "")
    api_secret = os.getenv("CCXT_API_SECRET", "")
    api_pass   = os.getenv("CCXT_API_PASSWORD", "")
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "apiKey": api_key, "secret": api_secret, "password": api_pass or None,
        "enableRateLimit": True, "options": {"adjustForTimeDifference": True}
    })
    return ex

def get_usd_balance(ex: Any) -> float:
    try:
        bal = ex.fetch_balance()
    except Exception as e:
        print(f"[WARN] fetch_balance error: {e}")
        return 0.0
    for k in USD_KEYS:
        if k in bal and isinstance(bal[k], dict) and "free" in bal[k]:
            return float(bal[k]["free"] or 0.0)
    free_map = bal.get("free") or {}
    for k in USD_KEYS:
        if k in free_map:
            return float(free_map.get(k) or 0.0)
    return 0.0

def list_positions(ex: Any, all_symbols: List[str]) -> Dict[str, float]:
    try:
        bal = ex.fetch_balance()
    except Exception as e:
        print(f"[WARN] fetch_balance (positions) error: {e}")
        return {}
    pos: Dict[str, float] = {}
    for s in all_symbols:
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
    if mid <= 0: return 1e9
    return to_bps((ask - bid) / mid)

def usd_to_base(usd: float, ask: float) -> float:
    if ask <= 0: return 0.0
    return usd / ask

def fetch_ohlcv_change(ex: Any, symbol: str, timeframe: str, bars: int) -> float:
    """
    Returns (last_close / first_close - 1) over 'bars' bars for given timeframe.
    """
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=bars)
        if not ohlcv or len(ohlcv) < 2: return 0.0
        first_close = float(ohlcv[0][4]); last_close = float(ohlcv[-1][4])
        if first_close <= 0: return 0.0
        return (last_close / first_close) - 1.0
    except Exception as e:
        print(f"[WARN] fetch_ohlcv {timeframe} for {symbol} failed: {e}")
        return 0.0

def estimate_24h_quote_volume_usd(ex: Any, symbol: str) -> float:
    """
    Approximate 24h quote volume in USD using 24 x 1h bars: sum(volume_base * close_price).
    """
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1h", limit=24)
        if not ohlcv: return 0.0
        total = 0.0
        for ts, o, h, l, c, v in ohlcv:
            total += float(v) * float(c)
        return total
    except Exception as e:
        print(f"[WARN] vol24h {symbol}: {e}")
        return 0.0

# --------- Selection with tiers --------- #

CORE_LIST: List[str] = [s.strip() for s in WHITELIST_CSV.split(",") if s.strip()]
SPEC_LIST = SPEC_LIST if SPEC_ENABLE else []

@dataclass
class Candidate:
    symbol: str
    tier: str  # "core" or "spec"
    spread_bps: float
    mom_24h: float
    mom_15m: float
    vol24h_usd: float
    hot_bias_bps: float
    score: float
    bid: float
    ask: float

def core_candidates(ex: Any) -> List[Candidate]:
    cands: List[Candidate] = []
    for sym in CORE_LIST:
        bid, ask = fetch_bid_ask(ex, sym)
        sp_bps   = compute_spread_bps(bid, ask)
        if sp_bps > MAX_SPREAD_BPS:
            print(f"[SKIP CORE] {sym} spread {sp_bps:.1f} bps > {MAX_SPREAD_BPS:.1f} bps")
            continue
        mom24 = fetch_ohlcv_change(ex, sym, "1h", 25)
        mom15 = fetch_ohlcv_change(ex, sym, "5m", 4)  # ~15m
        hot_bps = HOT_BIAS_BPS if sym in HOT_LIST else 0.0
        score = to_bps(mom24) + hot_bps - sp_bps * 0.25
        cands.append(Candidate(sym, "core", sp_bps, mom24, mom15, 0.0, hot_bps, score, bid, ask))
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands[:max(TOP_K, 1)]

def spec_candidates(ex: Any) -> List[Candidate]:
    if not SPEC_ENABLE or not SPEC_LIST:
        return []
    cands: List[Candidate] = []
    for sym in SPEC_LIST:
        bid, ask = fetch_bid_ask(ex, sym)
        sp_bps   = compute_spread_bps(bid, ask)
        if sp_bps > SPEC_MAX_SPREAD_BPS:
            print(f"[SKIP SPEC] {sym} spread {sp_bps:.1f} bps > {SPEC_MAX_SPREAD_BPS:.1f} bps")
            continue
        vol24 = estimate_24h_quote_volume_usd(ex, sym)
        if vol24 < SPEC_MIN_VOL24H_USD:
            print(f"[SKIP SPEC] {sym} vol24h ${vol24:,.0f} < ${SPEC_MIN_VOL24H_USD:,.0f}")
            continue
        mom24 = fetch_ohlcv_change(ex, sym, "1h", 25)
        if to_bps(mom24) < SPEC_MIN_MOM24H_BPS:
            print(f"[SKIP SPEC] {sym} mom24h {to_bps(mom24):.0f}bps < {SPEC_MIN_MOM24H_BPS:.0f}bps")
            continue
        mom15 = fetch_ohlcv_change(ex, sym, "5m", 4)
        if SPEC_REQUIRE_MOM15M_POS and mom15 <= 0:
            print(f"[SKIP SPEC] {sym} mom15m {to_bps(mom15):.0f}bps <= 0bps")
            continue
        # Spec score: overweight short-term momentum, underweight spread penalty
        score = to_bps(mom24) * 0.7 + to_bps(mom15) * 0.6 - sp_bps * 0.15
        cands.append(Candidate(sym, "spec", sp_bps, mom24, mom15, vol24, 0.0, score, bid, ask))
    cands.sort(key=lambda x: x.score, reverse=True)
    # only consider top 1–2 spec names
    return cands[: min(2, len(cands))]

# --------- Orders --------- #

def place_market_buy(ex: Any, symbol: str, usd_notional: float, ask: float, tier: str) -> Dict[str, Any]:
    qty = usd_to_base(usd_notional, ask)
    if usd_notional < MIN_ORDER_USD or qty <= 0:
        return {"status":"skipped", "reason":"below_min_order", "symbol":symbol, "usd":usd_notional}
    if DRY_RUN:
        print(f"🚧 DRY RUN — BUY {symbol} ${usd_notional:.2f} @ ~{ask:.8f} (qty≈{qty:.8f}) [{tier}]")
        return {"status":"simulated", "side":"buy", "symbol":symbol, "usd":usd_notional, "price":ask, "qty":qty, "tier":tier}
    try:
        order = ex.create_market_buy_order(symbol, qty)
        print(f"[LIVE] BUY {symbol} qty={qty:.8f} [{tier}]")
        return {"status":"filled", "order":order, "price":ask, "qty":qty, "tier":tier}
    except Exception as e:
        print(f"[ERROR] create_market_buy_order {symbol}: {e}")
        return {"status":"error", "error":str(e), "tier":tier}

def place_market_sell(ex: Any, symbol: str, qty: float, bid: float, reason: str, tier: str) -> Dict[str, Any]:
    if qty <= 0:
        return {"status":"skipped", "reason":"qty<=0", "symbol":symbol}
    if DRY_RUN:
        usd = qty * (bid if bid > 0 else 0.0)
        tag = reason.upper()
        print(f"🚧 DRY RUN — SELL {symbol} qty={qty:.8f} @ ~{bid:.8f} (≈${usd:.2f}) [{tag}|{tier}]")
        return {"status":"simulated", "side":"sell", "symbol":symbol, "qty":qty, "price":bid, "usd":usd, "reason":tag, "tier":tier}
    try:
        order = ex.create_market_sell_order(symbol, qty)
        print(f"[LIVE] SELL {symbol} qty={qty:.8f} [{reason}|{tier}]")
        return {"status":"filled", "order":order, "reason":reason, "tier":tier}
    except Exception as e:
        print(f"[ERROR] create_market_sell_order {symbol}: {e}")
        return {"status":"error", "error":str(e), "tier":tier}

# --------- Anchors / exits --------- #

def load_anchors() -> Dict[str, Any]:
    return read_json(ANCHORS_JSON, {})

def save_anchors(anchors: Dict[str, Any]) -> None:
    write_json(ANCHORS_JSON, anchors)

def tier_params(tier: str) -> Tuple[float,float,float]:
    if tier == "spec":
        return (SPEC_TP_PCT, SPEC_SL_PCT, SPEC_TRAIL_PCT)
    return (TP_PCT, SL_PCT, TRAIL_PCT)

def ensure_anchor_on_buy(anchors: Dict[str, Any], symbol: str, entry_price: float, tier: str) -> None:
    if symbol in anchors:  # don't overwrite
        return
    tp, sl, tr = tier_params(tier)
    anchors[symbol] = {
        "tier": tier,
        "entry": entry_price,
        "tp": entry_price * (1.0 + tp),
        "sl": entry_price * (1.0 - sl),
        "peak": entry_price,          # for trailing stop
        "opened_at": now_utc().isoformat()
    }

def manage_positions(ex: Any, positions: Dict[str, float], anchors: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    actions: List[str] = []
    for sym, qty in list(positions.items()):
        bid, ask = fetch_bid_ask(ex, sym)
        price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
        a = anchors.get(sym)
        if not a:
            # fallback seed
            ensure_anchor_on_buy(anchors, sym, price, "core")
            a = anchors.get(sym)
        tier = a.get("tier", "core")
        tp_pct, sl_pct, trail_pct = tier_params(tier)

        # track peak
        if price > a["peak"]:
            a["peak"] = price

        # compute conditions
        tp_hit = price >= a["tp"]
        sl_hit = price <= a["sl"]
        trail_floor = a["peak"] * (1.0 - trail_pct)
        trail_hit = price <= trail_floor and price > 0

        # spec fast-drop kill within window
        fast_hit = False
        if tier == "spec":
            try:
                opened_at = datetime.fromisoformat(a.get("opened_at"))
            except Exception:
                opened_at = now_utc()
            age_min = max(0, int((now_utc() - opened_at).total_seconds() // 60))
            if age_min <= SPEC_FAST_DROP_WINDOW_MIN:
                fast_floor = a["entry"] * (1.0 - SPEC_FAST_DROP_PCT)
                fast_hit = price <= fast_floor

        reason = None
        if tp_hit:
            reason = "TAKE_PROFIT"
        elif sl_hit:
            reason = "STOP_LOSS"
        elif fast_hit:
            reason = "FAST_DROP"
        elif trail_hit:
            reason = "TRAILING_STOP"

        if reason:
            res = place_market_sell(ex, sym, qty, bid if bid > 0 else price, reason, tier)
            actions.append(f"sell:{sym}:{reason}:{res.get('status')}")
            anchors.pop(sym, None)
        else:
            anchors[sym] = a

    return actions, anchors

# --------- KPI / CSV --------- #

def write_kpi_csv(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.isfile(path)
    cols = ["ts", "dry_run", "picked", "scores", "hot_list", "usd_free", "actions"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists: w.writeheader()
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
    print(f"=== Crypto Live — Core + Spec (DRY_RUN={DRY_RUN}, SPEC_ENABLE={SPEC_ENABLE}) ===")
    if RUN_SWITCH == "OFF":
        print("[SKIP] RUN_SWITCH=OFF"); return

    ex = build_exchange()
    try: ex.load_markets()
    except Exception as e: print(f"[WARN] load_markets failed: {e}")

    all_syms = CORE_LIST + SPEC_LIST
    usd_free = get_usd_balance(ex)
    positions = list_positions(ex, all_syms)
    anchors = load_anchors()

    core = core_candidates(ex)
    spec = spec_candidates(ex)

    # pretty print
    def fmt(c: Candidate) -> str:
        base = f"{c.symbol} | {c.tier} | score={c.score:+.1f} bps"
        extra = f"(24h={to_bps(c.mom_24h):+.0f}bps, 15m={to_bps(c.mom_15m):+.0f}bps, spread={c.spread_bps:.1f}bps"
        if c.tier == "spec": extra += f", vol24h=${c.vol24h_usd/1_000_000:.1f}M"
        if c.tier == "core": extra += f", hot={c.hot_bias_bps:.0f}bps"
        extra += ")"
        return f"{base} {extra}"

    for i, c in enumerate(core, 1): print(f"[CORE {i}] {fmt(c)}")
    for i, c in enumerate(spec, 1): print(f"[SPEC {i}] {fmt(c)}")

    # caps
    core_can_open = max(0, min(MAX_POSITIONS - len([p for p in positions if p in CORE_LIST]), MAX_DAILY_NEW_ENTRIES))
    spec_can_open = max(0, min(SPEC_MAX_POSITIONS - len([p for p in positions if p in SPEC_LIST]), SPEC_MAX_DAILY_NEW_ENTRIES))

    actions: List[str] = []
    picked_syms = [c.symbol for c in (core + spec)]
    score_blurbs = [f"{c.symbol}:{c.tier}:{c.score:+.0f}bps" for c in (core + spec)]

    # Buy loop: core then spec
    for c in core:
        if core_can_open <= 0: break
        if c.symbol in positions: continue
        if usd_free - USD_PER_TRADE < MIN_USD_BAL:
            print(f"[HALT] Reserve floor reached: need >= ${MIN_USD_BAL:.2f} after buy"); break
        res = place_market_buy(ex, c.symbol, USD_PER_TRADE, c.ask, "core")
        actions.append(f"buy:{c.symbol}:{res.get('status')}")
        if res.get("status") in ("filled","simulated"):
            entry_price = float(res.get("price") or c.ask)
            ensure_anchor_on_buy(anchors, c.symbol, entry_price, "core")
            usd_free -= USD_PER_TRADE
            core_can_open -= 1

    for c in spec:
        if spec_can_open <= 0: break
        if c.symbol in positions: continue
        if usd_free - SPEC_USD_PER_TRADE < MIN_USD_BAL:
            print(f"[HALT SPEC] Reserve floor reached for spec buy; need >= ${MIN_USD_BAL:.2f} after buy"); break
        res = place_market_buy(ex, c.symbol, SPEC_USD_PER_TRADE, c.ask, "spec")
        actions.append(f"buy:{c.symbol}:{res.get('status')}")
        if res.get("status") in ("filled","simulated"):
            entry_price = float(res.get("price") or c.ask)
            ensure_anchor_on_buy(anchors, c.symbol, entry_price, "spec")
            usd_free -= SPEC_USD_PER_TRADE
            spec_can_open -= 1

    # Manage exits
    exit_actions, anchors = manage_positions(ex, positions, anchors)
    actions.extend(exit_actions)

    # Persist and report
    save_anchors(anchors)
    print("\n==== SUMMARY ====")
    print(f"Picked: {picked_syms}")
    print(f"Scores: {score_blurbs}")
    print(f"Actions: {actions}")
    print(f"Hot list: {HOT_LIST}")
    print(f"Spec list (enabled={SPEC_ENABLE}): {SPEC_LIST}")
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
