# main.py — Auto-TopK Movers + Cash-Short Rotation + TP/SL/TRAIL
# - NEW: AUTO_TOPK dynamic universe from exchange tickers (biggest movers)
# - Safety filters: MAX_SPREAD_PCT, MIN_QUOTE_VOL_USD, EXCLUDE_BASES
# - Rotation: sell worst → buy best when cash < reserve and edge ≥ threshold
# - Guards: TAKE_PROFIT / STOP_LOSS / TRAIL with simple state
#
# Env (highlights):
#   AUTO_TOPK="6"                 # if >0, auto-pick top-K movers; if 0, use SYMBOL_WHITELIST
#   MAX_SPREAD_PCT="0.8"          # skip if (ask-bid)/mid * 100 > this
#   MIN_QUOTE_VOL_USD="20000"     # skip if 24h quote vol < this (needs exchange support; else fallback)
#   EXCLUDE_BASES="USD,USDT,USDC,EUR,GBP,SPX,BABY,PUMP"  # bases to exclude from candidates
#
#   ROTATE_WHEN_CASH_SHORT="true"
#   ROTATE_MIN_EDGE_PCT="2.0"
#   COOLDOWN_RUNS="1"
#
#   TAKE_PROFIT_PCT="3.5"  STOP_LOSS_PCT="2.0"  TRAIL_ARM_PCT="1.0"  TRAIL_PCT="1.5"

from __future__ import annotations
import os, json, pathlib, traceback
from typing import Dict, List, Tuple, Optional

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- helpers ----------

def as_bool(v: Optional[str], d: bool) -> bool:
    if v is None: return d
    return v.strip().lower() in ("1","true","yes","y","on")

def as_float(v: Optional[str], d: float) -> float:
    try: return float(v) if v is not None else d
    except: return d

def as_int(v: Optional[str], d: int) -> int:
    try: return int(v) if v is not None else d
    except: return d

def env_list(v: Optional[str], d: List[str]) -> List[str]:
    if not v: return d
    return [s.strip() for s in v.split(",") if s.strip()] or d

# ---------- ENV ----------

EXCHANGE_ID = os.getenv("EXCHANGE","kraken").lower()
API_KEY     = os.getenv("API_KEY") or os.getenv("KRAKEN_API_KEY") or os.getenv("CCXT_API_KEY") or ""
API_SECRET  = os.getenv("API_SECRET") or os.getenv("KRAKEN_API_SECRET") or os.getenv("CCXT_API_SECRET") or ""

DRY_RUN     = as_bool(os.getenv("DRY_RUN"), True)
MAX_POS     = as_int(os.getenv("MAX_POSITIONS"), 6)
USD_PER_TRADE = as_float(os.getenv("USD_PER_TRADE"), 15.0)
RESERVE_USD = as_float(os.getenv("RESERVE_USD"), 80.0)

# Rotation
ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)
ROTATE_MIN_EDGE_PCT    = as_float(os.getenv("ROTATE_MIN_EDGE_PCT"), 2.0)
COOLDOWN_RUNS          = as_int(os.getenv("COOLDOWN_RUNS"), 1)

# Universe
SYMBOL_WHITELIST = env_list(os.getenv("SYMBOL_WHITELIST"),
    ["BTC/USD","ETH/USD","SOL/USD","DOGE/USD","ZEC/USD","ENA/USD"])
AUTO_TOPK          = as_int(os.getenv("AUTO_TOPK"), 0)  # if > 0, use auto universe

# Auto-universe filters
MAX_SPREAD_PCT     = as_float(os.getenv("MAX_SPREAD_PCT"), 0.8)     # %
MIN_QUOTE_VOL_USD  = as_float(os.getenv("MIN_QUOTE_VOL_USD"), 20000) # $
EXCLUDE_BASES      = env_list(os.getenv("EXCLUDE_BASES"),
    ["USD","USDT","USDC","EUR","GBP","SPX","PUMP","BABY"])

# Sell-guards
TAKE_PROFIT_PCT = as_float(os.getenv("TAKE_PROFIT_PCT"), 3.5)
STOP_LOSS_PCT   = as_float(os.getenv("STOP_LOSS_PCT"), 2.0)
TRAIL_ARM_PCT   = as_float(os.getenv("TRAIL_ARM_PCT"), 1.0)
TRAIL_PCT       = as_float(os.getenv("TRAIL_PCT"), 1.5)

# State
STATE_DIR = pathlib.Path(".state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
COOLDOWN_PATH = STATE_DIR / "rotation_cooldowns.json"
ENTRIES_PATH  = STATE_DIR / "entries.json"
HIGHS_PATH    = STATE_DIR / "highs.json"

USD_KEYS    = ("USD","ZUSD")
STABLE_KEYS = ("USDT",)

# ---------- file io ----------

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
    for k in USD_KEYS: total += float(bal.get(k, {}).get("free",0) or bal.get(k,0) or 0)
    for k in STABLE_KEYS: total += float(bal.get(k, {}).get("free",0) or bal.get(k,0) or 0)
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

# ---------- scoring ----------

def momentum_score_1h(ex: ccxt.Exchange, symbol: str) -> float:
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1h", limit=3)
        if len(ohlcv) < 2: return 0.0
        open_prev  = ohlcv[-2][1]
        close_last = ohlcv[-1][4]
        if open_prev <= 0: return 0.0
        return (close_last - open_prev) / open_prev * 100.0
    except: return 0.0

def rank_by_momentum(ex: ccxt.Exchange, symbols: List[str]) -> List[Tuple[str,float]]:
    scored = [(s, momentum_score_1h(ex, s)) for s in symbols]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

# ---------- auto-universe ----------

def calc_spread_pct(tick: dict) -> float:
    bid = float(tick.get("bid") or 0)
    ask = float(tick.get("ask") or 0)
    if bid <= 0 or ask <= 0: return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0 if mid > 0 else 999.0

def quote_vol_usd(tick: dict) -> float:
    # CCXT often exposes 'quoteVolume' (24h volume in quote currency) or 'baseVolume'
    qv = float(tick.get("quoteVolume") or 0)
    if qv > 0: return qv
    # fallback: baseVolume * last
    bv = float(tick.get("baseVolume") or 0)
    last = float(tick.get("last") or tick.get("close") or 0)
    return bv * last

def base_of(symbol: str) -> str:
    return symbol.split("/")[0]

def build_auto_topk(ex: ccxt.Exchange, k: int,
                    max_spread_pct: float,
                    min_quote_vol_usd: float,
                    exclude_bases: List[str]) -> List[str]:
    tickers = ex.fetch_tickers()
    candidates: List[Tuple[str,float]] = []
    for sym, t in tickers.items():
        # only spot symbols containing /USD or /USDT
        if not (sym.endswith("/USD") or sym.endswith("/USDT")): 
            continue
        b = base_of(sym).upper()
        if b in exclude_bases: 
            continue
        sp = calc_spread_pct(t)
        if sp > max_spread_pct:
            continue
        qvol = quote_vol_usd(t)
        if qvol < min_quote_vol_usd:
            continue
        # rank by 24h percentage if present, else by 1h momentum fallback
        pct = t.get("percentage")
        if pct is None:
            pct = momentum_score_1h(ex, sym)
        try:
            score = float(pct or 0.0)
        except:
            score = 0.0
        candidates.append((sym, score))
    # sort by score desc and dedupe by base preferring USD over USDT
    candidates.sort(key=lambda x: x[1], reverse=True)
    picked: List[str] = []
    seen_bases: set[str] = set()
    for sym, _ in candidates:
        b = base_of(sym)
        if b in seen_bases: 
            continue
        seen_bases.add(b)
        picked.append(sym)
        if len(picked) >= k:
            break
    return picked

# ---------- market limits & orders (Kraken-safe) ----------

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
    try:
        free, total = _free_and_total_base(ex, symbol)
        if free <= 0 and total > 0:
            try: open_orders = ex.fetch_open_orders(symbol)
            except Exception: open_orders = []
            if not open_orders: free = total
        amt = max(0.0, min(free, total)) * max(0.0, min(1.0, pct_of_position))
        if amt <= 0:
            return False, f"SELL skip {symbol} — no free/total amount"
        price = last_price(ex, symbol)
        if price <= 0:
            return False, f"SELL skip {symbol} — price unknown"
        min_amt, min_cost = _market_limits(ex, symbol)
        amt_precise = float(ex.amount_to_precision(symbol, amt))
        est_cost = amt_precise * price
        if min_amt and amt_precise < min_amt:
            return False, (f"SELL skip {symbol} — amount {amt_precise:.8f} < min_amount {min_amt} "
                           f"(price {price:.8f}, est_cost {est_cost:.4f})")
        if min_cost and est_cost < min_cost:
            return False, (f"SELL skip {symbol} — est_cost ${est_cost:.4f} < min_cost ${min_cost
