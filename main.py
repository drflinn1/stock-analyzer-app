# main.py — Crypto Live with Core + Spec tiers
# NEW: Spec Gate Report — prints a one-liner per SPEC symbol with PASS/FAIL for:
#  spread, vol24h, mom24h, mom15m, and reserve (incl. auto-bumped min notional).
# Keeps: exchange-aware min-notional bump, HOT bias, TP/SL/Trailing/FAST_DROP.

from __future__ import annotations
import os, json, math, csv, time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple, Optional

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- Utils ----------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def as_bool(v: Optional[str], default: bool) -> bool:
    if v is None: return default
    t = v.strip().lower()
    if t in ("1","true","yes","on","y","t"): return True
    if t in ("0","false","no","off","n","f"): return False
    return default

def as_float(v: Optional[str], default: float) -> float:
    try: return float(v) if v is not None else default
    except: return default

def as_int(v: Optional[str], default: int) -> int:
    try: return int(v) if v is not None else default
    except: return default

def to_bps(x: float) -> float: return x * 10000.0

def ensure_dir(p: str) -> None:
    if p: os.makedirs(p, exist_ok=True)

def read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception:
        return default

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def fmt_millions(x: float) -> str:
    return f"${x/1_000_000:.1f}M"

# ---------- ENV ----------

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
DRY_RUN     = as_bool(os.getenv("DRY_RUN"), True)
RUN_SWITCH  = os.getenv("RUN_SWITCH", "ON").upper()

# Core
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

# Spec sandbox
SPEC_ENABLE                 = as_bool(os.getenv("SPEC_ENABLE"), False)
SPEC_LIST                   = [s.strip() for s in os.getenv("SPEC_LIST", "").split(",") if s.strip()]
SPEC_MAX_POSITIONS          = as_int(os.getenv("SPEC_MAX_POSITIONS"), 1)
SPEC_MAX_DAILY_NEW_ENTRIES  = as_int(os.getenv("SPEC_MAX_DAILY_NEW_ENTRIES"), 1)
SPEC_USD_PER_TRADE          = as_float(os.getenv("SPEC_USD_PER_TRADE"), 5.0)
SPEC_MAX_SPREAD_BPS         = as_float(os.getenv("SPEC_MAX_SPREAD_BPS"), 120.0)
SPEC_MIN_VOL24H_USD         = as_float(os.getenv("SPEC_MIN_VOL24H_USD"), 3_000_000.0)
SPEC_MIN_MOM24H_BPS         = as_float(os.getenv("SPEC_MIN_MOM24H_BPS"), 1000.0)
SPEC_REQUIRE_MOM15M_POS     = as_bool(os.getenv("SPEC_REQUIRE_MOM15M_POS"), True)
SPEC_TP_PCT                 = as_float(os.getenv("SPEC_TP_PCT"), 0.050)
SPEC_SL_PCT                 = as_float(os.getenv("SPEC_SL_PCT"), 0.030)
SPEC_TRAIL_PCT              = as_float(os.getenv("SPEC_TRAIL_PCT"), 0.030)
SPEC_FAST_DROP_PCT          = as_float(os.getenv("SPEC_FAST_DROP_PCT"), 0.040)
SPEC_FAST_DROP_WINDOW_MIN   = as_int(os.getenv("SPEC_FAST_DROP_WINDOW_MIN"), 60)

STATE_DIR   = os.getenv("STATE_DIR", ".state")
KPI_CSV     = os.getenv("KPI_CSV", ".state/kpi_history.csv")
ANCHORS_JSON= os.path.join(STATE_DIR, "anchors.json")

USD_KEYS = ("USD", "ZUSD")

# ---------- Exchange helpers ----------

def build_exchange() -> Any:
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "apiKey": os.getenv("CCXT_API_KEY", ""),
        "secret": os.getenv("CCXT_API_SECRET", ""),
        "password": os.getenv("CCXT_API_PASSWORD") or None,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })
    return ex

def get_usd_balance(ex: Any) -> float:
    try: bal = ex.fetch_balance()
    except Exception as e: print(f"[WARN] fetch_balance error: {e}"); return 0.0
    for k in USD_KEYS:
        if k in bal and isinstance(bal[k], dict) and "free" in bal[k]:
            return float(bal[k]["free"] or 0.0)
    free = bal.get("free") or {}
    for k in USD_KEYS:
        if k in free: return float(free.get(k) or 0.0)
    return 0.0

def list_positions(ex: Any, all_symbols: List[str]) -> Dict[str, float]:
    try: bal = ex.fetch_balance()
    except Exception as e: print(f"[WARN] fetch_balance (positions) error: {e}"); return {}
    pos: Dict[str, float] = {}
    for s in all_symbols:
        base, _ = s.split("/")
        qty = float((bal.get(base) or {}).get("total") or (bal.get(base) or {}).get("free") or 0.0)
        if qty > 0: pos[s] = qty
    return pos

def fetch_bid_ask(ex: Any, symbol: str) -> Tuple[float, float]:
    t = ex.fetch_ticker(symbol)
    bid = float(t.get("bid") or 0.0); ask = float(t.get("ask") or 0.0)
    if bid <= 0 or ask <= 0:
        last = float(t.get("last") or 0.0)
        if last > 0: bid, ask = last*0.99975, last*1.00025
    return bid, ask

def compute_spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 1e9 if mid <= 0 else to_bps((ask - bid) / mid)

def usd_to_base(usd: float, ask: float) -> float:
    return 0.0 if ask <= 0 else usd / ask

def fetch_ohlcv_change(ex: Any, symbol: str, timeframe: str, bars: int) -> float:
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=bars)
        if not o or len(o) < 2: return 0.0
        first_c = float(o[0][4]); last_c = float(o[-1][4])
        if first_c <= 0: return 0.0
        return (last_c / first_c) - 1.0
    except Exception as e:
        print(f"[WARN] fetch_ohlcv {timeframe} for {symbol} failed: {e}"); return 0.0

def estimate_24h_quote_volume_usd(ex: Any, symbol: str) -> float:
    try:
        o = ex.fetch_ohlcv(symbol, timeframe="1h", limit=24)
        if not o: return 0.0
        return sum(float(v) * float(c) for _,_,_,_,c,v in o)
    except Exception as e:
        print(f"[WARN] vol24h {symbol}: {e}"); return 0.0

# --- Exchange min not
