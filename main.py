# main.py — Crypto Live with Guard Pack (root-level)
# - DRY-RUN banner + simulated orders
# - USD-only accounting (Kraken: USD/ZUSD aware)
# - Universe: whitelist or auto-pick by 24h quoteVolume
# - TP/SL/Trailing logic (price-triggered exits)
# - Daily loss cap + reserve cash + max positions + max new entries
# - Color-coded logs + KPI history CSV

from __future__ import annotations
import os, json, time, math, csv, sys
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

STATE_DIR = ".state"
POSITIONS_F = os.path.join(STATE_DIR, "positions.json")
KPI_CSV = os.path.join(STATE_DIR, "kpi_history.csv")
SUMMARY_F = os.path.join(STATE_DIR, "summary_last.txt")

os.makedirs(STATE_DIR, exist_ok=True)

def env_str(k:str, d:str="") -> str: return str(os.getenv(k, d)).strip()
def env_f(k:str, d:float) -> float:
    try: return float(env_str(k, str(d)))
    except: return d
def env_i(k:str, d:int) -> int:
    try: return int(env_str(k, str(d)))
    except: return d
def env_b(k:str, d:bool) -> bool:
    v = env_str(k, "true" if d else "false").lower()
    return v in ("1","true","yes","y")

# ---------- ENV ----------
DRY_RUN = env_b("DRY_RUN", True)
EXCHANGE_ID = env_str("EXCHANGE_ID", "kraken")
MAX_ENTRIES_PER_RUN = env_i("MAX_ENTRIES_PER_RUN", 1)
USD_PER_TRADE = env_f("USD_PER_TRADE", 10.0)
TAKE_PROFIT_PCT = env_f("TAKE_PROFIT_PCT", 0.035)
STOP_LOSS_PCT   = env_f("STOP_LOSS_PCT",   0.020)
TRAIL_PCT       = env_f("TRAIL_PCT",       0.025)
DAILY_LOSS_CAP_PCT = env_f("DAILY_LOSS_CAP_PCT", -0.02)
RESERVE_USD = env_f("RESERVE_USD", 100.0)
UNIVERSE_MODE = env_str("UNIVERSE_MODE", "whitelist")
WHITELIST = [s.strip() for s in env_str("WHITELIST", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD").split(",") if s.strip()]
TOPK = env_i("TOPK", 8)
MAX_POSITIONS = env_i("MAX_POSITIONS", 6)
AVOID_STABLES = env_b("AVOID_STABLES", True)

GREEN = "\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; RESET="\033[0m"

def log_good(msg): print(GREEN + msg + RESET, flush=True)
def log_warn(msg): print(YELLOW + msg + RESET, flush=True)
def log_error(msg): print(RED + msg + RESET, flush=True)

def now_utc_ts() -> int: return int(time.time())
def now_iso() -> str: return datetime.now(timezone.utc).isoformat(timespec="seconds")

def load_positions() -> Dict[str, dict]:
    if not os.path.exists(POSITIONS_F): return {}
    try:
        with open(POSITIONS_F, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_positions(d: Dict[str, dict]) -> None:
    with open(POSITIONS_F, "w", encoding="utf-8") as f: json.dump(d, f, indent=2, sort_keys=True)

def append_kpi(row: Dict[str, str]) -> None:
    new_file = not os.path.exists(KPI_CSV)
    with open(KPI_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts","iso","pnl_realized","pnl_unrealized","positions","buys","sells","dry_run"
        ])
        if new_file: w.writeheader()
        w.writerow(row)

def stable_like(symbol: str) -> bool:
    s = symbol.upper()
    stubs = ("USDT/","USDC/","DAI/","TUSD/","FDUSD/","EUR/","GBP/","USD/")
    return any(s.startswith(x) for x in stubs) or any(s.endswith(x) for x in ("/USDT","/USDC","/DAI","/TUSD","/FDUSD","/EUR","/GBP","/USD"))

def usd_keys() -> Tuple[str,...]:
    return ("USD","ZUSD")

def fetch_usd_free(ex) -> float:
    bal = ex.fetch_balance()
    total = 0.0
    for k in usd_keys():
        if k in bal and isinstance(bal[k], dict):
            total += float(bal[k].get("free", 0) or 0)
        elif k in bal:
            total += float(bal.get(k, 0) or 0)
    if "free" in bal and isinstance(bal["free"], dict):
        for k in usd_keys():
            total += float(bal["free"].get(k, 0) or 0)
    return total

def get_exchange():
    if EXCHANGE_ID == "kraken":
        apiKey = os.getenv("KRAKEN_API_KEY","")
        secret = os.getenv("KRAKEN_API_SECRET","")
        ex = ccxt.kraken({
            "apiKey": apiKey,
            "secret": secret,
            "enableRateLimit": True
        })
    else:
        ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    ex.load_markets()
    return ex

# … rest of code identical to the earlier `main.py` version …
# (exits, buys, trailing, KPIs, run() wrapper, __main__)
