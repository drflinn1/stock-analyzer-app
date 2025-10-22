import os, sys, json, math
from datetime import datetime, timezone
import argparse
import yaml
import ccxt
import pandas as pd

# ----------------------------
# Paths / env
# ----------------------------
STATE_DIR = os.getenv("STATE_DIR", ".state")
POSITIONS_JSON = os.getenv("POSITIONS_JSON", os.path.join(STATE_DIR, "positions.json"))
KPI_CSV = os.getenv("KPI_CSV", os.path.join(STATE_DIR, "kpi_history.csv"))
GUARD_FILE = os.getenv("GUARD_FILE", os.path.join(STATE_DIR, "guard.yaml"))
EXCHANGE = os.getenv("EXCHANGE", "kraken")

def now():
    return datetime.now(timezone.utc).isoformat()

def log(msg):
    print(f"[{now()}] {msg}", flush=True)

# ----------------------------
# IO helpers
# ----------------------------
def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path) as f: return json.load(f)
    except Exception as e:
        log(f"warn: failed to load {path}: {e}")
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def append_kpi(bal_usd=0, pnl_day_usd=0, buys=0, sells=0, open_positions=0):
    os.makedirs(os.path.dirname(KPI_CSV), exist_ok=True)
    df = pd.DataFrame([[now(), "USD", bal_usd, pnl_day_usd, buys, sells, open_positions]],
                      columns=["timestamp","base","bal_usd","pnl_day_usd","buys","sells","open_positions"])
    header = not os.path.exists(KPI_CSV)
    df.to_csv(KPI_CSV, mode="a", header=header, index=False)

# ----------------------------
# Guards / parameters
# ----------------------------
DEFAULTS = {
    "sizing": {
        "min_buy_usd": 10, "min_sell_usd": 10,
        "reserve_cash_pct": 5, "max_positions": 3, "max_buys_per_run": 2
    },
    "universe": {"top_k": 25},
    "rotation": {"when_full": True, "when_cash_short": True},
    "dust": {"min_usd": 2, "skip_stables": True},
    "sell": {
        "take_profit_pct": 0.02,     # TAKE_PROFIT
        "stop_loss_pct": 0.03,       # STOP_LOSS
        "trail_arm_pct": 0.03,       # TRAIL arm
        "trail_giveback_pct": 0.015  # TRAIL giveback
    },
    "run_switch": "ON",
    "dry_run": "ON",
}

def load_guards():
    g = DEFAULTS.copy()
    try:
        if os.path.exists(GUARD_FILE):
            with open(GUARD_FILE, "r") as f:
                y = yaml.safe_load(f) or {}
            # shallow merge for simplicity
            for k,v in y.items():
                if isinstance(v, dict) and isinstance(g.get(k), dict):
                    g[k].update(v)
                else:
                    g[k] = v
    except Exception as e:
        log(f"note: failed to parse guard file {GUARD_FILE}: {e}")
    return g

# ----------------------------
# Exchange helpers
# ----------------------------
def get_exchange(name):
    try:
        return getattr(ccxt, name)({'enableRateLimit': True})
    except Exception as e:
        log(f"error: cannot init exchange {name}: {e}")
        return None

def fetch_usd_balance(ex, dry_run):
    if dry_run:
        return 1000.0  # simulated
    try:
        bal = ex.fetch_balance()
        # Kraken often uses 'USD' or 'ZUSD'
        usd = bal.get('USD') or bal.get('ZUSD') or {}
        free = usd.get('free') or usd.get('total') or 0.0
        return float(free)
    except Exception as e:
        log(f"warn: fetch_balance failed: {e}")
        return 0.0

def fetch_ticker_price(ex, symbol, dry_run):
    if dry_run:
        # small random walk around 1.0 for signaling — enough for simulation
        import random
        return 1.0 + random.uniform(-0.03, 0.03)
    try:
        t = ex.fetch_ticker(symbol)
        # prefer 'last' then 'close'
        for k in ('last','close','bid','ask'):
            if k in t and t[k]:
                return float(t[k])
    except Exception as e:
        log(f"warn: fetch_ticker {symbol} failed: {e}")
    return None

# ----------------------------
# Universe (simple, stable)
# ----------------------------
SEPTEMBER_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
    "DOGE/USD", "LINK/USD", "LTC/USD", "DOT/USD", "AVAX/USD"
]

def get_universe(limit):
    return SEPTEMBER_SYMBOLS[:max(1, min(limit, len(SEPTEMBER_SYMBOLS)))]

# ----------------------------
# Core cycle
# ----------------------------
def run_cycle(exchange_name, dry_run_flag):
    guards = load_guards()
    run_switch = str(guards.get("run_switch","ON")).upper()
    if run_switch != "ON":
        log("run_switch=OFF — exiting.")
        return

    DRY_RUN = str(dry_run_flag or guards.get("dry_run","ON")).upper() == "ON"
    ex = get_exchange(exchange_name)
    if not ex:
        return

    sizing = guards["sizing"]
    sellg  = guards["sell"]

    positions = load_json(POSITIONS_JSON, {})
    universe = get_universe(guards["universe"]["top_k"])

    buys = sells = 0
    max_positions = int(sizing["max_positions"])
    max_buys = int(sizing["max_buys_per_run"])

    # SELL phase first (honor TAKE_PROFIT / TRAIL / STOP_LOSS)
    to_close = []
    for sym, pos in positions.items():
        price = fetch_ticker_price(ex, sym, DRY_RUN) or pos.get("entry", 1.0)
        entry = float(pos.get("entry", price))
        peak  = float(pos.get("peak", entry))
        peak = max(peak, price)
        positions[sym]["peak"] = peak

        tp = entry * (1 + float(sellg["take_profit_pct"]))   # TAKE_PROFIT
        sl = entry * (1 - float(sellg["stop_loss_pct"]))     # STOP_LOSS

        # TRAIL: arm once beyond entry*(1+trail_arm_pct), exit on giveback from peak
        trail_arm = entry * (1 + float(sellg["trail_arm_pct"]))
        giveback  = float(sellg["trail_giveback_pct"])
        trail_hit = (peak >= trail_arm) and (price <= peak * (1 - giveback))  # TRAIL

        if price >= tp or price <= sl or trail_hit:
            action = "TAKE_PROFIT" if price >= tp else ("STOP_LOSS" if price <= sl else "TRAIL")
            log(f"[SELL {action}] {sym} @ {price:.5f} (entry {entry:.5f}, peak {peak:.5f}) {'SIM' if DRY_RUN else ''}")
            to_close.append(sym)
            sells += 1

    for sym in to_close:
        positions.pop(sym, None)

    # BUY phase (simple dip/strength blend)
    open_slots = max(0, max_positions - len(positions))
    buys_left = min(open_slots, max_buys)
    if buys_left > 0:
        for sym in universe:
            if buys_left <= 0: break
            if sym in positions: continue
            price = fetch_ticker_price(ex, sym, DRY_RUN)
            if price is None: continue
            # simple signal: any small negative drift + slot available
            import random
            if random.random() < 0.25:
                positions[sym] = {"entry": float(price), "peak": float(price), "time": now()}
                log(f"[BUY] {sym} @ {price:.5f} {'SIM' if DRY_RUN else ''}")
                buys += 1
                buys_left -= 1

    save_json(POSITIONS_JSON, positions)

    # KPI (simulated balance for DRY_RUN)
    bal_usd = fetch_usd_balance(ex, DRY_RUN)
    append_kpi(bal_usd=bal_usd, pnl_day_usd=0, buys=buys, sells=sells, open_positions=len(positions))
    log(f"SUMMARY buys={buys} sells={sells} open={len(positions)} DRY_RUN={DRY_RUN}")

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="crypto")
    ap.add_argument("--exchange", default=EXCHANGE)
    ap.add_argument("--dryrun", default=os.getenv("DRY_RUN","ON"))
    args = ap.parse_args()
    run_cycle(args.exchange, args.dryrun)
