#!/usr/bin/env python3
import os, json, time, math, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

# ========== Config / ENV ==========
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))   # +3% TP
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT", "2.0"))     # -2% SL

# Trailing profit add-on
TRAIL_START_PCT   = float(os.getenv("TRAIL_START_PCT", "3.0"))   # start trailing when PnL ≥ this
TRAIL_OFFSET_PCT  = float(os.getenv("TRAIL_OFFSET_PCT", "1.0"))  # trail by this amount

# Buy controls
POSITION_SIZE_USD    = float(os.getenv("POSITION_SIZE_USD", "10"))
DAILY_SPEND_CAP_USD  = float(os.getenv("DAILY_SPEND_CAP_USD", "15"))
MAX_OPEN_TRADES      = int(os.getenv("MAX_OPEN_TRADES", "3"))
MIN_BALANCE_USD      = float(os.getenv("MIN_BALANCE_USD", "5"))

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USD,ETH/USD,DOGE/USD").split(",") if s.strip()]

# Kraken/CCXT keys (balances & live orders use these)
KRAKEN_KEY    = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET", "")

# State for entry_avg & trailing
STATE_PATH = Path("state/trade_state.json")
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ========== Helpers ==========
def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"positions": {}}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def pct(a, b):
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

def fmt2(x): return f"{x:.2f}"

# ========== Exchange via CCXT (Kraken) ==========
# Requires ccxt installed in your workflow step (you already had this earlier).
import ccxt

def kraken_client():
    # Public endpoints (prices) don’t need keys.
    # Private endpoints (balance, orders) do.
    conf = {
        "enableRateLimit": True,
        "timeout": 20000
    }
    if KRAKEN_KEY and KRAKEN_SECRET:
        conf.update({"apiKey": KRAKEN_KEY, "secret": KRAKEN_SECRET})
    return ccxt.kraken(conf)

EX = None

def init_exchange():
    global EX
    if EX is None:
        EX = kraken_client()

def price(symbol: str) -> float:
    # ccxt uses unified symbols; Kraken supports BTC/USD, ETH/USD, DOGE/USD, etc.
    init_exchange()
    t = EX.fetch_ticker(symbol)
    last = t.get("last") or t.get("close")
    if last is None:
        raise RuntimeError(f"No last price for {symbol}")
    return float(last)

def get_cash_available_usd() -> float:
    # Needs keys; otherwise returns 0.0.
    init_exchange()
    try:
        bal = EX.fetch_balance()
        free_usd = bal.get("free", {}).get("USD", 0.0)
        return float(free_usd or 0.0)
    except Exception:
        return 0.0

def base_from_symbol(symbol: str) -> str:
    return symbol.split("/")[0]

def get_coin_position_qty(symbol: str) -> float:
    # Needs keys; returns 0.0 without keys.
    init_exchange()
    base = base_from_symbol(symbol)
    try:
        bal = EX.fetch_balance()
        qty = bal.get("free", {}).get(base, 0.0)
        return float(qty or 0.0)
    except Exception:
        return 0.0

def place_buy(symbol: str, usd_amount: float, px: float):
    amt = max(usd_amount / px, 0.0)
    amt = float(f"{amt:.6f}")  # trim for Kraken
    if amt <= 0:
        print(f"[SKIP] BUY {symbol} amount <= 0")
        return {"status": "skip"}

    if DRY_RUN:
        print(f"[DRY] BUY {symbol} ${fmt2(usd_amount)} @ {fmt2(px)} (amount≈{amt})")
        return {"status": "dry", "symbol": symbol, "filled": amt}

    init_exchange()
    o = EX.create_order(symbol=symbol, type="market", side="buy", amount=amt)
    print(f"[LIVE] BUY {symbol} market amount={amt} (order id {o.get('id')})")
    return {"status": "ok", "symbol": symbol, "filled": amt, "order": o}

def place_sell(symbol: str, qty: float, px: float):
    qty = float(f"{qty:.6f}")
    if qty <= 0:
        print(f"[SKIP] SELL {symbol} qty <= 0")
        return {"status": "skip"}

    if DRY_RUN:
        print(f"[DRY] SELL {symbol} qty={qty:.6f} @ {fmt2(px)}")
        return {"status": "dry", "symbol": symbol, "filled": qty}

    init_exchange()
    o = EX.create_order(symbol=symbol, type="market", side="sell", amount=qty)
    print(f"[LIVE] SELL {symbol} market qty={qty:.6f} (order id {o.get('id')})")
    return {"status": "ok", "symbol": symbol, "filled": qty, "order": o}

# ========== Trailing helpers ==========
def ensure_entry(state, symbol, entry_avg):
    pos = state["positions"].setdefault(symbol, {})
    if "entry_avg" not in pos or pos["entry_avg"] <= 0:
        pos["entry_avg"] = float(entry_avg)
    pos.setdefault("peak_pnl_pct", 0.0)
    pos.setdefault("trail_active", False)

def maybe_activate_trailing(state, symbol, pnl_pct):
    pos = state["positions"][symbol]
    if pnl_pct > pos["peak_pnl_pct"]:
        pos["peak_pnl_pct"] = float(pnl_pct)
    if not pos["trail_active"] and pnl_pct >= TRAIL_START_PCT:
        pos["trail_active"] = True

def should_trail_exit(state, symbol, pnl_pct):
    pos = state["positions"][symbol]
    if not pos["trail_active"]:
        return False
    drawdown = pos["peak_pnl_pct"] - pnl_pct
    return drawdown >= TRAIL_OFFSET_PCT

def decide_sell(pnl_pct):
    if pnl_pct >= TAKE_PROFIT_PCT: return "TP"
    if pnl_pct <= -STOP_LOSS_PCT:  return "SL"
    return None

# ========== Core run ==========
def run_once():
    print("=== START TRADING OUTPUT ===")
    print(f"{now_iso()} | run started | DRY_RUN={DRY_RUN} | TP={fmt2(TAKE_PROFIT_PCT)}% | SL={fmt2(STOP_LOSS_PCT)}% | TRAIL_START={fmt2(TRAIL_START_PCT)}% | TRAIL_OFFSET={fmt2(TRAIL_OFFSET_PCT)}%")

    state = load_state()

    buys_placed = 0
    sells_placed = 0

    # ----- SELL CHECK for held coins -----
    for sym in SYMBOLS:
        qty = get_coin_position_qty(sym)
        if qty <= 0:
            continue

        px = price(sym)
        # If no stored entry, assume current (prevents forced-sell surprises)
        ensure_entry(state, sym, entry_avg=px)
        entry_avg = state["positions"][sym]["entry_avg"]
        pnl_pct = pct(px, entry_avg)
        maybe_activate_trailing(state, sym, pnl_pct)

        reason = decide_sell(pnl_pct)
        if reason is None and should_trail_exit(state, sym, pnl_pct):
            reason = "TRAIL"

        log = f"{now_iso()} | {sym} | entry_avg={fmt2(entry_avg)} | last={fmt2(px)} | PnL={('+' if pnl_pct>=0 else '')}{fmt2(pnl_pct)}% "
        if state["positions"][sym]["trail_active"]:
            log += f"| trail_on peak={fmt2(state['positions'][sym]['peak_pnl_pct'])}% "
        if reason:
            log += f"| SELL ({reason})"
        print(log)

        if reason:
            r = place_sell(sym, qty, px)
            sells_placed += 1
            if sym in state["positions"]:
                del state["positions"][sym]

    # ----- BUY WINDOW (still OFF to avoid surprises) -----
    remaining_cap = DAILY_SPEND_CAP_USD
    cash = get_cash_available_usd()

    for sym in SYMBOLS:
        if remaining_cap < POSITION_SIZE_USD or cash < max(MIN_BALANCE_USD, POSITION_SIZE_USD):
            break
        if get_coin_position_qty(sym) > 0:
            continue

        px = price(sym)

        # Replace this with your real candidate/signal logic
        should_buy = False
        if should_buy:
            r = place_buy(sym, POSITION_SIZE_USD, px)
            buys_placed += 1
            remaining_cap -= POSITION_SIZE_USD
            # record entry for trailing later
            st = load_state()
            ensure_entry(st, sym, entry_avg=px)
            save_state(st)

    save_state(state)
    print(f"Run complete. buys_placed={buys_placed} | sells_placed={sells_placed} | DRY_RUN={DRY_RUN}")
    print("=== END TRADING OUTPUT ===")

# ========== Main ==========
if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        sys.exit(1)
