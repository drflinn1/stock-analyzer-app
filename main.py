#!/usr/bin/env python3
import os, json, time, math, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

# ========== Config / ENV ==========
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))   # e.g., 3.0 = +3%
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT", "2.0"))     # e.g., 2.0 = -2%

# Trailing profit add-on:
TRAIL_START_PCT   = float(os.getenv("TRAIL_START_PCT", "3.0"))   # when PnL ≥ this, start trailing
TRAIL_OFFSET_PCT  = float(os.getenv("TRAIL_OFFSET_PCT", "1.0"))  # trail by this amount

# Basic buy controls (you already had these envs in workflows; keeping here for completeness)
POSITION_SIZE_USD    = float(os.getenv("POSITION_SIZE_USD", "10"))
DAILY_SPEND_CAP_USD  = float(os.getenv("DAILY_SPEND_CAP_USD", "15"))
MAX_OPEN_TRADES      = int(os.getenv("MAX_OPEN_TRADES", "3"))
MIN_BALANCE_USD      = float(os.getenv("MIN_BALANCE_USD", "5"))

# Symbols universe (bot may pick among these if your selection logic does that)
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USD,ETH/USD,DOGE/USD").split(",") if s.strip()]

# State file to remember entry_avg and trailing stop data for bot-managed coins
STATE_PATH = Path("state/trade_state.json")
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ========== Helpers ==========
def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"positions": {}}  # positions[symbol] = {"entry_avg": float, "peak_pnl_pct": float, "trail_active": bool}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def pct(a, b):
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

def fmt2(x):
    return f"{x:.2f}"

# ========== Exchange Adapter (stubbed for simplicity) ==========
# Replace these with your existing exchange adapter calls if you already have them.
# The shapes are kept intentionally simple so you can swap in your real functions.

class DummyPriceFeed:
    # In your real code, replace with ccxt Kraken calls:
    #   - fetch_balance() for USD + coin balances
    #   - fetch_ticker(symbol)["last"] for prices
    #   - create_order(...) for buys/sells (when not DRY_RUN)
    def __init__(self):
        # You will replace with actual CCXT client if you already have it wired
        self.prices = {}  # runtime-populated
    def price(self, symbol: str) -> float:
        # TODO: replace with real price fetch; for now raise if unset
        if symbol not in self.prices:
            raise RuntimeError(f"No price for {symbol}. Wire your price fetch here.")
        return float(self.prices[symbol])
    def set_price(self, symbol: str, px: float):
        self.prices[symbol] = float(px)

def get_cash_available_usd() -> float:
    # Replace with real USD balance fetch
    try:
        return float(os.getenv("SIM_BALANCE_USD", "100.0"))
    except:
        return 0.0

def get_coin_position_qty(symbol: str) -> float:
    # Replace with your real coin balance fetch (spot)
    # For safety here, return 0 so we don't accidentally "sell" in stub mode.
    return 0.0

def place_buy(symbol: str, usd_amount: float, price: float):
    if DRY_RUN:
        print(f"[DRY] BUY {symbol} ${fmt2(usd_amount)} @ {fmt2(price)}")
        return {"status": "dry", "symbol": symbol, "filled": usd_amount/price}
    # TODO: replace with real exchange order
    print(f"[LIVE] BUY {symbol} ${fmt2(usd_amount)} @ {fmt2(price)}")
    return {"status": "ok", "symbol": symbol, "filled": usd_amount/price}

def place_sell(symbol: str, qty: float, price: float):
    if DRY_RUN:
        print(f"[DRY] SELL {symbol} qty={qty:.8f} @ {fmt2(price)}")
        return {"status": "dry", "symbol": symbol, "filled": qty}
    # TODO: replace with real exchange order
    print(f"[LIVE] SELL {symbol} qty={qty:.8f} @ {fmt2(price)}")
    return {"status": "ok", "symbol": symbol, "filled": qty}

# ========== Core Logic ==========
def ensure_entry(state, symbol, entry_avg):
    pos = state["positions"].setdefault(symbol, {})
    if "entry_avg" not in pos or pos["entry_avg"] <= 0:
        pos["entry_avg"] = float(entry_avg)
    pos.setdefault("peak_pnl_pct", 0.0)
    pos.setdefault("trail_active", False)

def maybe_activate_trailing(state, symbol, pnl_pct):
    pos = state["positions"][symbol]
    # track peak pnl
    if pnl_pct > pos["peak_pnl_pct"]:
        pos["peak_pnl_pct"] = float(pnl_pct)
    # activate trailing at threshold
    if not pos["trail_active"] and pnl_pct >= TRAIL_START_PCT:
        pos["trail_active"] = True

def should_trail_exit(state, symbol, pnl_pct):
    pos = state["positions"][symbol]
    if not pos["trail_active"]:
        return False
    # compute drawdown from the peak
    drawdown = pos["peak_pnl_pct"] - pnl_pct
    return drawdown >= TRAIL_OFFSET_PCT

def decide_sell(pnl_pct):
    # Fixed TP/SL cut first
    if pnl_pct >= TAKE_PROFIT_PCT:
        return "TP"
    if pnl_pct <= -STOP_LOSS_PCT:
        return "SL"
    return None

def run_once():
    print("=== START TRADING OUTPUT ===")
    print(f"{now_iso()} | run started | DRY_RUN={DRY_RUN} | TP={fmt2(TAKE_PROFIT_PCT)}% | SL={fmt2(STOP_LOSS_PCT)}% | TRAIL_START={fmt2(TRAIL_START_PCT)}% | TRAIL_OFFSET={fmt2(TRAIL_OFFSET_PCT)}%")

    state = load_state()
    feed = DummyPriceFeed()

    # TODO: Replace these with real price fetches per symbol
    # For safety, require the caller to inject runtime prices via env like PRICE_BTC_USD, PRICE_ETH_USD, etc., when testing.
    # In your production code, delete this block and fetch from exchange.
    for sym in SYMBOLS:
        env_key = "PRICE_" + sym.replace("/", "_").replace("-", "_")
        px_env = os.getenv(env_key)
        if px_env:
            feed.set_price(sym, float(px_env))

    buys_placed = 0
    sells_placed = 0

    # ===== SELL CHECK for held coins =====
    for sym in SYMBOLS:
        qty = get_coin_position_qty(sym)
        if qty <= 0:
            continue  # nothing to evaluate

        price = feed.price(sym)
        # If we don't have an entry price on file (e.g., bot didn't open it), assume current (prevents false forced sells).
        ensure_entry(state, sym, entry_avg=price)
        entry_avg = state["positions"][sym]["entry_avg"]
        pnl_pct = pct(price, entry_avg)
        maybe_activate_trailing(state, sym, pnl_pct)

        reason = decide_sell(pnl_pct)
        trailed = False
        if reason is None and should_trail_exit(state, sym, pnl_pct):
            reason = "TRAIL"
            trailed = True

        # log line as promised:
        log = f"{now_iso()} | {sym} | entry_avg={fmt2(entry_avg)} | last={fmt2(price)} | PnL={('+' if pnl_pct>=0 else '')}{fmt2(pnl_pct)}% "
        if state["positions"][sym]["trail_active"]:
            log += f"| trail_on peak={fmt2(state['positions'][sym]['peak_pnl_pct'])}% "
        if reason:
            log += f"| SELL ({reason})"
        print(log)

        if reason:
            # Place sell
            r = place_sell(sym, qty, price)
            sells_placed += 1
            # Clear position tracking (bot is flat now)
            if sym in state["positions"]:
                del state["positions"][sym]

    # ===== BUY WINDOW (stub) =====
    # Your existing “best candidate” dip-buy logic should be here.
    # Below is a minimal skeleton honoring spend caps and DRY_RUN,
    # but it does NOT select candidates. Replace with your real logic.
    remaining_cap = DAILY_SPEND_CAP_USD  # you probably track daily spend elsewhere; keep this simple here
    cash = get_cash_available_usd()

    # Example: try to buy the first symbol if we have no position and enough cash (replace with your signal logic)
    for sym in SYMBOLS:
        if remaining_cap < POSITION_SIZE_USD or cash < max(MIN_BALANCE_USD, POSITION_SIZE_USD):
            break
        qty = get_coin_position_qty(sym)
        if qty > 0:
            continue  # already holding

        # Use price if available
        price = feed.price(sym)
        # *** Replace this condition with your real signal (RSI/dip rule/etc.) ***
        should_buy = False  # default: off, so we don't buy unexpectedly
        if should_buy:
            r = place_buy(sym, POSITION_SIZE_USD, price)
            buys_placed += 1
            remaining_cap -= POSITION_SIZE_USD
            # Set entry for trailing logic later
            ensure_entry(load_state(), sym, entry_avg=price)

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
