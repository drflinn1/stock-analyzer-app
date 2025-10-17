#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CryptoBOT main — 24/7 runner with:
- Emergency stop (RUN_SWITCH / EMERGENCY_STOP / .state/STOP file)
- Universe scan + rotation
- SELL RULES: TAKE_PROFIT, TRAIL (trailing stop), STOP_LOSS
- KPI logging to .state/kpi_history.csv
- Lightweight position state persistence in .state/positions.json
"""

import os, sys, json, math
from datetime import datetime, timezone

# --- Third-party ---
try:
    import ccxt  # noqa: F401
except Exception as e:
    print(f"[init] Missing ccxt: {e}")
    sys.exit(1)

# --- Local engine helpers ---
from trader.crypto_engine import (
    build_exchange,
    get_cash_balance_usd,
    fetch_positions_snapshot,
    pick_candidates,
    place_market_buy,
    place_market_sell,
    estimate_equity_usd,
)

# -------------------- Env helpers --------------------
def env_str(name, default): return os.environ.get(name, default)
def env_bool(name, default):
    v = os.environ.get(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","on","yes")
def env_int(name, default):
    try: return int(os.environ.get(name, default))
    except: return default
def env_float(name, default):
    try: return float(os.environ.get(name, default))
    except: return default

# -------------------- Config --------------------
DRY_RUN = env_str("DRY_RUN", "ON").upper() == "ON"      # ON = simulate
RUN_SWITCH = env_str("RUN_SWITCH", "ON").upper() == "ON"
EMERGENCY_STOP = env_str("EMERGENCY_STOP", "OFF").upper() == "ON"

MIN_BUY_USD  = env_float("MIN_BUY_USD", 10.0)
MIN_SELL_USD = env_float("MIN_SELL_USD", 10.0)
MAX_POSITIONS     = env_int("MAX_POSITIONS", 3)
MAX_BUYS_PER_RUN  = env_int("MAX_BUYS_PER_RUN", 2)
UNIVERSE_TOP_K    = env_int("UNIVERSE_TOP_K", 25)
RESERVE_CASH_PCT  = env_int("RESERVE_CASH_PCT", 5)

ROTATE_WHEN_FULL       = env_bool("ROTATE_WHEN_FULL", True)
ROTATE_WHEN_CASH_SHORT = env_bool("ROTATE_WHEN_CASH_SHORT", True)

# --- Sell rule knobs ---
# TAKE_PROFIT: sell when price >= avg_cost * (1 + TAKE_PROFIT_PCT/100)
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 4.0)
# STOP_LOSS: sell when price <= avg_cost * (1 - STOP_LOSS_PCT/100)
STOP_LOSS_PCT   = env_float("STOP_LOSS_PCT", 8.0)
# TRAIL: activate once price >= avg_cost * (1 + TRAIL_ACTIVATE_PCT/100),
# then sell if price <= high_water * (1 - TRAIL_PCT/100)
TRAIL_PCT           = env_float("TRAIL_PCT", 3.0)
TRAIL_ACTIVATE_PCT  = env_float("TRAIL_ACTIVATE_PCT", 2.0)

# --- Dust cleanup (not aggressive; safe defaults) ---
DUST_MIN_USD     = env_float("DUST_MIN_USD", 2.0)
DUST_SKIP_STABLES= env_bool("DUST_SKIP_STABLES", True)

# --- Paths ---
STATE_DIR = env_str("STATE_DIR", ".state")
KPI_CSV   = env_str("KPI_CSV", os.path.join(STATE_DIR, "kpi_history.csv"))
POS_STATE = os.path.join(STATE_DIR, "positions.json")
STOP_FILE = os.path.join(STATE_DIR, "STOP")
os.makedirs(STATE_DIR, exist_ok=True)

# -------------------- Emergency stop --------------------
def emergency_stop_active() -> bool:
    if not RUN_SWITCH:
        print("[STOP] RUN_SWITCH=OFF — trading halted.")
        return True
    if EMERGENCY_STOP:
        print("[STOP] EMERGENCY_STOP=ON — trading halted.")
        return True
    if os.path.exists(STOP_FILE):
        print("[STOP] .state/STOP present — trading halted.")
        return True
    return False

# -------------------- State I/O --------------------
def load_pos_state() -> dict:
    try:
        with open(POS_STATE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_pos_state(state: dict):
    tmp = POS_STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, POS_STATE)

# -------------------- KPI --------------------
def append_kpi_row(equity_usd: float, positions_count: int):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    header_needed = not os.path.exists(KPI_CSV)
    with open(KPI_CSV, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,equity,positions\n")
        f.write(f"{ts},{equity_usd:.2f},{positions_count}\n")
    print(f"[KPI] {ts} equity={equity_usd:.2f} pos={positions_count}")

# -------------------- Rotation --------------------
def rotate_if_needed(exchange, positions, candidates, cash_usd, reserve_cash_usd):
    if not positions:
        return cash_usd
    portfolio_full = len(positions) >= MAX_POSITIONS
    cash_short = cash_usd < reserve_cash_usd
    should_rotate = (portfolio_full and ROTATE_WHEN_FULL) or (cash_short and ROTATE_WHEN_CASH_SHORT)
    if not should_rotate:
        return cash_usd

    change_map = {c["symbol"]: c.get("change24h", -999) for c in candidates}
    ranked = sorted(positions, key=lambda p: change_map.get(p["symbol"], -999))
    worst = ranked[0]
    worst_usd = worst["usd_value"]
    if worst_usd < MIN_SELL_USD:
        print(f"[rotate] Worst {worst['symbol']} only ${worst_usd:.2f} — skip rotation.")
        return cash_usd

    held_syms = {p["symbol"] for p in positions}
    best = next((c for c in candidates if c["symbol"] not in held_syms), None)
    if not best:
        print("[rotate] No stronger candidate available.")
        return cash_usd

    print(f"[rotate] SELL {worst['symbol']} ~ ${worst_usd:.2f} → rotate into {best['symbol']}.")
    if not DRY_RUN:
        try:
            place_market_sell(exchange, worst["symbol"], worst["base_qty"])
        except Exception as e:
            print(f"[rotate] Sell failed: {e}")
            return cash_usd
    cash_usd += worst_usd
    return cash_usd

# -------------------- SELL RULES --------------------
def evaluate_sell_rules(symbol: str, price: float, st: dict) -> str:
    """
    Returns reason string if a sell should trigger, else "".
    State keys per symbol:
      - avg_cost: float
      - high_water: float
      - trail_armed: bool
    """
    avg_cost = st.get("avg_cost", price)
    high = st.get("high_water", price)
    st["avg_cost"] = avg_cost
    st["high_water"] = max(high, price)

    # TAKE_PROFIT
    tp_price = avg_cost * (1.0 + TAKE_PROFIT_PCT / 100.0)
    if price >= tp_price:
        return "TAKE_PROFIT"  # keyword required by guard

    # Activate trailing once minimal profit achieved
    activate_price = avg_cost * (1.0 + TRAIL_ACTIVATE_PCT / 100.0)
    if price >= activate_price:
        st["trail_armed"] = True

    # TRAIL
    if st.get("trail_armed"):
        trail_line = st["high_water"] * (1.0 - TRAIL_PCT / 100.0)
        if price <= trail_line:
            return "TRAIL"  # keyword required by guard

    # STOP_LOSS
    sl_price = avg_cost * (1.0 - STOP_LOSS_PCT / 100.0)
    if price <= sl_price:
        return "STOP_LOSS"  # keyword required by guard

    return ""

def perform_sells(exchange, positions):
    """
    Walk all positions and apply sell rules.
    """
    state = load_pos_state()
    sells = 0
    for p in positions:
        sym = p["symbol"]
        price = float(p["price"])
        usd_val = float(p["usd_value"])
        if usd_val < MIN_SELL_USD:
            continue

        st = state.get(sym, {})
        reason = evaluate_sell_rules(sym, price, st)
        state[sym] = st  # updated (avg/high/trail)

        if reason:
            print(f"[sell] {reason} → SELL {sym} (${usd_val:.2f}) @ {price:.6f}")
            if not DRY_RUN:
                try:
                    place_market_sell(exchange, sym, p["base_qty"])
                except Exception as e:
                    print(f"[sell] Failed: {e}")
                    continue
            sells += 1
    # prune symbols no longer held
    held_now = {p["symbol"] for p in positions}
    for sym in list(state.keys()):
        if sym not in held_now:
            state.pop(sym, None)
    save_pos_state(state)
    return sells

# -------------------- Main --------------------
def main():
    print("=== Crypto Live — SAFE ===", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    print(f"Mode: {'DRY' if DRY_RUN else 'LIVE'}  RUN_SWITCH: {'ON' if RUN_SWITCH else 'OFF'}")
    print(f"MAX_POSITIONS={MAX_POSITIONS}  MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}")
    print(f"UNIVERSE_TOP_K={UNIVERSE_TOP_K}  RESERVE_CASH_PCT={RESERVE_CASH_PCT}")
    print(f"SELLS: TAKE_PROFIT={TAKE_PROFIT_PCT}%  TRAIL={TRAIL_PCT}% (arm {TRAIL_ACTIVATE_PCT}%)  STOP_LOSS={STOP_LOSS_PCT}%")

    # If stopped, still log KPI and exit.
    if emergency_stop_active():
        try:
            ex = build_exchange(os.environ.get("KRAKEN_API_KEY"), os.environ.get("KRAKEN_API_SECRET"), DRY_RUN)
            equity = estimate_equity_usd(ex)
            append_kpi_row(equity, 0)
        except Exception as e:
            print(f"[stop] KPI log while stopped failed: {e}")
        return

    api_key = os.environ.get("KRAKEN_API_KEY")
    api_secret = os.environ.get("KRAKEN_API_SECRET")
    exchange = build_exchange(api_key, api_secret, DRY_RUN)

    # Universe & candidates
    candidates = pick_candidates(exchange, top_k=UNIVERSE_TOP_K)
    print(f"[scan] top{len(candidates)}:",
          ", ".join([f"{c['symbol']}({c['change24h']:+.2f}%)" for c in candidates[:10]]) +
          (", ..." if len(candidates) > 10 else ""))

    # Positions & balances
    positions = fetch_positions_snapshot(exchange)
    cash_usd = get_cash_balance_usd(exchange)
    equity = estimate_equity_usd(exchange)
    reserve_cash_usd = equity * (RESERVE_CASH_PCT / 100.0)
    print(f"[acct] cash=${cash_usd:.2f} reserve=${reserve_cash_usd:.2f} equity=${equity:.2f} positions={len(positions)}")

    # --- SELL first (respect risk & free cash) ---
    sells = perform_sells(exchange, positions)
    if sells:
        # refresh after sells
        positions = fetch_positions_snapshot(exchange)
        cash_usd = get_cash_balance_usd(exchange)

    # --- Rotation if needed ---
    cash_usd = rotate_if_needed(exchange, positions, candidates, cash_usd, reserve_cash_usd)

    # --- BUY flow ---
    buys_made = 0
    held = {p["symbol"] for p in positions}
    for coin in candidates:
        if buys_made >= MAX_BUYS_PER_RUN: break
        if len(held) >= MAX_POSITIONS: break
        if coin["symbol"] in held: continue

        # allocation: aim to distribute risk across max positions while keeping reserve
        alloc = max(MIN_BUY_USD, (equity - reserve_cash_usd) / max(1, MAX_POSITIONS))
        free_to_spend = max(0.0, cash_usd - reserve_cash_usd)
        spend = min(alloc, free_to_spend)
        if spend < MIN_BUY_USD:
            continue

        print(f"[buy] BUY {coin['symbol']} for ~${spend:.2f}")
        if not DRY_RUN:
            try:
                place_market_buy(exchange, coin["symbol"], spend)
            except Exception as e:
                print(f"[buy] failed: {e}")
                continue
        buys_made += 1
        cash_usd -= spend
        held.add(coin["symbol"])

    # KPI after trades
    positions_after = fetch_positions_snapshot(exchange)
    equity_after = estimate_equity_usd(exchange)
    append_kpi_row(equity_after, len(positions_after))

    print("SUMMARY:")
    print(f"  sells={sells}  buys={buys_made}  pos={len(positions_after)}  equity=${equity_after:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fatal] {e}")
        sys.exit(1)
