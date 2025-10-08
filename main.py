#!/usr/bin/env python3
"""
Crypto Live Bot — Guarded, with Daily Cap + Rebuy Cooldown
- Exchange: Kraken via CCXT
- Features:
  * USD-only whitelist (BTC/ETH/SOL/DOGE by default)
  * Auto-pick top movers (simple momentum) with rotation when cash is short
  * Daily notional cap: skip new buys after hitting cap
  * Per-symbol rebuy cooldown to avoid churn
  * Dust sweep under DUST_MIN_USD
  * SL/TP1/Trailing stop exits on each run (15m schedule is fine)
  * DRY_RUN simulation with verbose logs
  * State persisted in .state/
"""

from __future__ import annotations
import os, json, math, time, csv, pathlib, random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

# -------------------------- Utilities & State -------------------------- #

ROOT = pathlib.Path(".")
STATE_DIR = ROOT / ".state"
STATE_DIR.mkdir(exist_ok=True)

def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)

def env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None: return default
    return str(val).strip().lower() in ("1","true","yes","on")

def env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except Exception:
        return default

def env_int(key: str, default: int) -> int:
    try:
        return int(float(os.environ.get(key, str(default))))
    except Exception:
        return default

def now_ts() -> float:
    return time.time()

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def today_key() -> str:
    # Use user's TZ? Actions runners use UTC; daily cap is fine in UTC.
    return datetime.now(timezone.utc).strftime("%Y%m%d")

def load_json(path: pathlib.Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default

def save_json(path: pathlib.Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)

def append_csv(path: pathlib.Path, row: List[Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts","event","symbol","side","qty","price","notional","note"])
        w.writerow(row)

# -------------------------- Config -------------------------- #

RUN_SWITCH          = env_str("RUN_SWITCH", "ON")
DRY_RUN             = env_str("DRY_RUN", "ON").upper()  # "ON"/"OFF"
USD_ONLY            = env_bool("USD_ONLY", True)
WHITELIST           = [s.strip() for s in env_str("WHITELIST", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD").split(",") if s.strip()]

MAX_POSITIONS       = env_int("MAX_POSITIONS", 6)
MAX_BUYS_PER_RUN    = env_int("MAX_BUYS_PER_RUN", 1)

SL_PCT              = env_float("SL_PCT", 0.04)     # 4%
TRAIL_PCT           = env_float("TRAIL_PCT", 0.035) # 3.5%
TP1_PCT             = env_float("TP1_PCT", 0.05)    # 5%
TP1_SIZE            = env_float("TP1_SIZE", 0.25)   # 25% size at TP1

RESERVE_CASH_PCT    = env_float("RESERVE_CASH_PCT", 0.10)
DAILY_CAP_USD       = env_float("DAILY_NOTIONAL_CAP_USD", 0.0)  # 0 disables
REBUY_COOLDOWN_MIN  = env_int("REBUY_COOLDOWN_MIN", 30)         # minutes between buys of same symbol

ROTATE_WHEN_CASH_SHORT = env_bool("ROTATE_WHEN_CASH_SHORT", True)
ROTATE_WHEN_FULL       = env_bool("ROTATE_WHEN_FULL", False)

DUST_MIN_USD        = env_float("DUST_MIN_USD", 2.0)
BOT_NAME            = env_str("BOT_NAME", "crypto-live")

KRAKEN_API_KEY      = env_str("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET   = env_str("KRAKEN_API_SECRET", "")
HAS_KEYS            = bool(KRAKEN_API_KEY and KRAKEN_API_SECRET)

# -------------------------- Exchange -------------------------- #

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"[FATAL] ccxt is required: {e}")

def make_exchange(dry: bool) -> Any:
    # Even in DRY_RUN we init client for tickers/balances.
    params = {"enableRateLimit": True}
    if HAS_KEYS and not dry:
        params.update({"apiKey": KRAKEN_API_KEY, "secret": KRAKEN_API_SECRET})
    return ccxt.kraken(params)

# -------------------------- State Files -------------------------- #

POS_STATE_FILE   = STATE_DIR / "positions.json"      # avg_cost, qty, peak, last_buy_ts
COOLDOWN_FILE    = STATE_DIR / "cooldown.json"       # symbol -> last_buy_ts
KPI_FILE         = STATE_DIR / "kpi_history.csv"
DAILY_FILE       = STATE_DIR / f"notional_{today_key()}.json"  # {"spent": float}

positions: Dict[str, Dict[str, float]] = load_json(POS_STATE_FILE, {})
cooldown: Dict[str, float]             = load_json(COOLDOWN_FILE, {})
daily: Dict[str, float]                 = load_json(DAILY_FILE, {"spent": 0.0})

def bump_daily_spent(amount: float) -> None:
    daily["spent"] = round(float(daily.get("spent", 0.0)) + max(amount, 0.0), 2)
    save_json(DAILY_FILE, daily)

def record_trade(event: str, symbol: str, side: str, qty: float, price: float, note: str="") -> None:
    append_csv(STATE_DIR / "trades.csv", [now_iso(), event, symbol, side, f"{qty:.8f}", f"{price:.8f}", f"{qty*price:.2f}", note])

def kpi(note: str) -> None:
    append_csv(KPI_FILE, [now_iso(), "kpi", note, "", "", "", "", ""])

# -------------------------- Helpers -------------------------- #

def log_header():
    print("========== CONFIG ==========")
    print(f"BOT={BOT_NAME}  RUN_SWITCH={RUN_SWITCH}  DRY_RUN={DRY_RUN}  HAS_KEYS={HAS_KEYS}")
    print(f"MAX_POS={MAX_POSITIONS}  MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}")
    print(f"SL={SL_PCT}  TRAIL={TRAIL_PCT}  TP1={TP1_PCT} x {TP1_SIZE}")
    print(f"RESERVE_CASH_PCT={RESERVE_CASH_PCT}  DUST_MIN_USD={DUST_MIN_USD}")
    print(f"DAILY_CAP_USD={DAILY_CAP_USD} (spent_today={daily.get('spent',0.0)})  REBUY_COOLDOWN_MIN={REBUY_COOLDOWN_MIN}")
    print(f"WHITELIST={','.join(WHITELIST)}  USD_ONLY={USD_ONLY}")
    print(f"ROTATE_WHEN_CASH_SHORT={ROTATE_WHEN_CASH_SHORT}  ROTATE_WHEN_FULL={ROTATE_WHEN_FULL}")
    print("============================")

def fetch_market_snapshot(ex):
    markets = ex.load_markets()
    # Tickers may fail intermittently; retry once.
    try:
        tickers = ex.fetch_tickers(WHITELIST)
    except Exception:
        time.sleep(1.0)
        tickers = ex.fetch_tickers(WHITELIST)
    # Build simple metrics (last, 24h change, ask/bid, spread)
    data = {}
    for s in WHITELIST:
        t = tickers.get(s, {})
        last = float(t.get("last") or t.get("close") or 0.0)
        percent = float(t.get("percentage") or 0.0)  # 24h change %
        ask = float(t.get("ask") or last or 0.0)
        bid = float(t.get("bid") or last or 0.0)
        spread = (ask - bid) / ask if ask else 0.0
        data[s] = {"last": last, "pct": percent, "ask": ask, "bid": bid, "spread": spread}
    return data

def fetch_balances(ex) -> Tuple[float, Dict[str, float]]:
    bal = ex.fetch_free_balance()
    usd = float(bal.get("USD", 0.0)) + float(bal.get("ZUSD", 0.0))  # Kraken sometimes uses ZUSD
    coins = {}
    for s in WHITELIST:
        base = s.split("/")[0]
        coins[base] = float(bal.get(base, 0.0))
    return usd, coins

def value_of(symbol: str, qty: float, price: float) -> float:
    return float(qty) * float(price)

def want_slots(current_positions: int) -> int:
    return max(0, MAX_POSITIONS - current_positions)

def can_buy_symbol(sym: str, nowt: float) -> bool:
    # Cooldown
    last = cooldown.get(sym, 0.0)
    if last and (nowt - last) < (REBUY_COOLDOWN_MIN * 60):
        return False
    return True

def update_after_buy(sym: str, qty: float, price: float):
    nowt = now_ts()
    p = positions.get(sym, {"avg_cost": 0.0, "qty": 0.0, "peak": 0.0, "last_buy_ts": 0.0})
    new_qty = p["qty"] + qty
    new_cost = (p["avg_cost"]*p["qty"] + qty*price) / new_qty if new_qty > 0 else price
    p["qty"] = new_qty
    p["avg_cost"] = new_cost
    p["peak"] = max(p.get("peak", 0.0), price)
    p["last_buy_ts"] = nowt
    positions[sym] = p
    cooldown[sym] = nowt
    save_json(POS_STATE_FILE, positions)
    save_json(COOLDOWN_FILE, cooldown)

def update_after_sell(sym: str, sell_qty: float, price: float):
    p = positions.get(sym, {"avg_cost": 0.0, "qty": 0.0, "peak": 0.0, "last_buy_ts": 0.0})
    p["qty"] = max(0.0, p["qty"] - sell_qty)
    p["peak"] = max(p.get("peak", 0.0), price)  # keep peak sane
    if p["qty"] <= 1e-9:
        # remove when fully closed
        positions.pop(sym, None)
        cooldown.pop(sym, None)
    else:
        positions[sym] = p
    save_json(POS_STATE_FILE, positions)
    save_json(COOLDOWN_FILE, cooldown)

# -------------------------- Order Execution (Sim or Live) -------------------------- #

def place_market_buy(ex, symbol: str, usd_alloc: float, price_hint: float) -> Tuple[float,float]:
    # Convert USD allocation to base qty.
    price = price_hint if price_hint > 0 else ex.fetch_ticker(symbol)["last"]
    qty = round(usd_alloc / price, 8)
    if qty <= 0:
        raise RuntimeError("qty <= 0")

    if DRY_RUN == "ON":
        print(f"[SIM BUY] {symbol} qty={qty:.8f} @ ~{price:.4f}  notional=${usd_alloc:.2f}")
        record_trade("sim", symbol, "buy", qty, price, "dry_run")
        return qty, price

    if not HAS_KEYS:
        raise RuntimeError("Live BUY requested but API keys not present")

    try:
        order = ex.create_market_buy_order(symbol, qty)
        fill_price = price
        if isinstance(order, dict):
            fill_price = float(order.get("average") or order.get("price") or price)
        print(f"[LIVE BUY] {symbol} qty={qty:.8f} @ ~{fill_price:.4f}  notional≈${qty*fill_price:.2f}")
        record_trade("live", symbol, "buy", qty, fill_price, "market")
        return qty, fill_price
    except Exception as e:
        print(f"[ERROR BUY] {symbol}: {e}")
        raise

def place_market_sell(ex, symbol: str, qty: float, price_hint: float) -> Tuple[float,float]:
    price = price_hint if price_hint > 0 else ex.fetch_ticker(symbol)["last"]
    qty = round(qty, 8)
    if qty <= 0:
        return 0.0, price

    if DRY_RUN == "ON":
        print(f"[SIM SELL] {symbol} qty={qty:.8f} @ ~{price:.4f}  notional≈${qty*price:.2f}")
        record_trade("sim", symbol, "sell", qty, price, "dry_run")
        return qty, price

    if not HAS_KEYS:
        raise RuntimeError("Live SELL requested but API keys not present")

    try:
        order = ex.create_market_sell_order(symbol, qty)
        fill_price = price
        if isinstance(order, dict):
            fill_price = float(order.get("average") or order.get("price") or price)
        print(f"[LIVE SELL] {symbol} qty={qty:.8f} @ ~{fill_price:.4f}  notional≈${qty*fill_price:.2f}")
        record_trade("live", symbol, "sell", qty, fill_price, "market")
        return qty, fill_price
    except Exception as e:
        print(f"[ERROR SELL] {symbol}: {e}")
        raise

# -------------------------- Exit Logic (SL / TP1 / Trailing / Dust) -------------------------- #

def evaluate_exits(ex, snapshot, coin_balances) -> Tuple[float, float]:
    """Returns tuple (usd_from_sells, usd_dusted)"""
    usd_realized = 0.0
    usd_dusted = 0.0
    for symbol in WHITELIST:
        base = symbol.split("/")[0]
        qty = float(coin_balances.get(base, 0.0))
        if qty <= 0: 
            continue
        price = snapshot[symbol]["bid"] or snapshot[symbol]["last"]
        p = positions.get(symbol, {"avg_cost": 0.0, "qty": qty, "peak": price})
        avg = float(p.get("avg_cost", 0.0)) or price
        peak = float(p.get("peak", price))
        # Maintain peak for trailing
        if price > peak:
            p["peak"] = price
            positions[symbol] = p
            save_json(POS_STATE_FILE, positions)

        value = value_of(symbol, qty, price)

        # Dust sweep
        if value < DUST_MIN_USD and qty > 0:
            print(f"[DUST] {symbol} value=${value:.2f} < {DUST_MIN_USD} → sweeping")
            sold_qty, sold_px = place_market_sell(ex, symbol, qty, price)
            update_after_sell(symbol, sold_qty, sold_px)
            usd_dusted += sold_qty * sold_px
            continue

        # Stop-loss
        if price <= avg * (1.0 - SL_PCT):
            print(f"[SL] {symbol} price {price:.4f} <= avg {avg:.4f} * (1-SL) → exit ALL")
            sold_qty, sold_px = place_market_sell(ex, symbol, qty, price)
            update_after_sell(symbol, sold_qty, sold_px)
            usd_realized += sold_qty * sold_px
            continue

        # Trailing stop (only if in profit)
        if peak > avg and price <= peak * (1.0 - TRAIL_PCT):
            print(f"[TRAIL] {symbol} price {price:.4f} <= peak {peak:.4f} * (1-TRAIL) → exit ALL")
            sold_qty, sold_px = place_market_sell(ex, symbol, qty, price)
            update_after_sell(symbol, sold_qty, sold_px)
            usd_realized += sold_qty * sold_px
            continue

        # Take-profit 1 (partial)
        if price >= avg * (1.0 + TP1_PCT) and TP1_SIZE > 0.0:
            take_qty = round(qty * TP1_SIZE, 8)
            if take_qty > 0:
                print(f"[TP1] {symbol} price {price:.4f} >= avg {avg:.4f} * (1+TP1) → sell {TP1_SIZE*100:.0f}%")
                sold_qty, sold_px = place_market_sell(ex, symbol, take_qty, price)
                update_after_sell(symbol, sold_qty, sold_px)
                usd_realized += sold_qty * sold_px
    return usd_realized, usd_dusted

# -------------------------- Entry Logic (Momentum + Rotation + Caps) -------------------------- #

def pick_candidates(snapshot, held_symbols: List[str]) -> List[str]:
    # Simple quality filter: remove bizarre spreads
    # Rank by 24h % change DESC (chase strength) — you can flip the sign for mean reversion if desired.
    scored = []
    for s, d in snapshot.items():
        if d["last"] <= 0 or d["ask"] <= 0 or d["bid"] <= 0:
            continue
        if d["spread"] > 0.004:  # 0.4% max spread guard
            continue
        scored.append((s, d["pct"], d["last"]))
    # Sort by percentage change descending (momentum)
    scored.sort(key=lambda x: x[1], reverse=True)
    # Prefer not-held first
    ordered = [s for (s,_,_) in scored if s not in held_symbols] + [s for (s,_,_) in scored if s in held_symbols]
    return ordered

def maybe_rotate_to_free_cash(ex, snapshot, usd_free: float, need_usd: float) -> float:
    if usd_free >= need_usd:
        return usd_free
    if not ROTATE_WHEN_CASH_SHORT:
        return usd_free
    # Sell the weakest held position by 24h % to free cash
    held = [(s, snapshot[s]["pct"]) for s in WHITELIST if positions.get(s,{}).get("qty",0) > 0]
    if not held:
        return usd_free
    weakest = sorted(held, key=lambda x: x[1])[0][0]
    base = weakest.split("/")[0]
    qty = positions.get(weakest, {}).get("qty", 0.0)
    if qty <= 0:
        return usd_free
    price = snapshot[weakest]["bid"] or snapshot[weakest]["last"]
    print(f"[ROTATE] Selling weakest {weakest} to free cash for a better entry")
    sold_qty, sold_px = place_market_sell(ex, weakest, qty, price)
    update_after_sell(weakest, sold_qty, sold_px)
    return usd_free + sold_qty * sold_px

def place_entries(ex, snapshot, usd_free: float, held_count: int) -> float:
    if MAX_BUYS_PER_RUN <= 0:
        return 0.0
    slots = want_slots(held_count)
    buys_allowed = min(MAX_BUYS_PER_RUN, max(0, slots))
    if buys_allowed <= 0:
        if ROTATE_WHEN_FULL:
            print("[INFO] At max positions; ROTATE_WHEN_FULL=true not implemented (kept off to reduce churn).")
        return 0.0

    # Respect reserve cash
    spendable = max(0.0, usd_free * (1.0 - RESERVE_CASH_PCT))
    if spendable <= 10.0:
        print("[INFO] Not enough spendable USD after reserve buffer.")
        return 0.0

    # Daily cap: if set, block new buys when cap reached
    cap_left = max(0.0, DAILY_CAP_USD - daily.get("spent", 0.0)) if DAILY_CAP_USD > 0 else float("inf")
    if cap_left <= 5.0:
        print(f"[DAILYCAP] Reached cap: spent={daily.get('spent',0.0)} / cap={DAILY_CAP_USD}. No new buys.")
        return 0.0

    per_buy = spendable / buys_allowed

    ordered = pick_candidates(snapshot, held_symbols=[s for s in WHITELIST if positions.get(s,{}).get("qty",0)>0])
    buys_done = 0
    usd_spent_total = 0.0
    nowt = now_ts()

    for sym in ordered:
        if buys_done >= buys_allowed:
            break
        if not can_buy_symbol(sym, nowt):
            # Cooldown in effect
            continue
        price = snapshot[sym]["ask"] or snapshot[sym]["last"]
        if price <= 0:
            continue

        # Respect daily cap per order
        allowed = min(per_buy, cap_left - usd_spent_total) if DAILY_CAP_USD > 0 else per_buy
        if allowed <= 5.0:
            break

        # If we don't have enough USD, rotate from weakest
        if allowed > usd_free:
            usd_free = maybe_rotate_to_free_cash(ex, snapshot, usd_free, allowed)

        if usd_free < allowed:
            continue

        qty, fill = place_market_buy(ex, sym, allowed, price)
        update_after_buy(sym, qty, fill)
        usd_free -= qty * fill
        usd_spent_total += qty * fill
        buys_done += 1

    if usd_spent_total > 0:
        bump_daily_spent(usd_spent_total)
        print(f"[BUY] Placed {buys_done} buy(s), spent ≈ ${usd_spent_total:.2f} (daily spent now ${daily.get('spent',0.0):.2f})")
    else:
        print("[BUY] No entries this run.")
    return usd_spent_total

# -------------------------- Main -------------------------- #

def main():
    if RUN_SWITCH != "ON":
        print(f"[SKIP] RUN_SWITCH={RUN_SWITCH} → exiting.")
        return

    log_header()
    ex = make_exchange(dry=(DRY_RUN=="ON"))

    # Snapshot & balances
    snapshot = fetch_market_snapshot(ex)
    usd_free, base_balances = fetch_balances(ex)

    # Sync positions qty with live balances (if missing)
    for s in WHITELIST:
        base = s.split("/")[0]
        live_qty = float(base_balances.get(base, 0.0))
        p = positions.get(s, {"avg_cost": 0.0, "qty": 0.0, "peak": snapshot[s]["last"]})
        if abs(p.get("qty", 0.0) - live_qty) > 1e-8:
            p["qty"] = live_qty
            p.setdefault("avg_cost", snapshot[s]["last"])
            p.setdefault("peak", snapshot[s]["last"])
            positions[s] = p
    save_json(POS_STATE_FILE, positions)

    held = [s for s in WHITELIST if positions.get(s,{}).get("qty",0) > 0]
    print(f"[BAL] USD free ≈ ${usd_free:.2f} | Held: {', '.join([f'{s}:{positions[s]['qty']:.6f}' for s in held]) or 'None'}")

    # 1) Exits (SL/TP/Trailing/Dust)
    usd_from_sells, usd_dusted = evaluate_exits(ex, snapshot, base_balances)
    if usd_from_sells or usd_dusted:
        # refresh balances after sells
        usd_free, base_balances = fetch_balances(ex)

    # 2) Entries
    held_count = sum(1 for s in WHITELIST if positions.get(s,{}).get("qty",0) > 0)
    _spent = place_entries(ex, snapshot, usd_free, held_count)

    # 3) KPIs / summary
    portfolio_value = float(usd_free)
    for s in WHITELIST:
        base = s.split("/")[0]
        qty = positions.get(s,{}).get("qty",0.0)
        if qty > 0:
            portfolio_value += qty * snapshot[s]["last"]
    print(f"[SUMMARY] Portfolio est ≈ ${portfolio_value:.2f}  | USD free ≈ ${usd_free:.2f}")
    kpi(f"pv=${portfolio_value:.2f}; usd_free=${usd_free:.2f}; spent_today=${daily.get('spent',0.0):.2f}; buys_this_run=${_spent:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        # Non-zero exit so Actions flags it; logs will show the reason.
        raise
