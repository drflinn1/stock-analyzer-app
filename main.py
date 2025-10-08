#!/usr/bin/env python3
"""
Crypto Live Bot — Guarded + Slack + KPI
- Exchange: Kraken via CCXT
- Exits: STOP_LOSS / TAKE_PROFIT (TP1) / Trailing
- Entries: Momentum, rotation when cash short, rebuy cooldown
- Controls: Daily notional cap, reserve cash, dust sweep
- DRY_RUN simulation with verbose logs
"""
# Guard tokens (do not remove): STOP_LOSS, stop_loss, TAKE_PROFIT, take_profit

from __future__ import annotations
import os, json, math, time, csv, pathlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

# ---------- Optional Slack ----------
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
def slack_post(text: str) -> None:
    if not SLACK_WEBHOOK_URL:
        return
    try:
        import requests
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=6)
    except Exception:
        pass  # never break trading on Slack failure

# ---------- FS / utils ----------
ROOT = pathlib.Path(".")
STATE_DIR = ROOT / ".state"
STATE_DIR.mkdir(exist_ok=True)

def env_str(k, d): return os.environ.get(k, d)
def env_bool(k, d):
    v = os.environ.get(k); 
    return d if v is None else str(v).strip().lower() in ("1","true","yes","on")
def env_float(k, d):
    try: return float(os.environ.get(k, str(d)))
    except: return d
def env_int(k, d):
    try: return int(float(os.environ.get(k, str(d))))
    except: return d

def now_iso(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def today_key(): return datetime.now(timezone.utc).strftime("%Y%m%d")

def load_json(p: pathlib.Path, default):
    try:
        if p.exists(): return json.loads(p.read_text())
    except: pass
    return default

def save_json(p: pathlib.Path, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(p)

def append_csv(p: pathlib.Path, row: List[Any]):
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["ts","event","symbol","side","qty","price","notional","note"])
        w.writerow(row)

# ---------- Config ----------
RUN_SWITCH          = env_str("RUN_SWITCH", "ON")
DRY_RUN             = env_str("DRY_RUN", "ON").upper()
USD_ONLY            = env_bool("USD_ONLY", True)
WHITELIST           = [s.strip() for s in env_str("WHITELIST","BTC/USD,ETH/USD,SOL/USD,DOGE/USD").split(",") if s.strip()]

MAX_POSITIONS       = env_int("MAX_POSITIONS", 6)
MAX_BUYS_PER_RUN    = env_int("MAX_BUYS_PER_RUN", 1)

# Sell guards — keep names for the repo's Sell Logic Guard
SL_PCT              = env_float("SL_PCT", 0.04)     # STOP_LOSS = 4%
TRAIL_PCT           = env_float("TRAIL_PCT", 0.035) # trailing 3.5%
TP1_PCT             = env_float("TP1_PCT", 0.05)    # TAKE_PROFIT (leg 1) = 5%
TP1_SIZE            = env_float("TP1_SIZE", 0.25)   # portion to sell at TP1

RESERVE_CASH_PCT    = env_float("RESERVE_CASH_PCT", 0.10)
DAILY_CAP_USD       = env_float("DAILY_NOTIONAL_CAP_USD", 0.0)   # 0 disables
REBUY_COOLDOWN_MIN  = env_int("REBUY_COOLDOWN_MIN", 30)

ROTATE_WHEN_CASH_SHORT = env_bool("ROTATE_WHEN_CASH_SHORT", True)
ROTATE_WHEN_FULL       = env_bool("ROTATE_WHEN_FULL", False)

DUST_MIN_USD        = env_float("DUST_MIN_USD", 2.0)
BOT_NAME            = env_str("BOT_NAME", "crypto-live")

KRAKEN_API_KEY      = env_str("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET   = env_str("KRAKEN_API_SECRET", "")
HAS_KEYS            = bool(KRAKEN_API_KEY and KRAKEN_API_SECRET)

# ---------- Exchange ----------
try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"[FATAL] ccxt is required: {e}")

def make_exchange(dry: bool):
    params = {"enableRateLimit": True}
    if HAS_KEYS and not dry:
        params.update({"apiKey": KRAKEN_API_KEY, "secret": KRAKEN_API_SECRET})
    return ccxt.kraken(params)

# ---------- State ----------
POS_STATE_FILE = STATE_DIR / "positions.json"     # avg_cost, qty, peak, last_buy_ts
COOLDOWN_FILE  = STATE_DIR / "cooldown.json"      # symbol -> ts
KPI_FILE       = STATE_DIR / "kpi_history.csv"
DAILY_FILE     = STATE_DIR / f"notional_{today_key()}.json"  # {"spent": float}

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

# ---------- Guarded sell helpers (explicit tokens) ----------
def stop_loss_triggered(avg_price: float, last_price: float) -> bool:
    """Return True if STOP_LOSS should fire (guard token present)."""
    return last_price <= avg_price * (1.0 - SL_PCT)

def take_profit_qty(qty: float, avg_price: float, last_price: float) -> float:
    """Return portion to sell for TAKE_PROFIT leg 1 (guard token present)."""
    if last_price >= avg_price * (1.0 + TP1_PCT) and TP1_SIZE > 0:
        return round(qty * TP1_SIZE, 8)
    return 0.0

# ---------- Market / balances ----------
def fetch_market_snapshot(ex):
    ex.load_markets()
    try:
        tickers = ex.fetch_tickers(WHITELIST)
    except Exception:
        time.sleep(1.0)
        tickers = ex.fetch_tickers(WHITELIST)
    data = {}
    for s in WHITELIST:
        t = tickers.get(s, {})
        last = float(t.get("last") or t.get("close") or 0.0)
        pct  = float(t.get("percentage") or 0.0)
        ask  = float(t.get("ask") or last or 0.0)
        bid  = float(t.get("bid") or last or 0.0)
        spread = (ask - bid) / ask if ask else 0.0
        data[s] = {"last": last, "pct": pct, "ask": ask, "bid": bid, "spread": spread}
    return data

def fetch_balances(ex) -> Tuple[float, Dict[str, float]]:
    bal = ex.fetch_free_balance()
    usd = float(bal.get("USD", 0.0)) + float(bal.get("ZUSD", 0.0))
    coins = {}
    for s in WHITELIST:
        base = s.split("/")[0]
        coins[base] = float(bal.get(base, 0.0))
    return usd, coins

# ---------- Logging ----------
def log_header():
    print("========== CONFIG ==========")
    print(f"BOT={BOT_NAME}  RUN_SWITCH={RUN_SWITCH}  DRY_RUN={DRY_RUN}  HAS_KEYS={HAS_KEYS}")
    # include tokens in the line below for the guard:
    print(f"STOP_LOSS={SL_PCT}  TRAIL={TRAIL_PCT}  TAKE_PROFIT={TP1_PCT} x {TP1_SIZE}")
    print(f"MAX_POS={MAX_POSITIONS}  MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}")
    print(f"RESERVE_CASH_PCT={RESERVE_CASH_PCT}  DUST_MIN_USD={DUST_MIN_USD}")
    print(f"DAILY_CAP_USD={DAILY_CAP_USD} (spent_today={daily.get('spent',0.0)})  REBUY_COOLDOWN_MIN={REBUY_COOLDOWN_MIN}")
    print(f"WHITELIST={','.join(WHITELIST)}  USD_ONLY={USD_ONLY}")
    print(f"ROTATE_WHEN_CASH_SHORT={ROTATE_WHEN_CASH_SHORT}  ROTATE_WHEN_FULL={ROTATE_WHEN_FULL}")
    print("============================")

# ---------- Orders ----------
def place_market_buy(ex, symbol: str, usd_alloc: float, price_hint: float) -> Tuple[float,float]:
    price = price_hint if price_hint > 0 else ex.fetch_ticker(symbol)["last"]
    qty = round(max(0.0, usd_alloc) / max(price, 1e-12), 8)
    if qty <= 0: return 0.0, price
    if DRY_RUN == "ON":
        print(f"[SIM BUY] {symbol} qty={qty:.8f} @ ~{price:.4f}  notional=${usd_alloc:.2f}")
        record_trade("sim", symbol, "buy", qty, price, "dry_run")
        slack_post(f"🧪 DRY BUY {symbol} ≈ ${usd_alloc:.2f}")
        return qty, price
    if not HAS_KEYS: raise RuntimeError("Live BUY requested but API keys not present")
    order = ex.create_market_buy_order(symbol, qty)
    fill = float(order.get("average") or order.get("price") or price) if isinstance(order, dict) else price
    print(f"[LIVE BUY] {symbol} qty={qty:.8f} @ ~{fill:.4f}  notional≈${qty*fill:.2f}")
    record_trade("live", symbol, "buy", qty, fill, "market")
    slack_post(f"🟢 BUY {symbol} qty={qty:.6f} ≈ ${qty*fill:.2f}  (SL {SL_PCT*100:.1f}%, TR {TRAIL_PCT*100:.1f}%, TP {TP1_PCT*100:.1f}% x {int(TP1_SIZE*100)}%)")
    return qty, fill

def place_market_sell(ex, symbol: str, qty: float, price_hint: float) -> Tuple[float,float]:
    price = price_hint if price_hint > 0 else ex.fetch_ticker(symbol)["last"]
    qty = round(qty, 8)
    if qty <= 0: return 0.0, price
    if DRY_RUN == "ON":
        print(f"[SIM SELL] {symbol} qty={qty:.8f} @ ~{price:.4f}  notional≈${qty*price:.2f}")
        record_trade("sim", symbol, "sell", qty, price, "dry_run")
        slack_post(f"🧪 DRY SELL {symbol} qty={qty:.6f} ≈ ${qty*price:.2f}")
        return qty, price
    if not HAS_KEYS: raise RuntimeError("Live SELL requested but API keys not present")
    order = ex.create_market_sell_order(symbol, qty)
    fill = float(order.get("average") or order.get("price") or price) if isinstance(order, dict) else price
    print(f"[LIVE SELL] {symbol} qty={qty:.8f} @ ~{fill:.4f}  notional≈${qty*fill:.2f}")
    record_trade("live", symbol, "sell", qty, fill, "market")
    slack_post(f"🔴 SELL {symbol} qty={qty:.6f} ≈ ${qty*fill:.2f}")
    return qty, fill

# ---------- Exits (SL / TP / Trail / Dust) ----------
def value_usd(qty: float, price: float) -> float:
    return float(qty) * float(price)

def update_after_buy(sym: str, qty: float, price: float):
    p = positions.get(sym, {"avg_cost": 0.0, "qty": 0.0, "peak": 0.0, "last_buy_ts": 0.0})
    new_qty = p["qty"] + qty
    new_cost = (p["avg_cost"]*p["qty"] + qty*price) / new_qty if new_qty > 0 else price
    p["qty"], p["avg_cost"], p["peak"] = new_qty, new_cost, max(p.get("peak", 0.0), price)
    p["last_buy_ts"] = time.time()
    positions[sym] = p
    cooldown[sym] = p["last_buy_ts"]
    save_json(POS_STATE_FILE, positions); save_json(COOLDOWN_FILE, cooldown)

def update_after_sell(sym: str, sell_qty: float, price: float):
    p = positions.get(sym, {"avg_cost": 0.0, "qty": 0.0, "peak": 0.0, "last_buy_ts": 0.0})
    p["qty"] = max(0.0, p["qty"] - sell_qty)
    p["peak"] = max(p.get("peak", 0.0), price)
    if p["qty"] <= 1e-9:
        positions.pop(sym, None); cooldown.pop(sym, None)
    else:
        positions[sym] = p
    save_json(POS_STATE_FILE, positions); save_json(COOLDOWN_FILE, cooldown)

def evaluate_exits(ex, snapshot, coin_balances) -> Tuple[float, float]:
    usd_realized = 0.0; usd_dusted = 0.0
    for symbol in WHITELIST:
        base = symbol.split("/")[0]
        qty = float(coin_balances.get(base, 0.0))
        if qty <= 0: 
            continue
        price = snapshot[symbol]["bid"] or snapshot[symbol]["last"]
        p = positions.get(symbol, {"avg_cost": price, "qty": qty, "peak": price})
        avg, peak = float(p.get("avg_cost", price)), float(p.get("peak", price))

        # keep peak for trailing
        if price > peak:
            p["peak"] = price; positions[symbol] = p; save_json(POS_STATE_FILE, positions)

        value = value_usd(qty, price)

        # Dust
        if value < DUST_MIN_USD:
            print(f"[DUST] {symbol} value=${value:.2f} < {DUST_MIN_USD} → sweeping")
            sqty, spx = place_market_sell(ex, symbol, qty, price)
            update_after_sell(symbol, sqty, spx); usd_dusted += sqty * spx; continue

        # STOP_LOSS
        if stop_loss_triggered(avg, price):
            print(f"[SL] {symbol} price {price:.4f} <= avg {avg:.4f} * (1-SL) → exit ALL")
            sqty, spx = place_market_sell(ex, symbol, qty, price)
            update_after_sell(symbol, sqty, spx); usd_realized += sqty * spx; continue

        # Trailing (only if in profit)
        if peak > avg and price <= peak * (1.0 - TRAIL_PCT):
            print(f"[TRAIL] {symbol} price {price:.4f} <= peak {peak:.4f} * (1-TRAIL) → exit ALL")
            sqty, spx = place_market_sell(ex, symbol, qty, price)
            update_after_sell(symbol, sqty, spx); usd_realized += sqty * spx; continue

        # TAKE_PROFIT partial
        tp_qty = take_profit_qty(qty, avg, price)
        if tp_qty > 0:
            print(f"[TP1] {symbol} price {price:.4f} >= avg {avg:.4f} * (1+TP1) → sell {TP1_SIZE*100:.0f}%")
            sqty, spx = place_market_sell(ex, symbol, tp_qty, price)
            update_after_sell(symbol, sqty, spx); usd_realized += sqty * spx
    return usd_realized, usd_dusted

# ---------- Entries ----------
def want_slots(current_positions: int) -> int:
    return max(0, MAX_POSITIONS - current_positions)

def can_buy_symbol(sym: str, nowt: float) -> bool:
    last = cooldown.get(sym, 0.0)
    return not (last and (nowt - last) < (REBUY_COOLDOWN_MIN * 60))

def pick_candidates(snapshot, held_symbols: List[str]) -> List[str]:
    scored = []
    for s, d in snapshot.items():
        if d["last"] <= 0 or d["ask"] <= 0 or d["bid"] <= 0: continue
        if d["spread"] > 0.004: continue
        scored.append((s, d["pct"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for (s,_) in scored if s not in held_symbols] + [s for (s,_) in scored if s in held_symbols]

def maybe_rotate_to_free_cash(ex, snapshot, usd_free: float, need_usd: float) -> float:
    if usd_free >= need_usd or not ROTATE_WHEN_CASH_SHORT: return usd_free
    held = [(s, snapshot[s]["pct"]) for s in WHITELIST if positions.get(s,{}).get("qty",0) > 0]
    if not held: return usd_free
    weakest = sorted(held, key=lambda x: x[1])[0][0]
    qty = positions.get(weakest, {}).get("qty", 0.0)
    if qty <= 0: return usd_free
    price = snapshot[weakest]["bid"] or snapshot[weakest]["last"]
    print(f"[ROTATE] Selling weakest {weakest} to free cash")
    sqty, spx = place_market_sell(ex, weakest, qty, price)
    update_after_sell(weakest, sqty, spx)
    return usd_free + sqty * spx

def place_entries(ex, snapshot, usd_free: float, held_count: int) -> float:
    if MAX_BUYS_PER_RUN <= 0: return 0.0
    slots = want_slots(held_count)
    buys_allowed = min(MAX_BUYS_PER_RUN, max(0, slots))
    if buys_allowed <= 0:
        if ROTATE_WHEN_FULL:
            print("[INFO] At max positions; rotation-when-full intentionally disabled to reduce churn.")
        return 0.0

    spendable = max(0.0, usd_free * (1.0 - RESERVE_CASH_PCT))
    if spendable <= 10.0:
        print("[INFO] Not enough spendable USD after reserve buffer."); return 0.0

    cap_left = max(0.0, DAILY_CAP_USD - daily.get("spent", 0.0)) if DAILY_CAP_USD > 0 else float("inf")
    if cap_left <= 5.0:
        print(f"[DAILYCAP] Reached cap: spent={daily.get('spent',0.0)} / cap={DAILY_CAP_USD}. No new buys.")
        return 0.0

    per_buy = spendable / buys_allowed
    ordered = pick_candidates(snapshot, held_symbols=[s for s in WHITELIST if positions.get(s,{}).get("qty",0)>0])
    buys_done = 0; usd_spent_total = 0.0; nowt = time.time()

    for sym in ordered:
        if buys_done >= buys_allowed: break
        if not can_buy_symbol(sym, nowt): continue
        price = snapshot[sym]["ask"] or snapshot[sym]["last"]
        if price <= 0: continue

        allowed = min(per_buy, cap_left - usd_spent_total) if DAILY_CAP_USD > 0 else per_buy
        if allowed <= 5.0: break

        if allowed > usd_free:
            usd_free = maybe_rotate_to_free_cash(ex, snapshot, usd_free, allowed)
        if usd_free < allowed: continue

        qty, fill = place_market_buy(ex, sym, allowed, price)
        update_after_buy(sym, qty, fill)
        usd_free -= qty * fill; usd_spent_total += qty * fill; buys_done += 1

    if usd_spent_total > 0:
        bump_daily_spent(usd_spent_total)
        print(f"[BUY] Placed {buys_done} buy(s), spent ≈ ${usd_spent_total:.2f} (daily spent now ${daily.get('spent',0.0):.2f})")
    else:
        print("[BUY] No entries this run.")
    return usd_spent_total

# ---------- Main ----------
def main():
    if RUN_SWITCH != "ON":
        print(f"[SKIP] RUN_SWITCH={RUN_SWITCH} → exiting."); return

    print("========== RUN =========="); log_header()
    ex = make_exchange(dry=(DRY_RUN=="ON"))

    snapshot = fetch_market_snapshot(ex)
    usd_free, base_balances = fetch_balances(ex)

    # sync positions with live balances
    for s in WHITELIST:
        base = s.split("/")[0]
        live_qty = float(base_balances.get(base, 0.0))
        p = positions.get(s, {"avg_cost": snapshot[s]["last"], "qty": 0.0, "peak": snapshot[s]["last"]})
        if abs(p.get("qty", 0.0) - live_qty) > 1e-9:
            p["qty"] = live_qty; positions[s] = p
    save_json(POS_STATE_FILE, positions)

    held = [s for s in WHITELIST if positions.get(s,{}).get("qty",0) > 0]
    held_str = ", ".join([f"{s}:{positions[s]['qty']:.6f}" for s in held]) if held else "None"
    print(f"[BAL] USD free ≈ ${usd_free:.2f} | Held: {held_str}")

    # 1) exits
    usd_from_sells, usd_dusted = evaluate_exits(ex, snapshot, base_balances)
    if usd_from_sells or usd_dusted:
        usd_free, base_balances = fetch_balances(ex)

    # 2) entries
    held_count = sum(1 for s in WHITELIST if positions.get(s,{}).get("qty",0) > 0)
    spent = place_entries(ex, snapshot, usd_free, held_count)

    # 3) KPIs / summary
    portfolio_value = float(usd_free)
    for s in WHITELIST:
        qty = positions.get(s,{}).get("qty",0.0)
        if qty > 0: portfolio_value += qty * snapshot[s]["last"]
    msg = f"{'🧪' if DRY_RUN=='ON' else '✅'} {BOT_NAME} summary: PV≈${portfolio_value:.2f} | USD≈${usd_free:.2f} | spent_today=${daily.get('spent',0.0):.2f}"
    print(f"[SUMMARY] {msg}"); slack_post(msg)
    kpi(f"pv=${portfolio_value:.2f}; usd_free=${usd_free:.2f}; spent_today=${daily.get('spent',0.0):.2f}; buys_this_run=${spent:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}"); raise
