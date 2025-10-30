# -*- coding: utf-8 -*-
"""
Crypto engine with tight SELL GUARD:
- Universe scan + whitelist
- BUY up to limits
- SELL losers quickly (hard stop) and trail winners (trailing stop)
- Quick take-profit to bank small gains

Env knobs (GitHub Variables):
  DRY_RUN=ON|OFF
  MIN_BUY_USD=10
  MAX_BUYS_PER_RUN=2
  MAX_POSITIONS=6
  RESERVE_CASH_PCT=0
  UNIVERSE_TOP_K=25
  MIN_24H_PCT=0
  MIN_BASE_VOL_USD=10000
  WHITELIST="BTC/USD,DOGE/USD,..."

  SELL_HARD_STOP_PCT=3       # sell if current <= entry*(1-3%)
  SELL_TRAIL_PCT=2           # sell if current <= high_since_entry*(1-2%)
  SELL_TAKE_PROFIT_PCT=5     # optional quick bank at +5%

  MIN_SELL_USD=10
  DUST_MIN_USD=2
  DUST_SKIP_STABLES=true

Secrets:
  KRAKEN_API_KEY, KRAKEN_API_SECRET
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import json
import math
import time

# ---------- State files ----------
STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
LEDGER = STATE_DIR / "positions.json"      # entry/highwater ledger (bot-tracked)
BUY_GATES_MD = STATE_DIR / "buy_gates.md"

# ---------- Static filters ----------
STABLES = {"USD", "USDT", "USDC", "EUR", "GBP"}
EXCLUDE_TICKERS = {"SPX", "PUMP", "BABY", "ALKIMI"}  # expand as needed


# ---------- small helpers ----------
def _bool(s: str, default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return int(default)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------- exchange ----------
def build_exchange(api_key: str, api_secret: str, dry_run: bool) -> ccxt.Exchange:
    ex = ccxt.kraken({
        "apiKey": api_key or "",
        "secret": api_secret or "",
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    ex.load_markets()
    return ex


# ---------- market utils ----------
def _is_bad_coin(symbol: str) -> bool:
    base, quote = symbol.split("/")
    if base.upper() in EXCLUDE_TICKERS:
        return True
    if quote.upper() != "USD":
        return True
    if base.upper() in STABLES:
        return True
    return False


def pick_candidates(ex: ccxt.Exchange, top_k: int) -> List[Dict]:
    tickers = ex.fetch_tickers()
    out: List[Dict] = []
    for sym, t in tickers.items():
        if "/" not in sym or not sym.endswith("/USD"):
            continue
        if _is_bad_coin(sym):
            continue
        change = t.get("percentage")
        if change is None:
            last = t.get("last") or t.get("close")
            open_ = t.get("open")
            if last and open_:
                change = (last - open_) / open_ * 100.0
            else:
                continue
        qv = float(t.get("quoteVolume") or 0.0)
        if qv < 10000:
            continue
        out.append({
            "symbol": sym,
            "price": float(t.get("last") or t.get("close") or 0.0),
            "change24h": float(change),
            "quoteVolUsd": qv,
        })
    out.sort(key=lambda r: (r["change24h"], r["quoteVolUsd"]), reverse=True)
    return out[:max(1, int(top_k))]


def fetch_positions_snapshot(ex: ccxt.Exchange):
    balances = ex.fetch_balance()
    prices = ex.fetch_tickers()
    positions = []
    for sym, m in ex.markets.items():
        if _is_bad_coin(sym) or not sym.endswith("/USD"):
            continue
        base = m["base"]
        qty = float((balances.get(base, {}) or {}).get("total") or 0.0)
        if qty <= 0:
            continue
        price = float((prices.get(sym, {}) or {}).get("last") or 0.0)
        usd_value = qty * price
        positions.append({"symbol": sym, "base": base, "qty": qty, "price": price, "usd_value": usd_value})
    return positions


def get_cash_balance_usd(ex: ccxt.Exchange) -> float:
    bal = ex.fetch_balance()
    usd = bal.get("USD", {})
    free = float(usd.get("free") or 0.0)
    total = float(usd.get("total") or free)
    return max(free, total)


# ---------- orders ----------
def _ensure_min_notional(ex: ccxt.Exchange, symbol: str, spend_usd: float) -> float:
    m = ex.markets[symbol]
    min_cost = float(m.get("limits", {}).get("cost", {}).get("min") or 0.0)
    return max(spend_usd, min_cost)


def buy_market(ex: ccxt.Exchange, symbol: str, notional: float, dry_run: bool):
    notional = _ensure_min_notional(ex, symbol, notional)
    last = float(ex.fetch_ticker(symbol)["last"])
    qty = ex.amount_to_precision(symbol, notional / max(last, 1e-9))
    if dry_run:
        print(f"[order][dry-run] BUY {symbol} ${notional:.2f} qty≈{qty}")
        return {"id": "dry", "symbol": symbol, "amount": float(qty), "cost": notional, "price": last}
    print(f"[order] BUY {symbol} ${notional:.2f} qty≈{qty}")
    return ex.create_market_buy_order(symbol, float(qty))


def sell_market(ex: ccxt.Exchange, symbol: str, qty: float, dry_run: bool):
    qty_p = ex.amount_to_precision(symbol, qty)
    if dry_run:
        print(f"[order][dry-run] SELL {symbol} qty≈{qty_p}")
        return {"id": "dry", "symbol": symbol, "amount": float(qty_p)}
    print(f"[order] SELL {symbol} qty≈{qty_p}")
    return ex.create_market_sell_order(symbol, float(qty_p))


# ---------- local ledger (for entry/highest) ----------
def load_ledger() -> Dict[str, Dict]:
    if LEDGER.exists():
        try:
            return json.loads(LEDGER.read_text() or "{}")
        except Exception:
            return {}
    return {}


def save_ledger(d: Dict[str, Dict]) -> None:
    LEDGER.write_text(json.dumps(d, indent=2))


def ensure_entry(ledger: Dict[str, Dict], symbol: str, entry_price: float) -> None:
    rec = ledger.get(symbol.upper())
    if not rec:
        ledger[symbol.upper()] = {"entry": float(entry_price), "high": float(entry_price), "added": _now()}
    else:
        # if new buy at higher/lower price, reset entry to weighted blend? keep simple: keep min(entry, new price)
        if entry_price > 0:
            rec["entry"] = float(rec.get("entry", entry_price))
            rec["high"] = float(max(rec.get("high", entry_price), entry_price))


def update_high(ledger: Dict[str, Dict], symbol: str, current_price: float) -> None:
    rec = ledger.get(symbol.upper())
    if not rec:
        ledger[symbol.upper()] = {"entry": float(current_price), "high": float(current_price), "added": _now()}
    else:
        rec["high"] = float(max(rec.get("high", current_price), current_price))


# ---------- core trading loop ----------
def run_trading_loop() -> int:
    # env
    DRY_RUN = os.getenv("DRY_RUN", "ON").upper()
    dry = DRY_RUN != "OFF"

    api_key = os.getenv("KRAKEN_API_KEY", "")
    api_secret = os.getenv("KRAKEN_API_SECRET", "")

    MIN_BUY_USD = _env_float("MIN_BUY_USD", 10.0)
    MAX_BUYS_PER_RUN = _env_int("MAX_BUYS_PER_RUN", 2)
    MAX_POSITIONS = _env_int("MAX_POSITIONS", 6)
    RESERVE_CASH_PCT = _env_float("RESERVE_CASH_PCT", 0.0)

    UNIVERSE_TOP_K = _env_int("UNIVERSE_TOP_K", 25)
    MIN_24H_PCT = _env_float("MIN_24H_PCT", 0.0)
    MIN_BASE_VOL_USD = _env_float("MIN_BASE_VOL_USD", 10000.0)

    SELL_HARD_STOP_PCT = _env_float("SELL_HARD_STOP_PCT", 3.0)      # tighter
    SELL_TRAIL_PCT = _env_float("SELL_TRAIL_PCT", 2.0)              # tighter
    SELL_TAKE_PROFIT_PCT = _env_float("SELL_TAKE_PROFIT_PCT", 5.0)  # quick bank

    WHITELIST = [s.strip().upper() for s in (os.getenv("WHITELIST", "") or "").split(",") if s.strip()]

    ex = build_exchange(api_key, api_secret, dry)

    # state
    ledger = load_ledger()
    positions = fetch_positions_snapshot(ex)
    held_syms = {p["symbol"].upper() for p in positions}
    pos_count = len(held_syms)

    # ---------- SELL GUARD: scan all holdings ----------
    sells_done = 0
    for p in positions:
        sym = p["symbol"].upper()
        price = float(p["price"] or 0.0)
        qty = float(p["qty"] or 0.0)
        if qty <= 0 or price <= 0:
            continue

        # update local high-water
        update_high(ledger, sym, price)
        entry = float(ledger.get(sym, {}).get("entry", price))
        high = float(ledger.get(sym, {}).get("high", price))

        change_from_entry = (price / entry - 1.0) * 100.0 if entry > 0 else 0.0
        draw_from_high = (price / high - 1.0) * 100.0 if high > 0 else 0.0

        # quick take-profit
        if change_from_entry >= SELL_TAKE_PROFIT_PCT:
            print(f"[SELL] TAKE_PROFIT {sym} @ {price:.6f} (+{change_from_entry:.2f}%)")
            sell_market(ex, sym, qty, dry)
            sells_done += 1
            # remove from ledger; new buys will recreate
            ledger.pop(sym, None)
            continue

        # trailing stop (from high)
        if draw_from_high <= -abs(SELL_TRAIL_PCT):
            print(f"[SELL] TRAIL {sym} @ {price:.6f} (drawdown {draw_from_high:.2f}% from high {high:.6f})")
            sell_market(ex, sym, qty, dry)
            sells_done += 1
            ledger.pop(sym, None)
            continue

        # hard stop (from entry)
        if change_from_entry <= -abs(SELL_HARD_STOP_PCT):
            print(f"[SELL] STOP_LOSS {sym} @ {price:.6f} ({change_from_entry:.2f}% vs entry {entry:.6f})")
            sell_market(ex, sym, qty, dry)
            sells_done += 1
            ledger.pop(sym, None)
            continue

    # save ledger after sell scans
    save_ledger(ledger)

    # refresh counts after potential sells
    if sells_done:
        positions = fetch_positions_snapshot(ex)
        held_syms = {p["symbol"].upper() for p in positions}
        pos_count = len(held_syms)

    # ---------- BUY side ----------
    cash = get_cash_balance_usd(ex)
    spendable = max(0.0, cash - cash * (RESERVE_CASH_PCT / 100.0))

    # discovery
    universe = pick_candidates(ex, UNIVERSE_TOP_K)
    if WHITELIST:
        universe = [c for c in universe if c["symbol"].upper() in WHITELIST]
    filtered = [c for c in universe if c["change24h"] >= MIN_24H_PCT and c["quoteVolUsd"] >= MIN_BASE_VOL_USD]
    filtered = [c for c in filtered if c["symbol"].upper() not in held_syms]  # avoid adding duplicates

    # quick diag
    BUY_GATES_MD.write_text(
        "\n".join([
            "## BUY_DIAGNOSTIC",
            f"- time: {_now()}",
            f"- cash: {cash:.2f} spendable: {spendable:.2f}",
            f"- positions: {pos_count}",
            f"- candidates: total={len(universe)} after_filters={len(filtered)}",
            f"- whitelist: {'on' if WHITELIST else 'off'}",
        ]) + "\n"
    )

    buys_done = 0
    max_new = max(0, MAX_POSITIONS - pos_count)
    max_slots = min(MAX_BUYS_PER_RUN, max_new)

    for c in filtered:
        if buys_done >= max_slots:
            break
        if spendable < MIN_BUY_USD:
            break
        sym = c["symbol"].upper()

        # try to buy; if region-restricted Kraken will reject
        try:
            order = buy_market(ex, sym, MIN_BUY_USD, dry)
            # Register entry in ledger so trailing has a baseline
            price = float(ex.fetch_ticker(sym)["last"])
            ensure_entry(ledger, sym, price)
            save_ledger(ledger)
            buys_done += 1
            spendable -= MIN_BUY_USD
        except Exception as e:
            print(f"[engine] BUY failed for {sym}: {e}")

    if buys_done == 0 and sells_done == 0:
        print("[engine] No buys executed this run.")
    else:
        if buys_done:
            print(f"[engine] Buys executed: {buys_done}")
        if sells_done:
            print(f"[engine] Sells executed: {sells_done}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_trading_loop())
