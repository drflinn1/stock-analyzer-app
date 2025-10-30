# -*- coding: utf-8 -*-
"""
Crypto engine:

- Builds ccxt Kraken exchange
- Discovers candidates (universe scan + optional whitelist)
- Enforces gates (cash, positions, thresholds)
- Places market BUY/SELL (BUY only by default here)
- Writes diagnostics to .state/buy_gates.md for quick debugging

Env knobs (read from GitHub Variables/Secrets):
  DRY_RUN=ON|OFF
  MIN_BUY_USD=10
  MAX_BUYS_PER_RUN=2
  MAX_POSITIONS=3
  RESERVE_CASH_PCT=0
  UNIVERSE_TOP_K=25
  MIN_24H_PCT=0
  MIN_BASE_VOL_USD=10000
  ROTATE_WHEN_FULL=true|false
  ROTATE_WHEN_CASH_SHORT=true|false
  WHITELIST="BTC/USD,DOGE/USD,..."   (optional)
  KRAKEN_API_KEY, KRAKEN_API_SECRET
"""

import os
import math
from typing import Dict, List, Tuple

import ccxt
from pathlib import Path

# ---------- Helpers ----------
STABLES = {"USD", "USDT", "USDC", "EUR", "GBP"}
EXCLUDE_TICKERS = {"SPX", "PUMP", "BABY", "ALKIMI"}  # expand as needed

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
BUY_GATES_MD = STATE_DIR / "buy_gates.md"


def _bool(s: str, default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def build_exchange(api_key: str, api_secret: str, dry_run: bool) -> ccxt.Exchange:
    ex = ccxt.kraken({
        "apiKey": api_key or "",
        "secret": api_secret or "",
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    ex.load_markets()
    return ex


def symbol_to_usd_market(ex: ccxt.Exchange, symbol: str) -> Dict:
    m = ex.markets.get(symbol)
    if m:
        return m
    return None


def _is_bad_coin(symbol: str) -> bool:
    base, quote = symbol.split("/")
    if base.upper() in EXCLUDE_TICKERS:
        return True
    if quote.upper() != "USD":
        return True
    if base.upper() in STABLES:  # skip stables on base side
        return True
    return False


def pick_candidates(ex: ccxt.Exchange, top_k: int = 25) -> List[Dict]:
    """
    Rank USD pairs by 24h percent change (descending) with liquidity filter.
    Returns list of dicts: {symbol, price, change24h, quoteVolUsd}
    """
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

        quote_vol = t.get("quoteVolume") or 0.0
        if quote_vol is None:
            quote_vol = 0.0
        if float(quote_vol) < 10000:  # tiny coins out
            continue

        out.append({
            "symbol": sym,
            "price": float(t.get("last") or t.get("close") or 0.0),
            "change24h": float(change),
            "quoteVolUsd": float(quote_vol),
        })

    out.sort(key=lambda r: (r["change24h"], r["quoteVolUsd"]), reverse=True)
    return out[:max(1, int(top_k))]


def fetch_positions_snapshot(ex: ccxt.Exchange):
    """
    Return list of spot holdings (>0) as:
    {symbol, base, base_qty, price, usd_value}
    """
    balances = ex.fetch_balance()
    prices = ex.fetch_tickers()
    positions = []
    for sym, m in ex.markets.items():
        if _is_bad_coin(sym):
            continue
        if not sym.endswith("/USD"):
            continue
        base = m["base"]
        free = balances.get(base, {}).get("free")
        total = balances.get(base, {}).get("total")
        qty = None
        if total:
            qty = float(total)
        elif free:
            qty = float(free)
        if not qty or qty <= 0:
            continue

        price = prices.get(sym, {}).get("last") or prices.get(sym, {}).get("close") or 0.0
        price = float(price or 0.0)
        usd_value = qty * price
        positions.append({
            "symbol": sym,
            "base": base,
            "base_qty": qty,
            "price": price,
            "usd_value": usd_value,
        })
    return positions


def get_cash_balance_usd(ex: ccxt.Exchange) -> float:
    bal = ex.fetch_balance()
    usd = bal.get("USD", {})
    free = float(usd.get("free") or 0.0)
    total = float(usd.get("total") or free)
    return max(free, total)


# ---------- Order wrappers ----------
def _ensure_min_notional(ex: ccxt.Exchange, symbol: str, spend_usd: float) -> float:
    m = ex.markets[symbol]
    min_cost = float(m.get("limits", {}).get("cost", {}).get("min") or 0.0)
    return max(spend_usd, min_cost)


def place_market_buy(ex: ccxt.Exchange, symbol: str, spend_usd: float, dry_run: bool):
    spend_usd = _ensure_min_notional(ex, symbol, spend_usd)
    price = float(ex.fetch_ticker(symbol)["last"])
    amount = spend_usd / max(price, 1e-9)
    amount = ex.amount_to_precision(symbol, amount)
    if dry_run:
        print(f"[order][dry-run] BUY {symbol} notional≈${spend_usd:.2f} qty≈{amount}")
        return {"id": "dry-run", "symbol": symbol, "amount": amount, "cost": spend_usd}
    print(f"[order] BUY {symbol} notional≈${spend_usd:.2f} qty≈{amount}")
    return ex.create_market_buy_order(symbol, float(amount))


# ---------- Trading loop ----------
def parse_whitelist(env_val: str) -> List[str]:
    if not env_val:
        return []
    return [s.strip().upper() for s in env_val.split(",") if s.strip()]


def write_buy_diag(md: str) -> None:
    BUY_GATES_MD.write_text(md)


def run_trading_loop() -> int:
    # --- Read env knobs ---
    DRY_RUN = os.getenv("DRY_RUN", "ON").upper()
    dry_run_flag = (DRY_RUN != "OFF")

    api_key = os.getenv("KRAKEN_API_KEY", "")
    api_secret = os.getenv("KRAKEN_API_SECRET", "")

    MIN_BUY_USD = _env_float("MIN_BUY_USD", 10.0)
    MAX_BUYS_PER_RUN = _env_int("MAX_BUYS_PER_RUN", 2)
    MAX_POSITIONS = _env_int("MAX_POSITIONS", 3)
    RESERVE_CASH_PCT = _env_float("RESERVE_CASH_PCT", 0.0)

    UNIVERSE_TOP_K = _env_int("UNIVERSE_TOP_K", 25)
    MIN_24H_PCT = _env_float("MIN_24H_PCT", 0.0)
    MIN_BASE_VOL_USD = _env_float("MIN_BASE_VOL_USD", 10000.0)

    ROTATE_WHEN_FULL = _bool(os.getenv("ROTATE_WHEN_FULL", "true"), True)
    ROTATE_WHEN_CASH_SHORT = _bool(os.getenv("ROTATE_WHEN_CASH_SHORT", "true"), True)

    WHITELIST = parse_whitelist(os.getenv("WHITELIST", ""))

    # --- Build exchange ---
    ex = build_exchange(api_key, api_secret, dry_run_flag)

    # --- Current state ---
    positions = fetch_positions_snapshot(ex)
    held_symbols = {p["symbol"].upper() for p in positions}
    pos_count = len(held_symbols)
    cash_usd = get_cash_balance_usd(ex)

    # spendable cash after reserve
    reserve_amt = cash_usd * (RESERVE_CASH_PCT / 100.0)
    spendable = max(0.0, cash_usd - reserve_amt)

    # --- Discovery ---
    all_candidates = pick_candidates(ex, UNIVERSE_TOP_K)

    # Apply whitelist if provided
    if WHITELIST:
        all_candidates = [c for c in all_candidates if c["symbol"].upper() in WHITELIST]

    # Apply thresholds
    filtered = []
    for c in all_candidates:
        if c["change24h"] < MIN_24H_PCT:
            continue
        if c["quoteVolUsd"] < MIN_BASE_VOL_USD:
            continue
        filtered.append(c)

    # Remove ones we already hold
    filtered_not_held = [c for c in filtered if c["symbol"].upper() not in held_symbols]

    # --- Gates decisions ---
    reasons = []
    if pos_count >= MAX_POSITIONS:
        reasons.append(f"positions_full: {pos_count} >= MAX_POSITIONS({MAX_POSITIONS})")

    if spendable < MIN_BUY_USD:
        reasons.append(f"cash_short: spendable ${spendable:.2f} < MIN_BUY_USD {MIN_BUY_USD:.2f}")

    if not filtered_not_held:
        reasons.append("no_candidates_passed_filters")

    # Write diagnostic
    md = []
    md.append("## BUY_DIAGNOSTIC\n")
    md.append(f"- DRY_RUN: {DRY_RUN}\n")
    md.append(f"- Cash USD: {cash_usd:.2f} (spendable {spendable:.2f} after {RESERVE_CASH_PCT:.1f}% reserve)\n")
    md.append(f"- Positions held: {pos_count} / MAX_POSITIONS {MAX_POSITIONS}\n")
    md.append(f"- Universe checked: {len(all_candidates)} (top_k={UNIVERSE_TOP_K})\n")
    md.append(f"- After filters: {len(filtered)} (MIN_24H_PCT={MIN_24H_PCT}, MIN_BASE_VOL_USD={MIN_BASE_VOL_USD})\n")
    md.append(f"- After not-held filter: {len(filtered_not_held)}\n")
    if WHITELIST:
        md.append(f"- Whitelist active: {','.join(WHITELIST)}\n")
    if reasons:
        md.append(f"- Blockers: {', '.join(reasons)}\n")
    write_buy_diag("".join(md))

    # --- Execute buys if allowed ---
    if reasons and not (ROTATE_WHEN_FULL and pos_count >= MAX_POSITIONS):
        # If we’re full, ROTATE_WHEN_FULL could sell to make room — not implemented here.
        print("[engine] No buy action - " + ", ".join(reasons))
        return 0

    buys_done = 0
    per_buy = MIN_BUY_USD
    max_buys_allowed = min(MAX_BUYS_PER_RUN, max(0, MAX_POSITIONS - pos_count))

    for c in filtered_not_held:
        if buys_done >= max_buys_allowed:
            break
        if spendable < per_buy:
            if ROTATE_WHEN_CASH_SHORT:
                # Could sell weakest to free cash; out-of-scope for this engine stub
                print("[engine] Cash short; rotation not implemented in this stub.")
            else:
                print("[engine] Cash short; skipping buys.")
            break

        sym = c["symbol"]
        try:
            place_market_buy(ex, sym, per_buy, dry_run_flag)
            spendable -= per_buy
            buys_done += 1
        except Exception as e:
            print(f"[engine] BUY failed for {sym}: {e}")

    if buys_done == 0:
        print("[engine] No buys executed this run.")
    else:
        print(f"[engine] Buys executed: {buys_done}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_trading_loop())
