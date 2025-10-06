# main.py — Crypto Live (Kraken) with Guard Pack + TAKE_PROFIT / STOP_LOSS / TRAILING_STOP
# - Robust env-var credential mapping (KRAKEN_API_KEY/SECRET/OTP and fallbacks)
# - Dust sweeping (≤ DUST_MIN_USD)
# - Auto Top-K selection by 24h % change (filters out stables/fiat/index)
# - Rotation (when full or cash-short), max positions, per-run buy cap
# - TAKE_PROFIT, STOP_LOSS, TRAILING_STOP using lightweight local state
# - KPI summary + CSV history, color-tagged logs
#
# NOTE: For cost basis we maintain a tiny state file .state/pos_state.json that
# tracks per-symbol avg_price, trail_max, and last_qty. If we first see an open
# position without prior state, we seed avg_price from current market price.

from __future__ import annotations
import os, sys, time, json, csv, math, traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

# -------- Dependencies -------- #
try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# -------- Helpers / ENV -------- #
def getenv_any(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return v
    return default

def as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1","true","yes","on"):
        return True
    if s in ("0","false","no","off"):
        return False
    return default

def as_float(v: Optional[str], default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default

def as_int(v: Optional[str], default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return default

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC (%Y_%m%d)")

def log(tag: str, msg: str) -> None:
    # tags: OK, WARN, ERR, KPI, SUM
    color = {"OK":"\033[92m","WARN":"\033[93m","ERR":"\033[91m","KPI":"\033[96m","SUM":"\033[95m"}.get(tag, "")
    endc = "\033[0m" if color else ""
    print(f"{color}[{tag}]{endc} {msg}", flush=True)

# -------- ENV Defaults -------- #
DRY_RUN           = as_bool(os.getenv("DRY_RUN"), True)
RUN_SWITCH        = getenv_any("RUN_SWITCH", default="on").lower() == "on"

EXCHANGE          = getenv_any("EXCHANGE", default="kraken").lower()
BASE_QUOTE        = getenv_any("BASE_QUOTE", default="USD").upper()

MAX_POSITIONS     = as_int(os.getenv("MAX_POSITIONS"), 12)
MAX_BUYS_PER_RUN  = as_int(os.getenv("MAX_BUYS_PER_RUN"), 1)
ROTATE_WHEN_FULL  = as_bool(os.getenv("ROTATE_WHEN_FULL"), True)
ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)

MIN_NOTIONAL_USD  = as_float(os.getenv("MIN_NOTIONAL_USD"), 5.0)
DUST_MIN_USD      = as_float(os.getenv("DUST_MIN_USD"), 2.0)
RESERVE_CASH_PCT  = as_float(os.getenv("RESERVE_CASH_PCT"), 5.0)

TP_PCT            = as_float(os.getenv("TP_PCT"), 2.0)   # TAKE_PROFIT %
SL_PCT            = as_float(os.getenv("SL_PCT"), 3.5)   # STOP_LOSS %
TRAIL_PCT         = as_float(os.getenv("TRAIL_PCT"), 1.2) # TRAILING_STOP %

DAILY_LOSS_CAP    = as_float(os.getenv("DAILY_LOSS_CAP"), 3.0)
AUTO_PAUSE_ON_ERROR = as_bool(os.getenv("AUTO_PAUSE_ON_ERROR"), True)

LOG_LEVEL         = getenv_any("LOG_LEVEL", default="INFO")
KPI_CSV_PATH      = getenv_any("KPI_CSV_PATH", default=".state/kpi_history.csv")
STATE_DIR         = getenv_any("STATE_DIR", default=".state")
POS_STATE_PATH    = os.path.join(STATE_DIR, "pos_state.json")

TIMEFRAME         = getenv_any("TIMEFRAME", default="15m")
TOPK              = as_int(os.getenv("TOPK"), 6)
COOL_DOWN_MINUTES = as_int(os.getenv("COOL_DOWN_MINUTES"), 30)
SHOW_BANNER       = as_bool(os.getenv("SHOW_BANNER"), True)

# -------- Exchange Init (robust env mapping) -------- #
def make_exchange() -> ccxt.Exchange:
    if EXCHANGE != "kraken":
        raise SystemExit("This build is wired for Kraken only right now (EXCHANGE=kraken).")

    api_key = getenv_any("KRAKEN_API_KEY", "KRAKEN_KEY", "KRAKEN_API")
    api_sec = getenv_any("KRAKEN_API_SECRET", "KRAKEN_SECRET", "KRAKEN_APISECRET")
    api_otp = getenv_any("KRAKEN_API_OTP", "KRAKEN_OTP", "KRAKEN_TOTP")

    if not api_key or not api_sec:
        raise SystemExit("Kraken credentials missing: expected KRAKEN_API_KEY and KRAKEN_API_SECRET (or compatible fallbacks).")

    kwargs: Dict[str, Any] = {"apiKey": api_key, "secret": api_sec}
    if api_otp:
        kwargs["password"] = api_otp  # only if your key uses API 2FA

    ex = ccxt.kraken(kwargs)
    return ex

# -------- Strategy: symbols / filters -------- #
STABLE_FILTER = {"USDT", "USDC"}
BLOCKLIST = set(["SPX/USD", "EUR/USD", "GBP/USD", "USD/USD"])

def is_valid_symbol(s: str) -> bool:
    if not s.endswith(f"/{BASE_QUOTE}"):
        return False
    if s.upper() in BLOCKLIST:
        return False
    base = s.split("/")[0].upper()
    if base in STABLE_FILTER:
        return False
    return True

# -------- State I/O -------- #
def ensure_state() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)

def load_pos_state() -> Dict[str, Dict[str, float]]:
    ensure_state()
    if not os.path.exists(POS_STATE_PATH):
        return {}
    try:
        with open(POS_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_pos_state(st: Dict[str, Dict[str, float]]) -> None:
    ensure_state()
    with open(POS_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)

def write_summary(lines: List[str]) -> None:
    ensure_state()
    with open(os.path.join(STATE_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def append_kpi_row(row: Dict[str, Any]) -> None:
    ensure_state()
    new_file = not os.path.exists(KPI_CSV_PATH)
    with open(KPI_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts","dry_run","positions","cash_usd","equity_usd","pnl_day_pct","buys","sells","rotations"
        ])
        if new_file:
            w.writeheader()
        w.writerow(row)

# -------- Portfolio helpers -------- #
def fetch_cash_positions(ex: ccxt.Exchange) -> Tuple[float, Dict[str, Dict[str, float]]]:
    balance = ex.fetch_balance()
    cash = float(balance.get("total", {}).get(BASE_QUOTE, 0.0))
    positions: Dict[str, Dict[str, float]] = {}
    for coin, amt in balance.get("total", {}).items():
        if coin.upper() in (BASE_QUOTE, "USDT", "USDC"):
            continue
        try:
            amt_f = float(amt)
        except Exception:
            amt_f = 0.0
        if amt_f <= 0:
            continue
        symbol = f"{coin.upper()}/{BASE_QUOTE}"
        if not is_valid_symbol(symbol):
            continue
        try:
            ticker = ex.fetch_ticker(symbol)
            price = float(ticker["last"]) if ticker and ticker.get("last") else 0.0
        except Exception:
            price = 0.0
        positions[symbol] = {"amount": amt_f, "price": price, "value": amt_f * price}
    return cash, positions

def fetch_topk_symbols(ex: ccxt.Exchange, topk: int) -> List[Tuple[str, float]]:
    markets = ex.load_markets()
    candidates = [s for s in markets.keys() if is_valid_symbol(s)]
    scored: List[Tuple[str, float]] = []
    for s in candidates:
        try:
            t = ex.fetch_ticker(s)
            chg = float(t.get("percentage") or 0.0)  # 24h %
        except Exception:
            chg = 0.0
        scored.append((s, chg))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max(0, topk)]

# -------- Orders -------- #
def market_buy(ex: ccxt.Exchange, symbol: str, usd_amount: float, dry: bool) -> Optional[Dict[str, Any]]:
    if usd_amount < MIN_NOTIONAL_USD:
        log("WARN", f"Buy skipped — below MIN_NOTIONAL_USD ${MIN_NOTIONAL_USD:.2f}")
        return None
    try:
        price = float(ex.fetch_ticker(symbol)["last"])
        qty = usd_amount / max(price, 1e-12)
        if dry:
            log("OK", f"DRY BUY {symbol} for ${usd_amount:.2f} (~{qty:.6f}) @ {price:.4f}")
            return {"id":"dry","type":"market","side":"buy","symbol":symbol,"cost":usd_amount,"amount":qty,"price":price}
        else:
            o = ex.create_order(symbol, "market", "buy", qty)
            log("OK", f"BUY {symbol} ${usd_amount:.2f} (qty ~{qty:.6f})")
            return o
    except Exception as e:
        log("ERR", f"Buy failed {symbol}: {e}")
        return None

def market_sell(ex: ccxt.Exchange, symbol: str, qty: float, dry: bool, reason: str = "") -> Optional[Dict[str, Any]]:
    if qty <= 0:
        return None
    try:
        if dry:
            log("OK", f"DRY SELL {symbol} qty {qty:.6f} {'— '+reason if reason else ''}")
            return {"id":"dry","type":"market","side":"sell","symbol":symbol,"amount":qty}
        else:
            o = ex.create_order(symbol, "market", "sell", qty)
            log("OK", f"SELL {symbol} qty {qty:.6f} {'— '+reason if reason else ''}")
            return o
    except Exception as e:
        log("ERR", f"Sell failed {symbol}: {e}")
        return None

# -------- Dust Sweeper -------- #
def apply_dust_sweeper(ex: ccxt.Exchange, positions: Dict[str, Dict[str, float]], dry: bool) -> float:
    freed = 0.0
    for sym, pos in list(positions.items()):
        val = float(pos["value"])
        if val <= DUST_MIN_USD and val > 0:
            qty = float(pos["amount"])
            log("WARN", f"🧹 Sweeping dust: {sym} (${val:.2f})")
            market_sell(ex, sym, qty, dry, reason="dust")
            freed += val
            del positions[sym]
    return freed

# -------- Sell Rules: TAKE_PROFIT / STOP_LOSS / TRAILING_STOP -------- #
def seed_or_update_pos_state(st: Dict[str, Dict[str, float]], sym: str, price: float, qty: float) -> None:
    ps = st.get(sym)
    if not ps:
        st[sym] = {"avg_price": price, "trail_max": price, "last_qty": qty}
        return
    # If qty increased, update avg_price (simple weighted)
    prev_qty = float(ps.get("last_qty", 0.0))
    prev_avg = float(ps.get("avg_price", price))
    if qty > prev_qty:
        added = qty - prev_qty
        new_avg = (prev_avg * prev_qty + price * added) / max(qty, 1e-12)
        ps["avg_price"] = new_avg
    # Update trail_max if new high
    if price > float(ps.get("trail_max", price)):
        ps["trail_max"] = price
    ps["last_qty"] = qty

def take_profit_triggered(price: float, avg_price: float) -> bool:
    if avg_price <= 0:
        return False
    gain = (price - avg_price) / avg_price * 100.0
    return gain >= TP_PCT

def stop_loss_triggered(price: float, avg_price: float) -> bool:
    if avg_price <= 0:
        return False
    dd = (price - avg_price) / avg_price * 100.0
    return dd <= -SL_PCT

def trailing_stop_triggered(price: float, trail_max: float, avg_price: float) -> bool:
    if trail_max <= 0 or price <= 0:
        return False
    # Only trail once we're above avg (lock gains, not losses)
    if price <= avg_price:
        return False
    drop = (trail_max - price) / trail_max * 100.0
    return drop >= TRAIL_PCT

def apply_sell_rules(ex: ccxt.Exchange, positions: Dict[str, Dict[str, float]], st: Dict[str, Dict[str, float]], dry: bool) -> Tuple[int,int]:
    sells = 0
    rot_sells = 0  # reserved for rotation path
    for sym, pos in list(positions.items()):
        qty = float(pos["amount"])
        if qty <= 0:
            continue
        # refresh live price
        try:
            price = float(ex.fetch_ticker(sym)["last"])
        except Exception:
            price = float(pos.get("price", 0.0))
        ps = st.get(sym)
        if not ps:
            seed_or_update_pos_state(st, sym, price, qty)
            ps = st[sym]
        avg = float(ps.get("avg_price", price))
        trail_max = float(ps.get("trail_max", price))
        # Update trail_max each run if we made a new high
        if price > trail_max:
            ps["trail_max"] = price
            trail_max = price

        # --- TAKE_PROFIT ---
        if take_profit_triggered(price, avg):
            log("OK", f"TAKE_PROFIT hit on {sym}: price {price:.6f} ≥ avg {avg:.6f} + {TP_PCT:.2f}%")
            market_sell(ex, sym, qty, dry, reason="TAKE_PROFIT")
            sells += 1
            del positions[sym]
            st.pop(sym, None)
            continue

        # --- STOP_LOSS ---
        if stop_loss_triggered(price, avg):
            log("WARN", f"STOP_LOSS hit on {sym}: price {price:.6f} ≤ avg {avg:.6f} - {SL_PCT:.2f}%")
            market_sell(ex, sym, qty, dry, reason="STOP_LOSS")
            sells += 1
            del positions[sym]
            st.pop(sym, None)
            continue

        # --- TRAILING_STOP ---
        if trailing_stop_triggered(price, trail_max, avg):
            log("WARN", f"TRAILING_STOP hit on {sym}: drawdown from {trail_max:.6f} ≥ {TRAIL_PCT:.2f}% (price {price:.6f})")
            market_sell(ex, sym, qty, dry, reason="TRAILING_STOP")
            sells += 1
            del positions[sym]
            st.pop(sym, None)
            continue

        # Keep state updated when we simply hold
        seed_or_update_pos_state(st, sym, price, qty)

    return sells, rot_sells

# -------- Rotation helpers -------- #
def weakest_symbol(positions: Dict[str, Dict[str, float]], ex: ccxt.Exchange) -> Optional[Tuple[str,float]]:
    weakest = None
    weakest_chg = 1e9
    for sym in positions.keys():
        try:
            t = ex.fetch_ticker(sym)
            chg = float(t.get("percentage") or 0.0)
        except Exception:
            chg = -999.0
        if chg < weakest_chg:
            weakest_chg = chg
            weakest = sym
    if weakest is None:
        return None
    return weakest, weakest_chg

# -------- Main -------- #
def main() -> int:
    ensure_state()

    if SHOW_BANNER:
        log("SUM", "===================================================")
        log("SUM", f"Crypto Live — {now_utc_iso()}")
        log("SUM", f"DRY_RUN={'ON' if DRY_RUN else 'OFF'}  RUN_SWITCH={'ON' if RUN_SWITCH else 'OFF'}")
        log("SUM", f"Quote: {BASE_QUOTE}, Max positions: {MAX_POSITIONS}, Max buys/run: {MAX_BUYS_PER_RUN}")
        log("SUM", f"Dust: <= ${DUST_MIN_USD:.2f}, MinNotional: ${MIN_NOTIONAL_USD:.2f}, Reserve cash: {RESERVE_CASH_PCT:.1f}%")
        log("SUM", f"TP {TP_PCT:.2f}%, SL {SL_PCT:.2f}%, TRAIL {TRAIL_PCT:.2f}%")
        log("SUM", "===================================================")

    if not RUN_SWITCH:
        log("WARN", "RUN_SWITCH=off — skipping trading loop.")
        write_summary(["RUN_SWITCH=off — no trading performed."])
        return 0

    try:
        ex = make_exchange()
        _ = ex.fetch_ticker(f"BTC/{BASE_QUOTE}")  # sanity public call
    except Exception as e:
        log("ERR", f"Exchange init/fetch_ticker failed: {e}")
        if AUTO_PAUSE_ON_ERROR:
            write_summary([f"Hard error: {e}"])
            return 1
        return 0

    try:
        cash, positions = fetch_cash_positions(ex)
        equity = cash + sum(p["value"] for p in positions.values())
        log("OK", f"Start — Cash ${cash:.2f}, Equity ${equity:.2f}, Positions {len(positions)}")

        # Load per-position state and apply sells first
        pos_state = load_pos_state()
        sells_tp_sl_trail, _ = apply_sell_rules(ex, positions, pos_state, DRY_RUN)

        # Dust sweep after rule-based sells (sweeps tiny leftovers)
        freed = apply_dust_sweeper(ex, positions, DRY_RUN)
        if freed > 0:
            cash += freed
            log("OK", f"Dust sweep freed ~${freed:.2f}; Cash now ${cash:.2f}")

        # Desired targets
        top_syms = fetch_topk_symbols(ex, TOPK)
        log("OK", "Top-K by 24h%: " + ", ".join([f"{s}({c:+.1f}%)" for s,c in top_syms]))

        have_syms = set(positions.keys())
        buys_done = 0
        rotations = 0
        sells_rotation = 0

        # Buy loop (capacity & cash-aware; one buy/run by default)
        for s,_chg in top_syms:
            if buys_done >= MAX_BUYS_PER_RUN:
                break
            if s in have_syms:
                continue

            reserve = equity * (RESERVE_CASH_PCT / 100.0)
            spendable = max(0.0, cash - reserve)

            if spendable < MIN_NOTIONAL_USD and ROTATE_WHEN_CASH_SHORT and len(positions) > 0:
                w = weakest_symbol(positions, ex)
                if w:
                    wsym, wchg = w
                    qty = positions[wsym]["amount"]
                    log("WARN", f"ROTATE_WHEN_CASH_SHORT true → selling weakest {wsym} ({wchg:+.1f}%) to fund {s}")
                    if market_sell(ex, wsym, qty, DRY_RUN, reason="cash_short"):
                        sells_rotation += 1
                        rotations += 1
                        cash, positions = fetch_cash_positions(ex)
                        have_syms = set(positions.keys())
                        reserve = equity * (RESERVE_CASH_PCT / 100.0)
                        spendable = max(0.0, cash - reserve)

            if spendable >= MIN_NOTIONAL_USD and buys_done < MAX_BUYS_PER_RUN and len(positions) < MAX_POSITIONS:
                buy_amt = max(MIN_NOTIONAL_USD, spendable / max(1, (MAX_POSITIONS - len(positions))))
                if market_buy(ex, s, buy_amt, DRY_RUN):
                    buys_done += 1
                    # refresh + seed state
                    cash, positions = fetch_cash_positions(ex)
                    have_syms = set(positions.keys())
                    try:
                        price = float(ex.fetch_ticker(s)["last"])
                    except Exception:
                        price = 0.0
                    qty = positions.get(s, {}).get("amount", 0.0)
                    seed_or_update_pos_state(pos_state, s, price, float(qty))

        # If full, swap weakest for a stronger candidate once/run
        if ROTATE_WHEN_FULL and len(positions) >= MAX_POSITIONS and buys_done < MAX_BUYS_PER_RUN:
            candidate = next((sym for sym,_ in top_syms if sym not in have_syms), None)
            if candidate:
                w = weakest_symbol(positions, ex)
                if w:
                    wsym, wchg = w
                    if wsym != candidate:
                        qty = positions[wsym]["amount"]
                        log("WARN", f"ROTATE_WHEN_FULL true → {wsym} -> {candidate}")
                        if market_sell(ex, wsym, qty, DRY_RUN, reason="rotate_full"):
                            sells_rotation += 1
                            rotations += 1
                            cash, positions = fetch_cash_positions(ex)
                            reserve = equity * (RESERVE_CASH_PCT / 100.0)
                            spendable = max(0.0, cash - reserve)
                            if spendable >= MIN_NOTIONAL_USD and buys_done < MAX_BUYS_PER_RUN:
                                buy_amt = max(MIN_NOTIONAL_USD, spendable / max(1, (MAX_POSITIONS - len(positions))))
                                if market_buy(ex, candidate, buy_amt, DRY_RUN):
                                    buys_done += 1
                                    # seed state
                                    try:
                                        price = float(ex.fetch_ticker(candidate)["last"])
                                    except Exception:
                                        price = 0.0
                                    qty2 = positions.get(candidate, {}).get("amount", 0.0)
                                    seed_or_update_pos_state(pos_state, candidate, price, float(qty2))

        # Recalc and persist
        cash_end, positions_end = fetch_cash_positions(ex)
        equity_end = cash_end + sum(p["value"] for p in positions_end.values())
        pnl_day_pct = 0.0  # placeholder unless you track prior-day equity

        save_pos_state(pos_state)

        log("KPI", f"End — Cash ${cash_end:.2f}, Equity ${equity_end:.2f}, Positions {len(positions_end)}")
        append_kpi_row({
            "ts": now_utc_iso(),
            "dry_run": DRY_RUN,
            "positions": len(positions_end),
            "cash_usd": round(cash_end, 2),
            "equity_usd": round(equity_end, 2),
            "pnl_day_pct": round(pnl_day_pct, 3),
            "buys": buys_done,
            "sells": int(sells_tp_sl_trail + sells_rotation),
            "rotations": rotations,
        })
        write_summary([
            f"Mode: {'DRY' if DRY_RUN else 'LIVE'}",
            f"Buys: {buys_done}, Sells: {int(sells_tp_sl_trail + sells_rotation)}, Rotations: {rotations}",
            f"End Equity: ${equity_end:.2f} (Cash ${cash_end:.2f}, Positions {len(positions_end)})",
        ])
        return 0

    except ccxt.AuthenticationError as e:
        log("ERR", f"Authentication error: {e}")
        write_summary([f"Authentication error: {e}"])
        return 1 if AUTO_PAUSE_ON_ERROR else 0
    except Exception as e:
        log("ERR", f"Unhandled error: {e}")
        traceback.print_exc()
        write_summary([f"Unhandled error: {e}"])
        return 1 if AUTO_PAUSE_ON_ERROR else 0

if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
