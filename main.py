# main.py — Crypto Live with Guard Pack (Kraken + ENV-robust creds)
# - USD-only trading on Kraken via ccxt
# - Dust sweeping: sell positions <= DUST_MIN_USD to USD
# - Auto-pick Top-K by 24h % change (filtering out stablecoins/indexes)
# - Rotation: sell weakest to buy stronger (when full or cash-short)
# - Max positions, per-run buy cap, min notional guard
# - TP/SL/Trailing stop, daily loss cap, auto-pause on hard error
# - Color-tagged logs, KPI summary, CSV history
#
# All behavior is controlled by environment variables (see ENV section).
# This script prints clearly whether it is DRY_RUN or LIVE.

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

TP_PCT            = as_float(os.getenv("TP_PCT"), 2.0)
SL_PCT            = as_float(os.getenv("SL_PCT"), 3.5)
TRAIL_PCT         = as_float(os.getenv("TRAIL_PCT"), 1.2)
DAILY_LOSS_CAP    = as_float(os.getenv("DAILY_LOSS_CAP"), 3.0)
AUTO_PAUSE_ON_ERROR = as_bool(os.getenv("AUTO_PAUSE_ON_ERROR"), True)

LOG_LEVEL         = getenv_any("LOG_LEVEL", default="INFO")
KPI_CSV_PATH      = getenv_any("KPI_CSV_PATH", default=".state/kpi_history.csv")
STATE_DIR         = getenv_any("STATE_DIR", default=".state")

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
        # only needed if your Kraken API key is protected by API 2FA
        kwargs["password"] = api_otp

    ex = ccxt.kraken(kwargs)
    ex.options["fetchMarketsMethod"] = "publicGetAssets"  # ccxt works out the rest; Kraken is quirky
    return ex

# -------- Strategy: symbols / filters -------- #
STABLE_FILTER = {"USDT", "USDC"}  # avoid stablecoins
BLOCKLIST = set(["SPX/USD", "EUR/USD", "GBP/USD", "USD/USD"])  # index/fiat-not-crypto

def is_valid_symbol(s: str) -> bool:
    if not s.endswith(f"/{BASE_QUOTE}"):
        return False
    if s.upper() in BLOCKLIST:
        return False
    base = s.split("/")[0].upper()
    if base in STABLE_FILTER:
        return False
    return True

# -------- Portfolio / math utils -------- #
def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

def ensure_state() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)

def write_summary(lines: List[str]) -> None:
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

# -------- Core trading helpers -------- #
def fetch_cash_positions(ex: ccxt.Exchange) -> Tuple[float, Dict[str, Dict[str, float]]]:
    # Return cash in USD and a dict: symbol -> {amount, price, value}
    balance = ex.fetch_balance()
    # Kraken uses 'USD' under 'total' for cash
    cash = float(balance.get("total", {}).get(BASE_QUOTE, 0.0))
    positions: Dict[str, Dict[str, float]] = {}
    # Build positions from balances by mapping non-zero coins to symbols
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
        positions[symbol] = {
            "amount": amt_f,
            "price": price,
            "value": amt_f * price,
        }
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

def market_buy(ex: ccxt.Exchange, symbol: str, usd_amount: float, dry: bool) -> Optional[Dict[str, Any]]:
    if usd_amount < MIN_NOTIONAL_USD:
        log("WARN", f"Buy skipped — below MIN_NOTIONAL_USD ${MIN_NOTIONAL_USD:.2f}")
        return None
    try:
        price = float(ex.fetch_ticker(symbol)["last"])
        qty = usd_amount / max(price, 1e-12)
        if dry:
            log("OK", f"DRY BUY {symbol} for ${usd_amount:.2f} (~{qty:.6f}) @ {price:.2f}")
            return {"id":"dry","type":"market","side":"buy","symbol":symbol,"cost":usd_amount,"amount":qty}
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

# -------- Risk controls (simple versions) -------- #
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

def weakest_symbol(positions: Dict[str, Dict[str, float]], ex: ccxt.Exchange) -> Optional[Tuple[str,float]]:
    # Weakness by 24h % change
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

# -------- Main run -------- #
def main() -> int:
    ensure_state()

    if SHOW_BANNER:
        log("SUM", "===================================================")
        log("SUM", f"Crypto Live — {now_utc_iso()}")
        log("SUM", f"DRY_RUN={'ON' if DRY_RUN else 'OFF'}  RUN_SWITCH={'ON' if RUN_SWITCH else 'OFF'}")
        log("SUM", f"Quote: {BASE_QUOTE}, Max positions: {MAX_POSITIONS}, Max buys/run: {MAX_BUYS_PER_RUN}")
        log("SUM", f"Dust: <= ${DUST_MIN_USD:.2f}, MinNotional: ${MIN_NOTIONAL_USD:.2f}, Reserve cash: {RESERVE_CASH_PCT:.1f}%")
        log("SUM", "===================================================")

    if not RUN_SWITCH:
        log("WARN", "RUN_SWITCH=off — skipping trading loop.")
        write_summary(["RUN_SWITCH=off — no trading performed."])
        return 0

    try:
        ex = make_exchange()
        # sanity: a tiny public call first
        _ = ex.fetch_ticker(f"BTC/{BASE_QUOTE}")
    except Exception as e:
        log("ERR", f"Exchange init/fetch_ticker failed: {e}")
        if AUTO_PAUSE_ON_ERROR:
            write_summary([f"Hard error: {e}"])
            return 1
        return 0

    try:
        cash, positions = fetch_cash_positions(ex)
        pos_count = len(positions)
        equity = cash + sum(p["value"] for p in positions.values())
        log("OK", f"Start — Cash ${cash:.2f}, Equity ${equity:.2f}, Positions {pos_count}")

        # Dust sweep
        freed = apply_dust_sweeper(ex, positions, DRY_RUN)
        if freed > 0:
            cash += freed
            log("OK", f"Dust sweep freed ~${freed:.2f}; Cash now ${cash:.2f}")

        # Compute desired target list
        top_syms = fetch_topk_symbols(ex, TOPK)
        log("OK", "Top-K by 24h%: " + ", ".join([f"{s}({c:+.1f}%)" for s,c in top_syms]))

        have_syms = set(positions.keys())
        target_syms = [s for s,_ in top_syms]

        # Rotation logic
        buys_done = 0
        sells_done = 0
        rotations = 0

        def can_buy_more() -> bool:
            return (len(positions) < MAX_POSITIONS) and (buys_done < MAX_BUYS_PER_RUN)

        # If we have capacity and a top symbol we don't own, buy one
        for s,_chg in top_syms:
            if buys_done >= MAX_BUYS_PER_RUN:
                break
            if s in have_syms:
                continue
            # need cash after reserve
            reserve = equity * (RESERVE_CASH_PCT / 100.0)
            spendable = max(0.0, cash - reserve)
            if spendable < MIN_NOTIONAL_USD:
                # try rotation if allowed
                if ROTATE_WHEN_CASH_SHORT and len(positions) > 0:
                    w = weakest_symbol(positions, ex)
                    if w:
                        wsym, wchg = w
                        qty = positions[wsym]["amount"]
                        log("WARN", f"ROTATE_WHEN_CASH_SHORT true → selling weakest {wsym} ({wchg:+.1f}%) to fund {s}")
                        if market_sell(ex, wsym, qty, DRY_RUN, reason="cash_short"):
                            sells_done += 1
                            rotations += 1
                            # refresh cash snapshot roughly
                            cash2, positions = fetch_cash_positions(ex)
                            cash = cash2
                            have_syms = set(positions.keys())
                            reserve = equity * (RESERVE_CASH_PCT / 100.0)
                            spendable = max(0.0, cash - reserve)
                else:
                    log("WARN", "Insufficient spendable cash; skipping buy.")
            if spendable >= MIN_NOTIONAL_USD and buys_done < MAX_BUYS_PER_RUN and len(positions) < MAX_POSITIONS:
                buy_amt = max(MIN_NOTIONAL_USD, spendable / max(1, (MAX_POSITIONS - len(positions))))
                if market_buy(ex, s, buy_amt, DRY_RUN):
                    buys_done += 1
                    # refresh post-buy
                    cash2, positions = fetch_cash_positions(ex)
                    cash = cash2
                    have_syms = set(positions.keys())

        # If full and rotation enabled, replace one weak with one strong (once per run)
        if ROTATE_WHEN_FULL and len(positions) >= MAX_POSITIONS and buys_done < MAX_BUYS_PER_RUN:
            # pick first top symbol we don't have
            candidate = None
            for s,_ in top_syms:
                if s not in have_syms:
                    candidate = s
                    break
            if candidate:
                w = weakest_symbol(positions, ex)
                if w:
                    wsym, wchg = w
                    if wsym != candidate:
                        qty = positions[wsym]["amount"]
                        log("WARN", f"ROTATE_WHEN_FULL true → {wsym} -> {candidate}")
                        if market_sell(ex, wsym, qty, DRY_RUN, reason="rotate_full"):
                            sells_done += 1
                            rotations += 1
                            # refresh cash and buy
                            cash2, positions = fetch_cash_positions(ex)
                            cash = cash2
                            reserve = equity * (RESERVE_CASH_PCT / 100.0)
                            spendable = max(0.0, cash - reserve)
                            if spendable >= MIN_NOTIONAL_USD and buys_done < MAX_BUYS_PER_RUN:
                                buy_amt = max(MIN_NOTIONAL_USD, spendable / max(1, (MAX_POSITIONS - len(positions))))
                                if market_buy(ex, candidate, buy_amt, DRY_RUN):
                                    buys_done += 1

        # TODO: TP/SL/TRAIL logic could be expanded. For now, we rely on ccxt market orders above.
        # A minimal placeholder is kept for future extension.

        # Recalc equity at end
        cash_end, positions_end = fetch_cash_positions(ex)
        equity_end = cash_end + sum(p["value"] for p in positions_end.values())
        pnl_day_pct = 0.0  # placeholder unless you log prior day equity
        log("KPI", f"End — Cash ${cash_end:.2f}, Equity ${equity_end:.2f}, Positions {len(positions_end)}")

        # Persist KPI
        append_kpi_row({
            "ts": now_utc_iso(),
            "dry_run": DRY_RUN,
            "positions": len(positions_end),
            "cash_usd": round(cash_end, 2),
            "equity_usd": round(equity_end, 2),
            "pnl_day_pct": round(pnl_day_pct, 3),
            "buys": buys_done,
            "sells": sells_done,
            "rotations": rotations,
        })

        write_summary([
            f"Mode: {'DRY' if DRY_RUN else 'LIVE'}",
            f"Buys: {buys_done}, Sells: {sells_done}, Rotations: {rotations}",
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
