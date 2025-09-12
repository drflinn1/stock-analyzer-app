#!/usr/bin/env python3
import os, json, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

# =========================
# ENV / KNOBS
# =========================
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# Exits
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))   # +3% TP
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT", "2.0"))     # -2% SL

# Trailing profit
TRAIL_START_PCT   = float(os.getenv("TRAIL_START_PCT", "3.0"))   # start trailing when PnL ≥ this
TRAIL_OFFSET_PCT  = float(os.getenv("TRAIL_OFFSET_PCT", "1.0"))  # exit when drawdown from peak ≥ this

# Buy controls
POSITION_SIZE_USD    = float(os.getenv("POSITION_SIZE_USD", "10"))
DAILY_SPEND_CAP_USD  = float(os.getenv("DAILY_SPEND_CAP_USD", "15"))
MAX_OPEN_TRADES      = int(os.getenv("MAX_OPEN_TRADES", "3"))
MIN_BALANCE_USD      = float(os.getenv("MIN_BALANCE_USD", "5"))

# Picker (Balanced)
DROP_PCT_GATE        = float(os.getenv("DROP_PCT", "2.5"))       # buy if 15m change ≤ -2.5%
TIMEFRAME            = os.getenv("CANDLES_TIMEFRAME", "15m")

# RSI filter
ENABLE_RSI           = os.getenv("ENABLE_RSI", "true").lower() == "true"
RSI_LEN              = int(os.getenv("RSI_LEN", "14"))
RSI_MAX              = float(os.getenv("RSI_MAX", "35"))

# Universe
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USD,ETH/USD,DOGE/USD").split(",") if s.strip()]

# Kraken keys
KRAKEN_API_KEY    = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

# Optional simulated balance for DRY runs (quote-currency units, e.g., USD)
SIM_BALANCE_QUOTE = os.getenv("SIM_BALANCE_QUOTE")  # e.g., "100.0" to test buys in DRY mode

# =========================
# STATE
# =========================
STATE_PATH = Path("state/trade_state.json")
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"positions": {}, "spend": {}}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))

def today_key():
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d")

def add_daily_spend(state, usd):
    k = today_key()
    state["spend"][k] = float(state["spend"].get(k, 0.0) + float(usd))

def spent_today(state):
    return float(state["spend"].get(today_key(), 0.0))

# =========================
# Utils
# =========================
def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def pct(a, b):
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

def fmt2(x): return f"{float(x):.2f}"

# =========================
# CCXT / Kraken
# =========================
import ccxt
EX = None

def kraken_client():
    conf = {"enableRateLimit": True, "timeout": 20000}
    if KRAKEN_API_KEY and KRAKEN_API_SECRET:
        conf.update({"apiKey": KRAKEN_API_KEY, "secret": KRAKEN_API_SECRET})
    return ccxt.kraken(conf)

def init_exchange():
    global EX
    if EX is None:
        EX = kraken_client()

def price(symbol: str) -> float:
    init_exchange()
    t = EX.fetch_ticker(symbol)
    last = t.get("last") or t.get("close")
    if last is None:
        raise RuntimeError(f"No last price for {symbol}")
    return float(last)

def fetch_closes(symbol: str, timeframe: str, limit: int):
    init_exchange()
    ohlcv = EX.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return [float(c[4]) for c in ohlcv]  # close prices

def change_pct_15m(symbol: str) -> float:
    init_exchange()
    ohlcv = EX.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=2)
    if ohlcv and len(ohlcv) >= 2:
        prev_close = float(ohlcv[-2][4])
        cur_last   = float(ohlcv[-1][4])
        return pct(cur_last, prev_close)
    t = EX.fetch_ticker(symbol)
    o = t.get("open")
    l = t.get("last") or t.get("close")
    return pct(float(l), float(o)) if (o and l) else 0.0

def balances():
    init_exchange()
    try:
        return EX.fetch_balance()
    except Exception:
        return {"free": {}}

def base(sym: str) -> str:
    return sym.split("/")[0]

def quote(sym: str) -> str:
    return sym.split("/")[1] if "/" in sym else "USD"

def qty_free(sym: str) -> float:
    bal = balances()
    return float(bal.get("free", {}).get(base(sym), 0.0))

def free_balance_for_quote(q: str) -> float:
    """Handle Kraken's fiat aliases like ZUSD and allow DRY simulation."""
    bal = balances()
    free = bal.get("free", {}) or {}
    candidates = [q, q.upper()]
    if len(q) == 3:
        candidates += [f"Z{q.upper()}", f"X{q.upper()}"]  # Kraken aliases
    for k in candidates:
        if k in free:
            try:
                return float(free[k])
            except Exception:
                pass
    if DRY_RUN and SIM_BALANCE_QUOTE:
        try:
            return float(SIM_BALANCE_QUOTE)
        except Exception:
            pass
    return 0.0

def place_buy(symbol: str, quote_amount: float, px: float):
    amount = max(quote_amount / px, 0.0)  # base units
    amount = float(f"{amount:.6f}")       # Kraken precision
    if amount <= 0:
        print(f"[SKIP] BUY {symbol} amount<=0")
        return {"status": "skip"}
    if DRY_RUN:
        print(f"[DRY] BUY {symbol} ${fmt2(quote_amount)} @ {fmt2(px)} (≈{amount})")
        return {"status": "dry", "symbol": symbol, "filled": amount}
    init_exchange()
    o = EX.create_order(symbol=symbol, type="market", side="buy", amount=amount)
    print(f"[LIVE] BUY {symbol} market amount={amount} (order {o.get('id')})")
    return {"status": "ok", "symbol": symbol, "filled": amount, "order": o}

def place_sell(symbol: str, qty: float, px: float):
    qty = float(f"{qty:.6f}")
    if qty <= 0:
        print(f"[SKIP] SELL {symbol} qty<=0")
        return {"status": "skip"}
    if DRY_RUN:
        print(f"[DRY] SELL {symbol} qty={qty:.6f} @ {fmt2(px)}")
        return {"status": "dry", "symbol": symbol, "filled": qty}
    init_exchange()
    o = EX.create_order(symbol=symbol, type="market", side="sell", amount=qty)
    print(f"[LIVE] SELL {symbol} market qty={qty:.6f} (order {o.get('id')})")
    return {"status": "ok", "symbol": symbol, "filled": qty, "order": o}

# =========================
# RSI (Wilder)
# =========================
def compute_rsi(closes, length: int) -> float:
    if len(closes) < length + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, length + 1):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length
    for i in range(length + 1, len(closes)):
        ch = closes[i] - closes[i-1]
        gain = max(ch, 0.0)
        loss = max(-ch, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def rsi_now(symbol: str, length: int) -> float:
    closes = fetch_closes(symbol, TIMEFRAME, limit=length + 50)
    return compute_rsi(closes, length)

# =========================
# Trailing + exit helpers
# =========================
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

# =========================
# Core run
# =========================
def run_once():
    private_api = bool(KRAKEN_API_KEY and KRAKEN_API_SECRET)
    print("=== START TRADING OUTPUT ===")
    print(f"{now_iso()} | run started | DRY_RUN={DRY_RUN} | TP={fmt2(TAKE_PROFIT_PCT)}% | SL={fmt2(STOP_LOSS_PCT)}% | "
          f"TRAIL_START={fmt2(TRAIL_START_PCT)}% | TRAIL_OFFSET={fmt2(TRAIL_OFFSET_PCT)}% | "
          f"DROP_GATE={fmt2(DROP_PCT_GATE)}% | TF={TIMEFRAME} | RSI({RSI_LEN}) {'ON' if ENABLE_RSI else 'OFF'} max≤{fmt2(RSI_MAX)} | "
          f"private_api={'ON' if private_api else 'OFF'}")

    state = load_state()
    buys_placed = 0
    sells_placed = 0

    # ---------- SELL CHECK ----------
    for sym in SYMBOLS:
        qty = qty_free(sym)
        if qty <= 0:
            continue

        px = price(sym)
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
            place_sell(sym, qty, px)
            sells_placed += 1
            state["positions"].pop(sym, None)

    # ---------- BUY WINDOW (best-candidate dip + optional RSI) ----------
    # Use the quote currency of your first symbol for budgeting
    q = quote(SYMBOLS[0]) if SYMBOLS else "USD"
    quote_free = free_balance_for_quote(q)
    daily_remaining = max(0.0, DAILY_SPEND_CAP_USD - spent_today(state))
    can_spend = min(quote_free, daily_remaining)
    open_trades = sum(1 for s in SYMBOLS if qty_free(s) > 0)

    print(f"{now_iso()} | budget | {q}_free=${fmt2(quote_free)} | daily_remaining=${fmt2(daily_remaining)} | open_trades={open_trades}/{MAX_OPEN_TRADES}")

    if can_spend >= max(MIN_BALANCE_USD, POSITION_SIZE_USD) and open_trades < MAX_OPEN_TRADES:
        # Build candidate list
        candidates = []
        for sym in SYMBOLS:
            if qty_free(sym) > 0:
                continue
            chg15 = change_pct_15m(sym)
            rsi = rsi_now(sym, RSI_LEN) if ENABLE_RSI else None
            ok_drop = (chg15 <= -abs(DROP_PCT_GATE))
            ok_rsi  = (True if not ENABLE_RSI else (rsi is not None and rsi <= RSI_MAX))
            candidates.append((sym, chg15, rsi, ok_drop and ok_rsi))

        # pick strongest qualifying dip
        picked = None
        for sym, chg, rsi, ok in sorted(candidates, key=lambda x: x[1]):  # most negative first
            if ok:
                picked = (sym, chg, rsi)
                break

        if picked:
            psym, pchg, prsi = picked
            px = price(psym)
            rsi_txt = f", RSI={fmt2(prsi)}" if ENABLE_RSI else ""
            print(f"{now_iso()} | Best candidate | {psym} 15m_change={fmt2(pchg)}% (gate {fmt2(-abs(DROP_PCT_GATE))}%){rsi_txt} → BUY ${fmt2(POSITION_SIZE_USD)} @ {fmt2(px)}")
            r = place_buy(psym, POSITION_SIZE_USD, px)
            if r.get("status") in ("ok", "dry"):
                buys_placed += 1
                add_daily_spend(state, POSITION_SIZE_USD)
                ensure_entry(state, psym, entry_avg=px)
        else:
            if candidates:
                sym0, chg0, rsi0, ok0 = sorted(candidates, key=lambda x: x[1])[0]
                reasons = []
                if chg0 > -abs(DROP_PCT_GATE): reasons.append(f"drop {fmt2(chg0)}% > gate")
                if ENABLE_RSI and (rsi0 is None or rsi0 > RSI_MAX): reasons.append(f"RSI {fmt2(rsi0 or 0)} > {fmt2(RSI_MAX)}")
                print(f"{now_iso()} | Best candidate | {sym0} (did not pass: {', '.join(reasons) or 'gate'}) → NO BUY")
            else:
                print(f"{now_iso()} | Best candidate | none → NO BUY")
    else:
        reason = []
        if can_spend < max(MIN_BALANCE_USD, POSITION_SIZE_USD): reason.append("insufficient budget")
        if open_trades >= MAX_OPEN_TRADES: reason.append("max open trades reached")
        print(f"{now_iso()} | BUY window skipped ({', '.join(reason) if reason else 'guardrail'})")

    save_state(state)
    print(f"Run complete. buys_placed={buys_placed} | sells_placed={sells_placed} | DRY_RUN={DRY_RUN}")
    print("=== END TRADING OUTPUT ===")

# =========================
# Main
# =========================
if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        sys.exit(1)
