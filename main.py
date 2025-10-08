# main.py — Crypto Live with Guard Pack (Kraken, USD)
# - Auto-pick Top-K from UNIVERSE using 24h momentum
# - DRY_RUN banner and safe execution
# - Rotation when cash short (optional) and when full (optional)
# - Take-profit / Stop-loss / Trailing stop
# - Daily caps (max loss, max entries)
# - Slack pings for trades and risk flips (no line continuations)
# - Artifacts: .state/positions.json, .state/kpi_history.csv, .state/spec_gate_report.txt
#
# Env keys expected (all have sane defaults):
# EXCHANGE, QUOTE, UNIVERSE, MAX_POSITIONS, USD_PER_TRADE, RESERVE_CASH_PCT,
# ROTATE_WHEN_CASH_SHORT, ROTATE_WHEN_FULL, DUST_MIN_USD,
# TAKE_PROFIT_PCT, STOP_LOSS_PCT, TRAIL_STOP_PCT,
# MAX_DAILY_LOSS_PCT, MAX_DAILY_ENTRIES, EMERGENCY_SL_PCT,
# RUN_SWITCH, DRY_RUN,
# STATE_DIR, POSITIONS_JSON, KPI_CSV, SPEC_GATE_REPORT,
# KRAKEN_API_KEY, KRAKEN_API_SECRET, SLACK_WEBHOOK_URL

from __future__ import annotations
import os, sys, json, time, csv, math, traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- Tiny utils ----------

def utcnow() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def as_bool(val: Optional[str], default: bool) -> bool:
    if val is None or val == "":
        return default
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def as_float(val: Optional[str], default: float) -> float:
    try:
        return float(val) if val is not None else default
    except Exception:
        return default

def as_int(val: Optional[str], default: int) -> int:
    try:
        return int(val) if val is not None else default
    except Exception:
        return default

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def append_csv(path: str, row: List[Any]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts_utc","event","symbol","side","qty","price","notional","pnl","note"])
        w.writerow(row)

# ---------- Slack helpers (safe, no backslashes) ----------

import requests

def slack_post(webhook: str, payload: Dict[str, Any]) -> None:
    try:
        r = requests.post(
            webhook,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[warn] Slack post failed: {e}")

def slack_trade_ping(
    webhook: str,
    *,
    text: str,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    color: str = "#36a64f",
) -> None:
    fields: List[Dict[str, Any]] = [
        {"title": "Symbol",   "value": symbol,               "short": True},
        {"title": "Side",     "value": side.upper(),         "short": True},
        {"title": "Qty",      "value": f"{qty:.6f}",         "short": True},
        {"title": "Price",    "value": f"${price:.2f}",      "short": True},
        {"title": "Notional", "value": f"${qty*price:.2f}",  "short": True},
    ]
    payload = {
        "text": text,
        "attachments": [
            {
                "color": color,
                "fields": fields,
            }
        ],
    }
    slack_post(webhook, payload)

def slack_info(webhook: str, text: str) -> None:
    slack_post(webhook, {"text": text})

def slack_risk_flip(webhook: str, *, new_state: str, reason: str) -> None:
    payload = {
        "text": f"⚠️ Risk signal changed → *{new_state.upper()}*",
        "attachments": [
            {
                "color": "#e01e5a" if new_state.lower() == "off" else "#2eb886",
                "fields": [
                    {"title": "Reason", "value": reason or "n/a", "short": False}
                ],
            }
        ],
    }
    slack_post(webhook, payload)

# ---------- ENV ----------

EXCHANGE           = os.getenv("EXCHANGE", "kraken")
QUOTE              = os.getenv("QUOTE", "USD").upper()
UNIVERSE           = [s.strip().upper() for s in os.getenv("UNIVERSE", "BTC,ETH,SOL,DOGE,ADA,ZEC").split(",") if s.strip()]
MAX_POSITIONS      = as_int(os.getenv("MAX_POSITIONS"), 6)
USD_PER_TRADE      = as_float(os.getenv("USD_PER_TRADE"), 20.0)
RESERVE_CASH_PCT   = as_float(os.getenv("RESERVE_CASH_PCT"), 10.0)

ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)
ROTATE_WHEN_FULL       = as_bool(os.getenv("ROTATE_WHEN_FULL"), False)
DUST_MIN_USD       = as_float(os.getenv("DUST_MIN_USD"), 2.0)

TAKE_PROFIT_PCT    = as_float(os.getenv("TAKE_PROFIT_PCT"), 3.0)
STOP_LOSS_PCT      = as_float(os.getenv("STOP_LOSS_PCT"), 2.0)
TRAIL_STOP_PCT     = as_float(os.getenv("TRAIL_STOP_PCT"), 1.0)
MAX_DAILY_LOSS_PCT = as_float(os.getenv("MAX_DAILY_LOSS_PCT"), 5.0)
MAX_DAILY_ENTRIES  = as_int(os.getenv("MAX_DAILY_ENTRIES"), 6)
EMERGENCY_SL_PCT   = as_float(os.getenv("EMERGENCY_SL_PCT"), 8.0)

RUN_SWITCH         = os.getenv("RUN_SWITCH","ON").upper()
DRY_RUN            = os.getenv("DRY_RUN","ON").upper()

STATE_DIR          = os.getenv("STATE_DIR",".state")
POSITIONS_JSON     = os.getenv("POSITIONS_JSON", f"{STATE_DIR}/positions.json")
KPI_CSV            = os.getenv("KPI_CSV", f"{STATE_DIR}/kpi_history.csv")
SPEC_GATE_REPORT   = os.getenv("SPEC_GATE_REPORT", f"{STATE_DIR}/spec_gate_report.txt")

SLACK_WEBHOOK      = os.getenv("SLACK_WEBHOOK_URL","").strip()

API_KEY            = os.getenv("KRAKEN_API_KEY","")
API_SECRET         = os.getenv("KRAKEN_API_SECRET","")

# ---------- Exchange wiring ----------

def make_exchange() -> ccxt.Exchange:
    if EXCHANGE.lower() != "kraken":
        raise SystemExit("Only kraken is supported in this file (set EXCHANGE=kraken).")
    cfg = {
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    }
    if DRY_RUN != "ON":
        cfg.update({"apiKey": API_KEY, "secret": API_SECRET})
    ex = ccxt.kraken(cfg)
    ex.load_markets()
    return ex

def pair(base: str) -> str:
    return f"{base}/{QUOTE}"

# Kraken returns balances sometimes as 'USD' or 'ZUSD'
def pick_usd_key(bal: Dict[str, float]) -> str:
    for k in ("USD","ZUSD","XUSD"):
        if k in bal:
            return k
    # fall back to QUOTE key if present
    return QUOTE if QUOTE in bal else "USD"

# ---------- State ----------

def default_state() -> Dict[str, Any]:
    return {
        "positions": {},  # symbol -> {"qty":float, "entry_price":float, "trail":float}
        "last_risk": "ON",
        "today": datetime.utcnow().strftime("%Y-%m-%d"),
        "daily_pnl": 0.0,
        "daily_entries": 0,
    }

def load_state() -> Dict[str, Any]:
    s = load_json(POSITIONS_JSON, default_state())
    # migrate keys
    if "positions" not in s:
        s = default_state()
    # reset day counters if date changed
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if s.get("today") != today:
        s["today"] = today
        s["daily_pnl"] = 0.0
        s["daily_entries"] = 0
    return s

def save_state(s: Dict[str, Any]) -> None:
    ensure_dir(STATE_DIR)
    save_json(POSITIONS_JSON, s)

# ---------- Risk Signal (simple momentum breadth) ----------

def compute_risk_signal(ex: ccxt.Exchange, bases: List[str]) -> Tuple[str, str]:
    # Simple: ON if average 24h change of universe >= -2%; else OFF
    changes: List[float] = []
    for b in bases:
        sym = pair(b)
        try:
            t = ex.fetch_ticker(sym)
            pct = t.get("percentage", 0.0) or 0.0
            changes.append(float(pct))
        except Exception:
            pass
    if not changes:
        return "ON", "No data; default ON"
    avg = sum(changes)/len(changes)
    if avg < -2.0:
        return "OFF", f"Avg 24h% {avg:.2f} < -2%"
    return "ON", f"Avg 24h% {avg:.2f} >= -2%"

# ---------- Guards ----------

def violated_daily_caps(state: Dict[str, Any], equity_usd: float) -> Optional[str]:
    # MAX_DAILY_LOSS_PCT: compare daily_pnl vs equity
    if MAX_DAILY_LOSS_PCT > 0 and equity_usd > 0:
        if (state.get("daily_pnl", 0.0) / equity_usd) * 100 <= -MAX_DAILY_LOSS_PCT:
            return f"Daily loss cap hit ({state.get('daily_pnl',0.0):.2f} vs equity {equity_usd:.2f})"
    if MAX_DAILY_ENTRIES > 0 and state.get("daily_entries",0) >= MAX_DAILY_ENTRIES:
        return f"Max daily entries reached ({state['daily_entries']})"
    return None

# ---------- Sizing / balances ----------

def fetch_cash_and_equity(ex: ccxt.Exchange) -> Tuple[float, float, Dict[str,float]]:
    bal = ex.fetch_free_balance()
    usd_key = pick_usd_key(bal)
    cash = float(bal.get(usd_key, 0.0))
    equity = cash
    # rough equity: add positions using last price
    try:
        for b in UNIVERSE:
            qty = float(bal.get(b, 0.0))
            if qty > 0:
                t = ex.fetch_ticker(pair(b))
                px = float(t.get("last") or t.get("close") or t.get("ask") or 0.0)
                equity += qty * px
    except Exception:
        pass
    return cash, equity, bal

def min_cash_reserved(cash: float) -> float:
    reserve = max(0.0, cash * (RESERVE_CASH_PCT/100.0))
    return reserve

# ---------- Order exec ----------

def place_order(
    ex: ccxt.Exchange, base: str, side: str, usd_notional: float, price_hint: Optional[float]=None
) -> Tuple[float,float]:
    # returns (qty, price)
    sym = pair(base)
    t = ex.fetch_ticker(sym)
    price = float(t.get("last") or t.get("close") or t.get("ask") or t.get("bid") or 0.0)
    if price <= 0:
        raise RuntimeError(f"Bad price for {sym}")
    qty = usd_notional / price
    if DRY_RUN == "ON":
        print(f"[DRY] {side.upper()} {sym} qty={qty:.8f} px={price:.2f} notional=${usd_notional:.2f}")
        return qty, price
    # LIVE
    typ = "market"
    if side.lower() == "buy":
        o = ex.create_order(sym, typ, "buy", qty, None, {"cost": usd_notional})
    else:
        o = ex.create_order(sym, typ, "sell", qty, None, {})
    # best-effort fill price
    filled = float(o.get("filled") or qty)
    avg = float(o.get("average") or price)
    return filled, avg

# ---------- Core strategy ----------

def rank_universe_by_momentum(ex: ccxt.Exchange, bases: List[str]) -> List[Tuple[str, float]]:
    ranks: List[Tuple[str, float]] = []
    for b in bases:
        sym = pair(b)
        try:
            t = ex.fetch_ticker(sym)
            pct = float(t.get("percentage") or 0.0)
            ranks.append((b, pct))
        except Exception:
            pass
    ranks.sort(key=lambda x: x[1], reverse=True)
    return ranks

def trailing_update(trail: float, price: float, trail_pct: float, is_long: bool=True) -> float:
    if is_long:
        # ratchet up only
        new_trail = price * (1.0 - trail_pct/100.0)
        return max(trail, new_trail) if trail > 0 else new_trail
    return trail

# ---------- Spec gate report ----------

def append_spec(msg: str) -> None:
    ensure_dir(STATE_DIR)
    with open(SPEC_GATE_REPORT, "a", encoding="utf-8") as f:
        f.write(f"[{utcnow()}] {msg}\n")

# ---------- Main run ----------

def run() -> None:
    print(f"=== Crypto Live — {EXCHANGE.upper()} — {utcnow()} ===")
    print(f"Mode: {DRY_RUN} | RUN_SWITCH: {RUN_SWITCH} | UNIVERSE: {','.join(UNIVERSE)}")
    ensure_dir(STATE_DIR)
    # clear spec report for this run
    with open(SPEC_GATE_REPORT, "w", encoding="utf-8") as f:
        f.write(f"Run started {utcnow()}\n")

    if RUN_SWITCH != "ON":
        append_spec("RUN_SWITCH is OFF — exiting.")
        print("RUN_SWITCH is OFF — exiting.")
        return

    ex = make_exchange()
    state = load_state()

    # risk
    risk, reason = compute_risk_signal(ex, UNIVERSE)
    append_spec(f"Risk={risk} ({reason})")

    if risk != state.get("last_risk","ON"):
        if SLACK_WEBHOOK:
            slack_risk_flip(SLACK_WEBHOOK, new_state=risk, reason=reason)
        state["last_risk"] = risk
        save_state(state)

    # balances
    cash, equity, raw_bal = fetch_cash_and_equity(ex)
    append_spec(f"Cash={cash:.2f} Equity~={equity:.2f}")

    # daily caps
    cap_msg = violated_daily_caps(state, equity)
    if cap_msg:
        append_spec(f"Daily cap active: {cap_msg}")
        print(cap_msg)
        save_state(state)
        save_json(POSITIONS_JSON, state)  # ensure file
        return

    # build current holdings qty (exchange balances)
    holdings: Dict[str,float] = {}
    usd_key = pick_usd_key(raw_bal)
    for b in UNIVERSE:
        q = float(raw_bal.get(b, 0.0))
        if q > 0:
            holdings[b] = q

    # mark dust for sell
    to_sell_dust: List[str] = []
    for b, q in list(holdings.items()):
        try:
            px = float(ex.fetch_ticker(pair(b)).get("last") or 0.0)
            notional = q * px
            if notional < DUST_MIN_USD:
                to_sell_dust.append(b)
        except Exception:
            pass

    ranks = rank_universe_by_momentum(ex, UNIVERSE)
    append_spec(f"Top ranks: {ranks[:5]}")

    # desired set = top MAX_POSITIONS
    desired = [b for (b,_) in ranks[:MAX_POSITIONS]]

    actions: List[Tuple[str,str,float]] = []  # (side, base, usd_notional)

    # SELL rules first: TP/SL/Trail, Dust, Rotation
    for b, q in list(holdings.items()):
        sym = pair(b)
        try:
            t = ex.fetch_ticker(sym)
            px = float(t.get("last") or 0.0)
        except Exception:
            px = 0.0

        # trailing stop tracking in state
        pos = state["positions"].get(b, {"qty": q, "entry_price": px, "trail": 0.0})
        if pos.get("qty",0) <= 0:
            pos["qty"] = q
            pos["entry_price"] = px if px > 0 else pos.get("entry_price", 0.0)
            pos["trail"] = pos.get("trail", 0.0)

        entry = float(pos.get("entry_price", px))
        if entry <= 0 and px > 0:
            entry = px
            pos["entry_price"] = entry

        # update trailing
        pos["trail"] = trailing_update(float(pos.get("trail",0.0)), px, TRAIL_STOP_PCT, True)
        state["positions"][b] = pos

        pnl_pct = (px/entry - 1.0) * 100.0 if entry > 0 and px > 0 else 0.0

        should_sell = False
        sell_reason = ""

        # Dust first
        if b in to_sell_dust:
            should_sell = True
            sell_reason = f"Dust < ${DUST_MIN_USD}"

        # Emergency SL
        if not should_sell and EMERGENCY_SL_PCT > 0 and pnl_pct <= -EMERGENCY_SL_PCT:
            should_sell = True
            sell_reason = f"Emergency SL {pnl_pct:.2f}%"

        # Take profit / stop loss
        if not should_sell and TAKE_PROFIT_PCT > 0 and pnl_pct >= TAKE_PROFIT_PCT:
            should_sell = True
            sell_reason = f"TP {pnl_pct:.2f}%"
        if not should_sell and STOP_LOSS_PCT > 0 and pnl_pct <= -STOP_LOSS_PCT:
            should_sell = True
            sell_reason = f"SL {pnl_pct:.2f}%"

        # Trailing stop check
        trail = float(pos.get("trail", 0.0))
        if not should_sell and trail > 0 and px > 0 and px <= trail:
            should_sell = True
            sell_reason = f"Trail hit (px {px:.4f} <= {trail:.4f})"

        # Rotation: if full and not in desired, or cash short and found better rank
        in_desired = b in desired
        if not should_sell and ROTATE_WHEN_FULL and len(holdings) >= MAX_POSITIONS and not in_desired:
            should_sell = True
            sell_reason = "Rotate when full"
        if not should_sell and ROTATE_WHEN_CASH_SHORT and len(holdings) >= 1:
            # sell worst ranked holding if we have no cash for a buy and this isn't among top N
            pass  # handled below when planning buys

        if should_sell and q > 0 and px > 0:
            actions.append(("sell", b, q * px))  # notional for logging
            append_spec(f"Queue SELL {b}: {sell_reason}")

    # BUY planning
    # compute available cash after reserve
    cash_after_reserve = max(0.0, cash - min_cash_reserved(cash))
    # plan top picks not already held
    buys: List[str] = []
    for b in desired:
        if b not in holdings:
            buys.append(b)

    # if no cash but buys exist and ROTATE_WHEN_CASH_SHORT, sell worst-ranked current holding to fund
    if buys and cash_after_reserve < USD_PER_TRADE and ROTATE_WHEN_CASH_SHORT and holdings:
        # pick lowest ranked holding
        rank_map = {b:i for i,(b,_) in enumerate(ranks)}
        worst = None
        worst_idx = -1
        for b in holdings.keys():
            idx = rank_map.get(b, 9999)
            if idx > worst_idx:
                worst_idx = idx
                worst = b
        if worst and worst not in [b for _,b,_ in actions if _=="sell"]:
            # sell worst to free cash
            q = holdings.get(worst, 0.0)
            try:
                px = float(ex.fetch_ticker(pair(worst)).get("last") or 0.0)
            except Exception:
                px = 0.0
            if q > 0 and px > 0:
                actions.append(("sell", worst, q*px))
                append_spec(f"Queue SELL {worst}: rotate to fund buy")

    # execute SELLS first
    for side, b, notion in actions:
        if side != "sell":
            continue
        try:
            sym = pair(b)
            # compute qty from balance fresh
            bal = ex.fetch_free_balance()
            q = float(bal.get(b, 0.0))
            if q <= 0:
                continue
            qty, px = place_order(ex, b, "sell", q * px if False else q * (ex.fetch_ticker(sym).get("last") or 0.0))  # qty-based sell
            # update state
            st_pos = state["positions"].get(b, {"qty":0.0,"entry_price":px,"trail":0.0})
            realized = (px - float(st_pos.get("entry_price",px))) * qty
            state["daily_pnl"] = float(state.get("daily_pnl",0.0)) + realized
            state["positions"].pop(b, None)
            append_csv(KPI_CSV, [utcnow(),"SELL", b, "SELL", f"{qty:.8f}", f"{px:.6f}", f"{qty*px:.2f}", f"{realized:.2f}", ""])
            if SLACK_WEBHOOK:
                slack_trade_ping(
                    SLACK_WEBHOOK,
                    text=f"Sold {b} — qty {qty:.6f} @ ${px:.2f}",
                    symbol=b, side="sell", qty=qty, price=px, color="#e01e5a"
                )
        except Exception as e:
            append_spec(f"Sell error {b}: {e}")

    # refresh balances after sells
    cash, equity, raw_bal = fetch_cash_and_equity(ex)
    cash_after_reserve = max(0.0, cash - min_cash_reserved(cash))

    # recompute holdings map
    holdings = {}
    usd_key = pick_usd_key(raw_bal)
    for b in UNIVERSE:
        q = float(raw_bal.get(b, 0.0))
        if q > 0:
            holdings[b] = q

    # Limit buy count by cap
    buys_to_do = min(len(buys), max(0, MAX_POSITIONS - len(holdings)))

    for b in buys[:buys_to_do]:
        if cash_after_reserve < USD_PER_TRADE:
            append_spec(f"No cash for buy {b} (need {USD_PER_TRADE}, have {cash_after_reserve:.2f})")
            continue
        try:
            qty, px = place_order(ex, b, "buy", USD_PER_TRADE)
            # update state
            state["positions"][b] = {"qty": qty, "entry_price": px, "trail": 0.0}
            state["daily_entries"] = int(state.get("daily_entries",0)) + 1
            append_csv(KPI_CSV, [utcnow(),"BUY", b, "BUY", f"{qty:.8f}", f"{px:.6f}", f"{qty*px:.2f}", "", ""])
            if SLACK_WEBHOOK:
                slack_trade_ping(
                    SLACK_WEBHOOK,
                    text=f"Bought {b} — qty {qty:.6f} @ ${px:.2f}",
                    symbol=b, side="buy", qty=qty, price=px, color="#2eb886"
                )
            cash_after_reserve -= USD_PER_TRADE
        except Exception as e:
            append_spec(f"Buy error {b}: {e}")

    save_state(state)
    print("Run complete.")
    append_spec("Run complete.")

# ---------- Entry ----------

if __name__ == "__main__":
    try:
        if DRY_RUN == "ON":
            print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")
        else:
            print("✅ LIVE MODE — REAL ORDERS ENABLED")
        run()
    except Exception as e:
        err = "".join(traceback.format_exception(*sys.exc_info()))
        ensure_dir(STATE_DIR)
        with open(SPEC_GATE_REPORT, "a", encoding="utf-8") as f:
            f.write("\n[ERROR]\n")
            f.write(err)
        print(f"[FATAL] {e}")
        sys.exit(1)
