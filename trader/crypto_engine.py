# trader/crypto_engine.py
# Minimal, reliable crypto live engine for Kraken via ccxt
# - Auto-universe (top-K movers) + optional CORE_WHITELIST + SPEC_SYMBOLS
# - Position caps & cash reserve
# - BUY entries with guaranteed min new positions (GUARANTEE_MIN_NEW)
# - DRY_RUN support (no real orders)
#
# Logs are shaped to match your previous runs.

from __future__ import annotations
import os, math, time, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ENV ----------
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
RUN_SWITCH = os.getenv("RUN_SWITCH", "on").lower().strip()

EXCHANGE = os.getenv("EXCHANGE", "kraken").lower()

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

USD_PER_TRADE = float(os.getenv("USD_PER_TRADE", "25"))
RESERVE_USD   = float(os.getenv("RESERVE_USD", "100"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "6"))
DAILY_MAX_TRADES = int(os.getenv("DAILY_MAX_TRADES", "6"))

# Guaranteed entries if nothing else fires
GUARANTEE_MIN_NEW = int(os.getenv("GUARANTEE_MIN_NEW", "0"))  # e.g., "1" to force one buy when funds allow

# Universe building
QUOTE_ALLOW = [q.strip().upper() for q in os.getenv("QUOTE_ALLOW", "USD,USDT").split(",") if q.strip()]
CORE_WHITELIST = [s.strip().upper() for s in os.getenv("CORE_WHITELIST", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD").split(",") if s.strip()]
SPEC_SYMBOLS   = [s.strip().upper() for s in os.getenv("SPEC_SYMBOLS", "").split(",") if s.strip()]
PAIR_BLOCKLIST = set([s.strip().upper() for s in os.getenv("PAIR_BLOCKLIST", "").split(",") if s.strip()])

TOP_K = int(os.getenv("TOP_K", "6"))  # for the "Universe (auto): top 6" display
MIN_NOTIONAL_USD = float(os.getenv("MIN_NOTIONAL_USD", "20"))  # safety to avoid too-small orders

# ---------- Helpers ----------
def now_utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

def new_exchange():
    if EXCHANGE != "kraken":
        raise SystemExit(f"Only kraken is supported in this engine right now (got {EXCHANGE})")
    return ccxt.kraken({
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_API_SECRET,
        "enableRateLimit": True,
        "options": {
            "adjustForTimeDifference": True,
        },
    })

def usd_balance(balances: Dict[str, Any]) -> float:
    total = balances.get("total") or {}
    # Prefer USD cash; do NOT auto-convert USDT here to avoid surprise sells
    return float(total.get("USD", 0.0))

def list_open_bases(balances: Dict[str, Any]) -> Dict[str, float]:
    total = balances.get("total") or {}
    bases: Dict[str, float] = {}
    for asset, amt in total.items():
        if not isinstance(amt, (int, float)):
            continue
        a = float(amt)
        # treat these as quotes/cash
        if asset in ("USD", "USDT", "EUR", "GBP") or a <= 0.0:
            continue
        bases[asset] = a
    return bases

def allowed_symbol(markets: Dict[str, Any], base: str) -> Optional[str]:
    # prefer quotes in QUOTE_ALLOW order
    for q in QUOTE_ALLOW:
        sym = f"{base}/{q}"
        m = markets.get(sym)
        if m and (m.get("active", True)) and sym.upper() not in PAIR_BLOCKLIST:
            return sym
    return None

def fetch_change_pct(exch, symbol: str) -> Optional[float]:
    try:
        t = exch.fetch_ticker(symbol)
        # percent change may be in "percentage" on some ccxt exchanges
        pc = t.get("percentage")
        if pc is None:
            last = t.get("last") or t.get("close")
            openp = t.get("open")
            if last is None or openp in (None, 0):
                return None
            pc = (float(last) - float(openp)) / float(openp) * 100.0
        return float(pc)
    except Exception:
        return None

def price_precision(market: Dict[str, Any]) -> int:
    return int((market.get("precision", {}) or {}).get("price", 8) or 8)

def amount_precision(market: Dict[str, Any]) -> int:
    return int((market.get("precision", {}) or {}).get("amount", 8) or 8)

def min_amount(market: Dict[str, Any]) -> float:
    return float(((market.get("limits", {}) or {}).get("amount", {}) or {}).get("min", 0.0) or 0.0)

def min_cost(market: Dict[str, Any]) -> float:
    return float(((market.get("limits", {}) or {}).get("cost", {}) or {}).get("min", 0.0) or 0.0)

def round_to(x: float, prec: int) -> float:
    factor = 10 ** prec
    return math.floor(x * factor) / factor

# ---------- Engine ----------
def build_universe(exch, markets: Dict[str, Any]) -> List[str]:
    # Start with CORE whitelist
    symbols: List[str] = []
    for s in CORE_WHITELIST:
        if s and s in markets and s.upper() not in PAIR_BLOCKLIST:
            symbols.append(s)

    # Merge SPEC symbols explicitly
    for s in SPEC_SYMBOLS:
        if s and s in markets and s.upper() not in PAIR_BLOCKLIST:
            if s not in symbols:
                symbols.append(s)

    # Auto-pick movers for the display (and to fill up if we have room)
    movers: List[Tuple[str, float]] = []
    # Consider all markets with allowed quotes
    seen = set(symbols)
    bases_seen = set(sym.split("/")[0] for sym in seen)
    # To keep cost down, only scan bases that look like typical alts
    for sym, m in markets.items():
        try:
            base, quote = sym.split("/")
        except Exception:
            continue
        if quote.upper() not in QUOTE_ALLOW:
            continue
        if sym.upper() in PAIR_BLOCKLIST:
            continue
        if base in ("USD", "USDT", "EUR", "GBP"):
            continue
        if sym in seen:
            continue
        # skip if we already have this base via another quote
        if base in bases_seen:
            continue
        pct = fetch_change_pct(exch, sym)
        if pct is None:
            continue
        # prefer positive movers
        movers.append((sym, float(pct)))
    movers.sort(key=lambda x: x[1], reverse=True)
    for sym, _pct in movers[:max(0, TOP_K - len(symbols))]:
        symbols.append(sym)

    return symbols[:TOP_K]

def ensure_trade_allowed(bal: float) -> Tuple[float, float]:
    """returns (avail, per_trade) after reserve/sanity"""
    avail = max(0.0, bal - RESERVE_USD)
    return avail, USD_PER_TRADE

def create_market_buy(exch, symbol: str, usd_size: float) -> Tuple[bool, str]:
    """Place a market buy for ~usd_size notional, respecting min amount/cost and precision."""
    market = exch.markets[symbol]
    last = None
    try:
        t = exch.fetch_ticker(symbol)
        last = float(t.get("last") or t.get("close"))
    except Exception as e:
        return False, f"ticker failed: {e}"
    if last is None or last <= 0:
        return False, "no price"

    amt_prec = amount_precision(market)
    prc_prec = price_precision(market)
    min_amt = min_amount(market)
    cost_min = min_cost(market)

    # amount from target notional
    amt = usd_size / last
    amt = round_to(amt, amt_prec)

    # check mins
    if min_amt and amt < min_amt:
        # bump amount to min
        amt = min_amt
    notional = amt * last
    if cost_min and notional < cost_min:
        # bump notional to min cost
        target = max(usd_size, cost_min)
        amt = round_to(target / last, amt_prec)
        notional = amt * last

    # final sanity
    if notional < MIN_NOTIONAL_USD:
        return False, f"notional ${notional:.2f} < MIN_NOTIONAL_USD ${MIN_NOTIONAL_USD:.2f}"

    if DRY_RUN:
        return True, f"DRY_RUN BUY {symbol} amt={amt} last={round(last, prc_prec)} notional≈${notional:.2f}"

    try:
        o = exch.create_order(symbol, type="market", side="buy", amount=amt)
        oid = o.get("id") or "?"
        return True, f"BUY ok {symbol} amt={amt} notional≈${notional:.2f} order_id={oid}"
    except Exception as e:
        return False, f"BUY error {symbol}: {e}"

def main() -> None:
    print("============================================================")
    print("🟢 LIVE TRADING")
    print("============================================================")
    print(f"{now_utc()} INFO: Starting trader in CRYPTO mode. Dry run={DRY_RUN}. Broker=ccxt")

    if RUN_SWITCH != "on":
        print(f"{now_utc()} INFO: RUN_SWITCH={RUN_SWITCH} → exiting early.")
        return

    exch = new_exchange()
    markets = exch.load_markets()

    # Balance / holdings
    balances = exch.fetch_balance()
    usd = usd_balance(balances)
    print(f"{now_utc()} INFO: USD balance detected: ${usd:.2f}")

    open_bases = list_open_bases(balances)
    open_count = len(open_bases)

    # Universe
    symbols = build_universe(exch, markets)
    print(f"{now_utc()} INFO: Universe (auto): top {TOP_K} → {symbols}")

    # Caps
    cap_left = max(0, MAX_POSITIONS - open_count)
    avail, per_trade = ensure_trade_allowed(usd)

    # Entry loop (simple: try in listed order)
    buys = 0
    reasons: List[str] = []

    if cap_left <= 0:
        reasons.append("cap_left=0")

    if avail < per_trade:
        reasons.append(f"avail ${avail:.2f} < per_trade ${per_trade:.2f}")

    if not reasons:
        for sym in symbols:
            if cap_left <= 0:
                break
            if sym.upper() in PAIR_BLOCKLIST:
                continue
            ok, info = create_market_buy(exch, sym, per_trade)
            if ok:
                print(f"{now_utc()} INFO: {info}")
                buys += 1
                cap_left -= 1
                avail -= per_trade
                if buys >= DAILY_MAX_TRADES:
                    break
            else:
                # keep first 3 reasons for visibility
                if len(reasons) < 3:
                    reasons.append(f"{sym}: {info}")

    # Guarantee block — if we still didn't open anything and user asked for it
    if buys < GUARANTEE_MIN_NEW and cap_left > 0:
        # Re-check funds
        need_buys = GUARANTEE_MIN_NEW - buys
        # Pick from universe again; try up to need_buys entries
        forced = 0
        for sym in symbols:
            if forced >= need_buys:
                break
            if avail < per_trade:
                reasons.append("guarantee: insufficient avail")
                break
            ok, info = create_market_buy(exch, sym, per_trade)
            if ok:
                print(f"{now_utc()} INFO: GUARANTEE BUY → {info}")
                forced += 1
                buys += 1
                cap_left -= 1
                avail -= per_trade
            else:
                if len(reasons) < 5:
                    reasons.append(f"guarantee {sym}: {info}")

    if buys == 0:
        msg = f"No entry (cap_left={cap_left}, per_trade=${per_trade:.2f}, avail=${avail:.2f})"
        if reasons:
            msg += " — " + "; ".join(reasons)
        print(f"{now_utc()} INFO: {msg}")

    # KPI-ish tail line (compat with your log scanning)
    print(f"{now_utc()} INFO: KPI SUMMARY: entries={buys} open={open_count} cap_left={cap_left} usd=${usd:.2f}")
    print(f"{now_utc()} INFO: DONE.")

if __name__ == "__main__":
    main()
