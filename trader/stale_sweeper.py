# trader/stale_sweeper.py
# Time-based "stale" exit for spot crypto on Kraken via CCXT.
# - Closes positions that have gone sideways too long.
# - Safe in DRY_RUN=true (logs only).
# - Persists per-symbol meta in .state/stale_meta.json and cooldowns in .state/stale_cooldown.json

from __future__ import annotations
import os, json, time, math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"[STALE] ccxt required: {e}")

STATE_DIR = Path(".state")
STATE_DIR.mkdir(exist_ok=True, parents=True)
META_PATH = STATE_DIR / "stale_meta.json"
COOLDOWN_PATH = STATE_DIR / "stale_cooldown.json"

# ---- Config from env (with sane defaults) ----
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

STALE_MAX_HOURS       = float(os.getenv("STALE_MAX_HOURS", "24"))     # age threshold
STALE_MIN_MOVE_PCT    = float(os.getenv("STALE_MIN_MOVE_PCT", "0.8")) # below this absolute move → stale
STALE_MIN_RANGE_PCT   = float(os.getenv("STALE_MIN_RANGE_PCT", "1.0"))# below this hi/lo range → stale
STALE_COOLDOWN_HOURS  = float(os.getenv("STALE_COOLDOWN_HOURS", "6")) # don't re-enter immediately

QUOTE_ALLOW = [q.strip().upper() for q in os.getenv("QUOTE_ALLOW", "USD,USDT").split(",") if q.strip()]
PAIR_BLOCKLIST = set([p.strip().upper() for p in os.getenv("PAIR_BLOCKLIST", "").split(",") if p.strip()])

KRAKEN_API_KEY    = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

def _now() -> float:
    return time.time()

def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def pick_symbol_for_base(markets: Dict[str, Any], base: str) -> Optional[str]:
    # Prefer USD, then USDT (or whatever QUOTE_ALLOW orders)
    for q in QUOTE_ALLOW:
        sym = f"{base}/{q}"
        m = markets.get(sym)
        if m and (m.get("active", True)) and sym.upper() not in PAIR_BLOCKLIST:
            return sym
    return None

def fetch_last(exchange, symbol: str) -> Optional[float]:
    try:
        t = exchange.fetch_ticker(symbol)
        return float(t.get("last") or t.get("close"))
    except Exception as e:
        print(f"[STALE] WARN: fetch_ticker({symbol}) failed: {e}")
        return None

def place_market_sell(exchange, symbol: str, amount: float) -> Tuple[bool, str]:
    try:
        if DRY_RUN:
            print(f"[STALE] DRY_RUN SELL {symbol} amount={amount}")
            return True, "DRY_RUN"
        o = exchange.create_order(symbol, type="market", side="sell", amount=amount)
        oid = o.get("id") or "?"
        print(f"[STALE] SELL OK {symbol} amount={amount} order_id={oid}")
        return True, str(oid)
    except Exception as e:
        print(f"[STALE] ERROR selling {symbol}: {e}")
        return False, str(e)

def main() -> None:
    print("========================================================")
    print("🕒 STALE SWEEPER — time-based exits (STALE rule)")
    print(f"cfg: max_hours={STALE_MAX_HOURS}h  min_move={STALE_MIN_MOVE_PCT}%  min_range={STALE_MIN_RANGE_PCT}%  cooldown={STALE_COOLDOWN_HOURS}h  dry_run={DRY_RUN}")
    print(f"cfg: quote_allow={QUOTE_ALLOW}  pair_blocklist={sorted(PAIR_BLOCKLIST) if PAIR_BLOCKLIST else '∅'}")

    # Exchange
    kraken = ccxt.kraken({
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_API_SECRET,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })

    markets = kraken.load_markets()
    balances = kraken.fetch_balance()
    total = balances.get("total", {})

    meta = load_json(META_PATH, {})
    cooldowns = load_json(COOLDOWN_PATH, {})

    now = _now()
    cooldown_secs = int(STALE_COOLDOWN_HOURS * 3600.0)
    max_age_secs = int(STALE_MAX_HOURS * 3600.0)

    # Walk through non-stable holdings
    for base, amt in sorted(total.items()):
        if not isinstance(amt, (int, float)):
            continue
        if base in ("USD", "USDT", "EUR", "GBP") or amt <= 0:
            continue

        symbol = pick_symbol_for_base(markets, base)
        if not symbol:
            print(f"[STALE] Skip {base} — no allowed market with quotes {QUOTE_ALLOW}")
            continue

        market = markets[symbol]
        # respect min trade size
        min_amt = (market.get("limits", {}).get("amount", {}) or {}).get("min", 0.0) or 0.0
        precision = market.get("precision", {}).get("amount", 8) or 8

        last = fetch_last(kraken, symbol)
        if last is None:
            continue

        # Initialize meta for this symbol
        m = meta.get(symbol)
        if not m:
            m = {
                "first_seen": now,
                "entry_price": last,     # first seen approximation
                "hhv": last,             # high since entry
                "llv": last,             # low since entry
            }
        else:
            # Track range since entry
            m["hhv"] = max(float(m.get("hhv", last)), last)
            m["llv"] = min(float(m.get("llv", last)), last)

        age_secs = now - float(m.get("first_seen", now))
        age_hours = age_secs / 3600.0

        move_pct = abs((last - float(m["entry_price"])) / float(m["entry_price"])) * 100.0
        range_pct = ((float(m["hhv"]) - float(m["llv"])) / float(m["entry_price"])) * 100.0

        # Cooldown?
        cd_until = float(cooldowns.get(symbol, 0))
        if now < cd_until:
            rem_h = (cd_until - now) / 3600.0
            print(f"[STALE] {symbol}: on cooldown for {rem_h:.1f}h — skipping")
            meta[symbol] = m
            continue

        # Decide stale exit
        stale = (age_secs >= max_age_secs) and (move_pct < STALE_MIN_MOVE_PCT) and (range_pct <= STALE_MIN_RANGE_PCT)

        print(f"[STALE] {symbol}: age={age_hours:.1f}h move={move_pct:.2f}% range={range_pct:.2f}% amount={amt}")

        if stale:
            # Ensure amount meets min size; otherwise skip to avoid rejection
            sell_amt = float(amt)
            if min_amt and sell_amt < min_amt:
                print(f"[STALE] {symbol}: amount {sell_amt} < min {min_amt} — cannot place market sell")
            else:
                ok, info = place_market_sell(kraken, symbol, round(sell_amt, precision))
                if ok:
                    # Start cooldown, and reset meta so if it remains in wallet we re-track
                    cooldowns[symbol] = now + cooldown_secs
                    m = {
                        "first_seen": now,    # reset tracking
                        "entry_price": last,
                        "hhv": last,
                        "llv": last,
                    }
                    print(f"[STALE] SELL — STALE ({age_hours:.1f}h, move {move_pct:.2f}%, range {range_pct:.2f}%)")
                else:
                    print(f"[STALE] ERROR — sell failed: {info}")

        meta[symbol] = m

    # Persist
    save_json(META_PATH, meta)
    save_json(COOLDOWN_PATH, cooldowns)
    print("[STALE] Done. Meta saved to", META_PATH, "Cooldowns to", COOLDOWN_PATH)

if __name__ == "__main__":
    main()
