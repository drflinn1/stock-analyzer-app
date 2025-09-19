# SELL-ONLY Dust Sweeper for Kraken via ccxt
# - Sells tiny balances ("dust") when they are individually sellable
# - Respects exchange min amount and min notional per market
# - Never buys. Safe to run every schedule before main.py

import os, time, json, pathlib, datetime as dt
from typing import Dict, Tuple
import ccxt

def env_str(k, d=""): return os.environ.get(k, d)
def env_f(k, d=0.0):  return float(os.environ.get(k, d))
def env_b(k, d="false"): return os.environ.get(k, d).lower() == "true"

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG = print

# ---- env ----
DRY = env_b("DRY_RUN", "false")          # default false in live
EXCHANGE = env_str("EXCHANGE","kraken")
BASE = env_str("BASE_CCY","USD")

DUST_ENABLED = env_b("DUST_SWEEP_ENABLED","true")
DUST_TRIGGER = env_f("DUST_SWEEP_USD_TRIGGER", 15.0)  # only sweep if sum(dust) >= trigger
DUST_MAX_USD = env_f("DUST_MAX_NOTIONAL_USD", 12.0)   # classify positions <= this as "dust"
DUST_SKIP = {s.strip().upper() for s in env_str("DUST_SKIP_ASSETS","USD,USDT,USDC").split(",") if s.strip()}

if not DUST_ENABLED:
    LOG("[dust] disabled; skipping")
    raise SystemExit(0)

def make_exchange():
    if EXCHANGE.lower()=="kraken":
        return ccxt.kraken({
            "apiKey": env_str("KRAKEN_API_KEY",""),
            "secret": env_str("KRAKEN_API_SECRET",""),
            "enableRateLimit": True,
        })
    raise RuntimeError("Only Kraken wired.")

ex = make_exchange()
MARKETS = ex.load_markets()

def best_symbol_for_asset(asset: str) -> str | None:
    # Prefer BASE (USD), fallback USDT
    for q in (BASE, "USDT"):
        s = f"{asset}/{q}"
        if s in MARKETS:
            return s
    return None

def market_limits(symbol: str) -> Tuple[float,float]:
    m = MARKETS.get(symbol, {})
    amt_min = float(((m.get("limits") or {}).get("amount") or {}).get("min") or 0)
    cost_min = float(((m.get("limits") or {}).get("cost") or {}).get("min") or 0)
    return max(0.0, amt_min), max(0.0, cost_min)

def fetch_equity_snapshot():
    bal = ex.fetch_balance()
    total = bal.get("total",{})
    positions = {}
    usd_equity = float(total.get(BASE,0) or 0)
    for asset, qty in total.items():
        if asset.upper() in DUST_SKIP or asset.upper()==BASE.upper(): continue
        q = float(qty or 0.0)
        if q <= 0: continue
        sym = best_symbol_for_asset(asset)
        if not sym: continue
        try:
            px = float(ex.fetch_ticker(sym)["last"])
        except Exception:
            continue
        notional = q * px
        positions[sym] = {"asset":asset,"qty":q,"price":px,"usd":notional}
        usd_equity += notional
    return positions, usd_equity

def create_sell(symbol, qty):
    if DRY:
        return {"id":"DRY", "symbol":symbol, "side":"sell", "qty":qty}
    try:
        return ex.create_market_sell_order(symbol, qty)
    except Exception as e:
        LOG(f"[dust] sell fail {symbol} {qty}: {e}")
        return None

def main():
    LOG("[dust] starting sweep (SELLS ONLY)")
    positions, _ = fetch_equity_snapshot()
    if not positions:
        LOG("[dust] no positions found"); return

    # classify dust
    dust = {sym:info for sym,info in positions.items() if info["usd"] <= DUST_MAX_USD}
    total_dust = sum(info["usd"] for info in dust.values())
    LOG(f"[dust] candidates={len(dust)} total_dust=${total_dust:.2f} (trigger=${DUST_TRIGGER:.2f})")

    if total_dust < DUST_TRIGGER:
        LOG("[dust] under trigger; skip entire sweep"); return

    # attempt to sell each dust position if above exchange min
    sold = 0; skipped = 0
    for sym, info in sorted(dust.items(), key=lambda kv: kv[1]["usd"], reverse=True):
        amt_min, cost_min = market_limits(sym)
        qty = info["qty"]; px = info["price"]; usd = qty * px
        min_cost = max(cost_min, 0.0)
        if qty < max(amt_min, 0.0) or usd < min_cost:
            LOG(f"[dustskip] {sym} qty={qty:.8g} ${usd:.2f} < min (amount>={amt_min} cost>={min_cost})")
            skipped += 1
            continue
        od = create_sell(sym, qty)
        if od:
            LOG(f"[dustsell] {sym} qty={qty:.8g} ~${usd:.2f} (min_cost={min_cost}) id={od.get('id')}")
            sold += 1
        else:
            skipped += 1
        time.sleep(0.2)  # be nice to API

    LOG(f"[dust] done: sold={sold} skipped={skipped}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOG(f"[dust] error: {e}")
        raise
