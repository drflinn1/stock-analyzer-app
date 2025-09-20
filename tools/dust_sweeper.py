# SELL-ONLY Dust Sweeper for Kraken via ccxt
# - Tries USD or USDT quote (lower min wins)
# - Writes DUST_SELL rows into data/tax_ledger.csv (so exports include them)
import os, time, pathlib, csv
from typing import Dict, Tuple, Optional
import ccxt

def env_str(k, d=""): return os.environ.get(k, d)
def env_f(k, d=0.0):  return float(os.environ.get(k, d))
def env_b(k, d="false"): return os.environ.get(k, d).lower() == "true"

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
LEDGER = DATA_DIR / "tax_ledger.csv"
TAX_FIELDS = ["ts","event","side","symbol","qty","price","notional","fee","fee_ccy","order_id","note"]
LOG = print

DRY = env_b("DRY_RUN", "false")
EXCHANGE = env_str("EXCHANGE","kraken")
BASE = env_str("BASE_CCY","USD")

DUST_ENABLED = env_b("DUST_SWEEP_ENABLED","true")
DUST_TRIGGER = env_f("DUST_SWEEP_USD_TRIGGER", 1.0)
DUST_MAX_USD = env_f("DUST_MAX_NOTIONAL_USD", 20.0)
DUST_SKIP = {s.strip().upper() for s in env_str("DUST_SKIP_ASSETS","USD,USDT,USDC").split(",") if s.strip()}

if not DUST_ENABLED:
    LOG("[dust] disabled; skipping"); raise SystemExit(0)

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

def market_limits(symbol: str) -> Tuple[float,float]:
    m = MARKETS.get(symbol, {})
    amt_min = float(((m.get("limits") or {}).get("amount") or {}).get("min") or 0)
    cost_min = float(((m.get("limits") or {}).get("cost") or {}).get("min") or 0)
    return max(0.0, amt_min), max(0.0, cost_min)

def best_pair_with_lowest_min(asset: str) -> Optional[str]:
    candidates = []
    for q in (BASE, "USDT"):
        s = f"{asset}/{q}"
        if s in MARKETS:
            amt_min, cost_min = market_limits(s)
            candidates.append((cost_min, amt_min, s))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]

def append_tax(row: Dict):
    write_header = not LEDGER.exists()
    with LEDGER.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TAX_FIELDS)
        if write_header: w.writeheader()
        w.writerow(row)

def _fee_from_order(od):
    fee = 0.0; fee_ccy = None
    f = (od or {}).get("fee")
    if f:
        fee = float(f.get("cost") or 0); fee_ccy = f.get("currency")
    fl = (od or {}).get("fees") or []
    if fl and fee == 0 and isinstance(fl, list):
        fee = float(fl[0].get("cost") or 0); fee_ccy = fl[0].get("currency")
    return fee, fee_ccy

def fetch_positions() -> Dict[str,dict]:
    bal = ex.fetch_balance()
    total = bal.get("total",{})
    positions = {}
    for asset, qty in total.items():
        if asset.upper() in DUST_SKIP or asset.upper()==BASE.upper(): 
            continue
        q = float(qty or 0.0)
        if q <= 0: 
            continue
        sym = best_pair_with_lowest_min(asset)
        if not sym: 
            continue
        try:
            px = float(ex.fetch_ticker(sym)["last"])
        except Exception:
            continue
        positions[sym] = {"asset":asset,"qty":q,"price":px,"usd":q*px}
    return positions

def create_sell(symbol, qty):
    if DRY:
        return {"id":"DRY", "symbol":symbol, "side":"sell", "qty":qty, "average": None, "cost": None, "filled": qty}
    try:
        return ex.create_market_sell_order(symbol, qty)
    except Exception as e:
        LOG(f"[dust] sell fail {symbol} {qty}: {e}")
        return None

def main():
    LOG("[dust] starting sweep (SELLS ONLY)")
    positions = fetch_positions()
    if not positions:
        LOG("[dust] no positions found"); return

    dust = {sym:info for sym,info in positions.items() if info["usd"] <= DUST_MAX_USD}
    total_dust = sum(info["usd"] for info in dust.values())
    LOG(f"[dust] candidates={len(dust)} total_dust=${total_dust:.2f} (trigger=${DUST_TRIGGER:.2f})")

    if total_dust < DUST_TRIGGER:
        LOG("[dust] under trigger; skip entire sweep"); return

    sold = 0; skipped = 0
    for sym, info in sorted(dust.items(), key=lambda kv: kv[1]["usd"], reverse=True):
        amt_min, cost_min = market_limits(sym)
        qty = info["qty"]; px = info["price"]; usd = qty * px
        min_cost = max(cost_min, 0.0)
        if qty < max(amt_min, 0.0) or usd < min_cost:
            LOG(f"[dustskip] {sym} qty={qty:.8g} ${usd:.2f} < min (amount>={amt_min} cost>={min_cost})")
            skipped += 1; continue
        od = create_sell(sym, qty)
        if od:
            avg = od.get("average") or ( (od.get("cost") or 0.0) / (od.get("filled") or qty or 1.0) ) or px
            fee, fee_ccy = _fee_from_order(od)
            append_tax({
                "ts": int(time.time()),
                "event": "DUST_SELL",
                "side": "sell",
                "symbol": sym,
                "qty": qty,
                "price": float(avg),
                "notional": float(avg) * qty,
                "fee": float(fee),
                "fee_ccy": fee_ccy or BASE,
                "order_id": od.get("id"),
                "note": "DUST"
            })
            LOG(f"[dustsell] {sym} qty={qty:.8g} ~${usd:.2f} (min_cost={min_cost}) id={od.get('id')}")
            sold += 1
        else:
            skipped += 1
        time.sleep(0.2)

    LOG(f"[dust] done: sold={sold} skipped={skipped}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOG(f"[dust] error: {e}")
        raise
