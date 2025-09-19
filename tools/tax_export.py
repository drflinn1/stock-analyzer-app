import csv, json, pathlib, datetime as dt
from collections import deque, defaultdict

LEDGER = pathlib.Path("data/tax_ledger.csv")
OUT_8949 = pathlib.Path("data/tax_8949.csv")
OUT_SUM  = pathlib.Path("data/tax_summary.json")

BASE = "USD"  # treat USDT as USD-equivalent for reporting; adjust if needed

def parse_ts(ts):
    return dt.datetime.utcfromtimestamp(int(ts))

def is_usd_like(ccy):
    return (ccy or "").upper() in {"USD","USDT","USD.S"}

def main():
    if not LEDGER.exists():
        print("[tax] no ledger found"); return

    # Read trades
    trades = []
    with LEDGER.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("event") not in (None,"","TRADE","DUST_SELL"):  # ignore non-trade events if any
                pass
            side = (row["side"] or "").lower()
            if side not in {"buy","sell"}: 
                continue
            ts = int(float(row["ts"]))
            sym = row["symbol"]
            qty = float(row["qty"])
            px  = float(row["price"])
            notional = float(row.get("notional") or (qty*px))
            fee = float(row.get("fee") or 0.0)
            fee_ccy = row.get("fee_ccy") or BASE
            note = row.get("note") or ""
            trades.append({
                "ts": ts, "dt": parse_ts(ts), "side": side, "symbol": sym,
                "qty": qty, "price": px, "notional": notional,
                "fee": fee, "fee_ccy": fee_ccy, "note": note
            })

    # sort by time
    trades.sort(key=lambda x: x["ts"])

    # Maintain FIFO lots per base asset (from SYMBOL like BASE/QUOTE)
    lots = defaultdict(deque)
    rows_8949 = []
    totals = {"short_term_gain":0.0, "long_term_gain":0.0, "proceeds":0.0, "basis":0.0}

    def asset_of(symbol):
        # expects "ASSET/USD" or "ASSET/USDT"
        return symbol.split("/")[0].upper()

    for t in trades:
        asset = asset_of(t["symbol"])
        q = t["qty"]
        px = t["price"]
        side = t["side"]
        fee = t["fee"]
        fee_is_usd = is_usd_like(t["fee_ccy"])

        if side == "buy":
            # cost is USD notional; add USD fee to basis; if fee in asset, reduce qty
            qty = q
            cost = q * px
            if fee_is_usd:
                cost += fee
            else:
                # fee in asset (rare on Kraken): reduce qty; treat basis as (qty_eff * px)
                qty = max(q - fee, 0.0)
                cost = qty * px
            lots[asset].append({"acq_dt": t["dt"], "qty": qty, "cost": cost})
        else:  # sell
            qty_to_sell = q
            proceeds = q * px
            if fee_is_usd:
                proceeds -= fee
            else:
                # fee in asset: effectively sells slightly less; adjust qty_to_sell
                qty_to_sell = max(q - fee, 0.0)
                proceeds = qty_to_sell * px

            # consume FIFO lots
            while qty_to_sell > 1e-12 and lots[asset]:
                lot = lots[asset][0]
                take = min(qty_to_sell, lot["qty"])
                basis = lot["cost"] * (take / lot["qty"])
                gain = proceeds * (take / max(q,1e-12)) - basis
                term = "LT" if (t["dt"] - lot["acq_dt"]).days > 365 else "ST"

                rows_8949.append({
                    "Description": f"{asset}",
                    "Date Acquired": lot["acq_dt"].date().isoformat(),
                    "Date Sold": t["dt"].date().isoformat(),
                    "Proceeds": round(proceeds * (take / max(q,1e-12)), 10),
                    "Cost Basis": round(basis, 10),
                    "Adjustment": 0.0,
                    "Gain/Loss": round(gain, 10),
                    "Term": term,
                    "Notes": t["note"]
                })

                totals["proceeds"] += proceeds * (take / max(q,1e-12))
                totals["basis"] += basis
                if term=="LT": totals["long_term_gain"] += gain
                else: totals["short_term_gain"] += gain

                # update lot
                lot["qty"] -= take
                lot["cost"] -= basis
                qty_to_sell -= take
                if lot["qty"] <= 1e-12:
                    lots[asset].popleft()

    # write 8949-style CSV
    OUT_8949.parent.mkdir(parents=True, exist_ok=True)
    with OUT_8949.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Description","Date Acquired","Date Sold",
            "Proceeds","Cost Basis","Adjustment","Gain/Loss","Term","Notes"
        ])
        w.writeheader()
        for row in rows_8949:
            w.writerow(row)

    with OUT_SUM.open("w") as f:
        json.dump(totals, f, indent=2)

    print(f"[tax] wrote {OUT_8949} and {OUT_SUM}")

if __name__ == "__main__":
    main()
