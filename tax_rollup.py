#!/usr/bin/env python3
import csv, os, sys
from collections import defaultdict

LEDGER = os.getenv("TAX_LEDGER_PATH", "data/tax_ledger.csv")

def main():
    if not os.path.exists(LEDGER):
        print(f"No ledger at {LEDGER}")
        sys.exit(0)

    agg = defaultdict(lambda: {"profit_usd": 0.0, "reserved_usd": 0.0})
    with open(LEDGER, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y = r.get("year") or "unknown"
            try:
                agg[y]["profit_usd"] += float(r.get("profit_usd", "0") or 0)
                agg[y]["reserved_usd"] += float(r.get("reserved_usd", "0") or 0)
            except: pass

    os.makedirs("data", exist_ok=True)
    for y, sums in agg.items():
        outp = f"data/tax_summary_{y}.csv"
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["year", "total_profit_usd", "total_reserved_usd"])
            w.writerow([y, f"{sums['profit_usd']:.2f}", f"{sums['reserved_usd']:.2f}"])
        print(f"Wrote {outp}")

if __name__ == "__main__":
    main()
