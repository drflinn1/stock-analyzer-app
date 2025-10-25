#!/usr/bin/env python3
import pathlib, pandas as pd, matplotlib.pyplot as plt

CSV = pathlib.Path(".state/spike_scan/momentum_candidates.csv")
OUT = pathlib.Path(".state/spike_scan/momentum_top10.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

if not CSV.exists() or CSV.stat().st_size == 0:
    print("No CSV to chart."); raise SystemExit(0)

df = pd.read_csv(CSV)
if df.empty:
    print("Empty CSV, nothing to chart."); raise SystemExit(0)

top = df.nlargest(10, "pct_24h")[["symbol", "pct_24h"]].iloc[::-1]
plt.figure(figsize=(6,3))
plt.barh(top["symbol"], top["pct_24h"])
plt.title("Momentum Spike — Top 10 by 24h %")
plt.xlabel("% change (24h)")
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved chart → {OUT}")
