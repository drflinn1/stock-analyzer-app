#!/usr/bin/env python3
# Usage: python tools/make_kpi_chart.py .state/kpi_history.csv .state/kpi_chart.png
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def pick_time_column(df):
    for c in ["timestamp", "time", "date", "datetime", "run_time"]:
        if c in df.columns:
            return c
    return df.columns[0]  # fall back to first column

def pick_value_column(df):
    # try common KPI fields in preference order
    for c in ["equity", "balance", "nav", "pnl", "pnl_pct", "daily_pnl", "kpi"]:
        if c in df.columns:
            return c
    # pick last numeric column
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[-1] if nums else df.columns[-1]

def main(csv_path, out_path):
    csv = Path(csv_path)
    out = Path(out_path)
    if not csv.exists():
        print(f"[KPI] CSV not found: {csv}")
        return

    df = pd.read_csv(csv)
    if df.empty:
        print("[KPI] CSV is empty; nothing to plot.")
        return

    tcol = pick_time_column(df)
    vcol = pick_value_column(df)

    # parse time if looks like datetime
    try:
        df[tcol] = pd.to_datetime(df[tcol])
    except Exception:
        pass

    plt.figure(figsize=(9, 4.5))
    plt.plot(df[tcol], df[vcol], linewidth=2)
    plt.title(f"KPI — {vcol}")
    plt.xlabel(tcol)
    plt.ylabel(vcol)
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    print(f"[KPI] Chart saved to {out}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: make_kpi_chart.py <kpi_csv> <out_png>")
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
