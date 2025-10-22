#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

STATE_DIR = ".state"
CSV = os.path.join(STATE_DIR, "kpi_history.csv")
IMG = os.path.join(STATE_DIR, "kpi_chart.png")

os.makedirs(STATE_DIR, exist_ok=True)

if not os.path.isfile(CSV):
    # Create a harmless placeholder chart so upload-artifact never warns
    plt.figure()
    plt.title("KPI chart (placeholder)")
    plt.plot([0, 1], [0, 1])
    plt.savefig(IMG, bbox_inches="tight")
    print(f"Created placeholder chart at {IMG} (no {CSV})")
else:
    df = pd.read_csv(CSV)
    # Pick sensible columns if present
    y = None
    for col in ["equity", "balance", "pnl_cum", "pnl"]:
        if col in df.columns:
            y = col
            break
    if y is None:
        # Fallback to first numeric column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                y = col
                break
    if y is None:
        plt.figure()
        plt.title("KPI chart (no numeric columns)")
        plt.plot([0, 1], [0, 1])
    else:
        plt.figure()
        plt.plot(df[y])
        plt.title(f"KPI: {y}")
        plt.xlabel("run")
        plt.ylabel(y)
    plt.savefig(IMG, bbox_inches="tight")
    print(f"Wrote {IMG}")
