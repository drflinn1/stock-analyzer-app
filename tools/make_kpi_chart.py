#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

STATE_DIR = ".state"
CSV = os.path.join(STATE_DIR, "kpi_history.csv")
IMG = os.path.join(STATE_DIR, "kpi_chart.png")

os.makedirs(STATE_DIR, exist_ok=True)

def numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def pick_best_series(df: pd.DataFrame):
    preferred = ["equity", "balance", "pnl_cum", "pnl", "cash", "positions_value"]
    for col in preferred:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    nums = numeric_columns(df)
    return nums[0] if nums else None

def placeholder(msg: str):
    plt.figure()
    plt.title(f"KPI chart\n{msg}")
    plt.plot([0, 1], [0, 1])
    plt.tight_layout()
    plt.savefig(IMG, bbox_inches="tight")
    print(msg)

if not os.path.isfile(CSV):
    placeholder(f"No {CSV} found yet (first runs). Created placeholder chart at {IMG}.")
    sys.exit(0)

try:
    df = pd.read_csv(CSV)
except Exception as e:
    placeholder(f"Could not read {CSV}: {e}. Created placeholder chart.")
    sys.exit(0)

target = pick_best_series(df)
if target is None:
    placeholder(f"No numeric columns detected in {CSV}. Created placeholder chart.")
    sys.exit(0)

plt.figure()
plt.plot(df[target])
plt.title(f"KPI: {target}")
plt.xlabel("run index")
plt.ylabel(target)
plt.tight_layout()
plt.savefig(IMG, bbox_inches="tight")
print(f"Wrote {IMG} using column '{target}'.")
