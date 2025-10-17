import os
import sys

# Lazy imports so the step can "succeed" even if matplotlib isn't present in some contexts
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"[make_kpi_chart] Missing deps: {e}")
    sys.exit(0)  # non-fatal for the workflow

STATE_DIR = os.environ.get("STATE_DIR", ".state")
CSV_PATH  = os.environ.get("KPI_CSV", os.path.join(STATE_DIR, "kpi_history.csv"))
IMG_PATH  = os.environ.get("KPI_IMG", os.path.join(STATE_DIR, "kpi_chart.png"))

os.makedirs(STATE_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"[make_kpi_chart] No KPI CSV at {CSV_PATH}; skipping.")
    sys.exit(0)

try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"[make_kpi_chart] Could not read KPI CSV: {e}")
    sys.exit(0)

if df.shape[0] >= 1 and "equity" in df.columns:
    try:
        plt.figure()
        plt.plot(df.index, df["equity"])
        plt.title("CryptoBOT Equity (run history)")
        plt.xlabel("Run #")
        plt.ylabel("Equity (USD)")
        plt.tight_layout()
        plt.savefig(IMG_PATH)
        print(f"[make_kpi_chart] Saved KPI chart -> {IMG_PATH}")
    except Exception as e:
        print(f"[make_kpi_chart] Failed to render/save chart: {e}")
else:
    print("[make_kpi_chart] KPI CSV present but missing 'equity' column; skipping chart.")
