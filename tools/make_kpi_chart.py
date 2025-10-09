# tools/make_kpi_chart.py
# Reads .state/kpi_history.csv and writes .state/kpi_chart.png (last ~500 points)
import os, csv
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # headless renderer for CI
import matplotlib.pyplot as plt  # noqa: E402

STATE_DIR = os.getenv("STATE_DIR", ".state")
KPI_CSV = os.getenv("KPI_CSV", f"{STATE_DIR}/kpi_history.csv")
OUT_PNG = os.path.join(STATE_DIR, "kpi_chart.png")

def read_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            ts = (d.get("ts_utc") or "").strip()
            try:
                if "UTC" in ts:
                    when = datetime.strptime(ts.split(".")[0], "%Y-%m-%d %H:%M:%S %Z")
                else:
                    when = datetime.fromisoformat(ts.split(".")[0])
            except Exception:
                when = datetime.utcnow()
            pnl = d.get("pnl")
            try:
                pnl = float(pnl) if pnl not in (None, "") else 0.0
            except Exception:
                pnl = 0.0
            rows.append((when, pnl))
    rows.sort(key=lambda x: x[0])
    return rows[-500:]  # keep recent

def make_chart(rows):
    os.makedirs(STATE_DIR, exist_ok=True)
    plt.figure(figsize=(10, 4))
    if not rows:
        plt.title("Cumulative P&L — no data yet")
        plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
        return
    times = [r[0] for r in rows]
    pnls  = [r[1] for r in rows]
    cum = []
    s = 0.0
    for x in pnls:
        s += x
        cum.append(s)
    plt.plot(times, cum, linewidth=2)
    plt.title("Cumulative P&L")
    plt.xlabel("Time")
    plt.ylabel("USD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)

if __name__ == "__main__":
    rows = read_rows(KPI_CSV)
    make_chart(rows)
    print(f"Wrote {OUT_PNG}")
