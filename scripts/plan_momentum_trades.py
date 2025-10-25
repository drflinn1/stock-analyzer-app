#!/usr/bin/env python3
"""
Momentum Spike — Paper Trade Planner (PRIORITIZE SPIKES)
Reads .state/spike_scan/momentum_candidates.csv and builds a dry-run trade plan.

Priority logic:
1) Take coins with pct_24h >= SPIKE_FOCUS_PCT first (sorted by pct_24h desc, then volume).
2) If we still have capacity, fill remaining slots from the rest (sorted by pct_24h desc, then volume).
"""
import os, json, pathlib, pandas as pd
from datetime import datetime, timezone

# ---- Paths ----
CSV_PATH = pathlib.Path(".state/spike_scan/momentum_candidates.csv")
PLAN_DIR = pathlib.Path(".state/spike_plan")
PLAN_DIR.mkdir(parents=True, exist_ok=True)
PLAN_PATH = PLAN_DIR / "trades.json"
REPORT_PATH = PLAN_DIR / "plan_report.txt"

# ---- Config (env/inputs) ----
ACCOUNT_BAL_USD   = float(os.getenv("ACCOUNT_BAL_USD",   "500"))
MAX_POSITIONS     = int(os.getenv("MAX_POSITIONS",       "3"))
MIN_BUY_USD       = float(os.getenv("MIN_BUY_USD",       "10"))
MAX_BUY_USD       = float(os.getenv("MAX_BUY_USD",       "50"))
RESERVE_CASH_PCT  = float(os.getenv("RESERVE_CASH_PCT",  "10"))
EXCLUDE_STABLES   = os.getenv("EXCLUDE_STABLES", "true").lower() == "true"

# NEW: prioritize big movers first
SPIKE_FOCUS_PCT   = float(os.getenv("SPIKE_FOCUS_PCT", "50"))  # e.g., 50%+

STABLES = {"USDT","USDC","DAI","TUSD","FDUSD","GUSD","USDP","USD"}

# ---- Load candidates ----
if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
    print("No momentum_candidates.csv found; nothing to plan.")
    raise SystemExit(0)

df = pd.read_csv(CSV_PATH)
if df.empty:
    print("Empty candidate list; aborting.")
    raise SystemExit(0)

if EXCLUDE_STABLES:
    df = df[~df["quote"].isin(STABLES)]

# Sort helper: highest % first, then highest volume
df = df.sort_values(["pct_24h","quote_volume_24h"], ascending=[False, False])

# Split into spikes vs rest
spikes = df[df["pct_24h"] >= SPIKE_FOCUS_PCT]
rest   = df[df["pct_24h"] < SPIKE_FOCUS_PCT]

# Select up to MAX_POSITIONS
selected = []
selected.extend(spikes.head(MAX_POSITIONS).to_dict("records"))
if len(selected) < MAX_POSITIONS:
    slots = MAX_POSITIONS - len(selected)
    selected.extend(rest.head(slots).to_dict("records"))

# Budgeting
available_cash = ACCOUNT_BAL_USD * (1 - RESERVE_CASH_PCT / 100)
if len(selected) == 0:
    budget_per = 0
else:
    # Clamp per-trade between MIN and MAX
    budget_per = max(MIN_BUY_USD, min(MAX_BUY_USD, available_cash / len(selected)))

trades = []
for r in selected:
    price = float(r["last_price"])
    alloc = round(budget_per, 2)
    est_qty = round(alloc / price, 6) if price > 0 else 0.0
    trades.append({
        "symbol": r["symbol"],
        "allocation_usd": alloc,
        "last_price": price,
        "est_qty": est_qty,
        "pct_24h": float(r["pct_24h"]),
        "quote_vol": float(r["quote_volume_24h"]),
        "ema_slope": float(r["ema_slope"]),
        "priority_bucket": "SPIKE" if r["pct_24h"] >= SPIKE_FOCUS_PCT else "REST",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

plan = {
    "summary": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "positions_planned": len(trades),
        "account_balance_usd": ACCOUNT_BAL_USD,
        "reserve_pct": RESERVE_CASH_PCT,
        "budget_per_trade_usd": budget_per,
        "spike_focus_pct": SPIKE_FOCUS_PCT,
    },
    "trades": trades
}

PLAN_PATH.write_text(json.dumps(plan, indent=2))

lines = [
    f"Momentum Spike — Paper Plan  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    f"Selected: {len(trades)}  |  Spike focus: ≥ {SPIKE_FOCUS_PCT:.0f}%",
    f"Cash available: ${available_cash:,.2f}  |  Budget per trade: ${budget_per:,.2f}",
    ""
]
if trades:
    for t in trades:
        lines.append(
            f"{t['symbol']:<12} [{t['priority_bucket']}]  "
            f"${t['allocation_usd']:>6.2f}  qty≈{t['est_qty']}  "
            f"24h%={t['pct_24h']:.2f}  vol≈${t['quote_vol']:,.0f}"
        )
else:
    lines.append("No trades selected. Re-run the scanner with looser thresholds or set SPIKE_FOCUS_PCT lower.")

REPORT_PATH.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved trade plan → {PLAN_PATH}")
