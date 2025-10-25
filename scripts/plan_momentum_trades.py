#!/usr/bin/env python3
"""
Momentum Spike — Paper Trade Planner
Reads the latest .state/spike_scan/momentum_candidates.csv
and builds a dry-run trade plan with guardrails.
"""

import os, json, pathlib, pandas as pd
from datetime import datetime, timezone

# ---------- Config ---------- #
CSV_PATH = pathlib.Path(".state/spike_scan/momentum_candidates.csv")
PLAN_DIR = pathlib.Path(".state/spike_plan")
PLAN_DIR.mkdir(parents=True, exist_ok=True)
PLAN_PATH = PLAN_DIR / "trades.json"
REPORT_PATH = PLAN_DIR / "plan_report.txt"

# Basic env vars / defaults
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
MIN_BUY_USD = float(os.getenv("MIN_BUY_USD", "10"))
MAX_BUY_USD = float(os.getenv("MAX_BUY_USD", "50"))
RESERVE_CASH_PCT = float(os.getenv("RESERVE_CASH_PCT", "10"))
ACCOUNT_BAL_USD = float(os.getenv("ACCOUNT_BAL_USD", "500"))
EXCLUDE_STABLES = os.getenv("EXCLUDE_STABLES", "true").lower() == "true"

STABLES = {"USDT","USDC","DAI","TUSD","FDUSD","GUSD","USDP","USD"}

# ---------- Load candidates ---------- #
if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
    print("No momentum_candidates.csv found; nothing to plan.")
    raise SystemExit(0)

df = pd.read_csv(CSV_PATH)
if df.empty:
    print("Empty candidate list; aborting.")
    raise SystemExit(0)

# Filter stables if needed
if EXCLUDE_STABLES:
    df = df[~df["quote"].isin(STABLES)]

# Take top N
df = df.head(MAX_POSITIONS)

# Budget per trade
available_cash = ACCOUNT_BAL_USD * (1 - RESERVE_CASH_PCT / 100)
budget_per = min(MAX_BUY_USD, available_cash / max(1, len(df)))

# Build plan
trades = []
for row in df.itertuples(index=False):
    trades.append({
        "symbol": row.symbol,
        "allocation_usd": round(budget_per, 2),
        "last_price": float(row.last_price),
        "est_qty": round(budget_per / float(row.last_price), 6),
        "pct_24h": float(row.pct_24h),
        "quote_vol": float(row.quote_volume_24h),
        "ema_slope": float(row.ema_slope),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

plan = {
    "summary": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "positions_planned": len(trades),
        "account_balance_usd": ACCOUNT_BAL_USD,
        "reserve_pct": RESERVE_CASH_PCT,
        "budget_per_trade_usd": budget_per
    },
    "trades": trades
}

PLAN_PATH.write_text(json.dumps(plan, indent=2))

lines = [
    f"Momentum Spike — Paper Plan  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    f"Candidates planned: {len(trades)}",
    f"Cash available: ${available_cash:,.2f}  |  Budget per trade: ${budget_per:,.2f}",
    ""
]
for t in trades:
    lines.append(f"{t['symbol']:<10} → ${t['allocation_usd']:>6.2f}  qty≈{t['est_qty']}  24h%={t['pct_24h']:.2f}")
REPORT_PATH.write_text("\n".join(lines))

print("\n".join(lines))
print(f"\nSaved trade plan → {PLAN_PATH}")
