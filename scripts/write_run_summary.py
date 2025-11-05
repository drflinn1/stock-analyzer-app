#!/usr/bin/env python3
"""
Always-write run summary for Crypto workflows.

- Writes both .state/run_summary.json and .state/run_summary.md
- Never touches trading logic; it only *reads* .state/position.json if present.
"""

from pathlib import Path
import json
import datetime as dt

STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)

pos_file = STATE / "position.json"
summary_json = STATE / "run_summary.json"
summary_md   = STATE / "run_summary.md"

now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

summary = {
    "when": now,
    "status": "FLAT",
    "note": "No open position",
    "position": None,
}

if pos_file.exists():
    try:
        pos = json.loads(pos_file.read_text())
        summary["status"] = "HOLD"
        summary["note"] = "Tracking current position"
        summary["position"] = {
            "symbol": pos.get("symbol"),
            "qty": pos.get("qty"),
            "entry_price": pos.get("entry_price"),
            "since": pos.get("since"),
            "last_price": pos.get("last_price"),
            "change_pct": pos.get("change_pct"),
        }
    except Exception as e:
        summary["status"] = "UNKNOWN"
        summary["note"] = f"position.json parse error: {e}"

# Write JSON
summary_json.write_text(json.dumps(summary, indent=2))

# Write Markdown
lines = [
    f"**When:** {summary['when']}",
    f"**Status:** {summary['status']}",
    f"**Note:** {summary['note']}",
]
if summary["position"]:
    p = summary["position"]
    lines += [
        "",
        "### Position",
        f"- **Symbol:** {p.get('symbol')}",
        f"- **Qty:** {p.get('qty')}",
        f"- **Entry:** {p.get('entry_price')}",
        f"- **Last Price:** {p.get('last_price')}",
        f"- **Change %:** {p.get('change_pct')}",
        f"- **Since:** {p.get('since')}",
    ]

summary_md.write_text("\n".join(lines))
print(f"Wrote {summary_json} and {summary_md}")
