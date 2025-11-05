#!/usr/bin/env python3
from pathlib import Path
import json, datetime as dt

STATE = Path(".state")
STATE.mkdir(exist_ok=True)
pos_file = STATE / "position.json"
summary_json = STATE / "run_summary.json"
summary_md   = STATE / "run_summary.md"

now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

data = {"when": now, "status": "FLAT", "note": "No open position", "position": None}

if pos_file.exists():
    try:
        pos = json.loads(pos_file.read_text())
        data["status"] = "HOLD"
        data["note"]   = "Tracking current position"
        data["position"] = {
            "symbol": pos.get("symbol"),
            "qty": pos.get("qty"),
            "entry_price": pos.get("entry_price"),
            "since": pos.get("since"),
        }
    except Exception as e:
        data["status"] = "UNKNOWN"
        data["note"]   = f"position.json parse error: {e}"

# write JSON
summary_json.write_text(json.dumps(data, indent=2))

# write Markdown
lines = [
    f"**When:** {data['when']}",
    f"**Status:** {data['status']}",
    f"**Note:** {data['note']}",
]
if data["position"]:
    p = data["position"]
    lines += [
        "",
        "### Position",
        f"- **Symbol:** {p.get('symbol')}",
        f"- **Qty:** {p.get('qty')}",
        f"- **Entry:** {p.get('entry_price')}",
        f"- **Since:** {p.get('since')}",
    ]
summary_md.write_text("\n".join(lines))
print(f"Wrote {summary_json} and {summary_md}")
