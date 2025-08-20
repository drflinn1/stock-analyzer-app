#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone

def parse_bool(s: str) -> bool:
    if s is None:
        return False
    s = s.strip().lower()
    return s in {"1", "true", "yes", "y", "on"}

def main():
    parser = argparse.ArgumentParser(description="Headless daily rebalance runner")
    parser.add_argument("--dry-run", required=False, default="true",
                        help="true/false; if true, do not make external changes")
    parser.add_argument("--ignore-market-hours", required=False, default="false",
                        help="true/false; if true, allow running outside US market hours")
    args = parser.parse_args()

    dry_run = parse_bool(args.dry_run)
    ignore_hours = parse_bool(args.ignore_market_hours)

    # ---- placeholder business logic ---------------------------------------
    # Put your actual signal generation here. For now, we emit no-op signals
    # so the workflow is healthy and schedulable.
    signals = []  # e.g. [{"ticker":"AAPL","action":"BUY","weight":0.10}, ...]
    note = "No signals generated (placeholder script)."
    # ------------------------------------------------------------------------

    # Write a small log the workflow can upload
    with open("rebalance.log", "a", encoding="utf-8") as lf:
        lf.write(f"[{datetime.now(timezone.utc).isoformat()}] dry_run={dry_run}, "
                 f"ignore_market_hours={ignore_hours}\n")
        lf.write(note + "\n")

    # Emit a stable machine-readable artifact
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "ignore_market_hours": ignore_hours,
        "signals": signals,
        "summary": note,
    }
    with open("signals.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Append to the job summary if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write("## Rebalance summary\n")
            sf.write(f"- Dry run: **{dry_run}**\n")
            sf.write(f"- Ignore market hours: **{ignore_hours}**\n")
            sf.write(f"- Signals generated: **{len(signals)}**\n")
            sf.write("\n_No signals generated (placeholder)._ \n")

    print("Rebalance completed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
