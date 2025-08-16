# runner.py
import os
import json
import argparse
import time
import pathlib
import csv
from typing import Dict, Any

import engine

# Where to write artifacts (override with env if desired)
ART_DIR = pathlib.Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
ART_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(returns: Dict[str, float], path: pathlib.Path) -> None:
    """Write header + rows: symbol, percent_return."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "percent_return"])
        # Sort for stable order
        for sym, val in sorted(returns.items()):
            # keep empty string for None to match your previous behavior
            w.writerow([sym, "" if val is None else f"{float(val):.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", default=os.environ.get("DRY_RUN", "true"))
    args = parser.parse_args()

    res: Dict[str, Any] = engine.run_engine()

    # Attach dry_run flag
    dry = str(args.dry_run).lower() in ("1", "true", "yes", "y")
    res["dry_run"] = dry

    # Persist artifacts
    ts = time.strftime("%Y%m%d-%H%M%S")
    csv_path = ART_DIR / f"signals-{ts}.csv"
    json_path = ART_DIR / f"signals-{ts}.json"

    returns = res.get("computed_returns") or {}
    write_csv(returns, csv_path)

    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(res, jf, ensure_ascii=False, indent=2)

    # Helpful log lines for the Actions run
    print(f'[artifact] wrote "{csv_path.resolve()}"')
    print(f'[artifact] wrote "{json_path.resolve()}"')

    # Print compact JSON to the job log (unchanged)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
