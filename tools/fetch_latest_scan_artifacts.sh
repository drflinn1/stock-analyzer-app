#!/usr/bin/env bash
# tools/fetch_latest_scan_artifacts.sh
# Pulls the most recent artifacts from Momentum — Scan (Real) and Momentum — Spike Scan (Real)
# and extracts them into .state/ so the rotation job can read them.

set -euo pipefail

REPO="${GITHUB_REPOSITORY}"
TOKEN="${GITHUB_TOKEN}"

mkdir -p .state

# Ensure tools available
sudo apt-get update -y >/dev/null
sudo apt-get install -y jq unzip >/dev/null

pull_artifact () {
  local NAME="$1"
  local API_URL="https://api.github.com/repos/${REPO}/actions/artifacts?per_page=100"

  echo "Fetching latest artifact named: ${NAME}"
  local ARCHIVE_URL
  ARCHIVE_URL="$(curl -s \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "${API_URL}" \
    | jq -r --arg NAME "${NAME}" \
      '[.artifacts[] | select(.name==$NAME and .expired==false)] | sort_by(.created_at) | last | .archive_download_url')"

  if [[ -n "${ARCHIVE_URL}" && "${ARCHIVE_URL}" != "null" ]]; then
    curl -sSL -H "Authorization: Bearer ${TOKEN}" -o "/tmp/${NAME}.zip" "${ARCHIVE_URL}"
    unzip -o "/tmp/${NAME}.zip" -d .state/ >/dev/null
    echo "✓ Downloaded and extracted ${NAME}"
  else
    echo "⚠️  No non-expired artifact found for ${NAME} (yet)."
  fi
}

pull_artifact "momentum_scan_outputs"
pull_artifact "spike_scan_outputs"

# Best-effort normalize to our schema (symbol, quote, rank)
python tools/csv_schema_fix.py .state/momentum_candidates.csv || true
python tools/csv_schema_fix.py .state/spike_candidates.csv || true

echo "Contents of .state after fetch:"
ls -la .state || true
