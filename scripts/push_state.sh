#!/usr/bin/env bash
set -euo pipefail

# Usage: .github/scripts/push_state.sh "Update .state (Momentum Scan)"

COMMIT_MSG="${1:-"Update .state artifacts"}"

git config user.name  "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

# Make sure we’re on the checked-out branch (Actions defaults to detached unless fetch-depth=0).
# The checkout step in workflows already sets fetch-depth: 0 and checks out the ref.

# Stage only what we care about
git add .state || true

# If there’s nothing to commit, we’re done
if git diff --cached --quiet; then
  echo "No .state changes to commit."
  exit 0
fi

# Rebase onto latest remote and attempt push with lease, retrying a couple times if a race happens.
MAX_TRIES=3
TRY=1
while true; do
  echo "Attempt $TRY/$MAX_TRIES: rebase-and-push…"
  git fetch origin
  git pull --rebase origin "$(git rev-parse --abbrev-ref HEAD)"
  git commit -m "$COMMIT_MSG" || true
  set +e
  git push --force-with-lease
  STATUS=$?
  set -e
  if [ $STATUS -eq 0 ]; then
    echo "Push succeeded."
    break
  fi
  if [ $TRY -ge $MAX_TRIES ]; then
    echo "Push failed after $MAX_TRIES attempts."
    exit 1
  fi
  TRY=$((TRY+1))
  # brief backoff
  sleep $((2*TRY))
done
