#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/push_state.sh "Commit message"
COMMIT_MSG="${1:-"Update .state artifacts"}"

git config user.name  "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
# Silence the "ignored by .gitignore" hint (cosmetic)
git config advice.addIgnoredFile false || true

# Always force-add .state even if .gitignore ignores it
# (The -f is the key change)
git add -f .state || true

# If nothing staged, we're done
if git diff --cached --quiet; then
  echo "No .state changes to commit."
  exit 0
fi

# Figure out branch name safely (Actions can be detached)
BRANCH="$(git branch --show-current || echo "${GITHUB_REF_NAME:-main}")"

MAX_TRIES=3
TRY=1
while true; do
  echo "Attempt $TRY/$MAX_TRIES: rebase-and-push…"
  git fetch origin
  git pull --rebase origin "$BRANCH" || true
  git commit -m "$COMMIT_MSG" || true
  set +e
  git push --force-with-lease origin "$BRANCH"
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
  sleep $((2*TRY))
done
