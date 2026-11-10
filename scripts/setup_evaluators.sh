#!/usr/bin/env bash
# Fetch the official Spider 1 evaluator at a pinned commit.
# (Spider 2.0-Lite's official evaluator ships inside data/Spider2.)
set -euo pipefail

REPO_URL="https://github.com/taoyds/test-suite-sql-eval.git"
PINNED_COMMIT="e97acc546ecbee8fa27fa8dbf025ef61493a876c"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/third_party/test-suite-sql-eval"

if [ ! -d "$TARGET/.git" ]; then
  git clone --quiet "$REPO_URL" "$TARGET"
fi
git -C "$TARGET" fetch --quiet origin
git -C "$TARGET" checkout --quiet "$PINNED_COMMIT"
echo "test-suite-sql-eval at $(git -C "$TARGET" rev-parse --short HEAD) -> $TARGET"
