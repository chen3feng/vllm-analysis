#!/bin/bash
# Update vllm submodule and notify to check docs
set -e

COMMIT="${1}"
if [ -z "$COMMIT" ]; then
    echo "Usage: $0 <commit-hash-or-tag>"
    echo "Example: $0 v0.11.0"
    exit 1
fi

echo "Updating vllm submodule to $COMMIT..."

cd "$(dirname "$0")/.."
cd vllm
git fetch origin
git checkout "$COMMIT"
cd ..

echo ""
echo "vllm submodule updated to:"
git -C vllm log -1 --oneline
echo ""
echo "Remember to:"
echo "  1. Diff the changed files: git -C vllm diff <old-commit>..<new-commit>"
echo "  2. Check and update affected docs in docs/"
echo "  3. Commit both the submodule pointer and doc updates"
