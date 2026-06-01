#!/usr/bin/env bash
#
# batch-commit.sh
# Stage, commit, and push uncommitted files in batches of 100.
# Works across a NESTED directory structure, regardless of which
# subdirectory you run it from. Handles filenames with spaces/special chars.

set -euo pipefail

# ---- Config (override via env or flags) ------------------------------------
BATCH_SIZE="${BATCH_SIZE:-50}"
REMOTE="${REMOTE:-origin}"
DRY_RUN="${DRY_RUN:-0}"          # set DRY_RUN=1 to preview without changing anything
# ----------------------------------------------------------------------------

# Make sure we're inside a git repo, then move to its ROOT.
# This is essential for nested dirs: git status paths are repo-root-relative,
# so `git add` must run from the root or pathspecs won't match.
repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Error: not inside a git repository." >&2
  exit 1
}
cd "$repo_root"

BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

echo "Repo root: $repo_root"
echo "Branch: $BRANCH | Remote: $REMOTE | Batch size: $BATCH_SIZE | Dry run: $DRY_RUN"

# Collect changed files, NUL-delimited for safety.
#   --untracked-files=all  -> list every nested file individually instead of
#                             collapsing an untracked dir into a single entry
#                             (critical so batches of 100 are accurate).
mapfile -d '' -t files < <(
  git status --porcelain -z --untracked-files=all | while IFS= read -r -d '' entry; do
    path="${entry:3}"   # entry is "XY <path>"; path starts at index 3
    # Renames/copies emit the old name as a separate NUL field; read & discard it.
    if [[ "${entry:0:2}" == R* || "${entry:0:2}" == C* ]]; then
      IFS= read -r -d '' _oldname || true
    fi
    printf '%s\0' "$path"
  done
)

total="${#files[@]}"
if (( total == 0 )); then
  echo "Nothing to commit. Working tree is clean."
  exit 0
fi

echo "Found $total changed file(s) across the tree to process."

batch_num=0
i=0
while (( i < total )); do
  batch_num=$(( batch_num + 1 ))
  end=$(( i + BATCH_SIZE ))
  (( end > total )) && end=$total

  batch=( "${files[@]:i:BATCH_SIZE}" )
  count="${#batch[@]}"

  echo "----------------------------------------------------------------"
  echo "Batch $batch_num: staging files $((i + 1))-$end of $total ($count files)"

  if (( DRY_RUN == 1 )); then
    printf '  would add: %s\n' "${batch[@]}"
  else
    # Paths are repo-root-relative and we're at the root, so they match.
    # -- guards against filenames starting with a dash.
    git add -- "${batch[@]}"
    git commit -q -m "Batch $batch_num: add $count files ($((i + 1))-$end of $total)"
    echo "Pushing batch $batch_num..."
    git push "$REMOTE" "$BRANCH"
    echo "Batch $batch_num committed and pushed."
  fi

  i=$end
done

echo "================================================================"
echo "Done. Processed $total file(s) in $batch_num batch(es)."
