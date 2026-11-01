#!/bin/bash
# Standalone Git repair and launcher for the CUDA4AS M2A native probe.
# Acquire this tracked file with the one-time Git pull when the checkout cannot
# yet obtain the M2A branch, then run: /bin/bash /actual/path/bootstrap-m2a-native-aot-probe.sh
set -u

TARGET_BRANCH="codex/m2a-native-aot-vector-add"
REMOTE_NAME="origin"
REMOTE_REF="refs/remotes/$REMOTE_NAME/$TARGET_BRANCH"
UPSTREAM_REF="$REMOTE_NAME/$TARGET_BRANCH"
REPO_ROOT="${CUDA4AS_M2A_REPO:-/Users/yaminocellist/git_repos/cuda4AS-m1}"
LOG_ROOT="${CUDA4AS_M2A_LOG_ROOT:-$HOME/cuda4as-m2a-bootstrap-runs}"
PACKAGE_NAME="cuda4as-m2a-native-aot-vector-add-v1.tgz"
PACKAGE_SHA="1184651ff56d76fbea08fce7c68ca857615d1f27fe78934470d218313fbbe1c0"
PACKAGE_BYTES="22535"

fail() {
  code="$1"
  shift
  printf 'ERROR: %s\n' "$*" >&2
  exit "$code"
}

command -v git >/dev/null 2>&1 || fail 2 "git is unavailable"
command -v bash >/dev/null 2>&1 || fail 2 "bash is unavailable"
command -v shasum >/dev/null 2>&1 || fail 2 "shasum is unavailable"
command -v mktemp >/dev/null 2>&1 || fail 2 "mktemp is unavailable"
command -v tee >/dev/null 2>&1 || fail 2 "tee is unavailable"
command -v grep >/dev/null 2>&1 || fail 2 "grep is unavailable"

mkdir -p "$LOG_ROOT" || fail 2 "cannot create log root: $LOG_ROOT"
TASK_ROOT="$(mktemp -d "$LOG_ROOT/run.XXXXXX")" ||
  fail 2 "cannot create a fresh bootstrap task directory"
LOG_FILE="$TASK_ROOT/bootstrap.console.txt"
exec > >(tee "$LOG_FILE") 2>&1

printf 'CUDA4AS M2A Git repair and probe launch\n'
printf 'Repository: %s\n' "$REPO_ROOT"
printf 'Target branch: %s\n' "$TARGET_BRANCH"
printf 'Bootstrap task root: %s\n' "$TASK_ROOT"

if [ "${CUDA4AS_M2A_DRY_RUN:-0}" != "1" ]; then
  OS_NAME="$(uname -s 2>/dev/null || true)"
  CPU_ARCH="$(uname -m 2>/dev/null || true)"
  [ "$OS_NAME" = "Darwin" ] || fail 2 "target OS is $OS_NAME; run this on the Mac"
  [ "$CPU_ARCH" = "arm64" ] || fail 2 "target architecture is $CPU_ARCH; expected arm64"
else
  printf 'DRY_RUN=1: target OS/architecture check bypassed for fixture validation only\n'
fi

[ -d "$REPO_ROOT" ] || fail 2 "repository directory does not exist: $REPO_ROOT"
git -C "$REPO_ROOT" rev-parse --show-toplevel >/dev/null 2>&1 ||
  fail 2 "repository is not a Git checkout: $REPO_ROOT"
git -C "$REPO_ROOT" remote get-url "$REMOTE_NAME" >/dev/null 2>&1 ||
  fail 2 "remote '$REMOTE_NAME' is unavailable"

FETCH_SPEC="refs/heads/$TARGET_BRANCH:refs/remotes/$REMOTE_NAME/$TARGET_BRANCH"
if ! git -C "$REPO_ROOT" config --get-all "remote.$REMOTE_NAME.fetch" |
  grep -Fqx "$FETCH_SPEC"; then
  printf 'Adding the target ref to the local narrow-fetch configuration\n'
  git -C "$REPO_ROOT" config --add "remote.$REMOTE_NAME.fetch" "$FETCH_SPEC"
  CONFIG_EXIT="$?"
  [ "$CONFIG_EXIT" -eq 0 ] ||
    fail 3 "cannot add target ref to local fetch configuration"
fi

printf 'Fetching the target ref explicitly (narrow/single-branch safe)\n'
git -C "$REPO_ROOT" fetch "$REMOTE_NAME" \
  "$FETCH_SPEC"
FETCH_EXIT="$?"
[ "$FETCH_EXIT" -eq 0 ] ||
  fail 3 "target ref fetch failed; local work was not changed"

git -C "$REPO_ROOT" show-ref --verify --quiet "$REMOTE_REF" ||
  fail 3 "fetched remote ref is still unavailable: $REMOTE_REF"
REMOTE_HEAD="$(git -C "$REPO_ROOT" rev-parse "$REMOTE_REF")" ||
  fail 3 "cannot resolve fetched remote head"
printf 'Fetched remote head: %s\n' "$REMOTE_HEAD"

if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  printf 'Existing local target branch found; switching without discarding work\n'
  git -C "$REPO_ROOT" switch "$TARGET_BRANCH"
  SWITCH_EXIT="$?"
  [ "$SWITCH_EXIT" -eq 0 ] ||
    fail 3 "cannot switch safely to existing target branch; local work preserved"
else
  printf 'Creating local target branch from fetched remote head\n'
  git -C "$REPO_ROOT" switch --track -c "$TARGET_BRANCH" "$UPSTREAM_REF"
  SWITCH_EXIT="$?"
  [ "$SWITCH_EXIT" -eq 0 ] ||
    fail 3 "cannot create target branch safely; local work preserved"
fi

git -C "$REPO_ROOT" branch --set-upstream-to="$UPSTREAM_REF" "$TARGET_BRANCH"
UPSTREAM_EXIT="$?"
[ "$UPSTREAM_EXIT" -eq 0 ] ||
  fail 3 "cannot repair target branch upstream; local work preserved"

STATUS_BEFORE="$(git -C "$REPO_ROOT" status --short)"
if [ -n "$STATUS_BEFORE" ]; then
  printf 'Existing worktree changes (preserved):\n%s\n' "$STATUS_BEFORE"
else
  printf 'Worktree is clean before fast-forward check\n'
fi

AHEAD="$(git -C "$REPO_ROOT" rev-list --count "$UPSTREAM_REF..$TARGET_BRANCH")" ||
  fail 3 "cannot measure local branch divergence"
BEHIND="$(git -C "$REPO_ROOT" rev-list --count "$TARGET_BRANCH..$UPSTREAM_REF")" ||
  fail 3 "cannot measure remote branch divergence"
printf 'DIVERGENCE_AHEAD=%s\nDIVERGENCE_BEHIND=%s\n' "$AHEAD" "$BEHIND"

if [ "$AHEAD" -ne 0 ]; then
  fail 3 "local target branch has unpushed commits; refusing reset, merge, or overwrite"
fi

if [ "$BEHIND" -ne 0 ]; then
  printf 'Fast-forwarding the target branch\n'
  git -C "$REPO_ROOT" pull --ff-only
  PULL_EXIT="$?"
  [ "$PULL_EXIT" -eq 0 ] ||
    fail 3 "fast-forward pull failed; local work and commits were preserved"
else
  printf 'Target branch is already at the fetched remote head\n'
fi

LOCAL_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)" ||
  fail 3 "cannot resolve local target head"
REMOTE_HEAD_AFTER="$(git -C "$REPO_ROOT" rev-parse "$UPSTREAM_REF")" ||
  fail 3 "cannot resolve fetched remote head after pull"
[ "$LOCAL_HEAD" = "$REMOTE_HEAD_AFTER" ] ||
  fail 3 "local head does not equal fetched remote head; refusing to launch"
printf 'Verified checkout head: %s\n' "$LOCAL_HEAD"

PROBE="$REPO_ROOT/tools/m2a/run-m2a-native-aot-vector-add-all-in-one.sh"
PACKAGE="$REPO_ROOT/tools/m2a/artifacts/$PACKAGE_NAME"
[ -f "$PROBE" ] || fail 3 "tracked M2A probe is missing after repair: $PROBE"
[ -f "$PACKAGE" ] || fail 3 "tracked M2A package is missing after repair: $PACKAGE"

ACTUAL_BYTES="$(wc -c <"$PACKAGE" | tr -d '[:space:]')"
ACTUAL_SHA="$(shasum -a 256 "$PACKAGE" | awk '{print $1}')"
printf 'Tracked package bytes: %s\nTracked package SHA-256: %s\n' "$ACTUAL_BYTES" "$ACTUAL_SHA"
[ "$ACTUAL_BYTES" = "$PACKAGE_BYTES" ] ||
  fail 3 "tracked package byte count mismatch"
[ "$ACTUAL_SHA" = "$PACKAGE_SHA" ] ||
  fail 3 "tracked package SHA-256 mismatch"

printf 'Verified probe: %s\n' "$PROBE"
printf 'Launching the tracked all-in-one probe\n'
exec /bin/bash "$PROBE"
