#!/bin/sh
# Assemble cuda4AS/drop/as-phase0/ from the sources of truth.
#
#     sh cuda4AS/tools/make_drop.sh [nstr] [plane]
#
# The drop is a COPY of probe/ plus a copy of two files out of lmz/ plus
# generated data.  Assembling it by hand is how a drop ends up carrying a stale
# probe, and a stale probe costs a round trip on hardware this box does not
# have.  So it is assembled here, every time, from scratch.
#
# `run.sh` and `README.md` inside the drop are written by hand and are NOT
# regenerated; this script leaves them alone.
#
# Rule 4: `lmz/` is not ours.  Its two files are copied OUT, unmodified, and
# PROVENANCE.md records the commit they came from.  Nothing is written into it.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
CUDA4AS=$(dirname "$HERE")
YFCE=$(dirname "$CUDA4AS")
DROP="$CUDA4AS/drop/as-phase0"
LMZSRC="$YFCE/lmz/scratchpad/gpu/metal"

NSTR=${1:-512}
PLANE=${2:-32768}

mkdir -p "$DROP/probes" "$DROP/lmz" "$DROP/data"

echo "probes:"
rm -f "$DROP"/probes/*.swift "$DROP"/probes/*.metal
cp "$CUDA4AS"/probe/common.swift "$CUDA4AS"/probe/p*.swift \
   "$CUDA4AS"/probe/p6_matmul_*.metal "$DROP/probes/"
ls "$DROP/probes" | sed 's/^/  /'

echo "lmz (not ours -- copied out, unmodified):"
cp "$LMZSRC/bench.swift" "$LMZSRC/lmz_rans.metal" "$DROP/lmz/"
COMMIT=$(git -C "$YFCE/lmz" rev-parse HEAD)
SUBJ=$(git -C "$YFCE/lmz" log -1 --format=%s)
CDATE=$(git -C "$YFCE/lmz" log -1 --format=%cI)
DIRTY=$(git -C "$YFCE/lmz" status --porcelain | wc -l | tr -d ' ')
if [ "$DIRTY" != "0" ]; then
    echo "  WARNING: the lmz working tree is dirty; the copies may not match commit $COMMIT" >&2
fi
echo "  bench.swift lmz_rans.metal @ $COMMIT"

{
    echo "# Provenance — these two files are not ours"
    echo
    echo '`bench.swift` and `lmz_rans.metal` are copied **unmodified** from the `lmz`'
    echo 'project, which is developed in parallel by someone else and moves on its own'
    echo 'schedule (YFCE `CLAUDE.md` rule 4: read it, run it, copy out of it with'
    echo 'provenance noted; never edit it, never commit inside it, never stage its'
    echo 'gitlink).'
    echo
    echo '| | |'
    echo '|---|---|'
    echo '| source path | `lmz/scratchpad/gpu/metal/bench.swift` |'
    echo '| | `lmz/scratchpad/gpu/metal/lmz_rans.metal` |'
    echo "| lmz commit | \`$COMMIT\` |"
    echo "| commit subject | $SUBJ |"
    echo "| commit date | $CDATE |"
    echo "| copied on | $(date -u '+%Y-%m-%d') |"
    echo '| modified | **no** — byte-for-byte as committed |'
    echo
    echo 'Verify with, from an lmz checkout at that commit:'
    echo
    echo '```sh'
    echo 'shasum -a 256 scratchpad/gpu/metal/bench.swift scratchpad/gpu/metal/lmz_rans.metal'
    echo '```'
    echo
    echo 'Expected:'
    echo
    echo '```'
    ( cd "$DROP/lmz" && sha256sum bench.swift lmz_rans.metal | sed 's/  / /' )
    echo '```'
    echo
    cat "$HERE/provenance_body.md"
} > "$DROP/lmz/PROVENANCE.md"

echo "data:"
python3 "$HERE/prep_synth_shared.py" "$DROP/data" "$NSTR" "$PLANE" | sed 's/^/  /'

echo
echo "self-contained check (no path outside the drop):"
if grep -rn "/home/rog\|/mnt/\|\.\./\.\./" "$DROP" 2>/dev/null | grep -v '^Binary'; then
    echo "  FAILED -- the references above point outside the drop" >&2
    exit 1
fi
echo "  ok"
echo
printf "drop size: %s bytes (%.1f MB)\n" \
    "$(du -sb "$DROP" | cut -f1)" "$(echo "$(du -sb "$DROP" | cut -f1)/1000000" | bc -l)"
echo "the user runs:  cd as-phase0 && ./run.sh"
