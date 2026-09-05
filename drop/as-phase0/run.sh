#!/bin/sh
# YFCE cuda4AS -- phase 0a probe drop for Apple silicon.
#
#     ./run.sh
#
# Builds and runs every probe in turn, captures each build's and each run's
# output, writes a status table, and tars the results up for the trip back.
#
# THE DROP MUST NEVER STOP ON A FAILURE.  Nothing in here can be compiled or run
# on the Linux box it was written on, so this run is the first time any of it
# meets a Swift compiler or a Metal driver, and ONE round trip has to bring back
# ALL the errors.  Every step therefore records its outcome and continues:
# a probe that will not build, a probe that crashes, a capability this chip does
# not have -- each is a RESULT, written down and moved past.
#
# Nothing is installed and nothing outside this directory is written.
# POSIX sh on purpose: no bashisms, no GNU coreutils (macOS has neither
# `timeout` nor GNU `date -d`).

set -u

HERE=$(cd "$(dirname "$0")" && pwd)
RESULTS="$HERE/results"
BUILD="$HERE/build"
PROBES="$HERE/probes"
STATUS="$RESULTS/status.txt"

PER_PROBE_TIMEOUT=900        # seconds; p2's sweep is the long one

rm -rf "$RESULTS" "$BUILD"
mkdir -p "$RESULTS" "$BUILD"
: > "$STATUS"

say() { echo "$@"; }
row() { printf '%-22s %-14s %s\n' "$1" "$2" "$3" >> "$STATUS"; }

# Portable per-probe timeout: macOS has no `timeout(1)`.  A watchdog process
# kills the probe if it outlives the budget; the probe's own spin caps mean this
# should never fire, and if it does that is itself worth knowing.
run_bounded() {
    _secs=$1; shift
    _log=$1; shift
    ( "$@" > "$_log" 2>&1 ) &
    _pid=$!
    ( sleep "$_secs"; kill -9 "$_pid" 2>/dev/null ) > /dev/null 2>&1 &
    _watch=$!
    wait "$_pid"
    _rc=$?
    kill -9 "$_watch" 2>/dev/null
    wait "$_watch" 2>/dev/null
    return $_rc
}

# ---------------------------------------------------------------- environment
say "=== environment ==="
{
    echo "date:          $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "hostname:      $(hostname)"
    echo "uname:         $(uname -a)"
    echo "sw_vers:"; sw_vers 2>&1 | sed 's/^/  /'
    echo "arch:          $(uname -m)"
    echo "hw.model:      $(sysctl -n hw.model 2>&1)"
    echo "cpu:           $(sysctl -n machdep.cpu.brand_string 2>&1)"
    echo "memsize:       $(sysctl -n hw.memsize 2>&1)"
    echo "xcode-select:  $(xcode-select -p 2>&1)"
    echo "swiftc:"; xcrun swiftc --version 2>&1 | sed 's/^/  /'
    echo "sdk:           $(xcrun --show-sdk-version 2>&1)"
    echo "metal:"; xcrun -sdk macosx metal --version 2>&1 | sed 's/^/  /'
    echo "gpu-core-count:"; ioreg -rd1 -c AGXAccelerator 2>/dev/null \
        | grep -i 'gpu-core-count' | sed 's/^/  /'
} > "$RESULTS/env.txt" 2>&1
cat "$RESULTS/env.txt"
say ""

if ! xcrun swiftc --version > /dev/null 2>&1; then
    say "swiftc not found.  Run:  xcode-select --install"
    row "PREFLIGHT" "FAILED" "swiftc missing; xcode-select --install"
    exit 1
fi

# ------------------------------------------------------------------- p6 metallib
# The tensor-op probe has TWO compilation routes and reports both (see the probe
# source).  This builds the ahead-of-time one if the Metal toolchain is present.
# If it is absent the probe still runs and reports the runtime-compile result --
# an absent toolchain is a result, not a failure of the drop.
say "=== ahead-of-time Metal shaders for p6 (optional) ==="
for T in fp16 bf16; do
    SRC="$PROBES/p6_matmul_$T.metal"
    OUT="$BUILD/p6_matmul_$T.metallib"
    LOG="$RESULTS/build-p6_metallib_$T.log"
    : > "$LOG"
    BUILT=no
    for STD in metal4.1 metal4.0 macos-metal4.0 metal3.2 ""; do
        if [ -n "$STD" ]; then FLAG="-std=$STD"; else FLAG=""; fi
        echo "--- xcrun -sdk macosx metal $FLAG ---" >> "$LOG"
        if xcrun -sdk macosx metal $FLAG "$SRC" -o "$OUT" >> "$LOG" 2>&1; then
            echo "OK with '$FLAG'" >> "$LOG"
            row "p6.metallib.$T" "built" "xcrun metal $FLAG"
            say "  $T: built with '$FLAG'"
            BUILT=yes
            break
        fi
    done
    if [ "$BUILT" = no ]; then
        row "p6.metallib.$T" "absent" "no -std spelling worked; see build-p6_metallib_$T.log"
        say "  $T: no ahead-of-time metallib (see the log; the probe still runs)"
    fi
done
say ""

# ---------------------------------------------------------------------- probes
PROBE_LIST="p0_device p1_heap p1b_vmremap p1c_machvmremap p2_coresident \
p3_lockstep p4_bandwidth p5_simd p6_tensorops p7_fp64_atomics"

# The probes read these: where to write JSON, and where p6 finds its .metal
# sources and any ahead-of-time metallib run.sh managed to build.
NO_GPU=0

PROBE_RESULTS="$RESULTS"; export PROBE_RESULTS
P6_SRC_DIR="$PROBES";     export P6_SRC_DIR
P6_LIB_DIR="$BUILD";      export P6_LIB_DIR

say "=== probes ==="
for P in $PROBE_LIST; do
    SRC="$PROBES/$P.swift"
    BIN="$BUILD/$P"
    BLOG="$RESULTS/build-$P.log"
    RLOG="$RESULTS/run-$P.log"
    : > "$BLOG"

    if [ ! -f "$SRC" ]; then
        row "$P" "MISSING" "no $P.swift in probes/"
        say "  $P: source missing"
        continue
    fi

    # Route 1: multi-file compilation.  No file is named main.swift, so the
    # @main attribute in the probe supplies the entry point.
    echo "=== route 1: swiftc -O common.swift $P.swift ===" >> "$BLOG"
    if xcrun swiftc -O "$PROBES/common.swift" "$SRC" -o "$BIN" >> "$BLOG" 2>&1; then
        ROUTE="multi-file"
    else
        # Route 2: concatenate into one script-mode file and call probeMain()
        # directly.  A different mechanism on purpose -- if the two Swift
        # versions disagree about entry points, one of these still builds, and
        # this drop does not get a second round trip.
        echo "" >> "$BLOG"
        echo "=== route 2: concatenated script mode ===" >> "$BLOG"
        CAT="$BUILD/${P}_script.swift"
        cat "$PROBES/common.swift" "$SRC" > "$CAT"
        echo "probeMain()" >> "$CAT"
        if xcrun swiftc -O -D PROBE_SCRIPT_MODE "$CAT" -o "$BIN" >> "$BLOG" 2>&1; then
            ROUTE="script-mode"
        else
            ROUTE=""
        fi
    fi

    if [ -z "$ROUTE" ]; then
        row "$P" "BUILD FAILED" "see build-$P.log"
        say "  $P: build failed (both routes) -- the compiler text is the result"
        continue
    fi

    if run_bounded "$PER_PROBE_TIMEOUT" "$RLOG" "$BIN"; then
        if [ -f "$RESULTS/$P.json" ]; then
            # "wrote a JSON" is not "measured something".  Every probe reports
            # no_metal_device when Metal handed out no device, and that is a
            # failed run whatever the exit status said.
            if grep -q '"no_metal_device": true' "$RESULTS/$P.json"; then
                row "$P" "NO GPU" "ran, but Metal gave no device -- measured nothing"
                say "  $P: NO METAL DEVICE -- nothing measured"
                NO_GPU=$((NO_GPU + 1))
            else
                row "$P" "ok" "built $ROUTE; $P.json written"
                say "  $P: ok ($ROUTE)"
            fi
        else
            row "$P" "ran, NO JSON" "built $ROUTE; see run-$P.log"
            say "  $P: ran but wrote no JSON"
        fi
    else
        RC=$?
        if [ -f "$RESULTS/$P.json" ]; then
            row "$P" "PARTIAL rc=$RC" "built $ROUTE; $P.json has what it reached"
            say "  $P: exited rc=$RC after writing partial JSON"
        else
            row "$P" "RUN FAILED rc=$RC" "built $ROUTE; see run-$P.log"
            say "  $P: run failed rc=$RC"
        fi
    fi
done
say ""

# ---------------------------------------------------------------- p8: lmz decode
# lmz's own harness and shader, unmodified (see lmz/PROVENANCE.md).  It compiles
# its Metal at run time, so it needs no Metal toolchain.  It is run from its own
# directory because it looks for lmz_rans.metal next to itself first.
say "=== p8: lmz rANS decoder on the Apple GPU ==="
BLOG="$RESULTS/build-p8_lmz_decoder.log"
RLOG="$RESULTS/run-p8_lmz_decoder.log"
if [ ! -f "$HERE/data/streams.bin" ] || [ ! -f "$HERE/data/ref.bin" ]; then
    # 22.6 MB of generated streams, shipped as a release asset rather than in
    # git so this repository stays cloneable on a metered link.  Absent data is
    # a reported result like any other -- p0 through p7 have already run.
    row "p8_lmz_decoder" "DATA ABSENT" "22.6 MB not in git; see data/README.md"
    say "  data/streams.bin and data/ref.bin are not here.  From this directory:"
    say ""
    say "    gh release download as-phase0 --repo FanxinSun/cuda4AS --pattern 'as-phase0.tgz*'"
    say "    shasum -a 256 -c as-phase0.tgz.sha256"
    say "    tar xzf as-phase0.tgz --strip-components=1 -C . as-phase0/data"
    say ""
    say "  Every probe above ran without them; only p8 needs them."
elif xcrun swiftc -O "$HERE/lmz/bench.swift" -o "$BUILD/lmzmetal" > "$BLOG" 2>&1; then
    ( cd "$HERE/lmz" && run_bounded "$PER_PROBE_TIMEOUT" "$RLOG" \
        "$BUILD/lmzmetal" "$HERE/data" )
    RC=$?
    if [ $RC -eq 0 ] && grep -q 'byte-identical' "$RLOG" 2>/dev/null; then
        row "p8_lmz_decoder" "ok" "byte-identical; see run-p8_lmz_decoder.log"
        say "  byte-identical:"
        grep -E 'plane|fused|device|streams' "$RLOG" | sed 's/^/    /'
    elif [ $RC -eq 0 ]; then
        row "p8_lmz_decoder" "ran, NOT identical" "see run-p8_lmz_decoder.log"
        say "  ran, but did not print byte-identical -- see the log"
        sed -n '1,20p' "$RLOG" | sed 's/^/    /'
    else
        row "p8_lmz_decoder" "RUN FAILED rc=$RC" "see run-p8_lmz_decoder.log"
        say "  run failed rc=$RC"
    fi
else
    row "p8_lmz_decoder" "BUILD FAILED" "see build-p8_lmz_decoder.log"
    say "  build failed -- the compiler text is the result"
fi
say ""

# ------------------------------------------------------------------------ pack
if [ "$NO_GPU" -gt 0 ]; then
    say "########################################################################"
    say "#  $NO_GPU probes got NO METAL DEVICE.  Nothing below is a measurement."
    say "#"
    say "#  MTLCreateSystemDefaultDevice() and MTLCopyAllDevices() both came back"
    say "#  empty.  That is almost never a fact about the GPU -- it is a fact"
    say "#  about the session.  Metal hands out no device to a process with no"
    say "#  window-server (Aqua) session, which means:"
    say "#"
    say "#    * over SSH                    -> run it from the Mac's own screen"
    say "#    * from a launchd/cron job     -> run it from a Terminal"
    say "#    * inside a sandboxed tool     -> run it from a plain Terminal"
    say "#"
    say "#  The session block in each JSON records launchctl managername and the"
    say "#  SSH environment, so the results say which of these it was."
    say "#"
    say "#  Please rerun from Terminal.app or iTerm ON the Mac, logged in at the"
    say "#  screen, and send the new tarball."
    say "########################################################################"
    say ""
fi

say "=== status ==="
printf '%-22s %-14s %s\n' "probe" "outcome" "note"
printf '%s\n' "--------------------------------------------------------------------------"
cat "$STATUS"
say ""

TAR="results-$(hostname -s)-$(date -u '+%Y%m%d-%H%M%S').tgz"
( cd "$HERE" && tar czf "$TAR" results )
say "Send back:  $HERE/$TAR"
say "            ($(cd "$HERE" && wc -c < "$TAR" | tr -d ' ') bytes)"
say ""
say "Nothing was installed and nothing outside this directory was written."
