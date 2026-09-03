"""Write a *shared-table* lmz stream set from synthetic data, for the Mac drop.

Why this file exists at all
---------------------------
`lmz/scratchpad/gpu/metal/bench.swift` reads one container shape:

    <u32 nstr><u32 plane><516 B shared table><nstr x (u64 off, u64 len)><payload>

and its three kernels take that 516-byte table as `buffer(3)` and build one
decode LUT per threadgroup from it -- 16 streams share the LUT, so a per-stream
table cannot work with these kernels at all.  lmz has two producers of that
container and *neither* is usable for a drop:

  * `scratchpad/gpu/prep_shared.py` writes the right shape but needs a real
    Qwen2.5 safetensors file and emits ~3 GB, which is neither on this box in
    the place it looks for it nor small enough to carry to a Mac.
  * `scratchpad/gpu/prep_synth.py` needs no checkpoint, but writes the
    *per-chunk* variant -- `<u32 nstr><u32 plane><offsets><payload>`, with no
    shared table -- because its consumer, `scratchpad/gpu/cuda/libbench.cu`,
    reads both shapes behind a `shared` flag.  `bench.swift` reads only one,
    at a hard-coded offset of `8 + 516`, so feeding it prep_synth's output
    would misparse the offset table and decode garbage.

So this script is prep_synth's synthetic distribution poured into
prep_shared's container.  It lives in *our* tree: `lmz/` is not ours to edit
(YFCE `CLAUDE.md` rule 4).  It calls lmz's own encoder through lmz's own
Python API and verifies through lmz's own decoder, so the streams in the drop
are real lmz streams and the Metal kernel has nothing easier to do than the
CUDA kernel had.

Derived from `lmz/scratchpad/gpu/prep_synth.py` (the distribution) and
`lmz/scratchpad/gpu/prep_shared.py` (the container and the round-trip check),
lmz commit 5e90439.

    python3 cuda4AS/tools/prep_synth_shared.py <outdir> [nstr] [plane]

Defaults to 512 x 32768 B = 16.8 MB of plane, ~6 MB coded, which is the size
the drop can carry.  Both files are written to <outdir>: `streams.bin` and
`ref.bin`, the two names `bench.swift` looks for.
"""
import os
import random
import struct
import sys
from math import log2

HERE = os.path.dirname(os.path.abspath(__file__))
YFCE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(YFCE, "lmz"))

from lmz import kernels   # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lmz-as-drop"
NSTR = int(sys.argv[2]) if len(sys.argv) > 2 else 512
PLANE = int(sys.argv[3]) if len(sys.argv) > 3 else 32768

if not kernels.have_rans():
    sys.exit("the native lmz kernel is needed to write the streams")

# A BF16 exponent plane is a few dozen live values with a sharp mode, measured
# at ~2.8 bits a symbol over real weights.  That number is load-bearing rather
# than decoration: how often an rANS state refills is a direct function of the
# entropy, and refill rate is what this kernel's speed is made of.  Coding a
# flatter buffer would measure a different machine.  (prep_synth.py's comment,
# and its constants.)
TARGET_BITS = 2.8
CENTRE = 0x86
SEED = 20260903


def _slots(decay):
    """256 translate slots in proportion to a two-sided geometric.

    Largest remainder, so the slots sum to exactly 256 and the buffer gets the
    distribution that was asked for rather than one random sampling of it.
    """
    w = [decay ** abs(v - CENTRE) for v in range(256)]
    tot = sum(w)
    exact = [256 * x / tot for x in w]
    n = [int(x) for x in exact]
    for v in sorted(range(256), key=lambda i: exact[i] - n[i], reverse=True)[
            :256 - sum(n)]:
        n[v] += 1
    return n


def _bits(n):
    return -sum((c / 256) * log2(c / 256) for c in n if c)


lo, hi = 0.01, 0.999
for _ in range(60):
    mid = (lo + hi) / 2
    if _bits(_slots(mid)) < TARGET_BITS:
        lo = mid
    else:
        hi = mid
slots = _slots((lo + hi) / 2)
table256 = bytes(v for v in range(256) for _ in range(slots[v]))
assert len(table256) == 256, len(table256)
live = sum(1 for c in slots if c)
print(f"synthetic plane: {_bits(slots):.2f} bits/symbol over {live} live values")

# ---- the planes ----------------------------------------------------------
rnd = random.Random(SEED)
planes = bytearray()
for i in range(NSTR):
    planes += rnd.randbytes(PLANE).translate(table256)
planes = bytes(planes)

# ---- one table for the whole set ----------------------------------------
# The counts must cover every stream the table will code: a symbol with zero
# frequency cannot be represented, and rans_encode_shared refuses rather than
# mis-coding it.  Histogramming the concatenation is what guarantees that.
hist = kernels.histogram(planes)
shared_header = kernels.rans_table(hist)
assert shared_header is not None and len(shared_header) == kernels.RANS_HEADER
freqs = [shared_header[4 + 2 * s] | (shared_header[5 + 2 * s] << 8)
         for s in range(256)]
print(f"shared table: {sum(1 for f in freqs if f)} symbols with non-zero "
      f"frequency, sum {sum(freqs)}")

# ---- encode, 16-byte aligned starts as prep_shared.py does ---------------
streams, offs = bytearray(), []
per_stream_total = 0
for i in range(NSTR):
    seg = planes[i * PLANE:(i + 1) * PLANE]
    solo = kernels.rans_encode(seg)
    if solo is None:
        sys.exit(f"the coder declined stream {i}")
    per_stream_total += len(solo)
    coded = kernels.rans_encode_shared(seg, shared_header)
    if coded is None:
        sys.exit(f"shared encode failed on stream {i}")
    offs.append((len(streams), len(coded)))
    streams += coded
    streams += b"\0" * (-len(streams) % 16)     # 16 B aligned starts
    if i % 128 == 0:
        print(f"\r  {i}/{NSTR}", end="", flush=True)

raw = NSTR * PLANE
shared_total = sum(n for _, n in offs) + kernels.RANS_HEADER
print(f"\r{'':22}{'bytes':>14}{'of raw':>10}")
print(f"{'per-stream tables':22}{per_stream_total:14,}{100*per_stream_total/raw:9.2f}%")
print(f"{'one shared table':22}{shared_total:14,}{100*shared_total/raw:9.2f}%")

# ---- round-trip through lmz's own decoder -------------------------------
# Prepending the shared header to a shared-table stream yields an ordinary lmz
# stream, so lmz's untouched decoder is the oracle rather than a second
# implementation of the format.  Every stream, not a sample: 512 is cheap.
bad = 0
for i in range(NSTR):
    o, n = offs[i]
    got = kernels.rans_decode(shared_header + bytes(streams[o:o + n]), PLANE)
    if got != planes[i * PLANE:(i + 1) * PLANE]:
        bad += 1
print(f"round-trip through lmz_rans_decode: "
      f"{'OK' if not bad else f'{bad} FAILURES'} over all {NSTR} streams")
assert not bad, "the drop must not carry streams that do not decode here"

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/streams.bin", "wb") as fh:
    fh.write(struct.pack("<II", NSTR, PLANE))
    fh.write(shared_header)
    for o, n in offs:
        fh.write(struct.pack("<QQ", o, n))
    fh.write(bytes(streams))
with open(f"{OUT}/ref.bin", "wb") as fh:
    fh.write(planes)
sz = os.path.getsize(f"{OUT}/streams.bin") + os.path.getsize(f"{OUT}/ref.bin")
print(f"wrote {OUT}/streams.bin + ref.bin -- {NSTR} x {PLANE} B, "
      f"{sz/1e6:.1f} MB on disk")
