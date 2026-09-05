# ref — the nine reference outputs

The `.json` files are here and tracked. The `.bin` files they describe are
**not**, for the same reason as `drop/as-phase0/data/`: 9.5 MB of binary in git
is paid for on every clone, over a link that may be metered, with no resume.

That costs nothing, because a reference here is fully determined by three
things that *are* tracked:

- the `.cu` source in `../src/`,
- the build command and `-arch`, recorded in each `.json`,
- the `sha256` of the bytes, recorded in each `.json`.

Regenerate them in about fifteen seconds on the development box:

```sh
python3 cuda4AS/oracle/run_oracle.py
```

and every checksum must match what the JSON already says. If one does not,
that is a finding — the toolchain, the card or the source changed — and it is
worth more than a stored copy would have been, because a stored copy would
simply have been overwritten.

A snapshot is attached to the `as-phase0` release as `oracle-ref.tgz`
(4.3 MB) for archival, so the bytes exist somewhere other than this machine:

```sh
gh release download as-phase0 --repo FanxinSun/cuda4AS --pattern 'oracle-ref.tgz*'
```

## What they are for

These are the correctness oracle for the phase-0b translator. When it emits MSL
for one of the nine `.cu` sources and the Apple GPU runs it, the output is
compared against these bytes. Every kernel's inputs were chosen so that
comparison can be **exact** — small multiples of a negative power of two, sized
so every partial sum is exactly representable — because a translated kernel
will not reduce in the same order, and a reference needing a tolerance cannot
tell a rounding difference from a real bug.

They were produced on the RTX 5080 in the development box, which is a
**correctness oracle and nothing else**: no timing was recorded and no
performance target is set from that card. Each `.json` carries the machine
block so a reference made on different hardware is never silently mixed in.
