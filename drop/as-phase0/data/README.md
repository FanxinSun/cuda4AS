# data — 22.6 MB of lmz streams, deliberately not in git

`p8_lmz_decoder` needs two files here:

    streams.bin   5,848,764 B   512 shared-table lmz rANS streams
    ref.bin      16,777,216 B   what they must decode to, byte for byte

Neither is kept in the repository. 22.6 MB of generated binary inside a git
repository has to be transferred on **every** clone, in one pack, with no way
to resume — and this project's third standing rule is that the network may be
metered. A clone that dies at 80 % starts again from zero. A release asset is
served from a CDN and resumes.

So the data ships as a release asset, and the repository stays small enough to
clone over anything.

## Getting them

```sh
gh release download as-phase0 --repo FanxinSun/cuda4AS --pattern 'as-phase0.tgz*'
shasum -a 256 -c as-phase0.tgz.sha256
tar xzf as-phase0.tgz --strip-components=1 -C .. as-phase0/data
```

run from this directory. Or simply unpack the whole tarball somewhere and run
the drop from there — it is the same directory, complete.

`run.sh` does not need them for anything else: **p0 through p7 all run without
this data**, and if it is missing p8 says so and the run continues. That is the
same rule as everywhere else in this drop — an absent thing is reported, never
fatal.

## Where they come from

`cuda4AS/tools/prep_synth_shared.py`, which needs `lmz` and so can only run on
the development box. It generates a synthetic BF16-exponent plane at 2.78
bits/symbol from a fixed seed, codes every stream with **lmz's own encoder**
against one shared 516-byte table, and verifies all 512 through **lmz's own
decoder** before writing. Regenerating reproduces both files byte for byte:

    streams.bin  5684932f59df7b4f062ebbd8288696376578dacd7820ff1bb01bb3d23e0ebae3
    ref.bin      e04cac37377fab6ad6edf2f7ad47d96161e39b5855f6d0717d84661808e28f3d

They cannot be regenerated on the Mac, and not only because lmz is not here:
decoding those streams is precisely what p8 is testing, so a reference the Mac
produced for itself would prove nothing.
