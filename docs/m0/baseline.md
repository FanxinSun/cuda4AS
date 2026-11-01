# M0 baseline and evidence inventory

This record captures the M0 evidence snapshot taken on 2026-09-06
(Asia/Shanghai). M0 validates preserved artifacts and defines what they can
support; it does not rerun the NVIDIA oracle, regenerate goldens, compile a new
toolchain, or run new Apple-GPU work.

## Evidence vocabulary

- **Observed locally** means M0 read or computed the fact from files or tools on
  this WSL host.
- **Returned observation** means an archived Mac result reports the fact. M0 did
  not reproduce it.
- **Upstream-reported** means another project or vendor reports the fact. It is
  not a cuda4AS result.
- **Proposed** means a future mechanism or acceptance rule. It has not passed.
- **Unresolved** means the available evidence cannot settle the claim.

## Repository seal and working state

| Item | M0 observation | Result |
| --- | --- | --- |
| Repository | `/home/rog/business/YFCE/cuda4AS`; origin `https://github.com/FanxinSun/cuda4AS.git` | observed locally |
| Sealed commit | `ad30bbd16554da5cea55e13f2c7459ed64b893d5` | local `HEAD`, remote `master`, and peeled tag all match |
| Annotated tag | `pre-codex-takeover-2026-09-06`; tag object `9342af7e0afa8d8c73cf771bb94d2ffe2f76fd43` | unchanged locally and on origin |
| Initial branch | `master`, ahead/behind origin `+0/-0` | clean except the manager change below |
| M0 branch | `codex/m0-evidence-contract`, created directly from the sealed commit | active; no commit or push made |
| Preserved manager edit | `.gitignore` adds `/planning-codex/` and its explanatory comment | retained unstaged; not authored by M0 |
| Ignored plan | `planning-codex/cuda4AS-project-plan-codex.md` and `.pdf` | both present and ignored |

The tree had 154 tracked files before the M0 additions. The local oracle
binaries, oracle executables, decoder data, distribution archive, and planning
copies are intentionally ignored. Historical files under `RESULTS/`, the
external blueprint at `/home/rog/business/YFCE/docs/cuda-on-apple-silicon.md`,
and the prior report at
`/home/rog/.claude/handover/2026-09-03-cuda-as-phase0.REPORT.md` were treated as
immutable evidence.

## Repository evidence map

| Path | Role and provenance | M0 validation |
| --- | --- | --- |
| [`probe/`](../../probe/) | Swift/MSL source of truth: 13 shipped source files plus `SELF_REVIEW.md` | all 13 shipped files exactly match `drop/as-phase0/probes/` |
| [`drop/as-phase0/`](../../drop/as-phase0/) | generated Mac probe drop, runner, copied lmz files, and ignored decoder inputs | 21-file distribution archive round-trip matches this tree; `run.sh` retains mode `0755` |
| [`oracle/src/`](../../oracle/src/) | nine CUDA sources and shared harness | sources are tracked at the sealed commit; source hashes recorded below |
| [`oracle/ref/`](../../oracle/ref/) | tracked metadata plus nine ignored exact-output binaries | all nine binaries match both `INDEX.json` and per-kernel metadata |
| [`oracle/log/`](../../oracle/log/) | preserved NVIDIA build/run logs | present for all nine kernels; not replayed in M0 |
| [`RESULTS/as-phase0/Studio/`](../../RESULTS/as-phase0/Studio/) | two returned runs from one M1 Pro configuration | all 20 probe JSON files parse as schema 1; limitations below |
| [`tools/`](../../tools/) | drop assembly and decoder-data generation sources | inspected only; generation was not rerun |
| [`dist/as-phase0.tgz`](../../dist/as-phase0.tgz) | ignored release drop | SHA-256 `7f0dd094b96e6ba9edddd21d6b9331f0673566a7a2c21fe1bbc5ca2cfd6a7c79` matches its sidecar; all 21 archived files match the drop |

The earlier report calls the copied probe set “14 probe files.” There are 14
files in `probe/` only when `SELF_REVIEW.md` is counted; 13 source files are
actually copied into `drop/as-phase0/probes/`.

## NVIDIA oracle validation

The recorded producer was `DESKTOP-V6NN9KI`, Linux
`6.18.33.2-microsoft-standard-WSL2`, NVIDIA GeForce RTX 5080, driver 610.88,
16,303 MiB, compute capability 12.0, with
`/usr/local/cuda-13/bin/nvcc` reporting
`Build cuda_13.2.r13.2/compiler.37953736_0`; target `sm_120`; collected
2026-09-03T07:21:13Z. That machine is a correctness oracle only. M0 observed
that the same CUDA 13.2 compiler remains installed, while the current driver is
616.64; this later host state does not rewrite the recorded producer block.

Expected values come from `oracle/ref/INDEX.json` and the matching per-kernel
JSON. Actual values were recomputed from the preserved `.bin` files. The
per-kernel metadata also agrees with the index on kernel, bytes, hash, comparison
rule, architecture, and producer machine.

| Kernel | Source SHA-256 | Expected bytes | Actual bytes | Expected SHA-256 | Actual SHA-256 | Result |
| --- | --- | ---: | ---: | --- | --- | --- |
| `vector_add` | `b3f205cabc42a697276244d5810ce62ff40a5c9cc6a8e1078420ccbefa88d0e2` | 4,194,304 | 4,194,304 | `ed551637cf393112d0093037a0b41b9d1e9bd213c6037e8efcde30cf480f0332` | `ed551637cf393112d0093037a0b41b9d1e9bd213c6037e8efcde30cf480f0332` | MATCH |
| `saxpy` | `03ccab042fdfb416764ff0e9b6c0c60eb9964f17f9cf2380b6adb0d96522d54a` | 4,194,304 | 4,194,304 | `70949ff55979f4a3eaebb6d904fa2115d90328cf4f2d7040460159057f2462ca` | `70949ff55979f4a3eaebb6d904fa2115d90328cf4f2d7040460159057f2462ca` | MATCH |
| `reduce_sum` | `34eec8bcf34886b49d4c161de89b6e333013a962510cf7515ec243ffb756dfbd` | 16,384 | 16,384 | `1e9a9b2b2a225e70b13ed112181d70b6d6be1cad00577bd0986907fffa3e12b0` | `1e9a9b2b2a225e70b13ed112181d70b6d6be1cad00577bd0986907fffa3e12b0` | MATCH |
| `matmul_tiled` | `dad87624d6e25cba676bf6a8d92d3e9bc5745662d269d8d2627d9f8e15b4f3ac` | 1,048,576 | 1,048,576 | `97f09e437a4f24d2cd22fd5eece3374cc68365fa537c43ee01e859d34472b102` | `97f09e437a4f24d2cd22fd5eece3374cc68365fa537c43ee01e859d34472b102` | MATCH |
| `histogram_atomics` | `8d6acb49309e4168146b89cfb82bd53f5c6d16f51b5530fee553886fd2f2a2b8` | 1,024 | 1,024 | `1c3febdc0e0a6affa4a2d05ba1f98d0fe1604d3c3cfdf398a84da96f651ead4c` | `1c3febdc0e0a6affa4a2d05ba1f98d0fe1604d3c3cfdf398a84da96f651ead4c` | MATCH |
| `shared_48kb` | `d35c24654f5f883b297c1e7c4d597c0d4bb81c6840252b33fd526fa0062538f9` | 65,536 | 65,536 | `02de6cdeaee2558edd7d428420a539a1e9bc68a12722a49d91a522d0098886dc` | `02de6cdeaee2558edd7d428420a539a1e9bc68a12722a49d91a522d0098886dc` | MATCH |
| `mma_inline_ptx` | `0ea8cdc08e29e53d49d565226c7886aab9d217255e3579b89353db4cf54022dd` | 512 | 512 | `a1ca504830bd3182aa3a4bdad30e51985fb2ab0cb98c205a74f518dae1e75319` | `a1ca504830bd3182aa3a4bdad30e51985fb2ab0cb98c205a74f518dae1e75319` | MATCH |
| `grid_sync` | `2de90bb06407a445f6a9a72bcedc963b9beb81a01cb0b635f6fce6300d38f7c2` | 2,016 | 2,016 | `4fbd942528ebeb6cb546cbbbd2177efc399f9ec609356d727d864368cb66b089` | `4fbd942528ebeb6cb546cbbbd2177efc399f9ec609356d727d864368cb66b089` | MATCH |
| `double_dot` | `107886875a914abc7471aa5bad1a89fc67843593ab12a5796ba0f14f464c0f31` | 32,768 | 32,768 | `10857aa39522c523ccdbc34ff62c041e07c6ae239034003f84bb695942b7891b` | `10857aa39522c523ccdbc34ff62c041e07c6ae239034003f84bb695942b7891b` | MATCH |

All nine recorded producer statuses are `ok` and all comparison rules are
`exact`. This establishes that the local golden bytes are internally
consistent with their metadata. It does not establish cuda4AS compilation or
Apple-GPU execution; every cuda4AS corpus status remains `NOT_RUN` in M0.

## Source/drop correspondence

The SHA-256 below is the same for each source and its generated copy.

| Source-of-truth file | SHA-256 | Drop copy |
| --- | --- | --- |
| `common.swift` | `15f80b8c7296c1ccfa106d583c1b02bc24ef758de69f445977f64c169c030b1c` | MATCH |
| `p0_device.swift` | `fe8caa16c8919d7d40f38f2c9f29d247848b2e4e590b3f3f8e18966c5851261a` | MATCH |
| `p1_heap.swift` | `8d62bafd33a10ff4c872357b75f98c70e5f15f23640bf56cd26602bb262ac34c` | MATCH |
| `p1b_vmremap.swift` | `d143020ba98d6f7fb72de90038772b0f41b0800e2935616b9592cf9e42641967` | MATCH |
| `p1c_machvmremap.swift` | `7c0c1387586a3f9f7a7d2e783a41a6f8f77f94c804c11088b018faaf7ae98922` | MATCH |
| `p2_coresident.swift` | `58fbf6ed12dd9da2ff0d89e5ebee6f3fe38c770b3d23797a2093d86354b78471` | MATCH |
| `p3_lockstep.swift` | `5a439e2626a2c9428973ce67b07d04baf1ccc9e3033b48d98681a8c3fb62716c` | MATCH |
| `p4_bandwidth.swift` | `440ac85003b413081a1d48569174d2b16826db907a74e613e1350312ab17f453` | MATCH |
| `p5_simd.swift` | `05e1f010115c88cd88a19f6991df9344607d383a7c030837423e3bac09b4dbfb` | MATCH |
| `p6_matmul_bf16.metal` | `fba5ebb45304176ac71e6eae14fbe3fa99703e7074c3264014018f2749f0a59b` | MATCH |
| `p6_matmul_fp16.metal` | `0c02ed9004af2e6ceab3a79b4fbc714321d787a5f8dca79b96162961a4d387fb` | MATCH |
| `p6_tensorops.swift` | `3d04543f5ee1696231fda06b96186b93da6182c9691fc1503b8e139182cb456d` | MATCH |
| `p7_fp64_atomics.swift` | `7b545e01554fbc13c04ca0df8412abfa09db50002fedd39b1f95290ba1cae647` | MATCH |

There are no extra files in the generated probe directory.

## Decoder data and copied lmz sources

The copied lmz sources name commit
`34dadef47e0f70166ee27ba2eb733105740e2857`, subject “Record the lmz that
ran, not the one that was asked for,” committed 2026-09-04T21:49:55+08:00.
M0 observed the sibling checkout clean at that same commit and read the named
files directly from the commit object; no sibling file was changed.

| Artifact | Expected bytes | Actual bytes | Expected SHA-256 | Actual SHA-256 | Result |
| --- | ---: | ---: | --- | --- | --- |
| `drop/as-phase0/lmz/bench.swift` | not recorded | 7,139 | `988df856859e27643af5124c99c5274dbd68f8fb884cf665fce2a3e1fb6041e1` | `988df856859e27643af5124c99c5274dbd68f8fb884cf665fce2a3e1fb6041e1` | MATCH commit object |
| `drop/as-phase0/lmz/lmz_rans.metal` | not recorded | 10,884 | `6128cce0e510cd34c8f60d81429c2eb6bce78ebc069dbe841d8b15ecc3d5cdfa` | `6128cce0e510cd34c8f60d81429c2eb6bce78ebc069dbe841d8b15ecc3d5cdfa` | MATCH commit object |
| `drop/as-phase0/data/streams.bin` | 5,848,764 | 5,848,764 | `5684932f59df7b4f062ebbd8288696376578dacd7820ff1bb01bb3d23e0ebae3` | `5684932f59df7b4f062ebbd8288696376578dacd7820ff1bb01bb3d23e0ebae3` | MATCH |
| `drop/as-phase0/data/ref.bin` | 16,777,216 | 16,777,216 | `e04cac37377fab6ad6edf2f7ad47d96161e39b5855f6d0717d84661808e28f3d` | `e04cac37377fab6ad6edf2f7ad47d96161e39b5855f6d0717d84661808e28f3d` | MATCH |

The container parses as 512 streams of 32,768 decoded bytes, with a 516-byte
`R1` shared table, 4,096 total frequency slots over 14 live symbols, 512
monotone nonempty in-bounds entries, and every stream offset 16-byte aligned.
`ref.bin` is exactly `512 * 32,768` bytes. M0 did not decode or regenerate the
streams. The prior report records a 512/512 lmz decoder round-trip during
generation; that remains historical evidence.

The generator docstring names derivation commit
`5e904393fe0a6d3a4f6bbb15ea477c48d3cf005a`, while the later copied decoder
sources name `34dadef...`. The data directory has deterministic hashes and a
fixed seed but no standalone generation manifest recording the exact lmz code,
native-library build, Python version, command, and timestamp used for these
bytes. This is a provenance gap to close before regenerating an accepted asset;
it is not a hash mismatch.

## Returned Mac runs

Both runs report the same physical configuration: MacBookPro18,3, Apple M1
Pro, 14 GPU cores, 16 GiB unified memory, arm64, macOS 14.4 build 23E214,
Xcode 15, Swift 5.9.2, and Metal compiler 32023.101. The exact SDK version is
unknown: `xcrun --show-sdk-version` failed while looking for
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`, and the JSON records
`sdk_version: null`. Claims from these runs therefore carry the OS, Xcode, and
Metal compiler versions, not an inferred SDK version.

| Run | JSON/schema validation | What it establishes | Packaging check |
| --- | --- | --- | --- |
| `run1-20260905-130047` | 10/10 JSON files parse; schema 1; every JSON status says no Metal device | the initial acquisition path measured no GPU data; the status table incorrectly calls all ten probes `ok`; p8 also reports no device | all 36 preserved payload files match returned tar SHA-256 `30935b5f29ab7b50330abc17b74e5534ccd9018c343ef85b97bd4dc1e706f450`; tar additionally contains ten AppleDouble `._p*.json` sidecars |
| `run2-20260905-134706` | 10/10 JSON files parse; schema 1; every JSON top-level status is `ok` | probes p0–p7 (including p1b/p1c) completed after fallback to `MTLCopyAllDevices().first`; p8 still did not run | all 36 preserved payload files match returned tar SHA-256 `369796774f7eca53abaa13cf5cf5dfe3f87d09bf791069c43faa3ad828bc5345`; the same ten AppleDouble sidecars are extra |

The runner has 13 rows: two optional p6 ahead-of-time builds, ten JSON-writing
probe executables, and p8. A top-level probe status of `ok` means the probe
completed its own recording; it is not one of the cuda4AS contract results in
[`compatibility.md`](compatibility.md). For example, p6 is `ok` while both
tensor paths are absent, and p8 has no JSON at all.

Important returned observations from run 2 are:

| Probe | Returned observation | Bounded reading |
| --- | --- | --- |
| p0 | SIMD width 32; 1,024 threads/threadgroup; 32,768 B threadgroup memory; 8 GiB `maxBufferLength`; 10.67 GiB recommended working set; runtime MSL accepts 2.4/3.0/3.1 | one device/OS/compiler configuration |
| p1/p1b/p1c | both 8 GiB single buffers succeed without touching backing pages; `bytesNoCopy` is coherent; CPU/GPU addresses differ; both fixed-address remaps return `KERN_NO_SPACE` | does not measure aggregate committed capacity or rule out other managed-memory designs |
| p2 | measured co-resident counts `[15,37,7]`, `[73,37,7]`, and `[37,7,7]` for 64/256/1,024 threads across minimal/16 KiB/32 KiB threadgroup memory | bounded spin-probe observations, not a scheduler guarantee |
| p3 | divergent cases produced recorded values; 3/256 lanes acquired the bounded lock; ballot/prefix checks matched | the barrier pattern violates the specification participation precondition; it cannot prove decoder safety |
| p4/p5 | 1 GiB best-of-five bandwidth and short calibrated arithmetic rates were recorded | useful local measurements, not sustained rates or target constants |
| p6 | Metal 4 family false in the old runtime view; MSL 4.0/4.1 and `metal_tensor` paths unavailable | cannot distinguish hardware support from old OS/SDK/API availability |
| p7 | `double` and tested `atomic_ulong` operations were rejected; uint and float-add controls ran correctly | native capability result for this compiler/device combination only |
| p8 | unmodified lmz host harness exits after the default device call returns nil | decoder GPU correctness remains `NOT_RUN` |

## Documentation and provenance findings

These historical statements remain preserved but must not control new
compatibility claims:

1. [`README.md`](../../README.md) says G-C1 is blocked only by a clang fetch,
   promotes p2 co-residency into a grid-wide safety mechanism, treats the p3
   divergent barrier as correct, treats the 8 GiB single-buffer limit as the
   effective `cudaMalloc` ceiling, and says every blocker has a known mechanism.
   Each conclusion is broader than the evidence.
2. The prior phase-0 report says the p3 result makes lmz prefetch safe and that
   co-residency can make arbitrary `grid.sync()` safe. The MSL participation
   rule and scheduler-contract gap leave both unresolved.
3. The external blueprint is a historical design proposal. Statements about a
   common CPU/GPU pointer model, automatic CPU execution semantics, universal
   32 KiB limits, Metal 4 availability, sparse-resource absence, and
   lock-backed atomics require explicit conformance and capability checks.
4. Run 1 demonstrates that file existence and exit code are insufficient result
   validation. Future runners must validate schema, execution route, required
   fields, and assertions before reporting success.
5. The returned tarballs preserve their intended payload but include AppleDouble
   sidecars. Future packaging should suppress or explicitly allow and hash such
   metadata.

The corrected boundaries and future tests are recorded in
[`compatibility.md`](compatibility.md), [`corpus.json`](corpus.json), and
[`assumptions.md`](assumptions.md).
