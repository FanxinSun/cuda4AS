# M1 carry-forward: shape-aware `grid_sync` contract

M1 preserves the accepted M0 contract without reinterpreting or executing it.
This provision does not authorize M3 synchronization implementation.

## Historical enrolled oracle

- Case: `oracle.grid_sync`.
- Source: `oracle/src/grid_sync.cu`, SHA-256
  `2de90bb06407a445f6a9a72bcedc963b9beb81a01cb0b635f6fce6300d38f7c2`.
- Reference metadata: `oracle/ref/grid_sync.json`.
- Reference data: `oracle/ref/grid_sync.bin`, 2,016 bytes, SHA-256
  `4fbd942528ebeb6cb546cbbbd2177efc399f9ec609356d727d864368cb66b089`.
- Type and shape: `u32[504]`, exact comparison.
- Producer observation: 504 blocks came from 6 resident blocks per SM over 84
  SMs on the recorded NVIDIA RTX 5080 correctness-oracle run. The producer
  command used CUDA 13.2, `sm_120`, relocatable device code, and `cudadevrt`.
- Current cuda4AS classification: `NOT_RUN`.

The value 504 is an enrolled reference shape. It is not an Apple target
constant, portable residency claim, scheduler guarantee, or permission to
invent Apple device properties. A qualifying execution may compare with this
golden only when it deliberately runs the same 504-block fixture contract.

## Different device-derived shapes

A legitimate device-derived shape other than 504 requires one of these before
execution can produce a compatibility result:

1. a separately enrolled exact expectation that records the derivation,
   source hash, dimensions, data type, byte count, expected bytes/hash, and
   applicable machine conditions; or
2. a successor fixed-shape fixture whose expected result is declared and
   reviewed before execution.

The historical golden and metadata remain immutable. A runner must never
rewrite, resize, truncate, extend, or regenerate them from its observed target
shape.

## Future semantic and diagnostic gates

`semantic.grid_sync_eligible` still requires a specified correctness-preserving
mechanism, with kernel splitting and live-state transfer recorded when used,
and adversarial occupancy, side-effect, loop, and divergence coverage.
`diagnostic.grid_sync_ineligible` still requires source-located rejection and
the failed eligibility condition when no implemented mechanism is safe.

Measured co-residency or bounded global-atomic behavior cannot establish
general forward progress. A process exit code, completed probe, or output file
does not turn either proposed M0 case into a pass. These cases remain `NOT_RUN`
through M1 unless separately enrolled and executed under a later authorized
scope.
