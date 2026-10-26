# S1c brief — two small follow-ups, for the cuda4AS executor

Written 2026-10-26 by the Committee. First read "Committee review of S1b" at the bottom of
`samples/caloml/PLAN.md`. Same rules as before: no commits or pushes, every number must be in
the JSON, and installs go into `~/system-env` (none are expected).

1. **P9, CUB's own tile pattern.** Add a case where `tile = threadgroup_position_in_grid.x`, with no
   ticket at all. This is what CUB does (`cub/agent/agent_scan.cuh:424`).
   - Sweep 16 to 4,096 tiles, at threadgroup sizes 128, 256 and 512, as before.
   - Report timeouts, max spins, and how often dispatch arrival order differed from
     `threadgroup_position_in_grid` order: take an arrival ticket alongside and compare.
   - Then run the same sweep with the arrival-ticket remap (`blockIdx := ticket`) as the control.
2. **MSL language version.** Record the MSL language version that the `memory_order_release` /
   `memory_order_acquire` refusal came from. Retry at the newest version this SDK offers, and record
   each refusal verbatim.
3. **`atomicAdd(double*)` reference.** Add a host *sequential* binary64 sum, in GPU arrival order if
   the kernel can log it, otherwise in index order. Report the GPU result's ulp distance from it,
   next to the Kahan figure.

Hand back with a short "S1c results" section in PLAN.md.
