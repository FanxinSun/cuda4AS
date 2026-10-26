# gpu-mode/lectures on tesla-t4-sm75

Commit `77a8df418834e5789c12da23e7d2719e0efabef1`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (0): none

## reference (9)

ran clean, two runs agree: its stdout SHA-256 is a usable reference

- `lecture_002/vector_addition/vector_addition.cu` `6e88ffe149125616`
- `lecture_008/occupancy.cu` `fc61c9eaa83adc30`
- `lecture_009/control_divergence_reduce.cu` `40f2572bbfa9dffe`
- `lecture_009/multistream-reduce.cu` `1f4c60448cd5fe14`
- `lecture_009/reduce_coarsening.cu` `ac9bd54bd175f209`
- `lecture_009/segment_reduce.cu` `a9f78e1232673f23`
- `lecture_009/shared_reduce.cu` `40f2572bbfa9dffe`
- `lecture_009/simple_reduce.cu` `40f2572bbfa9dffe`
- `lecture_012/main.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- main.cu allocates head dim 64 and launches 32x16 threads for both kernels, but the kernels are compiled for head dim 128 with their own tiles (16x16 / 8x32) and blocks (16x16 / 128x8): out-of-bounds reads, 'illegal memory access' on any NVIDIA GPU. The copy uses d=128 and gives each kernel its own tile count, block and grid.] [linked with lecture_012/flash_attention.cu, lecture_012/flash_attention_spilling_from_registers.cu] `170d396a6677069f`

## silent (3)

ran clean, prints nothing: only the exit status is checkable, there is no output to compare

- `lecture_008/coalesce.cu`
- `lecture_008/privatization.cu`
- `lecture_008/privatization2.cu`

## pass-only (2)

ran clean but output varies between runs (timings): judge by exit status only

- `lecture_008/coarsening.cu`
- `lecture_008/divergence.cu`
