# CoffeeBeforeArch/cuda_programming on tesla-t4-sm75

Commit `8711be08e2060680820304594f3691cc0016edd1`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (0): none

## reference (20)

ran clean, two runs agree: its stdout SHA-256 is a usable reference

- `01_vector_addition/baseline/vectorAdd.cu` `9018c16075667c3d`
- `01_vector_addition/pinned_memory/vectorAdd_pinned.cu` `9018c16075667c3d`
- `01_vector_addition/unified_memory/vectorAdd_um_baseline.cu` `4993c78d9956735e`
- `01_vector_addition/unified_memory/vectorAdd_um_prefetch.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- written for CUDA <= 12 -- not a bug in the program. In CUDA 13 both cudaMemPrefetchAsync(ptr, bytes, device) and cudaMemAdvise(ptr, bytes, advice, device) take a cudaMemLocation instead of a device id (the same break as in cuda-training-series hw6). The copy uses the CUDA 13 form at all ten call sites (4 cudaMemAdvise, 6 cudaMemPrefetchAsync).] `4993c78d9956735e`
- `02_matrix_mul/baseline/mmul.cu` `9018c16075667c3d`
- `02_matrix_mul/noAliasing/restrict/mmul.cu` `9018c16075667c3d`
- `02_matrix_mul/noAliasing/tmp/mmul.cu` `9018c16075667c3d`
- `02_matrix_mul/nonMultiple/mmul.cu` `9018c16075667c3d`
- `02_matrix_mul/tiled/mmul.cu` `9018c16075667c3d`
- `03_sum_reduction/bank_conflicts/sumReduction.cu` `9018c16075667c3d`
- `03_sum_reduction/cooperative_groups/sumReduction.cu` `9018c16075667c3d`
- `03_sum_reduction/device_function/sumReduction.cu` `9018c16075667c3d`
- `03_sum_reduction/diverged/sumReduction.cu` `9018c16075667c3d`
- `03_sum_reduction/no_confliicts/sumReduction.cu` `9018c16075667c3d`
- `03_sum_reduction/reduce_idle/sumReduction.cu` `9018c16075667c3d`
- `05_convolution/1d_cache/convolution.cu` `9018c16075667c3d`
- `05_convolution/1d_constant_memory/convolution.cu` `9018c16075667c3d`
- `05_convolution/1d_naive/convolution.cu` `9018c16075667c3d`
- `05_convolution/1d_tiled/convolution.cu` `9018c16075667c3d`
- `05_convolution/2d_constant_memory/convolution.cu` `da8d0b082da4aec6`

## silent (2)

ran clean, prints nothing: only the exit status is checkable, there is no output to compare

- `04_histogram/global_atomic/histogram.cu`
- `04_histogram/shmem_atomic/histogram.cu`

## pass-only (1)

ran clean but output varies between runs (timings): judge by exit status only

- `02_matrix_mul/rectangular/mmul.cu` [rescued: restarted with a longer budget; attempts 60s -> rc 124, 180s -> rc 0]
