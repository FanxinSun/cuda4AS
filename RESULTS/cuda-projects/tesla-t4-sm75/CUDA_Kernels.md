# ridash2005/CUDA_Kernels on tesla-t4-sm75

Commit `95b69c7636e3b6df4e6864f14344cf99b839b0ca`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (1): modules/07_ecosystem/06_nvtx/src/profiling.cu (NVTX v2 header was removed from recent toolkits)

## reference (8)

ran clean, two runs agree: its stdout SHA-256 is a usable reference

- `modules/01_fundamentals/01_kernel_basics/src/kernel.cu` `0507a4036d209b43`
- `modules/01_fundamentals/03_hardware/src/props.cu` `1e91eac63bd804a5`
- `modules/02_memory_management/03_constant_memory/src/constant_vec_dot.cu` `d21a12c86d873598`
- `modules/02_memory_management/04_unified_memory/src/unified_mem.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- written for CUDA <= 12 -- not a bug in the program. In CUDA 13 cudaMemPrefetchAsync(ptr, bytes, device) takes a cudaMemLocation and a flags argument instead of a device id (the same break as in cuda-training-series hw6). The copy uses the CUDA 13 form at all three call sites.] `84cb010a866de456`
- `modules/03_advanced_math/02_fft/src/main.cu` [linked with modules/03_advanced_math/02_fft/src/fft_radix2.cu, modules/03_advanced_math/02_fft/src/fft_stockham.cu] `15153bb61aa2637c`
- `modules/07_ecosystem/04_cusparse/src/sparse.cu` `c2c143c3317cc9f8`
- `modules/07_ecosystem/05_cusolver/src/cholesky.cu` `4edb0cec12b81439`
- `modules/07_ecosystem/07_dynamic_parallelism/src/cdp.cu` `abe5fd91eb57f639`

## pass-only (11)

ran clean but output varies between runs (timings): judge by exit status only

- `modules/01_fundamentals/02_grid_block/src/checkDimension.cu`
- `modules/02_memory_management/01_vector_ops/src/main.cu` [linked with modules/02_memory_management/01_vector_ops/src/vector_ops.cu]
- `modules/02_memory_management/02_vector_dot/src/vecDot_non_reduction.cu`
- `modules/03_advanced_math/01_matrix_vector_ops/src/main.cu` [linked with modules/03_advanced_math/01_matrix_vector_ops/src/mv_ops.cu]
- `modules/04_optimizations/01_tiled_matmul/src/tiled_matmul.cu`
- `modules/04_optimizations/02_reduction/src/reduction.cu`
- `modules/05_concurrency/01_streams/src/streams.cu`
- `modules/06_advanced_algorithms/01_histogram/src/histogram.cu`
- `modules/07_ecosystem/01_cublas/src/gemm.cu`
- `modules/07_ecosystem/02_thrust/src/thrust_demo.cu`
- `modules/07_ecosystem/03_curand/src/monte_carlo.cu`
