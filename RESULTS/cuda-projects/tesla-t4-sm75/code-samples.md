# NVIDIA-developer-blog/code-samples on tesla-t4-sm75

Commit `3350d216083a902ccbf5b31665e3b82096a75b55`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (1): posts/nvtx/manual_nvtx.cu (NVTX v2 header was removed from recent toolkits)

## reference (6)

ran clean, two runs agree: its stdout SHA-256 is a usable reference

- `c++11_cuda/c++11_cuda.cu` `502b09af79514ee4`
- `posts/mixed-precision/haxpy.cu` `bcdac8668dfadd5c`
- `posts/unified-memory/dataElem.cu` `9a9cc7b75f0b1f76`
- `posts/unified-memory/dataElem_um.cu` `9a9cc7b75f0b1f76`
- `posts/unified-memory/dataElem_um_c++_1.cu` `9a9cc7b75f0b1f76`
- `posts/unified-memory/dataElem_um_c++_2.cu` `f6560c0aac836bc6`

## silent (4)

ran clean, prints nothing: only the exit status is checkable, there is no output to compare

- `posts/cuda-vmm/sync_main.cu`
- `posts/nvtx/compiler_inst_nvtx.cu`
- `series/cuda-cpp/optimize-data-transfers/profile.cu`
- `series/cuda-cpp/shared-memory/shared-memory.cu`

## pass-only (14)

ran clean but output varies between runs (timings): judge by exit status only

- `posts/american-options/longstaff_schwartz_svd_2.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- written for CUDA <= 8: __any(pred) was removed for sm_70+ (ptxas: 'vote without .sync is not supported'). The copy uses __any_sync(0xffffffff, pred), the full-warp form that call site already assumes.]
- `posts/gups/gups.cu`
- `posts/math-libraries-intro/cublas-example.cu`
- `posts/nvml/nvml_test.cu` [linked with posts/nvml/matrixMul.cu]
- `posts/parallel_reduction_with_shfl/main.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- the vendored 2016 NVlabs/cub submodule collides with the CCCL cub that ships in CUDA 13 (#error CUB_NS_QUALIFIER); the copy includes the toolkit's <cub/cub.cuh> instead, and replaces cub::Sum() (removed in CCCL 3) with cuda::std::plus<int>() and the init value 0 that CCCL 3's Reduce now requires. The header warp_reduce.h calls __shfl_down() without a mask (removed for sm_70+); the build force-includes a one-function shim giving it the full-warp __shfl_down_sync form.]
- `posts/printf-alternative/main.cu` [failure words in the output are expected: demonstrates reporting device-side errors from the kernel on purpose (large value, negative sqrt); the 'ERROR' lines are its output]
- `posts/rnn/LSTM.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- upstream bug: `return time < 0;` compares the libc function time() with 0, which g++ now rejects. The copy returns totalTime < 0 (always 0), the evident intent.]
- `posts/tensor-cores/simpleTensorCoreGEMM.cu`
- `posts/unified-memory-oversubscription/uvm_oversubs.cu`
- `series/cuda-cpp/coalescing-global/coalescing.cu`
- `series/cuda-cpp/finite-difference/finite-difference.cu`
- `series/cuda-cpp/optimize-data-transfers/bandwidthtest.cu`
- `series/cuda-cpp/overlap-data-transfers/async.cu`
- `series/cuda-cpp/transpose/transpose.cu`

## suspect (2)

exit 0 but output contains a failure-like word: read before trusting

- `posts/cufft-callbacks/cufft_callbacks.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- CUFFT_LICENSE_ERROR no longer exists in cuFFT (the license check was dropped); the copy removes that one branch and keeps the status check.] -- `*** FAILED ***` (raw: `tesla-t4-sm75/results/code-samples.json`)
- `posts/cufft-callbacks/cufft_no_callbacks.cu` -- `*** FAILED ***` (raw: `tesla-t4-sm75/results/code-samples.json`)
