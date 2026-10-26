# NVIDIA-developer-blog/code-samples on a100-sm80

Commit `3350d216083a902ccbf5b31665e3b82096a75b55`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (1): posts/nvtx/manual_nvtx.cu (NVTX v2 header was removed from recent toolkits)

## suspect (2)

exit 0 but output contains a failure-like word: read before trusting

- `posts/cufft-callbacks/cufft_callbacks.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- CUFFT_LICENSE_ERROR no longer exists in cuFFT (the license check was dropped); the copy removes that one branch and keeps the status check.] -- `*** FAILED ***` (raw: `a100-sm80/results/code-samples.json`)
- `posts/cufft-callbacks/cufft_no_callbacks.cu` -- `*** FAILED ***` (raw: `a100-sm80/results/code-samples.json`)

## build-fail (1)

nvcc/cmake failed on the GPU host: see first error; often not a translator target

- `posts/parallel_reduction_with_shfl/main.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- the vendored 2016 NVlabs/cub submodule collides with the CCCL cub that ships in CUDA 13 (#error CUB_NS_QUALIFIER); the copy includes the toolkit's <cub/cub.cuh> instead.] [ok on tesla-t4-sm75] -- `4 errors detected in the compilation of "/content/out/build/code-samples/timing1__posts__parallel_reduction_with_shfl__main.cu".`
