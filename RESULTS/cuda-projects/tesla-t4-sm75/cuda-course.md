# Infatoshi/cuda-course on tesla-t4-sm75

Commit `79681bfd003d403170dd93c0436c6f2a45cb390f`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (5): 05_Writing_your_First_Kernels/03 Profiling/00 nvtx_matmul.cu (NVTX v2 header was removed from recent toolkits); 06_CUDA_APIs/02 CUDNN/00 Tanh.cu (needs cuDNN); 06_CUDA_APIs/02 CUDNN/01 Conv2d_NCHW.cu (needs cuDNN); 06_CUDA_APIs/02 CUDNN/02 compare_conv.cu (needs cuDNN); 06_CUDA_APIs/optional/CUTLASS/compare.cu (needs the CUTLASS include path)

## reference (8)

ran clean, two runs agree: its stdout SHA-256 is a usable reference

- `03_C_and_C++_Review/06 Makefiles/03.cu` `962dbd7362c34a20`
- `05_Writing_your_First_Kernels/04 Atomics/00_atomicAdd.cu` `6a2f5178612c3c5f`
- `05_Writing_your_First_Kernels/05 Streams/01_stream_basics.cu` `19b43e0a8c307c64`
- `05_Writing_your_First_Kernels/05 Streams/02_stream_advanced.cu` `62831b6f1eabb441`
- `06_CUDA_APIs/01 CUBLAS/01 cuBLAS/01_Hgemm_Sgemm.cu` `b7b99b7815a0451d`
- `06_CUDA_APIs/01 CUBLAS/02 cuBLASLt/01_LtMatmul.cu` `10bc59eccb0975d0`
- `06_CUDA_APIs/01 CUBLAS/03 cuBLASXt/01_demo.cu` `cecacd2b6ec2bf30`
- `08_Triton/02_softmax.cu` `de766c88d4863937`

## silent (2)

ran clean, prints nothing: only the exit status is checkable, there is no output to compare

- `05_Writing_your_First_Kernels/03 Profiling/01_naive_matmul.cu`
- `05_Writing_your_First_Kernels/03 Profiling/02_tiled_matmul.cu`

## pass-only (7)

ran clean but output varies between runs (timings): judge by exit status only

- `05_Writing_your_First_Kernels/01 CUDA Basics/01_idxing.cu`
- `05_Writing_your_First_Kernels/02 Kernels/00_vector_add_v1.cu`
- `05_Writing_your_First_Kernels/02 Kernels/01_vector_add_v2.cu`
- `05_Writing_your_First_Kernels/02 Kernels/02 matmul.cu`
- `06_CUDA_APIs/01 CUBLAS/02 cuBLASLt/02_compare.cu`
- `06_CUDA_APIs/01 CUBLAS/03 cuBLASXt/02_compare.cu` [rescued: restarted with a longer budget; attempts 60s -> rc 124, 180s -> rc 0]
- `07_Faster_Matmul/unrolling_example.cu`
