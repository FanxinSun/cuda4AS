# yasin-arafat-05/Learning_CUDA_From_Scratch on tesla-t4-sm75

Commit `adc00f0430fa11730ac042741ee3797a8d0e02cd`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (0): none

## not-tested (1)

needs hardware, libraries or a toolkit the bench host does not have (2 GPUs, OpenGL/EGL, a newer CUDA); no verdict about the program

- `Problems/L8-Production Quality Code/Multi-GPU.cu` [same result in 3 further session(s)] -- needs 2 GPUs with peer access; Colab gives one -- NOTE: needs two GPUs with peer access ('Peer access not supported between devices'); Colab has one -- a hardware limit, not a defect

## reference (64)

ran clean, two runs agree: its stdout SHA-256 is a usable reference

- `Day 13/Flash_Attention_V1.cu` `172c1bd3951782bd`
- `Day 14/dotproduct.cu` `a583798abdf10580`
- `Day 14/leetgpuDotProduct.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- free() on the stack arrays h_A/h_B aborts with 'double free or corruption'; removed.] `3b292072d2312c15`
- `Day 16/hadamardProduct.cu` `e164b8c5355762e3`
- `Day 17/GEMM.cu` `9eaa1fcde1ab37ef`
- `Day 18/Relu.cu` `d913dadc3ee9fc43`
- `Day 18/SigmoidAct.cu` `61bfb4318da067a3`
- `Day 2/vector_addition.cu` `12fe735a8986c3da`
- `Day 20/SoftmaxKernel.cu` `b31819d94530ff82`
- `Day 21/SigmoidLinearUnit.cu` `dd5fa593c1095bb5`
- `Day 3/vector_subtraction.cu` `10eb062b114c6248`
- `Problems/L0-Setup and foundations/hello_from_gpu.cu` `e285fc2e32a7f44a`
- `Problems/L0-Setup and foundations/mem_cpy.cu` `936f6629c3b35f8c`
- `Problems/L0-Setup and foundations/thread_index.cu` `35680fba9e783efd`
- `Problems/L1-Map Patterns/SAXPY.cu` `4e5d1c1908906306`
- `Problems/L1-Map Patterns/polynomial_Evaluation.cu` `6c4afb47800943da`
- `Problems/L1-Map Patterns/vector_Subtraction.cu` `1e57e4a16329ed64`
- `Problems/L1-Map Patterns/vector_division.cu` `db636b7ace993574`
- `Problems/L1-Map Patterns/vector_multiply.cu` `0f1b90f42feb590c`
- `Problems/L1-Map Patterns/vector_scaling.cu` `77f573acc8352d07`
- `Problems/L3-Memory Patterns/constat memory test.cu` `343542b0d1095817`
- `Problems/L3-Memory Patterns/matrix transpose (optimized).cu` `bf034f465aff70f1`
- `Problems/L3-Memory Patterns/matrix transpose(optimized with padding).cu` `bf034f465aff70f1`
- `Problems/L3-Memory Patterns/transpose (naive).cu` `f93478defd9d2ae5`
- `Problems/L4-Scan Pattern - Prefix Sums/Partition.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- free() on the stack arrays h_input/h_output aborts with 'double free or corruption'; removed.] `ec93ff35afacb744`
- `Problems/L4-Scan Pattern - Prefix Sums/Run length encoding.cu` `84b8643d8b891359`
- `Problems/L4-Scan Pattern - Prefix Sums/prefix sum (Blelloch).cu` `a530446eeceb10ed`
- `Problems/L4-Scan Pattern - Prefix Sums/prefix sum (Multi Block).cu` `61ce60ebe7a5762f`
- `Problems/L4-Scan Pattern - Prefix Sums/prefix-sum (Hills-Steele).cu` [PATCHED COPY (source changed to build/run; clone untouched) -- each step does `out += in[tid-offset]` into the other buffer, which was never initialised (the known GPU Gems listing bug); it must be `out = in[tid] + in[tid-offset]`. The program prints 'Mismatch' yet exits 0.] `a530446eeceb10ed`
- `Problems/L4-Scan Pattern - Prefix Sums/scan with comparison.cu` `3ebe4298de1d6a17`
- `Problems/L4-Scan Pattern - Prefix Sums/scan_with_comparison_hills_steele.cu` `67e59b53ba40a6a3`
- `Problems/L4-Scan Pattern - Prefix Sums/segmented scan.cu` `9109686b9e835fbf`
- `Problems/L4-Scan Pattern - Prefix Sums/sparse to dense.cu` `f17a5bed9411ba5a`
- `Problems/L4-Scan Pattern - Prefix Sums/stream compaction.cu` `9ab1de948e2364fb`
- `Problems/L5-Stencil Pattern - Neighbout Access/1D Heat Equation.cu` `5caf638290de706a`
- `Problems/L5-Stencil Pattern - Neighbout Access/1D Wave Equation.cu` `3ee999fb2c84f0ad`
- `Problems/L5-Stencil Pattern - Neighbout Access/2D Box Filter.cu` `5ea9dece170e0f16`
- `Problems/L5-Stencil Pattern - Neighbout Access/2D Convolution.cu` `b1acf440ea0bb254`
- `Problems/L5-Stencil Pattern - Neighbout Access/2D Heat Equation.cu` `6efdee00cf8fb300`
- `Problems/L5-Stencil Pattern - Neighbout Access/2D Laplace Equation.cu` `e0c012320b41b2a9`
- `Problems/L5-Stencil Pattern - Neighbout Access/Box Filter (1D).cu` `95ad04ac697549ea`
- `Problems/L5-Stencil Pattern - Neighbout Access/Game of Life.cu` `09b964119c4c89cd`
- `Problems/L5-Stencil Pattern - Neighbout Access/Gaussuan Blur (1D).cu` `54cc4236148f5e4a`
- `Problems/L5-Stencil Pattern - Neighbout Access/sobel edge detection.cu` `049560b9efe576a4`
- `Problems/L6-Advanced Patterns/Atomic Histogram.cu` `cd8b3fe14e75d09c`
- `Problems/L6-Advanced Patterns/Atomic Operations Comparison.cu` `c363f1d301202461`
- `Problems/L6-Advanced Patterns/Connected Components (BFS).cu` `289e28dab72c734c`
- `Problems/L6-Advanced Patterns/Graph Traversal.cu` `ad210831acb5e44f`
- `Problems/L6-Advanced Patterns/LockFreeQueue.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- d_head/d_tail are declared `unsigned int` but used everywhere as device pointers (cudaMalloc(&d_head), atomicAdd(d_head, 1), *d_head); declared as `unsigned int*`.] `c75fe2a7291dc929`
- `Problems/L6-Advanced Patterns/Radix Sort (1 bit).cu` `dcccdacbe174d51b`
- `Problems/L6-Advanced Patterns/Tree Reduction vs Atomic Reduction.cu` `5388dc122259dc2a`
- `Problems/L6-Advanced Patterns/Triangular Solve.cu` `c118e321b3c1e88e`
- `Problems/L6-Advanced Patterns/sparse matrix vector multiply.cu` `e9a034060824c2d6`
- `Problems/L7-Optimization Masterclass/Fast Math Study.cu` `d9f2b541ddad1766`
- `Problems/L7-Optimization Masterclass/Loop Unrolling.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- three 4 MB arrays on the stack (12 MB > the 8 MB limit: the segfault), and <<<1, N/4>>> asks for 262144 threads in one block (limit 1024). Copy: static arrays, 256-thread blocks.] `fa1b5332b2d7f3c4`
- `Problems/L7-Optimization Masterclass/Register Pressure Tuning.cu` `31749f85d957d30f`
- `Problems/L7-Optimization Masterclass/Texture Memory Example.cu` `8fbffc4d356bc661`
- `Problems/L7-Optimization Masterclass/Zero-Copy Memory.cu` `7d72dbc1cb84b71e`
- `Problems/L7-Optimization Masterclass/occupancy calculator.cu` `2cb8acbdabe4135b`
- `Problems/L8-Production Quality Code/Stream Callbacks.cu` `93d4393bb101d8f4`
- `Problems/L8-Production Quality Code/template kernel library.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- the host wrapper uses `N` (lines 32, 37) where its parameter is `n`.] `24e0067219434831`
- `Problems/L9-Domain Applications/2D Poisson Solver.cu` `db2dc4be13ad899c`
- `Problems/L9-Domain Applications/Matrix Multiply.cu` `205ee528c5db7258`
- `Problems/L9-Domain Applications/batch normalization.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- __syncthreads() inside `if (tid < s)` in both reductions: a barrier only some threads reach, which deadlocks (the 'timeout'); every thread reads mean[0] with no barrier after thread 0 writes it; and 256 threads for N=1024 averages only the first 256 elements. Copy: barriers moved out of the branch, a barrier before mean[0] is read, one block of N threads.] `d63ff4c6c001ba75`

## silent (4)

ran clean, prints nothing: only the exit status is checkable, there is no output to compare

- `Problems/L7-Optimization Masterclass/grid stride loop perfomance.cu`
- `Problems/L8-Production Quality Code/basic streams.cu`
- `Problems/L8-Production Quality Code/multi stream data processing.cu`
- `Problems/L8-Production Quality Code/overlapping compute and copy.cu`

## pass-only (42)

ran clean but output varies between runs (timings): judge by exit status only

- `Day 11/reduction_kernel_v5.cu`
- `Day 11/reduction_kernel_v6.cu`
- `Day 11/reduction_kernel_v7.cu`
- `Day 12/reduction_kernel_v8.cu`
- `Day 14/OptimizedDotProduct.cu`
- `Day 14/reduction_sum_cpu_vs_gpu.cu`
- `Day 15/matrixTranspose.cu`
- `Day 17/L2Norm.cu`
- `Day 17/OptimizedGEMM.cu`
- `Day 18/OptimizedL2BlockSizes.cu`
- `Day 18/OptimizedL2Norm.cu`
- `Day 19/max_reduction.cu`
- `Day 4/arrayMultiplication.cu`
- `Day 4/vectorAdditionTiming.cu`
- `Day 5/additionofvectorby2.cu` [output differs between Colab sessions, so not a reference]
- `Day 5/arrayMulCPUvsGPU.cu`
- `Day 6/matrixAddition.cu`
- `Day 7/matrixAdditionCPUvsGPU.cu`
- `Day 8/arrayMulOptimized.cu` [output differs between Colab sessions, so not a reference]
- `Day 8/matrixMulTiled.cu`
- `Day 9/reduction_kernel_v1.cu`
- `Day 9/reduction_kernel_v2.cu`
- `Day 9/reduction_kernel_v3.cu`
- `Day 9/reduction_kernel_v4.cu`
- `Problems/L1-Map Patterns/vector_addition.cu`
- `Problems/L3-Memory Patterns/SOA_VS_AOS.cu`
- `Problems/L3-Memory Patterns/matrix multiplication optimized.cu`
- `Problems/L3-Memory Patterns/matrix_multiply(naive).cu`
- `Problems/L3-Memory Patterns/shared memory bank conflicts.cu`
- `Problems/L3-Memory Patterns/shared memory double buffering.cu`
- `Problems/L3-Memory Patterns/strided_access_analysis.cu`
- `Problems/L4-Scan Pattern - Prefix Sums/segmented_Scan_blocklevel.cu`
- `Problems/L6-Advanced Patterns/K-Means Clustering.cu`
- `Problems/L6-Advanced Patterns/Radix Sort (Complete).cu` [PATCHED COPY (source changed to build/run; clone untouched) -- two extern __shared__ kernels launched with no shared-memory size (the illegal address); the one-block scan assumes M*BINS <= 1024 (its own comment) but N=2^20 gives 65536; and the histogram is block-major where a radix sort needs digit-major. Copy: N=2^14 (M*BINS=1024), shared sizes passed, digit-major histogram and offsets.]
- `Problems/L7-Optimization Masterclass/Fused Multiply Add.cu`
- `Problems/L7-Optimization Masterclass/Pinned Memory Tranfers.cu`
- `Problems/L7-Optimization Masterclass/Unified Memory Perfomance.cu`
- `Problems/L7-Optimization Masterclass/block size tuning.cu`
- `Problems/L7-Optimization Masterclass/cache blocking study.cu`
- `Problems/L8-Production Quality Code/perfomance benchmarking.cu`
- `Profiling_Lab/Sum Reduction/reduction_kernel2.cu`
- `Profiling_Lab/Sum Reduction/reduction_sum.cu`
