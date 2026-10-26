#!/usr/bin/env python3
"""The third-party CUDA corpus under cuda-projects/: manifest, and Colab baseline.

    python3 -I tools/cuda_projects.py manifest            # Mac: write cuda-projects/corpus.json
    python3 -I tools/cuda_projects.py notebook            # Mac: write oracle/colab/bench_<repo>.ipynb
    python3 -I tools/cuda_projects.py ledger              # Mac: write RESULTS/cuda-projects/LEDGER.md
    python3 cuda_projects.py bench --root R --out O       # Colab: build + run the bench tier

Three tiers, decided per repository (see REPOS):

  bench           standalone programs that build with nvcc alone.  Whether the
                  translator got them right is only knowable against a run on
                  a real NVIDIA GPU, so these go to Colab first.  The run
                  records exit status, a hash of stdout and whether two runs
                  agree, so a program that prints timings is not mistaken for
                  a reference.
  translate-only  needs torch, model weights, a library the Mac does not have
                  yet, or a GPU Colab does not offer.  There is nothing to
                  baseline; the translator is checked by "does it build" (G-C1)
                  and, later, by the program's own self-test.
  reference       no CUDA source to translate (an index, a Numba course).

THE MACHINE RULE (same as oracle/run_oracle.py).  A Colab T4 is one sample of a
CUDA host, not a target.  The machine block is recorded with every result and
no timing is read from it.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECTS = os.path.join(ROOT, "cuda-projects")

# slug -> (tier, why).  The folder name is the last path component of the slug.
REPOS = {
    "NVIDIA/cuda-samples": ("bench", "per-sample CMake; most print PASS/FAIL themselves"),
    "CoffeeBeforeArch/cuda_programming": ("bench", "one program per directory"),
    "olcf/cuda-training-series": ("bench", "homework templates skipped (FIXME); *_solution.cu run"),
    "siboehm/SGEMM_CUDA": ("bench", "13 SGEMM kernels checked against cuBLAS; special-cased"),
    "karpathy/llm.c": ("bench", "dev/cuda: each kernel file is standalone with a CPU reference"),
    "NVIDIA-developer-blog/code-samples": ("bench", "blog posts; standalone programs"),
    "yasin-arafat-05/Learning_CUDA_From_Scratch": ("bench", "graded problems; standalone"),
    "ridash2005/CUDA_Kernels": ("bench", "numbered modules with CPU verification"),
    "hoshuaclawdbot/cuda-learning-lab": ("bench", "small CMake project"),
    "Infatoshi/cuda-course": ("bench", "course kernels; cuDNN/Triton/PyTorch parts auto-skipped"),
    "gpu-mode/lectures": ("bench", "per-lecture programs; torch-based ones auto-skipped"),
    "rogerallen/raytracinginoneweekendincuda": ("bench", "pinned to branch ch12_where_next_cuda (master is the CPU C++ original, no CUDA); main.cu writes a PPM to stdout"),
    "rahul238xaviers/cuda_training": ("bench", "standalone .cu; templates skipped"),
    "NVIDIA/CUDALibrarySamples": ("translate-only", "one Makefile/CMake per sample and each needs a CUDA library the translator does not provide yet; add to bench once the library shims exist"),
    "NVIDIA/cutlass": ("translate-only", "G-C1 names the Ampere examples; they need sm_80 and Colab's free T4 is sm_75"),
    "NVIDIA/cccl": ("translate-only", "Thrust/CUB/libcu++ headers; the translator's stress test, not a program"),
    "HazyResearch/ThunderKittens": ("translate-only", "needs sm_90a (H100) for most kernels"),
    "Dao-AILab/flash-attention": ("translate-only", "builds as a torch extension; G-C3"),
    "vllm-project/vllm": ("translate-only", "csrc/ is torch-bound; G-C3"),
    "NVIDIA/TensorRT-LLM": ("translate-only", "needs TensorRT, H100-class GPUs and engines"),
    "rapidsai/cudf": ("translate-only", "needs the RAPIDS stack (rmm, libcudf build)"),
    "rapidsai/cuml": ("translate-only", "needs the RAPIDS stack"),
    "ggml-org/llama.cpp": ("translate-only", "G-C1 names GGML_CUDA=ON; running it needs model weights"),
    "ankan-ban/llama_cu_awq": ("translate-only", "needs AWQ model weights"),
    "xlite-dev/LeetCUDA": ("translate-only", "kernels are torch extensions; tests need PyTorch"),
    "hkevin01/cuda-kernel": ("translate-only", "Qt/GUI application around a few kernels"),
    "rkinas/cuda-learning": ("reference", "an index of links; no source"),
    "srush/GPU-Puzzles": ("reference", "Numba (Python), not .cu"),
}

# Headers a standalone nvcc build cannot satisfy on Colab without more setup.
UNSUPPORTED = [
    (r"GL/|glut|GLFW|cuda_gl_interop|EGL|X11", "needs OpenGL / a display"),
    (r"cudnn", "needs cuDNN"),
    (r"nccl", "needs NCCL"),
    (r"\bmpi\.h", "needs MPI"),
    (r"vulkan|cuda_d3d|dxgi", "needs Vulkan / D3D"),
    (r"torch/|pybind11", "needs PyTorch"),
    (r"cutlass/|cute/", "needs the CUTLASS include path"),
    (r"nvToolsExt", "NVTX v2 header was removed from recent toolkits"),
]
# include -> link flags
LIBS = [
    (r"cublasLt\.h", ["-lcublasLt", "-lcublas"]),
    (r"cublas(_v2|Xt)?\.h", ["-lcublas"]),
    (r"curand", ["-lcurand"]),
    (r"cufft", ["-lcufft"]),
    (r"cusparse", ["-lcusparse"]),
    (r"cusolver", ["-lcusolver"]),
    (r"nvml\.h", ["-lnvidia-ml"]),
    (r"nvrtc\.h", ["-lnvrtc", "-lcuda"]),
    (r"[<\"]cuda\.h[>\"]", ["-lcuda"]),
]
TEMPLATE = re.compile(r"FIXME|YOUR CODE HERE|TODO:?\s*(implement|fill|complete)", re.I)
SKIP_DIRS = {".git", "build", "node_modules", "third_party", "3rdparty", "__pycache__"}
SHARED_DIRS = {"common", "Common", "include", "inc", "utils", "helpers"}
SHADOW_HEADERS = ("unistd.h", "dirent.h", "direct.h", "io.h", "windows.h")
# Programs that are heavy by design get longer than the default run timeout.
RUN_TIMEOUT = {"karpathy/llm.c": 300}
# A run that hits its limit is restarted once with this many times the budget.
# A kernel that is merely slow finishes; one that hangs times out again and is
# recorded with every attempt.
RESCUE_FACTOR = 3
# Timing loops that make a run take minutes.  Each program checks its result
# (validate_result) *before* its timing loop, so a copy with one repeat still
# checks every block size.  The copy is compiled; the clone is never edited, and
# the result is marked "shortened" -- its time is not a reference.
TIMING_SHORTEN = {
    "karpathy/llm.c": {
        "dev/cuda/matmul_forward.cu": [("int repeat_times = 100;", "int repeat_times = 1;")],
        # one benchmark call takes 86-188 s per block size on a T4: time one size, not six
        "dev/cuda/attention_backward.cu": [("int repeat_times = 10;", "int repeat_times = 1;"),
                                           ("int block_sizes[] = {32, 64, 128, 256, 512, 1024};",
                                            "int block_sizes[] = {256};")],
        "dev/cuda/classifier_fused.cu": [("int repeat_times = 1000;", "int repeat_times = 1;")],
    }
}
# Sources that are CUDA but do not end in .cu.
EXTRA_SOURCES = {}
# Where in a repo the single-file tier looks.
SUBTREE = {"karpathy/llm.c": "dev/cuda"}
# Build flags the program's own Makefile sets and nvcc does not default to.  A flag starting
# with "!" drops that flag from the detected libraries; "-I@dir" is an include dir inside the
# repo; "@cuda13_shim" force-includes CUDA13_SHIM (see below).
EXTRA_FLAGS = {
    "ridash2005/CUDA_Kernels": {"modules/07_ecosystem/07_dynamic_parallelism/src/cdp.cu": ["-rdc=true"]},
    "NVIDIA-developer-blog/code-samples": {
        "c++11_cuda/c++11_cuda.cu": ["-I@c++11_cuda/cpp11-range"],
        "posts/cufft-callbacks/cufft_callbacks.cu": ["!-lcufft", "-rdc=true", "-lcufft_static", "-lculibos"],
        "posts/parallel_reduction_with_shfl/main.cu": ["@legacy_shfl"],
        "posts/nvml/nvml_test.cu": ["-L/usr/local/cuda/lib64/stubs", "-lnvidia-ml"],
        "posts/unified-memory-oversubscription/uvm_oversubs.cu": ["@cuda13_shim"],
    },
}
# Programs that print a usage line and exit 0 when run bare: the arguments their README/Makefile use.
RUN_ARGS = {
    "NVIDIA-developer-blog/code-samples": {"posts/parallel_reduction_with_shfl/main.cu": ["1048576", "100"]},
}
# Git submodules the shallow clones do not carry, fetched at the commit the repo pins.
# repo slug -> [(path inside the repo, url, commit)]
SUBMODULES = {
    "NVIDIA-developer-blog/code-samples": [
        ("c++11_cuda/cpp11-range", "https://github.com/harrism/cpp11-range", "70f844968c5f669ce85f8ce4cbd24a3584c57f4b"),
    ],
}
# __shfl_down(v, delta) without a mask was removed after CUDA 8 (sm_70+ needs __shfl_down_sync).  This
# gives the old call back, for a full warp, to a header the harness cannot patch (it only copies the .cu).
LEGACY_SHFL = """#pragma once
#include <cuda_runtime.h>
template <class T> __device__ __forceinline__ T __shfl_down(T v, unsigned delta, int width = 32) {
  return __shfl_down_sync(0xffffffffu, v, delta, width);
}
"""
# CUDA 13 changed cudaMemPrefetchAsync / cudaMemAdvise to take a cudaMemLocation instead of a device
# id.  This header gives the old call shape back so a program written for CUDA <= 12 builds unedited.
CUDA13_SHIM = """#pragma once
#include <cuda_runtime.h>
#if CUDART_VERSION >= 13000
static inline cudaMemLocation cuda13_loc(int dev) {
  cudaMemLocation l; l.type = dev == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  l.id = dev == cudaCpuDeviceId ? 0 : dev; return l;
}
static inline cudaError_t cudaMemPrefetchAsync(const void *p, size_t n, int dev, cudaStream_t s = 0) {
  return cudaMemPrefetchAsync(p, n, cuda13_loc(dev), 0, s);
}
static inline cudaError_t cudaMemAdvise(const void *p, size_t n, enum cudaMemoryAdvise a, int dev) {
  return cudaMemAdvise(p, n, a, cuda13_loc(dev));
}
#endif
"""
# Upstream programs that are wrong as written.  The clone is never edited; the harness
# compiles a patched copy, and the result says so ("patched") so the translator is not
# held to a program whose behaviour is undefined.  Each entry: (why, [(old, new), ...]).
PATCHED_COPIES = {
    "NVIDIA-developer-blog/code-samples": {
        "posts/american-options/longstaff_schwartz_svd_2.cu": (
            "written for CUDA <= 8: __any(pred) was removed for sm_70+ (ptxas: 'vote without .sync is not supported'). The copy uses __any_sync(0xffffffff, pred), the full-warp form that call site already assumes.",
            [("__any(in_the_money)", "__any_sync(0xffffffff, in_the_money)")]),
        "posts/cufft-callbacks/cufft_callbacks.cu": (
            "CUFFT_LICENSE_ERROR no longer exists in cuFFT (the license check was dropped); the copy removes that one branch and keeps the status check.",
            [("if (status == CUFFT_LICENSE_ERROR) {", "if (false) {")]),
        "posts/parallel_reduction_with_shfl/main.cu": (
            "the vendored 2016 NVlabs/cub submodule collides with the CCCL cub that ships in CUDA 13 (#error CUB_NS_QUALIFIER); the copy includes the toolkit's <cub/cub.cuh> instead, and replaces cub::Sum() (removed in CCCL 3) with cuda::std::plus<int>() and the init value 0 that CCCL 3's Reduce now requires. The header warp_reduce.h calls __shfl_down() without a mask (removed for sm_70+); the build force-includes a one-function shim giving it the full-warp __shfl_down_sync form.",
            [('#include "cub/cub/cub.cuh"', "#include <cub/cub.cuh>"),
             ("N, cub::Sum());\n  cudaMalloc", "N, cuda::std::plus<int>(), 0);\n  cudaMalloc"),
             ("REPEAT;i++) {\n    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, in, out, N, cub::Sum());",
              "REPEAT;i++) {\n    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, in, out, N, cuda::std::plus<int>(), 0);")]),
        "posts/rnn/LSTM.cu": (
            "upstream bug: `return time < 0;` compares the libc function time() with 0, which g++ now rejects. The copy returns totalTime < 0 (always 0), the evident intent.",
            [("return time < 0;", "return totalTime < 0;")]),
    },
    "ridash2005/CUDA_Kernels": {
        "modules/02_memory_management/04_unified_memory/src/unified_mem.cu": (
            "written for CUDA <= 12 -- not a bug in the program. In CUDA 13 cudaMemPrefetchAsync(ptr, bytes, device) "
            "takes a cudaMemLocation and a flags argument instead of a device id (the same break as in "
            "cuda-training-series hw6). The copy uses the CUDA 13 form at all three call sites.",
            [("cudaMemPrefetchAsync(mData_A, bytes, deviceId)",
              "cudaMemPrefetchAsync(mData_A, bytes, cudaMemLocation{cudaMemLocationTypeDevice, deviceId}, 0, 0)"),
             ("cudaMemPrefetchAsync(mData_B, bytes, deviceId)",
              "cudaMemPrefetchAsync(mData_B, bytes, cudaMemLocation{cudaMemLocationTypeDevice, deviceId}, 0, 0)"),
             ("cudaMemPrefetchAsync(mData_C, bytes, deviceId)",
              "cudaMemPrefetchAsync(mData_C, bytes, cudaMemLocation{cudaMemLocationTypeDevice, deviceId}, 0, 0)")]),
    },
    "CoffeeBeforeArch/cuda_programming": {
        "01_vector_addition/unified_memory/vectorAdd_um_prefetch.cu": (
            "written for CUDA <= 12 -- not a bug in the program. In CUDA 13 both cudaMemPrefetchAsync(ptr, bytes, "
            "device) and cudaMemAdvise(ptr, bytes, advice, device) take a cudaMemLocation instead of a device id "
            "(the same break as in cuda-training-series hw6). The copy uses the CUDA 13 form at all ten call sites (4 cudaMemAdvise, 6 cudaMemPrefetchAsync).",
            [('  cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeHost, 0}; cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, loc); }'),
             ('  cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeHost, 0}; cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, loc); }'),
             ('  cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeDevice, id}; cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, loc); }'),
             ('  cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeDevice, id}; cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, loc); }'),
             ('  cudaMemPrefetchAsync(c, bytes, id);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeDevice, id}; cudaMemPrefetchAsync(c, bytes, loc, 0, 0); }'),
             ('  cudaMemPrefetchAsync(a, bytes, id);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeDevice, id}; cudaMemPrefetchAsync(a, bytes, loc, 0, 0); }'),
             ('  cudaMemPrefetchAsync(b, bytes, id);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeDevice, id}; cudaMemPrefetchAsync(b, bytes, loc, 0, 0); }'),
             ('  cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeHost, 0}; cudaMemPrefetchAsync(a, bytes, loc, 0, 0); }'),
             ('  cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeHost, 0}; cudaMemPrefetchAsync(b, bytes, loc, 0, 0); }'),
             ('  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);',
              '  { cudaMemLocation loc = {cudaMemLocationTypeHost, 0}; cudaMemPrefetchAsync(c, bytes, loc, 0, 0); }')]),
    },
    "olcf/cuda-training-series": {
        "exercises/hw6/array_inc_solution.cu": (
            "written for CUDA <= 12 -- not a bug in the program. cudaMemPrefetchAsync(ptr, bytes, device) was "
            "replaced in CUDA 13 by a form taking a cudaMemLocation and flags (nvcc 13: 'no suitable constructor "
            "exists to convert from int to cudaMemLocation'). The copy uses the CUDA 13 form.",
            [("cudaMemPrefetchAsync(h_array, ds*sizeof(h_array[0]), 0); // add in step 2c",
              "{ cudaMemLocation loc = {cudaMemLocationTypeDevice, 0}; cudaMemPrefetchAsync(h_array, ds*sizeof(h_array[0]), loc, 0, 0); } // add in step 2c"),
             ("cudaMemPrefetchAsync(h_array, ds*sizeof(h_array[0]), cudaCpuDeviceId); // add in step 2c",
              "{ cudaMemLocation loc = {cudaMemLocationTypeHost, 0}; cudaMemPrefetchAsync(h_array, ds*sizeof(h_array[0]), loc, 0, 0); } // add in step 2c")]),
    },
    "yasin-arafat-05/Learning_CUDA_From_Scratch": {
        "Problems/L6-Advanced Patterns/LockFreeQueue.cu": (
            "d_head/d_tail are declared `unsigned int` but used everywhere as device pointers "
            "(cudaMalloc(&d_head), atomicAdd(d_head, 1), *d_head); declared as `unsigned int*`.",
            [("        unsigned int d_head;\n        unsigned int d_tail;",
              "        unsigned int* d_head;\n        unsigned int* d_tail;")]),
        "Problems/L8-Production Quality Code/template kernel library.cu": (
            "the host wrapper uses `N` (lines 32, 37) where its parameter is `n`.",
            [("int blocks = (N + threadsPerBlock * 2 - 1)", "int blocks = (n + threadsPerBlock * 2 - 1)"),
             ("(d_input, d_intermediate, N, op, identity)", "(d_input, d_intermediate, n, op, identity)")]),
        "Day 14/leetgpuDotProduct.cu": (
            "free() on the stack arrays h_A/h_B aborts with 'double free or corruption'; removed.",
            [("\tfree(h_A);\n\tfree(h_B);\n", "\t// free(h_A); free(h_B); -- stack arrays, removed\n")]),
        "Problems/L4-Scan Pattern - Prefix Sums/Partition.cu": (
            "free() on the stack arrays h_input/h_output aborts with 'double free or corruption'; removed.",
            [("    free(h_input); free(h_output);\n", "    // free(h_input); free(h_output); -- stack arrays, removed\n")]),
        "Problems/L6-Advanced Patterns/Radix Sort (Complete).cu": (
            "two extern __shared__ kernels launched with no shared-memory size (the illegal address); the "
            "one-block scan assumes M*BINS <= 1024 (its own comment) but N=2^20 gives 65536; and the "
            "histogram is block-major where a radix sort needs digit-major. Copy: N=2^14 (M*BINS=1024), "
            "shared sizes passed, digit-major histogram and offsets.",
            [("#define N 1048576             // 2^20 elements",
              "#define N 16384               // 2^14: keeps M*BINS = 1024, the scan's one-block assumption"),
             ("histogram_kernel<<<M, THREADS_PER_BLOCK>>>(d_in, d_hist, N, bit);",
              "histogram_kernel<<<M, THREADS_PER_BLOCK, BINS * sizeof(int)>>>(d_in, d_hist, N, bit);"),
             ("scan_kernel<<<1, 1024>>>(d_hist, d_scan, M * BINS);",
              "scan_kernel<<<1, 1024, 1024 * sizeof(int)>>>(d_hist, d_scan, M * BINS);"),
             ("global_hist[blockIdx.x * BINS + tid] = local_hist[tid];",
              "global_hist[tid * gridDim.x + blockIdx.x] = local_hist[tid];"),
             ("local_offsets[tid] = scan_offsets[blockIdx.x * BINS + tid];",
              "local_offsets[tid] = scan_offsets[tid * gridDim.x + blockIdx.x];")]),
        "Problems/L7-Optimization Masterclass/Loop Unrolling.cu": (
            "three 4 MB arrays on the stack (12 MB > the 8 MB limit: the segfault), and <<<1, N/4>>> asks "
            "for 262144 threads in one block (limit 1024). Copy: static arrays, 256-thread blocks.",
            [("    float h_a[N], h_b[N], h_c[N];", "    static float h_a[N], h_b[N], h_c[N];"),
             ("manual_unroll<<<1, N / 4>>>", "manual_unroll<<<(N / 4 + 255) / 256, 256>>>"),
             ("pragma_unroll<<<1, N / 4>>>", "pragma_unroll<<<(N / 4 + 255) / 256, 256>>>")]),
        "Problems/L9-Domain Applications/batch normalization.cu": (
            "__syncthreads() inside `if (tid < s)` in both reductions: a barrier only some threads reach, "
            "which deadlocks (the 'timeout'); every thread reads mean[0] with no barrier after thread 0 "
            "writes it; and 256 threads for N=1024 averages only the first 256 elements. Copy: barriers "
            "moved out of the branch, a barrier before mean[0] is read, one block of N threads.",
            [("            sdata[tid] += sdata[tid + s];\n            __syncthreads();\n        }\n    }\n"
              "    if (tid == 0) {\n        mean[blockIdx.x] = sdata[0] / N;\n    }\n",
              "            sdata[tid] += sdata[tid + s];\n        }\n        __syncthreads();\n    }\n"
              "    if (tid == 0) {\n        mean[blockIdx.x] = sdata[0] / N;\n    }\n"
              "    __syncthreads();   // every thread reads mean[0] below\n"),
             ("            sdata[tid] += sdata[tid + s];\n            __syncthreads();\n        }\n    }\n"
              "    if (tid == 0) {\n        var[blockIdx.x]",
              "            sdata[tid] += sdata[tid + s];\n        }\n        __syncthreads();\n    }\n"
              "    if (tid == 0) {\n        var[blockIdx.x]"),
             ("batchNorm<<<1, 256, 256 * sizeof(float)>>>", "batchNorm<<<1, N, N * sizeof(float)>>>")]),
        "Problems/L4-Scan Pattern - Prefix Sums/prefix-sum (Hills-Steele).cu": (
            "each step does `out += in[tid-offset]` into the other buffer, which was never initialised (the "
            "known GPU Gems listing bug); it must be `out = in[tid] + in[tid-offset]`. The program prints "
            "'Mismatch' yet exits 0.",
            [("sdata[pout * n + tid] += sdata[pin * n + tid - offset];",
              "sdata[pout * n + tid] = sdata[pin * n + tid] + sdata[pin * n + tid - offset];")]),
    },
    "gpu-mode/lectures": {
        "lecture_012/main.cu": (
            "main.cu allocates head dim 64 and launches 32x16 threads for both kernels, but the kernels are "
            "compiled for head dim 128 with their own tiles (16x16 / 8x32) and blocks (16x16 / 128x8): "
            "out-of-bounds reads, 'illegal memory access' on any NVIDIA GPU. The copy uses d=128 and "
            "gives each kernel its own tile count, block and grid.",
            [("int d = 64;        // feature dimension", "int d = 128;       // feature dimension"),
             ("flash_attention_spilling_from_registers_k<<<blocks, threads>>>(",
              "flash_attention_spilling_from_registers_k<<<blocks, dim3(16, 16)>>>("),
             ("    flash_attention_k<<<blocks, threads>>>(",
              "    T_r = cdiv(n, 8);        // flash_attention_k tiles are B_r=8, B_c=32\n"
              "    T_c = cdiv(n_inp, 32);\n"
              "    flash_attention_k<<<dim3(T_r), dim3(128, 8)>>>(")]),
    },
}
# Notebooks whose first cell run should also redo some non-timeout statuses, because
# the harness was fixed after the first Colab run (lectures/lecture_012 needed its sibling .cu linked).
RERUN_PRESET = {"lectures": "build-fail,run-fail", "cuda-samples": "run-fail",
                "Learning_CUDA_From_Scratch": "build-fail,run-fail",
                "CUDA_Kernels": "build-fail"}
# Per-notebook budget for those reruns: seconds per run in round 1 (the restart gets 3x), and rounds.
# batch normalization.cu got 120 s + 360 s last time and printed nothing, so give it more, once.
RUN_TIMEOUT_PRESET = {}
ROUNDS_PRESET = {}
# Programs a notebook should run once more even though they passed (cleared after one use per session).
RERUN_PATHS_PRESET = {"lectures": "lecture_012", "Learning_CUDA_From_Scratch": "Hills-Steele",
                      "cuda-training-series": "array_inc_solution",
                      "cuda_programming": "vectorAdd_um_prefetch"}
SAMPLE_CATEGORIES = ["0_Introduction", "1_Utilities", "2_Concepts_and_Techniques",
                     "3_CUDA_Features", "4_CUDA_Libraries", "6_Performance"]
MAIN = re.compile(r"\bmain\s*\(")
FAIL = re.compile(r"\b(fail(ed|ure)?|error|mismatch|incorrect|wrong)\b", re.I)
# "Max error: 0.000000e+00" is a pass, not a failure.
ZERO_ERR = re.compile(r"\berror\s*[:=]?\s*[-+]?0*\.?0*(e[-+]?\d+)?\s*$", re.I)


# A number after "error" is a measurement ("Absolute error: 1.1e-04", "Error amount = 0.000000"),
# not a failure; so is a deliberate one ("Device assert failed as expected").
METRIC_ERR = re.compile(r"\berror(\s+amount)?\s*[:=]\s*[-+]?\d[\d.]*(e[-+]?\d+)?|\berror\b[\w ]{0,20}[:=]\s*[-+]?\d[\d.]*(e[-+]?\d+)?\s*$|\berror\s+vs\b", re.I)


def has_fail(text):
    for line in text.splitlines():
        if re.search(r"as expected", line, re.I):
            continue
        line = METRIC_ERR.sub("", line)
        if FAIL.search(line) and not ZERO_ERR.search(line.strip()):
            return True
    return False


PASSM = re.compile(r"\b(pass(ed)?|success(ful)?|ok|correct)\b", re.I)


def folder(slug):
    return slug.split("/")[-1]


def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def sh(cmd, cwd=None, timeout=60):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return 124, out, "timeout after %ss" % timeout
    except OSError as e:
        return 127, "", str(e)


# ---------------------------------------------------------------- discovery

def local_headers(path, text, depth=3, seen=None):
    """Text of the repo-local headers a file pulls in with #include "..." (transitively).

    llm.c's kernels get cuBLAS from common.h, not from themselves, so the file
    alone does not say what it must link against or whether it needs cuDNN.
    """
    seen = set() if seen is None else seen
    out = []
    if depth == 0:
        return out
    for m in re.finditer(r'^\s*#\s*include\s*"([^"]+)"', text, re.M):
        p = os.path.normpath(os.path.join(os.path.dirname(path), m.group(1)))
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        t = read(p)
        out.append(t)
        out += local_headers(p, t, depth - 1, seen)
    return out


def classify_source(path, text):
    """None if the file can be built alone, else the reason it cannot."""
    if TEMPLATE.search(text):
        return "exercise template"
    text = text + "\n" + "\n".join(local_headers(path, text))
    includes = "\n".join(l for l in text.splitlines() if l.lstrip().startswith("#include"))
    for pat, why in UNSUPPORTED:
        if re.search(pat, includes, re.I):
            return why
    return None


def link_flags(text):
    includes = "\n".join(l for l in text.splitlines() if l.lstrip().startswith("#include"))
    flags = []
    for pat, fl in LIBS:
        if re.search(pat, includes):
            flags += [f for f in fl if f not in flags]
    if re.search(r"this_grid|grid_group|cudaLaunchCooperativeKernel", text):
        flags += ["-rdc=true", "-lcudadevrt"]
    return flags


def include_dirs(repo_dir, src_dir):
    dirs, cur = [], src_dir
    while True:
        # An ancestor holding a stand-in for a system header (llm.c's dev/unistd.h
        # is a Windows shim) must not go on -I: it would shadow the real one.
        if cur == src_dir or not any(os.path.exists(os.path.join(cur, h)) for h in SHADOW_HEADERS):
            dirs.append(cur)
        for d in sorted(os.listdir(cur)):
            if d in SHARED_DIRS and os.path.isdir(os.path.join(cur, d)):
                dirs.append(os.path.join(cur, d))
                # common/include/ (CUDA_Kernels) is the usual nesting
                nested = os.path.join(cur, d, "include")
                if os.path.isdir(nested):
                    dirs.append(nested)
        if os.path.abspath(cur) == os.path.abspath(repo_dir):
            break
        cur = os.path.dirname(cur)
    return dirs


def discover_single(slug, repo_dir):
    """Every standalone program in the repo, with why the others are skipped."""
    start = os.path.join(repo_dir, SUBTREE.get(slug, ""))
    extra = set(EXTRA_SOURCES.get(slug, []))
    found, skipped = [], []
    for dp, dns, fns in os.walk(start):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
        for fn in sorted(fns):
            if not (fn.endswith(".cu") or fn in extra):
                continue
            path = os.path.join(dp, fn)
            rel = os.path.relpath(path, repo_dir)
            text = read(path)
            if not MAIN.search(text):
                continue  # a kernel/header-style file; some other file owns main()
            why = classify_source(path, text)
            if why:
                skipped.append({"path": rel, "reason": why})
            else:
                found.append({"path": rel, "kind": "single",
                              "libs": link_flags(text + "\n" + "\n".join(local_headers(path, text)))})
    return found, skipped


def sample_dirs(repo_dir, categories):
    out = []
    for cat in categories:
        base = os.path.join(repo_dir, "cpp", cat)
        if os.path.isdir(base):
            for d in sorted(os.listdir(base)):
                if os.path.isfile(os.path.join(base, d, "CMakeLists.txt")):
                    out.append(os.path.relpath(os.path.join(base, d), repo_dir))
    return out


def discover(slug, repo_dir):
    if slug == "NVIDIA/cuda-samples":
        return [{"path": p, "kind": "cmake-sample"} for p in sample_dirs(repo_dir, SAMPLE_CATEGORIES)], []
    if slug == "siboehm/SGEMM_CUDA":
        return [{"path": "sgemm.cu", "kind": "sgemm", "kernels": list(range(13))}], []
    return discover_single(slug, repo_dir)


# ----------------------------------------------------------------- manifest

def git_head(d):
    rc, out, _ = sh(["git", "-C", d, "rev-parse", "HEAD"])
    return out.strip() if rc == 0 else None


def counts(repo_dir):
    c = {"cu": 0, "cuh": 0}
    for dp, dns, fns in os.walk(repo_dir):
        dns[:] = [d for d in dns if d != ".git"]
        for fn in fns:
            for e in c:
                if fn.endswith("." + e):
                    c[e] += 1
    return c


def cmd_manifest(args):
    if os.path.exists(os.path.join(PROJECTS, "ready.json")) and not args.force:
        sys.exit("refusing: the clones were thinned by `ready --apply` (see cuda-projects/ready.json), so discovery would now "
                 "see only the programs that did not pass and corpus.json would lose the baseline's targets. "
                 "Use --force only on fresh clones.")
    repos = []
    for slug, (tier, why) in REPOS.items():
        d = os.path.join(PROJECTS, folder(slug))
        entry = {"slug": slug, "dir": folder(slug), "url": "https://github.com/%s.git" % slug,
                 "tier": tier, "why": why, "commit": None, "present": os.path.isdir(d)}
        if entry["present"]:
            entry["commit"] = git_head(d)
            entry.update(counts(d))
            if tier == "bench":
                found, skipped = discover(slug, d)
                entry["targets"] = found
                entry["skipped"] = skipped
        repos.append(entry)
    doc = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "note": "commit pins what the translator was fed; the Colab bench checks out exactly these",
           "repos": repos}
    path = os.path.join(PROJECTS, "corpus.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    for e in repos:
        print("%-16s %-44s %s" % (e["tier"], e["slug"],
              "%d targets, %d skipped" % (len(e["targets"]), len(e["skipped"])) if "targets" in e else ""))
    print("wrote", path)


# -------------------------------------------------------------------- bench

def machine_block(nvcc):
    _, smi, _ = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader"], timeout=30)
    _, ver, _ = sh([nvcc, "--version"], timeout=30)
    return {"gpu": smi.strip(), "nvcc": ver.strip().splitlines()[-1:] or None,
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def compute_cap():
    _, out, _ = sh(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], timeout=30)
    first = out.strip().splitlines()[:1]
    return first[0].replace(".", "") if first else None


def gpu_count():
    _, out, _ = sh(["nvidia-smi", "-L"], timeout=30)
    return len([l for l in out.splitlines() if l.startswith("GPU ")])


def unmet(needs):
    """Why this host cannot run a target that declares `needs`, or None."""
    if not needs:
        return None
    cc = compute_cap()
    if needs.get("min_cc") and (not cc or int(cc) < needs["min_cc"]):
        return "needs compute capability >= %s.%s, this GPU is %s" % (
            needs["min_cc"] // 10, needs["min_cc"] % 10, cc or "unknown")
    if needs.get("min_gpus") and gpu_count() < needs["min_gpus"]:
        return "needs %d GPUs, this host has %d" % (needs["min_gpus"], gpu_count())
    return None


def run_twice(exe, args, cwd, timeout):
    """Run twice; stdout hash is only a reference if the two agree.

    A run that times out is restarted once with RESCUE_FACTOR x the budget.  A
    rescued program is not run a second time (its budget is already large), so
    it is never "deterministic" and triages as pass-only at best.
    """
    # Line-buffered stdout (GNU stdbuf, present on Colab) so a timed-out run still
    # leaves its partial output in the record instead of an empty string.
    pre = ["stdbuf", "-oL"] if shutil.which("stdbuf") else []
    attempts = []

    def once(budget):
        t0 = time.monotonic()
        rc, out, err = sh(pre + [exe] + args, cwd=cwd, timeout=budget)
        attempts.append({"timeout": budget, "rc": rc, "seconds": round(time.monotonic() - t0, 1)})
        return rc, out, err

    def record(rc, out, err, rc2, deterministic, rescued):
        return {
            "rc": rc, "rc2": rc2,
            "stdout_sha256": hashlib.sha256(out.encode()).hexdigest(),
            "deterministic": deterministic,
            "pass_marker": bool(PASSM.search(out)), "fail_marker": has_fail(out + err),
            "stdout_head": out[:1500], "stderr_head": err[:500],
            "rescued": rescued, "attempts": attempts,
        }

    rc, out, err = once(timeout)
    if rc != 124:
        rc2, out2, _ = once(timeout)
        return record(rc, out, err, rc2, rc == rc2 and out == out2, False)
    # Rescue: restart the same program once with a longer budget.
    rc, out, err = once(timeout * RESCUE_FACTOR)
    return record(rc, out, err, None, False, True)


def cmake_waived(r):
    """CUDA samples exit 2 to say 'waived': this GPU lacks a feature the sample needs."""
    if r.get("kind") != "cmake-sample":
        return False
    runs = [r["run"]] if r.get("run") else (
        list(r["runs"].values()) if isinstance(r.get("runs"), dict) else r.get("runs") or [])
    return bool(runs) and any(x["rc"] == 2 for x in runs) and all(x["rc"] in (0, 2) for x in runs)


def overall_status(r):
    """build-fail > timeout > run-fail > ok, over every run the target made."""
    if r.get("build_rc") != 0:
        return "build-fail"
    runs = [r["run"]] if r.get("run") else (
        list(r["runs"].values()) if isinstance(r.get("runs"), dict) else r.get("runs") or [])
    if not runs:
        return "no-executable"
    rcs = [x["rc"] for x in runs]
    if 124 in rcs:
        return "timeout"
    if cmake_waived(r):
        return "waived"
    return "ok" if all(rc == 0 for rc in rcs) else "run-fail"


def shortened_copy(slug, repo_dir, rel, work):
    """Path of a derived copy (timing loops cut and/or a caller bug patched), or None if the file needs none."""
    subs = list(TIMING_SHORTEN.get(slug, {}).get(rel) or []) + list((PATCHED_COPIES.get(slug, {}).get(rel) or ("", []))[1])
    if not subs:
        return None
    text = read(os.path.join(repo_dir, rel))
    for old, new in subs:
        if text.count(old) != 1:
            raise RuntimeError("expected exactly one %r in %s" % (old, rel))
        text = text.replace(old, new)
    path = os.path.join(work, "timing1__" + rel.replace(os.sep, "__"))
    with open(path, "w") as f:
        f.write(text)
    return path


def link_siblings(orig, log):
    """Sibling .cu files (no main, buildable alone) that define a symbol the link step could not find."""
    names = set(re.findall(r"undefined reference to `([^'(]+)", log))
    d = os.path.dirname(orig)
    found = []
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if not fn.endswith(".cu") or p == orig:
            continue
        text = read(p)
        if MAIN.search(text) or classify_source(p, text):
            continue
        if any(re.search(r"\b%s\b" % re.escape(n), text) for n in names):
            found.append(p)
    return found


def bench_single(t, repo_dir, work, nvcc, args):
    orig = os.path.join(repo_dir, t["path"])
    src = shortened_copy(t.get("slug"), repo_dir, t["path"], work) or orig
    exe = os.path.join(work, t["path"].replace(os.sep, "__"))
    # Includes resolve from the original location, never from the copy's directory.
    cmd = [nvcc, "-arch=native", "-O2", "-std=c++17", "-x", "cu", src, "-o", exe]
    for d in include_dirs(repo_dir, os.path.dirname(orig)):
        cmd.insert(1, "-I" + d)
    extra = EXTRA_FLAGS.get(t.get("slug"), {}).get(t["path"], [])
    cmd += [l for l in t.get("libs", []) if "!" + l not in extra]
    for fl in extra:
        if fl.startswith("!"):
            continue
        if fl == "@cuda13_shim":
            shim = os.path.join(work, "cuda13_shim.h")
            with open(shim, "w") as f:
                f.write(CUDA13_SHIM)
            cmd += ["-include", shim]
        elif fl == "@legacy_shfl":
            shim = os.path.join(work, "legacy_shfl.h")
            with open(shim, "w") as f:
                f.write(LEGACY_SHFL)
            cmd += ["-include", shim]
        elif fl.startswith("-I@"):
            cmd.append("-I" + os.path.join(repo_dir, fl[3:]))
        else:
            cmd.append(fl)
    rc, out, err = sh(cmd, cwd=os.path.dirname(orig), timeout=args.build_timeout)
    linked = []
    if rc != 0 and "undefined reference" in out + err:
        # A multi-file program: the missing symbol is defined in a sibling .cu.  Try
        # plain linking first, then relocatable device code.
        extra = link_siblings(orig, out + err)
        for rdc in ([], ["-rdc=true", "-lcudadevrt"]) if extra else []:
            cmd2 = list(cmd)
            at = cmd2.index(src) + 1
            cmd2[at:at] = extra
            rc2, out2, err2 = sh(cmd2 + rdc, cwd=os.path.dirname(orig), timeout=args.build_timeout)
            if rc2 == 0:
                rc, out, err, linked = rc2, out2, err2, [os.path.relpath(x, repo_dir) for x in extra]
                break
    rel = t["path"]
    argv = RUN_ARGS.get(t.get("slug"), {}).get(rel, [])
    run = run_twice(exe, argv, os.path.dirname(orig), args.run_timeout) if rc == 0 else None
    if run and argv:
        run["args"] = argv
    return {"build_rc": rc, "build_log": (out + err)[-6000:], "run": run,
            "shortened": bool(TIMING_SHORTEN.get(t.get("slug"), {}).get(rel)),
            "patched": (PATCHED_COPIES.get(t.get("slug"), {}).get(rel) or ("",))[0],
            "linked_with": linked}


def bench_sgemm(t, repo_dir, work, nvcc, args):
    exe = os.path.join(work, "sgemm")
    srcs = [os.path.join(repo_dir, "sgemm.cu")] + sorted(
        os.path.join(repo_dir, "src", f) for f in os.listdir(os.path.join(repo_dir, "src")) if f.endswith(".cu"))
    # CMakeLists.txt hard-codes sm_86, which a T4 (sm_75) cannot run.
    cmd = [nvcc, "-arch=native", "-O2", "-std=c++17", "-I" + os.path.join(repo_dir, "src"),
           "-I" + repo_dir] + srcs + ["-o", exe, "-lcublas"]
    rc, out, err = sh(cmd, cwd=repo_dir, timeout=args.build_timeout)
    res = {"build_rc": rc, "build_log": (out + err)[-2000:], "runs": {}}
    if rc == 0:
        for k in t["kernels"]:
            res["runs"][str(k)] = run_twice(exe, [str(k)], repo_dir, args.run_timeout)
    return res


def sample_args(repo_dir, name):
    path = os.path.join(repo_dir, "test_args.json")
    cfg = json.loads(read(path)).get(name, {}) if os.path.exists(path) else {}
    if cfg.get("skip"):
        return None
    return [r.get("args", []) for r in cfg["runs"]] if "runs" in cfg else [cfg.get("args", [])]


def bench_sample(t, repo_dir, work, nvcc, args):
    name = os.path.basename(t["path"])
    arg_sets = sample_args(repo_dir, name)
    if arg_sets is None:
        return {"skipped": "marked skip in test_args.json"}
    bdir = os.path.join(work, name)
    cfg = ["cmake", "-S", os.path.join(repo_dir, t["path"]), "-B", bdir,
           "-DCMAKE_CUDA_ARCHITECTURES=" + (args.arch or "native"),
           "-DCMAKE_CUDA_COMPILER=" + nvcc]
    rc, o1, e1 = sh(cfg, timeout=120)
    log = o1 + e1
    if rc == 0:
        rc, o2, e2 = sh(["cmake", "--build", bdir, "-j2"], timeout=args.build_timeout)
        log += o2 + e2
    res = {"build_rc": rc, "build_log": log[-2000:], "runs": []}
    if rc != 0:
        return res
    exes = [os.path.join(dp, f) for dp, dns, fns in os.walk(bdir) if "CMakeFiles" not in dp
            for f in fns if os.access(os.path.join(dp, f), os.X_OK) and "." not in f]
    # The sample's executable is named like its directory.  CPM-fetched dependencies (CCCL) drop
    # their own executable scripts into the build tree, so "the first executable" is not enough.
    named = [e for e in exes if os.path.basename(e) == name]
    for exe in (named or sorted(exes))[:1]:
        # NVIDIA's run_tests.py runs each sample from the executable's directory: the Driver API
        # samples load <name>_kernel64.fatbin/.ptx from there.
        cwd = os.path.dirname(exe)
        for a in arg_sets:
            r = run_twice(exe, a, cwd, args.run_timeout)
            r["args"] = a
            res["runs"].append(r)
    return res


def bench_repo(slug, entry, root, out, nvcc, args):
    repo_dir = os.path.join(root, folder(slug))
    work = os.path.join(out, "build", folder(slug))
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    handler = {"single": bench_single, "sgemm": bench_sgemm, "cmake-sample": bench_sample}
    args = argparse.Namespace(**vars(args))  # the override must not leak to other repos
    args.run_timeout = max(args.run_timeout, RUN_TIMEOUT.get(slug, 0))
    res_path = os.path.join(out, "results", folder(slug) + ".json")
    every = entry["targets"][:args.limit] if args.limit else entry["targets"]
    only_paths = [x for x in args.paths.split(",") if x]
    if only_paths:  # a subset run (e.g. the sm_80-only samples on an A100)
        every = [t for t in every if any(x in t["path"] for x in only_paths)]
    targets, done = every, {}
    if args.resume and os.path.exists(res_path):
        # Keep every finished result; rerun what timed out and what never finished
        # (the cell was stopped or the session dropped mid-run).
        done = {r["path"]: r for r in json.loads(read(res_path))["results"]}
        again = {"timeout"} | set(filter(None, args.rerun_status.split(",")))
        paths = [x for x in args.rerun_paths.split(",") if x]
        targets = [t for t in every if t["path"] not in done or done[t["path"]]["status"] in again
                   or any(x in t["path"] for x in paths)]
        print("resuming %s: %d kept, rerunning %d (run timeout %ss): %s" % (
            slug, len(every) - len(targets), len(targets), args.run_timeout,
            ", ".join(t["path"] for t in targets) or "none"), flush=True)
    lock = threading.Lock()

    def save(complete):
        """Write the results file; called after every program, so nothing finished is lost."""
        rows = [done[t["path"]] for t in every if t["path"] in done]
        tally = {}
        for r in rows:
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        missing = len(every) - len(rows)
        if missing:
            tally["not-run-yet"] = missing
        doc = {"slug": slug, "commit": entry["commit"], "complete": complete and not missing, "tally": tally,
               "skipped_at_discovery": entry["skipped"], "results": rows}
        tmp = res_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp, res_path)
        return tally

    def one(t):
        t0 = datetime.datetime.now()
        t = dict(t, slug=slug)
        try:
            why = unmet(t.get("needs"))
            r = {"skipped": "not run: " + why} if why else handler[t["kind"]](t, repo_dir, work, nvcc, args)
        except Exception as e:  # never stop on a failure
            r = {"build_rc": None, "build_log": "harness error: %r" % (e,)}
        r.update(path=t["path"], kind=t["kind"], seconds=round((datetime.datetime.now() - t0).total_seconds(), 1))
        r["status"] = "skipped" if r.get("skipped") else overall_status(r)
        with lock:
            if t["path"] in done:  # it timed out before and has now been run again
                r["retries"] = done[t["path"]].get("retries", 0) + 1
            done[t["path"]] = r
            save(False)
        print("%-10s %s/%s" % (r["status"], folder(slug), t["path"]), flush=True)
        subs = list(expand(r))
        if len(subs) > 1:  # one target, many runs (SGEMM's 13 kernels): show each verdict
            for s_ in subs:
                print("  %-10s %s" % (s_["status"], s_["path"]), flush=True)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(one, targets))
    print("==", slug, save(True), flush=True)


def cmd_bench(args):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    args.arch = args.arch or compute_cap()
    corpus = json.loads(read(args.manifest))
    wanted = set(args.only.split(",")) if args.only else None
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "machine.json"), "w") as f:
        json.dump(machine_block(nvcc), f, indent=1)
    for entry in corpus["repos"]:
        if entry["tier"] != "bench" or (wanted and entry["dir"] not in wanted):
            continue
        if not os.path.isdir(os.path.join(args.root, entry["dir"])):
            print("MISSING", entry["dir"])
            continue
        bench_repo(entry["slug"], entry, args.root, args.out, nvcc, args)


# ------------------------------------------------------------------- ledger

RESULTS = os.path.join(ROOT, "RESULTS", "cuda-projects")


EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def first_runs(r):
    runs = [r["run"]] if r.get("run") else (
        list(r["runs"].values()) if isinstance(r.get("runs"), dict) else r.get("runs") or [])
    return runs


def expand(r):
    """A target that made several runs (SGEMM's 13 kernels) becomes one row per run."""
    if isinstance(r.get("runs"), dict):
        for k, x in r["runs"].items():
            st = "timeout" if x["rc"] == 124 else "ok" if x["rc"] == 0 else "run-fail"
            yield {"path": "%s kernel %s" % (r["path"], k), "kind": r["kind"], "status": st,
                   "build_rc": r["build_rc"], "run": x}
    else:
        yield r


def run_note(r):
    x = first_runs(r)[0] if first_runs(r) else {}
    text = (x.get("stdout_head") or "") + "\n" + (x.get("stderr_head") or "")
    for l in text.splitlines():
        if re.search(r"fail|diverg|error|timeout", l, re.I):
            return l.strip()[:140]
    return ""


def triage(r):
    """One word per program: what a later session may rely on."""
    st = r["status"]
    if st == "run-fail" and cmake_waived(r):
        return "waived"       # also reads results recorded before the harness knew about exit code 2
    if st == "no-executable" and "will not build sample" in (r.get("build_log") or ""):
        return "not-tested"   # CMake declined: OpenGL / EGL / NvSci is missing on the host
    if st != "ok":
        return st
    runs = first_runs(r)
    if any(has_fail((x.get("stdout_head") or "") + "\n" + (x.get("stderr_head") or "")) for x in runs):
        return "suspect"      # exited 0 but printed a failure-like word; read the log
    if all(x["stdout_sha256"] == EMPTY_SHA for x in runs):
        return "silent"       # nothing printed: the hash proves nothing
    return "reference" if all(x["deterministic"] for x in runs) else "pass-only"


def first_error(r):
    lines = [l.strip() for l in (r.get("build_log") or "").splitlines()
             if re.search(r"error|fatal", l, re.I)]
    return (lines[0] if lines else (r.get("build_log") or "")[-120:].strip())[:160]


# Why a program is not a usable test, from reading its output/source.  slug -> path -> note.
NOTES = {
    "yasin-arafat-05/Learning_CUDA_From_Scratch": {
        "Problems/L8-Production Quality Code/Multi-GPU.cu":
            "needs two GPUs with peer access ('Peer access not supported between devices'); Colab has one -- a hardware limit, not a defect",
    },
}

# Programs that cannot be tried on the bench host at all (hardware it does not have): the run
# says nothing about the program, so the ledger records "not-tested", not a failure.
UNTESTABLE = {
    "yasin-arafat-05/Learning_CUDA_From_Scratch": {
        "Problems/L8-Production Quality Code/Multi-GPU.cu": "needs 2 GPUs with peer access; Colab gives one",
    },
    "olcf/cuda-training-series": {
        "exercises/hw7/multi_solution.cu": "needs 2 GPUs ('invalid device ordinal'); Colab gives one",
    },
    "NVIDIA/cuda-samples": {
        "cpp/3_CUDA_Features/localityDomains": "needs a CUDA toolkit newer than Colab's 13.0 (the samples target 13.4)",
        "cpp/3_CUDA_Features/localityDomainsDrv": "needs a CUDA toolkit newer than Colab's 13.0: CUmemLocation.localized and CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL do not exist in 13.0",
    },
}

# Teaching exercises that are unfinished or deliberately broken (the repo ships the solution as
# a separate file, which is the real test).  They exit 0 or fail, but neither says anything
# about CUDA behaviour.  slug -> path -> why.
NOT_A_TEST = {
    "olcf/cuda-training-series": {
        "exercises/hw12/task1.cu": "deliberately broken debugging exercise (off-by-one `e <= BLOCK_SIZE`, missing barrier); task1_solution.cu is the fixed program",
        "exercises/hw12/task2.cu": "deliberately broken debugging exercise (prints -inf); task2_solution.cu prints 0.693148",
        "exercises/hw13/axpy_cublas_from_scratch.cu": "unfinished student exercise (`thread` for `threads`, missing semicolons, wrong block count); the solution is in hw13/Solutions/",
        "exercises/hw6/linked_list.cu": "unfinished exercise: host malloc'd list handed to the GPU; linked_list_solution.cu uses managed memory",
        "exercises/hw5/max_reduction.cu": "unfinished exercise: still computes the sum; max_reduction_solution.cu computes the max",
    },
}

# Whole families of unfinished exercises, matched by path.  Only a program that failed is
# reclassified: a template that happens to pass stays what it is.
NOT_A_TEST_PATTERN = {
    "cuda_training": (re.compile(r"/exercise/(beginner|intermediate|champion)_workbook\.cu$"),
                      "unfinished fill-in-the-blank workbook (TODO stubs); it reports its own tests as FAILED until the student writes the kernels. "
                      "The matching solution/ folders are empty in the repo (they are the student's workspace), so there is no finished version to run"),
}


def not_a_test(slug, path, status=None):
    if path in NOT_A_TEST.get(slug, {}):
        return NOT_A_TEST[slug][path]
    for key, (rx, why) in NOT_A_TEST_PATTERN.items():
        if slug.endswith("/" + key) and rx.search("/" + path) and status in (None, "run-fail", "build-fail"):
            return why
    return None


# Programs whose output legitimately contains failure words (they provoke the failure on purpose).
EXPECTED_FAIL_WORDS = {
    "NVIDIA/cuda-samples": {
        "cpp/0_Introduction/simpleAssert": "triggers a device-side assert on purpose and exits 0 ('failed as expected')",
    },
    "NVIDIA-developer-blog/code-samples": {
        "posts/printf-alternative/main.cu": "demonstrates reporting device-side errors from the kernel on purpose (large value, negative sqrt); the 'ERROR' lines are its output",
    },
    "rahul238xaviers/cuda_training": {
        "kernels/practice/vector_dot_reduction.cu": "prints [FAILED] on purpose: the float CPU reference loses precision at a total of ~1e19 (rel err 3.5e-3 vs the 1e-4 threshold); the program's own comment says the GPU sum is the accurate one",
    },
}

MEANING = {
    "waived": "ran and declined: the sample's own check says this GPU lacks a feature (exit code 2 is the CUDA samples' convention); rerun on a newer GPU",
    "not-a-test": "an unfinished or deliberately broken teaching exercise; the repo's solution file is the test",
    "not-tested": "needs hardware, libraries or a toolkit the bench host does not have (2 GPUs, OpenGL/EGL, a newer CUDA); no verdict about the program",
    "reference": "ran clean, two runs agree: its stdout SHA-256 is a usable reference",
    "silent": "ran clean, prints nothing: only the exit status is checkable, there is no output to compare",
    "pass-only": "ran clean but output varies between runs (timings): judge by exit status only",
    "suspect": "exit 0 but output contains a failure-like word: read before trusting",
    "build-fail": "nvcc/cmake failed on the GPU host: see first error; often not a translator target",
    "run-fail": "built, non-zero exit", "timeout": "built, exceeded the run timeout",
    "skipped": "marked skip upstream", "no-executable": "built but no executable found",
}


def classify(d, extras):
    """Every row of one repo's result file with its final triage and cross-session evidence."""
    results = [x for r in d["results"] for x in expand(r)]
    for r in results:
        r["triage"] = triage(r)
        for rows in extras:
            other = rows.get(r["path"])
            if other is None or bool(other.get("patched")) != bool(r.get("patched")):
                continue      # a session that built different source is not a repeat
            # Only a session where the program also ran ok can disagree with this one; an
            # earlier session that failed (for instance before a harness fix) is not evidence.
            if r["triage"] == "reference" and other["status"] == "ok" and (triage(other) != "reference" or
                    first_runs(other)[0]["stdout_sha256"] != first_runs(r)[0]["stdout_sha256"]):
                r["triage"] = "pass-only"      # agreed within a session but not across sessions
                r["unstable"] = True
            if r["status"] != "ok" and other["status"] == r["status"]:
                r["reproduced"] = r.get("reproduced", 0) + 1
        if r["path"] in UNTESTABLE.get(d["slug"], {}):
            r["triage"] = "not-tested"
        if r["triage"] == "suspect" and r["path"] in EXPECTED_FAIL_WORDS.get(d["slug"], {}):
            r["triage"] = "pass-only"
            r["expected_fail_note"] = EXPECTED_FAIL_WORDS[d["slug"]][r["path"]]
        if not_a_test(d["slug"], r["path"], r["status"]):
            r["triage"] = "not-a-test"
    return results


def cmd_ledger(_):
    gpus = sorted(d for d in os.listdir(RESULTS) if os.path.isfile(os.path.join(RESULTS, d, "machine.json")))
    top = ["# Bench ledger -- third-party CUDA programs run on a real NVIDIA GPU", "",
           "Generated by `python3 -I tools/cuda_projects.py ledger`; do not edit by hand.",
           "Sources and pinned commits: `cuda-projects/corpus.json`. Raw data: `RESULTS/cuda-projects/<gpu>/results/<repo>.json`.",
           "One GPU is one sample of a CUDA host, never a target; no timing is read from these runs.", "",
           "How to read a program's `triage`:", ""]
    top += ["- **passed** (a column, not a label): ran with exit code 0 and no failure words = reference + pass-only + silent. "
            "Patched copies count as passed; their page says so.", ""]
    top += ["- **%s**: %s" % (k, v) for k, v in MEANING.items()] + [""]
    # every GPU's primary result rows, to say "this passes on <other gpu>" next to a waived/failed row
    allrows = {}
    for g in gpus:
        rd = os.path.join(RESULTS, g, "results")
        for fn in sorted(os.listdir(rd)):
            if fn.endswith(".json") and not re.match(r".+\.run\d+\.json$", fn):
                allrows.setdefault(g, {})[fn[:-5]] = {x["path"]: x for x in json.load(open(os.path.join(rd, fn)))["results"]}
    for gpu in gpus:
        m = json.load(open(os.path.join(RESULTS, gpu, "machine.json")))
        top += ["## %s" % gpu, "", "`%s`, %s, %s" % (m["gpu"], " ".join(m.get("nvcc") or []), m["date"]), ""]
        if m.get("note"):
            top += ["*%s*" % m["note"], ""]
        top += [
                "| repo | commit | programs | **passed** | reference | pass-only | silent | suspect | build-fail | other | detail |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        tot = [0] * 8     # programs, passed, reference, pass-only, silent, suspect, build-fail, other
        rdir = os.path.join(RESULTS, gpu, "results")
        # <repo>.run2.json, <repo>.run3.json ... are further Colab sessions of the same repo: they
        # are not rows of their own, they test whether the first session's result repeats.
        extras = {}
        for fn in sorted(os.listdir(rdir)):
            m = re.match(r"(.+)\.run\d+\.json$", fn)
            if m:
                rows = {x["path"]: x for r in json.load(open(os.path.join(rdir, fn)))["results"] for x in expand(r)}
                extras.setdefault(m.group(1), []).append(rows)
        for fn in sorted(os.listdir(rdir)):
            if re.match(r".+\.run\d+\.json$", fn):
                continue
            d = json.load(open(os.path.join(rdir, fn)))
            name = fn[:-5]
            tri = {}
            results = classify(d, extras.get(name, []))
            for r in results:
                tri.setdefault(r["triage"], []).append(r)
            other = sum(len(v) for k, v in tri.items() if k not in ("reference", "pass-only", "suspect", "build-fail", "silent"))
            c = lambda k: len(tri.get(k, []))
            passed = c("reference") + c("pass-only") + c("silent")
            for i, v in enumerate((len(results), passed, c("reference"), c("pass-only"), c("silent"),
                                   c("suspect"), c("build-fail"), other)):
                tot[i] += v
            top.append("| %s | `%s` | %d | **%d** | %d | %d | %d | %d | %d | %d | [%s.md](%s/%s.md) |" % (
                d["slug"] + ("" if d.get("complete", True) else " (incomplete run)"),
                (d["commit"] or "")[:9], len(results), passed, c("reference"), c("pass-only"), c("silent"),
                c("suspect"), c("build-fail"), other, name, gpu, name))
            page = ["# %s on %s" % (d["slug"], gpu), "", "Commit `%s`. Back to [LEDGER.md](../LEDGER.md)." % d["commit"], ""]
            page += ["Skipped before running (%d): %s" % (len(d["skipped_at_discovery"]),
                     "; ".join("%s (%s)" % (s["path"], s["reason"]) for s in d["skipped_at_discovery"][:30]) or "none"), ""]
            for k in MEANING:
                if k not in tri:
                    continue
                page += ["## %s (%d)" % (k, len(tri[k])), "", MEANING[k], ""]
                for r in tri[k]:
                    extra = " [timing loops cut (1 repeat, fewer sizes): correctness only]" if r.get("shortened") else ""
                    if r.get("patched"):
                        extra += " [PATCHED COPY (source changed to build/run; clone untouched) -- %s]" % r["patched"]
                    runs = first_runs(r)
                    if r["status"] != "ok":
                        for og, repos in allrows.items():
                            orow = repos.get(name, {}).get(r["path"]) if og != gpu else None
                            if orow and orow["status"] == "ok":
                                extra += " [ok on %s]" % og
                    if r.get("expected_fail_note"):
                        extra += " [failure words in the output are expected: %s]" % r["expected_fail_note"]
                    if r.get("unstable"):
                        extra += " [output differs between Colab sessions, so not a reference]"
                    if r.get("reproduced"):
                        extra += " [same result in %d further session(s)]" % r["reproduced"]
                    fix = PATCHED_COPIES.get(d["slug"], {}).get(r["path"])
                    if fix and not r.get("patched"):
                        extra += " -- a patched copy now exists: " + fix[0]
                    why = not_a_test(d["slug"], r["path"], r["status"]) or UNTESTABLE.get(d["slug"], {}).get(r["path"])
                    if why:
                        extra += " -- " + why
                    note = NOTES.get(d["slug"], {}).get(r["path"])
                    if note:
                        extra += " -- NOTE: " + note
                    if r.get("linked_with"):
                        extra += " [linked with %s]" % ", ".join(r["linked_with"])
                    if r.get("retries"):
                        extra += " [retried %dx]" % r["retries"]
                    if any(x.get("rescued") for x in runs):
                        extra += " [rescued: restarted with a longer budget; attempts %s]" % (
                            ", ".join("%ss -> rc %s" % (a["timeout"], a["rc"]) for a in runs[0]["attempts"]))
                    if k == "reference" and runs:
                        extra += " `%s`" % runs[0]["stdout_sha256"][:16]
                    elif k == "build-fail":
                        extra += " -- `%s`" % first_error(r).replace("`", "'")
                    elif k in ("suspect", "run-fail", "timeout"):
                        rn = run_note(r)
                        extra += " -- %s(raw: `%s`)" % ("`%s` " % rn.replace("`", "'") if rn else "",
                                                       os.path.join(gpu, "results", name + ".json"))
                    page.append("- `%s`%s" % (r["path"], extra))
                page.append("")
            with open(os.path.join(RESULTS, gpu, name + ".md"), "w") as f:
                f.write("\n".join(page))
        top.append("| **total** | | %d | **%d** | %d | %d | %d | %d | %d | %d | |" % tuple(tot))
        top.append("")
    with open(os.path.join(RESULTS, "LEDGER.md"), "w") as f:
        f.write("\n".join(top) + "\n")
    print("wrote", os.path.join(RESULTS, "LEDGER.md"))


# ----------------------------------------------------------------- notebook

# ---------------------------------------------------------- ready / retest

PASSED = ("reference", "pass-only", "silent")
READY = os.path.join(PROJECTS, "ready")


def load_all():
    """Every classified row of every GPU: [(gpu, doc, rows)]."""
    out = []
    for g in sorted(os.listdir(RESULTS)):
        rd = os.path.join(RESULTS, g, "results")
        if not os.path.isdir(rd):
            continue
        for fn in sorted(os.listdir(rd)):
            if re.match(r".+\.run\d+\.json$", fn):
                continue
            d = json.load(open(os.path.join(rd, fn)))
            extras = [{x["path"]: x for r in json.load(open(os.path.join(rd, f2)))["results"] for x in expand(r)}
                      for f2 in sorted(os.listdir(rd)) if re.match(re.escape(fn[:-5]) + r"\.run\d+\.json$", f2)]
            out.append((g, d, classify(d, extras)))
    return out


def verdicts():
    """(slug, path) -> {gpu: row}; one program, its row on every GPU that ran it."""
    v = {}
    for g, d, rows in load_all():
        for r in rows:
            v.setdefault((d["slug"], r["path"]), {})[g] = dict(r, commit=d["commit"])
    return v


def split_verdicts():
    """passed / not_a_test / failed, each a list of (slug, path, {gpu: row})."""
    passed, nat, failed = [], [], []
    for (slug, path), by in sorted(verdicts().items()):
        tri = {g: r["triage"] for g, r in by.items()}
        if any(t in PASSED for t in tri.values()):
            passed.append((slug, path, by))
        elif all(t == "not-a-test" for t in tri.values()):
            nat.append((slug, path, by))
        else:
            failed.append((slug, path, by))
    return passed, nat, failed


def target_path(path):
    """A row of SGEMM's kernel k belongs to the one target sgemm.cu."""
    m = re.match(r"^(sgemm\.cu) kernel \d+$", path)
    return m.group(1) if m else path


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def header_closure(repo_dir, files):
    """Repo-local files a set of sources includes (#include "x" or <x>), transitively."""
    found, todo = set(), list(files)
    while todo:
        f = todo.pop()
        text = read(os.path.join(repo_dir, f))
        for m in re.finditer(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', text, re.M):
            cands = [os.path.join(os.path.dirname(os.path.join(repo_dir, f)), m.group(1))]
            cands += [os.path.join(d, m.group(1)) for d in include_dirs(repo_dir, os.path.dirname(os.path.join(repo_dir, f)))]
            for c in cands:
                c = os.path.normpath(c)
                if os.path.isfile(c) and os.path.abspath(c).startswith(os.path.abspath(repo_dir) + os.sep):
                    rel = os.path.relpath(c, repo_dir)
                    if rel not in found and rel not in files:
                        found.add(rel)
                        todo.append(rel)
                    break
    return found


def program_files(repo_dir, t, path):
    """Source files one program needs, as repo-relative paths."""
    if t["kind"] == "cmake-sample":
        return None
    if t["kind"] == "sgemm":
        srcs = ["sgemm.cu"] + sorted("src/" + f for f in os.listdir(os.path.join(repo_dir, "src")) if f.endswith(".cu"))
        return srcs + sorted(header_closure(repo_dir, srcs) | {"src/" + f for f in os.listdir(os.path.join(repo_dir, "src")) if f.endswith((".cuh", ".h"))})
    return [path] + list(t.get("linked_with") or [])


def patched_text(slug, rel, text):
    subs = (PATCHED_COPIES.get(slug, {}).get(rel) or ("", []))[1]
    for old, new in subs:
        if text.count(old) != 1:
            raise RuntimeError("expected exactly one %r in %s" % (old, rel))
        text = text.replace(old, new)
    return text


def cmd_ready(args):
    ready_json = os.path.join(PROJECTS, "ready.json")
    corpus = {r["slug"]: r for r in json.load(open(os.path.join(PROJECTS, "corpus.json")))["repos"]}
    passed, nat, failed = split_verdicts()
    old = json.load(open(ready_json)) if os.path.exists(ready_json) and args.add else None
    if old:
        have = {(r["slug"], p["path"]) for r in old["repos"] for p in r["programs"]}
        passed = [x for x in passed if (x[0], x[1]) not in have]
        nat = []
    elif os.path.exists(ready_json) and not args.rebuild:
        doc = json.load(open(ready_json))
        bad = 0
        for repo in doc["repos"]:
            for rel, sha in repo["files"].items():
                p = os.path.join(READY, repo["dir"], rel)
                if not os.path.isfile(p) or sha256_file(p) != sha:
                    bad += 1
                    print("MISMATCH", repo["dir"], rel)
        print("ready.json: %d programs in %d repos, %d file mismatches" % (
            sum(len(r["programs"]) for r in doc["repos"]), len(doc["repos"]), bad))
        if args.apply and not bad:
            n = 0
            for rel in doc["removed_from_clone"]:
                p = os.path.join(PROJECTS, rel)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                    n += 1
                elif os.path.exists(p):
                    os.remove(p)
                    n += 1
            print("removed %d paths from the clones" % n)
        annotate(doc)
        with open(ready_json, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        write_ready_md(doc)
        return
    repos, taken = ({r["slug"]: r for r in old["repos"]} if old else {}), {}
    # what every program that stays in the clone still reads (a program must not be removed from under it)
    needed = {}
    for slug, path, by in failed:
        e = corpus[slug]
        rd = os.path.join(PROJECTS, e["dir"])
        t = next((x for x in e["targets"] if x["path"] == target_path(path)), None)
        if t and t["kind"] == "single" and os.path.isfile(os.path.join(rd, path)):
            for f in [path] + sorted(header_closure(rd, [path])):
                needed.setdefault(slug, set()).add(f)
    removed = []
    for slug, path, by in passed:
        e = corpus[slug]
        rd = os.path.join(PROJECTS, e["dir"])
        tpath = target_path(path)
        t = next(x for x in e["targets"] if x["path"] == tpath)
        gpu0 = next(g for g, r in by.items() if r["triage"] in PASSED)
        row = by[gpu0]
        ev = [{"gpu": g, "triage": r["triage"], "stdout_sha256": (first_runs(r)[0]["stdout_sha256"] if first_runs(r) else None),
               "deterministic": all(x["deterministic"] for x in first_runs(r)) if first_runs(r) else None}
              for g, r in sorted(by.items()) if r["triage"] in PASSED]
        entry = {"path": path, "kind": t["kind"], "triage": row["triage"], "verified_on": ev,
                 "build": {"libs": t.get("libs", []), "extra": [f for f in EXTRA_FLAGS.get(slug, {}).get(tpath, [])],
                           "linked_with": row.get("linked_with") or []},
                 "patched": (PATCHED_COPIES.get(slug, {}).get(tpath) or ("",))[0] or None,
                 "timing_shortened": bool(TIMING_SHORTEN.get(slug, {}).get(tpath)),
                 "files": []}
        if t["kind"] == "sgemm":
            entry["kernel"] = int(path.rsplit(" kernel ", 1)[1])
        repo = repos.setdefault(slug, {"slug": slug, "dir": e["dir"], "url": e["url"], "commit": e["commit"], "files": {}, "programs": []})
        files = program_files(rd, t, tpath) if t["kind"] != "cmake-sample" else None
        dest_root = os.path.join(READY, e["dir"])
        copies = []   # (relative path, bytes)
        if files is None:     # a CMake sample: its directory plus the repo-level pieces its CMakeLists reaches
            for dp, dns, fns in os.walk(os.path.join(rd, tpath)):
                dns[:] = [d for d in dns if d not in ("build", ".git")]
                for fn in fns:
                    copies.append(os.path.relpath(os.path.join(dp, fn), rd))
            for top in ("Common", "cmake"):
                for dp, dns, fns in os.walk(os.path.join(rd, top)):
                    copies += [os.path.relpath(os.path.join(dp, fn), rd) for fn in fns]
            copies += [f for f in ("test_args.json", "LICENSE") if os.path.isfile(os.path.join(rd, f))]
        else:
            copies = files + sorted(header_closure(rd, files))
        for rel in sorted(set(copies)):
            src = os.path.join(rd, rel)
            if not os.path.isfile(src):
                continue            # an earlier program of this run already moved it
            data = open(src, "rb").read()
            if rel == tpath and entry["patched"]:
                entry["upstream_sha256"] = hashlib.sha256(data).hexdigest()
                data = patched_text(slug, rel, data.decode()).encode()
            dst = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) and open(dst, "rb").read() != data:
                raise RuntimeError("two programs need different %s" % rel)
            with open(dst, "wb") as f:
                f.write(data)
            if files is not None or rel.startswith(tpath + os.sep):
                entry["files"].append(rel)      # a CMake sample lists only its own directory; Common/ and cmake/ are repo-wide
            repo["files"][rel] = hashlib.sha256(data).hexdigest()
        repo["programs"].append(entry)
        # what leaves the clone: the program itself, unless a program that stays still reads it
        if t["kind"] == "sgemm":
            continue
        if t["kind"] == "cmake-sample":
            taken[(slug, tpath)] = os.path.join(rd, tpath)
        elif tpath not in needed.get(slug, set()):
            taken[(slug, tpath)] = os.path.join(rd, tpath)
    for slug, path, by in nat:
        e = corpus[slug]
        why = next((not_a_test(slug, path, r["status"]) for r in by.values() if not_a_test(slug, path, r["status"])), "")
        removed.append({"slug": slug, "path": path, "why": why})
        p = os.path.join(PROJECTS, e["dir"], path)
        if os.path.isfile(p):
            taken[(slug, path)] = p
    doc = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "note": "Programs that built and ran clean on a real NVIDIA GPU (reference / pass-only / silent), copied unedited "
                   "(except where `patched`) into cuda-projects/ready/<dir>/ with the headers and sibling sources they need. "
                   "`removed_from_clone` lists what was taken out of cuda-projects/<dir>/ afterwards; "
                   "`git -C cuda-projects/<dir> checkout -- <path>` restores any of it.",
           "repos": list(repos.values()), "removed_not_a_test": old["removed_not_a_test"] if old else removed,
           "removed_from_clone": sorted(set(old["removed_from_clone"] if old else []) | {"%s/%s" % (corpus[k[0]]["dir"], k[1]) for k in taken})}
    # verify every copy before anything is deleted
    for repo in doc["repos"]:
        for rel, sha in repo["files"].items():
            assert sha256_file(os.path.join(READY, repo["dir"], rel)) == sha, rel
    annotate(doc)
    with open(ready_json, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    n = 0
    if args.apply:
        for (slug, path), p in sorted(taken.items()):
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
            n += 1
    print("ready: %d programs in %d repos; %d not-a-test; %d paths %s" % (
        sum(len(r["programs"]) for r in doc["repos"]), len(doc["repos"]), len(doc["removed_not_a_test"]), len(taken),
        "removed from the clones" if args.apply else "would be removed (rerun with --apply)"))
    write_ready_md(doc)


def annotate(doc):
    """Run arguments and shim sources that a program's verdict depended on (kept current from the tables)."""
    shims = {"@cuda13_shim": CUDA13_SHIM, "@legacy_shfl": LEGACY_SHFL}
    for repo in doc["repos"]:
        for pr in repo["programs"]:
            tp = target_path(pr["path"])
            args = RUN_ARGS.get(repo["slug"], {}).get(tp)
            if args:
                pr["build"]["run_args"] = args
            used = {k: shims[k] for k in pr["build"]["extra"] if k in shims}
            if used:
                pr["build"]["shims"] = used


def write_ready_md(doc):
    L = ["# Ready to convert -- programs that passed on a real GPU", "",
         "Generated by `python3 -I tools/cuda_projects.py ready`; do not edit by hand.",
         "Machine-readable: `cuda-projects/ready.json` (every file with its SHA-256). Sources: `cuda-projects/ready/<dir>/`, "
         "laid out as in the upstream repo so relative `#include`s still resolve.", "",
         "A program is here when it built with nvcc and exited 0 with no failure words on at least one GPU "
         "(reference, pass-only or silent in [LEDGER.md](LEDGER.md)). `patched` means the file is the repaired copy that passed; "
         "`upstream_sha256` is the file as upstream ships it. Output hashes are only a reference where `triage` is `reference`.", "",
         "| repo | commit | ready | reference | pass-only | silent | patched |", "|---|---|---|---|---|---|---|"]
    tot = [0] * 5
    for r in doc["repos"]:
        ps = r["programs"]
        c = [len(ps)] + [sum(1 for p in ps if p["triage"] == t) for t in PASSED] + [sum(1 for p in ps if p["patched"])]
        tot = [a + b for a, b in zip(tot, c)]
        L.append("| %s | `%s` | **%d** | %d | %d | %d | %d |" % ((r["slug"], r["commit"][:9]) + tuple(c)))
    L += ["| **total** | | **%d** | %d | %d | %d | %d |" % tuple(tot), ""]
    for r in doc["repos"]:
        L += ["## %s (`%s`)" % (r["slug"], r["commit"][:9]), ""]
        for p in r["programs"]:
            bits = [p["triage"]] + ["on " + ", ".join(e["gpu"] for e in p["verified_on"])]
            if p["patched"]:
                bits.append("patched: " + p["patched"])
            if p["timing_shortened"]:
                bits.append("timing loops shortened for the run")
            if p["build"]["libs"] or p["build"]["extra"]:
                bits.append("flags " + " ".join(p["build"]["libs"] + p["build"]["extra"]))
            if p["build"].get("run_args"):
                bits.append("run with args " + " ".join(p["build"]["run_args"]))
            if p["build"]["linked_with"]:
                bits.append("linked with " + ", ".join(p["build"]["linked_with"]))
            L.append("- `%s` -- %s" % (p["path"], "; ".join(bits)))
        L.append("")
    L += ["## Removed as not programs (%d)" % len(doc["removed_not_a_test"]), "",
          "Unfinished or deliberately broken teaching exercises; taken out of the clones, still listed in the ledger as `not-a-test`.", ""]
    L += ["- `%s/%s`" % (x["slug"], x["path"]) for x in doc["removed_not_a_test"]]
    with open(os.path.join(RESULTS, "READY.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", os.path.join(RESULTS, "READY.md"))


# --------------------------------------------------------------- retest

# Where a program that did not pass can be tried again.
WHERE = {
    "any": "any CUDA GPU -- the Colab T4 is enough",
    "sm80": "an sm_80+ GPU -- Colab A100 or L4",
    "multi": "a host with 2+ GPUs and peer access -- not available on Colab",
    "display": "a GPU with a Linux desktop (X + OpenGL, freeglut and GLEW installed) -- the RTX 5080 box, not Colab",
    "egl": "NVIDIA EGL / NvSci hardware (Jetson or DRIVE; the CrossGPU one also needs 2 GPUs) -- not Colab",
    "toolkit134": "a CUDA 13.4 toolkit on a recent GPU -- Colab ships 13.0",
    "none": "tried on a T4 and an A100 and failed the same way -- no further hardware to try",
}
# path (substring) -> (where, what is different this time).  A failed program with no entry stops `retest`.
RETEST = [
    ("c++11_cuda/c++11_cuda.cu", "any", "needs the cpp11-range git submodule, which the shallow clone does not carry: the notebook now fetches it at the pinned commit"),
    ("longstaff_schwartz_svd_2.cu", "any", "patched copy: __any() -> __any_sync() (removed for sm_70+)"),
    ("cuda-vmm/sync_main.cu", "any", "timed out at 60 s: the run budget is now 300 s (it may be a spin-wait that needs more CPU cores than Colab's 2)"),
    ("cufft_callbacks.cu", "none", "failed identically on a T4 and an A100 (the wrong values differ, the expected (16.59, 247.90) is the same): the callback does not give the reference result under CUDA 13. Nothing more to try on Colab"),
    ("cufft_no_callbacks.cu", "none", "failed on a T4 and an A100 with printed values equal to the digit: the program compares floats with !=, so its verdict is rounding noise. Upstream bug, nothing more to try"),
    ("nvml_test.cu", "any", "now links the NVML stub (-L.../stubs -lnvidia-ml) and the sibling matrixMul.cu"),
    ("parallel_reduction_with_shfl/main.cu", "any", "the vendored 2016 cub submodule clashed with CUDA 13's CCCL cub (#error CUB_NS_QUALIFIER); patched copy now includes the toolkit's <cub/cub.cuh>"),
    ("posts/rnn/LSTM.cu", "any", "patched copy: `return time < 0` (libc time()) -> `totalTime < 0`"),
    ("uvm_oversubs.cu", "any", "CUDA 13 prefetch/advise: built with a force-included shim that restores the old call shape"),
    ("sgemm.cu kernel 1", "any", "timed out at 60 s on the T4: the run budget is now 300 s"),
    ("sgemm.cu kernel 11", "none", "diverges from cuBLAS on a T4 and on an A100 (different numbers each time): the kernel is wrong or non-deterministic as shipped. Nothing more to try on Colab"),
    ("Multi-GPU.cu", "multi", "peer access between two GPUs"),
    ("multi_solution.cu", "multi", "two GPUs ('invalid device ordinal' on one)"),
    ("simpleP2P", "multi", "waived on one GPU: peer-to-peer needs 2"),
    ("streamOrderedAllocationP2P", "multi", "waived on one GPU: peer-to-peer needs 2"),
    ("conjugateGradientMultiDeviceCG", "multi", "waived on one GPU: multi-device cooperative groups needs 2"),
    ("simpleCUDA2GL", "display", "OpenGL interop; marked skip upstream for headless runs"),
    ("simpleTexture3D", "display", "OpenGL interop"), ("FunctionPointers", "display", "OpenGL interop"),
    ("boxFilter", "display", "OpenGL interop"), ("imageDenoising", "display", "OpenGL interop"),
    ("particles", "display", "OpenGL interop"), ("bindlessTexture", "display", "OpenGL interop"),
    ("oceanFFT", "display", "OpenGL interop"), ("randomFog", "display", "OpenGL interop"),
    ("EGLStream_CUDA_Interop", "egl", "EGL stream interop"), ("EGLStream_CUDA_CrossGPU", "egl", "EGL stream between two GPUs"),
    ("cudaNvSci", "egl", "NvSci interop"),
    ("localityDomains", "toolkit134", "uses CUDA 13.4 API (CUmemLocation.localized, locality-domain attribute)"),
]
NEEDS = {"sm80": {"min_cc": 80}, "multi": {"min_gpus": 2}}


def cmd_retest(_):
    corpus = {r["slug"]: r for r in json.load(open(os.path.join(PROJECTS, "corpus.json")))["repos"]}
    _, _, failed = split_verdicts()
    items = []
    for slug, path, by in failed:
        hit = next(((w, why) for key, w, why in sorted(RETEST, key=lambda r: -len(r[0])) if key in path), None)
        if not hit:
            sys.exit("no retest entry for %s %s -- decide where it can be tried again" % (slug, path))
        gpu, row = next(iter(by.items()))
        last = by[sorted(by)[0]]
        items.append({"slug": slug, "dir": corpus[slug]["dir"], "path": path, "target": target_path(path),
                      "failed_as": last["triage"], "why_failed": (first_error(last) if last["triage"] == "build-fail" else run_note(last)) or last.get("build_log", "")[-160:],
                      "where": hit[0], "change": hit[1], "colab": hit[0] in ("any", "sm80")})
    doc = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "where": WHERE, "items": items}
    with open(os.path.join(PROJECTS, "retest.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    L = ["# Retest -- every program that has not passed", "",
         "Generated by `python3 -I tools/cuda_projects.py retest`; do not edit by hand. Machine-readable: `cuda-projects/retest.json`.",
         "Unfinished or deliberately broken exercises are not here: they were removed (see [READY.md](READY.md)). "
         "Everything below is a real program that failed, was waived, or could not be tried on the T4.", "",
         "One Colab notebook, `oracle/colab/bench_retest.ipynb`, runs the rows whose GPU is Colab-able; on a GPU below what a row needs it records "
         "'not run: needs ...' instead of a false failure. Run it once on a T4 and once on an A100 (or L4).", ""]
    for key, label in WHERE.items():
        rows = [i for i in items if i["where"] == key]
        if not rows:
            continue
        L += ["## %s -- %d programs" % (label, len(rows)), ""]
        L += ["- `%s/%s` (was %s) -- %s" % (i["dir"], i["path"], i["failed_as"], i["change"]) for i in rows]
        L.append("")
    with open(os.path.join(RESULTS, "RETEST.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    runnable = [i for i in items if i["colab"]]
    repos = []
    for slug in sorted({i["slug"] for i in runnable}):
        e = dict(corpus[slug])
        tg, seen = [], set()
        for i in runnable:
            if i["slug"] != slug or i["target"] in seen:
                continue
            seen.add(i["target"])
            t = next(x for x in e["targets"] if x["path"] == i["target"])
            t = dict(t)
            if t["kind"] == "sgemm":
                t["kernels"] = sorted({int(x["path"].rsplit(" ", 1)[1]) for x in runnable if x["target"] == i["target"]})
            if NEEDS.get(i["where"]):
                t["needs"] = NEEDS[i["where"]]
            tg.append(t)
        e["targets"] = tg
        repos.append(e)
    path = os.path.join(ROOT, "oracle", "colab", "bench_retest.ipynb")
    with open(path, "w") as f:
        json.dump(build_notebook(repos, "retest", items=items), f, indent=1)
        f.write("\n")
    print("%d programs to retest (%d on Colab); wrote %s, RETEST.md, retest.json" % (len(items), len(runnable), path))


def cmd_merge(args):
    """Fold a retest tarball's results into the repo result files of the same GPU."""
    gdir = os.path.join(RESULTS, args.gpu)
    src = os.path.join(args.dir, "out", "results")
    for fn in sorted(os.listdir(src)):
        new = json.load(open(os.path.join(src, fn)))
        dst = os.path.join(gdir, "results", fn)
        if os.path.exists(dst):
            old = json.load(open(dst))
        else:       # this GPU has not run the repo before: start from an empty result of the same repo
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            old = dict(new, results=[], tally={}, complete=True)
        by = {r["path"]: r for r in old["results"]}
        for r in new["results"]:
            if r["status"] == "skipped":
                continue            # not run on this GPU: says nothing
            r["retest"] = True
            if r["kind"] == "sgemm" and r["path"] in by:
                runs = dict(by[r["path"]]["runs"])
                runs.update(r["runs"])
                r = dict(by[r["path"]], runs=runs, retest=True)
                r["status"] = overall_status(r)
            by[r["path"]] = r
        old["results"] = list(by.values())
        tally = {}
        for r in old["results"]:
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        old["tally"] = tally
        with open(dst, "w") as f:
            json.dump(old, f, indent=1)
        print("merged", fn, tally)
    os.makedirs(os.path.join(gdir, "logs"), exist_ok=True)
    for fn in os.listdir(os.path.join(args.dir, "logs")):
        shutil.copy(os.path.join(args.dir, "logs", fn), os.path.join(gdir, "logs", fn))


def cmd_notebook(_):
    with open(os.path.join(PROJECTS, "corpus.json")) as f:
        corpus = json.load(f)
    bench = [r for r in corpus["repos"] if r["tier"] == "bench"]
    out = os.path.join(ROOT, "oracle", "colab")
    os.makedirs(out, exist_ok=True)
    for r in bench:
        path = os.path.join(out, "bench_%s.ipynb" % r["dir"])
        with open(path, "w") as f:
            json.dump(build_notebook([r], r["dir"]), f, indent=1)
            f.write("\n")
        print("wrote", path)
    # A second-GPU notebook: only the samples a T4 waived because it lacks sm_80 features.
    cs = [r for r in bench if r["dir"] == "cuda-samples"]
    if cs:
        path = os.path.join(out, "bench_cuda-samples_a100.ipynb")
        with open(path, "w") as f:
            json.dump(build_notebook(cs, "cuda-samples", suffix="_a100", paths=A100_SAMPLES), f, indent=1)
            f.write("\n")
        print("wrote", path)


# The T4 waived these because it lacks sm_80 features (2-GPU samples cannot run on Colab at all).
A100_SAMPLES = "bf16TensorCoreGemm,tf32TensorCoreGemm,dmmaTensorCoreGemm,simpleAttributes,cudaCompressibleMemory"


def build_notebook(bench, only, suffix="", paths="", items=None):
    pins = {r["dir"]: [r["url"], r["commit"]] for r in bench}
    src = read(os.path.abspath(__file__))
    manifest = json.dumps({"repos": [{k: r[k] for k in ("slug", "dir", "tier", "commit", "targets", "skipped")}
                                     for r in bench]})

    def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}
    def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(True)}

    cells = [
        md("# cuda4AS baseline run\n\n"
           "Runs the *bench* tier of `cuda-projects/` on a real NVIDIA GPU and tars up what happened: exit status, a "
           "hash of stdout, whether two runs agree, every compiler message. Nothing here is a performance number.\n\n"
           "**Runtime → Change runtime type → a GPU** first (T4 is fine). Then Run all. A failure never stops the run. "
           "This notebook runs one repo (`ONLY` is preset). `LIMIT` caps programs for a quick smoke test. "
           "Everything printed is also saved to `/content/logs/bench_<repo>.log`, which goes into the returned tarball.\n\n"
           "The sources are checked out at the exact commits the translator was fed."),
        code("import os, sys\nos.makedirs('/content/logs', exist_ok=True)\n"
             "LOG = " + repr("/content/logs/bench_%s.log" % (only + suffix)) + "\n"
             "class Tee:\n"
             "    def __init__(self, stream, f):\n        self.stream, self.f = stream, f\n"
             "    def write(self, s):\n        self.stream.write(s); self.f.write(s); return len(s)\n"
             "    def flush(self):\n        self.stream.flush(); self.f.flush()\n"
             "    def __getattr__(self, name):\n        return getattr(self.stream, name)\n"
             "if not isinstance(sys.stdout, Tee):\n"
             "    _logf = open(LOG, 'a', buffering=1)\n"
             "    sys.stdout, sys.stderr = Tee(sys.stdout, _logf), Tee(sys.stderr, _logf)\n"
             "print('logging all output to', LOG)"),
        code("!nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv\n"
             "!nvcc --version | tail -2\n"
             "!cmake --version 2>&1 | head -1"),
        code("import json, os, subprocess\nPINS = json.loads(%r)\nROOT = '/content/cuda-projects'\nos.makedirs(ROOT, exist_ok=True)\n"
             "for d, (url, sha) in PINS.items():\n    p = f'{ROOT}/{d}'\n    if os.path.isdir(p + '/.git'):\n        continue\n"
             "    os.makedirs(p, exist_ok=True)\n"
             "    for c in (['init', '-q'], ['remote', 'add', 'origin', url], ['fetch', '-q', '--depth', '1', 'origin', sha], ['checkout', '-q', 'FETCH_HEAD']):\n"
             "        r = subprocess.run(['git', '-C', p] + c, capture_output=True, text=True)\n"
             "        if r.returncode:\n            print('FAILED', d, c[0], r.stderr.strip()[-200:]); break\n"
             "    else:\n        print('ok', d)\n"
             "SUBS = json.loads(%r)   # git submodules the shallow clone does not carry\n"
             "for d, mods in SUBS.items():\n    for rel, url, sha in mods:\n        p = f'{ROOT}/{d}/{rel}'\n"
             "        if os.path.isdir(p + '/.git'):\n            continue\n        os.makedirs(p, exist_ok=True)\n"
             "        for c in (['init', '-q'], ['remote', 'add', 'origin', url], ['fetch', '-q', '--depth', '1', 'origin', sha], ['checkout', '-q', 'FETCH_HEAD']):\n"
             "            r = subprocess.run(['git', '-C', p] + c, capture_output=True, text=True)\n"
             "            if r.returncode:\n                print('FAILED', d, rel, c[0], r.stderr.strip()[-200:]); break\n"
             "        else:\n            print('ok', d, rel)\n" % (json.dumps(pins), json.dumps(
                 {r["dir"]: SUBMODULES[r["slug"]] for r in bench if r["slug"] in SUBMODULES}))),
        code("%%writefile cuda_projects.py\n" + src),
        code("%%writefile manifest.json\n" + manifest),
        code("import json, os, shlex\n"
             "ONLY = " + repr(only) + "    # the one repo this notebook runs\n"
             "LIMIT = 0         # programs per repo, 0 = all\n"
             "RUN_TIMEOUT = " + repr(RUN_TIMEOUT_PRESET.get(only, 0)) + "   # seconds per run in round 1, 0 = harness default\n"
             "ROUNDS = " + repr(ROUNDS_PRESET.get(only, 2)) + "        # rounds in one go: each later round reruns ONLY what timed out, with double the timeout\n"
             "RERUN = " + repr("" if suffix else RERUN_PRESET.get(only, "")) + "   # statuses to run again in round 1 besides timeouts, e.g. 'build-fail' after a harness fix\n"
             "rerun = set(filter(None, RERUN.split(',')))\n"
             "RERUN_PATHS = " + repr("" if suffix else RERUN_PATHS_PRESET.get(only, "")) + "   # programs to run again once even though they passed, e.g. 'lecture_012'\n"
             "PATHS = " + repr(paths) + "   # run only programs whose path contains one of these (comma-separated); blank = all\n"
             "MARK = '/content/out/.rerun_paths_done'   # so the cell does not repeat them every time it is run\n"
             "if RERUN_PATHS and os.path.exists(MARK) and open(MARK).read() == RERUN_PATHS:\n"
             "    RERUN_PATHS = ''\n"
             "# Results are saved after every program, so this cell is also safe to stop and run again:\n"
             "# it keeps everything that finished and continues with what timed out or never ran.\n"
             "RES = f'/content/out/results/{ONLY}.json'\n"
             "def load():\n"
             "    return json.load(open(RES)) if os.path.exists(RES) else None\n"
             "def first_budget(d):  # the timeout the timed-out programs had on their first attempt\n"
             "    b = []\n"
             "    for r in d['results']:\n"
             "        if r['status'] != 'timeout':\n"
             "            continue\n"
             "        runs = [r['run']] if r.get('run') else list((r.get('runs') or {}).values()) if isinstance(r.get('runs'), dict) else (r.get('runs') or [])\n"
             "        b += [x['attempts'][0]['timeout'] for x in runs if x.get('attempts')]\n"
             "    return max(b) if b else 0\n"
             "timeout = RUN_TIMEOUT\n"
             "for rnd in range(1, ROUNDS + 1):\n"
             "    d = load()\n"
             "    again = rerun if rnd == 1 else set()\n"
             "    paths = RERUN_PATHS if rnd == 1 else ''\n"
             "    if d is not None and d.get('complete') and not d['tally'].get('timeout') and not any(d['tally'].get(x) for x in again) and not paths:\n"
             "        print('done:', d['tally'])\n"
             "        break\n"
             "    if d is not None and rnd > 1:\n"
             "        timeout = 2 * (first_budget(d) or timeout or 60)\n"
             "    print(f'--- round {rnd}/{ROUNDS}' + (f', run timeout {timeout}s' if timeout else ', default run timeout'), flush=True)\n"
             "    flags = f'--only {ONLY}'\n"
             "    flags += f' --limit {LIMIT}' if LIMIT else ''\n"
             "    flags += f' --run-timeout {timeout}' if timeout else ''\n"
             "    flags += ' --resume' if d is not None else ''\n"
             "    flags += f' --rerun-status {RERUN}' if d is not None and again else ''\n"
             "    flags += f' --rerun-paths {paths}' if d is not None and paths else ''\n"
             "    flags += ' --paths ' + shlex.quote(PATHS) if PATHS else ''\n"
             "    get_ipython().system('python3 cuda_projects.py bench --manifest manifest.json '\n"
             "                         '--root /content/cuda-projects --out /content/out ' + flags)\n"
             "    if paths:\n"
             "        open(MARK, 'w').write(RERUN_PATHS)\n"
             "else:\n"
             "    d = load()\n"
             "    if d is None:\n"
             "        print('no results file was written: the bench command failed -- read its output above')\n"
             "    elif d['tally'].get('timeout') or d['tally'].get('not-run-yet'):\n"
             "        print(f'still not finished after {ROUNDS} rounds: {d[\"tally\"]} -- recorded as timeouts; '\n"
             "              'run this cell again to keep going')\n"
             "    else:\n"
             "        print('done:', d['tally'])"),
        code("import json, glob\nfor f in sorted(glob.glob('/content/out/results/*.json')):\n    d = json.load(open(f))\n    print(f'{d[\"slug\"]:48s}', d['tally'])"),
        code("!tar czf /content/cuda-projects-baseline-" + only + suffix + ".tgz --exclude=out/build -C /content out logs\n"
             "from google.colab import files\nfiles.download('/content/cuda-projects-baseline-" + only + suffix + ".tgz')"),
        md("Hand the tarball back and it is unpacked under `RESULTS/cuda-projects/<gpu>/`."),
    ]
    if items is not None:
        cells = retest_cells(cells, [r["dir"] for r in bench], items)
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


def retest_cells(cells, dirs, items):
    """The bench notebook, turned into the one that retries everything that has not passed."""
    table = "\n".join("| `%s/%s` | %s | %s |" % (i["dir"], i["path"], WHERE[i["where"]], i["change"]) for i in items)
    cells[0] = {"cell_type": "markdown", "metadata": {}, "source": (
        "# cuda4AS retest -- everything that has not passed\n\n"
        "Retries each real program that failed, was waived, or could not be tried on the T4. Pick a GPU runtime, Run all. "
        "A program that needs more than this GPU has (compute capability, GPU count) is recorded as `not run: needs ...`, not as a failure; "
        "so run the notebook once on a **T4** and once on an **A100 or L4**. The last column says what changed since the first attempt.\n\n"
        "| program | where it can run | what is different |\n|---|---|---|\n" + table + "\n\n"
        "Rows whose host is not Colab (2+ GPUs, an OpenGL desktop, Jetson, CUDA 13.4) are listed here for completeness and are not run.").splitlines(True)}
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if c["cell_type"] == "code" and src.startswith("import json, os, shlex\nONLY"):
            cells[i]["source"] = (
                "import json, glob\n"
                "ONLY = " + repr(",".join(dirs)) + "\n"
                "RUN_TIMEOUT = 300   # seconds per run in round 1; a timed-out program is restarted at 3x, later rounds double it\n"
                "ROUNDS = 2\n"
                "def docs():\n    return [json.load(open(f)) for f in sorted(glob.glob('/content/out/results/*.json'))]\n"
                "def pending():\n    return sum(d['tally'].get('timeout', 0) + d['tally'].get('not-run-yet', 0) for d in docs())\n"
                "# Results are saved after every program: stopping this cell and running it again keeps what finished.\n"
                "for rnd in range(1, ROUNDS + 1):\n"
                "    if docs() and not pending():\n        break\n"
                "    t = RUN_TIMEOUT * 2 ** (rnd - 1)\n"
                "    print(f'--- round {rnd}/{ROUNDS}, run timeout {t}s', flush=True)\n"
                "    get_ipython().system(f'python3 cuda_projects.py bench --manifest manifest.json --root /content/cuda-projects '\n"
                "                         f'--out /content/out --only {ONLY} --run-timeout {t}' + (' --resume' if docs() else ''))\n"
                "for d in docs():\n    print(d['slug'], d['tally'])").splitlines(True)
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mf = sub.add_parser("manifest")
    mf.add_argument("--force", action="store_true")
    mf.set_defaults(fn=cmd_manifest)
    sub.add_parser("notebook").set_defaults(fn=cmd_notebook)
    sub.add_parser("ledger").set_defaults(fn=cmd_ledger)
    sub.add_parser("retest").set_defaults(fn=cmd_retest)
    mg = sub.add_parser("merge")
    mg.add_argument("gpu")
    mg.add_argument("dir", help="the unpacked retest tarball (holds out/ and logs/)")
    mg.set_defaults(fn=cmd_merge)
    rd = sub.add_parser("ready")
    rd.add_argument("--apply", action="store_true", help="remove the moved programs from the clones")
    rd.add_argument("--rebuild", action="store_true")
    rd.add_argument("--add", action="store_true", help="add programs that passed since ready.json was written")
    rd.set_defaults(fn=cmd_ready)
    b = sub.add_parser("bench")
    b.add_argument("--manifest", required=True)
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--only")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--jobs", type=int, default=2)
    b.add_argument("--arch", help="override compute capability, e.g. 75")
    b.add_argument("--build-timeout", type=int, default=600)
    b.add_argument("--run-timeout", type=int, default=60)
    b.add_argument("--paths", default="",
                   help="run only targets whose path contains one of these (comma-separated); results then hold just that subset")
    b.add_argument("--rerun-paths", default="",
                   help="with --resume, also rerun targets whose path contains any of these (comma-separated), even if they passed")
    b.add_argument("--rerun-status", default="",
                   help="with --resume, also rerun targets whose last status is one of these (comma-separated), e.g. build-fail after a harness fix")
    b.add_argument("--resume", action="store_true",
                   help="keep finished results; rerun only targets that timed out or never finished")
    b.set_defaults(fn=cmd_bench)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
