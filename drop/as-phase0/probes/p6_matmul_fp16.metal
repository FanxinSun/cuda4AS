// Metal 4 tensor ops, FP16 -- blueprint unknown #2 on real hardware.
//
// §2.4 puts cuBLAS and cuDNN on "Metal 4 tensor ops on the Neural Accelerators,
// via Metal Performance Primitives"; §3 says the A19 microbenchmark measured
// ~1,024 FP16 FLOP per GPU core per cycle through this path.  Neither has ever
// been run on the machine this drop goes to.  This file is the smallest thing
// that answers "does the path exist here, and at what rate".
//
// The spelling follows the working example at
// github.com/liuliu/example_matmul_metal4 (Sources/matmul/shader.metal),
// checked 2026-09-03: mpp::tensor_ops, matmul2d_descriptor(M, N, K, transpose
// left, transpose right, reduced precision, mode), and static slices so the
// tile shapes are compile-time constants as §1's table says Metal requires.
//
// IMPORTANT, and the reason this is a .metal file rather than a Swift string:
// that example's author notes the MetalPerformancePrimitives header does not
// appear to be available to the runtime compiler, so a JIT
// `makeLibrary(source:)` may fail even where the hardware is capable.  The
// probe tries BOTH -- runtime compile (which is also the nvrtc path of §2.1)
// and this file compiled ahead of time by `xcrun metal` (the AOT path §2.1
// actually chose) -- and reports each separately.  A JIT refusal with an AOT
// success is not a failure; it is a finding about which of our two compilation
// paths can reach the accelerators.
//
// Values are 0 or 1 so the 1024-term dot products are small integers, exact in
// FP16, and checkable on the CPU without a 1 G-MAC reference run.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// #define rather than a program-scope constexpr variable: TILE_M, TILE_N and
// TILE_K are used as TEMPLATE ARGUMENTS (slice<>, matmul2d_descriptor), where
// MSL's address-space rules for program-scope variables are the kind of detail
// that costs a whole round trip if it is wrong.  A macro cannot be wrong.
#define MDIM 1024
#define NDIM 1024
#define KDIM 1024

#define TILE_M 64
#define TILE_N 32
#define TILE_K 16

kernel void fill_ab(device half *A [[buffer(0)]],
                    device half *B [[buffer(1)]],
                    uint2 gid [[thread_position_in_grid]])
{
    // A is M x K, row-major: A[m * KDIM + k].  B is K x N: B[k * NDIM + n].
    uint x = gid.x, y = gid.y;
    if (x < uint(KDIM) && y < uint(MDIM)) {
        A[y * uint(KDIM) + x] = ((y + x) % 5u == 0u) ? half(1.0f) : half(0.0f);
    }
    if (x < uint(NDIM) && y < uint(KDIM)) {
        B[y * uint(NDIM) + x] = ((y + x) % 3u == 0u) ? half(1.0f) : half(0.0f);
    }
}

kernel void zero_c(device half *C [[buffer(0)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < uint(MDIM * NDIM)) C[gid] = half(0.0f);
}

kernel void matmul_tensorops(device half *A_buf [[buffer(0)]],
                             device half *B_buf [[buffer(1)]],
                             device half *C_buf [[buffer(2)]],
                             uint2 tgid [[threadgroup_position_in_grid]])
{
    // Metal tensor extents are (inner, outer) = (columns, rows).
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(KDIM, MDIM));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(NDIM, KDIM));
    auto C = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(NDIM, MDIM));

    constexpr auto desc = matmul2d_descriptor(
        TILE_M, TILE_N, TILE_K,
        /* transpose_left  */ false,
        /* transpose_right */ false,
        /* reduced_precision */ false,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<4>> op;

    for (int k = 0; k < KDIM; k += TILE_K) {
        auto mA = A.slice<TILE_K, TILE_M>(k, int(tgid.y) * TILE_M);
        auto mB = B.slice<TILE_N, TILE_K>(int(tgid.x) * TILE_N, k);
        auto mC = C.slice<TILE_N, TILE_M>(int(tgid.x) * TILE_N,
                                          int(tgid.y) * TILE_M);
        op.run(mA, mB, mC);
    }
}
