// Metal 4 tensor ops, BF16 -- the single most valuable number in this drop.
//
// This is the question §3 hangs a strategy on.  "The A19 measurement reports no
// BF16 path on the accelerators.  PyTorch trains in BF16."  If BF16 compiles and
// runs here, the block-scaled FP16 GEMM of §3 becomes an optimisation rather
// than a requirement and §2.4's "BF16 is a decision, not an accident" resolves
// the easy way.  If it does not, the fallback is mandatory and the cost is real.
// Either answer is a result; a compiler error text is the finding, not a bug.
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
// Values are 0 or 1 so the 1024-term dot products are small integers.  BF16
// carries 8 mantissa bits, so sums up to 256 are exact and the expected ~68 is
// well inside that; the CPU check is exact, not tolerant.

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

kernel void fill_ab(device bfloat *A [[buffer(0)]],
                    device bfloat *B [[buffer(1)]],
                    uint2 gid [[thread_position_in_grid]])
{
    // A is M x K, row-major: A[m * KDIM + k].  B is K x N: B[k * NDIM + n].
    uint x = gid.x, y = gid.y;
    if (x < uint(KDIM) && y < uint(MDIM)) {
        A[y * uint(KDIM) + x] = ((y + x) % 5u == 0u) ? bfloat(1.0f) : bfloat(0.0f);
    }
    if (x < uint(NDIM) && y < uint(KDIM)) {
        B[y * uint(NDIM) + x] = ((y + x) % 3u == 0u) ? bfloat(1.0f) : bfloat(0.0f);
    }
}

kernel void zero_c(device bfloat *C [[buffer(0)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < uint(MDIM * NDIM)) C[gid] = bfloat(0.0f);
}

kernel void matmul_tensorops(device bfloat *A_buf [[buffer(0)]],
                             device bfloat *B_buf [[buffer(1)]],
                             device bfloat *C_buf [[buffer(2)]],
                             uint2 tgid [[threadgroup_position_in_grid]])
{
    // Metal tensor extents are (inner, outer) = (columns, rows).
    auto A = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(KDIM, MDIM));
    auto B = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(NDIM, KDIM));
    auto C = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
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
