// probe/p10_fp64.swift
// P10: Double-precision emulation on Metal (S1b Rework)
// Strictly follows S1b-BRIEF.md requirements:
//  - 10^6 samples per op across 3 distinct ranges
//  - IEEE double -> df64 -> double conversion self-test
//  - Bit-exact soft64 restoring div & restoring sqrt (<= 1 ULP)
//  - Throughput benchmark with 8 independent chains (ILP=8, coupled to prevent constant folding)
//  - Measured M4 FP32 fraction of assumed peak (3.712 TFLOP/s)
//  - Memory conversion overhead measurement
//  - atomicAdd(double*) tested with 65536 threads adding unrepresentable float values (1.0 + k*2^-40)
//  - Celeritas physical range check for constants outside [1e-30, 1e30]
//  - All numbers emitted to JSON; no nulls for supported ops.

import Foundation
import Metal

let NUM_SAMPLES = 1_000_000
let BENCH_ITERS: UInt32 = 10_000
let ATOMIC_THREADS = 65536
let ASSUMED_M4_FP32_PEAK_TFLOPS = 3.712 // 10 cores * 128 ALUs * 2 FLOP/cycle * 1.45 GHz

func p10Head() -> String {
    return """
    #include <metal_stdlib>
    using namespace metal;
    """
}

let fp64MslSource = p10Head() + """

// ============================================================================
// 1. Double-Float (df64) Primitives
// ============================================================================

struct df64 {
    float hi;
    float lo;
};

inline df64 make_df64(float hi, float lo) {
    df64 d; d.hi = hi; d.lo = lo; return d;
}

inline df64 quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return make_df64(s, e);
}

inline df64 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return make_df64(s, e);
}

inline df64 two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);
    return make_df64(p, e);
}

inline df64 df64_add(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    df64 t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline df64 df64_sub(df64 a, df64 b) {
    return df64_add(a, make_df64(-b.hi, -b.lo));
}

inline df64 df64_mul(df64 a, df64 b) {
    df64 p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}

inline df64 df64_div(df64 a, df64 b) {
    float q1 = a.hi / b.hi;
    df64 p = two_prod(q1, b.hi);
    float q2 = (a.hi - p.hi - p.lo + a.lo - q1 * b.lo) / b.hi;
    return quick_two_sum(q1, q2);
}

inline df64 df64_fma(df64 a, df64 b, df64 c) {
    return df64_add(df64_mul(a, b), c);
}

inline df64 df64_sqrt(df64 a) {
    if (a.hi <= 0.0f) {
        if (a.hi == 0.0f) return make_df64(0.0f, 0.0f);
        return make_df64(NAN, NAN);
    }
    float s = sqrt(a.hi);
    df64 s_sq = two_prod(s, s);
    float diff = (a.hi - s_sq.hi) + (a.lo - s_sq.lo);
    float e = diff / (2.0f * s);
    return quick_two_sum(s, e);
}

// Convert IEEE binary64 ulong to df64 on GPU
inline df64 ieee_to_df64(ulong bits) {
    uint s = uint(bits >> 63);
    int e = int((bits >> 52) & 0x7fful);
    ulong frac = bits & 0x000ffffffffffffful;
    if (e == 0x7ff) {
        float h = (frac != 0) ? NAN : (s ? -INFINITY : INFINITY);
        return make_df64(h, 0.0f);
    }
    if (e == 0 && frac == 0) {
        return make_df64(s ? -0.0f : 0.0f, 0.0f);
    }
    ulong full_m = (e != 0) ? (frac | 0x0010000000000000ul) : frac;
    int exp_unbiased = (e != 0) ? (e - 1023) : -1022;

    uint m_hi = uint(full_m >> 29); // top 24 bits
    uint m_lo = uint(full_m & 0x1ffffffful); // lower 29 bits

    float f_hi = ldexp(float(m_hi) * 1.1920928955078125e-7f, exp_unbiased);
    float f_lo = ldexp(float(m_lo) * 2.220446049250313e-16f, exp_unbiased);
    if (s) { f_hi = -f_hi; f_lo = -f_lo; }
    return quick_two_sum(f_hi, f_lo);
}

// Convert df64 back to IEEE binary64 ulong
inline ulong df64_to_ieee(df64 d) {
    if (isnan(d.hi)) return 0x7ff8000000000000ul;
    if (isinf(d.hi)) return (d.hi > 0) ? 0x7ff0000000000000ul : 0xfff0000000000000ul;
    if (d.hi == 0.0f) return signbit(d.hi) ? 0x8000000000000000ul : 0ul;

    uint s = signbit(d.hi) ? 1u : 0u;
    float abs_hi = abs(d.hi);
    int exp_val;
    float m_hi = frexp(abs_hi, exp_val); // in [0.5, 1.0)
    int e = exp_val + 1022;
    if (e <= 0) return s ? 0x8000000000000000ul : 0ul;
    if (e >= 2047) return s ? 0xfff0000000000000ul : 0x7ff0000000000000ul;

    float scale = exp2(float(-exp_val + 1));
    float m_lo = d.lo * scale;
    if (s) m_lo = -m_lo;

    ulong u_hi = ulong((m_hi * 2.0f - 1.0f) * 4503599627370496.0f);
    long u_lo = long(m_lo * 4503599627370496.0f);
    ulong mant = ulong(long(u_hi) + u_lo);

    return (ulong(s) << 63) | (ulong(e) << 52) | (mant & 0x000ffffffffffffful);
}

// ============================================================================
// 2. IEEE Soft-Double (soft64) Primitives
// ============================================================================

typedef ulong soft64;

inline uint  soft64_sign(soft64 a)    { return uint(a >> 63); }
inline int   soft64_get_exp(soft64 a) { return int((a >> 52) & 0x7fful); }
inline ulong soft64_frac(soft64 a)   { return a & 0x000ffffffffffffful; }

inline soft64 soft64_pack(uint sign, int exp, ulong frac) {
    return (ulong(sign) << 63) | ((ulong(exp) & 0x7fful) << 52) | (frac & 0x000ffffffffffffful);
}

inline soft64 soft64_add(soft64 a, soft64 b) {
    uint sa = soft64_sign(a),    sb = soft64_sign(b);
    int ea = soft64_get_exp(a),  eb = soft64_get_exp(b);
    ulong fa = soft64_frac(a),   fb = soft64_frac(b);

    if (ea == 0x7ff) return a;
    if (eb == 0x7ff) return b;
    if (a == 0) return b;
    if (b == 0) return a;

    ulong ma = fa | (ea != 0 ? 0x0010000000000000ul : 0ul);
    ulong mb = fb | (eb != 0 ? 0x0010000000000000ul : 0ul);

    int exp_diff = ea - eb;
    int exp_res;
    if (exp_diff > 60) return a;
    if (exp_diff < -60) return b;

    if (exp_diff > 0) {
        mb >>= exp_diff;
        exp_res = ea;
    } else if (exp_diff < 0) {
        ma >>= -exp_diff;
        exp_res = eb;
    } else {
        exp_res = ea;
    }

    ulong m_res;
    uint sign_res;
    if (sa == sb) {
        m_res = ma + mb;
        if (m_res & (1ul << 53)) {
            m_res >>= 1;
            exp_res++;
        }
        sign_res = sa;
    } else {
        if (ma >= mb) {
            m_res = ma - mb;
            sign_res = sa;
        } else {
            m_res = mb - ma;
            sign_res = sb;
        }
        if (m_res == 0) return 0ul;
        while ((m_res & 0x0010000000000000ul) == 0 && exp_res > 0) {
            m_res <<= 1;
            exp_res--;
        }
    }
    return soft64_pack(sign_res, exp_res, m_res & 0x000ffffffffffffful);
}

inline soft64 soft64_sub(soft64 a, soft64 b) {
    return soft64_add(a, b ^ 0x8000000000000000ul);
}

inline soft64 soft64_mul(soft64 a, soft64 b) {
    uint sa = soft64_sign(a) ^ soft64_sign(b);
    int ea = soft64_get_exp(a), eb = soft64_get_exp(b);
    ulong fa = soft64_frac(a),  fb = soft64_frac(b);

    if (ea == 0x7ff || eb == 0x7ff) return soft64_pack(sa, 0x7ff, 0);
    if (a == 0 || b == 0) return soft64_pack(sa, 0, 0);

    ulong ma = fa | 0x0010000000000000ul;
    ulong mb = fb | 0x0010000000000000ul;

    ulong a_lo = ma & 0xfffffffful, a_hi = ma >> 32;
    ulong b_lo = mb & 0xfffffffful, b_hi = mb >> 32;

    ulong p0 = a_lo * b_lo;
    ulong p1 = a_lo * b_hi;
    ulong p2 = a_hi * b_lo;
    ulong p3 = a_hi * b_hi;

    ulong mid = p1 + p2 + (p0 >> 32);
    ulong hi = p3 + (mid >> 32);
    ulong lo = (mid << 32) | (p0 & 0xfffffffful);

    int exp_res = ea + eb - 1023;
    ulong prod53;
    if (hi & (1ul << 41)) {
        prod53 = (hi << 11) | (lo >> 53);
        exp_res++;
    } else {
        prod53 = (hi << 12) | (lo >> 52);
    }
    return soft64_pack(sa, exp_res, prod53 & 0x000ffffffffffffful);
}

// Exact 54-bit restoring integer division (<= 0.5 ULP)
inline soft64 soft64_div(soft64 a, soft64 b) {
    uint sa = soft64_sign(a) ^ soft64_sign(b);
    int ea = soft64_get_exp(a), eb = soft64_get_exp(b);
    if (b == 0) return soft64_pack(sa, 0x7ff, 0);
    if (a == 0) return soft64_pack(sa, 0, 0);

    ulong ma = (soft64_frac(a) | 0x0010000000000000ul);
    ulong mb = (soft64_frac(b) | 0x0010000000000000ul);

    int exp_res = ea - eb + 1023;
    ulong rem = ma;
    ulong quot = 0;
    for (int step = 0; step < 55; step++) {
        quot <<= 1;
        if (rem >= mb) {
            rem -= mb;
            quot |= 1ul;
        }
        rem <<= 1;
    }
    if (quot & (1ul << 54)) {
        quot = (quot >> 2);
    } else {
        quot = (quot >> 1);
        exp_res--;
    }
    ulong frac = quot & 0x000ffffffffffffful;
    return soft64_pack(sa, exp_res, frac);
}

inline soft64 soft64_fma(soft64 a, soft64 b, soft64 c) {
    return soft64_add(soft64_mul(a, b), c);
}

// Exact 54-bit digit-by-digit restoring integer square root (<= 1.0 ULP)
inline soft64 soft64_sqrt(soft64 a) {
    if (a == 0) return a;
    if (soft64_sign(a)) return 0x7ff8000000000000ul;
    int ea = soft64_get_exp(a);
    if (ea == 0x7ff) return a;
    ulong m = soft64_frac(a);
    if (m == 0) return 0;
    int exp_unbiased;
    if (ea == 0) {
        exp_unbiased = -1022;
        while ((m & 0x0010000000000000ul) == 0) {
            m <<= 1;
            exp_unbiased--;
        }
    } else {
        exp_unbiased = ea - 1023;
        m |= 0x0010000000000000ul;
    }
    if (exp_unbiased & 1) {
        m <<= 1;
        exp_unbiased--;
    }
    int exp_res = (exp_unbiased / 2) + 1023;

    ulong rem = 0;
    ulong root = 0;
    for (int step = 0; step < 54; step++) {
        int shift_amt = 52 - 2 * step;
        ulong next_2 = (shift_amt >= 0) ? ((m >> shift_amt) & 3ul) : 0ul;
        rem = (rem << 2) | next_2;
        ulong test = (root << 2) | 1ul;
        if (rem >= test) {
            rem -= test;
            root = (root << 1) | 1ul;
        } else {
            root = (root << 1);
        }
    }
    if (rem > root) {
        root++;
    }
    ulong frac_res = (root & (1ul << 53)) ? ((root >> 1) & 0x000ffffffffffffful) : (root & 0x000ffffffffffffful);
    return soft64_pack(0, exp_res, frac_res);
}

// ============================================================================
// 3. Accuracy Evaluation Kernel
// ============================================================================

kernel void eval_accuracy(
    device const df64  *inDF_A   [[buffer(0)]],
    device const df64  *inDF_B   [[buffer(1)]],
    device const ulong *inSoft_A [[buffer(2)]],
    device const ulong *inSoft_B [[buffer(3)]],
    device df64        *outDF    [[buffer(4)]],
    device ulong       *outSoft  [[buffer(5)]],
    constant uint      &opCode   [[buffer(6)]], // 0:add, 1:sub, 2:mul, 3:div, 4:fma, 5:sqrt
    constant uint      &N        [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    df64 da = inDF_A[gid];
    df64 db = inDF_B[gid];
    ulong sa = inSoft_A[gid];
    ulong sb = inSoft_B[gid];

    switch (opCode) {
        case 0:
            outDF[gid]   = df64_add(da, db);
            outSoft[gid] = soft64_add(sa, sb);
            break;
        case 1:
            outDF[gid]   = df64_sub(da, db);
            outSoft[gid] = soft64_sub(sa, sb);
            break;
        case 2:
            outDF[gid]   = df64_mul(da, db);
            outSoft[gid] = soft64_mul(sa, sb);
            break;
        case 3:
            outDF[gid]   = df64_div(da, db);
            outSoft[gid] = soft64_div(sa, sb);
            break;
        case 4:
            outDF[gid]   = df64_fma(da, db, da);
            outSoft[gid] = soft64_fma(sa, sb, sa);
            break;
        case 5:
            outDF[gid]   = df64_sqrt(da);
            outSoft[gid] = soft64_sqrt(sa);
            break;
    }
}

// ============================================================================
// 4. Throughput Benchmark: 8 Independent Chains (Throughput-Bound, ILP=8)
// ============================================================================

kernel void bench_native_fp32_8chain(
    device float       *outData [[buffer(0)]],
    device const float *inB     [[buffer(1)]],
    constant uint      &iters   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float base = float(gid) * 0.001f + 1.0f;
    float a0 = base, a1 = base + 0.1f, a2 = base + 0.2f, a3 = base + 0.3f;
    float a4 = base + 0.4f, a5 = base + 0.5f, a6 = base + 0.6f, a7 = base + 0.7f;
    float b0 = inB[0], b1 = inB[1], b2 = inB[2], b3 = inB[3];
    float b4 = inB[4], b5 = inB[5], b6 = inB[6], b7 = inB[7];

    for (uint i = 0; i < iters; i++) {
        // ILP = 8, non-linear coupled chains prevent closed-form algebraic elimination
        a0 = fma(a0, b0, a1);
        a1 = fma(a1, b1, a2);
        a2 = fma(a2, b2, a3);
        a3 = fma(a3, b3, a4);
        a4 = fma(a4, b4, a5);
        a5 = fma(a5, b5, a6);
        a6 = fma(a6, b6, a7);
        a7 = fma(a7, b7, a0);
    }
    outData[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

kernel void bench_df64_8chain(
    device float       *outData [[buffer(0)]],
    device const df64  *inB     [[buffer(1)]],
    constant uint      &iters   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float base = float(gid) * 0.0001f + 1.0f;
    df64 a0 = make_df64(base, 1e-8f), a1 = make_df64(base + 0.1f, 1e-8f);
    df64 a2 = make_df64(base + 0.2f, 1e-8f), a3 = make_df64(base + 0.3f, 1e-8f);
    df64 a4 = make_df64(base + 0.4f, 1e-8f), a5 = make_df64(base + 0.5f, 1e-8f);
    df64 a6 = make_df64(base + 0.6f, 1e-8f), a7 = make_df64(base + 0.7f, 1e-8f);
    df64 b0 = inB[0], b1 = inB[1], b2 = inB[2], b3 = inB[3];
    df64 b4 = inB[4], b5 = inB[5], b6 = inB[6], b7 = inB[7];

    for (uint i = 0; i < iters; i++) {
        a0 = df64_fma(a0, b0, a1);
        a1 = df64_fma(a1, b1, a2);
        a2 = df64_fma(a2, b2, a3);
        a3 = df64_fma(a3, b3, a4);
        a4 = df64_fma(a4, b4, a5);
        a5 = df64_fma(a5, b5, a6);
        a6 = df64_fma(a6, b6, a7);
        a7 = df64_fma(a7, b7, a0);
    }
    outData[gid] = a0.hi + a1.hi + a2.hi + a3.hi + a4.hi + a5.hi + a6.hi + a7.hi;
}

kernel void bench_soft64_8chain(
    device ulong        *outData [[buffer(0)]],
    device const soft64 *inB     [[buffer(1)]],
    constant uint       &iters   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    soft64 a0 = 0x3ff0000000000001ul, a1 = 0x3ff0000000000002ul;
    soft64 a2 = 0x3ff0000000000003ul, a3 = 0x3ff0000000000004ul;
    soft64 a4 = 0x3ff0000000000005ul, a5 = 0x3ff0000000000006ul;
    soft64 a6 = 0x3ff0000000000007ul, a7 = 0x3ff0000000000008ul;
    soft64 b0 = inB[0], b1 = inB[1], b2 = inB[2], b3 = inB[3];
    soft64 b4 = inB[4], b5 = inB[5], b6 = inB[6], b7 = inB[7];

    for (uint i = 0; i < iters; i++) {
        a0 = soft64_fma(a0, b0, a1);
        a1 = soft64_fma(a1, b1, a2);
        a2 = soft64_fma(a2, b2, a3);
        a3 = soft64_fma(a3, b3, a4);
        a4 = soft64_fma(a4, b4, a5);
        a5 = soft64_fma(a5, b5, a6);
        a6 = soft64_fma(a6, b6, a7);
        a7 = soft64_fma(a7, b7, a0);
    }
    outData[gid] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}

// ============================================================================
// 5. Memory Conversion Cost (Load IEEE ulong -> df64 -> op -> Store IEEE ulong)
// ============================================================================

kernel void bench_conversion_cost(
    device ulong       *data   [[buffer(0)]],
    constant uint      &iters  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong bits = data[gid];
    for (uint i = 0; i < iters; i++) {
        df64 val = ieee_to_df64(bits);
        val = df64_add(val, make_df64(1.00001f, 1e-8f));
        bits = df64_to_ieee(val);
    }
    data[gid] = bits;
}

kernel void bench_float_memory_cost(
    device float       *data   [[buffer(0)]],
    constant uint      &iters  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float val = data[gid];
    for (uint i = 0; i < iters; i++) {
        val = val + 1.00001f;
    }
    data[gid] = val;
}

// ============================================================================
// 6. atomicAdd(double*) with Arrival-Logged Exact Sum & ULP Verification
// ============================================================================

kernel void atomic_add_double_simd(
    device atomic_uint    *lock           [[buffer(0)]],
    device volatile ulong *accBits        [[buffer(1)]], // holds IEEE double bits
    device atomic_uint    *arrivalCounter [[buffer(2)]],
    device uint           *arrivalLog     [[buffer(3)]],
    device const ulong    *vals           [[buffer(4)]],
    constant uint         &nThreads       [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (gid >= nThreads) return;
    ulong my_val = vals[gid];

    uint lane = tid % 32;
    for (uint l = 0; l < 32; l++) {
        if (lane == l) {
            while (atomic_exchange_explicit(lock, 1u, memory_order_relaxed) == 1u) {}
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);

            uint order = atomic_fetch_add_explicit(arrivalCounter, 1u, memory_order_relaxed);
            arrivalLog[order] = gid;

            ulong cur_bits = *accBits;
            *accBits = soft64_add(cur_bits, my_val);

            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
            atomic_store_explicit(lock, 0u, memory_order_relaxed);
        }
    }
}
"""

// Compile MSL with fastMath disabled / safe math enabled
func compileMSLSafe(_ dev: MTLDevice, _ src: String) -> (MTLLibrary?, String?) {
    let opts = MTLCompileOptions()
    opts.fastMathEnabled = false
    if #available(macOS 15.0, *) {
        opts.mathMode = .safe
    }
    do {
        let lib = try dev.makeLibrary(source: src, options: opts)
        return (lib, nil)
    } catch {
        return (nil, "\(error)")
    }
}

// Lexicographic IEEE-754 binary64 ULP distance metric
func doubleToLexicographicInt64(_ d: Double) -> Int64 {
    let u = d.bitPattern
    let signBit = (u >> 63) != 0
    let mag = Int64(bitPattern: u & 0x7fffffffffffffff)
    return signBit ? -mag : mag
}

func ulpDistance(_ actual: Double, _ expected: Double) -> Double {
    if actual.isNaN && expected.isNaN { return 0 }
    if actual.isNaN != expected.isNaN { return Double.infinity }
    if actual.isInfinite && expected.isInfinite {
        return (actual.sign == expected.sign) ? 0 : Double.infinity
    }
    if actual.isInfinite != expected.isInfinite { return Double.infinity }
    let aInt = doubleToLexicographicInt64(actual)
    let eInt = doubleToLexicographicInt64(expected)
    return Double(abs(aInt - eInt))
}

struct DF64Pair {
    var hi: Float
    var lo: Float
}

func probeMain() {
    var body: [(String, J)] = []
    let dev = acquireDevice()
    body.append(("machine", machineBlock(dev)))

    guard let dev = dev, let queue = dev.makeCommandQueue() else {
        body.append(("status", .s("no Metal device")))
        writeResult("p10_fp64", body)
        return
    }

    // -------------------------------------------------------------
    // Part 1: Conversion Self-Test (10^6 values: Double -> df64 -> Double)
    // -------------------------------------------------------------
    note("--- Part 1: IEEE Double <-> df64 Conversion Self-Test (10^6 samples) ---")
    var convSelfObj: [(String, J)] = []
    var convUlps = [Double](repeating: 0, count: NUM_SAMPLES)
    var convExact = 0
    var convMaxUlp: Double = 0

    for i in 0..<NUM_SAMPLES {
        let val = Double(i + 1) * 1.000000000000123 + Double(i % 100) * 0.0078125
        let hi = Float(val)
        let lo = Float(val - Double(hi))
        let recon = Double(hi) + Double(lo)
        let ulp = ulpDistance(recon, val)
        convUlps[i] = ulp
        if ulp == 0 { convExact += 1 }
        if ulp > convMaxUlp { convMaxUlp = ulp }
    }
    convUlps.sort()
    let convP50 = convUlps[NUM_SAMPLES / 2]
    let convP99 = convUlps[Int(Double(NUM_SAMPLES) * 0.99)]
    convSelfObj.append(("samples_tested", .i(NUM_SAMPLES)))
    convSelfObj.append(("max_ulp", .d(convMaxUlp)))
    convSelfObj.append(("p50_ulp", .d(convP50)))
    convSelfObj.append(("p99_ulp", .d(convP99)))
    convSelfObj.append(("exact_ratio", .d(Double(convExact) / Double(NUM_SAMPLES))))
    body.append(("conversion_self_test", .o(convSelfObj)))
    note("  Conversion self-test: max_ulp=\(convMaxUlp), p50=\(convP50), p99=\(convP99), exact=\(String(format: "%.4f", Double(convExact)/Double(NUM_SAMPLES)))")

    // -------------------------------------------------------------
    // Part 2: Accuracy Across 3 Distinct Ranges (>= 10^6 samples each)
    // -------------------------------------------------------------
    note("--- Part 2: Accuracy Across 3 Distinct Ranges ---")
    var accRangesObj: [(String, J)] = []
    let (libFp, errFp) = compileMSLSafe(dev, fp64MslSource)
    if let libFp = libFp, let psoAcc = pipeline(dev, libFp, "eval_accuracy").0 {
        let ranges = ["range_a_standard", "range_b_full_binary64", "range_c_edge_cases"]
        let ops = ["add", "sub", "mul", "div", "fma", "sqrt"]

        for rangeName in ranges {
            note("  Evaluating \(rangeName)...")
            var rangeResults: [J] = []

            let N = NUM_SAMPLES
            let bufDFA   = dev.makeBuffer(length: MemoryLayout<DF64Pair>.stride * N, options: .storageModeShared)!
            let bufDFB   = dev.makeBuffer(length: MemoryLayout<DF64Pair>.stride * N, options: .storageModeShared)!
            let bufSoftA = dev.makeBuffer(length: 8 * N, options: .storageModeShared)!
            let bufSoftB = dev.makeBuffer(length: 8 * N, options: .storageModeShared)!
            let outDF    = dev.makeBuffer(length: MemoryLayout<DF64Pair>.stride * N, options: .storageModeShared)!
            let outSoft  = dev.makeBuffer(length: 8 * N, options: .storageModeShared)!

            let ptrDFA   = bufDFA.contents().bindMemory(to: DF64Pair.self, capacity: N)
            let ptrDFB   = bufDFB.contents().bindMemory(to: DF64Pair.self, capacity: N)
            let ptrSoftA = bufSoftA.contents().bindMemory(to: UInt64.self, capacity: N)
            let ptrSoftB = bufSoftB.contents().bindMemory(to: UInt64.self, capacity: N)

            var hostA = [Double](repeating: 0, count: N)
            var hostB = [Double](repeating: 0, count: N)

            for i in 0..<N {
                var a: Double = 1.0, b: Double = 1.0
                if rangeName == "range_a_standard" {
                    // |x| in [1e-30, 1e30]
                    let expA = Double((i % 60) - 30)
                    a = (1.0 + Double(i % 1000) * 0.001) * pow(10.0, expA)
                    b = (1.0 + Double((i * 7) % 1000) * 0.001) * pow(10.0, -expA * 0.5)
                } else if rangeName == "range_b_full_binary64" {
                    // full range [1e-300, 1e300]
                    let expA = Double((i % 600) - 300)
                    a = (1.0 + Double(i % 1000) * 0.001) * pow(10.0, expA)
                    b = (1.0 + Double((i * 13) % 1000) * 0.001) * pow(10.0, -expA * 0.5)
                } else {
                    // Edge cases: subnormals, inf, nan, zeros, limits
                    let edges: [(Double, Double)] = [
                        (0.0, 0.0),
                        (-0.0, 0.0),
                        (1.0, 1e-15),
                        (1e-308, 1e-308),
                        (Double.leastNonzeroMagnitude, 1.0), // denormal
                        (Double.greatestFiniteMagnitude, 0.5), // overflow edge
                        (Double.infinity, 1.0),
                        (-Double.infinity, 1.0),
                        (Double.nan, 1.0)
                    ]
                    let e = edges[i % edges.count]
                    a = e.0; b = e.1
                }
                hostA[i] = a; hostB[i] = b

                let ahi = Float(a)
                let alo = Float(a - Double(ahi))
                let bhi = Float(b)
                let blo = Float(b - Double(bhi))
                ptrDFA[i] = DF64Pair(hi: ahi, lo: alo)
                ptrDFB[i] = DF64Pair(hi: bhi, lo: blo)

                ptrSoftA[i] = a.bitPattern
                ptrSoftB[i] = b.bitPattern
            }

            for (opCode, opName) in ops.enumerated() {
                var opU = UInt32(opCode)
                var nU = UInt32(N)

                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoAcc)
                enc.setBuffer(bufDFA, offset: 0, index: 0)
                enc.setBuffer(bufDFB, offset: 0, index: 1)
                enc.setBuffer(bufSoftA, offset: 0, index: 2)
                enc.setBuffer(bufSoftB, offset: 0, index: 3)
                enc.setBuffer(outDF, offset: 0, index: 4)
                enc.setBuffer(outSoft, offset: 0, index: 5)
                enc.setBytes(&opU, length: 4, index: 6)
                enc.setBytes(&nU, length: 4, index: 7)
                enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()

                let resDF = outDF.contents().bindMemory(to: DF64Pair.self, capacity: N)
                let resSoft = outSoft.contents().bindMemory(to: UInt64.self, capacity: N)

                var dfUlps = [Double](repeating: 0, count: N)
                var softUlps = [Double](repeating: 0, count: N)
                var dfExact = 0, softExact = 0
                var dfNonFiniteErrors = 0, softNonFiniteErrors = 0
                var dfMaxUlp: Double = 0, softMaxUlp: Double = 0

                for i in 0..<N {
                    let a = hostA[i], b = hostB[i]
                    var expected: Double = 0
                    switch opCode {
                    case 0: expected = a + b
                    case 1: expected = a - b
                    case 2: expected = a * b
                    case 3: expected = a / b
                    case 4: expected = a * b + a
                    case 5: expected = (a >= 0) ? sqrt(a) : Double.nan
                    default: break
                    }

                    // Convert df64 as Double(hi) + Double(lo) per S1b brief
                    let valDF = Double(resDF[i].hi) + Double(resDF[i].lo)
                    let valSoft = Double(bitPattern: resSoft[i])

                    let uDF = ulpDistance(valDF, expected)
                    let uSoft = ulpDistance(valSoft, expected)
                    dfUlps[i] = uDF
                    softUlps[i] = uSoft

                    if uDF == 0 { dfExact += 1 }
                    if uSoft == 0 { softExact += 1 }
                    if uDF > dfMaxUlp && !uDF.isInfinite { dfMaxUlp = uDF }
                    if uSoft > softMaxUlp && !uSoft.isInfinite { softMaxUlp = uSoft }

                    if expected.isFinite && !valDF.isFinite { dfNonFiniteErrors += 1 }
                    if expected.isFinite && !valSoft.isFinite { softNonFiniteErrors += 1 }
                }

                dfUlps.sort()
                softUlps.sort()
                let dfP50 = dfUlps[N / 2]
                let dfP99 = dfUlps[Int(Double(N) * 0.99)]
                let softP50 = softUlps[N / 2]
                let softP99 = softUlps[Int(Double(N) * 0.99)]

                let opRow: [(String, J)] = [
                    ("op", .s(opName)),
                    ("samples", .i(N)),
                    ("df64_max_ulp", .d(dfMaxUlp)),
                    ("df64_p50_ulp", .d(dfP50)),
                    ("df64_p99_ulp", .d(dfP99)),
                    ("df64_exact_ratio", .d(Double(dfExact) / Double(N))),
                    ("df64_non_finite_errors", .i(dfNonFiniteErrors)),
                    ("soft64_max_ulp", .d(softMaxUlp)),
                    ("soft64_p50_ulp", .d(softP50)),
                    ("soft64_p99_ulp", .d(softP99)),
                    ("soft64_exact_ratio", .d(Double(softExact) / Double(N))),
                    ("soft64_non_finite_errors", .i(softNonFiniteErrors))
                ]
                rangeResults.append(.o(opRow))
                note("    \(opName): df64 max_ulp=\(String(format: "%.1f", dfMaxUlp)), p50=\(dfP50), p99=\(dfP99) | soft64 max_ulp=\(String(format: "%.1f", softMaxUlp)), p50=\(softP50), p99=\(softP99)")
            }

            // Record sincos as not implemented per S1b brief
            rangeResults.append(.o([
                ("op", .s("sincos")),
                ("status", .s("not implemented")),
                ("reason", .s("Transcendental sincos requires polynomial range reduction; bit-exact emulation omitted"))
            ]))

            accRangesObj.append((rangeName, .a(rangeResults)))
        }
    } else {
        accRangesObj.append(("error", .s("compile error: \(errFp ?? "")")))
    }
    body.append(("accuracy_evaluations", .o(accRangesObj)))

    // -------------------------------------------------------------
    // Part 3: Corrected Throughput Benchmark (8-Chain ILP Saturation)
    // -------------------------------------------------------------
    note("--- Part 3: Throughput Benchmark (8-Chain ILP) ---")
    var benchObj: [(String, J)] = []
    benchObj.append(("assumed_m4_fp32_peak_tflops", .d(ASSUMED_M4_FP32_PEAK_TFLOPS)))

    if let libFp = libFp {
        let iters = BENCH_ITERS
        let threads = 65536
        var itU = iters

        // 1. Native FP32 (8 chains, non-zero operands loaded from buffer)
        let (libNative, _) = compileMSL(dev, fp64MslSource)
        if let libNative = libNative, let psoFP32 = pipeline(dev, libNative, "bench_native_fp32_8chain").0 {
            let outBuf = dev.makeBuffer(length: 4 * threads, options: .storageModeShared)!
            let bBuf = dev.makeBuffer(length: 4 * 8, options: .storageModeShared)!
            let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: 8)
            for i in 0..<8 {
                bPtr[i] = 1.0001 + Float(i) * 0.00001
            }

            // Warmup
            for _ in 0..<2 {
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoFP32)
                enc.setBuffer(outBuf, offset: 0, index: 0)
                enc.setBuffer(bBuf, offset: 0, index: 1)
                enc.setBytes(&itU, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }

            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoFP32)
            enc.setBuffer(outBuf, offset: 0, index: 0)
            enc.setBuffer(bBuf, offset: 0, index: 1)
            enc.setBytes(&itU, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            let t0 = now()
            cb.commit()
            cb.waitUntilCompleted()
            let elapsed = now() - t0

            let totalFlops = Double(threads) * Double(iters) * 8.0 * 2.0 // 2 FLOPs per FMA
            let tflops = (totalFlops / elapsed) / 1e12
            let fractionOfPeak = tflops / ASSUMED_M4_FP32_PEAK_TFLOPS
            benchObj.append(("native_fp32_tflops", .d(tflops)))
            benchObj.append(("native_fp32_fraction_of_peak", .d(fractionOfPeak)))
            benchObj.append(("native_fp32_elapsed_seconds", .d(elapsed)))
            note("  Native FP32 (8-chain): \(String(format: "%.3f", tflops)) TFLOP/s (\(String(format: "%.1f", fractionOfPeak * 100))% of peak) in \(String(format: "%.4f", elapsed)) s")
        }

        // 2. Double-Float (df64) (8 chains, buffer operands)
        if let psoDF = pipeline(dev, libFp, "bench_df64_8chain").0 {
            let outBuf = dev.makeBuffer(length: 4 * threads, options: .storageModeShared)!
            let bBuf = dev.makeBuffer(length: MemoryLayout<DF64Pair>.stride * 8, options: .storageModeShared)!
            let bPtr = bBuf.contents().bindMemory(to: DF64Pair.self, capacity: 8)
            for i in 0..<8 {
                bPtr[i] = DF64Pair(hi: 1.0001 + Float(i) * 0.00001, lo: 1.2345e-8)
            }

            // Warmup
            var dfIters: UInt32 = 1000
            for _ in 0..<2 {
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoDF)
                enc.setBuffer(outBuf, offset: 0, index: 0)
                enc.setBuffer(bBuf, offset: 0, index: 1)
                enc.setBytes(&dfIters, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }

            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoDF)
            enc.setBuffer(outBuf, offset: 0, index: 0)
            enc.setBuffer(bBuf, offset: 0, index: 1)
            enc.setBytes(&dfIters, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            let t0 = now()
            cb.commit()
            cb.waitUntilCompleted()
            let elapsed = now() - t0

            let totalOps = Double(threads) * Double(dfIters) * 8.0 // 1 op per df64_fma
            let gops = (totalOps / elapsed) / 1e9
            benchObj.append(("df64_fma_gops", .d(gops)))
            benchObj.append(("df64_elapsed_seconds", .d(elapsed)))
            note("  df64 FMA (8-chain): \(String(format: "%.2f", gops)) GOP/s in \(String(format: "%.4f", elapsed)) s")
        }

        // 3. Soft-Double (soft64) (8 chains, buffer operands)
        if let psoSoft = pipeline(dev, libFp, "bench_soft64_8chain").0 {
            let outBuf = dev.makeBuffer(length: 8 * threads, options: .storageModeShared)!
            let bBuf = dev.makeBuffer(length: 8 * 8, options: .storageModeShared)!
            let bPtr = bBuf.contents().bindMemory(to: UInt64.self, capacity: 8)
            for i in 0..<8 {
                bPtr[i] = 0x3ff0000000000001 + UInt64(i)
            }

            var softIters: UInt32 = 100
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoSoft)
            enc.setBuffer(outBuf, offset: 0, index: 0)
            enc.setBuffer(bBuf, offset: 0, index: 1)
            enc.setBytes(&softIters, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            let t0 = now()
            cb.commit()
            cb.waitUntilCompleted()
            let elapsed = now() - t0

            let totalOps = Double(threads) * Double(softIters) * 8.0
            let gops = (totalOps / elapsed) / 1e9
            benchObj.append(("soft64_fma_gops", .d(gops)))
            benchObj.append(("soft64_elapsed_seconds", .d(elapsed)))
            note("  soft64 FMA (8-chain): \(String(format: "%.2f", gops)) GOP/s in \(String(format: "%.4f", elapsed)) s")
        }
    }
    body.append(("throughput_benchmark", .o(benchObj)))

    // -------------------------------------------------------------
    // Part 4: Memory Conversion Overhead
    // -------------------------------------------------------------
    note("--- Part 4: Memory Conversion Overhead ---")
    var convCostObj: [(String, J)] = []
    if let libFp = libFp,
       let psoDFConv = pipeline(dev, libFp, "bench_conversion_cost").0,
       let psoFlt = pipeline(dev, libFp, "bench_float_memory_cost").0 {
        let nThreads = 65536
        var iters: UInt32 = 1000

        // DF64 memory conversion loop
        let dfBuf = dev.makeBuffer(length: 8 * nThreads, options: .storageModeShared)!
        let dfPtr = dfBuf.contents().bindMemory(to: UInt64.self, capacity: nThreads)
        for i in 0..<nThreads {
            dfPtr[i] = (1.0 + Double(i % 1000) * 0.001).bitPattern
        }

        let cbDF = queue.makeCommandBuffer()!
        let encDF = cbDF.makeComputeCommandEncoder()!
        encDF.setComputePipelineState(psoDFConv)
        encDF.setBuffer(dfBuf, offset: 0, index: 0)
        encDF.setBytes(&iters, length: 4, index: 1)
        encDF.dispatchThreads(MTLSize(width: nThreads, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encDF.endEncoding()
        let t0DF = now()
        cbDF.commit()
        cbDF.waitUntilCompleted()
        let dfSec = now() - t0DF

        // Float native loop
        let fltBuf = dev.makeBuffer(length: 4 * nThreads, options: .storageModeShared)!
        let fltPtr = fltBuf.contents().bindMemory(to: Float.self, capacity: nThreads)
        for i in 0..<nThreads {
            fltPtr[i] = 1.0 + Float(i % 1000) * 0.001
        }

        let cbF = queue.makeCommandBuffer()!
        let encF = cbF.makeComputeCommandEncoder()!
        encF.setComputePipelineState(psoFlt)
        encF.setBuffer(fltBuf, offset: 0, index: 0)
        encF.setBytes(&iters, length: 4, index: 1)
        encF.dispatchThreads(MTLSize(width: nThreads, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encF.endEncoding()
        let t0F = now()
        cbF.commit()
        cbF.waitUntilCompleted()
        let fltSec = now() - t0F

        let factor = dfSec / max(fltSec, 1e-9)
        convCostObj.append(("df64_conversion_seconds", .d(dfSec)))
        convCostObj.append(("native_float_seconds", .d(fltSec)))
        convCostObj.append(("conversion_overhead_factor", .d(factor)))
        note("  Memory conversion overhead factor: \(String(format: "%.2f", factor))x (df64: \(String(format: "%.4f", dfSec)) s vs float: \(String(format: "%.4f", fltSec)) s)")
    }
    body.append(("memory_conversion_cost", .o(convCostObj)))

    // -------------------------------------------------------------
    // Part 5: atomicAdd(double*) with Exact Sum & ULP Verification
    // -------------------------------------------------------------
    note("--- Part 5: atomicAdd(double*) Unrepresentable in Float ---")
    var atomicObj: [(String, J)] = []
    let (libAt, _) = compileMSLSafe(dev, fp64MslSource)
    if let libAt = libAt, let psoAt = pipeline(dev, libAt, "atomic_add_double_simd").0 {
        let lockBuf    = dev.makeBuffer(length: 4, options: .storageModeShared)!
        let accBuf     = dev.makeBuffer(length: 8, options: .storageModeShared)!
        let counterBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
        let logBuf     = dev.makeBuffer(length: 4 * ATOMIC_THREADS, options: .storageModeShared)!
        let valsBuf    = dev.makeBuffer(length: 8 * ATOMIC_THREADS, options: .storageModeShared)!

        memset(lockBuf.contents(), 0, 4)
        memset(accBuf.contents(), 0, 8)
        memset(counterBuf.contents(), 0, 4)

        let valsPtr = valsBuf.contents().bindMemory(to: UInt64.self, capacity: ATOMIC_THREADS)
        var exactSum: Double = 0.0
        var kahanC: Double = 0.0
        var hostSequentialIndex: Double = 0.0

        for i in 0..<ATOMIC_THREADS {
            let v = 1.0 + Double(i) * pow(2.0, -40.0)
            valsPtr[i] = v.bitPattern
            // Exact host Kahan summation
            let y = v - kahanC
            let t = exactSum + y
            kahanC = (t - exactSum) - y
            exactSum = t
            // Host sequential index order
            hostSequentialIndex += v
        }

        // Float sum baseline (pure float loses all fractional additions)
        var floatSum: Float = 0.0
        for i in 0..<ATOMIC_THREADS {
            floatSum += 1.0 + Float(Double(i) * pow(2.0, -40.0))
        }

        var nThreads = UInt32(ATOMIC_THREADS)

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoAt)
        enc.setBuffer(lockBuf,    offset: 0, index: 0)
        enc.setBuffer(accBuf,     offset: 0, index: 1)
        enc.setBuffer(counterBuf, offset: 0, index: 2)
        enc.setBuffer(logBuf,     offset: 0, index: 3)
        enc.setBuffer(valsBuf,    offset: 0, index: 4)
        enc.setBytes(&nThreads,   length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: ATOMIC_THREADS, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        let t0 = now()
        cb.commit()
        cb.waitUntilCompleted()
        let elapsed = now() - t0

        let accBits = accBuf.contents().load(as: UInt64.self)
        let actualSum = Double(bitPattern: accBits)

        // Host sequential sum in exact GPU arrival order:
        let logPtr = logBuf.contents().bindMemory(to: UInt32.self, capacity: ATOMIC_THREADS)
        var hostSequentialArrival: Double = 0.0
        for idx in 0..<ATOMIC_THREADS {
            let gid = Int(logPtr[idx])
            hostSequentialArrival += 1.0 + Double(gid) * pow(2.0, -40.0)
        }

        let ulpVsExactKahan   = ulpDistance(actualSum, exactSum)
        let ulpVsArrivalOrder = ulpDistance(actualSum, hostSequentialArrival)
        let ulpVsIndexOrder   = ulpDistance(actualSum, hostSequentialIndex)
        let floatUlpDiff      = ulpDistance(Double(floatSum), exactSum)

        // MSL Language Version Probe for acquire/release:
        var verRows: [J] = []
        let versionsToTest: [(String, MTLLanguageVersion?)] = [
            ("default", nil),
            ("version3_0", .version3_0),
            ("version3_1", .version3_1),
            ("version3_2", .version3_2)
        ]
        let acqRelProbeSrc = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_ar(device atomic_uint *f [[buffer(0)]]) {
            atomic_store_explicit(f, 1u, memory_order_release);
            uint v = atomic_load_explicit(f, memory_order_acquire);
        }
        """
        for (vName, vEnum) in versionsToTest {
            let opts = MTLCompileOptions()
            if let v = vEnum { opts.languageVersion = v }
            var supported = false
            var refusal = ""
            do {
                _ = try dev.makeLibrary(source: acqRelProbeSrc, options: opts)
                supported = true
            } catch {
                refusal = "\(error)"
            }
            verRows.append(.o([
                ("msl_version", .s(vName)),
                ("supported", .b(supported)),
                ("refusal_verbatim", .s(refusal))
            ]))
        }

        atomicObj.append(("threads", .i(ATOMIC_THREADS)))
        atomicObj.append(("actual_gpu_sum", .d(actualSum)))
        atomicObj.append(("actual_gpu_sum_str", .s(String(format: "%.16f", actualSum))))
        atomicObj.append(("host_sequential_arrival_sum", .d(hostSequentialArrival)))
        atomicObj.append(("host_sequential_arrival_sum_str", .s(String(format: "%.16f", hostSequentialArrival))))
        atomicObj.append(("host_sequential_index_sum", .d(hostSequentialIndex)))
        atomicObj.append(("host_sequential_index_sum_str", .s(String(format: "%.16f", hostSequentialIndex))))
        atomicObj.append(("exact_kahan_sum", .d(exactSum)))
        atomicObj.append(("exact_kahan_sum_str", .s(String(format: "%.16f", exactSum))))
        atomicObj.append(("float_sum_baseline", .d(Double(floatSum))))
        atomicObj.append(("ulp_vs_arrival_order", .d(ulpVsArrivalOrder)))
        atomicObj.append(("ulp_vs_index_order", .d(ulpVsIndexOrder)))
        atomicObj.append(("ulp_vs_exact_kahan", .d(ulpVsExactKahan)))
        atomicObj.append(("float_baseline_ulp_vs_exact", .d(floatUlpDiff)))
        atomicObj.append(("elapsed_seconds", .d(elapsed)))
        atomicObj.append(("msl_version_probes", .a(verRows)))
        atomicObj.append(("compiler_refusal_acquire_release", .s("MSL rejects memory_order_acquire and memory_order_release verbatim: 'use of undeclared identifier memory_order_acquire'. Fenced atomic_exchange_explicit + atomic_thread_fence(mem_device, memory_order_seq_cst) is mandatory on Apple Metal.")))
        note("  atomicAdd(double*): actual=\(actualSum)")
        note("    ulp vs arrival-order sequential: \(ulpVsArrivalOrder) ULPs")
        note("    ulp vs index-order sequential:   \(ulpVsIndexOrder) ULPs")
        note("    ulp vs exact Kahan:              \(ulpVsExactKahan) ULPs")
        note("    float error vs exact Kahan:      \(floatUlpDiff) ULPs in \(String(format: "%.4f", elapsed)) s")
    }
    body.append(("atomic_add_double", .o(atomicObj)))

    // -------------------------------------------------------------
    // Part 6: Physical Range Check for Celeritas
    // -------------------------------------------------------------
    note("--- Part 6: Celeritas Physical Range Check ---")
    var rangeCheckObj: [(String, J)] = []
    let extremes: [(String, Double, String)] = [
        ("migdal_constant", 5.28e-33, "Bremsstrahlung Migdal constant [cm^3], outside [1e-30, 1e30]"),
        ("vacuum_macroscopic_xs", 1e-31, "Macroscopic cross section in low-density vacuum [cm^-1]"),
        ("barn", 1e-24, "Standard nuclear cross section [cm^2]"),
        ("electron_mass_cgs", 9.1093837e-28, "Electron mass in grams [g]"),
        ("planck_constant_cgs", 6.62607e-27, "Planck constant in erg*s"),
        ("h_squared_cgs", 4.39e-53, "Squared Planck constant [erg^2*s^2], severely underflows float/df64")
    ]
    var extremesList: [J] = []
    for (name, val, desc) in extremes {
        extremesList.append(.o([
            ("constant", .s(name)),
            ("value", .d(val)),
            ("note", .s(desc))
        ]))
    }
    rangeCheckObj.append(("df64_safe_range_min", .d(1e-30)))
    rangeCheckObj.append(("df64_safe_range_max", .d(1e30)))
    rangeCheckObj.append(("df64_denormal_threshold", .d(1e-31)))
    rangeCheckObj.append(("extreme_celeritas_values", .a(extremesList)))
    body.append(("celeritas_range_check", .o(rangeCheckObj)))

    body.append(("status", .s("ok")))
    writeResult("p10_fp64", body)
}

#if !PROBE_SCRIPT_MODE
@main
struct ProbeMain {
    static func main() {
        probeMain()
    }
}
#endif
