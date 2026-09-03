// Shared scaffolding for the phase-0 CUDA kernel corpus.
//
// These nine kernels are the CORRECTNESS ORACLE for the translator of phase 0b:
// each is built with nvcc, run on the RTX 5080 in the development box, and its
// output written to ref/<kernel>.bin.  When the translator emits MSL for the
// same source, the Apple GPU's output is compared against these bytes.
//
// THE MACHINE RULE, restated where it bites hardest.  The RTX 5080 numbers here
// are an ORACLE FOR CORRECTNESS AND NOTHING ELSE.  No timing is recorded, no
// performance target is set from this card, and nothing about sm_120 -- its
// 100 KB dynamic shared memory, its FP64 rate, its warp scheduler -- may leak
// into a design decision.  The kernels are deliberately written to Ampere-class
// features only, because §2.1 fixes __CUDA_ARCH__ at 860 for the Apple target.
//
// Every kernel is arranged so its answer is BIT-EXACT regardless of summation
// order: inputs are small multiples of a negative power of two, chosen so every
// partial sum is exactly representable.  That matters because a translated
// kernel will not reduce in the same order, and a reference that needs a
// tolerance cannot tell a rounding difference from a real bug.

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_OK(x) do {                                                       \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x,          \
                cudaGetErrorString(_e));                                      \
        exit(1);                                                              \
    }                                                                         \
} while (0)

// One fixed-seed LCG for every kernel, so a rerun on any machine writes the
// same inputs.  Knuth's multiplier; the low bits are never used.
static inline uint64_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}

/// A float that is an exact multiple of 1/den, in [0, hi/den].  Exactness is
/// the point: sums of these stay exact as long as the total times den fits in
/// 24 bits, which every kernel below arranges.
static inline float lcg_exact_f32(uint64_t *s, int hi, int den) {
    uint32_t u = (uint32_t)(lcg_next(s) >> 40);
    return (float)(u % (uint32_t)(hi + 1)) / (float)den;
}

static inline double lcg_exact_f64(uint64_t *s, int hi, int den) {
    uint32_t u = (uint32_t)(lcg_next(s) >> 40);
    return (double)(u % (uint32_t)(hi + 1)) / (double)den;
}

static inline uint32_t lcg_u32(uint64_t *s, uint32_t mod) {
    return (uint32_t)((lcg_next(s) >> 33) % mod);
}

static void write_ref(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fwrite(data, 1, bytes, f) != bytes) {
        fprintf(stderr, "short write to %s\n", path); exit(1);
    }
    fclose(f);
}

/// One machine-readable line the driver parses.  `compare` says how a
/// translated implementation's output must be checked against these bytes.
static void emit_json(const char *kernel, const char *dtype, const char *shape,
                      size_t bytes, const char *compare, const char *notes) {
    printf("ORACLE_JSON {\"kernel\": \"%s\", \"dtype\": \"%s\", "
           "\"shape\": %s, \"bytes\": %zu, \"compare\": \"%s\", "
           "\"notes\": \"%s\"}\n",
           kernel, dtype, shape, bytes, compare, notes);
}
