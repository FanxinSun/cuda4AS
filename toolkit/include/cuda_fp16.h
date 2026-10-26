#pragma once

#include "cuda_runtime.h"

#ifdef __cplusplus
struct __half {
    union {
        unsigned short __x;
        _Float16 __f;
    };

    constexpr __host__ __device__ __half() : __x(0) {}
    constexpr __host__ __device__ __half(_Float16 f) : __f(f) {}
    constexpr __host__ __device__ explicit __half(unsigned short x) : __x(x) {}
    constexpr __host__ __device__ operator float() const { return (float)__f; }
    constexpr __host__ __device__ operator _Float16() const { return __f; }
};

static inline __host__ __device__ __half __float2half(float f) {
    return __half((_Float16)f);
}

static inline __host__ __device__ __half __double2half(double d) {
    return __half((_Float16)(float)d);
}

static inline __host__ __device__ float __half2float(__half h) {
    return (float)h.__f;
}

static inline __host__ __device__ double __half2double(__half h) {
    return (double)(float)h.__f;
}

struct __half2 {
    __half x;
    __half y;

    constexpr __host__ __device__ __half2() : x(), y() {}
    constexpr __host__ __device__ __half2(__half a, __half b) : x(a), y(b) {}
};

static inline __host__ __device__ __half __hadd(__half a, __half b) { return (float)a + (float)b; }
static inline __host__ __device__ __half __hsub(__half a, __half b) { return (float)a - (float)b; }
static inline __host__ __device__ __half __hmul(__half a, __half b) { return (float)a * (float)b; }
static inline __host__ __device__ __half __hdiv(__half a, __half b) { return (float)a / (float)b; }
static inline __host__ __device__ __half __hmax(__half a, __half b) { return (float)a > (float)b ? a : b; }
static inline __host__ __device__ __half __hmin(__half a, __half b) { return (float)a < (float)b ? a : b; }

static inline __host__ __device__ __half2 __hadd2(__half2 a, __half2 b) { return __half2(__hadd(a.x, b.x), __hadd(a.y, b.y)); }
static inline __host__ __device__ __half2 __hsub2(__half2 a, __half2 b) { return __half2(__hsub(a.x, b.x), __hsub(a.y, b.y)); }
static inline __host__ __device__ __half2 __hmul2(__half2 a, __half2 b) { return __half2(__hmul(a.x, b.x), __hmul(a.y, b.y)); }
static inline __host__ __device__ __half2 __hmax2(__half2 a, __half2 b) { return __half2(__hmax(a.x, b.x), __hmax(a.y, b.y)); }
static inline __host__ __device__ __half2 __hmin2(__half2 a, __half2 b) { return __half2(__hmin(a.x, b.x), __hmin(a.y, b.y)); }
#else
typedef struct {
    unsigned short __x;
} __half;
#endif
