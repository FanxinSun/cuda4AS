#pragma once

#include "host_defines.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "vector_types.h"

#ifdef __cplusplus
template <class T>
static inline cudaError_t cudaMalloc(T **devPtr, size_t size) {
    return cudaMalloc((void**)devPtr, size);
}

template <typename T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, T func, int blockSize, size_t dynamicSMemSize) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, (const void*)func, blockSize, dynamicSMemSize);
}

template <typename T>
static inline cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int *minGridSize,
    int *blockSize,
    T func,
    size_t dynamicSMemSize = 0,
    int blockSizeLimit = 0)
{
    if (minGridSize) *minGridSize = 64;
    if (blockSize) *blockSize = (blockSizeLimit > 0 && blockSizeLimit < 256) ? blockSizeLimit : 256;
    return cudaSuccess;
}

template <typename T, typename UnaryFunction>
static inline cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(
    int *minGridSize,
    int *blockSize,
    T func,
    UnaryFunction blockSizeToDynamicSMemSize,
    int blockSizeLimit = 0)
{
    if (minGridSize) *minGridSize = 64;
    if (blockSize) *blockSize = (blockSizeLimit > 0 && blockSizeLimit < 256) ? blockSizeLimit : 256;
    return cudaSuccess;
}

template <typename F, typename... Args>
static inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *config, F func, Args... args) {
    if constexpr (sizeof...(Args) > 0) {
        void *kernelParams[] = { const_cast<void*>(static_cast<const void*>(&args))... };
        return cudaLaunchKernelEx(config, reinterpret_cast<const void*>(func), kernelParams);
    } else {
        return cudaLaunchKernelEx(config, reinterpret_cast<const void*>(func), nullptr);
    }
}
#endif

#if defined(__CUDA__) && defined(__clang__)
#pragma clang force_cuda_host_device begin
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#define __CLANG_CUDA_WRAPPERS_CMATH 1
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
namespace std {
using ::acos; using ::acosf; using ::asin; using ::asinf;
using ::atan; using ::atanf; using ::atan2; using ::atan2f;
using ::cbrt; using ::cbrtf; using ::ceil; using ::ceilf;
using ::copysign; using ::copysignf; using ::cos; using ::cosf;
using ::cosh; using ::coshf; using ::erf; using ::erff;
using ::erfc; using ::erfcf; using ::exp; using ::expf;
using ::exp2; using ::exp2f; using ::expm1; using ::expm1f;
using ::fabs; using ::fabsf; using ::fdim; using ::fdimf;
using ::floor; using ::floorf; using ::fma; using ::fmaf;
using ::fmax; using ::fmaxf; using ::fmin; using ::fminf;
using ::fmod; using ::fmodf; using ::frexp; using ::frexpf;
using ::hypot; using ::hypotf; using ::ilogb; using ::ilogbf;
using ::ldexp; using ::ldexpf; using ::lgamma; using ::lgammaf;
using ::llrint; using ::llrintf; using ::llround; using ::llroundf;
using ::log; using ::logf; using ::log10; using ::log10f;
using ::log1p; using ::log1pf; using ::log2; using ::log2f;
using ::logb; using ::logbf; using ::lrint; using ::lrintf;
using ::lround; using ::lroundf; using ::modf; using ::modff;
using ::nan; using ::nanf; using ::nearbyint; using ::nearbyintf;
using ::nextafter; using ::nextafterf; using ::pow; using ::powf;
using ::remainder; using ::remainderf; using ::remquo; using ::remquof;
using ::rint; using ::rintf; using ::round; using ::roundf;
using ::scalbln; using ::scalblnf; using ::scalbn; using ::scalbnf;
using ::sin; using ::sinf; using ::sinh; using ::sinhf;
using ::sqrt; using ::sqrtf; using ::tan; using ::tanf;
using ::tanh; using ::tanhf; using ::tgamma; using ::tgammaf;
using ::trunc; using ::truncf;
using ::isfinite; using ::isinf; using ::isnan; using ::isnormal;
using ::signbit; using ::fpclassify;
using ::isgreater; using ::isgreaterequal; using ::isless;
using ::islessequal; using ::islessgreater; using ::isunordered;
}
#pragma clang force_cuda_host_device end
#else
#include <cmath>
#include <cstdlib>
#include <cstdio>
#endif

#ifdef __cplusplus
template <typename T>
static inline __host__ __device__ T __ldg(const T *ptr) {
    return *ptr;
}
#endif

static inline __host__ __device__ float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

static inline __host__ __device__ double rsqrt(double x) {
    return 1.0 / sqrt(x);
}

#ifdef __cplusplus
static inline __host__ __device__ void sincosf(float x, float *sptr, float *cptr) {
    *sptr = std::sin(x);
    *cptr = std::cos(x);
}
static inline __host__ __device__ void sincos(double x, double *sptr, double *cptr) {
    *sptr = std::sin(x);
    *cptr = std::cos(x);
}
static inline __host__ __device__ void sincospif(float x, float *sptr, float *cptr) {
    constexpr float pi = 3.14159265358979323846f;
    *sptr = std::sin(x * pi);
    *cptr = std::cos(x * pi);
}
static inline __host__ __device__ void sincospi(double x, double *sptr, double *cptr) {
    constexpr double pi = 3.14159265358979323846;
    *sptr = std::sin(x * pi);
    *cptr = std::cos(x * pi);
}
static inline __host__ __device__ void sincospi(float x, float *sptr, float *cptr) {
    sincospif(x, sptr, cptr);
}
static inline __host__ __device__ float sinpif(float x) {
    return std::sin(x * 3.14159265358979323846f);
}
static inline __host__ __device__ double sinpi(double x) {
    return std::sin(x * 3.14159265358979323846);
}
static inline __host__ __device__ float cospif(float x) {
    return std::cos(x * 3.14159265358979323846f);
}
static inline __host__ __device__ double cospi(double x) {
    return std::cos(x * 3.14159265358979323846);
}
#endif

// Host-side launch configuration routines called by Clang compiler lowering
#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void *func);

#ifdef __cplusplus
}
#endif
