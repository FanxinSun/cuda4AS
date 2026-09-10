// cuda4AS Clang CUDA facade v1.0.0
//
// This is the deliberately small CUDA surface used by Native AOT Core v1.
// It is a compiler facade, not a CUDA runtime implementation or toolkit copy.
// The Mac runner passes -nocudainc -nocudalib and resolves this file through
// an explicit -I path. Unsupported CUDA APIs are intentionally undeclared.

#ifndef CUDA4AS_CLANG_CUDA_FACADE_V1
#define CUDA4AS_CLANG_CUDA_FACADE_V1 1
#define CUDA4AS_CLANG_CUDA_FACADE_VERSION "1.0.0"

#if !defined(__clang__)
#error "cuda4AS Clang CUDA facade requires Clang"
#endif

#include <stddef.h>
#include <stdint.h>

// Clang's CUDA frontend exposes these attributes through the same spellings
// used by its CUDA runtime wrapper. Empty annotation macros would hide kernel
// and device semantics from the AST, so every annotation maps to an attribute.
#ifndef __host__
#define __host__ __attribute__((host))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __device_builtin__
#define __device_builtin__ __attribute__((device_builtin))
#endif

#ifndef __CUDACC__
#define __CUDACC__ 1
#endif
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 860
#endif

typedef struct __device_builtin__ dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    __host__ __device__ constexpr dim3(unsigned int vx = 1,
                                       unsigned int vy = 1,
                                       unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
} dim3;

// CUDA execution builtins. The device_builtin attribute is required so the
// AST records their compiler-defined meaning rather than ordinary globals.
extern "C" __device__ __device_builtin__ const dim3 blockIdx;
extern "C" __device__ __device_builtin__ const dim3 blockDim;
extern "C" __device__ __device_builtin__ const dim3 threadIdx;

typedef enum cudaError {
    cudaSuccess = 0,
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
} cudaMemcpyKind;

// Only declarations needed to parse the unchanged oracle's host boundary are
// present. There is no implementation and no hidden CPU fallback here.
template <typename T>
__host__ cudaError_t cudaMalloc(T **device_pointer, size_t bytes);
__host__ cudaError_t cudaMemcpy(void *destination, const void *source,
                                size_t bytes, cudaMemcpyKind kind);
__host__ cudaError_t cudaGetLastError();
__host__ cudaError_t cudaDeviceSynchronize();
__host__ const char *cudaGetErrorString(cudaError_t error);

#endif  // CUDA4AS_CLANG_CUDA_FACADE_V1
