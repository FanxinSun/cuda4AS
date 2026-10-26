#ifndef HOST_DEFINES_H
#define HOST_DEFINES_H

#if defined(__CUDA__) && defined(__clang__)
#ifndef __host__
#define __host__ __attribute__((host))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#endif
#ifndef __grid_constant__
#define __grid_constant__
#endif
#else
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#endif

#ifndef __inline__
#define __inline__ inline
#endif

#ifndef __forceinline__
#if defined(__GNUC__) || defined(__clang__)
#define __forceinline__ inline __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif

#ifndef CUDARTAPI
#if defined(_WIN32)
#define CUDARTAPI __stdcall
#else
#define CUDARTAPI
#endif
#endif

#ifndef CUDAAPI
#if defined(_WIN32)
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif
#endif

#endif // HOST_DEFINES_H
