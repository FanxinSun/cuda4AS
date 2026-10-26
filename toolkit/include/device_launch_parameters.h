#pragma once

#include "host_defines.h"
#include "vector_types.h"
#include "driver_types.h"

#if defined(__CUDA__) && defined(__clang__)

#define __CUDA_DEVICE_BUILTIN(FIELD, INTRINSIC)                                \
  __declspec(property(get = __fetch_builtin_##FIELD)) unsigned int FIELD;      \
  static inline __attribute__((always_inline))                                 \
      __attribute__((host)) __attribute__((device)) unsigned int __fetch_builtin_##FIELD(void) { \
    return (unsigned int)INTRINSIC;                                            \
  }

extern "C" {
__host__ __device__ unsigned int __cuda4as_read_tid_x(void);
__host__ __device__ unsigned int __cuda4as_read_tid_y(void);
__host__ __device__ unsigned int __cuda4as_read_tid_z(void);
__host__ __device__ unsigned int __cuda4as_read_ctaid_x(void);
__host__ __device__ unsigned int __cuda4as_read_ctaid_y(void);
__host__ __device__ unsigned int __cuda4as_read_ctaid_z(void);
__host__ __device__ unsigned int __cuda4as_read_ntid_x(void);
__host__ __device__ unsigned int __cuda4as_read_ntid_y(void);
__host__ __device__ unsigned int __cuda4as_read_ntid_z(void);
__host__ __device__ unsigned int __cuda4as_read_nctaid_x(void);
__host__ __device__ unsigned int __cuda4as_read_nctaid_y(void);
__host__ __device__ unsigned int __cuda4as_read_nctaid_z(void);
}

struct __cuda_builtin_threadIdx_t {
  __CUDA_DEVICE_BUILTIN(x, __cuda4as_read_tid_x());
  __CUDA_DEVICE_BUILTIN(y, __cuda4as_read_tid_y());
  __CUDA_DEVICE_BUILTIN(z, __cuda4as_read_tid_z());
  __host__ __device__ operator dim3() const { return dim3(x, y, z); }
  __host__ __device__ operator uint3() const { uint3 r = {x, y, z}; return r; }
};

struct __cuda_builtin_blockIdx_t {
  __CUDA_DEVICE_BUILTIN(x, __cuda4as_read_ctaid_x());
  __CUDA_DEVICE_BUILTIN(y, __cuda4as_read_ctaid_y());
  __CUDA_DEVICE_BUILTIN(z, __cuda4as_read_ctaid_z());
  __host__ __device__ operator dim3() const { return dim3(x, y, z); }
  __host__ __device__ operator uint3() const { uint3 r = {x, y, z}; return r; }
};

struct __cuda_builtin_blockDim_t {
  __CUDA_DEVICE_BUILTIN(x, __cuda4as_read_ntid_x());
  __CUDA_DEVICE_BUILTIN(y, __cuda4as_read_ntid_y());
  __CUDA_DEVICE_BUILTIN(z, __cuda4as_read_ntid_z());
  __host__ __device__ operator dim3() const { return dim3(x, y, z); }
  __host__ __device__ operator uint3() const { uint3 r = {x, y, z}; return r; }
};

struct __cuda_builtin_gridDim_t {
  __CUDA_DEVICE_BUILTIN(x, __cuda4as_read_nctaid_x());
  __CUDA_DEVICE_BUILTIN(y, __cuda4as_read_nctaid_y());
  __CUDA_DEVICE_BUILTIN(z, __cuda4as_read_nctaid_z());
  __host__ __device__ operator dim3() const { return dim3(x, y, z); }
  __host__ __device__ operator uint3() const { uint3 r = {x, y, z}; return r; }
};

#undef __CUDA_DEVICE_BUILTIN

extern const __attribute__((device)) __attribute__((weak)) __cuda_builtin_threadIdx_t threadIdx;
extern const __attribute__((device)) __attribute__((weak)) __cuda_builtin_blockIdx_t blockIdx;
extern const __attribute__((device)) __attribute__((weak)) __cuda_builtin_blockDim_t blockDim;
extern const __attribute__((device)) __attribute__((weak)) __cuda_builtin_gridDim_t gridDim;

__attribute__((device)) static const int warpSize = 32;

// Builtin device functions
extern "C" {
__host__ __device__ void __cuda4as_syncthreads(void);
__host__ __device__ void __cuda4as_trap(void);
static inline __host__ __device__ void __trap(void) { __cuda4as_trap(); }
}
#define __syncthreads() __cuda4as_syncthreads()

static inline __device__ int __syncthreads_count(int predicate) {
    __syncthreads();
    return predicate;
}

static inline __device__ int __syncthreads_and(int predicate) {
    __syncthreads();
    return predicate;
}

static inline __device__ int __syncthreads_or(int predicate) {
    __syncthreads();
    return predicate;
}

static inline __host__ __device__ void __syncwarp(unsigned int mask = 0xffffffff) {
    (void)mask;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline __host__ __device__ float __shfl_sync(unsigned int mask, float var, int srcLane, int width = 32) {
    (void)mask; (void)srcLane; (void)width;
    return var;
}

static inline __host__ __device__ int __shfl_sync(unsigned int mask, int var, int srcLane, int width = 32) {
    (void)mask; (void)srcLane; (void)width;
    return var;
}

static inline __host__ __device__ unsigned int __shfl_sync(unsigned int mask, unsigned int var, int srcLane, int width = 32) {
    (void)mask; (void)srcLane; (void)width;
    return var;
}

static inline __host__ __device__ float __shfl_up_sync(unsigned int mask, float var, unsigned int delta, int width = 32) {
    (void)mask; (void)delta; (void)width;
    return var;
}

static inline __host__ __device__ int __shfl_up_sync(unsigned int mask, int var, unsigned int delta, int width = 32) {
    (void)mask; (void)delta; (void)width;
    return var;
}

static inline __host__ __device__ unsigned int __shfl_up_sync(unsigned int mask, unsigned int var, unsigned int delta, int width = 32) {
    (void)mask; (void)delta; (void)width;
    return var;
}

static inline __host__ __device__ float __shfl_xor_sync(unsigned int mask, float var, int laneMask, int width = 32) {
    (void)mask; (void)laneMask; (void)width;
    return var;
}

static inline __host__ __device__ int __shfl_xor_sync(unsigned int mask, int var, int laneMask, int width = 32) {
    (void)mask; (void)laneMask; (void)width;
    return var;
}

static inline __host__ __device__ unsigned int __shfl_xor_sync(unsigned int mask, unsigned int var, int laneMask, int width = 32) {
    (void)mask; (void)laneMask; (void)width;
    return var;
}

static inline __host__ __device__ float __shfl_down_sync(unsigned int mask, float var, unsigned int delta, int width = 32) {
    (void)mask; (void)delta; (void)width;
    return var;
}

static inline __host__ __device__ int __shfl_down_sync(unsigned int mask, int var, unsigned int delta, int width = 32) {
    (void)mask; (void)delta; (void)width;
    return var;
}

static inline __host__ __device__ unsigned int __shfl_down_sync(unsigned int mask, unsigned int var, unsigned int delta, int width = 32) {
    (void)mask; (void)delta; (void)width;
    return var;
}

static inline __host__ __device__ unsigned int __ballot_sync(unsigned int mask, int predicate) {
    return predicate ? mask : 0;
}

static inline __host__ __device__ int __any_sync(unsigned int mask, int predicate) {
    return predicate != 0;
}

static inline __host__ __device__ int __all_sync(unsigned int mask, int predicate) {
    return predicate != 0;
}

static inline __host__ __device__ void __threadfence(void) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline __host__ __device__ void __threadfence_block(void) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline __host__ __device__ void __threadfence_system(void) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned int __activemask(void) {
    return 0xffffffff;
}

// atomic builtins
static inline __host__ __device__ int atomicAdd(int *address, int val) {
    return __atomic_fetch_add(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned int atomicAdd(unsigned int *address, unsigned int val) {
    return __atomic_fetch_add(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {
    return __atomic_fetch_add(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ float atomicAdd(float *address, float val) {
    unsigned int old_u = *reinterpret_cast<const unsigned int*>(address);
    for (;;) {
        float old_f;
        __builtin_memcpy(&old_f, &old_u, sizeof(float));
        float new_f = old_f + val;
        unsigned int new_u;
        __builtin_memcpy(&new_u, &new_f, sizeof(float));
        if (__atomic_compare_exchange_n(reinterpret_cast<unsigned int*>(address), &old_u, new_u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            return old_f;
        }
    }
}

static inline __host__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long old_u = *reinterpret_cast<const unsigned long long*>(address);
    for (;;) {
        double old_d;
        __builtin_memcpy(&old_d, &old_u, sizeof(double));
        double new_d = old_d + val;
        unsigned long long new_u;
        __builtin_memcpy(&new_u, &new_d, sizeof(double));
        if (__atomic_compare_exchange_n(reinterpret_cast<unsigned long long*>(address), &old_u, new_u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            return old_d;
        }
    }
}

static inline __host__ __device__ int atomicSub(int *address, int val) {
    return __atomic_fetch_sub(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned int atomicSub(unsigned int *address, unsigned int val) {
    return __atomic_fetch_sub(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ int atomicExch(int *address, int val) {
    return __atomic_exchange_n(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned int atomicExch(unsigned int *address, unsigned int val) {
    return __atomic_exchange_n(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {
    return __atomic_exchange_n(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ float atomicExch(float *address, float val) {
    unsigned int val_u;
    __builtin_memcpy(&val_u, &val, sizeof(float));
    unsigned int old_u = __atomic_exchange_n(reinterpret_cast<unsigned int*>(address), val_u, __ATOMIC_SEQ_CST);
    float old_f;
    __builtin_memcpy(&old_f, &old_u, sizeof(float));
    return old_f;
}

static inline __host__ __device__ int atomicCAS(int *address, int compare, int val) {
    int expected = compare;
    __atomic_compare_exchange_n(address, &expected, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}

static inline __host__ __device__ unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val) {
    unsigned int expected = compare;
    __atomic_compare_exchange_n(address, &expected, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}

static inline __host__ __device__ unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {
    unsigned long long expected = compare;
    __atomic_compare_exchange_n(address, &expected, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}

static inline __host__ __device__ unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {
    unsigned short expected = compare;
    __atomic_compare_exchange_n(address, &expected, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}

static inline __host__ __device__ int atomicMin(int *address, int val) {
    int old = *address;
    while (val < old && !__atomic_compare_exchange_n(address, &old, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}

static inline __host__ __device__ unsigned int atomicMin(unsigned int *address, unsigned int val) {
    unsigned int old = *address;
    while (val < old && !__atomic_compare_exchange_n(address, &old, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}

static inline __host__ __device__ long long atomicMin(long long *address, long long val) {
    long long old = *address;
    while (val < old && !__atomic_compare_exchange_n(address, &old, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}

static inline __host__ __device__ unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {
    unsigned long long old = *address;
    while (val < old && !__atomic_compare_exchange_n(address, &old, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}

static inline __host__ __device__ int atomicMax(int *address, int val) {
    int old = *address;
    while (val > old && !__atomic_compare_exchange_n(address, &old, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}

static inline __host__ __device__ unsigned int atomicMax(unsigned int *address, unsigned int val) {
    unsigned int old = *address;
    while (val > old && !__atomic_compare_exchange_n(address, &old, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}

static inline __host__ __device__ long long atomicMax(long long *address, long long val) {
    long long old = *address;
    while (val > old && !__atomic_compare_exchange_n(address, &old, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}

static inline __host__ __device__ unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {
    unsigned long long old = *address;
    while (val > old && !__atomic_compare_exchange_n(address, &old, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}

static inline __host__ __device__ unsigned int atomicInc(unsigned int *address, unsigned int val) {
    unsigned int old = *address;
    unsigned int next;
    do {
        next = ((old >= val) ? 0 : (old + 1));
    } while (!__atomic_compare_exchange_n(address, &old, next, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return old;
}

static inline __host__ __device__ unsigned int atomicDec(unsigned int *address, unsigned int val) {
    unsigned int old = *address;
    unsigned int next;
    do {
        next = (((old == 0) || (old > val)) ? val : (old - 1));
    } while (!__atomic_compare_exchange_n(address, &old, next, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return old;
}

static inline __host__ __device__ int atomicAnd(int *address, int val) {
    return __atomic_fetch_and(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned int atomicAnd(unsigned int *address, unsigned int val) {
    return __atomic_fetch_and(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {
    return __atomic_fetch_and(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ int atomicOr(int *address, int val) {
    return __atomic_fetch_or(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned int atomicOr(unsigned int *address, unsigned int val) {
    return __atomic_fetch_or(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {
    return __atomic_fetch_or(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ int atomicXor(int *address, int val) {
    return __atomic_fetch_xor(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned int atomicXor(unsigned int *address, unsigned int val) {
    return __atomic_fetch_xor(address, val, __ATOMIC_SEQ_CST);
}

static inline __host__ __device__ unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {
    return __atomic_fetch_xor(address, val, __ATOMIC_SEQ_CST);
}

#endif

#ifdef __cplusplus
static inline __host__ __device__ unsigned int __isGlobal(const void *ptr) { return ptr != nullptr ? 1u : 0u; }
static inline __host__ __device__ unsigned int __isShared(const void *ptr) { (void)ptr; return 0u; }
static inline __host__ __device__ unsigned int __isConstant(const void *ptr) { (void)ptr; return 0u; }
static inline __host__ __device__ unsigned int __isLocal(const void *ptr) { (void)ptr; return 0u; }
static inline __host__ __device__ unsigned int __isGridConstant(const void *ptr) { (void)ptr; return 0u; }

static inline __host__ __device__ int __mulhi(int a, int b) {
    return (int)(((long long)a * (long long)b) >> 32);
}
static inline __host__ __device__ unsigned int __umulhi(unsigned int a, unsigned int b) {
    return (unsigned int)(((unsigned long long)a * (unsigned long long)b) >> 32);
}
static inline __host__ __device__ long long __mul64hi(long long a, long long b) {
    return (long long)(((__int128)a * (__int128)b) >> 64);
}
static inline __host__ __device__ unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return (unsigned long long)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
}

static inline __host__ __device__ size_t __cvta_generic_to_shared(const void *ptr) { return (size_t)(uintptr_t)ptr; }
static inline __host__ __device__ size_t __cvta_generic_to_global(const void *ptr) { return (size_t)(uintptr_t)ptr; }
static inline __host__ __device__ void* __cvta_shared_to_generic(size_t off) { return (void*)(uintptr_t)off; }
static inline __host__ __device__ void* __cvta_global_to_generic(size_t off) { return (void*)(uintptr_t)off; }

static inline __host__ __device__ int __popc(unsigned int x) { return __builtin_popcount(x); }
static inline __host__ __device__ int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
static inline __host__ __device__ int __clz(int x) { return __builtin_clz(x); }
static inline __host__ __device__ int __clzll(long long x) { return __builtin_clzll(x); }
static inline __host__ __device__ int __ffs(int x) { return __builtin_ffs(x); }
static inline __host__ __device__ int __ffsll(long long x) { return __builtin_ffsll(x); }
static inline __host__ __device__ unsigned int __brev(unsigned int x) { return __builtin_bitreverse32(x); }
static inline __host__ __device__ unsigned long long __brevll(unsigned long long x) { return __builtin_bitreverse64(x); }
static inline __host__ __device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    unsigned char bytes[8];
    __builtin_memcpy(bytes, &x, 4);
    __builtin_memcpy(bytes + 4, &y, 4);
    unsigned char result[4];
    result[0] = bytes[s & 0x7];
    result[1] = bytes[(s >> 4) & 0x7];
    result[2] = bytes[(s >> 8) & 0x7];
    result[3] = bytes[(s >> 12) & 0x7];
    unsigned int ret;
    __builtin_memcpy(&ret, result, 4);
    return ret;
}
#endif
