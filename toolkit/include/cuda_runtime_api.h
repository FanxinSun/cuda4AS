#pragma once

#define CUDART_VERSION 12000

#include "driver_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device Management
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);

// Error Handling
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

// Stream & Event Types
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;

// Event Management
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);

// Memory Management
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
cudaError_t cudaMallocHost(void **pHost, size_t size);
cudaError_t cudaFreeHost(void *pHost);
cudaError_t cudaHostMalloc(void **pHost, size_t size, unsigned int flags);
cudaError_t cudaHostFree(void *pHost);
#ifdef __cplusplus
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
#else
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
#endif

// Async Memory Pools
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream);
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream);
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *mem_pool, int device);
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t mem_pool, enum cudaMemPoolAttr attr, void *value);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t mem_pool, enum cudaMemPoolAttr attr, void *value);

// Stream Management
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamGetId(cudaStream_t stream, unsigned long long *streamId);

// Execution Control & Occupancy
cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *entry);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *config, const void *func, void **args);

#ifdef __cplusplus
}
#endif
