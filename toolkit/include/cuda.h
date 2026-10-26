#pragma once

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CUDAAPI
#define CUDAAPI
#endif

typedef unsigned long long cuuint64_t;
typedef unsigned int cuuint32_t;
typedef cuuint64_t CUdeviceptr;
typedef int CUdevice;

typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUevent_st* CUevent;
typedef struct CUstream_st* CUstream;
typedef struct CUmemoryPool_st* CUmemoryPool;
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphNode_st* CUgraphNode;
typedef struct CUlibrary_st* CUlibrary;
typedef struct CUkernel_st* CUkernel;
typedef struct CUgreenCtx_st* CUgreenCtx;

typedef enum cudaError_enum {
    CUDA_SUCCESS                              = 0,
    CUDA_ERROR_INVALID_VALUE                  = 1,
    CUDA_ERROR_OUT_OF_MEMORY                  = 2,
    CUDA_ERROR_NOT_INITIALIZED                = 3,
    CUDA_ERROR_DEINITIALIZED                  = 4,
    CUDA_ERROR_PROFILER_DISABLED              = 5,
    CUDA_ERROR_NO_DEVICE                      = 100,
    CUDA_ERROR_INVALID_DEVICE                 = 101,
    CUDA_ERROR_INVALID_IMAGE                  = 200,
    CUDA_ERROR_INVALID_CONTEXT                = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,
    CUDA_ERROR_MAP_FAILED                     = 205,
    CUDA_ERROR_UNMAP_FAILED                   = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED                = 207,
    CUDA_ERROR_ALREADY_MAPPED                 = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,
    CUDA_ERROR_ALREADY_ACQUIRED               = 210,
    CUDA_ERROR_NOT_MAPPED                     = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE              = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,
    CUDA_ERROR_INVALID_PTX                    = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE           = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND         = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION        = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED       = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY      = 224,
    CUDA_ERROR_INVALID_SOURCE                 = 300,
    CUDA_ERROR_FILE_NOT_FOUND                 = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,
    CUDA_ERROR_OPERATING_SYSTEM               = 304,
    CUDA_ERROR_INVALID_HANDLE                 = 400,
    CUDA_ERROR_ILLEGAL_STATE                  = 401,
    CUDA_ERROR_NOT_FOUND                      = 500,
    CUDA_ERROR_NOT_READY                      = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS                = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,
    CUDA_ERROR_ASSERT                         = 710,
    CUDA_ERROR_TOO_MANY_PEERS                 = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR           = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION            = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS             = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE          = 717,
    CUDA_ERROR_INVALID_PC                     = 718,
    CUDA_ERROR_LAUNCH_FAILED                  = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720,
    CUDA_ERROR_NOT_PERMITTED                  = 800,
    CUDA_ERROR_NOT_SUPPORTED                  = 801,
    CUDA_ERROR_SYSTEM_NOT_READY               = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH         = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED          = 805,
    CUDA_ERROR_MPS_RPC_FAILURE                = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY           = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED        = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED    = 809,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED     = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED     = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE           = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED       = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED        = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION       = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT        = 906,
    CUDA_ERROR_CAPTURED_EVENT                 = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONGLIST       = 908,
    CUDA_ERROR_UNKNOWN                        = 999
} CUresult;

typedef enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS                = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND       = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2
} CUdriverProcAddressQueryResult;

enum {
    CU_GET_PROC_ADDRESS_DEFAULT = 0,
    CU_EVENT_WAIT_DEFAULT = 0,
    CU_GREEN_CTX_DEFAULT_STREAM = 0
};

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK           = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                 = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                 = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                 = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                  = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                  = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                  = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK     = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY           = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE                       = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH                       = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK         = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE                      = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT               = 14,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT            = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT             = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED                      = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY             = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                    = 20,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS              = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED                     = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                      = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                   = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER                      = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE               = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH         = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                   = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR  = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT              = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING              = 41,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR        = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR        = 76,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH              = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96
} CUdevice_attribute;

typedef enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK            = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES                = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES                 = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES                 = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS                         = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION                      = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION                   = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA                    = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES    = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;

typedef enum CUmemPool_attribute_enum {
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES   = 1,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC        = 2,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD                 = 4,
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT              = 5,
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH                 = 6,
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT                  = 7,
    CU_MEMPOOL_ATTR_USED_MEM_HIGH                     = 8
} CUmemPool_attribute;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT                    = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE                = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER             = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER               = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS                 = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS                = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID                  = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED                 = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL             = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR           = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE                 = 12,
    CU_POINTER_ATTRIBUTE_MAPPED                     = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES       = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS               = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE             = 17
} CUpointer_attribute;

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST    = 1,
    CU_MEMORYTYPE_DEVICE  = 2,
    CU_MEMORYTYPE_ARRAY   = 3,
    CU_MEMORYTYPE_UNIFIED = 4
} CUmemorytype;

typedef enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE      = 0,
    CU_MEM_ACCESS_FLAGS_PROT_READ      = 1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
} CUmemAccess_flags;

typedef enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0,
    CU_MEM_LOCATION_TYPE_DEVICE  = 1
} CUmemLocationType;

typedef struct CUmemLocation_st {
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0,
    CU_MEM_ALLOCATION_TYPE_PINNED  = 1
} CUmemAllocationType;

typedef enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0
} CUmemAllocationHandleType;

typedef struct CUmemPoolProps_st {
    CUmemAllocationType allocType;
    CUmemAllocationHandleType handleTypes;
    CUmemLocation location;
    void *win32SecurityAttributes;
    unsigned char reserved[64];
} CUmemPoolProps;

typedef struct CUmemAccessDesc_st {
    CUmemLocation location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

typedef void (CUDAAPI *CUhostFn)(void *userData);
typedef void (CUDAAPI *CUstreamCallback)(CUstream hStream, CUresult status, void *userData);

typedef struct CUlaunchConfig_st {
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **attrs;
    unsigned int numAttrs;
} CUlaunchConfig;

typedef enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0
} CUjit_option;

typedef enum CUlibraryOption_enum {
    CU_LIBRARY_HOST_UNIVERSAL_SUPPORT = 0
} CUlibraryOption;

typedef enum CUkernelNodeAttrID_enum {
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
} CUkernelNodeAttrID;

typedef union CUkernelNodeAttrValue_union {
    int dummy;
} CUkernelNodeAttrValue;

typedef struct CUDA_KERNEL_NODE_PARAMS_st {
    CUfunction func;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    void **kernelParams;
    void **extra;
} CUDA_KERNEL_NODE_PARAMS;

typedef struct CUmemcpyAttributes_st {
    int dummy;
} CUmemcpyAttributes;

typedef struct CUtensorMap_st {
    unsigned long long opaque[16];
} CUtensorMap;

typedef enum CUtensorMapDataType_enum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8         = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16        = 1,
    CU_TENSOR_MAP_DATA_TYPE_UINT32        = 2,
    CU_TENSOR_MAP_DATA_TYPE_INT32         = 3,
    CU_TENSOR_MAP_DATA_TYPE_UINT64        = 4,
    CU_TENSOR_MAP_DATA_TYPE_INT64         = 5,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16       = 6,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32       = 7,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64       = 8,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16      = 9,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ   = 10,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32      = 11,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ  = 12,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B  = 13,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B = 14,
    CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B = 15
} CUtensorMapDataType;

typedef enum CUtensorMapInterleave_enum {
    CU_TENSOR_MAP_INTERLEAVE_NONE = 0
} CUtensorMapInterleave;

typedef enum CUtensorMapSwizzle_enum {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0
} CUtensorMapSwizzle;

typedef enum CUtensorMapL2promotion_enum {
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0
} CUtensorMapL2promotion;

typedef enum CUtensorMapFloatOOBfill_enum {
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
} CUtensorMapFloatOOBfill;

// Driver functions
CUresult CUDAAPI cuInit(unsigned int Flags);
CUresult CUDAAPI cuDriverGetVersion(int *driverVersion);
CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal);
CUresult CUDAAPI cuDeviceGetCount(int *count);
CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx);
CUresult CUDAAPI cuCtxGetDevice(CUdevice *device);
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyBatchAsync(CUdeviceptr *dsts, CUdeviceptr *srcs, size_t *sizes, size_t count, CUmemcpyAttributes *attrs, size_t *flags, size_t numFlags, CUstream hStream);
CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps);
CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool);
CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool *pool, CUdevice dev);
CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count);
CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool pool, CUmemLocation *location);
CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize);
CUresult CUDAAPI cuMemFree(CUdeviceptr dptr);
CUresult CUDAAPI cuMemFreeHost(void *p);
CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);
CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
CUresult CUDAAPI cuStreamDestroy(CUstream hStream);
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
CUresult CUDAAPI cuStreamQuery(CUstream hStream);
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority);
CUresult CUDAAPI cuStreamGetId(CUstream hStream, unsigned long long *streamId);
CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx);
CUresult CUDAAPI cuStreamGetCtx_v2(CUstream hStream, CUcontext *pctx, CUgreenCtx *pgctx);
CUresult CUDAAPI cuStreamGetDevice(CUstream hStream, CUdevice *pdev);
CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult CUDAAPI cuEventDestroy(CUevent hEvent);
CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult CUDAAPI cuEventQuery(CUevent hEvent);
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI cuEventSynchronize(CUevent hEvent);
CUresult CUDAAPI cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel);
CUresult CUDAAPI cuKernelGetAttribute(int *pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev);
CUresult CUDAAPI cuKernelGetName(const char **name, CUkernel kernel);
CUresult CUDAAPI cuKernelGetLibrary(CUlibrary *library, CUkernel kernel);
CUresult CUDAAPI cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
CUresult CUDAAPI cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name);
CUresult CUDAAPI cuLibraryUnload(CUlibrary library);
CUresult CUDAAPI cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name);
CUresult CUDAAPI cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name);
CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData);
CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra);
CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID id, const CUkernelNodeAttrValue *value);
CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev);
CUresult CUDAAPI cuGreenCtxCreate(CUgreenCtx *pgctx, void *desc, CUdevice dev, unsigned int flags);
CUresult CUDAAPI cuGreenCtxDestroy(CUgreenCtx gctx);
CUresult CUDAAPI cuCtxFromGreenCtx(CUcontext *pctx, CUgreenCtx gctx);
CUresult CUDAAPI cuGreenCtxGetId(CUgreenCtx gctx, unsigned long long *id);
CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const cuuint32_t *boxDim, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);
CUresult CUDAAPI cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);

#ifdef __cplusplus
}
#endif
