#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess                           = 0,
    cudaErrorInvalidValue                 = 1,
    cudaErrorMemoryAllocation             = 2,
    cudaErrorInitializationError          = 3,
    cudaErrorLaunchFailure                = 4,
    cudaErrorProfilerNotInitialized       = 6,
    cudaErrorProfilerAlreadyStarted       = 7,
    cudaErrorProfilerAlreadyStopped       = 8,
    cudaErrorInvalidDeviceFunction        = 8,
    cudaErrorInvalidConfiguration         = 9,
    cudaErrorInvalidPitchValue            = 12,
    cudaErrorInvalidSymbol                = 13,
    cudaErrorMapBufferObjectFailed        = 14,
    cudaErrorUnmapBufferObjectFailed      = 15,
    cudaErrorArrayIsMapped                = 16,
    cudaErrorInvalidDevicePointer         = 17,
    cudaErrorAlreadyMapped                = 18,
    cudaErrorInvalidTexture               = 19,
    cudaErrorInvalidTextureBinding        = 20,
    cudaErrorInvalidChannelDescriptor     = 21,
    cudaErrorInvalidMemcpyDirection       = 22,
    cudaErrorInvalidFilterSetting         = 23,
    cudaErrorInvalidNormSetting           = 24,
    cudaErrorCudartUnloading              = 29,
    cudaErrorInvalidResourceHandle        = 33,
    cudaErrorNotReady                     = 34,
    cudaErrorInsufficientDriver           = 35,
    cudaErrorNoDevice                     = 38,
    cudaErrorECCUncorrectable             = 39,
    cudaErrorSharedObjectSymbolNotFound   = 40,
    cudaErrorSharedObjectInitFailed       = 41,
    cudaErrorUnsupportedLimit             = 42,
    cudaErrorDuplicateVariableName        = 43,
    cudaErrorDuplicateTextureName         = 44,
    cudaErrorDuplicateSurfaceName         = 45,
    cudaErrorDevicesUnavailable           = 46,
    cudaErrorInvalidKernelImage           = 47,
    cudaErrorNoKernelImageForDevice       = 48,
    cudaErrorIncompatibleDriverContext    = 49,
    cudaErrorMissingConfiguration         = 52,
    cudaErrorPriorLaunchFailure           = 53,
    cudaErrorDeviceAlreadyInUse           = 54,
    cudaErrorProfilerDisabled             = 55,
    cudaErrorLaunchMaxDepthExceeded       = 65,
    cudaErrorLaunchFileScopedTex          = 66,
    cudaErrorLaunchFileScopedSurf         = 67,
    cudaErrorSyncDepthExceeded            = 68,
    cudaErrorLaunchPendingCountExceeded   = 69,
    cudaErrorInvalidDevice                = 101,
    cudaErrorDeviceNotLicensed            = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure               = 127,
    cudaErrorIllegalAddress               = 700,
    cudaErrorLaunchOutOfResources         = 701,
    cudaErrorLaunchTimeout                = 702,
    cudaErrorLaunchIncompatibleTexturing  = 703,
    cudaErrorPeerAccessAlreadyEnabled     = 704,
    cudaErrorPeerAccessNotEnabled         = 705,
    cudaErrorOperatingSystem              = 706,
    cudaErrorPeerAccessUnsupported        = 707,
    cudaErrorSetOnActiveProcess           = 708,
    cudaErrorContextIsDestroyed           = 709,
    cudaErrorAssert                       = 710,
    cudaErrorTooManyPeers                 = 711,
    cudaErrorHostMemoryAlreadyRegistered  = 712,
    cudaErrorHostMemoryNotRegistered      = 713,
    cudaErrorHardwareStackError           = 714,
    cudaErrorIllegalInstruction           = 715,
    cudaErrorMisalignedAddress            = 716,
    cudaErrorInvalidAddressSpace          = 717,
    cudaErrorInvalidPc                    = 718,
    cudaErrorIllegalState                 = 719,
    cudaErrorNotPermitted                 = 800,
    cudaErrorNotSupported                 = 801,
    cudaErrorSystemNotReady               = 802,
    cudaErrorSystemDriverMismatch         = 803,
    cudaErrorCompatNotSupportedOnDevice   = 804,
    cudaErrorMpsConnectionFailed          = 805,
    cudaErrorMpsRpcFailure                = 806,
    cudaErrorMpsServerNotReady            = 807,
    cudaErrorMpsMaxClientsReached         = 808,
    cudaErrorMpsMaxConnectionsReached     = 809,
    cudaErrorUnknown                      = 999
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
} cudaMemcpyKind;

typedef enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock               = 1,
    cudaDevAttrMaxBlockDimX                     = 2,
    cudaDevAttrMaxBlockDimY                     = 3,
    cudaDevAttrMaxBlockDimZ                     = 4,
    cudaDevAttrMaxGridDimX                      = 5,
    cudaDevAttrMaxGridDimY                      = 6,
    cudaDevAttrMaxGridDimZ                      = 7,
    cudaDevAttrMaxSharedMemoryPerBlock          = 8,
    cudaDevAttrTotalConstantMemory              = 9,
    cudaDevAttrWarpSize                         = 10,
    cudaDevAttrMaxPitch                         = 11,
    cudaDevAttrMaxRegistersPerBlock             = 12,
    cudaDevAttrClockRate                        = 13,
    cudaDevAttrTextureAlignment                 = 14,
    cudaDevAttrMultiProcessorCount              = 16,
    cudaDevAttrKernelExecTimeout                = 17,
    cudaDevAttrIntegrated                       = 18,
    cudaDevAttrCanMapHostMemory                 = 19,
    cudaDevAttrComputeMode                      = 20,
    cudaDevAttrConcurrentKernels                = 31,
    cudaDevAttrEccEnabled                       = 32,
    cudaDevAttrPciBusId                         = 33,
    cudaDevAttrPciDeviceId                      = 34,
    cudaDevAttrTccDriver                        = 35,
    cudaDevAttrMemoryClockRate                  = 36,
    cudaDevAttrGlobalMemoryBusWidth             = 37,
    cudaDevAttrL2CacheSize                      = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor      = 39,
    cudaDevAttrAsyncEngineCount                 = 40,
    cudaDevAttrUnifiedAddressing                = 41,
    cudaDevAttrComputeCapabilityMajor           = 75,
    cudaDevAttrComputeCapabilityMinor           = 76,
    cudaDevAttrCooperativeLaunch                = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch     = 96
} cudaDeviceAttr;

typedef enum cudaLimit {
    cudaLimitStackSize                    = 0x00,
    cudaLimitPrintfFifoSize               = 0x01,
    cudaLimitMallocHeapSize               = 0x02,
    cudaLimitDevRuntimeSyncDepth          = 0x03,
    cudaLimitDevRuntimePendingLaunchCount = 0x04,
    cudaLimitMaxL2FetchGranularity        = 0x05,
    cudaLimitPersistingL2CacheSize        = 0x06
} cudaLimit;

typedef struct CUmemoryPool_st* cudaMemPool_t;

typedef enum cudaMemPoolAttr {
    cudaMemPoolReuseFollowEventDependencies   = 1,
    cudaMemPoolReuseAllowOpportunistic        = 2,
    cudaMemPoolReuseAllowInternalDependencies = 3,
    cudaMemPoolAttrReleaseThreshold           = 4,
    cudaMemPoolAttrReservedMemCurrent         = 5,
    cudaMemPoolAttrReservedMemHigh            = 6,
    cudaMemPoolAttrUsedMemCurrent             = 7,
    cudaMemPoolAttrUsedMemHigh                = 8
} cudaMemPoolAttr;

typedef enum cudaMemLocationType {
    cudaMemLocationTypeInvalid  = 0,
    cudaMemLocationTypeDevice   = 1,
    cudaMemLocationTypeHost     = 2,
    cudaMemLocationTypeHostNuma = 3
} cudaMemLocationType;

typedef struct cudaMemLocation {
    enum cudaMemLocationType type;
    int id;
} cudaMemLocation;

typedef enum cudaSharedMemConfig {
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
} cudaSharedMemConfig;

struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    int binaryVersion;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
};

#define cudaHostAllocDefault        0x00
#define cudaHostAllocPortable       0x01
#define cudaHostAllocMapped         0x02
#define cudaHostAllocWriteCombined  0x04

#define cudaMemAttachGlobal         0x01
#define cudaMemAttachHost           0x02
#define cudaMemAttachSingle         0x04

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
};

enum cudaLaunchAttributeID {
    cudaLaunchAttributeInvalid = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8
};

struct cudaLaunchAttribute {
    enum cudaLaunchAttributeID id;
    union {
        int programmaticStreamSerializationAllowed;
        int cooperative;
        int priority;
    } val;
};

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};
#else
typedef struct dim3 {
    unsigned int x, y, z;
} dim3;
#endif

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;

#define cudaStreamDefault       ((cudaStream_t)0x00)
#define cudaStreamNonBlocking   0x01
#define cudaStreamLegacy        ((cudaStream_t)0x01)
#define cudaStreamPerThread     ((cudaStream_t)0x02)

#define cudaEventDefault        0x00
#define cudaEventBlockingSync   0x01
#define cudaEventDisableTiming  0x02
#define cudaEventInterprocess   0x04

typedef struct cudaLaunchConfig_st {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    struct cudaLaunchAttribute *attrs;
    unsigned int numAttrs;
} cudaLaunchConfig_t;
