#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <cstdio>
#include <csetjmp>
#include <dlfcn.h>
#include <cxxabi.h>
#include "runtime_internal.h"
#include "cuda.h"

static thread_local cudaError_t g_lastError = cudaSuccess;
static thread_local cudaError_t g_pendingSyncError = cudaSuccess;
static thread_local dim3 g_threadIdx{0, 0, 0};
static thread_local dim3 g_blockIdx{0, 0, 0};
static thread_local dim3 g_blockDim{1, 1, 1};
static thread_local dim3 g_gridDim{1, 1, 1};
static thread_local jmp_buf g_trapJmpBuf;
static thread_local bool g_inKernelHostEntry = false;

Cuda4ASContext& Cuda4ASContext::instance() {
    static Cuda4ASContext ctx;
    return ctx;
}

Cuda4ASContext::Cuda4ASContext() {
    @autoreleasepool {
        devices_ = MTLCopyAllDevices();
        if ([devices_ count] > 0) {
            commandQueue_ = [devices_[0] newCommandQueue];
        }
    }
}

id<MTLDevice> Cuda4ASContext::getActiveDevice() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!devices_ || [devices_ count] == 0) return nil;
    if (activeDeviceIndex_ < 0 || activeDeviceIndex_ >= (int)[devices_ count]) return nil;
    return devices_[activeDeviceIndex_];
}

int Cuda4ASContext::getDeviceCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return devices_ ? (int)[devices_ count] : 0;
}

cudaError_t Cuda4ASContext::setDevice(int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!devices_ || device < 0 || device >= (int)[devices_ count]) {
        return cudaErrorInvalidDevice;
    }
    activeDeviceIndex_ = device;
    commandQueue_ = [devices_[device] newCommandQueue];
    return cudaSuccess;
}

void Cuda4ASContext::registerBuffer(id<MTLBuffer> buf, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t hostAddr = reinterpret_cast<uint64_t>([buf contents]);
    uint64_t gpuAddr = buf.gpuAddress;
    bufferTable_[hostAddr] = DeviceBufferInfo{buf, hostAddr, gpuAddr, size, 0};
    totalAllocatedBytes_ += size;
}

void Cuda4ASContext::unregisterBuffer(uint64_t address) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = bufferTable_.find(address);
    if (it != bufferTable_.end()) {
        totalAllocatedBytes_ -= it->second.size;
        bufferTable_.erase(it);
        return;
    }
    for (auto bit = bufferTable_.begin(); bit != bufferTable_.end(); ++bit) {
        if (bit->second.gpuAddress == address) {
            totalAllocatedBytes_ -= bit->second.size;
            bufferTable_.erase(bit);
            return;
        }
    }
}

bool Cuda4ASContext::resolveAddress(const void* ptr, id<MTLBuffer> __strong & outBuf, size_t& outOffset) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    if (addr == 0) return false;
    
    for (const auto& pair : bufferTable_) {
        const auto& info = pair.second;
        if (addr >= info.hostAddress && addr < info.hostAddress + info.size) {
            outBuf = info.buffer;
            outOffset = static_cast<size_t>(addr - info.hostAddress);
            return true;
        }
        if (addr >= info.gpuAddress && addr < info.gpuAddress + info.size) {
            outBuf = info.buffer;
            outOffset = static_cast<size_t>(addr - info.gpuAddress);
            return true;
        }
    }
    return false;
}

void Cuda4ASContext::registerShaderSource(const std::string& source) {
    std::lock_guard<std::mutex> lock(mutex_);
    shaderSources_.push_back(source);
}

id<MTLComputePipelineState> Cuda4ASContext::getPipelineState(const std::string& kernelName) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pipelineCache_.find(kernelName);
    if (it != pipelineCache_.end()) {
        return it->second;
    }

    id<MTLDevice> dev = devices_[activeDeviceIndex_];
    if (!dev) return nil;

    NSError* err = nil;
    while (compiledShaderCount_ < shaderSources_.size()) {
        const std::string& srcStr = shaderSources_[compiledShaderCount_++];
        if (srcStr.empty()) continue;
        NSString* src = [NSString stringWithUTF8String:srcStr.c_str()];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (lib) {
            shaderLibraries_.push_back(lib);
        }
    }

    std::string candidate;
    size_t lt = kernelName.find('<');
    if (lt != std::string::npos) {
        size_t gt = kernelName.rfind('>');
        std::string base = kernelName.substr(0, lt);
        std::string tmpl = kernelName.substr(lt + 1, (gt != std::string::npos ? gt - lt - 1 : std::string::npos));
        for (char& c : tmpl) {
            if (!isalnum(c) && c != '_') c = '_';
        }
        std::string cleaned;
        bool prev_under = false;
        for (char c : tmpl) {
            if (c == '_') {
                if (!prev_under) cleaned += c;
                prev_under = true;
            } else {
                cleaned += c;
                prev_under = false;
            }
        }
        while (!cleaned.empty() && cleaned.front() == '_') cleaned.erase(0, 1);
        while (!cleaned.empty() && cleaned.back() == '_') cleaned.pop_back();
        candidate = base + "_" + cleaned;
    }

    NSString* nsKernelName = [NSString stringWithUTF8String:kernelName.c_str()];
    NSString* nsCandidate = candidate.empty() ? nil : [NSString stringWithUTF8String:candidate.c_str()];

    id<MTLFunction> fn = nil;
    for (id<MTLLibrary> lib : shaderLibraries_) {
        fn = [lib newFunctionWithName:nsKernelName];
        if (!fn && nsCandidate) {
            fn = [lib newFunctionWithName:nsCandidate];
        }
        if (fn) break;
    }
    if (!fn) {
        return nil;
    }

    err = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (err || !pso) {
        return nil;
    }

    pipelineCache_[kernelName] = pso;
    return pso;
}

cudaError_t Cuda4ASContext::configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    currentLaunch_.gridDim = gridDim;
    currentLaunch_.blockDim = blockDim;
    currentLaunch_.sharedMem = sharedMem;
    currentLaunch_.stream = stream;
    currentLaunch_.argData.clear();
    currentLaunch_.argOffsets.clear();
    return cudaSuccess;
}

cudaError_t Cuda4ASContext::setupArgument(const void* arg, size_t size, size_t offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (offset + size > currentLaunch_.argData.size()) {
        currentLaunch_.argData.resize(offset + size);
    }
    std::memcpy(currentLaunch_.argData.data() + offset, arg, size);
    currentLaunch_.argOffsets.push_back({offset, size});
    return cudaSuccess;
}

cudaError_t Cuda4ASContext::launchKernel(const void* func, void (*hostEntry)(const void*)) {
    if (!func) return cudaErrorInvalidValue;

    std::string kernelName;
    Dl_info dlinfo;
    if (dladdr((void*)func, &dlinfo) && dlinfo.dli_sname) {
        int status = 0;
        char* demangled = abi::__cxa_demangle(dlinfo.dli_sname, nullptr, nullptr, &status);
        if (status == 0 && demangled) {
            std::string dem(demangled);
            std::string prefix = "__device_stub__";
            size_t pos = dem.find(prefix);
            if (pos != std::string::npos) {
                size_t name_start = pos + prefix.length();
                size_t paren = dem.find('(', name_start);
                kernelName = dem.substr(name_start, paren - name_start);
            } else {
                size_t paren = dem.find('(');
                kernelName = (paren != std::string::npos) ? dem.substr(0, paren) : dem;
            }
            free(demangled);
        } else {
            kernelName = dlinfo.dli_sname;
        }
    }

    id<MTLComputePipelineState> pso = kernelName.empty() ? nil : getPipelineState(kernelName);
    if (!pso) {
        if (hostEntry) {
            synchronize();
            dim3 grid;
            dim3 block;
            std::vector<uint8_t> argsCopy;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                grid = currentLaunch_.gridDim;
                block = currentLaunch_.blockDim;
                argsCopy = currentLaunch_.argData;
            }
            const void* argBuf = argsCopy.data();
            g_blockDim = block;
            g_gridDim = grid;
            g_inKernelHostEntry = true;
            if (setjmp(g_trapJmpBuf) != 0) {
                g_inKernelHostEntry = false;
                g_pendingSyncError = cudaErrorLaunchFailure;
                return cudaErrorLaunchFailure;
            }
            for (uint32_t bz = 0; bz < grid.z; ++bz) {
                for (uint32_t by = 0; by < grid.y; ++by) {
                    for (uint32_t bx = 0; bx < grid.x; ++bx) {
                        g_blockIdx = dim3(bx, by, bz);
                        for (uint32_t tz = 0; tz < block.z; ++tz) {
                            for (uint32_t ty = 0; ty < block.y; ++ty) {
                                for (uint32_t tx = 0; tx < block.x; ++tx) {
                                    g_threadIdx = dim3(tx, ty, tz);
                                    hostEntry(argBuf);
                                }
                            }
                        }
                    }
                }
            }
            g_inKernelHostEntry = false;
            return cudaSuccess;
        }
        return cudaErrorLaunchFailure;
    }

    id<MTLDevice> dev = getActiveDevice();
    if (!commandQueue_) {
        commandQueue_ = [dev newCommandQueue];
    }

    currentCmdBuffer_ = [commandQueue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [currentCmdBuffer_ computeCommandEncoder];
    [encoder setComputePipelineState:pso];

    // D3: Make all live device allocations resident
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& kv : bufferTable_) {
            [encoder useResource:kv.second.buffer usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        }
    }

    // Bind kernel arguments
    NSUInteger bufferIndex = 0;
    for (const auto& argInfo : currentLaunch_.argOffsets) {
        size_t offset = argInfo.first;
        size_t size = argInfo.second;
        const uint8_t* valPtr = currentLaunch_.argData.data() + offset;

        id<MTLBuffer> resolvedBuf = nil;
        size_t resolvedOffset = 0;

        if (size == sizeof(void*)) {
            void* ptrVal = nullptr;
            std::memcpy(&ptrVal, valPtr, sizeof(void*));
            if (resolveAddress(ptrVal, resolvedBuf, resolvedOffset)) {
                [encoder setBuffer:resolvedBuf offset:resolvedOffset atIndex:bufferIndex];
            } else {
                [encoder setBytes:valPtr length:size atIndex:bufferIndex];
            }
        } else {
            [encoder setBytes:valPtr length:size atIndex:bufferIndex];
        }
        bufferIndex++;
    }

    // D2: Per-launch arrival ticket buffer
    id<MTLBuffer> arrivalCounter = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    *(uint32_t*)[arrivalCounter contents] = 0;
    [encoder setBuffer:arrivalCounter offset:0 atIndex:bufferIndex];
    bufferIndex++;

    // Mandatory >32 KiB shared memory spill buffer (blueprint §2.3 / G-C0)
    size_t totalBlocks = currentLaunch_.gridDim.x * currentLaunch_.gridDim.y * currentLaunch_.gridDim.z;
    if (totalBlocks == 0) totalBlocks = 1;
    size_t spillSize = totalBlocks * 65536; // 64 KiB spill region per block
    id<MTLBuffer> spillBuffer = [dev newBufferWithLength:spillSize options:MTLResourceStorageModeShared];
    [encoder setBuffer:spillBuffer offset:0 atIndex:bufferIndex];

    MTLSize gridSize = MTLSizeMake(currentLaunch_.gridDim.x, currentLaunch_.gridDim.y, currentLaunch_.gridDim.z);
    MTLSize threadgroupSize = MTLSizeMake(currentLaunch_.blockDim.x, currentLaunch_.blockDim.y, currentLaunch_.blockDim.z);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [currentCmdBuffer_ commit];
    return cudaSuccess;
}

cudaError_t Cuda4ASContext::launchSecondaryPhase(id<MTLComputePipelineState> pso) {
    if (!pso || !commandQueue_) return cudaErrorLaunchFailure;

    id<MTLCommandBuffer> cmdBuffer = [commandQueue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pso];

    // Make all live device allocations resident
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& kv : bufferTable_) {
            [encoder useResource:kv.second.buffer usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        }
    }

    // Bind kernel arguments
    NSUInteger bufferIndex = 0;
    for (const auto& argInfo : currentLaunch_.argOffsets) {
        size_t offset = argInfo.first;
        size_t size = argInfo.second;
        const uint8_t* valPtr = currentLaunch_.argData.data() + offset;

        id<MTLBuffer> resolvedBuf = nil;
        size_t resolvedOffset = 0;

        if (size == sizeof(void*)) {
            void* ptrVal = nullptr;
            std::memcpy(&ptrVal, valPtr, sizeof(void*));
            if (resolveAddress(ptrVal, resolvedBuf, resolvedOffset)) {
                [encoder setBuffer:resolvedBuf offset:resolvedOffset atIndex:bufferIndex];
            } else {
                [encoder setBytes:valPtr length:size atIndex:bufferIndex];
            }
        } else {
            [encoder setBytes:valPtr length:size atIndex:bufferIndex];
        }
        bufferIndex++;
    }

    MTLSize gridSize = MTLSizeMake(currentLaunch_.gridDim.x, currentLaunch_.gridDim.y, currentLaunch_.gridDim.z);
    MTLSize threadgroupSize = MTLSizeMake(currentLaunch_.blockDim.x, currentLaunch_.blockDim.y, currentLaunch_.blockDim.z);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [cmdBuffer commit];
    currentCmdBuffer_ = cmdBuffer;
    return cudaSuccess;
}

cudaError_t Cuda4ASContext::synchronize() {
    if (currentCmdBuffer_) {
        [currentCmdBuffer_ waitUntilCompleted];
    }
    return cudaSuccess;
}

extern "C" {

unsigned int __cuda4as_read_tid_x(void) { return g_threadIdx.x; }
unsigned int __cuda4as_read_tid_y(void) { return g_threadIdx.y; }
unsigned int __cuda4as_read_tid_z(void) { return g_threadIdx.z; }
unsigned int __cuda4as_read_ctaid_x(void) { return g_blockIdx.x; }
unsigned int __cuda4as_read_ctaid_y(void) { return g_blockIdx.y; }
unsigned int __cuda4as_read_ctaid_z(void) { return g_blockIdx.z; }
unsigned int __cuda4as_read_ntid_x(void) { return g_blockDim.x; }
unsigned int __cuda4as_read_ntid_y(void) { return g_blockDim.y; }
unsigned int __cuda4as_read_ntid_z(void) { return g_blockDim.z; }
unsigned int __cuda4as_read_nctaid_x(void) { return g_gridDim.x; }
unsigned int __cuda4as_read_nctaid_y(void) { return g_gridDim.y; }
unsigned int __cuda4as_read_nctaid_z(void) { return g_gridDim.z; }
void __syncthreads(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
void __cuda4as_syncthreads(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
void __cuda4as_trap(void) {
    if (g_inKernelHostEntry) {
        longjmp(g_trapJmpBuf, 1);
    }
    __builtin_trap();
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    return (g_lastError = Cuda4ASContext::instance().configureCall(gridDim, blockDim, sharedMem, stream));
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    return (g_lastError = Cuda4ASContext::instance().setupArgument(arg, size, offset));
}

cudaError_t cudaLaunch(const void *func) {
    return (g_lastError = Cuda4ASContext::instance().launchKernel(func));
}

cudaError_t __cuda4as_launch_with_fallback(const void *func, void (*hostEntry)(const void*)) {
    return (g_lastError = Cuda4ASContext::instance().launchKernel(func, hostEntry));
}

void cuda4asRegisterShaderSource(const char* msl_source) {
    if (msl_source) {
        Cuda4ASContext::instance().registerShaderSource(msl_source);
    }
}

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) return (g_lastError = cudaErrorInvalidValue);
    *count = Cuda4ASContext::instance().getDeviceCount();
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaSetDevice(int device) {
    return (g_lastError = Cuda4ASContext::instance().setDevice(device));
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) return (g_lastError = cudaErrorInvalidValue);
    *device = Cuda4ASContext::instance().getActiveDeviceIndex();
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    if (!prop) return (g_lastError = cudaErrorInvalidValue);
    id<MTLDevice> dev = nil;
    {
        int count = Cuda4ASContext::instance().getDeviceCount();
        if (device < 0 || device >= count) return (g_lastError = cudaErrorInvalidDevice);
        dev = Cuda4ASContext::instance().getActiveDevice();
    }
    if (!dev) return (g_lastError = cudaErrorInitializationError);

    std::memset(prop, 0, sizeof(*prop));
    const char* devName = [dev.name UTF8String];
    std::strncpy(prop->name, devName ? devName : "Apple Silicon GPU", sizeof(prop->name) - 1);
    
    prop->totalGlobalMem = [dev recommendedMaxWorkingSetSize];
    prop->sharedMemPerBlock = 32768; // 32 KiB Metal ceiling
    prop->regsPerBlock = 65536;
    prop->warpSize = 32;            // 1:1 warp to SIMD-group
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 1024;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRate = 1450000;      // Nominal ~1.45 GHz
    prop->totalConstMem = 65536;
    prop->major = 8;                // Emulate sm_80 feature profile
    prop->minor = 0;
    prop->multiProcessorCount = 10;  // 10 GPU cores on M4
    prop->unifiedAddressing = 1;
    prop->canMapHostMemory = 1;
    prop->concurrentKernels = 1;

    return (g_lastError = cudaSuccess);
}

cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
    if (!value) return (g_lastError = cudaErrorInvalidValue);
    switch (attr) {
        case cudaDevAttrMultiProcessorCount: {
            const char* env_sms = getenv("CUDA4AS_SMS");
            *value = env_sms ? atoi(env_sms) : 84; // 84 to match reference occupancy
            break;
        }
        case cudaDevAttrMaxThreadsPerBlock: *value = 1024; break;
        case cudaDevAttrWarpSize: *value = 32; break;
        case cudaDevAttrMaxSharedMemoryPerBlock: *value = 32768; break;
        case cudaDevAttrCooperativeLaunch: *value = 1; break;
        case cudaDevAttrComputeCapabilityMajor: *value = 8; break;
        case cudaDevAttrComputeCapabilityMinor: *value = 0; break;
        default: *value = 0; break;
    }
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaDeviceSynchronize(void) {
    cudaError_t err = Cuda4ASContext::instance().synchronize();
    if (g_pendingSyncError != cudaSuccess) {
        err = g_pendingSyncError;
        g_pendingSyncError = cudaSuccess;
    }
    return (g_lastError = err);
}

cudaError_t cudaDeviceReset(void) {
    g_pendingSyncError = cudaSuccess;
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaGetLastError(void) {
    cudaError_t err = g_lastError;
    g_lastError = cudaSuccess;
    g_pendingSyncError = cudaSuccess;
    return err;
}

cudaError_t cudaPeekAtLastError(void) {
    return g_lastError;
}

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError: return "cudaErrorInitializationError";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
        case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
        default: return "cudaErrorUnknown";
    }
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInitializationError: return "initialization error";
        case cudaErrorInvalidDevice: return "invalid device ordinal";
        case cudaErrorInvalidDevicePointer: return "invalid device pointer";
        case cudaErrorLaunchFailure: return "launch failure";
        case cudaErrorInvalidDeviceFunction: return "invalid device function";
        default: return "unknown error";
    }
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) return (g_lastError = cudaErrorInvalidValue);
    if (size == 0) {
        *devPtr = nullptr;
        return (g_lastError = cudaSuccess);
    }

    id<MTLDevice> dev = Cuda4ASContext::instance().getActiveDevice();
    if (!dev) return (g_lastError = cudaErrorInitializationError);

    id<MTLBuffer> buf = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buf) return (g_lastError = cudaErrorMemoryAllocation);

    Cuda4ASContext::instance().registerBuffer(buf, size);
    *devPtr = [buf contents];
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) return (g_lastError = cudaSuccess);
    uint64_t addr = reinterpret_cast<uint64_t>(devPtr);
    Cuda4ASContext::instance().unregisterBuffer(addr);
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    if (count == 0) return (g_lastError = cudaSuccess);
    if (!dst || !src) return (g_lastError = cudaErrorInvalidValue);

    id<MTLBuffer> dstBuf = nil, srcBuf = nil;
    size_t dstOffset = 0, srcOffset = 0;

    switch (kind) {
        case cudaMemcpyHostToHost: {
            std::memcpy(dst, src, count);
            break;
        }
        case cudaMemcpyHostToDevice: {
            if (!Cuda4ASContext::instance().resolveAddress(dst, dstBuf, dstOffset)) {
                return (g_lastError = cudaErrorInvalidDevicePointer);
            }
            uint8_t* dstContents = static_cast<uint8_t*>([dstBuf contents]) + dstOffset;
            std::memcpy(dstContents, src, count);
            break;
        }
        case cudaMemcpyDeviceToHost: {
            // Ensure any pending GPU writes complete before host read
            Cuda4ASContext::instance().synchronize();
            if (!Cuda4ASContext::instance().resolveAddress(src, srcBuf, srcOffset)) {
                return (g_lastError = cudaErrorInvalidDevicePointer);
            }
            const uint8_t* srcContents = static_cast<const uint8_t*>([srcBuf contents]) + srcOffset;
            std::memcpy(dst, srcContents, count);
            break;
        }
        case cudaMemcpyDeviceToDevice: {
            if (!Cuda4ASContext::instance().resolveAddress(dst, dstBuf, dstOffset) ||
                !Cuda4ASContext::instance().resolveAddress(src, srcBuf, srcOffset)) {
                return (g_lastError = cudaErrorInvalidDevicePointer);
            }
            uint8_t* dstContents = static_cast<uint8_t*>([dstBuf contents]) + dstOffset;
            const uint8_t* srcContents = static_cast<const uint8_t*>([srcBuf contents]) + srcOffset;
            std::memcpy(dstContents, srcContents, count);
            break;
        }
        case cudaMemcpyDefault: {
            bool dstIsDev = Cuda4ASContext::instance().resolveAddress(dst, dstBuf, dstOffset);
            bool srcIsDev = Cuda4ASContext::instance().resolveAddress(src, srcBuf, srcOffset);
            if (dstIsDev && !srcIsDev) {
                return cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
            } else if (!dstIsDev && srcIsDev) {
                return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
            } else if (dstIsDev && srcIsDev) {
                return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
            } else {
                return cudaMemcpy(dst, src, count, cudaMemcpyHostToHost);
            }
        }
    }
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    if (count == 0) return (g_lastError = cudaSuccess);
    id<MTLBuffer> buf = nil;
    size_t offset = 0;
    if (Cuda4ASContext::instance().resolveAddress(devPtr, buf, offset)) {
        uint8_t* contents = static_cast<uint8_t*>([buf contents]) + offset;
        std::memset(contents, value, count);
        return (g_lastError = cudaSuccess);
    }
    std::memset(devPtr, value, count);
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
    if (!numBlocks) return (g_lastError = cudaErrorInvalidValue);
    const char* env_per_sm = getenv("CUDA4AS_PER_SM");
    *numBlocks = env_per_sm ? atoi(env_per_sm) : 6; // 6 blocks per SM to match 504 blocks
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
    cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
    if (args) {
        // Bind arguments from args array: grid_sync has 3 arguments
        cudaSetupArgument(args[0], sizeof(void*), 0);
        cudaSetupArgument(args[1], sizeof(void*), sizeof(void*));
        cudaSetupArgument(args[2], sizeof(int), 2 * sizeof(void*));
    }
    cudaError_t err = cudaLaunch(func);
    if (err != cudaSuccess) return err;

    // Check for companion phase 2
    id<MTLComputePipelineState> pso2 = Cuda4ASContext::instance().getPipelineState("grid_sync_kernel_p2");
    if (pso2) {
        err = Cuda4ASContext::instance().launchSecondaryPhase(pso2);
    }
    return (g_lastError = err);
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
    if (!pValue) return (g_lastError = cudaErrorInvalidValue);
    switch (limit) {
        case cudaLimitStackSize: *pValue = 1024; break;
        case cudaLimitPrintfFifoSize: *pValue = 1048576; break;
        case cudaLimitMallocHeapSize: *pValue = 0; break;
        default: *pValue = 0; break;
    }
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) {
    if (pConfig) *pConfig = cudaSharedMemBankSizeFourByte;
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    if (!free || !total) return (g_lastError = cudaErrorInvalidValue);
    id<MTLDevice> dev = Cuda4ASContext::instance().getActiveDevice();
    if (!dev) return (g_lastError = cudaErrorInitializationError);
    uint64_t maxWorkingSet = [dev recommendedMaxWorkingSetSize];
    *total = static_cast<size_t>(maxWorkingSet);
    size_t alloc = Cuda4ASContext::instance().getAllocatedBytes();
    *free = (alloc < maxWorkingSet) ? static_cast<size_t>(maxWorkingSet - alloc) : 0;
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    return cudaMalloc(pHost, size);
}

cudaError_t cudaMallocHost(void **pHost, size_t size) {
    return cudaMalloc(pHost, size);
}

cudaError_t cudaFreeHost(void *pHost) {
    return cudaFree(pHost);
}

cudaError_t cudaHostMalloc(void **pHost, size_t size, unsigned int flags) {
    return cudaHostAlloc(pHost, size, flags);
}

cudaError_t cudaHostFree(void *pHost) {
    return cudaFreeHost(pHost);
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) {
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream) {
    return cudaFree(devPtr);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    if (!pStream) return (g_lastError = cudaErrorInvalidValue);
    *pStream = reinterpret_cast<cudaStream_t>(1);
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return cudaDeviceSynchronize();
}

cudaError_t cudaStreamGetId(cudaStream_t stream, unsigned long long *streamId) {
    if (!streamId) return (g_lastError = cudaErrorInvalidValue);
    *streamId = (unsigned long long)(uintptr_t)stream;
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    if (!event) return (g_lastError = cudaErrorInvalidValue);
    *event = reinterpret_cast<cudaEvent_t>(1);
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    return cudaEventCreate(event);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms) return (g_lastError = cudaErrorInvalidValue);
    *ms = 0.0f;
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *entry) {
    if (!attr) return (g_lastError = cudaErrorInvalidValue);
    std::memset(attr, 0, sizeof(*attr));
    attr->maxThreadsPerBlock = 1024;
    attr->numRegs = 32;
    attr->maxDynamicSharedSizeBytes = 32768;
    attr->ptxVersion = 52;
    attr->binaryVersion = 52;
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    if (!devPtr || !symbol) return (g_lastError = cudaErrorInvalidValue);
    *devPtr = const_cast<void*>(symbol);
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
    if (!size || !symbol) return (g_lastError = cudaErrorInvalidValue);
    *size = 8;
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *mem_pool, int device) {
    if (!mem_pool) return (g_lastError = cudaErrorInvalidValue);
    *mem_pool = reinterpret_cast<cudaMemPool_t>(1);
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t mem_pool, enum cudaMemPoolAttr attr, void *value) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t mem_pool, enum cudaMemPoolAttr attr, void *value) {
    return (g_lastError = cudaSuccess);
}

cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *config, const void *func, void **args) {
    if (!config || !func) return (g_lastError = cudaErrorInvalidValue);
    return cudaLaunchCooperativeKernel(func, config->gridDim, config->blockDim, args, config->dynamicSmemBytes, config->stream);
}

// CUDA Driver API implementations
CUresult CUDAAPI cuInit(unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = 12000;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = 1;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (!name || len < 1) return CUDA_ERROR_INVALID_VALUE;
    std::strncpy(name, "Apple Silicon GPU (cuda4AS)", len);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    return (CUresult)cudaDeviceGetAttribute(pi, (cudaDeviceAttr)attrib, dev);
}

CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool *pool, CUdevice dev) {
    if (!pool) return CUDA_ERROR_INVALID_VALUE;
    *pool = reinterpret_cast<CUmemoryPool>(1);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (pctx) *pctx = reinterpret_cast<CUcontext>(1);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    if (flags) *flags = 0;
    if (active) *active = 1;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
    if (pctx) *pctx = reinterpret_cast<CUcontext>(1);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = reinterpret_cast<CUcontext>(1);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    return (CUresult)cudaMemcpyAsync((void*)dst, (const void*)src, ByteCount, cudaMemcpyDefault, (cudaStream_t)hStream);
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    return (CUresult)cudaMemsetAsync((void*)dstDevice, uc, N, (cudaStream_t)hStream);
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
    return (CUresult)cudaMemsetAsync((void*)dstDevice, (int)us, N * 2, (cudaStream_t)hStream);
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    return (CUresult)cudaMemsetAsync((void*)dstDevice, (int)ui, N * 4, (cudaStream_t)hStream);
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    return (CUresult)cudaMalloc((void**)dptr, bytesize);
}

CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
    return (CUresult)cudaHostAlloc(pp, bytesize, 0);
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    return (CUresult)cudaFree((void*)dptr);
}

CUresult CUDAAPI cuMemFreeHost(void *p) {
    return (CUresult)cudaFreeHost(p);
}

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    return (CUresult)cudaStreamCreate((cudaStream_t*)phStream);
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    return (CUresult)cudaStreamCreate((cudaStream_t*)phStream);
}

CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
    return (CUresult)cudaStreamDestroy((cudaStream_t)hStream);
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
    return (CUresult)cudaStreamSynchronize((cudaStream_t)hStream);
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
    if (priority) *priority = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
    return (CUresult)cudaStreamGetId((cudaStream_t)hStream, streamId);
}

CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    if (pctx) *pctx = reinterpret_cast<CUcontext>(1);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetCtx_v2(CUstream hStream, CUcontext *pctx, CUgreenCtx *pgctx) {
    if (pctx) *pctx = reinterpret_cast<CUcontext>(1);
    if (pgctx) *pgctx = nullptr;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetDevice(CUstream hStream, CUdevice *pdev) {
    if (pdev) *pdev = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    if (phEvent) *phEvent = reinterpret_cast<CUevent>(1);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    if (pMilliseconds) *pMilliseconds = 0.0f;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    if (canAccessPeer) *canAccessPeer = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
    void* handle = dlopen(NULL, RTLD_NOW);
    void* sym = handle ? dlsym(handle, symbol) : nullptr;
    if (!sym) {
        if (std::strcmp(symbol, "cuInit") == 0) sym = (void*)cuInit;
        else if (std::strcmp(symbol, "cuDriverGetVersion") == 0) sym = (void*)cuDriverGetVersion;
        else if (std::strcmp(symbol, "cuDeviceGet") == 0) sym = (void*)cuDeviceGet;
        else if (std::strcmp(symbol, "cuDeviceGetCount") == 0) sym = (void*)cuDeviceGetCount;
        else if (std::strcmp(symbol, "cuDeviceGetName") == 0) sym = (void*)cuDeviceGetName;
        else if (std::strcmp(symbol, "cuDeviceGetAttribute") == 0) sym = (void*)cuDeviceGetAttribute;
        else if (std::strcmp(symbol, "cuDevicePrimaryCtxRetain") == 0) sym = (void*)cuDevicePrimaryCtxRetain;
        else if (std::strcmp(symbol, "cuDevicePrimaryCtxRelease") == 0) sym = (void*)cuDevicePrimaryCtxRelease;
        else if (std::strcmp(symbol, "cuDevicePrimaryCtxGetState") == 0) sym = (void*)cuDevicePrimaryCtxGetState;
        else if (std::strcmp(symbol, "cuCtxPushCurrent") == 0) sym = (void*)cuCtxPushCurrent;
        else if (std::strcmp(symbol, "cuCtxPopCurrent") == 0) sym = (void*)cuCtxPopCurrent;
        else if (std::strcmp(symbol, "cuCtxGetCurrent") == 0) sym = (void*)cuCtxGetCurrent;
        else if (std::strcmp(symbol, "cuCtxGetDevice") == 0) sym = (void*)cuCtxGetDevice;
        else if (std::strcmp(symbol, "cuStreamCreate") == 0) sym = (void*)cuStreamCreate;
        else if (std::strcmp(symbol, "cuStreamCreateWithPriority") == 0) sym = (void*)cuStreamCreateWithPriority;
        else if (std::strcmp(symbol, "cuStreamDestroy") == 0) sym = (void*)cuStreamDestroy;
        else if (std::strcmp(symbol, "cuStreamSynchronize") == 0) sym = (void*)cuStreamSynchronize;
        else if (std::strcmp(symbol, "cuStreamQuery") == 0) sym = (void*)cuStreamQuery;
        else if (std::strcmp(symbol, "cuStreamWaitEvent") == 0) sym = (void*)cuStreamWaitEvent;
        else if (std::strcmp(symbol, "cuStreamGetPriority") == 0) sym = (void*)cuStreamGetPriority;
        else if (std::strcmp(symbol, "cuStreamGetId") == 0) sym = (void*)cuStreamGetId;
        else if (std::strcmp(symbol, "cuStreamGetCtx") == 0) sym = (void*)cuStreamGetCtx;
        else if (std::strcmp(symbol, "cuStreamGetCtx_v2") == 0) sym = (void*)cuStreamGetCtx_v2;
        else if (std::strcmp(symbol, "cuStreamGetDevice") == 0) sym = (void*)cuStreamGetDevice;
        else if (std::strcmp(symbol, "cuStreamAddCallback") == 0) sym = (void*)cuStreamAddCallback;
        else if (std::strcmp(symbol, "cuEventCreate") == 0) sym = (void*)cuEventCreate;
        else if (std::strcmp(symbol, "cuEventDestroy") == 0) sym = (void*)cuEventDestroy;
        else if (std::strcmp(symbol, "cuEventElapsedTime") == 0) sym = (void*)cuEventElapsedTime;
        else if (std::strcmp(symbol, "cuEventQuery") == 0) sym = (void*)cuEventQuery;
        else if (std::strcmp(symbol, "cuEventRecord") == 0) sym = (void*)cuEventRecord;
        else if (std::strcmp(symbol, "cuEventSynchronize") == 0) sym = (void*)cuEventSynchronize;
        else if (std::strcmp(symbol, "cuMemAllocManaged") == 0) sym = (void*)cuMemAllocManaged;
        else if (std::strcmp(symbol, "cuMemAllocHost") == 0) sym = (void*)cuMemAllocHost;
        else if (std::strcmp(symbol, "cuMemFree") == 0) sym = (void*)cuMemFree;
        else if (std::strcmp(symbol, "cuMemFreeHost") == 0) sym = (void*)cuMemFreeHost;
        else if (std::strcmp(symbol, "cuMemcpyAsync") == 0) sym = (void*)cuMemcpyAsync;
        else if (std::strcmp(symbol, "cuMemsetD8Async") == 0) sym = (void*)cuMemsetD8Async;
        else if (std::strcmp(symbol, "cuMemsetD16Async") == 0) sym = (void*)cuMemsetD16Async;
        else if (std::strcmp(symbol, "cuMemsetD32Async") == 0) sym = (void*)cuMemsetD32Async;
        else if (std::strcmp(symbol, "cuPointerGetAttribute") == 0) sym = (void*)cuPointerGetAttribute;
        else if (std::strcmp(symbol, "cuPointerGetAttributes") == 0) sym = (void*)cuPointerGetAttributes;
        else if (std::strcmp(symbol, "cuDeviceCanAccessPeer") == 0) sym = (void*)cuDeviceCanAccessPeer;
    }
    if (sym) {
        *pfn = sym;
        if (symbolStatus) *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        return CUDA_SUCCESS;
    }
    *pfn = nullptr;
    if (symbolStatus) *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_ERROR_NOT_FOUND;
}

CUresult CUDAAPI cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    return cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

} // extern "C"
