#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include "cuda_runtime.h"

struct DeviceBufferInfo {
    id<MTLBuffer> buffer;
    uint64_t hostAddress;
    uint64_t gpuAddress;
    size_t size;
    size_t offset;
};

struct LaunchConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem{0};
    cudaStream_t stream{nullptr};
    std::vector<uint8_t> argData;
    std::vector<std::pair<size_t, size_t>> argOffsets; // (offset, size)
};

class Cuda4ASContext {
public:
    static Cuda4ASContext& instance();
    
    id<MTLDevice> getActiveDevice();
    int getDeviceCount();
    cudaError_t setDevice(int device);
    int getActiveDeviceIndex() const { return activeDeviceIndex_; }
    
    // Address lookup for unified memory + Metal buffer residency
    void registerBuffer(id<MTLBuffer> buf, size_t size);
    void unregisterBuffer(uint64_t address);
    bool resolveAddress(const void* ptr, id<MTLBuffer> __strong & outBuf, size_t& outOffset);
    size_t getAllocatedBytes() const { return totalAllocatedBytes_; }

    // Launch pipeline
    cudaError_t configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
    cudaError_t setupArgument(const void* arg, size_t size, size_t offset);
    cudaError_t launchKernel(const void* func, void (*hostEntry)(const void*) = nullptr);
    cudaError_t launchSecondaryPhase(id<MTLComputePipelineState> pso);
    cudaError_t synchronize();

    // Shader registry & pipeline management
    void registerShaderSource(const std::string& source);
    id<MTLComputePipelineState> getPipelineState(const std::string& kernelName);

private:
    Cuda4ASContext();
    ~Cuda4ASContext() = default;

    std::mutex mutex_;
    NSArray<id<MTLDevice>>* devices_;
    int activeDeviceIndex_{0};
    id<MTLCommandQueue> commandQueue_{nil};
    id<MTLCommandBuffer> currentCmdBuffer_{nil};
    std::unordered_map<uint64_t, DeviceBufferInfo> bufferTable_;
    size_t totalAllocatedBytes_{0};

    LaunchConfig currentLaunch_;
    std::vector<std::string> shaderSources_;
    std::vector<id<MTLLibrary>> shaderLibraries_;
    size_t compiledShaderCount_{0};
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineCache_;
};
