#pragma once

#include "cuda_runtime.h"

#ifdef __cplusplus
namespace cooperative_groups {

class grid_group {
public:
    __device__ void sync() const {
#if defined(__CUDA_ARCH__) || defined(__NVPTX__)
        __syncthreads();
#endif
    }
    __device__ unsigned int num_threads() const {
        return gridDim.x * blockDim.x;
    }
    __device__ unsigned int thread_rank() const {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }
};

static inline __device__ grid_group this_grid() {
    return grid_group();
}

} // namespace cooperative_groups
#endif
