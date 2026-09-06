#pragma once

#include <cuda_runtime.h>

#include <cstdint>

__device__ __noinline__ std::uint32_t mix_value(std::uint32_t value,
                                                 std::uint32_t index);

__global__ void transform_kernel(const std::uint32_t* input,
                                 std::uint32_t* output,
                                 int count);
