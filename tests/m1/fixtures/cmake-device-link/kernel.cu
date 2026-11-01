#include "device-api.cuh"

__global__ void transform_kernel(const std::uint32_t* input,
                                 std::uint32_t* output,
                                 int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        output[index] = mix_value(input[index], static_cast<std::uint32_t>(index));
    }
}
