#include "device-api.cuh"

__device__ __noinline__ std::uint32_t mix_value(std::uint32_t value,
                                                 std::uint32_t index) {
    value ^= index * 0x9e3779b9u;
    value = (value << 7) | (value >> 25);
    return value * 2246822519u + 3266489917u;
}
