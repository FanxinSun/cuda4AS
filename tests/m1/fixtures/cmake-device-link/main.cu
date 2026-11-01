#include "device-api.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr int kElements = 2048;

std::uint32_t host_mix(std::uint32_t value, std::uint32_t index) {
    value ^= index * 0x9e3779b9u;
    value = (value << 7) | (value >> 25);
    return value * 2246822519u + 3266489917u;
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "FAIL: %s returned CUDA error %d\n", operation,
                 static_cast<int>(status));
    return false;
}
}  // namespace

int main(int argc, char** argv) {
    const char* output_path = argc > 1 ? argv[1] : "cmake-device-link.bin";
    constexpr std::size_t bytes = kElements * sizeof(std::uint32_t);

    auto* host_input = static_cast<std::uint32_t*>(std::malloc(bytes));
    auto* host_output = static_cast<std::uint32_t*>(std::malloc(bytes));
    if (!host_input || !host_output) {
        std::fprintf(stderr, "FAIL: host allocation\n");
        return 1;
    }
    for (int i = 0; i < kElements; ++i) {
        host_input[i] = static_cast<std::uint32_t>(i) * 747796405u + 2891336453u;
    }

    std::uint32_t* device_input = nullptr;
    std::uint32_t* device_output = nullptr;
    if (!cuda_ok(cudaMalloc(&device_input, bytes), "cudaMalloc(input)") ||
        !cuda_ok(cudaMalloc(&device_output, bytes), "cudaMalloc(output)") ||
        !cuda_ok(cudaMemcpy(device_input, host_input, bytes,
                           cudaMemcpyHostToDevice),
                 "cudaMemcpy(input)")) {
        return 1;
    }

    transform_kernel<<<(kElements + 127) / 128, 128>>>(
        device_input, device_output, kElements);
    if (!cuda_ok(cudaGetLastError(), "transform_kernel launch") ||
        !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
        !cuda_ok(cudaMemcpy(host_output, device_output, bytes,
                           cudaMemcpyDeviceToHost),
                 "cudaMemcpy(output)")) {
        return 1;
    }

    for (int i = 0; i < kElements; ++i) {
        const std::uint32_t expected =
            host_mix(host_input[i], static_cast<std::uint32_t>(i));
        if (host_output[i] != expected) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %d: got %u, expected %u\n", i,
                         host_output[i], expected);
            return 1;
        }
    }

    std::FILE* output = std::fopen(output_path, "wb");
    if (!output || std::fwrite(host_output, 1, bytes, output) != bytes ||
        std::fclose(output) != 0) {
        std::fprintf(stderr, "FAIL: could not write %s\n", output_path);
        return 1;
    }

    cudaFree(device_output);
    cudaFree(device_input);
    std::free(host_output);
    std::free(host_input);
    std::printf("PASS: cmake-device-link wrote %zu exact bytes to %s\n", bytes,
                output_path);
    return 0;
}
