#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr int kElements = 4096;

__global__ void vector_add(const std::uint32_t* a,
                           const std::uint32_t* b,
                           std::uint32_t* output,
                           int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        output[index] = a[index] + b[index];
    }
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
    const char* output_path = argc > 1 ? argv[1] : "cmake-vector-add.bin";
    constexpr std::size_t bytes = kElements * sizeof(std::uint32_t);

    auto* host_a = static_cast<std::uint32_t*>(std::malloc(bytes));
    auto* host_b = static_cast<std::uint32_t*>(std::malloc(bytes));
    auto* host_output = static_cast<std::uint32_t*>(std::malloc(bytes));
    if (!host_a || !host_b || !host_output) {
        std::fprintf(stderr, "FAIL: host allocation\n");
        return 1;
    }
    for (int i = 0; i < kElements; ++i) {
        host_a[i] = static_cast<std::uint32_t>(i) * 2654435761u + 17u;
        host_b[i] = (static_cast<std::uint32_t>(i) ^ 0xa5a5a5a5u) * 2246822519u;
    }

    std::uint32_t* device_a = nullptr;
    std::uint32_t* device_b = nullptr;
    std::uint32_t* device_output = nullptr;
    if (!cuda_ok(cudaMalloc(&device_a, bytes), "cudaMalloc(a)") ||
        !cuda_ok(cudaMalloc(&device_b, bytes), "cudaMalloc(b)") ||
        !cuda_ok(cudaMalloc(&device_output, bytes), "cudaMalloc(output)") ||
        !cuda_ok(cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(a)") ||
        !cuda_ok(cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(b)")) {
        return 1;
    }

    vector_add<<<(kElements + 255) / 256, 256>>>(
        device_a, device_b, device_output, kElements);
    if (!cuda_ok(cudaGetLastError(), "vector_add launch") ||
        !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
        !cuda_ok(cudaMemcpy(host_output, device_output, bytes,
                           cudaMemcpyDeviceToHost),
                 "cudaMemcpy(output)")) {
        return 1;
    }

    for (int i = 0; i < kElements; ++i) {
        const std::uint32_t expected = host_a[i] + host_b[i];
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
    cudaFree(device_b);
    cudaFree(device_a);
    std::free(host_output);
    std::free(host_b);
    std::free(host_a);
    std::printf("PASS: cmake-vector-add wrote %zu exact bytes to %s\n", bytes,
                output_path);
    return 0;
}
