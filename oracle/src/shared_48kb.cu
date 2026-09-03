// shared_48kb -- a STATIC 48 KiB __shared__ array.  This is the spill case of
// §2.3: Apple GPUs have 32 KiB of threadgroup memory on every chip to date, so
// this kernel cannot run as written and the compiler must slice the excess into
// a per-threadgroup region of device memory, addressed through the same index.
// The kernel is deliberately arranged to touch the whole array and to depend on
// __syncthreads between phases, so a transform that gets the addressing wrong
// produces the wrong bytes rather than merely being slow.
//
// Unsigned integer arithmetic throughout, so the reference is exact.
#include "oracle.h"

#define SHARED_WORDS (48 * 1024 / 4)   /* 12288 words = 49152 B */
#define TPB 256
#define BLOCKS 64

__global__ void shared_48kb(unsigned int *out) {
    __shared__ unsigned int buf[SHARED_WORDS];
    unsigned int seed = blockIdx.x * 2654435761u + 1u;
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x)
        buf[i] = seed ^ ((unsigned int)i * 2246822519u);
    __syncthreads();
    // A stride that is coprime with the array length, so every thread's walk
    // touches words other threads wrote and the barrier above is load-bearing.
    unsigned int acc = 0u;
    for (int i = 0; i < 64; i++) {
        int idx = (int)(((unsigned int)(threadIdx.x * 97 + i * 1543)) % SHARED_WORDS);
        acc = acc * 31u + buf[idx];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x)
        buf[i] = buf[i] + acc;
    __syncthreads();
    unsigned int total = 0u;
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x) total += buf[i];
    out[blockIdx.x * TPB + threadIdx.x] = total;
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "shared_48kb.bin";
    size_t n = (size_t)BLOCKS * TPB;
    unsigned int *hout = (unsigned int *)malloc(n * sizeof(unsigned int));
    unsigned int *dout;
    CUDA_OK(cudaMalloc(&dout, n * sizeof(unsigned int)));
    shared_48kb<<<BLOCKS, TPB>>>(dout);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(hout, dout, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    write_ref(out, hout, n * sizeof(unsigned int));
    emit_json("shared_48kb", "u32", "[64, 256]", n * sizeof(unsigned int),
              "exact", "static 49152 B __shared__; the >32 KiB spill case of blueprint 2.3");
    return 0;
}
