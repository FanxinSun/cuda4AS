// histogram_atomics -- shared-memory atomics then global atomics, the pattern
// every reduction in the wild uses.  Integers, so the reference is exact and
// the only thing under test is that every increment happened exactly once.
#include "oracle.h"

#define N (1 << 22)
#define BINS 256
#define TPB 256

__global__ void histogram(const unsigned char *in, unsigned int *out, int n) {
    __shared__ unsigned int local[BINS];
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) local[i] = 0;
    __syncthreads();
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&local[in[i]], 1u);
    __syncthreads();
    for (int i = threadIdx.x; i < BINS; i += blockDim.x)
        if (local[i]) atomicAdd(&out[i], local[i]);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "histogram_atomics.bin";
    unsigned char *hin = (unsigned char *)malloc(N);
    unsigned int hout[BINS];
    uint64_t s = 0x484953544f470005ULL;
    for (int i = 0; i < N; i++) hin[i] = (unsigned char)lcg_u32(&s, 256);
    unsigned char *din; unsigned int *dout;
    CUDA_OK(cudaMalloc(&din, N));
    CUDA_OK(cudaMalloc(&dout, BINS * sizeof(unsigned int)));
    CUDA_OK(cudaMemset(dout, 0, BINS * sizeof(unsigned int)));
    CUDA_OK(cudaMemcpy(din, hin, N, cudaMemcpyHostToDevice));
    histogram<<<1024, TPB>>>(din, dout, N);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(hout, dout, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int ref[BINS]; memset(ref, 0, sizeof(ref));
    for (int i = 0; i < N; i++) ref[hin[i]]++;
    int bad = 0;
    for (int i = 0; i < BINS; i++) if (ref[i] != hout[i]) bad++;
    if (bad) { fprintf(stderr, "histogram: %d bin mismatches\n", bad); return 1; }
    write_ref(out, hout, sizeof(hout));
    emit_json("histogram_atomics", "u32", "[256]", sizeof(hout),
              "exact", "shared then global atomicAdd over 4 Mi bytes");
    return 0;
}
