// reduce_sum -- shared memory plus __shfl_down_sync, the two CUDA mechanisms
// §1 maps directly onto Metal (threadgroup memory, simd_shuffle).  The output
// is the per-block partial sums, not just the total, so a translated kernel is
// checked block by block rather than on one number that could hide compensating
// errors.
//
// Inputs are multiples of 1/16 in [0, 15/16]; a block of 1024 sums to at most
// 960, which times 16 is 15360 -- exact in FP32 by a wide margin, so summation
// order cannot change the answer.
#include "oracle.h"

#define N (1 << 22)
#define TPB 1024
#define BLOCKS (N / TPB)

__inline__ __device__ float warp_reduce(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__global__ void reduce_sum(const float *in, float *out, int n) {
    __shared__ float warp_sums[TPB / 32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;
    v = warp_reduce(v);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < TPB / 32) ? warp_sums[lane] : 0.0f;
        v = warp_reduce(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "reduce_sum.bin";
    float *hin = (float *)malloc((size_t)N * sizeof(float));
    float *hout = (float *)malloc(BLOCKS * sizeof(float));
    uint64_t s = 0x5245445543450003ULL;
    for (int i = 0; i < N; i++) hin[i] = lcg_exact_f32(&s, 15, 16);
    float *din, *dout;
    CUDA_OK(cudaMalloc(&din, (size_t)N * sizeof(float)));
    CUDA_OK(cudaMalloc(&dout, BLOCKS * sizeof(float)));
    CUDA_OK(cudaMemcpy(din, hin, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    reduce_sum<<<BLOCKS, TPB>>>(din, dout, N);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(hout, dout, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int b = 0; b < BLOCKS; b++) {
        float ref = 0.0f;
        for (int j = 0; j < TPB; j++) ref += hin[(size_t)b * TPB + j];
        if (hout[b] != ref) bad++;
    }
    if (bad) { fprintf(stderr, "reduce_sum: %d block mismatches\n", bad); return 1; }
    write_ref(out, hout, BLOCKS * sizeof(float));
    emit_json("reduce_sum", "f32", "[4096]", (size_t)BLOCKS * sizeof(float),
              "exact", "per-block partial sums; shared memory + __shfl_down_sync; exact by construction");
    return 0;
}
