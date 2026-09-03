// matmul_tiled -- 32x32 shared-memory tiles, the classic.  Two things are being
// referenced: the tiling itself, and __syncthreads inside a loop, which is the
// pattern §1 maps onto threadgroup_barrier.
//
// 512x512x512 with inputs that are multiples of 1/4 in [0, 3/4]: each product is
// a multiple of 1/16 no larger than 9/16, and a 512-term sum is at most 288.
// 288*16 = 4608, exact in FP32, so the answer does not depend on accumulation
// order or on whether the multiply-add is contracted.
#include "oracle.h"

#define M 512
#define K 512
#define NN 512
#define T 32

__global__ void matmul_tiled(const float *A, const float *B, float *C) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < K / T; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t * T + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * T + threadIdx.y) * NN + col];
        __syncthreads();
        for (int k = 0; k < T; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * NN + col] = acc;
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "matmul_tiled.bin";
    size_t na = (size_t)M * K, nb = (size_t)K * NN, nc = (size_t)M * NN;
    float *hA = (float *)malloc(na * sizeof(float));
    float *hB = (float *)malloc(nb * sizeof(float));
    float *hC = (float *)malloc(nc * sizeof(float));
    uint64_t s = 0x4d41544d554c0004ULL;
    for (size_t i = 0; i < na; i++) hA[i] = lcg_exact_f32(&s, 3, 4);
    for (size_t i = 0; i < nb; i++) hB[i] = lcg_exact_f32(&s, 3, 4);
    float *dA, *dB, *dC;
    CUDA_OK(cudaMalloc(&dA, na * sizeof(float)));
    CUDA_OK(cudaMalloc(&dB, nb * sizeof(float)));
    CUDA_OK(cudaMalloc(&dC, nc * sizeof(float)));
    CUDA_OK(cudaMemcpy(dA, hA, na * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB, nb * sizeof(float), cudaMemcpyHostToDevice));
    dim3 grid(NN / T, M / T), blk(T, T);
    matmul_tiled<<<grid, blk>>>(dA, dB, dC);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(hC, dC, nc * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < M; r += 37) {
        for (int c = 0; c < NN; c += 41) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++) ref += hA[(size_t)r * K + k] * hB[(size_t)k * NN + c];
            if (hC[(size_t)r * NN + c] != ref) bad++;
        }
    }
    if (bad) { fprintf(stderr, "matmul_tiled: %d sampled mismatches\n", bad); return 1; }
    write_ref(out, hC, nc * sizeof(float));
    emit_json("matmul_tiled", "f32", "[512, 512]", nc * sizeof(float),
              "exact", "32x32 shared tiles; exact by construction, order-independent");
    return 0;
}
