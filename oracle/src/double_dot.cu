// double_dot -- FP64, the row §1 answers with "none; Apple's Metal compiler
// rejects double arithmetic".  p7 asks that compiler directly; this is the
// reference the software path (double-float, IEEE soft-double, or the CPU
// device of 2.5) has to reproduce.
//
// Inputs are multiples of 1/256 in [0,1), 1 Mi of them, so every partial sum is
// exact in FP64 (the total times 256 stays under 2^53) and the reference does
// not depend on summation order.  A fast48 double-float path with a ~48-bit
// mantissa will still reproduce it exactly at this magnitude, which is the
// point: this file measures whether the path is CORRECT, not how wide it is.
#include "oracle.h"

#define N (1 << 20)
#define TPB 256
#define BLOCKS (N / TPB)

__global__ void double_dot(const double *a, const double *b, double *out, int n) {
    __shared__ double s[TPB];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();
    for (int off = TPB / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "double_dot.bin";
    double *ha = (double *)malloc((size_t)N * sizeof(double));
    double *hb = (double *)malloc((size_t)N * sizeof(double));
    double *ho = (double *)malloc(BLOCKS * sizeof(double));
    uint64_t s = 0x444f55424c450009ULL;
    for (int i = 0; i < N; i++) { ha[i] = lcg_exact_f64(&s, 255, 256);
                                  hb[i] = lcg_exact_f64(&s, 255, 256); }
    double *da, *db, *dout;
    CUDA_OK(cudaMalloc(&da, (size_t)N * sizeof(double)));
    CUDA_OK(cudaMalloc(&db, (size_t)N * sizeof(double)));
    CUDA_OK(cudaMalloc(&dout, BLOCKS * sizeof(double)));
    CUDA_OK(cudaMemcpy(da, ha, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    double_dot<<<BLOCKS, TPB>>>(da, db, dout, N);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(ho, dout, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int b = 0; b < BLOCKS; b++) {
        double ref = 0.0;
        for (int j = 0; j < TPB; j++) ref += ha[(size_t)b * TPB + j] * hb[(size_t)b * TPB + j];
        if (ho[b] != ref) bad++;
    }
    if (bad) { fprintf(stderr, "double_dot: %d block mismatches\n", bad); return 1; }
    write_ref(out, ho, BLOCKS * sizeof(double));
    emit_json("double_dot", "f64", "[4096]", (size_t)BLOCKS * sizeof(double),
              "exact", "per-block FP64 dot products; exact by construction");
    return 0;
}
