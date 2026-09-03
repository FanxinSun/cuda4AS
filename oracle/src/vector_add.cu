// vector_add -- the smallest thing that has to work.  Every element is an exact
// multiple of 1/256 and the sum of two of them is exact, so this reference is
// bitwise for any correct implementation.
#include "oracle.h"

#define N (1 << 20)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "vector_add.bin";
    float *ha = (float *)malloc(N * sizeof(float));
    float *hb = (float *)malloc(N * sizeof(float));
    float *hc = (float *)malloc(N * sizeof(float));
    uint64_t s = 0x5645434144440001ULL;
    for (int i = 0; i < N; i++) { ha[i] = lcg_exact_f32(&s, 255, 256);
                                  hb[i] = lcg_exact_f32(&s, 255, 256); }
    float *da, *db, *dc;
    CUDA_OK(cudaMalloc(&da, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&db, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&dc, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice));
    vector_add<<<(N + 255) / 256, 256>>>(da, db, dc, N);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; i++) if (hc[i] != ha[i] + hb[i]) bad++;
    if (bad) { fprintf(stderr, "vector_add: %d host mismatches\n", bad); return 1; }
    write_ref(out, hc, N * sizeof(float));
    emit_json("vector_add", "f32", "[1048576]", (size_t)N * sizeof(float),
              "exact", "c=a+b; inputs are exact multiples of 1/256");
    return 0;
}
