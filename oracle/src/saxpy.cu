// saxpy -- y = a*x + y.  nvcc contracts this into an FMA by default and §2.2
// says the translator does the same; the inputs are chosen so contracted and
// uncontracted forms agree bitwise, which means this reference does not depend
// on that choice being made correctly.  a = 5/2, x and y multiples of 1/256,
// so every result is an exact multiple of 1/512.
#include "oracle.h"

#define N (1 << 20)
#define ALPHA 2.5f

__global__ void saxpy(float a, const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "saxpy.bin";
    float *hx = (float *)malloc(N * sizeof(float));
    float *hy = (float *)malloc(N * sizeof(float));
    float *hr = (float *)malloc(N * sizeof(float));
    uint64_t s = 0x5341585059000002ULL;
    for (int i = 0; i < N; i++) { hx[i] = lcg_exact_f32(&s, 255, 256);
                                  hy[i] = lcg_exact_f32(&s, 255, 256); }
    float *dx, *dy;
    CUDA_OK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dy, hy, N * sizeof(float), cudaMemcpyHostToDevice));
    saxpy<<<(N + 255) / 256, 256>>>(ALPHA, dx, dy, N);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(hr, dy, N * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; i++) if (hr[i] != ALPHA * hx[i] + hy[i]) bad++;
    if (bad) { fprintf(stderr, "saxpy: %d host mismatches\n", bad); return 1; }
    write_ref(out, hr, N * sizeof(float));
    emit_json("saxpy", "f32", "[1048576]", (size_t)N * sizeof(float),
              "exact", "y=2.5*x+y; exact whether or not the multiply-add is contracted");
    return 0;
}
