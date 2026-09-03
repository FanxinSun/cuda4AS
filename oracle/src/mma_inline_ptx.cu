// mma_inline_ptx -- one mma.sync.m16n8k16 in inline asm.
//
// §2.1: "Performance code carries asm volatile(mma.sync...) ... This is why
// the source product still needs a PTX front-end."  CUTLASS and flash-attention
// are full of exactly this instruction, so it is the first inline-PTX corpus
// item and its reference has to exist before the translator does.
//
// Fragment layout is the PTX ISA's for m16n8k16 with .f16 operands and .f32
// accumulate: groupID = laneid >> 2, threadID_in_group = laneid % 4.  Operands
// are small non-negative integers, so every product and every 16-term sum is
// exact in FP32 and the reference is bitwise.
#include "oracle.h"
#include <cuda_fp16.h>

#define MM 16
#define NNN 8
#define KK 16

__global__ void mma_kernel(const __half *A, const __half *B, float *D) {
    int lane = threadIdx.x;                 // one warp
    int group = lane >> 2;                  // 0..7
    int tig = lane & 3;                     // 0..3

    __half a[8], b[4];
    // A is 16x16 row-major.
    a[0] = A[(group)     * KK + (tig * 2 + 0)];
    a[1] = A[(group)     * KK + (tig * 2 + 1)];
    a[2] = A[(group + 8) * KK + (tig * 2 + 0)];
    a[3] = A[(group + 8) * KK + (tig * 2 + 1)];
    a[4] = A[(group)     * KK + (tig * 2 + 8)];
    a[5] = A[(group)     * KK + (tig * 2 + 9)];
    a[6] = A[(group + 8) * KK + (tig * 2 + 8)];
    a[7] = A[(group + 8) * KK + (tig * 2 + 9)];
    // B is 16x8 (k by n).
    b[0] = B[(tig * 2 + 0) * NNN + group];
    b[1] = B[(tig * 2 + 1) * NNN + group];
    b[2] = B[(tig * 2 + 8) * NNN + group];
    b[3] = B[(tig * 2 + 9) * NNN + group];

    unsigned int ra[4], rb[2];
    memcpy(ra, a, sizeof(ra));
    memcpy(rb, b, sizeof(rb));
    float d[4] = {0.f, 0.f, 0.f, 0.f};

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
          "r"(rb[0]), "r"(rb[1]));

    // D is 16x8 row-major.
    D[(group)     * NNN + (tig * 2 + 0)] = d[0];
    D[(group)     * NNN + (tig * 2 + 1)] = d[1];
    D[(group + 8) * NNN + (tig * 2 + 0)] = d[2];
    D[(group + 8) * NNN + (tig * 2 + 1)] = d[3];
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "mma_inline_ptx.bin";
    __half hA[MM * KK]; __half hB[KK * NNN]; float hD[MM * NNN];
    uint64_t s = 0x4d4d415054580006ULL;
    for (int i = 0; i < MM * KK; i++) hA[i] = __float2half((float)lcg_u32(&s, 4));
    for (int i = 0; i < KK * NNN; i++) hB[i] = __float2half((float)lcg_u32(&s, 4));
    __half *dA, *dB; float *dD;
    CUDA_OK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_OK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_OK(cudaMalloc(&dD, sizeof(hD)));
    CUDA_OK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    mma_kernel<<<1, 32>>>(dA, dB, dD);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int m = 0; m < MM; m++) for (int n = 0; n < NNN; n++) {
        float ref = 0.0f;
        for (int k = 0; k < KK; k++)
            ref += __half2float(hA[m * KK + k]) * __half2float(hB[k * NNN + n]);
        if (hD[m * NNN + n] != ref) bad++;
    }
    if (bad) { fprintf(stderr, "mma: %d mismatches against the host reference\n", bad); return 1; }
    write_ref(out, hD, sizeof(hD));
    emit_json("mma_inline_ptx", "f32", "[16, 8]", sizeof(hD),
              "exact", "one mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 in inline asm");
    return 0;
}
