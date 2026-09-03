// grid_sync -- cooperative groups and grid.sync().
//
// §1: "cooperative launch, grid.sync() | nothing grid-wide | persistent-threads
// barrier only when every block is co-resident -- a PROBED number, 2.8".  That
// probe is p2 in the Apple drop; this is the CUDA side of the same contract, and
// the reference is what a correct grid-wide barrier must produce.
//
// The kernel does a two-phase shuffle that is wrong unless every block sees
// every other block's phase-1 writes: phase 2 reads a slot written by a
// DIFFERENT block.  Integer arithmetic, so the reference is exact.
//
// Built with -rdc=true and launched with cudaLaunchCooperativeKernel, both of
// which the translator has to recognise.
#include "oracle.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define TPB 256

__global__ void grid_sync_kernel(unsigned int *buf, unsigned int *out, int nblocks) {
    cg::grid_group grid = cg::this_grid();
    unsigned int b = blockIdx.x;
    if (threadIdx.x == 0) buf[b] = b * 2654435761u + 1u;
    grid.sync();
    // Read the block "across the grid" from this one: only a real grid-wide
    // barrier makes this defined.
    unsigned int partner = (b + nblocks / 2) % (unsigned int)nblocks;
    unsigned int v = buf[partner];
    grid.sync();
    if (threadIdx.x == 0) out[b] = v ^ (b * 40503u);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "grid_sync.bin";
    int dev = 0; CUDA_OK(cudaGetDevice(&dev));
    int coop = 0;
    CUDA_OK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
    if (!coop) { fprintf(stderr, "cooperative launch unsupported here\n"); return 2; }
    int per_sm = 0, sms = 0;
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,
            (const void *)grid_sync_kernel, TPB, 0));
    CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    int nblocks = per_sm * sms;
    if (nblocks > 1024) nblocks = 1024;
    if (nblocks & 1) nblocks--;
    fprintf(stderr, "grid_sync: %d blocks (%d per SM x %d SMs, capped)\n",
            nblocks, per_sm, sms);
    unsigned int *dbuf, *dout;
    CUDA_OK(cudaMalloc(&dbuf, nblocks * sizeof(unsigned int)));
    CUDA_OK(cudaMalloc(&dout, nblocks * sizeof(unsigned int)));
    void *args[] = { &dbuf, &dout, &nblocks };
    CUDA_OK(cudaLaunchCooperativeKernel((const void *)grid_sync_kernel,
            dim3(nblocks), dim3(TPB), args, 0, 0));
    CUDA_OK(cudaDeviceSynchronize());
    unsigned int *hout = (unsigned int *)malloc(nblocks * sizeof(unsigned int));
    CUDA_OK(cudaMemcpy(hout, dout, nblocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int b = 0; b < nblocks; b++) {
        unsigned int partner = (unsigned int)((b + nblocks / 2) % nblocks);
        unsigned int ref = (partner * 2654435761u + 1u) ^ ((unsigned int)b * 40503u);
        if (hout[b] != ref) bad++;
    }
    if (bad) { fprintf(stderr, "grid_sync: %d mismatches\n", bad); return 1; }
    char shape[64]; snprintf(shape, sizeof(shape), "[%d]", nblocks);
    write_ref(out, hout, nblocks * sizeof(unsigned int));
    emit_json("grid_sync", "u32", shape, (size_t)nblocks * sizeof(unsigned int),
              "exact", "cooperative launch; block count is occupancy-derived on the oracle card and is NOT a target constant");
    return 0;
}
