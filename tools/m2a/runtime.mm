// cuda4AS Native AOT Core v1 Objective-C++ runtime.
// This file owns Metal object lifetimes and never executes a CPU kernel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "runtime-config.h"

static std::string hex_digest(const uint8_t *data, size_t bytes) {
  uint8_t digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data, (CC_LONG)bytes, digest);
  char out[CC_SHA256_DIGEST_LENGTH * 2 + 1];
  for (size_t i = 0; i < sizeof(digest); ++i) snprintf(out + i * 2, 3, "%02x", digest[i]);
  out[sizeof(digest) * 2] = 0;
  return std::string(out);
}

static void fail_json(const char *stage, const std::string &message, const char *path) {
  std::ofstream f(path);
  f << "{\n  \"schema\": \"cuda4as-m2a-result-v1\",\n  \"classification\": \"FAIL\",\n";
  f << "  \"failed_stage\": \"" << stage << "\",\n  \"message\": \"";
  for (char c : message) { if (c == '\\' || c == '"') f << '\\'; if (c == '\n') f << "\\n"; else f << c; }
  f << "\"\n}\n";
}

static bool write_file(const char *path, const void *data, size_t bytes) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  bool ok = fwrite(data, 1, bytes, f) == bytes;
  fclose(f);
  return ok;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <work-dir>\n", argv[0]); return 2; }
  const std::string work = argv[1];
  const std::string result_path = work + "/m2a-result.json";
  const std::string output_path = work + "/output.bin";
  auto t0 = std::chrono::steady_clock::now();
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fail_json("device_enumeration", "MTLCreateSystemDefaultDevice returned nil", result_path.c_str()); return 1; }
    NSString *deviceName = device.name ?: @"";
    std::string name = deviceName.UTF8String ? deviceName.UTF8String : "";
    uint64_t registry = device.registryID;
    if (name != std::string(CUDA4AS_EXPECTED_DEVICE_NAME) || registry != (uint64_t)CUDA4AS_EXPECTED_REGISTRY_ID) {
      std::string msg = "inventory mismatch: name=" + name + " registry=" + std::to_string(registry);
      fail_json("device_inventory", msg, result_path.c_str()); return 77;
    }
    if (device.maxThreadsPerThreadgroup.width < CUDA4AS_BLOCK_X) {
      fail_json("device_inventory", "device maxThreadsPerThreadgroup is below recorded block size", result_path.c_str()); return 77;
    }

    std::vector<float> ha(CUDA4AS_ELEMENT_COUNT), hb(CUDA4AS_ELEMENT_COUNT), hc(CUDA4AS_ELEMENT_COUNT);
    uint64_t state = CUDA4AS_SEED;
    auto next = [&]() -> uint64_t { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state; };
    auto exact = [&](int hi, int den) -> float { uint32_t u = (uint32_t)(next() >> 40); return (float)(u % (uint32_t)(hi + 1)) / (float)den; };
    for (uint32_t i = 0; i < CUDA4AS_ELEMENT_COUNT; ++i) { ha[i] = exact(CUDA4AS_HI_A, CUDA4AS_DEN_A); hb[i] = exact(CUDA4AS_HI_B, CUDA4AS_DEN_B); }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) { fail_json("queue", "newCommandQueue returned nil", result_path.c_str()); return 1; }
    const size_t bytes = sizeof(float) * CUDA4AS_ELEMENT_COUNT;
    id<MTLBuffer> a = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!a || !b || !c) { fail_json("allocation", "newBufferWithLength failed", result_path.c_str()); return 1; }
    memcpy(a.contents, ha.data(), bytes); memcpy(b.contents, hb.data(), bytes); memset(c.contents, 0, bytes);
    [a didModifyRange:NSMakeRange(0, bytes)]; [b didModifyRange:NSMakeRange(0, bytes)];

    NSString *libraryPath = [NSString stringWithFormat:@"%s/metallib/%s.metallib", work.c_str(), CUDA4AS_KERNEL_NAME];
    NSError *libraryError = nil;
    id<MTLLibrary> library = [device newLibraryWithFile:libraryPath error:&libraryError];
    if (!library) { fail_json("aot_load", libraryError.localizedDescription.UTF8String ?: "newLibraryWithFile failed", result_path.c_str()); return 1; }
    id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:CUDA4AS_KERNEL_NAME]];
    if (!fn) { fail_json("aot_load", "kernel entry point was not found in metallib", result_path.c_str()); return 1; }
    NSError *pipelineError = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&pipelineError];
    if (!pipeline) { fail_json("pipeline", pipelineError.localizedDescription.UTF8String ?: "pipeline creation failed", result_path.c_str()); return 1; }
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command) { fail_json("command_buffer", "new command buffer failed", result_path.c_str()); return 1; }
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a offset:0 atIndex:0]; [encoder setBuffer:b offset:0 atIndex:1]; [encoder setBuffer:c offset:0 atIndex:2];
    int n = (int)CUDA4AS_ELEMENT_COUNT;
    [encoder setBytes:&n length:sizeof(n) atIndex:3];
    MTLSize grid = MTLSizeMake(CUDA4AS_ELEMENT_COUNT, 1, 1);
    MTLSize group = MTLSizeMake(CUDA4AS_BLOCK_X, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      fail_json("launch", command.error.localizedDescription.UTF8String ?: "Metal command buffer did not complete", result_path.c_str()); return 1;
    }
    memcpy(hc.data(), c.contents, bytes);
    uint64_t mismatches = 0;
    for (uint32_t i = 0; i < CUDA4AS_ELEMENT_COUNT; ++i) if (hc[i] != ha[i] + hb[i]) ++mismatches;
    std::string actual = hex_digest(reinterpret_cast<const uint8_t *>(hc.data()), bytes);
    const auto t1 = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    if (mismatches || actual != CUDA4AS_EXPECTED_OUTPUT_SHA256 || !write_file(output_path.c_str(), hc.data(), bytes)) {
      std::ofstream f(result_path);
      f << "{\n  \"schema\": \"cuda4as-m2a-result-v1\",\n  \"classification\": \"FAIL\",\n  \"failed_stage\": \"validation\",\n  \"device\": {\"name\": \"" << name << "\", \"registry_id\": " << registry << "},\n  \"output\": {\"bytes\": " << bytes << ", \"sha256\": \"" << actual << "\", \"expected_sha256\": \"" << CUDA4AS_EXPECTED_OUTPUT_SHA256 << "\", \"mismatches\": " << mismatches << "},\n  \"cpu_fallback\": false\n}\n";
      return 1;
    }
    std::ofstream f(result_path);
    f << "{\n  \"schema\": \"cuda4as-m2a-result-v1\",\n  \"classification\": \"PASS_GPU\",\n  \"device\": {\"name\": \"" << name << "\", \"registry_id\": " << registry << ", \"route\": \"apple_gpu\"},\n  \"output\": {\"bytes\": " << bytes << ", \"sha256\": \"" << actual << "\", \"expected_sha256\": \"" << CUDA4AS_EXPECTED_OUTPUT_SHA256 << "\", \"mismatches\": 0},\n  \"launch\": {\"grid_x\": " << CUDA4AS_ELEMENT_COUNT << ", \"block_x\": " << CUDA4AS_BLOCK_X << ", \"completed\": true},\n  \"stages\": {\"allocation\": \"PASS\", \"h2d\": \"PASS\", \"aot_load\": \"PASS\", \"launch\": \"PASS\", \"d2h\": \"PASS\", \"validation\": \"PASS\"},\n  \"cpu_fallback\": false,\n  \"diagnostic_elapsed_ms\": " << elapsed_ms << "\n}\n";
  }
  return 0;
}
