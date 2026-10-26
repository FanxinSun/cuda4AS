# NVIDIA/cuda-samples on tesla-t4-sm75

Commit `5443602d89ed99aede2e4b7bf329daddeadb320e`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (0): none

## waived (8)

ran and declined: the sample's own check says this GPU lacks a feature (exit code 2 is the CUDA samples' convention); rerun on a newer GPU

- `cpp/0_Introduction/simpleAttributes` [ok on a100-sm80]
- `cpp/0_Introduction/simpleP2P`
- `cpp/2_Concepts_and_Techniques/streamOrderedAllocationP2P`
- `cpp/3_CUDA_Features/bf16TensorCoreGemm` [ok on a100-sm80]
- `cpp/3_CUDA_Features/cudaCompressibleMemory` [ok on a100-sm80]
- `cpp/3_CUDA_Features/dmmaTensorCoreGemm` [ok on a100-sm80]
- `cpp/3_CUDA_Features/tf32TensorCoreGemm` [ok on a100-sm80]
- `cpp/4_CUDA_Libraries/conjugateGradientMultiDeviceCG`

## not-tested (13)

needs hardware, libraries or a toolkit the bench host does not have (2 GPUs, OpenGL/EGL, a newer CUDA); no verdict about the program

- `cpp/0_Introduction/simpleTexture3D` [same result in 1 further session(s)]
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU` [same result in 1 further session(s)]
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop` [same result in 1 further session(s)]
- `cpp/2_Concepts_and_Techniques/FunctionPointers` [same result in 1 further session(s)]
- `cpp/2_Concepts_and_Techniques/boxFilter` [same result in 1 further session(s)]
- `cpp/2_Concepts_and_Techniques/imageDenoising` [same result in 1 further session(s)]
- `cpp/2_Concepts_and_Techniques/particles` [same result in 1 further session(s)]
- `cpp/3_CUDA_Features/bindlessTexture` [same result in 1 further session(s)]
- `cpp/3_CUDA_Features/localityDomains` [same result in 1 further session(s)] -- needs a CUDA toolkit newer than Colab's 13.0 (the samples target 13.4)
- `cpp/3_CUDA_Features/localityDomainsDrv` [same result in 1 further session(s)] -- needs a CUDA toolkit newer than Colab's 13.0: CUmemLocation.localized and CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL do not exist in 13.0
- `cpp/4_CUDA_Libraries/cudaNvSci` [same result in 1 further session(s)]
- `cpp/4_CUDA_Libraries/oceanFFT` [same result in 1 further session(s)]
- `cpp/4_CUDA_Libraries/randomFog` [same result in 1 further session(s)]

## reference (35)

ran clean, two runs agree: its stdout SHA-256 is a usable reference

- `cpp/0_Introduction/matrixMulDynlinkJIT` `450fbdc808a9212e`
- `cpp/0_Introduction/simpleAWBarrier` `8137ddb2141b684d`
- `cpp/0_Introduction/simpleCooperativeGroups` `a8d0520a69f89488`
- `cpp/0_Introduction/simpleDrvRuntime` `168cca287584cb25`
- `cpp/0_Introduction/simpleMPI` `49b5090612fca423`
- `cpp/0_Introduction/simpleMultiGPU` `4e1eb94e157cf19c`
- `cpp/0_Introduction/simpleVoteIntrinsics` `4079f468e0152c7b`
- `cpp/0_Introduction/simpleZeroCopy` `a95d840765ac7bed`
- `cpp/0_Introduction/systemWideAtomics` `7c030f0e31a95d32`
- `cpp/0_Introduction/vectorAdd` `b306d707d9dfe4da`
- `cpp/0_Introduction/vectorAddDrv` `2d756e974798f2e4`
- `cpp/0_Introduction/vectorAddMMAP` `e478fe6090c1e04e`
- `cpp/0_Introduction/vectorAdd_nvrtc` `bf8ce1c8c4a44db9`
- `cpp/1_Utilities/deviceQuery` `bcc0d527d1d5a521`
- `cpp/1_Utilities/deviceQueryDrv` `97944da3d80af97b`
- `cpp/1_Utilities/topologyQuery` `d467edd77467b688`
- `cpp/2_Concepts_and_Techniques/inlinePTX` `5f0b031c312c9976`
- `cpp/3_CUDA_Features/binaryPartitionCG` `73d99d313780b012`
- `cpp/3_CUDA_Features/cdpBezierTessellation` `a8e6bdeed479a24c`
- `cpp/3_CUDA_Features/cdpQuadtree` `901436af247b8b90`
- `cpp/3_CUDA_Features/cdpSimpleQuicksort` `15b4d66e9af79ee6`
- `cpp/3_CUDA_Features/dmabufInterop` `73ad81102ea345ae`
- `cpp/3_CUDA_Features/graphConditionalNodes` `dd5949caa461db85`
- `cpp/3_CUDA_Features/graphMemoryFootprint` `9d54efc2e529a940`
- `cpp/3_CUDA_Features/graphMemoryNodes` `9c0ce5f06bad1ced`
- `cpp/3_CUDA_Features/newdelete` `9597a73628862075`
- `cpp/3_CUDA_Features/simpleCudaGraphs` `3da3cd8c7a49633d`
- `cpp/3_CUDA_Features/warpAggregatedAtomicsCG` `5f412abfdc82c724`
- `cpp/4_CUDA_Libraries/conjugateGradientCudaGraphs` `a9d4a718f9fb2a72`
- `cpp/4_CUDA_Libraries/cubDeviceFind` `bcace28a22d297ff`
- `cpp/4_CUDA_Libraries/cubDeviceSegmentedScan` `0da7805d1079669d`
- `cpp/4_CUDA_Libraries/cubDeviceTransform` `f29baf3d372f5d92`
- `cpp/4_CUDA_Libraries/jitLto` `e7e9096354406501`
- `cpp/4_CUDA_Libraries/libcuxxMdspan` `c1ddb98a9cf9d1a3`
- `cpp/4_CUDA_Libraries/libcuxxRandom` `eb06aea355fcfa22`

## pass-only (59)

ran clean but output varies between runs (timings): judge by exit status only

- `cpp/0_Introduction/UnifiedMemoryStreams`
- `cpp/0_Introduction/asyncAPI`
- `cpp/0_Introduction/clock`
- `cpp/0_Introduction/cudaOpenMP` [output differs between Colab sessions, so not a reference]
- `cpp/0_Introduction/fp16ScalarProduct`
- `cpp/0_Introduction/matrixMul`
- `cpp/0_Introduction/matrixMulDrv`
- `cpp/0_Introduction/simpleAssert` [failure words in the output are expected: triggers a device-side assert on purpose and exits 0 ('failed as expected')]
- `cpp/0_Introduction/simpleAtomicIntrinsics`
- `cpp/0_Introduction/simpleCallback` [output differs between Colab sessions, so not a reference]
- `cpp/0_Introduction/simpleCubemapTexture`
- `cpp/0_Introduction/simpleIPC`
- `cpp/0_Introduction/simpleLayeredTexture`
- `cpp/0_Introduction/simpleMultiCopy`
- `cpp/0_Introduction/simplePitchLinearTexture`
- `cpp/0_Introduction/simplePrintf`
- `cpp/0_Introduction/simpleStreams`
- `cpp/0_Introduction/simpleSurfaceWrite`
- `cpp/0_Introduction/simpleTexture`
- `cpp/0_Introduction/simpleTextureDrv`
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP`
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ`
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiP`
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ`
- `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP`
- `cpp/2_Concepts_and_Techniques/convolutionSeparable`
- `cpp/2_Concepts_and_Techniques/convolutionTexture`
- `cpp/2_Concepts_and_Techniques/dct8x8`
- `cpp/2_Concepts_and_Techniques/eigenvalues`
- `cpp/2_Concepts_and_Techniques/histogram`
- `cpp/2_Concepts_and_Techniques/interval`
- `cpp/2_Concepts_and_Techniques/radixSortThrust`
- `cpp/2_Concepts_and_Techniques/reduction`
- `cpp/2_Concepts_and_Techniques/reductionMultiBlockCG`
- `cpp/2_Concepts_and_Techniques/scalarProd`
- `cpp/2_Concepts_and_Techniques/scan`
- `cpp/2_Concepts_and_Techniques/segmentationTreeThrust`
- `cpp/2_Concepts_and_Techniques/shfl_scan`
- `cpp/2_Concepts_and_Techniques/sortingNetworks`
- `cpp/2_Concepts_and_Techniques/streamOrderedAllocation`
- `cpp/2_Concepts_and_Techniques/streamOrderedAllocationIPC`
- `cpp/2_Concepts_and_Techniques/threadFenceReduction`
- `cpp/2_Concepts_and_Techniques/threadMigration`
- `cpp/3_CUDA_Features/StreamPriorities`
- `cpp/3_CUDA_Features/cdpAdvancedQuicksort`
- `cpp/3_CUDA_Features/cdpSimplePrint`
- `cpp/3_CUDA_Features/cudaTensorCoreGemm`
- `cpp/3_CUDA_Features/globalToShmemAsyncCopy`
- `cpp/3_CUDA_Features/immaTensorCoreGemm`
- `cpp/3_CUDA_Features/jacobiCudaGraphs`
- `cpp/3_CUDA_Features/memMapIPCDrv`
- `cpp/3_CUDA_Features/ptxjit`
- `cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG`
- `cpp/4_CUDA_Libraries/lineOfSight`
- `cpp/6_Performance/LargeKernelParameter`
- `cpp/6_Performance/UnifiedMemoryPerf`
- `cpp/6_Performance/alignedTypes`
- `cpp/6_Performance/cudaGraphsPerfScaling`
- `cpp/6_Performance/transpose`

## skipped (1)

marked skip upstream

- `cpp/0_Introduction/simpleCUDA2GL` [same result in 1 further session(s)]
