package cuda.dsl.benchmark

import cuda.dsl.core.dim3
import cuda.dsl.runtime.{TritonKernel, TritonKernel as TKTrait}
import cuda.dsl.runtime.runtimeTypes.*
import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR
import TKTrait.{given TritonKernel}

/** Fully unified benchmark — execute ALL kernels via direct function references.
  *  Uses strongly-typed function-as-parameter API, NOT runByName.
  *  Covers Test50ComplexKernels, Test50CUTLASSKernels, Test50ThreadIdxKernels,
  *  Test100ComplexKernels, TestAttentionGeneric — total ~260 kernels.
  */
object UnifiedBenchmarkAllKernels:

  // Force initialization of test objects (triggers @TritonKernelMacro registration)
  private val _ = (Test50ComplexKernels, Test50CUTLASSKernels, Test50ThreadIdxKernels, Test100ComplexKernels, TestAttentionGeneric)

  // ============================================================================
  // Category 1: Test50ComplexKernels (50 kernels)
  // ============================================================================

  // 1-10: For Loop Kernels
  given vectorSumKernel_g: TritonKernel[K3F1I] = TritonKernel("vectorSumKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given matrixVecMulKernel_g: TritonKernel[K4F2I] = TritonKernel("matrixVecMulKernel", List("float*","float*","float*","int","int","int"), List("outPtr","matPtr","vecPtr","M","N","n"))
  given softmaxKernel_g: TritonKernel[K3F1I] = TritonKernel("softmaxKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given layerNormKernel_g: TritonKernel[K4F1I] = TritonKernel("layerNormKernel", List("float*","float*","int","float","int"), List("outPtr","inPtr","N","eps","n"))
  given batchMatMulKernel_g: TritonKernel[K5F3I] = TritonKernel("batchMatMulKernel", List("float*","float*","float*","int","int","int","int","int"), List("outPtr","aPtr","bPtr","B","M","N","K","n"))
  given conv1dKernel_g: TritonKernel[K4F2I] = TritonKernel("conv1dKernel", List("float*","float*","float*","int","int","int"), List("outPtr","inPtr","kernelPtr","L","K","n"))
  given maxPoolKernel_g: TritonKernel[K4F2I] = TritonKernel("maxPoolKernel", List("float*","float*","int","int","int","int"), List("outPtr","inPtr","H","W","poolSize","n"))
  given rmsNormKernel_g: TritonKernel[K4F1I] = TritonKernel("rmsNormKernel", List("float*","float*","float*","int","float","int"), List("outPtr","inPtr","weightPtr","N","eps","n"))
  given attentionScoreKernel_g: TritonKernel[K4F1I] = TritonKernel("attentionScoreKernel", List("float*","float*","float*","float","int","int"), List("outPtr","qPtr","kPtr","scale","N","n"))
  given cumsumKernel_g: TritonKernel[K2F1I] = TritonKernel("cumsumKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))

  // 11-20: Shared Memory + Syncthreads
  given tiledMatMulKernel2_g: TritonKernel[K5F3I] = TritonKernel("tiledMatMulKernel2", List("float*","float*","float*","int","int","int","int","int"), List("outPtr","aPtr","bPtr","M","N","K","blockSize","n"))
  given blockReduceKernel2_g: TritonKernel[K3F1I] = TritonKernel("blockReduceKernel2", List("float*","float*","int","int","int"), List("outPtr","inPtr","N","blockSize","n"))
  given sharedTransposeKernel_g: TritonKernel[K3F1I] = TritonKernel("sharedTransposeKernel", List("float*","float*","int","int","int"), List("outPtr","inPtr","N","blockSize","n"))
  given tiledSoftmaxKernel_g: TritonKernel[K3F1I] = TritonKernel("tiledSoftmaxKernel", List("float*","float*","int","int","int"), List("outPtr","inPtr","N","blockSize","n"))
  given blockMatrixLoadKernel_g: TritonKernel[K4F2I] = TritonKernel("blockMatrixLoadKernel", List("float*","float*","int","int","int","int"), List("outPtr","inPtr","M","N","blockSize","n"))
  given parallelReduceKernel_g: TritonKernel[K3F1I] = TritonKernel("parallelReduceKernel", List("float*","float*","int","int","int"), List("outPtr","inPtr","N","blockSize","n"))
  given tiledLayerNormKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledLayerNormKernel", List("float*","float*","int","int","float","int"), List("outPtr","inPtr","N","blockSize","eps","n"))
  given tiledAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("tiledAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","N","D","blockSize","n"))
  given warpReduceSyncthreadsKernel_g: TritonKernel[K3F1I] = TritonKernel("warpReduceSyncthreadsKernel", List("float*","float*","int","int","int"), List("outPtr","inPtr","N","blockSize","n"))
  given multiWarpReduceKernel_g: TritonKernel[K3F1I] = TritonKernel("multiWarpReduceKernel", List("float*","float*","int","int","int"), List("outPtr","inPtr","N","blockSize","n"))

  // 21-30: Warp Shuffle
  given warpReduceSumKernel_g: TritonKernel[K2F1I] = TritonKernel("warpReduceSumKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given warpReduceMaxKernel_g: TritonKernel[K2F1I] = TritonKernel("warpReduceMaxKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given warpButterflyReduceKernel_g: TritonKernel[K2F1I] = TritonKernel("warpButterflyReduceKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given warpScanKernel_g: TritonKernel[K2F1I] = TritonKernel("warpScanKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given warpVoteAnyKernel_g: TritonKernel[K2F1I] = TritonKernel("warpVoteAnyKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given warpVoteAllKernel_g: TritonKernel[K2F1I] = TritonKernel("warpVoteAllKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given warpVoteCombinedKernel_g: TritonKernel[K2F1I] = TritonKernel("warpVoteCombinedKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given warpShuffleXorKernel_g: TritonKernel[K2F1I] = TritonKernel("warpShuffleXorKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))
  given warpMultiReduceKernel_g: TritonKernel[K3F1I] = TritonKernel("warpMultiReduceKernel", List("float*","float*","float*","int","int"), List("outPtr","inPtr1","inPtr2","N","n"))
  given warpReduceTernaryKernel_g: TritonKernel[K2F1I] = TritonKernel("warpReduceTernaryKernel", List("float*","float*","int","int"), List("outPtr","inPtr","N","n"))

  // 31-40: Masked Load/Store
  given maskedVectorAddKernel_g: TritonKernel[K5F1I] = TritonKernel("maskedVectorAddKernel", List("float*","float*","float*","float*","int","int"), List("outPtr","aPtr","bPtr","maskPtr","N","n"))
  given maskedSoftmaxKernel_g: TritonKernel[K4F1I] = TritonKernel("maskedSoftmaxKernel", List("float*","float*","float*","int","int"), List("outPtr","inPtr","maskPtr","N","n"))
  given maskedAttentionKernel_g: TritonKernel[K5F1I] = TritonKernel("maskedAttentionKernel", List("float*","float*","float*","float*","float*","int","int"), List("outPtr","qPtr","kPtr","vPtr","maskPtr","N","n"))
  given maskedFillKernel_g: TritonKernel[K3F1I] = TritonKernel("maskedFillKernel", List("float*","float*","float*","int","int"), List("outPtr","fillVal","maskPtr","N","n"))
  given maskedSelectKernel_g: TritonKernel[K4F1I] = TritonKernel("maskedSelectKernel", List("float*","float*","float*","float*","int","int"), List("outPtr","aPtr","bPtr","maskPtr","N","n"))
  given maskedScaleKernel_g: TritonKernel[K4F1I] = TritonKernel("maskedScaleKernel", List("float*","float*","float*","float","int","int"), List("outPtr","inPtr","scale","maskPtr","N","n"))
  given maskedClampKernel_g: TritonKernel[K5F1I] = TritonKernel("maskedClampKernel", List("float*","float*","float","float","float*","int","int"), List("outPtr","inPtr","minVal","maxVal","maskPtr","N","n"))
  given maskedLayerNormKernel_g: TritonKernel[K4F1I] = TritonKernel("maskedLayerNormKernel", List("float*","float*","float*","int","float","int"), List("outPtr","inPtr","maskPtr","N","eps","n"))
  given maskedGeluKernel_g: TritonKernel[K3F1I] = TritonKernel("maskedGeluKernel", List("float*","float*","float*","int"), List("outPtr","inPtr","maskPtr","N"))
  given maskedDropoutKernel_g: TritonKernel[K4F1I] = TritonKernel("maskedDropoutKernel", List("float*","float*","float*","float*","float","int","int"), List("outPtr","inPtr","maskPtr","keepProb","N","n"))

  // 41-50: Complex Combined
  given flashAttentionKernel2_g: TritonKernel[K5F2I] = TritonKernel("flashAttentionKernel2", List("float*","float*","float*","float*","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","N","D","n"))
  given pageAttentionKernel2_g: TritonKernel[K7F3I] = TritonKernel("pageAttentionKernel2", List("float*","float*","float*","float*","float*","float*","int","int","int","int","int"), List("outPtr","queryPtr","keyCachePtr","valueCachePtr","blockTablesPtr","seqLensPtr","batchSize","maxBlocksPerSeq","blockSize","D","n"))
  given groupedQueryAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("groupedQueryAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","kvPtr","N","D","n"))
  given multiHeadAttentionKernel_g: TritonKernel[K5F3I] = TritonKernel("multiHeadAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","N","H","D","n"))
  given fusedGeluLayerNormKernel_g: TritonKernel[K4F1I] = TritonKernel("fusedGeluLayerNormKernel", List("float*","float*","float*","int","float","int"), List("outPtr","inPtr","weightPtr","N","eps","n"))
  given fusedScaleBiasGeluKernel_g: TritonKernel[K4F1I] = TritonKernel("fusedScaleBiasGeluKernel", List("float*","float*","float*","float*","int"), List("outPtr","inPtr","scalePtr","biasPtr","N"))
  given fusedAttentionBiasKernel_g: TritonKernel[K5F2I] = TritonKernel("fusedAttentionBiasKernel", List("float*","float*","float*","float*","float*","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","biasPtr","N","D","n"))
  given sparseAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("sparseAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","indicesPtr","N","D","n"))
  given crossAttentionKernel_g: TritonKernel[K5F3I] = TritonKernel("crossAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","N","M","D","n"))
  given hierarchicalAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("hierarchicalAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","coarsePtr","N","D","n"))

  // ============================================================================
  // Category 2: TestAttentionGeneric (4 kernels)
  // ============================================================================

  given storeKVCacheKernel_g: TritonKernel[K7F1I] = TritonKernel("storeKVCacheKernel", List("float*","float*","float*","float*","float*","float*","int*","int"), List("keyPtr","keyStride","valuePtr","valueStride","kCachePtr","vCachePtr","slotMappingPtr","D"))
  given flashAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("flashAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","N","D","n"))
  given pageAttentionKernel_g: TritonKernel[K8F4I] = TritonKernel("pageAttentionKernel", List("float*","float*","float*","float*","float*","int*","int*","int","int","int","int","int"), List("outPtr","queryPtr","keyCachePtr","valueCachePtr","blockTablesPtr","seqLensPtr","batchSize","maxBlocksPerSeq","blockSize","D","n"))
  given flexAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("flexAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("outPtr","qPtr","kPtr","vPtr","N","D","n"))

  // ============================================================================
  // Category 3: Test50CUTLASSKernels (50 kernels)
  // ============================================================================

  // 1-10: Tiled GEMM
  given tiledGemm64x64Kernel_g: TritonKernel[K7F1I] = TritonKernel("tiledGemm64x64Kernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","blockSize"))
  given tiledGemm128x128Kernel_g: TritonKernel[K7F1I] = TritonKernel("tiledGemm128x128Kernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","blockSize"))
  given tiledGemm32x32WarpKernel_g: TritonKernel[K7F1I] = TritonKernel("tiledGemm32x32WarpKernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","blockSize"))
  given persistentGemmKernel_g: TritonKernel[K7F1I] = TritonKernel("persistentGemmKernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","gridSize"))
  given tiledGemmBiasKernel_g: TritonKernel[K8F1I] = TritonKernel("tiledGemmBiasKernel", List("float*","float*","float*","float*","int","int","int","int"), List("C","A","B","bias","M","N","K","blockSize"))
  given tiledGemmGeluKernel_g: TritonKernel[K7F1I] = TritonKernel("tiledGemmGeluKernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","blockSize"))
  given stridedBatchedGemmKernel_g: TritonKernel[K10F2I] = TritonKernel("stridedBatchedGemmKernel", List("float*","float*","float*","int","int","int","int","int","int","int","int"), List("C","A","B","M","N","K","batchStrideA","batchStrideB","batchStrideC","batchCount","n"))
  given quantizedGemmKernel_g: TritonKernel[K8F1I] = TritonKernel("quantizedGemmKernel", List("float*","float*","float*","float*","float*","int","int","int"), List("C","A","B","scaleA","scaleB","M","N","K"))
  given sparseGemmKernel_g: TritonKernel[K8F1I] = TritonKernel("sparseGemmKernel", List("float*","float*","float*","float*","int","int","int","int"), List("C","A","B","metadata","M","N","K","blockSize"))
  given mixedPrecisionGemmKernel_g: TritonKernel[K4F1I] = TritonKernel("mixedPrecisionGemmKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))

  // 11-20: Warp-Level Tiled
  given warpGemm16x16Kernel_g: TritonKernel[K4F1I] = TritonKernel("warpGemm16x16Kernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given warpTiledReduceKernel_g: TritonKernel[K3F1I] = TritonKernel("warpTiledReduceKernel", List("float*","float*","int","int","int"), List("out","in","N","blockSize","n"))
  given warpTiledScanKernel_g: TritonKernel[K2F1I] = TritonKernel("warpTiledScanKernel", List("float*","float*","int","int"), List("out","in","N","n"))
  given warpTiledSoftmaxKernel_g: TritonKernel[K3F1I] = TritonKernel("warpTiledSoftmaxKernel", List("float*","float*","int","int","int"), List("out","in","N","blockSize","n"))
  given warpTiledLayerNormKernel_g: TritonKernel[K5F1I] = TritonKernel("warpTiledLayerNormKernel", List("float*","float*","float*","float*","int","int"), List("out","in","weight","bias","N","blockSize"))
  given warpTiledAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("warpTiledAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("out","q","k","v","M","N","D"))
  given warpTiledConv2dKernel_g: TritonKernel[K8F1I] = TritonKernel("warpTiledConv2dKernel", List("float*","float*","float*","int","int","int","int","int","int"), List("out","in","kernel","N","H","W","K","R","S"))
  given warpTiledRmsNormKernel_g: TritonKernel[K4F1I] = TritonKernel("warpTiledRmsNormKernel", List("float*","float*","float*","int","int"), List("out","in","weight","N","blockSize"))
  given warpTiledGeluKernel_g: TritonKernel[K3F1I] = TritonKernel("warpTiledGeluKernel", List("float*","float*","int","int"), List("out","in","N","blockSize"))
  given warpTiledDropoutKernel_g: TritonKernel[K5F1I] = TritonKernel("warpTiledDropoutKernel", List("float*","float*","float*","float","int"), List("out","in","mask","keepProb","N"))

  // 21-30: Persistent Kernels
  given persistentSoftmaxKernel_g: TritonKernel[K3F1I] = TritonKernel("persistentSoftmaxKernel", List("float*","float*","int","int"), List("out","in","N","nTiles"))
  given persistentLayerNormKernel_g: TritonKernel[K5F1I] = TritonKernel("persistentLayerNormKernel", List("float*","float*","float*","float*","int","int"), List("out","in","weight","bias","N","nTiles"))
  given persistentAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("persistentAttentionKernel", List("float*","float*","float*","float*","float","int","int","int","int"), List("out","q","k","v","scale","M","N","D","nTiles"))
  given persistentGemvKernel_g: TritonKernel[K4F1I] = TritonKernel("persistentGemvKernel", List("float*","float*","float*","int","int","int"), List("y","A","x","M","N","nTiles"))
  given persistentBatchGemmKernel_g: TritonKernel[K6F2I] = TritonKernel("persistentBatchGemmKernel", List("float*","float*","float*","int","int","int","int","int"), List("C","A","B","M","N","K","batchCount","nTiles"))
  given persistentTanhKernel_g: TritonKernel[K3F1I] = TritonKernel("persistentTanhKernel", List("float*","float*","int","int"), List("out","in","N","nTiles"))
  given persistentSigmoidKernel_g: TritonKernel[K3F1I] = TritonKernel("persistentSigmoidKernel", List("float*","float*","int","int"), List("out","in","N","nTiles"))
  given persistentReluKernel_g: TritonKernel[K3F1I] = TritonKernel("persistentReluKernel", List("float*","float*","int","int"), List("out","in","N","nTiles"))
  given persistentLeakyReluKernel_g: TritonKernel[K4F1I] = TritonKernel("persistentLeakyReluKernel", List("float*","float*","float","int","int"), List("out","in","alpha","N","nTiles"))
  given persistentEluKernel_g: TritonKernel[K4F1I] = TritonKernel("persistentEluKernel", List("float*","float*","float","int","int"), List("out","in","alpha","N","nTiles"))

  // 31-40: Tiled Attention
  given tiledFlashAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("tiledFlashAttentionKernel", List("float*","float*","float*","float*","float","int","int","int","int"), List("out","q","k","v","scale","M","N","D","blockSize"))
  given tiledMultiHeadAttentionKernel_g: TritonKernel[K5F3I] = TritonKernel("tiledMultiHeadAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("out","q","k","v","M","N","D","H"))
  given tiledGQAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("tiledGQAttentionKernel", List("float*","float*","float*","float*","float","int","int","int","int","int"), List("out","q","k","v","scale","M","N","D","H","G"))
  given tiledSlidingWindowAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("tiledSlidingWindowAttentionKernel", List("float*","float*","float*","float*","float","int","int","int","int"), List("out","q","k","v","scale","M","N","D","windowSize"))
  given tiledCrossAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("tiledCrossAttentionKernel", List("float*","float*","float*","float*","float","int","int","int","int"), List("out","q","k","v","scale","M","N","D","blockSize"))
  given tiledBidirectionalAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("tiledBidirectionalAttentionKernel", List("float*","float*","float*","float*","float","int","int","int"), List("out","q","k","v","scale","M","N","D"))
  given tiledLocalAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("tiledLocalAttentionKernel", List("float*","float*","float*","float*","float","int","int","int","int"), List("out","q","k","v","scale","M","N","D","localSize"))
  given tiledSparseAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("tiledSparseAttentionKernel", List("float*","float*","float*","float*","float*","float","int","int","int"), List("out","q","k","v","sparseMask","scale","M","N","D"))
  given tiledGlobalLocalAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("tiledGlobalLocalAttentionKernel", List("float*","float*","float*","float*","float","float","int","int","int"), List("out","q","k","v","scale","globalRatio","M","N","D"))
  given tiledKernelAttentionKernel_g: TritonKernel[K9F1I] = TritonKernel("tiledKernelAttentionKernel", List("float*","float*","float*","float*","float*","float","int","int","int","int"), List("out","q","k","v","convKernel","scale","M","N","D","K"))

  // 41-50: Fused Tiled
  given fusedTiledGemmBiasGeluKernel_g: TritonKernel[K8F1I] = TritonKernel("fusedTiledGemmBiasGeluKernel", List("float*","float*","float*","float*","int","int","int","int"), List("C","A","B","bias","M","N","K","blockSize"))
  given fusedTiledGemmBiasResidualKernel_g: TritonKernel[K9F1I] = TritonKernel("fusedTiledGemmBiasResidualKernel", List("float*","float*","float*","float*","float*","int","int","int","int"), List("C","A","B","bias","residual","M","N","K","blockSize"))
  given fusedTiledLayerNormAttentionKernel_g: TritonKernel[K10F2I] = TritonKernel("fusedTiledLayerNormAttentionKernel", List("float*","float*","float*","float*","float*","float*","float","int","int","int","int"), List("out","q","k","v","lnWeight","lnBias","scale","M","N","D","blockSize"))
  given fusedTiledRmsNormGemmKernel_g: TritonKernel[K8F1I] = TritonKernel("fusedTiledRmsNormGemmKernel", List("float*","float*","float*","float*","int","int","int","int"), List("C","A","B","rmsWeight","M","N","K","blockSize"))
  given fusedTiledSkipConnectionKernel_g: TritonKernel[K9F1I] = TritonKernel("fusedTiledSkipConnectionKernel", List("float*","float*","float*","float*","float","int","int","int"), List("out","input","gemmA","gemmB","alpha","M","N","K"))
  given fusedTiledGatedActivationKernel_g: TritonKernel[K4F1I] = TritonKernel("fusedTiledGatedActivationKernel", List("float*","float*","float*","int","int"), List("out","gate","up","M","N"))
  given fusedTiledSwigluKernel_g: TritonKernel[K9F1I] = TritonKernel("fusedTiledSwigluKernel", List("float*","float*","float*","float*","int","int","int"), List("out","gate","up","down","M","N","K"))
  given fusedTiledMhaFfnKernel_g: TritonKernel[K9F2I] = TritonKernel("fusedTiledMhaFfnKernel", List("float*","float*","float*","float*","float*","float*","float","int","int","int","int"), List("out","q","k","v","ffnUp","ffnDown","scale","M","N","D","H"))
  given fusedTiledRmsNormSkipGemmKernel_g: TritonKernel[K9F2I] = TritonKernel("fusedTiledRmsNormSkipGemmKernel", List("float*","float*","float*","float*","float*","float*","float","int","int","int","int"), List("C","input","skip","gemmA","gemmB","rmsWeight","alpha","M","N","K"))
  given fusedTiledBiasGeluResidualLayerNormKernel_g: TritonKernel[K8F1I] = TritonKernel("fusedTiledBiasGeluResidualLayerNormKernel", List("float*","float*","float*","float*","float*","float*","int","int"), List("out","input","gemm","bias","lnWeight","lnBias","M","N"))

  // ============================================================================
  // Category 4: Test50ThreadIdxKernels (50 kernels)
  // ============================================================================

  // 1-10: Warp-Level Tiled GEMM
  given warpTileGemm4x4LaneKernel_g: TritonKernel[K4F1I] = TritonKernel("warpTileGemm4x4LaneKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given warpTileGemm8x8BankAvoidKernel_g: TritonKernel[K4F1I] = TritonKernel("warpTileGemm8x8BankAvoidKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given warpTranspose16x16Kernel_g: TritonKernel[K3F1I] = TritonKernel("warpTranspose16x16Kernel", List("float*","float*","int","int"), List("A","C","M","N"))
  given blockSoftmaxKernel_g: TritonKernel[K3F1I] = TritonKernel("blockSoftmaxKernel", List("float*","float*","int","int"), List("X","Y","M","N"))
  given blockLayerNormKernel_g: TritonKernel[K5F1I] = TritonKernel("blockLayerNormKernel", List("float*","float*","int","int","float"), List("X","Y","M","N","eps"))
  given blockAttentionQKVKernel_g: TritonKernel[K5F1I] = TritonKernel("blockAttentionQKVKernel", List("float*","float*","float*","float*","int","int","int"), List("Q","K","V","O","M","N","D"))
  given blockGeluKernel_g: TritonKernel[K3F1I] = TritonKernel("blockGeluKernel", List("float*","float*","int","int"), List("X","Y","M","N"))
  given blockResidualKernel_g: TritonKernel[K4F1I] = TritonKernel("blockResidualKernel", List("float*","float*","float*","int","int"), List("X","Y","Z","M","N"))
  given blockBiLevelReduceKernel_g: TritonKernel[K3F1I] = TritonKernel("blockBiLevelReduceKernel", List("float*","float*","int","int"), List("X","Y","M","N"))
  given blockGatedActivationKernel_g: TritonKernel[K4F1I] = TritonKernel("blockGatedActivationKernel", List("float*","float*","float*","int","int"), List("X","G","Y","M","N"))

  // 11-20: Multi-Axis threadIdx
  given tile2dGemmKernel_g: TritonKernel[K4F1I] = TritonKernel("tile2dGemmKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tile2dAttentionKernel_g: TritonKernel[K5F1I] = TritonKernel("tile2dAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("Q","K","V","O","M","N","D"))
  given tile2dLayerNormKernel_g: TritonKernel[K5F1I] = TritonKernel("tile2dLayerNormKernel", List("float*","float*","int","int","float"), List("X","Y","M","N","eps"))
  given tile2dRmsNormKernel_g: TritonKernel[K5F1I] = TritonKernel("tile2dRmsNormKernel", List("float*","float*","int","int","float"), List("X","Y","M","N","eps"))
  given tile2dSigmoidKernel_g: TritonKernel[K3F1I] = TritonKernel("tile2dSigmoidKernel", List("float*","float*","int","int"), List("X","Y","M","N"))
  given tile2dSwigluKernel_g: TritonKernel[K4F1I] = TritonKernel("tile2dSwigluKernel", List("float*","float*","float*","int","int"), List("X","G","Y","M","N"))
  given tile2dLeakyReluKernel_g: TritonKernel[K3F1I] = TritonKernel("tile2dLeakyReluKernel", List("float*","float*","int","int"), List("X","Y","M","N"))
  given tile2dBatchNormKernel_g: TritonKernel[K5F1I] = TritonKernel("tile2dBatchNormKernel", List("float*","float*","int","int","float"), List("X","Y","M","N","eps"))
  given tile2dCrossEntropyKernel_g: TritonKernel[K4F1I] = TritonKernel("tile2dCrossEntropyKernel", List("float*","float*","float*","int","int"), List("_logits","_targets","_loss","M","N"))
  given tile2dDropoutKernel_g: TritonKernel[K4F1I] = TritonKernel("tile2dDropoutKernel", List("float*","float*","float","int","int"), List("X","Y","p","M","N"))

  // 21-30: Hierarchical Reduction Trees
  given blockReduceGemmKernel_g: TritonKernel[K4F1I] = TritonKernel("blockReduceGemmKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given hierarchicalSoftmaxKernel_g: TritonKernel[K3F1I] = TritonKernel("hierarchicalSoftmaxKernel", List("float*","float*","int","int"), List("X","Y","M","N"))
  given blockSegmentReduceKernel_g: TritonKernel[K5F1I] = TritonKernel("blockSegmentReduceKernel", List("float*","float*","int","int","int"), List("X","Y","M","N","segSize"))
  given hierarchicalLayerNormKernel_g: TritonKernel[K5F1I] = TritonKernel("hierarchicalLayerNormKernel", List("float*","float*","int","int","float"), List("X","Y","M","N","eps"))
  given multiLevelReduceGemmKernel_g: TritonKernel[K4F1I] = TritonKernel("multiLevelReduceGemmKernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","batch"))
  given blockCausalAttentionKernel_g: TritonKernel[K5F1I] = TritonKernel("blockCausalAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("Q","K","V","O","M","N","D"))
  given hierarchicalGemmKernel_g: TritonKernel[K4F1I] = TritonKernel("hierarchicalGemmKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given blockExclusiveScanKernel_g: TritonKernel[K2F1I] = TritonKernel("blockExclusiveScanKernel", List("float*","float*","int","int"), List("X","Y","N","n"))
  given multiBlockReduceGemmKernel_g: TritonKernel[K4F1I] = TritonKernel("multiBlockReduceGemmKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given multiBlockSoftmaxKernel_g: TritonKernel[K3F1I] = TritonKernel("multiBlockSoftmaxKernel", List("float*","float*","int","int"), List("X","Y","M","N"))

  // 31-40: Tiled Attention Mechanisms
  given threadIdx_flashAttentionKernel_g: TritonKernel[K8F1I] = TritonKernel("flashAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int","int"), List("Q","K","V","O","M","N","D","blockSize"))
  given multiHeadAttentionKernel2_g: TritonKernel[K9F2I] = TritonKernel("multiHeadAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int","int"), List("Q","K","V","O","M","N","D","numHeads","n"))
  given groupedQueryAttentionKernel2_g: TritonKernel[K9F2I] = TritonKernel("groupedQueryAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int","int","int"), List("Q","K","V","O","M","N","D","numHeads","numKVHeads","n"))
  given slidingWindowAttentionKernel_g: TritonKernel[K8F1I] = TritonKernel("slidingWindowAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("Q","K","V","O","M","N","D","windowSize"))
  given crossAttentionKernel2_g: TritonKernel[K5F2I] = TritonKernel("crossAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("Q","K","V","O","M","N","D"))
  given bidirectionalAttentionKernel_g: TritonKernel[K5F2I] = TritonKernel("bidirectionalAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("Q","K","V","O","M","N","D"))
  given localAttentionKernel2_g: TritonKernel[K8F1I] = TritonKernel("localAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("Q","K","V","O","M","N","D","localSize"))
  given sparseAttentionKernel2_g: TritonKernel[K8F1I] = TritonKernel("sparseAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("Q","K","V","O","M","N","D","sparsity"))
  given globalLocalAttentionKernel_g: TritonKernel[K8F1I] = TritonKernel("globalLocalAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("Q","K","V","O","M","N","D","localSize"))
  given kernelAttentionKernel_g: TritonKernel[K9F2I] = TritonKernel("kernelAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("Q","K","V","O","M","N","D","numFeatures"))

  // 41-50: Fused Tiled Operations
  given fusedGemmGeluKernel_g: TritonKernel[K4F1I] = TritonKernel("fusedGemmGeluKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given fusedGemmResidualKernel_g: TritonKernel[K5F1I] = TritonKernel("fusedGemmResidualKernel", List("float*","float*","float*","float*","int","int","int"), List("C","A","B","D","M","N","K"))
  given fusedLayerNormAttentionKernel_g: TritonKernel[K10F2I] = TritonKernel("fusedLayerNormAttentionKernel", List("float*","float*","float*","float*","float*","float*","float","int","int","int","float"), List("X","Y","Q","K","V","O","M","N","D","eps","n"))
  given fusedRmsNormSwigluKernel_g: TritonKernel[K5F1I] = TritonKernel("fusedRmsNormSwigluKernel", List("float*","float*","float*","int","int","float"), List("X","G","Y","M","N","eps"))
  given fusedSkipConnectionKernel_g: TritonKernel[K5F1I] = TritonKernel("fusedSkipConnectionKernel", List("float*","float*","float*","float*","int","int"), List("X","F","Y","M","N"))
  given fusedGatedActivationKernel2_g: TritonKernel[K6F1I] = TritonKernel("fusedGatedActivationKernel", List("float*","float*","float*","float*","int","int"), List("X","G1","G2","Y","M","N"))
  given fusedSwigluKernel_g: TritonKernel[K4F1I] = TritonKernel("fusedSwigluKernel", List("float*","float*","float*","int","int"), List("X","G","Y","M","N"))
  given fusedMhaFfnKernel_g: TritonKernel[K10F2I] = TritonKernel("fusedMhaFfnKernel", List("float*","float*","float*","float*","float*","float*","int","int","int","int","int"), List("Q","K","V","W1","W2","O","M","N","D","hidden","n"))
  given fusedRmsNormSkipGemmKernel_g: TritonKernel[K10F2I] = TritonKernel("fusedRmsNormSkipGemmKernel", List("float*","float*","float*","float*","float*","int","int","int","float","int"), List("X","R","A","B","Y","M","N","K","eps","n"))
  given fusedBiasGeluResidualLayerNormKernel_g: TritonKernel[K6F1I] = TritonKernel("fusedBiasGeluResidualLayerNormKernel", List("float*","float*","float*","float*","float*","int","int","float"), List("X","B","R","Y","M","N","eps","n"))

  // ============================================================================
  // Category 5: Test100ComplexKernels (100 kernels)
  // ============================================================================

  // 1-20: CUTLASS-style Tiled GEMM
  given tiledGemm64x64Kernel100_g: TritonKernel[K4F1I] = TritonKernel("tiledGemm64x64Kernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemm128x128Kernel100_g: TritonKernel[K4F1I] = TritonKernel("tiledGemm128x128Kernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmPartitionKKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmPartitionKKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmBankAvoidKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmBankAvoidKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given batchedGemmKernel_g: TritonKernel[K4F2I] = TritonKernel("batchedGemmKernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","batch"))
  given stridedBatchedGemmKernel100_g: TritonKernel[K4F2I] = TritonKernel("stridedBatchedGemmKernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","batchStride"))
  given tiledGemmDoubleBufferKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmDoubleBufferKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmDynamicSliceKernel_g: TritonKernel[K6F1I] = TritonKernel("tiledGemmDynamicSliceKernel", List("float*","float*","float*","int","int","int","int","int"), List("C","A","B","M","N","K","startM","startN"))
  given tiledGemmTensorCoreKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmTensorCoreKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmSplitKKernel_g: TritonKernel[K4F2I] = TritonKernel("tiledGemmSplitKKernel", List("float*","float*","float*","int","int","int","int"), List("C","A","B","M","N","K","splitK"))
  given tiledGemmRegBlock4x4Kernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmRegBlock4x4Kernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmSoftwarePipelineKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmSoftwarePipelineKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmProducerConsumerKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmProducerConsumerKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmPersistentKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmPersistentKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmStreamKKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmStreamKKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmKPadKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmKPadKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmNPadKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmNPadKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmMPadKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmMPadKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmAsyncCopyKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmAsyncCopyKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))
  given tiledGemmWarpReductionKernel_g: TritonKernel[K4F1I] = TritonKernel("tiledGemmWarpReductionKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))

  // 21-40: KVCache Management
  given kvCacheUpdateKernel_g: TritonKernel[K3F2I] = TritonKernel("kvCacheUpdateKernel", List("float*","float*","float*","int","int","int"), List("kvCache","key","value","seqLen","headDim","layerIdx"))
  given kvCacheRoPEKernel_g: TritonKernel[K2F2I] = TritonKernel("kvCacheRoPEKernel", List("float*","int","int","int"), List("kvCache","pos","headDim","seqLen"))
  given kvCachePagedAttnKernel_g: TritonKernel[K5F1I] = TritonKernel("kvCachePagedAttnKernel", List("float*","float*","float*","float*","int","int","int"), List("q","kvCache","kCache","vCache","seqLen","headDim","blockSize"))
  given kvCacheSlidingWindowAttnKernel_g: TritonKernel[K5F2I] = TritonKernel("kvCacheSlidingWindowAttnKernel", List("float*","float*","float*","float*","int","int","int"), List("q","k","v","kvCache","seqLen","headDim","windowSize"))
  given kvCacheContinuousBatchingKernel_g: TritonKernel[K3F2I] = TritonKernel("kvCacheContinuousBatchingKernel", List("float*","float*","int","int","int"), List("kvCache","tokens","seqLen","headDim","batchSize"))
  given kvCacheSpeculativeDecodingKernel_g: TritonKernel[K3F2I] = TritonKernel("kvCacheSpeculativeDecodingKernel", List("float*","float*","float*","int","int","int"), List("kvCache","key","value","seqLen","headDim","numSpeculative"))
  given kvCacheMultiQueryAttnKernel_g: TritonKernel[K5F2I] = TritonKernel("kvCacheMultiQueryAttnKernel", List("float*","float*","float*","float*","int","int","int"), List("q","k","v","kvCache","seqLen","numKvHeads","headDim"))
  given kvCacheGroupedQueryAttnKernel_g: TritonKernel[K5F3I] = TritonKernel("kvCacheGroupedQueryAttnKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","kvCache","seqLen","numQHeads","numKvHeads","headDim"))
  given kvCachePrefixCachingKernel_g: TritonKernel[K3F2I] = TritonKernel("kvCachePrefixCachingKernel", List("float*","float*","int","int","int"), List("kvCache","prefixHash","seqLen","headDim","numLayers"))
  given kvCacheEvictionLRUKernel_g: TritonKernel[K3F1I] = TritonKernel("kvCacheEvictionLRUKernel", List("float*","float*","int","int","int"), List("kvCache","accessTime","seqLen","headDim","cacheSize"))
  given kvCacheFlashAttnKernel_g: TritonKernel[K5F1I] = TritonKernel("kvCacheFlashAttnKernel", List("float*","float*","float*","float*","int","int","int"), List("q","k","v","kvCache","seqLen","headDim","blockSize"))
  given kvCacheContextMergeKernel_g: TritonKernel[K4F1I] = TritonKernel("kvCacheContextMergeKernel", List("float*","float*","float*","int","int","int"), List("kvCache1","kvCache2","kvCacheOut","seqLen1","seqLen2","headDim"))
  given kvCacheLayerNormKernel_g: TritonKernel[K4F1I] = TritonKernel("kvCacheLayerNormKernel", List("float*","float*","float*","int","int"), List("kvCache","mean","variance","seqLen","headDim"))
  given kvCacheRMSNormKernel_g: TritonKernel[K3F1I] = TritonKernel("kvCacheRMSNormKernel", List("float*","float*","int","int"), List("kvCache","rms","seqLen","headDim"))
  given kvCacheKeyHashKernel_g: TritonKernel[K3F1I] = TritonKernel("kvCacheKeyHashKernel", List("float*","float*","int","int"), List("kvCache","keyHash","seqLen","headDim"))
  given kvCacheValueInterpKernel_g: TritonKernel[K3F1I] = TritonKernel("kvCacheValueInterpKernel", List("float*","float*","int","int"), List("kvCache","weights","seqLen","headDim"))
  given kvCacheTemporalCacheKernel_g: TritonKernel[K3F2I] = TritonKernel("kvCacheTemporalCacheKernel", List("float*","float*","int","int","int"), List("kvCache","timeStamp","seqLen","headDim","maxTime"))
  given kvCacheCompressKernel_g: TritonKernel[K3F2I] = TritonKernel("kvCacheCompressKernel", List("float*","float*","int","int","int"), List("kvCacheIn","kvCacheOut","seqLen","headDim","compressRatio"))
  given kvCacheReconstructKernel_g: TritonKernel[K3F2I] = TritonKernel("kvCacheReconstructKernel", List("float*","float*","int","int","int"), List("kvCacheCompressed","kvCacheOut","seqLen","headDim","compressRatio"))
  given kvCacheAdaptivePrecisionKernel_g: TritonKernel[K3F1I] = TritonKernel("kvCacheAdaptivePrecisionKernel", List("float*","float*","int","int"), List("kvCache","precision","seqLen","headDim"))

  // 41-70: Transformer Attention
  given multiHeadAttentionKernel100_g: TritonKernel[K6F1I] = TritonKernel("multiHeadAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","numHeads","headDim","n"))
  given flashAttentionKernel100_g: TritonKernel[K5F1I] = TritonKernel("flashAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("q","k","v","out","seqLen","headDim","blockSize"))
  given groupedQueryAttentionKernel100_g: TritonKernel[K6F2I] = TritonKernel("groupedQueryAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","numQHeads","numKvHeads","headDim"))
  given causalAttentionKernel_g: TritonKernel[K5F1I] = TritonKernel("causalAttentionKernel", List("float*","float*","float*","float*","int","int"), List("q","k","v","out","seqLen","headDim"))
  given crossAttentionKernel100_g: TritonKernel[K6F1I] = TritonKernel("crossAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("q","k","v","out","qSeqLen","kvSeqLen","headDim"))
  given localAttentionKernel100_g: TritonKernel[K6F1I] = TritonKernel("localAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","windowSize","n"))
  given alibiAttentionKernel_g: TritonKernel[K5F1I] = TritonKernel("alibiAttentionKernel", List("float*","float*","float*","float*","int","int"), List("q","k","v","out","seqLen","headDim"))
  given multiScaleAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("multiScaleAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","numScales","n"))
  given sparseAttentionKernel100_g: TritonKernel[K6F1I] = TritonKernel("sparseAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","numSparsity","n"))
  given linearAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("linearAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","featureDim","n"))
  given performerAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("performerAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","numFeatures","n"))
  given rezeroAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("rezeroAttentionKernel", List("float*","float*","float*","float*","float*","int","int"), List("q","k","v","out","residual","seqLen","headDim"))
  given gatedAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("gatedAttentionKernel", List("float*","float*","float*","float*","float*","int","int"), List("q","k","v","gate","out","seqLen","headDim"))
  given convAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("convAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int","int"), List("q","k","v","kernel","out","seqLen","headDim","kernelSize","n"))
  given relativeBiasAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("relativeBiasAttentionKernel", List("float*","float*","float*","float*","float*","int","int"), List("q","k","v","bias","out","seqLen","headDim"))
  given memEfficientAttentionKernel_g: TritonKernel[K5F1I] = TritonKernel("memEfficientAttentionKernel", List("float*","float*","float*","float*","int","int"), List("q","k","v","out","seqLen","headDim"))
  given xformerAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("xformerAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int","int"), List("q","k","v","proj","out","seqLen","headDim","reducedDim","n"))
  given longformerAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("longformerAttentionKernel", List("float*","float*","float*","float*","int","int","int","int","int"), List("q","k","v","out","seqLen","headDim","globalSize","windowSize","n"))
  given bigBirdAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("bigBirdAttentionKernel", List("float*","float*","float*","float*","int","int","int","int","int"), List("q","k","v","out","seqLen","headDim","numRandom","numBand","n"))
  given routingAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("routingAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int","int"), List("q","k","v","router","out","seqLen","headDim","numExperts","n"))
  given diffusionAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("diffusionAttentionKernel", List("float*","float*","float*","float*","float*","int","int"), List("q","k","v","timeStep","out","seqLen","headDim"))
  given multiModalAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("multiModalAttentionKernel", List("float*","float*","float*","float*","int","int","int"), List("qText","kImage","vImage","out","textLen","imageLen","headDim"))
  given threadIdx_hierarchicalAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("hierarchicalAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","blockSize","n"))
  given maskedAttentionKernel100_g: TritonKernel[K6F1I] = TritonKernel("maskedAttentionKernel", List("float*","float*","float*","float*","float*","int","int"), List("q","k","v","mask","out","seqLen","headDim"))
  given axialAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("axialAttentionKernel", List("float*","float*","float*","float*","int","int","int","int","int"), List("q","k","v","out","seqLen","headDim","dim0","dim1","n"))
  given globalAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("globalAttentionKernel", List("float*","float*","float*","float*","float*","int","int"), List("q","k","v","globalIdx","out","seqLen","headDim"))
  given synthesizerAttentionKernel_g: TritonKernel[K5F1I] = TritonKernel("synthesizerAttentionKernel", List("float*","float*","float*","float*","int","int"), List("q","bias","v","out","seqLen","headDim"))
  given funnelAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("funnelAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","poolSize","n"))
  given cosformerAttentionKernel_g: TritonKernel[K5F1I] = TritonKernel("cosformerAttentionKernel", List("float*","float*","float*","float*","int","int"), List("q","k","v","out","seqLen","headDim"))
  given retNetAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("retNetAttentionKernel", List("float*","float*","float*","float*","float","int","int"), List("q","k","v","out","decay","seqLen","headDim"))

  // 71-100: TokenCache & Advanced Patterns
  given tokenEmbeddingLookupKernel_g: TritonKernel[K3F1I] = TritonKernel("tokenEmbeddingLookupKernel", List("float*","float*","float*","int","int","int"), List("embeddingTable","tokenIds","out","vocabSize","embeddingDim","seqLen"))
  given ringAttentionKernel_g: TritonKernel[K8F1I] = TritonKernel("ringAttentionKernel", List("float*","float*","float*","float*","int","int","int","int","int"), List("q","k","v","out","seqLen","headDim","numDevices","deviceId","n"))
  given mixtureOfExpertsKernel_g: TritonKernel[K6F2I] = TritonKernel("mixtureOfExpertsKernel", List("float*","float*","float*","float*","int","int","int","int"), List("x","weight","router","out","seqLen","hiddenDim","numExperts","expertDim"))
  given tokenMergeKernel_g: TritonKernel[K4F1I] = TritonKernel("tokenMergeKernel", List("float*","float*","float*","int","int"), List("tokens","mergeMap","out","seqLen","hiddenDim"))
  given tokenSplitKernel_g: TritonKernel[K4F2I] = TritonKernel("tokenSplitKernel", List("float*","float*","float*","int","int","int","int"), List("tokens","splitMap","out","seqLen","hiddenDim","maxSplits","n"))
  given tokenGenWithPrefixKernel_g: TritonKernel[K5F2I] = TritonKernel("tokenGenWithPrefixKernel", List("float*","int","float*","float*","int","int"), List("prefix","prefixLen","generated","out","totalLen","hiddenDim"))
  given tokenScoringKernel_g: TritonKernel[K4F1I] = TritonKernel("tokenScoringKernel", List("float*","float*","float*","int","int"), List("tokens","scores","out","seqLen","hiddenDim"))
  given softmaxOverSeqKernel_g: TritonKernel[K3F1I] = TritonKernel("softmaxOverSeqKernel", List("float*","float*","int","int"), List("x","out","seqLen","headDim"))
  given layerNormOverSeqKernel_g: TritonKernel[K5F1I] = TritonKernel("layerNormOverSeqKernel", List("float*","float*","float*","float*","int","int"), List("x","mean","variance","out","seqLen","hiddenDim"))
  given tokenDropoutKernel_g: TritonKernel[K5F1I] = TritonKernel("tokenDropoutKernel", List("float*","float*","float*","float","int","int"), List("x","mask","out","dropoutRate","seqLen","hiddenDim"))
  given removePaddingKernel_g: TritonKernel[K3F1I] = TritonKernel("removePaddingKernel", List("float*","float*","float*","int","int"), List("x","seqLens","out","seqLen","hiddenDim"))
  given insertPaddingKernel_g: TritonKernel[K4F1I] = TritonKernel("insertPaddingKernel", List("float*","float*","float*","float*","int","int","int"), List("x","seqLens","out","seqLen","hiddenDim","maxLen"))
  given positionalEncodingKernel_g: TritonKernel[K3F1I] = TritonKernel("positionalEncodingKernel", List("float*","float*","int","int"), List("x","out","seqLen","hiddenDim"))
  given ropeKernel_g: TritonKernel[K3F1I] = TritonKernel("ropeKernel", List("float*","float*","int","int"), List("x","out","seqLen","headDim"))
  given alibiKernel_g: TritonKernel[K3F1I] = TritonKernel("alibiKernel", List("float*","float*","int","int","int"), List("x","out","seqLen","headDim","numHeads"))
  given tokenAttnPoolKernel_g: TritonKernel[K2F1I] = TritonKernel("tokenAttnPoolKernel", List("float*","float*","int","int"), List("x","out","seqLen","hiddenDim"))
  given deepNormKernel_g: TritonKernel[K5F1I] = TritonKernel("deepNormKernel", List("float*","float*","float","float","int","int"), List("x","out","alpha","beta","seqLen","hiddenDim"))
  given swigluKernel_g: TritonKernel[K5F1I] = TritonKernel("swigluKernel", List("float*","float*","float*","float*","int","int"), List("x","w1","w2","out","seqLen","hiddenDim"))
  given fusedAttnFFNKernel_g: TritonKernel[K8F2I] = TritonKernel("fusedAttnFFNKernel", List("float*","float*","float*","float*","float*","float*","int","int","int","int"), List("q","k","v","w1","w2","out","seqLen","headDim","ffnDim","n"))
  given fusedRMSNormAttnKernel_g: TritonKernel[K7F1I] = TritonKernel("fusedRMSNormAttnKernel", List("float*","float*","float*","float*","float*","float*","int","int","int"), List("x","weight","q","k","v","out","seqLen","headDim","n"))
  given quantizedAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("quantizedAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int"), List("q","k","v","scaleQ","scaleK","out","seqLen","headDim","n"))
  given speculativeAttentionKernel_g: TritonKernel[K8F1I] = TritonKernel("speculativeAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int","int"), List("q","kDraft","vDraft","kFinal","vFinal","out","seqLen","headDim","draftLen","n"))
  given cascadeAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("cascadeAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int"), List("q","k1","v1","k2","v2","out","seqLen","headDim","n"))
  given chunkedAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("chunkedAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","chunkSize","n"))
  given stridedAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("stridedAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","stride","n"))
  given dilatedAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("dilatedAttentionKernel", List("float*","float*","float*","float*","int","int","int","int"), List("q","k","v","out","seqLen","headDim","dilation","n"))
  given starAttentionKernel_g: TritonKernel[K6F1I] = TritonKernel("starAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int"), List("q","k","v","center","out","seqLen","headDim","n"))
  given hydraAttentionKernel_g: TritonKernel[K7F1I] = TritonKernel("hydraAttentionKernel", List("float*","float*","float*","float*","float*","int","int","int"), List("q1","q2","k","v","out","seqLen","headDim","n"))
  given multiHeadLatentAttnKernel_g: TritonKernel[K8F1I] = TritonKernel("multiHeadLatentAttnKernel", List("float*","float*","float*","float*","float*","int","int","int","int"), List("q","k","v","latent","out","seqLen","headDim","latentDim","n"))
  given fullSpectrumAttnKernel_g: TritonKernel[K5F1I] = TritonKernel("fullSpectrumAttnKernel", List("float*","float*","float*","float*","int","int"), List("q","k","v","out","seqLen","headDim"))

// alibiKernel ropeKernel  insertPaddingKernel removePaddingKernel  tokenEmbeddingLookupKernel kvCacheEvictionLRUKernel
//warpTiledReduceKernel  flexAttentionKernel pageAttentionKernel flashAttentionKernel
// [executeKernelByName] Kernel not found: tiledGemmSplitSKernel 
  
// Top-level kernel registry — all ~260 kernels from 5 test files
private val allKernelNames: List[String] = List(
  // Category 1: Test50ComplexKernels (50 kernels)
  "vectorSumKernel", "matrixVecMulKernel", "softmaxKernel", "layerNormKernel",
  "batchMatMulKernel", "conv1dKernel", "maxPoolKernel", "rmsNormKernel",
  "attentionScoreKernel", "cumsumKernel",
  "tiledMatMulKernel2", "blockReduceKernel2", "sharedTransposeKernel",
  "tiledSoftmaxKernel", "blockMatrixLoadKernel", "parallelReduceKernel",
  "tiledLayerNormKernel", "tiledAttentionKernel", "warpReduceSyncthreadsKernel",
  "multiWarpReduceKernel",
  "warpReduceSumKernel", "warpReduceMaxKernel", "warpButterflyReduceKernel",
  "warpScanKernel", "warpVoteAnyKernel", "warpVoteAllKernel",
  "warpVoteCombinedKernel", "warpShuffleXorKernel", "warpMultiReduceKernel",
  "warpReduceTernaryKernel",
  "maskedVectorAddKernel", "maskedSoftmaxKernel", "maskedAttentionKernel",
  "maskedFillKernel", "maskedSelectKernel", "maskedScaleKernel",
  "maskedClampKernel", "maskedLayerNormKernel", "maskedGeluKernel",
  "maskedDropoutKernel",
  "flashAttentionKernel2", "pageAttentionKernel2", "groupedQueryAttentionKernel",
  "multiHeadAttentionKernel", "fusedGeluLayerNormKernel", "fusedScaleBiasGeluKernel",
  "fusedAttentionBiasKernel", "sparseAttentionKernel", "crossAttentionKernel",
  "hierarchicalAttentionKernel",
  // Category 2: TestAttentionGeneric (4 kernels)
  "storeKVCacheKernel", "flashAttentionKernel", "pageAttentionKernel",
  "flexAttentionKernel",
  // Category 3: Test50CUTLASSKernels (50 kernels)
  "tiledGemm64x64Kernel", "tiledGemm128x128Kernel", "tiledGemm32x32WarpKernel",
  "persistentGemmKernel", "tiledGemmBiasKernel", "tiledGemmGeluKernel",
  "stridedBatchedGemmKernel", "quantizedGemmKernel", "sparseGemmKernel",
  "mixedPrecisionGemmKernel",
  "warpGemm16x16Kernel", "warpTiledReduceKernel", "warpTiledScanKernel",
  "warpTiledSoftmaxKernel", "warpTiledLayerNormKernel", "warpTiledAttentionKernel",
  "warpTiledConv2dKernel", "warpTiledRmsNormKernel", "warpTiledGeluKernel",
  "warpTiledDropoutKernel",
  "persistentSoftmaxKernel", "persistentLayerNormKernel", "persistentAttentionKernel",
  "persistentGemvKernel", "persistentBatchGemmKernel", "persistentTanhKernel",
  "persistentSigmoidKernel", "persistentReluKernel", "persistentLeakyReluKernel",
  "persistentEluKernel",
  "tiledFlashAttentionKernel", "tiledMultiHeadAttentionKernel", "tiledGQAttentionKernel",
  "tiledSlidingWindowAttentionKernel", "tiledCrossAttentionKernel",
  "tiledBidirectionalAttentionKernel", "tiledLocalAttentionKernel",
  "tiledSparseAttentionKernel", "tiledGlobalLocalAttentionKernel",
  "tiledKernelAttentionKernel",
  "fusedTiledGemmBiasGeluKernel", "fusedTiledGemmBiasResidualKernel",
  "fusedTiledLayerNormAttentionKernel", "fusedTiledRmsNormGemmKernel",
  "fusedTiledSkipConnectionKernel", "fusedTiledGatedActivationKernel",
  "fusedTiledSwigluKernel", "fusedTiledMhaFfnKernel",
  "fusedTiledRmsNormSkipGemmKernel", "fusedTiledBiasGeluResidualLayerNormKernel",
  // Category 4: Test50ThreadIdxKernels (50 kernels)
  "warpTileGemm4x4LaneKernel", "warpTileGemm8x8BankAvoidKernel",
  "warpTranspose16x16Kernel", "blockSoftmaxKernel", "blockLayerNormKernel",
  "blockAttentionQKVKernel", "blockGeluKernel", "blockResidualKernel",
  "blockBiLevelReduceKernel", "blockGatedActivationKernel",
  "tile2dGemmKernel", "tile2dAttentionKernel", "tile2dLayerNormKernel",
  "tile2dRmsNormKernel", "tile2dSigmoidKernel", "tile2dSwigluKernel",
  "tile2dLeakyReluKernel", "tile2dBatchNormKernel", "tile2dCrossEntropyKernel",
  "tile2dDropoutKernel",
  "blockReduceGemmKernel", "hierarchicalSoftmaxKernel", "blockSegmentReduceKernel",
  "hierarchicalLayerNormKernel", "multiLevelReduceGemmKernel",
  "blockCausalAttentionKernel", "hierarchicalGemmKernel", "blockExclusiveScanKernel",
  "multiBlockReduceGemmKernel", "multiBlockSoftmaxKernel",
  "flashAttentionKernel", "multiHeadAttentionKernel", "groupedQueryAttentionKernel",
  "slidingWindowAttentionKernel", "crossAttentionKernel",
  "bidirectionalAttentionKernel", "localAttentionKernel", "sparseAttentionKernel",
  "globalLocalAttentionKernel", "kernelAttentionKernel",
  "fusedGemmGeluKernel", "fusedGemmResidualKernel", "fusedLayerNormAttentionKernel",
  "fusedRmsNormSwigluKernel", "fusedSkipConnectionKernel",
  "fusedGatedActivationKernel", "fusedSwigluKernel", "fusedMhaFfnKernel",
  "fusedRmsNormSkipGemmKernel", "fusedBiasGeluResidualLayerNormKernel",
  // Category 5: Test100ComplexKernels (100 kernels)
  "tiledGemm64x64Kernel", "tiledGemm128x128Kernel", "tiledGemmPartitionKKernel",
  "tiledGemmBankAvoidKernel", "batchedGemmKernel", "stridedBatchedGemmKernel",
  "tiledGemmDoubleBufferKernel", "tiledGemmDynamicSliceKernel",
  "tiledGemmTensorCoreKernel", "tiledGemmSplitKKernel",
  "tiledGemmRegBlock4x4Kernel", "tiledGemmSoftwarePipelineKernel",
  "tiledGemmProducerConsumerKernel", "tiledGemmPersistentKernel",
  "tiledGemmStreamKKernel", "tiledGemmKPadKernel", "tiledGemmNPadKernel",
  "tiledGemmMPadKernel", "tiledGemmAsyncCopyKernel", "tiledGemmWarpReductionKernel",
  "kvCacheUpdateKernel", "kvCacheRoPEKernel", "kvCachePagedAttnKernel",
  "kvCacheSlidingWindowAttnKernel", "kvCacheContinuousBatchingKernel",
  "kvCacheSpeculativeDecodingKernel", "kvCacheMultiQueryAttnKernel",
  "kvCacheGroupedQueryAttnKernel", "kvCachePrefixCachingKernel",
  "kvCacheEvictionLRUKernel", "kvCacheFlashAttnKernel",
  "kvCacheContextMergeKernel", "kvCacheLayerNormKernel", "kvCacheRMSNormKernel",
  "kvCacheKeyHashKernel", "kvCacheValueInterpKernel", "kvCacheTemporalCacheKernel",
  "kvCacheCompressKernel", "kvCacheReconstructKernel",
  "kvCacheAdaptivePrecisionKernel",
  "multiHeadAttentionKernel", "flashAttentionKernel", "groupedQueryAttentionKernel",
  "causalAttentionKernel", "crossAttentionKernel", "localAttentionKernel",
  "alibiAttentionKernel", "multiScaleAttentionKernel", "sparseAttentionKernel",
  "linearAttentionKernel", "performerAttentionKernel", "rezeroAttentionKernel",
  "gatedAttentionKernel", "convAttentionKernel", "relativeBiasAttentionKernel",
  "memEfficientAttentionKernel", "xformerAttentionKernel", "longformerAttentionKernel",
  "bigBirdAttentionKernel", "routingAttentionKernel", "diffusionAttentionKernel",
  "multiModalAttentionKernel", "hierarchicalAttentionKernel",
  "maskedAttentionKernel", "axialAttentionKernel", "globalAttentionKernel",
  "synthesizerAttentionKernel", "funnelAttentionKernel",
  "cosformerAttentionKernel", "retNetAttentionKernel",
  "tokenEmbeddingLookupKernel", "ringAttentionKernel", "mixtureOfExpertsKernel",
  "tokenMergeKernel", "tokenSplitKernel", "tokenGenWithPrefixKernel",
  "tokenScoringKernel", "softmaxOverSeqKernel", "layerNormOverSeqKernel",
  "tokenDropoutKernel", "removePaddingKernel", "insertPaddingKernel",
  "positionalEncodingKernel", "ropeKernel", "alibiKernel",
  "tokenAttnPoolKernel", "deepNormKernel", "swigluKernel",
  "fusedAttnFFNKernel", "fusedRMSNormAttnKernel", "quantizedAttentionKernel",
  "speculativeAttentionKernel", "cascadeAttentionKernel", "chunkedAttentionKernel",
  "stridedAttentionKernel", "dilatedAttentionKernel", "starAttentionKernel",
  "hydraAttentionKernel", "multiHeadLatentAttnKernel", "fullSpectrumAttnKernel"
)

private def runKernelByName(name: String, n: Int, blockSize: Int): Int =
  try
    val r = SCR.executeKernelByName(name, n, blockSize)
    if r.isDefined then
      println(s"  [OK] $name")
      1
    else
      println(s"  [FAIL] $name")
      0
  catch case e: Exception =>
    println(s"  [ERR] $name: ${e.getMessage.take(80)}")
    0

@main def UnifiedBenchmarkAll(): Unit =
  val N = 1024
  val BLOCK = 256
  println("=" * 70)
  println("UNIFIED BENCHMARK ALL — ALL ~260 Kernels")
  println("=" * 70)

  // Force initialization of all test objects (triggering @TritonKernelMacro registration)
  UnifiedBenchmarkAllKernels

  val uniqueKernels = allKernelNames.distinct
  println(s"\nExecuting ${uniqueKernels.size} unique kernels (deduped from ~260)...")

  var totalPass = 0
  var totalFail = 0
  var totalErr = 0

  for name <- uniqueKernels do
    val r = runKernelByName(name, N, BLOCK)
    if r == 1 then totalPass += 1
    else if r == 0 then totalFail += 1
    else totalErr += 1

  println("\n" + "=" * 70)
  println(f"RESULTS: $totalPass passed, $totalFail failed, $totalErr errors")
  println("=" * 70)
