package cuda.dsl.benchmark

import cuda.dsl.core.{FloatPtr, IntPtr}
import cuda.dsl.core.dim3
import cuda.dsl.runtime.{TritonKernel, TritonKernel as TKTrait}
import cuda.dsl.runtime.runtimeTypes.*
import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR
import TKTrait.{given TritonKernel}

/** Comprehensive verification for all ~260 kernels from @TritonKernelMacro (pointer variant).
 *  Tests each kernel's output to ensure it's not all-zeros, not NaN, not Inf.
 *UnifiedBenchmarkVerificationPtr
 *  Executes kernels via ScalaCudaRuntime.executeKernelByName(),
 *  loading from generated kernel sources in /tmp/cuda_dsl_generated_kernels_ptr.txt.
 */
object UnifiedBenchmarkVerificationPtr:

  // Force initialization of test objects (pointer variants, triggers @TritonKernelMacro registration)
  private val _ = (Test50ComplexKernelsPtr, Test50CUTLASSKernelsPtr, Test50ThreadIdxKernelsPtr, Test100ComplexKernelsPtr, TestAttentionGenericPtr)

  // Unique kernel list (deduped from ~260) - all with Ptr suffix
  private val kernelNames: List[String] = List(
    // Category 1: Test50ComplexKernelsPtr (46 kernels)
    "vectorSumKernelPtr", "matrixVecMulKernelPtr", "softmaxKernelPtr", "layerNormKernelPtr",
    "batchMatMulKernelPtr", "conv1dKernelPtr", "maxPoolKernelPtr", "rmsNormKernelPtr",
    "attentionScoreKernelPtr", "cumsumKernelPtr",
    "tiledMatMulKernel2Ptr", "blockReduceKernel2Ptr", "sharedTransposeKernelPtr",
    "tiledSoftmaxKernelPtr", "blockMatrixLoadKernelPtr", "parallelReduceKernelPtr",
    "tiledLayerNormKernelPtr", "tiledAttentionKernelPtr", "warpReduceSyncthreadsKernelPtr",
    "multiWarpReduceKernelPtr",
    "warpReduceSumKernelPtr", "warpReduceMaxKernelPtr", "warpButterflyReduceKernelPtr",
    "warpScanKernelPtr", "warpVoteAnyKernelPtr", "warpVoteAllKernelPtr",
    "warpVoteCombinedKernelPtr", "warpShuffleXorKernelPtr", "warpMultiReduceKernelPtr",
    "warpReduceTernaryKernelPtr",
    "maskedVectorAddKernelPtr", "maskedSoftmaxKernelPtr", "maskedAttentionKernelPtr",
    "maskedFillKernelPtr", "maskedSelectKernelPtr", "maskedScaleKernelPtr",
    "maskedClampKernelPtr", "maskedLayerNormKernelPtr", "maskedGeluKernelPtr",
    "maskedDropoutKernelPtr",
    "flashAttentionKernel2Ptr", "pageAttentionKernel2Ptr", "groupedQueryAttentionKernelPtr",
    "multiHeadAttentionKernelPtr", "fusedGeluLayerNormKernelPtr", "fusedScaleBiasGeluKernelPtr",
    "fusedAttentionBiasKernelPtr", "sparseAttentionKernelPtr", "crossAttentionKernelPtr",
    "hierarchicalAttentionKernelPtr",
    // Category 2: TestAttentionGenericPtr (4 kernels)
    "storeKVCacheKernelSPtr", "flashAttentionKernelSPtr", "pageAttentionKernelSPtr",
    "flexAttentionKernelSPtr", "loadKVCacheKernelSPtr", "evictKVCacheKernelSPtr", "flashAttentionBiasKernelSPtr",
    "flashAttentionMaskKernelSPtr", "flashAttentionDropoutKernelSPtr", "flashAttentionSplitKernelSPtr",
    "pageAttentionPagedKernelSPtr", "pageAttentionPrefillKernelSPtr", "pageAttentionDecodeKernelSPtr", "pageAttentionSparseKernelSPtr",
    "flexAttentionBlockMaskKernelSPtr", "flexAttentionScoreModKernelSPtr", "flexAttentionCausalKernelSPtr", "flexAttentionSlidingWindowKernelSPtr",
    "multiHeadAttentionKernelSPtr", "groupedQueryAttentionKernelSPtr", "multiQueryAttentionKernelSPtr", "crossAttentionKernelSPtr",
    "bidirectionalAttentionKernelSPtr",
    // Category 3: Test50CUTLASSKernelsPtr (50 kernels)
    "tiledGemm64x64KernelPtr", "tiledGemm128x128KernelPtr", "tiledGemm32x32WarpKernelPtr",
    "persistentGemmKernelPtr", "tiledGemmBiasKernelPtr", "tiledGemmGeluKernelPtr",
    "stridedBatchedGemmKernelPtr", "quantizedGemmKernelPtr", "sparseGemmKernelPtr",
    "mixedPrecisionGemmKernelPtr",
    "warpGemm16x16KernelPtr", "warpTiledReduceKernelPtr", "warpTiledScanKernelPtr",
    "warpTiledSoftmaxKernelPtr", "warpTiledLayerNormKernelPtr", "warpTiledAttentionKernelPtr",
    "warpTiledConv2dKernelPtr", "warpTiledRmsNormKernelPtr", "warpTiledGeluKernelPtr",
    "warpTiledDropoutKernelPtr",
    "persistentSoftmaxKernelPtr", "persistentLayerNormKernelPtr", "persistentAttentionKernelPtr",
    "persistentGemvKernelPtr", "persistentBatchGemmKernelPtr", "persistentTanhKernelPtr",
    "persistentSigmoidKernelPtr", "persistentReluKernelPtr", "persistentLeakyReluKernelPtr",
    "persistentEluKernelPtr",
    "tiledFlashAttentionKernelPtr", "tiledMultiHeadAttentionKernelPtr", "tiledGQAttentionKernelPtr",
    "tiledSlidingWindowAttentionKernelPtr", "tiledCrossAttentionKernelPtr",
    "tiledBidirectionalAttentionKernelPtr", "tiledLocalAttentionKernelPtr",
    "tiledSparseAttentionKernelPtr", "tiledGlobalLocalAttentionKernelPtr",
    "tiledKernelAttentionKernelPtr",
    "fusedTiledGemmBiasGeluKernelPtr", "fusedTiledGemmBiasResidualKernelPtr",
    "fusedTiledLayerNormAttentionKernelPtr", "fusedTiledRmsNormGemmKernelPtr",
    "fusedTiledSkipConnectionKernelPtr", "fusedTiledGatedActivationKernelPtr",
    "fusedTiledSwigluKernelPtr", "fusedTiledMhaFfnKernelPtr",
    "fusedTiledRmsNormSkipGemmKernelPtr", "fusedTiledBiasGeluResidualLayerNormKernelPtr",
    // Category 4: Test50ThreadIdxKernelsPtr (50 kernels)
    "warpTileGemm4x4LaneKernelPtr", "warpTileGemm8x8BankAvoidKernelPtr",
    "warpTranspose16x16KernelPtr", "blockSoftmaxKernelPtr", "blockLayerNormKernelPtr",
    "blockAttentionQKVKernelPtr", "blockGeluKernelPtr", "blockResidualKernelPtr",
    "blockBiLevelReduceKernelPtr", "blockGatedActivationKernelPtr",
    "tile2dGemmKernelPtr", "tile2dAttentionKernelPtr", "tile2dLayerNormKernelPtr",
    "tile2dRmsNormKernelPtr", "tile2dSigmoidKernelPtr", "tile2dSwigluKernelPtr",
    "tile2dLeakyReluKernelPtr", "tile2dBatchNormKernelPtr", "tile2dCrossEntropyKernelPtr",
    "tile2dDropoutKernelPtr",
    "blockReduceGemmKernelPtr", "hierarchicalSoftmaxKernelPtr", "blockSegmentReduceKernelPtr",
    "hierarchicalLayerNormKernelPtr", "multiLevelReduceGemmKernelPtr",
    "blockCausalAttentionKernelPtr", "hierarchicalGemmKernelPtr", "blockExclusiveScanKernelPtr",
    "multiBlockReduceGemmKernelPtr", "multiBlockSoftmaxKernelPtr",
    "flashAttentionKernelPtr", "multiHeadAttentionKernelPtr", "groupedQueryAttentionKernelPtr",
    "slidingWindowAttentionKernelPtr", "crossAttentionKernelPtr",
    "bidirectionalAttentionKernelPtr", "localAttentionKernelPtr", "sparseAttentionKernelPtr",
    "globalLocalAttentionKernelPtr", "kernelAttentionKernelPtr",
    "fusedGemmGeluKernelPtr", "fusedGemmResidualKernelPtr", "fusedLayerNormAttentionKernelPtr",
    "fusedRmsNormSwigluKernelPtr", "fusedSkipConnectionKernelPtr",
    "fusedGatedActivationKernelPtr", "fusedSwigluKernelPtr", "fusedMhaFfnKernelPtr",
    "fusedRmsNormSkipGemmKernelPtr", "fusedBiasGeluResidualLayerNormKernelPtr",
    // Category 5: Test100ComplexKernelsPtr (100 kernels)
    "tiledGemm64x64KernelPtr", "tiledGemm128x128KernelPtr", "tiledGemmPartitionKKernelPtr",
    "tiledGemmBankAvoidKernelPtr", "batchedGemmKernelPtr", "stridedBatchedGemmKernelPtr",
    "tiledGemmDoubleBufferKernelPtr", "tiledGemmDynamicSliceKernelPtr",
    "tiledGemmTensorCoreKernelPtr", "tiledGemmSplitKKernelPtr",
    "tiledGemmRegBlock4x4KernelPtr", "tiledGemmSoftwarePipelineKernelPtr",
    "tiledGemmProducerConsumerKernelPtr", "tiledGemmPersistentKernelPtr",
    "tiledGemmStreamKKernelPtr", "tiledGemmKPadKernelPtr", "tiledGemmNPadKernelPtr",
    "tiledGemmMPadKernelPtr", "tiledGemmAsyncCopyKernelPtr", "tiledGemmWarpReductionKernelPtr",
    "kvCacheUpdateKernelPtr", "kvCacheRoPEKernelPtr", "kvCachePagedAttnKernelPtr",
    "kvCacheSlidingWindowAttnKernelPtr", "kvCacheContinuousBatchingKernelPtr",
    "kvCacheSpeculativeDecodingKernelPtr", "kvCacheMultiQueryAttnKernelPtr",
    "kvCacheGroupedQueryAttnKernelPtr", "kvCachePrefixCachingKernelPtr",
    "kvCacheEvictionLRUKernelPtr", "kvCacheFlashAttnKernelPtr",
    "kvCacheContextMergeKernelPtr", "kvCacheLayerNormKernelPtr", "kvCacheRMSNormKernelPtr",
    "kvCacheKeyHashKernelPtr", "kvCacheValueInterpKernelPtr", "kvCacheTemporalCacheKernelPtr",
    "kvCacheCompressKernelPtr", "kvCacheReconstructKernelPtr",
    "kvCacheAdaptivePrecisionKernelPtr",
    "multiHeadAttentionKernelPtr", "flashAttentionKernelPtr", "groupedQueryAttentionKernelPtr",
    "causalAttentionKernelPtr", "crossAttentionKernelPtr", "localAttentionKernelPtr",
    "alibiAttentionKernelPtr", "multiScaleAttentionKernelPtr", "sparseAttentionKernelPtr",
    "linearAttentionKernelPtr", "performerAttentionKernelPtr", "rezeroAttentionKernelPtr",
    "gatedAttentionKernelPtr", "convAttentionKernelPtr", "relativeBiasAttentionKernelPtr",
    "memEfficientAttentionKernelPtr", "xformerAttentionKernelPtr", "longformerAttentionKernelPtr",
    "bigBirdAttentionKernelPtr", "routingAttentionKernelPtr", "diffusionAttentionKernelPtr",
    "multiModalAttentionKernelPtr", "hierarchicalAttentionKernelPtr",
    "maskedAttentionKernelPtr", "axialAttentionKernelPtr", "globalAttentionKernelPtr",
    "synthesizerAttentionKernelPtr", "funnelAttentionKernelPtr",
    "cosformerAttentionKernelPtr", "retNetAttentionKernelPtr",
    "tokenEmbeddingLookupKernelPtr", "ringAttentionKernelPtr", "mixtureOfExpertsKernelPtr",
    "tokenMergeKernelPtr", "tokenSplitKernelPtr", "tokenGenWithPrefixKernelPtr",
    "tokenScoringKernelPtr", "softmaxOverSeqKernelPtr", "layerNormOverSeqKernelPtr",
    "tokenDropoutKernelPtr", "removePaddingKernelPtr", "insertPaddingKernelPtr",
    "positionalEncodingKernelPtr", "ropeKernelPtr", "alibiKernelPtr",
    "tokenAttnPoolKernelPtr", "deepNormKernelPtr", "swigluKernelPtr",
    "fusedAttnFFNKernelPtr", "fusedRMSNormAttnKernelPtr", "quantizedAttentionKernelPtr",
    "speculativeAttentionKernelPtr", "cascadeAttentionKernelPtr", "chunkedAttentionKernelPtr",
    "stridedAttentionKernelPtr", "dilatedAttentionKernelPtr", "starAttentionKernelPtr",
    "hydraAttentionKernelPtr", "multiHeadLatentAttnKernelPtr", "fullSpectrumAttnKernelPtr"
  ).distinct

  // Configuration
  private val N = 1024
  private val BLOCK_SIZE = 256

  // Test results tracker
  case class KernelResult(
    name: String,
    status: String,
    outputSample: Array[Float],
    stats: String
  )

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("UNIFIED BENCHMARK VERIFICATION (POINTER VARIANT) — All ~260 Kernels")
    println("=" * 80)

    val startTime = System.nanoTime()

    println(s"\nVerifying ${kernelNames.size} kernels...")

    // Execute and verify each kernel
    val results = scala.collection.mutable.ListBuffer[KernelResult]()

    kernelNames.zipWithIndex.foreach { (name, idx) =>
      print(f"\r[${idx + 1}/${kernelNames.size}] Testing $name...")
      val result = verifyKernel(name, N, BLOCK_SIZE)
      results += result
    }

    // Print summary
    val elapsed = (System.nanoTime() - startTime) / 1e6
    printSummary(results.toList, elapsed)
  }

  /** Verify a single kernel's output */
  private def verifyKernel(name: String, n: Int, blockSize: Int): KernelResult = {
    try {
      val output = SCR.executeKernelByName(name, n, blockSize)

      output match {
        case Some(out) =>
          val allZero = out.forall(v => v == 0.0f)
          if (allZero)
            return KernelResult(name, "FAIL_ZEROS", out, "ALL_ZERO")

          val hasNaN = out.exists(v => v.isNaN)
          if (hasNaN)
            return KernelResult(name, "FAIL_NAN", out, "NaN_FOUND")

          val hasInf = out.exists(v => v.isInfinite)
          if (hasInf)
            return KernelResult(name, "FAIL_INF", out, "INF_FOUND")

          val finite = out.count(v => !v.isNaN && !v.isInfinite)
          val sum = out.filter(v => !v.isNaN && !v.isInfinite).sum
          val stats = s"finite=$finite/${out.length}, sum=${sum.formatted("%.2f")}"
          KernelResult(name, "PASS", out, stats)

        case None =>
          KernelResult(name, "ERROR", Array.empty, "NONE_RETURN")
      }
    } catch {
      case e: Exception =>
        KernelResult(name, "ERROR", Array.empty, e.getMessage.take(60))
    }
  }

  /** Print summary report */
  private def printSummary(results: List[KernelResult], elapsedMs: Double): Unit = {
    val passCount = results.count(_.status == "PASS")
    val failZeros = results.count(_.status == "FAIL_ZEROS")
    val failNaN = results.count(_.status == "FAIL_NAN")
    val failInf = results.count(_.status == "FAIL_INF")
    val errorCount = results.count(_.status == "ERROR")
    val total = results.size

    println("\n" + "=" * 80)
    println("VERIFICATION SUMMARY (POINTER VARIANT)")
    println("=" * 80)
    println(f"Kernels tested:   $total")
    println(f"Correct:         $passCount (${passCount * 100.0 / total}%.1f%%)")
    println(f"All-zeros:       $failZeros  [kernels producing only zeros]")
    println(f"NaN output:      $failNaN   [kernels producing NaN]")
    println(f"Inf output:      $failInf   [kernels producing Inf]")
    println(f"Execution errors:$errorCount  [kernels that failed to execute]")
    println(f"Elapsed:         ${elapsedMs.toLong}ms")
    println("=" * 80)

    // Show problematic kernels
    val problems = results.filter(r => r.status != "PASS")
    if (problems.nonEmpty) {
      println("\nProblematic kernels:")
      problems.foreach { r =>
        val sample = if (r.outputSample.nonEmpty)
          r.outputSample.take(3).map(_.formatted("%.3f")).mkString(", ")
        else "N/A"
        println(f"  ${r.name} [${r.status}] sample=[$sample] stats: ${r.stats}")
      }
    }

    // Show sample outputs from passing kernels
    println("\nSample passing kernel outputs:")
    results.filter(_.status == "PASS").take(20).foreach { p =>
      val sample = p.outputSample.take(5).map(_.formatted("%.3f")).mkString(", ")
      println(f"  ${p.name}: [$sample ...]  (${p.stats})")
    }

    // Category breakdown
    println("\n" + "=" * 80)
    println("CATEGORY BREAKDOWN (POINTER VARIANT)")
    println("=" * 80)
    val cats = Map(
      "Complex" -> kernelNames.take(46),
      "Attention" -> List("storeKVCacheKernelPtr", "flashAttentionKernelPtr", "pageAttentionKernelPtr", "flexAttentionKernelPtr"),
      "CUTLASS" -> kernelNames.slice(46, 96),
      "ThreadIdx" -> kernelNames.slice(96, 146),
      "100Complex" -> kernelNames.drop(146)
    )
    cats.foreach { (cat, names) =>
      val passed = results.count(r => names.contains(r.name) && r.status == "PASS")
      val totalCat = names.distinct.size
      println(f"  $cat: $passed/$totalCat passed")
    }

    println("=" * 80)
    println("VERDICT: " + (if (failZeros == 0 && failNaN == 0 && failInf == 0 && errorCount == 0)
      "ALL KERNELS PRODUCING VALID OUTPUT" else "ISSUES DETECTED"))
    println("=" * 80)
  }
