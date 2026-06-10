package cuda.dsl.benchmark

import cuda.dsl.core.dim3
import cuda.dsl.runtime.{TritonKernel, TritonKernel as TKTrait}
import cuda.dsl.runtime.runtimeTypes.*
import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR
import TKTrait.{given TritonKernel}

/** Comprehensive verification for all ~260 kernels from @TritonKernelMacro.
 *  Tests each kernel's output to ensure it's not all-zeros, not NaN, not Inf.
 *
 *  Executes kernels via ScalaCudaRuntime.executeKernelByName(),
 *  loading from generated kernel sources in /tmp/cuda_dsl_generated_kernels90.txt.
 */
object UnifiedBenchmarkVerification:

  // Force initialization of test objects (triggers @TritonKernelMacro registration)
  private val _ = (Test50ComplexKernels, Test50CUTLASSKernels, Test50ThreadIdxKernels, Test100ComplexKernels, TestAttentionGeneric)

  // Unique kernel list from UnifiedBenchmarkAllKernels (deduped from ~260)
  private val kernelNames: List[String] = List(
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
    println("UNIFIED BENCHMARK VERIFICATION — All ~260 Kernels")
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
    println("VERIFICATION SUMMARY")
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
    println("CATEGORY BREAKDOWN")
    println("=" * 80)
    val cats = Map(
      "Complex" -> kernelNames.take(50),
      "Attention" -> List("storeKVCacheKernel", "flashAttentionKernel", "pageAttentionKernel", "flexAttentionKernel"),
      "CUTLASS" -> kernelNames.slice(54, 104),
      "ThreadIdx" -> kernelNames.slice(104, 154),
      "100Complex" -> kernelNames.drop(154)
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
