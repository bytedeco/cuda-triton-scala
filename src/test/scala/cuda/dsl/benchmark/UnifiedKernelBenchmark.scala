package cuda.dsl.benchmark

import cuda.dsl.core.dim3
import cuda.dsl.runtime._
import cuda.dsl.dsl._

/** Unified benchmark that executes kernels from all test files by-name.
  *  Accesses the test objects to trigger @TritonKernelMacro registration at compile time.
  */
object UnifiedBenchmark:

  // Import all test objects to trigger kernel registration at compile time
  val _ = (
    Test50ThreadIdxKernels,
    Test50ComplexKernels,
    Test50CUTLASSKernels,
    Test100ComplexKernels,
    TestAttentionGeneric
  )

  def runByName(name: String, n: Int): Int =
    try
      val desc = KernelDesc(name, List(OutputBuffer("out", n)), dim3((n+255)/256), dim3(256))
      val r = ScalaCudaRuntime.execute(desc)
      if r.isDefined then
        println(s"  [OK] $name")
        1
      else
        println(s"  [FAIL] $name")

        0
    catch case e: Exception =>
      println(s"  [ERR] $name: ${e.getMessage.take(60)}")
      0

@main def UnifiedKernelBenchmark(): Unit =
  val N = 1024
  println("=" * 70)
  println("Unified Kernel Benchmark — execute by name")
  println("=" * 70)

  // Trigger kernel registration
  println("\n[Phase 1] Registering kernels...")
  val startReg = System.nanoTime()
  // Force registration by importing objects
  val regTime = (System.nanoTime() - startReg) / 1e6
  println(f"Kernels registered in ${regTime}%.0fms")

  // Phase 2: Execute sample kernels by-name
  println("\n[Phase 2] Executing kernels by-name...")
  var totalPassed = 0
  var totalTests = 0

  // Test50ThreadIdxKernels samples
  println("\n-- Test50ThreadIdxKernels --")
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("blockSoftmaxKernel", N*N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("warpTileGemm4x4LaneKernel", N*N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("flashAttentionKernel", N*N)

  // Test50ComplexKernels samples
  println("\n-- Test50ComplexKernels --")
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("softmaxKernel", N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("warpReduceSumKernel", N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("maskedSoftmaxKernel", N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("flashAttentionKernel2", N)

  // Test50CUTLASSKernels samples
  println("\n-- Test50CUTLASSKernels --")
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("tiledGemm64x64Kernel", N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("persistentSoftmaxKernel", N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("tiledFlashAttentionKernel", N)

  // TestAttentionGeneric
  println("\n-- TestAttentionGeneric --")
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("storeKVCacheKernel", N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("flashAttentionKernel", N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("pageAttentionKernel", N)
  totalTests += 1; totalPassed += UnifiedBenchmark.runByName("flexAttentionKernel", N)

  println("\n" + "=" * 70)
  println(s"Results: $totalPassed/$totalTests passed")
  println("=" * 70)