package cuda.dsl.benchmark.cutlass

import cuda.dsl.cutlass.{CutlassGemm, CutlassConv}
import cuda.dsl.core.*
import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR

/** Benchmark for CUTLASS GEMM operations.
 *
 * Tests various GEMM configurations following CUTLASS patterns.
 */
object CUTLASSGemmBenchmark:

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("CUTLASS GEMM Benchmark")
    println("=" * 80)

    // Force initialization of CUTLASS modules
    val _ = CutlassGemm
    val __ = CutlassConv

    // GEMM kernel configs
    val gemmConfigs = List(
      ("tiledGemm64x64", 64, 64),
      ("tiledGemm128x128", 128, 128),
      ("tiledGemm128x64", 128, 64),
      ("tiledGemm64x128", 64, 128),
      ("tiledGemm128x32", 128, 32)
    )

    println("\n--- CUTLASS GEMM Kernels ---")
    gemmConfigs.foreach { (name, M, N) =>
      println(s"\n--- Testing $name ($M x $N) ---")
      val result = SCR.executeKernelByName(name, M, N)
      result match {
        case Some(out) =>
          val finite = out.count(v => !v.isNaN && !v.isInfinite)
          val sum = out.filter(v => !v.isNaN && !v.isInfinite).sum
          println(s"  Result: $finite/$M finite elements")
          println(s"  Sum: ${sum.formatted("%.2f")}")
        case None =>
          println(s"  ERROR: No result")
      }
    }

    // Row-major GEMM test
    println("\n--- Row-major GEMM ---")
    val rowResult = SCR.executeKernelByName("rowMajorGemmKernel", 64, 64)
    rowResult match {
      case Some(out) =>
        val finite = out.count(v => !v.isNaN && !v.isInfinite)
        val sum = out.filter(v => !v.isNaN && !v.isInfinite).sum
        println(s"  Result: $finite finite elements")
        println(s"  Sum: ${sum.formatted("%.2f")}")
      case None =>
        println(s"  ERROR: No result")
    }

    // Conv kernel configs
    val convConfigs = List(
      ("conv2dKernel", 1, 1),
      ("pointwiseConv2dKernel", 1, 1)
    )

    println("\n--- CUTLASS Conv Kernels ---")
    convConfigs.foreach { (name, N, K) =>
      println(s"\n--- Testing $name ---")
      val result = SCR.executeKernelByName(name, N, K)
      result match {
        case Some(out) =>
          val finite = out.count(v => !v.isNaN && !v.isInfinite)
          val sum = out.filter(v => !v.isNaN && !v.isInfinite).sum
          println(s"  Result: $finite finite elements")
          println(s"  Sum: ${sum.formatted("%.2f")}")
        case None =>
          println(s"  ERROR: No result")
      }
    }

    println("\n" + "=" * 80)
    println("CUTLASS GEMM Benchmark Complete")
    println("=" * 80)
  }

// Force initialization of CUTLASS modules for kernel registration
private val _init = (CutlassGemm)
private val _init2 = (CutlassConv)