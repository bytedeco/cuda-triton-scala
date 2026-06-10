package cuda.dsl.benchmark.cutlite

import cuda.dsl.cutlite.{CutliteMath, CutliteReduce}
import cuda.dsl.core.*
import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR

/** Benchmark for CUTLITE operations.
 *
 * Tests various CUTLITE-style operations.
 */
object CUTLITEMathBenchmark:

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("CUTLITE Math Benchmark")
    println("=" * 80)

    // Force initialization of CUTLITE modules
    val _ = CutliteMath
    val __ = CutliteReduce

    // Math kernel tests
    val mathConfigs = List(
      ("softMaxKernel", 32, 64),
      ("geluKernel", 256, 1),
      ("swigluKernel", 256, 1),
      ("sigmoidKernel", 256, 1),
      ("reluKernel", 256, 1),
      ("layerNormKernel", 32, 64),
      ("rmsNormKernel", 32, 64)
    )

    println("\n--- CUTLITE Math Kernels ---")
    mathConfigs.foreach { (name, N, D) =>
      println(s"\n--- Testing $name ($N x $D) ---")
      val result = SCR.executeKernelByName(name, N, D)
      result match {
        case Some(out) =>
          val finite = out.count(v => !v.isNaN && !v.isInfinite)
          val sum = out.filter(v => !v.isNaN && !v.isInfinite).sum
          println(s"  Result: $finite/$N finite elements")
          println(s"  Sum: ${sum.formatted("%.2f")}")
        case None =>
          println(s"  ERROR: No result")
      }
    }

    // Reduction kernel tests
    val reduceConfigs = List(
      ("rowReduceSumKernel", 32, 64),
      ("rowReduceMaxKernel", 32, 64),
      ("prefixSumKernel", 256, 1),
      ("cumsumKernel", 256, 1)
    )

    println("\n--- CUTLITE Reduction Kernels ---")
    reduceConfigs.foreach { (name, N, D) =>
      println(s"\n--- Testing $name ($N x $D) ---")
      val result = SCR.executeKernelByName(name, N, D)
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
    println("CUTLITE Math Benchmark Complete")
    println("=" * 80)
  }

// Force initialization of CUTLITE modules for kernel registration
private val _init = (CutliteMath)
private val _init2 = (CutliteReduce)