package cuda.dsl.benchmark

import cuda.dsl.core.dim3
import cuda.dsl.runtime._
import TestScalaCudaRuntime.{reluKernel, saxpyKernel, vectorAddKernel, triton_relu, triton_saxpy, triton_vectorAdd}
import cuda.dsl.dsl._

/** Benchmark: execute(fn, n, blockSize) — name from given, params auto-built.
 *  Usage:
 *    sbt "runMain cuda.dsl.benchmark.Benchmark100Kernels"
 */
@main
def Benchmark100Kernels(): Unit =

  val N = 1024

  println("=" * 60)
  println("Benchmark100Kernels: execute(fn, n, blockSize)")
  println("=" * 60)

  var totalPassed = 0

  // Core API: execute(fn, n, blockSize) — no KernelDesc needed
  println("\n[Category 1] execute(fn, n, blockSize) — auto name + params")

  if runKernel(reluKernel, N, 256) > 0 then totalPassed += 1
  if runKernel(saxpyKernel, N, 256) > 0 then totalPassed += 1
  if runKernel(vectorAddKernel, N, 256) > 0 then totalPassed += 1

  // With explicit params: execute(fn, params, n, blockSize)
  println("\n[Category 2] execute(fn, params, n, blockSize)")
  if runKernelParams(reluKernel, N, 256) > 0 then totalPassed += 1

  // Registered by name (no given)
  println("\n[Category 3] Registered kernels via Test100ComplexKernels.main()")
  Test100ComplexKernels.main(Array())
  val desc = KernelDesc("tiledGemm64x64Kernel",
    List(OutputBuffer("C", N*N), BufferParam("A", N*N), BufferParam("B", N*N), ScalarParam("int"), ScalarParam("int"), ScalarParam("int")),
    dim3((N*N + 255) / 256), dim3(256))
  if ScalaCudaRuntime.execute(desc).isDefined then
    println("  [OK] tiledGemm64x64Kernel (name-based)")
    totalPassed += 1

  println(s"\n  Total: $totalPassed passed")
  println("=" * 60)

/** execute(fn, n, blockSize) — name + params from given */
private def runKernel[F: cuda.dsl.runtime.TritonKernel](fn: F, n: Int, blockSize: Int): Int =
  try
    val result = ScalaCudaRuntime.execute(fn, n, blockSize)
    if result.isDefined then { println(s"  [OK] (fn, $n, $blockSize)"); 1 }
    else { println(s"  [FAIL]"); 0 }
  catch
    case e: Exception => println(s"  [ERROR] ${e.getMessage.take(60)}"); 0

/** execute(fn, params, n, blockSize) — explicit params */
private def runKernelParams[F: cuda.dsl.runtime.TritonKernel](fn: F, n: Int, blockSize: Int): Int =
  try
    val params = List(OutputBuffer("out", n), BufferParam("in", n), ScalarParam("int"))
    val result = ScalaCudaRuntime.execute(fn, params, n, blockSize)
    if result.isDefined then { println(s"  [OK] (fn, params, $n, $blockSize)"); 1 }
    else { println(s"  [FAIL]"); 0 }
  catch
    case e: Exception => println(s"  [ERROR] ${e.getMessage.take(60)}"); 0
