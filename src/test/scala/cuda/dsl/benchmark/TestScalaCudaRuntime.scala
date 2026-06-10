package cuda.dsl.benchmark

import cuda.dsl.core.dim3
import cuda.dsl.dsl.{TritonKernelMacro, tl}
import cuda.dsl.runtime.*

/** Test harness: execute(fn, n, blockSize) with given TritonKernel instances.
 *  Usage:
 *    sbt "runMain cuda.dsl.runtime.TestScalaCudaRuntime"
 */
object TestScalaCudaRuntime:

  @TritonKernelMacro
  def reluKernel(outPtr: Float, inPtr: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val x = tl.load(inPtr + i)
      val r = if x > 0.0f then x else 0.0f
      tl.store(outPtr + i, r)
    }
    ()
  }

  @TritonKernelMacro
  def saxpyKernel(outPtr: Float, xPtr: Float, yPtr: Float, a: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val x = tl.load(xPtr + i)
      val y = tl.load(yPtr + i)
      tl.store(outPtr + i, a * x + y)
    }
    ()
  }

  @TritonKernelMacro
  def vectorAddKernel(outPtr: Float, aPtr: Float, bPtr: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val a = tl.load(aPtr + i)
      val b = tl.load(bPtr + i)
      tl.store(outPtr + i, a + b)
    }
    ()
  }

  // Given instances for function-as-parameter API
  given triton_relu: TritonKernel[(Float, Float, Int) => Unit] =
    TritonKernel[(Float, Float, Int) => Unit]("reluKernel", List("float*", "float*", "int"), List("outPtr", "inPtr", "n"))

  given triton_saxpy: TritonKernel[(Float, Float, Float, Float, Int) => Unit] =
    TritonKernel[(Float, Float, Float, Float, Int) => Unit]("saxpyKernel", List("float*", "float*", "float*", "int"), List("outPtr", "xPtr", "yPtr", "a", "n"))

  given triton_vectorAdd: TritonKernel[(Float, Float, Float, Int) => Unit] =
    TritonKernel[(Float, Float, Float, Int) => Unit]("vectorAddKernel", List("float*", "float*", "float*", "int"), List("outPtr", "aPtr", "bPtr", "n"))

  def main(args: Array[String]): Unit =
    println("=" * 60)
    println("Test: execute(fn, n, blockSize) — name + params from given")
    println("=" * 60)

    // Auto: just pass fn + sizes
    val result1 = ScalaCudaRuntime.execute(reluKernel, 1024, 256)
    println(s"ReLU: ${result1.map(r => s"${r.take(3).mkString(", ")}...").getOrElse("FAILED")}")

    // With explicit params
    val params = List(OutputBuffer("out", 1024), BufferParam("x", 1024), BufferParam("y", 1024), ScalarParam("float"), ScalarParam("int"))
    val result2 = ScalaCudaRuntime.execute(saxpyKernel, params, 1024, 256)
    println(s"SAXPY: ${result2.map(r => s"${r.take(3).mkString(", ")}...").getOrElse("FAILED")}")

    println("=" * 60)
    println("Done!")
