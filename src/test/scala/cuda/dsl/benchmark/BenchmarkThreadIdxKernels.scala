package cuda.dsl.benchmark

import cuda.dsl.core.dim3
import cuda.dsl.runtime._
import cuda.dsl.dsl._

object ThreadIdxTestKernels:

  @TritonKernelMacro
  def testSoftmaxKernel(out: Float, in: Float, rows: Int, cols: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= rows) return
    val tx = tl.threadIdx(0)
    val rowStart = pid * cols
    var mv = Float.MinValue
    0.until(cols) foreach { i => mv = mv max tl.load(in + rowStart + i) }
    val rm = mv
    var se: Float = 0.0f
    0.until(cols) foreach { i => se = se + scala.math.exp((tl.load(in + rowStart + i) - rm).toDouble).toFloat }
    if (tx < cols) tl.store(out + rowStart + tx, scala.math.exp((tl.load(in + rowStart + tx) - rm).toDouble).toFloat / se)
    ()
  }

  @TritonKernelMacro
  def testLayerNormKernel(out: Float, in: Float, rows: Int, cols: Int, eps: Float): Unit = {
    val pid = tl.program_id(0)
    if (pid >= rows) return
    val tx = tl.threadIdx(0)
    val rowStart = pid * cols
    var mean: Float = 0.0f
    0.until(cols) foreach { i => mean = mean + tl.load(in + rowStart + i) }
    mean = mean / cols.toFloat
    var vari: Float = 0.0f
    0.until(cols) foreach { i => val d = tl.load(in + rowStart + i) - mean; vari = vari + d * d }
    vari = vari / cols.toFloat + eps
    val std = scala.math.sqrt(vari.toDouble).toFloat
    if (tx < cols) tl.store(out + rowStart + tx, (tl.load(in + rowStart + tx) - mean) / std)
    ()
  }

  @TritonKernelMacro
  def testGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
    val pid = tl.program_id(0)
    val bM = 32; val bN = 32
    val nBM = (M + bM - 1) / bM; val nBN = (N + bN - 1) / bN
    val nb = nBM * nBN
    if (pid >= nb) return
    val bIM = pid / nBN; val bIN = pid % nBN
    val rS = bIM * bM; val cS = bIN * bN
    val tx = tl.threadIdx(0); val ty = tl.threadIdx(1)
    val row = rS + tx; val col = cS + ty
    if (row < M && col < N) {
      var sum: Float = 0.0f
      0.until(K) foreach { k => sum = sum + tl.load(A + row * K + k) * tl.load(B + k * N + col) }
      tl.store(C + row * N + col, sum)
    }
    ()
  }

  given triton_softmax: cuda.dsl.runtime.TritonKernel[(Float, Float, Int, Int) => Unit] =
    cuda.dsl.runtime.TritonKernel[(Float, Float, Int, Int) => Unit]("testSoftmaxKernel", List("float*","float*","int","int"), List("out","in","rows","cols"))

  given triton_layernorm: cuda.dsl.runtime.TritonKernel[(Float, Float, Int, Int, Float) => Unit] =
    cuda.dsl.runtime.TritonKernel[(Float, Float, Int, Int, Float) => Unit]("testLayerNormKernel", List("float*","float*","int","int","float"), List("out","in","rows","cols","eps"))

  given triton_gemm: cuda.dsl.runtime.TritonKernel[(Float, Float, Float, Int, Int, Int) => Unit] =
    cuda.dsl.runtime.TritonKernel[(Float, Float, Float, Int, Int, Int) => Unit]("testGemmKernel", List("float*","float*","float*","int","int","int"), List("C","A","B","M","N","K"))

  def runAuto[F: cuda.dsl.runtime.TritonKernel](fn: F): Int =
    try
      val r = ScalaCudaRuntime.execute(fn, 1024, 256)
      val ok = r.isDefined
      val name = summon[cuda.dsl.runtime.TritonKernel[F]].name
      if ok then
        println("  [OK] " + name)
        1
      else
        println("  [FAIL] " + name)
        0
    catch case e: Exception =>
      println("  [ERR] " + e.getMessage.take(80))
      0

  def runParams[F: cuda.dsl.runtime.TritonKernel](fn: F, n: Int): Int =
    try
      val name = summon[cuda.dsl.runtime.TritonKernel[F]].name
      val ps: List[KernelParam] = name match
        case "testSoftmaxKernel" => List(OutputBuffer("out", n), BufferParam("in", n), ScalarParam("int"), ScalarParam("int"))
        case "testLayerNormKernel" => List(OutputBuffer("out", n), BufferParam("in", n), ScalarParam("int"), ScalarParam("int"), ScalarParam("float"))
        case "testGemmKernel" => List(OutputBuffer("C", n), BufferParam("A", n), BufferParam("B", n), ScalarParam("int"), ScalarParam("int"), ScalarParam("int"))
        case _ => List(OutputBuffer("out", n))
      val r = ScalaCudaRuntime.execute(fn, ps, n, 256)
      val ok = r.isDefined
      if ok then
        println("  [OK] " + name + " (params)")
        1
      else
        println("  [FAIL] " + name + " (params)")
        0
    catch case e: Exception =>
      println("  [ERR] " + e.getMessage.take(80))
      0

  def runByName(name: String, n: Int): Int =
    try
      val desc = KernelDesc(name, List(OutputBuffer("out", n)), dim3((n+255)/256), dim3(256))
      val r = ScalaCudaRuntime.execute(desc)
      if r.isDefined then
        println("  [OK] " + name + " (by-name)")
        1
      else
        println("  [FAIL] " + name)
        0
    catch case e: Exception =>
      println("  [ERR] " + name + ": " + e.getMessage.take(80))
      0

@main def BenchmarkThreadIdxKernels(): Unit =
  val N = 1024
  println("=" * 70)
  println("execute(fn, n, bs) vs execute(fn, params, n, bs)")
  println("=" * 70)

  var totalAuto = 0
  var totalParams = 0
  val start = System.nanoTime()

  import ThreadIdxTestKernels.{testSoftmaxKernel, testLayerNormKernel, testGemmKernel}
  import ThreadIdxTestKernels.given

  println("\n[1] execute(fn, n, blockSize) — auto")
  totalAuto += ThreadIdxTestKernels.runAuto(testSoftmaxKernel)
  totalAuto += ThreadIdxTestKernels.runAuto(testLayerNormKernel)
  totalAuto += ThreadIdxTestKernels.runAuto(testGemmKernel)

  println("\n[2] execute(fn, params, n, blockSize) — explicit")
  totalParams += ThreadIdxTestKernels.runParams(testSoftmaxKernel, N*N)
  totalParams += ThreadIdxTestKernels.runParams(testLayerNormKernel, N*N)
  totalParams += ThreadIdxTestKernels.runParams(testGemmKernel, N*N)

  println("\n[3] execute(desc) — by name")
  Test50ThreadIdxKernels.main(Array())
  totalParams += ThreadIdxTestKernels.runByName("warpTileGemm4x4LaneKernel", N*N)
  totalParams += ThreadIdxTestKernels.runByName("blockSoftmaxKernel", N*N)

  val elapsed = (System.nanoTime() - start) / 1e9
  println(f"\nCOMPLETE $elapsed%.1fs")
  println(s"auto: $totalAuto | params: $totalParams")
  println("=" * 70)
