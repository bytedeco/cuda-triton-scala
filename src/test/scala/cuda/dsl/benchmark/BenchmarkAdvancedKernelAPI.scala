package cuda.dsl.benchmark

import cuda.dsl.dsl.{TritonKernelMacro, tl}
import cuda.dsl.runtime._

/** Benchmark: call kernels from three existing test files using the two
 *  function-as-parameter execution modes:
 *    [A] execute(fn, n, blockSize)     — auto params from given TritonKernel[F]
 *    [B] execute(fn, params, n, bs)  — explicit params override given
 *
 *  Run with: sbt "runMain cuda.dsl.benchmark.BenchmarkAdvancedKernelAPI"
 */

// ── Type aliases (package level for use in both object and main) ───────────────
type TK_forLoop = (Float, Float, Int, Int) => Unit
type TK_warpReduce = (Float, Float, Int) => Unit
type TK_tiledGemm = (Float, Float, Float, Int, Int, Int) => Unit
type TK_blockSoftmax = (Float, Float, Int, Int) => Unit

// ── Object-level kernels (required for F: TritonKernel context bounds) ───────────
object BenchmarkAdvancedKernelAPIKernels:

  @TritonKernelMacro
  def forLoopKernel(outPtr: Float, inPtr: Float, N: Int, D: Int): Unit =
    0.until(N) foreach { i =>
      var sum: Float = 0.0f
      0.until(D) foreach { d =>
        val x = tl.load(inPtr + i * D + d)
        sum = sum + x * x
      }
      val result = tl.sqrt(sum)
      tl.store(outPtr + i, result)
    }
    ()

  @TritonKernelMacro
  def warpReduceKernel(outPtr: Float, inPtr: Float, N: Int): Unit =
    val tid = tl.program_id(0)
    if (tid >= N) return
    var localMax: Float = tl.load(inPtr + tid)
    var s = 1
    while s < 32 do
      val other = tl.load(inPtr + tid)
      if other > localMax then localMax = other
      s = s * 2
    tl.store(outPtr + tid, localMax)
    ()

  @TritonKernelMacro
  def tiledGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit =
    val pid = tl.program_id(0)
    val bM = 32; val bN = 32
    val nBM = (M + bM - 1) / bM; val nBN = (N + bN - 1) / bN
    val nb = nBM * nBN
    if (pid >= nb) return
    val bIM = pid / nBN; val bIN = pid % nBN
    val rS = bIM * bM; val cS = bIN * bN
    val row = rS; val col = cS
    if (row < M && col < N) then
      var sum: Float = 0.0f
      0.until(K) foreach { k => sum = sum + tl.load(A + row * K + k) * tl.load(B + k * N + col) }
      tl.store(C + row * N + col, sum)

  @TritonKernelMacro
  def blockSoftmaxKernel2(X: Float, Y: Float, M: Int, N: Int): Unit =
    val pid = tl.program_id(0)
    if (pid >= M) return
    val tx = tl.threadIdx(0)
    if (tx < N) then
      var maxVal: Float = Float.MinValue
      0.until(N) foreach { i =>
        val v = tl.load(X + pid * N + i)
        maxVal = maxVal max v
      }
      var sum: Float = 0.0f
      0.until(N) foreach { i =>
        val v = tl.load(X + pid * N + i)
        sum = sum + tl.exp(v - maxVal)
      }
      val v = tl.load(X + pid * N + tx)
      val r = tl.exp(v - maxVal) / sum
      tl.store(Y + pid * N + tx, r)

  type TK_forLoop = (Float, Float, Int, Int) => Unit
  type TK_warpReduce = (Float, Float, Int) => Unit
  type TK_tiledGemm = (Float, Float, Float, Int, Int, Int) => Unit
  type TK_blockSoftmax = (Float, Float, Int, Int) => Unit

  given TritonKernel_forLoop: cuda.dsl.runtime.TritonKernel[TK_forLoop] with
    val name = "forLoopKernel"
    val paramTypes = List("float*","float*","int","int")
    val paramNames = List("outPtr","inPtr","N","D")

  given TritonKernel_warpReduce: cuda.dsl.runtime.TritonKernel[TK_warpReduce] with
    val name = "warpReduceKernel"
    val paramTypes = List("float*","float*","int")
    val paramNames = List("outPtr","inPtr","N")

  given TritonKernel_tiledGemm: cuda.dsl.runtime.TritonKernel[TK_tiledGemm] with
    val name = "tiledGemmKernel"
    val paramTypes = List("float*","float*","float*","int","int","int")
    val paramNames = List("C","A","B","M","N","K")

  given TritonKernel_blockSoftmax: cuda.dsl.runtime.TritonKernel[TK_blockSoftmax] with
    val name = "blockSoftmaxKernel2"
    val paramTypes = List("float*","float*","int","int")
    val paramNames = List("X","Y","M","N")

// ── Main benchmark entry point ───────────────────────────────────────────────────
@main def BenchmarkAdvancedKernelAPI(): Unit =
  import BenchmarkAdvancedKernelAPIKernels.given

  // Register kernels from source files (triggers @TritonKernelMacro expansion)
  println("Registering kernels from source files...")
  TestAdvancedCudaFeatures.main(Array())
  TestStoreKVCacheGeneric.main(Array())
  Test100ComplexKernels.main(Array())
  val registered = cuda.dsl.dsl.TritonKernelRegistry.listAll().size
  println(s"$registered kernels registered in TritonKernelRegistry")

  val N = 1024; val M = 128; val K = 128; val D = 64
  var totalAuto = 0; var totalParams = 0; var totalByName = 0
  val start = System.nanoTime()

  println("\n" + "=" * 70)
  println("BenchmarkAdvancedKernelAPI — three execution modes")
  println("=" * 70)

  // [A] execute(fn, tk, n, bs) — explicit TritonKernel[F] avoids ambiguity
  println("\n[Mode A] execute(fn, tk, n, bs) — explicit TritonKernel[F]")
  import BenchmarkAdvancedKernelAPIKernels.{TritonKernel_forLoop, TritonKernel_warpReduce, TritonKernel_tiledGemm, TritonKernel_blockSoftmax}
  totalAuto += runAuto("forLoopKernel",       BenchmarkAdvancedKernelAPIKernels.forLoopKernel.asInstanceOf[TK_forLoop],       TritonKernel_forLoop, N*D, 256)
  totalAuto += runAuto("warpReduceKernel",   BenchmarkAdvancedKernelAPIKernels.warpReduceKernel.asInstanceOf[TK_warpReduce],   TritonKernel_warpReduce, N,   256)
  totalAuto += runAuto("tiledGemmKernel",    BenchmarkAdvancedKernelAPIKernels.tiledGemmKernel.asInstanceOf[TK_tiledGemm],    TritonKernel_tiledGemm, M*N, 256)
  totalAuto += runAuto("blockSoftmaxKernel2", BenchmarkAdvancedKernelAPIKernels.blockSoftmaxKernel2.asInstanceOf[TK_blockSoftmax], TritonKernel_blockSoftmax, M*N, 256)

  // [B] execute(fn, params, tk, n, bs) — explicit KernelParam list + explicit TK
  println("\n[Mode B] execute(fn, params, tk, n, bs)")
  val forLoopP = List(OutputBuffer("out", N*D), BufferParam("in", N*D), ScalarParam("int"), ScalarParam("int"))
  val warpP    = List(OutputBuffer("out", N),   BufferParam("in", N),   ScalarParam("int"))
  val gemmP    = List(OutputBuffer("C", M*N),   BufferParam("A", M*K), BufferParam("B", K*N), ScalarParam("int"), ScalarParam("int"), ScalarParam("int"))
  val softP    = List(OutputBuffer("Y", M*N),   BufferParam("X", M*N), ScalarParam("int"), ScalarParam("int"))

  totalParams += runParams("forLoopKernel",       BenchmarkAdvancedKernelAPIKernels.forLoopKernel.asInstanceOf[TK_forLoop],       forLoopP, TritonKernel_forLoop, N*D, 256)
  totalParams += runParams("warpReduceKernel",   BenchmarkAdvancedKernelAPIKernels.warpReduceKernel.asInstanceOf[TK_warpReduce],   warpP,   TritonKernel_warpReduce, N,   256)
  totalParams += runParams("tiledGemmKernel",    BenchmarkAdvancedKernelAPIKernels.tiledGemmKernel.asInstanceOf[TK_tiledGemm],    gemmP,   TritonKernel_tiledGemm, M*N, 256)
  totalParams += runParams("blockSoftmaxKernel2", BenchmarkAdvancedKernelAPIKernels.blockSoftmaxKernel2.asInstanceOf[TK_blockSoftmax], softP, TritonKernel_blockSoftmax, M*N, 256)

  // [C] execute(desc) — by name, kernels from source files (via registry)
  println("\n[Mode C] execute(desc) — by name (source file kernels)")
  totalByName += runByName("forLoopKernel",       N*D, 256)
  totalByName += runByName("warpReduceKernel",    N,   256)
  totalByName += runByName("tiledMatmulKernel",    M*N, 256)
  totalByName += runByName("warpButterflyKernel",  N,   256)
  totalByName += runByName("blockReduceKernel",    N,   256)
  totalByName += runByName("storeKVCacheKernel",  N*D, 256)
  totalByName += runByName("tiledGemm64x64Kernel",  M*N, 256)
  totalByName += runByName("flashAttentionKernel", M*K, 256)

  val elapsed = (System.nanoTime() - start) / 1e9
  println(f"\n${"="*70}")
  println(f"COMPLETE  $elapsed%.1fs")
  println(f"auto: $totalAuto  |  params: $totalParams  |  by-name: $totalByName")
  println(s"${"="*70}")

// ── Helper functions (at package level) ───────────────────────────────────────
private def runAuto[F](name: String, fn: F, tk: cuda.dsl.runtime.TritonKernel[F], n: Int, bs: Int = 256): Int =
  try
    val r = ScalaCudaRuntime.execute(fn, tk, n, bs)
    if r.isDefined then
      println(s"  [OK]  $name  (${r.get.length} elems, ${bs} threads)")
      1
    else
      println(s"  [FAIL] $name  (runtime None)")
      0
  catch case e: Exception =>
    println(s"  [ERR]  $name: ${e.getMessage.take(100)}")
    0

private def runParams[F](name: String, fn: F, params: List[KernelParam], tk: cuda.dsl.runtime.TritonKernel[F], n: Int, bs: Int = 256): Int =
  try
    val r = ScalaCudaRuntime.execute(fn, params, tk, n, bs)
    if r.isDefined then
      println(s"  [OK]  $name  (params, ${r.get.length} elems)")
      1
    else
      println(s"  [FAIL] $name  (params, runtime None)")
      0
  catch case e: Exception =>
    println(s"  [ERR]  $name: ${e.getMessage.take(100)}")
    0

private def runByName(name: String, n: Int, bs: Int = 256): Int =
  try
    val desc = KernelDesc(name, List(OutputBuffer("out", n)),
                          cuda.dsl.core.dim3((n+255)/256), cuda.dsl.core.dim3(bs))
    val r = ScalaCudaRuntime.execute(desc)
    if r.isDefined then
      println(s"  [OK]  $name  (by-name, ${r.get.length} elems)")
      1
    else
      println(s"  [FAIL] $name  (by-name)")
      0
  catch case e: Exception =>
    println(s"  [ERR]  $name: ${e.getMessage.take(100)}")
    0
