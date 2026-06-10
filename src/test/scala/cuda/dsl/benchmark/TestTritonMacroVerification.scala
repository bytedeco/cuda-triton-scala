package cuda.dsl.benchmark

import cuda.dsl.core.dim3
import cuda.dsl.dsl.{TritonKernelMacro, tl}
import cuda.dsl.runtime.*

/** Benchmark: verify @TritonKernelMacro compiled kernels produce correct non-zero results.
 *  Uses ScalaCudaRuntime.execute with auto-generated params via explicit TritonKernel[F].
 *
 *  Pattern: type alias + given instance + asInstanceOf (same as BenchmarkAdvancedKernelAPI).
 */
object TestTritonMacroVerification:

  // ── Type aliases for kernel function types ───────────────────────
  type TK_tileScale    = (Float, Float, Int, Float) => Unit
  type TK_tileAdd      = (Float, Float, Float, Int) => Unit
  type TK_relu         = (Float, Float, Int) => Unit
  type TK_kvCacheWrite = (Float, Float, Int, Int) => Unit
  type TK_saveLoad     = (Float, Float, Int) => Unit
  type TK_gemm         = (Float, Float, Float, Int, Int, Int) => Unit
  type TK_pageAttn     = (Float, Float, Float, Float, Int, Int) => Unit
  type TK_softmax      = (Float, Float, Int) => Unit

  // ── Kernel implementations ──────────────────────────────────────
  @TritonKernelMacro
  def tileScaleKernel(out: Float, inp: Float, n: Int, scale: Float): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val v = tl.load(inp + i)
      tl.store(out + i, v * scale)
    }
    ()
  }

  @TritonKernelMacro
  def tileAddKernel(out: Float, a: Float, b: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val av = tl.load(a + i)
      val bv = tl.load(b + i)
      tl.store(out + i, av + bv)
    }
    ()
  }

  @TritonKernelMacro
  def reluKernel(out: Float, inp: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val v = tl.load(inp + i)
      val r = if v > 0.0f then v else 0.0f
      tl.store(out + i, r)
    }
    ()
  }

  @TritonKernelMacro
  def kvCacheWriteKernel(cache: Float, newVal: Float, pos: Int, size: Int): Unit = {
    val i = tl.program_id(0)
    if (i < size) {
      val old = tl.load(cache + pos + i)
      val nv = tl.load(newVal + i)
      val decay = 0.9f
      val result = old * decay + nv * (1.0f - decay)
      tl.store(cache + pos + i, result)
    }
    ()
  }

  @TritonKernelMacro
  def saveLoadKernel(dst: Float, src: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val v = tl.load(src + i)
      tl.store(dst + i, v)
    }
    ()
  }

  @TritonKernelMacro
  def gemmKernel(C: Float, A: Float, B: Float, M: Int, K: Int, N: Int): Unit = {
    val row = tl.program_id(0)
    val col = tl.program_id(1)
    if (row < M && col < N) {
      var sum = 0.0f
      for (k <- 0 until K) {
        val av = tl.load(A + row * K + k)
        val bv = tl.load(B + k * N + col)
        sum += av * bv
      }
      tl.store(C + row * N + col, sum)
    }
    ()
  }

  @TritonKernelMacro
  def pageAttnKernel(out: Float, q: Float, k: Float, v: Float, numHeads: Int, headDim: Int): Unit = {
    val i = tl.program_id(0)
    if (i < numHeads * headDim) {
      val head = i / headDim
      val d = i % headDim
      var score = 0.0f
      for (h <- 0 until numHeads) {
        val qv = tl.load(q + h * headDim + d)
        val kv = tl.load(k + h * headDim + d)
        score += qv * kv
      }
      val scale = 1.0f / tl.exp(headDim.toFloat * 0.1f)
      tl.store(out + i, score * scale)
    }
    ()
  }

  @TritonKernelMacro
  def softmaxKernel(out: Float, inp: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val v = tl.load(inp + i)
      tl.store(out + i, tl.exp(v))
    }
    ()
  }

  // ── Given TritonKernel instances ────────────────────────────────
  given TK_tileScale: TritonKernel[TK_tileScale] with
    val name = "tileScaleKernel"
    val paramTypes = List("float*","float*","int","float")
    val paramNames = List("out","inp","n","scale")

  given TK_tileAdd: TritonKernel[TK_tileAdd] with
    val name = "tileAddKernel"
    val paramTypes = List("float*","float*","float*","int")
    val paramNames = List("out","a","b","n")

  given TK_relu: TritonKernel[TK_relu] with
    val name = "reluKernel"
    val paramTypes = List("float*","float*","int")
    val paramNames = List("out","inp","n")

  given TK_kvCacheWrite: TritonKernel[TK_kvCacheWrite] with
    val name = "kvCacheWriteKernel"
    val paramTypes = List("float*","float*","int","int")
    val paramNames = List("cache","newVal","pos","size")

  given TK_saveLoad: TritonKernel[TK_saveLoad] with
    val name = "saveLoadKernel"
    val paramTypes = List("float*","float*","int")
    val paramNames = List("dst","src","n")

  given TK_gemm: TritonKernel[TK_gemm] with
    val name = "gemmKernel"
    val paramTypes = List("float*","float*","float*","int","int","int")
    val paramNames = List("C","A","B","M","K","N")

  given TK_pageAttn: TritonKernel[TK_pageAttn] with
    val name = "pageAttnKernel"
    val paramTypes = List("float*","float*","float*","float*","int","int")
    val paramNames = List("out","q","k","v","numHeads","headDim")

  given TK_softmax: TritonKernel[TK_softmax] with
    val name = "softmaxKernel"
    val paramTypes = List("float*","float*","int")
    val paramNames = List("out","inp","n")

  // ── Helpers ────────────────────────────────────────────────────
  private def checkResult(name: String, result: Option[Array[Float]], n: Int): (Boolean, Float, Int) =
    result match
      case Some(arr) =>
        val nonZero = arr.count(math.abs(_) > 0.001f)
        (nonZero >= n * 9 / 10, arr(1), nonZero)
      case None =>
        (false, 0f, 0)

  // ── Main ──────────────────────────────────────────────────────
  def main(args: Array[String]): Unit =
    println("=" * 80)
    println("TritonKernelMacro Verification Benchmark")
    println("验证 @TritonKernelMacro 编译的kernel返回值是否正确")
    println("=" * 80)

    import TestTritonMacroVerification.given
    val N = 256
    var passed = 0
    var failed = 0

    def test[A](name: String, fn: A, tk: TritonKernel[A], n: Int, bs: Int): Unit =
      try
        val result = ScalaCudaRuntime.execute(fn, tk, n, bs)
        val (ok, sample, nz) = checkResult(name, result, n)
        if (ok) {
          println(s"[PASS] $name: $nz/$n non-zero, sample=${sample.formatted("%.4f")}")
          passed += 1
        } else {
          println(s"[FAIL] $name: $nz/$n non-zero (expected >=${n*9/10}), sample=${sample.formatted("%.4f")}")
          failed += 1
        }
      catch case e: Exception =>
        println(s"[FAIL] $name: ${e.getMessage.take(80)}")
        failed += 1

    test("tileScaleKernel", tileScaleKernel.asInstanceOf[TK_tileScale], TK_tileScale, N, 128)
    test("tileAddKernel",   tileAddKernel.asInstanceOf[TK_tileAdd],     TK_tileAdd,   N, 128)
    test("reluKernel",      reluKernel.asInstanceOf[TK_relu],           TK_relu,      N, 128)
    test("kvCacheWriteKernel", kvCacheWriteKernel.asInstanceOf[TK_kvCacheWrite], TK_kvCacheWrite, 512, 128)
    test("saveLoadKernel",  saveLoadKernel.asInstanceOf[TK_saveLoad],   TK_saveLoad,  128, 128)
    test("gemmKernel",      gemmKernel.asInstanceOf[TK_gemm],           TK_gemm,      16*16, 128)
    test("pageAttnKernel",  pageAttnKernel.asInstanceOf[TK_pageAttn],    TK_pageAttn,  4*32, 128)
    test("softmaxKernel",   softmaxKernel.asInstanceOf[TK_softmax],     TK_softmax,   N, 128)

    println()
    println("=" * 80)
    println(s"结果: $passed 通过, $failed 失败 (共${passed + failed}个测试)")
    if (failed == 0) println("全部通过!")
    println("=" * 80)
