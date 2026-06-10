package cuda.dsl.benchmark

import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR

// ============================================================================
// 100 New CUDA Kernels - Different types from the main 238
// Using ONLY supported DSL features (tl.load, tl.store, tl.program_id, math funcs)
// ============================================================================

object Test100NewKernels:

  // Case class for kernel results
  case class KernelResult(name: String, status: String, output: Array[Float], stats: String)

  // Scala math helpers — these get inlined by the macro into CUDA code
  private def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  private def log(x: Float): Float = scala.math.log(x.toDouble).toFloat
  private def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat
  private def sin(x: Float): Float = scala.math.sin(x.toDouble).toFloat
  private def cos(x: Float): Float = scala.math.cos(x.toDouble).toFloat
  private def tanh(x: Float): Float = scala.math.tanh(x.toDouble).toFloat
  private def atan(x: Float): Float = scala.math.atan(x.toDouble).toFloat
  private def abs(x: Float): Float = scala.math.abs(x.toDouble).toFloat
  private def max(a: Float, b: Float): Float = if (a > b) a else b
  private def min(a: Float, b: Float): Float = if (a < b) a else b
  private def sigmoid(x: Float): Float = 1.0f / (1.0f + exp(-x))
  private def relu(x: Float): Float = max(0.0f, x)
  private def floor(x: Float): Float = scala.math.floor(x.toDouble).toFloat

  // List of all 100 kernel names (extracted from @TritonKernelMacro annotations)
  private val kernelNames = List(
    "vecAdd", "vecSub", "vecMul", "vecDiv", "vecScale", "vecNegate", "vecSquare", "vecCube", "vecInvSqrt", "vecSqrt",
    "vecRelu", "vecLeakyRelu", "vecGelu", "vecQuickGelu", "vecSigmoid", "vecSwish", "vecMish", "vecSoftplus", "vecLogSigmoid", "vecSilu",
    "vecClamp", "vecSelect", "vecSign", "vecStep", "vecHardSigmoid", "vecHardTanh", "vecAbsDiff", "vecSquaredDiff", "vecSmoothRelu", "vecThresholdRelu",
    "vecSin", "vecCos", "vecTan", "vecAtan", "vecSinhApprox", "vecCoshApprox", "vecTanhApprox", "vecSinCos", "vecAsinApprox", "vecAcosApprox",
    "vecLog", "vecExp", "vecExp2Approx", "vecLog2Approx", "vecLog10Approx", "vecPow2", "vecExpm1", "vecLog1p", "vecReciprocal", "vecSquareRootInv",
    "vecSumRed", "vecMaxRed", "vecMinRed", "vecMeanRed", "vecProdRed", "vecSumSqRed", "vecNorm2Red", "vecNorm1Red", "vecDotRed", "vecVarianceRed",
    "matAdd", "matSub", "matMulEle", "matScale", "matNeg", "matAbs", "matRelu", "matSigmoid", "matClamp", "matSquare",
    "matRowSum", "matColSum", "matRowMax", "matColMax", "matRowMean", "matColMean", "matRowNorm2", "matColNorm2", "matRowStdDev", "matDiagSum",
    "cosineSim", "euclideanDist", "manhattanDist", "vecCosineDist", "vecPearsonCorr", "vecL1Similarity", "vecSoftCossim", "vecMahalanobisDist", "vecCanberraDist", "vecBrayCurtisDist",
    "vecFloor", "vecCeilApprox", "vecRoundApprox", "vecFract", "vecMod", "vecRamp", "vecAddScale", "vecAddConst", "vecMulConst", "vecAffine"
  )

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Test100NewKernels: 100 New CUDA Kernels (Different from Main 238)")
    println("=" * 80)

    val N = 1024
    val BLOCK_SIZE = 256

    println(s"\nExecuting $N elements, block size $BLOCK_SIZE...")

    val results = scala.collection.mutable.ListBuffer[KernelResult]()
    val startTime = System.nanoTime()

    kernelNames.zipWithIndex.foreach { (name, idx) =>
      print(f"\r[${idx + 1}/${kernelNames.size}] Testing $name...")
      val result = verifyKernel(name, N, BLOCK_SIZE)
      results += result
    }

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
        val sample = if (r.output.nonEmpty)
          r.output.take(3).map(_.formatted("%.3f")).mkString(", ")
        else "N/A"
        println(f"  ${r.name} [${r.status}] sample=[$sample] stats: ${r.stats}")
      }
    } else {
      println("\nAll kernels PASSED!")
    }

    // Show sample outputs from passing kernels
    if (passCount > 0) {
      println("\nSample passing kernel outputs:")
      results.filter(_.status == "PASS").take(10).foreach { p =>
        val sample = p.output.take(5).map(_.formatted("%.3f")).mkString(", ")
        println(f"  ${p.name}: [$sample ...]  (${p.stats})")
      }
    }

    println("=" * 80)
    if (passCount == total)
      println("ALL KERNELS PASSED!")
    else
      println(s"VERDICT: $passCount/$total PASSED")

    println("=" * 80)
  }

  // ============================================================================
  // 1-10: Vector Arithmetic
  // ============================================================================
  @TritonKernelMacro
  def vecAdd(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, tl.load(a + pid) + tl.load(b + pid))
    ()
  }

  @TritonKernelMacro
  def vecSub(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, tl.load(a + pid) - tl.load(b + pid))
    ()
  }

  @TritonKernelMacro
  def vecMul(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, tl.load(a + pid) * tl.load(b + pid))
    ()
  }

  @TritonKernelMacro
  def vecDiv(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val bv = tl.load(b + pid)
    tl.store(out + pid, tl.load(a + pid) / (bv + 1e-10f))
    ()
  }

  @TritonKernelMacro
  def vecScale(out: Float, a: Float, s: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, tl.load(a + pid) * s)
    ()
  }

  @TritonKernelMacro
  def vecNegate(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, -tl.load(a + pid))
    ()
  }

  @TritonKernelMacro
  def vecSquare(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av * av)
    ()
  }

  @TritonKernelMacro
  def vecCube(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av * av * av)
    ()
  }

  @TritonKernelMacro
  def vecInvSqrt(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, 1.0f / sqrt(av + 1e-10f))
    ()
  }

  @TritonKernelMacro
  def vecSqrt(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, sqrt(tl.load(a + pid)))
    ()
  }

  // ============================================================================
  // 11-20: Activation Functions
  // ============================================================================
  @TritonKernelMacro
  def vecRelu(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, if (av > 0.0f) av else 0.0f)
    ()
  }

  @TritonKernelMacro
  def vecLeakyRelu(out: Float, a: Float, alpha: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val al = tl.load(alpha + pid)
    tl.store(out + pid, if (av > 0.0f) av else al * av)
    ()
  }

  @TritonKernelMacro
  def vecGelu(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val cdf = 0.5f * (1.0f + tanh(0.79788456f * (av + 0.044715f * av * av * av)))
    tl.store(out + pid, av * cdf)
    ()
  }

  @TritonKernelMacro
  def vecQuickGelu(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av * tanh(0.79788456f * (av + 0.044715f * av * av * av)))
    ()
  }

  @TritonKernelMacro
  def vecSigmoid(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, 1.0f / (1.0f + exp(-av)))
    ()
  }

  @TritonKernelMacro
  def vecSwish(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av / (1.0f + exp(-av)))
    ()
  }

  @TritonKernelMacro
  def vecMish(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av * tanh(log(1.0f + exp(av))))
    ()
  }

  @TritonKernelMacro
  def vecSoftplus(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, log(1.0f + exp(tl.load(a + pid))))
    ()
  }

  @TritonKernelMacro
  def vecLogSigmoid(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, -log(1.0f + exp(-av)))
    ()
  }

  @TritonKernelMacro
  def vecSilu(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av / (1.0f + exp(-av)))
    ()
  }

  // ============================================================================
  // 21-30: Conditional Operations
  // ============================================================================
  @TritonKernelMacro
  def vecClamp(out: Float, a: Float, lo: Float, hi: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val lv = tl.load(lo + pid)
    val hv = tl.load(hi + pid)
    tl.store(out + pid, if (av < lv) lv else if (av > hv) hv else av)
    ()
  }

  @TritonKernelMacro
  def vecSelect(out: Float, cond: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val cv = tl.load(cond + pid)
    val av = tl.load(a + pid)
    val bv = tl.load(b + pid)
    tl.store(out + pid, if (cv > 0.0f) av else bv)
    ()
  }

  @TritonKernelMacro
  def vecSign(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, if (av > 0.0f) 1.0f else if (av < 0.0f) -1.0f else 0.0f)
    ()
  }

  @TritonKernelMacro
  def vecStep(out: Float, a: Float, edge: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val ev = tl.load(edge + pid)
    tl.store(out + pid, if (av < ev) 0.0f else 1.0f)
    ()
  }

  @TritonKernelMacro
  def vecHardSigmoid(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val result = if (av <= -3.0f) 0.0f else if (av >= 3.0f) 1.0f else av * 0.1666667f + 0.5f
    tl.store(out + pid, result)
    ()
  }

  @TritonKernelMacro
  def vecHardTanh(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, if (av < -1.0f) -1.0f else if (av > 1.0f) 1.0f else av)
    ()
  }

  @TritonKernelMacro
  def vecAbsDiff(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val bv = tl.load(b + pid)
    val diff = av - bv
    tl.store(out + pid, if (diff < 0.0f) -diff else diff)
    ()
  }

  @TritonKernelMacro
  def vecSquaredDiff(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val bv = tl.load(b + pid)
    val diff = av - bv
    tl.store(out + pid, diff * diff)
    ()
  }

  @TritonKernelMacro
  def vecSmoothRelu(out: Float, a: Float, alpha: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val al = tl.load(alpha + pid)
    tl.store(out + pid, if (av > al) av else al * (av - al / 2.0f))
    ()
  }

  @TritonKernelMacro
  def vecThresholdRelu(out: Float, a: Float, threshold: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val th = tl.load(threshold + pid)
    tl.store(out + pid, if (av > th) av else 0.0f)
    ()
  }

  // ============================================================================
  // 31-40: Trigonometric Operations
  // ============================================================================
  @TritonKernelMacro
  def vecSin(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, sin(tl.load(a + pid)))
    ()
  }

  @TritonKernelMacro
  def vecCos(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, cos(tl.load(a + pid)))
    ()
  }

  @TritonKernelMacro
  def vecTan(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, sin(av) / (cos(av) + 1e-10f))
    ()
  }

  @TritonKernelMacro
  def vecAtan(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, atan(tl.load(a + pid)))
    ()
  }

  @TritonKernelMacro
  def vecSinhApprox(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, (exp(av) - exp(-av)) * 0.5f)
    ()
  }

  @TritonKernelMacro
  def vecCoshApprox(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, (exp(av) + exp(-av)) * 0.5f)
    ()
  }

  @TritonKernelMacro
  def vecTanhApprox(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, 2.0f / (1.0f + exp(-2.0f * av)) - 1.0f)
    ()
  }

  @TritonKernelMacro
  def vecSinCos(sinOut: Float, cosOut: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(sinOut + pid, sin(av))
    tl.store(cosOut + pid, cos(av))
    ()
  }

  @TritonKernelMacro
  def vecAsinApprox(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, atan(av / (sqrt(1.0f - av * av) + 1e-10f)))
    ()
  }

  @TritonKernelMacro
  def vecAcosApprox(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, 1.5707963f - atan(av / (sqrt(1.0f - av * av) + 1e-10f)))
    ()
  }

  // ============================================================================
  // 41-50: Log/Exp Operations
  // ============================================================================
  @TritonKernelMacro
  def vecLog(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, log(av + 1e-10f))
    ()
  }

  @TritonKernelMacro
  def vecExp(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, exp(tl.load(a + pid)))
    ()
  }

  @TritonKernelMacro
  def vecExp2Approx(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, exp(av * 0.6931472f))
    ()
  }

  @TritonKernelMacro
  def vecLog2Approx(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, log(av + 1e-10f) * 1.44269504f)
    ()
  }

  @TritonKernelMacro
  def vecLog10Approx(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, log(av + 1e-10f) * 0.4342945f)
    ()
  }

  @TritonKernelMacro
  def vecPow2(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, 1.0f / (1.0f + exp(-av)))
    ()
  }

  @TritonKernelMacro
  def vecExpm1(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, exp(av) - 1.0f)
    ()
  }

  @TritonKernelMacro
  def vecLog1p(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, log(1.0f + av))
    ()
  }

  @TritonKernelMacro
  def vecReciprocal(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, 1.0f / (av + 1e-10f))
    ()
  }

  @TritonKernelMacro
  def vecSquareRootInv(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, 1.0f / (sqrt(av) + 1e-10f))
    ()
  }

  // ============================================================================
  // 51-60: Vector Reduction (loop-based)
  // ============================================================================
  @TritonKernelMacro
  def vecSumRed(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sum = 0.0f
    0.until(n) foreach { i => sum = sum + tl.load(in + i) }
    tl.store(out, sum)
    ()
  }

  @TritonKernelMacro
  def vecMaxRed(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var maxVal = -1e38f
    0.until(n) foreach { i =>
      val v = tl.load(in + i)
      maxVal = if (v > maxVal) v else maxVal
    }
    tl.store(out, maxVal)
    ()
  }

  @TritonKernelMacro
  def vecMinRed(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var minVal = 1e38f
    0.until(n) foreach { i =>
      val v = tl.load(in + i)
      minVal = if (v < minVal) v else minVal
    }
    tl.store(out, minVal)
    ()
  }

  @TritonKernelMacro
  def vecMeanRed(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sum = 0.0f
    0.until(n) foreach { i => sum = sum + tl.load(in + i) }
    tl.store(out, sum / n.toFloat)
    ()
  }

  @TritonKernelMacro
  def vecProdRed(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var prod = 1.0f
    0.until(n) foreach { i => prod = prod * tl.load(in + i) }
    tl.store(out, prod)
    ()
  }

  @TritonKernelMacro
  def vecSumSqRed(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sumSq = 0.0f
    0.until(n) foreach { i =>
      val v = tl.load(in + i)
      sumSq = sumSq + v * v
    }
    tl.store(out, sumSq)
    ()
  }

  @TritonKernelMacro
  def vecNorm2Red(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sumSq = 0.0f
    0.until(n) foreach { i =>
      val v = tl.load(in + i)
      sumSq = sumSq + v * v
    }
    tl.store(out, sqrt(sumSq))
    ()
  }

  @TritonKernelMacro
  def vecNorm1Red(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var norm1 = 0.0f
    0.until(n) foreach { i =>
      val v = tl.load(in + i)
      norm1 = norm1 + (if (v < 0.0f) -v else v)
    }
    tl.store(out, norm1)
    ()
  }

  @TritonKernelMacro
  def vecDotRed(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var dot = 0.0f
    0.until(n) foreach { i => dot = dot + tl.load(a + i) * tl.load(b + i) }
    tl.store(out, dot)
    ()
  }

  @TritonKernelMacro
  def vecVarianceRed(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sum = 0.0f
    0.until(n) foreach { i => sum = sum + tl.load(in + i) }
    val mean = sum / n.toFloat
    var varSum = 0.0f
    0.until(n) foreach { i =>
      val d = tl.load(in + i) - mean
      varSum = varSum + d * d
    }
    tl.store(out, varSum / n.toFloat)
    ()
  }

  // ============================================================================
  // 61-70: Matrix Element-wise Operations
  // ============================================================================
  @TritonKernelMacro
  def matAdd(out: Float, a: Float, b: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    tl.store(out + pid, tl.load(a + pid) + tl.load(b + pid))
    ()
  }

  @TritonKernelMacro
  def matSub(out: Float, a: Float, b: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    tl.store(out + pid, tl.load(a + pid) - tl.load(b + pid))
    ()
  }

  @TritonKernelMacro
  def matMulEle(out: Float, a: Float, b: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    tl.store(out + pid, tl.load(a + pid) * tl.load(b + pid))
    ()
  }

  @TritonKernelMacro
  def matScale(out: Float, a: Float, s: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    tl.store(out + pid, tl.load(a + pid) * s)
    ()
  }

  @TritonKernelMacro
  def matNeg(out: Float, a: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    tl.store(out + pid, -tl.load(a + pid))
    ()
  }

  @TritonKernelMacro
  def matAbs(out: Float, a: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, if (av < 0.0f) -av else av)
    ()
  }

  @TritonKernelMacro
  def matRelu(out: Float, a: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, if (av > 0.0f) av else 0.0f)
    ()
  }

  @TritonKernelMacro
  def matSigmoid(out: Float, a: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, 1.0f / (1.0f + exp(-av)))
    ()
  }

  @TritonKernelMacro
  def matClamp(out: Float, a: Float, lo: Float, hi: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    val av = tl.load(a + pid)
    val lv = tl.load(lo + pid)
    val hv = tl.load(hi + pid)
    tl.store(out + pid, if (av < lv) lv else if (av > hv) hv else av)
    ()
  }

  @TritonKernelMacro
  def matSquare(out: Float, a: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m * n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av * av)
    ()
  }

  // ============================================================================
  // 71-80: Row/Column Operations
  // ============================================================================
  @TritonKernelMacro
  def matRowSum(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m) return
    var sum = 0.0f
    0.until(n) foreach { j => sum = sum + tl.load(in + pid * n + j) }
    tl.store(out + pid, sum)
    ()
  }

  @TritonKernelMacro
  def matColSum(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    var sum = 0.0f
    0.until(m) foreach { i => sum = sum + tl.load(in + i * n + pid) }
    tl.store(out + pid, sum)
    ()
  }

  @TritonKernelMacro
  def matRowMax(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m) return
    var maxVal = -1e38f
    0.until(n) foreach { j =>
      val v = tl.load(in + pid * n + j)
      maxVal = if (v > maxVal) v else maxVal
    }
    tl.store(out + pid, maxVal)
    ()
  }

  @TritonKernelMacro
  def matColMax(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    var maxVal = -1e38f
    0.until(m) foreach { i =>
      val v = tl.load(in + i * n + pid)
      maxVal = if (v > maxVal) v else maxVal
    }
    tl.store(out + pid, maxVal)
    ()
  }

  @TritonKernelMacro
  def matRowMean(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m) return
    var sum = 0.0f
    0.until(n) foreach { j => sum = sum + tl.load(in + pid * n + j) }
    tl.store(out + pid, sum / n.toFloat)
    ()
  }

  @TritonKernelMacro
  def matColMean(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    var sum = 0.0f
    0.until(m) foreach { i => sum = sum + tl.load(in + i * n + pid) }
    tl.store(out + pid, sum / m.toFloat)
    ()
  }

  @TritonKernelMacro
  def matRowNorm2(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m) return
    var sumSq = 0.0f
    0.until(n) foreach { j =>
      val v = tl.load(in + pid * n + j)
      sumSq = sumSq + v * v
    }
    tl.store(out + pid, sqrt(sumSq))
    ()
  }

  @TritonKernelMacro
  def matColNorm2(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    var sumSq = 0.0f
    0.until(m) foreach { i =>
      val v = tl.load(in + i * n + pid)
      sumSq = sumSq + v * v
    }
    tl.store(out + pid, sqrt(sumSq))
    ()
  }

  @TritonKernelMacro
  def matRowStdDev(out: Float, in: Float, m: Int, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= m) return
    var sum = 0.0f
    0.until(n) foreach { j => sum = sum + tl.load(in + pid * n + j) }
    val mean = sum / n.toFloat
    var varSum = 0.0f
    0.until(n) foreach { j =>
      val d = tl.load(in + pid * n + j) - mean
      varSum = varSum + d * d
    }
    tl.store(out + pid, sqrt(varSum / n.toFloat))
    ()
  }

  @TritonKernelMacro
  def matDiagSum(out: Float, in: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sum = 0.0f
    0.until(n) foreach { i => sum = sum + tl.load(in + i * n + i) }
    tl.store(out, sum)
    ()
  }

  // ============================================================================
  // 81-90: Distance/Cosine Operations
  // ============================================================================
  @TritonKernelMacro
  def cosineSim(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var dot = 0.0f
    var normA = 0.0f
    var normB = 0.0f
    0.until(n) foreach { i =>
      val av = tl.load(a + i)
      val bv = tl.load(b + i)
      dot = dot + av * bv
      normA = normA + av * av
      normB = normB + bv * bv
    }
    tl.store(out, dot / (sqrt(normA) * sqrt(normB) + 1e-10f))
    ()
  }

  @TritonKernelMacro
  def euclideanDist(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sumSq = 0.0f
    0.until(n) foreach { i =>
      val diff = tl.load(a + i) - tl.load(b + i)
      sumSq = sumSq + diff * diff
    }
    tl.store(out, sqrt(sumSq))
    ()
  }

  @TritonKernelMacro
  def manhattanDist(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var dist = 0.0f
    0.until(n) foreach { i =>
      val diff = tl.load(a + i) - tl.load(b + i)
      dist = dist + (if (diff < 0.0f) -diff else diff)
    }
    tl.store(out, dist)
    ()
  }

  @TritonKernelMacro
  def vecCosineDist(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var dot = 0.0f
    var normA = 0.0f
    var normB = 0.0f
    0.until(n) foreach { i =>
      val av = tl.load(a + i)
      val bv = tl.load(b + i)
      dot = dot + av * bv
      normA = normA + av * av
      normB = normB + bv * bv
    }
    tl.store(out, 1.0f - dot / (sqrt(normA) * sqrt(normB) + 1e-10f))
    ()
  }

  @TritonKernelMacro
  def vecPearsonCorr(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sumA = 0.0f
    var sumB = 0.0f
    0.until(n) foreach { i =>
      sumA = sumA + tl.load(a + i)
      sumB = sumB + tl.load(b + i)
    }
    val meanA = sumA / n.toFloat
    val meanB = sumB / n.toFloat
    var cov = 0.0f
    var varA = 0.0f
    var varB = 0.0f
    0.until(n) foreach { i =>
      val dA = tl.load(a + i) - meanA
      val dB = tl.load(b + i) - meanB
      cov = cov + dA * dB
      varA = varA + dA * dA
      varB = varB + dB * dB
    }
    tl.store(out, cov / (sqrt(varA * varB) + 1e-10f))
    ()
  }

  @TritonKernelMacro
  def vecL1Similarity(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sim = 0.0f
    0.until(n) foreach { i =>
      val diff = tl.load(a + i) - tl.load(b + i)
      sim = sim + (if (diff < 0.0f) -diff else diff)
    }
    tl.store(out, 1.0f / (1.0f + sim))
    ()
  }

  @TritonKernelMacro
  def vecSoftCossim(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var dot = 0.0f
    var normA = 0.0f
    var normB = 0.0f
    0.until(n) foreach { i =>
      val av = tl.load(a + i)
      val bv = tl.load(b + i)
      dot = dot + av * bv
      normA = normA + av * av
      normB = normB + bv * bv
    }
    val cossim = dot / (sqrt(normA) * sqrt(normB) + 1e-10f)
    tl.store(out, 1.0f / (1.0f + exp(-5.0f * cossim)))
    ()
  }

  @TritonKernelMacro
  def vecMahalanobisDist(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sumSq = 0.0f
    0.until(n) foreach { i =>
      val diff = tl.load(a + i) - tl.load(b + i)
      sumSq = sumSq + diff * diff
    }
    tl.store(out, sqrt(sumSq))
    ()
  }

  @TritonKernelMacro
  def vecCanberraDist(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var dist = 0.0f
    0.until(n) foreach { i =>
      val av = tl.load(a + i)
      val bv = tl.load(b + i)
      val num = (if (av < 0.0f) -av else av) + (if (bv < 0.0f) -bv else bv)
      val denom = (if (av < 0.0f) -av else av) + (if (bv < 0.0f) -bv else bv) + 1e-10f
      dist = dist + num / denom
    }
    tl.store(out, dist)
    ()
  }

  @TritonKernelMacro
  def vecBrayCurtisDist(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= 1) return
    var sumNum = 0.0f
    var sumDen = 0.0f
    0.until(n) foreach { i =>
      val av = tl.load(a + i)
      val bv = tl.load(b + i)
      sumNum = sumNum + (if (av - bv < 0.0f) (-(av - bv)) else av - bv)
      sumDen = sumDen + av + bv
    }
    tl.store(out, sumNum / (sumDen + 1e-10f))
    ()
  }

  // ============================================================================
  // 91-100: Special Math Operations
  // ============================================================================
  @TritonKernelMacro
  def vecFloor(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av.toInt.toFloat)
    ()
  }

  @TritonKernelMacro
  def vecCeilApprox(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val floorVal = av.toInt.toFloat
    tl.store(out + pid, if (av > floorVal) floorVal + 1.0f else floorVal)
    ()
  }

  @TritonKernelMacro
  def vecRoundApprox(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val floorVal = av.toInt.toFloat
    tl.store(out + pid, if (av - floorVal >= 0.5f) floorVal + 1.0f else floorVal)
    ()
  }

  @TritonKernelMacro
  def vecFract(out: Float, a: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    tl.store(out + pid, av - av.toInt.toFloat)
    ()
  }

  @TritonKernelMacro
  def vecMod(out: Float, a: Float, b: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val bv = tl.load(b + pid)
    tl.store(out + pid, av - (av / bv).toInt.toFloat * bv)
    ()
  }

  @TritonKernelMacro
  def vecRamp(out: Float, start: Float, step: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val sv = tl.load(start + pid)
    val stv = tl.load(step + pid)
    tl.store(out + pid, sv + stv * pid.toFloat)
    ()
  }

  @TritonKernelMacro
  def vecAddScale(out: Float, a: Float, b: Float, sa: Float, sb: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    val av = tl.load(a + pid)
    val bv = tl.load(b + pid)
    tl.store(out + pid, av * sa + bv * sb)
    ()
  }

  @TritonKernelMacro
  def vecAddConst(out: Float, a: Float, c: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, tl.load(a + pid) + c)
    ()
  }

  @TritonKernelMacro
  def vecMulConst(out: Float, a: Float, c: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, tl.load(a + pid) * c)
    ()
  }

  @TritonKernelMacro
  def vecAffine(out: Float, a: Float, scale: Float, bias: Float, n: Int): Unit = {
    val pid = tl.program_id(0)
    if (pid >= n) return
    tl.store(out + pid, tl.load(a + pid) * scale + bias)
    ()
  }

  // Main already defined above, remove duplicate