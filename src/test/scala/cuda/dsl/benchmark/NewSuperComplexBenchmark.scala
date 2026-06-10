package cuda.dsl.benchmark

import cuda.dsl.core.dim3
import cuda.dsl.runtime.{TritonKernel, TritonKernel as TKTrait}
import cuda.dsl.runtime.runtimeTypes.*
import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR
import TKTrait.{given TritonKernel}

// ============================================================================
// Harness Benchmark: 100 Super Complex New Kernels
// Tests DSL coverage for advanced CUDA kernel patterns
// ============================================================================

@main def NewSuperComplexBenchmark(): Unit =
  val N = 1024
  val BLOCK = 256
  println("=" * 70)
  println("NEW SUPER COMPLEX BENCHMARK — 100 Advanced CUDA Kernels")
  println("Testing DSL coverage beyond the original 238 kernels")
  println("=" * 70)

  // Force initialization (triggers @TritonKernelMacro registration)
  Test100SuperComplexKernels

  val newKernels = List(
    // 1-10: Approximation math functions
    "vecSinhApprox", "vecCoshApprox", "vecAsinApprox", "vecAcosApprox",
    "vecErfApprox", "vecAtanhApprox", "vecErfcApprox", "vecLgammaApprox",
    "vecSinhCosh", "vecBezierEval",
    // 11-20: Advanced activations
    "vecSwish", "vecMish", "vecGELU", "vecSoftplus", "vecLogSumExp",
    "vecLog1p", "vecExpm1", "vecSoftmaxLogStable", "vecLogSoftmax", "vecClipGrad",
    // 21-30: Quantization
    "vecQuantizeInt8", "vecDequantize", "vecFp16ToFp32", "vecFp32ToFp16", "vecDynQuant",
    "matQRIter", "matCholeskyIter", "matSVDIter", "matPowerIter", "matInverseGJ",
    // 31-40: ODE solvers & optimizers
    "odeEuler", "odeRK2", "odeRK4", "optAdamStep", "optSGDMom",
    "sigMovAvg", "sigEMA", "sigIIR", "sigNotch", "sigMedian3",
    // 41-50: Spatial/image
    "imgBilinear", "imgAffine", "imgPerspective", "imgRotate90", "imgBlurHoriz",
    "conv3x3", "convDepthwise", "convGrouped", "convDilated", "convTranspose",
    // 51-60: Pooling & attention
    "poolAdaptiveAvg", "poolGlobalAvg", "poolGlobalMax", "poolStochastic", "poolLP",
    "attnHeadWeight", "attnScaledDot", "attnRelativePos", "attnLocalWindow", "attnCross",
    // 61-70: Embeddings & RNN
    "embTokenLookup", "embPosEncode", "embPosLearned", "embLayerNorm", "embRMSNorm",
    "rnnForward", "gruUpdate", "lstmCell", "rnnBiForward", "rnnPack",
    // 71-80: Distributed & memory
    "distAllReduceSum", "distRingGather", "distBroadcast", "distScatter", "distGradSync",
    "memPrefetch", "memFence", "memDoubleSwap", "memReshape", "memPoolAlloc",
    // 81-90: Loss functions & advanced optimizer
    "lossMSE", "lossMAE", "lossHuber", "lossCrossEnt", "lossHinge",
    "optAdagrad", "optRMSprop", "optAdaDelta", "optNadam", "optLAMB",
    // 91-100: Fusion & misc super complex
    "fuseLayerNormGeluResidual", "attnFlashApprox", "bioSmithWaterman",
    "mlKmeansAssign", "mlPCAProject", "mlDBSCANQuery",
    "mlGMEMStep", "bioHMMViterbi", "decodeBeamStep", "lossDistill"
  )

  println(s"\nExecuting ${newKernels.size} new super complex kernels...")

  var totalPass = 0
  var totalFail = 0
  var totalErr = 0

  for name <- newKernels do
    val r = runSCKernelByName(name, N, BLOCK)
    if r == 1 then totalPass += 1
    else if r == 0 then totalFail += 1
    else totalErr += 1

  println("\n" + "=" * 70)
  println(f"NEW KERNEL RESULTS: $totalPass passed, $totalFail failed, $totalErr errors")
  println("=" * 70)

private def runSCKernelByName(name: String, n: Int, blockSize: Int): Int =
  try
    val r = SCR.executeKernelByName(name, n, blockSize)
    if r.isDefined then
      println(s"  [OK] $name")
      1
    else
      println(s"  [FAIL] $name")
      0
  catch case e: Exception =>
    println(s"  [ERR] $name: ${e.getMessage.take(120)}")
    2