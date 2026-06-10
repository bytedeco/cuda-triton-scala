package cuda.dsl.operators.linalg

import cuda.dsl.core.*
import cuda.dsl.core.Types.*
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId
import cuda.dsl.core.Types.given

/** Quantized GEMM operations for LLM inference optimization.
 *
 * Supports:
 * - W8A16: INT8 weight with FP16 activation
 * - W8A32: INT8 weight with FP32 activation
 * - W4A16: INT4 weight with FP16 activation (optional)
 * - Dynamic quantization
 */
object QuantizedGemmOps {

  /** W8A16 GEMM: y = (W_int8 * scale_w) @ x + bias
   *
   * Weight is stored as INT8, activation is FP32.
   * Dequantization happens during the matrix multiply.
   */
  @cudaKernel
  def w8a16Gemm(
    wInt8: Ptr[Byte],      // INT8 weight (K x N) row-major
    scalesW: Ptr[Float],    // Scale per column (N)
    x: Ptr[Float],         // Input (M x K)
    bias: Ptr[Float],      // Bias (N)
    y: Ptr[Float],         // Output (M x N)
    M: Int, N: Int, K: Int,
    hasBias: Bool
  ): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < M && col < N) {
      var sum = 0.0f

      // Compute dot product with dequantization
      for (k <- 0 until K) {
        val wVal = wInt8(k * N + col).toFloat  // INT8 -> FP32
        val scale = scalesW(col)                 // Per-column scale
        val xVal = x(row * K + k)              // Input
        sum += wVal * scale * xVal
      }

      // Add bias if present
      if (hasBias) {
        sum += bias(col)
      }

      y(row * N + col) = sum
    }
  }

  /** W8A16 GEMM with row-wise quantization
   *
   * Weight is stored as INT8, scales are per-row for better accuracy.
   */
  @cudaKernel
  def w8a16GemmRowWise(
    wInt8: Ptr[Byte],      // INT8 weight (K x N)
    scalesW: Ptr[Float],    // Scale per row (K)
    x: Ptr[Float],         // Input (M x K)
    y: Ptr[Float],         // Output (M x N)
    M: Int, N: Int, K: Int
  ): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < M && col < N) {
      var sum = 0.0f

      for (k <- 0 until K) {
        val wVal = wInt8(k * N + col).toFloat
        val scale = scalesW(k)  // Per-row scale
        val xVal = x(row * K + k)
        sum += wVal * scale * xVal
      }

      y(row * N + col) = sum
    }
  }

  /** W8A32 GEMM: y = W_int8 @ x (all INT32 accumulation, then dequantize)
   *
   * High precision quantized matmul using INT32 accumulation.
   */
  @cudaKernel
  def w8a32Gemm(
    wInt8: Ptr[Byte],      // INT8 weight (K x N)
    scalesW: Ptr[Float],   // Scale per column (N)
    x: Ptr[Byte],         // INT8 input (M x K)
    scalesX: Ptr[Float],   // Scale per row of input (M)
    y: Ptr[Float],        // Output (M x N)
    M: Int, N: Int, K: Int
  ): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < M && col < N) {
      var sum32 = 0  // INT32 accumulation

      for (k <- 0 until K) {
        val wVal = wInt8(k * N + col).toInt
        val xVal = x(row * K + k).toInt
        sum32 += wVal * xVal
      }

      // Dequantize: y = sum32 * scale_w * scale_x
      val finalVal = sum32.toFloat * scalesW(col) * scalesX(row)
      y(row * N + col) = finalVal
    }
  }

  /** Dynamic quantization kernel
   *
   * Quantizes FP32 weights to INT8 dynamically.
   */
  @cudaOperator
  def dynamicQuantize(
    weight: Ptr[Float],    // FP32 weight (K x N)
    scales: Ptr[Float],    // Output scale per column (N)
    qWeight: Ptr[Byte],    // Output INT8 quantized (K x N)
    K: Int, N: Int,
    qBits: Int = 8         // Quantization bits (8 or 4)
  ): Unit = {
    val col = programId(0)

    if (col < N) {
      // Find max absolute value in column
      var maxAbs = 0.0f
      for (k <- 0 until K) {
        val v = scala.math.abs(weight(k * N + col))
        if (v > maxAbs) maxAbs = v
      }

      // Scale factor: quantize range [-128, 127] or [-8, 7] for INT4
      val qMax = if (qBits == 8) 127.0f else 7.0f
      val scale = if (maxAbs > 1e-6f) qMax / maxAbs else 1.0f
      scales(col) = maxAbs / qMax

      // Quantize column
      for (k <- 0 until K) {
        val fVal = weight(k * N + col)
        val qVal = (fVal * scale).toInt
        // Clamp to [-128, 127] or [-8, 7]
        val clamped = if (qBits == 8) {
          if (qVal > 127) 127 else if (qVal < -128) -128 else qVal
        } else {
          if (qVal > 7) 7 else if (qVal < -8) -8 else qVal
        }
        qWeight(k * N + col) = clamped.toByte
      }
    }
  }

  /** Weight-only quantization with per-channel scales
   *
   * Most common format for LLM inference (GPTQ, AWQ style).
   */
  @cudaOperator
  def weightOnlyQuantizePerChannel(
    weight: Ptr[Float],    // FP32 weight (K x N)
    scales: Ptr[Float],    // Scale per channel (N)
    qWeight: Ptr[Byte],    // INT8 quantized (K x N)
    zeros: Ptr[Byte],      // Zero point per channel (N)
    K: Int, N: Int
  ): Unit = {
    val col = programId(0)

    if (col < N) {
      // Find max in column
      var maxVal = 0.0f
      for (k <- 0 until K) {
        val v = scala.math.abs(weight(k * N + col))
        if (v > maxVal) maxVal = v
      }

      val scale = if (maxVal > 1e-6f) 127.0f / maxVal else 1.0f
      scales(col) = maxVal / 127.0f
      zeros(col) = 0  // symmetric quantization

      for (k <- 0 until K) {
        val fVal = weight(k * N + col)
        val qVal = (fVal * scale).toInt
        val clamped = if (qVal > 127) 127 else if (qVal < -128) -128 else qVal
        qWeight(k * N + col) = clamped.toByte
      }
    }
  }

  /** Dequantize INT8 to FP32
   *
   * Inverse of quantization: y = qx * scale
   */
  @cudaOperator
  def dequantize(
    qWeight: Ptr[Byte],    // INT8 quantized (K x N)
    scales: Ptr[Float],    // Scale per column (N)
    weight: Ptr[Float],    // FP32 output (K x N)
    K: Int, N: Int
  ): Unit = {
    val i = programId(0)

    if (i < K * N) {
      val col = i % N
      val qVal = qWeight(i).toFloat
      weight(i) = qVal * scales(col)
    }
  }

  /** INT4 weight quantization (for extreme compression)
   *
   * Stores 2 INT4 values per byte.
   */
  @cudaKernel
  def w4a16Gemm(
    wInt4: Ptr[Byte],      // INT4 weight packed (K x N / 2)
    scalesW: Ptr[Float],   // Scale per column (N)
    x: Ptr[Float],        // FP32 input (M x K)
    y: Ptr[Float],        // FP32 output (M x N)
    M: Int, N: Int, K: Int
  ): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < M && col < N) {
      var sum = 0.0f

      for (k <- 0 until K) {
        // Unpack INT4 from byte: lower 4 bits for even k, upper 4 bits for odd k
        val byteIdx = (k * N + col) / 2
        val qIdx = (k * N + col) % 2
        val packed = wInt4(byteIdx).toInt
        val wVal = if (qIdx == 0) (packed & 0x0F).toFloat - 8.0f
                   else ((packed >> 4) & 0x0F).toFloat - 8.0f

        val scale = scalesW(col)
        val xVal = x(row * K + k)
        sum += wVal * scale * xVal
      }

      y(row * N + col) = sum
    }
  }
}