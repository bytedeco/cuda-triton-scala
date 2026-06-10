package cuda.dsl.cutlass

import cuda.dsl.core.FloatPtr
import cuda.dsl.dsl._

/** CUTLASS-inspired Convolution kernels using @TritonKernelMacro.
 *
 * Provides 2D convolution operations following CUTLASS patterns:
 * - Direct convolution
 * - Depthwise convolution
 * - Pointwise convolution (1x1)
 *
 * All kernels use @TritonKernelMacro with tl. DSL.
 */
object CutlassConv {

  @TritonKernelMacro(name = "conv2dKernel", gridType = "3D", blockSize = 256)
  def conv2dKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr, bias: FloatPtr,
      N: Int, C: Int, H: Int, W_ch: Int,
      K: Int, R: Int, S: Int,
      P: Int, Q: Int,
      pad_h: Int, pad_w: Int,
      stride_h: Int, stride_w: Int,
      activation: Int): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= P || q >= Q) return

    val nkBlock = tl.program_id(0)
    val nIdx = nkBlock / K
    val kIdx = nkBlock % K

    if (nIdx >= N) return

    var sum: Float = 0.0f
    var ci = 0
    while (ci < C) {
      var r = 0
      while (r < R) {
        var s = 0
        while (s < S) {
          val h_in = p * stride_h + r - pad_h
          val w_in = q * stride_w + s - pad_w

          if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_ch) {
            val aVal = tl.load(A, nIdx * C * H * W_ch + ci * H * W_ch + h_in * W_ch + w_in)
            val wVal = tl.load(W_filter, kIdx * C * R * S + ci * R * S + r * S + s)
            sum = sum + aVal * wVal
          }
          s = s + 1
        }
        r = r + 1
      }
      ci = ci + 1
    }

    sum = sum + tl.load(bias, kIdx)

    if (activation == 1 && sum < 0.0f) sum = 0.0f

    tl.store(D, nIdx * K * P * Q + kIdx * P * Q + p * Q + q, sum)
    ()
  }

  @TritonKernelMacro(name = "conv2dDirectKernel", gridType = "3D", blockSize = 256)
  def conv2dDirectKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr,
      N: Int, C: Int, H: Int, W_ch: Int,
      K: Int, R: Int, S: Int,
      P: Int, Q: Int,
      pad_h: Int, pad_w: Int,
      stride_h: Int, stride_w: Int): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= P || q >= Q) return

    val nkBlock = tl.program_id(0)
    val nIdx = nkBlock / K
    val kIdx = nkBlock % K

    if (nIdx >= N) return

    var sum: Float = 0.0f
    var ci = 0
    while (ci < C) {
      var r = 0
      while (r < R) {
        var s = 0
        while (s < S) {
          val h_in = p * stride_h + r - pad_h
          val w_in = q * stride_w + s - pad_w

          if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_ch) {
            sum = sum + tl.load(A, nIdx * C * H * W_ch + ci * H * W_ch + h_in * W_ch + w_in) *
                       tl.load(W_filter, kIdx * C * R * S + ci * R * S + r * S + s)
          }
          s = s + 1
        }
        r = r + 1
      }
      ci = ci + 1
    }
    tl.store(D, nIdx * K * P * Q + kIdx * P * Q + p * Q + q, sum)
    ()
  }

  @TritonKernelMacro(name = "depthwiseConv2dKernel", gridType = "3D", blockSize = 256)
  def depthwiseConv2dKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr, bias: FloatPtr,
      N: Int, C: Int, H: Int, W_ch: Int,
      R: Int, S: Int,
      P: Int, Q: Int,
      pad_h: Int, pad_w: Int,
      stride_h: Int, stride_w: Int,
      activation: Int): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= P || q >= Q) return

    val ncBlock = tl.program_id(0)
    val nIdx = ncBlock / C
    val cIdx = ncBlock % C

    if (nIdx >= N) return

    var sum: Float = 0.0f
    var r = 0
    while (r < R) {
      var s = 0
      while (s < S) {
        val h_in = p * stride_h + r - pad_h
        val w_in = q * stride_w + s - pad_w

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_ch) {
          sum = sum + tl.load(A, nIdx * C * H * W_ch + cIdx * H * W_ch + h_in * W_ch + w_in) *
                     tl.load(W_filter, cIdx * R * S + r * S + s)
        }
        s = s + 1
      }
      r = r + 1
    }

    sum = sum + tl.load(bias, cIdx)
    if (activation == 1 && sum < 0.0f) sum = 0.0f

    tl.store(D, nIdx * C * P * Q + cIdx * P * Q + p * Q + q, sum)
    ()
  }

  @TritonKernelMacro(name = "pointwiseConv2dKernel", gridType = "3D", blockSize = 256)
  def pointwiseConv2dKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr, bias: FloatPtr,
      N: Int, C: Int, H: Int, W_ch: Int, K: Int,
      activation: Int): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= H || q >= W_ch) return

    val nkBlock = tl.program_id(0)
    val nIdx = nkBlock / K
    val kIdx = nkBlock % K

    if (nIdx >= N) return

    var sum: Float = 0.0f
    var ci = 0
    while (ci < C) {
      sum = sum + tl.load(A, nIdx * C * H * W_ch + ci * H * W_ch + p * W_ch + q) *
                 tl.load(W_filter, kIdx * C + ci)
      ci = ci + 1
    }

    sum = sum + tl.load(bias, kIdx)
    if (activation == 1 && sum < 0.0f) sum = 0.0f

    tl.store(D, nIdx * K * H * W_ch + kIdx * H * W_ch + p * W_ch + q, sum)
    ()
  }

  // ========================================================================
  // Transposed / Deconvolution
  // ========================================================================

  /** Transposed 2D Convolution (Deconvolution).
   *
   * @param D Output [N, K, P, Q]
   * @param A Input [N, C, H, W]
   * @param W_filter Weights [C, K, R, S]
   */
  @TritonKernelMacro(name = "conv2dTransposeKernel", gridType = "3D", blockSize = 256)
  def conv2dTransposeKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr,
      N: Int, C: Int, H: Int, Wc: Int,
      K: Int, R: Int, S: Int,
      P: Int, Q: Int,
      pad_h: Int, pad_w: Int,
      stride_h: Int, stride_w: Int): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= P || q >= Q) return

    val nkcBlock = tl.program_id(0)
    val nIdx = nkcBlock / (K * C)
    val kkc = nkcBlock % (K * C)
    val kIdx = kkc / C
    val cIdx = kkc % C

    if (nIdx >= N) return

    var sum: Float = 0.0f

    var r = 0
    while (r < R) {
      var s = 0
      while (s < S) {
        val h_in = p * stride_h - pad_h + r
        val w_in = q * stride_w - pad_w + s

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < Wc) {
          sum = sum + tl.load(A, nIdx * C * H * Wc + cIdx * H * Wc + h_in * Wc + w_in) *
                     tl.load(W_filter, kIdx * C * R * S + cIdx * R * S + (R - 1 - r) * S + (S - 1 - s))
        }
        s = s + 1
      }
      r = r + 1
    }

    tl.store(D, nIdx * K * P * Q + kIdx * P * Q + p * Q + q, sum)
    ()
  }

  /** 3D Convolution (volumetric).
   *
   * @param D Output [N, K, P, Q, R]
   * @param A Input [N, C, D, H, W]
   * @param W_filter Weights [K, C, T, R, S]
   */
  @TritonKernelMacro(name = "conv3dKernel", gridType = "3D", blockSize = 128)
  def conv3dKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr, bias: FloatPtr,
      N: Int, C: Int, D_in: Int, H: Int, Wc: Int,
      K: Int, T: Int, R: Int, S: Int,
      P: Int, Q: Int, R_out: Int,
      pad_d: Int, pad_h: Int, pad_w: Int,
      stride_d: Int, stride_h: Int, stride_w: Int,
      activation: Int): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= P || q >= Q) return

    val nkBlock = tl.program_id(0)
    val nIdx = nkBlock / K
    val kIdx = nkBlock % K

    if (nIdx >= N) return

    var sum: Float = 0.0f

    var ci = 0
    while (ci < C) {
      var dt = 0
      while (dt < T) {
        var r = 0
        while (r < R) {
          var s = 0
          while (s < S) {
            val d_in = p * stride_d + dt - pad_d
            val h_in = q * stride_h + r - pad_h
            val w_in = 0 * stride_w + s - pad_w

            if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H && w_in >= 0 && w_in < Wc) {
              sum = sum + tl.load(A, nIdx * C * D_in * H * Wc + ci * D_in * H * Wc + d_in * H * Wc + h_in * Wc + w_in) *
                         tl.load(W_filter, kIdx * C * T * R * S + ci * T * R * S + dt * R * S + r * S + s)
            }
            s = s + 1
          }
          r = r + 1
        }
        dt = dt + 1
      }
      ci = ci + 1
    }

    sum = sum + tl.load(bias, kIdx)
    if (activation == 1 && sum < 0.0f) sum = 0.0f

    tl.store(D, nIdx * K * P * Q * R_out + kIdx * P * Q * R_out + p * Q * R_out + q * R_out + 0, sum)
    ()
  }

  /** Grouped Convolution (separate channel groups).
   *
   * @param numGroups Number of channel groups
   */
  @TritonKernelMacro(name = "groupedConv2dKernel", gridType = "3D", blockSize = 256)
  def groupedConv2dKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr, bias: FloatPtr,
      N: Int, C: Int, H: Int, Wc: Int,
      K: Int, R: Int, S: Int,
      P: Int, Q: Int,
      pad_h: Int, pad_w: Int,
      stride_h: Int, stride_w: Int,
      numGroups: Int, activation: Int): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= P || q >= Q) return

    val nkBlock = tl.program_id(0)
    val nIdx = nkBlock / K
    val kIdx = nkBlock % K

    if (nIdx >= N) return

    val cPerGroup = C / numGroups
    val kPerGroup = K / numGroups
    val groupIdx = kIdx / kPerGroup
    val cStart = groupIdx * cPerGroup

    var sum: Float = 0.0f
    var ci = cStart
    while (ci < cStart + cPerGroup) {
      var r = 0
      while (r < R) {
        var s = 0
        while (s < S) {
          val h_in = p * stride_h + r - pad_h
          val w_in = q * stride_w + s - pad_w

          if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < Wc) {
            sum = sum + tl.load(A, nIdx * C * H * Wc + ci * H * Wc + h_in * Wc + w_in) *
                       tl.load(W_filter, groupIdx * kPerGroup * cPerGroup * R * S + (kIdx % kPerGroup) * cPerGroup * R * S + (ci - cStart) * R * S + r * S + s)
          }
          s = s + 1
        }
        r = r + 1
      }
      ci = ci + 1
    }

    sum = sum + tl.load(bias, kIdx)
    if (activation == 1 && sum < 0.0f) sum = 0.0f

    tl.store(D, nIdx * K * P * Q + kIdx * P * Q + p * Q + q, sum)
    ()
  }

  /** Depthwise Convolution with channel multiplier.
   *
   * Each input channel produces M output channels.
   */
  @TritonKernelMacro(name = "depthwiseConv2dMKernel", gridType = "3D", blockSize = 256)
  def depthwiseConv2dMKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr, bias: FloatPtr,
      N: Int, C: Int, H: Int, Wc: Int,
      R: Int, S: Int,
      P: Int, Q: Int,
      pad_h: Int, pad_w: Int,
      stride_h: Int, stride_w: Int,
      channelMultiplier: Int, activation: Int): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= P || q >= Q) return

    val ncBlock = tl.program_id(0)
    val nIdx = ncBlock / C
    val cIdx = ncBlock % C

    if (nIdx >= N) return

    var m = 0
    while (m < channelMultiplier) {
      val outChannel = cIdx * channelMultiplier + m
      var sum: Float = 0.0f

      var r = 0
      while (r < R) {
        var s = 0
        while (s < S) {
          val h_in = p * stride_h + r - pad_h
          val w_in = q * stride_w + s - pad_w

          if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < Wc) {
            sum = sum + tl.load(A, nIdx * C * H * Wc + cIdx * H * Wc + h_in * Wc + w_in) *
                       tl.load(W_filter, cIdx * channelMultiplier * R * S + m * R * S + r * S + s)
          }
          s = s + 1
        }
        r = r + 1
      }

      sum = sum + tl.load(bias, outChannel)
      if (activation == 1 && sum < 0.0f) sum = 0.0f

      tl.store(D, nIdx * (C * channelMultiplier) * P * Q + outChannel * P * Q + p * Q + q, sum)
      m = m + 1
    }
    ()
  }

  /** Fused Conv + BatchNorm.
   *
   * Combines convolution with batch normalization parameters.
   */
  @TritonKernelMacro(name = "fusedConvBatchNormKernel", gridType = "3D", blockSize = 256)
  def fusedConvBatchNormKernel(
      D: FloatPtr, A: FloatPtr, W_filter: FloatPtr,
      N: Int, C: Int, H: Int, Wc: Int,
      K: Int, R: Int, S: Int,
      P: Int, Q: Int,
      pad_h: Int, pad_w: Int,
      stride_h: Int, stride_w: Int,
      // BatchNorm params (fused)
      gamma: FloatPtr, beta: FloatPtr,
      mean: FloatPtr, varPtr: FloatPtr, eps: Float): Unit = {
    val p = tl.program_id(1)
    val q = tl.program_id(2)

    if (p >= P || q >= Q) return

    val nkBlock = tl.program_id(0)
    val nIdx = nkBlock / K
    val kIdx = nkBlock % K

    if (nIdx >= N) return

    var sum: Float = 0.0f
    var ci = 0
    while (ci < C) {
      var r = 0
      while (r < R) {
        var s = 0
        while (s < S) {
          val h_in = p * stride_h + r - pad_h
          val w_in = q * stride_w + s - pad_w

          if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < Wc) {
            sum = sum + tl.load(A, nIdx * C * H * Wc + ci * H * Wc + h_in * Wc + w_in) *
                       tl.load(W_filter, kIdx * C * R * S + ci * R * S + r * S + s)
          }
          s = s + 1
        }
        r = r + 1
      }
      ci = ci + 1
    }

    // BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    val m = tl.load(mean, kIdx)
    val v = tl.load(varPtr, kIdx)
    val g = tl.load(gamma, kIdx)
    val b = tl.load(beta, kIdx)
    val normalized = (sum - m) / scala.math.sqrt((v + eps).toDouble).toFloat * g + b

    tl.store(D, nIdx * K * P * Q + kIdx * P * Q + p * Q + q, normalized)
    ()
  }
}