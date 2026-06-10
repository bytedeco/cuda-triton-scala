package cuda.dsl.operators.tensor

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool, given_MemoryOps_Byte}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Tensor operations for multi-dimensional arrays.
  * Supports 3D and 4D tensor operations commonly used in deep learning.
  */
object TensorOps {

  /** 3D element-wise addition: C = A + B
    * A, B, C are tensors with shape (D, H, W)
    */
  @cudaKernel
  def tensor3DAdd(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], D: Int, H: Int, W: Int): Unit = {
    val d = blockIdx.z
    val h = blockIdx.y * blockDim.y + threadIdx.y
    val w = blockIdx.x * blockDim.x + threadIdx.x

    if (d < D && h < H && w < W) {
      val idx = d * H * W + h * W + w
      C(idx) = A(idx) + B(idx)
    }
  }

  /** 3D element-wise multiplication: C = A * B */
  @cudaKernel
  def tensor3DMul(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], D: Int, H: Int, W: Int): Unit = {
    val d = blockIdx.z
    val h = blockIdx.y * blockDim.y + threadIdx.y
    val w = blockIdx.x * blockDim.x + threadIdx.x

    if (d < D && h < H && w < W) {
      val idx = d * H * W + h * W + w
      C(idx) = A(idx) * B(idx)
    }
  }

  /** 3D scalar multiplication: B = alpha * A */
  @cudaKernel
  def tensor3DScalarMul(A: Ptr[Float], alpha: Float, B: Ptr[Float], D: Int, H: Int, W: Int): Unit = {
    val d = blockIdx.z
    val h = blockIdx.y * blockDim.y + threadIdx.y
    val w = blockIdx.x * blockDim.x + threadIdx.x

    if (d < D && h < H && w < W) {
      val idx = d * H * W + h * W + w
      B(idx) = alpha * A(idx)
    }
  }

  /** 3D bias addition: B = A + bias */
  @cudaKernel
  def tensor3DBiasAdd(A: Ptr[Float], bias: Ptr[Float], B: Ptr[Float], D: Int, H: Int, W: Int): Unit = {
    val d = blockIdx.z
    val h = blockIdx.y * blockDim.y + threadIdx.y
    val w = blockIdx.x * blockDim.x + threadIdx.x

    if (d < D && h < H && w < W) {
      val idx = d * H * W + h * W + w
      B(idx) = A(idx) + bias(d)
    }
  }

  /** 4D element-wise addition: C = A + B
    * A, B, C are tensors with shape (N, C, H, W)
    */
  @cudaKernel
  def tensor4DAdd(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], N: Int, numChannels: Int, H: Int, W: Int): Unit = {
    val n = blockIdx.z
    val c = blockIdx.y
    val h = blockIdx.x * blockDim.x + threadIdx.x
    val w = threadIdx.y

    if (n < N && c < numChannels && h < H && w < W) {
      val idx = ((n * numChannels + c) * H + h) * W + w
      C(idx) = A(idx) + B(idx)
    }
  }

  /** 4D batch normalization: Y = (X - mean) / sqrt(var + eps) * gamma + beta */
  @cudaKernel
  def tensor4DBatchNorm(X: Ptr[Float], mean: Ptr[Float], variance: Ptr[Float],
                        gamma: Ptr[Float], beta: Ptr[Float], Y: Ptr[Float],
                        N: Int, C: Int, H: Int, W: Int, eps: Float): Unit = {
    val n = blockIdx.z
    val c = blockIdx.y
    val h = blockIdx.x * blockDim.x + threadIdx.x
    val w = threadIdx.y

    if (n < N && c < C && h < H && w < W) {
      val m = mean(c)
      val v = variance(c)
      val invStd = 1.0f / scala.math.sqrt(v + eps).toFloat
      val g = gamma(c)
      val b = beta(c)

      val idx = ((n * C + c) * H + h) * W + w
      Y(idx) = (X(idx) - m) * invStd * g + b
    }
  }

  /** 4D average pooling: Y = avg_pool(X) */
  @cudaKernel
  def tensor4DAvgPool(X: Ptr[Float], Y: Ptr[Float], N: Int, C: Int, H: Int, W: Int,
                       poolH: Int, poolW: Int, strideH: Int, strideW: Int): Unit = {
    val n = blockIdx.z
    val c = blockIdx.y
    val h = blockIdx.x * blockDim.x + threadIdx.x
    val w = threadIdx.y

    if (n < N && c < C && h < H && w < W) {
      val outH = (H - poolH) / strideH + 1
      val outW = (W - poolW) / strideW + 1

      if (h < outH && w < outW) {
        var sum = 0.0f
        var count = 0
        for (ph <- 0 until poolH) {
          for (pw <- 0 until poolW) {
            val inH = h * strideH + ph
            val inW = w * strideW + pw
            if (inH < H && inW < W) {
              val idx = ((n * C + c) * H + inH) * W + inW
              sum += X(idx)
              count += 1
            }
          }
        }
        val outIdx = ((n * C + c) * outH + h) * outW + w
        Y(outIdx) = sum / count.toFloat
      }
    }
  }

  /** 4D max pooling: Y = max_pool(X) */
  @cudaKernel
  def tensor4DMaxPool(X: Ptr[Float], Y: Ptr[Float], N: Int, C: Int, H: Int, W: Int,
                      poolH: Int, poolW: Int, strideH: Int, strideW: Int): Unit = {
    val n = blockIdx.z
    val c = blockIdx.y
    val h = blockIdx.x * blockDim.x + threadIdx.x
    val w = threadIdx.y

    if (n < N && c < C && h < H && w < W) {
      val outH = (H - poolH) / strideH + 1
      val outW = (W - poolW) / strideW + 1

      if (h < outH && w < outW) {
        var maxVal = scala.Float.MinValue
        for (ph <- 0 until poolH) {
          for (pw <- 0 until poolW) {
            val inH = h * strideH + ph
            val inW = w * strideW + pw
            if (inH < H && inW < W) {
              val idx = ((n * C + c) * H + inH) * W + inW
              val v = X(idx)
              if (v > maxVal) maxVal = v
            }
          }
        }
        val outIdx = ((n * C + c) * outH + h) * outW + w
        Y(outIdx) = maxVal
      }
    }
  }

  /** 3D convolution forward: Y = conv(X, W)
    * X: (N, C, H, W), W: (K, C, R, S), Y: (N, K, P, Q)
    */
  @cudaKernel
  def conv2D(X: Ptr[Float], W: Ptr[Float], Y: Ptr[Float],
             N: Int, C: Int, H: Int, Ww: Int,
             K: Int, R: Int, S: Int,
             padH: Int, padW: Int, strideH: Int, strideW: Int): Unit = {
    val n = blockIdx.z
    val k = blockIdx.y
    val p = blockIdx.x * blockDim.x + threadIdx.x
    val q = threadIdx.y

    if (n < N && k < K && p < ((H - R + 2 * padH) / strideH + 1) && q < ((Ww - S + 2 * padW) / strideW + 1)) {
      var sum = 0.0f
      for (c <- 0 until C) {
        for (r <- 0 until R) {
          for (s <- 0 until S) {
            val inH = p * strideH + r - padH
            val inW = q * strideW + s - padW
            if (inH >= 0 && inH < H && inW >= 0 && inW < Ww) {
              val xIdx = ((n * C + c) * H + inH) * Ww + inW
              val wIdx = ((k * C + c) * R + r) * S + s
              sum += X(xIdx) * W(wIdx)
            }
          }
        }
      }
      val outH = (H - R + 2 * padH) / strideH + 1
      val outW = (Ww - S + 2 * padW) / strideW + 1
      val outIdx = ((n * K + k) * outH + p) * outW + q
      Y(outIdx) = sum
    }
  }

  /** Depthwise convolution: Y = depthwise_conv(X, W)
    * X: (N, C, H, W), W: (C, R, S), Y: (N, C, P, Q)
    */
  @cudaKernel
  def depthwiseConv2D(X: Ptr[Float], W: Ptr[Float], Y: Ptr[Float],
                      N: Int, C: Int, H: Int, Ww: Int,
                      R: Int, S: Int,
                      padH: Int, padW: Int, strideH: Int, strideW: Int): Unit = {
    val n = blockIdx.z
    val c = blockIdx.y
    val p = blockIdx.x * blockDim.x + threadIdx.x
    val q = threadIdx.y

    if (n < N && c < C && p < ((H - R + 2 * padH) / strideH + 1) && q < ((Ww - S + 2 * padW) / strideW + 1)) {
      var sum = 0.0f
      for (r <- 0 until R) {
        for (s <- 0 until S) {
          val inH = p * strideH + r - padH
          val inW = q * strideW + s - padW
          if (inH >= 0 && inH < H && inW >= 0 && inW < Ww) {
            val xIdx = ((n * C + c) * H + inH) * Ww + inW
            val wIdx = (c * R + r) * S + s
            sum += X(xIdx) * W(wIdx)
          }
        }
      }
      val outH = (H - R + 2 * padH) / strideH + 1
      val outW = (Ww - S + 2 * padW) / strideW + 1
      val outIdx = ((n * C + c) * outH + p) * outW + q
      Y(outIdx) = sum
    }
  }

  /** 3D transpose convolution (deconvolution): Y = deconv(X, W) */
  @cudaKernel
  def deconv2D(X: Ptr[Float], W: Ptr[Float], Y: Ptr[Float],
               N: Int, C: Int, H: Int, Ww: Int,
               K: Int, R: Int, S: Int,
               padH: Int, padW: Int, strideH: Int, strideW: Int): Unit = {
    val n = blockIdx.z
    val k = blockIdx.y
    val p = blockIdx.x * blockDim.x + threadIdx.x
    val q = threadIdx.y

    if (n < N && k < K && p < H && q < Ww) {
      var sum = 0.0f
      for (c <- 0 until C) {
        for (r <- 0 until R) {
          for (s <- 0 until S) {
            val inH = (p - r + padH) / strideH
            val inW = (q - s + padW) / strideW
            if (inH >= 0 && inH < H && inW >= 0 && inW < Ww && (p - r + padH) % strideH == 0 && (q - s + padW) % strideW == 0) {
              val xIdx = ((n * C + c) * H + inH) * Ww + inW
              val wIdx = ((k * C + c) * R + r) * S + s
              sum += X(xIdx) * W(wIdx)
            }
          }
        }
      }
      val outIdx = ((n * K + k) * H + p) * Ww + q
      Y(outIdx) = sum
    }
  }

  /** Tensor concatenation along axis 0 */
  @cudaKernel
  def concatAxis0(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float],
                  ASize: Int, BSize: Int, Dim: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < ASize + BSize) {
      if (i < ASize) {
        C(i) = A(i)
      } else {
        C(i) = B(i - ASize)
      }
    }
  }

  /** Tensor concatenation along axis 1 */
  @cudaKernel
  def concatAxis1(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float],
                  N: Int, ACols: Int, BCols: Int, Rows: Int): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < Rows) {
      val totalCols = ACols + BCols
      if (col < totalCols) {
        if (col < ACols) {
          C(row * totalCols + col) = A(row * ACols + col)
        } else {
          C(row * totalCols + col) = B(row * BCols + (col - ACols))
        }
      }
    }
  }

  /** Tensor split - extract portion along axis 0 */
  @cudaKernel
  def splitAxis0(A: Ptr[Float], Start: Int, End: Int, B: Ptr[Float], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < (End - Start) * N) {
      val row = i / N
      val col = i % N
      B(i) = A((Start + row) * N + col)
    }
  }

  /** Tensor reshape (flatten) */
  @cudaKernel
  def reshapeFlatten(A: Ptr[Float], B: Ptr[Float], D: Int, H: Int, W: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val total = D * H * W
    if (i < total) {
      val d = i / (H * W)
      val h = (i % (H * W)) / W
      val w = i % W
      B(i) = A((d * H + h) * W + w)
    }
  }

  /** Tensor expand (add leading dimension) */
  @cudaKernel
  def expandDims(A: Ptr[Float], B: Ptr[Float], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      B(i) = A(i)
    }
  }

  /** Tensor squeeze (remove dimension of size 1) */
  @cudaKernel
  def squeezeDims(A: Ptr[Float], B: Ptr[Float], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      B(i) = A(i)
    }
  }
}

/** Tensor index operations */
object TensorIndexOps {

  /** Gather tensor elements along axis 0 */
  @cudaKernel
  def gatherAxis0(A: Ptr[Float], indices: Ptr[Int], B: Ptr[Float], N: Int, D: Int): Unit = {
    val n = blockIdx.x
    val d = threadIdx.x

    if (n < N && d < D) {
      val idx = indices(n)
      B(n * D + d) = A(idx * D + d)
    }
  }

  /** Scatter tensor elements along axis 0 */
  @cudaKernel
  def scatterAxis0(indices: Ptr[Int], A: Ptr[Float], B: Ptr[Float], N: Int, D: Int): Unit = {
    val n = blockIdx.x
    val d = threadIdx.x

    if (n < N && d < D) {
      val idx = indices(n)
      B(idx * D + d) = A(n * D + d)
    }
  }

  /** Advanced indexing with multiple index arrays */
  @cudaKernel
  def advancedIndex(A: Ptr[Float], idx0: Ptr[Int], idx1: Ptr[Int],
                    B: Ptr[Float], N: Int, M: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x

    if (i < N * M) {
      val n = i / M
      val m = i % M
      val rowIdx = idx0(n)
      val colIdx = idx1(m)
      B(i) = A(rowIdx * M + colIdx)
    }
  }

  /** Boolean mask indexing */
  @cudaKernel
  def booleanMask(A: Ptr[Float], mask: Ptr[Bool], B: Ptr[Float], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x

    if (i < N && mask(i)) {
      var count = 0
      for (j <- 0 until i) {
        if (mask(j)) count += 1
      }
      B(count) = A(i)
    }
  }
}

/** Tensor math operations */
object TensorMathOps {

  /** Batch matrix multiplication: C = A * B
    * A: (N, M, K), B: (N, K, P), C: (N, M, P)
    */
  @cudaKernel
  def batchMatmul(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float],
                  N: Int, M: Int, K: Int, P: Int): Unit = {
    val n = blockIdx.z
    val m = blockIdx.y * blockDim.y + threadIdx.y
    val p = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && m < M && p < P) {
      var sum = 0.0f
      for (k <- 0 until K) {
        sum += A((n * M + m) * K + k) * B((n * K + k) * P + p)
      }
      C((n * M + m) * P + p) = sum
    }
  }

  /** Softmax along axis 1: C = softmax(A, axis=1) */
  @cudaKernel
  def softmaxAxis1(A: Ptr[Float], C: Ptr[Float], N: Int, M: Int): Unit = {
    val n = blockIdx.x
    val m = threadIdx.x

    if (n < N && m < M) {
      // Find max for numerical stability
      var maxVal = scala.Float.MinValue
      for (j <- 0 until M) {
        val v = A(n * M + j)
        if (v > maxVal) maxVal = v
      }

      // Compute exp and sum
      var sum = 0.0f
      for (j <- 0 until M) {
        sum += scala.math.exp(A(n * M + j) - maxVal).toFloat
      }
      val logSum = scala.math.log(sum).toFloat + maxVal

      // Compute softmax
      C(n * M + m) = scala.math.exp(A(n * M + m) - logSum).toFloat
    }
  }

  /** Layer norm: Y = (X - mean) / sqrt(var + eps) * gamma + beta
    * Normalizes over last D dimensions
    */
  @cudaKernel
  def layerNorm(X: Ptr[Float], gamma: Ptr[Float], beta: Ptr[Float],
                Y: Ptr[Float], N: Int, D: Int, eps: Float): Unit = {
    val n = blockIdx.x * blockDim.x + threadIdx.x
    val d = threadIdx.y

    if (n < N && d < D) {
      // Compute mean
      var mean = 0.0f
      for (j <- 0 until D) {
        mean += X(n * D + j)
      }
      mean /= D.toFloat

      // Compute variance
      var var_ = 0.0f
      for (j <- 0 until D) {
        val diff = X(n * D + j) - mean
        var_ += diff * diff
      }
      var_ /= D.toFloat

      val invStd = 1.0f / scala.math.sqrt(var_ + eps).toFloat

      // Normalize and scale
      val idx = n * D + d
      Y(idx) = (X(idx) - mean) * invStd * gamma(d) + beta(d)
    }
  }

  /** RMS norm: Y = X / sqrt(mean(X^2) + eps) * gamma + beta */
  @cudaKernel
  def rmsNorm(X: Ptr[Float], gamma: Ptr[Float], beta: Ptr[Float],
              Y: Ptr[Float], N: Int, D: Int, eps: Float): Unit = {
    val n = blockIdx.x * blockDim.x + threadIdx.x
    val d = threadIdx.y

    if (n < N && d < D) {
      // Compute mean of squares
      var ms = 0.0f
      for (j <- 0 until D) {
        val v = X(n * D + j)
        ms += v * v
      }
      ms /= D.toFloat

      val invRms = 1.0f / scala.math.sqrt(ms + eps).toFloat

      // Normalize and scale
      val idx = n * D + d
      Y(idx) = X(idx) * invRms * gamma(d) + beta(d)
    }
  }
}

/** Tensor comparison and selection operations */
object TensorCompareOps {

  /** Element-wise greater than: C = A > B */
  @cudaKernel
  def tensorGT(A: Ptr[Float], B: Ptr[Float], C: Ptr[Bool], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      C(i) = A(i) > B(i)
    }
  }

  /** Element-wise less than: C = A < B */
  @cudaKernel
  def tensorLT(A: Ptr[Float], B: Ptr[Float], C: Ptr[Bool], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      C(i) = A(i) < B(i)
    }
  }

  /** Element-wise equal: C = A == B */
  @cudaKernel
  def tensorEQ(A: Ptr[Float], B: Ptr[Float], C: Ptr[Bool], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      C(i) = A(i) == B(i)
    }
  }

  /** Where operation: C = A if cond else B */
  @cudaKernel
  def tensorWhere(cond: Ptr[Bool], A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      C(i) = if (cond(i)) A(i) else B(i)
    }
  }

  /** Clip operation: B = clamp(A, min, max) */
  @cudaKernel
  def tensorClip(A: Ptr[Float], minVal: Float, maxVal: Float, B: Ptr[Float], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      val v = A(i)
      B(i) = if (v < minVal) minVal else if (v > maxVal) maxVal else v
    }
  }

  /** Is NaN check: B = isnan(A) */
  @cudaKernel
  def tensorIsNaN(A: Ptr[Float], B: Ptr[Bool], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      B(i) = A(i).isNaN
    }
  }

  /** Is Inf check: B = isinf(A) */
  @cudaKernel
  def tensorIsInf(A: Ptr[Float], B: Ptr[Bool], N: Int): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N) {
      B(i) = A(i).isInfinite
    }
  }
}
