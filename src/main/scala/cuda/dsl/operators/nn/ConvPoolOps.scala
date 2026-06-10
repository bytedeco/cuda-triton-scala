package cuda.dsl.operators.nn

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Convolution operations for neural networks.
 *  Implements 2D and 3D convolutions with various options.
 */
object Conv2DOps {

  /** 2D convolution forward
   *  Input: (N, C, H, W), Kernel: (K, C, R, S), Output: (N, K, P, Q)
   */
  @cudaKernel
  def conv2d(input: Ptr[Float], kernel: Ptr[Float], output: Ptr[Float],
              N: Int, C: Int, H: Int, W: Int, K: Int, R: Int, S: Int,
              stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val k = blockIdx.z / N % K
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && k < K && p < ((H + 2 * padding - R) / stride + 1) && q < ((W + 2 * padding - S) / stride + 1)) {
      var sum = 0.0f
      for (c <- 0 until C) {
        for (r <- 0 until R) {
          for (s <- 0 until S) {
            val hIn = p * stride - padding + r
            val wIn = q * stride - padding + s
            if (hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
              val inputVal = input((n * C + c) * H * W + hIn * W + wIn)
              val kernelVal = kernel((k * C + c) * R * S + r * S + s)
              sum += inputVal * kernelVal
            }
          }
        }
      }
      val outH = (H + 2 * padding - R) / stride + 1
      val outW = (W + 2 * padding - S) / stride + 1
      output((n * K + k) * outH * outW + p * outW + q) = sum
    }
  }

  /** 2D convolution with bias: + bias */
  @cudaKernel
  def conv2dBias(input: Ptr[Float], kernel: Ptr[Float], bias: Ptr[Float], output: Ptr[Float],
                   N: Int, C: Int, H: Int, W: Int, K: Int, R: Int, S: Int,
                   stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val k = blockIdx.z / N % K
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && k < K && p < ((H + 2 * padding - R) / stride + 1) && q < ((W + 2 * padding - S) / stride + 1)) {
      var sum = 0.0f
      for (c <- 0 until C) {
        for (r <- 0 until R) {
          for (s <- 0 until S) {
            val hIn = p * stride - padding + r
            val wIn = q * stride - padding + s
            if (hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
              val inputVal = input((n * C + c) * H * W + hIn * W + wIn)
              val kernelVal = kernel((k * C + c) * R * S + r * S + s)
              sum += inputVal * kernelVal
            }
          }
        }
      }
      sum += bias(k)
      val outH = (H + 2 * padding - R) / stride + 1
      val outW = (W + 2 * padding - S) / stride + 1
      output((n * K + k) * outH * outW + p * outW + q) = sum
    }
  }

  /** 2D convolution gradient w.r.t. input */
  @cudaKernel
  def conv2dInputGrad(grad: Ptr[Float], kernel: Ptr[Float], inputGrad: Ptr[Float],
                        N: Int, C: Int, H: Int, W: Int, K: Int, R: Int, S: Int,
                        stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val c = blockIdx.z / N % C
    val h = blockIdx.y * blockDim.y + threadIdx.y
    val w = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && c < C && h < H && w < W) {
      var sum = 0.0f
      for (k <- 0 until K) {
        for (p <- 0 until ((H + 2 * padding - R) / stride + 1)) {
          for (q <- 0 until ((W + 2 * padding - S) / stride + 1)) {
            val hKer = h + padding - p * stride
            val wKer = w + padding - q * stride
            if (hKer >= 0 && hKer < R && wKer >= 0 && wKer < S) {
              val outH = (H + 2 * padding - R) / stride + 1
              val outW = (W + 2 * padding - S) / stride + 1
              val gradVal = grad((n * K + k) * outH * outW + p * outW + q)
              val kernelVal = kernel((k * C + c) * R * S + hKer * S + wKer)
              sum += gradVal * kernelVal
            }
          }
        }
      }
      inputGrad((n * C + c) * H * W + h * W + w) = sum
    }
  }

  /** Depthwise convolution */
  @cudaKernel
  def depthwiseConv2d(input: Ptr[Float], kernel: Ptr[Float], output: Ptr[Float],
                       N: Int, C: Int, H: Int, W: Int, R: Int, S: Int,
                       stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val c = blockIdx.z / N % C
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && c < C && p < ((H + 2 * padding - R) / stride + 1) && q < ((W + 2 * padding - S) / stride + 1)) {
      var sum = 0.0f
      for (r <- 0 until R) {
        for (s <- 0 until S) {
          val hIn = p * stride - padding + r
          val wIn = q * stride - padding + s
          if (hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
            val inputVal = input((n * C + c) * H * W + hIn * W + wIn)
            val kernelVal = kernel(c * R * S + r * S + s)
            sum += inputVal * kernelVal
          }
        }
      }
      val outH = (H + 2 * padding - R) / stride + 1
      val outW = (W + 2 * padding - S) / stride + 1
      output((n * C + c) * outH * outW + p * outW + q) = sum
    }
  }

  /** Transposed 2D convolution (deconvolution) */
  @cudaKernel
  def transposedConv2d(input: Ptr[Float], kernel: Ptr[Float], output: Ptr[Float],
                        N: Int, C: Int, H: Int, W: Int, K: Int, R: Int, S: Int,
                        stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val k = blockIdx.z / N % K
    val h = blockIdx.y * blockDim.y + threadIdx.y
    val w = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && k < K && h < H && w < W) {
      var sum = 0.0f
      for (c <- 0 until C) {
        for (r <- 0 until R) {
          for (s <- 0 until S) {
            val hIn = (h - r + padding) / stride
            val wIn = (w - s + padding) / stride
            if (hIn >= 0 && hIn < ((H + 2 * padding - R) / stride + 1) &&
                wIn >= 0 && wIn < ((W + 2 * padding - S) / stride + 1) &&
                (h - r + padding) % stride == 0 && (w - s + padding) % stride == 0) {
              val inH = (H + 2 * padding - R) / stride + 1
              val inW = (W + 2 * padding - S) / stride + 1
              val inputVal = input((n * C + c) * inH * inW + hIn * inW + wIn)
              val kernelVal = kernel((k * C + c) * R * S + r * S + s)
              sum += inputVal * kernelVal
            }
          }
        }
      }
      output((n * K + k) * H * W + h * W + w) = sum
    }
  }

  /** Grouped convolution */
  @cudaKernel
  def groupedConv2d(input: Ptr[Float], kernel: Ptr[Float], output: Ptr[Float],
                      N: Int, C: Int, H: Int, W: Int, K: Int, R: Int, S: Int,
                      stride: Int, padding: Int, groups: Int): Unit = {
    val g = blockIdx.z / N % groups
    val n = blockIdx.z % N
    val k = blockIdx.y * blockDim.y + threadIdx.y
    val p = blockIdx.x * blockDim.x + threadIdx.x

    val cPerGroup = C / groups
    val kPerGroup = K / groups

    if (n < N && g < groups && k < kPerGroup && p < ((H + 2 * padding - R) / stride + 1)) {
      val q = threadIdx.y
      if (q < ((W + 2 * padding - S) / stride + 1)) {
        var sum = 0.0f
        for (c <- 0 until cPerGroup) {
          for (r <- 0 until R) {
            for (s <- 0 until S) {
              val hIn = p * stride - padding + r
              val wIn = q * stride - padding + s
              if (hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
                val inputVal = input((n * C + g * cPerGroup + c) * H * W + hIn * W + wIn)
                val kernelVal = kernel((g * kPerGroup + k) * cPerGroup * R * S + c * R * S + r * S + s)
                sum += inputVal * kernelVal
              }
            }
          }
        }
        val outH = (H + 2 * padding - R) / stride + 1
        val outW = (W + 2 * padding - S) / stride + 1
        output((n * K + g * kPerGroup + k) * outH * outW + p * outW + q) = sum
      }
    }
  }
}

/** Pooling operations */
object Pool2DOps {

  /** 2D max pooling */
  @cudaKernel
  def maxPool2d(input: Ptr[Float], output: Ptr[Float], indices: Ptr[Int],
                 N: Int, C: Int, H: Int, W: Int, kH: Int, kW: Int,
                 stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val c = blockIdx.z / N % C
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    val outH = (H + 2 * padding - kH) / stride + 1
    val outW = (W + 2 * padding - kW) / stride + 1

    if (n < N && c < C && p < outH && q < outW) {
      var maxVal = -Float.MaxValue
      var maxIdx = 0
      for (r <- 0 until kH) {
        for (s <- 0 until kW) {
          val hIn = p * stride - padding + r
          val wIn = q * stride - padding + s
          if (hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
            val idx = (n * C + c) * H * W + hIn * W + wIn
            val v = input(idx)
            if (v > maxVal) {
              maxVal = v
              maxIdx = idx
            }
          }
        }
      }
      output((n * C + c) * outH * outW + p * outW + q) = maxVal
      indices((n * C + c) * outH * outW + p * outW + q) = maxIdx
    }
  }

  /** 2D average pooling */
  @cudaKernel
  def avgPool2d(input: Ptr[Float], output: Ptr[Float],
                 N: Int, C: Int, H: Int, W: Int, kH: Int, kW: Int,
                 stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val c = blockIdx.z / N % C
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    val outH = (H + 2 * padding - kH) / stride + 1
    val outW = (W + 2 * padding - kW) / stride + 1

    if (n < N && c < C && p < outH && q < outW) {
      var sum = 0.0f
      var count = 0
      for (r <- 0 until kH) {
        for (s <- 0 until kW) {
          val hIn = p * stride - padding + r
          val wIn = q * stride - padding + s
          if (hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
            sum += input((n * C + c) * H * W + hIn * W + wIn)
            count += 1
          }
        }
      }
      output((n * C + c) * outH * outW + p * outW + q) = sum / count.toFloat
    }
  }

  /** 2D max pooling gradient */
  @cudaKernel
  def maxPool2dGrad(grad: Ptr[Float], indices: Ptr[Int], inputGrad: Ptr[Float],
                     N: Int, C: Int, H: Int, W: Int, outH: Int, outW: Int): Unit = {
    val n = blockIdx.z % N
    val c = blockIdx.z / N % C
    val h = blockIdx.y * blockDim.y + threadIdx.y
    val w = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && c < C && h < H && w < W) {
      var gradVal = 0.0f
      for (p <- 0 until outH) {
        for (q <- 0 until outW) {
          val idx = (n * C + c) * outH * outW + p * outW + q
          if (indices(idx) == (n * C + c) * H * W + h * W + w) {
            gradVal += grad(idx)
          }
        }
      }
      inputGrad((n * C + c) * H * W + h * W + w) = gradVal
    }
  }

  /** Global max pooling (over entire spatial dimensions) */
  @cudaKernel
  def globalMaxPool(input: Ptr[Float], output: Ptr[Float], N: Int, C: Int, H: Int, W: Int): Unit = {
    val n = blockIdx.x % N
    val c = blockIdx.x / N % C

    if (n < N && c < C) {
      var maxVal = -Float.MaxValue
      for (h <- 0 until H) {
        for (w <- 0 until W) {
          val v = input((n * C + c) * H * W + h * W + w)
          if (v > maxVal) maxVal = v
        }
      }
      output(n * C + c) = maxVal
    }
  }

  /** Global average pooling */
  @cudaKernel
  def globalAvgPool(input: Ptr[Float], output: Ptr[Float], N: Int, C: Int, H: Int, W: Int): Unit = {
    val n = blockIdx.x % N
    val c = blockIdx.x / N % C

    if (n < N && c < C) {
      var sum = 0.0f
      for (h <- 0 until H) {
        for (w <- 0 until W) {
          sum += input((n * C + c) * H * W + h * W + w)
        }
      }
      output(n * C + c) = sum / (H * W).toFloat
    }
  }

  /** Adaptive max pooling to target size */
  @cudaKernel
  def adaptiveMaxPool2d(input: Ptr[Float], output: Ptr[Float], N: Int, C: Int,
                         H: Int, W: Int, outH: Int, outW: Int): Unit = {
    val n = blockIdx.z % N
    val c = blockIdx.z / N % C
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && c < C && p < outH && q < outW) {
      val hStart = (p * H) / outH
      val hEnd = ((p + 1) * H + outH - 1) / outH
      val wStart = (q * W) / outW
      val wEnd = ((q + 1) * W + outW - 1) / outW

      var maxVal = -Float.MaxValue
      for (h <- hStart until hEnd) {
        for (w <- wStart until wEnd) {
          val v = input((n * C + c) * H * W + h * W + w)
          if (v > maxVal) maxVal = v
        }
      }
      output((n * C + c) * outH * outW + p * outW + q) = maxVal
    }
  }

  /** Adaptive average pooling to target size */
  @cudaKernel
  def adaptiveAvgPool2d(input: Ptr[Float], output: Ptr[Float], N: Int, C: Int,
                         H: Int, W: Int, outH: Int, outW: Int): Unit = {
    val n = blockIdx.z % N
    val c = blockIdx.z / N % C
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && c < C && p < outH && q < outW) {
      val hStart = (p * H) / outH
      val hEnd = ((p + 1) * H + outH - 1) / outH
      val wStart = (q * W) / outW
      val wEnd = ((q + 1) * W + outW - 1) / outW

      var sum = 0.0f
      var count = 0
      for (h <- hStart until hEnd) {
        for (w <- wStart until wEnd) {
          sum += input((n * C + c) * H * W + h * W + w)
          count += 1
        }
      }
      output((n * C + c) * outH * outW + p * outW + q) = sum / count.toFloat
    }
  }
}

/** 3D convolution and pooling */
object Conv3DOps {

  /** 3D convolution */
  @cudaKernel
  def conv3d(input: Ptr[Float], kernel: Ptr[Float], output: Ptr[Float],
              N: Int, C: Int, D: Int, H: Int, W: Int, K: Int,
              T: Int, R: Int, S: Int, stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val k = blockIdx.z / N % K
    val d = blockIdx.z * blockDim.z + threadIdx.z
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    val outD = (D + 2 * padding - T) / stride + 1
    val outH = (H + 2 * padding - R) / stride + 1
    val outW = (W + 2 * padding - S) / stride + 1

    if (n < N && k < K && d < outD && p < outH && q < outW) {
      var sum = 0.0f
      for (c <- 0 until C) {
        for (t <- 0 until T) {
          for (r <- 0 until R) {
            for (s <- 0 until S) {
              val dIn = d * stride - padding + t
              val hIn = p * stride - padding + r
              val wIn = q * stride - padding + s
              if (dIn >= 0 && dIn < D && hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
                val inputVal = input((n * C + c) * D * H * W + dIn * H * W + hIn * W + wIn)
                val kernelVal = kernel((k * C + c) * T * R * S + t * R * S + r * S + s)
                sum += inputVal * kernelVal
              }
            }
          }
        }
      }
      output((n * K + k) * outD * outH * outW + d * outH * outW + p * outW + q) = sum
    }
  }

  /** 3D pooling */
  @cudaKernel
  def maxPool3d(input: Ptr[Float], output: Ptr[Float], N: Int, C: Int, D: Int, H: Int, W: Int,
                 kD: Int, kH: Int, kW: Int, stride: Int, padding: Int): Unit = {
    val n = blockIdx.z % N
    val c = blockIdx.z / N % C
    val d = blockIdx.z * blockDim.z + threadIdx.z
    val p = blockIdx.y * blockDim.y + threadIdx.y
    val q = blockIdx.x * blockDim.x + threadIdx.x

    val outD = (D + 2 * padding - kD) / stride + 1
    val outH = (H + 2 * padding - kH) / stride + 1
    val outW = (W + 2 * padding - kW) / stride + 1

    if (n < N && c < C && d < outD && p < outH && q < outW) {
      var maxVal = -Float.MaxValue
      for (t <- 0 until kD) {
        for (r <- 0 until kH) {
          for (s <- 0 until kW) {
            val dIn = d * stride - padding + t
            val hIn = p * stride - padding + r
            val wIn = q * stride - padding + s
            if (dIn >= 0 && dIn < D && hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
              val v = input((n * C + c) * D * H * W + dIn * H * W + hIn * W + wIn)
              if (v > maxVal) maxVal = v
            }
          }
        }
      }
      output((n * C + c) * outD * outH * outW + d * outH * outW + p * outW + q) = maxVal
    }
  }
}

/** Attention mechanisms */
object AttentionOps {

  /** Scaled dot-product attention: attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V */
  @cudaKernel
  def scaledDotProductAttention(q: Ptr[Float], k: Ptr[Float], v: Ptr[Float], output: Ptr[Float],
                                 scale: Float, N: Int, L: Int, M: Int): Unit = {
    // Simplified: N=batch, L=seq_len, M=head_dim
    // This is a basic implementation
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < N * L) {
      val n = i / L
      val l = i % L

      var sum = 0.0f
      for (j <- 0 until L) {
        var dot = 0.0f
        for (m <- 0 until M) {
          dot += q(n * L * M + l * M + m) * k(n * L * M + j * M + m)
        }
        dot *= scale
        // softmax would be applied here
        for (m <- 0 until M) {
          sum += scala.math.exp(dot).toFloat * v(n * L * M + j * M + m)
        }
      }
      output(i * M) = sum
    }
  }

  /** Multi-head attention output projection */
  @cudaKernel
  def multiHeadOutputProjection(input: Ptr[Float], weight: Ptr[Float], bias: Ptr[Float],
                                 output: Ptr[Float], N: Int, L: Int, H: Int, D: Int, O: Int): Unit = {
    val n = blockIdx.z % N
    val l = blockIdx.y * blockDim.y + threadIdx.y
    val o = blockIdx.x * blockDim.x + threadIdx.x

    if (n < N && l < L && o < O) {
      var sum = bias(o)
      for (h <- 0 until H) {
        for (d <- 0 until D) {
          sum += input((n * H + h) * L * D + l * D + d) * weight((h * D + d) * O + o)
        }
      }
      output((n * L + l) * O + o) = sum
    }
  }
}
