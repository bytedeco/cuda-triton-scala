package cuda.dsl.operators.nn

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Neural network activation functions.
 *  These operators implement common activation functions used in neural networks.
 */
object ActivationOps {

  /** ReLU: c = max(0, a) */
  @cudaOperator
  def relu(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = if (v > 0) v else 0
    }
  }

  /** ReLU gradient: grad = grad * (a > 0) */
  @cudaOperator
  def reluGrad(a: Ptr[Float], grad: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n && a(i) <= 0) {
      grad(i) = 0
    }
  }

  /** Leaky ReLU: c = a > 0 ? a : alpha * a */
  @cudaOperator
  def leakyRelu(a: Ptr[Float], alpha: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = if (v > 0) v else alpha * v
    }
  }

  /** Leaky ReLU gradient */
  @cudaOperator
  def leakyReluGrad(a: Ptr[Float], alpha: Float, grad: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      grad(i) = if (a(i) > 0) grad(i) else alpha * grad(i)
    }
  }

  /** PReLU (Parametric ReLU): c = a > 0 ? a : alpha(i) * a */
  @cudaKernel
  def prelu(a: Ptr[Float], alpha: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = if (v > 0) v else alpha(i) * v
    }
  }

  /** Sigmoid: c = 1 / (1 + exp(-a)) */
  @cudaOperator
  def sigmoid(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = 1.0f / (1.0f + scala.math.exp(-v).toFloat)
    }
  }

  /** Sigmoid gradient: grad = c * (1 - c) * grad */
  @cudaOperator
  def sigmoidGrad(a: Ptr[Float], grad: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      val s = 1.0f / (1.0f + scala.math.exp(-v).toFloat)
      grad(i) = s * (1.0f - s) * grad(i)
    }
  }

  /** Tanh: c = tanh(a) */
  @cudaOperator
  def tanhActivation(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      c(i) = scala.math.tanh(a(i)).toFloat
    }
  }

  /** Tanh gradient: grad = (1 - c^2) * grad */
  @cudaOperator
  def tanhGrad(a: Ptr[Float], grad: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      grad(i) = (1.0f - v * v) * grad(i)
    }
  }

  /** ELU: c = a > 0 ? a : alpha * (exp(a) - 1) */
  @cudaOperator
  def elu(a: Ptr[Float], alpha: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = if (v > 0) v else alpha * (scala.math.exp(v).toFloat - 1.0f)
    }
  }

  /** SELU: scaled exponential linear unit */
  @cudaOperator
  def selu(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val alpha = 1.67326324235f
    val scale = 1.05070098735f
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = scale * (if (v > 0) v else alpha * (scala.math.exp(v).toFloat - 1.0f))
    }
  }

  /** GELU: Gaussian Error Linear Unit approximation */
  @cudaOperator
  def gelu(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      val cdf = 0.5f * (1.0f + scala.math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)).toFloat)
      c(i) = v * cdf
    }
  }

  /** GELU gradient */
  @cudaOperator
  def geluGrad(a: Ptr[Float], grad: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      val tanhArg = 0.7978845608028654 * (v + 0.044715 * v * v * v)
      val tanhVal = scala.math.tanh(tanhArg).toFloat
      val cdf = 0.5f * (1.0f + tanhVal)
      val pdf = 0.5f * 0.7978845608028654 * (1.0f - tanhVal * tanhVal) * (1.0f + 0.134145 * v * v + tanhVal)
      grad(i) = (cdf + v * pdf).toFloat
    }
  }

  /** Swish: c = a * sigmoid(a) */
  @cudaOperator
  def swish(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      val s = 1.0f / (1.0f + scala.math.exp(-v).toFloat)
      c(i) = v * s
    }
  }

  /** Mish: c = a * tanh(softplus(a)) */
  @cudaOperator
  def mish(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      val sp = scala.math.log(1.0f + scala.math.exp(v)).toFloat
      c(i) = v * scala.math.tanh(sp).toFloat
    }
  }

  /** SiLU (Sigmoid Linear Unit, also Swish): c = a / (1 + exp(-a)) */
  @cudaOperator
  def silu(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = v / (1.0f + scala.math.exp(-v).toFloat)
    }
  }

  /** Hard Swish: c = a * clamp(a + 3, 0, 6) / 6 */
  @cudaOperator
  def hardSwish(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      val shifted = v + 3.0f
      val clamped = if (shifted < 0) 0.0f else if (shifted > 6) 6.0f else shifted
      c(i) = v * clamped / 6.0f
    }
  }

  /** Hard Sigmoid: c = clamp(a + 3, 0, 6) / 6 */
  @cudaOperator
  def hardSigmoid(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i) + 3.0f
      c(i) = if (v < 0) 0.0f else if (v > 6) 1.0f else v / 6.0f
    }
  }

  /** Softplus: c = log(1 + exp(a)) */
  @cudaOperator
  def softplus(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      c(i) = scala.math.log(1.0f + scala.math.exp(a(i))).toFloat
    }
  }

  /** Softsign: c = a / (1 + abs(a)) */
  @cudaOperator
  def softsign(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      val absV = if (v < 0) -v else v
      c(i) = v / (1.0f + absV)
    }
  }

  /** CELU (Continuously differentiable ELU) */
  @cudaOperator
  def celu(a: Ptr[Float], alpha: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = if (v > 0) v else alpha * (scala.math.exp(v / alpha).toFloat - 1.0f)
    }
  }

  /** Threshold ReLU: c = a > threshold ? a : 0 */
  @cudaOperator
  def thresholdRelu(a: Ptr[Float], threshold: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = if (v > threshold) v else 0.0f
    }
  }

  /** LogSigmoid: c = log(1 / (1 + exp(-a))) */
  @cudaOperator
  def logSigmoid(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = -scala.math.log(1.0f + scala.math.exp(-v)).toFloat
    }
  }
}

/** Softmax and normalization operations */
object SoftmaxOps {

  /** Softmax: c(i) = exp(a(i)) / sum(exp(a(j))) */
  @cudaKernel
  def softmax(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    // Find max for numerical stability
    var maxVal = a(0)
    for (i <- 1 until n) {
      if (a(i) > maxVal) maxVal = a(i)
    }

    // Compute exp and sum - direct computation without shared memory
    var sum = 0.0f
    for (i <- 0 until n) {
      sum += scala.math.exp(a(i) - maxVal).toFloat
    }

    // Normalize
    for (i <- 0 until n) {
      c(i) = scala.math.exp(a(i) - maxVal).toFloat / sum
    }
  }

  /** Log softmax: c(i) = a(i) - log(sum(exp(a(j)))) */
  @cudaOperator
  def logSoftmax(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    // Find max for numerical stability
    var maxVal = a(0)
    for (i <- 1 until n) {
      if (a(i) > maxVal) maxVal = a(i)
    }

    // Compute sum of exp
    var sum = 0.0f
    for (i <- 0 until n) {
      sum += scala.math.exp(a(i) - maxVal).toFloat
    }
    val logSum = scala.math.log(sum).toFloat + maxVal

    // Compute log softmax
    for (i <- 0 until n) {
      c(i) = a(i) - logSum
    }
  }

  /** Hardmax: c(i) = 1 if i == argmax(a) else 0 */
  @cudaOperator
  def hardmax(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    // Find argmax
    var maxIdx = 0
    var maxVal = a(0)
    for (i <- 1 until n) {
      if (a(i) > maxVal) {
        maxVal = a(i)
        maxIdx = i
      }
    }

    // Set hardmax
    for (i <- 0 until n) {
      c(i) = if (i == maxIdx) 1.0f else 0.0f
    }
  }

  /** Softmax gradient */
  @cudaOperator
  def softmaxGrad(a: Ptr[Float], grad: Ptr[Float], n: Int): Unit = {
    // Simplified softmax gradient
    for (i <- 0 until n) {
      // This is a simplified version; full implementation requires Jacobian
      grad(i) = grad(i) * a(i) * (1.0f - a(i))
    }
  }
}

/** Batch normalization operations */
object BatchNormOps {

  /** Batch normalization forward
   *  y = (x - mean) / sqrt(var + eps) * gamma + beta
   */
  @cudaKernel
  def batchNorm(x: Ptr[Float], mean: Ptr[Float], variance: Ptr[Float], gamma: Ptr[Float], beta: Ptr[Float], y: Ptr[Float], M: Int, N: Int, eps: Float): Unit = {
    val j = blockIdx.x * blockDim.x + threadIdx.x
    if (j < N) {
      val m = mean(j)
      val v = variance(j)
      val invStd = 1.0f / scala.math.sqrt(v + eps).toFloat
      val g = gamma(j)
      val b = beta(j)

      for (i <- 0 until M) {
        val idx = i * N + j
        y(idx) = (x(idx) - m) * invStd * g + b
      }
    }
  }

  /** Batch normalization backward */
  @cudaKernel
  def batchNormGrad(x: Ptr[Float], grad: Ptr[Float], mean: Ptr[Float], variance: Ptr[Float], gamma: Ptr[Float], gammaGrad: Ptr[Float], betaGrad: Ptr[Float], M: Int, N: Int, eps: Float): Unit = {
    val j = blockIdx.x * blockDim.x + threadIdx.x
    if (j < N) {
      var xGradSum = 0.0f
      var gammaGradSum = 0.0f
      val m = mean(j)
      val v = variance(j)
      val invStd = 1.0f / scala.math.sqrt(v + eps).toFloat

      // Compute gamma gradient and sum of gradients
      for (i <- 0 until M) {
        val idx = i * N + j
        val normalized = (x(idx) - m) * invStd
        gammaGradSum += grad(idx) * normalized
        xGradSum += grad(idx)
      }

      gammaGrad(j) = gammaGradSum
      betaGrad(j) = xGradSum
    }
  }

  /** Layer normalization
   *  y = (x - mean) / sqrt(var + eps) * gamma + beta
   */
  @cudaKernel
  def layerNorm(x: Ptr[Float], gamma: Ptr[Float], beta: Ptr[Float], y: Ptr[Float], n: Int, eps: Float): Unit = {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) {
      // Compute mean
      var mean = 0.0f
      for (j <- 0 until n) {
        mean += x(i * n + j)
      }
      mean /= n.toFloat

      // Compute variance
      var var_ = 0.0f
      for (j <- 0 until n) {
        val diff = x(i * n + j) - mean
        var_ += diff * diff
      }
      var_ /= n.toFloat

      val invStd = 1.0f / scala.math.sqrt(var_ + eps).toFloat

      // Normalize and scale
      for (j <- 0 until n) {
        val idx = i * n + j
        y(idx) = (x(idx) - mean) * invStd * gamma(j) + beta(j)
      }
    }
  }

  /** Instance normalization */
  @cudaKernel
  def instanceNorm(x: Ptr[Float], gamma: Ptr[Float], beta: Ptr[Float], y: Ptr[Float], C: Int, H: Int, W: Int, eps: Float): Unit = {
    val c = blockIdx.x
    val h = blockIdx.y * blockDim.y + threadIdx.y
    val w = threadIdx.x

    if (c < C && h < H && w < W) {
      // Compute mean over H,W
      var mean = 0.0f
      for (i <- 0 until H) {
        for (j <- 0 until W) {
          mean += x(c * H * W + i * W + j)
        }
      }
      mean /= (H * W).toFloat

      // Compute variance
      var var_ = 0.0f
      for (i <- 0 until H) {
        for (j <- 0 until W) {
          val diff = x(c * H * W + i * W + j) - mean
          var_ += diff * diff
        }
      }
      var_ /= (H * W).toFloat

      val invStd = 1.0f / scala.math.sqrt(var_ + eps).toFloat

      // Normalize and scale
      val idx = c * H * W + h * W + w
      y(idx) = (x(idx) - mean) * invStd * gamma(c) + beta(c)
    }
  }
}

/** Dropout operations */
object DropoutOps {

  /** Dropout forward (apply mask) */
  @cudaOperator
  def dropout(x: Ptr[Float], mask: Ptr[Float], p: Float, scale: Float, n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      if (mask(i) > p) {
        mask(i) = 1.0f / scale  // Store scaled value for backward
      } else {
        mask(i) = 0.0f
      }
    }
  }

  /** Dropout backward */
  @cudaOperator
  def dropoutGrad(grad: Ptr[Float], mask: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      grad(i) = grad(i) * mask(i)
    }
  }
}
