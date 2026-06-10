package cuda.dsl.benchmark

import cuda.dsl.dsl._

/** Test @TritonKernelMacro for Attention and Transformer kernels */
object TestTransformerKernels {

  // Float-returning math functions (scala.math functions return Double)
  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  def log(x: Float): Float = scala.math.log(x.toDouble).toFloat
  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat
  def sin(x: Float): Float = scala.math.sin(x.toDouble).toFloat
  def cos(x: Float): Float = scala.math.cos(x.toDouble).toFloat
  def tanh(x: Float): Float = scala.math.tanh(x.toDouble).toFloat
  def sigmoid(x: Float): Float = 1.0f / (1.0f + exp(-x))
  def pow(x: Float, y: Float): Float = scala.math.pow(x.toDouble, y.toDouble).toFloat

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("@TritonKernelMacro: Attention & Transformer Kernels")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels.txt for generated CUDA code.")
    println("=" * 80)

    // ========================================================================
    // 1. Softmax (simplified)
    // ========================================================================
    println("\n[1] Softmax Kernel")

    @TritonKernelMacro
    def softmax(x: Float, maxVal: Float, expSum: Float): Float = {
      val exp_x = exp(x - maxVal)
      exp_x / (expSum + 0.001f)
    }

    // ========================================================================
    // 2. LayerNorm (simplified)
    // y = (x - mean) / sqrt(variance + eps) * gamma + beta
    // ========================================================================
    println("\n[2] LayerNorm Kernel")

    @TritonKernelMacro
    def layerNorm(x: Float, mean: Float, variance: Float, gamma: Float, beta: Float): Float = {
      val normalized = (x - mean) / sqrt(variance + 0.00001f)
      normalized * gamma + beta
    }

    // ========================================================================
    // 3. RMSNorm (simplified)
    // y = x / rms * gamma, where rms = sqrt(sum(x^2) / n + eps)
    // ========================================================================
    println("\n[3] RMSNorm Kernel")

    @TritonKernelMacro
    def rmsNorm(x: Float, rms: Float, gamma: Float): Float = {
      (x / (rms + 0.00001f)) * gamma
    }

    // ========================================================================
    // 4. GELU activation
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // ========================================================================
    println("\n[4] GELU Activation")

    @TritonKernelMacro
    def gelu(x: Float): Float = {
      val c = sqrt(2.0f / 3.14159265f)
      val t = tanh(c * (x + 0.044715f * x * x * x))
      0.5f * x * (1.0f + t)
    }

    // ========================================================================
    // 5. SwiGLU (Swish + Gated Linear Unit)
    // swiglu(x) = x * sigmoid(beta * x) * gate
    // ========================================================================
    println("\n[5] SwiGLU Activation")

    @TritonKernelMacro
    def swiglu(x: Float, gate: Float, beta: Float): Float = {
      val swish = x * sigmoid(beta * x)
      swish * gate
    }

    // ========================================================================
    // 6. Scaled Dot-Product Attention (simplified)
    // score = (q * k^T) / sqrt(d) + mask
    // out = softmax(scores) * v
    // ========================================================================
    println("\n[6] Scaled Dot-Product Attention (simplified)")

    @TritonKernelMacro
    def attentionScore(q: Float, k: Float, scale: Float): Float = {
      q * k * scale
    }

    // ========================================================================
    // 7. Feed-Forward Network (MLP)
    // out = gelu(x * w1 + b1) * w2 + b2
    // ========================================================================
    println("\n[7] Feed-Forward Network (MLP)")

    @TritonKernelMacro
    def mlp(x: Float, w1: Float, b1: Float, w2: Float, b2: Float): Float = {
      val h = gelu(x * w1 + b1)
      h * w2 + b2
    }

    // ========================================================================
    // 8. Residual Connection + LayerNorm
    // out = layerNorm(x + residual)
    // ========================================================================
    println("\n[8] Residual + LayerNorm")

    @TritonKernelMacro
    def residualLayerNorm(x: Float, residual: Float, mean: Float, variance: Float, gamma: Float, beta: Float): Float = {
      val combined = x + residual
      val normalized = (combined - mean) / sqrt(variance + 0.00001f)
      normalized * gamma + beta
    }

    // ========================================================================
    // 9. Multi-Head Attention (simplified per-head computation)
    // out = softmax(q * k^T / sqrt(d)) * v
    // ========================================================================
    println("\n[9] Multi-Head Attention (per-head)")

    @TritonKernelMacro
    def multiHeadAttn(q: Float, k: Float, v: Float, scale: Float): Float = {
      val score = q * k * scale
      val expScore = exp(score)
      expScore * v
    }

    // ========================================================================
    // 10. Cross Entropy Loss (simplified)
    // loss = -sum(log(softmax(logits) * targets))
    // ========================================================================
    println("\n[10] Cross Entropy Loss")

    @TritonKernelMacro
    def crossEntropy(logits: Float, target: Float, maxLogit: Float): Float = {
      val expLogit = exp(logits - maxLogit)
      val softmaxProb = expLogit
      -target * log(softmaxProb + 0.000001f)
    }

    // ========================================================================
    // 11. Label Smoothing Cross Entropy
    // loss = -((1 - alpha) * target + alpha / num_classes) * log(pred)
    // ========================================================================
    println("\n[11] Label Smoothing Loss")

    @TritonKernelMacro
    def labelSmoothingLoss(pred: Float, target: Float, smooth: Float, numClasses: Float): Float = {
      val smoothTarget = (1.0f - smooth) * target + smooth / numClasses
      -smoothTarget * log(pred + 0.000001f)
    }

    // ========================================================================
    // 12. AdamW Update (simplified)
    // m = beta1 * m + (1 - beta1) * grad
    // v = beta2 * v + (1 - beta2) * grad^2
    // w = w - lr * m / (sqrt(v) + eps)
    // ========================================================================
    println("\n[12] AdamW Update")

    @TritonKernelMacro
    def adamUpdate(grad: Float, m: Float, v: Float, lr: Float, beta1: Float, beta2: Float, eps: Float): Float = {
      val mNew = beta1 * m + (1.0f - beta1) * grad
      val vNew = beta2 * v + (1.0f - beta2) * grad * grad
      val update = lr * mNew / (sqrt(vNew) + eps)
      -update
    }

    // ========================================================================
    // 13. RoPE (Rotary Position Embedding) - simplified
    // Rot(X, pos) = X * cos(theta) + rotate_90(X) * sin(theta)
    // ========================================================================
    println("\n[13] RoPE (Rotary Position Embedding)")

    @TritonKernelMacro
    def rope(x: Float, xRot: Float, theta: Float): Float = {
      x * cos(theta) + xRot * sin(theta)
    }

    // ========================================================================
    // 14. Attention with RoPE (combined)
    // ========================================================================
    println("\n[14] Attention with RoPE")

    @TritonKernelMacro
    def attentionRoPE(q: Float, k: Float, qRot: Float, kRot: Float, theta: Float, scale: Float): Float = {
      val qEmb = q * cos(theta) + qRot * sin(theta)
      val kEmb = k * cos(theta) + kRot * sin(theta)
      qEmb * kEmb * scale
    }

    // ========================================================================
    // 15. Fused Attention (Q, K, V, mask -> output)
    // Simplified: out = softmax(Q * K^T / sqrt(d)) * V + mask
    // ========================================================================
    println("\n[15] Fused Attention Kernel")

    @TritonKernelMacro
    def fusedAttention(q: Float, k: Float, v: Float, mask: Float, scale: Float): Float = {
      val score = q * k * scale + mask
      val expScore = exp(score)
      expScore * v
    }

    // ========================================================================
    // 16. Softmax with Mask
    // out = exp(x - max) / sum(exp(x - max))
    // ========================================================================
    println("\n[16] Softmax with Mask")

    @TritonKernelMacro
    def softmaxMasked(x: Float, maxVal: Float, mask: Float): Float = {
      val masked = x + mask
      val shifted = masked - maxVal
      exp(shifted)
    }

    // ========================================================================
    // 17. SiLU / Swish activation
    // silu(x) = x * sigmoid(x)
    // ========================================================================
    println("\n[17] SiLU / Swish Activation")

    @TritonKernelMacro
    def silu(x: Float): Float = {
      x * sigmoid(x)
    }

    // ========================================================================
    // 18. Complex Multi-Head Attention
    // Combines multiple operations
    // ========================================================================
    println("\n[18] Complex Attention Kernel")

    @TritonKernelMacro
    def complexAttention(q: Float, k: Float, v: Float, scale: Float, attnScale: Float): Float = {
      val score = q * k * scale
      val attnWeight = sigmoid(score * attnScale)
      attnWeight * v
    }

    println("\n" + "=" * 80)
    println("All transformer kernels defined. Check compiler output for generated code.")
    println("=" * 80)
  }
}