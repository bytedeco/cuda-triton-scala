package cuda.dsl.benchmark

import cuda.dsl.dsl._

/** Test 50 complex CUDA kernels heavily utilizing tl.threadIdx(axis)
  * for thread identification in tile operations, warp-level primitives,
  * and multi-dimensional data movement patterns.
  *
  * Focus areas:
  * - Warp-level tile operations with tl.threadIdx(axis)
  * - Multi-axis thread identification for 2D/3D tiles
  * - Hierarchical reduction trees using threadIdx
  * - Tiled attention with intra-warp thread coordination
  * - Complex CUTLASS-style epilogues with threadIdx-based addressing
  */
object Test50ThreadIdxKernels {

  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat
  def tanh(x: Float): Float = scala.math.tanh(x.toDouble).toFloat
  def abs(x: Float): Float = scala.math.abs(x.toDouble).toFloat
  def max(a: Float, b: Float): Float = if (a > b) a else b
  def min(a: Float, b: Float): Float = if (a < b) a else b
  def log(x: Float): Float = scala.math.log(x.toDouble).toFloat
  def sigmoid(x: Float): Float = 1.0f / (1.0f + exp(-x))
  def gelu(x: Float): Float = 0.5f * x * (1.0f + tanh(sqrt(2.0f / 3.14159f) * (x + 0.044715f * x * x * x)))
  def relu(x: Float): Float = max(0.0f, x)
  def leakyrelu(x: Float): Float = if (x > 0.0f) x else 0.01f * x

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Test50ThreadIdxKernels: 50 Complex Kernels with tl.threadIdx(axis)")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels.txt for generated CUDA code.")
    println("=" * 80)

    // ========================================================================
    // Category 1: Warp-Level Tiled GEMM (10 kernels)
    // ========================================================================
    println("\n[1-10] Warp-Level Tiled GEMM with threadIdx")

    // 1. Warp-level 4x4 tile GEMM using threadIdx for lane routing
    @TritonKernelMacro
    def warpTileGemm4x4LaneKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 64
      val blockN = 64
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val laneId = (ty * 32) + tx

      if (laneId < 16) {
        val tileRow = laneId / 4
        val tileCol = laneId % 4
        val row = rowStart + tileRow
        val col = colStart + tileCol
        if (row < M && col < N) {
          var sum: Float = 0.0f
          val numTilesK = (K + 3) / 4
          0.until(numTilesK) foreach { tileK =>
            val kOffset = tileK * 4
            val aRow = row
            val aCol = kOffset + tileCol
            val bRow = kOffset + tileRow
            val bCol = col
            if (aCol < K && bRow < K) {
              val aVal = tl.load(A + aRow * K + aCol)
              val bVal = tl.load(B + bRow * N + bCol)
              sum = sum + aVal * bVal
            }
          }
          tl.store(C + row * N + col, sum)
        }
      }
      ()
    }
    println("[1] warpTileGemm4x4LaneKernel defined")

    // 2. Warp-level 8x8 tile GEMM with threadIdx-based bank conflict avoidance
    @TritonKernelMacro
    def warpTileGemm8x8BankAvoidKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 64
      val blockN = 64
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val laneId = (ty * 32) + tx

      val tileRow = laneId / 8
      val tileCol = laneId % 8
      val row = rowStart + tileRow
      val col = colStart + tileCol
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 64 * 64)
        tl.sharedMem("float", "s_b", 64 * 64)
        var sum: Float = 0.0f
        val numTilesK = (K + 7) / 8
        0.until(numTilesK) foreach { tileK =>
          val kOffset = tileK * 8
          val aRow = row
          val aCol = kOffset + tileCol
          val bRow = kOffset + tileRow
          val bCol = col
          if (aCol < K && bRow < K) {
            val aVal = tl.load(A + aRow * K + aCol)
            val bVal = tl.load(B + bRow * N + bCol)
            val aIdx = (tileRow * 8) + tileCol
            val bIdx = (tileRow * 8) + tileCol
            tl.sharedStore("s_a", aIdx, aVal)
            tl.sharedStore("s_b", bIdx, bVal)
          }
          tl.syncthreads()
          0.until(8) foreach { kk =>
            val aVal = tl.sharedLoad("s_a", tileRow * 8 + kk)
            val bVal = tl.sharedLoad("s_b", kk * 8 + tileCol)
            sum = sum + aVal * bVal
          }
          tl.syncthreads()
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[2] warpTileGemm8x8BankAvoidKernel defined")

    // 3. Warp-level transpose with threadIdx-based 2D lane mapping
    @TritonKernelMacro
    def warpTranspose16x16Kernel(A: Float, C: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 64
      val blockN = 64
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val laneId = (ty * 32) + tx

      if (laneId < 32) {
        tl.sharedMem("float", "s_data", 16 * 16)
        val row = laneId / 16
        val col = laneId % 16
        val srcRow = rowStart + row
        val srcCol = colStart + col
        if (srcRow < M && srcCol < N) {
          val val1 = tl.load(A + srcRow * N + srcCol)
          tl.sharedStore("s_data", row * 16 + col, val1)
        }
        tl.syncthreads()
        val dstRow = colStart + row
        val dstCol = rowStart + col
        if (dstRow < M && dstCol < N) {
          val val2 = tl.sharedLoad("s_data", col * 16 + row)
          tl.store(C + dstRow * N + dstCol, val2)
        }
      }
      ()
    }
    println("[3] warpTranspose16x16Kernel defined")

    // 4. Block-level softmax with threadIdx for intra-block reduction
    @TritonKernelMacro
    def blockSoftmaxKernel(X: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { i =>
          val col = i
          val v = tl.load(X + row * N + col)
          maxVal = max(maxVal, v)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { i =>
          val col = i
          val v = tl.load(X + row * N + col)
          sum = sum + exp(v - maxVal)
        }
        val v = tl.load(X + row * N + tx)
        val result = exp(v - maxVal) / sum
        tl.store(Y + row * N + tx, result)
      }
      ()
    }
    println("[4] blockSoftmaxKernel defined")

    // 5. Block-level layer norm with threadIdx for parallel reduction
    @TritonKernelMacro
    def blockLayerNormKernel(X: Float, Y: Float, M: Int, N: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var sum: Float = 0.0f
        0.until(N) foreach { i =>
          val col = i
          val v = tl.load(X + row * N + col)
          sum = sum + v
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        0.until(N) foreach { i =>
          val col = i
          val v = tl.load(X + row * N + col)
          val diff = v - mean
          sqSum = sqSum + diff * diff
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + eps)
        val v = tl.load(X + row * N + tx)
        val result = (v - mean) * invStd
        tl.store(Y + row * N + tx, result)
      }
      ()
    }
    println("[5] blockLayerNormKernel defined")

    // 6. Block-level attention with threadIdx for Q/K/V projection
    @TritonKernelMacro
    def blockAttentionQKVKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          result = result + attn * v
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[6] blockAttentionQKVKernel defined")

    // 7. Block-level GELU activation with threadIdx-based element-wise ops
    @TritonKernelMacro
    def blockGeluKernel(X: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val col = tx
        val x = tl.load(X + row * N + col)
        val result = gelu(x)
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[7] blockGeluKernel defined")

    // 8. Block-level residual add with threadIdx for block-sparse addition
    @TritonKernelMacro
    def blockResidualKernel(X: Float, Y: Float, Z: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val col = tx
        val x = tl.load(X + row * N + col)
        val y = tl.load(Y + row * N + col)
        tl.store(Z + row * N + col, x + y)
      }
      ()
    }
    println("[8] blockResidualKernel defined")

    // 9. Block-level bi-level reduction (max then sum) for attention
    @TritonKernelMacro
    def blockBiLevelReduceKernel(X: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var localMax: Float = Float.MinValue
        0.until(N) foreach { i =>
          val v = tl.load(X + row * N + i)
          localMax = max(localMax, v)
        }
        var localSum: Float = 0.0f
        0.until(N) foreach { i =>
          val v = tl.load(X + row * N + i)
          localSum = localSum + exp(v - localMax)
        }
        val v = tl.load(X + row * N + tx)
        val result = exp(v - localMax) / localSum
        tl.store(Y + row * N + tx, result)
      }
      ()
    }
    println("[9] blockBiLevelReduceKernel defined")

    // 10. Block-level gated activation (SiLU/GELU) for transformers
    @TritonKernelMacro
    def blockGatedActivationKernel(X: Float, G: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val col = tx
        val x = tl.load(X + row * N + col)
        val g = tl.load(G + row * N + col)
        val result = x * sigmoid(x) * g
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[10] blockGatedActivationKernel defined")

    // ========================================================================
    // Category 2: Multi-Axis ThreadIdx Tiled Operations (10 kernels)
    // ========================================================================
    println("\n[11-20] Multi-Axis ThreadIdx Tiled Operations")

    // 11. 2D tile GEMM with threadIdx.x/y for 2D thread mapping
    @TritonKernelMacro
    def tile2dGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 32 * 32)
        tl.sharedMem("float", "s_b", 32 * 32)
        var sum: Float = 0.0f
        val numTilesK = (K + 31) / 32
        0.until(numTilesK) foreach { tileK =>
          val kOffset = tileK * 32
          val aRow = row
          val aCol = kOffset + tx
          val bRow = kOffset + ty
          val bCol = col
          if (aCol < K && bRow < K) {
            val aVal = tl.load(A + aRow * K + aCol)
            val bVal = tl.load(B + bRow * N + bCol)
            tl.sharedStore("s_a", ty * 32 + tx, aVal)
            tl.sharedStore("s_b", ty * 32 + tx, bVal)
          }
          tl.syncthreads()
          0.until(32) foreach { kk =>
            val aVal = tl.sharedLoad("s_a", ty * 32 + kk)
            val bVal = tl.sharedLoad("s_b", kk * 32 + tx)
            sum = sum + aVal * bVal
          }
          tl.syncthreads()
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[11] tile2dGemmKernel defined")

    // 12. 2D tile attention with threadIdx for 2D Q/K/V
    @TritonKernelMacro
    def tile2dAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + j)
          val k = tl.load(K + col * N + j)
          val score = q * k / sqrt(N.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + j)
          val k = tl.load(K + col * N + j)
          val score = q * k / sqrt(N.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + j)
          val k = tl.load(K + col * N + j)
          val score = q * k / sqrt(N.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + col * N + j)
          result = result + attn * v
        }
        tl.store(O + row * N + col, result)
      }
      ()
    }
    println("[12] tile2dAttentionKernel defined")

    // 13. 2D tile layer norm with threadIdx parallel reduction
    @TritonKernelMacro
    def tile2dLayerNormKernel(X: Float, Y: Float, M: Int, N: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(X + row * N + j)
          sum = sum + v
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(X + row * N + j)
          val diff = v - mean
          sqSum = sqSum + diff * diff
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + eps)
        val v = tl.load(X + row * N + col)
        val result = (v - mean) * invStd
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[13] tile2dLayerNormKernel defined")

    // 14. 2D tile RMSNorm with threadIdx
    @TritonKernelMacro
    def tile2dRmsNormKernel(X: Float, Y: Float, M: Int, N: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        var sqSum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(X + row * N + j)
          sqSum = sqSum + v * v
        }
        val rms = sqrt(sqSum / N + eps)
        val v = tl.load(X + row * N + col)
        val result = v / rms
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[14] tile2dRmsNormKernel defined")

    // 15. 2D tile sigmoid with threadIdx
    @TritonKernelMacro
    def tile2dSigmoidKernel(X: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        val x = tl.load(X + row * N + col)
        val result = sigmoid(x)
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[15] tile2dSigmoidKernel defined")

    // 16. 2D tile SwiGLU activation
    @TritonKernelMacro
    def tile2dSwigluKernel(X: Float, G: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        val x = tl.load(X + row * N + col)
        val g = tl.load(G + row * N + col)
        val result = x * sigmoid(x) * g
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[16] tile2dSwigluKernel defined")

    // 17. 2D tile LeakyReLU
    @TritonKernelMacro
    def tile2dLeakyReluKernel(X: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        val x = tl.load(X + row * N + col)
        val result = leakyrelu(x)
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[17] tile2dLeakyReluKernel defined")

    // 18. 2D tile batch normalization forward
    @TritonKernelMacro
    def tile2dBatchNormKernel(X: Float, Y: Float, M: Int, N: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        var sum: Float = 0.0f
        0.until(M) foreach { i =>
          val v = tl.load(X + i * N + col)
          sum = sum + v
        }
        val mean = sum / M
        var sqSum: Float = 0.0f
        0.until(M) foreach { i =>
          val v = tl.load(X + i * N + col)
          val diff = v - mean
          sqSum = sqSum + diff * diff
        }
        val variance = sqSum / M
        val invStd = 1.0f / sqrt(variance + eps)
        val v = tl.load(X + row * N + col)
        val result = (v - mean) * invStd
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[18] tile2dBatchNormKernel defined")

    // 19. 2D tile cross entropy loss
    @TritonKernelMacro
    def tile2dCrossEntropyKernel(_logits: Float, _targets: Float, _loss: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        var maxLogit: Float = Float.MinValue
        0.until(N) foreach { j =>
          val v = tl.load(_logits + row * N + j)
          maxLogit = max(maxLogit, v)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(_logits + row * N + j)
          sum = sum + exp(v - maxLogit)
        }
        val target = tl.load(_targets + row)
        val logits = tl.load(_logits + row * N + col)
        val logProb = logits - maxLogit - log(sum)
        val loss = -logProb
        tl.store(_loss + row, loss)
      }
      ()
    }
    println("[19] tile2dCrossEntropyKernel defined")

    // 20. 2D tile dropout forward
    @TritonKernelMacro
    def tile2dDropoutKernel(X: Float, Y: Float, M: Int, N: Int, p: Float): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        val x = tl.load(X + row * N + col)
        val scale = 1.0f / (1.0f - p)
        val result = x * scale
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[20] tile2dDropoutKernel defined")

    // ========================================================================
    // Category 3: Hierarchical Reduction Trees (10 kernels)
    // ========================================================================
    println("\n[21-30] Hierarchical Reduction Trees with threadIdx")

    // 21. Block-reduction GEMM with warp-level accumulation
    @TritonKernelMacro
    def blockReduceGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 64
      val blockN = 64
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val laneId = (ty * 32) + tx

      if (laneId < 32) {
        val tileRow = laneId / 4
        val tileCol = laneId % 4
        val row = rowStart + tileRow
        val col = colStart + tileCol
        if (row < M && col < N) {
          var partialSum: Float = 0.0f
          0.until(K) foreach { k =>
            val aVal = tl.load(A + row * K + k)
            val bVal = tl.load(B + k * N + col)
            partialSum = partialSum + aVal * bVal
          }
          if (laneId == 0) {
            tl.store(C + row * N + col, partialSum)
          }
        }
      }
      ()
    }
    println("[21] blockReduceGemmKernel defined")

    // 22. Hierarchical softmax with block reduction
    @TritonKernelMacro
    def hierarchicalSoftmaxKernel(X: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { i =>
          val v = tl.load(X + row * N + i)
          maxVal = max(maxVal, v)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { i =>
          val v = tl.load(X + row * N + i)
          sum = sum + exp(v - maxVal)
        }
        if (tx < N) {
          val v = tl.load(X + row * N + tx)
          val result = exp(v - maxVal) / sum
          tl.store(Y + row * N + tx, result)
        }
      }
      ()
    }
    println("[22] hierarchicalSoftmaxKernel defined")

    // 23. Block segmented reduction for sparse attention
    @TritonKernelMacro
    def blockSegmentReduceKernel(X: Float, Y: Float, M: Int, N: Int, segSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val segId = tx / segSize
        val idxInSeg = tx % segSize
        if (segId < (N / segSize)) {
          var maxVal: Float = Float.MinValue
          0.until(segSize) foreach { i =>
            val col = segId * segSize + i
            if (col < N) {
              val v = tl.load(X + row * N + col)
              maxVal = max(maxVal, v)
            }
          }
          var sum: Float = 0.0f
          0.until(segSize) foreach { i =>
            val col = segId * segSize + i
            if (col < N) {
              val v = tl.load(X + row * N + col)
              sum = sum + exp(v - maxVal)
            }
          }
          if (idxInSeg == 0) {
            tl.store(Y + row * (N / segSize) + segId, maxVal + log(sum))
          }
        }
      }
      ()
    }
    println("[23] blockSegmentReduceKernel defined")

    // 24. Hierarchical layer norm with block reduction
    @TritonKernelMacro
    def hierarchicalLayerNormKernel(X: Float, Y: Float, M: Int, N: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var sum: Float = 0.0f
        0.until(N) foreach { i =>
          val v = tl.load(X + row * N + i)
          sum = sum + v
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        0.until(N) foreach { i =>
          val v = tl.load(X + row * N + i)
          val diff = v - mean
          sqSum = sqSum + diff * diff
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + eps)
        if (tx < N) {
          val v = tl.load(X + row * N + tx)
          val result = (v - mean) * invStd
          tl.store(Y + row * N + tx, result)
        }
      }
      ()
    }
    println("[24] hierarchicalLayerNormKernel defined")

    // 25. Multi-level block reduction for batched GEMM
    @TritonKernelMacro
    def multiLevelReduceGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int, batch: Int): Unit = {
      val pid = tl.program_id(0)
      val b = pid / (M * N)
      val idx = pid % (M * N)
      val row = idx / N
      val col = idx % N
      if (b >= batch || row >= M || col >= N) return

      val tx = tl.threadIdx(0)
      if (tx == 0) {
        var sum: Float = 0.0f
        0.until(K) foreach { k =>
          val aVal = tl.load(A + b * M * K + row * K + k)
          val bVal = tl.load(B + b * K * N + k * N + col)
          sum = sum + aVal * bVal
        }
        tl.store(C + b * M * N + row * N + col, sum)
      }
      ()
    }
    println("[25] multiLevelReduceGemmKernel defined")

    // 26. Block tree attention with causal masking
    @TritonKernelMacro
    def blockCausalAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val col = tx
        val colLimit = col
        var maxVal: Float = Float.MinValue
        0.until(colLimit + 1) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        0.until(colLimit + 1) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        0.until(colLimit + 1) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          result = result + attn * v
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[26] blockCausalAttentionKernel defined")

    // 27. Hierarchical GEMM with block-level blocking
    @TritonKernelMacro
    def hierarchicalGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 128
      val blockN = 128
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val laneId = (ty * 32) + tx

      val tileRow = laneId / 8
      val tileCol = laneId % 8
      val row = rowStart + tileRow
      val col = colStart + tileCol
      if (row < M && col < N) {
        var sum: Float = 0.0f
        val numTilesK = (K + 7) / 8
        0.until(numTilesK) foreach { tileK =>
          var tileSum: Float = 0.0f
          0.until(8) foreach { kk =>
            val aVal = tl.load(A + row * K + tileK * 8 + kk)
            val bVal = tl.load(B + (tileK * 8 + kk) * N + col)
            tileSum = tileSum + aVal * bVal
          }
          sum = sum + tileSum
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[27] hierarchicalGemmKernel defined")

    // 28. Block-level exclusive scan for prefix sum
    @TritonKernelMacro
    def blockExclusiveScanKernel(X: Float, Y: Float, N: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= 1) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var val1: Float = tl.load(X + tx)
        if (tx > 0) {
          var prefix: Float = 0.0f
          0.until(tx) foreach { i =>
            prefix = prefix + tl.load(X + i)
          }
          tl.store(Y + tx, prefix)
        } else {
          tl.store(Y + tx, 0.0f)
        }
      }
      ()
    }
    println("[28] blockExclusiveScanKernel defined")

    // 29. Multi-blocks hierarchical reduction for massive GEMM
    @TritonKernelMacro
    def multiBlockReduceGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 256
      val blockN = 256
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val laneId = (ty * 32) + tx

      if (laneId < 128) {
        val tileRow = laneId / 16
        val tileCol = laneId % 16
        val row = rowStart + tileRow
        val col = colStart + tileCol
        if (row < M && col < N) {
          var sum: Float = 0.0f
          0.until(K) foreach { k =>
            val aVal = tl.load(A + row * K + k)
            val bVal = tl.load(B + k * N + col)
            sum = sum + aVal * bVal
          }
          if (laneId == 0) {
            tl.store(C + row * N + col, sum)
          }
        }
      }
      ()
    }
    println("[29] multiBlockReduceGemmKernel defined")

    // 30. Hierarchical softmax with multiple block levels
    @TritonKernelMacro
    def multiBlockSoftmaxKernel(X: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { i =>
          val v = tl.load(X + row * N + i)
          maxVal = max(maxVal, v)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { i =>
          val v = tl.load(X + row * N + i)
          sum = sum + exp(v - maxVal)
        }
        if (tx < N) {
          val v = tl.load(X + row * N + tx)
          val result = exp(v - maxVal) / sum
          tl.store(Y + row * N + tx, result)
        }
      }
      ()
    }
    println("[30] multiBlockSoftmaxKernel defined")

    // ========================================================================
    // Category 4: Tiled Attention Mechanisms (10 kernels)
    // ========================================================================
    println("\n[31-40] Tiled Attention Mechanisms with threadIdx")

    // 31. Flash-style tiled attention with threadIdx masking
    @TritonKernelMacro
    def flashAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < blockSize) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          result = result + attn * v
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[31] flashAttentionKernel defined")

    // 32. Multi-head attention with threadIdx for head routing
    @TritonKernelMacro
    def multiHeadAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, numHeads: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      val headDim = D / numHeads
      if (row >= M) return

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val headId = ty
      if (headId >= numHeads) return

      if (tx < headDim) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * D + headId * headDim + tx)
          val k = tl.load(K + row * D + headId * headDim + j)
          val score = q * k / sqrt(headDim.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * D + headId * headDim + tx)
          val k = tl.load(K + row * D + headId * headDim + j)
          val score = q * k / sqrt(headDim.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * D + headId * headDim + tx)
          val k = tl.load(K + row * D + headId * headDim + j)
          val score = q * k / sqrt(headDim.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * D + headId * headDim + j)
          result = result + attn * v
        }
        tl.store(O + row * D + headId * headDim + tx, result)
      }
      ()
    }
    println("[32] multiHeadAttentionKernel defined")

    // 33. Grouped query attention with threadIdx
    @TritonKernelMacro
    def groupedQueryAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, numHeads: Int, numKVHeads: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      val headDim = D / numHeads
      val kvHeadDim = D / numKVHeads
      if (row >= M) return

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val headId = ty
      val kvHeadId = (headId * numKVHeads) / numHeads
      if (headId >= numHeads) return

      if (tx < headDim) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * D + headId * headDim + tx)
          val kIdx = j * kvHeadDim / N
          val k = tl.load(K + row * D + kvHeadId * kvHeadDim + kIdx)
          val score = q * k / sqrt(headDim.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * D + headId * headDim + tx)
          val kIdx = j * kvHeadDim / N
          val k = tl.load(K + row * D + kvHeadId * kvHeadDim + kIdx)
          val score = q * k / sqrt(headDim.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * D + headId * headDim + tx)
          val kIdx = j * kvHeadDim / N
          val k = tl.load(K + row * D + kvHeadId * kvHeadDim + kIdx)
          val score = q * k / sqrt(headDim.toFloat)
          val attn = exp(score - maxVal) / sum
          val vIdx = j * kvHeadDim / N
          val v = tl.load(V + row * D + kvHeadId * kvHeadDim + vIdx)
          result = result + attn * v
        }
        tl.store(O + row * D + headId * headDim + tx, result)
      }
      ()
    }
    println("[33] groupedQueryAttentionKernel defined")

    // 34. Sliding window attention with threadIdx
    @TritonKernelMacro
    def slidingWindowAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, windowSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val startCol = max(0, row - windowSize).toInt
        val endCol = min(N, row + windowSize + 1).toInt
        var maxVal: Float = Float.MinValue
        startCol.until(endCol) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        startCol.until(endCol) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        startCol.until(endCol) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          result = result + attn * v
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[34] slidingWindowAttentionKernel defined")

    // 35. Cross attention with threadIdx for encoder-decoder
    @TritonKernelMacro
    def crossAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          result = result + attn * v
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[35] crossAttentionKernel defined")

    // 36. Bidirectional attention with threadIdx
    @TritonKernelMacro
    def bidirectionalAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var fwdMax: Float = Float.MinValue
        0.until(row + 1) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          fwdMax = max(fwdMax, score)
        }
        var fwdSum: Float = 0.0f
        0.until(row + 1) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          fwdSum = fwdSum + exp(score - fwdMax)
        }
        var bwdMax: Float = Float.MinValue
        row.until(M) foreach { i =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + i * N + tx)
          val score = q * k / sqrt(D.toFloat)
          bwdMax = max(bwdMax, score)
        }
        var bwdSum: Float = 0.0f
        row.until(M) foreach { i =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + i * N + tx)
          val score = q * k / sqrt(D.toFloat)
          bwdSum = bwdSum + exp(score - bwdMax)
        }
        val maxVal = max(fwdMax, bwdMax)
        val sum = fwdSum * exp(fwdMax - maxVal) + bwdSum * exp(bwdMax - maxVal)
        var result: Float = 0.0f
        0.until(row + 1) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          result = result + attn * v
        }
        row.until(M) foreach { i =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + i * N + tx)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + i * N + tx)
          result = result + attn * v
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[36] bidirectionalAttentionKernel defined")

    // 37. Local attention with threadIdx for sparse computation
    @TritonKernelMacro
    def localAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, localSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val startCol = ((row / localSize) * localSize).toInt
        val endCol = min(N, startCol + localSize).toInt
        var maxVal: Float = Float.MinValue
        startCol.until(endCol) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        startCol.until(endCol) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
        }
        var result: Float = 0.0f
        startCol.until(endCol) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          result = result + attn * v
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[37] localAttentionKernel defined")

    // 38. Sparse attention with threadIdx for pattern masking
    @TritonKernelMacro
    def sparseAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, sparsity: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { j =>
          if ((j % sparsity) == 0) {
            val q = tl.load(Q + row * N + tx)
            val k = tl.load(K + row * N + j)
            val score = q * k / sqrt(D.toFloat)
            maxVal = max(maxVal, score)
          }
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          if ((j % sparsity) == 0) {
            val q = tl.load(Q + row * N + tx)
            val k = tl.load(K + row * N + j)
            val score = q * k / sqrt(D.toFloat)
            sum = sum + exp(score - maxVal)
          }
        }
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          if ((j % sparsity) == 0) {
            val q = tl.load(Q + row * N + tx)
            val k = tl.load(K + row * N + j)
            val score = q * k / sqrt(D.toFloat)
            val attn = exp(score - maxVal) / sum
            val v = tl.load(V + row * N + j)
            result = result + attn * v
          }
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[38] sparseAttentionKernel defined")

    // 39. Global-local attention with threadIdx
    @TritonKernelMacro
    def globalLocalAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, localSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var globalMax: Float = Float.MinValue
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          globalMax = max(globalMax, score)
        }
        var globalSum: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          globalSum = globalSum + exp(score - globalMax)
        }
        val localStart = ((row / localSize) * localSize).toInt
        val localEnd = min(N, localStart + localSize).toInt
        var localMax: Float = Float.MinValue
        localStart.until(localEnd) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          localMax = max(localMax, score)
        }
        var localSum: Float = 0.0f
        localStart.until(localEnd) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          localSum = localSum + exp(score - localMax)
        }
        val maxVal = max(globalMax, localMax)
        val sum = globalSum * exp(globalMax - maxVal) + localSum * exp(localMax - maxVal)
        var result: Float = 0.0f
        localStart.until(localEnd) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          result = result + attn * v
        }
        tl.store(O + row * N + tx, result)
      }
      ()
    }
    println("[39] globalLocalAttentionKernel defined")

    // 40. Kernel attention with threadIdx for performer-style
    @TritonKernelMacro
    def kernelAttentionKernel(Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, numFeatures: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < D) {
        var sum: Float = 0.0f
        0.until(numFeatures) foreach { f =>
          var qk: Float = 0.0f
          0.until(D) foreach { d =>
            val q = tl.load(Q + row * D + d)
            val k = tl.load(K + f * D + d)
            qk = qk + q * k
          }
          val phi = exp(qk / sqrt(D.toFloat))
          var v: Float = 0.0f
          0.until(D) foreach { d =>
            val kv = tl.load(V + f * D + d)
            v = v + phi * kv
          }
          sum = sum + v
        }
        tl.store(O + row * D + tx, sum)
      }
      ()
    }
    println("[40] kernelAttentionKernel defined")

    // ========================================================================
    // Category 5: Fused Tiled Operations (10 kernels)
    // ========================================================================
    println("\n[41-50] Fused Tiled Operations with threadIdx")

    // 41. Fused GEMM + GELU with threadIdx
    @TritonKernelMacro
    def fusedGemmGeluKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        var sum: Float = 0.0f
        0.until(K) foreach { k =>
          val aVal = tl.load(A + row * K + k)
          val bVal = tl.load(B + k * N + col)
          sum = sum + aVal * bVal
        }
        val result = gelu(sum)
        tl.store(C + row * N + col, result)
      }
      ()
    }
    println("[41] fusedGemmGeluKernel defined")

    // 42. Fused GEMM + residual with threadIdx
    @TritonKernelMacro
    def fusedGemmResidualKernel(C: Float, A: Float, B: Float, D: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        var sum: Float = 0.0f
        0.until(K) foreach { k =>
          val aVal = tl.load(A + row * K + k)
          val bVal = tl.load(B + k * N + col)
          sum = sum + aVal * bVal
        }
        val residual = tl.load(D + row * N + col)
        val result = sum + residual
        tl.store(C + row * N + col, result)
      }
      ()
    }
    println("[42] fusedGemmResidualKernel defined")

    // 43. Fused LayerNorm + Attention with threadIdx
    @TritonKernelMacro
    def fusedLayerNormAttentionKernel(X: Float, Y: Float, Q: Float, K: Float, V: Float, O: Float, M: Int, N: Int, D: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < D) {
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(X + row * N + j)
          sum = sum + v
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(X + row * N + j)
          val diff = v - mean
          sqSum = sqSum + diff * diff
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + eps)
        val normed = (tl.load(X + row * N + tx) - mean) * invStd
        tl.store(Y + row * N + tx, normed)
        tl.store(Q + row * D + tx, normed)
      }
      ()
    }
    println("[43] fusedLayerNormAttentionKernel defined")

    // 44. Fused RMSNorm + SwiGLU with threadIdx
    @TritonKernelMacro
    def fusedRmsNormSwigluKernel(X: Float, G: Float, Y: Float, M: Int, N: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var sqSum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(X + row * N + j)
          sqSum = sqSum + v * v
        }
        val rms = sqrt(sqSum / N + eps)
        val x = tl.load(X + row * N + tx)
        val g = tl.load(G + row * N + tx)
        val normed = x / rms
        val result = normed * sigmoid(normed) * g
        tl.store(Y + row * N + tx, result)
      }
      ()
    }
    println("[44] fusedRmsNormSwigluKernel defined")

    // 45. Fused skip connection with threadIdx
    @TritonKernelMacro
    def fusedSkipConnectionKernel(X: Float, F: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        val x = tl.load(X + row * N + col)
        val f = tl.load(F + row * N + col)
        val result = x + f
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[45] fusedSkipConnectionKernel defined")

    // 46. Fused gated activation with threadIdx
    @TritonKernelMacro
    def fusedGatedActivationKernel(X: Float, G1: Float, G2: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        val x = tl.load(X + row * N + col)
        val g1 = tl.load(G1 + row * N + col)
        val g2 = tl.load(G2 + row * N + col)
        val result = gelu(x) * sigmoid(g1) * g2
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[46] fusedGatedActivationKernel defined")

    // 47. Fused SwiGLU with threadIdx
    @TritonKernelMacro
    def fusedSwigluKernel(X: Float, G: Float, Y: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val blockM = 32
      val blockN = 32
      val numBlocksM = (M + blockM - 1) / blockM
      val numBlocksN = (N + blockN - 1) / blockN
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockIdM = pid / numBlocksN
      val blockIdN = pid % numBlocksN
      val rowStart = blockIdM * blockM
      val colStart = blockIdN * blockN

      val tx = tl.threadIdx(0)
      val ty = tl.threadIdx(1)
      val row = rowStart + ty
      val col = colStart + tx

      if (row < M && col < N) {
        val x = tl.load(X + row * N + col)
        val g = tl.load(G + row * N + col)
        val result = x * sigmoid(x) * g
        tl.store(Y + row * N + col, result)
      }
      ()
    }
    println("[47] fusedSwigluKernel defined")

    // 48. Fused MHA + FFN with threadIdx
    @TritonKernelMacro
    def fusedMhaFfnKernel(Q: Float, K: Float, V: Float, W1: Float, W2: Float, O: Float, M: Int, N: Int, D: Int, hidden: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < D) {
        var attnOut: Float = 0.0f
        var maxVal: Float = Float.MinValue
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
        }
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
        }
        0.until(N) foreach { j =>
          val q = tl.load(Q + row * N + tx)
          val k = tl.load(K + row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V + row * N + j)
          attnOut = attnOut + attn * v
        }
        var ffnOut: Float = 0.0f
        0.until(hidden) foreach { h =>
          val w = tl.load(W1 + tx * hidden + h)
          ffnOut = ffnOut + attnOut * w
        }
        ffnOut = gelu(ffnOut)
        var finalOut: Float = 0.0f
        0.until(hidden) foreach { h =>
          val w = tl.load(W2 + h * D + tx)
          finalOut = finalOut + ffnOut * w
        }
        tl.store(O + row * D + tx, finalOut)
      }
      ()
    }
    println("[48] fusedMhaFfnKernel defined")

    // 49. Fused RMSNorm + skip + GEMM with threadIdx
    @TritonKernelMacro
    def fusedRmsNormSkipGemmKernel(X: Float, R: Float, A: Float, B: Float, Y: Float, M: Int, N: Int, K: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var sqSum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(X + row * N + j)
          sqSum = sqSum + v * v
        }
        val rms = sqrt(sqSum / N + eps)
        val x = tl.load(X + row * N + tx)
        val normed = x / rms
        val residual = tl.load(R + row * N + tx)
        val combined = normed + residual
        var sum: Float = 0.0f
        0.until(K) foreach { k =>
          sum = sum + combined * tl.load(A + row * K + k) * tl.load(B + k * N + tx)
        }
        tl.store(Y + row * N + tx, sum)
      }
      ()
    }
    println("[49] fusedRmsNormSkipGemmKernel defined")

    // 50. Fused Bias + GELU + Residual + LayerNorm with threadIdx
    @TritonKernelMacro
    def fusedBiasGeluResidualLayerNormKernel(X: Float, B: Float, R: Float, Y: Float, M: Int, N: Int, eps: Float): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val x = tl.load(X + row * N + tx)
        val bias = tl.load(B + tx)
        val residual = tl.load(R + row * N + tx)
        val combined = gelu(x + bias) + residual
        tl.store(Y + row * N + tx, combined)
        var sum: Float = 0.0f
        0.until(N) foreach { j =>
          sum = sum + tl.load(Y + row * N + j)
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        0.until(N) foreach { j =>
          val v = tl.load(Y + row * N + j)
          sqSum = sqSum + (v - mean) * (v - mean)
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + eps)
        val finalVal = (tl.load(Y + row * N + tx) - mean) * invStd
        tl.store(Y + row * N + tx, finalVal)
      }
      ()
    }
    println("[50] fusedBiasGeluResidualLayerNormKernel defined")

    // ========================================================================
    println("\n" + "=" * 80)
    println("All 50 threadIdx-based CUDA kernels defined successfully!")
    println("=" * 80)
  }
}