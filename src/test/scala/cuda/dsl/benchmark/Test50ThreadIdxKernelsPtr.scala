package cuda.dsl.benchmark

import cuda.dsl.core.{FloatPtr, IntPtr}
import cuda.dsl.dsl._

/** Pointer variant of Test50ThreadIdxKernels using FloatPtr/IntPtr types.
  * All kernel method names have Ptr suffix to avoid naming conflicts.
  */
object Test50ThreadIdxKernelsPtr {

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
    println("Test50ThreadIdxKernelsPtr: 50 Complex Kernels with tl.threadIdx (pointer variant)")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels_ptr.txt for generated CUDA code.")
    println("=" * 80)

    // ========================================================================
    // Category 1: Warp-Level Tiled GEMM (10 kernels)
    // ========================================================================
    println("\n[1-10] Warp-Level Tiled GEMM with threadIdx")

    // 1. Warp-level 4x4 tile GEMM using threadIdx for lane routing
    @TritonKernelMacro
    def warpTileGemm4x4LaneKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, M: Int, N: Int, K: Int): Unit = {
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
          var tileK: Int = 0
          while (tileK < numTilesK) {
            val kOffset = tileK * 4
            val aRow = row
            val aCol = kOffset + tileCol
            val bRow = kOffset + tileRow
            val bCol = col
            if (aCol < K && bRow < K) {
              val aVal = tl.load(A, aRow * K + aCol)
              val bVal = tl.load(B, bRow * N + bCol)
              sum = sum + aVal * bVal
            }
            tileK = tileK + 1
          }
          tl.store(C, row * N + col, sum)
        }
      }
      ()
    }
    println("[1] warpTileGemm4x4LaneKernelPtr defined")

    // 2. Warp-level 8x8 tile GEMM with threadIdx-based bank conflict avoidance
    @TritonKernelMacro
    def warpTileGemm8x8BankAvoidKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, M: Int, N: Int, K: Int): Unit = {
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
        var tileK: Int = 0
        while (tileK < numTilesK) {
          val kOffset = tileK * 8
          val aRow = row
          val aCol = kOffset + tileCol
          val bRow = kOffset + tileRow
          val bCol = col
          if (aCol < K && bRow < K) {
            val aVal = tl.load(A, aRow * K + aCol)
            val bVal = tl.load(B, bRow * N + bCol)
            val aIdx = (tileRow * 8) + tileCol
            val bIdx = (tileRow * 8) + tileCol
            tl.sharedStore("s_a", aIdx, aVal)
            tl.sharedStore("s_b", bIdx, bVal)
          }
          tl.syncthreads()
          var kk: Int = 0
          while (kk < 8) {
            val aVal = tl.sharedLoad("s_a", tileRow * 8 + kk)
            val bVal = tl.sharedLoad("s_b", kk * 8 + tileCol)
            sum = sum + aVal * bVal
            kk = kk + 1
          }
          tl.syncthreads()
          tileK = tileK + 1
        }
        tl.store(C, row * N + col, sum)
      }
      ()
    }
    println("[2] warpTileGemm8x8BankAvoidKernelPtr defined")

    // 3. Warp-level transpose with threadIdx-based 2D lane mapping
    @TritonKernelMacro
    def warpTranspose16x16KernelPtr(A: FloatPtr, C: FloatPtr, M: Int, N: Int): Unit = {
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
          val val1 = tl.load(A, srcRow * N + srcCol)
          tl.sharedStore("s_data", row * 16 + col, val1)
        }
        tl.syncthreads()
        val dstRow = colStart + row
        val dstCol = rowStart + col
        if (dstRow < M && dstCol < N) {
          val val2 = tl.sharedLoad("s_data", col * 16 + row)
          tl.store(C, dstRow * N + dstCol, val2)
        }
      }
      ()
    }
    println("[3] warpTranspose16x16KernelPtr defined")

    // 4. Block-level softmax with threadIdx for intra-block reduction
    @TritonKernelMacro
    def blockSoftmaxKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        var i: Int = 0
        while (i < N) {
          val col = i
          val v = tl.load(X, row * N + col)
          maxVal = max(maxVal, v)
          i = i + 1
        }
        var sum: Float = 0.0f
        i = 0
        while (i < N) {
          val col = i
          val v = tl.load(X, row * N + col)
          sum = sum + exp(v - maxVal)
          i = i + 1
        }
        val v = tl.load(X, row * N + tx)
        val result = exp(v - maxVal) / sum
        tl.store(Y, row * N + tx, result)
      }
      ()
    }
    println("[4] blockSoftmaxKernelPtr defined")

    // 5. Block-level layer norm with threadIdx for parallel reduction
    @TritonKernelMacro
    def blockLayerNormKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int, eps: FloatPtr): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      val epsVal = eps.get()
      if (tx < N) {
        var sum: Float = 0.0f
        var i: Int = 0
        while (i < N) {
          val col = i
          val v = tl.load(X, row * N + col)
          sum = sum + v
          i = i + 1
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        i = 0
        while (i < N) {
          val col = i
          val v = tl.load(X, row * N + col)
          val diff = v - mean
          sqSum = sqSum + diff * diff
          i = i + 1
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + epsVal)
        val v = tl.load(X, row * N + tx)
        val result = (v - mean) * invStd
        tl.store(Y, row * N + tx, result)
      }
      ()
    }
    println("[5] blockLayerNormKernelPtr defined")

    // 6. Block-level attention with threadIdx for Q/K/V projection
    @TritonKernelMacro
    def blockAttentionQKVKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        var j: Int = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[6] blockAttentionQKVKernelPtr defined")

    // 7. Block-level GELU activation with threadIdx-based element-wise ops
    @TritonKernelMacro
    def blockGeluKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val col = tx
        val x = tl.load(X, row * N + col)
        val result = gelu(x)
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[7] blockGeluKernelPtr defined")

    // 8. Block-level residual add with threadIdx for block-sparse addition
    @TritonKernelMacro
    def blockResidualKernelPtr(X: FloatPtr, Y: FloatPtr, Z: FloatPtr, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val col = tx
        val x = tl.load(X, row * N + col)
        val y = tl.load(Y, row * N + col)
        tl.store(Z, row * N + col, x + y)
      }
      ()
    }
    println("[8] blockResidualKernelPtr defined")

    // 9. Block-level bi-level reduction (max then sum) for attention
    @TritonKernelMacro
    def blockBiLevelReduceKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var localMax: Float = Float.MinValue
        var i: Int = 0
        while (i < N) {
          val v = tl.load(X, row * N + i)
          localMax = max(localMax, v)
          i = i + 1
        }
        var localSum: Float = 0.0f
        i = 0
        while (i < N) {
          val v = tl.load(X, row * N + i)
          localSum = localSum + exp(v - localMax)
          i = i + 1
        }
        val v = tl.load(X, row * N + tx)
        val result = exp(v - localMax) / localSum
        tl.store(Y, row * N + tx, result)
      }
      ()
    }
    println("[9] blockBiLevelReduceKernelPtr defined")

    // 10. Block-level gated activation (SiLU/GELU) for transformers
    @TritonKernelMacro
    def blockGatedActivationKernelPtr(X: FloatPtr, G: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val col = tx
        val x = tl.load(X, row * N + col)
        val g = tl.load(G, row * N + col)
        val result = x * sigmoid(x) * g
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[10] blockGatedActivationKernelPtr defined")

    // ========================================================================
    // Category 2: Multi-Axis ThreadIdx Tiled Operations (10 kernels)
    // ========================================================================
    println("\n[11-20] Multi-Axis ThreadIdx Tiled Operations")

    // 11. 2D tile GEMM with threadIdx.x/y for 2D thread mapping
    @TritonKernelMacro
    def tile2dGemmKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, M: Int, N: Int, K: Int): Unit = {
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
        var tileK: Int = 0
        while (tileK < numTilesK) {
          val kOffset = tileK * 32
          val aRow = row
          val aCol = kOffset + tx
          val bRow = kOffset + ty
          val bCol = col
          if (aCol < K && bRow < K) {
            val aVal = tl.load(A, aRow * K + aCol)
            val bVal = tl.load(B, bRow * N + bCol)
            tl.sharedStore("s_a", ty * 32 + tx, aVal)
            tl.sharedStore("s_b", ty * 32 + tx, bVal)
          }
          tl.syncthreads()
          var kk: Int = 0
          while (kk < 32) {
            val aVal = tl.sharedLoad("s_a", ty * 32 + kk)
            val bVal = tl.sharedLoad("s_b", kk * 32 + tx)
            sum = sum + aVal * bVal
            kk = kk + 1
          }
          tl.syncthreads()
          tileK = tileK + 1
        }
        tl.store(C, row * N + col, sum)
      }
      ()
    }
    println("[11] tile2dGemmKernelPtr defined")

    // 12. 2D tile attention with threadIdx for 2D Q/K/V
    @TritonKernelMacro
    def tile2dAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int): Unit = {
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
        var j: Int = 0
        while (j < N) {
          val q = tl.load(Q, row * N + j)
          val k = tl.load(K, col * N + j)
          val score = q * k / sqrt(N.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + j)
          val k = tl.load(K, col * N + j)
          val score = q * k / sqrt(N.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + j)
          val k = tl.load(K, col * N + j)
          val score = q * k / sqrt(N.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, col * N + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * N + col, result)
      }
      ()
    }
    println("[12] tile2dAttentionKernelPtr defined")

    // 13. 2D tile layer norm with threadIdx parallel reduction
    @TritonKernelMacro
    def tile2dLayerNormKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int, eps: FloatPtr): Unit = {
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
      val epsVal = eps.get()

      if (row < M && col < N) {
        var sum: Float = 0.0f
        var j: Int = 0
        while (j < N) {
          val v = tl.load(X, row * N + j)
          sum = sum + v
          j = j + 1
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        j = 0
        while (j < N) {
          val v = tl.load(X, row * N + j)
          val diff = v - mean
          sqSum = sqSum + diff * diff
          j = j + 1
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + epsVal)
        val v = tl.load(X, row * N + col)
        val result = (v - mean) * invStd
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[13] tile2dLayerNormKernelPtr defined")

    // 14. 2D tile RMSNorm with threadIdx
    @TritonKernelMacro
    def tile2dRmsNormKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int, eps: FloatPtr): Unit = {
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
      val epsVal = eps.get()

      if (row < M && col < N) {
        var sqSum: Float = 0.0f
        var j: Int = 0
        while (j < N) {
          val v = tl.load(X, row * N + j)
          sqSum = sqSum + v * v
          j = j + 1
        }
        val rms = sqrt(sqSum / N + epsVal)
        val v = tl.load(X, row * N + col)
        val result = v / rms
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[14] tile2dRmsNormKernelPtr defined")

    // 15. 2D tile sigmoid with threadIdx
    @TritonKernelMacro
    def tile2dSigmoidKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
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
        val x = tl.load(X, row * N + col)
        val result = sigmoid(x)
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[15] tile2dSigmoidKernelPtr defined")

    // 16. 2D tile SwiGLU activation
    @TritonKernelMacro
    def tile2dSwigluKernelPtr(X: FloatPtr, G: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
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
        val x = tl.load(X, row * N + col)
        val g = tl.load(G, row * N + col)
        val result = x * sigmoid(x) * g
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[16] tile2dSwigluKernelPtr defined")

    // 17. 2D tile LeakyReLU
    @TritonKernelMacro
    def tile2dLeakyReluKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
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
        val x = tl.load(X, row * N + col)
        val result = leakyrelu(x)
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[17] tile2dLeakyReluKernelPtr defined")

    // 18. 2D tile batch normalization forward
    @TritonKernelMacro
    def tile2dBatchNormKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int, eps: FloatPtr): Unit = {
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
      val epsVal = eps.get()

      if (row < M && col < N) {
        var sum: Float = 0.0f
        var i: Int = 0
        while (i < M) {
          val v = tl.load(X, i * N + col)
          sum = sum + v
          i = i + 1
        }
        val mean = sum / M
        var sqSum: Float = 0.0f
        i = 0
        while (i < M) {
          val v = tl.load(X, i * N + col)
          val diff = v - mean
          sqSum = sqSum + diff * diff
          i = i + 1
        }
        val variance = sqSum / M
        val invStd = 1.0f / sqrt(variance + epsVal)
        val v = tl.load(X, row * N + col)
        val result = (v - mean) * invStd
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[18] tile2dBatchNormKernelPtr defined")

    // 19. 2D tile cross entropy loss
    @TritonKernelMacro
    def tile2dCrossEntropyKernelPtr(logits: FloatPtr, targets: FloatPtr, loss: FloatPtr, M: Int, N: Int): Unit = {
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
        var j: Int = 0
        while (j < N) {
          val v = tl.load(logits, row * N + j)
          maxLogit = max(maxLogit, v)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          val v = tl.load(logits, row * N + j)
          sum = sum + exp(v - maxLogit)
          j = j + 1
        }
        val target = tl.load(targets, row)
        val logitsVal = tl.load(logits, row * N + col)
        val logProb = logitsVal - maxLogit - log(sum)
        val lossVal = -logProb
        tl.store(loss, row, lossVal)
      }
      ()
    }
    println("[19] tile2dCrossEntropyKernelPtr defined")

    // 20. 2D tile dropout forward
    @TritonKernelMacro
    def tile2dDropoutKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int, p: FloatPtr): Unit = {
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
      val pVal = p.get()

      if (row < M && col < N) {
        val x = tl.load(X, row * N + col)
        val scale = 1.0f / (1.0f - pVal)
        val result = x * scale
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[20] tile2dDropoutKernelPtr defined")

    // ========================================================================
    // Category 3: Hierarchical Reduction Trees (10 kernels)
    // ========================================================================
    println("\n[21-30] Hierarchical Reduction Trees with threadIdx")

    // 21. Block-reduction GEMM with warp-level accumulation
    @TritonKernelMacro
    def blockReduceGemmKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, M: Int, N: Int, K: Int): Unit = {
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
          var k: Int = 0
          while (k < K) {
            val aVal = tl.load(A, row * K + k)
            val bVal = tl.load(B, k * N + col)
            partialSum = partialSum + aVal * bVal
            k = k + 1
          }
          if (laneId == 0) {
            tl.store(C, row * N + col, partialSum)
          }
        }
      }
      ()
    }
    println("[21] blockReduceGemmKernelPtr defined")

    // 22. Hierarchical softmax with block reduction
    @TritonKernelMacro
    def hierarchicalSoftmaxKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        var i: Int = 0
        while (i < N) {
          val v = tl.load(X, row * N + i)
          maxVal = max(maxVal, v)
          i = i + 1
        }
        var sum: Float = 0.0f
        i = 0
        while (i < N) {
          val v = tl.load(X, row * N + i)
          sum = sum + exp(v - maxVal)
          i = i + 1
        }
        if (tx < N) {
          val v = tl.load(X, row * N + tx)
          val result = exp(v - maxVal) / sum
          tl.store(Y, row * N + tx, result)
        }
      }
      ()
    }
    println("[22] hierarchicalSoftmaxKernelPtr defined")

    // 23. Block segmented reduction for sparse attention
    @TritonKernelMacro
    def blockSegmentReduceKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int, segSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val segId = tx / segSize
        val idxInSeg = tx % segSize
        if (segId < (N / segSize)) {
          var maxVal: Float = Float.MinValue
          var i: Int = 0
          while (i < segSize) {
            val col = segId * segSize + i
            if (col < N) {
              val v = tl.load(X, row * N + col)
              maxVal = max(maxVal, v)
            }
            i = i + 1
          }
          var sum: Float = 0.0f
          i = 0
          while (i < segSize) {
            val col = segId * segSize + i
            if (col < N) {
              val v = tl.load(X, row * N + col)
              sum = sum + exp(v - maxVal)
            }
            i = i + 1
          }
          if (idxInSeg == 0) {
            tl.store(Y, row * (N / segSize) + segId, maxVal + log(sum))
          }
        }
      }
      ()
    }
    println("[23] blockSegmentReduceKernelPtr defined")

    // 24. Hierarchical layer norm with block reduction
    @TritonKernelMacro
    def hierarchicalLayerNormKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int, eps: FloatPtr): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      val epsVal = eps.get()
      if (tx < N) {
        var sum: Float = 0.0f
        var i: Int = 0
        while (i < N) {
          val v = tl.load(X, row * N + i)
          sum = sum + v
          i = i + 1
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        i = 0
        while (i < N) {
          val v = tl.load(X, row * N + i)
          val diff = v - mean
          sqSum = sqSum + diff * diff
          i = i + 1
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + epsVal)
        if (tx < N) {
          val v = tl.load(X, row * N + tx)
          val result = (v - mean) * invStd
          tl.store(Y, row * N + tx, result)
        }
      }
      ()
    }
    println("[24] hierarchicalLayerNormKernelPtr defined")

    // 25. Multi-level block reduction for batched GEMM
    @TritonKernelMacro
    def multiLevelReduceGemmKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, M: Int, N: Int, K: Int, batch: Int): Unit = {
      val pid = tl.program_id(0)
      val b = pid / (M * N)
      val idx = pid % (M * N)
      val row = idx / N
      val col = idx % N
      if (b >= batch || row >= M || col >= N) return

      val tx = tl.threadIdx(0)
      if (tx == 0) {
        var sum: Float = 0.0f
        var k: Int = 0
        while (k < K) {
          val aVal = tl.load(A, b * M * K + row * K + k)
          val bVal = tl.load(B, b * K * N + k * N + col)
          sum = sum + aVal * bVal
          k = k + 1
        }
        tl.store(C, b * M * N + row * N + col, sum)
      }
      ()
    }
    println("[25] multiLevelReduceGemmKernelPtr defined")

    // 26. Block tree attention with causal masking
    @TritonKernelMacro
    def blockCausalAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val col = tx
        val colLimit = col
        var maxVal: Float = Float.MinValue
        var j: Int = 0
        while (j < colLimit + 1) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < colLimit + 1) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = 0
        while (j < colLimit + 1) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[26] blockCausalAttentionKernelPtr defined")

    // 27. Hierarchical GEMM with block-level blocking
    @TritonKernelMacro
    def hierarchicalGemmKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, M: Int, N: Int, K: Int): Unit = {
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
        var tileK: Int = 0
        while (tileK < numTilesK) {
          var tileSum: Float = 0.0f
          var kk: Int = 0
          while (kk < 8) {
            val aVal = tl.load(A, row * K + tileK * 8 + kk)
            val bVal = tl.load(B, (tileK * 8 + kk) * N + col)
            tileSum = tileSum + aVal * bVal
            kk = kk + 1
          }
          sum = sum + tileSum
          tileK = tileK + 1
        }
        tl.store(C, row * N + col, sum)
      }
      ()
    }
    println("[27] hierarchicalGemmKernelPtr defined")

    // 28. Block-level exclusive scan for prefix sum
    @TritonKernelMacro
    def blockExclusiveScanKernelPtr(X: FloatPtr, Y: FloatPtr, N: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= 1) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var val1: Float = tl.load(X, tx)
        if (tx > 0) {
          var prefix: Float = 0.0f
          var i: Int = 0
          while (i < tx) {
            prefix = prefix + tl.load(X, i)
            i = i + 1
          }
          tl.store(Y, tx, prefix)
        } else {
          tl.store(Y, tx, 0.0f)
        }
      }
      ()
    }
    println("[28] blockExclusiveScanKernelPtr defined")

    // 29. Multi-blocks hierarchical reduction for massive GEMM
    @TritonKernelMacro
    def multiBlockReduceGemmKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, M: Int, N: Int, K: Int): Unit = {
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
          var k: Int = 0
          while (k < K) {
            val aVal = tl.load(A, row * K + k)
            val bVal = tl.load(B, k * N + col)
            sum = sum + aVal * bVal
            k = k + 1
          }
          if (laneId == 0) {
            tl.store(C, row * N + col, sum)
          }
        }
      }
      ()
    }
    println("[29] multiBlockReduceGemmKernelPtr defined")

    // 30. Hierarchical softmax with multiple block levels
    @TritonKernelMacro
    def multiBlockSoftmaxKernelPtr(X: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        var i: Int = 0
        while (i < N) {
          val v = tl.load(X, row * N + i)
          maxVal = max(maxVal, v)
          i = i + 1
        }
        var sum: Float = 0.0f
        i = 0
        while (i < N) {
          val v = tl.load(X, row * N + i)
          sum = sum + exp(v - maxVal)
          i = i + 1
        }
        if (tx < N) {
          val v = tl.load(X, row * N + tx)
          val result = exp(v - maxVal) / sum
          tl.store(Y, row * N + tx, result)
        }
      }
      ()
    }
    println("[30] multiBlockSoftmaxKernelPtr defined")

    // ========================================================================
    // Category 4: Tiled Attention Mechanisms (10 kernels)
    // ========================================================================
    println("\n[31-40] Tiled Attention Mechanisms with threadIdx")

    // 31. Flash-style tiled attention with threadIdx masking
    @TritonKernelMacro
    def flashAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < blockSize) {
        var maxVal: Float = Float.MinValue
        var j: Int = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[31] flashAttentionKernelPtr defined")

    // 32. Multi-head attention with threadIdx for head routing
    @TritonKernelMacro
    def multiHeadAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, numHeads: Int): Unit = {
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
        var j: Int = 0
        while (j < N) {
          val q = tl.load(Q, row * D + headId * headDim + tx)
          val k = tl.load(K, row * D + headId * headDim + j)
          val score = q * k / sqrt(headDim.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * D + headId * headDim + tx)
          val k = tl.load(K, row * D + headId * headDim + j)
          val score = q * k / sqrt(headDim.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * D + headId * headDim + tx)
          val k = tl.load(K, row * D + headId * headDim + j)
          val score = q * k / sqrt(headDim.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * D + headId * headDim + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * D + headId * headDim + tx, result)
      }
      ()
    }
    println("[32] multiHeadAttentionKernelPtr defined")

    // 33. Grouped query attention with threadIdx
    @TritonKernelMacro
    def groupedQueryAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, numHeads: Int, numKVHeads: Int): Unit = {
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
        var j: Int = 0
        while (j < N) {
          val q = tl.load(Q, row * D + headId * headDim + tx)
          val kIdx = j * kvHeadDim / N
          val k = tl.load(K, row * D + kvHeadId * kvHeadDim + kIdx)
          val score = q * k / sqrt(headDim.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * D + headId * headDim + tx)
          val kIdx = j * kvHeadDim / N
          val k = tl.load(K, row * D + kvHeadId * kvHeadDim + kIdx)
          val score = q * k / sqrt(headDim.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * D + headId * headDim + tx)
          val kIdx = j * kvHeadDim / N
          val k = tl.load(K, row * D + kvHeadId * kvHeadDim + kIdx)
          val score = q * k / sqrt(headDim.toFloat)
          val attn = exp(score - maxVal) / sum
          val vIdx = j * kvHeadDim / N
          val v = tl.load(V, row * D + kvHeadId * kvHeadDim + vIdx)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * D + headId * headDim + tx, result)
      }
      ()
    }
    println("[33] groupedQueryAttentionKernelPtr defined")

    // 34. Sliding window attention with threadIdx
    @TritonKernelMacro
    def slidingWindowAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, windowSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val startCol = max(0, row - windowSize).toInt
        val endCol = min(N, row + windowSize + 1).toInt
        var maxVal: Float = Float.MinValue
        var j: Int = startCol
        while (j < endCol) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = startCol
        while (j < endCol) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = startCol
        while (j < endCol) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[34] slidingWindowAttentionKernelPtr defined")

    // 35. Cross attention with threadIdx for encoder-decoder
    @TritonKernelMacro
    def crossAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        var j: Int = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[35] crossAttentionKernelPtr defined")

    // 36. Bidirectional attention with threadIdx
    @TritonKernelMacro
    def bidirectionalAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var fwdMax: Float = Float.MinValue
        var j: Int = 0
        while (j < row + 1) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          fwdMax = max(fwdMax, score)
          j = j + 1
        }
        var fwdSum: Float = 0.0f
        j = 0
        while (j < row + 1) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          fwdSum = fwdSum + exp(score - fwdMax)
          j = j + 1
        }
        var bwdMax: Float = Float.MinValue
        var i: Int = row
        while (i < M) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, i * N + tx)
          val score = q * k / sqrt(D.toFloat)
          bwdMax = max(bwdMax, score)
          i = i + 1
        }
        var bwdSum: Float = 0.0f
        i = row
        while (i < M) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, i * N + tx)
          val score = q * k / sqrt(D.toFloat)
          bwdSum = bwdSum + exp(score - bwdMax)
          i = i + 1
        }
        val maxVal = max(fwdMax, bwdMax)
        val sum = fwdSum * exp(fwdMax - maxVal) + bwdSum * exp(bwdMax - maxVal)
        var result: Float = 0.0f
        j = 0
        while (j < row + 1) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          result = result + attn * v
          j = j + 1
        }
        i = row
        while (i < M) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, i * N + tx)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, i * N + tx)
          result = result + attn * v
          i = i + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[36] bidirectionalAttentionKernelPtr defined")

    // 37. Local attention with threadIdx for sparse computation
    @TritonKernelMacro
    def localAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, localSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        val startCol = ((row / localSize) * localSize).toInt
        val endCol = min(N, startCol + localSize).toInt
        var maxVal: Float = Float.MinValue
        var j: Int = startCol
        while (j < endCol) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = startCol
        while (j < endCol) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        var result: Float = 0.0f
        j = startCol
        while (j < endCol) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[37] localAttentionKernelPtr defined")

    // 38. Sparse attention with threadIdx for pattern masking
    @TritonKernelMacro
    def sparseAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, sparsity: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var maxVal: Float = Float.MinValue
        var j: Int = 0
        while (j < N) {
          if ((j % sparsity) == 0) {
            val q = tl.load(Q, row * N + tx)
            val k = tl.load(K, row * N + j)
            val score = q * k / sqrt(D.toFloat)
            maxVal = max(maxVal, score)
          }
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          if ((j % sparsity) == 0) {
            val q = tl.load(Q, row * N + tx)
            val k = tl.load(K, row * N + j)
            val score = q * k / sqrt(D.toFloat)
            sum = sum + exp(score - maxVal)
          }
          j = j + 1
        }
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          if ((j % sparsity) == 0) {
            val q = tl.load(Q, row * N + tx)
            val k = tl.load(K, row * N + j)
            val score = q * k / sqrt(D.toFloat)
            val attn = exp(score - maxVal) / sum
            val v = tl.load(V, row * N + j)
            result = result + attn * v
          }
          j = j + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[38] sparseAttentionKernelPtr defined")

    // 39. Global-local attention with threadIdx
    @TritonKernelMacro
    def globalLocalAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, localSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < N) {
        var globalMax: Float = Float.MinValue
        var j: Int = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          globalMax = max(globalMax, score)
          j = j + 1
        }
        var globalSum: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          globalSum = globalSum + exp(score - globalMax)
          j = j + 1
        }
        val localStart = ((row / localSize) * localSize).toInt
        val localEnd = min(N, localStart + localSize).toInt
        var localMax: Float = Float.MinValue
        j = localStart
        while (j < localEnd) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          localMax = max(localMax, score)
          j = j + 1
        }
        var localSum: Float = 0.0f
        j = localStart
        while (j < localEnd) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          localSum = localSum + exp(score - localMax)
          j = j + 1
        }
        val maxVal = max(globalMax, localMax)
        val sum = globalSum * exp(globalMax - maxVal) + localSum * exp(localMax - maxVal)
        var result: Float = 0.0f
        j = localStart
        while (j < localEnd) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          result = result + attn * v
          j = j + 1
        }
        tl.store(O, row * N + tx, result)
      }
      ()
    }
    println("[39] globalLocalAttentionKernelPtr defined")

    // 40. Kernel attention with threadIdx for performer-style
    @TritonKernelMacro
    def kernelAttentionKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, numFeatures: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < D) {
        var sum: Float = 0.0f
        var f: Int = 0
        while (f < numFeatures) {
          var qk: Float = 0.0f
          var d: Int = 0
          while (d < D) {
            val q = tl.load(Q, row * D + d)
            val k = tl.load(K, f * D + d)
            qk = qk + q * k
            d = d + 1
          }
          val phi = exp(qk / sqrt(D.toFloat))
          var v: Float = 0.0f
          d = 0
          while (d < D) {
            val kv = tl.load(V, f * D + d)
            v = v + phi * kv
            d = d + 1
          }
          sum = sum + v
          f = f + 1
        }
        tl.store(O, row * D + tx, sum)
      }
      ()
    }
    println("[40] kernelAttentionKernelPtr defined")

    // ========================================================================
    // Category 5: Fused Tiled Operations (10 kernels)
    // ========================================================================
    println("\n[41-50] Fused Tiled Operations with threadIdx")

    // 41. Fused GEMM + GELU with threadIdx
    @TritonKernelMacro
    def fusedGemmGeluKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, M: Int, N: Int, K: Int): Unit = {
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
        var k: Int = 0
        while (k < K) {
          val aVal = tl.load(A, row * K + k)
          val bVal = tl.load(B, k * N + col)
          sum = sum + aVal * bVal
          k = k + 1
        }
        val result = gelu(sum)
        tl.store(C, row * N + col, result)
      }
      ()
    }
    println("[41] fusedGemmGeluKernelPtr defined")

    // 42. Fused GEMM + residual with threadIdx
    @TritonKernelMacro
    def fusedGemmResidualKernelPtr(C: FloatPtr, A: FloatPtr, B: FloatPtr, D: FloatPtr, M: Int, N: Int, K: Int): Unit = {
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
        var k: Int = 0
        while (k < K) {
          val aVal = tl.load(A, row * K + k)
          val bVal = tl.load(B, k * N + col)
          sum = sum + aVal * bVal
          k = k + 1
        }
        val residual = tl.load(D, row * N + col)
        val result = sum + residual
        tl.store(C, row * N + col, result)
      }
      ()
    }
    println("[42] fusedGemmResidualKernelPtr defined")

    // 43. Fused LayerNorm + Attention with threadIdx
    @TritonKernelMacro
    def fusedLayerNormAttentionKernelPtr(X: FloatPtr, Y: FloatPtr, Q: FloatPtr, K: FloatPtr, V: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, eps: FloatPtr): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      val epsVal = eps.get()
      if (tx < D) {
        var sum: Float = 0.0f
        var j: Int = 0
        while (j < N) {
          val v = tl.load(X, row * N + j)
          sum = sum + v
          j = j + 1
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        j = 0
        while (j < N) {
          val v = tl.load(X, row * N + j)
          val diff = v - mean
          sqSum = sqSum + diff * diff
          j = j + 1
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + epsVal)
        val normed = (tl.load(X, row * N + tx) - mean) * invStd
        tl.store(Y, row * N + tx, normed)
        tl.store(Q, row * D + tx, normed)
      }
      ()
    }
    println("[43] fusedLayerNormAttentionKernelPtr defined")

    // 44. Fused RMSNorm + SwiGLU with threadIdx
    @TritonKernelMacro
    def fusedRmsNormSwigluKernelPtr(X: FloatPtr, G: FloatPtr, Y: FloatPtr, M: Int, N: Int, eps: FloatPtr): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      val epsVal = eps.get()
      if (tx < N) {
        var sqSum: Float = 0.0f
        var j: Int = 0
        while (j < N) {
          val v = tl.load(X, row * N + j)
          sqSum = sqSum + v * v
          j = j + 1
        }
        val rms = sqrt(sqSum / N + epsVal)
        val x = tl.load(X, row * N + tx)
        val g = tl.load(G, row * N + tx)
        val normed = x / rms
        val result = normed * sigmoid(normed) * g
        tl.store(Y, row * N + tx, result)
      }
      ()
    }
    println("[44] fusedRmsNormSwigluKernelPtr defined")

    // 45. Fused skip connection with threadIdx
    @TritonKernelMacro
    def fusedSkipConnectionKernelPtr(X: FloatPtr, F: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
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
        val x = tl.load(X, row * N + col)
        val f = tl.load(F, row * N + col)
        val result = x + f
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[45] fusedSkipConnectionKernelPtr defined")

    // 46. Fused gated activation with threadIdx
    @TritonKernelMacro
    def fusedGatedActivationKernelPtr(X: FloatPtr, G1: FloatPtr, G2: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
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
        val x = tl.load(X, row * N + col)
        val g1 = tl.load(G1, row * N + col)
        val g2 = tl.load(G2, row * N + col)
        val result = gelu(x) * sigmoid(g1) * g2
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[46] fusedGatedActivationKernelPtr defined")

    // 47. Fused SwiGLU with threadIdx
    @TritonKernelMacro
    def fusedSwigluKernelPtr(X: FloatPtr, G: FloatPtr, Y: FloatPtr, M: Int, N: Int): Unit = {
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
        val x = tl.load(X, row * N + col)
        val g = tl.load(G, row * N + col)
        val result = x * sigmoid(x) * g
        tl.store(Y, row * N + col, result)
      }
      ()
    }
    println("[47] fusedSwigluKernelPtr defined")

    // 48. Fused MHA + FFN with threadIdx
    @TritonKernelMacro
    def fusedMhaFfnKernelPtr(Q: FloatPtr, K: FloatPtr, V: FloatPtr, W1: FloatPtr, W2: FloatPtr, O: FloatPtr, M: Int, N: Int, D: Int, hidden: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      if (tx < D) {
        var attnOut: Float = 0.0f
        var maxVal: Float = Float.MinValue
        var j: Int = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          maxVal = max(maxVal, score)
          j = j + 1
        }
        var sum: Float = 0.0f
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          sum = sum + exp(score - maxVal)
          j = j + 1
        }
        j = 0
        while (j < N) {
          val q = tl.load(Q, row * N + tx)
          val k = tl.load(K, row * N + j)
          val score = q * k / sqrt(D.toFloat)
          val attn = exp(score - maxVal) / sum
          val v = tl.load(V, row * N + j)
          attnOut = attnOut + attn * v
          j = j + 1
        }
        var ffnOut: Float = 0.0f
        var h: Int = 0
        while (h < hidden) {
          val w = tl.load(W1, tx * hidden + h)
          ffnOut = ffnOut + attnOut * w
          h = h + 1
        }
        ffnOut = gelu(ffnOut)
        var finalOut: Float = 0.0f
        h = 0
        while (h < hidden) {
          val w = tl.load(W2, h * D + tx)
          finalOut = finalOut + ffnOut * w
          h = h + 1
        }
        tl.store(O, row * D + tx, finalOut)
      }
      ()
    }
    println("[48] fusedMhaFfnKernelPtr defined")

    // 49. Fused RMSNorm + skip + GEMM with threadIdx
    @TritonKernelMacro
    def fusedRmsNormSkipGemmKernelPtr(X: FloatPtr, R: FloatPtr, A: FloatPtr, B: FloatPtr, Y: FloatPtr, M: Int, N: Int, K: Int, eps: FloatPtr): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      val epsVal = eps.get()
      if (tx < N) {
        var sqSum: Float = 0.0f
        var j: Int = 0
        while (j < N) {
          val v = tl.load(X, row * N + j)
          sqSum = sqSum + v * v
          j = j + 1
        }
        val rms = sqrt(sqSum / N + epsVal)
        val x = tl.load(X, row * N + tx)
        val normed = x / rms
        val residual = tl.load(R, row * N + tx)
        val combined = normed + residual
        var sum: Float = 0.0f
        var k: Int = 0
        while (k < K) {
          sum = sum + combined * tl.load(A, row * K + k) * tl.load(B, k * N + tx)
          k = k + 1
        }
        tl.store(Y, row * N + tx, sum)
      }
      ()
    }
    println("[49] fusedRmsNormSkipGemmKernelPtr defined")

    // 50. Fused Bias + GELU + Residual + LayerNorm with threadIdx
    @TritonKernelMacro
    def fusedBiasGeluResidualLayerNormKernelPtr(X: FloatPtr, B: FloatPtr, R: FloatPtr, Y: FloatPtr, M: Int, N: Int, eps: FloatPtr): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return

      val tx = tl.threadIdx(0)
      val epsVal = eps.get()
      if (tx < N) {
        val x = tl.load(X, row * N + tx)
        val bias = tl.load(B, tx)
        val residual = tl.load(R, row * N + tx)
        val combined = gelu(x + bias) + residual
        tl.store(Y, row * N + tx, combined)
        var sum: Float = 0.0f
        var j: Int = 0
        while (j < N) {
          sum = sum + tl.load(Y, row * N + j)
          j = j + 1
        }
        val mean = sum / N
        var sqSum: Float = 0.0f
        j = 0
        while (j < N) {
          val v = tl.load(Y, row * N + j)
          sqSum = sqSum + (v - mean) * (v - mean)
          j = j + 1
        }
        val variance = sqSum / N
        val invStd = 1.0f / sqrt(variance + epsVal)
        val finalVal = (tl.load(Y, row * N + tx) - mean) * invStd
        tl.store(Y, row * N + tx, finalVal)
      }
      ()
    }
    println("[50] fusedBiasGeluResidualLayerNormKernelPtr defined")

    // ========================================================================
    println("\n" + "=" * 80)
    println("All 50 threadIdx-based CUDA kernels (pointer variant) defined successfully!")
    println("=" * 80)
  }
}
