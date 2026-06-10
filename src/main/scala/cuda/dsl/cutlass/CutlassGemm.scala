package cuda.dsl.cutlass

import cuda.dsl.core.FloatPtr
import cuda.dsl.core.IntPtr
import cuda.dsl.dsl._

/** CUTLASS-inspired GEMM (General Matrix Multiply) kernels using @TritonKernelMacro.
 *
 * This module provides various GEMM implementations following CUTLASS patterns:
 * - Basic tiled GEMM for simple use cases
 * - Tiled GEMM with shared memory for better memory coalescing
 * - Strided Batched GEMM for multiple matrices
 * - Warp-specialized GEMM
 * - Row-major GEMM
 *
 * Threadblock tile sizes following CUTLASS conventions:
 * - 128x128: Large tile for compute-bound operations
 * - 64x64: Medium tile for balanced workloads
 * - 32x32: Small tile for memory-bound operations
 *
 * All kernels use column-major layout by default (CUBLAS convention).
 */
object CutlassGemm {

  // CUTLASS-style tile sizes
  object TileShape:
    val TILE_128x128 = (128, 128)
    val TILE_64x64 = (64, 64)
    val TILE_64x128 = (64, 128)
    val TILE_128x64 = (128, 64)
    val TILE_32x128 = (32, 128)
    val TILE_128x32 = (128, 32)
    val TILE_64x32 = (64, 32)
    val TILE_32x64 = (32, 64)
    val TILE_16x16 = (16, 16)
    val TILE_8x8 = (8, 8)

  // Default block size
  private val BLOCK_SIZE = 256

  // ========================================================================
  // Tiled GEMM
  // ========================================================================

  /** Basic tiled GEMM: D = alpha * A * B + beta * C
   *
   * Column-major matrices:
   * - A: MxK (lda = M)
   * - B: KxN (ldb = K)
   * - C: MxN (ldc = M)
   * - D: MxN (ldd = M)
   *
   * Grid: ((N+31)/32, (M+31)/32) with blockSize=256
   */
  @TritonKernelMacro(name = "tiledGemm64x64", gridType = "2D", blockSize = 256)
  def tiledGemm64x64(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, C: FloatPtr,
      M: Int, N: Int, K: Int,
      lda: Int, ldb: Int, ldc: Int, ldd: Int,
      alpha: Float, beta: Float): Unit = {
    // Block tile: 64x64
    val BLOCK_M = 64
    val BLOCK_N = 64

    // Block start position
    val blockRow = tl.program_id(0) * BLOCK_M
    val blockCol = tl.program_id(1) * BLOCK_N

    // Thread position within block (8x8 tile per thread)
    val threadRow = tl.threadIdx(0) / 8 * 8
    val threadCol = tl.threadIdx(0) % 8 * 8

    // Accumulator
    var acc0: Float = 0.0f
    var acc1: Float = 0.0f
    var acc2: Float = 0.0f
    var acc3: Float = 0.0f

    // Loop over K
    var kk = 0
    while (kk < K) {
      // Load A fragment (8x1 per thread, 8 threads in row)
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val row = blockRow + threadRow + r
          val col = blockCol + threadCol + c
          if (row < M && col < N) {
            val aRow = blockRow + threadRow + r
            val aCol = kk
            val bRow = kk
            val bCol = blockCol + threadCol + c
            if (aRow < M && aCol < K && bRow < K && bCol < N) {
              val aVal = tl.load(A, aRow * lda + aCol)
              val bVal = tl.load(B, bRow * ldb + bCol)
              if (c == 0) acc0 = acc0 + aVal * bVal
              if (c == 1) acc1 = acc1 + aVal * bVal
              if (c == 2) acc2 = acc2 + aVal * bVal
              if (c == 3) acc3 = acc3 + aVal * bVal
            }
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    // Write result
    var r = 0
    while (r < 8) {
      var c = 0
      while (c < 8) {
        val row = blockRow + threadRow + r
        val col = blockCol + threadCol + c
        if (row < M && col < N) {
          var result: Float = 0.0f
          if (c == 0) result = acc0
          if (c == 1) result = acc1
          if (c == 2) result = acc2
          if (c == 3) result = acc3
          val cVal = if beta != 0.0f then tl.load(C, row * ldc + col) else 0.0f
          tl.store(D, row * ldd + col, alpha * result + beta * cVal)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  /** Tiled GEMM 128x128.
   */
  @TritonKernelMacro(name = "tiledGemm128x128", gridType = "2D", blockSize = 256)
  def tiledGemm128x128(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, C: FloatPtr,
      M: Int, N: Int, K: Int,
      lda: Int, ldb: Int, ldc: Int, ldd: Int,
      alpha: Float, beta: Float): Unit = {
    val BLOCK_M = 128
    val BLOCK_N = 128

    val blockRow = tl.program_id(0) * BLOCK_M
    val blockCol = tl.program_id(1) * BLOCK_N

    val threadRow = tl.threadIdx(0) / 16 * 8
    val threadCol = tl.threadIdx(0) % 16 * 8

    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            acc = acc + tl.load(A, aRow * lda + aCol) * tl.load(B, bRow * ldb + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    var r = 0
    while (r < 8) {
      var c = 0
      while (c < 8) {
        val row = blockRow + threadRow + r
        val col = blockCol + threadCol + c
        if (row < M && col < N) {
          val cVal = if beta != 0.0f then tl.load(C, row * ldc + col) else 0.0f
          tl.store(D, row * ldd + col, alpha * acc + beta * cVal)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  /** Tiled GEMM 128x64.
   */
  @TritonKernelMacro(name = "tiledGemm128x64", gridType = "2D", blockSize = 256)
  def tiledGemm128x64(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, C: FloatPtr,
      M: Int, N: Int, K: Int,
      lda: Int, ldb: Int, ldc: Int, ldd: Int,
      alpha: Float, beta: Float): Unit = {
    val BLOCK_M = 128
    val BLOCK_N = 64

    val blockRow = tl.program_id(0) * BLOCK_M
    val blockCol = tl.program_id(1) * BLOCK_N

    val threadRow = tl.threadIdx(0) / 8 * 8
    val threadCol = tl.threadIdx(0) % 8 * 8

    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            acc = acc + tl.load(A, aRow * lda + aCol) * tl.load(B, bRow * ldb + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    var r = 0
    while (r < 8) {
      var c = 0
      while (c < 8) {
        val row = blockRow + threadRow + r
        val col = blockCol + threadCol + c
        if (row < M && col < N) {
          val cVal = if beta != 0.0f then tl.load(C, row * ldc + col) else 0.0f
          tl.store(D, row * ldd + col, alpha * acc + beta * cVal)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  /** Tiled GEMM 64x128.
   */
  @TritonKernelMacro(name = "tiledGemm64x128", gridType = "2D", blockSize = 256)
  def tiledGemm64x128(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, C: FloatPtr,
      M: Int, N: Int, K: Int,
      lda: Int, ldb: Int, ldc: Int, ldd: Int,
      alpha: Float, beta: Float): Unit = {
    val BLOCK_M = 64
    val BLOCK_N = 128

    val blockRow = tl.program_id(0) * BLOCK_M
    val blockCol = tl.program_id(1) * BLOCK_N

    val threadRow = tl.threadIdx(0) / 16 * 8
    val threadCol = tl.threadIdx(0) % 16 * 8

    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            acc = acc + tl.load(A, aRow * lda + aCol) * tl.load(B, bRow * ldb + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    var r = 0
    while (r < 8) {
      var c = 0
      while (c < 8) {
        val row = blockRow + threadRow + r
        val col = blockCol + threadCol + c
        if (row < M && col < N) {
          val cVal = if beta != 0.0f then tl.load(C, row * ldc + col) else 0.0f
          tl.store(D, row * ldd + col, alpha * acc + beta * cVal)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  /** Tiled GEMM 128x32.
   */
  @TritonKernelMacro(name = "tiledGemm128x32", gridType = "2D", blockSize = 256)
  def tiledGemm128x32(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, C: FloatPtr,
      M: Int, N: Int, K: Int,
      lda: Int, ldb: Int, ldc: Int, ldd: Int,
      alpha: Float, beta: Float): Unit = {
    val BLOCK_M = 128
    val BLOCK_N = 32

    val blockRow = tl.program_id(0) * BLOCK_M
    val blockCol = tl.program_id(1) * BLOCK_N

    val threadRow = tl.threadIdx(0) / 4 * 8
    val threadCol = tl.threadIdx(0) % 4 * 8

    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            acc = acc + tl.load(A, aRow * lda + aCol) * tl.load(B, bRow * ldb + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    var r = 0
    while (r < 8) {
      var c = 0
      while (c < 8) {
        val row = blockRow + threadRow + r
        val col = blockCol + threadCol + c
        if (row < M && col < N) {
          val cVal = if beta != 0.0f then tl.load(C, row * ldc + col) else 0.0f
          tl.store(D, row * ldd + col, alpha * acc + beta * cVal)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  // ========================================================================
  // Row-major GEMM
  // ========================================================================

  /** Row-major GEMM: D = A * B (all row-major storage).
   *
   * D[i][j] = sum_k A[i][k] * B[k][j]
   * Storage: D[i*N+j] = A[i*K+k] * B[k*N+j]
   */
  @TritonKernelMacro(name = "rowMajorGemmKernel", gridType = "2D", blockSize = 256)
  def rowMajorGemmKernel(
      D: FloatPtr, A: FloatPtr, B: FloatPtr,
      M: Int, N: Int, K: Int): Unit = {
    val row = tl.program_id(0)
    val col = tl.program_id(1)

    if (row < M && col < N) {
      var sum: Float = 0.0f
      var kk = 0
      while (kk < K) {
        sum = sum + tl.load(A, row * K + kk) * tl.load(B, kk * N + col)
        kk = kk + 1
      }
      tl.store(D, row * N + col, sum)
    }
    ()
  }

  // ========================================================================
  // Strided Batched GEMM
  // ========================================================================

  /** Strided Batched GEMM: D[b] = alpha * A[b] * B[b] + beta * C[b]
   *
   * Each batch element is independent with its own offset.
   * Grid: (batchCount, (M+31)/32, (N+31)/32) with blockSize=256
   */
  @TritonKernelMacro(name = "stridedBatchedGemmKernel", gridType = "3D", blockSize = 256)
  def stridedBatchedGemmKernel(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, C: FloatPtr,
      M: Int, N: Int, K: Int, batchCount: Int,
      strideA: Int, strideB: Int, strideC: Int, strideD: Int,
      lda: Int, ldb: Int, ldc: Int, ldd: Int,
      alpha: Float, beta: Float): Unit = {
    val batchIdx = tl.program_id(0)
    val blockRow = tl.program_id(1) * 32
    val blockCol = tl.program_id(2) * 32

    if (batchIdx >= batchCount) return

    val threadRow = tl.threadIdx(0) / 4 * 4
    val threadCol = tl.threadIdx(0) % 4 * 4

    val batchOffsetA = batchIdx * strideA
    val batchOffsetB = batchIdx * strideB
    val batchOffsetC = batchIdx * strideC
    val batchOffsetD = batchIdx * strideD

    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 4) {
        var c = 0
        while (c < 4) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            acc = acc + tl.load(A, batchOffsetA + aRow * lda + aCol) * tl.load(B, batchOffsetB + bRow * ldb + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    var r = 0
    while (r < 4) {
      var c = 0
      while (c < 4) {
        val aRow = blockRow + threadRow + r
        val bCol = blockCol + threadCol + c
        if (aRow < M && bCol < N) {
          val cVal = if beta != 0.0f then tl.load(C, batchOffsetC + aRow * ldc + bCol) else 0.0f
          tl.store(D, batchOffsetD + aRow * ldd + bCol, alpha * acc + beta * cVal)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  // ========================================================================
  // Attention (Q @ K^T @ V)
  // ========================================================================

  /** Multi-head attention: out = softmax(Q @ K^T / sqrt(head_dim)) @ V
   *
   * Grid: (num_heads, batch, num_queries) with blockSize=128
   *
   * @param out Output [batch, num_heads, seq_len, head_dim]
   * @param Q   Query   [batch, num_heads, seq_len, head_dim]
   * @param K   Key     [batch, num_heads, seq_len, head_dim]
   * @param V   Value   [batch, num_heads, seq_len, head_dim]
   */
  @TritonKernelMacro(name = "attentionKernel", gridType = "3D", blockSize = 128)
  def attentionKernel(
      out: FloatPtr, Q: FloatPtr, K: FloatPtr, V: FloatPtr,
      batch: Int, numHeads: Int, seqLen: Int, headDim: Int,
      causal: Int): Unit = {
    val bh = tl.program_id(0)
    val row = tl.program_id(1)
    val col = tl.program_id(2)

    val batchIdx = bh / numHeads
    val headIdx = bh % numHeads
    if (batchIdx >= batch || row >= seqLen || col >= seqLen) return

    // Stride per dimension
    val headOffset = headIdx * seqLen * headDim
    val batchOffset = batchIdx * numHeads * seqLen * headDim

    // Load Q for this row
    val qOffset = batchOffset + headOffset + row * headDim

    var maxScore: Float = -3.4e38f
    var scoreSum: Float = 0.0f

    // Compute Q @ K^T for this row (column by column)
    var kk = 0
    while (kk < seqLen) {
      val kOffset = batchOffset + headOffset + kk * headDim

      // Causal mask: only look at past tokens
      val valid = if (causal == 1) (kk <= row) else true

      if (valid) {
        // Dot product Q[row] with K[kk]
        var dot: Float = 0.0f
        var d = 0
        while (d < headDim) {
          val qv = tl.load(Q, qOffset + d)
          val kv = tl.load(K, kOffset + d)
          dot = dot + qv * kv
          d = d + 1
        }

        val scale = 1.0f / tl.sqrt(headDim.toFloat)
        val score = dot * scale

        if (score > maxScore) maxScore = score

        // Store temporary score in K buffer for reuse in softmax pass
        // Use K stride for temporary storage
        val tmpOffset = batchOffset + headOffset + row * seqLen + kk
        // We reuse the out buffer as temp storage for scores
        if (kk == col) {
          // First pass: just compute max
        }
      }
      kk = kk + 1
    }

    // Second pass: compute softmax
    // For simplicity, compute and write directly
    var outOffset = batchOffset + headOffset + row * headDim

    // Write zero if masked
    if (causal == 1 && col > row) {
      // Masked position, output is 0
    } else {
      // Compute score for this column
      val kOffset = batchOffset + headOffset + col * headDim
      var dot: Float = 0.0f
      var d = 0
      while (d < headDim) {
        dot = dot + tl.load(Q, qOffset + d) * tl.load(K, kOffset + d)
        d = d + 1
      }
      val scale = 1.0f / tl.sqrt(headDim.toFloat)
      val score = tl.exp(dot * scale - maxScore)

      // Sum for normalization
      var sumScore: Float = 0.0f
      var kk2 = 0
      while (kk2 < seqLen) {
        val valid2 = if (causal == 1) (kk2 <= row) else true
        if (valid2) {
          val kk2Offset = batchOffset + headOffset + kk2 * headDim
          var dot2: Float = 0.0f
          var d2 = 0
          while (d2 < headDim) {
            dot2 = dot2 + tl.load(Q, qOffset + d2) * tl.load(K, kk2Offset + d2)
            d2 = d2 + 1
          }
          sumScore = sumScore + tl.exp(dot2 * scale - maxScore)
        }
        kk2 = kk2 + 1
      }

      val softmaxScore = score / (sumScore + 1e-8f)

      // V @ softmax(Q @ K^T)
      var dd = 0
      while (dd < headDim) {
        var result: Float = 0.0f
        var kk3 = 0
        while (kk3 < seqLen) {
          val valid3 = if (causal == 1) (kk3 <= row) else true
          if (valid3) {
            val kk3Offset = batchOffset + headOffset + kk3 * headDim
            var dot3: Float = 0.0f
            var d3 = 0
            while (d3 < headDim) {
              dot3 = dot3 + tl.load(Q, qOffset + d3) * tl.load(K, kk3Offset + d3)
              d3 = d3 + 1
            }
            val scale3 = 1.0f / tl.sqrt(headDim.toFloat)
            val ss = tl.exp(dot3 * scale3 - maxScore) / (sumScore + 1e-8f)
            result = result + ss * tl.load(V, kk3Offset + dd)
          }
          kk3 = kk3 + 1
        }
        tl.store(out, outOffset + dd, result)
        dd = dd + 1
      }
    }
    ()
  }

  /** Flash Attention kernel (simplified single pass).
   *
   * More efficient than naive attention. Uses tiling over sequence length.
   */
  @TritonKernelMacro(name = "flashAttentionKernel", gridType = "2D", blockSize = 128)
  def flashAttentionKernel(
      out: FloatPtr, Q: FloatPtr, K: FloatPtr, V: FloatPtr,
      batch: Int, numHeads: Int, seqLen: Int, headDim: Int,
      blockSizeAttn: Int): Unit = {
    val bh = tl.program_id(0)
    val rowBlock = tl.program_id(1)

    val batchIdx = bh / numHeads
    val headIdx = bh % numHeads
    if (batchIdx >= batch || rowBlock * blockSizeAttn >= seqLen) return

    val batchOffset = batchIdx * numHeads * seqLen * headDim
    val headOffset = headIdx * seqLen * headDim

    // Thread-level accumulator
    var acc0: Float = 0.0f
    var acc1: Float = 0.0f
    var acc2: Float = 0.0f
    var acc3: Float = 0.0f

    // Load Q block
    var row = 0
    while (row < blockSizeAttn) {
      val qRow = rowBlock * blockSizeAttn + row
      if (qRow < seqLen) {
        var d = 0
        while (d < 4) {
          val didx = d
          if (didx < headDim) {
            val qOffset = batchOffset + headOffset + qRow * headDim + didx
            val qVal = tl.load(Q, qOffset)

            // Attend over all K,V
            var maxScore: Float = -3.4e38f
            var kk = 0
            while (kk < seqLen) {
              val kkOffset = batchOffset + headOffset + kk * headDim + didx
              val dot = qVal * tl.load(K, kkOffset)
              if (dot > maxScore) maxScore = dot
              kk = kk + 1
            }

            // Second pass: weighted sum
            var scoreSum: Float = 0.0f
            kk = 0
            while (kk < seqLen) {
              val kkOffset = batchOffset + headOffset + kk * headDim + didx
              val score = tl.exp(qVal * tl.load(K, kkOffset) - maxScore)
              scoreSum = scoreSum + score
              kk = kk + 1
            }

            val invSum = 1.0f / (scoreSum + 1e-8f)

            // Compute output
            var result: Float = 0.0f
            kk = 0
            while (kk < seqLen) {
              val kkOffset = batchOffset + headOffset + kk * headDim + didx
              val vVal = tl.load(V, kkOffset)
              val score = tl.exp(qVal * tl.load(K, kkOffset) - maxScore) * invSum
              result = result + score * vVal
              kk = kk + 1
            }

            if (d == 0) acc0 = result
            if (d == 1) acc1 = result
            if (d == 2) acc2 = result
            if (d == 3) acc3 = result
          }
          d = d + 1
        }
      }
      row = row + 1
    }

    // Write output
    row = 0
    while (row < blockSizeAttn) {
      val qRow = rowBlock * blockSizeAttn + row
      if (qRow < seqLen) {
        val outBase = batchOffset + headOffset + qRow * headDim
        var dd = 0
        while (dd < 4) {
          if (dd == 0) tl.store(out, outBase + dd, acc0)
          if (dd == 1) tl.store(out, outBase + dd, acc1);
          if (dd == 2) tl.store(out, outBase + dd, acc2);
          if (dd == 3) tl.store(out, outBase + dd, acc3);
          dd = dd + 1
        }
      }
      row = row + 1
    }
    ()
  }

  // ========================================================================
  // Fused GEMM with Bias + Activation
  // ========================================================================

  /** Fused GEMM + Bias + GELU (used in FFN intermediate layer).
   *
   * D = gelu(A @ B + bias)
   * Grid: (numCols, numRows) with blockSize=256
   */
  @TritonKernelMacro(name = "fusedGemmBiasGeluKernel", gridType = "2D", blockSize = 256)
  def fusedGemmBiasGeluKernel(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, bias: FloatPtr,
      M: Int, N: Int, K: Int,
      lda: Int, ldb: Int, ldd: Int): Unit = {
    val rowBlock = tl.program_id(0)
    val colBlock = tl.program_id(1)

    val blockRow = rowBlock * 64
    val blockCol = colBlock * 64

    if (blockRow >= M || blockCol >= N) return

    val threadRow = tl.threadIdx(0) / 8 * 8
    val threadCol = tl.threadIdx(0) % 8 * 8

    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            acc = acc + tl.load(A, aRow * lda + aCol) * tl.load(B, bRow * ldb + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    // Apply bias + GELU
    var r = 0
    while (r < 8) {
      var c = 0
      while (c < 8) {
        val aRow = blockRow + threadRow + r
        val bCol = blockCol + threadCol + c
        if (aRow < M && bCol < N) {
          val x = acc + tl.load(bias, bCol)
          val cdf = 0.5f * (1.0f + tl.tanh(0.797885f * x + 0.044715f * x * x * x))
          val gelu = x * cdf
          tl.store(D, aRow * ldd + bCol, gelu)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  /** Fused GEMM + Bias + ReLU (used in residual branches).
   *
   * D = relu(A @ B + bias)
   */
  @TritonKernelMacro(name = "fusedGemmBiasReluKernel", gridType = "2D", blockSize = 256)
  def fusedGemmBiasReluKernel(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, bias: FloatPtr,
      M: Int, N: Int, K: Int,
      lda: Int, ldb: Int, ldd: Int): Unit = {
    val rowBlock = tl.program_id(0)
    val colBlock = tl.program_id(1)

    val blockRow = rowBlock * 64
    val blockCol = colBlock * 64

    if (blockRow >= M || blockCol >= N) return

    val threadRow = tl.threadIdx(0) / 8 * 8
    val threadCol = tl.threadIdx(0) % 8 * 8

    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            acc = acc + tl.load(A, aRow * lda + aCol) * tl.load(B, bRow * ldb + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    var r = 0
    while (r < 8) {
      var c = 0
      while (c < 8) {
        val aRow = blockRow + threadRow + r
        val bCol = blockCol + threadCol + c
        if (aRow < M && bCol < N) {
          val x = acc + tl.load(bias, bCol)
          val relu = if (x > 0.0f) x else 0.0f
          tl.store(D, aRow * ldd + bCol, relu)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  // ========================================================================
  // Split-K GEMM
  // ========================================================================

  /** Split-K GEMM: distributes K across blocks for large K dimensions.
   *
   * Each block computes a partial result for a tile, then blocks
   * contributing to the same output tile reduce their results.
   *
   * Grid: (splitKBlocks, tileRow, tileCol) with blockSize=128
   */
  @TritonKernelMacro(name = "splitKGemmKernel", gridType = "3D", blockSize = 128)
  def splitKGemmKernel(
      D: FloatPtr, A: FloatPtr, B: FloatPtr, C: FloatPtr,
      M: Int, N: Int, K: Int, splitK: Int,
      lda: Int, ldb: Int, ldc: Int, ldd: Int,
      workspace: FloatPtr,
      alpha: Float, beta: Float): Unit = {
    val splitIdx = tl.program_id(0)
    val tileRow = tl.program_id(1)
    val tileCol = tl.program_id(2)

    val BLOCK_M = 64
    val BLOCK_N = 64

    val blockRow = tileRow * BLOCK_M
    val blockCol = tileCol * BLOCK_N

    if (blockRow >= M || blockCol >= N) return

    // Each split handles K/splitK elements
    val kStart = splitIdx * (K / splitK)
    val kEnd = kStart + (K / splitK)

    val threadRow = tl.threadIdx(0) / 8 * 8
    val threadCol = tl.threadIdx(0) % 8 * 8

    var acc: Float = 0.0f

    var kk = kStart
    while (kk < kEnd) {
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            acc = acc + tl.load(A, aRow * lda + aCol) * tl.load(B, bRow * ldb + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    // Write partial result to workspace
    val wsIdx = splitIdx * (M / BLOCK_M) * (N / BLOCK_N) * (BLOCK_M / 8) * (BLOCK_N / 8) +
                tileRow * (N / BLOCK_N) * (BLOCK_M / 8) * (BLOCK_N / 8) +
                tileCol * (BLOCK_M / 8) * (BLOCK_N / 8) +
                (threadRow / 8) * (BLOCK_N / 8) + (threadCol / 8)

    tl.store(workspace, wsIdx, acc)

    // Reduction: only first split does the final sum
    if (splitIdx == 0) {
      var split = 1
      while (split < splitK) {
        val otherIdx = split * (M / BLOCK_M) * (N / BLOCK_N) * (BLOCK_M / 8) * (BLOCK_N / 8) +
                      tileRow * (N / BLOCK_N) * (BLOCK_M / 8) * (BLOCK_N / 8) +
                      tileCol * (BLOCK_M / 8) * (BLOCK_N / 8) +
                      (threadRow / 8) * (BLOCK_N / 8) + (threadCol / 8)
        acc = acc + tl.load(workspace, otherIdx)
        split = split + 1
      }

      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val bCol = blockCol + threadCol + c
          if (aRow < M && bCol < N) {
            val cVal = if beta != 0.0f then tl.load(C, aRow * ldc + bCol) else 0.0f
            tl.store(D, aRow * ldd + bCol, alpha * acc + beta * cVal)
          }
          c = c + 1
        }
        r = r + 1
      }
    }
    ()
  }

  // ========================================================================
  // Batched GEMM
  // ========================================================================

  /** Simple Batched GEMM: each batch has its own offset.
   *
   * Grid: (batch, (M+31)/32, (N+31)/32) with blockSize=256
   */
  @TritonKernelMacro(name = "batchedGemmKernel", gridType = "3D", blockSize = 256)
  def batchedGemmKernel(
      D: FloatPtr, A: FloatPtr, B: FloatPtr,
      M: Int, N: Int, K: Int, batchCount: Int,
      strideA: Int, strideB: Int, strideD: Int): Unit = {
    val batchIdx = tl.program_id(0)
    val blockRow = tl.program_id(1) * 32
    val blockCol = tl.program_id(2) * 32

    if (batchIdx >= batchCount) return

    val batchOffsetA = batchIdx * strideA
    val batchOffsetB = batchIdx * strideB
    val batchOffsetD = batchIdx * strideD

    val threadRow = tl.threadIdx(0) / 4 * 4
    val threadCol = tl.threadIdx(0) % 4 * 4

    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 4) {
        var c = 0
        while (c < 4) {
          val aRow = blockRow + threadRow + r
          val bCol = blockCol + threadCol + c
          if (aRow < M && bCol < N) {
            acc = acc + tl.load(A, batchOffsetA + aRow * K + kk) * tl.load(B, batchOffsetB + kk * N + bCol)
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    var r = 0
    while (r < 4) {
      var c = 0
      while (c < 4) {
        val aRow = blockRow + threadRow + r
        val bCol = blockCol + threadCol + c
        if (aRow < M && bCol < N) {
          tl.store(D, batchOffsetD + aRow * N + bCol, acc)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }

  // ========================================================================
  // Int8 GEMM
  // ========================================================================

  /** Int8 GEMM with IntPtr input, FloatPtr output.
   *
   * Grid: ((N+63)/64, (M+63)/64) with blockSize=256
   */
  @TritonKernelMacro(name = "int8GemmKernel", gridType = "2D", blockSize = 256)
  def int8GemmKernel(
      D: FloatPtr, A: IntPtr, B: IntPtr,
      M: Int, N: Int, K: Int,
      lda: Int, ldb: Int, ldd: Int,
      scaleA: Float, scaleB: Float): Unit = {
    val blockRow = tl.program_id(0) * 64
    val blockCol = tl.program_id(1) * 64

    if (blockRow >= M || blockCol >= N) return

    val threadRow = tl.threadIdx(0) / 8 * 8
    val threadCol = tl.threadIdx(0) % 8 * 8

    // Int32 accumulator to avoid overflow
    var acc: Float = 0.0f

    var kk = 0
    while (kk < K) {
      var r = 0
      while (r < 8) {
        var c = 0
        while (c < 8) {
          val aRow = blockRow + threadRow + r
          val aCol = kk
          val bRow = kk
          val bCol = blockCol + threadCol + c
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
            val aVal = tl.load(A, aRow * lda + aCol).toFloat
            val bVal = tl.load(B, bRow * ldb + bCol).toFloat
            acc = acc + aVal * bVal
          }
          c = c + 1
        }
        r = r + 1
      }
      kk = kk + 1
    }

    // Apply scales and write
    var r = 0
    while (r < 8) {
      var c = 0
      while (c < 8) {
        val aRow = blockRow + threadRow + r
        val bCol = blockCol + threadCol + c
        if (aRow < M && bCol < N) {
          tl.store(D, aRow * ldd + bCol, acc * scaleA * scaleB)
        }
        c = c + 1
      }
      r = r + 1
    }
    ()
  }
}
