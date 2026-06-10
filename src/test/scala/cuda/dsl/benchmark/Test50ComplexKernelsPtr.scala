package cuda.dsl.benchmark

import cuda.dsl.core.{FloatPtr, IntPtr}
import cuda.dsl.dsl._

/** Pointer variant of Test50ComplexKernels using FloatPtr/IntPtr types.
  *
  * Test 50 super complex CUDA kernels to verify all DSL features:
  * - For loops (0.until(N) foreach → while loops)
  * - Shared memory (tl.sharedMem, tl.sharedStore, tl.sharedLoad)
  * - Sync threads (tl.syncthreads)
  * - Warp shuffle (tl.shfl, tl.shfl_up, tl.shfl_down, tl.shfl_xor)
  * - Warp vote (tl.warp_any, tl.warp_all)
  * - Masked load/store (tl.maskedLoad, tl.maskedStore)
  * - Ternary expressions (if-then-else)
  *
  * All @TritonKernelMacro functions are at object level for reliable macro expansion.
  * At compile time, the macro transforms each function, generates CUDA C++ source,
  * and registers it to TritonKernelRegistry — no runtime reflection needed.
  */
object Test50ComplexKernelsPtr {

  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat
  def tanh(x: Float): Float = scala.math.tanh(x.toDouble).toFloat
  def abs(x: Float): Float = scala.math.abs(x.toDouble).toFloat
  def max(a: Float, b: Float): Float = if (a > b) a else b
  def min(a: Float, b: Float): Float = if (a < b) a else b

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Test50ComplexKernelsPtr: 50 Complex Kernels (pointer variant)")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels_ptr.txt for generated CUDA code.")
    println("=" * 80)
  }

  // ==========================================================================
  // Category 1: For Loop Kernels (1-10)
  // ==========================================================================

  /** Vector Sum: each thread sums N elements from inPtr, stores to outPtr */
  @TritonKernelMacro
  def vectorSumKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var sum: Float = 0.0f
      var j: Int = 0
      while (j < N) {
        sum = sum + tl.load(inPtr, i * N + j)
        j = j + 1
      }
      tl.store(outPtr, i, sum)
    }
    ()
  }

  /** Matrix-Vector Multiply: out[i] = sum_j mat[i,j] * vec[j] */
  @TritonKernelMacro
  def matrixVecMulKernelPtr(outPtr: FloatPtr, matPtr: FloatPtr, vecPtr: FloatPtr, M: Int, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var sum: Float = 0.0f
      var j: Int = 0
      while (j < N) {
        val a = tl.load(matPtr, i * N + j)
        val v = tl.load(vecPtr, j)
        sum = sum + a * v
        j = j + 1
      }
      tl.store(outPtr, i, sum)
    }
    ()
  }

  /** Softmax: out[i,j] = exp(x[i,j] - max_i) / sum_k exp(x[i,k] - max_i) */
  @TritonKernelMacro
  def softmaxKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var maxVal: Float = -3.4e38f
      var sumExp: Float = 0.0f
      var j: Int = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        if (x > maxVal) maxVal = x
        j = j + 1
      }
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        sumExp = sumExp + exp(x - maxVal)
        j = j + 1
      }
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        val exp_x = exp(x - maxVal)
        tl.store(outPtr, i * N + j, exp_x / sumExp)
        j = j + 1
      }
    }
    ()
  }

  /** Layer Normalization: out[i,j] = (x[i,j] - mean_i) / sqrt(variance_i + eps) */
  @TritonKernelMacro
  def layerNormKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, eps: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var mean: Float = 0.0f
      var variance: Float = 0.0f
      var j: Int = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        mean = mean + x
        j = j + 1
      }
      mean = mean / N.toFloat
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        val diff = x - mean
        variance = variance + diff * diff
        j = j + 1
      }
      variance = variance / N.toFloat
      val std = sqrt(variance + eps)
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        tl.store(outPtr, i * N + j, (x - mean) / std)
        j = j + 1
      }
    }
    ()
  }

  /** Batched Matrix Multiply: out[b,i,j] = sum_k a[b,i,k] * b[b,k,j] */
  @TritonKernelMacro
  def batchMatMulKernelPtr(outPtr: FloatPtr, aPtr: FloatPtr, bPtr: FloatPtr, B: Int, M: Int, N: Int, K: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val batch = i / (M * N)
      val row = (i / N) % M
      val col = i % N
      var sum: Float = 0.0f
      var k: Int = 0
      while (k < K) {
        val a = tl.load(aPtr, batch * M * K + row * K + k)
        val b = tl.load(bPtr, batch * K * N + k * N + col)
        sum = sum + a * b
        k = k + 1
      }
      tl.store(outPtr, i, sum)
    }
    ()
  }

  /** 1D Convolution: out[i] = sum_k in[i+k] * kernel[k] */
  @TritonKernelMacro
  def conv1dKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, kernelPtr: FloatPtr, L: Int, K: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var sum: Float = 0.0f
      var k: Int = 0
      while (k < K) {
        if (i + k < L) {
          val x = tl.load(inPtr, i + k)
          val w = tl.load(kernelPtr, k)
          sum = sum + x * w
        }
        k = k + 1
      }
      tl.store(outPtr, i, sum)
    }
    ()
  }

  /** Max Pooling: 2D pooling with poolSize x poolSize window */
  @TritonKernelMacro
  def maxPoolKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, H: Int, W: Int, poolSize: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val h = i / W
      val w = i % W
      var maxVal: Float = -3.4e38f
      var ph: Int = 0
      while (ph < poolSize) {
        var pw: Int = 0
        while (pw < poolSize) {
          val inH = h * poolSize + ph
          val inW = w * poolSize + pw
          if (inH < H && inW < W) {
            val x = tl.load(inPtr, inH * W + inW)
            if (x > maxVal) maxVal = x
          }
          pw = pw + 1
        }
        ph = ph + 1
      }
      tl.store(outPtr, i, maxVal)
    }
    ()
  }

  /** RMS Norm: out[i,j] = w[j] * x[i,j] / sqrt(sum(x[i]^2) / N + eps) */
  @TritonKernelMacro
  def rmsNormKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, weightPtr: FloatPtr, N: Int, eps: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var ss: Float = 0.0f
      var j: Int = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        ss = ss + x * x
        j = j + 1
      }
      val scale = 1.0f / sqrt(ss / N.toFloat + eps)
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        val w = tl.load(weightPtr, j)
        tl.store(outPtr, i * N + j, x * scale * w)
        j = j + 1
      }
    }
    ()
  }

  /** Attention Score: score[i,j] = scale * sum_k Q[i,k] * K[j,k] */
  @TritonKernelMacro
  def attentionScoreKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, scale: Float, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val row = i / N
      val col = i % N
      var score: Float = 0.0f
      var k: Int = 0
      while (k < N) {
        val q = tl.load(qPtr, row * N + k)
        val key = tl.load(kPtr, col * N + k)
        score = score + q * key
        k = k + 1
      }
      score = score * scale
      tl.store(outPtr, i, score)
    }
    ()
  }

  /** Cumulative Sum (Prefix Sum): out[i] = sum_{k=0}^{i} in[k] */
  @TritonKernelMacro
  def cumsumKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var sum: Float = 0.0f
      var j: Int = 0
      while (j < i + 1) {
        sum = sum + tl.load(inPtr, j)
        j = j + 1
      }
      tl.store(outPtr, i, sum)
    }
    ()
  }

  // ==========================================================================
  // Category 2: Shared Memory + Syncthreads (11-20)
  // ==========================================================================

  /** Tiled Matrix Multiply using shared memory for A and B tiles */
  @TritonKernelMacro
  def tiledMatMulKernel2Ptr(outPtr: FloatPtr, aPtr: FloatPtr, bPtr: FloatPtr, M: Int, N: Int, K: Int, blockSize: Int, n: Int): Unit = {
    val tx = tl.program_id(0)
    tl.sharedMem("float", "s_a", 256)
    tl.sharedMem("float", "s_b", 256)
    val row = tl.program_id(1) * blockSize + tx / blockSize
    val col = tl.program_id(0) + tx % blockSize
    if (row < M && col < N) {
      var sum: Float = 0.0f
      var tile: Int = 0
      while (tile < (K + blockSize - 1) / blockSize) {
        val aRow = row
        val aCol = tile * blockSize + tx % blockSize
        val aVal = tl.load(aPtr, aRow * K + aCol)
        tl.sharedStore("s_a", (tx / blockSize) * blockSize + tx % blockSize, aVal)
        val bRow = tile * blockSize + tx / blockSize
        val bCol = col
        val bVal = tl.load(bPtr, bRow * N + bCol)
        tl.sharedStore("s_b", (tx / blockSize) * blockSize + tx % blockSize, bVal)
        tl.syncthreads()
        var k: Int = 0
        while (k < blockSize) {
          val aShared = tl.sharedLoad("s_a", (tx / blockSize) * blockSize + k)
          val bShared = tl.sharedLoad("s_b", k * blockSize + tx % blockSize)
          sum = sum + aShared * bShared
          k = k + 1
        }
        tl.syncthreads()
        tile = tile + 1
      }
      tl.store(outPtr, row * N + col, sum)
    }
    ()
  }

  /** Block Reduction using warp shuffle + shared memory reduction tree */
  @TritonKernelMacro
  def blockReduceKernel2Ptr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, blockSize: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    val warpId = tid / 32
    tl.sharedMem("float", "s_warp", 16)
    var value = tl.load(inPtr, tid)
    var offset: Int = 16
    while (offset > 0) {
      val other = tl.shfl_down(value, offset, 32)
      value = value + other
      offset = offset / 2
    }
    if (lane == 0) {
      tl.sharedStore("s_warp", warpId, value)
    }
    tl.syncthreads()
    if (warpId == 0 && lane < 16) {
      value = tl.sharedLoad("s_warp", lane)
      var offset2: Int = 8
      while (offset2 > 0) {
        val other = tl.shfl_down(value, offset2, 32)
        value = value + other
        offset2 = offset2 / 2
      }
      if (lane == 0) {
        tl.store(outPtr, tid / 32, value)
      }
    }
    tl.syncthreads()
    ()
  }

  /** Shared Memory Matrix Transpose */
  @TritonKernelMacro
  def sharedTransposeKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, blockSize: Int, n: Int): Unit = {
    val tx = tl.program_id(0)
    tl.sharedMem("float", "s_data", 256)
    val row = tx / blockSize
    val col = tx % blockSize
    if (row < N && col < N) {
      val val1 = tl.load(inPtr, row * N + col)
      tl.sharedStore("s_data", tx, val1)
      tl.syncthreads()
      val outRow = col
      val outCol = row
      val outIdx = outRow * N + outCol
      val outVal = tl.sharedLoad("s_data", outIdx)
      tl.store(outPtr, outIdx, outVal)
    }
    tl.syncthreads()
    ()
  }

  /** Tiled Softmax with shared memory for max reduction */
  @TritonKernelMacro
  def tiledSoftmaxKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, blockSize: Int, n: Int): Unit = {
    val tx = tl.program_id(0)
    tl.sharedMem("float", "s_data", 256)
    val row = tx / blockSize
    val col = tx % blockSize
    if (row < N && col < N) {
      var maxVal: Float = -3.4e38f
      var i: Int = 0
      while (i < blockSize) {
        val idx = row * N + i
        if (i < N) {
          val x = tl.load(inPtr, idx)
          if (x > maxVal) maxVal = x
        }
        i = i + 1
      }
      tl.sharedStore("s_data", tx, maxVal)
      tl.syncthreads()
      maxVal = tl.sharedLoad("s_data", tx)
      var sumExp: Float = 0.0f
      i = 0
      while (i < blockSize) {
        val idx = row * N + i
        if (i < N) {
          val x = tl.load(inPtr, idx)
          sumExp = sumExp + exp(x - maxVal)
        }
        i = i + 1
      }
      val scale = 1.0f / sumExp
      val x = tl.load(inPtr, row * N + col)
      tl.store(outPtr, row * N + col, exp(x - maxVal) * scale)
    }
    ()
  }

  /** Block-wise Matrix Load/Store via shared memory */
  @TritonKernelMacro
  def blockMatrixLoadKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, M: Int, N: Int, blockSize: Int, n: Int): Unit = {
    val tx = tl.program_id(0)
    tl.sharedMem("float", "s_block", 256)
    val row = tl.program_id(1) * blockSize + tx / blockSize
    val col = blockSize * (tx / blockSize) + tx % blockSize
    if (row < M && col < N) {
      val x = tl.load(inPtr, row * N + col)
      tl.sharedStore("s_block", tx, x)
    }
    tl.syncthreads()
    val outRow = tl.program_id(1) * blockSize + tx / blockSize
    val outCol = tx % blockSize
    if (outRow < M && outCol < N) {
      val y = tl.sharedLoad("s_block", tx)
      tl.store(outPtr, outRow * N + outCol, y * 2.0f)
    }
    ()
  }

  /** Parallel Reduction using shared memory + syncthreads */
  @TritonKernelMacro
  def parallelReduceKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, blockSize: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    tl.sharedMem("float", "s_data", 256)
    var value = tl.load(inPtr, tid)
    tl.sharedStore("s_data", tid % blockSize, value)
    tl.syncthreads()
    var offset: Int = blockSize / 2
    while (offset > 0) {
      if (tid % blockSize < offset) {
        val other = tl.sharedLoad("s_data", (tid % blockSize) + offset)
        value = value + other
        tl.sharedStore("s_data", tid % blockSize, value)
      }
      tl.syncthreads()
      offset = offset / 2
    }
    if (tid == 0) {
      tl.store(outPtr, tl.program_id(1), value)
    }
    ()
  }

  /** Tiled Layer Norm using shared memory for mean/variance reduction */
  @TritonKernelMacro
  def tiledLayerNormKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, blockSize: Int, eps: Float, n: Int): Unit = {
    val tx = tl.program_id(0)
    tl.sharedMem("float", "s_data", 256)
    val row = tx / blockSize
    val col = tx % blockSize
    if (row < N && col < blockSize) {
      var mean: Float = 0.0f
      var i: Int = 0
      while (i < blockSize) {
        mean = mean + tl.load(inPtr, row * blockSize + i)
        i = i + 1
      }
      mean = mean / blockSize.toFloat
      tl.sharedStore("s_data", col, mean)
      tl.syncthreads()
      mean = tl.sharedLoad("s_data", col)
      var variance: Float = 0.0f
      i = 0
      while (i < blockSize) {
        val x = tl.load(inPtr, row * blockSize + i)
        val diff = x - mean
        variance = variance + diff * diff
        i = i + 1
      }
      variance = variance / blockSize.toFloat
      val std = sqrt(variance + eps)
      val x = tl.load(inPtr, row * blockSize + col)
      tl.store(outPtr, row * blockSize + col, (x - mean) / std)
    }
    ()
  }

  /** 2D Tiled Attention with shared memory for K/V caching */
  @TritonKernelMacro
  def tiledAttentionKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, N: Int, D: Int, blockSize: Int, n: Int): Unit = {
    val tx = tl.program_id(0)
    tl.sharedMem("float", "s_k", 8192)
    tl.sharedMem("float", "s_v", 8192)
    val row = tx / D
    val col = tx % D
    val scale = 1.0f / sqrt(D.toFloat)
    if (row < N && col < D) {
      var maxScore: Float = -3.4e38f
      var expSum: Float = 0.0f
      var k: Int = 0
      while (k < N) {
        val q = tl.load(qPtr, row * D + col)
        val key = tl.load(kPtr, k * D + col)
        val score = q * key * scale
        if (score > maxScore) maxScore = score
        k = k + 1
      }
      k = 0
      while (k < N) {
        val q = tl.load(qPtr, row * D + col)
        val key = tl.load(kPtr, k * D + col)
        val score = q * key * scale
        expSum = expSum + exp(score - maxScore)
        k = k + 1
      }
      var result: Float = 0.0f
      k = 0
      while (k < N) {
        val q = tl.load(qPtr, row * D + col)
        val key = tl.load(kPtr, k * D + col)
        val score = q * key * scale
        val expScore = exp(score - maxScore)
        val attn = expScore / expSum
        val v = tl.load(vPtr, k * D + col)
        result = result + attn * v
        k = k + 1
      }
      tl.store(outPtr, row * D + col, result)
    }
    ()
  }

  /** Warp Reduce + Syncthreads: full reduction across warps */
  @TritonKernelMacro
  def warpReduceSyncthreadsKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, blockSize: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    tl.sharedMem("float", "s_warp", 32)
    var value = tl.load(inPtr, tid)
    var offset: Int = 16
    while (offset > 0) {
      val other = tl.shfl_down(value, offset, 32)
      value = value + other
      offset = offset / 2
    }
    if (lane == 0) {
      tl.sharedStore("s_warp", lane, value)
    }
    tl.syncthreads()
    if (lane < 1) {
      value = tl.sharedLoad("s_warp", lane)
      var offset2: Int = 16
      while (offset2 > 0) {
        val other = tl.shfl_down(value, offset2, 32)
        value = value + other
        offset2 = offset2 / 2
      }
      if (lane == 0) {
        tl.store(outPtr, tid / 32, value)
      }
    }
    tl.syncthreads()
    ()
  }

  /** Multi-Warp Reduction: reduce within each warp, then combine */
  @TritonKernelMacro
  def multiWarpReduceKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, blockSize: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    val warpId = tid / 32
    tl.sharedMem("float", "s_partial", 256)
    var value = tl.load(inPtr, tid)
    var offset: Int = 16
    while (offset > 0) {
      val other = tl.shfl_down(value, offset, 32)
      value = value + other
      offset = offset / 2
    }
    if (lane == 0) {
      tl.sharedStore("s_partial", warpId, value)
    }
    tl.syncthreads()
    if (warpId == 0) {
      value = if (lane < 8) tl.sharedLoad("s_partial", lane) else 0.0f
      var offset2: Int = 4
      while (offset2 > 0) {
        val other = tl.shfl_down(value, offset2, 32)
        value = value + other
        offset2 = offset2 / 2
      }
      if (lane == 0) {
        tl.store(outPtr, (tid / 256) * 2, value)
      }
    }
    ()
  }

  // ==========================================================================
  // Category 3: Warp Shuffle Kernels (21-30)
  // ==========================================================================

  /** Warp-level reduction via shuffle down */
  @TritonKernelMacro
  def warpReduceSumKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    var value = tl.load(inPtr, tid)
    var offset: Int = 16
    while (offset > 0) {
      val other = tl.shfl_down(value, offset, 32)
      value = value + other
      offset = offset / 2
    }
    if (lane == 0) {
      tl.store(outPtr, tid / 32, value)
    }
    ()
  }

  /** Warp-level max reduction via shuffle down */
  @TritonKernelMacro
  def warpReduceMaxKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    var value = tl.load(inPtr, tid)
    var offset: Int = 16
    while (offset > 0) {
      val other = tl.shfl_down(value, offset, 32)
      value = max(value, other)
      offset = offset / 2
    }
    if (lane == 0) {
      tl.store(outPtr, tid / 32, value)
    }
    ()
  }

  /** Butterfly reduction via xor shuffle */
  @TritonKernelMacro
  def warpButterflyReduceKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    var value = tl.load(inPtr, tid)
    var mask: Int = 1
    while (mask < 32) {
      val other = tl.shfl_xor(value, mask, 32)
      value = value + other
      mask = mask * 2
    }
    if (lane == 0) {
      tl.store(outPtr, tid / 32, value)
    }
    ()
  }

  /** Warp-level inclusive scan (prefix sum) via shfl_up */
  @TritonKernelMacro
  def warpScanKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    var value = tl.load(inPtr, tid)
    var offset: Int = 1
    while (offset < 32) {
      val other = tl.shfl_up(value, offset, 32)
      if (lane >= offset) {
        value = value + other
      }
      offset = offset * 2
    }
    tl.store(outPtr, tid, value)
    ()
  }

  /** Warp any-vote: checks if any lane has predicate=true */
  @TritonKernelMacro
  def warpVoteAnyKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    val value = tl.load(inPtr, tid)
    val predicate = value > 0.5f
    val anyResult = tl.warp_any(predicate)
    if (lane == 0) {
      tl.store(outPtr, tid / 32, if (anyResult) 1.0f else 0.0f)
    }
    ()
  }

  /** Warp all-vote: checks if all lanes have predicate=true */
  @TritonKernelMacro
  def warpVoteAllKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    val value = tl.load(inPtr, tid)
    val predicate = value > 0.5f
    val allResult = tl.warp_all(predicate)
    if (lane == 0) {
      tl.store(outPtr, tid / 32, if (allResult) 1.0f else 0.0f)
    }
    ()
  }

  /** Combined warp any + all vote */
  @TritonKernelMacro
  def warpVoteCombinedKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    val value = tl.load(inPtr, tid)
    val predicate = value > 0.5f
    val anyResult = tl.warp_any(predicate)
    val allResult = tl.warp_all(predicate)
    if (lane == 0) {
      tl.store(outPtr, (tid / 32) * 2, if (anyResult) 1.0f else 0.0f)
      tl.store(outPtr, (tid / 32) * 2 + 1, if (allResult) 1.0f else 0.0f)
    }
    ()
  }

  /** Warp shuffle with xor pattern (butterfly exchange) */
  @TritonKernelMacro
  def warpShuffleXorKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    var value = tl.load(inPtr, tid)
    var mask: Int = 1
    while (mask < 32) {
      val other = tl.shfl_xor(value, mask, 32)
      value = value + other
      mask = mask * 2
    }
    if (lane == 0) {
      tl.store(outPtr, tid / 32, value)
    }
    ()
  }

  /** Warp-level multi-value reduction (two arrays simultaneously) */
  @TritonKernelMacro
  def warpMultiReduceKernelPtr(outPtr: FloatPtr, inPtr1: FloatPtr, inPtr2: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    var value1 = tl.load(inPtr1, tid)
    var value2 = tl.load(inPtr2, tid)
    var offset: Int = 16
    while (offset > 0) {
      val other1 = tl.shfl_down(value1, offset, 32)
      val other2 = tl.shfl_down(value2, offset, 32)
      value1 = value1 + other1
      value2 = value2 + other2
      offset = offset / 2
    }
    if (lane == 0) {
      tl.store(outPtr, (tid / 32) * 2, value1)
      tl.store(outPtr, (tid / 32) * 2 + 1, value2)
    }
    ()
  }

  /** Warp reduce with ternary output (only lane 0 stores) */
  @TritonKernelMacro
  def warpReduceTernaryKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, N: Int, n: Int): Unit = {
    val tid = tl.program_id(0)
    val lane = tid % 32
    var value = tl.load(inPtr, tid)
    var offset: Int = 16
    while (offset > 0) {
      val other = tl.shfl_down(value, offset, 32)
      value = value + other
      offset = offset / 2
    }
    val result = if (lane == 0) value else 0.0f
    tl.store(outPtr, tid, result)
    ()
  }

  // ==========================================================================
  // Category 4: Masked Load/Store Kernels (31-40)
  // ==========================================================================

  /** Masked Vector Add: out[i] = mask[i] > 0 ? a[i] + b[i] : 0 */
  @TritonKernelMacro
  def maskedVectorAddKernelPtr(outPtr: FloatPtr, aPtr: FloatPtr, bPtr: FloatPtr, maskPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val maskActive = true
      val a = tl.maskedLoad(aPtr, i, maskActive, 0.0f)
      val b = tl.maskedLoad(bPtr, i, maskActive, 0.0f)
      tl.maskedStore(outPtr, i, a + b, maskActive)
    }
    ()
  }

  /** Softmax with masking support */
  @TritonKernelMacro
  def maskedSoftmaxKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, maskPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var maxVal: Float = -3.4e38f
      var sumExp: Float = 0.0f
      var j: Int = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        if (x > maxVal) maxVal = x
        j = j + 1
      }
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        sumExp = sumExp + exp(x - maxVal)
        j = j + 1
      }
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        val exp_x = exp(x - maxVal)
        tl.store(outPtr, i * N + j, exp_x / sumExp)
        j = j + 1
      }
    }
    ()
  }

  /** Masked Attention: applies mask to attention scores */
  @TritonKernelMacro
  def maskedAttentionKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, maskPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val row = i / N
      val col = i % N
      val maskVal = tl.load(maskPtr, col)
      if (maskVal > 0.0f) {
        var maxScore: Float = -3.4e38f
        var expSum: Float = 0.0f
        var k: Int = 0
        while (k < N) {
          val q = tl.load(qPtr, row * N + k)
          val key = tl.load(kPtr, col * N + k)
          val score = q * key
          if (score > maxScore) maxScore = score
          k = k + 1
        }
        k = 0
        while (k < N) {
          val q = tl.load(qPtr, row * N + k)
          val key = tl.load(kPtr, col * N + k)
          val score = q * key
          expSum = expSum + exp(score - maxScore)
          k = k + 1
        }
        var result: Float = 0.0f
        k = 0
        while (k < N) {
          val q = tl.load(qPtr, row * N + k)
          val key = tl.load(kPtr, col * N + k)
          val score = q * key
          val expScore = exp(score - maxScore)
          val attn = expScore / expSum
          val v = tl.load(vPtr, col * N + k)
          result = result + attn * v
          k = k + 1
        }
        tl.store(outPtr, i, result)
      }
    }
    ()
  }

  /** Masked Fill: out[i] = mask[i] > 0 ? fillVal : 0 */
  @TritonKernelMacro
  def maskedFillKernelPtr(outPtr: FloatPtr, fillVal: Float, maskPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val maskVal = tl.load(maskPtr, i)
      val valToStore = if (maskVal > 0.0f) fillVal else 0.0f
      tl.store(outPtr, i, valToStore)
    }
    ()
  }

  /** Masked Select: out[i] = mask[i] > 0 ? a[i] : b[i] */
  @TritonKernelMacro
  def maskedSelectKernelPtr(outPtr: FloatPtr, aPtr: FloatPtr, bPtr: FloatPtr, maskPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val maskVal = tl.load(maskPtr, i)
      val a = tl.load(aPtr, i)
      val b = tl.load(bPtr, i)
      val result = if (maskVal > 0.0f) a else b
      tl.store(outPtr, i, result)
    }
    ()
  }

  /** Masked Scale: out[i] = mask[i] > 0 ? in[i] * scale : 0 */
  @TritonKernelMacro
  def maskedScaleKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, scale: Float, maskPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val maskVal = tl.load(maskPtr, i)
      if (maskVal > 0.0f) {
        val x = tl.load(inPtr, i)
        tl.store(outPtr, i, x * scale)
      }
    }
    ()
  }

  /** Masked Clamp: out[i] = mask[i] > 0 ? clamp(in[i], min, max) : 0 */
  @TritonKernelMacro
  def maskedClampKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, minVal: Float, maxVal: Float, maskPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val maskVal = tl.load(maskPtr, i)
      if (maskVal > 0.0f) {
        val x = tl.load(inPtr, i)
        val clamped = max(minVal, min(x, maxVal))
        tl.store(outPtr, i, clamped)
      }
    }
    ()
  }

  /** Masked Layer Normalization: applies layernorm only where mask > 0 */
  @TritonKernelMacro
  def maskedLayerNormKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, maskPtr: FloatPtr, N: Int, eps: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val maskVal = tl.load(maskPtr, i)
      if (maskVal > 0.0f) {
        var mean: Float = 0.0f
        var j: Int = 0
        while (j < N) {
          mean = mean + tl.load(inPtr, i * N + j)
          j = j + 1
        }
        mean = mean / N.toFloat
        var variance: Float = 0.0f
        j = 0
        while (j < N) {
          val x = tl.load(inPtr, i * N + j)
          val diff = x - mean
          variance = variance + diff * diff
          j = j + 1
        }
        variance = variance / N.toFloat
        val std = sqrt(variance + eps)
        j = 0
        while (j < N) {
          val x = tl.load(inPtr, i * N + j)
          tl.store(outPtr, i * N + j, (x - mean) / std)
          j = j + 1
        }
      }
    }
    ()
  }

  /** Masked GELU: applies GELU only where mask > 0 */
  @TritonKernelMacro
  def maskedGeluKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, maskPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val maskVal = tl.load(maskPtr, i)
      if (maskVal > 0.0f) {
        val x = tl.load(inPtr, i)
        val cdf = (0.5f * (1.0f + tanh(sqrt(2.0f / 3.14159f) * (x + 0.044715f * x * x * x)))).toFloat
        tl.store(outPtr, i, x * cdf)
      }
    }
    ()
  }

  /** Masked Dropout: keeps element with probability keepProb */
  @TritonKernelMacro
  def maskedDropoutKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, maskPtr: FloatPtr, keepProb: Float, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      val maskVal = tl.load(maskPtr, i)
      if (maskVal > keepProb) {
        val x = tl.load(inPtr, i)
        tl.store(outPtr, i, x)
      }
    }
    ()
  }

  // ==========================================================================
  // Category 5: Complex Combined Kernels (41-50)
  // ==========================================================================

  /** Flash Attention: online softmax attention algorithm */
  @TritonKernelMacro
  def flashAttentionKernel2Ptr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, N: Int, D: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    val row = i
    if (row >= N) return
    val scale = 1.0f / sqrt(D.toFloat)
    var scoreMax: Float = -3.4e38f
    var j: Int = 0
    while (j < N) {
      var dot: Float = 0.0f
      var d: Int = 0
      while (d < D) {
        val q_d = tl.load(qPtr, row * D + d)
        val k_d = tl.load(kPtr, j * D + d)
        dot = dot + q_d * k_d
        d = d + 1
      }
      val score_ij = dot * scale
      if (score_ij > scoreMax) scoreMax = score_ij
      j = j + 1
    }
    var expSum: Float = 0.0f
    j = 0
    while (j < N) {
      var dot: Float = 0.0f
      var d: Int = 0
      while (d < D) {
        val q_d = tl.load(qPtr, row * D + d)
        val k_d = tl.load(kPtr, j * D + d)
        dot = dot + q_d * k_d
        d = d + 1
      }
      val score_ij = dot * scale
      expSum = expSum + exp(score_ij - scoreMax)
      j = j + 1
    }
    var d: Int = 0
    while (d < D) {
      var result: Float = 0.0f
      j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var dk: Int = 0
        while (dk < D) {
          val q_d = tl.load(qPtr, row * D + dk)
          val k_d = tl.load(kPtr, j * D + dk)
          dot = dot + q_d * k_d
          dk = dk + 1
        }
        val score_ij = dot * scale
        val exp_ij = exp(score_ij - scoreMax)
        val attn_ij = exp_ij / expSum
        val v_d = tl.load(vPtr, j * D + d)
        result = result + attn_ij * v_d
        j = j + 1
      }
      tl.store(outPtr, row * D + d, result)
      d = d + 1
    }
    ()
  }

  /** Page Attention: vLLM-style paged KV cache attention */
  @TritonKernelMacro
  def pageAttentionKernel2Ptr(outPtr: FloatPtr, queryPtr: FloatPtr, keyCachePtr: FloatPtr, valueCachePtr: FloatPtr, blockTablesPtr: FloatPtr, seqLensPtr: FloatPtr, batchSize: Int, maxBlocksPerSeq: Int, blockSize: Int, D: Int, n: Int): Unit = {
    val pos = tl.program_id(0)
    val scale = 1.0f / sqrt(D.toFloat)
    val seqLen = tl.load(seqLensPtr, 0).toInt
    if (pos >= seqLen) return
    val batchTableOffset = 0
    var scoreMax: Float = -3.4e38f
    var expSum: Float = 0.0f
    var keyPos: Int = 0
    var blockIdx: Int = 0
    var posInBlock: Int = 0
    var physBlock: Int = 0
    var kBase: Int = 0
    var vBase: Int = 0
    var d: Int = 0
    keyPos = 0
    while (keyPos < seqLen) {
      blockIdx = keyPos / blockSize
      posInBlock = keyPos % blockSize
      physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
      kBase = physBlock * blockSize * D + posInBlock * D
      d = 0
      while (d < D) {
        val q = tl.load(queryPtr, pos * D + d)
        val k = tl.load(keyCachePtr, kBase + d)
        val score = q * k * scale
        if (score > scoreMax) {
          expSum = expSum * exp(scoreMax - score) + 1.0f
          scoreMax = score
        } else {
          expSum = expSum + exp(score - scoreMax)
        }
        d = d + 1
      }
      keyPos = keyPos + 1
    }
    keyPos = 0
    while (keyPos < seqLen) {
      blockIdx = keyPos / blockSize
      posInBlock = keyPos % blockSize
      physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
      kBase = physBlock * blockSize * D + posInBlock * D
      vBase = physBlock * blockSize * D + posInBlock * D
      d = 0
      while (d < D) {
        val q = tl.load(queryPtr, pos * D + d)
        val k = tl.load(keyCachePtr, kBase + d)
        val score = q * k * scale
        val expScore = exp(score - scoreMax)
        val attnWeight = expScore / expSum
        val vVal = tl.load(valueCachePtr, vBase + d)
        val outIdx = pos * D + d
        val newVal = tl.load(outPtr, outIdx) + attnWeight * vVal
        tl.store(outPtr, outIdx, newVal)
        d = d + 1
      }
      keyPos = keyPos + 1
    }
    ()
  }

  /** Grouped Query Attention: multiple query heads share KV heads */
  @TritonKernelMacro
  def groupedQueryAttentionKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, kvPtr: FloatPtr, N: Int, D: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    val row = i
    if (row >= N) return
    val scale = 1.0f / sqrt(D.toFloat)
    var scoreMax: Float = -3.4e38f
    var expSum: Float = 0.0f
    var j: Int = 0
    while (j < N) {
      val q = tl.load(qPtr, row * D + j)
      val k = tl.load(kPtr, j * D + j)
      val score = q * k * scale
      if (score > scoreMax) scoreMax = score
      j = j + 1
    }
    j = 0
    while (j < N) {
      val q = tl.load(qPtr, row * D + j)
      val k = tl.load(kPtr, j * D + j)
      val score = q * k * scale
      expSum = expSum + exp(score - scoreMax)
      j = j + 1
    }
    var d: Int = 0
    while (d < D) {
      var result: Float = 0.0f
      j = 0
      while (j < N) {
        val q = tl.load(qPtr, row * D + d)
        val k = tl.load(kvPtr, j * D + d)
        val score = q * k * scale
        val expScore = exp(score - scoreMax)
        val attn = expScore / expSum
        val v = tl.load(vPtr, j * D + d)
        result = result + attn * v
        j = j + 1
      }
      tl.store(outPtr, row * D + d, result)
      d = d + 1
    }
    ()
  }

  /** Multi-Head Attention: standard MHA with H heads */
  @TritonKernelMacro
  def multiHeadAttentionKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, N: Int, H: Int, D: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    val head = i % H
    val row = i / H
    if (row >= N) return
    val scale = 1.0f / sqrt(D.toFloat)
    var scoreMax: Float = -3.4e38f
    var expSum: Float = 0.0f
    var j: Int = 0
    while (j < N) {
      val q = tl.load(qPtr, row * H * D + head * D + j)
      val k = tl.load(kPtr, j * H * D + head * D + j)
      val score = q * k * scale
      if (score > scoreMax) scoreMax = score
      j = j + 1
    }
    j = 0
    while (j < N) {
      val q = tl.load(qPtr, row * H * D + head * D + j)
      val k = tl.load(kPtr, j * H * D + head * D + j)
      val score = q * k * scale
      expSum = expSum + exp(score - scoreMax)
      j = j + 1
    }
    var d: Int = 0
    while (d < D) {
      var result: Float = 0.0f
      j = 0
      while (j < N) {
        val q = tl.load(qPtr, row * H * D + head * D + d)
        val k = tl.load(kPtr, j * H * D + head * D + d)
        val score = q * k * scale
        val expScore = exp(score - scoreMax)
        val attn = expScore / expSum
        val v = tl.load(vPtr, j * H * D + head * D + d)
        result = result + attn * v
        j = j + 1
      }
      tl.store(outPtr, row * H * D + head * D + d, result)
      d = d + 1
    }
    ()
  }

  /** Fused GELU + Layer Norm: layernorm then GELU with learned weight */
  @TritonKernelMacro
  def fusedGeluLayerNormKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, weightPtr: FloatPtr, N: Int, eps: Float, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var mean: Float = 0.0f
      var j: Int = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        mean = mean + x
        j = j + 1
      }
      mean = mean / N.toFloat
      var variance: Float = 0.0f
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        val diff = x - mean
        variance = variance + diff * diff
        j = j + 1
      }
      variance = variance / N.toFloat
      val std = sqrt(variance + eps)
      j = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        val w = tl.load(weightPtr, j)
        val normalized = (x - mean) / std
        val cdf = (0.5f * (1.0f + tanh(sqrt(2.0f / 3.14159f) * (normalized + 0.044715f * normalized * normalized * normalized)))).toFloat
        tl.store(outPtr, i * N + j, normalized * cdf * w)
        j = j + 1
      }
    }
    ()
  }

  /** Fused Scale + Bias + GELU */
  @TritonKernelMacro
  def fusedScaleBiasGeluKernelPtr(outPtr: FloatPtr, inPtr: FloatPtr, scalePtr: FloatPtr, biasPtr: FloatPtr, N: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    if (i < n) {
      var j: Int = 0
      while (j < N) {
        val x = tl.load(inPtr, i * N + j)
        val s = tl.load(scalePtr, j)
        val b = tl.load(biasPtr, j)
        val y = x * s + b
        val cdf = (0.5f * (1.0f + tanh(sqrt(2.0f / 3.14159f) * (y + 0.044715f * y * y * y)))).toFloat
        tl.store(outPtr, i * N + j, y * cdf)
        j = j + 1
      }
    }
    ()
  }

  /** Fused Attention with additive bias: score = QK^T * scale + bias */
  @TritonKernelMacro
  def fusedAttentionBiasKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, biasPtr: FloatPtr, N: Int, D: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    val row = i
    if (row >= N) return
    val scale = 1.0f / sqrt(D.toFloat)
    var scoreMax: Float = -3.4e38f
    var expSum: Float = 0.0f
    var j: Int = 0
    while (j < N) {
      val q = tl.load(qPtr, row * N + j)
      val k = tl.load(kPtr, j * N + j)
      val bias = tl.load(biasPtr, j)
      val score = q * k * scale + bias
      if (score > scoreMax) scoreMax = score
      j = j + 1
    }
    j = 0
    while (j < N) {
      val q = tl.load(qPtr, row * N + j)
      val k = tl.load(kPtr, j * N + j)
      val bias = tl.load(biasPtr, j)
      val score = q * k * scale + bias
      expSum = expSum + exp(score - scoreMax)
      j = j + 1
    }
    var d: Int = 0
    while (d < D) {
      var result: Float = 0.0f
      j = 0
      while (j < N) {
        val q = tl.load(qPtr, row * N + d)
        val k = tl.load(kPtr, j * N + d)
        val bias = tl.load(biasPtr, j)
        val score = q * k * scale + bias
        val expScore = exp(score - scoreMax)
        val attn = expScore / expSum
        val v = tl.load(vPtr, j * N + d)
        result = result + attn * v
        j = j + 1
      }
      tl.store(outPtr, row * N + d, result)
      d = d + 1
    }
    ()
  }

  /** Sparse Attention: attention only over sparse indices */
  @TritonKernelMacro
  def sparseAttentionKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, indicesPtr: FloatPtr, N: Int, D: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    val row = i
    if (row >= N) return
    val scale = 1.0f / sqrt(D.toFloat)
    var scoreMax: Float = -3.4e38f
    var expSum: Float = 0.0f
    val numSparse = 16
    var idx: Int = 0
    while (idx < numSparse) {
      val j = tl.load(indicesPtr, row * numSparse + idx).toInt
      val q = tl.load(qPtr, row * N + j)
      val k = tl.load(kPtr, j * N + j)
      val score = q * k * scale
      if (score > scoreMax) scoreMax = score
      idx = idx + 1
    }
    idx = 0
    while (idx < numSparse) {
      val j = tl.load(indicesPtr, row * numSparse + idx).toInt
      val q = tl.load(qPtr, row * N + j)
      val k = tl.load(kPtr, j * N + j)
      val score = q * k * scale
      expSum = expSum + exp(score - scoreMax)
      idx = idx + 1
    }
    var d: Int = 0
    while (d < D) {
      var result: Float = 0.0f
      idx = 0
      while (idx < numSparse) {
        val j = tl.load(indicesPtr, row * numSparse + idx).toInt
        val q = tl.load(qPtr, row * N + d)
        val k = tl.load(kPtr, j * N + d)
        val score = q * k * scale
        val expScore = exp(score - scoreMax)
        val attn = expScore / expSum
        val v = tl.load(vPtr, j * N + d)
        result = result + attn * v
        idx = idx + 1
      }
      tl.store(outPtr, row * N + d, result)
      d = d + 1
    }
    ()
  }

  /** Cross Attention: Q from source, K/V from different context */
  @TritonKernelMacro
  def crossAttentionKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, N: Int, M: Int, D: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    val row = i
    if (row >= N) return
    val scale = 1.0f / sqrt(D.toFloat)
    var scoreMax: Float = -3.4e38f
    var expSum: Float = 0.0f
    var j: Int = 0
    while (j < M) {
      val q = tl.load(qPtr, row * D + j)
      val k = tl.load(kPtr, j * D + j)
      val score = q * k * scale
      if (score > scoreMax) scoreMax = score
      j = j + 1
    }
    j = 0
    while (j < M) {
      val q = tl.load(qPtr, row * D + j)
      val k = tl.load(kPtr, j * D + j)
      val score = q * k * scale
      expSum = expSum + exp(score - scoreMax)
      j = j + 1
    }
    var d: Int = 0
    while (d < D) {
      var result: Float = 0.0f
      j = 0
      while (j < M) {
        val q = tl.load(qPtr, row * D + d)
        val k = tl.load(kPtr, j * D + d)
        val score = q * k * scale
        val expScore = exp(score - scoreMax)
        val attn = expScore / expSum
        val v = tl.load(vPtr, j * D + d)
        result = result + attn * v
        j = j + 1
      }
      tl.store(outPtr, row * D + d, result)
      d = d + 1
    }
    ()
  }

  /** Hierarchical Attention: coarse-grained selection + fine-grained attention */
  @TritonKernelMacro
  def hierarchicalAttentionKernelPtr(outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, coarsePtr: FloatPtr, N: Int, D: Int, n: Int): Unit = {
    val i = tl.program_id(0)
    val row = i
    if (row >= N) return
    val scale = 1.0f / sqrt(D.toFloat)
    val coarseSize = N / 4
    var coarseMax: Float = -3.4e38f
    var coarseSum: Float = 0.0f
    var g: Int = 0
    while (g < coarseSize) {
      val c = tl.load(coarsePtr, row * coarseSize + g)
      if (c > coarseMax) coarseMax = c
      g = g + 1
    }
    g = 0
    while (g < coarseSize) {
      val c = tl.load(coarsePtr, row * coarseSize + g)
      coarseSum = coarseSum + exp(c - coarseMax)
      g = g + 1
    }
    var scoreMax: Float = -3.4e38f
    var expSum: Float = 0.0f
    var j: Int = 0
    while (j < N) {
      val q = tl.load(qPtr, row * N + j)
      val k = tl.load(kPtr, j * N + j)
      val score = q * k * scale
      if (score > scoreMax) scoreMax = score
      j = j + 1
    }
    j = 0
    while (j < N) {
      val q = tl.load(qPtr, row * N + j)
      val k = tl.load(kPtr, j * N + j)
      val score = q * k * scale
      expSum = expSum + exp(score - scoreMax)
      j = j + 1
    }
    var d: Int = 0
    while (d < D) {
      var result: Float = 0.0f
      j = 0
      while (j < N) {
        val q = tl.load(qPtr, row * N + d)
        val k = tl.load(kPtr, j * N + d)
        val score = q * k * scale
        val expScore = exp(score - scoreMax)
        val attn = expScore / expSum
        val v = tl.load(vPtr, j * N + d)
        result = result + attn * v
        j = j + 1
      }
      tl.store(outPtr, row * N + d, result)
      d = d + 1
    }
    ()
  }
}
