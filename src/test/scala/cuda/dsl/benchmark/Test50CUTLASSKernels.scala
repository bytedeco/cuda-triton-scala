package cuda.dsl.benchmark

import cuda.dsl.dsl._

/** Test 50 CUTLASS-style complex CUDA kernels focused on:
  * - Tiled GEMM operations (various block sizes)
  * - CUTLASS persistent kernels
  * - Warp-specialized tiles
  * - Mixed-precision operations
  * - Strided batched operations
  * - CUTLASS-style epilogues
  */
object Test50CUTLASSKernels {

  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat
  def tanh(x: Float): Float = scala.math.tanh(x.toDouble).toFloat
  def abs(x: Float): Float = scala.math.abs(x.toDouble).toFloat
  def max(a: Float, b: Float): Float = if (a > b) a else b
  def min(a: Float, b: Float): Float = if (a < b) a else b
  // Bit shift helper: lshift(a, b) -> a << b (DSL helper for bit shifting in kernel code)
  def lshift(a: Int, b: Int): Int = a << b
  def rshift(a: Int, b: Int): Int = a >> b

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Test50CUTLASSKernels: 50 CUTLASS-style Tiled CUDA Kernels")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels.txt for generated CUDA code.")
    println("=" * 80)

    // ========================================================================
    // Category 1: Tiled GEMM Kernels (10 kernels) - CUTLASS Core
    // ========================================================================
    println("\n[1-10] Tiled GEMM Kernels (CUTLASS Core)")

    // 1. CUTLASS-style Tiled GEMM 64x64 Block
    @TritonKernelMacro
    def tiledGemm64x64Kernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val numBlocksM = (M + 63) / 64
      val numBlocksN = (N + 63) / 64
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockM = pid / numBlocksN
      val blockN = pid % numBlocksN
      val rowStart = blockM * 64
      val colStart = blockN * 64
      val row = pid / numBlocksN
      val col = pid % numBlocksN
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 64 * 64)
        tl.sharedMem("float", "s_b", 64 * 64)
        var sum: Float = 0.0f
        val numTilesK = (K + 63) / 64
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * 64
          0.until(64) foreach { i =>
            0.until(64) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              val bRow = kStart + i
              val bCol = colStart + j
              if (aRow < M && aCol < K && bRow < K && bCol < N) {
                val aVal = tl.load(A + aRow * K + aCol)
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_a", i * 64 + j, aVal)
                tl.sharedStore("s_b", i * 64 + j, bVal)
              }
            }
          }
          tl.syncthreads()
          0.until(64) foreach { kk =>
            0.until(64) foreach { i =>
              0.until(64) foreach { j =>
                val aVal = tl.sharedLoad("s_a", i * 64 + kk)
                val bVal = tl.sharedLoad("s_b", kk * 64 + j)
                sum = sum + aVal * bVal
              }
            }
          }
          tl.syncthreads()
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[1] tiledGemm64x64Kernel defined")

    // 2. CUTLASS-style Tiled GEMM 128x128 Block
    @TritonKernelMacro
    def tiledGemm128x128Kernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val numBlocksM = (M + 127) / 128
      val numBlocksN = (N + 127) / 128
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockM = pid / numBlocksN
      val blockN = pid % numBlocksN
      val rowStart = blockM * 128
      val colStart = blockN * 128
      val row = pid / numBlocksN
      val col = pid % numBlocksN
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 128 * 128)
        tl.sharedMem("float", "s_b", 128 * 128)
        var sum: Float = 0.0f
        val numTilesK = (K + 127) / 128
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * 128
          0.until(128) foreach { i =>
            0.until(128) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              val bRow = kStart + i
              val bCol = colStart + j
              if (aRow < M && aCol < K && bRow < K && bCol < N) {
                val aVal = tl.load(A + aRow * K + aCol)
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_a", i * 128 + j, aVal)
                tl.sharedStore("s_b", i * 128 + j, bVal)
              }
            }
          }
          tl.syncthreads()
          0.until(128) foreach { kk =>
            0.until(128) foreach { i =>
              0.until(128) foreach { j =>
                val aVal = tl.sharedLoad("s_a", i * 128 + kk)
                val bVal = tl.sharedLoad("s_b", kk * 128 + j)
                sum = sum + aVal * bVal
              }
            }
          }
          tl.syncthreads()
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[2] tiledGemm128x128Kernel defined")

    // 3. CUTLASS-style Tiled GEMM 32x32 Block (Warp-level)
    @TritonKernelMacro
    def tiledGemm32x32WarpKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val numBlocksM = (M + 31) / 32
      val numBlocksN = (N + 31) / 32
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockM = pid / numBlocksN
      val blockN = pid % numBlocksN
      val rowStart = blockM * 32
      val colStart = blockN * 32
      val row = pid / numBlocksN
      val col = pid % numBlocksN
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 32 * 32)
        tl.sharedMem("float", "s_b", 32 * 32)
        var sum: Float = 0.0f
        val numTilesK = (K + 31) / 32
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * 32
          0.until(32) foreach { i =>
            0.until(32) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              val bRow = kStart + i
              val bCol = colStart + j
              if (aRow < M && aCol < K && bRow < K && bCol < N) {
                val aVal = tl.load(A + aRow * K + aCol)
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_a", i * 32 + j, aVal)
                tl.sharedStore("s_b", i * 32 + j, bVal)
              }
            }
          }
          tl.syncthreads()
          0.until(32) foreach { kk =>
            0.until(32) foreach { i =>
              0.until(32) foreach { j =>
                val aVal = tl.sharedLoad("s_a", i * 32 + kk)
                val bVal = tl.sharedLoad("s_b", kk * 32 + j)
                sum = sum + aVal * bVal
              }
            }
          }
          tl.syncthreads()
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[3] tiledGemm32x32WarpKernel defined")

    // 4. CUTLASS Persistent GEMM Kernel
    @TritonJit
    @TritonKernelMacro
    def persistentGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int, gridSize: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= gridSize) return
      val tileM = 64
      val tileN = 64
      val tileK = 32
      val numTilesM = (M + tileM - 1) / tileM
      val numTilesN = (N + tileN - 1) / tileN
      val numTilesK = (K + tileK - 1) / tileK
      val blockId = pid
      val blockM = blockId / numTilesN
      val blockN = blockId % numTilesN
      val rowStart = blockM * tileM
      val colStart = blockN * tileN
      val row = rowStart
      val col = colStart
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 4096)
        tl.sharedMem("float", "s_b", 4096)
        var sum: Float = 0.0f
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * tileK
          0.until(tileM) foreach { i =>
            0.until(tileK) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              if (aRow < M && aCol < K) {
                val aVal = tl.load(A + aRow * K + aCol)
                tl.sharedStore("s_a", i * tileK + j, aVal)
              }
            }
          }
          0.until(tileK) foreach { i =>
            0.until(tileN) foreach { j =>
              val bRow = kStart + i
              val bCol = colStart + j
              if (bRow < K && bCol < N) {
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_b", i * tileN + j, bVal)
              }
            }
          }
          tl.syncthreads()
          0.until(tileM) foreach { i =>
            0.until(tileN) foreach { j =>
              0.until(tileK) foreach { kk =>
                val aVal = tl.sharedLoad("s_a", i * tileK + kk)
                val bVal = tl.sharedLoad("s_b", kk * tileN + j)
                sum = sum + aVal * bVal
              }
            }
          }
          tl.syncthreads()
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[4] persistentGemmKernel defined")

    // 5. Tiled GEMM with Bias Addition (CUTLASS Epilogue)
    @TritonKernelMacro
    def tiledGemmBiasKernel(C: Float, A: Float, B: Float, bias: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val numBlocksM = (M + 63) / 64
      val numBlocksN = (N + 63) / 64
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockM = pid / numBlocksN
      val blockN = pid % numBlocksN
      val rowStart = blockM * 64
      val colStart = blockN * 64
      val row = rowStart
      val col = colStart
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 64 * 64)
        tl.sharedMem("float", "s_b", 64 * 64)
        var sum: Float = 0.0f
        val numTilesK = (K + 63) / 64
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * 64
          0.until(64) foreach { i =>
            0.until(64) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              val bRow = kStart + i
              val bCol = colStart + j
              if (aRow < M && aCol < K && bRow < K && bCol < N) {
                val aVal = tl.load(A + aRow * K + aCol)
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_a", i * 64 + j, aVal)
                tl.sharedStore("s_b", i * 64 + j, bVal)
              }
            }
          }
          tl.syncthreads()
          0.until(64) foreach { kk =>
            0.until(64) foreach { i =>
              0.until(64) foreach { j =>
                val aVal = tl.sharedLoad("s_a", i * 64 + kk)
                val bVal = tl.sharedLoad("s_b", kk * 64 + j)
                sum = sum + aVal * bVal
              }
            }
          }
          tl.syncthreads()
        }
        val biasVal = tl.load(bias + col)
        tl.store(C + row * N + col, sum + biasVal)
      }
      ()
    }
    println("[5] tiledGemmBiasKernel defined")

    // 6. Tiled GEMM with GELU Activation (CUTLASS Epilogue)
    @TritonKernelMacro
    def tiledGemmGeluKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val numBlocksM = (M + 63) / 64
      val numBlocksN = (N + 63) / 64
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockM = pid / numBlocksN
      val blockN = pid % numBlocksN
      val rowStart = blockM * 64
      val colStart = blockN * 64
      val row = rowStart
      val col = colStart
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 64 * 64)
        tl.sharedMem("float", "s_b", 64 * 64)
        var sum: Float = 0.0f
        val numTilesK = (K + 63) / 64
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * 64
          0.until(64) foreach { i =>
            0.until(64) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              val bRow = kStart + i
              val bCol = colStart + j
              if (aRow < M && aCol < K && bRow < K && bCol < N) {
                val aVal = tl.load(A + aRow * K + aCol)
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_a", i * 64 + j, aVal)
                tl.sharedStore("s_b", i * 64 + j, bVal)
              }
            }
          }
          tl.syncthreads()
          0.until(64) foreach { kk =>
            0.until(64) foreach { i =>
              0.until(64) foreach { j =>
                val aVal = tl.sharedLoad("s_a", i * 64 + kk)
                val bVal = tl.sharedLoad("s_b", kk * 64 + j)
                sum = sum + aVal * bVal
              }
            }
          }
          tl.syncthreads()
        }
        val x = sum
        val cdf = 0.5f * (1.0f + tanh(sqrt(2.0f / 3.14159f) * (x + 0.044715f * x * x * x)))
        tl.store(C + row * N + col, cdf)
      }
      ()
    }
    println("[6] tiledGemmGeluKernel defined")

    // 7. Strided Batched GEMM (CUTLASS Batch)
    @TritonKernelMacro
    def stridedBatchedGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int, batchStrideA: Int, batchStrideB: Int, batchStrideC: Int, batchCount: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= batchCount * M * N) return
      val batchIdx = pid / (M * N)
      val row = (pid % (M * N)) / N
      val col = pid % N
      if (row < M && col < N) {
        var sum: Float = 0.0f
        0.until(K) foreach { k =>
          val aVal = tl.load(A + batchIdx * batchStrideA + row * K + k)
          val bVal = tl.load(B + batchIdx * batchStrideB + k * N + col)
          sum = sum + aVal * bVal
        }
        tl.store(C + batchIdx * batchStrideC + row * N + col, sum)
      }
      ()
    }
    println("[7] stridedBatchedGemmKernel defined")

    // 8. Quantized INT8 GEMM (CUTLASS Quantization)
    @TritonKernelMacro
    def quantizedGemmKernel(C: Float, A: Float, B: Float, scaleA: Float, scaleB: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= M * N) return
      val row = pid / N
      val col = pid % N
      if (row < M && col < N) {
        var sum: Float = 0.0f
        0.until(K) foreach { k =>
          val aVal = tl.load(A + row * K + k)
          val bVal = tl.load(B + k * N + col)
          val sa = tl.load(scaleA + row)
          val sb = tl.load(scaleB + col)
          sum = sum + aVal * sa * bVal * sb
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[8] quantizedGemmKernel defined")

    // 9. Tiled Sparse GEMM (CUTLASS Sparse)
    @TritonKernelMacro
    def sparseGemmKernel(C: Float, A: Float, B: Float, metadata: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val numBlocksM = (M + 63) / 64
      val numBlocksN = (N + 63) / 64
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockM = pid / numBlocksN
      val blockN = pid % numBlocksN
      val rowStart = blockM * 64
      val colStart = blockN * 64
      val row = rowStart
      val col = colStart
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 64 * 64)
        tl.sharedMem("float", "s_b", 64 * 64)
        tl.sharedMem("float", "s_meta", 64 * 8)
        var sum: Float = 0.0f
        val numTilesK = (K + 63) / 64
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * 64
          val metaIdx = (pid % 64) / 8
          val metaVal = tl.load(metadata + blockM * 8 * numTilesK + tileK * 8 + metaIdx)
          0.until(64) foreach { i =>
            0.until(64) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              val bRow = kStart + i
              val bCol = colStart + j
              if (aRow < M && aCol < K && bRow < K && bCol < N) {
                val aVal = tl.load(A + aRow * K + aCol)
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_a", i * 64 + j, aVal)
                tl.sharedStore("s_b", i * 64 + j, bVal)
              }
            }
          }
          if (metaVal > 0.0f) {
            tl.syncthreads()
            0.until(64) foreach { kk =>
              0.until(64) foreach { i =>
                0.until(64) foreach { j =>
                  val aVal = tl.sharedLoad("s_a", i * 64 + kk)
                  val bVal = tl.sharedLoad("s_b", kk * 64 + j)
                  sum = sum + aVal * bVal
                }
              }
            }
            tl.syncthreads()
          }
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[9] sparseGemmKernel defined")

    // 10. Mixed Precision GEMM (CUTLASS Mixed Precision)
    @TritonKernelMacro
    def mixedPrecisionGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= M * N) return
      val row = pid / N
      val col = pid % N
      if (row < M && col < N) {
        var sum: Float = 0.0f
        0.until(K) foreach { k =>
          val aVal = tl.load(A + row * K + k)
          val bVal = tl.load(B + k * N + col)
          sum = sum + aVal * bVal
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[10] mixedPrecisionGemmKernel defined")

    // ========================================================================
    // Category 2: Warp-Level Tiled Operations (10 kernels)
    // ========================================================================
    println("\n[11-20] Warp-Level Tiled Operations")

    // 11. Warp-Level Matrix Multiply 16x16
    @TritonKernelMacro
    def warpGemm16x16Kernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= M * N) return
      val row = pid / N
      val col = pid % N
      if (row < M && col < N) {
        var sum: Float = 0.0f
        0.until(K) foreach { k =>
          val aVal = tl.load(A + row * K + k)
          val bVal = tl.load(B + k * N + col)
          sum = sum + aVal * bVal
        }
        tl.store(C + row * N + col, sum)
      }
      ()
    }
    println("[11] warpGemm16x16Kernel defined")

    // 12. Warp-Level Tiled Reduce
    @TritonKernelMacro
    def warpTiledReduceKernel(out: Float, in: Float, N: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val tid = pid
      val lane = tid % 32
      val warpId = tid / 32
      val numWarps = blockSize / 32
      if (pid < N) {
        tl.sharedMem("float", "s_partial", 256)
        var value = tl.load(in + pid)
        0.until(5) foreach { stage =>
          val offset = lshift(1, stage)
          if (lane >= offset) {
            val other = tl.shfl_up(value, offset, 32)
            value = value + other
          }
        }
        if (lane == 0) {
          tl.sharedStore("s_partial", warpId * 32, value)
        }
      }
      tl.syncthreads()
      if (tid < 32) {
        var value = if (warpId == 0) tl.sharedLoad("s_partial", tid * 32) else 0.0f
        var sum = value
        0.until(5) foreach { i =>
          val offset = lshift(1, i)
          if (offset <= 16) {
            val other = tl.shfl_xor(sum, offset, 32)
            sum = sum + other
          }
        }
        if (tid == 0) {
          tl.store(out + pid, sum)
        }
      }
      ()
    }
    println("[12] warpTiledReduceKernel defined")

    // 13. Warp-Level Tiled Scan
    @TritonKernelMacro
    def warpTiledScanKernel(out: Float, in: Float, N: Int): Unit = {
      val pid = tl.program_id(0)
      val lane = pid % 32
      if (pid >= N) return
      var value = tl.load(in + pid)
      0.until(5) foreach { stage =>
        val offset = lshift(1, stage)
        if (lane >= offset) {
          val other = tl.shfl_up(value, offset, 32)
          value = value + other
        }
      }
      tl.store(out + pid, value)
      ()
    }
    println("[13] warpTiledScanKernel defined")

    // 14. Warp-Level Tiled Softmax
    @TritonKernelMacro
    def warpTiledSoftmaxKernel(out: Float, in: Float, N: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val tid = pid
      val lane = tid % 32
      val warpId = tid / 32
      val numWarps = blockSize / 32
      val tileSize = numWarps * 32
      tl.sharedMem("float", "s_max", 64)
      tl.sharedMem("float", "s_sum", 64)
      val tileStart = warpId * 32
      if (pid < N) {
        var maxVal = -3.4e38f
        tileStart.until(tileStart + 32) foreach { i =>
          if (i < N) {
            val v = tl.load(in + i)
            if (v > maxVal) maxVal = v
          }
        }
        var warpMax = maxVal
        0.until(5) foreach { stage =>
          val offset = lshift(1, stage)
          if (offset <= 16) {
            val other = tl.shfl_down(warpMax, offset, 32)
            warpMax = if (warpMax > other) warpMax else other
          }
        }
        if (lane == 0) {
          tl.sharedStore("s_max", warpId * 32, warpMax)
        }
      }
      tl.syncthreads()
      var blockMax = -3.4e38f
      if (tid < numWarps) {
        blockMax = tl.sharedLoad("s_max", tid * 32)
      }
      if (pid < N) {
        var sumVal = 0.0f
        tileStart.until(tileStart + 32) foreach { i =>
          if (i < N) {
            val v = tl.load(in + i)
            sumVal = sumVal + exp(v - blockMax)
          }
        }
        var warpSum = sumVal
        0.until(5) foreach { stage =>
          val offset = lshift(1, stage)
          if (offset <= 16) {
            val other = tl.shfl_up(warpSum, offset, 32)
            warpSum = warpSum + other
          }
        }
        if (lane == 0) {
          tl.sharedStore("s_sum", warpId * 32, warpSum)
        }
      }
      tl.syncthreads()
      var blockSum = 0.0f
      if (tid < numWarps) {
        blockSum = tl.sharedLoad("s_sum", tid * 32)
      }
      if (pid < N) {
        tileStart.until(tileStart + 32) foreach { i =>
          if (i < N) {
            val v = tl.load(in + i)
            val softmax = exp(v - blockMax) / blockSum
            tl.store(out + i, softmax)
          }
        }
      }
      ()
    }
    println("[14] warpTiledSoftmaxKernel defined")

    // 15. Warp-Level Tiled LayerNorm
    @TritonKernelMacro
    def warpTiledLayerNormKernel(out: Float, in: Float, weight: Float, bias: Float, N: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val tid = pid
      val lane = tid % 32
      val warpId = tid / 32
      val numWarps = blockSize / 32
      tl.sharedMem("float", "s_mean", 32)
      tl.sharedMem("float", "s_var", 32)
      val row = pid
      if (row < N) {
        var sum = 0.0f
        0.until(N) foreach { i =>
          val v = tl.load(in + row * N + i)
          sum = sum + v
        }
        var warpSum = sum
        0.until(5) foreach { stage =>
          val offset = lshift(1, stage)
          if (offset <= 16) {
            val other = tl.shfl_up(warpSum, offset, 32)
            warpSum = warpSum + other
          }
        }
        val mean = warpSum / N.toFloat
        if (lane == 0) {
          tl.sharedStore("s_mean", warpId, mean)
        }
      }
      tl.syncthreads()
      var blockMean = 0.0f
      if (tid < numWarps) {
        blockMean = tl.sharedLoad("s_mean", tid)
      }
      if (row < N) {
        var varSum = 0.0f
        0.until(N) foreach { i =>
          val v = tl.load(in + row * N + i)
          val diff = v - blockMean
          varSum = varSum + diff * diff
        }
        var warpVar = varSum
        0.until(5) foreach { stage =>
          val offset = lshift(1, stage)
          if (offset <= 16) {
            val other = tl.shfl_up(warpVar, offset, 32)
            warpVar = warpVar + other
          }
        }
        val variance = warpVar / N.toFloat
        val std = sqrt(variance + 1e-5f)
        if (lane == 0) {
          tl.sharedStore("s_var", warpId, std)
        }
      }
      tl.syncthreads()
      var blockStd = 1.0f
      if (tid < numWarps) {
        blockStd = tl.sharedLoad("s_var", tid)
      }
      if (row < N) {
        0.until(N) foreach { i =>
          val v = tl.load(in + row * N + i)
          val w = tl.load(weight + i)
          val b = tl.load(bias + i)
          val norm = ((v - blockMean) / blockStd) * w + b
          tl.store(out + row * N + i, norm)
        }
      }
      ()
    }
    println("[15] warpTiledLayerNormKernel defined")

    // 16. Warp-Level Tiled Attention
    @TritonKernelMacro
    def warpTiledAttentionKernel(out: Float, q: Float, k: Float, v: Float, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      val scale = 1.0f / sqrt(D.toFloat)
      tl.sharedMem("float", "s_q", 64)
      tl.sharedMem("float", "s_k", 64)
      tl.sharedMem("float", "s_v", 64)
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val qVal = tl.load(q + row * D + d)
          val kVal = tl.load(k + j * D + d)
          val score = qVal * kVal * scale
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * D + d)
          result = result + attn * vVal
        }
        tl.store(out + row * D + d, result)
      }
      ()
    }
    println("[16] warpTiledAttentionKernel defined")

    // 17. Warp-Level Tiled Conv2d (CUTLASS Style)
    @TritonKernelMacro
    def warpTiledConv2dKernel(out: Float, in: Float, kernel: Float, N: Int, H: Int, W: Int, K: Int, R: Int, S: Int): Unit = {
      val pid = tl.program_id(0)
      val batchIdx = pid / (H * W)
      val h = (pid / W) % H
      val w = pid % W
      if (batchIdx < N && h < H && w < W) {
        var acc: Float = 0.0f
        0.until(K) foreach { k =>
          0.until(R) foreach { r =>
            0.until(S) foreach { s =>
              val hIn = h + r - 1
              val wIn = w + s - 1
              if (hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
                val inVal = tl.load(in + batchIdx * H * W + hIn * W + wIn)
                val kVal = tl.load(kernel + k * R * S + r * S + s)
                acc = acc + inVal * kVal
              }
            }
          }
        }
        tl.store(out + batchIdx * H * W + h * W + w, acc)
      }
      ()
    }
    println("[17] warpTiledConv2dKernel defined")

    // 18. Warp-Level Tiled RMSNorm
    @TritonKernelMacro
    def warpTiledRmsNormKernel(out: Float, in: Float, weight: Float, N: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val lane = pid % 32
      if (pid >= N) return
      var sumSq = 0.0f
      0.until(N) foreach { i =>
        val v = tl.load(in + pid * N + i)
        sumSq = sumSq + v * v
      }
      var warpSumSq = sumSq
      0.until(5) foreach { stage =>
        val offset = lshift(1, stage)
        if (offset <= 16) {
          val other = tl.shfl_up(warpSumSq, offset, 32)
          warpSumSq = warpSumSq + other
        }
      }
      val rms = sqrt(warpSumSq / N.toFloat + 1e-5f)
      0.until(N) foreach { i =>
        val v = tl.load(in + pid * N + i)
        val w = tl.load(weight + i)
        val norm = (v / rms) * w
        tl.store(out + pid * N + i, norm)
      }
      ()
    }
    println("[18] warpTiledRmsNormKernel defined")

    // 19. Warp-Level Tiled GELU
    @TritonKernelMacro
    def warpTiledGeluKernel(out: Float, in: Float, N: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= N) return
      val v = tl.load(in + pid)
      val cdf = 0.5f * (1.0f + tanh(sqrt(2.0f / 3.14159f) * (v + 0.044715f * v * v * v)))
      tl.store(out + pid, cdf)
      ()
    }
    println("[19] warpTiledGeluKernel defined")

    // 20. Warp-Level Tiled Dropout
    @TritonKernelMacro
    def warpTiledDropoutKernel(out: Float, in: Float, mask: Float, keepProb: Float, N: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= N) return
      val v = tl.load(in + pid)
      val m = tl.load(mask + pid)
      val result = if (m > keepProb) v else 0.0f
      tl.store(out + pid, result)
      ()
    }
    println("[20] warpTiledDropoutKernel defined")

    // ========================================================================
    // Category 3: CUTLASS Persistent Kernels (10 kernels)
    // ========================================================================
    println("\n[21-30] CUTLASS Persistent Kernels")

    // 21. CUTLASS Persistent Softmax
    @TritonKernelMacro
    def persistentSoftmaxKernel(out: Float, in: Float, N: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= nTiles) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = pid * tileSize
      val end = start + tileSize
      var maxVal = -3.4e38f
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          if (v > maxVal) maxVal = v
        }
      }
      var sumExp = 0.0f
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          sumExp = sumExp + exp(v - maxVal)
        }
      }
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          val softmax = exp(v - maxVal) / sumExp
          tl.store(out + i, softmax)
        }
      }
      ()
    }
    println("[21] persistentSoftmaxKernel defined")

    // 22. CUTLASS Persistent LayerNorm
    @TritonKernelMacro
    def persistentLayerNormKernel(out: Float, in: Float, weight: Float, bias: Float, N: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= nTiles) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = pid * tileSize
      val end = start + tileSize
      var sum = 0.0f
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          sum = sum + v
        }
      }
      val mean = sum / N.toFloat
      var varSum = 0.0f
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          val diff = v - mean
          varSum = varSum + diff * diff
        }
      }
      val variance = varSum / N.toFloat
      val std = sqrt(variance + 1e-5f)
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          val w = tl.load(weight + i)
          val b = tl.load(bias + i)
          val norm = ((v - mean) / std) * w + b
          tl.store(out + i, norm)
        }
      }
      ()
    }
    println("[22] persistentLayerNormKernel defined")

    // 23. CUTLASS Persistent Attention
    @TritonKernelMacro
    def persistentAttentionKernel(out: Float, q: Float, k: Float, v: Float, scale: Float, M: Int, N: Int, D: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      val rowTile = pid / nTiles
      val tileIdx = pid % nTiles
      if (rowTile >= M) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = tileIdx * tileSize
      val end = start + tileSize
      val row = rowTile
      var maxScore = -3.4e38f
      start.until(end) foreach { j =>
        if (j < N) {
          val qVal = tl.load(q + row * D + j)
          val kVal = tl.load(k + j * D + j)
          val score = qVal * kVal * scale
          if (score > maxScore) maxScore = score
        }
      }
      var sumExp = 0.0f
      start.until(end) foreach { j =>
        if (j < N) {
          val qVal = tl.load(q + row * D + j)
          val kVal = tl.load(k + j * D + j)
          val score = qVal * kVal * scale
          sumExp = sumExp + exp(score - maxScore)
        }
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        start.until(end) foreach { j =>
          if (j < N) {
            val qVal = tl.load(q + row * D + d)
            val kVal = tl.load(k + j * D + d)
            val score = qVal * kVal * scale
            val attn = exp(score - maxScore) / sumExp
            val vVal = tl.load(v + j * D + d)
            result = result + attn * vVal
          }
        }
        tl.store(out + row * D + d, result)
      }
      ()
    }
    println("[23] persistentAttentionKernel defined")

    // 24. CUTLASS Persistent GEMV (Matrix-Vector Product)
    @TritonKernelMacro
    def persistentGemvKernel(y: Float, A: Float, x: Float, M: Int, N: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      val rowTile = pid / nTiles
      val tileIdx = pid % nTiles
      if (rowTile >= M) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = tileIdx * tileSize
      val end = start + tileSize
      val row = rowTile
      var acc: Float = 0.0f
      start.until(end) foreach { j =>
        if (j < N) {
          val aVal = tl.load(A + row * N + j)
          val xVal = tl.load(x + j)
          acc = acc + aVal * xVal
        }
      }
      tl.store(y + row, acc)
      ()
    }
    println("[24] persistentGemvKernel defined")

    // 25. CUTLASS Persistent Batch GEMM
    @TritonKernelMacro
    def persistentBatchGemmKernel(C: Float, A: Float, B: Float, M: Int, N: Int, K: Int, batchCount: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      val batchTile = pid / nTiles
      val tileIdx = pid % nTiles
      if (batchTile >= batchCount) return
      val tileSizeM = (M + nTiles - 1) / nTiles
      val startM = batchTile * tileSizeM
      val endM = startM + tileSizeM
      val tileSizeN = (N + nTiles - 1) / nTiles
      val startN = tileIdx * tileSizeN
      val endN = startN + tileSizeN
      val batch = batchTile
      startM.until(endM) foreach { i =>
        if (i < M) {
          startN.until(endN) foreach { j =>
            if (j < N) {
              var acc: Float = 0.0f
              0.until(K) foreach { k =>
                val aVal = tl.load(A + batch * M * K + i * K + k)
                val bVal = tl.load(B + batch * K * N + k * N + j)
                acc = acc + aVal * bVal
              }
              tl.store(C + batch * M * N + i * N + j, acc)
            }
          }
        }
      }
      ()
    }
    println("[25] persistentBatchGemmKernel defined")

    // 26. CUTLASS Persistent Tanh Activation
    @TritonKernelMacro
    def persistentTanhKernel(out: Float, in: Float, N: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= nTiles) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = pid * tileSize
      val end = start + tileSize
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          val result = tanh(v)
          tl.store(out + i, result)
        }
      }
      ()
    }
    println("[26] persistentTanhKernel defined")

    // 27. CUTLASS Persistent Sigmoid
    @TritonKernelMacro
    def persistentSigmoidKernel(out: Float, in: Float, N: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= nTiles) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = pid * tileSize
      val end = start + tileSize
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          val result = 1.0f / (1.0f + exp(-v))
          tl.store(out + i, result)
        }
      }
      ()
    }
    println("[27] persistentSigmoidKernel defined")

    // 28. CUTLASS Persistent ReLU
    @TritonKernelMacro
    def persistentReluKernel(out: Float, in: Float, N: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= nTiles) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = pid * tileSize
      val end = start + tileSize
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          val result = if (v > 0.0f) v else 0.0f
          tl.store(out + i, result)
        }
      }
      ()
    }
    println("[28] persistentReluKernel defined")

    // 29. CUTLASS Persistent LeakyReLU
    @TritonKernelMacro
    def persistentLeakyReluKernel(out: Float, in: Float, alpha: Float, N: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= nTiles) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = pid * tileSize
      val end = start + tileSize
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          val result = if (v > 0.0f) v else alpha * v
          tl.store(out + i, result)
        }
      }
      ()
    }
    println("[29] persistentLeakyReluKernel defined")

    // 30. CUTLASS Persistent ELU
    @TritonKernelMacro
    def persistentEluKernel(out: Float, in: Float, alpha: Float, N: Int, nTiles: Int): Unit = {
      val pid = tl.program_id(0)
      if (pid >= nTiles) return
      val tileSize = (N + nTiles - 1) / nTiles
      val start = pid * tileSize
      val end = start + tileSize
      start.until(end) foreach { i =>
        if (i < N) {
          val v = tl.load(in + i)
          val result = if (v > 0.0f) v else alpha * (exp(v) - 1.0f)
          tl.store(out + i, result)
        }
      }
      ()
    }
    println("[30] persistentEluKernel defined")

    // ========================================================================
    // Category 4: Tiled Attention Variants (10 kernels)
    // ========================================================================
    println("\n[31-40] Tiled Attention Variants")

    // 31. Tiled Flash Attention (CUTLASS Style)
    @TritonKernelMacro
    def tiledFlashAttentionKernel(out: Float, q: Float, k: Float, v: Float, scale: Float, M: Int, N: Int, D: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      val tileSize = blockSize
      val numTiles = (N + tileSize - 1) / tileSize
      tl.sharedMem("float", "s_q", 64)
      tl.sharedMem("float", "s_k", 1024)
      tl.sharedMem("float", "s_v", 1024)
      0.until(D) foreach { d =>
        val qVal = tl.load(q + row * D + d)
        tl.sharedStore("s_q", d, qVal)
      }
      var maxScore = -3.4e38f
      var sumExp = 0.0f
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(numTiles) foreach { t =>
          val tileStart = t * tileSize
          val tileEnd = tileStart + tileSize
          tl.syncthreads()
          0.until(tileEnd - tileStart) foreach { j =>
            if (tileStart + j < N) {
              val kVal = tl.load(k + (tileStart + j) * D + d)
              tl.sharedStore("s_k", j * D + d, kVal)
            }
          }
          0.until(tileEnd - tileStart) foreach { j =>
            if (tileStart + j < N) {
              val vVal = tl.load(v + (tileStart + j) * D + d)
              tl.sharedStore("s_v", j * D + d, vVal)
            }
          }
          tl.syncthreads()
          0.until(tileEnd - tileStart) foreach { j =>
            if (tileStart + j < N) {
              val qVal = tl.sharedLoad("s_q", d)
              val kVal = tl.sharedLoad("s_k", j * D + d)
              val score = qVal * kVal * scale
              val attn = exp(score - maxScore)
              result = result + attn * tl.sharedLoad("s_v", j * D + d)
              sumExp = sumExp + attn
            }
          }
        }
        val finalAttn = result / sumExp
        tl.store(out + row * D + d, finalAttn)
      }
      ()
    }
    println("[31] tiledFlashAttentionKernel defined")

    // 32. Tiled Multi-Head Attention (CUTLASS Style)
    @TritonKernelMacro
    def tiledMultiHeadAttentionKernel(out: Float, q: Float, k: Float, v: Float, M: Int, N: Int, D: Int, H: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid / H
      val head = pid % H
      if (row >= M) return
      val scale = 1.0f / sqrt(D.toFloat)
      val headOffsetQ = head * D
      val headOffsetK = head * D
      val headOffsetV = head * D
      val headOffsetO = head * D
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * H * D + headOffsetQ + j)
        val kVal = tl.load(k + j * H * D + headOffsetK + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * H * D + headOffsetQ + j)
        val kVal = tl.load(k + j * H * D + headOffsetK + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val qVal = tl.load(q + row * H * D + headOffsetQ + d)
          val kVal = tl.load(k + j * H * D + headOffsetK + d)
          val score = qVal * kVal * scale
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * H * D + headOffsetV + d)
          result = result + attn * vVal
        }
        tl.store(out + row * H * D + headOffsetO + d, result)
      }
      ()
    }
    println("[32] tiledMultiHeadAttentionKernel defined")

    // 33. Tiled Grouped Query Attention (GQA)
    @TritonKernelMacro
    def tiledGQAttentionKernel(out: Float, q: Float, k: Float, v: Float, scale: Float, M: Int, N: Int, D: Int, H: Int, G: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid / H
      val head = pid % H
      val group = head / G
      if (row >= M) return
      val headOffsetQ = head * D
      val headOffsetK = group * D
      val headOffsetV = group * D
      val headOffsetO = head * D
      val numKV = M / G
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * H * D + headOffsetQ + j)
        val kVal = tl.load(k + j * numKV * D + headOffsetK + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * H * D + headOffsetQ + j)
        val kVal = tl.load(k + j * numKV * D + headOffsetK + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val qVal = tl.load(q + row * H * D + headOffsetQ + d)
          val kVal = tl.load(k + j * numKV * D + headOffsetK + d)
          val score = qVal * kVal * scale
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * numKV * D + headOffsetV + d)
          result = result + attn * vVal
        }
        tl.store(out + row * H * D + headOffsetO + d, result)
      }
      ()
    }
    println("[33] tiledGQAttentionKernel defined")

    // 34. Tiled Sliding Window Attention
    @TritonKernelMacro
    def tiledSlidingWindowAttentionKernel(out: Float, q: Float, k: Float, v: Float, scale: Float, M: Int, N: Int, D: Int, windowSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      val halfWindow = windowSize / 2
      val start = row - halfWindow
      val end = row + halfWindow + 1
      val actualStart = if (start > 0) start else 0
      val actualEnd = if (end < N) end else N
      var maxScore = -3.4e38f
      actualStart.until(actualEnd) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      actualStart.until(actualEnd) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        actualStart.until(actualEnd) foreach { j =>
          val qVal = tl.load(q + row * D + d)
          val kVal = tl.load(k + j * D + d)
          val score = qVal * kVal * scale
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * D + d)
          result = result + attn * vVal
        }
        tl.store(out + row * D + d, result)
      }
      ()
    }
    println("[34] tiledSlidingWindowAttentionKernel defined")

    // 35. Tiled Cross Attention
    @TritonKernelMacro
    def tiledCrossAttentionKernel(out: Float, q: Float, k: Float, v: Float, scale: Float, M: Int, N: Int, D: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      val tileSize = blockSize
      val numTiles = (N + tileSize - 1) / tileSize
      tl.sharedMem("float", "s_k", 1024)
      tl.sharedMem("float", "s_v", 1024)
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(numTiles) foreach { t =>
          val tileStart = t * tileSize
          val tileEnd = tileStart + tileSize
          tl.syncthreads()
          0.until(tileEnd - tileStart) foreach { j =>
            if (tileStart + j < N) {
              val kVal = tl.load(k + (tileStart + j) * D + d)
              tl.sharedStore("s_k", j * D + d, kVal)
            }
          }
          0.until(tileEnd - tileStart) foreach { j =>
            if (tileStart + j < N) {
              val vVal = tl.load(v + (tileStart + j) * D + d)
              tl.sharedStore("s_v", j * D + d, vVal)
            }
          }
          tl.syncthreads()
          0.until(tileEnd - tileStart) foreach { j =>
            if (tileStart + j < N) {
              val qVal = tl.load(q + row * D + d)
              val kVal = tl.sharedLoad("s_k", j * D + d)
              val score = qVal * kVal * scale
              val attn = exp(score - maxScore) / sumExp
              result = result + attn * tl.sharedLoad("s_v", j * D + d)
            }
          }
        }
        tl.store(out + row * D + d, result)
      }
      ()
    }
    println("[35] tiledCrossAttentionKernel defined")

    // 36. Tiled Bi-Directional Attention
    @TritonKernelMacro
    def tiledBidirectionalAttentionKernel(out: Float, q: Float, k: Float, v: Float, scale: Float, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      var maxScoreF = -3.4e38f
      var maxScoreB = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kFVal = tl.load(k + j * D + j)
        val kBVal = tl.load(k + (N - 1 - j) * D + (N - 1 - j))
        val scoreF = qVal * kFVal * scale
        val scoreB = qVal * kBVal * scale
        if (scoreF > maxScoreF) maxScoreF = scoreF
        if (scoreB > maxScoreB) maxScoreB = scoreB
      }
      var sumExpF = 0.0f
      var sumExpB = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kFVal = tl.load(k + j * D + j)
        val kBVal = tl.load(k + (N - 1 - j) * D + (N - 1 - j))
        val scoreF = qVal * kFVal * scale
        val scoreB = qVal * kBVal * scale
        sumExpF = sumExpF + exp(scoreF - maxScoreF)
        sumExpB = sumExpB + exp(scoreB - maxScoreB)
      }
      0.until(D) foreach { d =>
        var resultF: Float = 0.0f
        var resultB: Float = 0.0f
        0.until(N) foreach { j =>
          val qVal = tl.load(q + row * D + d)
          val kFVal = tl.load(k + j * D + d)
          val kBVal = tl.load(k + (N - 1 - j) * D + d)
          val scoreF = qVal * kFVal * scale
          val scoreB = qVal * kBVal * scale
          val attnF = exp(scoreF - maxScoreF) / sumExpF
          val attnB = exp(scoreB - maxScoreB) / sumExpB
          val vFVal = tl.load(v + j * D + d)
          val vBVal = tl.load(v + (N - 1 - j) * D + d)
          resultF = resultF + attnF * vFVal
          resultB = resultB + attnB * vBVal
        }
        tl.store(out + row * D + d, (resultF + resultB) * 0.5f)
      }
      ()
    }
    println("[36] tiledBidirectionalAttentionKernel defined")

    // 37. Tiled Local Attention
    @TritonKernelMacro
    def tiledLocalAttentionKernel(out: Float, q: Float, k: Float, v: Float, scale: Float, M: Int, N: Int, D: Int, localSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      val start = (row / localSize) * localSize
      val end = start + localSize
      val actualEnd = if (end < N) end else N
      var maxScore = -3.4e38f
      start.until(actualEnd) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      start.until(actualEnd) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        start.until(actualEnd) foreach { j =>
          val qVal = tl.load(q + row * D + d)
          val kVal = tl.load(k + j * D + d)
          val score = qVal * kVal * scale
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * D + d)
          result = result + attn * vVal
        }
        tl.store(out + row * D + d, result)
      }
      ()
    }
    println("[37] tiledLocalAttentionKernel defined")

    // 38. Tiled Sparse Attention
    @TritonKernelMacro
    def tiledSparseAttentionKernel(out: Float, q: Float, k: Float, v: Float, sparseMask: Float, scale: Float, M: Int, N: Int, D: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val maskVal = tl.load(sparseMask + row * N + j)
        if (maskVal > 0.0f) {
          val qVal = tl.load(q + row * D + j)
          val kVal = tl.load(k + j * D + j)
          val score = qVal * kVal * scale
          if (score > maxScore) maxScore = score
        }
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val maskVal = tl.load(sparseMask + row * N + j)
        if (maskVal > 0.0f) {
          val qVal = tl.load(q + row * D + j)
          val kVal = tl.load(k + j * D + j)
          val score = qVal * kVal * scale
          sumExp = sumExp + exp(score - maxScore)
        }
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val maskVal = tl.load(sparseMask + row * N + j)
          if (maskVal > 0.0f) {
            val qVal = tl.load(q + row * D + d)
            val kVal = tl.load(k + j * D + d)
            val score = qVal * kVal * scale
            val attn = exp(score - maxScore) / sumExp
            val vVal = tl.load(v + j * D + d)
            result = result + attn * vVal
          }
        }
        tl.store(out + row * D + d, result)
      }
      ()
    }
    println("[38] tiledSparseAttentionKernel defined")

    // 39. Tiled Global+Local Attention
    @TritonKernelMacro
    def tiledGlobalLocalAttentionKernel(out: Float, q: Float, k: Float, v: Float, scale: Float, M: Int, N: Int, D: Int, globalRatio: Float): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      val globalSize = (N * globalRatio).toInt
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val qVal = tl.load(q + row * D + d)
          val kVal = tl.load(k + j * D + d)
          val score = qVal * kVal * scale
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * D + d)
          result = result + attn * vVal
        }
        tl.store(out + row * D + d, result)
      }
      ()
    }
    println("[39] tiledGlobalLocalAttentionKernel defined")

    // 40. Tiled Kernel Diverse Attention (Linear + Convolutional)
    @TritonKernelMacro
    def tiledKernelAttentionKernel(out: Float, q: Float, k: Float, v: Float, convKernel: Float, scale: Float, M: Int, N: Int, D: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      tl.sharedMem("float", "s_conv", 1024)
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * D + j)
        val kVal = tl.load(k + j * D + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(K) foreach { kk =>
        0.until(D) foreach { d =>
          val convVal = tl.load(convKernel + kk * D + d)
          tl.sharedStore("s_conv", kk * D + d, convVal)
        }
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val qVal = tl.load(q + row * D + d)
          val kVal = tl.load(k + j * D + d)
          val score = qVal * kVal * scale
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * D + d)
          var convVal: Float = 0.0f
          0.until(K) foreach { kk =>
            convVal = convVal + tl.sharedLoad("s_conv", kk * D + d)
          }
          result = result + attn * (vVal + convVal)
        }
        tl.store(out + row * D + d, result)
      }
      ()
    }
    println("[40] tiledKernelAttentionKernel defined")

    // ========================================================================
    // Category 5: Fused Tiled Operations (10 kernels)
    // ========================================================================
    println("\n[41-50] Fused Tiled Operations")

    // 41. Fused Tiled GEMM + Bias + GELU (CUTLASS Epilogue)
    @TritonKernelMacro
    def fusedTiledGemmBiasGeluKernel(C: Float, A: Float, B: Float, bias: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val numBlocksM = (M + 63) / 64
      val numBlocksN = (N + 63) / 64
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockM = pid / numBlocksN
      val blockN = pid % numBlocksN
      val rowStart = blockM * 64
      val colStart = blockN * 64
      val row = rowStart
      val col = colStart
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 64 * 64)
        tl.sharedMem("float", "s_b", 64 * 64)
        var sum: Float = 0.0f
        val numTilesK = (K + 63) / 64
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * 64
          0.until(64) foreach { i =>
            0.until(64) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              val bRow = kStart + i
              val bCol = colStart + j
              if (aRow < M && aCol < K && bRow < K && bCol < N) {
                val aVal = tl.load(A + aRow * K + aCol)
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_a", i * 64 + j, aVal)
                tl.sharedStore("s_b", i * 64 + j, bVal)
              }
            }
          }
          tl.syncthreads()
          0.until(64) foreach { kk =>
            0.until(64) foreach { i =>
              0.until(64) foreach { j =>
                val aVal = tl.sharedLoad("s_a", i * 64 + kk)
                val bVal = tl.sharedLoad("s_b", kk * 64 + j)
                sum = sum + aVal * bVal
              }
            }
          }
          tl.syncthreads()
        }
        val biasVal = tl.load(bias + col)
        val x = sum + biasVal
        val cdf = 0.5f * (1.0f + tanh(sqrt(2.0f / 3.14159f) * (x + 0.044715f * x * x * x)))
        tl.store(C + row * N + col, cdf)
      }
      ()
    }
    println("[41] fusedTiledGemmBiasGeluKernel defined")

    // 42. Fused Tiled GEMM + Bias + Residual
    @TritonKernelMacro
    def fusedTiledGemmBiasResidualKernel(C: Float, A: Float, B: Float, bias: Float, residual: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val numBlocksM = (M + 63) / 64
      val numBlocksN = (N + 63) / 64
      val numBlocks = numBlocksM * numBlocksN
      if (pid >= numBlocks) return
      val blockM = pid / numBlocksN
      val blockN = pid % numBlocksN
      val rowStart = blockM * 64
      val colStart = blockN * 64
      val row = rowStart
      val col = colStart
      if (row < M && col < N) {
        tl.sharedMem("float", "s_a", 64 * 64)
        tl.sharedMem("float", "s_b", 64 * 64)
        var sum: Float = 0.0f
        val numTilesK = (K + 63) / 64
        0.until(numTilesK) foreach { tileK =>
          val kStart = tileK * 64
          0.until(64) foreach { i =>
            0.until(64) foreach { j =>
              val aRow = rowStart + i
              val aCol = kStart + j
              val bRow = kStart + i
              val bCol = colStart + j
              if (aRow < M && aCol < K && bRow < K && bCol < N) {
                val aVal = tl.load(A + aRow * K + aCol)
                val bVal = tl.load(B + bRow * N + bCol)
                tl.sharedStore("s_a", i * 64 + j, aVal)
                tl.sharedStore("s_b", i * 64 + j, bVal)
              }
            }
          }
          tl.syncthreads()
          0.until(64) foreach { kk =>
            0.until(64) foreach { i =>
              0.until(64) foreach { j =>
                val aVal = tl.sharedLoad("s_a", i * 64 + kk)
                val bVal = tl.sharedLoad("s_b", kk * 64 + j)
                sum = sum + aVal * bVal
              }
            }
          }
          tl.syncthreads()
        }
        val biasVal = tl.load(bias + col)
        val resVal = tl.load(residual + row * N + col)
        tl.store(C + row * N + col, sum + biasVal + resVal)
      }
      ()
    }
    println("[42] fusedTiledGemmBiasResidualKernel defined")

    // 43. Fused Tiled LayerNorm + Attention
    @TritonKernelMacro
    def fusedTiledLayerNormAttentionKernel(out: Float, q: Float, k: Float, v: Float, lnWeight: Float, lnBias: Float, scale: Float, M: Int, N: Int, D: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      tl.sharedMem("float", "s_ln", 1024)
      var sum = 0.0f
      0.until(N) foreach { i =>
        val qi = tl.load(q + row * N + i)
        sum = sum + qi
      }
      val mean = sum / N.toFloat
      var varSum = 0.0f
      0.until(N) foreach { i =>
        val qi = tl.load(q + row * N + i)
        val diff = qi - mean
        varSum = varSum + diff * diff
      }
      val variance = varSum / N.toFloat
      val std = sqrt(variance + 1e-5f)
      0.until(N) foreach { i =>
        val qi = tl.load(q + row * N + i)
        val w = tl.load(lnWeight + i)
        val b = tl.load(lnBias + i)
        val norm = ((qi - mean) / std) * w + b
        tl.sharedStore("s_ln", i, norm)
      }
      tl.syncthreads()
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.sharedLoad("s_ln", j)
        val kVal = tl.load(k + j * N + j)
        val score = qVal * kVal * scale
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.sharedLoad("s_ln", j)
        val kVal = tl.load(k + j * N + j)
        val score = qVal * kVal * scale
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var result: Float = 0.0f
        0.until(N) foreach { j =>
          val qVal = tl.sharedLoad("s_ln", d)
          val kVal = tl.load(k + j * N + d)
          val score = qVal * kVal * scale
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * N + d)
          result = result + attn * vVal
        }
        tl.store(out + row * N + d, result)
      }
      ()
    }
    println("[43] fusedTiledLayerNormAttentionKernel defined")

    // 44. Fused Tiled RMSNorm + GEMM
    @TritonKernelMacro
    def fusedTiledRmsNormGemmKernel(C: Float, A: Float, B: Float, rmsWeight: Float, M: Int, N: Int, K: Int, blockSize: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      var sumSq = 0.0f
      0.until(N) foreach { i =>
        val v = tl.load(A + row * N + i)
        sumSq = sumSq + v * v
      }
      val rms = sqrt(sumSq / N.toFloat + 1e-5f)
      var acc: Float = 0.0f
      0.until(N) foreach { j =>
        val aVal = tl.load(A + row * N + j)
        val norm = (aVal / rms) * tl.load(rmsWeight + j)
        val bVal = tl.load(B + j * N + row)
        acc = acc + norm * bVal
      }
      tl.store(C + row * N + row, acc)
      ()
    }
    println("[44] fusedTiledRmsNormGemmKernel defined")

    // 45. Fused Tiled Skip Connection
    @TritonKernelMacro
    def fusedTiledSkipConnectionKernel(out: Float, input: Float, gemmA: Float, gemmB: Float, alpha: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      var gemmResult: Float = 0.0f
      0.until(N) foreach { j =>
        val aVal = tl.load(gemmA + row * K + j)
        val bVal = tl.load(gemmB + j * N + row)
        gemmResult = gemmResult + aVal * bVal
      }
      0.until(N) foreach { j =>
        val inputVal = tl.load(input + row * N + j)
        val result = inputVal + alpha * gemmResult
        tl.store(out + row * N + j, result)
      }
      ()
    }
    println("[45] fusedTiledSkipConnectionKernel defined")

    // 46. Fused Tiled Gated Activation (GLU)
    @TritonKernelMacro
    def fusedTiledGatedActivationKernel(out: Float, gate: Float, up: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      0.until(N) foreach { j =>
        val gateVal = tl.load(gate + row * N + j)
        val upVal = tl.load(up + row * N + j)
        val gated = gateVal * (1.0f / (1.0f + exp(-gateVal)))
        tl.store(out + row * N + j, gated * upVal)
      }
      ()
    }
    println("[46] fusedTiledGatedActivationKernel defined")

    // 47. Fused Tiled Swiglu Activation
    @TritonKernelMacro
    def fusedTiledSwigluKernel(out: Float, gate: Float, up: Float, down: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      0.until(N) foreach { j =>
        val gateVal = tl.load(gate + row * K + j)
        val upVal = tl.load(up + row * K + j)
        val swiglu = gateVal * (1.0f / (1.0f + exp(-gateVal)))
        var result: Float = 0.0f
        0.until(K) foreach { k =>
          result = result + swiglu * upVal * tl.load(down + k * N + j)
        }
        tl.store(out + row * N + j, result)
      }
      ()
    }
    println("[47] fusedTiledSwigluKernel defined")

    // 48. Fused Tiled Multi-Head Attention + FFN
    @TritonKernelMacro
    def fusedTiledMhaFfnKernel(out: Float, q: Float, k: Float, v: Float, ffnUp: Float, ffnDown: Float, scale: Float, M: Int, N: Int, D: Int, H: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid / H
      val head = pid % H
      if (row >= M) return
      val scaleVal = 1.0f / sqrt(D.toFloat)
      val headOffsetQ = head * D
      val headOffsetK = head * D
      val headOffsetV = head * D
      val headOffsetO = head * D
      var maxScore = -3.4e38f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * H * D + headOffsetQ + j)
        val kVal = tl.load(k + j * H * D + headOffsetK + j)
        val score = qVal * kVal * scaleVal
        if (score > maxScore) maxScore = score
      }
      var sumExp = 0.0f
      0.until(N) foreach { j =>
        val qVal = tl.load(q + row * H * D + headOffsetQ + j)
        val kVal = tl.load(k + j * H * D + headOffsetK + j)
        val score = qVal * kVal * scaleVal
        sumExp = sumExp + exp(score - maxScore)
      }
      0.until(D) foreach { d =>
        var attnResult: Float = 0.0f
        0.until(N) foreach { j =>
          val qVal = tl.load(q + row * H * D + headOffsetQ + d)
          val kVal = tl.load(k + j * H * D + headOffsetK + d)
          val score = qVal * kVal * scaleVal
          val attn = exp(score - maxScore) / sumExp
          val vVal = tl.load(v + j * H * D + headOffsetV + d)
          attnResult = attnResult + attn * vVal
        }
        var ffnResult: Float = 0.0f
        0.until(D) foreach { k =>
          val upVal = tl.load(ffnUp + (row * H * D + headOffsetO) + k)
          val downVal = tl.load(ffnDown + k * D + (row * H * D + headOffsetO) + d)
          ffnResult = ffnResult + upVal * downVal
        }
        val gated = attnResult * (1.0f / (1.0f + exp(-attnResult)))
        tl.store(out + row * H * D + headOffsetO + d, ffnResult * gated)
      }
      ()
    }
    println("[48] fusedTiledMhaFfnKernel defined")

    // 49. Fused Tiled RMSNorm + Skip + GEMM
    @TritonKernelMacro
    def fusedTiledRmsNormSkipGemmKernel(C: Float, input: Float, skip: Float, gemmA: Float, gemmB: Float, rmsWeight: Float, alpha: Float, M: Int, N: Int, K: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      var sumSq = 0.0f
      0.until(N) foreach { i =>
        val v = tl.load(input + row * N + i)
        sumSq = sumSq + v * v
      }
      val rms = sqrt(sumSq / N.toFloat + 1e-5f)
      0.until(N) foreach { j =>
        val inputVal = tl.load(input + row * N + j)
        val skipVal = tl.load(skip + row * N + j)
        val norm = (inputVal / rms) * tl.load(rmsWeight + j)
        val fused = norm + alpha * skipVal
        var gemmResult: Float = 0.0f
        0.until(K) foreach { k =>
          gemmResult = gemmResult + fused * tl.load(gemmA + row * K + k) * tl.load(gemmB + k * N + j)
        }
        tl.store(C + row * N + j, gemmResult)
      }
      ()
    }
    println("[49] fusedTiledRmsNormSkipGemmKernel defined")

    // 50. Fused Tiled Bias + GELU + Residual + LayerNorm
    @TritonKernelMacro
    def fusedTiledBiasGeluResidualLayerNormKernel(out: Float, input: Float, gemm: Float, bias: Float, lnWeight: Float, lnBias: Float, M: Int, N: Int): Unit = {
      val pid = tl.program_id(0)
      val row = pid
      if (row >= M) return
      0.until(N) foreach { j =>
        val inputVal = tl.load(input + row * N + j)
        val gemmVal = tl.load(gemm + row * N + j)
        val biasVal = tl.load(bias + j)
        val x = inputVal + gemmVal + biasVal
        val cdf = 0.5f * (1.0f + tanh(sqrt(2.0f / 3.14159f) * (x + 0.044715f * x * x * x)))
        tl.store(out + row * N + j, cdf)
      }
      var sum = 0.0f
      0.until(N) foreach { i =>
        sum = sum + tl.load(out + row * N + i)
      }
      val mean = sum / N.toFloat
      var varSum = 0.0f
      0.until(N) foreach { i =>
        val v = tl.load(out + row * N + i)
        val diff = v - mean
        varSum = varSum + diff * diff
      }
      val variance = varSum / N.toFloat
      val std = sqrt(variance + 1e-5f)
      0.until(N) foreach { i =>
        val v = tl.load(out + row * N + i)
        val w = tl.load(lnWeight + i)
        val b = tl.load(lnBias + i)
        val norm = ((v - mean) / std) * w + b
        tl.store(out + row * N + i, norm)
      }
      ()
    }
    println("[50] fusedTiledBiasGeluResidualLayerNormKernel defined")

    println("\n" + "=" * 80)
    println("All 50 CUTLASS-style CUDA kernels defined successfully!")
    println("=" * 80)
  }
}
