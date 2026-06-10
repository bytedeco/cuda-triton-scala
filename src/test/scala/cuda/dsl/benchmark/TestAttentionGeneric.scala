package cuda.dsl.benchmark

import cuda.dsl.dsl._

/** Test @TritonKernelMacro with generic Triton-like DSL for Attention kernels */
object TestAttentionGeneric {

  // DSL math helpers: recognized by @TritonKernelMacro → CUDA expf/sqrtf
  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat
  def float2int(x: Float): Float = x  // noop in Scala, translated to (int) cast in C

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("@TritonKernelMacro: Generic Triton-like Attention Kernels")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels.txt for generated CUDA code.")
    println("=" * 80)

    // ========================================================================
    // Test 1: Generic Triton-like StoreKVCache
    // ========================================================================
    println("\n[1] StoreKVCache Kernel (generic Triton-like)")
    @TritonKernelMacro
    def storeKVCacheKernel(
        keyPtr: Float, keyStride: Int,
        valuePtr: Float, valueStride: Int,
        kCachePtr: Float, vCachePtr: Float,
        slotMappingPtr: Int,
        D: Int): Unit = {
      // Triton-like thread identification
      val idx = tl.program_id(0)

      // Load slot mapping
      val slot = tl.load(slotMappingPtr + idx).toInt

      // Early exit for invalid slot
      if (slot == -1) return

      // Compute cache offset
      val cacheOffset = slot * D

      // Store key/value to cache
      var d: Int = 0
      while (d < D) {
        tl.store(kCachePtr + (cacheOffset + d).toFloat, keyPtr + (idx * keyStride + d).toFloat)
        tl.store(vCachePtr + (cacheOffset + d).toFloat, valuePtr + (idx * keyStride + d).toFloat)
        d = d + 1
      }
    }
    println("StoreKVCache kernel defined")

    // ========================================================================
    // Test 2: Generic Triton-like FlashAttention (vLLM-style)
    // ========================================================================
    println("\n[2] FlashAttention Kernel (generic Triton-like)")
    @TritonKernelMacro
    def flashAttentionKernel(
        outPtr: Float, qPtr: Float, kPtr: Float, vPtr: Float,
        N: Int, D: Int, n: Int): Unit = {
      // 每线程处理一个 query 位置 row，遍历全部 key 位置 j，计算完整 attention
      val i = tl.program_id(0)
      val row = i

      if (row >= N) return

      val scale = 1.0f / sqrt(D.toFloat)

      // ---- 第一遍：遍历所有 key 位置 j，计算 max score ----
      var scoreMax: Float = -3.4e38f
      var j: Int = 0
      while (j < N) {
        // 计算 q[row] · k[j]（先对 d 累加）
        var dot: Float = 0.0f
        var d: Int = 0
        while (d < D) {
          val q_d = tl.load(qPtr + (row * D + d).toFloat)
          val k_d = tl.load(kPtr + (j * D + d).toFloat)
          dot = dot + q_d * k_d
          d = d + 1
        }
        val score_ij = dot * scale
        if (score_ij > scoreMax) scoreMax = score_ij
        j = j + 1
      }

      // ---- 第二遍：计算 exp sum ----
      var expSum: Float = 0.0f
      j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d: Int = 0
        while (d < D) {
          val q_d = tl.load(qPtr + (row * D + d).toFloat)
          val k_d = tl.load(kPtr + (j * D + d).toFloat)
          dot = dot + q_d * k_d
          d = d + 1
        }
        val score_ij = dot * scale
        expSum = expSum + exp(score_ij - scoreMax)
        j = j + 1
      }

      // ---- 第三遍：计算输出 out[row * D + d] ----
      var d: Int = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          // 重新计算 dot product
          var dot: Float = 0.0f
          var dk: Int = 0
          while (dk < D) {
            val q_d = tl.load(qPtr + (row * D + dk).toFloat)
            val k_d = tl.load(kPtr + (j * D + dk).toFloat)
            dot = dot + q_d * k_d
            dk = dk + 1
          }
          val score_ij = dot * scale
          val exp_ij = exp(score_ij - scoreMax)
          val attn_ij = exp_ij / expSum
          val v_d = tl.load(vPtr + j * D + d)
          result = result + attn_ij * v_d
          j = j + 1
        }
        tl.store(outPtr + (row * D + d).toFloat, result)
        d = d + 1
      }
      ()
    }
    println("FlashAttention kernel defined")

    // ========================================================================
    // Test 3: Generic Triton-like PageAttention (vLLM-style)
    // ========================================================================
    println("\n[3] PageAttention Kernel (generic Triton-like)")
    @TritonKernelMacro
    def pageAttentionKernel(
        outPtr: Float, queryPtr: Float, keyCachePtr: Float, valueCachePtr: Float,
        blockTablesPtr: Int, seqLensPtr: Int,
        batchSize: Int, maxBlocksPerSeq: Int, blockSize: Int, D: Int): Unit = {
      // 每线程处理一个 query 位置 pos = i
      val pos = tl.program_id(0)
      val scale = 1.0f / sqrt(D.toFloat)

      // 获取序列长度（假设 batch_idx = 0）
      val seqLen = tl.load(seqLensPtr + 0)

      // pos 越界检查
      if (pos >= seqLen) return

      // 计算该 batch 的 block table 起始偏移: batch_idx * maxBlocksPerSeq
      // 假设 batch_idx = 0，多 batch 时需从 batch 索引表查
      val batchTableOffset = 0

      // ---- 第一遍：遍历所有 key positions，计算 max 和 exp(sum) ----
      var scoreMax: Float = -3.4e38f
      var expSum: Float = 0.0f

      // 循环变量统一在外部声明，使用 Int 类型（用于 array indexing）
      var keyPos: Int = 0
      var blockNum: Int = 0
      var posInBlock: Int = 0
      var physBlock: Int = 0
      var kBase: Int = 0
      var vBase: Int = 0
      var d: Int = 0

      keyPos = 0
      while (keyPos < seqLen) {
        blockNum = keyPos / blockSize
        posInBlock = keyPos % blockSize
        physBlock = tl.load(blockTablesPtr + (batchTableOffset + blockNum).toFloat).toInt
        kBase = physBlock * blockSize * D + posInBlock * D

        d = 0
        while (d < D) {
          val q = tl.load(queryPtr + (pos * D + d).toFloat)
          val k = tl.load(keyCachePtr + (kBase + d).toFloat)
          val score = q * k * scale

          // online softmax 在线更新（安全数值写法）
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

      // ---- 第二遍：直接 accumulate 到 outPtr ----
      keyPos = 0
      while (keyPos < seqLen) {
        blockNum = keyPos / blockSize
        posInBlock = keyPos % blockSize
        physBlock = tl.load(blockTablesPtr + (batchTableOffset + blockNum).toFloat).toInt
        kBase = physBlock * blockSize * D + posInBlock * D
        vBase = physBlock * blockSize * D + posInBlock * D

        d = 0
        while (d < D) {
          val q = tl.load(queryPtr + (pos * D + d).toFloat)
          val k = tl.load(keyCachePtr + (kBase + d).toFloat)
          val score = q * k * scale
          val expScore = exp(score - scoreMax)
          val attnWeight = expScore / expSum
          val v = tl.load(valueCachePtr + (vBase + d).toFloat)
          val outIdx = pos * D + d
          // accumulate 到 outPtr[outIdx]（kernel 调用前输出需清零）
          val newVal = tl.load(outPtr + outIdx.toFloat) + attnWeight * v
          tl.store(outPtr + outIdx.toFloat, newVal)
          d = d + 1
        }
        keyPos = keyPos + 1
      }
      ()
      ()
    }
    println("PageAttention kernel defined")

    // ========================================================================
    // Test 4: Generic Triton-like FlexAttention
    // ========================================================================
    println("\n[4] FlexAttention Kernel (generic Triton-like)")
    @TritonKernelMacro
    def flexAttentionKernel(
        outPtr: Float, qPtr: Float, kPtr: Float, vPtr: Float,
        N: Int, D: Int, n: Int): Unit = {
      // 每线程处理一个 query 位置 row，遍历全部 key 位置 j，支持 score modification
      val i = tl.program_id(0)
      val row = i

      if (row >= N) return

      val scale = 1.0f / sqrt(D.toFloat)

      // ---- 第一遍：遍历所有 key 位置 j，计算 max score ----
      var scoreMax: Float = -3.4e38f
      var j: Int = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d: Int = 0
        while (d < D) {
          val q_d = tl.load(qPtr + (row * D + d).toFloat)
          val k_d = tl.load(kPtr + (j * D + d).toFloat)
          dot = dot + q_d * k_d
          d = d + 1
        }
        val score_ij = dot * scale
        // [score modification placeholder]
        // score_ij = scoreMod(row, j, score_ij)
        if (score_ij > scoreMax) scoreMax = score_ij
        j = j + 1
      }

      // ---- 第二遍：计算 exp sum ----
      var expSum: Float = 0.0f
      j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d: Int = 0
        while (d < D) {
          val q_d = tl.load(qPtr + (row * D + d).toFloat)
          val k_d = tl.load(kPtr + (j * D + d).toFloat)
          dot = dot + q_d * k_d
          d = d + 1
        }
        val score_ij = dot * scale
        // [score modification placeholder]
        // score_ij = scoreMod(row, j, score_ij)
        expSum = expSum + exp(score_ij - scoreMax)
        j = j + 1
      }

      // ---- 第三遍：计算输出 out[row * D + d] ----
      var d: Int = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk: Int = 0
          while (dk < D) {
            val q_d = tl.load(qPtr + (row * D + dk).toFloat)
            val k_d = tl.load(kPtr + (j * D + dk).toFloat)
            dot = dot + q_d * k_d
            dk = dk + 1
          }
          val score_ij = dot * scale
          // [score modification placeholder]
          // score_ij = scoreMod(row, j, score_ij)
          val exp_ij = exp(score_ij - scoreMax)
          val attn_ij = exp_ij / expSum
          val v_d = tl.load(vPtr + j * D + d)
          result = result + attn_ij * v_d
          j = j + 1
        }
        tl.store(outPtr + (row * D + d).toFloat, result)
        d = d + 1
      }
      ()
    }
    println("FlexAttention kernel defined")

    println("\n" + "=" * 80)
    println("All Attention kernels defined successfully!")
    println("=" * 80)
  }
}