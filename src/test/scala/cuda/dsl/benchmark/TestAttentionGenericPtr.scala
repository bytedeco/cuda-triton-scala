package cuda.dsl.benchmark

import cuda.dsl.core.{FloatPtr, IntPtr}
import cuda.dsl.dsl._

/** Pointer variant of TestAttentionGeneric using FloatPtr/IntPtr types.
  *
  * Test @TritonKernelMacro with generic Triton-like DSL for Attention kernels.
  * Suffix SPtr distinguishes from Java JPtr versions.
  */
object TestAttentionGenericPtr {

  // DSL math helpers: recognized by @TritonKernelMacro -> CUDA expf/sqrtf
  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("@TritonKernelMacro: Generic Triton-like Attention Kernels (SPtr variant)")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels_ptr.txt for generated CUDA code.")
    println("=" * 80)

    // ========================================================================
    // KV Cache Operations
    // ========================================================================

    // 1. Store KV Cache
    @TritonKernelMacro(name = "storeKVCacheKernelSPtr", gridType = "1D")
    def storeKVCacheKernelPtr(
        kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        kPtr: FloatPtr, vPtr: FloatPtr,
        seqIdPtr: IntPtr,
        batch: Int, head: Int, seqLen: Int, maxSeqLen: Int): Unit = {
      val idx = tl.program_id(0)
      val slot = tl.load(seqIdPtr, idx).toInt
      if (slot == -1) return
      val cacheOffset = slot * maxSeqLen
      var d = 0
      while (d < seqLen) {
        tl.store(kCachePtr, (head * maxSeqLen + cacheOffset + d) * maxSeqLen + d, tl.load(kPtr, idx * seqLen + d))
        tl.store(vCachePtr, (head * maxSeqLen + cacheOffset + d) * maxSeqLen + d, tl.load(vPtr, idx * seqLen + d))
        d = d + 1
      }
      ()
    }

    // 2. Load KV Cache
    @TritonKernelMacro(name = "loadKVCacheKernelSPtr", gridType = "1D")
    def loadKVCacheKernelPtr(
        kPtr: FloatPtr, vPtr: FloatPtr,
        kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        seqIdPtr: IntPtr,
        batch: Int, head: Int, seqLen: Int, maxSeqLen: Int): Unit = {
      val idx = tl.program_id(0)
      val slot = tl.load(seqIdPtr, idx).toInt
      if (slot == -1) return
      val cacheOffset = slot * maxSeqLen
      var d = 0
      while (d < seqLen) {
        tl.store(kPtr, idx * seqLen + d, tl.load(kCachePtr, (head * maxSeqLen + cacheOffset + d) * maxSeqLen + d))
        tl.store(vPtr, idx * seqLen + d, tl.load(vCachePtr, (head * maxSeqLen + cacheOffset + d) * maxSeqLen + d))
        d = d + 1
      }
      ()
    }

    // 3. Update KV Cache
    @TritonKernelMacro(name = "updateKVCacheKernelSPtr", gridType = "1D")
    def updateKVCacheKernelPtr(
        kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        kDeltaPtr: FloatPtr, vDeltaPtr: FloatPtr,
        seqIdPtr: IntPtr,
        batch: Int, head: Int, pos: Int): Unit = {
      val idx = tl.program_id(0)
      val slot = tl.load(seqIdPtr, idx).toInt
      if (slot == -1) return
      val cacheOffset = slot + pos
      tl.store(kCachePtr, cacheOffset, tl.load(kCachePtr, cacheOffset) + tl.load(kDeltaPtr, idx))
      tl.store(vCachePtr, cacheOffset, tl.load(vCachePtr, cacheOffset) + tl.load(vDeltaPtr, idx))
      ()
    }

    // 4. Evict KV Cache
    @TritonKernelMacro(name = "evictKVCacheKernelSPtr", gridType = "1D")
    def evictKVCacheKernelPtr(
        kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        blockTablesPtr: IntPtr,
        batch: Int, head: Int, blockId: Int): Unit = {
      val idx = tl.program_id(0)
      val physBlock = tl.load(blockTablesPtr, blockId)
      var d = 0
      while (d < 128) {
        tl.store(kCachePtr, physBlock * 128 + d, 0.0f)
        tl.store(vCachePtr, physBlock * 128 + d, 0.0f)
        d = d + 1
      }
      ()
    }

    // 5. Prefix KV Cache
    @TritonKernelMacro(name = "prefixKVCacheKernelSPtr", gridType = "1D")
    def prefixKVCacheKernelPtr(
        kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        prefixKPtr: FloatPtr, prefixVPtr: FloatPtr,
        head: Int, prefixLen: Int, maxSeqLen: Int): Unit = {
      val idx = tl.program_id(0)
      if (idx >= prefixLen) return
      val cacheOffset = head * maxSeqLen + idx
      tl.store(kCachePtr, cacheOffset, tl.load(prefixKPtr, idx))
      tl.store(vCachePtr, cacheOffset, tl.load(prefixVPtr, idx))
      ()
    }

    // ========================================================================
    // Flash Attention Variants
    // ========================================================================

    // 6. Flash Attention (scale before D for template compatibility)
    @TritonKernelMacro(name = "flashAttentionKernelSPtr", gridType = "1D")
    def flashAttentionKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, scale: Float, D: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
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
        var d = 0
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
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
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

    // 7. Flash Attention with Bias (scale before D)
    @TritonKernelMacro(name = "flashAttentionBiasKernelSPtr", gridType = "1D")
    def flashAttentionBiasKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, biasPtr: FloatPtr,
        N: Int, scale: Float, D: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val q_d = tl.load(qPtr, row * D + d)
          val k_d = tl.load(kPtr, j * D + d)
          dot = dot + q_d * k_d
          d = d + 1
        }
        val bv = tl.load(biasPtr, row * N + j)
        val score_ij = dot * scale + bv
        if (score_ij > scoreMax) scoreMax = score_ij
        j = j + 1
      }
      var expSum: Float = 0.0f
      j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val q_d = tl.load(qPtr, row * D + d)
          val k_d = tl.load(kPtr, j * D + d)
          dot = dot + q_d * k_d
          d = d + 1
        }
        val bv = tl.load(biasPtr, row * N + j)
        val score_ij = dot * scale + bv
        expSum = expSum + exp(score_ij - scoreMax)
        j = j + 1
      }
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
          while (dk < D) {
            val q_d = tl.load(qPtr, row * D + dk)
            val k_d = tl.load(kPtr, j * D + dk)
            dot = dot + q_d * k_d
            dk = dk + 1
          }
          val bv = tl.load(biasPtr, row * N + j)
          val score_ij = dot * scale + bv
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

    // 8. Flash Attention with Mask
    @TritonKernelMacro(name = "flashAttentionMaskKernelSPtr", gridType = "1D")
    def flashAttentionMaskKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, maskPtr: IntPtr,
        N: Int, D: Int, scale: Float): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        val maskVal = tl.load(maskPtr, row * N + j)
        if (maskVal == 0) {
          j = j + 1
        } else {
          var dot: Float = 0.0f
          var d = 0
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
      }
      var expSum: Float = 0.0f
      j = 0
      while (j < N) {
        val maskVal = tl.load(maskPtr, row * N + j)
        if (maskVal == 0) {
          j = j + 1
        } else {
          var dot: Float = 0.0f
          var d = 0
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
      }
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          val maskVal = tl.load(maskPtr, row * N + j)
          if (maskVal == 0) {
            j = j + 1
          } else {
            var dot: Float = 0.0f
            var dk = 0
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
        }
        tl.store(outPtr, row * D + d, result)
        d = d + 1
      }
      ()
    }

    // 9. Flash Attention with Dropout
    @TritonKernelMacro(name = "flashAttentionDropoutKernelSPtr", gridType = "1D")
    def flashAttentionDropoutKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, dropoutMaskPtr: FloatPtr,
        prob: Float, N: Int, D: Int, scale: Float): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
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
        var d = 0
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
      val scaleDrop = 1.0f / prob
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
          while (dk < D) {
            val q_d = tl.load(qPtr, row * D + dk)
            val k_d = tl.load(kPtr, j * D + dk)
            dot = dot + q_d * k_d
            dk = dk + 1
          }
          val score_ij = dot * scale
          val exp_ij = exp(score_ij - scoreMax)
          val attn_ij = exp_ij / expSum
          val mask = if (tl.load(dropoutMaskPtr, row * N + j) > prob) 1.0f else 0.0f
          val v_d = tl.load(vPtr, j * D + d)
          result = result + attn_ij * v_d * mask * scaleDrop
          j = j + 1
        }
        tl.store(outPtr, row * D + d, result)
        d = d + 1
      }
      ()
    }

    // 10. Flash Attention Split
    @TritonKernelMacro(name = "flashAttentionSplitKernelSPtr", gridType = "1D")
    def flashAttentionSplitKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, D: Int, numBlocks: Int, scale: Float): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
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
        var d = 0
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
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
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

    // ========================================================================
    // Page Attention
    // ========================================================================

    // 11. Page Attention
    @TritonKernelMacro(name = "pageAttentionKernelSPtr", gridType = "1D")
    def pageAttentionKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        blockTablesPtr: IntPtr, seqLensPtr: IntPtr,
        batch: Int, maxBlocks: Int, blockSize: Int, D: Int): Unit = {
      val pos = tl.program_id(0)
      val scale = 1.0f / sqrt(D.toFloat)
      val seqLen = tl.load(seqLensPtr, 0).toInt
      if (pos >= seqLen) return
      val batchTableOffset = 0
      var scoreMax: Float = -3.4e38f
      var expSum: Float = 0.0f
      var keyPos = 0
      while (keyPos < seqLen) {
        val blockIdx = keyPos / blockSize
        val posInBlock = keyPos % blockSize
        val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
        val kBase = physBlock * blockSize * D + posInBlock * D
        var d = 0
        while (d < D) {
          val q = tl.load(qPtr, pos * D + d)
          val kv = tl.load(kCachePtr, kBase + d)
          val score = q * kv * scale
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
        val blockIdx = keyPos / blockSize
        val posInBlock = keyPos % blockSize
        val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
        val kBase = physBlock * blockSize * D + posInBlock * D
        val vBase = physBlock * blockSize * D + posInBlock * D
        var d = 0
        while (d < D) {
          val q = tl.load(qPtr, pos * D + d)
          val kv = tl.load(kCachePtr, kBase + d)
          val score = q * kv * scale
          val expScore = exp(score - scoreMax)
          val attnWeight = expScore / expSum
          val vVal = tl.load(vCachePtr, vBase + d)
          val outIdx = pos * D + d
          val newVal = tl.load(outPtr, outIdx) + attnWeight * vVal
          tl.store(outPtr, outIdx, newVal)
          d = d + 1
        }
        keyPos = keyPos + 1
      }
      ()
    }

    // 12. Page Attention Paged
    @TritonKernelMacro(name = "pageAttentionPagedKernelSPtr", gridType = "1D")
    def pageAttentionPagedKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        blockTablesPtr: IntPtr, seqLensPtr: IntPtr, positionIdsPtr: IntPtr,
        batch: Int, maxBlocks: Int, blockSize: Int, D: Int): Unit = {
      val pos = tl.program_id(0)
      val scale = 1.0f / sqrt(D.toFloat)
      val seqLen = tl.load(seqLensPtr, 0).toInt
      if (pos >= seqLen) return
      val batchTableOffset = 0
      var scoreMax: Float = -3.4e38f
      var expSum: Float = 0.0f
      var keyPos = 0
      while (keyPos < seqLen) {
        val blockIdx = keyPos / blockSize
        val posInBlock = keyPos % blockSize
        val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
        val kBase = physBlock * blockSize * D + posInBlock * D
        var d = 0
        while (d < D) {
          val q = tl.load(qPtr, pos * D + d)
          val kv = tl.load(kCachePtr, kBase + d)
          val score = q * kv * scale
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
        val blockIdx = keyPos / blockSize
        val posInBlock = keyPos % blockSize
        val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
        val kBase = physBlock * blockSize * D + posInBlock * D
        val vBase = physBlock * blockSize * D + posInBlock * D
        var d = 0
        while (d < D) {
          val q = tl.load(qPtr, pos * D + d)
          val kv = tl.load(kCachePtr, kBase + d)
          val score = q * kv * scale
          val expScore = exp(score - scoreMax)
          val attnWeight = expScore / expSum
          val vVal = tl.load(vCachePtr, vBase + d)
          val outIdx = pos * D + d
          val newVal = tl.load(outPtr, outIdx) + attnWeight * vVal
          tl.store(outPtr, outIdx, newVal)
          d = d + 1
        }
        keyPos = keyPos + 1
      }
      ()
    }

    // 13. Page Attention Prefill
    @TritonKernelMacro(name = "pageAttentionPrefillKernelSPtr", gridType = "1D")
    def pageAttentionPrefillKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        blockTablesPtr: IntPtr, seqLensPtr: IntPtr,
        batch: Int, maxBlocks: Int, blockSize: Int, D: Int): Unit = {
      val pos = tl.program_id(0)
      val scale = 1.0f / sqrt(D.toFloat)
      val seqLen = tl.load(seqLensPtr, 0).toInt
      if (pos >= seqLen) return
      val batchTableOffset = 0
      var scoreMax: Float = -3.4e38f
      var expSum: Float = 0.0f
      var keyPos = 0
      while (keyPos < seqLen) {
        val blockIdx = keyPos / blockSize
        val posInBlock = keyPos % blockSize
        val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
        val kBase = physBlock * blockSize * D + posInBlock * D
        var d = 0
        while (d < D) {
          val q = tl.load(qPtr, pos * D + d)
          val kv = tl.load(kCachePtr, kBase + d)
          val score = q * kv * scale
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
        val blockIdx = keyPos / blockSize
        val posInBlock = keyPos % blockSize
        val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
        val kBase = physBlock * blockSize * D + posInBlock * D
        val vBase = physBlock * blockSize * D + posInBlock * D
        var d = 0
        while (d < D) {
          val q = tl.load(qPtr, pos * D + d)
          val kv = tl.load(kCachePtr, kBase + d)
          val score = q * kv * scale
          val expScore = exp(score - scoreMax)
          val attnWeight = expScore / expSum
          val vVal = tl.load(vCachePtr, vBase + d)
          val outIdx = pos * D + d
          val newVal = tl.load(outPtr, outIdx) + attnWeight * vVal
          tl.store(outPtr, outIdx, newVal)
          d = d + 1
        }
        keyPos = keyPos + 1
      }
      ()
    }

    // 14. Page Attention Decode
    @TritonKernelMacro(name = "pageAttentionDecodeKernelSPtr", gridType = "1D")
    def pageAttentionDecodeKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        blockTablesPtr: IntPtr, lastTokensPtr: IntPtr,
        batch: Int, maxBlocks: Int, blockSize: Int, D: Int): Unit = {
      val pos = tl.program_id(0)
      val scale = 1.0f / sqrt(D.toFloat)
      val batchTableOffset = 0
      val seqLen = maxBlocks * blockSize
      var scoreMax: Float = -3.4e38f
      var expSum: Float = 0.0f
      var keyPos = 0
      while (keyPos < seqLen) {
        val blockIdx = keyPos / blockSize
        val posInBlock = keyPos % blockSize
        val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
        val kBase = physBlock * blockSize * D + posInBlock * D
        var d = 0
        while (d < D) {
          val q = tl.load(qPtr, pos * D + d)
          val kv = tl.load(kCachePtr, kBase + d)
          val score = q * kv * scale
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
        val blockIdx = keyPos / blockSize
        val posInBlock = keyPos % blockSize
        val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
        val kBase = physBlock * blockSize * D + posInBlock * D
        val vBase = physBlock * blockSize * D + posInBlock * D
        var d = 0
        while (d < D) {
          val q = tl.load(qPtr, pos * D + d)
          val kv = tl.load(kCachePtr, kBase + d)
          val score = q * kv * scale
          val expScore = exp(score - scoreMax)
          val attnWeight = expScore / expSum
          val vVal = tl.load(vCachePtr, vBase + d)
          val outIdx = pos * D + d
          val newVal = tl.load(outPtr, outIdx) + attnWeight * vVal
          tl.store(outPtr, outIdx, newVal)
          d = d + 1
        }
        keyPos = keyPos + 1
      }
      ()
    }

    // 15. Page Attention Sparse
    @TritonKernelMacro(name = "pageAttentionSparseKernelSPtr", gridType = "1D")
    def pageAttentionSparseKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        blockTablesPtr: IntPtr, sparseIdxPtr: IntPtr,
        batch: Int, maxBlocks: Int, blockSize: Int, D: Int): Unit = {
      val pos = tl.program_id(0)
      val scale = 1.0f / sqrt(D.toFloat)
      val seqLen = maxBlocks * blockSize
      if (pos >= seqLen) return
      val batchTableOffset = 0
      var scoreMax: Float = -3.4e38f
      var expSum: Float = 0.0f
      val numSparse = 16
      var idx = 0
      while (idx < numSparse) {
        val keyPos = tl.load(sparseIdxPtr, pos * numSparse + idx).toInt
        if (keyPos >= 0 && keyPos < seqLen) {
          val blockIdx = keyPos / blockSize
          val posInBlock = keyPos % blockSize
          val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
          val kBase = physBlock * blockSize * D + posInBlock * D
          var d = 0
          while (d < D) {
            val q = tl.load(qPtr, pos * D + d)
            val kv = tl.load(kCachePtr, kBase + d)
            val score = q * kv * scale
            if (score > scoreMax) {
              expSum = expSum * exp(scoreMax - score) + 1.0f
              scoreMax = score
            } else {
              expSum = expSum + exp(score - scoreMax)
            }
            d = d + 1
          }
        }
        idx = idx + 1
      }
      idx = 0
      while (idx < numSparse) {
        val keyPos = tl.load(sparseIdxPtr, pos * numSparse + idx).toInt
        if (keyPos >= 0 && keyPos < seqLen) {
          val blockIdx = keyPos / blockSize
          val posInBlock = keyPos % blockSize
          val physBlock = tl.load(blockTablesPtr, batchTableOffset + blockIdx).toInt
          val kBase = physBlock * blockSize * D + posInBlock * D
          val vBase = physBlock * blockSize * D + posInBlock * D
          var d = 0
          while (d < D) {
            val q = tl.load(qPtr, pos * D + d)
            val kv = tl.load(kCachePtr, kBase + d)
            val score = q * kv * scale
            val expScore = exp(score - scoreMax)
            val attnWeight = expScore / expSum
            val vVal = tl.load(vCachePtr, vBase + d)
            val outIdx = pos * D + d
            val newVal = tl.load(outPtr, outIdx) + attnWeight * vVal
            tl.store(outPtr, outIdx, newVal)
            d = d + 1
          }
        }
        idx = idx + 1
      }
      ()
    }

    // ========================================================================
    // Flex Attention
    // ========================================================================

    // 16. Flex Attention
    @TritonKernelMacro(name = "flexAttentionKernelSPtr", gridType = "1D")
    def flexAttentionKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, scoreModPtr: FloatPtr,
        N: Int, M: Int, D: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < M) {
        var dot: Float = 0.0f
        var d = 0
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
      while (j < M) {
        var dot: Float = 0.0f
        var d = 0
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
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < M) {
          var dot: Float = 0.0f
          var dk = 0
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

    // 17. Flex Attention Block Mask
    @TritonKernelMacro(name = "flexAttentionBlockMaskKernelSPtr", gridType = "1D")
    def flexAttentionBlockMaskKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr, blockMaskPtr: IntPtr,
        N: Int, M: Int, D: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < M) {
        val blockMask = tl.load(blockMaskPtr, row * M + j)
        if (blockMask != 0) {
          var dot: Float = 0.0f
          var d = 0
          while (d < D) {
            val q_d = tl.load(qPtr, row * D + d)
            val k_d = tl.load(kPtr, j * D + d)
            dot = dot + q_d * k_d
            d = d + 1
          }
          val score_ij = dot * scale
          if (score_ij > scoreMax) scoreMax = score_ij
        }
        j = j + 1
      }
      var expSum: Float = 0.0f
      j = 0
      while (j < M) {
        val blockMask = tl.load(blockMaskPtr, row * M + j)
        if (blockMask != 0) {
          var dot: Float = 0.0f
          var d = 0
          while (d < D) {
            val q_d = tl.load(qPtr, row * D + d)
            val k_d = tl.load(kPtr, j * D + d)
            dot = dot + q_d * k_d
            d = d + 1
          }
          val score_ij = dot * scale
          expSum = expSum + exp(score_ij - scoreMax)
        }
        j = j + 1
      }
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < M) {
          val blockMask = tl.load(blockMaskPtr, row * M + j)
          if (blockMask != 0) {
            var dot: Float = 0.0f
            var dk = 0
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
          }
          j = j + 1
        }
        tl.store(outPtr, row * D + d, result)
        d = d + 1
      }
      ()
    }

    // 18. Flex Attention Score Mod
    @TritonKernelMacro(name = "flexAttentionScoreModKernelSPtr", gridType = "1D")
    def flexAttentionScoreModKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        queryOffsetPtr: FloatPtr, keyOffsetPtr: FloatPtr,
        N: Int, M: Int, D: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < M) {
        var dot: Float = 0.0f
        var d = 0
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
      while (j < M) {
        var dot: Float = 0.0f
        var d = 0
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
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < M) {
          var dot: Float = 0.0f
          var dk = 0
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

    // 19. Flex Attention Causal
    @TritonKernelMacro(name = "flexAttentionCausalKernelSPtr", gridType = "1D")
    def flexAttentionCausalKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, D: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j <= row) {
        var dot: Float = 0.0f
        var d = 0
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
      while (j <= row) {
        var dot: Float = 0.0f
        var d = 0
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
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j <= row) {
          var dot: Float = 0.0f
          var dk = 0
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

    // 20. Flex Attention Sliding Window
    @TritonKernelMacro(name = "flexAttentionSlidingWindowKernelSPtr", gridType = "1D")
    def flexAttentionSlidingWindowKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, D: Int, windowSize: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      val startPos = scala.math.max(0, row - windowSize)
      var scoreMax: Float = -3.4e38f
      var j = startPos
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
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
      j = startPos
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
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
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = startPos
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
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

    // ========================================================================
    // Multi-head & Variants
    // ========================================================================

    // 21. Multi-Head Attention
    @TritonKernelMacro(name = "multiHeadAttentionKernelSPtr", gridType = "1D")
    def multiHeadAttentionKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, D: Int, H: Int): Unit = {
      val i = tl.program_id(0)
      val head = i % H
      val row = i / H
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * H * D + head * D + d)
          val kv = tl.load(kPtr, j * H * D + head * D + d)
          dot = dot + qv * kv
          d = d + 1
        }
        val score = dot * scale
        if (score > scoreMax) scoreMax = score
        j = j + 1
      }
      var expSum: Float = 0.0f
      j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * H * D + head * D + d)
          val kv = tl.load(kPtr, j * H * D + head * D + d)
          dot = dot + qv * kv
          d = d + 1
        }
        val score = dot * scale
        expSum = expSum + exp(score - scoreMax)
        j = j + 1
      }
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
          while (dk < D) {
            val qv = tl.load(qPtr, row * H * D + head * D + dk)
            val kv = tl.load(kPtr, j * H * D + head * D + dk)
            dot = dot + qv * kv
            dk = dk + 1
          }
          val score = dot * scale
          val expScore = exp(score - scoreMax)
          val attn = expScore / expSum
          val vv = tl.load(vPtr, j * H * D + head * D + d)
          result = result + attn * vv
          j = j + 1
        }
        tl.store(outPtr, row * H * D + head * D + d, result)
        d = d + 1
      }
      ()
    }

    // 22. Grouped Query Attention
    @TritonKernelMacro(name = "groupedQueryAttentionKernelSPtr", gridType = "1D")
    def groupedQueryAttentionKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, D: Int, H: Int, G: Int): Unit = {
      val i = tl.program_id(0)
      val head = i % H
      val row = i / H
      if (row >= N) return
      val kvHead = head / G
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * H * D + head * D + d)
          val kvV = tl.load(kPtr, j * (N / G) * D + kvHead * D + d)
          dot = dot + qv * kvV
          d = d + 1
        }
        val score = dot * scale
        if (score > scoreMax) scoreMax = score
        j = j + 1
      }
      var expSum: Float = 0.0f
      j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * H * D + head * D + d)
          val kvV = tl.load(kPtr, j * (N / G) * D + kvHead * D + d)
          dot = dot + qv * kvV
          d = d + 1
        }
        val score = dot * scale
        expSum = expSum + exp(score - scoreMax)
        j = j + 1
      }
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
          while (dk < D) {
            val qv = tl.load(qPtr, row * H * D + head * D + dk)
            val kvV = tl.load(kPtr, j * (N / G) * D + kvHead * D + dk)
            dot = dot + qv * kvV
            dk = dk + 1
          }
          val score = dot * scale
          val expScore = exp(score - scoreMax)
          val attn = expScore / expSum
          val vv = tl.load(vPtr, j * (N / G) * D + kvHead * D + d)
          result = result + attn * vv
          j = j + 1
        }
        tl.store(outPtr, row * H * D + head * D + d, result)
        d = d + 1
      }
      ()
    }

    // 23. Multi-Query Attention
    @TritonKernelMacro(name = "multiQueryAttentionKernelSPtr", gridType = "1D")
    def multiQueryAttentionKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, D: Int, H: Int): Unit = {
      val i = tl.program_id(0)
      val head = i % H
      val row = i / H
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * H * D + head * D + d)
          val kv = tl.load(kPtr, j * D + d)
          dot = dot + qv * kv
          d = d + 1
        }
        val score = dot * scale
        if (score > scoreMax) scoreMax = score
        j = j + 1
      }
      var expSum: Float = 0.0f
      j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * H * D + head * D + d)
          val kv = tl.load(kPtr, j * D + d)
          dot = dot + qv * kv
          d = d + 1
        }
        val score = dot * scale
        expSum = expSum + exp(score - scoreMax)
        j = j + 1
      }
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
          while (dk < D) {
            val qv = tl.load(qPtr, row * H * D + head * D + dk)
            val kv = tl.load(kPtr, j * D + dk)
            dot = dot + qv * kv
            dk = dk + 1
          }
          val score = dot * scale
          val expScore = exp(score - scoreMax)
          val attn = expScore / expSum
          val vv = tl.load(vPtr, j * D + d)
          result = result + attn * vv
          j = j + 1
        }
        tl.store(outPtr, row * H * D + head * D + d, result)
        d = d + 1
      }
      ()
    }

    // 24. Cross Attention
    @TritonKernelMacro(name = "crossAttentionKernelSPtr", gridType = "1D")
    def crossAttentionKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, M: Int, D: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < M) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * D + d)
          val kv = tl.load(kPtr, j * D + d)
          dot = dot + qv * kv
          d = d + 1
        }
        val score = dot * scale
        if (score > scoreMax) scoreMax = score
        j = j + 1
      }
      var expSum: Float = 0.0f
      j = 0
      while (j < M) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * D + d)
          val kv = tl.load(kPtr, j * D + d)
          dot = dot + qv * kv
          d = d + 1
        }
        val score = dot * scale
        expSum = expSum + exp(score - scoreMax)
        j = j + 1
      }
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < M) {
          var dot: Float = 0.0f
          var dk = 0
          while (dk < D) {
            val qv = tl.load(qPtr, row * D + dk)
            val kv = tl.load(kPtr, j * D + dk)
            dot = dot + qv * kv
            dk = dk + 1
          }
          val score = dot * scale
          val expScore = exp(score - scoreMax)
          val attn = expScore / expSum
          val vv = tl.load(vPtr, j * D + d)
          result = result + attn * vv
          j = j + 1
        }
        tl.store(outPtr, row * D + d, result)
        d = d + 1
      }
      ()
    }

    // 25. Bidirectional Attention
    @TritonKernelMacro(name = "bidirectionalAttentionKernelSPtr", gridType = "1D")
    def bidirectionalAttentionKernelPtr(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        N: Int, D: Int): Unit = {
      val i = tl.program_id(0)
      val row = i
      if (row >= N) return
      val scale = 1.0f / sqrt(D.toFloat)
      var scoreMax: Float = -3.4e38f
      var j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * D + d)
          val kv = tl.load(kPtr, j * D + d)
          dot = dot + qv * kv
          d = d + 1
        }
        val score = dot * scale
        if (score > scoreMax) scoreMax = score
        j = j + 1
      }
      var expSum: Float = 0.0f
      j = 0
      while (j < N) {
        var dot: Float = 0.0f
        var d = 0
        while (d < D) {
          val qv = tl.load(qPtr, row * D + d)
          val kv = tl.load(kPtr, j * D + d)
          dot = dot + qv * kv
          d = d + 1
        }
        val score = dot * scale
        expSum = expSum + exp(score - scoreMax)
        j = j + 1
      }
      var d = 0
      while (d < D) {
        var result: Float = 0.0f
        j = 0
        while (j < N) {
          var dot: Float = 0.0f
          var dk = 0
          while (dk < D) {
            val qv = tl.load(qPtr, row * D + dk)
            val kv = tl.load(kPtr, j * D + dk)
            dot = dot + qv * kv
            dk = dk + 1
          }
          val score = dot * scale
          val expScore = exp(score - scoreMax)
          val attn = expScore / expSum
          val vv = tl.load(vPtr, j * D + d)
          result = result + attn * vv
          j = j + 1
        }
        tl.store(outPtr, row * D + d, result)
        d = d + 1
      }
      ()
    }

    println("\n" + "=" * 80)
    println("All Attention kernels (SPtr variant) defined successfully!")
    println("=" * 80)
  }
}
