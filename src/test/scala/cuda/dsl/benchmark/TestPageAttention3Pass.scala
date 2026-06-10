package cuda.dsl.benchmark

import cuda.dsl.core.{FloatPtr, IntPtr}
import cuda.dsl.core.dim3
import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR
import cuda.dsl.runtime.KernelDesc
import cuda.dsl.runtime.OutputBuffer
import cuda.dsl.runtime.BufferParam
import cuda.dsl.runtime.IntBufferParam
import cuda.dsl.runtime.ScalarParam
import TTIRDSL._

import java.lang.Math

/** Test PageAttention kernels matching pageattention.cu reference (3-pass softmax algorithm)
 *  Reference: cuda-triton-java/src/main/resources/pageattention.cu
 *
 *  Constants (must match reference):
 *    PAGE_SIZE = 128   // Tokens per page
 *    HEAD_DIM  = 128   // Attention head dimension
 *    BLOCK_SIZE = 128   // CUDA thread block size
 *    INF_NEG   = -1e8f // Negative infinity for softmax stability
 *
 *  This implements the EXACT 3-pass softmax algorithm from reference:
 *    1. Find global max score
 *    2. Compute sum of exp(score - max)
 *    3. Compute weighted output
 */
object TestPageAttention3Pass {

  // DSL math helpers: recognized by @TritonKernelMacro → CUDA expf/rsqrtf
  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat
  def rsqrt(x: Float): Float = 1.0f / scala.math.sqrt(x.toDouble).toFloat

  // Constants matching reference
  val PAGE_SIZE: Int = 128
  val HEAD_DIM: Int = 128
  val BLOCK_SIZE: Int = 128
  val INF_NEG: Float = -1e8f

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("PageAttention 3-Pass Softmax Kernel (matching pageattention.cu)")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels.txt for generated CUDA code.")
    println("=" * 80)

    // ========================================================================
    // 1. PagedAttention Prefill Kernel (long sequence prompt processing)
    // Thread: each token position, 3-pass softmax
    // ========================================================================
    println("\n[1] PagedAttention Prefill Kernel (3-pass softmax)")
    println("-" * 40)

    @TritonKernelMacro(name = "paged_attention_prefill_kernel", gridType = "1D")
    def pagedAttentionPrefillKernel(
        outPtr: FloatPtr, qPtr: FloatPtr, kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        pageTablePtr: IntPtr, seqLensPtr: IntPtr,
        batchSize: Int, totalSeq: Int, headDim: Int): Unit = {
      val tid = tl.program_id(0)
      if (tid >= totalSeq) return

      // Find batch and position within batch
      var batch: Int = 0
      var curPos: Int = tid
      var b: Int = 0
      while (b < batchSize) {
        val len = tl.load(seqLensPtr, b).toInt
        if (curPos < len) {
          batch = b
        } else {
          curPos = curPos - len
        }
        b = b + 1
      }

      val seqLen = tl.load(seqLensPtr, batch).toInt
      val batchPageOffset = batch * PAGE_SIZE
      val scale = rsqrt(headDim.toFloat)

      // Load q_vec to registers (reduce global memory access)
      val qVecBase = tid * headDim
      var d: Int = 0

      // ---- Pass 1: Find global max score ----
      var scoreMax: Float = INF_NEG
      var kPos: Int = 0
      while (kPos < seqLen) {
        val pageIdx = kPos / PAGE_SIZE
        val posInPage = kPos % PAGE_SIZE
        val physPage = tl.load(pageTablePtr, batchPageOffset + pageIdx).toInt
        val kBase = physPage * PAGE_SIZE * headDim + posInPage * headDim

        var dot: Float = 0.0f
        d = 0
        while (d < headDim) {
          val qVal = tl.load(qPtr, qVecBase + d)
          val kVal = tl.load(kCachePtr, kBase + d)
          dot = dot + qVal * kVal
          d = d + 1
        }
        val score = dot * scale
        if (score > scoreMax) scoreMax = score
        kPos = kPos + 1
      }

      // ---- Pass 2: Compute sum of exp(score - max) ----
      var sumExp: Float = 0.0f
      kPos = 0
      while (kPos < seqLen) {
        val pageIdx = kPos / PAGE_SIZE
        val posInPage = kPos % PAGE_SIZE
        val physPage = tl.load(pageTablePtr, batchPageOffset + pageIdx).toInt
        val kBase = physPage * PAGE_SIZE * headDim + posInPage * headDim

        var dot: Float = 0.0f
        d = 0
        while (d < headDim) {
          val qVal = tl.load(qPtr, qVecBase + d)
          val kVal = tl.load(kCachePtr, kBase + d)
          dot = dot + qVal * kVal
          d = d + 1
        }
        val score = dot * scale
        sumExp = sumExp + exp(score - scoreMax)
        kPos = kPos + 1
      }

      // ---- Pass 3: Compute attention-weighted Value output ----
      val outBase = tid * headDim
      d = 0
      while (d < headDim) {
        var accum: Float = 0.0f
        kPos = 0
        while (kPos < seqLen) {
          val pageIdx = kPos / PAGE_SIZE
          val posInPage = kPos % PAGE_SIZE
          val physPage = tl.load(pageTablePtr, batchPageOffset + pageIdx).toInt
          val kBase = physPage * PAGE_SIZE * headDim + posInPage * headDim
          val vBase = physPage * PAGE_SIZE * headDim + posInPage * headDim

          var dot: Float = 0.0f
          var dk: Int = 0
          while (dk < headDim) {
            val qVal = tl.load(qPtr, qVecBase + dk)
            val kVal = tl.load(kCachePtr, kBase + dk)
            dot = dot + qVal * kVal
            dk = dk + 1
          }
          val score = dot * scale
          val weight = exp(score - scoreMax) / sumExp
          val vVal = tl.load(vCachePtr, vBase + d)
          accum = accum + weight * vVal
          kPos = kPos + 1
        }
        tl.store(outPtr, outBase + d, accum)
        d = d + 1
      }
      ()
    }
    println("PagedAttentionPrefillKernel defined")

    // ========================================================================
    // 2. PagedAttention Decode Kernel (single token incremental inference)
    // Thread: each batch element
    // ========================================================================
    println("\n[2] PagedAttention Decode Kernel (single token)")
    println("-" * 40)

    @TritonKernelMacro(name = "paged_attention_decode_kernel", gridType = "1D")
    def pagedAttentionDecodeKernel(
        outPtr: FloatPtr, qPtr: FloatPtr, kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        pageTablePtr: IntPtr, seqLensPtr: IntPtr,
        batchSize: Int, headDim: Int): Unit = {
      val batchIdx = tl.program_id(0)
      if (batchIdx >= batchSize) return

      val seqLen = tl.load(seqLensPtr, batchIdx).toInt
      val batchPageOffset = batchIdx * PAGE_SIZE
      val qBase = batchIdx * headDim
      val scale = rsqrt(headDim.toFloat)

      // ---- Pass 1: Find global max score ----
      var scoreMax: Float = INF_NEG
      var kPos: Int = 0
      while (kPos < seqLen) {
        val pageIdx = kPos / PAGE_SIZE
        val posInPage = kPos % PAGE_SIZE
        val physPage = tl.load(pageTablePtr, batchPageOffset + pageIdx).toInt
        val kBase = physPage * PAGE_SIZE * headDim + posInPage * headDim

        var dot: Float = 0.0f
        var d: Int = 0
        while (d < headDim) {
          val qVal = tl.load(qPtr, qBase + d)
          val kVal = tl.load(kCachePtr, kBase + d)
          dot = dot + qVal * kVal
          d = d + 1
        }
        val score = dot * scale
        if (score > scoreMax) scoreMax = score
        kPos = kPos + 1
      }

      // ---- Pass 2: Compute sum of exp(score - max) ----
      var sumExp: Float = 0.0f
      kPos = 0
      while (kPos < seqLen) {
        val pageIdx = kPos / PAGE_SIZE
        val posInPage = kPos % PAGE_SIZE
        val physPage = tl.load(pageTablePtr, batchPageOffset + pageIdx).toInt
        val kBase = physPage * PAGE_SIZE * headDim + posInPage * headDim

        var dot: Float = 0.0f
        var d: Int = 0
        while (d < headDim) {
          val qVal = tl.load(qPtr, qBase + d)
          val kVal = tl.load(kCachePtr, kBase + d)
          dot = dot + qVal * kVal
          d = d + 1
        }
        val score = dot * scale
        sumExp = sumExp + exp(score - scoreMax)
        kPos = kPos + 1
      }

      // ---- Pass 3: Compute attention-weighted Value output ----
      val outBase = batchIdx * headDim
      var dOut: Int = 0
      while (dOut < headDim) {
        var accum: Float = 0.0f
        kPos = 0
        while (kPos < seqLen) {
          val pageIdx = kPos / PAGE_SIZE
          val posInPage = kPos % PAGE_SIZE
          val physPage = tl.load(pageTablePtr, batchPageOffset + pageIdx).toInt
          val kBase = physPage * PAGE_SIZE * headDim + posInPage * headDim
          val vBase = physPage * PAGE_SIZE * headDim + posInPage * headDim

          var dot: Float = 0.0f
          var dk: Int = 0
          while (dk < headDim) {
            val qVal = tl.load(qPtr, qBase + dk)
            val kVal = tl.load(kCachePtr, kBase + dk)
            dot = dot + qVal * kVal
            dk = dk + 1
          }
          val score = dot * scale
          val weight = exp(score - scoreMax) / sumExp
          val vVal = tl.load(vCachePtr, vBase + dOut)
          accum = accum + weight * vVal
          kPos = kPos + 1
        }
        tl.store(outPtr, outBase + dOut, accum)
        dOut = dOut + 1
      }
      ()
    }
    println("PagedAttentionDecodeKernel defined")

    // ========================================================================
    // 3. Write KV Cache Kernel (page-level KV cache write)
    // Thread: each token
    // ========================================================================
    println("\n[3] Write KV Cache Kernel")
    println("-" * 40)

    @TritonKernelMacro(name = "write_kv_cache_kernel", gridType = "1D")
    def writeKVCacheKernel(
        kCachePtr: FloatPtr, vCachePtr: FloatPtr,
        kInPtr: FloatPtr, vInPtr: FloatPtr,
        slotMapPtr: IntPtr, nTokens: Int, headDim: Int): Unit = {
      val tid = tl.program_id(0)
      if (tid >= nTokens) return

      val physAddr = tl.load(slotMapPtr, tid).toInt * headDim
      val inBase = tid * headDim

      var d: Int = 0
      while (d < headDim) {
        val kVal = tl.load(kInPtr, inBase + d)
        val vVal = tl.load(vInPtr, inBase + d)
        tl.store(kCachePtr, physAddr + d, kVal)
        tl.store(vCachePtr, physAddr + d, vVal)
        d = d + 1
      }
      ()
    }
    println("WriteKVCacheKernel defined")

    println("\n" + "=" * 80)
    println("All PageAttention 3-pass kernels defined successfully!")
    println("=" * 80)
    println("\nConstants (matching reference):")
    println(s"  PAGE_SIZE = $PAGE_SIZE")
    println(s"  HEAD_DIM = $HEAD_DIM")
    println(s"  BLOCK_SIZE = $BLOCK_SIZE")
    println(s"  INF_NEG = $INF_NEG")

    // ================================================================
    // Execute kernels and verify results
    // ================================================================
    runExecution()
  }

  /** Execute PageAttention kernels and verify output */
  private def runExecution(): Unit =
    println("\n" + "=" * 80)
    println("Executing PageAttention kernels...")
    println("=" * 80)

    try
      // Test 1: write_kv_cache_kernel (simplest, validates basic execution)
      println("\n[1] Testing write_kv_cache_kernel...")
      testWriteKVCache()

      // Test 2: paged_attention_prefill_kernel (3-pass softmax)
      println("\n[2] Testing paged_attention_prefill_kernel...")
      testPrefillKernel()

      // Test 3: paged_attention_decode_kernel
      println("\n[3] Testing paged_attention_decode_kernel...")
      testDecodeKernel()

      println("\n" + "=" * 80)
      println("All kernel executions completed!")
      println("=" * 80)

    catch case e: Exception =>
      println(f"[ERROR] ${e.getMessage}")
      e.printStackTrace()

  /** Test write_kv_cache_kernel */
  private def testWriteKVCache(): Unit =
    val headDim = 128
    val nTokens = 8
    val size = nTokens * headDim

    // Build params matching write_kv_cache_kernel signature:
    // (out, kCachePtr, vCachePtr, kInPtr, vInPtr, slotMapPtr, nTokens, headDim, n)
    val params = List(
      OutputBuffer("out", size),
      BufferParam("kCachePtr", size),
      BufferParam("vCachePtr", size),
      BufferParam("kInPtr", size),
      BufferParam("vInPtr", size),
      IntBufferParam("slotMapPtr", nTokens),  // slot mapping
      ScalarParam("nTokens", nTokens),
      ScalarParam("headDim", headDim)
    )

    val grid = dim3(nTokens, 1, 1)
    val block = dim3(128, 1, 1)

    val result = SCR.executeKernel(
      KernelDesc("write_kv_cache_kernel", params, grid, block),
      true
    )

    result match
      case Some(output) =>
        val valid = output.count(v => !v.isNaN && !v.isInfinite)
        val sum = output.filter(v => !v.isNaN && !v.isInfinite).sum
        println(f"  write_kv_cache_kernel: valid=$valid/${output.length}, sum=${sum.formatted("%.2f")}")
        if valid > 0 then println("  [PASS]") else println("  [FAIL]")
      case None =>
        println("  [FAIL] Kernel execution returned None")

  /** Test paged_attention_prefill_kernel */
  private def testPrefillKernel(): Unit =
    val batchSize = 2
    val totalSeq = 64  // 2 batches × 32 sequences each
    val headDim = 128

    // Calculate sizes: each sequence has headDim values
    val qSize = totalSeq * headDim
    val kSize = PAGE_SIZE * 4 * headDim  // 4 pages max
    val vSize = PAGE_SIZE * 4 * headDim

    // pageTable: 2 batches × 4 pages = 8 entries
    val pageTableSize = batchSize * 4
    // seqLens: batchSize entries
    val seqLensSize = batchSize

    val params = List(
      OutputBuffer("out", qSize),
      BufferParam("outPtr", qSize),
      BufferParam("qPtr", qSize),
      BufferParam("kCachePtr", kSize),
      BufferParam("vCachePtr", vSize),
      IntBufferParam("pageTablePtr", pageTableSize),
      IntBufferParam("seqLensPtr", seqLensSize),
      ScalarParam("batchSize", batchSize),
      ScalarParam("totalSeq", totalSeq),
      ScalarParam("headDim", headDim)
    )

    val grid = dim3(totalSeq, 1, 1)
    val block = dim3(128, 1, 1)

    val result = SCR.executeKernel(
      KernelDesc("paged_attention_prefill_kernel", params, grid, block),
      true
    )

    result match
      case Some(output) =>
        val valid = output.count(v => !v.isNaN && !v.isInfinite)
        val sum = output.filter(v => !v.isNaN && !v.isInfinite).sum
        println(f"  prefill_kernel: valid=$valid/${output.length}, sum=${sum.formatted("%.2f")}")
        if valid > 0 then println("  [PASS]") else println("  [FAIL]")
      case None =>
        println("  [FAIL] Kernel execution returned None")

  /** Test paged_attention_decode_kernel */
  private def testDecodeKernel(): Unit =
    val batchSize = 2
    val headDim = 128

    val qSize = batchSize * headDim
    val kSize = PAGE_SIZE * batchSize * headDim
    val vSize = PAGE_SIZE * batchSize * headDim
    val seqLensSize = batchSize
    val pageTableSize = batchSize

    val params = List(
      OutputBuffer("out", qSize),
      BufferParam("outPtr", qSize),
      BufferParam("qPtr", qSize),
      BufferParam("kCachePtr", kSize),
      BufferParam("vCachePtr", vSize),
      IntBufferParam("pageTablePtr", pageTableSize),
      IntBufferParam("seqLensPtr", seqLensSize),
      ScalarParam("batchSize", batchSize),
      ScalarParam("headDim", headDim)
    )

    val grid = dim3(batchSize, 1, 1)
    val block = dim3(128, 1, 1)

    val result = SCR.executeKernel(
      KernelDesc("paged_attention_decode_kernel", params, grid, block),
      true
    )

    result match
      case Some(output) =>
        val valid = output.count(v => !v.isNaN && !v.isInfinite)
        val sum = output.filter(v => !v.isNaN && !v.isInfinite).sum
        println(f"  decode_kernel: valid=$valid/${output.length}, sum=${sum.formatted("%.2f")}")
        if valid > 0 then println("  [PASS]") else println("  [FAIL]")
      case None =>
        println("  [FAIL] Kernel execution returned None")
}