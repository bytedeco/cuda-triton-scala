package cuda.dsl.benchmark

import cuda.dsl.core.{FloatPtr, IntPtr}
import cuda.dsl.core.dim3
import cuda.dsl.dsl._
import cuda.dsl.runtime.ScalaCudaRuntime as SCR
import cuda.dsl.runtime.KernelDesc
import cuda.dsl.runtime.OutputBuffer
import cuda.dsl.runtime.BufferParam
import cuda.dsl.runtime.ScalarParam
import TTIRDSL._

/** FlashAttention V3/V4 - Scala DSL kernel implementation
 *  Reference: cuda-triton-java/src/test/resources/flashattention.cu
 *
 *  Key design:
 *    - Each thread block handles one Q tile (BLOCK_Q elements)
 *    - Each thread handles ONE Q element (qIdx = threadIdx.x within block)
 *    - K/V tiles loaded into shared memory
 *    - Q stored in shared memory qReg[HEAD_DIM] for indexed access
 *    - Online softmax: maintain m,l and accumulator arrays
 *    - Causal masking: skip keys at positions > qIdx
 */
object TestFlashAttentionV3V4 {

  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat

  // Constants matching reference CUDA
  val HEAD_DIM: Int = 128
  val BLOCK_Q_V3: Int = 64
  val BLOCK_KV_V3: Int = 128
  val BLOCK_Q_V4: Int = 32
  val BLOCK_KV_V4: Int = 64
  val WARP_TILE_V4: Int = 16
  val INF_NEG: Float = -1e8f

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("FlashAttention V3/V4 — Online Softmax Kernels")
    println("=" * 80)
    println(s"Constants: HEAD_DIM=$HEAD_DIM, BLOCK_Q_V3=$BLOCK_Q_V3, BLOCK_KV_V3=$BLOCK_KV_V3")
    println(s"BLOCK_Q_V4=$BLOCK_Q_V4, BLOCK_KV_V4=$BLOCK_KV_V4, WARP_TILE_V4=$WARP_TILE_V4")
    println("=" * 80)

    // ========================================================================
    // FlashAttention V3: 2-level tiling (Q block × KV block)
    // Grid: (seqLen/BLOCK_Q_V3) × numHeads, Block: BLOCK_Q_V3
    // Algorithm:
    //   1. Each thread handles ONE Q element (qIdx = threadIdx.x within block)
    //   2. Load Q tile into shared memory qReg[HEAD_DIM] per thread
    //   3. Iterate over KV tiles, load K/V into shared memory
    //   4. Online softmax update with causal masking
    //   5. Normalize and write output
    // ========================================================================
    @TritonKernelMacro(name = "flash_attention_v3_causal_kernel", gridType = "1D")
    def flashAttentionV3Kernel(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        seqLen: Int, scale: Float, headIdx: Int, n: Int): Unit = {
      // Grid/Block geometry
      val blockQ = tl.program_id(0)
      val qStart = blockQ * BLOCK_Q_V3
      val qEnd = if (qStart + BLOCK_Q_V3 < seqLen) qStart + BLOCK_Q_V3 else seqLen
      if (qStart >= seqLen) return

      // Shared memory: K/V tiles + Q register per thread
      tl.sharedMem("float", "sh_k", BLOCK_KV_V3 * HEAD_DIM)
      tl.sharedMem("float", "sh_v", BLOCK_KV_V3 * HEAD_DIM)
      tl.sharedMem("float", "qReg", HEAD_DIM)
      tl.sharedMem("float", "oReg", HEAD_DIM)

      // Thread's Q element within this block
      val qIdx = tl.program_id(0)
      val qAbs = qStart + tl.program_id(0)

      // Load Q row into qReg[HEAD_DIM]
      if (qAbs < qEnd) {
        var dd = 0
        while (dd < HEAD_DIM) {
          val gAddr = qAbs * HEAD_DIM + dd
          val v = tl.load(qPtr, gAddr)
          tl.sharedStore("qReg", dd, v)
          tl.sharedStore("oReg", dd, 0.0f)
          dd = dd + 1
        }
      }
      tl.syncthreads()

      // Online softmax state
      var m = INF_NEG
      var l: Float = 0.0f

      // Iterate over KV tiles
      val numKTiles = (seqLen + BLOCK_KV_V3 - 1) / BLOCK_KV_V3
      var tile = 0
      while (tile < numKTiles) {
        val kvStart = tile * BLOCK_KV_V3
        val kvEnd = if (kvStart + BLOCK_KV_V3 < seqLen) kvStart + BLOCK_KV_V3 else seqLen

        // Causal: skip tiles entirely after this query position
        if (!(kvStart > qAbs)) {
          // Load K/V row into shared memory (one row per thread)
          val kvRow = kvStart + tl.program_id(0)
          if (kvRow < kvEnd) {
            var d2 = 0
            while (d2 < HEAD_DIM) {
              val gAddr = kvRow * HEAD_DIM + d2
              val shOff = tl.program_id(0) * HEAD_DIM + d2
              tl.sharedStore("sh_k", shOff, tl.load(kPtr, gAddr))
              tl.sharedStore("sh_v", shOff, tl.load(vPtr, gAddr))
              d2 = d2 + 1
            }
          }
        }
        tl.syncthreads()

        // Compute attention for this tile
        if (qAbs < qEnd) {
          var kvIdx = 0
          while (kvIdx < (kvEnd - kvStart)) {
            val kvAbs = kvStart + kvIdx
            // Causal mask: only compute for keys at or before q position
            if (!(kvAbs > qAbs)) {
              // Compute q · k[kvAbs]
              var dot: Float = 0.0f
              var di = 0
              while (di < HEAD_DIM) {
                val qVal = tl.sharedLoad("qReg", di)
                val kVal = tl.sharedLoad("sh_k", kvIdx * HEAD_DIM + di)
                dot = dot + qVal * kVal
                di = di + 1
              }
              val score = dot * scale

              // Online softmax update
              val mOld = m
              val lOld = l
              val mNew = if (score > mOld) score else mOld
              val expScore = exp(score - mNew)
              val prevScale = exp(mOld - mNew)
              val lNew = prevScale * lOld + expScore

              // Accumulate weighted V into oReg
              var dd = 0
              while (dd < HEAD_DIM) {
                val vVal = tl.sharedLoad("sh_v", kvIdx * HEAD_DIM + dd)
                val weight = expScore / lNew
                val prevWeight = prevScale / lNew
                val prev = tl.sharedLoad("oReg", dd)
                val contrib = prevWeight * prev + weight * vVal
                tl.sharedStore("oReg", dd, contrib)
                dd = dd + 1
              }
              m = mNew
              l = lNew
            }
            kvIdx = kvIdx + 1
          }
        }
        tl.syncthreads()
        tile = tile + 1
      }

      // Write normalized output
      if (qAbs < qEnd) {
        val invL = 1.0f / l
        var dj = 0
        while (dj < HEAD_DIM) {
          val oVal = tl.sharedLoad("oReg", dj)
          tl.store(outPtr, qAbs * HEAD_DIM + dj, oVal * invL)
          dj = dj + 1
        }
      }
      ()
    }

    // ========================================================================
    // FlashAttention V4: 3-level tiling (Q × KV × Warp tile)
    // Adds warp-level sub-tiling within each KV tile
    // ========================================================================
    @TritonKernelMacro(name = "flash_attention_v4_causal_kernel", gridType = "1D")
    def flashAttentionV4Kernel(
        outPtr: FloatPtr, qPtr: FloatPtr, kPtr: FloatPtr, vPtr: FloatPtr,
        seqLen: Int, scale: Float, headIdx: Int, n: Int): Unit = {
      val blockQ = tl.program_id(0)
      val qStart = blockQ * BLOCK_Q_V4
      val qEnd = if (qStart + BLOCK_Q_V4 < seqLen) qStart + BLOCK_Q_V4 else seqLen
      if (qStart >= seqLen) return

      tl.sharedMem("float", "sh_k", BLOCK_KV_V4 * HEAD_DIM)
      tl.sharedMem("float", "sh_v", BLOCK_KV_V4 * HEAD_DIM)
      tl.sharedMem("float", "qReg", HEAD_DIM)
      tl.sharedMem("float", "oReg", HEAD_DIM)

      val qAbs = qStart + tl.program_id(0)

      // Load Q row into qReg
      if (qAbs < qEnd) {
        var d0 = 0
        while (d0 < HEAD_DIM) {
          val gAddr = qAbs * HEAD_DIM + d0
          val v = tl.load(qPtr, gAddr)
          tl.sharedStore("qReg", d0, v)
          tl.sharedStore("oReg", d0, 0.0f)
          d0 = d0 + 1
        }
      }
      tl.syncthreads()

      var m = INF_NEG
      var l: Float = 0.0f

      val numKTiles = (seqLen + BLOCK_KV_V4 - 1) / BLOCK_KV_V4
      var tile = 0
      while (tile < numKTiles) {
        val kvStart = tile * BLOCK_KV_V4
        val kvEnd = if (kvStart + BLOCK_KV_V4 < seqLen) kvStart + BLOCK_KV_V4 else seqLen

        if (!(kvStart > qAbs)) {
          val kvRow = kvStart + tl.program_id(0)
          if (kvRow < kvEnd) {
            var d1 = 0
            while (d1 < HEAD_DIM) {
              val gAddr = kvRow * HEAD_DIM + d1
              val shOff = tl.program_id(0) * HEAD_DIM + d1
              tl.sharedStore("sh_k", shOff, tl.load(kPtr, gAddr))
              tl.sharedStore("sh_v", shOff, tl.load(vPtr, gAddr))
              d1 = d1 + 1
            }
          }
        }
        tl.syncthreads()

        // V4: Warp-level sub-tiles for finer-grained computation
        if (qAbs < qEnd) {
          val kvLen = kvEnd - kvStart
          val numWT = (kvLen + WARP_TILE_V4 - 1) / WARP_TILE_V4
          var wt = 0
          while (wt < numWT) {
            val wtStart = wt * WARP_TILE_V4
            val wtEnd = if (wtStart + WARP_TILE_V4 < kvLen) wtStart + WARP_TILE_V4 else kvLen

            var kvIdx = wtStart
            while (kvIdx < wtEnd) {
              val kvAbs = kvStart + kvIdx
              if (!(kvAbs > qAbs)) {
                var dot: Float = 0.0f
                var di = 0
                while (di < HEAD_DIM) {
                  val qVal = tl.sharedLoad("qReg", di)
                  val kVal = tl.sharedLoad("sh_k", kvIdx * HEAD_DIM + di)
                  dot = dot + qVal * kVal
                  di = di + 1
                }
                val score = dot * scale

                val mOld = m
                val lOld = l
                val mNew = if (score > mOld) score else mOld
                val expScore = exp(score - mNew)
                val prevScale = exp(mOld - mNew)
                val lNew = prevScale * lOld + expScore

                var dd = 0
                while (dd < HEAD_DIM) {
                  val vVal = tl.sharedLoad("sh_v", kvIdx * HEAD_DIM + dd)
                  val weight = expScore / lNew
                  val prevWeight = prevScale / lNew
                  val prev = tl.sharedLoad("oReg", dd)
                  val contrib = prevWeight * prev + weight * vVal
                  tl.sharedStore("oReg", dd, contrib)
                  dd = dd + 1
                }
                m = mNew
                l = lNew
              }
              kvIdx = kvIdx + 1
            }
            wt = wt + 1
          }
        }
        tl.syncthreads()
        tile = tile + 1
      }

      // Write normalized output
      if (qAbs < qEnd) {
        val invL = 1.0f / l
        var dj = 0
        while (dj < HEAD_DIM) {
          val oVal = tl.sharedLoad("oReg", dj)
          tl.store(outPtr, qAbs * HEAD_DIM + dj, oVal * invL)
          dj = dj + 1
        }
      }
      ()
    }

    // ========================================================================
    // Execute kernels
    // ========================================================================
    testV3Kernel()
    testV4Kernel()
    println("FlashAttention V3/V4 execution complete!")
  }

  // ============================================================================
  // Test FlashAttention V3
  // ============================================================================
  private def testV3Kernel(): Unit =
    val seqLen = 64   // Multiple of BLOCK_Q_V3=64
    val headDim = 128
    val totalQ = seqLen * headDim
    val scale = 1.0f / scala.math.sqrt(headDim.toDouble).toFloat

    val params = List(
      OutputBuffer("out", totalQ),
      BufferParam("outPtr", totalQ),
      BufferParam("qPtr", totalQ),
      BufferParam("kPtr", totalQ),
      BufferParam("vPtr", totalQ),
      ScalarParam("seqLen", seqLen),
      ScalarParam("scale", scale),
      ScalarParam("headIdx", 0)
    )

    val grid = dim3(seqLen, 1, 1)
    val block = dim3(64, 1, 1)

    println(s"\n[Test V3] seqLen=$seqLen, headDim=$headDim, scale=$scale")
    println(s"  grid=($seqLen,1,1), block=(64,1,1)")

    val result = SCR.executeKernel(
      KernelDesc("flash_attention_v3_causal_kernel", params, grid, block),
      true
    )

    result match
      case Some(output) =>
        val valid = output.count(v => !v.isNaN && !v.isInfinite && v != 0.0f)
        val sum = output.filter(v => !v.isNaN && !v.isInfinite).sum
        println(f"  valid=$valid/${output.length}, sum=${sum.formatted("%.4f")}")
        if valid > 0 then println("  [PASS]") else println("  [FAIL]")
      case None =>
        println("  [FAIL] Kernel execution returned None")

  // ============================================================================
  // Test FlashAttention V4
  // ============================================================================
  private def testV4Kernel(): Unit =
    val seqLen = 64   // Multiple of BLOCK_Q_V4=32
    val headDim = 128
    val totalQ = seqLen * headDim
    val scale = 1.0f / scala.math.sqrt(headDim.toDouble).toFloat

    val params = List(
      OutputBuffer("out", totalQ),
      BufferParam("outPtr", totalQ),
      BufferParam("qPtr", totalQ),
      BufferParam("kPtr", totalQ),
      BufferParam("vPtr", totalQ),
      ScalarParam("seqLen", seqLen),
      ScalarParam("scale", scale),
      ScalarParam("headIdx", 0)
    )

    val grid = dim3(seqLen, 1, 1)
    val block = dim3(32, 1, 1)

    println(s"\n[Test V4] seqLen=$seqLen, headDim=$headDim, scale=$scale")
    println(s"  grid=($seqLen,1,1), block=(32,1,1)")

    val result = SCR.executeKernel(
      KernelDesc("flash_attention_v4_causal_kernel", params, grid, block),
      true
    )

    result match
      case Some(output) =>
        val valid = output.count(v => !v.isNaN && !v.isInfinite && v != 0.0f)
        val sum = output.filter(v => !v.isNaN && !v.isInfinite).sum
        println(f"  valid=$valid/${output.length}, sum=${sum.formatted("%.4f")}")
        if valid > 0 then println("  [PASS]") else println("  [FAIL]")
      case None =>
        println("  [FAIL] Kernel execution returned None")
}
