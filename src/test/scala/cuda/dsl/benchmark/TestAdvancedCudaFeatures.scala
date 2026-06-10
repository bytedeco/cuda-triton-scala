package cuda.dsl.benchmark

import cuda.dsl.core.FloatPtr
import cuda.dsl.dsl.*

/** Test advanced CUDA DSL features: shared memory, warp primitives, for loops */
object TestAdvancedCudaFeatures {

  // DSL math helpers: recognized by @TritonKernelMacro → CUDA expf/sqrtf
  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat
  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("@TritonKernelMacro: Advanced CUDA DSL Features")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels.txt for generated CUDA code.")
    println("=" * 80)

    // ========================================================================
    // Feature 1: For Loop Support (0.until(N) foreach {...})
    // ========================================================================
    println("\n[1] For Loop Kernel (0.until(N) foreach)")
    @TritonKernelMacro
    def forLoopKernel(
        outPtr: Float, inPtr: Float, N: Int, D: Int): Unit = {
      // Scala 3: 0.until(N) foreach { i => ... }
      0.until(N) foreach { i =>
        var sum: Float = 0.0f
        0.until(D) foreach { d =>
          val x = tl.load(inPtr + i * D + d)
          sum = sum + x * x
        }
        val result = sqrt(sum)
        tl.store(outPtr + i, result)
      }
      ()
    }
    println("For loop kernel defined")

    // ========================================================================
    // Feature 2: Shared Memory + Syncthreads (tiled matrix multiply)
    // ========================================================================
    println("\n[2] Shared Memory Kernel (tiled matmul with __syncthreads)")
    @TritonKernelMacro
    def tiledMatmulKernel(
        outPtr: Float, aPtr: Float, bPtr: Float,
        M: Int, N: Int, K: Int,
        blockSize: Int): Unit = {
      // Block indices
      val bx = tl.program_id(0)
      val by = tl.program_id(1)

      // Thread within block
      val tx = tl.program_id(0)  // simplified: 1D

      // Shared memory declaration
      tl.sharedMem("float", "s_a", 256)
      tl.sharedMem("float", "s_b", 256)

      // Global indices
      val row = by * blockSize + tx
      val col = bx * blockSize + tx

      if (row < M && col < N) {
        var sum: Float = 0.0f
        // Loop over tiles
        0.until((K + blockSize - 1) / blockSize) foreach { tile =>
          // Load A tile into shared memory
          val aRow = row
          val aCol = tile * blockSize + tx
          val aIdx = aRow * K + aCol
          val aVal = tl.load(aPtr + aIdx)
          tl.sharedStore("s_a", tx * blockSize + tx, aVal)

          // Load B tile into shared memory
          val bRow = tile * blockSize + tx
          val bCol = col
          val bIdx = bRow * N + bCol
          val bVal = tl.load(bPtr + bIdx)
          tl.sharedStore("s_b", tx * blockSize + tx, bVal)

          // Barrier
          tl.syncthreads()

          // Compute tile contribution
          0.until(blockSize) foreach { k =>
            val aShared = tl.sharedLoad("s_a", tx * blockSize + k)
            val bShared = tl.sharedLoad("s_b", k * blockSize + tx)
            sum = sum + aShared * bShared
          }

          // Barrier before next tile
          tl.syncthreads()
        }
        tl.store(outPtr + row * N + col, sum)
      }
      ()
    }
    println("Tiled matmul kernel defined")

    // ========================================================================
    // Feature 3: Warp Shuffle Primitives (parallel reduction)
    // ========================================================================
    println("\n[3] Warp Shuffle Kernel (parallel reduction with __shfl)")
    @TritonKernelMacro
    def warpReduceKernel(
        outPtr: Float, inPtr: Float, N: Int): Unit = {
      val tid = tl.program_id(0)
      val lane = tid % 32

      // Load value
      var value = tl.load(inPtr + tid)

      // Warp-level reduction using shuffle down
      var offset = 16
      while (offset > 0) {
        val other = tl.shfl_down(value, offset, 32)
        value = value + other
        offset = offset / 2
      }

      // Store result (lane 0 has final sum)
      if (lane == 0) {
        tl.store(outPtr + (tid / 32), value)
      }
      ()
    }
    println("Warp reduce kernel defined")

    // ========================================================================
    // Feature 4: Warp Vote Primitives (predicate broadcasting)
    // ========================================================================
    println("\n[4] Warp Vote Kernel (__any_sync / __all_sync)")
    @TritonKernelMacro
    def warpVoteKernel(
        outPtr: Float, inPtr: Float, N: Int): Unit = {
      val tid = tl.program_id(0)
      val lane = tid % 32

      val threshold: Float = 0.5f
      val value = tl.load(inPtr + tid)
      val predicate = value > threshold

      // Warp any: does any thread in warp have predicate true?
      val anyResult = tl.warp_any(predicate)
      // Warp all: do all threads in warp have predicate true?
      val allResult = tl.warp_all(predicate)

      // Store results (only lane 0 writes)
      if (lane == 0) {
        tl.store(outPtr + (tid / 32) * 2, if (anyResult) 1.0f else 0.0f)
        tl.store(outPtr + (tid / 32) * 2 + 1, if (allResult) 1.0f else 0.0f)
      }
      ()
    }
    println("Warp vote kernel defined")

    // ========================================================================
    // Feature 5: Masked Load/Store (predicated memory access)
    // ========================================================================
    println("\n[5] Masked Load/Store Kernel (predicated memory access)")
    @TritonKernelMacro
    def maskedAccessKernel(
        outPtr: Float, inPtr: Float, maskPtr: Int, N: Int): Unit = {
      val i = tl.program_id(0)

      if (i < N) {
        // Masked load: only load if predicate is true
        val maskActive = true  // simplified predicate
        val value = tl.maskedLoad(inPtr, i, maskActive, 0.0f)
        // Masked store: only store if predicate is true
        tl.maskedStore(outPtr, i, value * 2.0f, maskActive)
      }
      ()
    }
    println("Masked access kernel defined")

    // ========================================================================
    // Feature 6: Warp Shuffle XOR (butterfly reduction pattern)
    // ========================================================================
    println("\n[6] Warp Butterfly Kernel (__shfl_xor)")
    @TritonKernelMacro
    def warpButterflyKernel(
        outPtr: Float, inPtr: Float, N: Int): Unit = {
      val tid = tl.program_id(0)
      val lane = tid % 32

      var value = tl.load(inPtr + tid)

      // Butterfly reduction: xor-based shuffle
      // Sum across warp using xor shuffle
      var mask = 1
      while (mask < 32) {
        val other = tl.shfl_xor(value, mask, 32)
        value = value + other
        mask = mask * 2
      }

      if (lane == 0) {
        tl.store(outPtr + (tid / 32), value)
      }
      ()
    }
    println("Warp butterfly kernel defined")

    // ========================================================================
    // Feature 7: Shared Memory + Warp Shuffle combined
    //    (block-level reduction using shared memory + warp shuffle)
    // ========================================================================
    println("\n[7] Block Reduction Kernel (shared mem + warp shuffle)")
    @TritonKernelMacro
    def blockReduceKernel(
        outPtr: Float, inPtr: Float, N: Int, blockSize: Int): Unit = {
      val tid = tl.program_id(0)
      val lane = tid % 32
      val warpId = tid / 32

      // Shared memory for per-warp partial results
      tl.sharedMem("float", "s_warp", 16)  // max 16 warps per block

      // Load and do warp-level reduction
      var value = tl.load(inPtr + tid)
      var offset = 16
      while (offset > 0) {
        val other = tl.shfl_down(value, offset, 32)
        value = value + other
        offset = offset / 2
      }

      // Lane 0 of each warp writes its warp-reduced sum to shared memory
      if (lane == 0) {
        tl.sharedStore("s_warp", warpId, value)
      }

      // Barrier
      tl.syncthreads()

      // Warp 0 does final reduction across warp partial results
      if (warpId == 0 && lane < 16) {
        value = tl.sharedLoad("s_warp", lane)
        offset = 8
        while (offset > 0) {
          val other = tl.shfl_down(value, offset, 32)
          value = value + other
          offset = offset / 2
        }
        if (lane == 0) {
          tl.store(outPtr + (tid / 32), value)
        }
      }

      // Barrier before exit
      tl.syncthreads()
      ()
    }
    println("Block reduction kernel defined")

    // ========================================================================
    println("\n================================================================================")
    println("All advanced CUDA kernels defined successfully!")
    println("================================================================================")
  }
}
