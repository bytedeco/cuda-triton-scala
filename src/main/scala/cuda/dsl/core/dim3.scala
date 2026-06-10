package cuda.dsl.core

import org.bytedeco.cuda.cudart.dim3 as JCudaDim3

/** 3D dimension type for CUDA grid and block dimensions.
  * Extends JavaCPP's dim3 to provide seamless interoperability with CUDA Runtime API.
  *
  * CUDA concepts:
  * - Grid: total number of thread blocks (launched with kernel<<<grid, block>>>)
  * - Block: group of threads that share shared memory and synchronize
  * - Thread: individual execution unit within a block
  * - Warp: group of 32 threads that execute in lockstep (SIMD)
  * - Tensor Core: specialized matrix multiplication unit ( Volta+ )
  */
class dim3(override val x: Int, override val y: Int = 1, override val z: Int = 1)
    extends JCudaDim3(x, y, z) {

  // Constructor for 1D
  def this(x: Int) = this(x, 1, 1)

  // Constructor for 2D
  def this(x: Int, y: Int) = this(x, y, 1)

  inline def product: Int = x * y * z

  override def toString: String = s"dim3($x, $y, $z)"

  override def equals(other: Any): Boolean = other match
    case that: dim3 => this.x == that.x && this.y == that.y && this.z == that.z
    case _ => false

  override def hashCode: Int = (x, y, z).##

  // ---- CUDA Grid/Block/Tensor/Core dimension helpers ----

  /** Total number of blocks in the grid */
  inline def totalBlocks: Int = x * y * z

  /** Total number of threads per block (for standard 1D/2D/3D blocks) */
  inline def threadsPerBlock: Int = x * y * z

  /** Number of warps per block (assuming warp size of 32) */
  inline def warpsPerBlock: Int = (threadsPerBlock + 31) / 32

  /** Estimated SM usage (blocks per SM, assuming 16 max blocks/SM on Ampere) */
  inline def estimatedSMUsage(smCount: Int = 108): Float = totalBlocks.toFloat / smCount

  /** Maximum threads per SM (Ampere: 2048) */
  inline def threadsPerSM: Int = 2048

  /** Occupancy estimate: how many blocks can run concurrently */
  inline def occupancyBlocks(maxBlocksPerSM: Int = 16, threadsPerBlockActual: Int = threadsPerBlock): Int =
    math.min(maxBlocksPerSM, threadsPerSM / math.max(1, threadsPerBlockActual))
}

object dim3 {
  import org.bytedeco.cuda.cudart.dim3 as JCudaDim3

  /** Create dim3 with single value */
  inline def apply(x: Int): dim3 = new dim3(x)

  /** Create dim3 with x, y values */
  inline def apply(x: Int, y: Int): dim3 = new dim3(x, y)

  /** Create dim3 with x, y, z values */
  inline def apply(x: Int, y: Int, z: Int): dim3 = new dim3(x, y, z)

  /** Create dim3 from JavaCPP dim3 (for seamless interop) */
  inline def apply(j: JCudaDim3): dim3 = new dim3(j.x, j.y, j.z)

  // ---- Predefined common CUDA dimensions ----

  /** Maximum threads per block (CUDA limit) */
  val CUDA_MAX_BLOCK_SIZE = 1024

  /** Warp size (constant across all NVIDIA GPUs) */
  val WARP_SIZE = 32

  /** Maximum blocks per SM (Ampere and later) */
  val MAX_BLOCKS_PER_SM = 16

  /** Maximum threads per SM (Ampere) */
  val MAX_THREADS_PER_SM = 2048

  /** Returns a dim3 for a 1D block of given size */
  inline def block1D(n: Int): dim3 = new dim3(math.min(n, CUDA_MAX_BLOCK_SIZE))

  /** Returns a dim3 for a 2D block of given size */
  inline def block2D(x: Int, y: Int): dim3 = new dim3(
    math.min(x, 1024),
    math.min(y, 1024 / x)
  )

  /** Returns a dim3 for a 3D block of given size.
    * Caps z so total threads (x*y*z) <= 1024.
    */
  inline def block3D(x: Int, y: Int, z: Int): dim3 = {
    val cappedZ = math.min(z, 1024 / math.max(1, x * y))
    new dim3(
      math.min(x, 1024),
      math.min(y, 1024 / math.max(1, x)),
      math.max(1, cappedZ)
    )
  }

  // ---- Grid dimension factories ----

  /** Compute optimal 1D grid from N elements and block size */
  inline def grid1D(n: Int, blockSize: Int = 256): dim3 = new dim3((n + blockSize - 1) / blockSize)

  /** Compute 2D grid from rows/cols and block dimensions */
  inline def grid2D(rows: Int, cols: Int, blockX: Int = 16, blockY: Int = 16): dim3 =
    new dim3((cols + blockX - 1) / blockX, (rows + blockY - 1) / blockY)

  /** Compute 3D grid from (rows, cols, depth) and block (x, y, z).
    * Returns dim3(cols/blockX, rows/blockY, depth/blockZ).
    */
  inline def grid3D(rows: Int, cols: Int, depth: Int = 1,
                    blockX: Int = 16, blockY: Int = 16, blockZ: Int = 1): dim3 =
    new dim3(
      math.max(1, (cols + blockX - 1) / blockX),
      math.max(1, (rows + blockY - 1) / blockY),
      math.max(1, (depth + blockZ - 1) / blockZ)
    )

  // ---- Convenience pre-defined grid/block combos ----

  /** Standard CUDA launch: grid(1024, 256) for 1D element-wise ops */
  val STANDARD_1D: (dim3, dim3) = (grid1D(1024), block1D(256))

  /** Standard CUDA launch for GEMM: grid(M/16, N/16), block(16, 16) */
  val GEMM_16X16: (Int => dim3, dim3) = (n => grid2D(n, n, 16, 16), block2D(16, 16))

  /** Tensor Core friendly launch: block(128, 2) for WMMA ops */
  val TENSOR_CORE: (Int => dim3, dim3) = (n => grid1D((n + 127) / 128, 128), block1D(128))

  /** Occupancy-optimized: max threads per block */
  val MAX_OCCUPANCY: (Int => dim3, dim3) = (n => grid1D(n, CUDA_MAX_BLOCK_SIZE), block1D(CUDA_MAX_BLOCK_SIZE))

  // ---- CUDA Architecture constants ----

  /** SM counts per architecture */
  val SM_COUNT_GP100: Int = 56   // Pascal
  val SM_COUNT_V100: Int = 80    // Volta
  val SM_COUNT_T4: Int = 40      // Turing
  val SM_COUNT_A100: Int = 108   // Ampere
  val SM_COUNT_H100: Int = 132   // Hopper
  val SM_COUNT_H200: Int = 144   // Hopper-2 (H200 = H100 + HBM3)
  val SM_COUNT_B100: Int = 144   // Blackwell (B100)
  val SM_COUNT_B200: Int = 192   // Blackwell (B200)
  val SM_COUNT_B300: Int = 192   // Blackwell (B300 = B200 with more memory)

  /** Shared memory per SM (bytes) */
  val SHARED_MEMORY_PER_SM: Int = 65536   // 64 KB (Ampere)

  /** Registers per SM */
  val REGISTERS_PER_SM: Int = 65536       // 65536 32-bit registers (Ampere)
}
