/** CUTLASS module - CUDA Templates for Linear Algebra Subroutines.
 *
 * This module provides high-performance CUDA kernels inspired by NVIDIA CUTLASS library.
 * It includes:
 * - GEMM (General Matrix Multiply) kernels with various configurations
 * - Tiled and persistent GEMM operations
 * - Tensor Core operations (MMA)
 * - Epilogue fusions (bias, activation)
 *
 * Design patterns from CUTLASS:
 * - Threadblock tiles for memory coalescing
 * - Warp-level operations for tensor core efficiency
 * - Epilogue pattern for fusing activations
 *
 * Example usage:
 * {{{
 * // Basic GEMM: D = alpha * A * B + beta * C
 * @TritonKernelMacro(name = "cutlassGemm", gridType = "2D")
 * def cutlassGemmKernel(
 *     D: FloatPtr, A: FloatPtr, B: FloatPtr, C: FloatPtr,
 *     M: Int, N: Int, K: Int,
 *     lda: Int, ldb: Int, ldc: Int, ldd: Int,
 *     alpha: Float, beta: Float
 * ): Unit = {
 *     // Threadblock coordinates
 *     val row = tl.program_id(0) * BLOCK_M
 *     val col = tl.program_id(1) * BLOCK_N
 *     // ... GEMM computation
 * }
 * }}}
 */
package cuda.dsl.cutlass


