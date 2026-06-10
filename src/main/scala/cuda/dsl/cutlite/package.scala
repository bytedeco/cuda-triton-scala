/** CUTLITE module - Lightweight CUDA operations inspired by NVIDIA CUTLITE.
 *
 * This module provides high-level operations similar to the CUTLITE Python library:
 * - Element-wise operations: exp, log, sqrt, sin, cos, tanh, etc.
 * - Reduction operations: sum, max, min, reduce
 * - Memory operations: load, store, gather, scatter
 * - Tiling abstractions: Tile, TiledView
 * - Math functions: matmul, softmax, attention
 *
 * All kernels use @TritonKernelMacro with the tl. DSL for CUDA code generation.
 *
 * Example usage:
 * {{{
 * // Softmax using CUTLITE-style operations
 * @TritonKernelMacro(name = "softMaxKernel", gridType = "1D")
 * def softmaxKernel(out: FloatPtr, inp: FloatPtr, N: Int, D: Int): Unit = {
 *   val row = tl.program_id(0)
 *   if (row >= N) return
 *
 *   // Compute max
 *   var rowMax: Float = -3.4e38f
 *   var j = 0
 *   while (j < D) {
 *     val v = tl.load(inp, row * D + j)
 *     if (v > rowMax) rowMax = v
 *     j = j + 1
 *   }
 *
 *   // Compute exp sum
 *   var expSum: Float = 0.0f
 *   j = 0
 *   while (j < D) {
 *     expSum = expSum + exp(tl.load(inp, row * D + j) - rowMax)
 *     j = j + 1
 *   }
 *
 *   // Normalize
 *   j = 0
 *   while (j < D) {
 *     val val_ = exp(tl.load(inp, row * D + j) - rowMax) / expSum
 *     tl.store(out, row * D + j, val_)
 *     j = j + 1
 *   }
 *   ()
 * }
 * }}}
 */
package cuda.dsl.cutlite


