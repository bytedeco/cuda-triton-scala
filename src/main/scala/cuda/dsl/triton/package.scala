package cuda.dsl

/** Convenience package for `@triton.jit` annotation.
  *
  * Usage:
  * {{{
  * import cuda.dsl.triton.jit
  *
  * @jit
  * def myKernel(out: Float, a: Float, b: Float, n: Int): Unit = { ... }
  * }}}
  *
  * Or equivalently:
  * {{{
  * import cuda.dsl.dsl.TritonKernelMacro
  *
  * @TritonKernelMacro
  * def myKernel(out: Float, a: Float, b: Float, n: Int): Unit = { ... }
  * }}}
  */
package object triton:
  /** Alias for [[cuda.dsl.dsl.TritonKernelMacro]].
    * Marks a method as a GPU kernel to be compiled via Triton DSL.
    */
  type jit = cuda.dsl.dsl.TritonKernelMacro
