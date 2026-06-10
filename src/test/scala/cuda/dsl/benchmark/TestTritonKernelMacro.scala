package cuda.dsl.benchmark

import cuda.dsl.dsl._

/** Test @TritonKernelMacro annotation */
object TestTritonKernelMacro {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("@TritonKernelMacro Test")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels.txt for generated CUDA code.")
    println("=" * 80)

    // Simple expression (no assignment needed)
    @TritonKernelMacro
    def simpleAdd(x: Float, y: Float): Float = x + y

    // SAXPY simplified
    @TritonKernelMacro
    def simpleSaxpy(a: Float, b: Float): Float = a * 2.0f + b * 3.0f

    // Math expression
    @TritonKernelMacro
    def simpleMath(x: Float, y: Float): Float =
      (x * x + y * y) / (x * y + 0.001f)

    // Vector add with loop
    @TritonKernelMacro
    def vectorAdd(x: Float, y: Float, out: Float): Float = {
      var sum = 0.0f
      for (j <- 0 until 10) {
        sum = sum + x + y
      }
      sum
    }

    // If statement test
    @TritonKernelMacro
    def conditionalMax(a: Float, b: Float): Float = {
      var result = 0.0f
      if (a > b) {
        result = a
      } else {
        result = b
      }
      result
    }

    println("\nKernels defined. Check compiler output for generated code.")
  }
}
