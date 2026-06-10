package cuda.dsl.benchmark

import cuda.dsl.macros.mpsKernel
import scala.annotation.experimental

/** Test to verify @mpsKernel macro generates code */
@experimental
object TestMPSKernelMacro {

  @mpsKernel
  def simpleAdd(x: Float, y: Float): Float = x + y

  @mpsKernel
  def simpleMul(x: Float, y: Float): Float = x * y

  def main(args: Array[String]): Unit = {
    println("Testing @mpsKernel macro...")
    println(s"simpleAdd(2, 3) = ${simpleAdd(2, 3)}")
    println(s"simpleMul(4, 5) = ${simpleMul(4, 5)}")

    // Read and display the generated code
    println("\nGenerated code should be in /tmp/cuda_dsl_generated_mps_kernels.txt")
    try {
      val file = new java.io.File("/tmp/cuda_dsl_generated_mps_kernels.txt")
      if (file.exists()) {
        println("\n--- Generated MPS Kernels ---")
        println(scala.io.Source.fromFile(file).mkString)
      } else {
        println("File not found - macro may not have generated code")
      }
    } catch {
      case e: Exception => println(s"Error reading file: ${e.getMessage}")
    }
  }
}
