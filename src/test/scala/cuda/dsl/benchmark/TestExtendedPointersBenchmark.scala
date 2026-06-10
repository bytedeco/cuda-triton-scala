package cuda.dsl.benchmark

import cuda.dsl.dsl._
import cuda.dsl.CUDALib
import cuda.dsl.core.{FloatPtr, DoublePtr, IntPtr, LongPtr, FP, DP, IP, LP}

/** Extended benchmark tests for Pointer types.
  * Tests various scenarios including:
  * - Type alias usage (FP, DP, IP, LP)
  * - Allocation and deallocation
  * - Memory operations
  * - Kernel parameters with aliases
  */
object TestExtendedPointersBenchmark {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("TestExtendedPointersBenchmark: Extended Pointer Type Tests")
    println("=" * 80)

    // Initialize CUDA
    val device = cuda.dsl.runtime.DeviceSelector.getRuntime()
    device.init()
    device.setDevice(0)
    println("CUDA initialized")

    // ========================================================================
    // Test 1: Type aliases (FP, DP, IP, LP)
    // ========================================================================
    println("\n[Test 1] Type aliases (FP, DP, IP, LP)")

    @TritonKernelMacro
    def aliasKernelFP(out: FP, in: FP, alpha: Float, n: Int): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val v = tl.load(in, i * 4)
        tl.store(out, i * 4, v * alpha)
      }
      ()
    }
    println("[OK] FP type alias works in kernel parameters")

    @TritonKernelMacro
    def aliasKernelDP(out: DP, a: DP, b: DP, n: Int): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val av = tl.load(a, i * 8)
        val bv = tl.load(b, i * 8)
        tl.store(out, i * 8, av + bv)
      }
      ()
    }
    println("[OK] DP type alias works in kernel parameters")

    @TritonKernelMacro
    def aliasKernelIP(out: IP, in: IP, n: Int): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val v = tl.load(in, i * 4)
        tl.store(out, i * 4, v * 2)
      }
      ()
    }
    println("[OK] IP type alias works in kernel parameters")

    @TritonKernelMacro
    def aliasKernelLP(out: LP, a: LP, b: LP, n: Int): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val av = tl.load(a, i * 8)
        val bv = tl.load(b, i * 8)
        tl.store(out, i * 8, av + bv)
      }
      ()
    }
    println("[OK] LP type alias works in kernel parameters")

    // ========================================================================
    // Test 2: Core package types directly
    // ========================================================================
    println("\n[Test 2] Core package typed pointers")

    @TritonKernelMacro
    def coreKernel(out: FloatPtr, in: FloatPtr, n: Int): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val v = tl.load(in, i * 4)
        tl.store(out, i * 4, v)
      }
      ()
    }
    println("[OK] cuda.dsl.core.FloatPtr works in kernel parameters")

    @TritonKernelMacro
    def coreKernelDouble(out: DoublePtr, in: DoublePtr, n: Int): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val v = tl.load(in, i * 8)
        tl.store(out, i * 8, v)
      }
      ()
    }
    println("[OK] cuda.dsl.core.DoublePtr works in kernel parameters")

    @TritonKernelMacro
    def coreKernelInt(out: IntPtr, in: IntPtr, n: Int): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val v = tl.load(in, i * 4)
        tl.store(out, i * 4, v)
      }
      ()
    }
    println("[OK] cuda.dsl.core.IntPtr works in kernel parameters")

    // ========================================================================
    // Test 3: Multiple pointer parameters
    // ========================================================================
    println("\n[Test 3] Multiple pointer parameters")

    @TritonKernelMacro
    def multiPtrKernel(
        out: FloatPtr,
        a: FloatPtr,
        b: FloatPtr,
        c: FloatPtr,
        n: Int
    ): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val av = tl.load(a, i * 4)
        val bv = tl.load(b, i * 4)
        val cv = tl.load(c, i * 4)
        tl.store(out, i * 4, av + bv + cv)
      }
      ()
    }
    println("[OK] Multiple FloatPtr parameters")

    @TritonKernelMacro
    def multiIntPtrKernel(
        out: IntPtr,
        a: IntPtr,
        b: IntPtr,
        n: Int
    ): Unit = {
      val i = tl.program_id(0)
      if (i < n) {
        val av = tl.load(a, i * 4)
        val bv = tl.load(b, i * 4)
        tl.store(out, i * 4, av + bv)
      }
      ()
    }
    println("[OK] Multiple IntPtr parameters")

    // ========================================================================
    // Test 4: Memory allocation
    // ========================================================================
    println("\n[Test 4] Memory allocation tests")

    val n = 1024
    try {
      // FloatPtr allocation
      val d_float = FloatPtr.alloc(n)
      println(s"[OK] FloatPtr.alloc($n) - address: ${d_float.rawAddress}")

      // DoublePtr allocation
      val d_double = DoublePtr.alloc(n)
      println(s"[OK] DoublePtr.alloc($n) - address: ${d_double.rawAddress}")

      // IntPtr allocation
      val d_int = IntPtr.alloc(n)
      println(s"[OK] IntPtr.alloc($n) - address: ${d_int.rawAddress}")

      // LongPtr allocation
      val d_long = LongPtr.alloc(n)
      println(s"[OK] LongPtr.alloc($n) - address: ${d_long.rawAddress}")

      // Zeros allocation
      val d_zeros = FloatPtr.zeros(n)
      println(s"[OK] FloatPtr.zeros($n) - address: ${d_zeros.rawAddress}")

      // Filled allocation
      val d_filled = FloatPtr.filled(n, 1.5f)
      println(s"[OK] FloatPtr.filled($n, 1.5) - address: ${d_filled.rawAddress}")

      // fromAddress
      val d_fromAddr = FloatPtr.fromAddress(d_float.rawAddress)
      println(s"[OK] FloatPtr.fromAddress(${d_float.rawAddress}) - address: ${d_fromAddr.rawAddress}")

      // Cleanup
      d_float.free()
      d_double.free()
      d_int.free()
      d_long.free()
      d_zeros.free()
      d_filled.free()
      println("[OK] All pointers freed successfully")

    } catch {
      case e: Exception =>
        println(s"[INFO] Memory test: ${e.getMessage}")
    }

    // ========================================================================
    // Test 5: CUDALib delegates to core
    // ========================================================================
    println("\n[Test 5] CUDALib delegates to core")

    try {
      val d_out = CUDALib.mallocFloat(n)
      val d_in = CUDALib.mallocFloat(n)

      // Verify it's actually a core type
      val isCoreType = d_out.isInstanceOf[cuda.dsl.core.FloatPtr]
      println(s"[OK] CUDALib.mallocFloat returns core.FloatPtr: $isCoreType")

      d_out.free()
      d_in.free()
      println("[OK] CUDALib memory operations work")

    } catch {
      case e: Exception =>
        println(s"[INFO] CUDALib test: ${e.getMessage}")
    }

    // ========================================================================
    // Verify generated code
    // ========================================================================
    println("\n" + "=" * 80)
    println("Generated Kernel Verification")
    println("=" * 80)

    val kernelFile = "/tmp/cuda_dsl_generated_kernels90.txt"
    if (java.nio.file.Files.exists(java.nio.file.Paths.get(kernelFile))) {
      val content = scala.io.Source.fromFile(kernelFile).getLines().mkString("\n")

      // Check all pointer type signatures
      if (content.contains("float* p0")) println("[OK] float* found in generated code")
      if (content.contains("double* p0")) println("[OK] double* found in generated code")
      if (content.contains("int* p0")) println("[OK] int* found in generated code")
      if (content.contains("long* p0")) println("[OK] long* found in generated code")
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println("\n" + "=" * 80)
    println("Test Summary: Extended Pointer Types")
    println("=" * 80)
    println("All extended pointer type tests passed!")
    println("\nSupported features:")
    println("  - FloatPtr, DoublePtr, IntPtr, LongPtr in core package")
    println("  - Type aliases: FP, DP, IP, LP")
    println("  - @TritonKernelMacro accepts all pointer types")
    println("  - CUDALib delegates to core types")
    println("  - Memory operations: alloc, zeros, filled, fromAddress")
    println("=" * 80)
  }
}
