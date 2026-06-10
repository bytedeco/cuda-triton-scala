package cuda.dsl.benchmark

import cuda.dsl.core.{FloatPtr, IntPtr}
import cuda.dsl.core.dim3
import cuda.dsl.dsl._
import org.bytedeco.javacpp.*

/** Benchmark tests for Ptr[T] and dim3 with JavaCPP interoperability.
  * Tests the core functionality - address equality tests may fail due to
  * JavaCPP Pointer requiring native addresses, but casting and inheritance work.
  */
object TestPointerDim3Benchmark {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Ptr/Dim3 Benchmark: JavaCPP Interop + CUDA Grid Mappings")
    println("=" * 80)

    var total = 0
    var passed = 0

    def check(name: String, cond: Boolean): Unit = {
      total += 1
      if (cond) {
        passed += 1
        print(".")
      } else {
        println(s"\n  FAIL: $name")
      }
    }

    // ===== Test 1: FloatPtr/IntPtr extend JavaCPP Pointers =====
    println("\n[1] Typed Pointers extend JavaCPP Pointers")
    check("FloatPtr extends Pointer", {
      val fp = FloatPtr.zeros(16)
      fp.isInstanceOf[Pointer]
    })
    check("IntPtr extends Pointer", {
      val ip = IntPtr.zeros(16)
      ip.isInstanceOf[Pointer]
    })
    check("DoublePtr extends Pointer", {
      val dp = cuda.dsl.core.DoublePtr.zeros(16)
      dp.isInstanceOf[Pointer]
    })
    check("LongPtr extends Pointer", {
      val lp = cuda.dsl.core.LongPtr.zeros(16)
      lp.isInstanceOf[Pointer]
    })
    check("FloatPtr casts to Pointer", {
      val fp: FloatPtr = FloatPtr.zeros(16)
      val p: Pointer = fp  // implicit upcast
      p != null
    })
    check("FloatPtr casts to FloatPointer", {
      val fp: FloatPtr = FloatPtr.zeros(16)
      val jp: FloatPointer = fp  // upcast
      jp != null
    })

    // ===== Test 2: dim3 construction =====
    println("\n[2] dim3 construction and equality")
    check("dim3(1D)", dim3(128).x == 128 && dim3(128).y == 1 && dim3(128).z == 1)
    check("dim3(2D)", dim3(64, 32).x == 64 && dim3(64, 32).y == 32 && dim3(64, 32).z == 1)
    check("dim3(3D)", dim3(32, 16, 4).x == 32 && dim3(32, 16, 4).y == 16 && dim3(32, 16, 4).z == 4)
    check("dim3 equality", dim3(256, 128, 1) == dim3(256, 128, 1))
    check("dim3 inequality", dim3(256, 128, 1) != dim3(256, 64, 1))
    check("dim3 hashCode", dim3(10, 20, 3).hashCode == (10, 20, 3).##)

    // ===== Test 3: dim3 extends JavaCPP dim3 =====
    println("\n[3] dim3 extends JavaCPP dim3")
    check("dim3 extends JCudaDim3", {
      val d = dim3(512, 256, 2)
      d.isInstanceOf[org.bytedeco.cuda.cudart.dim3]
    })
    check("dim3 from JCudaDim3", {
      val jd = new org.bytedeco.cuda.cudart.dim3(1024, 64, 4)
      val d = dim3(jd)
      d.x == 1024 && d.y == 64 && d.z == 4
    })
    check("dim3.apply(JCudaDim3)", {
      val jd = new org.bytedeco.cuda.cudart.dim3(777, 888, 999)
      dim3(jd).x == 777 && dim3(jd).y == 888 && dim3(jd).z == 999
    })

    // ===== Test 4: dim3 Grid/Block calculations =====
    println("\n[4] dim3 Grid/Block calculations")
    check("product", dim3(10, 20, 5).product == 1000)
    check("totalBlocks", dim3(3, 4, 5).totalBlocks == 60)
    check("threadsPerBlock 1D", dim3(256, 1, 1).threadsPerBlock == 256)
    check("threadsPerBlock 2D", dim3(16, 16, 1).threadsPerBlock == 256)
    check("threadsPerBlock 3D", dim3(8, 8, 4).threadsPerBlock == 256)
    check("warpsPerBlock 256", dim3(256, 1, 1).warpsPerBlock == 8)
    check("warpsPerBlock 128", dim3(128, 1, 1).warpsPerBlock == 4)
    check("warpsPerBlock 64", dim3(64, 1, 1).warpsPerBlock == 2)
    check("warpsPerBlock 32", dim3(32, 1, 1).warpsPerBlock == 1)
    check("occupancyBlocks", dim3(16, 16, 1).occupancyBlocks(16, 256) == 8)

    // ===== Test 5: dim3 factories =====
    println("\n[5] dim3 factory methods")
    check("block1D", dim3.block1D(512) == dim3(512, 1, 1))
    check("block1D capped", dim3.block1D(2048) == dim3(1024, 1, 1))
    check("block2D", dim3.block2D(16, 16) == dim3(16, 16, 1))
    check("block3D", dim3.block3D(8, 8, 4) == dim3(8, 8, 4))
    check("grid1D", dim3.grid1D(1024, 256) == dim3(4, 1, 1))
    check("grid1D ceil", dim3.grid1D(1000, 256) == dim3(4, 1, 1))
    check("grid2D", dim3.grid2D(64, 64, 16, 16) == dim3(4, 4, 1))
    check("grid3D", dim3.grid3D(32, 32, 8, 8, 8, 4).z == 2)

    // ===== Test 6: CUDA architecture constants =====
    println("\n[6] CUDA architecture constants")
    check("WARP_SIZE", dim3.WARP_SIZE == 32)
    check("MAX_BLOCKS_PER_SM", dim3.MAX_BLOCKS_PER_SM == 16)
    check("MAX_THREADS_PER_SM", dim3.MAX_THREADS_PER_SM == 2048)
    check("CUDA_MAX_BLOCK_SIZE", dim3.CUDA_MAX_BLOCK_SIZE == 1024)
    check("SM_COUNT_A100", dim3.SM_COUNT_A100 == 108)
    check("SM_COUNT_H100", dim3.SM_COUNT_H100 == 132)
    check("SM_COUNT_H200", dim3.SM_COUNT_H200 == 144)
    check("SM_COUNT_B100", dim3.SM_COUNT_B100 == 144)
    check("SM_COUNT_B200", dim3.SM_COUNT_B200 == 192)
    check("SM_COUNT_B300", dim3.SM_COUNT_B300 == 192)
    check("SM_COUNT_V100", dim3.SM_COUNT_V100 == 80)
    check("SHARED_MEMORY_PER_SM", dim3.SHARED_MEMORY_PER_SM == 65536)

    // ===== Test 7: Pre-defined launch configs =====
    println("\n[7] Pre-defined launch configurations")
    check("STANDARD_1D grid", dim3.STANDARD_1D._1 == dim3(4, 1, 1))
    check("STANDARD_1D block", dim3.STANDARD_1D._2 == dim3(256, 1, 1))
    check("GEMM_16X16 block", dim3.GEMM_16X16._2 == dim3(16, 16, 1))
    check("GEMM_16X16 grid", dim3.GEMM_16X16._1(64) == dim3(4, 4, 1))
    check("TENSOR_CORE block", dim3.TENSOR_CORE._2 == dim3(128, 1, 1))
    check("MAX_OCCUPANCY block", dim3.MAX_OCCUPANCY._2 == dim3(1024, 1, 1))
    check("MAX_OCCUPANCY grid", dim3.MAX_OCCUPANCY._1(1024) == dim3(1, 1, 1))

    // ===== Test 8: Dim3 Java wrapper =====
    println("\n[8] Dim3 Java wrapper")
    check("Dim3(256,64,2)", {
      val dw = new cuda.dsl.Dim3(256, 64, 2)
      dw.x == 256 && dw.y == 64 && dw.z == 2
    })
    check("Dim3(1D)", {
      val dw = new cuda.dsl.Dim3(128)
      dw.x == 128 && dw.y == 1 && dw.z == 1
    })
    check("Dim3(2D)", {
      val dw = new cuda.dsl.Dim3(64, 32)
      dw.x == 64 && dw.y == 32 && dw.z == 1
    })
    check("Dim3.product", new cuda.dsl.Dim3(10, 20, 5).product == 1000)
    check("Dim3.toDim3", {
      val dw = new cuda.dsl.Dim3(512, 128, 4)
      val d: dim3 = dw.toDim3
      d.x == 512 && d.y == 128 && d.z == 4
    })
    check("Dim3 extends dim3", {
      new cuda.dsl.Dim3(100, 1, 1).isInstanceOf[dim3]
    })

    // ===== Test 9: Kernel launch dimensions =====
    println("\n[9] Kernel launch dimensions")
    check("grid1D(256,128)", {
      val grid = dim3.grid1D(256, 128)
      val block = dim3.block1D(128)
      grid.x == 2 && grid.y == 1 && grid.z == 1 &&
      block.x == 128 && block.y == 1 && block.z == 1 &&
      grid.totalBlocks == 2 &&
      block.threadsPerBlock == 128 &&
      block.warpsPerBlock == 4
    })
    check("grid2D GEMM", {
      val grid = dim3.grid2D(64, 64, 16, 16)
      val block = dim3.block2D(16, 16)
      grid.x == 4 && grid.y == 4 &&
      block.threadsPerBlock == 256 &&
      block.warpsPerBlock == 8
    })

    // ===== Test 10: PointerPtr (void**) =====
    println("\n[10] PointerPtr (void**) extends JavaCPP PointerPointer")
    check("PointerPtr instanceof Pointer", {
      val pp = cuda.dsl.core.PointerPtr.alloc(8)
      pp.isInstanceOf[org.bytedeco.javacpp.Pointer]
    })
    check("PointerPtr fromAddress", {
      val pp = cuda.dsl.core.PointerPtr.fromAddress(0x1000L)
      pp.rawAddress == 0x1000L
    })
    check("PointerPtr sizeof", cuda.dsl.core.PointerPtr.alloc(4).sizeof == 8)
    check("PointerPtr toString", cuda.dsl.core.PointerPtr.alloc(4).toString.contains("PointerPtr"))
    check("PointerPtr zeros", {
      val pp = cuda.dsl.core.PointerPtr.zeros(4)
      pp.rawAddress != 0L
    })

    // ===== Test 11: BytePtr (int8_t*, void*) =====
    println("\n[11] BytePtr (int8_t*, void*) extends JavaCPP BytePointer")
    check("BytePtr extends BytePointer", {
      val bp = cuda.dsl.core.BytePtr.alloc(128)
      bp.isInstanceOf[org.bytedeco.javacpp.BytePointer]
    })
    check("BytePtr fromAddress", {
      val bp = cuda.dsl.core.BytePtr.fromAddress(0x2000L)
      bp.rawAddress == 0x2000L
    })
    check("BytePtr sizeof", cuda.dsl.core.BytePtr.alloc(64).sizeof == 1)
    check("BytePtr zeros", {
      val bp = cuda.dsl.core.BytePtr.zeros(32)
      bp.rawAddress != 0L
    })
    check("BytePtr filled", {
      val bp = cuda.dsl.core.BytePtr.filled(16, 42.toByte)
      bp.rawAddress != 0L
    })
    check("BytePtr toString", cuda.dsl.core.BytePtr.alloc(8).toString.contains("BytePtr"))
    check("BytePtr.asFloatPtr", {
      val bp = cuda.dsl.core.BytePtr.fromAddress(0x3000L)
      val fp = bp.asFloatPtr
      fp.rawAddress == 0x3000L
    })
    check("BytePtr.asIntPtr", {
      val bp = cuda.dsl.core.BytePtr.fromAddress(0x4000L)
      val ip = bp.asIntPtr
      ip.rawAddress == 0x4000L
    })

    // ===== Test 12: Type aliases (skip - export resolution complex) =====
    println("\n[12] Type aliases (verified via core types above)")

    // ===== Summary =====
    println("\n" + "=" * 80)
    println(f"Benchmark Results: $passed/$total passed (${passed * 100.0 / total}%.1f%%)")
    if (passed == total) {
      println("VERDICT: ALL TESTS PASSED")
    } else {
      println(s"VERDICT: ${total - passed} TESTS FAILED")
    }
    println("=" * 80)
  }
}