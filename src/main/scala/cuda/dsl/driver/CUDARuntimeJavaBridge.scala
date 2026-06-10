package cuda.dsl.driver

import cuda.dsl.runtime.*
import cuda.dsl.core.{Ptr, dim3}
import cuda.dsl.core.Types.given_MemoryOps_Float
import cuda.dsl.core.Types.given_MemoryOps_Int
import org.bytedeco.javacpp.*

/**
 * Java-friendly bridge for CUDARuntime operations.
 * Provides simplified float-only APIs without generic type parameters.
 */
object CUDARuntimeJavaBridge:

  // Mutable map to track result buffers for readResult
  // key = result address (Long), value = result data (Array[Float])
  private val resultStore = scala.collection.mutable.Map[Long, Array[Float]]()

  // Mutable map to track compiled kernels by name
  private val kernelCache = scala.collection.mutable.Map[String, CompiledKernel]()

  /** Initialize CUDA runtime */
  def initialize(): Boolean = {
    CUDARuntime.init()
    true
  }

  /** Allocate float buffer on device. Returns address as Long. */
  def malloc(count: Int): Long = {
    val ptr = CUDARuntime.malloc[Float](count)
    ptr.rawAddress
  }

  /** Free float buffer */
  def free(addr: Long): Unit = {
    val ptr = Ptr.fromAddress[Float](addr)
    CUDARuntime.free(ptr)
  }

  /** Copy from host to device */
  def memcpyHtoD(addr: Long, data: Array[Float], count: Int): Unit = {
    val ptr = Ptr.fromAddress[Float](addr)
    CUDARuntime.memcpyHtoD(ptr, data, count)
  }

  /** Copy from device to host */
  def memcpyDtoH(data: Array[Float], addr: Long, count: Int): Unit = {
    val ptr = Ptr.fromAddress[Float](addr)
    CUDARuntime.memcpyDtoH(data, ptr, count)
  }

  /** Allocate and copy data to device in one call */
  def allocAndCopy(data: Array[Float]): Long = {
    val addr = malloc(data.length)
    memcpyHtoD(addr, data, data.length)
    addr
  }

  /** Allocate float buffer with given initial values */
  def allocWithData(data: Array[Float]): Long = {
    val addr = malloc(data.length)
    memcpyHtoD(addr, data, data.length)
    addr
  }

  /** Allocate int buffer on device. Returns address as Long. */
  def mallocInt(count: Int): Long = {
    val ptr = CUDARuntime.malloc[Int](count)
    ptr.rawAddress
  }

  /** Copy int array from host to device */
  def memcpyHtoDInt(addr: Long, data: Array[Int], count: Int): Unit = {
    val ptr = Ptr.fromAddress[Int](addr)
    CUDARuntime.memcpyHtoD(ptr, data, count)
  }

  /** Allocate and copy int data to device */
  def allocAndCopyInt(data: Array[Int]): Long = {
    val addr = mallocInt(data.length)
    memcpyHtoDInt(addr, data, data.length)
    addr
  }

  /** Copy int data from device to host */
  def memcpyDtoHInt(data: Array[Int], addr: Long, count: Int): Unit = {
    val ptr = Ptr.fromAddress[Int](addr)
    CUDARuntime.memcpyDtoH(data, ptr, count)
  }

  // ========================================================================
  // Kernel Compilation & Execution
  // ========================================================================

  /**
   * Compile CUDA source code to kernel.
   * Returns kernel name string on success, null on failure.
   */
  def compileKernel(name: String, cudaSource: String): String = {
    val kernel = CUDARuntime.compileKernel(name, cudaSource)
    if (kernel != null) {
      kernelCache(name) = kernel
      name
    } else null
  }

  /**
   * Launch a kernel by name.
   * @param name kernel name (must have been compiled first)
   * @param gridX grid dimension X
   * @param gridY grid dimension Y
   * @param gridZ grid dimension Z
   * @param blockX block dimension X
   * @param blockY block dimension Y
   * @param blockZ block dimension Z
   * @param bufferAddrs buffer addresses (Long), first = output, rest = inputs
   * @param scalarVals scalar Float values, can be null
   * @param scalarKeys scalar names for stub resolution, can be null
   * @return result buffer address, or -1L on failure
   */
  def launchKernel(
    name: String,
    gridX: Int, gridY: Int, gridZ: Int,
    blockX: Int, blockY: Int, blockZ: Int,
    bufferAddrs: Array[Long],
    scalarVals: Array[Float],
    scalarKeys: Array[String]
  ): Long = {
    val ptrs = bufferAddrs.map(a => new LongPointer(a))

    val scalars: Map[String, Any] = (scalarVals, scalarKeys) match {
      case (v: Array[Float], k: Array[String]) if v != null && k != null && v.length == k.length =>
        k.zip(v).toMap
      case (v: Array[Float], _) if v != null =>
        v.zipWithIndex.map((a, b) => s"scalar$b" -> a).toMap
      case _ => Map.empty
    }

    val cached = kernelCache.get(name).orNull
    if (cached == null) {
      System.err.println(s"[CUDARuntimeJavaBridge] Kernel '$name' not found in cache")
      return -1L
    }

    val grid = dim3(gridX, gridY, gridZ)
    val block = dim3(blockX, blockY, blockZ)

    val resultAddr = CUDARuntime.launchKernel(cached, grid, block, ptrs.toSeq, scalars).getOrElse(-1L)

    // Capture stub results so readResult can return them
    if (resultAddr > 0) {
      try {
        val resultBuf = CUDARuntime.getStubBuffer(resultAddr)
        resultBuf.foreach { buf =>
          val captured = new Array[Float](buf.length)
          java.lang.System.arraycopy(buf, 0, captured, 0, buf.length)
          resultStore(resultAddr) = captured
        }
      } catch {
        case _: Throwable =>
      }
    }

    resultAddr
  }

  /** Simple launch with no scalars */
  def launchKernelSimple(
    name: String,
    gx: Int, gy: Int, gz: Int,
    bx: Int, by: Int, bz: Int,
    bufAddrs: Array[Long]
  ): Long = launchKernel(name, gx, gy, gz, bx, by, bz, bufAddrs, null, null)

  /**
   * Read result from a result buffer address into a host array.
   */
  def readResult(addr: Long, size: Int): Array[Float] = {
    if (addr <= 0) return Array.empty
    val bufOpt = resultStore.get(addr)
    val result = new Array[Float](size)
    bufOpt.foreach { buf =>
      val cnt = math.min(size, buf.length)
      java.lang.System.arraycopy(buf, 0, result, 0, cnt)
    }
    result
  }

  /**
   * Convenience: compile, allocate, copy inputs, launch, read output.
   * Returns output array.
   */
  def execKernel(
    kernelName: String,
    cudaSource: String,
    gx: Int, gy: Int, gz: Int,
    bx: Int, by: Int, bz: Int,
    inputAddrs: Array[Long],
    outputSize: Int,
    scalarVals: Array[Float],
    scalarKeys: Array[String]
  ): Array[Float] = {
    val kn = compileKernel(kernelName, cudaSource)
    if (kn == null) return Array.empty

    val outAddr = malloc(outputSize)
    val allAddrs = outAddr +: inputAddrs
    val resultAddr = launchKernel(kn, gx, gy, gz, bx, by, bz, allAddrs, scalarVals, scalarKeys)

    readResult(resultAddr, outputSize)
  }

  /**
   * Get raw buffer content from resultStore (for debugging).
   */
  def getBuffer(addr: Long): Array[Float] = resultStore.getOrElse(addr, Array.empty)

  /** Synchronize device */
  def synchronize(): Unit = CUDARuntime.synchronize

  /** Shutdown runtime */
  def shutdown(): Unit = CUDARuntime.shutdown
