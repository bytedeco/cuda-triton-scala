package cuda.dsl

/** Main DSL entry point and imports.
  * Provides convenient access to all CUDA DSL components.
  */
object DSL {
  // Core types
  export cuda.dsl.core.dim3
  export cuda.dsl.core.Ptr
  export cuda.dsl.core.Types.*
  export cuda.dsl.core.MemcpyKind
  export cuda.dsl.core.Memcpy

  // Typed Pointers (extends JavaCPP Pointer)
  export cuda.dsl.core.FloatPtr
  export cuda.dsl.core.DoublePtr
  export cuda.dsl.core.IntPtr
  export cuda.dsl.core.LongPtr
  export cuda.dsl.core.PointerPtr
  export cuda.dsl.core.BytePtr
  export cuda.dsl.core.{FP, DP, IP, LP, BP, PP}

  // Thread builtins
  export cuda.dsl.core.threadIdx
  export cuda.dsl.core.blockIdx
  export cuda.dsl.core.blockDim
  export cuda.dsl.core.gridDim
  export cuda.dsl.core.warp
  export cuda.dsl.core.syncthreads
  export cuda.dsl.core.globalThreadId
  export cuda.dsl.core.localThreadId

  // Runtime
  export cuda.dsl.runtime.CUDARuntime
  export cuda.dsl.runtime.DeviceRuntime
  export cuda.dsl.runtime.DeviceSelector
  export cuda.dsl.runtime.MPSRuntime
  export cuda.dsl.runtime.CompiledKernel
  export cuda.dsl.runtime.CUDAKernel
  export cuda.dsl.runtime.MPSKernel
  export cuda.dsl.runtime.CPUKernel

  // DSL (Triton-like kernels)
  export cuda.dsl.dsl.TritonKernel
  export cuda.dsl.dsl.TritonKernelRunner
  export cuda.dsl.dsl.TritonKernelMacro
  export cuda.dsl.dsl.TritonJit
  export cuda.dsl.triton.jit

  // Macros
  export cuda.dsl.macros.cudaKernel
  export cuda.dsl.macros.cudaOperator
  export cuda.dsl.macros.mpsKernel
  export cuda.dsl.macros.mpsOperator

  // GPU Collections
  export cuda.dsl.collections.GPUArray
  export cuda.dsl.collections.GPUSeq
  export cuda.dsl.collections.GPUType
  export cuda.dsl.collections.GPUOps
  export cuda.dsl.collections.Implicits

  // GPU DataFrame
  export cuda.dsl.dataframe.GPUDataFrame
  export cuda.dsl.dataframe.GPUColumn
  export cuda.dsl.dataframe.DataType
  export cuda.dsl.dataframe.StructType
  export cuda.dsl.dataframe.StructField
  export cuda.dsl.dataframe.Row
  export cuda.dsl.dataframe.JoinType
  export cuda.dsl.dataframe.SortDirection
  export cuda.dsl.dataframe.Aggregation
  export cuda.dsl.dataframe.AggregationType
  export cuda.dsl.dataframe.GroupedDataFrame
  export cuda.dsl.dataframe.DataFrameBackend

  // programId helper for kernel code
  inline def programId(axis: Int): Int = axis match {
    case 0 => blockIdx.x * blockDim.x + threadIdx.x
    case 1 => blockIdx.y * blockDim.y + threadIdx.y
    case 2 => blockIdx.z * blockDim.z + threadIdx.z
    case _ => blockIdx.x * blockDim.x + threadIdx.x
  }
}

/** Java-friendly API for CUDA DSL.
  * This object provides static methods that can be called directly from Java.
  * Now delegates to core package typed pointers.
  */
object CUDALib {
  // Import MemoryOps for copy operations
  import cuda.dsl.core.Types.given_MemoryOps_Float
  import cuda.dsl.core.Types.given_MemoryOps_Double
  import cuda.dsl.core.Types.given_MemoryOps_Int
  import cuda.dsl.core.Types.given_MemoryOps_Long

  // Import JavaCPP Pointer
  import org.bytedeco.javacpp.Pointer

  // Convenience methods for common operations
  def init(): Unit = cuda.dsl.runtime.DeviceSelector.getRuntime().init()

  def getDeviceCount: Int = cuda.dsl.runtime.DeviceSelector.getRuntime().getDeviceCount

  def setDevice(device: Int): Unit = cuda.dsl.runtime.DeviceSelector.getRuntime().setDevice(device)

  // Delegate to core typed pointers
  type FloatPtr = cuda.dsl.core.FloatPtr
  type DoublePtr = cuda.dsl.core.DoublePtr
  type IntPtr = cuda.dsl.core.IntPtr
  type LongPtr = cuda.dsl.core.LongPtr
  type BytePtr = cuda.dsl.core.BytePtr
  type PointerPtr = cuda.dsl.core.PointerPtr

  def mallocFloat(n: Int): FloatPtr = cuda.dsl.core.FloatPtr.alloc(n)
  def mallocDouble(n: Int): DoublePtr = cuda.dsl.core.DoublePtr.alloc(n)
  def mallocInt(n: Int): IntPtr = cuda.dsl.core.IntPtr.alloc(n)
  def mallocLong(n: Int): LongPtr = cuda.dsl.core.LongPtr.alloc(n)
  def mallocByte(n: Int): BytePtr = cuda.dsl.core.BytePtr.alloc(n)
  def mallocPointer(n: Int): PointerPtr = cuda.dsl.core.PointerPtr.alloc(n)

  def freeFloat(ptr: FloatPtr): Unit = ptr.free()
  def freeDouble(ptr: DoublePtr): Unit = ptr.free()
  def freeInt(ptr: IntPtr): Unit = ptr.free()
  def freeLong(ptr: LongPtr): Unit = ptr.free()
  def freeByte(ptr: BytePtr): Unit = ptr.free()
  def freePointer(ptr: PointerPtr): Unit = ptr.free()

  def memcpyFloatHtoD(dst: FloatPtr, src: Array[Float], n: Int): Unit =
    cuda.dsl.runtime.DeviceSelector.getRuntime().memcpyHtoD(dst.toPtr, src, n)

  def memcpyFloatDtoH(dst: Array[Float], src: FloatPtr, n: Int): Unit =
    cuda.dsl.runtime.DeviceSelector.getRuntime().memcpyDtoH(dst, src.toPtr, n)

  /** Launch kernel from function pointer (for Java compatibility).
   */
  def launch(functionPtr: Pointer, gridX: Int, gridY: Int, gridZ: Int,
             blockX: Int, blockY: Int, blockZ: Int): Unit = {
    val kernel = new cuda.dsl.runtime.CUDAKernel(functionPtr, functionPtr)
    cuda.dsl.runtime.DeviceSelector.getRuntime().launchKernel(kernel,
      new cuda.dsl.core.dim3(gridX, gridY, gridZ), new cuda.dsl.core.dim3(blockX, blockY, blockZ), Nil)
  }

  def synchronize(): Unit = cuda.dsl.runtime.DeviceSelector.getRuntime().synchronize()

  def getFreeMemory(): Long = cuda.dsl.runtime.DeviceSelector.getRuntime().getMemoryInfo()._1
  def getTotalMemory(): Long = cuda.dsl.runtime.DeviceSelector.getRuntime().getMemoryInfo()._2

  // Factory methods for creating pointer wrappers from device addresses
  object FloatPtr {
    def apply(deviceAddr: Long, elementCount: Long): FloatPtr =
      cuda.dsl.core.FloatPtr.fromAddress(deviceAddr)
  }
  object DoublePtr {
    def apply(deviceAddr: Long, elementCount: Long): DoublePtr =
      cuda.dsl.core.DoublePtr.fromAddress(deviceAddr)
  }
  object IntPtr {
    def apply(deviceAddr: Long, elementCount: Long): IntPtr =
      cuda.dsl.core.IntPtr.fromAddress(deviceAddr)
  }
  object LongPtr {
    def apply(deviceAddr: Long, elementCount: Long): LongPtr =
      cuda.dsl.core.LongPtr.fromAddress(deviceAddr)
  }
  object BytePtr {
    def apply(deviceAddr: Long, elementCount: Long): BytePtr =
      cuda.dsl.core.BytePtr.fromAddress(deviceAddr)
  }
  object PointerPtr {
    def apply(deviceAddr: Long, pointerCount: Long): PointerPtr =
      cuda.dsl.core.PointerPtr.fromAddress(deviceAddr)
  }
}

/** Dim3 wrapper for Java (extends JavaCPP dim3 for full interoperability).
  * Can be used from Java code and implicitly converted to cuda.dsl.core.dim3.
  */
class Dim3(override val x: Int, override val y: Int = 1, override val z: Int = 1)
    extends cuda.dsl.core.dim3(x, y, z) {

  def this(x: Int) = this(x, 1, 1)
  def this(x: Int, y: Int) = this(x, y, 1)

  /** Convert to Scala DSL dim3 */
  def toDim3: cuda.dsl.core.dim3 = this

  override def toString: String = s"Dim3($x, $y, $z)"
}
