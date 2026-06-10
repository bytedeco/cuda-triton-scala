package cuda.dsl.runtime

import cuda.dsl.core._
import cuda.dsl.core.Types._
import cuda.dsl.core.Ptr
import cuda.dsl.harness.Harness
import org.bytedeco.javacpp.Pointer

/** Abstract device runtime for multi-backend support (CUDA, MPS, CPU).
 *  This trait defines the interface that all device runtimes must implement.
 *  The CUDA DSL uses this interface to remain backend-agnostic.
 */
trait DeviceRuntime {
  /** Backend name: "CUDA", "MPS", or "CPU" */
  def backendName: String

  /** Whether this backend is available on the current system */
  def isAvailable: Boolean

  // Initialization
  def init(): Unit
  def getDeviceCount: Int
  def setDevice(deviceId: Int): Unit
  def getDevice: Int
  def synchronize(): Unit
  def shutdown(): Unit

  // Memory operations
  def malloc[T](n: Int)(using ops: MemoryOps[T]): Ptr[T]
  def free[T](ptr: Ptr[T])(using ops: MemoryOps[T]): Unit
  def memcpyHtoD[T](dst: Ptr[T], src: Array[T], n: Int)(using ops: MemoryOps[T]): Unit
  def memcpyDtoH[T](dst: Array[T], src: Ptr[T], n: Int)(using ops: MemoryOps[T]): Unit
  def memcpyDtoD[T](dst: Ptr[T], src: Ptr[T], n: Int)(using ops: MemoryOps[T]): Unit
  def getMemoryInfo(): (Long, Long)

  // Kernel operations
  // CUDA: Compiles NVRTC source to PTX and loads into module
  // MPS: Uses PyTorch tensor operations (compileKernel is a no-op)
  def compileKernel(name: String, cudaSource: String): CompiledKernel
  def launchKernel(kernel: CompiledKernel, grid: dim3, block: dim3, args: Seq[Pointer],
                 scalars: Map[String, Any] = Map.empty, invIdx: Int = 0): Option[Long]

  // Stub buffer retrieval — returns None in real mode
  def getStubBuffer(addr: Long): Option[Array[Float]] = None
}

/** Handle for compiled kernels (device-specific implementation) */
sealed trait CompiledKernel

/** CUDA kernel handle - contains module and function pointers */
class CUDAKernel(
  val modulePointer: Pointer,
  val functionPointer: Pointer,
  val name: String = ""
) extends CompiledKernel {
  override def toString: String = s"CUDAKernel(module=${modulePointer}, func=${functionPointer})"
}

/** MPS kernel handle - represents a PyTorch operation chain */
class MPSKernel(
  val operationName: String,
  val operationCode: String  // Generated PyTorch code
) extends CompiledKernel {
  override def toString: String = s"MPSKernel($operationName)"
}

/** CPU kernel handle - placeholder for fallback */
class CPUKernel(
  val name: String
) extends CompiledKernel {
  override def toString: String = s"CPUKernel($name)"
}
