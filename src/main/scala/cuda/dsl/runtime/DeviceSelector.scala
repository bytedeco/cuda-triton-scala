package cuda.dsl.runtime

import cuda.dsl.core._
import cuda.dsl.core.Types._
import cuda.dsl.core.Ptr

/** Factory for creating and managing device runtimes.
 *  Selects the best available backend based on hardware:
 *  1. NVIDIA GPU via CUDA (fastest, full-featured)
 *  2. Apple Silicon via MPS (when MP_USE_MPS=1)
 *  3. CPU fallback (always works)
 */
object DeviceSelector {
  private var currentRuntime: DeviceRuntime = null
  private var forcedRuntime: DeviceRuntime = null

  /** Get the current runtime, initializing if needed.
   *  Thread-safe singleton pattern.
   */
  def getRuntime(): DeviceRuntime = synchronized {
    forcedRuntime match {
      case null =>
        if (currentRuntime == null) {
          currentRuntime = selectBestRuntime()
        }
        currentRuntime
      case runtime =>
        runtime
    }
  }

  /** Get the backend name of the current runtime */
  def backendName: String = getRuntime().backendName

  /** Check if MPS is the current backend */
  def isMPS: Boolean = backendName == "MPS"

  /** Check if CUDA is the current backend */
  def isCUDA: Boolean = backendName == "CUDA"

  /** Check if CPU fallback is active */
  def isCPU: Boolean = backendName == "CPU"

  /** Select the best available runtime.
   *  Priority: CUDA > MPS (if enabled) > CPU
   */
  private def selectBestRuntime(): DeviceRuntime = {
    // Try CUDA first (fastest, full-featured)
    val cudaRuntime = tryCUDA()
    if (cudaRuntime != null) {
      return cudaRuntime
    }

    // Try MPS if enabled via environment variable
    val mpsRuntime = tryMPS()
    if (mpsRuntime != null) {
      return mpsRuntime
    }

    // Fallback to CPU
    println("[DeviceSelector] No GPU available, using CPU fallback")
    new CPUFallbackRuntime()
  }

  private def tryCUDA(): DeviceRuntime = {
    try {
      val cuda = new CUDARuntime()
      // init() checks PyTorch CUDA first
      cuda.init()
      val dc = cuda.getDeviceCount
      println(s"[DeviceSelector] CUDA init: deviceCount=$dc, isAvailable=${cuda.isAvailable}")
      if (dc > 0 || cuda.isAvailable) {
        println("[DeviceSelector] Selected: CUDA backend (NVIDIA GPU)")
        return cuda
      }
    } catch {
      case e: Throwable =>
        println(s"[DeviceSelector] CUDA not available: ${e.getMessage}")
    }
    null
  }

  private def tryMPS(): DeviceRuntime = {
    // Check if MPS is explicitly enabled
    val mpsEnv = System.getenv("MP_USE_MPS")
    println(s"[DeviceSelector] MP_USE_MPS env = '$mpsEnv'")
    if (mpsEnv != "1") {
      println("[DeviceSelector] MPS not enabled (set MP_USE_MPS=1 to enable)")
      return null
    }

    try {
      val mps = new MPSRuntime()
      mps.init()
      if (mps.isAvailable) {
        println("[DeviceSelector] Selected: MPS backend (Apple Silicon)")
        return mps
      }
    } catch {
      case e: Throwable =>
        println(s"[DeviceSelector] MPS not available: ${e.getMessage}")
    }
    null
  }

  /** Force a specific runtime (useful for testing).
   *  Clears any existing runtime and sets the forced one.
   */
  def forceRuntime(runtime: DeviceRuntime): Unit = synchronized {
    if (currentRuntime != null) {
      currentRuntime.shutdown()
    }
    forcedRuntime = runtime
    currentRuntime = runtime
    println(s"[DeviceSelector] Forced runtime: ${runtime.backendName}")
  }

  /** Reset to auto-select mode (clears forced runtime) */
  def resetToAuto(): Unit = synchronized {
    forcedRuntime = null
    if (currentRuntime != null) {
      currentRuntime.shutdown()
      currentRuntime = null
    }
  }

  /** Shutdown the current runtime */
  def shutdown(): Unit = synchronized {
    if (currentRuntime != null) {
      currentRuntime.shutdown()
      currentRuntime = null
    }
    forcedRuntime = null
  }
}

/** CPU fallback runtime - stub implementation that works on any machine.
 *  This is used when no GPU is available.
 *  Operations are no-ops or use host memory only.
 */
class CPUFallbackRuntime extends DeviceRuntime {
  override def backendName: String = "CPU"
  override def isAvailable: Boolean = true

  override def init(): Unit = {
    println("[CPUFallbackRuntime] Initialized (stub implementation)")
  }

  override def getDeviceCount: Int = 1
  override def setDevice(deviceId: Int): Unit = ()
  override def getDevice: Int = 0
  override def synchronize(): Unit = ()
  override def shutdown(): Unit = ()

  override def malloc[T](n: Int)(using ops: MemoryOps[T]): Ptr[T] = {
    val fakeAddr = (System.nanoTime() % 1000000).toLong + 0x1000
    Ptr.fromAddress[T](fakeAddr)
  }

  override def free[T](ptr: Ptr[T])(using ops: MemoryOps[T]): Unit = ()

  override def memcpyHtoD[T](dst: Ptr[T], src: Array[T], n: Int)(using ops: MemoryOps[T]): Unit = {
    // No-op for CPU fallback
  }

  override def memcpyDtoH[T](dst: Array[T], src: Ptr[T], n: Int)(using ops: MemoryOps[T]): Unit = {
    // No-op for CPU fallback - data stays in host memory
  }

  override def memcpyDtoD[T](dst: Ptr[T], src: Ptr[T], n: Int)(using ops: MemoryOps[T]): Unit = {
    // No-op for CPU fallback
  }

  override def getMemoryInfo(): (Long, Long) = {
    // Report reasonable host memory values
    val runtime = Runtime.getRuntime
    val freeMem = runtime.freeMemory()
    val totalMem = runtime.totalMemory()
    (freeMem, totalMem)
  }

  override def compileKernel(name: String, cudaSource: String): CPUKernel = {
    println(s"[CPUFallbackRuntime] compileKernel called (no-op)")
    new CPUKernel(name)
  }

  override def launchKernel(kernel: CompiledKernel, grid: dim3, block: dim3, args: Seq[org.bytedeco.javacpp.Pointer],
                    scalars: Map[String, Any] = Map.empty, invIdx: Int = 0): Option[Long] = {
    kernel match {
      case k: CPUKernel =>
        println(s"[CPUFallbackRuntime] launchKernel called for ${k.name} (no-op)")
        None
      case _ =>
        println(s"[CPUFallbackRuntime] launchKernel called for $kernel (no-op)")
        None
    }
  }
}
