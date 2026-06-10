package cuda.dsl.driver

import cuda.dsl.runtime.*
import scala.collection.mutable
import org.bytedeco.cuda.global.cudart.*
import org.bytedeco.cuda.cudart.{CUstream_st, CUevent_st, CUgraph_st, CUgraphExec_st, cudaDeviceProp}
import org.bytedeco.javacpp.{IntPointer, FloatPointer, Pointer, BytePointer}

/** Event flags */
object CUeventFlags:
  val DEFAULT = 0
  val BLOCKING = 1
  val DISABLE_TIMING = 2
  val INTERPROCESS = 4

/** Stream flags */
object CUstreamFlags:
  val DEFAULT = 0
  val NONBLOCKING = 1

/** CUDA Event with native CUDART bindings - self-initializing */
class NativeCUDAEvent(val event: CUevent_st = new CUevent_st()) extends AutoCloseable:
  private var created = false

  def this(flags: Int) = {
    this()
    create(flags)
  }

  def create(flags: Int = CUeventFlags.DEFAULT): NativeCUDAEvent = {
    if created then return this
    try
      val res = cudaEventCreateWithFlags(event, flags)
      if (res == 0) created = true else ()
    catch case _: Throwable => ()
    this
  }

  def record(stream: NativeCUDAStream): Int =
    if created then try cudaEventRecord(event, stream.stream) catch case _: Throwable => -1 else -1

  def recordWithFlags(stream: NativeCUDAStream, flags: Int): Int =
    if created then try cudaEventRecordWithFlags(event, stream.stream, flags) catch case _: Throwable => -1 else -1

  def synchronize(): Int =
    if created then try cudaEventSynchronize(event) catch case _: Throwable => -1 else -1

  def query: Int =
    if created then try cudaEventQuery(event) catch case _: Throwable => -1 else -1

  def elapsedTimeMs(endEvent: NativeCUDAEvent): Float = {
    if !created || !endEvent.created then return 0.0f
    try
      val timePtr = new FloatPointer(1L)
      val res = cudaEventElapsedTime(timePtr, event, endEvent.event)
      if (res == 0) { val t = timePtr.get(0); timePtr.close(); t }
      else { timePtr.close(); 0.0f }
    catch case _: Throwable => 0.0f
  }

  def destroy(): Int =
    if created then
      try
        val res = cudaEventDestroy(event)
        if res == 0 then created = false
        res
      catch case _: Throwable => -1
    else 0

  override def close(): Unit = { try destroy() catch case _: Throwable => () }
  override def finalize(): Unit = { try destroy() catch case _: Throwable => () }

object NativeCUDAEvent:
  def create(flags: Int = CUeventFlags.DEFAULT): NativeCUDAEvent =
    new NativeCUDAEvent().create(flags)

/** CUDA Stream with native CUDART bindings - self-initializing */
class NativeCUDAStream(val stream: CUstream_st = new CUstream_st()) extends AutoCloseable:
  private var created = false

  def this(flags: Int) = {
    this()
    create(flags)
  }

  def create(flags: Int = CUstreamFlags.DEFAULT): NativeCUDAStream = {
    if created then return this
    try
      val res = cudaStreamCreateWithFlags(stream, flags)
      if (res == 0) created = true else ()
    catch case _: Throwable => ()
    this
  }

  def createWithPriority(priority: Int, flags: Int): NativeCUDAStream = {
    if created then return this
    try cudaStreamCreateWithPriority(stream, priority, flags)
    catch case _: Throwable => ()
    this
  }

  def synchronize(): Int =
    if created then try cudaStreamSynchronize(stream) catch case _: Throwable => -1 else -1

  def query: Int =
    if created then try cudaStreamQuery(stream) catch case _: Throwable => -1 else -1

  def destroy(): Int =
    if created then
      try
        val res = cudaStreamDestroy(stream)
        if res == 0 then created = false
        res
      catch case _: Throwable => -1
    else 0

  override def close(): Unit = { try destroy() catch case _: Throwable => () }
  override def finalize(): Unit = { try destroy() catch case _: Throwable => () }

object NativeCUDAStream:
  def create(flags: Int = CUstreamFlags.DEFAULT): NativeCUDAStream =
    new NativeCUDAStream().create(flags)

/** Device properties from native CUDART */
class NativeDeviceProperties(val prop: cudaDeviceProp = new cudaDeviceProp()):
  def this(device: Int) = {
    this()
    try cudaGetDeviceProperties(prop, device) catch case _: Throwable => ()
  }

  def name: String =
    try
      val bp = prop.name()
      if bp != null then bp.getString else s"CUDA Device"
    catch case _: Throwable => s"CUDA Device"

  def totalGlobalMem: Long = try prop.totalGlobalMem catch case _: Throwable => 0L
  def sharedMemPerBlock: Long = try prop.sharedMemPerBlock catch case _: Throwable => 0L
  def regsPerBlock: Int = try prop.regsPerBlock catch case _: Throwable => 0
  def warpSize: Int = try prop.warpSize catch case _: Throwable => 32
  def maxThreadsPerBlock: Int = try prop.maxThreadsPerBlock catch case _: Throwable => 1024
  def memPitch: Long = try prop.memPitch catch case _: Throwable => 0L

  def maxThreadsDim: (Int, Int, Int) =
    try (prop.maxThreadsDim(0), prop.maxThreadsDim(1), prop.maxThreadsDim(2))
    catch case _: Throwable => (1024, 1024, 64)

  def maxGridSize: (Int, Int, Int) =
    try (prop.maxGridSize(0), prop.maxGridSize(1), prop.maxGridSize(2))
    catch case _: Throwable => (2147483647, 65535, 65535)

  def totalConstMem: Long = try prop.totalConstMem catch case _: Throwable => 0L
  def major: Int = try prop.major catch case _: Throwable => 0
  def minor: Int = try prop.minor catch case _: Throwable => 0
  def computeCapability: (Int, Int) = (major, minor)
  def multiProcessorCount: Int = try prop.multiProcessorCount catch case _: Throwable => 0
  def num_sm: Int = multiProcessorCount
  def textureAlignment: Long = try prop.textureAlignment catch case _: Throwable => 0L
  def texturePitchAlignment: Long = try prop.texturePitchAlignment catch case _: Throwable => 0L
  def asyncEngineCount: Int = try prop.asyncEngineCount catch case _: Throwable => 0
  def unifiedAddressing: Boolean = try prop.unifiedAddressing != 0 catch case _: Throwable => false
  def maxThreadsPerMultiProcessor: Int = try prop.maxThreadsPerMultiProcessor catch case _: Throwable => 2048
  def max_threads_per_sm: Int = maxThreadsPerMultiProcessor
  def sharedMemPerMultiprocessor: Long = try prop.sharedMemPerMultiprocessor catch case _: Throwable => 0L
  def regsPerMultiprocessor: Int = try prop.regsPerMultiprocessor catch case _: Throwable => 0
  def globalL1CacheSupported: Boolean = try prop.globalL1CacheSupported != 0 catch case _: Throwable => false
  def localL1CacheSupported: Boolean = try prop.localL1CacheSupported != 0 catch case _: Throwable => false
  def max_shared_memory_per_block: Long = sharedMemPerBlock
  def max_threads_per_block: Int = maxThreadsPerBlock
  def max_block_dim: (Int, Int, Int) = maxThreadsDim
  def max_grid_dim: (Int, Int, Int) = maxGridSize
  def max_constant_memory: Long = totalConstMem
  def compute_capability: (Int, Int) = computeCapability
  def sharedMemPerSM: Long = sharedMemPerMultiprocessor
  def maxThreadsPerSM: Int = maxThreadsPerMultiProcessor
  def maxSharedMemPerBlock: Long = sharedMemPerBlock
  def integrated: Boolean = try prop.integrated != 0 catch case _: Throwable => false
  def canMapHostMemory: Boolean = try prop.canMapHostMemory != 0 catch case _: Throwable => false
  def eccEnabled: Boolean = try prop.ECCEnabled != 0 catch case _: Throwable => false
  def pciBusID: Int = try prop.pciBusID catch case _: Throwable => 0
  def pciDeviceID: Int = try prop.pciDeviceID catch case _: Throwable => 0
  def memoryClockRate: Long = 0L
  def memoryBusWidth: Int = try prop.memoryBusWidth catch case _: Throwable => 0
  def l2CacheSize: Long = try prop.l2CacheSize catch case _: Throwable => 0L
  def clockRate: Long = 0L
  def streamPrioritiesSupported: Boolean = try prop.streamPrioritiesSupported != 0 catch case _: Throwable => false
  def global_l1_cache_supported: Boolean = globalL1CacheSupported
  def local_l1_cache_supported: Boolean = localL1CacheSupported
  def num_async_engines: Int = asyncEngineCount
  def shared_memory: Long = sharedMemPerBlock
  def l2_cache_size: Long = l2CacheSize
  def memory_clock_rate: Long = memoryClockRate
  def clock_rate: Long = clockRate
  def memory_bus_width: Int = memoryBusWidth
  def is_multi_gpu_board: Boolean = try prop.isMultiGpuBoard != 0 catch case _: Throwable => false
  def cooperative_launch_supported: Boolean = try prop.cooperativeLaunch != 0 catch case _: Throwable => false
  def max_shared_mem: Long = sharedMemPerBlock
  def total_memory: Long = totalGlobalMem

/** CUDA Graph with native CUDART bindings */
class NativeCUDAGraph(
  val graph: CUgraph_st = new CUgraph_st(),
  val exec: CUgraphExec_st = new CUgraphExec_st()
) extends AutoCloseable:
  private var instantiated = false

  def create(): NativeCUDAGraph = {
    try
      val res = cudaGraphCreate(graph, 0)
      if (res == 0) this else this
    catch case _: Throwable => this
  }

  def instantiate(): Int = {
    try
      val res = cudaGraphInstantiate(exec, graph, 0)
      if (res == 0) { instantiated = true; res } else res
    catch case _: Throwable => -1
  }

  def launch(stream: NativeCUDAStream): Int = {
    try
      if !instantiated then instantiate()
      cudaGraphLaunch(exec, stream.stream)
    catch case _: Throwable => -1
  }

  def launchOnDefaultStream(): Int = {
    try
      if !instantiated then instantiate()
      val defaultStream = new NativeCUDAStream().create()
      cudaGraphLaunch(exec, defaultStream.stream)
      defaultStream.destroy()
    catch case _: Throwable => -1
  }

  def destroy(): Int = {
    var res = 0
    try if instantiated then res |= cudaGraphExecDestroy(exec) catch case _: Throwable => ()
    try res |= cudaGraphDestroy(graph) catch case _: Throwable => ()
    res
  }

  override def close(): Unit = { try destroy() catch case _: Throwable => () }
  override def finalize(): Unit = { try destroy() catch case _: Throwable => () }

object NativeCUDAGraph:
  def create(): NativeCUDAGraph = new NativeCUDAGraph().create()

/** Runtime API with native CUDART bindings */
object CUDARuntimeNatives:
  private var initialized = false

  def initialize(): Boolean = {
    if initialized then return true
    try
      val countPtr = new IntPointer(1L)
      val res = cudaGetDeviceCount(countPtr)
      if res == 0 then { countPtr.close(); initialized = true; true }
      else { countPtr.close(); false }
    catch case _: Throwable => false
  }

  def getDeviceCount: Int = {
    try
      val countPtr = new IntPointer(1L)
      val res = cudaGetDeviceCount(countPtr)
      if res == 0 then { val c = countPtr.get(0).toInt; countPtr.close(); c } else { countPtr.close(); 0 }
    catch case _: Throwable => 0
  }

  def getCurrentDevice: Int = {
    try
      val devPtr = new IntPointer(1L)
      val res = cudaGetDevice(devPtr)
      if res == 0 then { val d = devPtr.get(0).toInt; devPtr.close(); d } else { devPtr.close(); 0 }
    catch case _: Throwable => 0
  }

  def setCurrentDevice(device: Int): Int =
    try cudaSetDevice(device) catch case _: Throwable => -1

  def getDeviceProperties(device: Int): NativeDeviceProperties =
    try new NativeDeviceProperties(device)
    catch case _: Throwable => new NativeDeviceProperties()

  def isAvailable: Boolean = getDeviceCount > 0

  def synchronize(): Unit =
    try
      val s = new NativeCUDAStream().create()
      s.synchronize()
      s.close()
    catch case _: Throwable => ()
