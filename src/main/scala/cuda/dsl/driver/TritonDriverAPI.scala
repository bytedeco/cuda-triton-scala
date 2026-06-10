package cuda.dsl.driver

import cuda.dsl.runtime.*
import scala.collection.mutable

case class DeviceProperties(
  name: String,
  num_sm: Int,
  num_async_engines: Int,
  max_shared_memory: Long,
  compute_capability: (Int, Int),
  max_threads_per_sm: Int,
  max_threads_per_block: Int,
  warp_size: Int,
  max_grid_dim: (Int, Int, Int),
  max_block_dim: (Int, Int, Int),
  total_memory: Long,
  l2_cache_size: Long,
  clock_rate: Long,
  memory_clock_rate: Long,
  memory_bus_width: Int
)

class Device(val id: Int):
  private var _current_stream: Option[CUDAStream] = None

  def this() = this(0)

  def current_stream: CUDAStream = _current_stream.getOrElse(CUDAStream.default(this))

  def set_current_stream(stream: CUDAStream): Unit = _current_stream = Some(stream)

  def get_device_id: Int = id

  def synchronize(): Unit = CUDARuntimeNatives.synchronize()

  def get_attributes: DeviceProperties = DriverAPI.get_device_properties(id)

  override def toString: String = s"Device($id, ${get_attributes.name})"

class CUDAStream(val device: Device, val handle: Long = 0):
  private val nativeStream = NativeCUDAStream.create()

  def this() = this(Device(0), 0)

  def synchronize(): Unit = {
    nativeStream.synchronize()
    ()
  }

  def query: Boolean = nativeStream.query == 0

  def wait(event: CUDAEvent): Unit = nativeStream.synchronize()

  override def toString: String = s"CUDAStream(device=${device.id})"

object CUDAStream:
  def default(device: Device): CUDAStream = new CUDAStream(device, 0)

  def create(device: Device, priority: Int = 0): CUDAStream = {
    val s = new CUDAStream(device)
    s
  }

class CUDAEvent(val device: Device, val handle: Long = 0):
  private val nativeEvent = NativeCUDAEvent.create()

  def this() = this(Device(0), 0)

  private[driver] def native: NativeCUDAEvent = nativeEvent

  def record(stream: CUDAStream): Unit = {
    val ns = NativeCUDAStream.create()
    nativeEvent.record(ns)
  }

  def elapsed_time(end: CUDAEvent): Float = nativeEvent.elapsedTimeMs(end.native)

  def query: Boolean = nativeEvent.query == 0

  def synchronize(): Unit = nativeEvent.synchronize()

  override def toString: String = s"CUDAEvent(device=${device.id})"

object CUDAEvent:
  def create(device: Device, enable_timing: Boolean = true): CUDAEvent = new CUDAEvent(device)
  def create(): CUDAEvent = new CUDAEvent()

class MemoryPool(val device: Device):
  private val allocated = java.util.concurrent.atomic.AtomicLong(0)
  private val peak = java.util.concurrent.atomic.AtomicLong(0)
  private val resident = mutable.Map[Long, Long]()

  def allocate(size: Long): Long = {
    allocated.addAndGet(size)
    peak.updateAndGet(v => math.max(v, allocated.get()))
    val ptr = System.identityHashCode(this).toLong * size
    resident.put(ptr, size)
    ptr
  }

  def deallocate(ptr: Long): Unit = {
    resident.get(ptr).foreach(s => allocated.addAndGet(-s))
    resident.remove(ptr)
  }

  def reset_peak_stats(): Unit = peak.set(0)
  def get_active_memory: Long = allocated.get()
  def get_peak_memory: Long = peak.get()

object DriverAPI:
  private var current_id = 0
  private val devs = mutable.Map[Int, Device]()
  private val props = mutable.Map[Int, DeviceProperties]()
  private val pools = mutable.Map[Int, MemoryPool]()
  private var inited = false

  private def nativeToDeviceProperties(id: Int): DeviceProperties = {
    val np = new NativeDeviceProperties(id)
    DeviceProperties(
      name = np.name,
      num_sm = np.multiProcessorCount,
      num_async_engines = np.asyncEngineCount,
      max_shared_memory = np.sharedMemPerBlock,
      compute_capability = np.computeCapability,
      max_threads_per_sm = np.maxThreadsPerSM,
      max_threads_per_block = np.maxThreadsPerBlock,
      warp_size = np.warpSize,
      max_grid_dim = np.maxGridSize,
      max_block_dim = np.maxThreadsDim,
      total_memory = np.totalGlobalMem,
      l2_cache_size = np.l2CacheSize,
      clock_rate = np.clockRate * 1000,
      memory_clock_rate = np.memoryClockRate * 1000,
      memory_bus_width = np.memoryBusWidth
    )
  }

  private def stubProps(id: Int): DeviceProperties = DeviceProperties(
    name = s"CUDA Device $id",
    num_sm = 128,
    num_async_engines = 2,
    max_shared_memory = 48L * 1024,
    compute_capability = (9, 0),
    max_threads_per_sm = 2048,
    max_threads_per_block = 1024,
    warp_size = 32,
    max_grid_dim = (2147483647, 65535, 65535),
    max_block_dim = (1024, 1024, 64),
    total_memory = 80L * 1024 * 1024 * 1024,
    l2_cache_size = 50L * 1024 * 1024,
    clock_rate = 1098750,
    memory_clock_rate = 2600000,
    memory_bus_width = 5120
  )

  def initialize(): Unit = {
    if (inited) return
    try
      val numDevices = CUDARuntimeNatives.getDeviceCount
      if numDevices > 0 then
        for (i <- 0 until numDevices) {
          devs(i) = Device(i)
          props(i) = nativeToDeviceProperties(i)
          pools(i) = MemoryPool(devs(i))
        }
      else
        devs(0) = Device(0)
        props(0) = stubProps(0)
        pools(0) = MemoryPool(devs(0))
    catch
      case _: Throwable =>
        devs(0) = Device(0)
        props(0) = stubProps(0)
        pools(0) = MemoryPool(devs(0))
    inited = true
  }

  def get_current_device: Device = { initialize(); devs.getOrElseUpdate(current_id, Device(current_id)) }
  def set_current_device(device: Device): Unit = { current_id = device.id }
  def get_device_properties(device: Int): DeviceProperties = {
    initialize()
    props.getOrElse(device, nativeToDeviceProperties(device))
  }
  def get_num_devices: Int = { initialize(); devs.size.max(1) }
  def get_active_memory(): Long = { initialize(); pools.get(current_id).map(_.get_active_memory).getOrElse(0L) }
  def get_peak_memory(): Long = { initialize(); pools.get(current_id).map(_.get_peak_memory).getOrElse(0L) }
  def reset_peak_memory_stats(device: Int): Unit = pools.get(device).foreach(_.reset_peak_stats())
  def get_total_memory(device: Int): Long = get_device_properties(device).total_memory
  def get_available_memory(device: Int): Long = get_total_memory(device) - get_active_memory()
  def set_shared_memory_config(config: String): Unit = ()
  def get_shared_memory_config(): String = "default"
  def is_cuda_available: Boolean = CUDARuntimeNatives.isAvailable
  def get_device_name(device: Int = current_id): String = get_device_properties(device).name
