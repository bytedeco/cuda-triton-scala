package cuda.dsl.driver

/**
 * Java-friendly bridge for Driver API.
 */
object DriverJavaBridge:

  def initialize(): Boolean = CUDARuntimeNatives.initialize()

  def getDeviceCount: Int = CUDARuntimeNatives.getDeviceCount

  def getCurrentDevice: Int = CUDARuntimeNatives.getCurrentDevice

  def setCurrentDevice(device: Int): Int = CUDARuntimeNatives.setCurrentDevice(device)

  def getDeviceProperties(device: Int): NativeDeviceProperties =
    CUDARuntimeNatives.getDeviceProperties(device)

  def synchronize(): Unit = CUDARuntimeNatives.synchronize()

  def createStream(flags: Int = 0): NativeCUDAStream = NativeCUDAStream.create(flags)

  def createEvent(flags: Int = 0): NativeCUDAEvent = NativeCUDAEvent.create(flags)
