package cuda.dsl.runtime

import org.bytedeco.javacpp._
import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import java.util.Map
import java.util.HashMap

/** MPS (Metal Performance Shaders) helper for PyTorch operations.
 *  This is a Scala wrapper around PyTorch's MPS backend for GPU operations on Apple Silicon.
 */
object MPSHelper {
  private val TAG = "[MPSHelper] "

  // Simple counter to use as key
  private var tensorIdCounter: Long = 0
  private val tensorById: Map[Long, Tensor] = new HashMap[Long, Tensor]()

  /** Check if MPS is available */
  def hasMPS(): Boolean = {
    try {
      println(TAG + "Checking MPS availability...")
      val available = torch.hasMPS()
      println(TAG + "hasMPS() = " + available)
      available
    } catch {
      case e: Exception =>
        println(TAG + "hasMPS check failed: " + e.getMessage)
        e.printStackTrace()
        false
    }
  }

  /** Create an empty tensor on MPS device for Float type
   *  Returns a unique ID that can be used to reference this tensor
   */
  def createMPSFloatTensor(size: Long): Long = {
    try {
      println(TAG + "Creating MPS tensor, size=" + size)

      // Create device for MPS
      val mpsDevice = new Device(torch.DeviceType.MPS)

      // Create tensor options
      val opts = new TensorOptions()
        .device(new DeviceOptional(mpsDevice))
        .dtype(new ScalarTypeOptional(torch.ScalarType.Float))

      // Create tensor using zeros with shape array
      val shape: Array[Long] = Array(size)
      val tensor = torch.zeros(shape, opts)

      // Use a unique ID as key
      tensorIdCounter += 1
      val tensorId = tensorIdCounter
      tensorById.put(tensorId, tensor)

      val addr = tensor.data_ptr().address()
      println(TAG + "Created tensor id=" + tensorId + " at address 0x" + java.lang.Long.toHexString(addr) + " with " + size + " elements")

      tensorId
    } catch {
      case e: Exception =>
        println(TAG + "createMPSFloatTensor failed: " + e.getMessage)
        e.printStackTrace()
        -1
    }
  }

  /** Copy data from host array to MPS tensor */
  def copyHostToDevice(tensorId: Long, data: Array[Float], size: Long): Boolean = {
    try {
      val tensor = tensorById.get(tensorId)
      if (tensor == null) {
        println(TAG + "copyHostToDevice: tensor not found for id " + tensorId)
        return false
      }

      println(TAG + "Copying " + size + " floats to device tensor id=" + tensorId)
      println(TAG + "  Host array length: " + data.length)

      // Create host tensor using torch.from_blob with explicit size
      // Use FloatPointer directly with the array - convert to FloatBuffer
      val floatBuffer = java.nio.FloatBuffer.wrap(data)
      val floatPtr = new FloatPointer(floatBuffer)
      val shape: Array[Long] = Array(size)

      // Create tensor on CPU with explicit shape using from_blob
      val cpuOpts = new TensorOptions()
        .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
      val hostTensor = torch.from_blob(floatPtr, shape, cpuOpts)

      // Copy to device tensor
      tensor.copy_(hostTensor)

      println(TAG + "Copy successful")
      true
    } catch {
      case e: Exception =>
        println(TAG + "copyHostToDevice failed: " + e.getMessage)
        e.printStackTrace()
        false
    }
  }

  /** Copy data from MPS tensor to host array */
  def copyDeviceToHost(tensorId: Long, data: Array[Float], size: Long): Boolean = {
    try {
      val tensor = tensorById.get(tensorId)
      if (tensor == null) {
        println(TAG + "copyDeviceToHost: tensor not found for id " + tensorId)
        return false
      }

      println(TAG + "Copying " + size + " floats from device tensor id=" + tensorId)

      // Move tensor to CPU
      val cpuDevice = new Device(torch.DeviceType.CPU)
      val cpuTensor = tensor.to(cpuDevice, torch.ScalarType.Float)

      // Get data pointer and create a FloatPointer from it
      val dataPtr = cpuTensor.data_ptr()
      // Position to start and read floats directly
      dataPtr.position(0)

      // Use FloatPointer to read from the pointer
      val floatPtr = new FloatPointer(dataPtr)

      // Copy to array
      var i = 0
      while (i < size && i < data.length) {
        data(i) = floatPtr.get(i)
        i += 1
      }

      println(TAG + "Copy successful")
      true
    } catch {
      case e: Exception =>
        println(TAG + "copyDeviceToHost failed: " + e.getMessage)
        e.printStackTrace()
        false
    }
  }

  /** Free an MPS tensor */
  def freeMPSTensor(tensorId: Long): Unit = {
    try {
      println(TAG + "Freeing tensor id=" + tensorId)
      tensorById.remove(tensorId)
    } catch {
      case e: Exception =>
        println(TAG + "freeMPSTensor failed: " + e.getMessage)
    }
  }
}
