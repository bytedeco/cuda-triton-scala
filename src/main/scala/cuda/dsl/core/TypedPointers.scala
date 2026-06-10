package cuda.dsl.core

import org.bytedeco.javacpp.*
import cuda.dsl.runtime.DeviceSelector
import java.lang.reflect.Field
import cuda.dsl.core.Types.{given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Byte,
  given_compileTime_Float, given_compileTime_Double, given_compileTime_Int, given_compileTime_Long, given_compileTime_Byte}

/** Typed pointer classes extending JavaCPP Pointer types.
  * These provide type-safe access to CUDA device memory.
  */

// =============================================================================
// FloatPtr - extends FloatPointer for float* in CUDA
// =============================================================================

/** Float pointer for CUDA device memory (float*). */
class FloatPtr private[core] (
    private[core] val ptr: cuda.dsl.core.Ptr[Float]
) extends org.bytedeco.javacpp.FloatPointer {

  // Initialize from the internal Ptr address
  address = ptr.rawAddress

  /** Get underlying Ptr */
  def toPtr: cuda.dsl.core.Ptr[Float] = ptr

  /** Read element at index */
  def apply(idx: Int): Float = {
    val ops = summon[MemoryOps[Float]]
    ops.read(ptr, idx)
  }

  /** Write element at index */
  def update(idx: Int, value: Float): Unit = {
    val ops = summon[MemoryOps[Float]]
    ops.write(ptr, idx, value)
  }

  /** Get raw device address */
  def rawAddress: Long = ptr.rawAddress

  /** Free device memory */
  def free(): Unit = DeviceSelector.getRuntime().free[Float](ptr)

  /** Element size in bytes */
  override def sizeof(): Int = 4

  override def toString: String = s"FloatPtr(${ptr.rawAddress})"
}

object FloatPtr {
  /** Allocate device memory for n floats */
  def alloc(n: Int): FloatPtr = {
    val device = DeviceSelector.getRuntime()
    val p = device.malloc[Float](n)
    new FloatPtr(p)
  }

  /** Create from device address */
  def fromAddress(addr: Long): FloatPtr = {
    val p = cuda.dsl.core.Ptr.fromAddress[Float](addr)
    new FloatPtr(p)
  }

  /** Zero-initialize */
  def zeros(n: Int): FloatPtr = {
    val p = cuda.dsl.core.Ptr.zeros[Float](n)
    new FloatPtr(p)
  }

  /** Allocate and fill with value */
  def filled(n: Int, value: Float): FloatPtr = {
    val p = cuda.dsl.core.Ptr.filled[Float](n, value)
    new FloatPtr(p)
  }
}

// =============================================================================
// DoublePtr - extends DoublePointer for double* in CUDA
// =============================================================================

/** Double pointer for CUDA device memory (double*). */
class DoublePtr private[core] (
    private[core] val ptr: cuda.dsl.core.Ptr[Double]
) extends org.bytedeco.javacpp.DoublePointer {

  address = ptr.rawAddress

  def toPtr: cuda.dsl.core.Ptr[Double] = ptr

  def apply(idx: Int): Double = {
    val ops = summon[MemoryOps[Double]]
    ops.read(ptr, idx)
  }

  def update(idx: Int, value: Double): Unit = {
    val ops = summon[MemoryOps[Double]]
    ops.write(ptr, idx, value)
  }

  def rawAddress: Long = ptr.rawAddress
  def free(): Unit = DeviceSelector.getRuntime().free[Double](ptr)
  override def sizeof(): Int = 8
  override def toString: String = s"DoublePtr(${ptr.rawAddress})"
}

object DoublePtr {
  def alloc(n: Int): DoublePtr = {
    val device = DeviceSelector.getRuntime()
    new DoublePtr(device.malloc[Double](n))
  }

  def fromAddress(addr: Long): DoublePtr = {
    new DoublePtr(cuda.dsl.core.Ptr.fromAddress[Double](addr))
  }

  def zeros(n: Int): DoublePtr = {
    val p = cuda.dsl.core.Ptr.zeros[Double](n)
    new DoublePtr(p)
  }

  def filled(n: Int, value: Double): DoublePtr = {
    val p = cuda.dsl.core.Ptr.filled[Double](n, value)
    new DoublePtr(p)
  }
}

// =============================================================================
// IntPtr - extends IntPointer for int* in CUDA
// =============================================================================

/** Int pointer for CUDA device memory (int*). */
class IntPtr private[core] (
    private[core] val ptr: cuda.dsl.core.Ptr[Int]
) extends org.bytedeco.javacpp.IntPointer {

  address = ptr.rawAddress

  def toPtr: cuda.dsl.core.Ptr[Int] = ptr

  def apply(idx: Int): Int = {
    val ops = summon[MemoryOps[Int]]
    ops.read(ptr, idx)
  }

  def update(idx: Int, value: Int): Unit = {
    val ops = summon[MemoryOps[Int]]
    ops.write(ptr, idx, value)
  }

  def rawAddress: Long = ptr.rawAddress
  def free(): Unit = DeviceSelector.getRuntime().free[Int](ptr)
  override def sizeof(): Int = 4
  override def toString: String = s"IntPtr(${ptr.rawAddress})"
}

object IntPtr {
  def alloc(n: Int): IntPtr = {
    val device = DeviceSelector.getRuntime()
    new IntPtr(device.malloc[Int](n))
  }

  def fromAddress(addr: Long): IntPtr = {
    new IntPtr(cuda.dsl.core.Ptr.fromAddress[Int](addr))
  }

  def zeros(n: Int): IntPtr = {
    val p = cuda.dsl.core.Ptr.zeros[Int](n)
    new IntPtr(p)
  }

  def filled(n: Int, value: Int): IntPtr = {
    val p = cuda.dsl.core.Ptr.filled[Int](n, value)
    new IntPtr(p)
  }
}

// =============================================================================
// LongPtr - extends LongPointer for long* in CUDA
// =============================================================================

/** Long pointer for CUDA device memory (long*). */
class LongPtr private[core] (
    private[core] val ptr: cuda.dsl.core.Ptr[Long]
) extends org.bytedeco.javacpp.LongPointer {

  address = ptr.rawAddress

  def toPtr: cuda.dsl.core.Ptr[Long] = ptr

  def apply(idx: Int): Long = {
    val ops = summon[MemoryOps[Long]]
    ops.read(ptr, idx)
  }

  def update(idx: Int, value: Long): Unit = {
    val ops = summon[MemoryOps[Long]]
    ops.write(ptr, idx, value)
  }

  def rawAddress: Long = ptr.rawAddress
  def free(): Unit = DeviceSelector.getRuntime().free[Long](ptr)
  override def sizeof(): Int = 8
  override def toString: String = s"LongPtr(${ptr.rawAddress})"
}

object LongPtr {
  def alloc(n: Int): LongPtr = {
    val device = DeviceSelector.getRuntime()
    new LongPtr(device.malloc[Long](n))
  }

  def fromAddress(addr: Long): LongPtr = {
    new LongPtr(cuda.dsl.core.Ptr.fromAddress[Long](addr))
  }

  def zeros(n: Int): LongPtr = {
    val p = cuda.dsl.core.Ptr.zeros[Long](n)
    new LongPtr(p)
  }

  def filled(n: Int, value: Long): LongPtr = {
    val p = cuda.dsl.core.Ptr.filled[Long](n, value)
    new LongPtr(p)
  }
}

// =============================================================================
// Type aliases for convenience
// =============================================================================

/** Short alias for FloatPtr */
type FP = FloatPtr

/** Short alias for DoublePtr */
type DP = DoublePtr

/** Short alias for IntPtr */
type IP = IntPtr

/** Short alias for LongPtr */
type LP = LongPtr

// =============================================================================
// PointerPointer - extends JavaCPP PointerPointer for void** in CUDA
// =============================================================================

/** Pointer pointer for CUDA device memory (void**, used for multi-pointer kernels).
  * Commonly used in batched GEMM, attention with multiple Q/K/V heads, etc.
  */
class PointerPtr private[core] (
    private[core] val ptr: cuda.dsl.core.Ptr[Long]
) extends org.bytedeco.javacpp.PointerPointer {

  address = ptr.rawAddress

  def toPtr: cuda.dsl.core.Ptr[Long] = ptr

  /** Read element pointer at index */
  def apply(idx: Int): Long = {
    val ops = summon[MemoryOps[Long]]
    ops.read(ptr, idx)
  }

  /** Write element pointer at index */
  def update(idx: Int, value: Long): Unit = {
    val ops = summon[MemoryOps[Long]]
    ops.write(ptr, idx, value)
  }

  def rawAddress: Long = ptr.rawAddress
  def free(): Unit = DeviceSelector.getRuntime().free[Long](ptr)
  override def sizeof(): Int = 8
  override def toString: String = s"PointerPtr(${ptr.rawAddress})"

  /** Get underlying PointerPointer.
    * Uses no-arg constructor + reflection to avoid native allocation.
    */
  def toPointerPointer: org.bytedeco.javacpp.PointerPointer[org.bytedeco.javacpp.Pointer] = {
    val pp = new org.bytedeco.javacpp.PointerPointer[org.bytedeco.javacpp.Pointer]()
    try {
      val f = classOf[org.bytedeco.javacpp.Pointer].getDeclaredField("address")
      f.setAccessible(true)
      f.setLong(pp, ptr.rawAddress)
    } catch {
      case e: Exception => throw new RuntimeException("Failed to set pointer address", e)
    }
    pp
  }
}

object PointerPtr {
  /** Allocate device memory for n pointers (8 bytes each) */
  def alloc(n: Int): PointerPtr = {
    val device = DeviceSelector.getRuntime()
    val p = device.malloc[Long](n)
    new PointerPtr(p)
  }

  /** Create from device address */
  def fromAddress(addr: Long): PointerPtr = {
    new PointerPtr(cuda.dsl.core.Ptr.fromAddress[Long](addr))
  }

  /** Zero-initialize */
  def zeros(n: Int): PointerPtr = {
    val p = cuda.dsl.core.Ptr.zeros[Long](n)
    new PointerPtr(p)
  }
}

// =============================================================================
// BytePtr - extends JavaCPP BytePointer for void* / int8_t* in CUDA
// =============================================================================

/** Byte pointer for CUDA device memory (void*, int8_t*).
  * Used for generic memory operations, mixed-type data, and raw device memory access.
  */
class BytePtr private[core] (
    private[core] val ptr: cuda.dsl.core.Ptr[Byte]
) extends org.bytedeco.javacpp.BytePointer {

  address = ptr.rawAddress

  def toPtr: cuda.dsl.core.Ptr[Byte] = ptr

  /** Read byte at index */
  def apply(idx: Int): Byte = {
    val ops = summon[MemoryOps[Byte]]
    ops.read(ptr, idx)
  }

  /** Write byte at index */
  def update(idx: Int, value: Byte): Unit = {
    val ops = summon[MemoryOps[Byte]]
    ops.write(ptr, idx, value)
  }

  def rawAddress: Long = ptr.rawAddress
  def free(): Unit = DeviceSelector.getRuntime().free[Byte](ptr)
  override def sizeof(): Int = 1
  override def toString: String = s"BytePtr(${ptr.rawAddress})"

  /** Convert to typed FloatPtr (unsafe but useful for void* cast) */
  def asFloatPtr: FloatPtr = FloatPtr.fromAddress(rawAddress)

  /** Convert to typed DoublePtr (unsafe but useful for void* cast) */
  def asDoublePtr: DoublePtr = DoublePtr.fromAddress(rawAddress)

  /** Convert to typed IntPtr (unsafe but useful for void* cast) */
  def asIntPtr: IntPtr = IntPtr.fromAddress(rawAddress)

  /** Convert to typed LongPtr (unsafe but useful for void* cast) */
  def asLongPtr: LongPtr = LongPtr.fromAddress(rawAddress)
}

object BytePtr {
  /** Allocate device memory for n bytes */
  def alloc(n: Int): BytePtr = {
    val device = DeviceSelector.getRuntime()
    val p = device.malloc[Byte](n)
    new BytePtr(p)
  }

  /** Create from device address */
  def fromAddress(addr: Long): BytePtr = {
    new BytePtr(cuda.dsl.core.Ptr.fromAddress[Byte](addr))
  }

  /** Zero-initialize */
  def zeros(n: Int): BytePtr = {
    val p = cuda.dsl.core.Ptr.zeros[Byte](n)
    new BytePtr(p)
  }

  /** Allocate and fill with value */
  def filled(n: Int, value: Byte): BytePtr = {
    val p = cuda.dsl.core.Ptr.filled[Byte](n, value)
    new BytePtr(p)
  }
}

/** Short alias for BytePtr */
type BP = BytePtr

/** Short alias for PointerPtr */
type PP = PointerPtr
