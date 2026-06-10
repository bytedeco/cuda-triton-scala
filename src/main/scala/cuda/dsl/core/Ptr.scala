package cuda.dsl.core

import cuda.dsl.runtime.CUDARuntime
import java.lang.reflect.Field

/** Device pointer type with element access.
  * Ptr[T] represents a pointer to T on the CUDA device.
  */
class Ptr[T] private[core] (val address: Long) {

  /** Read element at index */
  inline def apply(idx: Int)(using ops: MemoryOps[T]): T = ops.read(this, idx)

  /** Write element at index */
  inline def update(idx: Int, value: T)(using ops: MemoryOps[T]): Unit = ops.write(this, idx, value)

  /** Get raw address */
  inline def rawAddress: Long = address

  /** Check if pointer is null */
  inline def isNull: Boolean = address == 0L

  /** Pointer arithmetic - advance by n elements */
  inline def + (n: Int)(using ops: MemoryOps[T]): Ptr[T] = ops.offset(this, n)

  /** Pointer arithmetic - move back by n elements */
  inline def - (n: Int)(using ops: MemoryOps[T]): Ptr[T] = ops.offset(this, -n)

  /** Dereference - equivalent to apply(0) */
  inline def *(using ops: MemoryOps[T]): T = ops.read(this, 0)

  /** Offset pointer */
  inline def offset(n: Int)(using ops: MemoryOps[T]): Ptr[T] = ops.offset(this, n)

  /** Convert to JavaCPP Pointer (returns BytePointer wrapping raw address).
    * For typed access, use implicit conversions in PtrImplicits.
    * Uses no-arg constructor + reflection to avoid native allocation.
    */
  def toPointer: org.bytedeco.javacpp.BytePointer = {
    val bp = new org.bytedeco.javacpp.BytePointer()
    try {
      val f = classOf[org.bytedeco.javacpp.Pointer].getDeclaredField("address")
      f.setAccessible(true)
      f.setLong(bp, address)
    } catch {
      case e: Exception => throw new RuntimeException("Failed to set pointer address", e)
    }
    bp
  }

  override def toString: String = s"Ptr($address)"
}

/** Extension methods to convert Ptr[T] to/from JavaCPP Pointer types.
  * These are in a separate package object so they are available via implicit scope.
  */
package object PtrImplicits {
  import org.bytedeco.javacpp.*
  import java.lang.reflect.Field

  /** Create a JavaCPP Pointer wrapping a raw device address WITHOUT native allocation.
    * Uses no-arg constructor + reflection to set the protected address field.
    * This mirrors the fix applied to the Java Ptr classes (IntPtr, FloatPtr, etc.).
    */
  private def createPointerFromAddress(cls: Class[_], addr: Long): Pointer =
    try
      val ctor = cls.getConstructor()
      val ptr = ctor.newInstance()
      val addressField = classOf[Pointer].getDeclaredField("address")
      addressField.setAccessible(true)
      addressField.setLong(ptr, addr)
      ptr.asInstanceOf[Pointer]
    catch case e: Exception =>
      throw new RuntimeException(s"Failed to create ${cls.getSimpleName} from address 0x${addr.toHexString}", e)

  /** Convert DSL Ptr[Float] to JavaCPP FloatPointer.
    * Creates a FloatPointer wrapping the raw device address without native allocation.
    */
  given floatPtrFromPtr: Conversion[Ptr[Float], FloatPointer] = ptr => {
    val fp = createPointerFromAddress(classOf[FloatPointer], ptr.address).asInstanceOf[FloatPointer]
    fp.limit(ptr.address + 4)
    fp.position(0)
    fp
  }

  /** Convert DSL Ptr[Double] to JavaCPP DoublePointer. */
  given doublePtrFromPtr: Conversion[Ptr[Double], DoublePointer] = ptr => {
    val dp = createPointerFromAddress(classOf[DoublePointer], ptr.address).asInstanceOf[DoublePointer]
    dp.limit(ptr.address + 8)
    dp.position(0)
    dp
  }

  /** Convert DSL Ptr[Int] to JavaCPP IntPointer. */
  given intPtrFromPtr: Conversion[Ptr[Int], IntPointer] = ptr => {
    val ip = createPointerFromAddress(classOf[IntPointer], ptr.address).asInstanceOf[IntPointer]
    ip.limit(ptr.address + 4)
    ip.position(0)
    ip
  }

  /** Convert DSL Ptr[Long] to JavaCPP LongPointer. */
  given longPtrFromPtr: Conversion[Ptr[Long], LongPointer] = ptr => {
    val lp = createPointerFromAddress(classOf[LongPointer], ptr.address).asInstanceOf[LongPointer]
    lp.limit(ptr.address + 8)
    lp.position(0)
    lp
  }

  /** Convert DSL Ptr[Short] to JavaCPP ShortPointer. */
  given shortPtrFromPtr: Conversion[Ptr[Short], ShortPointer] = ptr => {
    val sp = createPointerFromAddress(classOf[ShortPointer], ptr.address).asInstanceOf[ShortPointer]
    sp.limit(ptr.address + 2)
    sp.position(0)
    sp
  }

  /** Convert DSL Ptr[Byte] to JavaCPP BytePointer. */
  given bytePtrFromPtr: Conversion[Ptr[Byte], BytePointer] = ptr =>
    createPointerFromAddress(classOf[BytePointer], ptr.address).asInstanceOf[BytePointer]

  /** Convert JavaCPP Pointer address to DSL Ptr[Float] */
  given ptrFromFloatPointer: Conversion[FloatPointer, Ptr[Float]] = p =>
    Ptr.fromAddress[Float](p.address)

  /** Convert JavaCPP Pointer address to DSL Ptr[Double] */
  given ptrFromDoublePointer: Conversion[DoublePointer, Ptr[Double]] = p =>
    Ptr.fromAddress[Double](p.address)

  /** Convert JavaCPP Pointer address to DSL Ptr[Int] */
  given ptrFromIntPointer: Conversion[IntPointer, Ptr[Int]] = p =>
    Ptr.fromAddress[Int](p.address)

  /** Convert JavaCPP Pointer address to DSL Ptr[Long] */
  given ptrFromLongPointer: Conversion[LongPointer, Ptr[Long]] = p =>
    Ptr.fromAddress[Long](p.address)

  /** Generic conversion: any JavaCPP Pointer to Ptr[Float] (unsafe but useful) */
  given ptrFromPointer: Conversion[Pointer, Ptr[Float]] = p =>
    Ptr.fromAddress[Float](p.address)
}

/** Companion object for Ptr with factory methods and type class evidence */
object Ptr {
  import scala.compiletime.*

  /** Allocate device memory for n elements of type T */
  inline def alloc[T](n: Int)(using ops: MemoryOps[T], ct: compileTime[T]): Ptr[T] =
    ops.alloc(ct.makeDefault(), n)

  /** Free device memory */
  inline def free[T](ptr: Ptr[T])(using ops: MemoryOps[T]): Unit = ops.free(ptr)

  /** Create Ptr from raw address */
  def fromAddress[T](addr: Long): Ptr[T] = new Ptr[T](addr)

  /** Create Ptr from host array */
  inline def fromArray[T](arr: Array[T], n: Int)(using ops: MemoryOps[T]): Ptr[T] =
    ops.fromHostArray(arr, n)

  /** Zero-initialize a device pointer */
  inline def zeros[T](n: Int)(using ops: MemoryOps[T], ct: compileTime[T]): Ptr[T] = {
    val ptr = alloc[T](n)
    ops.memset(ptr, 0, n)
    ptr
  }

  /** Allocate and fill with value */
  inline def filled[T](n: Int, value: T)(using ops: MemoryOps[T], ct: compileTime[T]): Ptr[T] = {
    val ptr = alloc[T](n)
    ops.memset(ptr, value, n)
    ptr
  }
}

/** Type class for memory operations on type T */
trait MemoryOps[T] {
  def elementSize: Int
  def read(ptr: Ptr[T], idx: Int): T
  def write(ptr: Ptr[T], idx: Int, value: T): Unit
  def alloc(default: T, n: Int): Ptr[T]
  def free(ptr: Ptr[T]): Unit
  def offset(ptr: Ptr[T], n: Int): Ptr[T]
  def memset(ptr: Ptr[T], value: Any, n: Int): Unit
  def memcpy(dst: Ptr[T], src: Ptr[T], n: Int, kind: MemcpyKind): Unit
  def fromHostArray(arr: Array[T], n: Int): Ptr[T]
  def toHostArray(ptr: Ptr[T], n: Int): Array[T]
  def createArray(n: Int): Array[T]
}

/** compileTime type class for compile-time type information */
trait compileTime[T] {
  def makeDefault(): T
  def cudaTypeName: String
  def createArray(n: Int): Array[T]
}

/** Memory copy kinds */
enum MemcpyKind {
  case HostToDevice
  case DeviceToHost
  case DeviceToDevice
  case HostToHost
}

/** Memory copy parameters */
case class Memcpy(dst: Ptr[?], src: Ptr[?], n: Int, kind: MemcpyKind)

object Memcpy {
  inline def apply[T](dst: Ptr[T], src: Ptr[T], n: Int, kind: MemcpyKind)(using ops: MemoryOps[T]): Unit =
    ops.memcpy(dst, src, n, kind)
}
