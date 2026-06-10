package cuda.dsl.core

import cuda.dsl.runtime.DeviceSelector

/** Primitive CUDA types with compile-time type information */
object Types {

  /** Float type - 32-bit floating point */
  type Float = scala.Float
  object Float {
    inline def apply(v: scala.Float): Float = v
    inline def unapply(f: Float): Some[scala.Float] = Some(f)
  }

  given compileTime[Float] with {
    inline def makeDefault(): Float = 0.0f
    inline def cudaTypeName: String = "float"
    inline def createArray(n: Int): Array[Float] = Array.ofDim[Float](n)
  }

  given MemoryOps[Float] with {
    val elementSize = 4
    inline def read(ptr: Ptr[Float], idx: Int): Float = {
      val arr = new Array[Float](1)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr + idx, 1)
      arr(0)
    }
    inline def write(ptr: Ptr[Float], idx: Int, value: Float): Unit = {
      val arr = Array(value)
      val dstPtr = ptr + idx
      DeviceSelector.getRuntime().memcpyHtoD(dstPtr, arr, 1)
    }
    inline def alloc(default: Float, n: Int): Ptr[Float] = {
      val p = DeviceSelector.getRuntime().malloc[Float](n)
      if (default != 0.0f) {
        val arr = Array.fill(n)(default)
        DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      }
      p
    }
    inline def free(ptr: Ptr[Float]): Unit = DeviceSelector.getRuntime().free(ptr)
    inline def offset(ptr: Ptr[Float], n: Int): Ptr[Float] = {
      // For MPS, tensor IDs are encoded with a marker bit to distinguish from real addresses
      // Address format: [marker bit 63][tensor id bits 32-62][offset bits 0-31]
      // This preserves tensor ID through pointer arithmetic
      val MARKER: Long = 0x8000000000000000L
      val TENSOR_ID_MASK: Long = 0x7FFFFFFFFF000000L  // Bits 32-62 (tensor id)
      val OFFSET_MASK: Long = 0x00000000FFFFFFFFL     // Bits 0-31 (offset)

      val addr = ptr.rawAddress
      if ((addr & MARKER) != 0) {
        // MPS pointer - preserve tensor ID and add offset
        val tensorId = (addr & TENSOR_ID_MASK) >> 32
        val currentOffset = addr & OFFSET_MASK
        val newOffset = currentOffset + n * elementSize
        Ptr.fromAddress[Float](MARKER | (tensorId << 32) | newOffset)
      } else {
        // Regular address (CUDA/CPU)
        Ptr.fromAddress[Float](ptr.rawAddress + n * elementSize)
      }
    }
    inline def memset(ptr: Ptr[Float], value: Any, n: Int): Unit = {
      val arr = Array.fill(n)(value.asInstanceOf[Float])
      DeviceSelector.getRuntime().memcpyHtoD(ptr, arr, n)
    }
    inline def memcpy(dst: Ptr[Float], src: Ptr[Float], n: Int, kind: MemcpyKind): Unit = {
      kind match {
        case MemcpyKind.HostToDevice => sys.error("Use fromHostArray for H2D")
        case MemcpyKind.DeviceToHost => sys.error("Use toHostArray for D2H")
        case _ => DeviceSelector.getRuntime().memcpyDtoD(dst, src, n)
      }
    }
    inline def fromHostArray(arr: Array[Float], n: Int): Ptr[Float] = {
      val p = DeviceSelector.getRuntime().malloc[Float](n)
      DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      p
    }
    inline def toHostArray(ptr: Ptr[Float], n: Int): Array[Float] = {
      val arr = new Array[Float](n)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr, n)
      arr
    }
    inline def createArray(n: Int): Array[Float] = Array.ofDim[Float](n)
  }

  /** Double type - 64-bit floating point */
  type Double = scala.Double
  object Double {
    inline def apply(v: scala.Double): Double = v
    inline def unapply(d: Double): Some[scala.Double] = Some(d)
  }

  given compileTime[Double] with {
    inline def makeDefault(): Double = 0.0
    inline def cudaTypeName: String = "double"
    inline def createArray(n: Int): Array[Double] = Array.ofDim[Double](n)
  }

  given MemoryOps[Double] with {
    val elementSize = 8
    inline def read(ptr: Ptr[Double], idx: Int): Double = {
      val arr = new Array[Double](1)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr + idx, 1)
      arr(0)
    }
    inline def write(ptr: Ptr[Double], idx: Int, value: Double): Unit = {
      val arr = Array(value)
      DeviceSelector.getRuntime().memcpyHtoD(ptr + idx, arr, 1)
    }
    inline def alloc(default: Double, n: Int): Ptr[Double] = {
      val p = DeviceSelector.getRuntime().malloc[Double](n)
      if (default != 0.0) {
        val arr = Array.fill(n)(default)
        DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      }
      p
    }
    inline def free(ptr: Ptr[Double]): Unit = DeviceSelector.getRuntime().free(ptr)
    inline def offset(ptr: Ptr[Double], n: Int): Ptr[Double] = {
      Ptr.fromAddress[Double](ptr.rawAddress + n * elementSize)
    }
    inline def memset(ptr: Ptr[Double], value: Any, n: Int): Unit = {
      val arr = Array.fill(n)(value.asInstanceOf[Double])
      DeviceSelector.getRuntime().memcpyHtoD(ptr, arr, n)
    }
    inline def memcpy(dst: Ptr[Double], src: Ptr[Double], n: Int, kind: MemcpyKind): Unit = {
      kind match {
        case MemcpyKind.HostToDevice => sys.error("Use fromHostArray for H2D")
        case MemcpyKind.DeviceToHost => sys.error("Use toHostArray for D2H")
        case _ => DeviceSelector.getRuntime().memcpyDtoD(dst, src, n)
      }
    }
    inline def fromHostArray(arr: Array[Double], n: Int): Ptr[Double] = {
      val p = DeviceSelector.getRuntime().malloc[Double](n)
      DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      p
    }
    inline def toHostArray(ptr: Ptr[Double], n: Int): Array[Double] = {
      val arr = new Array[Double](n)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr, n)
      arr
    }
    inline def createArray(n: Int): Array[Double] = Array.ofDim[Double](n)
  }

  /** Int type - 32-bit signed integer */
  type Int = scala.Int
  object Int {
    inline def apply(v: scala.Int): Int = v
    inline def unapply(i: Int): Some[scala.Int] = Some(i)
  }

  given compileTime[Int] with {
    inline def makeDefault(): Int = 0
    inline def cudaTypeName: String = "int"
    inline def createArray(n: Int): Array[Int] = Array.ofDim[Int](n)
  }

  given MemoryOps[Int] with {
    val elementSize = 4
    inline def read(ptr: Ptr[Int], idx: Int): Int = {
      val arr = new Array[Int](1)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr + idx, 1)
      arr(0)
    }
    inline def write(ptr: Ptr[Int], idx: Int, value: Int): Unit = {
      val arr = Array(value)
      DeviceSelector.getRuntime().memcpyHtoD(ptr + idx, arr, 1)
    }
    inline def alloc(default: Int, n: Int): Ptr[Int] = {
      val p = DeviceSelector.getRuntime().malloc[Int](n)
      if (default != 0) {
        val arr = Array.fill(n)(default)
        DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      }
      p
    }
    inline def free(ptr: Ptr[Int]): Unit = DeviceSelector.getRuntime().free(ptr)
    inline def offset(ptr: Ptr[Int], n: Int): Ptr[Int] = {
      Ptr.fromAddress[Int](ptr.rawAddress + n * elementSize)
    }
    inline def memset(ptr: Ptr[Int], value: Any, n: Int): Unit = {
      val arr = Array.fill(n)(value.asInstanceOf[Int])
      DeviceSelector.getRuntime().memcpyHtoD(ptr, arr, n)
    }
    inline def memcpy(dst: Ptr[Int], src: Ptr[Int], n: Int, kind: MemcpyKind): Unit = {
      kind match {
        case MemcpyKind.HostToDevice => sys.error("Use fromHostArray for H2D")
        case MemcpyKind.DeviceToHost => sys.error("Use toHostArray for D2H")
        case _ => DeviceSelector.getRuntime().memcpyDtoD(dst, src, n)
      }
    }
    inline def fromHostArray(arr: Array[Int], n: Int): Ptr[Int] = {
      val p = DeviceSelector.getRuntime().malloc[Int](n)
      DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      p
    }
    inline def toHostArray(ptr: Ptr[Int], n: Int): Array[Int] = {
      val arr = new Array[Int](n)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr, n)
      arr
    }
    inline def createArray(n: Int): Array[Int] = Array.ofDim[Int](n)
  }

  /** Long type - 64-bit signed integer */
  type Long = scala.Long
  object Long {
    inline def apply(v: scala.Long): Long = v
    inline def unapply(l: Long): Some[scala.Long] = Some(l)
  }

  given compileTime[Long] with {
    inline def makeDefault(): Long = 0L
    inline def cudaTypeName: String = "long"
    inline def createArray(n: Int): Array[Long] = Array.ofDim[Long](n)
  }

  given MemoryOps[Long] with {
    val elementSize = 8
    inline def read(ptr: Ptr[Long], idx: Int): Long = {
      val arr = new Array[Long](1)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr + idx, 1)
      arr(0)
    }
    inline def write(ptr: Ptr[Long], idx: Int, value: Long): Unit = {
      val arr = Array(value)
      DeviceSelector.getRuntime().memcpyHtoD(ptr + idx, arr, 1)
    }
    inline def alloc(default: Long, n: Int): Ptr[Long] = {
      val p = DeviceSelector.getRuntime().malloc[Long](n)
      if (default != 0L) {
        val arr = Array.fill(n)(default)
        DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      }
      p
    }
    inline def free(ptr: Ptr[Long]): Unit = DeviceSelector.getRuntime().free(ptr)
    inline def offset(ptr: Ptr[Long], n: Int): Ptr[Long] = {
      Ptr.fromAddress[Long](ptr.rawAddress + n * elementSize)
    }
    inline def memset(ptr: Ptr[Long], value: Any, n: Int): Unit = {
      val arr = Array.fill(n)(value.asInstanceOf[Long])
      DeviceSelector.getRuntime().memcpyHtoD(ptr, arr, n)
    }
    inline def memcpy(dst: Ptr[Long], src: Ptr[Long], n: Int, kind: MemcpyKind): Unit = {
      kind match {
        case MemcpyKind.HostToDevice => sys.error("Use fromHostArray for H2D")
        case MemcpyKind.DeviceToHost => sys.error("Use toHostArray for D2H")
        case _ => DeviceSelector.getRuntime().memcpyDtoD(dst, src, n)
      }
    }
    inline def fromHostArray(arr: Array[Long], n: Int): Ptr[Long] = {
      val p = DeviceSelector.getRuntime().malloc[Long](n)
      DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      p
    }
    inline def toHostArray(ptr: Ptr[Long], n: Int): Array[Long] = {
      val arr = new Array[Long](n)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr, n)
      arr
    }
    inline def createArray(n: Int): Array[Long] = Array.ofDim[Long](n)
  }

  /** Half type - 16-bit floating point (FP16) */
  type Half = scala.Short
  object Half {
    inline def apply(v: scala.Float): Half = floatToHalfBits(v)
    inline def toFloat(h: Half): scala.Float = halfBitsToFloat(h)
  }

  given compileTime[Half] with {
    inline def makeDefault(): Half = 0
    inline def cudaTypeName: String = "half"
    inline def createArray(n: Int): Array[Half] = Array.ofDim[Half](n)
  }

  given MemoryOps[Half] with {
    val elementSize = 2
    inline def read(ptr: Ptr[Half], idx: Int): Half = {
      val arr = new Array[Short](1)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr + idx, 1)
      arr(0)
    }
    inline def write(ptr: Ptr[Half], idx: Int, value: Half): Unit = {
      val arr = Array(value)
      DeviceSelector.getRuntime().memcpyHtoD(ptr + idx, arr, 1)
    }
    inline def alloc(default: Half, n: Int): Ptr[Half] = {
      val p = DeviceSelector.getRuntime().malloc[Half](n)
      if (default != 0) {
        val arr = Array.fill(n)(default)
        DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      }
      p
    }
    inline def free(ptr: Ptr[Half]): Unit = DeviceSelector.getRuntime().free(ptr)
    inline def offset(ptr: Ptr[Half], n: Int): Ptr[Half] = {
      Ptr.fromAddress[Half](ptr.rawAddress + n * elementSize)
    }
    inline def memset(ptr: Ptr[Half], value: Any, n: Int): Unit = {
      val arr = Array.fill(n)(value.asInstanceOf[Half])
      DeviceSelector.getRuntime().memcpyHtoD(ptr, arr, n)
    }
    inline def memcpy(dst: Ptr[Half], src: Ptr[Half], n: Int, kind: MemcpyKind): Unit = {
      kind match {
        case MemcpyKind.HostToDevice => sys.error("Use fromHostArray for H2D")
        case MemcpyKind.DeviceToHost => sys.error("Use toHostArray for D2H")
        case _ => DeviceSelector.getRuntime().memcpyDtoD(dst, src, n)
      }
    }
    inline def fromHostArray(arr: Array[Half], n: Int): Ptr[Half] = {
      val p = DeviceSelector.getRuntime().malloc[Half](n)
      DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      p
    }
    inline def toHostArray(ptr: Ptr[Half], n: Int): Array[Half] = {
      val arr = new Array[Half](n)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr, n)
      arr
    }
    inline def createArray(n: Int): Array[Half] = Array.ofDim[Half](n)
  }

  /** BFloat16 type - 16-bit brain floating point (BF16) */
  /** Format: 1 sign bit, 8 exponent bits, 7 mantissa bits */
  /** Same exponent range as FP32, but lower precision */
  type BFloat16 = scala.Short
  object BFloat16 {
    inline def apply(v: scala.Float): BFloat16 = floatToBf16Bits(v)
    inline def toFloat(b: BFloat16): scala.Float = bf16BitsToFloat(b)
  }

  /** Byte type - 8-bit signed integer */
  type Byte = scala.Byte
  object Byte {
    inline def apply(v: scala.Byte): Byte = v
    inline def unapply(b: Byte): Some[scala.Byte] = Some(b)
  }

  given compileTime[Byte] with {
    inline def makeDefault(): Byte = 0
    inline def cudaTypeName: String = "int8_t"
    inline def createArray(n: Int): Array[Byte] = Array.ofDim[Byte](n)
  }

  given MemoryOps[Byte] with {
    val elementSize = 1
    inline def read(ptr: Ptr[Byte], idx: Int): Byte = {
      val arr = new Array[Byte](1)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr + idx, 1)
      arr(0)
    }
    inline def write(ptr: Ptr[Byte], idx: Int, value: Byte): Unit = {
      val arr = Array(value)
      DeviceSelector.getRuntime().memcpyHtoD(ptr + idx, arr, 1)
    }
    inline def alloc(default: Byte, n: Int): Ptr[Byte] = {
      val p = DeviceSelector.getRuntime().malloc[Byte](n)
      if (default != 0) {
        val arr = Array.fill(n)(default)
        DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      }
      p
    }
    inline def free(ptr: Ptr[Byte]): Unit = DeviceSelector.getRuntime().free(ptr)
    inline def offset(ptr: Ptr[Byte], n: Int): Ptr[Byte] = {
      Ptr.fromAddress[Byte](ptr.rawAddress + n * elementSize)
    }
    inline def memset(ptr: Ptr[Byte], value: Any, n: Int): Unit = {
      val arr = Array.fill(n)(value.asInstanceOf[Byte])
      DeviceSelector.getRuntime().memcpyHtoD(ptr, arr, n)
    }
    inline def memcpy(dst: Ptr[Byte], src: Ptr[Byte], n: Int, kind: MemcpyKind): Unit = {
      kind match {
        case MemcpyKind.HostToDevice => sys.error("Use fromHostArray for H2D")
        case MemcpyKind.DeviceToHost => sys.error("Use toHostArray for D2H")
        case _ => DeviceSelector.getRuntime().memcpyDtoD(dst, src, n)
      }
    }
    inline def fromHostArray(arr: Array[Byte], n: Int): Ptr[Byte] = {
      val p = DeviceSelector.getRuntime().malloc[Byte](n)
      DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      p
    }
    inline def toHostArray(ptr: Ptr[Byte], n: Int): Array[Byte] = {
      val arr = new Array[Byte](n)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr, n)
      arr
    }
    inline def createArray(n: Int): Array[Byte] = Array.ofDim[Byte](n)
  }

  /** Boolean type */
  type Bool = scala.Boolean
  object Bool {
    inline def apply(v: scala.Boolean): Bool = v
    inline def unapply(b: Bool): Some[scala.Boolean] = Some(b)
  }

  given compileTime[Bool] with {
    inline def makeDefault(): Bool = false
    inline def cudaTypeName: String = "bool"
    inline def createArray(n: Int): Array[Bool] = Array.ofDim[Bool](n)
  }

  given MemoryOps[Bool] with {
    val elementSize = 1
    inline def read(ptr: Ptr[Bool], idx: Int): Bool = {
      val arr = new Array[scala.Boolean](1)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr + idx, 1)
      arr(0)
    }
    inline def write(ptr: Ptr[Bool], idx: Int, value: Bool): Unit = {
      val arr = Array(value)
      DeviceSelector.getRuntime().memcpyHtoD(ptr + idx, arr, 1)
    }
    inline def alloc(default: Bool, n: Int): Ptr[Bool] = {
      val p = DeviceSelector.getRuntime().malloc[Bool](n)
      if (default != false) {
        val arr = Array.fill(n)(default)
        DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      }
      p
    }
    inline def free(ptr: Ptr[Bool]): Unit = DeviceSelector.getRuntime().free(ptr)
    inline def offset(ptr: Ptr[Bool], n: Int): Ptr[Bool] = {
      Ptr.fromAddress[Bool](ptr.rawAddress + n * elementSize)
    }
    inline def memset(ptr: Ptr[Bool], value: Any, n: Int): Unit = {
      val arr = Array.fill(n)(value.asInstanceOf[scala.Boolean])
      DeviceSelector.getRuntime().memcpyHtoD(ptr, arr, n)
    }
    inline def memcpy(dst: Ptr[Bool], src: Ptr[Bool], n: Int, kind: MemcpyKind): Unit = {
      kind match {
        case MemcpyKind.HostToDevice => sys.error("Use fromHostArray for H2D")
        case MemcpyKind.DeviceToHost => sys.error("Use toHostArray for D2H")
        case _ => DeviceSelector.getRuntime().memcpyDtoD(dst, src, n)
      }
    }
    inline def fromHostArray(arr: Array[scala.Boolean], n: Int): Ptr[Bool] = {
      val p = DeviceSelector.getRuntime().malloc[Bool](n)
      DeviceSelector.getRuntime().memcpyHtoD(p, arr, n)
      p
    }
    inline def toHostArray(ptr: Ptr[Bool], n: Int): Array[scala.Boolean] = {
      val arr = new Array[scala.Boolean](n)
      DeviceSelector.getRuntime().memcpyDtoH(arr, ptr, n)
      arr
    }
    inline def createArray(n: Int): Array[Bool] = Array.ofDim[Bool](n)
  }

  // =========================================================================
  // FP16/BF16 Conversion Helpers
  // =========================================================================

  /** Convert Float to FP16 bits (IEEE 754 half-precision) */
  private inline def floatToHalfBits(value: scala.Float): scala.Short = {
    val bits = java.lang.Float.floatToIntBits(value)
    val sign = (bits >> 16) & 0x8000
    val exp = (bits >> 23) & 0xff
    val mantissa = bits & 0x7fffff

    if (exp == 255) {
      if (mantissa != 0) (sign | 0x7fff).toShort
      else (sign | 0x7c00).toShort
    } else {
      val biasedExp = exp - 127 + 15
      if (biasedExp <= 0) {
        if (biasedExp < -10) sign.toShort
        else {
          val mant = mantissa | 0x800000
          val shift = 14 - (exp - 127 + 1)
          ((sign | ((mant >> shift) & 0x3ff))).toShort
        }
      } else if (biasedExp >= 31) {
        (sign | 0x7c00).toShort
      } else {
        (sign | (biasedExp << 10) | (mantissa >> 13)).toShort
      }
    }
  }

  /** Convert FP16 bits to Float */
  private inline def halfBitsToFloat(bits: scala.Short): scala.Float = {
    val sign = (bits >> 15) & 1
    val exp = (bits >> 10) & 0x1f
    val mantissa = bits & 0x3ff

    if (exp == 31) {
      if (mantissa != 0) scala.Float.NaN
      else if (sign != 0) scala.Float.NegativeInfinity else scala.Float.PositiveInfinity
    } else if (exp == 0) {
      if (mantissa == 0) {
        java.lang.Float.intBitsToFloat(sign << 31)
      } else {
        val mant = mantissa << 13
        var shift = -1
        var m = mant
        while ((m & 0x800000) == 0 && m != 0) {
          m <<= 1
          shift -= 1
        }
        val biasedExp = 127 - 15 + (shift + 1)
        java.lang.Float.intBitsToFloat((sign << 31) | (biasedExp << 23) | (m & 0x7fffff))
      }
    } else {
      java.lang.Float.intBitsToFloat((sign << 31) | ((exp - 15 + 127) << 23) | (mantissa << 13))
    }
  }

  /** Convert Float to BF16 bits (brain floating point) */
  private inline def floatToBf16Bits(value: scala.Float): scala.Short = {
    // BF16 is simply the top 16 bits of FP32
    val bits = java.lang.Float.floatToIntBits(value)
    ((bits >> 16) & 0xFFFF).toShort
  }

  /** Convert BF16 bits to Float */
  private inline def bf16BitsToFloat(bits: scala.Short): scala.Float = {
    // Expand BF16 to FP32 by shifting left 16 bits
    val bits32 = (bits & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(bits32)
  }
}
