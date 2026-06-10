package cuda.dsl.dataframe

/** Data types supported by GPU DataFrame */
enum DataType:
  // Integer types
  case Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64

  // Float types
  case Float32, Float64

  // Other numeric
  case Boolean

  // String types
  case String, Binary

  // DateTime types
  case Date, Timestamp, Timedelta

  // Complex types
  case List, Map, Struct

  /** Get the element size in bytes for primitive types */
  def elementSize: Int = this match
    case Int8 | UInt8 => 1
    case Int16 | UInt16 => 2
    case Int32 | UInt32 | Float32 => 4
    case Int64 | UInt64 | Float64 | Timestamp | Timedelta => 8
    case Boolean => 1
    case Date => 4 // days since epoch
    case String | Binary | List | Map | Struct => -1 // variable size
    case _ => -1

  /** Check if this is a fixed-size primitive type */
  def isPrimitive: Boolean = elementSize > 0

  /** Convert to cuDF/Pandas dtype string */
  def toPandas: String = this match
    case Int8 => "int8"
    case Int16 => "int16"
    case Int32 => "int32"
    case Int64 => "int64"
    case UInt8 => "uint8"
    case UInt16 => "uint16"
    case UInt32 => "uint32"
    case UInt64 => "uint64"
    case Float32 => "float32"
    case Float64 => "float64"
    case Boolean => "bool"
    case String => "object"
    case Binary => "bytes"
    case Date => "datetime64[D]"
    case Timestamp => "datetime64[ns]"
    case Timedelta => "timedelta64[ns]"
    case _ => "object"

object DataType:
  /** Get DataType from Java Class */
  def fromClass(cls: Class[_]): DataType = cls match
    case c if c == classOf[Byte] => Int8
    case c if c == classOf[Short] => Int16
    case c if c == classOf[Int] => Int32
    case c if c == classOf[Long] => Int64
    case c if c == classOf[Float] => Float32
    case c if c == classOf[Double] => Float64
    case c if c == classOf[Boolean] => Boolean
    case c if c == classOf[String] => String
    case _ => String

  /** Get default value for type */
  def defaultValue(dtype: DataType): Any = dtype match
    case Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 => 0L
    case Float32 => 0.0f
    case Float64 => 0.0
    case Boolean => false
    case String => ""
    case _ => null
