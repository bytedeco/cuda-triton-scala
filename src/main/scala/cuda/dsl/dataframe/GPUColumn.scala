package cuda.dsl.dataframe

import cuda.dsl.collections.GPUArray
import cuda.dsl.core.Types.given_MemoryOps_Float
import cuda.dsl.runtime.DeviceSelector

/** A single column in a GPU DataFrame.
 *  Stores data in a GPUArray with associated type information.
 */
class GPUColumn(
  val name: String,
  private[dataframe] val data: GPUArray[_],
  val dtype: DataType
) {
  // Require name and non-null data
  require(name != null && name.nonEmpty, "Column name cannot be empty")
  require(data != null, "Column data cannot be null")

  /** Number of elements in this column */
  def length: Int = data.length

  /** Check if column is empty */
  def isEmpty: Boolean = length == 0

  /** Get element at index - materializes if needed */
  def apply(i: Int): Any = {
    require(i >= 0 && i < length, s"Index $i out of bounds [0, $length)")
    dtype match
      case DataType.Float32 => data.asInstanceOf[GPUArray[Float]].hostData(i)
      case DataType.Int32 => data.asInstanceOf[GPUArray[Int]].hostData(i)
      case DataType.Int64 => data.asInstanceOf[GPUArray[Long]].hostData(i)
      case DataType.Float64 => data.asInstanceOf[GPUArray[Double]].hostData(i)
      case DataType.Boolean => data.asInstanceOf[GPUArray[Boolean]].hostData(i)
      case DataType.String => data.asInstanceOf[GPUArray[String]].hostData(i)
      case _ => throw new UnsupportedOperationException(s"Unsupported dtype: $dtype")
  }

  /** Get all values as Scala sequence */
  def toSeq: Seq[Any] = {
    dtype match
      case DataType.Float32 => data.asInstanceOf[GPUArray[Float]].hostData.toSeq
      case DataType.Int32 => data.asInstanceOf[GPUArray[Int]].hostData.toSeq
      case DataType.Int64 => data.asInstanceOf[GPUArray[Long]].hostData.toSeq
      case DataType.Float64 => data.asInstanceOf[GPUArray[Double]].hostData.toSeq
      case DataType.Boolean => data.asInstanceOf[GPUArray[Boolean]].hostData.toSeq
      case DataType.String => data.asInstanceOf[GPUArray[String]].hostData.toSeq
      case _ => throw new UnsupportedOperationException(s"Unsupported dtype: $dtype")
  }

  /** Check if element at index is null */
  def isNullAt(i: Int): Boolean = apply(i) == null

  /** Create a null-mask column */
  def isNull: GPUColumn = {
    val nulls = GPUArray.ofLength[Boolean](length)
    val nullsHost = nulls.hostData
    for (i <- 0 until length) {
      nullsHost(i) = isNullAt(i)
    }
    new GPUColumn(name + "_is_null", nulls, DataType.Boolean)
  }

  /** Fill null values with specified value */
  def fillNull(value: Any): GPUColumn = {
    val filled = GPUArray.ofLength[Float](length)
    val filledHost = filled.hostData
    for (i <- 0 until length) {
      if (isNullAt(i)) {
        value match
          case v: Float => filledHost(i) = v
          case v: Int => filledHost(i) = v.toFloat
          case _ => filledHost(i) = 0f
      } else {
        filledHost(i) = apply(i) match
          case v: Float => v
          case v: Int => v.toFloat
          case _ => 0f
      }
    }
    new GPUColumn(name, filled, dtype)
  }

  /** Drop null values */
  def dropNull: (GPUColumn, GPUColumn) = {
    // Returns (non-null data, null mask)
    val notNull = GPUArray.ofLength[Boolean](length)
    val notNullHost = notNull.hostData
    var nullCount = 0
    for (i <- 0 until length) {
      val notNullVal = !isNullAt(i)
      notNullHost(i) = notNullVal
      if (!notNullVal) nullCount += 1
    }

    // Count non-null for new column
    val resultData = GPUArray.ofLength[Float](length - nullCount)
    val resultHost = resultData.hostData
    var j = 0
    for (i <- 0 until length) {
      if (notNullHost(i)) {
        apply(i) match
          case v: Float => resultHost(j) = v
          case v: Int => resultHost(j) = v.toFloat
          case _ =>
        j += 1
      }
    }

    val nullMask = new GPUColumn(name + "_is_null", notNull, DataType.Boolean)
    (new GPUColumn(name, resultData, dtype), nullMask)
  }

  /** Cast to different type */
  def cast(newType: DataType): GPUColumn = {
    if (dtype == newType) return this

    (dtype, newType) match
      case (DataType.Int32, DataType.Float32) =>
        val newData = GPUArray.ofLength[Float](length)
        val newHost = newData.hostData
        for (i <- 0 until length) {
          newHost(i) = apply(i) match
            case v: Int => v.toFloat
            case v: Float => v
            case _ => 0f
        }
        new GPUColumn(name, newData, newType)

      case (DataType.Float32, DataType.Int32) =>
        val newData = GPUArray.ofLength[Int](length)
        val newHost = newData.hostData
        for (i <- 0 until length) {
          newHost(i) = apply(i) match
            case v: Float => v.toInt
            case v: Int => v
            case _ => 0
        }
        new GPUColumn(name, newData, newType)

      case _ =>
        throw new UnsupportedOperationException(s"Cast from $dtype to $newType not supported")
  }

  /** Get basic statistics */
  def stats(): ColumnStats = {
    dtype match
      case DataType.Float32 | DataType.Int32 | DataType.Int64 =>
        var minVal = Double.MaxValue
        var maxVal = Double.MinValue
        var sum = 0.0
        var count = 0L

        for (i <- 0 until length) {
          if (!isNullAt(i)) {
            val v = apply(i) match
              case v: Float => v.toDouble
              case v: Int => v.toDouble
              case v: Long => v.toDouble
              case _ => Double.NaN
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
            sum += v
            count += 1
          }
        }

        val mean = if (count > 0) sum / count else Double.NaN

        ColumnStats(
          count = count,
          nullCount = length - count,
          min = minVal,
          max = maxVal,
          mean = mean,
          sum = sum
        )

      case _ =>
        ColumnStats(length, 0, Double.NaN, Double.NaN, Double.NaN, 0.0)
  }

  override def toString: String = s"GPUColumn($name, $dtype, $length elements)"
}

/** Statistics for a numeric column */
case class ColumnStats(
  count: Long,
  nullCount: Long,
  min: Double,
  max: Double,
  mean: Double,
  sum: Double
)

/** Factory methods for creating columns */
object GPUColumn {
  /** Create column from Scala sequence */
  def apply(name: String, values: Seq[Any], dtype: DataType): GPUColumn = {
    val length = values.length
    dtype match
      case DataType.Float32 =>
        val data = GPUArray.ofLength[Float](length)
        val host = data.hostData
        for (i <- 0 until length) {
          values(i) match
            case v: Float => host(i) = v
            case v: Int => host(i) = v.toFloat
            case v: Double => host(i) = v.toFloat
            case v: Long => host(i) = v.toFloat
            case null | _ => host(i) = 0f
        }
        new GPUColumn(name, data, dtype)

      case DataType.Int32 =>
        val data = GPUArray.ofLength[Int](length)
        val host = data.hostData
        for (i <- 0 until length) {
          values(i) match
            case v: Int => host(i) = v
            case v: Float => host(i) = v.toInt
            case v: Double => host(i) = v.toInt
            case null | _ => host(i) = 0
        }
        new GPUColumn(name, data, dtype)

      case DataType.String =>
        val data = GPUArray.ofLength[String](length)
        val host = data.hostData
        for (i <- 0 until length) {
          host(i) = values(i) match
            case v: String => v
            case null => ""
            case _ => values(i).toString
        }
        new GPUColumn(name, data, dtype)

      case _ =>
        throw new UnsupportedOperationException(s"Unsupported dtype: $dtype")
  }

  /** Create column from array */
  def apply(name: String, data: Array[Float], dtype: DataType = DataType.Float32): GPUColumn = {
    val gpuArray = GPUArray.fromArray[Float](data)
    new GPUColumn(name, gpuArray, dtype)
  }

  /** Create integer column */
  def int(name: String, data: Array[Int]): GPUColumn = {
    val gpuArray = GPUArray.fromArray[Int](data)
    new GPUColumn(name, gpuArray, DataType.Int32)
  }

  /** Create float column */
  def float(name: String, data: Array[Float]): GPUColumn = {
    val gpuArray = GPUArray.fromArray[Float](data)
    new GPUColumn(name, gpuArray, DataType.Float32)
  }

  /** Create string column */
  def string(name: String, data: Array[String]): GPUColumn = {
    val gpuArray = GPUArray.fromArray[String](data)
    new GPUColumn(name, gpuArray, DataType.String)
  }
}
