package cuda.dsl.dataframe

import cuda.dsl.collections.GPUArray
import cuda.dsl.runtime.DeviceRuntime
import cuda.dsl.runtime.DeviceSelector

/** Backend trait for DataFrame operations.
 *  Implementations: CPU (Scala), CUDA (native kernels), MPS (PyTorch).
 */
trait DataFrameBackend {
  /** Backend name */
  def name: String

  /** Check if backend is available */
  def isAvailable: Boolean

  /** Get underlying DeviceRuntime if GPU-backed */
  def deviceRuntime: Option[DeviceRuntime]

  // =========================================================================
  // Column Operations
  // =========================================================================

  /** Filter column by predicate */
  def colFilter(col: GPUColumn, predicate: Any => Boolean): GPUColumn

  /** Map column values */
  def colMap(col: GPUColumn, f: Any => Any): GPUColumn

  /** Reduce column to single value */
  def colReduce(col: GPUColumn, op: (Any, Any) => Any): Any

  // =========================================================================
  // Selection Operations
  // =========================================================================

  /** Select columns */
  def select(df: GPUDataFrame, cols: Seq[String]): GPUDataFrame

  /** Drop columns */
  def drop(df: GPUDataFrame, cols: Seq[String]): GPUDataFrame

  // =========================================================================
  // Filtering Operations
  // =========================================================================

  /** Filter rows by condition */
  def filter(df: GPUDataFrame, condition: GPUColumn): GPUDataFrame

  /** Limit rows */
  def limit(df: GPUDataFrame, n: Int): GPUDataFrame

  // =========================================================================
  // Aggregation Operations
  // =========================================================================

  /** Group by and aggregate */
  def groupByAgg(
    df: GPUDataFrame,
    groupCols: Seq[String],
    aggs: Seq[Aggregation]
  ): GPUDataFrame

  // =========================================================================
  // Join Operations
  // =========================================================================

  /** Hash join */
  def hashJoin(
    left: GPUDataFrame,
    right: GPUDataFrame,
    on: String,
    how: JoinType
  ): GPUDataFrame

  // =========================================================================
  // Set Operations
  // =========================================================================

  /** Union two DataFrames */
  def union(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame

  /** Intersect */
  def intersect(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame

  /** Except */
  def except(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame
}

/** CPU fallback implementation */
class CPUDataFrameBackend extends DataFrameBackend {
  def name: String = "CPU"
  def isAvailable: Boolean = true
  def deviceRuntime: Option[DeviceRuntime] = None

  def colFilter(col: GPUColumn, predicate: Any => Boolean): GPUColumn = {
    val result = new Array[Boolean](col.length)
    var count = 0
    for (i <- 0 until col.length) {
      val v = col(i)
      val keep = try predicate(v) catch { case _: Exception => false }
      result(i) = keep
      if (keep) count += 1
    }

    // Create filtered column
    col.dtype match {
      case DataType.Float32 =>
        val oldData = col.data.asInstanceOf[GPUArray[Float]]
        val newData = GPUArray.ofLength[Float](count)
        var j = 0
        for (i <- 0 until col.length) {
          if (result(i)) {
            newData.hostData(j) = oldData.hostData(i)
            j += 1
          }
        }
        new GPUColumn(col.name, newData, col.dtype)
      case _ =>
        throw new UnsupportedOperationException(s"Filter not supported for ${col.dtype}")
    }
  }

  def colMap(col: GPUColumn, f: Any => Any): GPUColumn = {
    col.dtype match {
      case DataType.Float32 =>
        val oldData = col.data.asInstanceOf[GPUArray[Float]]
        val newData = GPUArray.ofLength[Float](col.length)
        for (i <- 0 until col.length) {
          val v = oldData.hostData(i)
          newData.hostData(i) = f(v) match {
            case x: Float => x
            case x: Int => x.toFloat
            case x: Double => x.toFloat
            case _ => v
          }
        }
        new GPUColumn(col.name + "_mapped", newData, col.dtype)
      case _ =>
        throw new UnsupportedOperationException(s"Map not supported for ${col.dtype}")
    }
  }

  def colReduce(col: GPUColumn, op: (Any, Any) => Any): Any = {
    if (col.length == 0) return null
    var result = col(0)
    for (i <- 1 until col.length) {
      result = op(result, col(i))
    }
    result
  }

  def select(df: GPUDataFrame, cols: Seq[String]): GPUDataFrame = df.select(cols*)
  def drop(df: GPUDataFrame, cols: Seq[String]): GPUDataFrame = df.drop(cols*)
  def filter(df: GPUDataFrame, condition: GPUColumn): GPUDataFrame = df.filter(condition)
  def limit(df: GPUDataFrame, n: Int): GPUDataFrame = df.limit(n)

  def groupByAgg(df: GPUDataFrame, groupCols: Seq[String], aggs: Seq[Aggregation]): GPUDataFrame = {
    df.groupBy(groupCols*).agg(aggs*)
  }

  def hashJoin(left: GPUDataFrame, right: GPUDataFrame, on: String, how: JoinType): GPUDataFrame = {
    left.join(right, on, how)
  }

  def union(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame = left.union(right)
  def intersect(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame = left
  def except(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame = left
}

/** Get default backend based on DeviceSelector */
object DataFrameBackend {
  def getDefault(): DataFrameBackend = {
    val runtime = DeviceSelector.getRuntime()
    runtime.backendName match {
      case "CUDA" => new CUDADataFrameBackend
      case "MPS" => new MPSDataFrameBackend
      case _ => new CPUDataFrameBackend
    }
  }
}
