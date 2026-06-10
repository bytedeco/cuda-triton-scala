package cuda.dsl.dataframe

/** Represents a single row in a DataFrame.
 *  Provides row-based access to columnar data.
 */
class Row(private val df: GPUDataFrame, rowIndex: Int) {
  require(rowIndex >= 0 && rowIndex < df.numRows, s"Row index $rowIndex out of bounds")

  /** Get value by column name */
  def apply(col: String): Any = df(col)(rowIndex)

  /** Get value by column index */
  def apply(colIndex: Int): Any = {
    val colName = df.columns(colIndex)
    df(colName)(rowIndex)
  }

  /** Get value by column name (alternative) */
  def get(col: String): Any = apply(col)

  /** Get string value */
  def getString(col: String): String = apply(col) match {
    case s: String => s
    case null => null
    case other => other.toString
  }

  /** Get int value */
  def getInt(col: String): Int = apply(col) match {
    case i: Int => i
    case f: Float => f.toInt
    case d: Double => d.toInt
    case l: Long => l.toInt
    case null => 0
    case other => other.toString.toInt
  }

  /** Get long value */
  def getLong(col: String): Long = apply(col) match {
    case l: Long => l
    case i: Int => i.toLong
    case f: Float => f.toLong
    case d: Double => d.toLong
    case null => 0L
    case other => other.toString.toLong
  }

  /** Get float value */
  def getFloat(col: String): Float = apply(col) match {
    case f: Float => f
    case i: Int => i.toFloat
    case d: Double => d.toFloat
    case l: Long => l.toFloat
    case null => 0f
    case other => other.toString.toFloat
  }

  /** Get double value */
  def getDouble(col: String): Double = apply(col) match {
    case d: Double => d
    case f: Float => f.toDouble
    case i: Int => i.toDouble
    case l: Long => l.toDouble
    case null => 0.0
    case other => other.toString.toDouble
  }

  /** Get boolean value */
  def getBoolean(col: String): Boolean = apply(col) match {
    case b: Boolean => b
    case i: Int => i != 0
    case s: String => s.toBoolean
    case null => false
    case other => other.toString.toBoolean
  }

  /** Check if value at column is null */
  def isNullAt(col: String): Boolean = df(col).isNullAt(rowIndex)

  /** Convert row to Map */
  def toMap: Map[String, Any] = {
    df.columnData.map { case (name, _) => (name, apply(name)) }.toMap
  }

  /** Convert row to Seq */
  def toSeq: Seq[Any] = {
    df.columnData.map { case (name, _) => apply(name) }.toSeq
  }

  /** Convert row to Array */
  def toArray: Array[Any] = toSeq.toArray

  /** Number of columns */
  def size: Int = df.numCols

  override def toString: String = {
    s"Row(${df.columns.map(c => s"$c=${apply(c)}").mkString(", ")})"
  }

  override def equals(other: Any): Boolean = other match {
    case r: Row =>
      if (df.numCols != r.df.numCols) return false
      val thisMap = toMap
      val otherMap = r.toMap
      thisMap == otherMap
    case m: Map[_, _] =>
      toMap == m
    case _ => false
  }

  override def hashCode: Int = toMap.hashCode
}

/** Factory and utility methods for Row */
object Row {
  /** Create Row from a Map */
  def fromMap(map: Map[String, Any]): Row = {
    throw new UnsupportedOperationException("Cannot create Row directly from Map - use GPUDataFrame")
  }

  /** Create Row from index in DataFrame */
  private[dataframe] def fromIndex(df: GPUDataFrame, index: Int): Row = {
    new Row(df, index)
  }

  /** Concatenate rows */
  def concat(rows: Seq[Row]): Seq[Row] = rows
}
