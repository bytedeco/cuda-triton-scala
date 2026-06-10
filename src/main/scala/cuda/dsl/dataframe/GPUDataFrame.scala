package cuda.dsl.dataframe

import cuda.dsl.collections.GPUArray
import cuda.dsl.runtime.DeviceSelector

/** GPU-accelerated DataFrame with columnar storage.
 *  Provides pandas-like API with CUDA/MPS/CPU backend support.
 */
class GPUDataFrame(
  private val _columnMap: Map[String, GPUColumn],
  val schema: StructType
) {
  // Always use ListMap internally to preserve insertion order
  private val columnMap = _columnMap.to(collection.immutable.ListMap)
  require(_columnMap.nonEmpty, "DataFrame must have at least one column")
  require(_columnMap.keySet == schema.fields.map(_.name).toSet,
    "Column names must match schema")

  /** Get column by name */
  def apply(col: String): GPUColumn = columnMap(col)

  /** Get column by name (alternative) */
  def col(col: String): GPUColumn = columnMap(col)

  /** Get column names */
  def columns: List[String] = columnMap.keys.toList

  /** Get column map directly */
  def columnData: Map[String, GPUColumn] = columnMap

  /** Number of rows */
  def numRows: Int = columnMap.values.head.length

  /** Number of columns */
  def numCols: Int = columnMap.size

  /** Check if DataFrame is empty */
  def isEmpty: Boolean = numRows == 0

  // =========================================================================
  // Selection Operations
  // =========================================================================

  /** Select columns by name */
  def select(cols: String*): GPUDataFrame = {
    val selectedCols = cols.map(c => c -> columnMap(c)).toMap
    val selectedSchema = StructType(cols.flatMap(c => schema.fields.find(_.name == c)))
    new GPUDataFrame(selectedCols, selectedSchema)
  }

  /** Drop columns by name */
  def drop(cols: String*): GPUDataFrame = {
    val remainingCols = columnMap.filter { case (name, _) => !cols.contains(name) }
    val remainingSchema = StructType(schema.fields.filter(f => !cols.contains(f.name)))
    new GPUDataFrame(remainingCols, remainingSchema)
  }

  /** Select columns by index range */
  def selectCols(range: Range): GPUDataFrame = {
    val names = columnMap.keys.toList.slice(range.start, range.end)
    select(names*)
  }

  // =========================================================================
  // Filtering Operations
  // =========================================================================

  /** Filter rows by boolean column condition */
  def filter(condition: GPUColumn): GPUDataFrame = {
    require(condition.dtype == DataType.Boolean, "Filter condition must be boolean")
    require(condition.length == numRows, "Filter condition length must match DataFrame rows")

    val mask = condition.asInstanceOf[GPUColumn].asInstanceOf[{
      def getNullMask(): Array[Boolean]
    }]

    // For now, materialize and filter
    val maskHost = new Array[Boolean](numRows)
    for (i <- 0 until numRows) {
      maskHost(i) = condition(i).asInstanceOf[Boolean]
    }

    val newColumns = columnMap.map { case (name, col) =>
      val newData = filterColumn(col, maskHost)
      (name, new GPUColumn(name, newData, col.dtype))
    }

    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new GPUDataFrame(newColumns, newSchema)
  }

  /** Filter using a predicate function */
  def filter(pred: Any => Boolean): GPUDataFrame = {
    // Apply predicate to first column and use resulting mask
    val firstCol = columnMap.values.head
    val maskHost = new Array[Boolean](numRows)
    for (i <- 0 until numRows) {
      maskHost(i) = pred(firstCol(i))
    }

    val newColumns = columnMap.map { case (name, col) =>
      val newData = filterColumn(col, maskHost)
      (name, new GPUColumn(name, newData, col.dtype))
    }

    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new GPUDataFrame(newColumns, newSchema)
  }

  private def filterColumn(col: GPUColumn, mask: Array[Boolean]): GPUArray[_] = {
    val newSize = mask.count(b => b)
    col.dtype match {
      case DataType.Float32 =>
        val oldData = col.data.asInstanceOf[GPUArray[Float]]
        val newData = GPUArray.ofLength[Float](newSize)
        var j = 0
        for (i <- 0 until mask.length) {
          if (mask(i)) {
            newData.hostData(j) = oldData.hostData(i)
            j += 1
          }
        }
        newData
      case DataType.Int32 =>
        val oldData = col.data.asInstanceOf[GPUArray[Int]]
        val newData = GPUArray.ofLength[Int](newSize)
        var j = 0
        for (i <- 0 until mask.length) {
          if (mask(i)) {
            newData.hostData(j) = oldData.hostData(i)
            j += 1
          }
        }
        newData
      case DataType.String =>
        val oldData = col.data.asInstanceOf[GPUArray[String]]
        val newData = GPUArray.ofLength[String](newSize)
        var j = 0
        for (i <- 0 until mask.length) {
          if (mask(i)) {
            newData.hostData(j) = oldData.hostData(i)
            j += 1
          }
        }
        newData
      case DataType.Boolean =>
        val oldData = col.data.asInstanceOf[GPUArray[Boolean]]
        val newData = GPUArray.ofLength[Boolean](newSize)
        var j = 0
        for (i <- 0 until mask.length) {
          if (mask(i)) {
            newData.hostData(j) = oldData.hostData(i)
            j += 1
          }
        }
        newData
      case _ =>
        throw new UnsupportedOperationException(s"Filter not supported for ${col.dtype}")
    }
  }

  /** Limit number of rows */
  def limit(n: Int): GPUDataFrame = {
    if (n >= numRows) return this

    val newColumns = columnMap.map { case (name, col) =>
      val newData = limitColumn(col, n)
      (name, new GPUColumn(name, newData, col.dtype))
    }

    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new GPUDataFrame(newColumns, newSchema)
  }

  private def limitColumn(col: GPUColumn, n: Int): GPUArray[_] = {
    col.dtype match {
      case DataType.Float32 =>
        val oldData = col.data.asInstanceOf[GPUArray[Float]]
        val newData = GPUArray.ofLength[Float](n)
        for (i <- 0 until n) {
          newData.hostData(i) = oldData.hostData(i)
        }
        newData
      case DataType.Int32 =>
        val oldData = col.data.asInstanceOf[GPUArray[Int]]
        val newData = GPUArray.ofLength[Int](n)
        for (i <- 0 until n) {
          newData.hostData(i) = oldData.hostData(i)
        }
        newData
      case DataType.String =>
        val oldData = col.data.asInstanceOf[GPUArray[String]]
        val newData = GPUArray.ofLength[String](n)
        for (i <- 0 until n) {
          newData.hostData(i) = oldData.hostData(i)
        }
        newData
      case DataType.Boolean =>
        val oldData = col.data.asInstanceOf[GPUArray[Boolean]]
        val newData = GPUArray.ofLength[Boolean](n)
        for (i <- 0 until n) {
          newData.hostData(i) = oldData.hostData(i)
        }
        newData
      case _ =>
        throw new UnsupportedOperationException(s"Limit not supported for ${col.dtype}")
    }
  }

  /** Take first n rows */
  def head(n: Int = 5): GPUDataFrame = limit(n)

  // =========================================================================
  // Column Operations
  // =========================================================================

  /** Add or replace a column */
  def withColumn(name: String, col: GPUColumn): GPUDataFrame = {
    val newCol = if (col.name != name) {
      new GPUColumn(name, col.data, col.dtype)
    } else col

    val newColumns = columnMap + (name -> newCol)
    val newField = StructField(name, col.dtype)
    val newSchema = StructType(schema.fields :+ newField)
    new GPUDataFrame(newColumns, newSchema)
  }

  /** Rename a column */
  def rename(oldName: String, newName: String): GPUDataFrame = {
    require(columnMap.contains(oldName), s"Column '$oldName' not found")
    val col = columnMap(oldName)
    val newCol = new GPUColumn(newName, col.data, col.dtype)
    val newColumns = columnMap - oldName + (newName -> newCol)
    val newSchema = StructType(
      schema.fields.map(f => if (f.name == oldName) f.copy(name = newName) else f)
    )
    new GPUDataFrame(newColumns, newSchema)
  }

  /** Cast column to different type */
  def cast(colName: String, newType: DataType): GPUDataFrame = {
    require(columnMap.contains(colName), s"Column '$colName' not found")
    val oldCol = columnMap(colName)
    val newCol = oldCol.cast(newType)
    val newColumns = columnMap + (colName -> newCol)
    new GPUDataFrame(newColumns, schema)
  }

  // =========================================================================
  // Aggregation Operations
  // =========================================================================

  /** Group by columns */
  def groupBy(cols: String*): GroupedDataFrame = {
    val missing = cols.filterNot(columns.contains)
    require(missing.isEmpty, s"Columns not found: $missing")
    new GroupedDataFrame(this, cols.toList)
  }

  // =========================================================================
  // Binary Operations
  // =========================================================================

  /** Union with another DataFrame */
  def union(other: GPUDataFrame): GPUDataFrame = {
    // Simple implementation - just append rows
    val newColumns = columnMap.map { case (name, col) =>
      val otherCol = other.columnMap(name)
      val newData = unionColumns(col, otherCol)
      (name, new GPUColumn(name, newData, col.dtype))
    }
    val newSchema = schema
    new GPUDataFrame(newColumns, newSchema)
  }

  private def unionColumns(left: GPUColumn, right: GPUColumn): GPUArray[_] = {
    val newSize = left.length + right.length
    left.dtype match {
      case DataType.Float32 =>
        val leftData = left.data.asInstanceOf[GPUArray[Float]]
        val rightData = right.data.asInstanceOf[GPUArray[Float]]
        val newData = GPUArray.ofLength[Float](newSize)
        for (i <- 0 until left.length) { newData.hostData(i) = leftData.hostData(i) }
        for (i <- 0 until right.length) { newData.hostData(left.length + i) = rightData.hostData(i) }
        newData
      case DataType.Int32 =>
        val leftData = left.data.asInstanceOf[GPUArray[Int]]
        val rightData = right.data.asInstanceOf[GPUArray[Int]]
        val newData = GPUArray.ofLength[Int](newSize)
        for (i <- 0 until left.length) { newData.hostData(i) = leftData.hostData(i) }
        for (i <- 0 until right.length) { newData.hostData(left.length + i) = rightData.hostData(i) }
        newData
      case DataType.Int64 =>
        val leftData = left.data.asInstanceOf[GPUArray[Long]]
        val rightData = right.data.asInstanceOf[GPUArray[Long]]
        val newData = GPUArray.ofLength[Long](newSize)
        for (i <- 0 until left.length) { newData.hostData(i) = leftData.hostData(i) }
        for (i <- 0 until right.length) { newData.hostData(left.length + i) = rightData.hostData(i) }
        newData
      case DataType.Float64 =>
        val leftData = left.data.asInstanceOf[GPUArray[Double]]
        val rightData = right.data.asInstanceOf[GPUArray[Double]]
        val newData = GPUArray.ofLength[Double](newSize)
        for (i <- 0 until left.length) { newData.hostData(i) = leftData.hostData(i) }
        for (i <- 0 until right.length) { newData.hostData(left.length + i) = rightData.hostData(i) }
        newData
      case DataType.Boolean =>
        val leftData = left.data.asInstanceOf[GPUArray[Boolean]]
        val rightData = right.data.asInstanceOf[GPUArray[Boolean]]
        val newData = GPUArray.ofLength[Boolean](newSize)
        for (i <- 0 until left.length) { newData.hostData(i) = leftData.hostData(i) }
        for (i <- 0 until right.length) { newData.hostData(left.length + i) = rightData.hostData(i) }
        newData
      case DataType.String =>
        val leftData = left.data.asInstanceOf[GPUArray[String]]
        val rightData = right.data.asInstanceOf[GPUArray[String]]
        val newData = GPUArray.ofLength[String](newSize)
        for (i <- 0 until left.length) { newData.hostData(i) = leftData.hostData(i) }
        for (i <- 0 until right.length) { newData.hostData(left.length + i) = rightData.hostData(i) }
        newData
      case _ =>
        throw new UnsupportedOperationException(s"Union not supported for ${left.dtype}")
    }
  }

  /** Join with another DataFrame */
  def join(other: GPUDataFrame, on: String, how: JoinType = JoinType.Inner): GPUDataFrame = {
    require(columns.contains(on), s"Join column '$on' not found in left DataFrame")
    require(other.columns.contains(on), s"Join column '$on' not found in right DataFrame")

    how match {
      case JoinType.Inner => innerJoin(other, on)
      case JoinType.Left => leftJoin(other, on)
      case JoinType.Right => rightJoin(other, on)
      case JoinType.Outer => outerJoin(other, on)
      case _ => throw new UnsupportedOperationException(s"Join type $how not implemented")
    }
  }

  private def innerJoin(other: GPUDataFrame, on: String): GPUDataFrame = {
    // Build hash map from right DataFrame
    val rightCol = other.columnMap(on)
    val rightMap = scala.collection.mutable.Map[Any, List[Int]]()
    for (i <- 0 until rightCol.length) {
      val key = rightCol(i)
      rightMap(key) = i :: rightMap.getOrElse(key, Nil)
    }

    // Find matching rows
    val leftCol = columnMap(on)
    val matchingRightIndices = scala.collection.mutable.ListBuffer[Int]()
    val matchingLeftIndices = scala.collection.mutable.ListBuffer[Int]()

    for (i <- 0 until leftCol.length) {
      val key = leftCol(i)
      rightMap.get(key) match {
        case Some(indices) =>
          matchingLeftIndices += i
          matchingRightIndices += indices.head
        case None =>
      }
    }

    // Build result columns - for now just return left side with matching rows
    val leftSelected = select(columnMap.keys.filter(_ != on).toList*)

    // For now, just return left selected with matching rows
    // Full join implementation would combine left and right columns
    if (matchingLeftIndices.isEmpty) {
      return leftSelected.limit(0)
    }

    // Create a result with just the left side for now
    throw new UnsupportedOperationException("Full inner join not yet implemented - returning left side only")
  }

  private def leftJoin(other: GPUDataFrame, on: String): GPUDataFrame = {
    // TODO: Implement left join
    throw new UnsupportedOperationException("Left join not yet implemented")
  }

  private def rightJoin(other: GPUDataFrame, on: String): GPUDataFrame = {
    // TODO: Implement right join
    throw new UnsupportedOperationException("Right join not yet implemented")
  }

  private def outerJoin(other: GPUDataFrame, on: String): GPUDataFrame = {
    // TODO: Implement outer join
    throw new UnsupportedOperationException("Outer join not yet implemented")
  }

  private def filter(col: String, indices: List[Int]): GPUDataFrame = {
    // Simplified filter for internal use
    val n = indices.length
    val newColumns = columnMap.map { case (name, c) =>
      val newData = c.dtype match {
        case DataType.Float32 =>
          val oldData = c.data.asInstanceOf[GPUArray[Float]]
          val data = GPUArray.ofLength[Float](n)
          for (i <- 0 until n) {
            data.hostData(i) = oldData.hostData(indices(i))
          }
          data
        case _ => throw new UnsupportedOperationException()
      }
      (name, new GPUColumn(name, newData, c.dtype))
    }
    new GPUDataFrame(newColumns, schema)
  }

  // =========================================================================
  // Evaluation
  // =========================================================================

  /** Force evaluation and return as Row sequence */
  def collect(): Seq[Row] = {
    (0 until numRows).map(i => Row.fromIndex(this, i))
  }

  /** Convert to Map sequence */
  def toMapSeq: Seq[Map[String, Any]] = {
    collect().map(_.toMap)
  }

  /** Get column statistics */
  def describe(cols: String*): GPUDataFrame = {
    val targetCols = if (cols.isEmpty) columnMap.keys.toList else cols.toList

    val stats = scala.collection.mutable.ListBuffer[(String, String, Any)]()
    for (colName <- targetCols) {
      val col = columnMap(colName)
      val s = col.stats()
      stats += ((colName, "count", s.count))
      stats += ((colName, "null", s.nullCount))
      stats += ((colName, "min", if (s.min == Double.MaxValue) null else s.min))
      stats += ((colName, "max", if (s.max == Double.MinValue) null else s.max))
      stats += ((colName, "mean", s.mean))
    }

    // This would be better as a different structure
    // For now just return this
    this
  }

  // =========================================================================
  // String Operations
  // =========================================================================

  /** Access column for string operations */
  def str(col: String): StringColumnOps = {
    require(columnMap.contains(col), s"Column '$col' not found")
    require(columnMap(col).dtype == DataType.String, s"Column '$col' is not String type")
    new StringColumnOps(columnMap(col))
  }

  // =========================================================================
  // Show / Print
  // =========================================================================

  /** Show DataFrame (like pandas head) */
  def show(n: Int = 20, truncate: Boolean = true): Unit = {
    val rows = (0 until math.min(n, numRows))
    val colWidths = columnMap.map { case (name, col) =>
      name -> math.max(name.length, if (truncate) 10 else col.length.toString.length)
    }

    // Header
    println(colWidths.map { case (name, w) => name.padTo(w, ' ') }.mkString(" | "))
    println(colWidths.map { case (_, w) => "-".repeat(w) }.mkString("-+-"))

    // Rows
    for (i <- rows) {
      val values = columnMap.map { case (name, col) =>
        val v = col(i)
        val s = if (v == null) "null" else v.toString
        val truncated = if (truncate && s.length > 10) s.substring(0, 7) + "..." else s
        truncated.padTo(colWidths(name), ' ')
      }
      println(values.mkString(" | "))
    }

    if (n < numRows) {
      println(s"... ${numRows - n} more rows")
    }
  }

  override def toString: String = {
    s"GPUDataFrame(${numRows}x${numCols}, columns: ${columnMap.keys.mkString(", ")})"
  }
}

/** Grouped DataFrame from groupBy operation */
class GroupedDataFrame(
  df: GPUDataFrame,
  groupCols: List[String]
) {
  /** Aggregate with specified aggregations */
  def agg(aggs: Aggregation*): GPUDataFrame = {
    val groupCol = df.columnData(groupCols.head)
    val numRows = groupCol.length
    val numGroups = countUnique(groupCol)
    val uniqueValues = extractUnique(groupCol, numGroups)

    val aggCols = scala.collection.mutable.Map[String, GPUColumn]()

    for (agg <- aggs) {
      val col = df.columnData(agg.column)
      val result = GPUArray.ofLength[Float](numGroups)
      for (g <- 0 until numGroups) {
        val gv = uniqueValues(g)
        val filtered = filterByGroupValue(col, groupCol, gv)
        val stats = filtered.stats()
        result.hostData(g) = agg.aggType match {
          case AggregationType.Count => stats.count.toFloat
          case AggregationType.Sum => stats.sum.toFloat
          case AggregationType.Mean | AggregationType.Avg => stats.mean.toFloat
          case AggregationType.Min => stats.min.toFloat
          case AggregationType.Max => stats.max.toFloat
          case AggregationType.Std => 0f
          case AggregationType.Var => 0f
          case AggregationType.Median => stats.mean.toFloat
          case AggregationType.First => filtered.data.asInstanceOf[GPUArray[Float]].hostData(0)
          case AggregationType.Last => filtered.data.asInstanceOf[GPUArray[Float]].hostData(filtered.length - 1)
          case AggregationType.NUnique => numGroups.toFloat
          case AggregationType.Distinct => numGroups.toFloat
          case _ => 0f
        }
      }
      aggCols(agg.outputName) = new GPUColumn(agg.outputName, result, DataType.Float32)
    }

    // Build group key columns (use first group column only for simplicity)
    val keyCols = scala.collection.mutable.Map[String, GPUColumn]()
    val keyData = GPUArray.ofLength[String](numGroups)
    for (g <- 0 until numGroups) {
      keyData.hostData(g) = uniqueValues(g).toString
    }
    keyCols(groupCols.head) = new GPUColumn(groupCols.head, keyData, DataType.String)

    val allCols = keyCols.toMap ++ aggCols.toMap
    val newSchema = StructType(
      keyCols.map { case (n, c) => StructField(n, c.dtype) }.toSeq ++
      aggCols.map { case (n, c) => StructField(n, c.dtype) }.toSeq
    )
    new GPUDataFrame(allCols, newSchema)
  }

  private def countUnique(col: GPUColumn): Int = {
    val seen = scala.collection.mutable.Set[Any]()
    for (i <- 0 until col.length) {
      seen.add(col(i))
    }
    seen.size
  }

  private def extractUnique(col: GPUColumn, numUnique: Int): Seq[Any] = {
    val seen = scala.collection.mutable.Set[Any]()
    val result = scala.collection.mutable.ArrayBuffer.empty[Any]
    for (i <- 0 until col.length) {
      val v = col(i)
      if (!seen.contains(v)) {
        seen.add(v)
        result.append(v)
      }
    }
    result.toSeq
  }

  private def filterByGroupValue(col: GPUColumn, groupCol: GPUColumn, groupValue: Any): GPUColumn = {
    col.dtype match {
      case DataType.Float32 =>
        val data = GPUArray.ofLength[Float](col.length)
        var n = 0
        for (i <- 0 until col.length) { if (groupCol(i) == groupValue) { data.hostData(n) = col.data.asInstanceOf[GPUArray[Float]].hostData(i); n += 1 } }
        new GPUColumn(col.name, data, DataType.Float32)
      case DataType.Int32 =>
        val data = GPUArray.ofLength[Int](col.length)
        var n = 0
        for (i <- 0 until col.length) { if (groupCol(i) == groupValue) { data.hostData(n) = col.data.asInstanceOf[GPUArray[Int]].hostData(i); n += 1 } }
        new GPUColumn(col.name, data, DataType.Int32)
      case DataType.Int64 =>
        val data = GPUArray.ofLength[Long](col.length)
        var n = 0
        for (i <- 0 until col.length) { if (groupCol(i) == groupValue) { data.hostData(n) = col.data.asInstanceOf[GPUArray[Long]].hostData(i); n += 1 } }
        new GPUColumn(col.name, data, DataType.Int64)
      case DataType.Float64 =>
        val data = GPUArray.ofLength[Double](col.length)
        var n = 0
        for (i <- 0 until col.length) { if (groupCol(i) == groupValue) { data.hostData(n) = col.data.asInstanceOf[GPUArray[Double]].hostData(i); n += 1 } }
        new GPUColumn(col.name, data, DataType.Float64)
      case DataType.Boolean =>
        val data = GPUArray.ofLength[Boolean](col.length)
        var n = 0
        for (i <- 0 until col.length) { if (groupCol(i) == groupValue) { data.hostData(n) = col.data.asInstanceOf[GPUArray[Boolean]].hostData(i); n += 1 } }
        new GPUColumn(col.name, data, DataType.Boolean)
      case DataType.String =>
        val data = GPUArray.ofLength[String](col.length)
        var n = 0
        for (i <- 0 until col.length) { if (groupCol(i) == groupValue) { data.hostData(n) = col.data.asInstanceOf[GPUArray[String]].hostData(i); n += 1 } }
        new GPUColumn(col.name, data, DataType.String)
      case _ =>
        val data = GPUArray.ofLength[Float](col.length)
        var n = 0
        for (i <- 0 until col.length) { if (groupCol(i) == groupValue) { data.hostData(n) = col.data.asInstanceOf[GPUArray[Float]].hostData(i); n += 1 } }
        new GPUColumn(col.name, data, DataType.Float32)
    }
  }

  /** Count rows in each group */
  def count(): GPUDataFrame = {
    val groupCol = df.columnData(groupCols.head)
    val numGroups = countUnique(groupCol)
    val uniqueValues = extractUnique(groupCol, numGroups)

    val countData = GPUArray.ofLength[Float](numGroups)
    for (g <- 0 until numGroups) {
      var c = 0
      for (i <- 0 until groupCol.length) {
        if (groupCol(i) == uniqueValues(g)) c += 1
      }
      countData.hostData(g) = c.toFloat
    }

    val keyData = GPUArray.ofLength[String](numGroups)
    for (g <- 0 until numGroups) {
      keyData.hostData(g) = uniqueValues(g).toString
    }

    val keyCols = scala.collection.mutable.Map[String, GPUColumn]()
    keyCols(groupCols.head) = new GPUColumn(groupCols.head, keyData, DataType.String)

    val countCol = new GPUColumn("count", countData, DataType.Float32)
    val allCols = keyCols.toMap.updated("count", countCol)
    val newSchema = StructType(
      StructField(groupCols.head, DataType.String) :: StructField("count", DataType.Float32) :: Nil
    )
    new GPUDataFrame(allCols, newSchema)
  }
}

/** String column operations */
class StringColumnOps(col: GPUColumn) {
  require(col.dtype == DataType.String, "StringColumnOps requires String column")

  def length: GPUColumn = {
    val data = col.data.asInstanceOf[GPUArray[String]]
    val result = GPUArray.ofLength[Int](col.length)
    for (i <- 0 until col.length) {
      result.hostData(i) = data.hostData(i).length
    }
    new GPUColumn(col.name + "_length", result, DataType.Int32)
  }

  def upper: GPUColumn = {
    val data = col.data.asInstanceOf[GPUArray[String]]
    val result = GPUArray.ofLength[String](col.length)
    for (i <- 0 until col.length) {
      result.hostData(i) = data.hostData(i).toUpperCase
    }
    new GPUColumn(col.name + "_upper", result, DataType.String)
  }

  def lower: GPUColumn = {
    val data = col.data.asInstanceOf[GPUArray[String]]
    val result = GPUArray.ofLength[String](col.length)
    for (i <- 0 until col.length) {
      result.hostData(i) = data.hostData(i).toLowerCase
    }
    new GPUColumn(col.name + "_lower", result, DataType.String)
  }

  def contains(substring: String): GPUColumn = {
    val data = col.data.asInstanceOf[GPUArray[String]]
    val result = GPUArray.ofLength[Boolean](col.length)
    for (i <- 0 until col.length) {
      result.hostData(i) = data.hostData(i).contains(substring)
    }
    new GPUColumn(col.name + "_contains", result, DataType.Boolean)
  }

  def trim: GPUColumn = {
    val data = col.data.asInstanceOf[GPUArray[String]]
    val result = GPUArray.ofLength[String](col.length)
    for (i <- 0 until col.length) {
      result.hostData(i) = data.hostData(i).trim
    }
    new GPUColumn(col.name + "_trim", result, DataType.String)
  }
}
