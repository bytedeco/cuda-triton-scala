package cuda.dsl

/** GPU DataFrame package.
 *  Provides GPU-accelerated DataFrame with pandas-like API.
 */
package object dataframe {
  // Re-export common types
  type DataFrame = GPUDataFrame
  type Column = GPUColumn

  /** Create DataFrame from column map */
  def apply(columns: Map[String, GPUColumn]): GPUDataFrame = {
    require(columns.nonEmpty, "DataFrame must have at least one column")
    val schema = StructType(
      columns.map { case (name, col) => StructField(name, col.dtype) }.toSeq
    )
    new GPUDataFrame(columns, schema)
  }

  /** Create DataFrame from rows */
  def fromRows(rows: Seq[Map[String, Any]]): GPUDataFrame = {
    if (rows.isEmpty) {
      return new GPUDataFrame(Map.empty, StructType(Nil))
    }

    // Infer schema from first row
    val sample = rows.head
    val columns = scala.collection.mutable.LinkedHashMap[String, Seq[Any]]()

    for ((k, v) <- sample) {
      columns(k) = rows.map(_.getOrElse(k, null))
    }

    // Create GPUColumns
    val gpuColumns = columns.map { case (name, values) =>
      val dtype = values.head match {
        case _: Int => DataType.Int32
        case _: Long => DataType.Int64
        case _: Float => DataType.Float32
        case _: Double => DataType.Float64
        case _: Boolean => DataType.Boolean
        case _: String => DataType.String
        case _ => DataType.String
      }
      (name, GPUColumn(name, values, dtype))
    }.toMap

    val schema = StructType(
      gpuColumns.map { case (name, col) => StructField(name, col.dtype) }.toSeq
    )
    new GPUDataFrame(gpuColumns, schema)
  }

  /** Create DataFrame from CSV file */
  def readCSV(path: String, delimiter: Char = ','): GPUDataFrame = {
    // Simple CSV reader - for production use PyTorch's CSV loader
    import scala.io.Source
    val source = Source.fromFile(path)
    val lines = source.getLines().toVector
    source.close()

    if (lines.isEmpty) {
      return new GPUDataFrame(Map.empty, StructType(Nil))
    }

    val header = lines.head.split(delimiter.toString).map(_.trim)
    val dataLines = lines.tail

    val columnData = scala.collection.mutable.LinkedHashMap[String, Seq[String]]()
    for (col <- header) {
      columnData(col) = dataLines.map(line => {
        val cols = line.split(delimiter.toString)
        if (cols.length > header.indexOf(col)) cols(header.indexOf(col)).trim else ""
      })
    }

    // Infer types and create columns
    val gpuColumns = columnData.map { case (name, values) =>
      val dtype = inferType(values)
      val col = GPUColumn(name, values, dtype)
      (name, col)
    }.toMap

    val schema = StructType(
      gpuColumns.map { case (name, col) => StructField(name, col.dtype) }.toSeq
    )
    new GPUDataFrame(gpuColumns, schema)
  }

  private def inferType(values: Seq[String]): DataType = {
    val sample = values.filter(_.nonEmpty).take(100)
    if (sample.isEmpty) return DataType.String

    // Check if all are integers
    if (sample.forall(v => v.toLongOption.isDefined)) {
      return DataType.Int64
    }

    // Check if all are floats
    if (sample.forall(v => v.toDoubleOption.isDefined)) {
      return DataType.Float64
    }

    // Check if all are booleans
    if (sample.forall(v => v.toBooleanOption.isDefined)) {
      return DataType.Boolean
    }

    DataType.String
  }

  /** Create empty DataFrame with schema */
  def empty(schema: StructType): GPUDataFrame = {
    import cuda.dsl.collections.GPUArray
    val columns = scala.collection.mutable.LinkedHashMap[String, GPUColumn]()
    for (field <- schema.fields) {
      val data = field.dtype match {
        case DataType.Float32 => GPUArray.ofLength[Float](0)
        case DataType.Int32 => GPUArray.ofLength[Int](0)
        case DataType.Int64 => GPUArray.ofLength[Long](0)
        case DataType.String => GPUArray.ofLength[String](0)
        case DataType.Boolean => GPUArray.ofLength[Boolean](0)
        case _ => GPUArray.ofLength[Float](0)
      }
      columns(field.name) = new GPUColumn(field.name, data, field.dtype)
    }
    new GPUDataFrame(columns.toMap, schema)
  }

  /** Concatenate DataFrames vertically */
  def concat(dfs: Seq[GPUDataFrame]): GPUDataFrame = {
    if (dfs.isEmpty) return new GPUDataFrame(Map.empty, StructType(Nil))
    if (dfs.length == 1) return dfs.head

    var result = dfs.head
    for (df <- dfs.tail) {
      result = result.union(df)
    }
    result
  }
}
