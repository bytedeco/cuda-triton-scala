package cuda.dsl.dataframe

import cuda.dsl.collections.GPUArray
import scala.jdk.CollectionConverters._

/**
 * Java-friendly bridge for GPUDataFrame operations.
 */
object DataFrameJavaBridge:

  // ========================================================================
  // DataType Bridge
  // ========================================================================

  def dataTypeFromOrdinal(ordinal: Int): DataType = DataType.fromOrdinal(ordinal)

  def dataTypeElementSize(dtype: DataType): Int = dtype.elementSize

  def dataTypeIsPrimitive(dtype: DataType): Boolean = dtype.isPrimitive

  def dataTypeToPandas(dtype: DataType): String = dtype.toPandas

  // ========================================================================
  // StructType Bridge
  // ========================================================================

  def createStructType(fields: Array[StructField]): StructType = StructType(fields.toSeq)

  def createStructField(name: String, dtype: DataType): StructField = StructField(name, dtype)

  def createStructFieldWithNullable(name: String, dtype: DataType, nullable: Boolean): StructField =
    StructField(name, dtype, nullable)

  def structTypeSize(st: StructType): Int = st.size

  def structTypeIndexOf(st: StructType, name: String): Int = st.indexOf(name)

  def structTypeContains(st: StructType, name: String): Boolean = st.contains(name)

  def structTypeGetField(st: StructType, name: String): StructField = st(name)

  def structFieldDtypeName(sf: StructField): String = sf.dtype.toString()

  def structTypeSelect(st: StructType, names: Array[String]): StructType = st.select(names*)

  def structTypeDrop(st: StructType, names: Array[String]): StructType = st.drop(names*)

  def structTypeMerge(left: StructType, right: StructType): StructType = left.merge(right)

  // ========================================================================
  // GPUColumn Bridge
  // ========================================================================

  def createColumn(name: String, floatData: Array[Float]): GPUColumn = {
    val data = GPUArray.fromArray[Float](floatData)
    new GPUColumn(name, data, DataType.Float32)
  }

  def createColumnInt(name: String, intData: Array[Int]): GPUColumn = {
    val data = GPUArray.fromArray[Int](intData)
    new GPUColumn(name, data, DataType.Int32)
  }

  def createColumnLong(name: String, longData: Array[Long]): GPUColumn = {
    val data = GPUArray.fromArray[Long](longData)
    new GPUColumn(name, data, DataType.Int64)
  }

  def createColumnDouble(name: String, doubleData: Array[Double]): GPUColumn = {
    val data = GPUArray.fromArray[Double](doubleData)
    new GPUColumn(name, data, DataType.Float64)
  }

  def createColumnBoolean(name: String, boolData: Array[Boolean]): GPUColumn = {
    val data = GPUArray.fromArray[Boolean](boolData)
    new GPUColumn(name, data, DataType.Boolean)
  }

  def createColumnString(name: String, stringData: Array[String]): GPUColumn = {
    val data = GPUArray.fromArray[String](stringData)
    new GPUColumn(name, data, DataType.String)
  }

  def createColumnFromSeq(name: String, values: Array[Any], dtype: DataType): GPUColumn = {
    GPUColumn(name, values.toSeq, dtype)
  }

  def columnLength(col: GPUColumn): Int = col.length

  def columnIsEmpty(col: GPUColumn): Boolean = col.isEmpty

  def columnName(col: GPUColumn): String = col.name

  def columnDtype(col: GPUColumn): DataType = col.dtype

  def columnDtypeName(col: GPUColumn): String = col.dtype.toString()

  def columnGet(col: GPUColumn, index: Int): Any = col(index)

  def columnToSeqFloat(col: GPUColumn): Array[Float] =
    col.data.asInstanceOf[GPUArray[Float]].hostData

  def columnToSeqInt(col: GPUColumn): Array[Int] =
    col.data.asInstanceOf[GPUArray[Int]].hostData

  def columnToSeqLong(col: GPUColumn): Array[Long] =
    col.data.asInstanceOf[GPUArray[Long]].hostData

  def columnToSeqDouble(col: GPUColumn): Array[Double] =
    col.data.asInstanceOf[GPUArray[Double]].hostData

  def columnToSeqBoolean(col: GPUColumn): Array[Boolean] =
    col.data.asInstanceOf[GPUArray[Boolean]].hostData

  def columnToSeqString(col: GPUColumn): Array[String] =
    col.data.asInstanceOf[GPUArray[String]].hostData

  def columnIsNullAt(col: GPUColumn, index: Int): Boolean = col.isNullAt(index)

  def columnIsNull(col: GPUColumn): GPUColumn = col.isNull

  def columnFillNull(col: GPUColumn, value: Float): GPUColumn = col.fillNull(value)

  def columnDropNull(col: GPUColumn): Array[Object] = {
    val (newCol, mask) = col.dropNull
    Array(newCol.asInstanceOf[Object], mask.asInstanceOf[Object])
  }

  def columnCast(col: GPUColumn, newType: DataType): GPUColumn = col.cast(newType)

  def columnStats(col: GPUColumn): Array[Double] = {
    val s = col.stats()
    Array(s.count.toDouble, s.nullCount.toDouble, s.min, s.max, s.mean, s.sum)
  }

  // ========================================================================
  // GPUDataFrame Bridge
  // ========================================================================

  def createDataFrame(columns: java.util.Map[String, GPUColumn]): GPUDataFrame = {
    import scala.collection.mutable.ArrayBuffer
    val colMap = collection.mutable.LinkedHashMap[String, GPUColumn]()
    val fields = ArrayBuffer.empty[StructField]
    val iter = columns.entrySet().iterator()
    while iter.hasNext do
      val e = iter.next()
      colMap.put(e.getKey, e.getValue)
      fields.append(StructField(e.getKey, e.getValue.dtype))
    val schema = StructType(fields.toSeq)
    // Use ListMap to preserve insertion order
    new GPUDataFrame(collection.immutable.ListMap(colMap.toSeq*), schema)
  }

  def dataFrameNumRows(df: GPUDataFrame): Int = df.numRows

  def dataFrameNumCols(df: GPUDataFrame): Int = df.numCols

  def dataFrameIsEmpty(df: GPUDataFrame): Boolean = df.isEmpty

  def dataFrameColumns(df: GPUDataFrame): Array[String] = df.columns.toArray

  def dataFrameSchema(df: GPUDataFrame): StructType = df.schema

  def dataFrameGetColumn(df: GPUDataFrame, name: String): GPUColumn = df(name)

  def dataFrameSelect(df: GPUDataFrame, cols: Array[String]): GPUDataFrame = df.select(cols*)

  def dataFrameDrop(df: GPUDataFrame, cols: Array[String]): GPUDataFrame = df.drop(cols*)

  def dataFrameSelectCols(df: GPUDataFrame, start: Int, end: Int): GPUDataFrame =
    df.selectCols(start until end)

  def dataFrameFilter(df: GPUDataFrame, condition: GPUColumn): GPUDataFrame = df.filter(condition)

  def dataFrameLimit(df: GPUDataFrame, n: Int): GPUDataFrame = df.limit(n)

  def dataFrameHead(df: GPUDataFrame, n: Int): GPUDataFrame = df.head(n)

  def dataFrameWithColumn(df: GPUDataFrame, name: String, col: GPUColumn): GPUDataFrame =
    df.withColumn(name, col)

  def dataFrameRename(df: GPUDataFrame, oldName: String, newName: String): GPUDataFrame =
    df.rename(oldName, newName)

  def dataFrameCast(df: GPUDataFrame, colName: String, newType: DataType): GPUDataFrame =
    df.cast(colName, newType)

  def dataFrameUnion(df: GPUDataFrame, other: GPUDataFrame): GPUDataFrame = df.union(other)

  def dataFrameJoin(df: GPUDataFrame, other: GPUDataFrame, on: String, how: Int): GPUDataFrame = {
    val joinType = JoinType.fromOrdinal(how)
    df.join(other, on, joinType)
  }

  def dataFrameCollect(df: GPUDataFrame): Array[Array[Object]] = {
    df.collect().map(row => row.toSeq.toArray.asInstanceOf[Array[Object]]).toArray
  }

  def dataFrameToMapSeq(df: GPUDataFrame): Array[java.util.Map[String, Any]] = {
    df.toMapSeq.map(_.asJava).toArray
  }

  def dataFrameDescribe(df: GPUDataFrame, cols: Array[String]): GPUDataFrame = df.describe(cols*)

  // ========================================================================
  // GroupedDataFrame Bridge
  // ========================================================================

  def dataFrameGroupBy(df: GPUDataFrame, cols: Array[String]): GroupedDataFrame = df.groupBy(cols*)

  def groupedDataFrameAgg(gdf: GroupedDataFrame, aggs: Array[Aggregation]): GPUDataFrame =
    gdf.agg(aggs*)

  def groupedDataFrameCount(gdf: GroupedDataFrame): GPUDataFrame = gdf.count()

  // ========================================================================
  // Row Bridge
  // ========================================================================

  def createRow(df: GPUDataFrame, index: Int): Row = Row.fromIndex(df, index)

  def collectRows(df: GPUDataFrame): Array[Row] = {
    (0 until df.numRows).map(i => Row.fromIndex(df, i)).toArray
  }

  def rowGet(row: Row, colName: String): Any = row(colName)

  def rowGetAt(row: Row, colIndex: Int): Any = row(colIndex)

  def rowGetString(row: Row, colName: String): String = row.getString(colName)

  def rowGetInt(row: Row, colName: String): Int = row.getInt(colName)

  def rowGetLong(row: Row, colName: String): Long = row.getLong(colName)

  def rowGetFloat(row: Row, colName: String): Float = row.getFloat(colName)

  def rowGetDouble(row: Row, colName: String): Double = row.getDouble(colName)

  def rowGetBoolean(row: Row, colName: String): Boolean = row.getBoolean(colName)

  def rowIsNullAt(row: Row, colName: String): Boolean = row.isNullAt(colName)

  def rowToMap(row: Row): java.util.Map[String, Any] = row.toMap.asJava

  def rowToSeq(row: Row): Array[Any] = row.toSeq.toArray

  def rowToArray(row: Row): Array[Any] = row.toArray

  def rowSize(row: Row): Int = row.size

  // ========================================================================
  // StringColumnOps Bridge (StringColumnOps takes GPUColumn directly)
  // ========================================================================

  def stringOpsLength(col: GPUColumn): GPUColumn = new StringColumnOps(col).length

  def stringOpsUpper(col: GPUColumn): GPUColumn = new StringColumnOps(col).upper

  def stringOpsLower(col: GPUColumn): GPUColumn = new StringColumnOps(col).lower

  def stringOpsContains(col: GPUColumn, substring: String): GPUColumn =
    new StringColumnOps(col).contains(substring)

  def stringOpsTrim(col: GPUColumn): GPUColumn = new StringColumnOps(col).trim

  // ========================================================================
  // Aggregation Bridge
  // ========================================================================

  def aggregationSum(col: String): Aggregation = Aggregation.sum(col)

  def aggregationMean(col: String): Aggregation = Aggregation.mean(col)

  def aggregationAvg(col: String): Aggregation = Aggregation.avg(col)

  def aggregationCount(col: String): Aggregation = Aggregation.count(col)

  def aggregationMin(col: String): Aggregation = Aggregation.min(col)

  def aggregationMax(col: String): Aggregation = Aggregation.max(col)

  def aggregationStd(col: String): Aggregation = Aggregation.std(col)

  def aggregationVar(col: String): Aggregation = Aggregation.var_(col)

  def aggregationMedian(col: String): Aggregation = Aggregation.median(col)

  def aggregationFirst(col: String): Aggregation = Aggregation.first(col)

  def aggregationLast(col: String): Aggregation = Aggregation.last(col)

  def aggregationNUnique(col: String): Aggregation = Aggregation.nunique(col)

  def aggregationDistinct(col: String): Aggregation = Aggregation.distinct(col)

  def aggregationOutputName(agg: Aggregation): String = agg.outputName

  def aggregationColumn(agg: Aggregation): String = agg.column

  def aggregationType(agg: Aggregation): AggregationType = agg.aggType

  // ========================================================================
  // Package-level Bridge
  // ========================================================================

  def dataFrameFromRows(rows: Array[java.util.Map[String, Any]]): GPUDataFrame = {
    val scalaRows = rows.map(m => m.asScala.toMap)
    gpuRowsToDf(scalaRows.toSeq)
  }

  private def gpuRowsToDf(rows: Seq[Map[String, Any]]): GPUDataFrame = {
    if (rows.isEmpty) {
      return new GPUDataFrame(Map.empty, new StructType())
    }
    val sample = rows.head
    val columns = scala.collection.mutable.Map[String, Seq[Any]]()
    for ((k, v) <- sample) {
      columns(k) = rows.map(_.getOrElse(k, null))
    }
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
    val schema = new StructType(
      gpuColumns.map { case (name, col) => StructField(name, col.dtype) }.toSeq
    )
    new GPUDataFrame(gpuColumns, schema)
  }

  def dataFrameEmpty(st: StructType): GPUDataFrame = new GPUDataFrame(emptyColumns(st), st)

  private def emptyColumns(st: StructType): Map[String, GPUColumn] = {
    st.fields.map { field =>
      val data = field.dtype match {
        case DataType.Float32 => GPUArray.ofLength[Float](0)
        case DataType.Int32 => GPUArray.ofLength[Int](0)
        case DataType.Int64 => GPUArray.ofLength[Long](0)
        case DataType.String => GPUArray.ofLength[String](0)
        case DataType.Boolean => GPUArray.ofLength[Boolean](0)
        case _ => GPUArray.ofLength[Float](0)
      }
      (field.name, new GPUColumn(field.name, data, field.dtype))
    }.toMap
  }

  def dataFrameConcat(dfs: Array[GPUDataFrame]): GPUDataFrame = {
    if (dfs.isEmpty) return new GPUDataFrame(Map.empty, new StructType())
    if (dfs.length == 1) return dfs(0)
    var result = dfs(0)
    for (df <- dfs.tail) {
      result = result.union(df)
    }
    result
  }

  def dataFrameSelectFromDF(cols: java.util.Map[String, GPUColumn]): GPUDataFrame = {
    val scalaMap = cols.asScala.toMap
    val schema = new StructType(scalaMap.values.map(c => StructField(c.name, c.dtype)).toSeq)
    new GPUDataFrame(scalaMap, schema)
  }
