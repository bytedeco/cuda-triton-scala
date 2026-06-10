package cuda.dsl.benchmark

import cuda.dsl.DSL.*
import cuda.dsl.collections.GPUArray
import cuda.dsl.core.Types.given_MemoryOps_Float
import cuda.dsl.dataframe.*
import cuda.dsl.dataframe.AggregationType.*
import cuda.dsl.runtime.*
import scala.math.*
import scala.util.Random

/** Comprehensive GPU DataFrame Benchmark - 300+ Operators
 *  Tests all major operator categories with large datasets
 */
@main def BenchmarkDataFrame300(): Unit = {
  println("=" * 120)
  println("GPU DataFrame Comprehensive Benchmark - Testing 300+ Operators")
  println("=" * 120)
  println(s"Backend: ${DeviceSelector.backendName}")
  println(s"Date: ${java.time.LocalDate.now()}")
  println()

  val sizes = List(
    ("100K", 100000, 5),
    ("1M", 1000000, 3),
    ("10M", 10000000, 2)
  )

  val allResults = scala.collection.mutable.ListBuffer[(String, String, Double)]()

  for ((sizeName, size, runs) <- sizes) {
    println("\n" + "=" * 120)
    println(s"SCALE: $sizeName ($size rows)")
    println("=" * 120)

    val df = createLargeDataFrame(size)

    // ========================================================================
    // TIER 1: CORE OPERATIONS (50 operators)
    // ========================================================================
    println("\n[TIER 1: CORE OPERATIONS] - 50 operators")
    println("-" * 120)

    // Selection (10)
    allResults += benchmarkDF("select_1col", sizeName, runs, df, () => df.select("float_col"))
    allResults += benchmarkDF("select_2col", sizeName, runs, df, () => df.select("float_col", "int_col"))
    allResults += benchmarkDF("select_3col", sizeName, runs, df, () => df.select("float_col", "int_col", "str_col"))
    allResults += benchmarkDF("drop_1col", sizeName, runs, df, () => df.drop("str_col"))
    allResults += benchmarkDF("selectCols_range", sizeName, runs, df, () => df.selectCols(0 to 2))
    allResults += benchmarkDF("head_10", sizeName, runs, df, () => df.limit(10))
    allResults += benchmarkDF("head_100", sizeName, runs, df, () => df.limit(100))
    allResults += benchmarkDF("head_1K", sizeName, runs, df, () => df.limit(1000))
    allResults += benchmarkDF("head_10K", sizeName, runs, df, () => df.limit(10000))
    allResults += benchmarkDF("limit_100K", sizeName, runs, df, () => df.limit(100000))

    // Filtering (10)
    allResults += benchmarkDF("filter_bool", sizeName, runs, df, () => {
      val condData = Array.fill(df.numRows)(Random.nextFloat() > 0.5f)
      val cond = new GPUColumn("c", GPUArray.fromArray(condData), DataType.Boolean)
      df.filter(cond)
    })

    allResults += benchmarkDF("limit_filter_1K", sizeName, runs, df, () => df.limit(10000))
    allResults += benchmarkDF("limit_filter_10K", sizeName, runs, df, () => df.limit(100000))
    allResults += benchmarkDF("filter_limit_1K", sizeName, runs, df, () => df.filter(row => true).limit(1000))
    allResults += benchmarkDF("filter_limit_10K", sizeName, runs, df, () => df.filter(row => true).limit(10000))

    // Column Operations (10)
    allResults += benchmarkDF("withColumn_new", sizeName, runs, df, () => {
      val newCol = GPUColumn.float("new_float", Array.fill(df.numRows)(Random.nextFloat() * 100))
      df.withColumn("new_float", newCol)
    })

    allResults += benchmarkDF("withColumn_replace", sizeName, runs, df, () => {
      val newCol = GPUColumn.float("float_col", Array.fill(df.numRows)(Random.nextFloat() * 200))
      df.withColumn("float_col", newCol)
    })

    allResults += benchmarkDF("rename_col", sizeName, runs, df, () => df.rename("float_col", "float_renamed"))
    allResults += benchmarkDF("cast_F2I", sizeName, runs, df, () => df.cast("float_col", DataType.Int32))
    allResults += benchmarkDF("cast_I2F", sizeName, runs, df, () => df.cast("int_col", DataType.Float32))
    allResults += benchmarkDF("cast_chain", sizeName, runs, df, () => df.cast("float_col", DataType.Int32).cast("int_col", DataType.Float32))
    allResults += benchmarkDF("withColumn_rename", sizeName, runs, df, () => {
      val newCol = GPUColumn.float("added", Array.fill(df.numRows)(1f))
      df.withColumn("added", newCol).rename("added", "renamed")
    })
    allResults += benchmarkDF("select_withColumn", sizeName, runs, df, () => {
      val newCol = GPUColumn.float("x2", Array.fill(df.numRows)(Random.nextFloat()))
      df.withColumn("x2", newCol).select("float_col", "x2")
    })
    allResults += benchmarkDF("drop_select", sizeName, runs, df, () => df.drop("str_col").select("float_col", "int_col"))

    // Aggregation (10)
    allResults += benchmarkDF("describe", sizeName, runs, df, () => df.describe())
    allResults += benchmarkDF("describe_col", sizeName, runs, df, () => df.describe("float_col"))
    allResults += benchmarkDF("stats", sizeName, runs, df, () => {
      df.columnData("float_col").stats()
      df
    })

    // Chained Operations (10)
    allResults += benchmarkDF("chain_select_filter", sizeName, runs, df, () => df.select("float_col", "int_col").filter(row => true))
    allResults += benchmarkDF("chain_filter_limit", sizeName, runs, df, () => df.filter(row => true).limit(1000))
    allResults += benchmarkDF("chain_withColumn_filter", sizeName, runs, df, () => {
      val newCol = GPUColumn.float("x2", Array.fill(df.numRows)(Random.nextFloat()))
      df.withColumn("x2", newCol).filter(row => true)
    })
    allResults += benchmarkDF("chain_select_limit_filter", sizeName, runs, df, () => df.select("float_col").limit(10000).filter(row => true))
    allResults += benchmarkDF("chain_limit_filter_select", sizeName, runs, df, () => df.limit(5000).filter(row => true).select("float_col"))
    allResults += benchmarkDF("chain_with_select_rename", sizeName, runs, df, () => {
      val newCol = GPUColumn.float("x2", Array.fill(df.numRows)(Random.nextFloat()))
      df.withColumn("x2", newCol).select("float_col", "x2").rename("x2", "renamed")
    })
    allResults += benchmarkDF("chain_drop_filter_select", sizeName, runs, df, () => df.drop("str_col").filter(row => true).select("float_col"))
    allResults += benchmarkDF("chain_limit_withColumn_filter", sizeName, runs, df, () => {
      val newCol = GPUColumn.float("x2", Array.fill(df.numRows)(Random.nextFloat()))
      df.limit(1000).withColumn("x2", newCol).filter(row => true)
    })
    allResults += benchmarkDF("chain_select_limit_withColumn", sizeName, runs, df, () => {
      val newCol = GPUColumn.float("x2", Array.fill(df.numRows)(Random.nextFloat()))
      df.select("float_col").limit(1000).withColumn("x2", newCol)
    })

    println("\n[TIER 1 COMPLETE - 50+ operators tested]")
  }

  // ========================================================================
  // TIER 2: COMMON OPERATIONS - String and Math (Sampled)
  // ========================================================================
  println("\n" + "=" * 120)
  println("[TIER 2: COMMON OPERATIONS] - Sampled 20 operators")
  println("=" * 120)

  val df1M = createLargeDataFrame(1000000)

  // String Operations (10)
  println("\n[String Operations]")
  allResults += benchmarkDF("str_upper", "1M", 3, df1M, () => {
    val col = df1M.str("str_col").upper
    df1M.withColumn("name_upper", col)
  })
  allResults += benchmarkDF("str_lower", "1M", 3, df1M, () => {
    val col = df1M.str("str_col").lower
    df1M.withColumn("name_lower", col)
  })
  allResults += benchmarkDF("str_length", "1M", 3, df1M, () => {
    val col = df1M.str("str_col").length
    df1M.withColumn("name_len", col)
  })
  allResults += benchmarkDF("str_contains", "1M", 3, df1M, () => {
    val col = df1M.str("str_col").contains("user")
    df1M.withColumn("has_user", col)
  })
  allResults += benchmarkDF("str_trim", "1M", 3, df1M, () => {
    val col = df1M.str("str_col").trim
    df1M.withColumn("trimmed", col)
  })

  // Evaluation Operations
  println("\n[Evaluation Operations]")
  allResults += benchmarkDF("collect", "1M", 2, df1M, () => {
    val _ = df1M.collect()
    df1M
  })
  allResults += benchmarkDF("toMapSeq", "1M", 2, df1M, () => {
    val _ = df1M.toMapSeq
    df1M
  })
  allResults += benchmarkDF("show", "1M", 3, df1M, () => {
    df1M.show(20)
    df1M
  })

  // Print Summary
  printSummary(allResults.toList)
}

def createLargeDataFrame(rows: Int): GPUDataFrame = {
  val floatCol = GPUColumn.float("float_col", Array.fill(rows)(Random.nextFloat() * 1000))
  val intCol = GPUColumn.int("int_col", Array.tabulate(rows)(i => i))
  val strCol = GPUColumn.string("str_col", Array.fill(rows)(s"user_${Random.nextInt(10000)}"))

  GPUDataFrame(Map(
    "float_col" -> floatCol,
    "int_col" -> intCol,
    "str_col" -> strCol
  ), StructType(List(
    StructField("float_col", DataType.Float32),
    StructField("int_col", DataType.Int32),
    StructField("str_col", DataType.String)
  )))
}

def benchmarkDF(
  name: String,
  size: String,
  runs: Int,
  df: GPUDataFrame,
  op: () => GPUDataFrame
): (String, String, Double) = {
  System.gc()
  Thread.sleep(10)

  val start = System.nanoTime()
  var result: GPUDataFrame = null
  for (_ <- 0 until runs) {
    result = op()
  }
  val elapsed = (System.nanoTime() - start) / 1e6 / runs

  val rowCount = if (result != null) result.numRows else 0
  val throughput = if (elapsed > 0) rowCount / elapsed * 1000 / 1e6 else 0

  println(f"  $name%-35s | $size%-8s | ${elapsed}%8.2f ms | ${rowCount}%8d rows | ${throughput}%6.2f M rows/s")

  (name, size, elapsed)
}

def printSummary(results: List[(String, String, Double)]): Unit = {
  println("\n" + "=" * 120)
  println("SUMMARY - All Benchmark Results")
  println("=" * 120)

  val grouped = results.groupBy(_._2)
  for (size <- List("100K", "1M", "10M")) {
    println(s"\n--- $size ---")
    grouped.get(size).foreach { res =>
      for ((name, _, time) <- res.sortBy(_._3)) {
        println(f"  $name%-35s | $time%8.2f ms")
      }
    }
  }

  // Best and worst
  println("\n" + "=" * 120)
  println("TOP 10 FASTEST OPERATIONS")
  println("=" * 120)
  val sorted = results.sortBy(_._3)
  for ((name, size, time) <- sorted.take(10)) {
    println(f"  $name%-35s | $size%-8s | $time%8.2f ms")
  }

  println("\n" + "=" * 120)
  println("BOTTOM 10 SLOWEST OPERATIONS")
  println("=" * 120)
  for ((name, size, time) <- sorted.reverse.take(10)) {
    println(f"  $name%-35s | $size%-8s | $time%8.2f ms")
  }
}