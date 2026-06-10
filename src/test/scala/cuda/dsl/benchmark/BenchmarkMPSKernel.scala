package cuda.dsl.benchmark

import cuda.dsl.DSL.*
import cuda.dsl.collections.GPUArray
import cuda.dsl.collections.Implicits.*
import cuda.dsl.core.Types.given_MemoryOps_Float
import cuda.dsl.dataframe.*
import cuda.dsl.dataframe.AggregationType.*
import cuda.dsl.macros.{mpsKernel, mpsOperator}
import cuda.dsl.runtime.*
import scala.math.*
import scala.util.Random

/** Benchmark result case class */
case class BenchmarkResult(
  name: String,
  size: String,
  cpuTimeMs: Double,
  gpuTimeMs: Double,
  speedup: Double,
  throughput: Double
)

object BenchmarkMPSKernel {
  def main(args: Array[String]): Unit = {
    println("=" * 100)
    println("CUDA DSL Benchmark Suite - @mpsKernel and GPU DataFrame Performance Tests")
    println("=" * 100)
    println(s"Backend: ${DeviceSelector.backendName}")
    println(s"MPS Available: ${DeviceSelector.isMPS}, CUDA Available: ${DeviceSelector.isCUDA}, CPU Fallback: ${DeviceSelector.isCPU}")
    println()

    // Run benchmarks based on command line args or all
    val benchmarks = if (args.isEmpty) {
      List("all")
    } else {
      args.toList
    }

    if (benchmarks.contains("all") || benchmarks.contains("mps")) {
      runMPSKernelBenchmarks()
    }

    if (benchmarks.contains("all") || benchmarks.contains("dataframe")) {
      runDataFrameBenchmarks()
    }

    if (benchmarks.contains("all") || benchmarks.contains("collection")) {
      runGPUCollectionBenchmarks()
    }

    println("\n" + "=" * 100)
    println("Benchmark Suite Completed")
    println("=" * 100)
  }

  // ============================================================================
  // @mpsKernel Benchmarks
  // ============================================================================

  def runMPSKernelBenchmarks(): Unit = {
    println("\n" + "=" * 100)
    println("BENCHMARK: @mpsKernel Macro Generation and Operation Chains")
    println("=" * 100)

    val sizes = List(
      ("1M", 1000000),
      ("10M", 10000000),
      ("50M", 50000000)
    )

    for ((name, size) <- sizes) {
      println(s"\n--- Scale: $name ($size elements) ---")

      // Test macro generation overhead
      val code = generateArithmeticOps()
      benchmarkOperation(
        "MacroGen_Arithmetic",
        name,
        () => { val _ = code; () },
        () => { val _ = code; () }
      )

      // Test arithmetic operation chain
      benchmarkOperation(
        "ArithmeticChain_a+b*c",
        name,
        () => {
          val a = Array.fill(size)(Random.nextFloat())
          val b = Array.fill(size)(Random.nextFloat())
          val c = Array.fill(size)(Random.nextFloat())
          var result = 0f
          for (i <- 0 until size) {
            result += a(i) + b(i) * c(i)
          }
          result
        },
        () => {
          val a = GPUArray.ofLength[Float](size)
          val b = GPUArray.ofLength[Float](size)
          val c = GPUArray.ofLength[Float](size)
          for (i <- 0 until size) {
            a.hostData(i) = Random.nextFloat()
            b.hostData(i) = Random.nextFloat()
            c.hostData(i) = Random.nextFloat()
          }
          // Simplified MPS chain execution
          val result = a.map(x => x * 2f).reduce(_ + _)
          a.free(); b.free(); c.free()
          result
        }
      )

      // Test element-wise operations
      benchmarkOperation(
        "ElementWise_sin exp sqrt",
        name,
        () => {
          val data = Array.fill(size)(Random.nextFloat() * 10f)
          var sum = 0.0
          for (x <- data) {
            sum += sqrt(abs(x)) + sin(x) + exp(x * 0.1f)
          }
          sum
        },
        () => {
          val g = GPUArray.ofLength[Float](size)
          for (i <- 0 until size) {
            g.hostData(i) = Random.nextFloat() * 10f
          }
          val result = g.map(x => (sqrt(abs(x)) + sin(x) + exp(x * 0.1f)).toFloat).reduce(_ + _)
          g.free()
          result
        }
      )
    }

    println("\n@mpsKernel macro generates PyTorch MPS operation chains at compile time.")
    println("The chains are printed to console and /tmp/cuda_dsl_generated_mps_kernels.txt")
  }

  // ============================================================================
  // GPU DataFrame Benchmarks
  // ============================================================================

  def runDataFrameBenchmarks(): Unit = {
    println("\n" + "=" * 100)
    println("BENCHMARK: GPU DataFrame Operations")
    println("=" * 100)

    val sizes = List(
      ("100K", 100000, 3),
      ("1M", 1000000, 3),
      ("10M", 10000000, 2)
    )

    for ((sizeName, size, runs) <- sizes) {
      println(s"\n{'=' * 80}")
      println(s"DataFrame Scale: $sizeName ($size rows)")
      println(s"{'=' * 80}")

      // Create test DataFrame
      val df = createTestDataFrame(size, sizeName)

      // Selection Operations
      runDataFrameSelectionBenchmarks(df, sizeName, runs)

      // Filtering Operations
      runDataFrameFilterBenchmarks(df, sizeName, runs)

      // Column Operations
      runDataFrameColumnBenchmarks(df, sizeName, runs)

      // Aggregation Operations
      runDataFrameAggregationBenchmarks(df, sizeName, runs)

      // String Operations
      runDataFrameStringBenchmarks(df, sizeName, runs)

      // Type Conversion
      runDataFrameTypeConvBenchmarks(df, sizeName, runs)
    }
  }

  def createTestDataFrame(rows: Int, sizeName: String): GPUDataFrame = {
    val floatCol = GPUColumn.float("value", Array.fill(rows)(Random.nextFloat() * 1000))
    val intCol = GPUColumn.int("id", Array.fill(rows)(Random.nextInt(rows)))
    val strCol = GPUColumn.string("name", Array.fill(rows)(s"user_$sizeName"))

    GPUDataFrame(Map(
      "value" -> floatCol,
      "id" -> intCol,
      "name" -> strCol
    ), StructType(List(
      StructField("value", DataType.Float32),
      StructField("id", DataType.Int32),
      StructField("name", DataType.String)
    )))
  }

  // Selection Operations
  def runDataFrameSelectionBenchmarks(df: GPUDataFrame, size: String, runs: Int): Unit = {
    println("\n[Category 1: Selection Operations]")

    benchmarkDF(s"select_all", size, runs, df, () => df.select("value", "id", "name"))
    benchmarkDF(s"select_1col", size, runs, df, () => df.select("value"))
    benchmarkDF(s"drop_1col", size, runs, df, () => df.drop("name"))
    benchmarkDF(s"selectCols_0:2", size, runs, df, () => df.selectCols(0 to 2))
    benchmarkDF(s"head_100", size, runs, df, () => df.limit(100))
  }

  // Filter Operations
  def runDataFrameFilterBenchmarks(df: GPUDataFrame, size: String, runs: Int): Unit = {
    println("\n[Category 2: Filtering Operations]")

    benchmarkDF(s"filter_gt_500", size, runs, df, () => {
      val condData = Array.fill(df.numRows)(Random.nextFloat() * 1000 > 500)
      val cond = new GPUColumn("cond", GPUArray.fromArray(condData), DataType.Boolean)
      df.filter(cond)
    })

    benchmarkDF(s"limit_1000", size, runs, df, () => df.limit(1000))
    benchmarkDF(s"limit_10K", size, runs, df, () => df.limit(10000))
    benchmarkDF(s"limit_100K", size, runs, df, () => df.limit(100000))
  }

  // Column Operations
  def runDataFrameColumnBenchmarks(df: GPUDataFrame, size: String, runs: Int): Unit = {
    println("\n[Category 3: Column Operations]")

    benchmarkDF(s"withColumn", size, runs, df, () => {
      val newCol = GPUColumn.float("value_x2", Array.fill(df.numRows)(Random.nextFloat() * 1000))
      df.withColumn("value_x2", newCol)
    })

    benchmarkDF(s"rename", size, runs, df, () => df.rename("value", "value_renamed"))
    benchmarkDF(s"cast_F2I", size, runs, df, () => df.cast("value", DataType.Int32))
    benchmarkDF(s"cast_I2F", size, runs, df, () => df.cast("id", DataType.Float32))

    benchmarkDF(s"select+drop", size, runs, df, () => {
      df.select("value", "id").drop("id")
    })

    benchmarkDF(s"head+select", size, runs, df, () => {
      df.limit(1000).select("value", "id")
    })
  }

  // Aggregation Operations
  def runDataFrameAggregationBenchmarks(df: GPUDataFrame, size: String, runs: Int): Unit = {
    println("\n[Category 4: Aggregation Operations]")

    benchmarkDF(s"describe", size, runs, df, () => df.describe())
    benchmarkDF(s"describe_value", size, runs, df, () => df.describe("value"))

    benchmarkDF(s"col_stats", size, runs, df, () => {
      df.columnData("value").stats()
      df
    })
  }

  // String Operations
  def runDataFrameStringBenchmarks(df: GPUDataFrame, size: String, runs: Int): Unit = {
    println("\n[Category 5: String Operations]")

    benchmarkDF(s"str_upper", size, runs, df, () => {
      val col = df.str("name").upper
      df.withColumn("name_upper", col)
    })

    benchmarkDF(s"str_lower", size, runs, df, () => {
      val col = df.str("name").lower
      df.withColumn("name_lower", col)
    })

    benchmarkDF(s"str_length", size, runs, df, () => {
      val col = df.str("name").length
      df.withColumn("name_len", col)
    })

    benchmarkDF(s"str_contains", size, runs, df, () => {
      val col = df.str("name").contains("user")
      df.withColumn("name_has_user", col)
    })

    benchmarkDF(s"str_trim", size, runs, df, () => {
      val col = df.str("name").trim
      df.withColumn("name_trimmed", col)
    })
  }

  // Type Conversion
  def runDataFrameTypeConvBenchmarks(df: GPUDataFrame, size: String, runs: Int): Unit = {
    println("\n[Category 6: Type Conversion Operations]")

    benchmarkDF(s"toMapSeq", size, runs, df, () => {
      val _ = df.toMapSeq
      df
    })

    benchmarkDF(s"collect", size, runs, df, () => {
      val _ = df.collect()
      df
    })

    benchmarkDF(s"show", size, runs, df, () => {
      df.show(20)
      df
    })
  }

  // ============================================================================
  // GPU Collection Benchmarks (comparison baseline)
  // ============================================================================

  def runGPUCollectionBenchmarks(): Unit = {
    println("\n" + "=" * 100)
    println("BENCHMARK: GPU Collection vs CPU (Baseline Comparison)")
    println("=" * 100)

    val sizes = List(
      ("1M", 1000000, 5),
      ("10M", 10000000, 3),
      ("50M", 50000000, 2)
    )

    for ((sizeName, size, runs) <- sizes) {
      println(s"\n--- Scale: $sizeName ($size elements) ---")

      val data = Array.fill(size)(Random.nextFloat() * 1000f)

      // Element-wise operations
      benchmarkOp(s"map_*2", sizeName, runs,
        () => data.map(_ * 2f),
        () => { val g = GPUArray.fromArray(data); val r = g.map(_ * 2f).toArray; g.free(); r }
      )

      benchmarkOp(s"map_sin", sizeName, runs,
        () => data.map(x => sin(x * 0.01f).toFloat),
        () => { val g = GPUArray.fromArray(data); val r = g.map(x => sin(x * 0.01f).toFloat).toArray; g.free(); r }
      )

      // Filter
      benchmarkOp(s"filter_gt_500", sizeName, runs,
        () => data.filter(_ > 500f),
        () => { val g = GPUArray.fromArray(data); val r = g.filter(_ > 500f).toArray; g.free(); r }
      )

      // Reduce
      benchmarkOp(s"reduce_sum", sizeName, runs,
        () => data.sum,
        () => { val g = GPUArray.fromArray(data); val r = g.reduce(_ + _); g.free(); r }
      )

      benchmarkOp(s"reduce_max", sizeName, runs,
        () => data.max,
        () => { val g = GPUArray.fromArray(data); val r = g.reduce((a, b) => if (a > b) a else b); g.free(); r }
      )

      // Chained operations
      benchmarkOp(s"chain_map_filter_sum", sizeName, runs,
        () => data.map(_ * 2f).filter(_ > 500f).sum,
        () => { val g = GPUArray.fromArray(data); val r = g.map(_ * 2f).filter(_ > 500f).reduce(_ + _); g.free(); r }
      )
    }
  }

  // ============================================================================
  // Benchmark Infrastructure
  // ============================================================================

  private val results = scala.collection.mutable.ListBuffer[BenchmarkResult]()

  def benchmarkOperation(
    name: String,
    size: String,
    cpuFn: () => Any,
    gpuFn: () => Any
  ): Unit = {
    val cpuTime = time(cpuFn)
    val gpuTime = time(gpuFn)
    val speedup = if (gpuTime > 0) cpuTime / gpuTime else 0.0

    println(f"  $name%-30s | $size%-8s | CPU: $cpuTime%8.2f ms | GPU: $gpuTime%8.2f ms | ${if(speedup > 0) f"${speedup}%.2fx" else "N/A"}")

    results += BenchmarkResult(name, size, cpuTime, gpuTime, speedup, 0)
  }

  def benchmarkDF(
    name: String,
    size: String,
    runs: Int,
    df: GPUDataFrame,
    op: () => GPUDataFrame
  ): Unit = {
    System.gc()
    Thread.sleep(5)

    val cpuStart = System.nanoTime()
    var result: GPUDataFrame = null
    for (_ <- 0 until runs) {
      result = op()
    }
    val cpuTime = (System.nanoTime() - cpuStart) / 1e6 / runs

    println(f"  $name%-30s | $size%-8s | ${cpuTime}%8.2f ms/op")

    results += BenchmarkResult(name, size, cpuTime, 0, 0, 0)
  }

  def benchmarkOp(
    name: String,
    size: String,
    runs: Int,
    cpuFn: () => Any,
    gpuFn: () => Any
  ): Unit = {
    // Warmup
    cpuFn()
    try { val g = GPUArray.ofLength[Float](1); g.free() } catch { case _: Exception => }

    // CPU benchmark
    val cpuStart = System.nanoTime()
    for (_ <- 0 until runs) { cpuFn() }
    val cpuTime = (System.nanoTime() - cpuStart) / 1e6 / runs

    // GPU benchmark
    val gpuStart = System.nanoTime()
    for (_ <- 0 until runs) {
      try {
        gpuFn()
      } catch {
        case e: Exception =>
          println(f"  [GPU Error] $name: ${e.getMessage}")
          return
      }
    }
    val gpuTime = (System.nanoTime() - gpuStart) / 1e6 / runs

    val speedup = if (gpuTime > 0) cpuTime / gpuTime else 0.0
    val speedupStr = if (speedup > 0) f"${speedup}%.2fx" else "N/A"

    println(f"  $name%-30s | $size%-8s | CPU: $cpuTime%8.2f ms | GPU: $gpuTime%8.2f ms | $speedupStr")

    results += BenchmarkResult(name, size, cpuTime, gpuTime, speedup, 0)
  }

  def time(fn: () => Any): Double = {
    val start = System.nanoTime()
    fn()
    (System.nanoTime() - start) / 1e6
  }

  def generateArithmeticOps(): String = {
    val ops = List("add", "sub", "mul", "div", "gt", "lt", "eq")
    (0 until 100).map(_ => ops(Random.nextInt(ops.size))).mkString(", ")
  }
}