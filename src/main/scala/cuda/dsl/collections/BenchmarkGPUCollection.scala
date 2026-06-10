package cuda.dsl.collections

import cuda.dsl.DSL._
import cuda.dsl.collections.Implicits._
import cuda.dsl.core.Types.given_MemoryOps_Float
import scala.math._

/** Comprehensive GPU vs CPU benchmark for GPU collection operations.
 *  Tests 35+ operators at 4 different scales with focus on compute-intensive workloads.
 */
@main def BenchmarkGPUCollection(): Unit = {
  println("=" * 100)
  println("GPU Collection Benchmark - High-Performance GPU Computing Test")
  println("=" * 100)
  println(s"Backend: ${DeviceSelector.backendName}")
  println(s"MPS: ${DeviceSelector.isMPS}, CUDA: ${DeviceSelector.isCUDA}, CPU: ${DeviceSelector.isCPU}")
  println()

  // Different scale sizes (adjusted to avoid OOM)
  val sizes = List(
    ("1M", 1000000),
    ("10M", 10000000),
    ("20M", 20000000),
    ("50M", 50000000)
  )

  val benchmarkRuns = 3

  // ============================================================================
  // Test Categories:
  // 1. Element-wise Operations (8 operators)
  // 2. Filtering Operations (4 operators)
  // 3. Aggregation Operations (10 operators)
  // 4. Transformation Operations (2 operators)
  // 5. Grouping Operations (3 operators)
  // 6. Scanning Operations (2 operators)
  // 7. Chained Operations (8 operators)
  // Total: 37 operators
  // ============================================================================

  for ((scaleName, size) <- sizes) {
    println("=" * 100)
    println(s"SCALE: $scaleName (${size} elements)")
    println("=" * 100)

    // Generate test data
    val data = (0 until size).map(i => (i % 10000).toFloat).toArray

    // ========== Category 1: Element-wise Operations (8) ==========
    runElementWiseBenchmarks(data, size, scaleName, benchmarkRuns)

    // ========== Category 2: Filtering Operations (4) ==========
    runFilterBenchmarks(data, size, scaleName, benchmarkRuns)

    // ========== Category 3: Aggregation Operations (10) ==========
    runAggregationBenchmarks(data, size, scaleName, benchmarkRuns)

    // ========== Category 4: Transformation Operations (2) ==========
    runTransformBenchmarks(data, size, scaleName, benchmarkRuns)

    // ========== Category 5: Grouping Operations (3) ==========
    runGroupingBenchmarks(data, size, scaleName, benchmarkRuns)

    // ========== Category 6: Scanning Operations (2) ==========
    runScanBenchmarks(data, size, scaleName, benchmarkRuns)

    // ========== Category 7: Chained Operations (8) ==========
    runChainedBenchmarks(data, size, scaleName, benchmarkRuns)

    println()
  }

  printSummary()

  DeviceSelector.backendName match {
    case "MPS" =>
      println("\nNote: For true 10x-100x speedup, native CUDA kernels are required.")
      println("MPS backend uses PyTorch which has inherent wrapper overhead.")
    case "CUDA" =>
      println("\nNote: CUDA backend should achieve 10x-100x speedup for compute-intensive ops.")
    case _ =>
      println("\nNote: CPU fallback mode.")
  }
}

// ============================================================================
// Benchmark Runners by Category
// ============================================================================

/** Element-wise operations: map */
def runElementWiseBenchmarks(data: Array[Float], size: Int, scale: String, runs: Int): Unit = {
  println("\n[Category 1: Element-wise Operations] (8 operators)")
  println("-" * 100)

  // Map: x * 2.0f
  benchmark(s"map(*2)", scale, size, runs,
    () => data.map(_ * 2.0f),
    () => { val g = GPUArray.fromArray(data); g.map(_ * 2.0f).toArray; g.free() }
  )

  // Map: x + 10.0f
  benchmark(s"map(+10)", scale, size, runs,
    () => data.map(_ + 10.0f),
    () => { val g = GPUArray.fromArray(data); g.map(_ + 10.0f).toArray; g.free() }
  )

  // Map: x / 2.0f
  benchmark(s"map(/2)", scale, size, runs,
    () => data.map(_ / 2.0f),
    () => { val g = GPUArray.fromArray(data); g.map(_ / 2.0f).toArray; g.free() }
  )

  // Map: sin(x * 0.01)
  benchmark(s"map(sin)", scale, size, runs,
    () => data.map(x => sin(x * 0.01f).toFloat),
    () => { val g = GPUArray.fromArray(data); g.map(x => sin(x * 0.01f).toFloat).toArray; g.free() }
  )

  // Map: x * x (square)
  benchmark(s"map(x²)", scale, size, runs,
    () => data.map(x => x * x),
    () => { val g = GPUArray.fromArray(data); g.map(x => x * x).toArray; g.free() }
  )

  // Map: sqrt(abs(x))
  benchmark(s"map(sqrt)", scale, size, runs,
    () => data.map(x => sqrt(abs(x)).toFloat),
    () => { val g = GPUArray.fromArray(data); g.map(x => sqrt(abs(x)).toFloat).toArray; g.free() }
  )

  // Map: log(abs(x) + 1)
  benchmark(s"map(log)", scale, size, runs,
    () => data.map(x => log(abs(x) + 1).toFloat),
    () => { val g = GPUArray.fromArray(data); g.map(x => log(abs(x) + 1).toFloat).toArray; g.free() }
  )

  // Map chain: x * 2 + 1
  benchmark(s"map(x*2+1)", scale, size, runs,
    () => data.map(x => x * 2 + 1),
    () => { val g = GPUArray.fromArray(data); g.map(x => x * 2 + 1).toArray; g.free() }
  )
}

/** Filtering operations: filter */
def runFilterBenchmarks(data: Array[Float], size: Int, scale: String, runs: Int): Unit = {
  println("\n[Category 2: Filtering Operations] (4 operators)")
  println("-" * 100)

  // Filter: x > 5000
  benchmark(s"filter(>5000)", scale, size, runs,
    () => data.filter(_ > 5000),
    () => { val g = GPUArray.fromArray(data); g.filter(_ > 5000).toArray; g.free() }
  )

  // Filter: x % 2 == 0
  benchmark(s"filter(even)", scale, size, runs,
    () => data.filter(_ % 2 == 0),
    () => { val g = GPUArray.fromArray(data); g.filter(_ % 2 == 0).toArray; g.free() }
  )

  // Filter: x > 100 and x < 9000
  benchmark(s"filter(range)", scale, size, runs,
    () => data.filter(x => x > 100 && x < 9000),
    () => { val g = GPUArray.fromArray(data); g.filter(x => x > 100 && x < 9000).toArray; g.free() }
  )

  // Filter: sin(x) > 0
  benchmark(s"filter(sin>0)", scale, size, runs,
    () => data.filter(x => sin(x * 0.01) > 0),
    () => { val g = GPUArray.fromArray(data); g.filter(x => sin(x * 0.01) > 0).toArray; g.free() }
  )
}

/** Aggregation operations: reduce, fold */
def runAggregationBenchmarks(data: Array[Float], size: Int, scale: String, runs: Int): Unit = {
  println("\n[Category 3: Aggregation Operations] (10 operators)")
  println("-" * 100)

  // Sum
  benchmark(s"sum", scale, size, runs,
    () => data.sum,
    () => { val g = GPUArray.fromArray(data); val r = g.reduce(_ + _); g.free(); r }
  )

  // Product
  benchmark(s"product", scale, size, runs,
    () => data.product,
    () => { val g = GPUArray.fromArray(data); val r = g.reduce(_ * _); g.free(); r }
  )

  // Min
  benchmark(s"min", scale, size, runs,
    () => data.min,
    () => { val g = GPUArray.fromArray(data); val r = g.reduce((a, b) => if (a < b) a else b); g.free(); r }
  )

  // Max
  benchmark(s"max", scale, size, runs,
    () => data.max,
    () => { val g = GPUArray.fromArray(data); val r = g.reduce((a, b) => if (a > b) a else b); g.free(); r }
  )

  // Count
  benchmark(s"count(>5000)", scale, size, runs,
    () => data.count(_ > 5000),
    () => { val g = GPUArray.fromArray(data); val r = g.filter(_ > 5000).toArray.length; g.free(); r }
  )

  // Average
  benchmark(s"avg", scale, size, runs,
    () => data.sum / data.length,
    () => { val g = GPUArray.fromArray(data); val r = g.reduce(_ + _) / data.length; g.free(); r }
  )

  // Variance (2-pass)
  benchmark(s"variance", scale, size, runs,
    () => {
      val mean = data.sum / data.length
      data.map(x => (x - mean) * (x - mean)).sum / data.length
    },
    () => {
      val g = GPUArray.fromArray(data)
      val mean = g.reduce(_ + _) / data.length
      val r = g.map(x => (x - mean) * (x - mean)).reduce(_ + _) / data.length
      g.free()
      r
    }
  )

  // Fold: sum with initial value
  benchmark(s"fold(sum)", scale, size, runs,
    () => data.fold(1000f)(_ + _),
    () => { val g = GPUArray.fromArray(data); val r = g.fold(1000f)(_ + _); g.free(); r }
  )

  // Fold: max with initial
  benchmark(s"fold(max)", scale, size, runs,
    () => data.fold(-1e9f)((a, b) => if (a > b) a else b),
    () => { val g = GPUArray.fromArray(data); val r = g.fold(-1e9f)((a, b) => if (a > b) a else b); g.free(); r }
  )

  // Sum of squares
  benchmark(s"sumSquares", scale, size, runs,
    () => data.map(x => x * x).sum,
    () => { val g = GPUArray.fromArray(data); val r = g.map(x => x * x).reduce(_ + _); g.free(); r }
  )
}

/** Transformation operations: zip, union */
def runTransformBenchmarks(data: Array[Float], size: Int, scale: String, runs: Int): Unit = {
  println("\n[Category 4: Transformation Operations] (2 operators)")
  println("-" * 100)

  // Zip two arrays
  val data2 = (0 until size).map(i => (i * 2 % 10000).toFloat).toArray
  benchmark(s"zip", scale, size, runs,
    () => data.zip(data2).map { case (a, b) => a + b },
    () => {
      val g1 = GPUArray.fromArray(data)
      val g2 = GPUArray.fromArray(data2)
      val r = g1.map(_ + 1f).toArray  // Simplified
      g1.free()
      g2.free()
      r
    },
    skipGpu = true  // Zip not fully implemented
  )

  // Union simulation (concatenation)
  benchmark(s"concat", scale, size, runs,
    () => data ++ data,
    () => {
      val g = GPUArray.fromArray(data)
      val r = g.toArray ++ g.toArray
      g.free()
      r
    },
    skipGpu = true
  )
}

/** Grouping operations */
def runGroupingBenchmarks(data: Array[Float], size: Int, scale: String, runs: Int): Unit = {
  println("\n[Category 5: Grouping Operations] (3 operators)")
  println("-" * 100)

  // Group by key (mod 100)
  benchmark(s"groupBy(mod100)", scale, size, runs,
    () => data.groupBy(_ % 100).view.mapValues(_.sum).toMap,
    () => {
      val g = GPUArray.fromArray(data)
      val r = g.toArray.groupBy(_ % 100).view.mapValues(_.sum).toMap
      g.free()
      r
    },
    skipGpu = true
  )

  // Partition by predicate
  benchmark(s"partition", scale, size, runs,
    () => data.partition(_ > 5000),
    () => {
      val g = GPUArray.fromArray(data)
      val r = g.toArray.partition(_ > 5000)
      g.free()
      r
    },
    skipGpu = true
  )

  // Partition by even/odd
  benchmark(s"partition(even)", scale, size, runs,
    () => data.partition(_ % 2 == 0),
    () => {
      val g = GPUArray.fromArray(data)
      val r = g.toArray.partition(_ % 2 == 0)
      g.free()
      r
    },
    skipGpu = true
  )
}

/** Scanning operations */
def runScanBenchmarks(data: Array[Float], size: Int, scale: String, runs: Int): Unit = {
  println("\n[Category 6: Scanning Operations] (2 operators)")
  println("-" * 100)

  // Running sum (prefix sum)
  benchmark(s"scan(sum)", scale, size, runs,
    () => {
      var sum = 0f
      data.map(x => { sum += x; sum })
    },
    () => {
      val g = GPUArray.fromArray(data)
      val arr = g.toArray
      var sum = 0f
      val result = arr.map(x => { sum += x; sum })
      g.free()
      result
    },
    skipGpu = true
  )

  // Running max
  benchmark(s"scan(max)", scale, size, runs,
    () => {
      var max = -1e9f
      data.map(x => { max = math.max(max, x); max })
    },
    () => {
      val g = GPUArray.fromArray(data)
      val arr = g.toArray
      var max = -1e9f
      val result = arr.map(x => { max = math.max(max, x); max })
      g.free()
      result
    },
    skipGpu = true
  )
}

/** Chained operations - multiple operations combined */
def runChainedBenchmarks(data: Array[Float], size: Int, scale: String, runs: Int): Unit = {
  println("\n[Category 7: Chained Operations] (8 operators)")
  println("-" * 100)

  // Map -> Filter -> Reduce
  benchmark(s"map->filter->sum", scale, size, runs,
    () => data.map(_ * 2).filter(_ > 5000).sum,
    () => { val g = GPUArray.fromArray(data); val r = g.map(_ * 2).filter(_ > 5000).reduce(_ + _); g.free(); r }
  )

  // Filter -> Map -> Reduce
  benchmark(s"filter->map->sum", scale, size, runs,
    () => data.filter(_ > 5000).map(_ * 2).sum,
    () => { val g = GPUArray.fromArray(data); val r = g.filter(_ > 5000).map(_ * 2).reduce(_ + _); g.free(); r }
  )

  // Map -> Map -> Reduce
  benchmark(s"map->map->sum", scale, size, runs,
    () => data.map(_ * 2).map(_ + 1).sum,
    () => { val g = GPUArray.fromArray(data); val r = g.map(_ * 2).map(_ + 1).reduce(_ + _); g.free(); r }
  )

  // Filter -> Map -> Filter -> Sum
  benchmark(s"filter->map->filter->sum", scale, size, runs,
    () => data.filter(_ > 100).map(_ * 2).filter(_ < 20000).sum,
    () => { val g = GPUArray.fromArray(data); val r = g.filter(_ > 100).map(_ * 2).filter(_ < 20000).reduce(_ + _); g.free(); r }
  )

  // Complex chain: map(sin) -> filter -> sum
  benchmark(s"map(sin)->filter->sum", scale, size, runs,
    () => data.map(x => sin(x * 0.01f)).filter(_ > 0).sum,
    () => { val g = GPUArray.fromArray(data); val r = g.map(x => sin(x * 0.01f).toFloat).filter(_ > 0).reduce(_ + _); g.free(); r }
  )

  // Map with expensive function -> sum
  benchmark(s"map(exp)->sum", scale, size, runs,
    () => data.map(x => exp(x * 0.001f).toFloat).sum,
    () => { val g = GPUArray.fromArray(data); val r = g.map(x => exp(x * 0.001f).toFloat).reduce(_ + _); g.free(); r }
  )

  // Multiple aggregations on same data
  benchmark(s"multiAgg(sum,max)", scale, size, runs,
    () => (data.sum, data.max),
    () => {
      val g = GPUArray.fromArray(data)
      val s = g.reduce(_ + _)
      val m = g.reduce((a, b) => if (a > b) a else b)
      g.free()
      (s, m)
    }
  )

  // Fold -> Map -> Reduce
  benchmark(s"fold->map->sum", scale, size, runs,
    () => data.fold(0f)(_ + _),
    () => { val g = GPUArray.fromArray(data); val r = g.fold(0f)(_ + _); g.free(); r }
  )
}

// ============================================================================
// Core Benchmark Infrastructure
// ============================================================================

case class BenchmarkResult(
  name: String,
  scale: String,
  cpuTime: Double,
  gpuTime: Double,
  speedup: Double,
  skipped: Boolean = false
)

private var allResults: List[BenchmarkResult] = List.empty

/** Run a single benchmark and print results */
def benchmark(
  name: String,
  scale: String,
  size: Int,
  runs: Int,
  cpuFn: () => Any,
  gpuFn: () => Any,
  skipGpu: Boolean = false
): Unit = {
  // CPU benchmark
  val cpuTimes = for (_ <- 0 until runs) yield {
    val start = System.nanoTime()
    cpuFn()
    val end = System.nanoTime()
    (end - start) / 1000000.0
  }
  val cpuAvgTime = cpuTimes.sum / cpuTimes.length

  // GPU benchmark (skip if requested)
  val (gpuAvgTime, speedup, skipped) = if (skipGpu) {
    (0.0, 0.0, true)
  } else {
    try {
      val gpuTimes = for (_ <- 0 until runs) yield {
        val start = System.nanoTime()
        gpuFn()
        val end = System.nanoTime()
        (end - start) / 1000000.0
      }
      val avgTime = gpuTimes.sum / gpuTimes.length
      val sp = if (avgTime > 0) cpuAvgTime / avgTime else 0.0
      (avgTime, sp, false)
    } catch {
      case e: Exception =>
        println(f"  [GPU Error] ${e.getMessage}")
        (0.0, 0.0, true)
    }
  }

  val speedupStr = if (skipped) "SKIPPED" else if (speedup > 0) f"${speedup}%.2fx" else "N/A"

  println(f"  ${name}%-25s | ${scale}%-5s | ${cpuAvgTime}%9.2f ms | ${if(skipped) "-" else f"${gpuAvgTime}%9.2f ms"} | ${speedupStr}")

  allResults = BenchmarkResult(name, scale, cpuAvgTime, gpuAvgTime, speedup, skipped) :: allResults
}

/** Print summary table of all results */
def printSummary(): Unit = {
  println("\n" + "=" * 100)
  println("SUMMARY: Best GPU Accelerations by Scale")
  println("=" * 100)

  val validResults = allResults.filter(r => !r.skipped && r.speedup > 0)

  for (scale <- List("1M", "10M", "50M", "100M")) {
    val scaleResults = validResults.filter(_.scale == scale)
    if (scaleResults.nonEmpty) {
      val best = scaleResults.maxByOption(_.speedup)
      best.foreach { r =>
        println(f"  ${scale}: ${r.name}%-25s ${r.speedup}%6.2fx speedup (CPU: ${r.cpuTime}%8.2f ms, GPU: ${r.gpuTime}%8.2f ms)")
      }
    }
  }

  println("\n" + "=" * 100)
  println("ALL RESULTS (sorted by speedup)")
  println("=" * 100)
  println("%-25s | %-5s | %-10s | %-10s | %-8s".format("Operation", "Scale", "CPU (ms)", "GPU (ms)", "Speedup"))
  println("-" * 80)

  for (r <- validResults.sortBy(-_.speedup).take(30)) {
    val spStr = if (r.skipped) "SKIPPED" else f"${r.speedup}%6.2fx"
    println("%-25s | %-5s | %10.2f | %10s | %8s".format(r.name, r.scale,
      r.cpuTime,
      if (r.skipped) "-" else r.gpuTime.toString,
      spStr))
  }
}
