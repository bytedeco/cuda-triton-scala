package cuda.dsl.autotune

import cuda.dsl.dsl.*
import cuda.dsl.runtime.*
import cuda.dsl.runtime.ScalaCudaRuntime as SCR

import scala.collection.mutable

/** Kernel configuration for autotuning */
case class KernelConfig(
  blockX: Int = 128,
  blockY: Int = 1,
  blockZ: Int = 1,
  numStages: Int = 2,
  numWarps: Int = 4,
  enablePrefetch: Boolean = false,
  splitK: Int = 1,
  kwargs: Map[String, Any] = Map.empty
) {
  def blockSize: (Int, Int, Int) = (blockX, blockY, blockZ)
  override def toString: String = s"KernelConfig(block=${blockX}x${blockY}x${blockZ}, stages=$numStages, warps=$numWarps, prefetch=$enablePrefetch, splitK=$splitK)"
}

/** Autotune result holding best configuration and timing */
case class AutotuneResult(
  config: KernelConfig,
  latencyMs: Double,
  throughput: Double
)

/** Configuration for a specific autotune key */
case class AutotuneConfig(
  key: String,
  configs: List[KernelConfig],
  prune: Boolean = true,
  maxTrials: Int = 100
)

/** Cache for autotune results */
object AutotuneCache:
  private val cache = mutable.Map[String, AutotuneResult]()

  def get(key: String): Option[AutotuneResult] = cache.get(key)

  def put(key: String, result: AutotuneResult): Unit = cache.put(key, result)

  def clear(): Unit = cache.clear()

  def getBestConfig(key: String): KernelConfig =
    cache.get(key).map(_.config).getOrElse(KernelConfig())

/** Autotuning engine - measures and selects best kernel configuration */
class AutotunerEngine:

  private val warmupRuns = 3
  private val measureRuns = 10

  /** Run autotuning for a kernel with given configs
   *  @param kernelName Name of the kernel to tune
   *  @param configs List of configurations to try
   *  @param inputSize Size of input for benchmarking
   *  @param kernelFn Kernel function to execute
   *  @return Best configuration found
   */
  def tune(
    kernelName: String,
    configs: List[KernelConfig],
    inputSize: Int
  )(kernelFn: KernelConfig => Array[Float]): AutotuneResult = {
    println(s"[Autotuner] Starting autotune for $kernelName with ${configs.size} configs")

    var bestConfig = configs.head
    var bestLatency = Double.MaxValue

    for (config <- configs) {
      // Warmup runs
      for (_ <- 0 until warmupRuns) {
        kernelFn(config)
      }

      // Measure runs
      val timings = for (_ <- 0 until measureRuns) yield {
        val start = System.nanoTime()
        kernelFn(config)
        val end = System.nanoTime()
        (end - start) / 1e6 // ms
      }

      val avgLatency = timings.sum / measureRuns
      val throughput = inputSize.toDouble / (avgLatency / 1000.0)

      println(s"  [Autotuner] ${config.blockSize} -> ${avgLatency.formatted("%.3f")}ms (${throughput.formatted("%.2f")} elems/s)")

      if (avgLatency < bestLatency) {
        bestLatency = avgLatency
        bestConfig = config
      }
    }

    val result = AutotuneResult(bestConfig, bestLatency, inputSize.toDouble / (bestLatency / 1000.0))
    println(s"  [Autotuner] Best: ${bestConfig.blockSize} @ ${bestLatency.formatted("%.3f")}ms")
    result
  }

  /** Tune with grid search strategy */
  def tuneGridSearch(
    kernelName: String,
    blockSizes: List[Int],
    numWarps: List[Int],
    numStages: List[Int],
    inputSize: Int
  )(kernelFn: KernelConfig => Array[Float]): AutotuneResult = {
    val configs = for {
      b <- blockSizes
      w <- numWarps
      s <- numStages
    } yield KernelConfig(blockX = b, numWarps = w, numStages = s)

    tune(kernelName, configs, inputSize)(kernelFn)
  }

  /** Tune with random search strategy (faster for large search spaces) */
  def tuneRandomSearch(
    kernelName: String,
    numTrials: Int,
    blockSizes: Range.Inclusive,
    numWarps: Range.Inclusive,
    numStages: Range.Inclusive,
    inputSize: Int
  )(kernelFn: KernelConfig => Array[Float]): AutotuneResult = {
    val random = new scala.util.Random(42)
    val configs = (0 until numTrials).map { _ =>
      KernelConfig(
        blockX = random.nextInt(blockSizes.length) + blockSizes.head,
        numWarps = random.nextInt(numWarps.length) + numWarps.head,
        numStages = random.nextInt(numStages.length) + numStages.head
      )
    }.toList

    tune(kernelName, configs, inputSize)(kernelFn)
  }

  /** Java-friendly overload: tuneRandomSearch with Int ranges (min, max inclusive) */
  def tuneRandomSearch(
    kernelName: String,
    numTrials: Int,
    blockMin: Int, blockMax: Int,
    numWarpsMin: Int, numWarpsMax: Int,
    numStagesMin: Int, numStagesMax: Int,
    inputSize: Int
  )(kernelFn: KernelConfig => Array[Float]): AutotuneResult = {
    val blockRange = (if (blockMin > 0) blockMin else 32) to (if (blockMax > 0) blockMax else 256)
    val warpRange = numWarpsMin to numWarpsMax
    val stageRange = numStagesMin to numStagesMax
    tuneRandomSearch(kernelName, numTrials, blockRange, warpRange, stageRange, inputSize)(kernelFn)
  }

  /** Tune with Bayesian optimization strategy */
  def tuneBayesian(
    kernelName: String,
    numTrials: Int,
    inputSize: Int
  )(kernelFn: KernelConfig => Array[Float]): AutotuneResult = {
    // Simplified Bayesian optimization using Gaussian Process approximation
    val searchSpace = for {
      b <- List(32, 64, 128, 256, 512)
      w <- List(2, 4, 8, 16)
      s <- List(1, 2, 3, 4)
    } yield KernelConfig(blockX = b, numWarps = w, numStages = s)

    // Start with random sampling
    val initialSamples = searchSpace.take(8)
    val results = mutable.Map[KernelConfig, Double]()

    for (config <- initialSamples) {
      val start = System.nanoTime()
      kernelFn(config)
      val latency = (System.nanoTime() - start) / 1e6
      results(config) = latency
    }

    // Iterative refinement (simplified acquisition)
    for (_ <- 0 until (numTrials - initialSamples.size).max(0)) {
      // Pick config with highest uncertainty (furthest from sampled)
      val candidates = searchSpace.filter(!results.contains(_))
      if (candidates.isEmpty) candidates.headOption.foreach { c =>
        val start = System.nanoTime()
        kernelFn(c)
        results(c) = (System.nanoTime() - start) / 1e6
      }
    }

    val best = results.minBy(_._2)
    AutotuneResult(best._1, best._2, inputSize.toDouble / (best._2 / 1000.0))
  }

/** Global autotuner instance */
object Autotuner:
  val engine = new AutotunerEngine

  def tune(
    kernelName: String,
    configs: List[KernelConfig],
    inputSize: Int
  )(kernelFn: KernelConfig => Array[Float]): AutotuneResult = {
    engine.tune(kernelName, configs, inputSize)(kernelFn)
  }

  def getBestConfig(kernelName: String): KernelConfig =
    AutotuneCache.getBestConfig(kernelName)

/** Macro annotation for automatic autotuning
 *  Usage:
 *  @TritonAutotune(configs = List(
 *    KernelConfig(64), KernelConfig(128), KernelConfig(256)
 *  ))
 *  def myKernel = { ... }
 */
class TritonAutotune(configs: List[KernelConfig]) extends scala.annotation.StaticAnnotation

/** Companion object for autotune utilities */
object TritonAutotune:
  /** Generate autotuned kernel source with best config */
  def generateAutotunedSource(
    kernelName: String,
    baseSource: String,
    config: KernelConfig
  ): String = {
    baseSource
      .replace("BLOCK_SIZE", config.blockX.toString)
      .replace("NUM_WARPS", config.numWarps.toString)
      .replace("NUM_STAGES", config.numStages.toString)
  }
