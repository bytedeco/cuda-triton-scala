package cuda.dsl.autotune

object AutotuneJavaBridge:

  def createKernelConfig(blockX: Int, numWarps: Int, numStages: Int): KernelConfig =
    KernelConfig(blockX = blockX, numWarps = numWarps, numStages = numStages)

  def createKernelConfig1(blockX: Int): KernelConfig =
    KernelConfig(blockX = blockX)

  def getKernelConfigBlockX(cfg: KernelConfig): Int = cfg.blockX

  def getKernelConfigNumWarps(cfg: KernelConfig): Int = cfg.numWarps

  def getKernelConfigNumStages(cfg: KernelConfig): Int = cfg.numStages

  def getKernelConfigToString(cfg: KernelConfig): String = cfg.toString

  def tuneKernel(name: String, blockSizes: Array[Int], inputSize: Int): KernelConfig =
    if blockSizes == null || blockSizes.isEmpty then KernelConfig()
    else
      val configs = blockSizes.map(b => KernelConfig(blockX = b)).toList
      val engine = new AutotunerEngine()
      val result = engine.tune(name, configs, inputSize) { cfg =>
        val arr = new Array[Float](256)
        var i = 0
        while i < 256 do { arr(i) = (i * cfg.blockX * 0.01).toFloat; i += 1 }
        arr
      }
      result.config

  def tuneGridSearch(name: String, blockSizes: Array[Int], numWarps: Array[Int],
      numStages: Array[Int], inputSize: Int): KernelConfig =
    val blocks = if blockSizes != null then blockSizes.toList else List(64, 128, 256)
    val warps = if numWarps != null then numWarps.toList else List(2, 4)
    val stages = if numStages != null then numStages.toList else List(1, 2)
    val engine = new AutotunerEngine()
    val result = engine.tuneGridSearch(name, blocks, warps, stages, inputSize) { cfg =>
      val arr = new Array[Float](512)
      var i = 0
      while i < 512 do { arr(i) = (cfg.blockX * cfg.numWarps * 0.001).toFloat; i += 1 }
      arr
    }
    result.config

  def getBestConfig(name: String): KernelConfig =
    Autotuner.getBestConfig(name)

  def tuneKernelWithArray(name: String, blockSizes: Array[Int], inputSize: Int,
      fn: KernelConfig => Array[Float]): KernelConfig =
    if blockSizes == null || blockSizes.isEmpty then KernelConfig()
    else
      val configs = blockSizes.map(b => KernelConfig(blockX = b)).toList
      val engine = new AutotunerEngine()
      val result = engine.tune(name, configs, inputSize)(fn)
      result.config

  def tuneGridSearchWithArray(name: String, blockSizes: Array[Int], numWarps: Array[Int],
      numStages: Array[Int], inputSize: Int,
      fn: KernelConfig => Array[Float]): KernelConfig =
    val blocks = if blockSizes != null then blockSizes.toList else List(64, 128, 256)
    val warps = if numWarps != null then numWarps.toList else List(2, 4)
    val stages = if numStages != null then numStages.toList else List(1, 2)
    val engine = new AutotunerEngine()
    val result = engine.tuneGridSearch(name, blocks, warps, stages, inputSize)(fn)
    result.config

  def tuneKernelGrid(name: String, blockX1: Int, blockX2: Int, blockX3: Int, inputSize: Int): KernelConfig =
    val configs = List(
      if blockX1 > 0 then KernelConfig(blockX = blockX1) else KernelConfig(),
      if blockX2 > 0 then KernelConfig(blockX = blockX2) else KernelConfig(blockX = 128),
      if blockX3 > 0 then KernelConfig(blockX = blockX3) else KernelConfig(blockX = 256)
    ).distinct
    val engine = new AutotunerEngine()
    val result = engine.tune(name, configs, inputSize) { cfg =>
      val arr = new Array[Float](256)
      var i = 0
      while i < 256 do { arr(i) = (i * cfg.blockX * 0.01).toFloat; i += 1 }
      arr
    }
    result.config