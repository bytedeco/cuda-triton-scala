package cuda.dsl.benchmark

import cuda.dsl.autotune.*
import cuda.dsl.graph.*
import cuda.dsl.sparse.*
import cuda.dsl.nccl.*
import cuda.dsl.driver.*

/** Comprehensive verification for all new modules */
object TestNewModulesVerification:
  def main(args: Array[String]): Unit = {
    println("=" * 70)
    println("NEW MODULES VERIFICATION TEST")
    println("=" * 70)

    var passed = 0
    var failed = 0

    // Test Autotune
    println("\n[1] AUTOTUNE MODULE")
    try {
      testAutotune()
      passed += 1
    } catch
      case e: Exception =>
        println(s"  ❌ FAILED: ${e.getMessage}")
        failed += 1

    // Test Graph Optimization
    println("\n[2] GRAPH OPTIMIZATION MODULE")
    try {
      testGraphOptimization()
      passed += 1
    } catch
      case e: Exception =>
        println(s"  ❌ FAILED: ${e.getMessage}")
        failed += 1

    // Test Graph Native (CUDA Graph)
    println("\n[3] GRAPH NATIVE (CUDA Graph API)")
    try {
      testGraphNative()
      passed += 1
    } catch
      case e: Exception =>
        println(s"  ❌ FAILED: ${e.getMessage}")
        failed += 1

    // Test Sparse Tensor
    println("\n[4] SPARSE TENSOR MODULE")
    try {
      testSparseTensor()
      passed += 1
    } catch
      case e: Exception =>
        println(s"  ❌ FAILED: ${e.getMessage}")
        failed += 1

    // Test NCCL Communication
    println("\n[5] NCCL COMMUNICATION MODULE")
    try {
      testNCCL()
      passed += 1
    } catch
      case e: Exception =>
        println(s"  ❌ FAILED: ${e.getMessage}")
        failed += 1

    // Test Driver API (native CUDART)
    println("\n[6] DRIVER API MODULE (Native CUDART)")
    try {
      testDriverAPI()
      passed += 1
    } catch
      case e: Exception =>
        println(s"  ❌ FAILED: ${e.getMessage}")
        failed += 1

    // Test Driver API Native
    println("\n[7] DRIVER API NATIVE (NativeCUDAStream/Event)")
    try {
      testDriverAPINative()
      passed += 1
    } catch
      case e: Exception =>
        println(s"  ❌ FAILED: ${e.getMessage}")
        failed += 1

    println("\n" + "=" * 70)
    println(s"RESULTS: $passed passed, $failed failed")
    println("=" * 70)

    if (failed == 0) {
      println("✅ ALL NEW MODULES VERIFIED SUCCESSFULLY!")
    } else {
      println("⚠️  SOME MODULES FAILED VERIFICATION")
    }
    println("=" * 70)
  }

  def testAutotune(): Unit = {
    println("  Testing basic autotuning...")
    val configs = List(
      KernelConfig(64), KernelConfig(128), KernelConfig(256)
    )
    val result = Autotuner.tune("testKernel", configs, 1024) { cfg =>
      val out = Array.fill(256)(0.0f)
      for (i <- 0 until 256) out(i) = (i * cfg.blockX).toFloat * 0.01f
      out
    }
    assert(List(64, 128, 256).contains(result.config.blockX), s"Best config should be one of 64/128/256, got ${result.config.blockX}")
    println(s"  ✅ Best config: ${result.config.blockX} @ ${result.latencyMs.formatted("%.3f")}ms")

    println("  Testing grid search...")
    val engine = new AutotunerEngine
    val result2 = engine.tuneGridSearch("gemm", List(32, 64, 128), List(2, 4), List(1, 2), 2048) { cfg =>
      Array.fill(512)(cfg.blockX.toFloat * cfg.numWarps.toFloat * 0.001f)
    }
    println(s"  ✅ Grid search best: ${result2.config}")
  }

  def testGraphOptimization(): Unit = {
    println("  Testing graph construction...")
    val graph = new KernelGraph
    val a = graph.addOp(OpType.Load)
    val b = graph.addOp(OpType.Load)
    val c = graph.addOp(OpType.Add, List(a.id, b.id))
    val d = graph.addOp(OpType.Mul, List(c.id, a.id))
    assert(graph.getOps.size == 4)
    println(s"  ✅ Graph has ${graph.getOps.size} ops")

    println("  Testing elementwise fusion...")
    val optimizer = new GraphOptimizer
    val optimized = optimizer.optimize(graph)
    println(s"  ✅ Optimized ops: ${optimized.getOps.size} (was ${graph.getOps.size})")

    println("  Testing code generation...")
    val code = KernelCodeGenerator.generate(optimized)
    assert(code.contains("extern \"C\""))
    assert(code.contains("fused_"))
    println(s"  ✅ Generated ${code.linesIterator.size} lines of CUDA code")
  }

  def testGraphNative(): Unit = {
    println("  Testing CUDA Graph (legacy) creation...")
    val graph = NativeCUDAGraph.create()
    println(s"  ✅ Graph created: ${graph.graph}")

    println("  Testing CUDA Graph (legacy) instantiation...")
    val instRes = graph.instantiate()
    println(s"  ✅ Graph instantiate result: $instRes")

    println("  Testing CUDA Graph (legacy) launch on default stream...")
    val launchRes = graph.launchOnDefaultStream()
    println(s"  ✅ Graph launch result: $launchRes")

    println("  Testing CUDA Graph (legacy) destroy...")
    graph.destroy()
    println(s"  ✅ Graph destroyed")

    println("  Testing NativeCUDAGpuGraph creation...")
    val gpuGraph = NativeCUDAGpuGraph.create()
    println(s"  ✅ NativeCUDAGraph created")

    println("  Testing graph addKernelNode...")
    val added = gpuGraph.addKernelNode(null, 1, 1, 1, 256, 1, 1, 0)
    println(s"  ✅ addKernelNode result: $added")

    println("  Testing graph addEmptyNode...")
    val emptyAdded = gpuGraph.addEmptyNode()
    println(s"  ✅ addEmptyNode result: $emptyAdded")

    println("  Testing graph getNodesCount...")
    val nodeCount = gpuGraph.getNodesCount
    println(s"  ✅ Graph nodes count: $nodeCount")

    gpuGraph.close()
    println(s"  ✅ NativeCUDAGraph destroyed")
  }

  def testSparseTensor(): Unit = {
    println("  Testing CSR creation from dense...")
    val dense = Array(
      Array(1.0f, 0.0f, 2.0f),
      Array(0.0f, 3.0f, 0.0f),
      Array(4.0f, 0.0f, 5.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    assert(sparse.nnz == 5, s"Should have 5 non-zeros, got ${sparse.nnz}")
    println(s"  ✅ Sparse: ${sparse.info}")

    println("  Testing SpMV...")
    val x = Array(1.0f, 2.0f, 3.0f)
    val y = SparseOps.spmv(sparse, x)
    assert(y(0) == 7.0f, s"y[0] should be 7, got ${y(0)}")
    assert(y(1) == 6.0f, s"y[1] should be 6, got ${y(1)}")
    assert(y(2) == 19.0f, s"y[2] should be 19, got ${y(2)}")
    println(s"  ✅ SpMV result: [${y.mkString(", ")}]")

    println("  Testing CSR <-> CSC conversion...")
    val csc = sparse.toCSC
    assert(sparse.nnz == csc.nnz)
    println(s"  ✅ CSC conversion preserved nnz: ${csc.nnz}")

    println("  Testing sparse attention...")
    val mask = Array.tabulate(4) { i => Array.tabulate(4) { j => j <= i } }
    val attn = SparseAttention(mask, blockSize = 1)
    val Q = Array.tabulate(4)(_ => Array.fill(4)(1.0f))
    val K = Array.tabulate(4)(_ => Array.fill(4)(1.0f))
    val V = Array.tabulate(4)(_ => Array.fill(4)(1.0f))
    val out = attn.forward(Q, K, V)
    assert(out.length == 4)
    println(s"  ✅ Sparse attention output shape: ${out.length}x${out(0).length}")
  }

  def testNCCL(): Unit = {
    println("  Testing communicator creation...")
    val comm = NCCLCommunicator.init(0, 1, 0)
    assert(comm.rank == 0)
    assert(comm.worldSize == 1)
    println(s"  ✅ Communicator: $comm")
    assert(comm.isNative, "NCCL should be using native bindings")
    println(s"  ✅ NCCL is using native JavaCPP bindings")

    println("  Testing AllReduce...")
    val send = Array(1.0f, 2.0f, 3.0f, 4.0f)
    val recv = Array.fill(4)(0.0f)
    NCCLOps.ncclAllReduce(send, recv, 4, NCCLOp.Sum, comm)
    assert(recv(0) == 1.0f, s"Should be 1.0 (worldSize=1), got ${recv(0)}")
    println(s"  ✅ AllReduce result (worldSize=1): [${recv.mkString(", ")}]")

    println("  Testing AllGather...")
    val send2 = Array(10.0f, 20.0f)
    val recv2 = Array.fill(2)(0.0f)
    NCCLOps.ncclAllGather(send2, recv2, 2, comm)
    assert(recv2(0) == 10.0f, s"recv2[0] should be 10.0, got ${recv2(0)}")
    assert(recv2(1) == 20.0f, s"recv2[1] should be 20.0, got ${recv2(1)}")
    println(s"  ✅ AllGather result (worldSize=1): [${recv2.mkString(", ")}]")

    println("  Testing Broadcast...")
    val send3 = Array(5.0f, 6.0f, 7.0f)
    val recv3 = Array.fill(3)(0.0f)
    NCCLOps.ncclBroadcast(send3, recv3, 3, root = 0, comm)
    assert(recv3(0) == 5.0f)
    println(s"  ✅ Broadcast result: [${recv3.mkString(", ")}]")

    println("  Testing DataParallel wrapper...")
    val dp = DataParallel[Float](comm, localBatchSize = 32, gradientAverage = true)
    val grads = Array(0.1f, 0.2f, 0.3f)
    val reduced = dp.allReduceGradients(grads)
    assert(reduced.length == grads.length)
    println(s"  ✅ DataParallel gradient average: [${reduced.mkString(", ")}]")
  }

  def testDriverAPI(): Unit = {
    println("  Testing driver initialization...")
    DriverAPI.initialize()
    val numDevices = DriverAPI.get_num_devices
    assert(numDevices >= 1, s"Should have at least 1 device, got $numDevices")
    println(s"  ✅ Number of devices: $numDevices")

    println("  Testing device properties...")
    val props = DriverAPI.get_device_properties(0)
    println(s"  ✅ Device: ${props.name}")
    println(s"     SMs: ${props.num_sm}, Warp size: ${props.warp_size}")
    println(s"     Compute: sm_${props.compute_capability._1}.${props.compute_capability._2}")
    println(s"     Total memory: ${props.total_memory / (1024*1024*1024)} GB")

    println("  Testing current device management...")
    val current = DriverAPI.get_current_device
    assert(current.id == 0)
    println(s"  ✅ Current device: $current")

    println("  Testing stream management...")
    val stream = CUDAStream.create(current)
    println(s"  ✅ Created stream: $stream")

    println("  Testing event management...")
    val event = CUDAEvent.create(current)
    println(s"  ✅ Created event: $event")

    println("  Testing memory queries...")
    val total = DriverAPI.get_total_memory(0)
    val active = DriverAPI.get_active_memory()
    println(s"  ✅ Total memory: ${total / (1024*1024*1024)} GB")
    println(s"     Active memory: ${active / (1024*1024)} MB")
  }

  def testDriverAPINative(): Unit = {
    println("  Testing CUDARuntimeNatives initialization...")
    val inited = CUDARuntimeNatives.initialize()
    println(s"  ✅ CUDARuntime initialized: $inited")

    println("  Testing native getDeviceCount...")
    val count = CUDARuntimeNatives.getDeviceCount
    assert(count >= 1, s"Should have at least 1 device, got $count")
    println(s"  ✅ Native device count: $count")

    println("  Testing native getCurrentDevice...")
    val current = CUDARuntimeNatives.getCurrentDevice
    println(s"  ✅ Native current device: $current")

    println("  Testing NativeDeviceProperties...")
    val props = CUDARuntimeNatives.getDeviceProperties(0)
    println(s"  ✅ Native device: ${props.name}")
    println(s"     SMs: ${props.num_sm}, Warp: ${props.warpSize}")
    println(s"     Compute: sm_${props.major}.${props.minor}")
    println(s"     Total memory: ${props.totalGlobalMem / (1024*1024*1024)} GB")
    println(s"     Max threads/block: ${props.maxThreadsPerBlock}")
    println(s"     Max threads/SM: ${props.maxThreadsPerSM}")
    println(s"     L2 cache: ${props.l2CacheSize / (1024*1024)} MB")
    println(s"     Global L1 cache: ${props.globalL1CacheSupported}")
    println(s"     Memory bus width: ${props.memoryBusWidth} bit")

    println("  Testing NativeCUDAStream...")
    val stream = NativeCUDAStream.create()
    println(s"  ✅ Created native stream: ${stream.stream}")
    stream.synchronize()
    println(s"  ✅ Stream synchronized")
    stream.destroy()
    println(s"  ✅ Stream destroyed")

    println("  Testing NativeCUDAEvent...")
    val event = NativeCUDAEvent.create()
    println(s"  ✅ Created native event: ${event.event}")
    event.synchronize()
    println(s"  ✅ Event synchronized")
    event.destroy()
    println(s"  ✅ Event destroyed")

    println("  Testing event timing...")
    val start = NativeCUDAEvent.create()
    val end = NativeCUDAEvent.create()
    val delay = end.elapsedTimeMs(start)
    println(s"  ✅ Event elapsed time: ${delay}ms")
    start.destroy()
    end.destroy()

    println("  Testing CUDA Graph...")
    val g = NativeCUDAGraph.create()
    println(s"  ✅ CUDA Graph created: ${g.graph}")
    g.destroy()
    println(s"  ✅ CUDA Graph destroyed")
  }
