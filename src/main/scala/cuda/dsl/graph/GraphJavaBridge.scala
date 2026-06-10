package cuda.dsl.graph

import cuda.dsl.driver.NativeCUDAStream

/**
 * Java-friendly bridge for Graph operations.
 */
object GraphJavaBridge:

  def createGpuGraph(): NativeCUDAGpuGraph = NativeCUDAGpuGraph.create()

  def createGraphExecutor(): CudaGraphExecutor = CudaGraphExecutor.create()

  def createKernelGraph(): KernelGraph = new KernelGraph()

  def optimizeGraph(g: KernelGraph): KernelGraph = GraphOptimizer.optimize(g)

  def gpuGraphAddKernelNode(g: NativeCUDAGpuGraph, gridX: Int, gridY: Int, gridZ: Int,
      blockX: Int, blockY: Int, blockZ: Int): Boolean =
    g.addKernelNode(null, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, null)

  def gpuGraphAddEmptyNode(g: NativeCUDAGpuGraph): Boolean =
    g.addEmptyNode()

  def gpuGraphGetNodesCount(g: NativeCUDAGpuGraph): Int =
    g.getNodesCount

  def gpuGraphGetRootNodesCount(g: NativeCUDAGpuGraph): Int =
    g.getRootNodesCount

  def gpuGraphDestroy(g: NativeCUDAGpuGraph): Unit =
    g.close()

  def executorInstantiate(exec: CudaGraphExecutor, graph: NativeCUDAGpuGraph): Boolean =
    exec.instantiate(graph)

  def executorLaunchDefault(exec: CudaGraphExecutor): Boolean =
    exec.launchOnDefaultStream()

  def executorDestroy(exec: CudaGraphExecutor): Unit =
    exec.close()

  def kernelGraphAddOp(kg: KernelGraph, opTypeName: String, inputIds: Array[Int]): Int = {
    val opt = opTypeName match
      case "Load" => OpType.Load
      case "Store" => OpType.Store
      case "Add" => OpType.Add
      case "Mul" => OpType.Mul
      case "Sub" => OpType.Sub
      case "Div" => OpType.Div
      case "Exp" => OpType.Exp
      case "Log" => OpType.Log
      case "Sqrt" => OpType.Sqrt
      case "MatMul" => OpType.MatMul
      case "ReduceSum" => OpType.ReduceSum
      case "ReduceMax" => OpType.ReduceMax
      case "ReduceMin" => OpType.ReduceMin
      case "Softmax" => OpType.Softmax
      case "LayerNorm" => OpType.LayerNorm
      case "Attention" => OpType.Attention
      case "Relu" => OpType.Relu
      case "Gelu" => OpType.Gelu
      case _ => OpType.Load
    val inputs = if inputIds != null then scala.collection.immutable.List.from(inputIds.map(_.toInt)) else scala.collection.immutable.List.empty
    val op = kg.addOp(opt, inputs, scala.collection.immutable.Map.empty)
    op.id
  }

  def kernelGraphGetOpsCount(kg: KernelGraph): Int =
    kg.getOps.size

  def kernelGraphTopoSort(kg: KernelGraph): Array[Int] =
    kg.topologicalSort.map(_.id).toArray
