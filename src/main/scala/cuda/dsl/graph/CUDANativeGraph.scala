package cuda.dsl.graph

import cuda.dsl.driver.{NativeCUDAStream, NativeCUDAEvent}
import org.bytedeco.cuda.global.cudart.*
import org.bytedeco.cuda.cudart.{CUgraph_st, CUgraphExec_st, cudaKernelNodeParams, dim3}
import org.bytedeco.javacpp.{IntPointer, SizeTPointer, Pointer}

import scala.collection.mutable

/** CUDA Graph with simplified native API */
class NativeCUDAGpuGraph(
  val graph: CUgraph_st = new CUgraph_st()
) extends AutoCloseable:
  private var created = false
  private val nodes = mutable.ListBuffer[org.bytedeco.cuda.cudart.CUgraphNode_st]()

  def create(): NativeCUDAGpuGraph = {
    if created then return this
    try
      val res = cudaGraphCreate(graph, 0)
      if res == 0 then created = true else ()
    catch case _: Throwable => ()
    this
  }

  def isCreated: Boolean = created

  def addKernelNode(
    func: Pointer,
    gridX: Int, gridY: Int, gridZ: Int,
    blockX: Int, blockY: Int, blockZ: Int,
    sharedMem: Int = 0,
    extra: Pointer = null
  ): Boolean = {
    if !created then return false
    try
      val params = new cudaKernelNodeParams()
      params.func(func)
      params.gridDim(new dim3(gridX, gridY, gridZ))
      params.blockDim(new dim3(blockX, blockY, blockZ))
      params.sharedMemBytes(sharedMem)

      val node = new org.bytedeco.cuda.cudart.CUgraphNode_st()
      val res = cudaGraphAddKernelNode(node, graph, null, 0L, params)
      if res == 0 then
        nodes += node
        true
      else false
    catch case _: Throwable => false
  }

  def addEmptyNode(): Boolean = {
    if !created then return false
    try
      val node = new org.bytedeco.cuda.cudart.CUgraphNode_st()
      val res = cudaGraphAddEmptyNode(node, graph, null, 0L)
      if res == 0 then
        nodes += node
        true
      else false
    catch case _: Throwable => false
  }

  def getNodesCount: Int = {
    if !created then return 0
    try
      val numPtr = new SizeTPointer(1L)
      val res = cudaGraphGetNodes(graph, null, numPtr)
      if res == 0 then
        val n = numPtr.get(0).toInt
        numPtr.close()
        n
      else
        numPtr.close()
        0
    catch case _: Throwable => 0
  }

  def getRootNodesCount: Int = {
    if !created then return 0
    try
      val numPtr = new SizeTPointer(1L)
      val res = cudaGraphGetRootNodes(graph, null, numPtr)
      if res == 0 then
        val n = numPtr.get(0).toInt
        numPtr.close()
        n
      else
        numPtr.close()
        0
    catch case _: Throwable => 0
  }

  def destroy(): Int = {
    if !created then return 0
    try
      nodes.foreach(_ => ())
      nodes.clear()
      val res = cudaGraphDestroy(graph)
      if res == 0 then created = false
      res
    catch case _: Throwable => -1
  }

  override def close(): Unit = { try destroy() catch case _: Throwable => () }

object NativeCUDAGpuGraph:
  def create(): NativeCUDAGpuGraph = new NativeCUDAGpuGraph().create()

/** Graph executor */
class CudaGraphExecutor(
  val exec: CUgraphExec_st = new CUgraphExec_st()
) extends AutoCloseable:
  private var instantiated = false

  def instantiate(graph: NativeCUDAGpuGraph): Boolean = {
    if instantiated then return true
    try
      val res = cudaGraphInstantiate(exec, graph.graph, 0L)
      if res == 0 then { instantiated = true; true } else false
    catch case _: Throwable => false
  }

  def launch(stream: NativeCUDAStream): Boolean = {
    if !instantiated then return false
    try
      val res = cudaGraphLaunch(exec, stream.stream)
      res == 0
    catch case _: Throwable => false
  }

  def launchOnDefaultStream(): Boolean = {
    if !instantiated then return false
    try
      val s = NativeCUDAStream.create()
      val res = cudaGraphLaunch(exec, s.stream)
      s.close()
      res == 0
    catch case _: Throwable => false
  }

  def destroy(): Int = {
    if !instantiated then return 0
    try
      val res = cudaGraphExecDestroy(exec)
      if res == 0 then instantiated = false
      res
    catch case _: Throwable => -1
  }

  override def close(): Unit = { try destroy() catch case _: Throwable => () }

object CudaGraphExecutor:
  def create(): CudaGraphExecutor = new CudaGraphExecutor()

/** CUDA Graph stream capture builder */
class CudaGraphCaptureManager(val stream: NativeCUDAStream) extends AutoCloseable:
  private var capturing = false

  def beginCapture(mode: Int = 0): Boolean = {
    if capturing then return false
    try
      val res = cudaStreamBeginCapture(stream.stream, mode)
      if res == 0 then { capturing = true; true } else false
    catch case _: Throwable => false
  }

  def endCapture(): Option[NativeCUDAGpuGraph] = {
    if !capturing then return None
    try
      val g = new CUgraph_st()
      val res = cudaStreamEndCapture(stream.stream, g)
      if res == 0 then
        capturing = false
        Some(new NativeCUDAGpuGraph(g))
      else None
    catch case _: Throwable => None
  }

  def isCapturing: Boolean = capturing

  override def close(): Unit = { capturing = false }

object CudaGraphCaptureManager:
  def create(mode: Int = 0): Option[CudaGraphCaptureManager] = {
    val s = NativeCUDAStream.create()
    if s.stream == null then return None
    val manager = new CudaGraphCaptureManager(s)
    if manager.beginCapture(mode) then Some(manager) else None
  }

/** High-level Graph API wrapper */
object CudaGraphAPI:
  def profileGraph(graph: NativeCUDAGpuGraph): Map[String, Any] = {
    Map(
      "nodes" -> graph.getNodesCount,
      "roots" -> graph.getRootNodesCount
    )
  }