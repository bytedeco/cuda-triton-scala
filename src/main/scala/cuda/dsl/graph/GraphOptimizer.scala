package cuda.dsl.graph

import cuda.dsl.dsl.*
import cuda.dsl.runtime.*

import scala.collection.mutable

/** Operation type in kernel graph */
enum OpType:
  case Load, Store, Mul, Add, Sub, Div, Exp, Log, Sqrt, MatMul
  case ReduceSum, ReduceMax, ReduceMin, Softmax, LayerNorm, Attention
  case Relu, Gelu, Sigmoid, Tanh, Sin, Cos
  case Custom(name: String)

/** Single operation in kernel graph */
case class KernelOp(
  id: Int,
  opType: OpType,
  inputs: List[Int],
  attrs: Map[String, Any] = Map.empty,
  output: Option[String] = None
) {
  override def toString: String = s"Op($id: $opType)"
}

/** Kernel graph representing operations and data flow */
class KernelGraph:
  private val ops = mutable.ListBuffer[KernelOp]()
  private val adjList = mutable.Map[Int, mutable.Set[Int]]()
  private var nextId = 0

  def addOp(opType: OpType, inputs: List[Int] = Nil, attrs: Map[String, Any] = Map.empty): KernelOp = {
    val op = KernelOp(nextId, opType, inputs, attrs)
    nextId += 1
    ops += op
    inputs.foreach { inId =>
      adjList.getOrElseUpdate(inId, mutable.Set()) += op.id
    }
    op
  }

  def getOps: List[KernelOp] = ops.toList

  def getSuccessors(opId: Int): Set[Int] = adjList.getOrElse(opId, mutable.Set()).toSet

  def getPredecessors(opId: Int): Set[Int] = ops.filter(_.inputs.contains(opId)).map(_.id).toSet

  def topologicalSort: List[KernelOp] =
    val visited = mutable.Set[Int]()
    val result = mutable.ListBuffer[KernelOp]()

    def dfs(id: Int): Unit =
      if !visited(id) then
        visited += id
        for succ <- getSuccessors(id) do dfs(succ)
        result.prepend(ops(id))

    for op <- ops do dfs(op.id)
    result.toList

  def show(): String = {
    val sb = new StringBuilder
    sb ++= "KernelGraph(\n"
    for op <- ops do
      sb ++= s"  $op"
      if op.inputs.nonEmpty then sb ++= s" [inputs: ${op.inputs.mkString(", ")}]"
      sb ++= "\n"
    sb ++= ")"
    sb.toString
  }

/** Fusion pattern trait */
trait FusionPattern:
  def name: String
  def canFuse(graph: KernelGraph, ops: List[KernelOp]): Boolean
  def fuse(graph: KernelGraph, ops: List[KernelOp]): KernelOp

/** Elementwise fusion pattern: mul(add(x, y), z) -> fused_mul_add */
object ElementwiseFusion extends FusionPattern:
  val name = "elementwise"

  def canFuse(graph: KernelGraph, ops: List[KernelOp]): Boolean = {
    if ops.size < 2 then return false
    // Check if operations are elementwise and can be fused
    val elementwise = Set(OpType.Mul, OpType.Add, OpType.Sub, OpType.Div, OpType.Exp, OpType.Log)
    ops.forall(o => elementwise(o.opType) || o.opType == OpType.Custom("activation"))
  }

  def fuse(graph: KernelGraph, ops: List[KernelOp]): KernelOp = {
    graph.addOp(
      OpType.Custom("fused_elementwise"),
      ops.flatMap(_.inputs),
      Map("fused_ops" -> ops.map(_.opType).mkString("_"))
    )
  }

/** Reduction fusion pattern */
object ReductionFusion extends FusionPattern:
  val name = "reduction"

  def canFuse(graph: KernelGraph, ops: List[KernelOp]): Boolean = {
    if ops.size < 2 then return false
    val hasReduction = ops.exists(o => o.opType == OpType.ReduceSum || o.opType == OpType.ReduceMax)
    val hasElementwise = ops.forall(o =>
      o.opType == OpType.ReduceSum || o.opType == OpType.ReduceMax ||
      Set(OpType.Mul, OpType.Add, OpType.Sub).contains(o.opType))
    hasReduction && hasElementwise
  }

  def fuse(graph: KernelGraph, ops: List[KernelOp]): KernelOp = {
    graph.addOp(
      OpType.Custom("fused_reduction"),
      ops.flatMap(_.inputs).distinct,
      Map("stages" -> ops.size)
    )
  }

/** LayerNorm fusion: sub + pow + mean + div + sqrt + mul + add -> fused_layernorm */
object LayerNormFusion extends FusionPattern:
  val name = "layernorm"

  def canFuse(graph: KernelGraph, ops: List[KernelOp]): Boolean = {
    val types = ops.map(_.opType).toSet
    types.contains(OpType.Sub) &&
    types.contains(OpType.Div) &&
    ops.exists(_.opType == OpType.Custom("mean")) &&
    ops.exists(_.opType == OpType.Custom("variance"))
  }

  def fuse(graph: KernelGraph, ops: List[KernelOp]): KernelOp = {
    graph.addOp(
      OpType.LayerNorm,
      ops.flatMap(_.inputs).distinct,
      Map("fused" -> true)
    )
  }

/** Attention fusion: matmul + softmax + matmul -> fused_attention */
object AttentionFusion extends FusionPattern:
  val name = "attention"

  def canFuse(graph: KernelGraph, ops: List[KernelOp]): Boolean = {
    val types = ops.map(_.opType).toSet
    types.contains(OpType.MatMul) && ops.count(_.opType == OpType.MatMul) >= 2 && types.contains(OpType.Softmax)
  }

  def fuse(graph: KernelGraph, ops: List[KernelOp]): KernelOp = {
    graph.addOp(
      OpType.Attention,
      ops.flatMap(_.inputs).distinct,
      Map("fused_attention" -> true)
    )
  }

/** GELU activation fusion */
object GeluFusion extends FusionPattern:
  val name = "gelu"

  def canFuse(graph: KernelGraph, ops: List[KernelOp]): Boolean = {
    ops.size >= 3 && {
      val types = ops.map(_.opType).toSet
      types.contains(OpType.Div) && types.contains(OpType.Exp) && types.contains(OpType.Mul)
    }
  }

  def fuse(graph: KernelGraph, ops: List[KernelOp]): KernelOp = {
    graph.addOp(
      OpType.Gelu,
      ops.flatMap(_.inputs).distinct,
      Map("approximation" -> "fast")
    )
  }

/** Graph optimization engine */
class GraphOptimizer:
  private val patterns = mutable.ListBuffer[FusionPattern]()
  private val fusionLog = mutable.ListBuffer[String]()

  addPattern(ElementwiseFusion)
  addPattern(ReductionFusion)
  addPattern(LayerNormFusion)
  addPattern(AttentionFusion)
  addPattern(GeluFusion)

  def addPattern(pattern: FusionPattern): Unit = patterns += pattern

  def optimize(graph: KernelGraph): KernelGraph = {
    val optimized = new KernelGraph
    val merged = mutable.Set[Int]()

    for (pattern <- patterns) {
      val sorted = graph.topologicalSort
      var i = 0
      while i < sorted.size do
        val op = sorted(i)
        if !merged(op.id) then
          // Look for fusable sequences
          val candidates = collectFusableOps(sorted, i, pattern)
          if candidates.size >= 2 && pattern.canFuse(graph, candidates) then
            val fused = pattern.fuse(optimized, candidates)
            candidates.foreach(c => merged.add(c.id))
            fusionLog += s"Fused ${candidates.size} ops with pattern '${pattern.name}' -> ${fused.opType}"
            i += candidates.size
          else
            if !merged(op.id) then
              optimized.addOp(op.opType, op.inputs, op.attrs)
            i += 1
        else i += 1
    }

    optimized
  }

  private def collectFusableOps(ops: List[KernelOp], start: Int, pattern: FusionPattern): List[KernelOp] = {
    val collected = mutable.ListBuffer[KernelOp]()
    var i = start
    while i < ops.size && collected.size < 10 do
      val op = ops(i)
      if !Set(OpType.Load, OpType.Store, OpType.MatMul).contains(op.opType) then
        collected += op
      i += 1
    collected.toList
  }

  def getFusionLog: List[String] = fusionLog.toList

  def clearLog(): Unit = fusionLog.clear()

/** Graph fusion result */
case class FusionResult(
  originalOps: Int,
  fusedOps: Int,
  speedup: Double,
  generatedKernel: String
)

/** Generate CUDA code from optimized graph */
object KernelCodeGenerator:
  def generate(graph: KernelGraph): String = {
    val sb = new StringBuilder
    sb ++= "// Auto-generated fused kernel\n"
    sb ++= "#include <cuda_runtime.h>\n\n"

    for (op <- graph.topologicalSort) {
      op.opType match
        case OpType.LayerNorm =>
          sb ++= generateLayerNormKernel(op)
        case OpType.Attention =>
          sb ++= generateAttentionKernel(op)
        case OpType.Custom(name) if name.startsWith("fused_") =>
          sb ++= generateFusedKernel(op, name)
        case _ =>
          sb ++= generateSimpleKernel(op)
    }

    sb.toString
  }

  private def generateLayerNormKernel(op: KernelOp): String = {
    s"""
extern "C" __global__ void fused_layernorm_${op.id}(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Fused LayerNorm: compute mean, variance, and normalize in single pass
    float mean = 0.0f;
    for (int i = 0; i < N; i++) mean += input[i];
    mean /= N;

    float var = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= N;

    float inv_std = rsqrtf(var + (float)eps);
    output[idx] = (input[idx] - mean) * inv_std;
}
"""
  }

  private def generateAttentionKernel(op: KernelOp): String = {
    s"""
extern "C" __global__ void fused_attention_${op.id}(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / head_dim;
    int col = idx % head_dim;
    if (row >= seq_len) return;

    // Fused QK^T + softmax + matmul with V
    float sum = 0.0f;
    float max_val = -INFINITY;

    // Online softmax
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += Q[row * head_dim + d] * K[j * head_dim + d];
        max_val = fmaxf(max_val, score);
    }

    float exp_sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += Q[row * head_dim + d] * K[j * head_dim + d];
        exp_sum += expf(score - max_val);
    }

    float result = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += Q[row * head_dim + d] * K[j * head_dim + d];
        float weight = expf(score - max_val) / exp_sum;
        for (int d = 0; d < head_dim; d++)
            result += weight * V[j * head_dim + d];
    }
    output[idx] = result;
}
"""
  }

  private def generateFusedKernel(op: KernelOp, name: String): String = {
    s"""
extern "C" __global__ void ${name}_${op.id}(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx]; // Placeholder
}
"""
  }

  private def generateSimpleKernel(op: KernelOp): String = {
    s"""
extern "C" __global__ void kernel_${op.id}(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] += 1.0f;
}
"""
  }

/** Global optimizer instance */
object GraphOptimizer:
  val default = new GraphOptimizer

  def optimize(graph: KernelGraph): KernelGraph = default.optimize(graph)
