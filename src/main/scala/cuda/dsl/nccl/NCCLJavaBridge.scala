package cuda.dsl.nccl

/**
 * Java-friendly bridge for NCCL operations.
 * Provides static methods with String op parameters for Java interop.
 */
object NCCLJavaBridge:

  def allReduce(send: Array[Float], recv: Array[Float], count: Int, op: String, comm: NCCLCommunicator): Unit = {
    val ncclOp = op match
      case "Prod" => NCCLOp.Prod
      case "Min" => NCCLOp.Min
      case "Max" => NCCLOp.Max
      case "Avg" => NCCLOp.Avg
      case _ => NCCLOp.Sum
    NCCLOps.ncclAllReduce(send, recv, count, ncclOp, comm)
  }

  def broadcast(send: Array[Float], recv: Array[Float], count: Int, root: Int, comm: NCCLCommunicator): Unit =
    NCCLOps.ncclBroadcast(send, recv, count, root, comm)

  def allGather(send: Array[Float], recv: Array[Float], sendCount: Int, comm: NCCLCommunicator): Unit =
    NCCLOps.ncclAllGather(send, recv, sendCount, comm)

  def reduce(send: Array[Float], recv: Array[Float], count: Int, op: String, root: Int, comm: NCCLCommunicator): Unit = {
    val ncclOp = op match
      case "Prod" => NCCLOp.Prod
      case "Min" => NCCLOp.Min
      case "Max" => NCCLOp.Max
      case "Avg" => NCCLOp.Avg
      case _ => NCCLOp.Sum
    NCCLOps.ncclReduce(send, recv, count, ncclOp, root, comm)
  }

  def gather(send: Array[Float], recv: Array[Float], count: Int, root: Int, comm: NCCLCommunicator): Unit =
    NCCLOps.ncclGather(send, recv, count, root, comm)

  def scatter(send: Array[Float], recv: Array[Float], count: Int, root: Int, comm: NCCLCommunicator): Unit =
    NCCLOps.ncclScatter(send, recv, count, root, comm)

  def barrier(comm: NCCLCommunicator): Unit =
    NCCLOps.ncclBarrier(comm)
