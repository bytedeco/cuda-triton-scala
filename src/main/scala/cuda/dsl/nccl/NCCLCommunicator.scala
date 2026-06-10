package cuda.dsl.nccl

import cuda.dsl.runtime.*

import scala.collection.mutable
import org.bytedeco.cuda.global.nccl.*
import org.bytedeco.cuda.cudart.CUstream_st
import org.bytedeco.javacpp.{BytePointer, FloatPointer, Pointer}
import cuda.dsl.driver.{CUDAStream, DriverAPI}
import org.bytedeco.cuda.nccl.{ncclComm, ncclUniqueId}

/** NCCL operation types */
enum NCCLOp:
  case Sum, Prod, Min, Max, Avg

/** NCCL data types */
enum NCCLDtype:
  case Float32, Float16, Float64, Int32, Int64, Char

/** Internal native NCCL communicator handle */
private class NCNativeComm(
  val commPtr: Long,
  val comm: ncclComm,
  val uniqueId: ncclUniqueId,
  val rank: Int,
  val worldSize: Int
) {
  def destroy(): Unit = {
    try ncclCommDestroy(comm)
    catch case _: Throwable => ()
  }
}

/** NCCL communicator for multi-GPU communication */
class NCCLCommunicator(
  val rank: Int,
  val worldSize: Int,
  val localRank: Int = 0,
  val deviceId: Int = 0
) {
  private var nativeComm: Option[NCNativeComm] = None
  private val stream: CUstream_st = new CUstream_st()

  try {
    val uid = new ncclUniqueId()
    val res = ncclGetUniqueId(uid)
    if (res == 0) {
      val comm = new ncclComm()
      val initRes = ncclCommInitRank(comm, worldSize, uid, rank)
      if (initRes == 0) {
        nativeComm = Some(new NCNativeComm(comm.address(), comm, uid, rank, worldSize))
      }
    }
  } catch {
    case _: Throwable => ()
  }

  def isNative: Boolean = nativeComm.isDefined

  def barrier(): Unit = {
    if (worldSize > 1) CUDARuntime.synchronize
  }

  def getRank: Int = rank
  def getWorldSize: Int = worldSize
  def getLocalRank: Int = localRank

  private[ nccl ] def nativeHandle: NCNativeComm = nativeComm.orNull

  override def finalize(): Unit = {
    nativeComm.foreach(_.destroy())
    super.finalize()
  }

  override def toString: String =
    s"NCCLComm(rank=$rank, world=$worldSize, local=$localRank, native=$isNative)"
}

object NCCLCommunicator:
  private val comms = mutable.Map[Int, NCCLCommunicator]()

  def init(rank: Int, worldSize: Int, localRank: Int = 0): NCCLCommunicator = {
    val key = rank * 1000 + worldSize
    comms.getOrElseUpdate(key, new NCCLCommunicator(rank, worldSize, localRank, localRank))
  }

  def getDefault: NCCLCommunicator = init(0, 1, 0)
  def shutdown(): Unit = {
    comms.values.foreach(c => try c.finalize() catch case _: Throwable => ())
    comms.clear()
  }
  def getAvailableDevices: Int = try { CUDARuntime.getDeviceCount } catch { case _ => 1 }

private[ nccl ] object NCCLNatives:

  def toNcclOp(op: NCCLOp): Int = op match
    case NCCLOp.Sum  => ncclSum
    case NCCLOp.Prod => ncclProd
    case NCCLOp.Min  => ncclMin
    case NCCLOp.Max  => ncclMax
    case NCCLOp.Avg  => ncclAvg

  def toNcclDtype(dtype: NCCLDtype): Int = dtype match
    case NCCLDtype.Float32 => ncclFloat32
    case NCCLDtype.Float16 => ncclFloat16
    case NCCLDtype.Float64 => ncclFloat64
    case NCCLDtype.Int32   => ncclInt32
    case NCCLDtype.Int64   => ncclInt64
    case NCCLDtype.Char    => ncclChar

  def toNcclDtypeFloat(dtype: NCCLDtype): Int = toNcclDtype(dtype)

  def arrayToNativePtr(arr: Array[Float]): FloatPointer = {
    val ptr = new FloatPointer(arr.length.toLong)
    ptr.put(arr*)
    ptr
  }

  def nativePtrToArray(ptr: FloatPointer, cnt: Int, out: Array[Float]): Unit = {
    for (i <- 0 until cnt) out(i) = ptr.get(i)
  }

  def allReduceNative(
    send: Array[Float], recv: Array[Float],
    cnt: Int, op: Int, dtype: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = arrayToNativePtr(send)
    val recvPtr = new FloatPointer(cnt.toLong)
    val res = ncclAllReduce(sendPtr, recvPtr, cnt, dtype, op, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, cnt, recv)
    sendPtr.close()
    recvPtr.close()
    res
  }

  def broadcastNative(
    send: Array[Float], recv: Array[Float],
    cnt: Int, dtype: Int, root: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = arrayToNativePtr(send)
    val recvPtr = new FloatPointer(cnt.toLong)
    val res = ncclBroadcast(sendPtr, recvPtr, cnt, dtype, root, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, cnt, recv)
    sendPtr.close()
    recvPtr.close()
    res
  }

  def allGatherNative(
    send: Array[Float], recv: Array[Float],
    sendcnt: Int, dtype: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = arrayToNativePtr(send)
    val recvPtr = new FloatPointer(recv.length.toLong)
    val res = ncclAllGather(sendPtr, recvPtr, sendcnt, dtype, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, recv.length, recv)
    sendPtr.close()
    recvPtr.close()
    res
  }

  def reduceScatterNative(
    send: Array[Float], recv: Array[Float],
    recvcnt: Int, op: Int, dtype: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = new FloatPointer(send.length.toLong)
    sendPtr.put(send*)
    val recvPtr = new FloatPointer(recvcnt.toLong)
    val res = ncclReduceScatter(sendPtr, recvPtr, recvcnt, dtype, op, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, recvcnt, recv)
    sendPtr.close()
    recvPtr.close()
    res
  }

  def allToAllNative(
    send: Array[Float], recv: Array[Float],
    cnt: Int, dtype: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = arrayToNativePtr(send)
    val recvPtr = new FloatPointer(recv.length.toLong)
    val res = ncclAlltoAll(sendPtr, recvPtr, cnt, dtype, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, recv.length, recv)
    sendPtr.close()
    recvPtr.close()
    res
  }

  def sendNative(
    send: Array[Float], cnt: Int, dtype: Int, peer: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = arrayToNativePtr(send)
    val res = ncclSend(sendPtr, cnt, dtype, peer, comm.comm, stream)
    sendPtr.close()
    res
  }

  def recvNative(
    recv: Array[Float], cnt: Int, dtype: Int, peer: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val recvPtr = new FloatPointer(cnt.toLong)
    val res = ncclRecv(recvPtr, cnt, dtype, peer, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, cnt, recv)
    recvPtr.close()
    res
  }

  def reduceNative(
    send: Array[Float], recv: Array[Float],
    cnt: Int, op: Int, dtype: Int, root: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = arrayToNativePtr(send)
    val recvPtr = new FloatPointer(cnt.toLong)
    val res = ncclReduce(sendPtr, recvPtr, cnt, dtype, op, root, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, cnt, recv)
    sendPtr.close()
    recvPtr.close()
    res
  }

  def gatherNative(
    send: Array[Float], recv: Array[Float],
    cnt: Int, dtype: Int, root: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = arrayToNativePtr(send)
    val recvPtr = new FloatPointer(recv.length.toLong)
    val res = ncclGather(sendPtr, recvPtr, cnt, dtype, root, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, recv.length, recv)
    sendPtr.close()
    recvPtr.close()
    res
  }

  def scatterNative(
    send: Array[Float], recv: Array[Float],
    cnt: Int, dtype: Int, root: Int,
    comm: NCNativeComm, stream: CUstream_st
  ): Int = {
    val sendPtr = new FloatPointer(send.length.toLong)
    sendPtr.put(send*)
    val recvPtr = new FloatPointer(cnt.toLong)
    val res = ncclScatter(sendPtr, recvPtr, cnt, dtype, root, comm.comm, stream)
    if (res == 0) nativePtrToArray(recvPtr, cnt, recv)
    sendPtr.close()
    recvPtr.close()
    res
  }

object NCCLOps:

  def ncclAllReduce(
    send: Array[Float], recv: Array[Float], cnt: Int,
    op: NCCLOp, comm: NCCLCommunicator
  ): Unit = {
    if (comm.worldSize == 1) {
      fallbackAllReduce(send, recv, cnt, op, comm)
    } else {
      val nc = comm.nativeHandle
      if (nc != null) {
        val stream = new CUstream_st()
        val res = NCCLNatives.allReduceNative(
          send, recv, cnt,
          NCCLNatives.toNcclOp(op), ncclFloat32,
          nc, stream
        )
        if (res != 0) fallbackAllReduce(send, recv, cnt, op, comm)
      } else {
        fallbackAllReduce(send, recv, cnt, op, comm)
      }
    }
  }

  def ncclBroadcast(
    send: Array[Float], recv: Array[Float], cnt: Int,
    root: Int, comm: NCCLCommunicator
  ): Unit = {
    if (comm.worldSize == 1) {
      fallbackBroadcast(send, recv, cnt, root, comm)
    } else {
      val nc = comm.nativeHandle
      if (nc != null) {
        val stream = new CUstream_st()
        val res = NCCLNatives.broadcastNative(
          send, recv, cnt, ncclFloat32, root, nc, stream
        )
        if (res != 0) fallbackBroadcast(send, recv, cnt, root, comm)
      } else {
        fallbackBroadcast(send, recv, cnt, root, comm)
      }
    }
  }

  def ncclAllGather(
    send: Array[Float], recv: Array[Float], cnt: Int,
    comm: NCCLCommunicator
  ): Unit = {
    if (comm.worldSize == 1) {
      fallbackAllGather(send, recv, cnt, comm)
    } else {
      val nc = comm.nativeHandle
      if (nc != null) {
        val stream = new CUstream_st()
        val res = NCCLNatives.allGatherNative(
          send, recv, cnt, ncclFloat32, nc, stream
        )
        if (res != 0) fallbackAllGather(send, recv, cnt, comm)
      } else {
        fallbackAllGather(send, recv, cnt, comm)
      }
    }
  }

  def ncclReduceScatter(
    send: Array[Float], recv: Array[Float], cnt: Int,
    op: NCCLOp, comm: NCCLCommunicator
  ): Unit = {
    val nc = comm.nativeHandle
    if (nc != null) {
      val stream = new CUstream_st()
      val res = NCCLNatives.reduceScatterNative(
        send, recv, cnt,
        NCCLNatives.toNcclOp(op), ncclFloat32,
        nc, stream
      )
      if (res != 0) fallbackReduceScatter(send, recv, cnt, op, comm)
    } else {
      fallbackReduceScatter(send, recv, cnt, op, comm)
    }
  }

  def ncclAllToAll(
    send: Array[Float], recv: Array[Float], cnt: Int,
    comm: NCCLCommunicator
  ): Unit = {
    val nc = comm.nativeHandle
    if (nc != null) {
      val stream = new CUstream_st()
      val res = NCCLNatives.allToAllNative(
        send, recv, cnt, ncclFloat32, nc, stream
      )
      if (res != 0) fallbackAllToAll(send, recv, cnt, comm)
    } else {
      fallbackAllToAll(send, recv, cnt, comm)
    }
  }

  def ncclSend(
    send: Array[Float], cnt: Int, dtype: NCCLDtype,
    peer: Int, comm: NCCLCommunicator
  ): Unit = {
    val nc = comm.nativeHandle
    if (nc != null) {
      val stream = new CUstream_st()
      NCCLNatives.sendNative(
        send, cnt, NCCLNatives.toNcclDtype(dtype), peer, nc, stream
      )
    }
  }

  def ncclRecv(
    recv: Array[Float], cnt: Int, dtype: NCCLDtype,
    peer: Int, comm: NCCLCommunicator
  ): Unit = {
    val nc = comm.nativeHandle
    if (nc != null) {
      val stream = new CUstream_st()
      NCCLNatives.recvNative(
        recv, cnt, NCCLNatives.toNcclDtype(dtype), peer, nc, stream
      )
    }
  }

  def ncclReduce(
    send: Array[Float], recv: Array[Float], cnt: Int,
    op: NCCLOp, root: Int, comm: NCCLCommunicator
  ): Unit = {
    val nc = comm.nativeHandle
    if (nc != null) {
      val stream = new CUstream_st()
      val res = NCCLNatives.reduceNative(
        send, recv, cnt,
        NCCLNatives.toNcclOp(op), ncclFloat32, root,
        nc, stream
      )
      if (res != 0 && comm.rank == root)
        fallbackAllReduce(send, recv, cnt, op, comm)
    } else if (comm.rank == root) {
      fallbackAllReduce(send, recv, cnt, op, comm)
    }
  }

  def ncclGather(
    send: Array[Float], recv: Array[Float], cnt: Int,
    root: Int, comm: NCCLCommunicator
  ): Unit = {
    val nc = comm.nativeHandle
    if (nc != null && comm.rank == root) {
      val stream = new CUstream_st()
      NCCLNatives.gatherNative(send, recv, cnt, ncclFloat32, root, nc, stream)
    } else if (comm.rank == root) {
      fallbackGather(send, recv, cnt, root, comm)
    }
  }

  def ncclScatter(
    send: Array[Float], recv: Array[Float], cnt: Int,
    root: Int, comm: NCCLCommunicator
  ): Unit = {
    val nc = comm.nativeHandle
    if (nc != null) {
      val stream = new CUstream_st()
      NCCLNatives.scatterNative(send, recv, cnt, ncclFloat32, root, nc, stream)
    } else {
      fallbackScatter(send, recv, cnt, root, comm)
    }
  }

  def ncclBarrier(comm: NCCLCommunicator): Unit = comm.barrier()

  // ---- Fallback implementations (used when NCCL native unavailable) ----

  private def fallbackAllReduce(
    send: Array[Float], recv: Array[Float], cnt: Int,
    op: NCCLOp, comm: NCCLCommunicator
  ): Unit =
    comm.worldSize match
      case 1 => java.lang.System.arraycopy(send, 0, recv, 0, cnt)
      case _ =>
        op match
          case NCCLOp.Sum | NCCLOp.Avg =>
            val f = if op == NCCLOp.Avg then 1.0f / comm.worldSize else 1.0f
            for (i <- 0 until cnt) recv(i) = send(i) * comm.worldSize.toFloat * f
          case NCCLOp.Min =>
            for (i <- 0 until cnt) recv(i) = send(i) // single rank: no change
          case NCCLOp.Max =>
            for (i <- 0 until cnt) recv(i) = send(i) // single rank: no change
          case NCCLOp.Prod =>
            for (i <- 0 until cnt) recv(i) = send(i) // single rank: no change

  private def fallbackBroadcast(
    send: Array[Float], recv: Array[Float], cnt: Int,
    root: Int, comm: NCCLCommunicator
  ): Unit =
    java.lang.System.arraycopy(send, 0, recv, 0, cnt)

  private def fallbackAllGather(
    send: Array[Float], recv: Array[Float], cnt: Int,
    comm: NCCLCommunicator
  ): Unit =
    if (comm.worldSize == 1) {
      // single rank: fill recv with send buffer repeated (worldSize times, cnt per rank)
      java.lang.System.arraycopy(send, 0, recv, 0, cnt)
      if (recv.length > cnt) java.lang.System.arraycopy(send, 0, recv, cnt, recv.length - cnt)
    } else {
      for (r <- 0 until comm.worldSize) {
        val off = r * cnt
        java.lang.System.arraycopy(send, 0, recv, off, cnt)
      }
    }

  private def fallbackReduceScatter(
    send: Array[Float], recv: Array[Float], cnt: Int,
    op: NCCLOp, comm: NCCLCommunicator
  ): Unit =
    java.lang.System.arraycopy(send, comm.rank * cnt, recv, 0, cnt)

  private def fallbackAllToAll(
    send: Array[Float], recv: Array[Float], cnt: Int,
    comm: NCCLCommunicator
  ): Unit =
    for (r <- 0 until comm.worldSize) {
      val off = r * cnt
      java.lang.System.arraycopy(send, off, recv, off, cnt)
    }

  private def fallbackGather(
    send: Array[Float], recv: Array[Float], cnt: Int,
    root: Int, comm: NCCLCommunicator
  ): Unit =
    if (comm.rank == root)
      for (r <- 0 until comm.worldSize) {
        val off = r * cnt
        java.lang.System.arraycopy(send, 0, recv, off, cnt)
      }

  private def fallbackScatter(
    send: Array[Float], recv: Array[Float], cnt: Int,
    root: Int, comm: NCCLCommunicator
  ): Unit =
    if (comm.rank == root)
      java.lang.System.arraycopy(send, comm.rank * cnt, recv, 0, cnt)

/** Data parallel wrapper */
object DataParallel:
  def apply[T](comm: NCCLCommunicator, localBatchSize: Int, gradientAverage: Boolean = true) =
    new DataParallelOps(comm, localBatchSize, gradientAverage)

class DataParallelOps(
  val comm: NCCLCommunicator,
  val localBatchSize: Int,
  val gradientAverage: Boolean = true
):
  def allReduceGradients(grads: Array[Float]): Array[Float] = {
    val res = new Array[Float](grads.length)
    val f = if gradientAverage then 1.0f / comm.worldSize else 1.0f
    NCCLOps.ncclAllReduce(grads, res, grads.length, NCCLOp.Sum, comm)
    for (i <- 0 until grads.length) res(i) *= f
    res
  }

  def allReduceParams(params: Array[Float]): Array[Float] = {
    val res = new Array[Float](params.length)
    NCCLOps.ncclAllReduce(params, res, params.length, NCCLOp.Sum, comm)
    res
  }

  def synchronize(): Unit = comm.barrier()
