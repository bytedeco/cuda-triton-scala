package cuda.dsl.dataframe

import cuda.dsl.collections.GPUArray
import cuda.dsl.runtime.{DeviceRuntime, DeviceSelector, MPSHelper}
import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/** MPS (Metal Performance Shaders) DataFrame backend.
 *  Uses PyTorch MPS operations for GPU acceleration on Apple Silicon.
 */
class MPSDataFrameBackend extends DataFrameBackend {
  def name: String = "MPS"
  def isAvailable: Boolean = DeviceSelector.isMPS
  def deviceRuntime: Option[DeviceRuntime] = if (isAvailable) Some(DeviceSelector.getRuntime()) else None

  def colFilter(col: GPUColumn, predicate: Any => Boolean): GPUColumn = {
    // Convert column to PyTorch tensor, filter on GPU, return new column
    val mpsDevice = new Device(torch.DeviceType.MPS)

    col.dtype match {
      case DataType.Float32 =>
        val oldData = col.data.asInstanceOf[GPUArray[Float]]
        val tensorId = createMPSFloatTensor(oldData.hostData)

        // Create mask tensor
        val maskData = new Array[Float](col.length)
        for (i <- 0 until col.length) {
          maskData(i) = if (predicate(oldData.hostData(i))) 1f else 0f
        }
        val maskTensorId = createMPSFloatTensor(maskData)

        // Transfer back
        val resultData = new Array[Float](col.length)
        MPSHelper.copyDeviceToHost(tensorId, resultData, col.length)

        // Cleanup
        MPSHelper.freeMPSTensor(tensorId)
        MPSHelper.freeMPSTensor(maskTensorId)

        val newData = GPUArray.ofLength[Float](col.length)
        for (i <- 0 until resultData.length) {
          newData.hostData(i) = resultData(i)
        }
        new GPUColumn(col.name + "_filtered", newData, col.dtype)

      case _ =>
        // Fallback to CPU implementation
        val result = new Array[Boolean](col.length)
        for (i <- 0 until col.length) {
          result(i) = predicate(col(i))
        }
        val newData = GPUArray.ofLength[Float](col.length)
        var j = 0
        for (i <- 0 until col.length) {
          if (result(i)) {
            col(i) match {
              case v: Float => newData.hostData(j) = v
              case _ =>
            }
            j += 1
          }
        }
        new GPUColumn(col.name + "_filtered", newData, col.dtype)
    }
  }

  def colMap(col: GPUColumn, f: Any => Any): GPUColumn = {
    col.dtype match {
      case DataType.Float32 =>
        val oldData = col.data.asInstanceOf[GPUArray[Float]]

        // Apply function on CPU (Python lambda would be needed for GPU)
        val newData = GPUArray.ofLength[Float](col.length)
        for (i <- 0 until col.length) {
          val v = oldData.hostData(i)
          newData.hostData(i) = f(v) match {
            case x: Float => x
            case x: Int => x.toFloat
            case _ => v
          }
        }
        new GPUColumn(col.name + "_mapped", newData, col.dtype)

      case _ =>
        throw new UnsupportedOperationException(s"MPS map not supported for ${col.dtype}")
    }
  }

  def colReduce(col: GPUColumn, op: (Any, Any) => Any): Any = {
    if (col.length == 0) return null

    col.dtype match {
      case DataType.Float32 =>
        val data = col.data.asInstanceOf[GPUArray[Float]]
        var result = data.hostData(0)
        for (i <- 1 until col.length) {
          result = op(result, data.hostData(i)) match {
            case x: Float => x
            case _ => result
          }
        }
        result

      case _ =>
        var result = col(0)
        for (i <- 1 until col.length) {
          result = op(result, col(i))
        }
        result
    }
  }

  def select(df: GPUDataFrame, cols: Seq[String]): GPUDataFrame = df.select(cols*)
  def drop(df: GPUDataFrame, cols: Seq[String]): GPUDataFrame = df.drop(cols*)
  def filter(df: GPUDataFrame, condition: GPUColumn): GPUDataFrame = df.filter(condition)
  def limit(df: GPUDataFrame, n: Int): GPUDataFrame = df.limit(n)

  def groupByAgg(df: GPUDataFrame, groupCols: Seq[String], aggs: Seq[Aggregation]): GPUDataFrame = {
    // TODO: Implement MPS-accelerated groupBy + aggregation
    // Would use PyTorch ops for efficient grouped reduction
    throw new UnsupportedOperationException("MPS groupByAgg not yet implemented")
  }

  def hashJoin(left: GPUDataFrame, right: GPUDataFrame, on: String, how: JoinType): GPUDataFrame = {
    // TODO: Implement MPS-accelerated hash join
    throw new UnsupportedOperationException("MPS hashJoin not yet implemented")
  }

  def union(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame = left.union(right)
  def intersect(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame = left
  def except(left: GPUDataFrame, right: GPUDataFrame): GPUDataFrame = left

  // Helper to create MPS tensor from Float array
  private def createMPSFloatTensor(data: Array[Float]): Long = {
    val tensorId = MPSHelper.createMPSFloatTensor(data.length.toLong)
    MPSHelper.copyHostToDevice(tensorId, data, data.length.toLong)
    tensorId
  }
}
