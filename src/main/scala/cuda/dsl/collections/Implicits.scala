package cuda.dsl.collections

import cuda.dsl.core._
import scala.reflect.ClassTag

/** Implicit conversions to enable GPU operations on Scala collections.
 *  Import this object to enable the `.enable_gpu` method on collections.
 *
 *  Example:
 *  {{{
 *  import cuda.dsl.collections.Implicits._
 *
 *  val result = Seq(1.0f, 2.0f, 3.0f)
 *    .enable_gpu
 *    .map(_ * 2.0f)
 *    .filter(_ > 3.0f)
 *    .toSeq
 *  }}}
 */
object Implicits {

  /** Implicit class to add GPU capabilities to Seq */
  implicit class GPUSeqOps[T: GPUType: ClassTag](val seq: Seq[T]) {
    /** Transfer sequence to GPU for accelerated operations */
    def enable_gpu(using gpu: GPUType[T]): GPUSeq[T] = {
      GPUSeq.fromSeq(seq)
    }
  }

  /** Implicit class to add GPU capabilities to List */
  implicit class GPUListOps[T: GPUType: ClassTag](val list: List[T]) {
    /** Transfer list to GPU for accelerated operations */
    def enable_gpu(using gpu: GPUType[T]): GPUSeq[T] = {
      GPUSeq.fromList(list)
    }
  }

  /** Implicit class to add GPU capabilities to Array */
  implicit class GPUArrayOps[T: GPUType: ClassTag](val arr: Array[T]) {
    /** Transfer array to GPU for accelerated operations */
    def enable_gpu(using gpu: GPUType[T]): GPUArray[T] = {
      GPUArray.fromArray(arr)
    }
  }

  /** Extension methods for GPUArray */
  implicit class GPUArrayExtensions[T: GPUType: ClassTag](val gpuArr: GPUArray[T]) {
    /** Collect back to Seq */
    def toScalaSeq(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): Seq[T] = {
      gpuArr.toSeq
    }

    /** Collect back to List */
    def toScalaList(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): List[T] = {
      gpuArr.toList
    }
  }

  /** Extension methods for GPUSeq */
  implicit class GPUSeqExtensions[T: GPUType: ClassTag](val gpuSeq: GPUSeq[T]) {
    /** Collect back to Seq */
    def toScalaSeq(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): Seq[T] = {
      gpuSeq.toSeq
    }

    /** Collect back to List */
    def toScalaList(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): List[T] = {
      gpuSeq.toList
    }

    /** Collect back to Array */
    def toScalaArray(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): Array[T] = {
      gpuSeq.toArray
    }
  }
}

/** Rich GPU-accelerated collection operations.
 *  Provides a fluent API for chaining GPU operations.
 */
class RichGPUCollection[T: GPUType: GPUOps: ClassTag](private val gpuColl: GPUArray[T]) {
  import Implicits._

  /** Map operation */
  def map(f: T => T)(using ops: MemoryOps[T], gpuOps: GPUOps[T], gpu: GPUType[T]): RichGPUCollection[T] = {
    new RichGPUCollection(gpuColl.map(f))
  }

  /** Filter operation */
  def filter(p: T => Boolean)(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): RichGPUCollection[T] = {
    new RichGPUCollection(gpuColl.filter(p))
  }

  /** Reduce operation - returns result, not collection */
  def reduce(op: (T, T) => T)(using ops: MemoryOps[T], gpuOps: GPUOps[T], gpu: GPUType[T]): T = {
    gpuColl.reduce(op)
  }

  /** Fold operation */
  def fold(z: T)(op: (T, T) => T)(using ops: MemoryOps[T], gpuOps: GPUOps[T], gpu: GPUType[T]): T = {
    gpuColl.fold(z)(op)
  }

  /** Foreach operation */
  def foreach(f: T => Unit)(using ops: MemoryOps[T], gpuOps: GPUOps[T], gpu: GPUType[T]): Unit = {
    gpuColl.foreach(f)
  }

  /** Convert to Scala Seq */
  def toSeq(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): Seq[T] = {
    gpuColl.toSeq
  }

  /** Convert to Scala List */
  def toList(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): List[T] = {
    gpuColl.toList
  }

  /** Convert to Scala Array */
  def toArray(using ops: MemoryOps[T], gpu: GPUType[T], ct: ClassTag[T]): Array[T] = {
    gpuColl.toArray
  }
}
