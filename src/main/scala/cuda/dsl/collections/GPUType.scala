package cuda.dsl.collections

import cuda.dsl.core._
import scala.reflect.ClassTag

/** Type class for GPU-supported element types */
trait GPUType[T] {
  def elementSize: Int
  def zero: T
  def scalarOne: T
  def makeArray(n: Int): Array[T]
}

object GPUType {
  given GPUType[Float] with
    val elementSize = 4
    val zero = 0.0f
    val scalarOne = 1.0f
    def makeArray(n: Int) = new Array[Float](n)

  given GPUType[Int] with
    val elementSize = 4
    val zero = 0
    val scalarOne = 1
    def makeArray(n: Int) = new Array[Int](n)

  given GPUType[Double] with
    val elementSize = 8
    val zero = 0.0
    val scalarOne = 1.0
    def makeArray(n: Int) = new Array[Double](n)

  given GPUType[Long] with
    val elementSize = 8
    val zero = 0L
    val scalarOne = 1L
    def makeArray(n: Int) = new Array[Long](n)

  given GPUType[String] with
    val elementSize = 8 // pointer size
    val zero = ""
    val scalarOne = "1"
    def makeArray(n: Int) = new Array[String](n)

  given GPUType[Boolean] with
    val elementSize = 1
    val zero = false
    val scalarOne = true
    def makeArray(n: Int) = new Array[Boolean](n)
}
