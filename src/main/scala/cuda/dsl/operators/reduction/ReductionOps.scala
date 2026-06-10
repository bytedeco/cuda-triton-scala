package cuda.dsl.operators.reduction

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Reduction operators for parallel reduction algorithms.
 *  These operators perform reduction operations (sum, max, min, etc.)
 *  using shared memory and tree-based reduction patterns.
 */
object ReductionOps {

  /** Sum reduction: output = sum(input) */
  @cudaKernel
  def sum(input: Ptr[Float], output: Ptr[Float], n: Int, blockSize: Int): Unit = {
    val i = blockIdx.x * blockSize + threadIdx.x

    // Simple reduction without shared memory
    var sumVal = 0.0f
    var idx = i
    while (idx < n) {
      sumVal += input(idx)
      idx += blockDim.x * gridDim.x
    }

    // Write result
    if (threadIdx.x == 0) {
      output(blockIdx.x) = sumVal
    }
  }

  /** Simple sum for contiguous arrays */
  @cudaOperator
  def sumSimple(input: Ptr[Float], n: Int): Float = {
    var total = 0.0f
    val i = programId(0)
    if (i < n) {
      total = input(i)
    }
    total
  }

  /** Max reduction: output = max(input) */
  @cudaOperator
  def max(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      output(0) = if (i == 0) input(0) else {
        val currentMax = output(0)
        if (input(i) > currentMax) input(i) else currentMax
      }
    }
  }

  /** Min reduction: output = min(input) */
  @cudaOperator
  def min(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      output(0) = if (i == 0) input(0) else {
        val currentMin = output(0)
        if (input(i) < currentMin) input(i) else currentMin
      }
    }
  }

  /** Argmax: output = index of maximum value */
  @cudaOperator
  def argmax(input: Ptr[Float], output: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val currentMaxIdx = output(0)
      if (i == 0) {
        output(0) = 0
      } else if (input(i) > input(currentMaxIdx)) {
        output(0) = i
      }
    }
  }

  /** Argmin: output = index of minimum value */
  @cudaOperator
  def argmin(input: Ptr[Float], output: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val currentMinIdx = output(0)
      if (i == 0) {
        output(0) = 0
      } else if (input(i) < input(currentMinIdx)) {
        output(0) = i
      }
    }
  }

  /** Product reduction: output = product(input) */
  @cudaOperator
  def prod(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      output(0) = if (i == 0) input(0) else output(0) * input(i)
    }
  }

  /** Mean: output = mean(input) */
  @cudaOperator
  def mean(input: Ptr[Float], n: Int): Float = {
    var total = 0.0f
    val i = programId(0)
    if (i < n) {
      total = input(i)
    }
    total / n.toFloat
  }

  /** Variance: output = variance(input) */
  @cudaOperator
  def variance(input: Ptr[Float], meanVal: Float, n: Int): Float = {
    var sumSq = 0.0f
    val i = programId(0)
    if (i < n) {
      val diff = input(i) - meanVal
      sumSq = diff * diff
    }
    sumSq / n.toFloat
  }

  /** Standard deviation: output = std(input) */
  @cudaOperator
  def std(input: Ptr[Float], meanVal: Float, n: Int): Float = {
    var sumSq = 0.0f
    val i = programId(0)
    if (i < n) {
      val diff = input(i) - meanVal
      sumSq = diff * diff
    }
    scala.math.sqrt(sumSq / n.toFloat).toFloat
  }

  /** L1 norm (sum of absolute values) */
  @cudaOperator
  def norm1(input: Ptr[Float], n: Int): Float = {
    var total = 0.0f
    val i = programId(0)
    if (i < n) {
      total = if (input(i) < 0) -input(i) else input(i)
    }
    total
  }

  /** L2 norm (Euclidean distance) */
  @cudaOperator
  def norm2(input: Ptr[Float], n: Int): Float = {
    var sumSq = 0.0f
    val i = programId(0)
    if (i < n) {
      val v = input(i)
      sumSq = v * v
    }
    scala.math.sqrt(sumSq).toFloat
  }

  /** Sum of squares */
  @cudaOperator
  def sumSquare(input: Ptr[Float], n: Int): Float = {
    var sumSq = 0.0f
    val i = programId(0)
    if (i < n) {
      val v = input(i)
      sumSq = v * v
    }
    sumSq
  }

  /** Count of true values (for boolean arrays) */
  @cudaOperator
  def count(input: Ptr[Bool], n: Int): Int = {
    var cnt = 0
    val i = programId(0)
    if (i < n && input(i)) {
      cnt = 1
    }
    cnt
  }

  /** Logical AND (all values true) */
  @cudaOperator
  def all(input: Ptr[Bool], n: Int): Bool = {
    var result = true
    val i = programId(0)
    if (i < n && !input(i)) {
      result = false
    }
    result
  }

  /** Logical OR (any value true) */
  @cudaOperator
  def any(input: Ptr[Bool], n: Int): Bool = {
    var result = false
    val i = programId(0)
    if (i < n && input(i)) {
      result = true
    }
    result
  }
}

/** Scan (prefix sum) operations */
object ScanOps {

  /** Inclusive prefix sum */
  @cudaOperator
  def inclusiveScan(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      if (i == 0) {
        output(i) = input(i)
      } else {
        output(i) = output(i - 1) + input(i)
      }
    }
  }

  /** Exclusive prefix sum */
  @cudaOperator
  def exclusiveScan(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      if (i == 0) {
        output(i) = 0.0f
      } else {
        output(i) = output(i - 1) + input(i - 1)
      }
    }
  }

  /** Cumulative sum (same as inclusive scan) */
  @cudaOperator
  def cumsum(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      if (i == 0) {
        output(i) = input(i)
      } else {
        output(i) = output(i - 1) + input(i)
      }
    }
  }

  /** Cumulative product */
  @cudaOperator
  def cumprod(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      if (i == 0) {
        output(i) = input(i)
      } else {
        output(i) = output(i - 1) * input(i)
      }
    }
  }

  /** Cumulative minimum */
  @cudaOperator
  def cummin(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      if (i == 0) {
        output(i) = input(i)
      } else {
        output(i) = if (input(i) < output(i - 1)) input(i) else output(i - 1)
      }
    }
  }

  /** Cumulative maximum */
  @cudaOperator
  def cummax(input: Ptr[Float], output: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      if (i == 0) {
        output(i) = input(i)
      } else {
        output(i) = if (input(i) > output(i - 1)) input(i) else output(i - 1)
      }
    }
  }
}

/** 2D reduction operations */
object Reduction2DOps {

  /** Sum over rows: output(j) = sum_i input(i, j) */
  @cudaOperator
  def sumRows(input: Ptr[Float], output: Ptr[Float], m: Int, n: Int): Unit = {
    val j = programId(0)
    if (j < n) {
      var total = 0.0f
      for (i <- 0 until m) {
        total += input(i * n + j)
      }
      output(j) = total
    }
  }

  /** Sum over columns: output(i) = sum_j input(i, j) */
  @cudaOperator
  def sumCols(input: Ptr[Float], output: Ptr[Float], m: Int, n: Int): Unit = {
    val i = programId(0)
    if (i < m) {
      var total = 0.0f
      for (j <- 0 until n) {
        total += input(i * n + j)
      }
      output(i) = total
    }
  }

  /** Mean over rows */
  @cudaOperator
  def meanRows(input: Ptr[Float], output: Ptr[Float], m: Int, n: Int): Unit = {
    val j = programId(0)
    if (j < n) {
      var total = 0.0f
      for (i <- 0 until m) {
        total += input(i * n + j)
      }
      output(j) = total / m.toFloat
    }
  }

  /** Mean over columns */
  @cudaOperator
  def meanCols(input: Ptr[Float], output: Ptr[Float], m: Int, n: Int): Unit = {
    val i = programId(0)
    if (i < m) {
      var total = 0.0f
      for (j <- 0 until n) {
        total += input(i * n + j)
      }
      output(i) = total / n.toFloat
    }
  }

  /** Max over rows */
  @cudaOperator
  def maxRows(input: Ptr[Float], output: Ptr[Float], m: Int, n: Int): Unit = {
    val j = programId(0)
    if (j < n) {
      var maxVal = input(j)
      for (i <- 1 until m) {
        val v = input(i * n + j)
        if (v > maxVal) maxVal = v
      }
      output(j) = maxVal
    }
  }

  /** Max over columns */
  @cudaOperator
  def maxCols(input: Ptr[Float], output: Ptr[Float], m: Int, n: Int): Unit = {
    val i = programId(0)
    if (i < m) {
      var maxVal = input(i * n)
      for (j <- 1 until n) {
        val v = input(i * n + j)
        if (v > maxVal) maxVal = v
      }
      output(i) = maxVal
    }
  }
}
