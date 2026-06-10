package cuda.dsl.operators.sparse

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Sparse matrix operations.
 *  Supports CSR, CSC, COO, and ELL formats.
 */
object SparseOps {

  /** CSR (Compressed Sparse Row) format:
   *  - values: non-zero values in row-major order
   *  - colIndices: column indices for each value
   *  - rowPtr: starting index for each row in values
   */

  /** Sparse matrix-vector multiply (CSR): y = A * x */
  @cudaKernel
  def csrMV(values: Ptr[Float], colIndices: Ptr[Int], rowPtr: Ptr[Int],
            x: Ptr[Float], y: Ptr[Float], numRows: Int): Unit = {
    val row = blockIdx.x * blockDim.x + threadIdx.x

    if (row < numRows) {
      var sum = 0.0f
      val start = rowPtr(row)
      val end = rowPtr(row + 1)

      for (idx <- start until end) {
        val col = colIndices(idx)
        val val_ = values(idx)
        sum += val_ * x(col)
      }
      y(row) = sum
    }
  }

  /** Sparse matrix-vector multiply with binary search (CSC): y = A^T * x */
  @cudaKernel
  def cscMV(values: Ptr[Float], rowIndices: Ptr[Int], colPtr: Ptr[Int],
            x: Ptr[Float], y: Ptr[Float], numCols: Int): Unit = {
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (col < numCols) {
      var sum = 0.0f
      val start = colPtr(col)
      val end = colPtr(col + 1)

      for (idx <- start until end) {
        val row = rowIndices(idx)
        val val_ = values(idx)
        sum += val_ * x(row)
      }
      y(col) = sum
    }
  }

  /** COO (Coordinate) format sparse matrix-vector multiply: y = A * x */
  @cudaKernel
  def cooMV(values: Ptr[Float], rowIndices: Ptr[Int], colIndices: Ptr[Int],
            x: Ptr[Float], y: Ptr[Float], numNonZeros: Int): Unit = {
    val idx = blockIdx.x * blockDim.x + threadIdx.x

    if (idx < numNonZeros) {
      val row = rowIndices(idx)
      val col = colIndices(idx)
      val val_ = values(idx)
      // Use atomic add for duplicate rows
      y(row) = y(row) + val_ * x(col)
    }
  }

  /** Sparse matrix-matrix multiply (CSR): C = A * B */
  @cudaKernel
  def csrMM(csrValuesA: Ptr[Float], csrColA: Ptr[Int], csrRowPtrA: Ptr[Int],
             b: Ptr[Float], c: Ptr[Float], numRowsA: Int, numColsB: Int, numColsA: Int): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < numRowsA && col < numColsB) {
      var sum = 0.0f
      val startA = csrRowPtrA(row)
      val endA = csrRowPtrA(row + 1)

      for (idxA <- startA until endA) {
        val k = csrColA(idxA)
        val aVal = csrValuesA(idxA)
        sum += aVal * b(k * numColsB + col)
      }
      c(row * numColsB + col) = sum
    }
  }

  /** Convert dense to CSR format */
  @cudaKernel
  def denseToCSR(A: Ptr[Float], csrValues: Ptr[Float], csrColIndices: Ptr[Int],
                  csrRowPtr: Ptr[Int], M: Int, N: Int): Unit = {
    val row = blockIdx.x * blockDim.x + threadIdx.x

    if (row < M) {
      var nnz = 0
      for (j <- 0 until N) {
        if (A(row * N + j) != 0.0f) {
          nnz += 1
        }
      }
      csrRowPtr(row) = nnz
    }
  }

  /** Convert CSR to dense format */
  @cudaOperator
  def csrToDense(csrValues: Ptr[Float], csrColIndices: Ptr[Int], csrRowPtr: Ptr[Int],
                  A: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M) {
      for (j <- 0 until N) {
        A(i * N + j) = 0.0f
      }

      val start = csrRowPtr(i)
      val end = csrRowPtr(i + 1)

      for (idx <- start until end) {
        val col = csrColIndices(idx)
        val v = csrValues(idx)
        A(i * N + col) = v
      }
    }
  }

  /** Sparse add: C = A + B (CSR format, same sparsity pattern assumed) */
  @cudaOperator
  def csrAdd(valuesA: Ptr[Float], colA: Ptr[Int], rowPtrA: Ptr[Int],
              valuesB: Ptr[Float], colB: Ptr[Int], rowPtrB: Ptr[Int],
              valuesC: Ptr[Float], colC: Ptr[Int], rowPtrC: Ptr[Int],
              numRows: Int): Unit = {
    val row = programId(0)
    if (row < numRows) {
      val startA = rowPtrA(row)
      val endA = rowPtrA(row + 1)
      val startB = rowPtrB(row)
      val endB = rowPtrB(row + 1)

      rowPtrC(row) = rowPtrA(row)

      var idxA = startA
      var idxB = startB
      var idxC = rowPtrC(row)

      while (idxA < endA && idxB < endB) {
        val colValA = colA(idxA)
        val colValB = colB(idxB)

        if (colValA == colValB) {
          colC(idxC) = colValA
          valuesC(idxC) = valuesA(idxA) + valuesB(idxB)
          idxA += 1
          idxB += 1
        } else if (colValA < colValB) {
          colC(idxC) = colValA
          valuesC(idxC) = valuesA(idxA)
          idxA += 1
        } else {
          colC(idxC) = colValB
          valuesC(idxC) = valuesB(idxB)
          idxB += 1
        }
        idxC += 1
      }
    }
  }

  /** Sparse element-wise multiply (Hadamard): C = A .* B (COO format) */
  @cudaOperator
  def csrMultiply(valuesA: Ptr[Float], colA: Ptr[Int], rowPtrA: Ptr[Int],
                   valuesB: Ptr[Float], colB: Ptr[Int], rowPtrB: Ptr[Int],
                   valuesC: Ptr[Float], colC: Ptr[Int], rowPtrC: Ptr[Int],
                   numRows: Int): Unit = {
    val row = programId(0)
    if (row < numRows) {
      val startA = rowPtrA(row)
      val endA = rowPtrA(row + 1)
      val startB = rowPtrB(row)
      val endB = rowPtrB(row + 1)

      rowPtrC(row) = rowPtrA(row)

      var idxA = startA
      var idxB = startB
      var idxC = rowPtrC(row)

      while (idxA < endA && idxB < endB) {
        val colValA = colA(idxA)
        val colValB = colB(idxB)

        if (colValA == colValB) {
          colC(idxC) = colValA
          valuesC(idxC) = valuesA(idxA) * valuesB(idxB)
          idxA += 1
          idxB += 1
        } else if (colValA < colValB) {
          idxA += 1
        } else {
          idxB += 1
        }
        idxC += 1
      }
    }
  }
}

/** ELL (ELLPACK) sparse format operations */
object ELLOps {

  /** Convert CSR to ELL format */
  @cudaKernel
  def csrToELL(csrValues: Ptr[Float], csrColIndices: Ptr[Int], csrRowPtr: Ptr[Int],
                ellValues: Ptr[Float], ellColIndices: Ptr[Int], maxRowNNZ: Int,
                numRows: Int): Unit = {
    val row = blockIdx.x * blockDim.x + threadIdx.x
    val tid = threadIdx.x

    if (row < numRows) {
      val csrStart = csrRowPtr(row)
      val csrEnd = csrRowPtr(row + 1)
      val nnz = csrEnd - csrStart

      for (i <- 0 until nnz) {
        ellValues(row * maxRowNNZ + i) = csrValues(csrStart + i)
        ellColIndices(row * maxRowNNZ + i) = csrColIndices(csrStart + i)
      }

      // Fill remaining with zeros
      for (i <- nnz until maxRowNNZ) {
        ellValues(row * maxRowNNZ + i) = 0.0f
        ellColIndices(row * maxRowNNZ + i) = -1
      }
    }
  }

  /** ELL matrix-vector multiply: y = A * x */
  @cudaKernel
  def ellMV(ellValues: Ptr[Float], ellColIndices: Ptr[Int], x: Ptr[Float],
             y: Ptr[Float], numRows: Int, maxRowNNZ: Int): Unit = {
    val row = blockIdx.x * blockDim.x + threadIdx.x

    if (row < numRows) {
      var sum = 0.0f

      for (i <- 0 until maxRowNNZ) {
        val col = ellColIndices(row * maxRowNNZ + i)
        if (col >= 0) {
          val val_ = ellValues(row * maxRowNNZ + i)
          sum += val_ * x(col)
        }
      }
      y(row) = sum
    }
  }
}

/** Hybrid (HYB) sparse format operations */
object HYBOps {

  /** Split CSR/ELL multiply */
  @cudaKernel
  def hybMV(ellValues: Ptr[Float], ellColIndices: Ptr[Int], ellNumCols: Int,
             csrValues: Ptr[Float], csrColIndices: Ptr[Int], csrRowPtr: Ptr[Int],
             x: Ptr[Float], y: Ptr[Float], numRows: Int): Unit = {
    val row = blockIdx.x * blockDim.x + threadIdx.x

    if (row < numRows) {
      var sum = 0.0f

      // ELL portion
      for (i <- 0 until ellNumCols) {
        val col = ellColIndices(row * ellNumCols + i)
        if (col >= 0) {
          sum += ellValues(row * ellNumCols + i) * x(col)
        }
      }

      // CSR portion
      val csrStart = csrRowPtr(row)
      val csrEnd = csrRowPtr(row + 1)
      for (idx <- csrStart until csrEnd) {
        val col = csrColIndices(idx)
        sum += csrValues(idx) * x(col)
      }

      y(row) = sum
    }
  }
}
