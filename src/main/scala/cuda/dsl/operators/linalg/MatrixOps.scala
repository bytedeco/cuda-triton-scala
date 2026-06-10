package cuda.dsl.operators.linalg

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Linear algebra operations - matrix and vector operations.
 *  Includes matrix multiplication, transpose, dot product, etc.
 */
object MatrixOps {

  /** Matrix multiplication: C = A * B
   *  A is MxK, B is KxN, C is MxN
   */
  @cudaKernel
  def matmul(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], M: Int, N: Int, K: Int): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < M && col < N) {
      var sum = 0.0f
      for (k <- 0 until K) {
        sum += A(row * K + k) * B(k * N + col)
      }
      C(row * N + col) = sum
    }
  }

  /** Matrix multiplication with bias: C = A * B + bias */
  @cudaKernel
  def matmulAddBias(A: Ptr[Float], B: Ptr[Float], bias: Ptr[Float], C: Ptr[Float], M: Int, N: Int, K: Int): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < M && col < N) {
      var sum = 0.0f
      for (k <- 0 until K) {
        sum += A(row * K + k) * B(k * N + col)
      }
      C(row * N + col) = sum + bias(col)
    }
  }

  /** Transpose matrix: B = A^T
   *  A is MxN, B is NxM
   */
  @cudaOperator
  def transpose(A: Ptr[Float], B: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M * N) {
      val row = i / N
      val col = i % N
      B(col * M + row) = A(row * N + col)
    }
  }

  /** 2D transpose with tiling for better memory access pattern */
  @cudaKernel
  def transpose2D(A: Ptr[Float], B: Ptr[Float], M: Int, N: Int): Unit = {
    // Note: Shared memory tiling requires compile-time size known
    // Using direct access pattern for now
    val x = blockIdx.x * 16 + threadIdx.x
    val y = blockIdx.y * 16 + threadIdx.y

    if (x < N && y < M) {
      B(x * M + y) = A(y * N + x)
    }
  }

  /** Dot product: result = sum(a * b) */
  @cudaOperator
  def dot(a: Ptr[Float], b: Ptr[Float], n: Int): Float = {
    var sum = 0.0f
    for (i <- 0 until n) {
      sum += a(i) * b(i)
    }
    sum
  }

  /** Vector addition: c = a + b */
  @cudaOperator
  def vecAdd(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) + b(i)
  }

  /** Vector subtraction: c = a - b */
  @cudaOperator
  def vecSub(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) - b(i)
  }

  /** Vector scaling: b = alpha * a */
  @cudaOperator
  def scale(alpha: Float, a: Ptr[Float], b: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) b(i) = alpha * a(i)
  }

  /** Outer product: C = a * b^T
   *  a is Mx1, b is Nx1, C is MxN
   */
  @cudaOperator
  def outer(a: Ptr[Float], b: Ptr[Float], C: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M * N) {
      val row = i / N
      val col = i % N
      C(row * N + col) = a(row) * b(col)
    }
  }

  /** Diagonal extraction: d = diag(A) */
  @cudaOperator
  def diag(A: Ptr[Float], d: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) d(i) = A(i * n + i)
  }

  /** Identity matrix: I = eye(n) */
  @cudaOperator
  def eye(I: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n * n) {
      val row = i / n
      val col = i % n
      I(row * n + col) = if (row == col) 1.0f else 0.0f
    }
  }

  /** Upper triangular matrix */
  @cudaOperator
  def triu(A: Ptr[Float], n: Int, k: Int): Unit = {
    val i = programId(0)
    if (i < n * n) {
      val row = i / n
      val col = i % n
      A(row * n + col) = if (col >= row + k) A(row * n + col) else 0.0f
    }
  }

  /** Lower triangular matrix */
  @cudaOperator
  def tril(A: Ptr[Float], n: Int, k: Int): Unit = {
    val i = programId(0)
    if (i < n * n) {
      val row = i / n
      val col = i % n
      A(row * n + col) = if (col <= row + k) A(row * n + col) else 0.0f
    }
  }

  /** Matrix trace: tr = sum(diag(A)) */
  @cudaOperator
  def trace(A: Ptr[Float], n: Int): Float = {
    var tr = 0.0f
    for (i <- 0 until n) {
      tr += A(i * n + i)
    }
    tr
  }

  /** Matrix-vector multiplication: y = A * x */
  @cudaOperator
  def gemv(A: Ptr[Float], x: Ptr[Float], y: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M) {
      var sum = 0.0f
      for (j <- 0 until N) {
        sum += A(i * N + j) * x(j)
      }
      y(i) = sum
    }
  }

  /** Rank-1 update: A = A + alpha * x * y^T */
  @cudaOperator
  def ger(alpha: Float, x: Ptr[Float], y: Ptr[Float], A: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M * N) {
      val row = i / N
      val col = i % N
      A(row * N + col) = A(row * N + col) + alpha * x(row) * y(col)
    }
  }

  /** Matrix addition: C = A + B */
  @cudaOperator
  def matAdd(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M * N) {
      C(i) = A(i) + B(i)
    }
  }

  /** Matrix subtraction: C = A - B */
  @cudaOperator
  def matSub(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M * N) {
      C(i) = A(i) - B(i)
    }
  }

  /** Scalar-matrix multiplication: B = alpha * A */
  @cudaOperator
  def matScale(alpha: Float, A: Ptr[Float], B: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M * N) {
      B(i) = alpha * A(i)
    }
  }

  /** Hadamard (element-wise) product: C = A * B */
  @cudaOperator
  def hadamard(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], M: Int, N: Int): Unit = {
    val i = programId(0)
    if (i < M * N) {
      C(i) = A(i) * B(i)
    }
  }
}

/** BLAS-style operations using cuBLAS */
object BLASOps {

  /** SGEMM - Single precision matrix multiply */
  @cudaKernel
  def sgemm(A: Ptr[Float], B: Ptr[Float], C: Ptr[Float], M: Int, N: Int, K: Int, alpha: Float, beta: Float): Unit = {
    val row = blockIdx.y * blockDim.y + threadIdx.y
    val col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < M && col < N) {
      var sum = 0.0f
      for (k <- 0 until K) {
        sum += A(row * K + k) * B(k * N + col)
      }
      C(row * N + col) = alpha * sum + beta * C(row * N + col)
    }
  }

  /** SYMV - Symmetric matrix-vector multiply */
  @cudaOperator
  def symv(A: Ptr[Float], x: Ptr[Float], y: Ptr[Float], n: Int, alpha: Float, beta: Float): Unit = {
    val i = programId(0)
    if (i < n) {
      var sum = 0.0f
      for (j <- 0 until n) {
        sum += A(i * n + j) * x(j)
      }
      y(i) = alpha * sum + beta * y(i)
    }
  }

  /** TRMV - Triangular matrix-vector multiply */
  @cudaOperator
  def trmv(A: Ptr[Float], x: Ptr[Float], n: Int, unit: Bool, upper: Bool): Unit = {
    val i = programId(0)
    if (i < n) {
      var sum = x(i)
      val start = if (upper) 0 else i + 1
      val end = if (upper) i else n
      for (j <- start until end) {
        sum += A(i * n + j) * x(j)
      }
      x(i) = if (!unit) sum else sum + x(i)
    }
  }

  /** SPR - Symmetric packed matrix rank-1 update */
  @cudaOperator
  def spr(alpha: Float, x: Ptr[Float], A: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      for (j <- 0 until n) {
        A(i * n + j) = A(i * n + j) + alpha * x(i) * x(j)
      }
    }
  }

  /** DNRM2 - Euclidean norm of vector */
  @cudaOperator
  def dnrm2(x: Ptr[Float], n: Int): Float = {
    var sumSq = 0.0f
    for (i <- 0 until n) {
      val v = x(i)
      sumSq += v * v
    }
    scala.math.sqrt(sumSq).toFloat
  }

  /** DASUM - Sum of absolute values */
  @cudaOperator
  def dasum(x: Ptr[Float], n: Int): Float = {
    var sum = 0.0f
    for (i <- 0 until n) {
      val v = x(i)
      val absV = if (v < 0) -v else v
      sum += absV
    }
    sum
  }

  /** IDAMAX - Index of maximum absolute value */
  @cudaOperator
  def idamax(x: Ptr[Float], n: Int): Int = {
    var maxIdx = 0
    var maxVal = 0.0f
    for (i <- 0 until n) {
      val v = x(i)
      val absV = if (v < 0) -v else v
      if (absV > maxVal) {
        maxVal = absV
        maxIdx = i
      }
    }
    maxIdx
  }
}

/** Strided operations for advanced indexing */
object StridedOps {

  /** Gather: y(i) = x(stride * i) */
  @cudaOperator
  def gather(x: Ptr[Float], indices: Ptr[Int], y: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      y(i) = x(indices(i))
    }
  }

  /** Scatter: y(stride * i) = x(i) */
  @cudaOperator
  def scatter(indices: Ptr[Int], x: Ptr[Float], y: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      y(indices(i)) = x(i)
    }
  }

  /** Strided slice */
  @cudaOperator
  def stridedSlice(x: Ptr[Float], start: Int, step: Int, y: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      y(i) = x(start + i * step)
    }
  }

  /** Strided write */
  @cudaOperator
  def stridedWrite(start: Int, step: Int, x: Ptr[Float], y: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      y(start + i * step) = x(i)
    }
  }

  /** 2D gather */
  @cudaOperator
  def gather2D(A: Ptr[Float], rowIndices: Ptr[Int], colIndices: Ptr[Int], B: Ptr[Float], n: Int, cols: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val row = rowIndices(i)
      val col = colIndices(i)
      B(i) = A(row * cols + col)
    }
  }

  /** 2D scatter */
  @cudaOperator
  def scatter2D(rowIndices: Ptr[Int], colIndices: Ptr[Int], A: Ptr[Float], B: Ptr[Float], n: Int, cols: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val row = rowIndices(i)
      val col = colIndices(i)
      A(row * cols + col) = B(i)
    }
  }
}
