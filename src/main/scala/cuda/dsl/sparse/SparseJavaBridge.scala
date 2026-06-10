package cuda.dsl.sparse

/**
 * Java-friendly bridge for Sparse operations.
 */
object SparseJavaBridge:

  def fromDense(dense: Array[Array[Float]], tol: Float = 0.0f): SparseMatrix =
    SparseMatrix.fromDense(dense, tol)

  def identity(n: Int): SparseMatrix = SparseMatrix.identity(n)

  def random(rows: Int, cols: Int, density: Double, seed: Int = 42): SparseMatrix =
    SparseMatrix.random(rows, cols, density, seed)

  def banded(n: Int, lower: Int, upper: Int): SparseMatrix =
    SparseMatrix.banded(n, lower, upper)

  def spmv(A: SparseMatrix, x: Array[Float]): Array[Float] = SparseOps.spmv(A, x)

  def spmm(A: SparseMatrix, B: Array[Array[Float]]): Array[Array[Float]] = SparseOps.spmm(A, B)

  def transpose(A: SparseMatrix): SparseMatrix = SparseOps.transpose(A)

  def spadd(A: SparseMatrix, B: SparseMatrix, alpha: Float, beta: Float): SparseMatrix =
    SparseOps.spadd(A, B, alpha, beta)

  def diagonal(A: SparseMatrix): Array[Float] = SparseOps.diagonal(A)

  def isSymmetric(A: SparseMatrix): Boolean = SparseOps.isSymmetric(A)

  def sparseAttention(seqLen: Int, blockSize: Int = 32): SparseAttention =
    SparseAttention.fromLowerTriangular(seqLen, blockSize)

  def blockSparseFromDense(dense: Array[Array[Float]], blockSize: Int, tol: Float = 0.0f): BlockSparseMatrix =
    BlockSparseMatrix.fromDense(dense, blockSize, tol)

  def blockSpMV(bsm: BlockSparseMatrix, x: Array[Float]): Array[Float] = bsm.spmv(x)
