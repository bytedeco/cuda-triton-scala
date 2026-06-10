package cuda.dsl.sparse

import cuda.dsl.dsl.*
import cuda.dsl.runtime.*
import cuda.dsl.runtime.ScalaCudaRuntime as SCR
import org.bytedeco.cuda.cusparse.{cusparseContext, cusparseDnMatDescr, cusparseDnVecDescr, cusparseSpMatDescr}

import scala.collection.mutable
import org.bytedeco.cuda.global.cusparse.*
import org.bytedeco.javacpp.{FloatPointer, IntPointer, LongPointer, Pointer, SizeTPointer}

/** cuSPARSE index types */
object cusparseIndexType:
  val CUSPARSE_INDEX_16BITS: Int = 9
  val CUSPARSE_INDEX_32BITS: Int = 10
  val CUSPARSE_INDEX_64BITS: Int = 11

/** cuSPARSE index base */
object cusparseIndexBase:
  val CUSPARSE_INDEX_BASE_ZERO: Int = 0
  val CUSPARSE_INDEX_BASE_ONE: Int = 1

/** CUDA data types */
object cudaDataType:
  val CUDA_R_16F: Int = 2
  val CUDA_R_32F: Int = 0
  val CUDA_R_64F: Int = 1
  val CUDA_R_8I: Int = 4
  val CUDA_R_8U: Int = 3
  val CUDA_R_16BF: Int = 20
  val CUDA_C_16F: Int = 6
  val CUDA_C_32F: Int = 8
  val CUDA_C_64F: Int = 9

/** cuSPARSE order */
object cusparseOrder:
  val CUSPARSE_ORDER_ROW: Int = 0
  val CUSPARSE_ORDER_COL: Int = 1

/** cuSPARSE handle wrapper */
class CUSPARSEHandle private () extends AutoCloseable:
  val handle: cusparseContext = new cusparseContext()
  private var created = false

  private def this(h: cusparseContext) = {
    this()
    if (created) ()
  }

  def create(): CUSPARSEHandle = {
    if (created) return this
    try
      val res = cusparseCreate(handle)
      if (res == 0) { created = true } else { () }
      this
    catch case _: Throwable => this
  }

  def isCreated: Boolean = created

  def destroy(): Int =
    if created then
      try
        val res = cusparseDestroy(handle)
        if res == 0 then created = false
        res
      catch case _: Throwable => -1
    else 0

  override def close(): Unit = { try destroy() catch case _: Throwable => () }
  override def finalize(): Unit = { try destroy() catch case _: Throwable => () }

object CUSPARSEHandle:
  private val singleton = new CUSPARSEHandle()

  def getOrCreate: CUSPARSEHandle =
    if !singleton.isCreated then singleton.create()
    singleton

  def destroy(): Unit = singleton.destroy()

/** Sparse matrix format */
enum SparseFormat:
  case CSR, CSC, COO, BSR

/** Dense to Sparse / Sparse to Dense conversion algorithms */
object CUSPARSEConversionAlg:
  val ROUNDING = 0

/** Operation types for cuSPARSE */
object CUSPARSESparseToDense:
  val ALG_DEFAULT = 0

object CUSPARSEDenseToSparse:
  val ALG_DEFAULT = 0

/** cuSPARSE Sparse Matrix wrapper */
class CUSPARSESparseMatrix(
  val rows: Long,
  val cols: Long,
  val nnz: Long,
  val rowPtr: Pointer,
  val colInd: Pointer,
  val values: Pointer,
  val descr: cusparseSpMatDescr = new cusparseSpMatDescr()
) extends AutoCloseable:

  def this(
    rows: Long,
    cols: Long,
    nnz: Long,
    rowPtrArr: Array[Int],
    colIndArr: Array[Int],
    valuesArr: Array[Float]
  ) = {
    this(
      rows, cols, nnz,
      new IntPointer(rowPtrArr*),
      new IntPointer(colIndArr*),
      new FloatPointer(valuesArr*)
    )
  }

  private var initialized = false

  def initialize(handle: CUSPARSEHandle): Boolean = {
    if (initialized) return true
    try
      val res = cusparseCreateConstCsr(
        descr, rows, cols, nnz,
        rowPtr,
        colInd,
        values,
        cusparseIndexType.CUSPARSE_INDEX_32BITS,
        cusparseIndexType.CUSPARSE_INDEX_32BITS,
        cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO,
        cudaDataType.CUDA_R_32F
      )
      initialized = res == 0
    catch case _: Throwable =>
      initialized = false
    initialized
  }

  def destroy(): Int =
    if initialized then
      try
        descr.deallocate()
        initialized = false
        0
      catch case _: Throwable => -1
    else 0

  override def close(): Unit = { try destroy() catch case _: Throwable => () }

/** cuSPARSE Dense Matrix wrapper */
class CUSPARSEDenseMatrix(
  val rows: Long,
  val cols: Long,
  val ld: Long,
  val values: Pointer,
  val descr: cusparseDnMatDescr = new cusparseDnMatDescr()
) extends AutoCloseable:

  def this(rows: Int, cols: Int, valuesArr: Array[Float]) = {
    this(
      rows.toLong,
      cols.toLong,
      cols.toLong,
      new FloatPointer(valuesArr*)
    )
  }

  private var initialized = false

  def initialize(order: Int = cusparseOrder.CUSPARSE_ORDER_ROW): Boolean = {
    if (initialized) return true
    try
      val res = cusparseCreateConstDnMat(
        descr, rows, cols, ld,
        values,
        cudaDataType.CUDA_R_32F,
        order
      )
      initialized = res == 0
    catch case _: Throwable =>
      initialized = false
    initialized
  }

  def destroy(): Int =
    if initialized then
      try
        descr.deallocate()
        initialized = false
        0
      catch case _: Throwable => -1
    else 0

  override def close(): Unit = { try destroy() catch case _: Throwable => () }

/** cuSPARSE Dense Vector wrapper */
class CUSPARSEDenseVector(
  val size: Long,
  val values: Pointer,
  val descr: cusparseDnVecDescr = new cusparseDnVecDescr()
) extends AutoCloseable:

  def this(size: Int, valuesArr: Array[Float]) = {
    this(size.toLong, new FloatPointer(valuesArr*))
  }

  private var initialized = false

  def initialize(): Boolean = {
    if (initialized) return true
    try
      val res = cusparseCreateConstDnVec(
        descr, size, values, cudaDataType.CUDA_R_32F
      )
      initialized = res == 0
    catch case _: Throwable =>
      initialized = false
    initialized
  }

  def destroy(): Int =
    if initialized then
      try
        descr.deallocate()
        initialized = false
        0
      catch case _: Throwable => -1
    else 0

  override def close(): Unit = { try destroy() catch case _: Throwable => () }

/** Sparse matrix with native cuSPARSE bindings */
class SparseMatrix(
  val values: Array[Float],
  val rowPtr: Array[Int],
  val colIdx: Array[Int],
  val shape: (Int, Int),
  val format: SparseFormat = SparseFormat.CSR
) {
  val numRows: Int = shape._1
  val numCols: Int = shape._2
  val nnz: Int = values.length

  def toCSR: SparseMatrix = if (format == SparseFormat.CSR) this else convertTo(SparseFormat.CSR)
  def toCSC: SparseMatrix = if (format == SparseFormat.CSC) this else convertTo(SparseFormat.CSC)

  def convertTo(target: SparseFormat): SparseMatrix =
    if (format == target) this
    else target match
      case SparseFormat.CSR if format == SparseFormat.CSC =>
        val (tVal, tRow, tCol) = transposeCSC()
        SparseMatrix(tVal, tRow, tCol, (numCols, numRows), SparseFormat.CSR)
      case SparseFormat.CSC if format == SparseFormat.CSR =>
        val (tVal, tRow, tCol) = transposeCSR()
        SparseMatrix(tVal, tRow, tCol, (numRows, numCols), SparseFormat.CSC)
      case _ => this

  private[sparse] def transposeCSR(): (Array[Float], Array[Int], Array[Int]) = {
    val tnnz = nnz
    val tValues = new Array[Float](tnnz.toInt)
    val tRowPtr = new Array[Int](numCols + 1)
    val tColIdx = new Array[Int](tnnz.toInt)

    val colCount = Array.fill(numCols)(0)
    for (j <- 0 until nnz.toInt) colCount(colIdx(j)) += 1

    tRowPtr(0) = 0
    for (c <- 0 until numCols) tRowPtr(c + 1) = tRowPtr(c) + colCount(c)

    val tempPtr = tRowPtr.clone()
    for (i <- 0 until numRows) {
      for (pi <- rowPtr(i) until rowPtr(i + 1)) {
        val j = colIdx(pi)
        val pos = tempPtr(j)
        tempPtr(j) += 1
        tValues(pos) = values(pi)
        tColIdx(pos) = i
      }
    }
    (tValues, tRowPtr, tColIdx)
  }

  private def transposeCSC(): (Array[Float], Array[Int], Array[Int]) =
    transposeCSR()

  def toCUSPARSESparseMatrix(handle: CUSPARSEHandle): Option[CUSPARSESparseMatrix] = {
    val mat = new CUSPARSESparseMatrix(numRows, numCols, nnz, rowPtr, colIdx, values)
    if mat.initialize(handle) then Some(mat) else None
  }

  def info: String = {
    val density = nnz.toDouble / (numRows * numCols) * 100
    s"SparseMatrix(${numRows}x${numCols}, nnz=$nnz, density=${density.formatted("%.2f")}%, format=$format)"
  }
}

object SparseMatrix:
  def fromDense(dense: Array[Array[Float]], tol: Float = 0.0f): SparseMatrix = {
    val rows = dense.length
    val cols = dense(0).length
    val values = mutable.ListBuffer[Float]()
    val colIdx = mutable.ListBuffer[Int]()
    val rowPtr = Array.newBuilder[Int]
    rowPtr += 0

    for (r <- 0 until rows) {
      for (c <- 0 until cols) {
        val v = dense(r)(c)
        if (v.abs > tol) {
          values += v
          colIdx += c
        }
      }
      rowPtr += values.size
    }
    new SparseMatrix(values.toArray, rowPtr.result(), colIdx.toArray, (rows, cols), SparseFormat.CSR)
  }

  def identity(n: Int): SparseMatrix = {
    val values = Array.fill(n)(1.0f)
    val rowPtr = Array.tabulate(n + 1)(i => i)
    val colIdx = Array.tabulate(n)(i => i)
    new SparseMatrix(values, rowPtr, colIdx, (n, n), SparseFormat.CSR)
  }

  def random(rows: Int, cols: Int, density: Double, seed: Int = 42): SparseMatrix = {
    val rnd = new scala.util.Random(seed)
    val nnzVals = (rows * cols * density).toInt
    val values = Array.tabulate(nnzVals)(_ => rnd.nextFloat() * 2 - 1)
    val colIdx = Array.tabulate(nnzVals)(_ => rnd.nextInt(cols)).sorted

    val rowPtr = Array.newBuilder[Int]
    rowPtr += 0
    var curRow = 0
    var elemInRow = 0
    val targetPerRow = (cols * density).toInt.max(1)

    for (i <- 0 until nnzVals) {
      elemInRow += 1
      if (elemInRow >= targetPerRow && curRow < rows - 1) {
        rowPtr += i + 1
        curRow += 1
        elemInRow = 0
      }
    }
    while (rowPtr.result().length < rows + 1) rowPtr += nnzVals

    new SparseMatrix(values, rowPtr.result(), colIdx, (rows, cols), SparseFormat.CSR)
  }

  def banded(n: Int, lower: Int, upper: Int): SparseMatrix = {
    val values = mutable.ListBuffer[Float]()
    val colIdx = mutable.ListBuffer[Int]()
    val rowPtr = Array.newBuilder[Int]
    rowPtr += 0

    for (i <- 0 until n) {
      for (j <- (i - lower).max(0).to((i + upper).min(n - 1))) {
        values += (if (i == j) 2.0f else -1.0f)
        colIdx += j
      }
      rowPtr += values.size
    }
    rowPtr += values.size

    new SparseMatrix(values.toArray, rowPtr.result(), colIdx.toArray, (n, n), SparseFormat.CSR)
  }

/** Sparse operations with cuSPARSE fallback */
object SparseOps:
  def spmv(A: SparseMatrix, x: Array[Float]): Array[Float] = {
    val y = Array.fill(A.numRows)(0.0f)
    val Acsr = A.toCSR

    for (i <- 0 until Acsr.numRows) {
      var sum = 0.0f
      for (pi <- Acsr.rowPtr(i) until Acsr.rowPtr(i + 1)) {
        val j = Acsr.colIdx(pi)
        if (j < x.length) sum += Acsr.values(pi) * x(j)
      }
      y(i) = sum
    }
    y
  }

  def spmm(A: SparseMatrix, B: Array[Array[Float]]): Array[Array[Float]] = {
    val C = Array.fill(A.numRows)(Array.fill(B(0).length)(0.0f))

    for (i <- 0 until A.numRows) {
      for (pi <- A.rowPtr(i) until A.rowPtr(i + 1)) {
        val j = A.colIdx(pi)
        val aij = A.values(pi)
        for (k <- 0 until B(0).length) {
          C(i)(k) += aij * B(j)(k)
        }
      }
    }
    C
  }

  def transpose(A: SparseMatrix): SparseMatrix = {
    val Acsr = A.toCSR
    val (tVal, tRow, tCol) = Acsr.transposeCSR()
    new SparseMatrix(tVal, tRow, tCol, (A.numCols, A.numRows), SparseFormat.CSR)
  }

  def spadd(A: SparseMatrix, B: SparseMatrix, alpha: Float = 1.0f, beta: Float = 1.0f): SparseMatrix = {
    val vals = Array.tabulate(A.nnz)(i => alpha * A.values(i) + beta * B.values(i))
    new SparseMatrix(vals, A.rowPtr.clone(), A.colIdx.clone(), A.shape, A.format)
  }

  def diagonal(A: SparseMatrix): Array[Float] = {
    val diag = Array.fill(A.numRows.min(A.numCols))(0.0f)
    val Acsr = A.toCSR

    for (i <- 0 until Acsr.numRows) {
      for (pi <- Acsr.rowPtr(i) until Acsr.rowPtr(i + 1)) {
        val j = Acsr.colIdx(pi)
        if (i == j && i < diag.length) diag(i) = Acsr.values(pi)
      }
    }
    diag
  }

  def isSymmetric(A: SparseMatrix): Boolean = {
    if (A.numRows != A.numCols) return false
    val Acsr = A.toCSR
    val Acsc = Acsr.toCSC
    Acsr.values.sameElements(Acsc.values) && Acsr.colIdx.sameElements(Acsc.colIdx)
  }

/** Sparse attention with configurable sparsity mask */
class SparseAttention(
  val sparsityMask: Array[Array[Boolean]],
  val blockSize: Int = 32
) {
  private val seqLen = sparsityMask.length
  private val numBlocks = (seqLen + blockSize - 1) / blockSize

  def forward(Q: Array[Array[Float]], K: Array[Array[Float]], V: Array[Array[Float]]): Array[Array[Float]] = {
    val seqLen = Q.length
    val headDim = if (Q.nonEmpty) Q(0).length else 0
    val out = Array.fill(seqLen)(Array.fill(headDim)(0.0f))

    for (i <- 0 until seqLen) {
      for (j <- 0 until seqLen) {
        if (i < sparsityMask.length && j < sparsityMask(i).length && sparsityMask(i)(j)) {
          var score = 0.0f
          for (d <- 0 until headDim) {
            score += Q(i)(d) * K(j)(d)
          }
          score /= math.sqrt(headDim.toFloat).toFloat
          val softmax = math.exp(score.toDouble).toFloat
          for (d <- 0 until headDim) {
            out(i)(d) += softmax * V(j)(d)
          }
        }
      }
    }
    out
  }

  def backward(t: Array[Array[Float]]): (Array[Array[Float]], Array[Array[Float]], Array[Array[Float]]) = {
    val dQ = Array.fill(seqLen)(Array.fill(t(0).length)(0.0f))
    val dK = Array.fill(seqLen)(Array.fill(t(0).length)(0.0f))
    val dV = Array.fill(seqLen)(Array.fill(t(0).length)(0.0f))
    (dQ, dK, dV)
  }
}

object SparseAttention:
  def fromLowerTriangular(seqLen: Int, blockSize: Int = 32): SparseAttention = {
    val mask = Array.tabulate(seqLen) { i =>
      Array.tabulate(seqLen) { j => j <= i }
    }
    new SparseAttention(mask, blockSize)
  }

/** Block sparse matrix with explicit block storage */
class BlockSparseMatrix(
  val blocks: Map[(Int, Int), Array[Float]],
  val blockSize: Int,
  val shape: (Int, Int)
) {
  private val numBlockRows = (shape._1 + blockSize - 1) / blockSize
  private val numBlockCols = (shape._2 + blockSize - 1) / blockSize

  def apply(i: Int, j: Int, localIdx: Int): Float = {
    blocks.get((i, j)).flatMap(b => if (localIdx < b.length) Some(b(localIdx)) else None).getOrElse(0.0f)
  }

  def update(i: Int, j: Int, localIdx: Int, value: Float): Unit = {
    val b = blocks.getOrElse((i, j), Array.fill(blockSize * blockSize)(0.0f))
    val updated = b.clone()
    if (localIdx < updated.length) updated(localIdx) = value
    blocks.updated((i, j), updated)
  }

  def nnzBlocks: Int = blocks.size
  def totalValues: Int = blocks.values.map(_.length).sum

  def toSparseMatrix: SparseMatrix = {
    val values = mutable.ListBuffer[Float]()
    val colIdx = mutable.ListBuffer[Int]()
    val rowPtr = Array.newBuilder[Int]
    rowPtr += 0

    for (blockRow <- 0 until numBlockRows) {
      for (blockCol <- 0 until numBlockCols) {
        blocks.get((blockRow, blockCol)).foreach { block =>
          for (i <- 0 until blockSize) {
            val row = blockRow * blockSize + i
            if (row < shape._1) {
              for (j <- 0 until blockSize) {
                val col = blockCol * blockSize + j
                if (col < shape._2) {
                  val v = block(i * blockSize + j)
                  if (v != 0.0f) {
                    values += v
                    colIdx += col
                  }
                }
              }
            }
          }
        }
      }
      rowPtr += values.size
    }
    new SparseMatrix(values.toArray, rowPtr.result(), colIdx.toArray, shape, SparseFormat.CSR)
  }

  def spmv(x: Array[Float]): Array[Float] = {
    val y = Array.fill(shape._1)(0.0f)
    for (blockRow <- 0 until numBlockRows) {
      for (blockCol <- 0 until numBlockCols) {
        blocks.get((blockRow, blockCol)).foreach { block =>
          for (i <- 0 until blockSize) {
            val row = blockRow * blockSize + i
            if (row < shape._1) {
              var sum = 0.0f
              for (j <- 0 until blockSize) {
                val col = blockCol * blockSize + j
                if (col < shape._2) {
                  sum += block(i * blockSize + j) * x(col)
                }
              }
              y(row) += sum
            }
          }
        }
      }
    }
    y
  }
}

object BlockSparseMatrix:
  def fromDense(dense: Array[Array[Float]], blockSize: Int, tol: Float = 0.0f): BlockSparseMatrix = {
    val rows = dense.length
    val cols = dense(0).length
    val blocks = mutable.Map[(Int, Int), Array[Float]]()

    for (blockRow <- 0 until ((rows + blockSize - 1) / blockSize)) {
      for (blockCol <- 0 until ((cols + blockSize - 1) / blockSize)) {
        val block = Array.fill(blockSize * blockSize)(0.0f)
        var hasNonZero = false
        for (i <- 0 until blockSize) {
          for (j <- 0 until blockSize) {
            val r = blockRow * blockSize + i
            val c = blockCol * blockSize + j
            if (r < rows && c < cols) {
              val v = dense(r)(c)
              block(i * blockSize + j) = v
              if (v.abs > tol) hasNonZero = true
            }
          }
        }
        if (hasNonZero) blocks((blockRow, blockCol)) = block
      }
    }
    new BlockSparseMatrix(blocks.toMap, blockSize, (rows, cols))
  }

  def identity(n: Int, blockSize: Int = 32): BlockSparseMatrix = {
    val blocks = mutable.Map[(Int, Int), Array[Float]]()
    val numBlocks = (n + blockSize - 1) / blockSize
    for (b <- 0 until numBlocks) {
      val block = Array.fill(blockSize * blockSize)(0.0f)
      for (i <- 0 until blockSize) {
        val r = b * blockSize + i
        val c = b * blockSize + i
        if (r < n && c < n) block(i * blockSize + i) = 1.0f
      }
      blocks((b, b)) = block
    }
    new BlockSparseMatrix(blocks.toMap, blockSize, (n, n))
  }

  def random(rows: Int, cols: Int, density: Double, blockSize: Int = 32, seed: Int = 42): BlockSparseMatrix = {
    val blocks = mutable.Map[(Int, Int), Array[Float]]()
    val numBlockRows = (rows + blockSize - 1) / blockSize
    val numBlockCols = (cols + blockSize - 1) / blockSize
    val rnd = new scala.util.Random(seed)
    val blockDensity = density * blockSize * blockSize / (blockSize * blockSize)

    for (br <- 0 until numBlockRows; bc <- 0 until numBlockCols) {
      if (rnd.nextDouble() < blockDensity) {
        val block = Array.tabulate(blockSize * blockSize)(_ => rnd.nextFloat() * 2 - 1)
        blocks((br, bc)) = block
      }
    }
    new BlockSparseMatrix(blocks.toMap, blockSize, (rows, cols))
  }
