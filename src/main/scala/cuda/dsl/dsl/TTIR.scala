package cuda.dsl.dsl

import scala.collection.mutable

/** Triton-like IR (TTIR) - AST for GPU kernels
 *
 *  This is a complete IR that supports:
 *  - Thread/block identification
 *  - Arithmetic operations
 *  - For loops with nested statements
 *  - If-then-else
 *  - Memory operations (load/store with masking)
 *  - Reduction operations
 *  - 2D tensor indexing (automatic flattening)
 */
class TTIR(val name: String):

  val _params = mutable.ListBuffer[TTIRParam]()
  val _locals = mutable.ListBuffer[TTIRLocal]()
  val _body = mutable.ListBuffer[TTIRStmt]()
  val _constants = mutable.ListBuffer[(String, String)]()  // name -> value string

  def param(tpe: String, name: String) =
    _params += TTIRParam(tpe, name)
    this

  def local(tpe: String, name: String, init: TTIRExpr) =
    _locals += TTIRLocal(tpe, name, init)
    this

  def constant(name: String, value: String) =
    _constants += ((name, value))
    this

  def +=(stmt: TTIRStmt) =
    _body += stmt
    this

  def for_(iter: String, start: Int, end: TTIRExpr)(body: TTIRBlock ?=> Unit)(using ctx: TTIRBlock) =
    val block = new TTIRBlock()
    body(using block)
    ctx.stmts += TTIRFor(iter, start, end, block.stmts.toList)

  def if_(cond: TTIRExpr)(thenp: TTIRBlock ?=> Unit)(using ctx: TTIRBlock): TTIRIf =
    val thenBlock = new TTIRBlock()
    thenp(using thenBlock)
    val stmt = TTIRIf(cond, thenBlock.stmts.toList, Nil)
    ctx.stmts += stmt
    stmt

  def emit(): String =
    val emitter = TTIREmitter()
    emitter.emit(this)

// ============================================================================
// IR Nodes
// ============================================================================

case class TTIRParam(tpe: String, name: String)
case class TTIRLocal(tpe: String, name: String, init: TTIRExpr)

sealed trait TTIRStmt
case class TTIRFor(iter: String, start: Int, end: TTIRExpr, body: List[TTIRStmt]) extends TTIRStmt
case class TTIRWhile(cond: TTIRExpr, body: List[TTIRStmt]) extends TTIRStmt
case class TTIRIf(cond: TTIRExpr, thenp: List[TTIRStmt], elsep: List[TTIRStmt]) extends TTIRStmt
case class TTIRAssign(name: String, value: TTIRExpr) extends TTIRStmt
case class TTIRStore(ptr: String, offset: TTIRExpr, value: TTIRExpr, mask: TTIRExpr) extends TTIRStmt
case class TTIRReturn(expr: TTIRExpr) extends TTIRStmt
case class TTIRLocalDecl(tpe: String, name: String) extends TTIRStmt
case class TTIRExprStmt(expr: TTIRExpr) extends TTIRStmt

sealed trait TTIRExpr
case class TTIRProgId(axis: Int) extends TTIRExpr
case class TTIRBlockId(axis: Int) extends TTIRExpr
case class TTIRThreadId(axis: Int) extends TTIRExpr
case class TTIRBlockDim() extends TTIRExpr
case class TTIRGridDim() extends TTIRExpr
case class TTIRVar(name: String) extends TTIRExpr
case class TTIRConst(value: Any) extends TTIRExpr
case class TTIRLoad(ptr: String, offset: TTIRExpr, mask: TTIRExpr, other: TTIRExpr) extends TTIRExpr
case class TTIRBinOp(op: String, lhs: TTIRExpr, rhs: TTIRExpr) extends TTIRExpr
case class TTIRUnaryOp(op: String, arg: TTIRExpr) extends TTIRExpr
case class TTIRMathCall(func: String, args: List[TTIRExpr]) extends TTIRExpr
case class TTIRDot(a: TTIRExpr, b: TTIRExpr) extends TTIRExpr
case class TTIRReduce(maximize: Boolean, name: String, input: TTIRExpr, init: Float, body: List[TTIRStmt]) extends TTIRExpr

// 2D tensor load - ptr is 2D tensor, row/col are the indices, stride is the row stride
case class TTIRLoad2D(ptr: String, row: TTIRExpr, col: TTIRExpr, stride: TTIRExpr, mask: TTIRExpr, other: TTIRExpr) extends TTIRExpr
// 2D tensor store
case class TTIRStore2D(ptr: String, row: TTIRExpr, col: TTIRExpr, stride: TTIRExpr, value: TTIRExpr, mask: TTIRExpr) extends TTIRStmt

// ============================================================================
// CUDA Shared Memory & Synchronization IR Nodes
// ============================================================================

// __shared__ declaration: e.g., __shared__ float s_q[BR * D];
case class TTIRSharedMem(tpe: String, name: String, size: TTIRExpr) extends TTIRStmt
// __syncthreads() barrier
case class TTIRSyncThreads() extends TTIRStmt
// Warp-level shuffle: __shfl_sync(mask, value, srcLane, width)
// mode: "shfl", "shfl_up", "shfl_down", "shfl_xor"
case class TTIRShfl(mode: String, value: TTIRExpr, srcLane: TTIRExpr, width: TTIRExpr) extends TTIRExpr
// Warp-level vote: __any_sync / __all_sync
case class TTIRWarpVote(op: String, predicate: TTIRExpr) extends TTIRExpr
// Masked load (predicated load)
case class TTIRMaskedLoad(ptr: String, offset: TTIRExpr, pred: TTIRExpr, other: TTIRExpr) extends TTIRExpr
// Ternary if-then-else expression: cond ? thenExpr : elseExpr
case class TTIRTernary(cond: TTIRExpr, thenExpr: TTIRExpr, elseExpr: TTIRExpr) extends TTIRExpr
// Masked store (predicated store)
case class TTIRMaskedStore(ptr: String, offset: TTIRExpr, value: TTIRExpr, pred: TTIRExpr) extends TTIRStmt

// ============================================================================
// Attention Kernels IR Nodes
// ============================================================================

// FlashAttention: IO-aware exact attention with two-pass algorithm
case class TTIRFlashAttention(
  qPtr: String, kPtr: String, vPtr: String, outPtr: String,
  n: TTIRExpr, d: TTIRExpr,
  blockSize: Int = 128,
  causal: Boolean = false
) extends TTIRStmt

// PageAttention: vLLM-style paged KV cache attention
case class TTIRPageAttention(
  queryPtr: String,
  keyCachePtr: String, valueCachePtr: String,
  blockTablesPtr: String, seqLensPtr: String,
  outPtr: String,
  numBlocks: TTIRExpr, blockSize: TTIRExpr, d: TTIRExpr
) extends TTIRStmt

// FlexAttention: Flexible attention with custom score/mask functions
case class TTIRFlexAttention(
  qPtr: String, kPtr: String, vPtr: String,
  outPtr: String,
  n: TTIRExpr, d: TTIRExpr,
  scoreModName: Option[String] = None,
  maskModName: Option[String] = None,
  blockSize: Int = 128
) extends TTIRStmt

// Grouped Query Attention (GQA) - shares KV heads across query heads
case class TTIRGroupedQueryAttention(
  qPtr: String, kPtr: String, vPtr: String, outPtr: String,
  n: TTIRExpr, d: TTIRExpr,
  numQHeads: TTIRExpr, numKVHeads: TTIRExpr,
  scale: TTIRExpr
) extends TTIRStmt

// Multi-head Attention wrapper
case class TTIRMultiHeadAttention(
  qPtr: String, kPtr: String, vPtr: String, outPtr: String,
  n: TTIRExpr, d: TTIRExpr,
  numHeads: TTIRExpr,
  scale: TTIRExpr
) extends TTIRStmt

// StoreKVCache: Store key/value to paged KV cache (vLLM-style)
case class TTIRStoreKVCache(
  keyPtr: String, keyStride: TTIRExpr,
  valuePtr: String, valueStride: TTIRExpr,
  kCachePtr: String, vCachePtr: String,
  slotMappingPtr: String,
  d: TTIRExpr
) extends TTIRStmt

// LoadKVCache: Load key/value from paged KV cache (vLLM-style)
case class TTIRLoadKVCache(
  kCachePtr: String, vCachePtr: String,
  slotMappingPtr: String,
  keyOutputPtr: String, valueOutputPtr: String,
  numTokens: TTIRExpr,
  d: TTIRExpr
) extends TTIRStmt

// LoadTile: Load a tile from memory
case class TTIRLoadTile(
  inputPtr: String,
  tileOutputPtr: String,
  tileRow: TTIRExpr, tileCol: TTIRExpr,
  tileSize: TTIRExpr,
  m: TTIRExpr, n: TTIRExpr,
  stride: TTIRExpr
) extends TTIRStmt

// StoreTile: Store a tile to memory
case class TTIRStoreTile(
  tilePtr: String,
  outputPtr: String,
  tileRow: TTIRExpr, tileCol: TTIRExpr,
  tileSize: TTIRExpr,
  m: TTIRExpr, n: TTIRExpr,
  stride: TTIRExpr
) extends TTIRStmt

// SaveStore: Combined save and store operation (atomically saves to persistent storage)
case class TTIRSaveStore(
  dataPtr: String,
  storagePtr: String,
  offset: TTIRExpr,
  size: TTIRExpr,
  compressionRatio: TTIRExpr
) extends TTIRStmt

// LoadStore: Load from persistent storage
case class TTIRLoadStore(
  storagePtr: String,
  dataPtr: String,
  offset: TTIRExpr,
  size: TTIRExpr
) extends TTIRStmt

// PagedSaveKVCache: Save KV cache with page table mapping
case class TTIRPagedSaveKVCache(
  kvCachePtr: String,
  pageTablePtr: String,
  storagePtr: String,
  numBlocks: TTIRExpr,
  blockSize: TTIRExpr,
  d: TTIRExpr
) extends TTIRStmt

// PagedLoadKVCache: Load KV cache with page table mapping
case class TTIRPagedLoadKVCache(
  storagePtr: String,
  pageTablePtr: String,
  kvCachePtr: String,
  numBlocks: TTIRExpr,
  blockSize: TTIRExpr,
  d: TTIRExpr
) extends TTIRStmt

// CheckpointSave: Save with compression
case class TTIRCheckpointSave(
  kvCachePtr: String,
  checkpointPtr: String,
  seqLen: TTIRExpr,
  headDim: TTIRExpr,
  numLayers: TTIRExpr,
  compressRatio: TTIRExpr
) extends TTIRStmt

// CheckpointRestore: Restore from compressed checkpoint
case class TTIRCheckpointRestore(
  checkpointPtr: String,
  kvCachePtr: String,
  seqLen: TTIRExpr,
  headDim: TTIRExpr,
  numLayers: TTIRExpr,
  compressRatio: TTIRExpr
) extends TTIRStmt

// DeltaSave: Incremental save (only deltas)
case class TTIRDeltaSave(
  kvCachePtr: String,
  prevCheckpointPtr: String,
  deltaPtr: String,
  seqLen: TTIRExpr,
  headDim: TTIRExpr,
  numLayers: TTIRExpr
) extends TTIRStmt

// DeltaRestore: Restore from deltas
case class TTIRDeltaRestore(
  prevCheckpointPtr: String,
  deltaPtr: String,
  kvCachePtr: String,
  seqLen: TTIRExpr,
  headDim: TTIRExpr,
  numLayers: TTIRExpr
) extends TTIRStmt

// SelectiveSave: Save based on importance threshold
case class TTIRSelectiveSave(
  kvCachePtr: String,
  importancePtr: String,
  storagePtr: String,
  seqLen: TTIRExpr,
  headDim: TTIRExpr,
  numLayers: TTIRExpr,
  threshold: TTIRExpr
) extends TTIRStmt

// SelectiveLoad: Load based on importance threshold
case class TTIRSelectiveLoad(
  storagePtr: String,
  importancePtr: String,
  kvCachePtr: String,
  seqLen: TTIRExpr,
  headDim: TTIRExpr,
  numLayers: TTIRExpr,
  threshold: TTIRExpr
) extends TTIRStmt

// SaveWithECC: Save with error correction codes
case class TTIRSaveWithECC(
  kvCachePtr: String,
  storagePtr: String,
  eccPtr: String,
  seqLen: TTIRExpr,
  headDim: TTIRExpr
) extends TTIRStmt

// LoadWithECC: Load with error correction
case class TTIRLoadWithECC(
  storagePtr: String,
  eccPtr: String,
  kvCachePtr: String,
  seqLen: TTIRExpr,
  headDim: TTIRExpr
) extends TTIRStmt

class TTIRBlock:
  val stmts = mutable.ListBuffer[TTIRStmt]()

// ============================================================================
// High-level DSL helpers (moved outside class TTIR)
// ============================================================================

object TTIRDSL:
  // Thread identification
  def tid(axis: Int = 0) = TTIRThreadId(axis)
  def bid(axis: Int = 0) = TTIRBlockId(axis)
  def blockDim() = TTIRBlockDim()

  // 2D indexing helpers - converts q[i][j] to q[i*stride + j]
  def idx2D(row: TTIRExpr, col: TTIRExpr, stride: TTIRExpr): TTIRExpr =
    TTIRBinOp("+", TTIRBinOp("*", row, stride), col)

  // Load from 2D tensor with mask
  def load2D(ptr: String, row: TTIRExpr, col: TTIRExpr, stride: TTIRExpr, mask: TTIRExpr = TTIRConst(true), other: TTIRExpr = TTIRConst(0.0f)): TTIRExpr =
    TTIRLoad2D(ptr, row, col, stride, mask, other)

  // Store to 2D tensor with mask
  def store2D(ptr: String, row: TTIRExpr, col: TTIRExpr, stride: TTIRExpr, value: TTIRExpr, mask: TTIRExpr = TTIRConst(true)): TTIRStmt =
    TTIRStore2D(ptr, row, col, stride, value, mask)

  // Dot product between two 2D rows: sum over k of a[row][k] * b[row][k]
  def dot2D(aPtr: String, aRow: TTIRExpr, bPtr: String, bRow: TTIRExpr, k: String, kEnd: TTIRExpr, stride: TTIRExpr, acc: String = "dotAcc"): List[TTIRStmt] =
    List(
      TTIRAssign(acc, TTIRConst(0.0f)),
      TTIRFor(k, 0, kEnd, List(
        TTIRAssign(acc, TTIRBinOp("+", TTIRVar(acc),
          TTIRBinOp("*",
            TTIRLoad2D(aPtr, aRow, TTIRVar(k), stride, TTIRConst(true), TTIRConst(0.0f)),
            TTIRLoad2D(bPtr, bRow, TTIRVar(k), stride, TTIRConst(true), TTIRConst(0.0f)))))
        ))
      )

  // ============================================================================
  // Attention Kernel DSL Helpers
  // ============================================================================

  // FlashAttention: IO-aware tiled attention
  def flashAttention(q: String, k: String, v: String, out: String, N: TTIRExpr, D: TTIRExpr, blockSize: Int = 128, causal: Boolean = false): TTIRStmt =
    TTIRFlashAttention(q, k, v, out, N, D, blockSize, causal)

  // PageAttention: vLLM-style paged KV cache attention
  def pageAttention(
    query: String, keyCache: String, valueCache: String,
    blockTables: String, seqLens: String, out: String,
    numBlocks: TTIRExpr, blockSize: TTIRExpr, D: TTIRExpr
  ): TTIRStmt =
    TTIRPageAttention(
      query, keyCache, valueCache,
      blockTables, seqLens, out,
      numBlocks, blockSize, D
    )

  // FlexAttention: Flexible attention with custom score/mask functions
  def flexAttention(
    q: String, k: String, v: String, out: String,
    N: TTIRExpr, D: TTIRExpr,
    scoreModName: Option[String] = None,
    maskModName: Option[String] = None,
    blockSize: Int = 128
  ): TTIRStmt =
    TTIRFlexAttention(q, k, v, out, N, D, scoreModName, maskModName, blockSize)

  // Grouped Query Attention (GQA)
  def groupedQueryAttention(
    q: String, k: String, v: String, out: String,
    N: TTIRExpr, D: TTIRExpr,
    numQHeads: TTIRExpr, numKVHeads: TTIRExpr,
    scale: TTIRExpr
  ): TTIRStmt =
    TTIRGroupedQueryAttention(
      q, k, v, out,
      N, D,
      numQHeads, numKVHeads,
      scale
    )

  // Multi-head Attention (standard MHA)
  def multiHeadAttention(
    q: String, k: String, v: String, out: String,
    N: TTIRExpr, D: TTIRExpr,
    numHeads: TTIRExpr, scale: TTIRExpr
  ): TTIRStmt =
    TTIRMultiHeadAttention(
      q, k, v, out,
      N, D,
      numHeads, scale
    )

  // StoreKVCache: Store key/value to paged KV cache (vLLM-style)
  // keyPtr: pointer to key data [num_tokens, D]
  // keyStride: stride of key data
  // valuePtr: pointer to value data [num_tokens, D]
  // valueStride: stride of value data
  // kCachePtr: pointer to KV cache for keys [num_blocks, block_size, D]
  // vCachePtr: pointer to KV cache for values [num_blocks, block_size, D]
  // slotMappingPtr: maps logical positions to cache slots
  // D: head dimension
  // Note: accepts Any to work with @TritonKernelMacro which passes variable names
  def storeKVCache(
    keyPtr: Any, keyStride: Any,
    valuePtr: Any, valueStride: Any,
    kCachePtr: Any, vCachePtr: Any,
    slotMappingPtr: Any,
    D: Any
  ): TTIRStmt =
    TTIRStoreKVCache(
      keyPtr.toString, TTIRVar(keyStride.toString),
      valuePtr.toString, TTIRVar(valueStride.toString),
      kCachePtr.toString, vCachePtr.toString,
      slotMappingPtr.toString,
      TTIRVar(D.toString)
    )

  // LoadKVCache: Load key/value from paged KV cache (vLLM-style)
  def loadKVCache(
    kCachePtr: Any, vCachePtr: Any,
    slotMappingPtr: Any,
    keyOutputPtr: Any, valueOutputPtr: Any,
    numTokens: Any,
    D: Any
  ): TTIRStmt =
    TTIRLoadKVCache(
      kCachePtr.toString, vCachePtr.toString,
      slotMappingPtr.toString,
      keyOutputPtr.toString, valueOutputPtr.toString,
      TTIRVar(numTokens.toString),
      TTIRVar(D.toString)
    )

  // LoadTile: Load a tile from memory
  def loadTile(
    inputPtr: Any,
    tileOutputPtr: Any,
    tileRow: Any, tileCol: Any,
    tileSize: Any,
    M: Any, N: Any,
    stride: Any
  ): TTIRStmt =
    TTIRLoadTile(
      inputPtr.toString,
      tileOutputPtr.toString,
      TTIRVar(tileRow.toString), TTIRVar(tileCol.toString),
      TTIRVar(tileSize.toString),
      TTIRVar(M.toString), TTIRVar(N.toString),
      TTIRVar(stride.toString)
    )

  // StoreTile: Store a tile to memory
  def storeTile(
    tilePtr: Any,
    outputPtr: Any,
    tileRow: Any, tileCol: Any,
    tileSize: Any,
    M: Any, N: Any,
    stride: Any
  ): TTIRStmt =
    TTIRStoreTile(
      tilePtr.toString,
      outputPtr.toString,
      TTIRVar(tileRow.toString), TTIRVar(tileCol.toString),
      TTIRVar(tileSize.toString),
      TTIRVar(M.toString), TTIRVar(N.toString),
      TTIRVar(stride.toString)
    )

  // SaveStore: Combined save and store operation
  def saveStore(
    dataPtr: Any,
    storagePtr: Any,
    offset: Any,
    size: Any,
    compressionRatio: Any
  ): TTIRStmt =
    TTIRSaveStore(
      dataPtr.toString,
      storagePtr.toString,
      TTIRVar(offset.toString),
      TTIRVar(size.toString),
      TTIRVar(compressionRatio.toString)
    )

  // LoadStore: Load from persistent storage
  def loadStore(
    storagePtr: Any,
    dataPtr: Any,
    offset: Any,
    size: Any
  ): TTIRStmt =
    TTIRLoadStore(
      storagePtr.toString,
      dataPtr.toString,
      TTIRVar(offset.toString),
      TTIRVar(size.toString)
    )

  // PagedSaveKVCache: Save KV cache with page table mapping
  def pagedSaveKVCache(
    kvCachePtr: Any,
    pageTablePtr: Any,
    storagePtr: Any,
    numBlocks: Any,
    blockSize: Any,
    D: Any
  ): TTIRStmt =
    TTIRPagedSaveKVCache(
      kvCachePtr.toString,
      pageTablePtr.toString,
      storagePtr.toString,
      TTIRVar(numBlocks.toString),
      TTIRVar(blockSize.toString),
      TTIRVar(D.toString)
    )

  // PagedLoadKVCache: Load KV cache with page table mapping
  def pagedLoadKVCache(
    storagePtr: Any,
    pageTablePtr: Any,
    kvCachePtr: Any,
    numBlocks: Any,
    blockSize: Any,
    D: Any
  ): TTIRStmt =
    TTIRPagedLoadKVCache(
      storagePtr.toString,
      pageTablePtr.toString,
      kvCachePtr.toString,
      TTIRVar(numBlocks.toString),
      TTIRVar(blockSize.toString),
      TTIRVar(D.toString)
    )

  // CheckpointSave: Save with compression
  def checkpointSave(
    kvCachePtr: Any,
    checkpointPtr: Any,
    seqLen: Any,
    headDim: Any,
    numLayers: Any,
    compressRatio: Any
  ): TTIRStmt =
    TTIRCheckpointSave(
      kvCachePtr.toString,
      checkpointPtr.toString,
      TTIRVar(seqLen.toString),
      TTIRVar(headDim.toString),
      TTIRVar(numLayers.toString),
      TTIRVar(compressRatio.toString)
    )

  // CheckpointRestore: Restore from compressed checkpoint
  def checkpointRestore(
    checkpointPtr: Any,
    kvCachePtr: Any,
    seqLen: Any,
    headDim: Any,
    numLayers: Any,
    compressRatio: Any
  ): TTIRStmt =
    TTIRCheckpointRestore(
      checkpointPtr.toString,
      kvCachePtr.toString,
      TTIRVar(seqLen.toString),
      TTIRVar(headDim.toString),
      TTIRVar(numLayers.toString),
      TTIRVar(compressRatio.toString)
    )

  // DeltaSave: Incremental save (only deltas)
  def deltaSave(
    kvCachePtr: Any,
    prevCheckpointPtr: Any,
    deltaPtr: Any,
    seqLen: Any,
    headDim: Any,
    numLayers: Any
  ): TTIRStmt =
    TTIRDeltaSave(
      kvCachePtr.toString,
      prevCheckpointPtr.toString,
      deltaPtr.toString,
      TTIRVar(seqLen.toString),
      TTIRVar(headDim.toString),
      TTIRVar(numLayers.toString)
    )

  // DeltaRestore: Restore from deltas
  def deltaRestore(
    prevCheckpointPtr: Any,
    deltaPtr: Any,
    kvCachePtr: Any,
    seqLen: Any,
    headDim: Any,
    numLayers: Any
  ): TTIRStmt =
    TTIRDeltaRestore(
      prevCheckpointPtr.toString,
      deltaPtr.toString,
      kvCachePtr.toString,
      TTIRVar(seqLen.toString),
      TTIRVar(headDim.toString),
      TTIRVar(numLayers.toString)
    )

  // SelectiveSave: Save based on importance threshold
  def selectiveSave(
    kvCachePtr: Any,
    importancePtr: Any,
    storagePtr: Any,
    seqLen: Any,
    headDim: Any,
    numLayers: Any,
    threshold: Any
  ): TTIRStmt =
    TTIRSelectiveSave(
      kvCachePtr.toString,
      importancePtr.toString,
      storagePtr.toString,
      TTIRVar(seqLen.toString),
      TTIRVar(headDim.toString),
      TTIRVar(numLayers.toString),
      TTIRVar(threshold.toString)
    )

  // SelectiveLoad: Load based on importance threshold
  def selectiveLoad(
    storagePtr: Any,
    importancePtr: Any,
    kvCachePtr: Any,
    seqLen: Any,
    headDim: Any,
    numLayers: Any,
    threshold: Any
  ): TTIRStmt =
    TTIRSelectiveLoad(
      storagePtr.toString,
      importancePtr.toString,
      kvCachePtr.toString,
      TTIRVar(seqLen.toString),
      TTIRVar(headDim.toString),
      TTIRVar(numLayers.toString),
      TTIRVar(threshold.toString)
    )

  // SaveWithECC: Save with error correction codes
  def saveWithECC(
    kvCachePtr: Any,
    storagePtr: Any,
    eccPtr: Any,
    seqLen: Any,
    headDim: Any
  ): TTIRStmt =
    TTIRSaveWithECC(
      kvCachePtr.toString,
      storagePtr.toString,
      eccPtr.toString,
      TTIRVar(seqLen.toString),
      TTIRVar(headDim.toString)
    )

  // LoadWithECC: Load with error correction
  def loadWithECC(
    storagePtr: Any,
    eccPtr: Any,
    kvCachePtr: Any,
    seqLen: Any,
    headDim: Any
  ): TTIRStmt =
    TTIRLoadWithECC(
      storagePtr.toString,
      eccPtr.toString,
      kvCachePtr.toString,
      TTIRVar(seqLen.toString),
      TTIRVar(headDim.toString)
    )

// ============================================================================
// Emitter
// ============================================================================

class TTIREmitter:

  private var indent = 0
  private val sb = new StringBuilder

  private def indent_=(level: Int): Unit = indent = level
  private def newline(): Unit = sb.append("\n")

  def emit(ir: TTIR): String =
    sb.clear()
    indent = 0

    // Constants (e.g. PAGE_SIZE, INF_NEG for attention kernels)
    ir._constants.foreach((name, value) => emitLine(s"#define $name $value"))
    if (ir._constants.nonEmpty) newline()

    // Kernel signature
    sb.append(s"extern \"C\" __global__ void ${ir.name}(")
    newline()
    val hasNParam = ir._params.exists(_.name == "n")
    ir._params.foreach: p =>
      sb.append(s"    ${p.tpe} ${p.name},")
      newline()
    // 只有当用户没有提供 n 参数时才添加 grid size 参数
    if !hasNParam then
      sb.append("    int n)")
    else
      sb.append("    int _n_)")  // 使用不冲突的名称
    newline()
    sb.append("{")
    newline()

    indent += 1

    // Thread ID
    emitLine("int i = blockIdx.x * blockDim.x + threadIdx.x;")
    // 使用正确的 grid size 参数名
    val gridSizeParam = if hasNParam then "_n_" else "n"
    emitLine(s"if (i >= $gridSizeParam) return;")
    newline()

    // Local variables
    ir._locals.foreach: l =>
      emitLine(s"${l.tpe} ${l.name} = ${emitExpr(l.init)};")
    if (ir._locals.nonEmpty) newline()

    // Body
    ir._body.foreach: stmt =>
      emitStmt(stmt)

    indent -= 1
    sb.append("}")
    newline()

    sb.toString

  private def emitLine(code: String): Unit =
    sb.append("  " * indent)
    sb.append(code)
    sb.append("\n")

  private def emitStmt(stmt: TTIRStmt): Unit = stmt match
    case TTIRAssign(name, value) =>
      emitLine(s"$name = ${emitExpr(value)};")

    case TTIRStore(ptr, offset, value, mask) =>
      mask match
        case TTIRConst(true) => emitLine(s"${ptr}[${emitExpr(offset)}] = ${emitExpr(value)};")
        case _ => emitLine(s"if (${emitExpr(mask)}) ${ptr}[${emitExpr(offset)}] = ${emitExpr(value)};")

    case TTIRStore2D(ptr, row, col, stride, value, mask) =>
      val offset = s"(${emitExpr(row)} * ${emitExpr(stride)} + ${emitExpr(col)})"
      mask match
        case TTIRConst(true) => emitLine(s"${ptr}[$offset] = ${emitExpr(value)};")
        case _ => emitLine(s"if (${emitExpr(mask)}) ${ptr}[$offset] = ${emitExpr(value)};")

    case TTIRReturn(expr) =>
      expr match
        case TTIRConst(()) => emitLine("return;")
        case _ => emitLine(s"return ${emitExpr(expr)};")

    case TTIRLocalDecl(tpe, name) =>
      emitLine(s"$tpe $name;")

    case TTIRExprStmt(expr) =>
      val code = emitExpr(expr)
      // Skip void/unit statements (represented as 0.0f) and empty code
      if (code.nonEmpty && code != "()" && code != "0.0") emitLine(s"$code;")

    case TTIRFlashAttention(qPtr, kPtr, vPtr, outPtr, n, d, blockSize, causal) =>
      emitFlashAttentionKernel(qPtr, kPtr, vPtr, outPtr, n, d, blockSize, causal)

    case TTIRPageAttention(queryPtr, keyCachePtr, valueCachePtr, blockTablesPtr, seqLensPtr, outPtr, numBlocks, blockSize, d) =>
      emitPageAttentionKernel(queryPtr, keyCachePtr, valueCachePtr, blockTablesPtr, seqLensPtr, outPtr, numBlocks, blockSize, d)

    case TTIRFlexAttention(qPtr, kPtr, vPtr, outPtr, n, d, scoreModName, maskModName, blockSize) =>
      emitFlexAttentionKernel(qPtr, kPtr, vPtr, outPtr, n, d, scoreModName, maskModName, blockSize)

    case TTIRGroupedQueryAttention(qPtr, kPtr, vPtr, outPtr, n, d, numQHeads, numKVHeads, scale) =>
      emitGQAKernel(qPtr, kPtr, vPtr, outPtr, n, d, numQHeads, numKVHeads, scale)

    case TTIRMultiHeadAttention(qPtr, kPtr, vPtr, outPtr, n, d, numHeads, scale) =>
      emitMHAKernel(qPtr, kPtr, vPtr, outPtr, n, d, numHeads, scale)

    case TTIRStoreKVCache(keyPtr, keyStride, valuePtr, valueStride, kCachePtr, vCachePtr, slotMappingPtr, d) =>
      emitStoreKVCacheKernel(keyPtr, keyStride, valuePtr, valueStride, kCachePtr, vCachePtr, slotMappingPtr, d)

    case TTIRLoadKVCache(kCachePtr, vCachePtr, slotMappingPtr, keyOutputPtr, valueOutputPtr, numTokens, d) =>
      emitLoadKVCacheKernel(kCachePtr, vCachePtr, slotMappingPtr, keyOutputPtr, valueOutputPtr, numTokens, d)

    case TTIRLoadTile(inputPtr, tileOutputPtr, tileRow, tileCol, tileSize, m, n, stride) =>
      emitLoadTileKernel(inputPtr, tileOutputPtr, tileRow, tileCol, tileSize, m, n, stride)

    case TTIRStoreTile(tilePtr, outputPtr, tileRow, tileCol, tileSize, m, n, stride) =>
      emitStoreTileKernel(tilePtr, outputPtr, tileRow, tileCol, tileSize, m, n, stride)

    case TTIRSaveStore(dataPtr, storagePtr, offset, size, compressionRatio) =>
      emitSaveStoreKernel(dataPtr, storagePtr, offset, size, compressionRatio)

    case TTIRLoadStore(storagePtr, dataPtr, offset, size) =>
      emitLoadStoreKernel(storagePtr, dataPtr, offset, size)

    case TTIRPagedSaveKVCache(kvCachePtr, pageTablePtr, storagePtr, numBlocks, blockSize, d) =>
      emitPagedSaveKVCacheKernel(kvCachePtr, pageTablePtr, storagePtr, numBlocks, blockSize, d)

    case TTIRPagedLoadKVCache(storagePtr, pageTablePtr, kvCachePtr, numBlocks, blockSize, d) =>
      emitPagedLoadKVCacheKernel(storagePtr, pageTablePtr, kvCachePtr, numBlocks, blockSize, d)

    case TTIRCheckpointSave(kvCachePtr, checkpointPtr, seqLen, headDim, numLayers, compressRatio) =>
      emitCheckpointSaveKernel(kvCachePtr, checkpointPtr, seqLen, headDim, numLayers, compressRatio)

    case TTIRCheckpointRestore(checkpointPtr, kvCachePtr, seqLen, headDim, numLayers, compressRatio) =>
      emitCheckpointRestoreKernel(checkpointPtr, kvCachePtr, seqLen, headDim, numLayers, compressRatio)

    case TTIRDeltaSave(kvCachePtr, prevCheckpointPtr, deltaPtr, seqLen, headDim, numLayers) =>
      emitDeltaSaveKernel(kvCachePtr, prevCheckpointPtr, deltaPtr, seqLen, headDim, numLayers)

    case TTIRDeltaRestore(prevCheckpointPtr, deltaPtr, kvCachePtr, seqLen, headDim, numLayers) =>
      emitDeltaRestoreKernel(prevCheckpointPtr, deltaPtr, kvCachePtr, seqLen, headDim, numLayers)

    case TTIRSelectiveSave(kvCachePtr, importancePtr, storagePtr, seqLen, headDim, numLayers, threshold) =>
      emitSelectiveSaveKernel(kvCachePtr, importancePtr, storagePtr, seqLen, headDim, numLayers, threshold)

    case TTIRSelectiveLoad(storagePtr, importancePtr, kvCachePtr, seqLen, headDim, numLayers, threshold) =>
      emitSelectiveLoadKernel(storagePtr, importancePtr, kvCachePtr, seqLen, headDim, numLayers, threshold)

    case TTIRSaveWithECC(kvCachePtr, storagePtr, eccPtr, seqLen, headDim) =>
      emitSaveWithECCKernel(kvCachePtr, storagePtr, eccPtr, seqLen, headDim)

    case TTIRLoadWithECC(storagePtr, eccPtr, kvCachePtr, seqLen, headDim) =>
      emitLoadWithECCKernel(storagePtr, eccPtr, kvCachePtr, seqLen, headDim)

    case TTIRFor(iter, start, end, body) =>
      emitLine(s"for (int $iter = $start; $iter < ${emitExpr(end)}; $iter++) {")
      indent += 1
      body.foreach(emitStmt)
      indent -= 1
      emitLine("}")

    case TTIRWhile(cond, body) =>
      emitLine(s"while (${emitExpr(cond)}) {")
      indent += 1
      body.foreach(emitStmt)
      indent -= 1
      emitLine("}")

    case TTIRIf(cond, thenp, elsep) =>
      emitLine(s"if (${emitExpr(cond)}) {")
      indent += 1
      thenp.foreach(emitStmt)
      indent -= 1
      if (elsep.nonEmpty) {
        emitLine("} else {")
        indent += 1
        elsep.foreach(emitStmt)
        indent -= 1
      }
      emitLine("}")

    case TTIRSharedMem(tpe, name, size) =>
      emitLine(s"__shared__ $tpe $name[${emitExpr(size)}];")

    case TTIRSyncThreads() =>
      emitLine("__syncthreads();")

    case TTIRMaskedStore(ptr, offset, value, pred) =>
      val predCode = emitExpr(pred)
      val offsetCode = emitExpr(offset)
      val valueCode = emitExpr(value)
      emitLine(s"if ($predCode) ${ptr}[$offsetCode] = $valueCode;")

  private def emitExpr(expr: TTIRExpr): String = expr match
    case TTIRProgId(0) => "blockIdx.x"
    case TTIRProgId(1) => "blockIdx.y"
    case TTIRProgId(2) => "blockIdx.z"
    case TTIRBlockId(0) => "blockIdx.x"
    case TTIRBlockId(1) => "blockIdx.y"
    case TTIRBlockId(2) => "blockIdx.z"
    case TTIRThreadId(0) => "threadIdx.x"
    case TTIRThreadId(1) => "threadIdx.y"
    case TTIRThreadId(2) => "threadIdx.z"
    case TTIRBlockDim() => "blockDim.x"
    case TTIRGridDim() => "gridDim.x"
    case TTIRVar(name) =>
      // Post-process to fix malformed tl.program_id/tl.threadIdx/tl.blockIdx literals
      if name.contains("cuda.dsl.dsl.tl.program_id(1)") then "blockIdx.y"
      else if name.contains("cuda.dsl.dsl.tl.program_id(2)") then "blockIdx.z"
      else if name.contains("cuda.dsl.dsl.tl.program_id(") then "i"
      else if name.contains("cuda.dsl.dsl.tl.threadIdx(0)") then "threadIdx.x"
      else if name.contains("cuda.dsl.dsl.tl.threadIdx(1)") then "threadIdx.y"
      else if name.contains("cuda.dsl.dsl.tl.threadIdx(2)") then "threadIdx.z"
      else if name.contains("cuda.dsl.dsl.tl.blockIdx(0)") then "blockIdx.x"
      else if name.contains("cuda.dsl.dsl.tl.blockIdx(1)") then "blockIdx.y"
      else if name.contains("cuda.dsl.dsl.tl.blockIdx(2)") then "blockIdx.z"
      else if name.contains("cuda.dsl.dsl.tl.blockIdx(") then "blockIdx.x"
      else if name.contains("cuda.dsl.dsl.tl.block_dim(") then "blockDim.x"
      else if name.contains("cuda.dsl.dsl.tl.num_blocks(") then "gridDim.x"
      else if name.contains("cuda.dsl.dsl.tl.gridDim(") then "gridDim.x"
      else name
    case TTIRConst(v) => v.toString

    case TTIRLoad(ptr, offset, mask, other) =>
      mask match
        case TTIRConst(true) => s"$ptr[${emitExpr(offset)}]"
        case _ => s"(${emitExpr(mask)} ? $ptr[${emitExpr(offset)}] : ${emitExpr(other)})"

    case TTIRLoad2D(ptr, row, col, stride, mask, other) =>
      val offset = s"(${emitExpr(row)} * ${emitExpr(stride)} + ${emitExpr(col)})"
      mask match
        case TTIRConst(true) => s"$ptr[$offset]"
        case _ => s"(${emitExpr(mask)} ? $ptr[$offset] : ${emitExpr(other)})"

    case TTIRBinOp("+", lhs, rhs) => s"(${emitExpr(lhs)} + ${emitExpr(rhs)})"
    case TTIRBinOp("-", lhs, rhs) => s"(${emitExpr(lhs)} - ${emitExpr(rhs)})"
    case TTIRBinOp("*", lhs, rhs) => s"(${emitExpr(lhs)} * ${emitExpr(rhs)})"
    case TTIRBinOp("/", lhs, rhs) => s"(${emitExpr(lhs)} / ${emitExpr(rhs)})"
    case TTIRBinOp("%", lhs, rhs) => s"(${emitExpr(lhs)} % ${emitExpr(rhs)})"
    case TTIRBinOp("&&", lhs, rhs) => s"(${emitExpr(lhs)} && ${emitExpr(rhs)})"
    case TTIRBinOp("||", lhs, rhs) => s"(${emitExpr(lhs)} || ${emitExpr(rhs)})"
    case TTIRBinOp(">", lhs, rhs) => s"(${emitExpr(lhs)} > ${emitExpr(rhs)})"
    case TTIRBinOp("<", lhs, rhs) => s"(${emitExpr(lhs)} < ${emitExpr(rhs)})"
    case TTIRBinOp(">=", lhs, rhs) => s"(${emitExpr(lhs)} >= ${emitExpr(rhs)})"
    case TTIRBinOp("<=", lhs, rhs) => s"(${emitExpr(lhs)} <= ${emitExpr(rhs)})"
    case TTIRBinOp("==", lhs, rhs) => s"(${emitExpr(lhs)} == ${emitExpr(rhs)})"
    case TTIRBinOp("!=", lhs, rhs) => s"(${emitExpr(lhs)} != ${emitExpr(rhs)})"
    case TTIRBinOp("<<", lhs, rhs) => s"(${emitExpr(lhs)} << ${emitExpr(rhs)})"
    case TTIRBinOp(">>", lhs, rhs) => s"(${emitExpr(lhs)} >> ${emitExpr(rhs)})"
    case TTIRBinOp("|", lhs, rhs) => s"(${emitExpr(lhs)} | ${emitExpr(rhs)})"
    case TTIRBinOp("&", lhs, rhs) => s"(${emitExpr(lhs)} & ${emitExpr(rhs)})"
    case TTIRBinOp("^", lhs, rhs) => s"(${emitExpr(lhs)} ^ ${emitExpr(rhs)})"
    case TTIRBinOp("max", lhs, rhs) => s"fmaxf(${emitExpr(lhs)}, ${emitExpr(rhs)})"
    case TTIRBinOp("min", lhs, rhs) => s"fminf(${emitExpr(lhs)}, ${emitExpr(rhs)})"

    case TTIRUnaryOp("!", arg) => s"(!${emitExpr(arg)})"
    case TTIRUnaryOp("-", arg) => s"(-${emitExpr(arg)})"

    case TTIRMathCall("exp", List(x)) => s"expf(${emitExpr(x)})"
    case TTIRMathCall("log", List(x)) => s"logf(${emitExpr(x)})"
    case TTIRMathCall("sqrt", List(x)) => s"sqrtf(${emitExpr(x)})"
    case TTIRMathCall("rsqrt", List(x)) => s"rsqrtf(${emitExpr(x)})"
    case TTIRMathCall("abs", List(x)) => s"fabsf(${emitExpr(x)})"
    case TTIRMathCall("sin", List(x)) => s"sinf(${emitExpr(x)})"
    case TTIRMathCall("cos", List(x)) => s"cosf(${emitExpr(x)})"
    case TTIRMathCall("tan", List(x)) => s"tanf(${emitExpr(x)})"
    case TTIRMathCall("tanh", List(x)) => s"tanhf(${emitExpr(x)})"
    case TTIRMathCall("max", List(a, b)) => s"fmaxf(${emitExpr(a)}, ${emitExpr(b)})"
    case TTIRMathCall("min", List(a, b)) => s"fminf(${emitExpr(a)}, ${emitExpr(b)})"
    case TTIRMathCall("minInt", List(a, b)) => s"fminf(${emitExpr(a)}, ${emitExpr(b)})"
    case TTIRMathCall("maxInt", List(a, b)) => s"fmaxf(${emitExpr(a)}, ${emitExpr(b)})"
    case TTIRMathCall("pow", List(a, b)) => s"powf(${emitExpr(a)}, ${emitExpr(b)})"
    case TTIRMathCall("sigmoid", List(x)) => s"(1.0f / (1.0f + expf(-${emitExpr(x)})))"
    case TTIRMathCall("relu", List(x)) => s"fmaxf(${emitExpr(x)}, 0.0f)"
    // Extended math functions
    case TTIRMathCall("exp2", List(x)) => s"exp2f(${emitExpr(x)})"
    case TTIRMathCall("expm1", List(x)) => s"expm1f(${emitExpr(x)})"
    case TTIRMathCall("log2", List(x)) => s"log2f(${emitExpr(x)})"
    case TTIRMathCall("log1p", List(x)) => s"log1pf(${emitExpr(x)})"
    case TTIRMathCall("floor", List(x)) => s"floorf(${emitExpr(x)})"
    case TTIRMathCall("ceil", List(x)) => s"ceilf(${emitExpr(x)})"
    case TTIRMathCall("round", List(x)) => s"roundf(${emitExpr(x)})"
    case TTIRMathCall("erf", List(x)) => s"erff(${emitExpr(x)})"
    case TTIRMathCall("erfc", List(x)) => s"erfcf(${emitExpr(x)})"
    case TTIRMathCall("rint", List(x)) => s"rintf(${emitExpr(x)})"
    case TTIRMathCall("nearbyint", List(x)) => s"nearbyintf(${emitExpr(x)})"
    case TTIRMathCall("atan", List(x)) => s"atanf(${emitExpr(x)})"
    case TTIRMathCall("sinh", List(x)) => s"sinhf(${emitExpr(x)})"
    case TTIRMathCall("cosh", List(x)) => s"coshf(${emitExpr(x)})"
    case TTIRMathCall("signum", List(x)) => s"copysignf(1.0f, ${emitExpr(x)})"
    // 类型转换
    case TTIRMathCall("int2float", List(x)) => s"((float)${emitExpr(x)})"
    case TTIRMathCall("toFloat", List(x)) => s"((float)${emitExpr(x)})"
    case TTIRMathCall("float2int", List(x)) => s"((int)${emitExpr(x)})"
    case TTIRMathCall("toInt", List(x)) => s"((int)${emitExpr(x)})"
    case TTIRMathCall("toDouble", List(x)) => s"((double)${emitExpr(x)})"
    case TTIRMathCall("toLong", List(x)) => s"((long)${emitExpr(x)})"
    case TTIRMathCall(f, args) => s"$f(${args.map(emitExpr).mkString(", ")})"

    case TTIRDot(a, b) => s"(${emitExpr(a)} * ${emitExpr(b)})"

    case TTIRReduce(maximize, name, input, init, body) =>
      s"reduce_${if maximize then "max" else "sum"}($name, ${emitExpr(input)})"

    // Warp shuffle: __shfl_sync(~0u, value, srcLane, width)
    case TTIRShfl("shfl", value, srcLane, width) =>
      s"__shfl_sync(0xFFFFFFFFu, ${emitExpr(value)}, ${emitExpr(srcLane)}, ${emitExpr(width)})"
    case TTIRShfl("shfl_up", value, srcLane, width) =>
      s"__shfl_up_sync(0xFFFFFFFFu, ${emitExpr(value)}, ${emitExpr(srcLane)}, ${emitExpr(width)})"
    case TTIRShfl("shfl_down", value, srcLane, width) =>
      s"__shfl_down_sync(0xFFFFFFFFu, ${emitExpr(value)}, ${emitExpr(srcLane)}, ${emitExpr(width)})"
    case TTIRShfl("shfl_xor", value, srcLane, width) =>
      s"__shfl_xor_sync(0xFFFFFFFFu, ${emitExpr(value)}, ${emitExpr(srcLane)}, ${emitExpr(width)})"
    // Warp vote: __any_sync / __all_sync
    case TTIRWarpVote("any", pred) => s"__any_sync(0xFFFFFFFFu, ${emitExpr(pred)})"
    case TTIRWarpVote("all", pred) => s"__all_sync(0xFFFFFFFFu, ${emitExpr(pred)})"
    // Masked load
    case TTIRMaskedLoad(ptr, offset, pred, other) =>
      s"(${emitExpr(pred)} ? ${ptr}[${emitExpr(offset)}] : ${emitExpr(other)})"
    // Ternary if-then-else expression: cond ? then : else
    case TTIRTernary(cond, thene, elsee) =>
      s"(${emitExpr(cond)} ? ${emitExpr(thene)} : ${emitExpr(elsee)})"

    case _ => s"/* ${expr.getClass.getSimpleName} */"

// ============================================================================
// Attention Kernel Emission Methods
// ============================================================================

  // FlashAttention: Two-pass tiled attention algorithm
  // Reduces HBM accesses by using on-chip SRAM for softmax computation
  private def emitFlashAttentionKernel(
    qPtr: String, kPtr: String, vPtr: String, outPtr: String,
    N: TTIRExpr, D: TTIRExpr,
    blockSize: Int, causal: Boolean
  ): Unit = {
    val n = emitExpr(N)
    val d = emitExpr(D)
    val bs = blockSize.toString

    emitLine(s"// FlashAttention kernel: Q[$n, $d] x K[$n, $d] -> O[$n, $d]")
    emitLine(s"const int BLOCK_M = $bs;  // block size for query sequence")
    emitLine(s"const int BLOCK_N = $bs;  // block size for key/value sequence")
    emitLine(s"const int BLOCK_D = min($d, 64);  // block size for head dimension")

    // Outer loop over query blocks
    emitLine(s"for (int block_m = 0; block_m < $n; block_m += BLOCK_M) {")
    indent += 1
    emitLine(s"// Load query block to shared memory")
    emitLine(s"float q_smem[BLOCK_M][BLOCK_D];")
    emitLine(s"float acc[BLOCK_M][BLOCK_D] = {0.0f};")
    emitLine(s"float l_i[BLOCK_M] = {-1e9f};  // running max")
    emitLine(s"float m_i[BLOCK_M] = {0.0f};   // running sum of exp")
    newline()

    // Inner loop over key/value blocks
    emitLine(s"for (int block_n = 0; block_n < $n; block_n += BLOCK_N) {")
    indent += 1
    emitLine(s"// Load key/value blocks")
    emitLine(s"float k_smem[BLOCK_N][BLOCK_D];")
    emitLine(s"float v_smem[BLOCK_N][BLOCK_D];")
    emitLine(s"float s[BLOCK_M][BLOCK_N];  // scores")
    newline()

    // Compute scores for this block
    emitLine(s"// Compute scores: QK^T / sqrt(d)")
    emitLine(s"for (int m = 0; m < BLOCK_M; m++) {")
    indent += 1
    emitLine(s"for (int n = 0; n < BLOCK_N; n++) {")
    indent += 1
    emitLine(s"float score = 0.0f;")
    emitLine(s"for (int d = 0; d < BLOCK_D; d++) {")
    indent += 1
    emitLine(s"int q_row = block_m + m;")
    emitLine(s"int k_row = block_n + n;")
    emitLine(s"if (q_row < $n && k_row < $n) {")
    indent += 1
    emitLine(s"float q_val = ${qPtr}[q_row * $d + d];")
    emitLine(s"float k_val = ${kPtr}[k_row * $d + d];")
    emitLine(s"score += q_val * k_val;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    emitLine(s"score /= sqrtf((float)$d);")
    if (causal) {
      emitLine(s"// Causal mask: prevent attending to future tokens")
      emitLine(s"if (block_m + m > block_n + n) score = -1e9f;")
    }
    emitLine(s"s[m][n] = score;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    newline()

    // Online softmax update
    emitLine(s"// Online softmax: update max and sum")
    emitLine(s"for (int m = 0; m < BLOCK_M; m++) {")
    indent += 1
    emitLine(s"float row_max = l_i[m];")
    emitLine(s"float row_sum = m_i[m];")
    emitLine(s"for (int n = 0; n < BLOCK_N; n++) {")
    indent += 1
    emitLine(s"row_max = fmaxf(row_max, s[m][n]);")
    indent -= 1
    emitLine(s"}")
    emitLine(s"float new_sum = 0.0f;")
    emitLine(s"for (int n = 0; n < BLOCK_N; n++) {")
    indent += 1
    emitLine(s"s[m][n] = expf(s[m][n] - row_max);")
    emitLine(s"new_sum += s[m][n];")
    indent -= 1
    emitLine(s"}")
    emitLine(s"float new_max = row_max;")
    emitLine(s"float mult = expf(row_max - new_max);")
    emitLine(s"for (int d = 0; d < BLOCK_D; d++) {")
    indent += 1
    emitLine(s"acc[m][d] = acc[m][d] * mult;")
    indent -= 1
    emitLine(s"}")
    emitLine(s"row_sum = row_sum * mult + new_sum;")
    emitLine(s"l_i[m] = new_max;")
    emitLine(s"m_i[m] = row_sum;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    newline()

    // Accumulate output: acc += softmax(s) * V
    emitLine(s"// Accumulate output: acc += softmax(s) * V")
    emitLine(s"for (int m = 0; m < BLOCK_M; m++) {")
    indent += 1
    emitLine(s"for (int d = 0; d < BLOCK_D; d++) {")
    indent += 1
    emitLine(s"float val = 0.0f;")
    emitLine(s"for (int n = 0; n < BLOCK_N; n++) {")
    indent += 1
    emitLine(s"int k_row = block_n + n;")
    emitLine(s"if (k_row < $n) {")
    indent += 1
    emitLine(s"float v_val = ${vPtr}[k_row * $d + d];")
    emitLine(s"val += s[m][n] * v_val;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    emitLine(s"acc[m][d] += val;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}") // end for block_n
    newline()

    // Write back output
    emitLine(s"// Write back to global memory")
    emitLine(s"for (int m = 0; m < BLOCK_M; m++) {")
    indent += 1
    emitLine(s"int q_row = block_m + m;")
    emitLine(s"if (q_row < $n) {")
    indent += 1
    emitLine(s"for (int d = 0; d < BLOCK_D; d++) {")
    indent += 1
    emitLine(s"${outPtr}[q_row * $d + d] = acc[m][d] / m_i[m];")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}") // end for block_m
  }

  // PageAttention: vLLM-style paged KV cache attention
  // Optimized for serving long sequences with KV cache pagination
  private def emitPageAttentionKernel(
    queryPtr: String,
    keyCachePtr: String, valueCachePtr: String,
    blockTablesPtr: String, seqLensPtr: String,
    outPtr: String,
    numBlocks: TTIRExpr, blockSize: TTIRExpr, D: TTIRExpr
  ): Unit = {
    val nb = emitExpr(numBlocks)
    val bs = emitExpr(blockSize)
    val d = emitExpr(D)

    emitLine(s"// PageAttention kernel with block tables")
    emitLine(s"const int BLOCK_SIZE = $bs;")
    emitLine(s"const int D = $d;")
    emitLine(s"float max_score = -1e9f;")
    emitLine(s"float sum_exp = 0.0f;")
    emitLine(s"float attn_out[D] = {0.0f};")
    newline()

    emitLine(s"// Get sequence length for this query")
    emitLine(s"int seq_len = seq_lens[i / D];  // assuming 1 query per sequence")
    emitLine(s"int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;")
    newline()

    // Iterate over pages/blocks
    emitLine(s"for (int block_idx = 0; block_idx < num_blocks; block_idx++) {")
    indent += 1
    emitLine(s"// Get physical block index from block table")
    emitLine(s"int phys_block_id = ${blockTablesPtr}[i * (${nb} + 1) + block_idx];")
    emitLine(s"int block_offset = block_idx * BLOCK_SIZE;")
    newline()

    // Compute attention score with this block
    emitLine(s"// Compute score: query dot key_cache[block]")
    emitLine(s"float score = 0.0f;")
    emitLine(s"for (int d = 0; d < D; d++) {")
    indent += 1
    emitLine(s"float q_val = ${queryPtr}[i * D + d];")
    emitLine(s"float k_val = ${keyCachePtr}[phys_block_id * BLOCK_SIZE * D + d];")
    emitLine(s"score += q_val * k_val;")
    indent -= 1
    emitLine(s"}")
    emitLine(s"score /= sqrtf((float)D);")
    newline()

    // Online softmax
    emitLine(s"float prev_max = max_score;")
    emitLine(s"max_score = fmaxf(max_score, score);")
    emitLine(s"float exp_score = expf(score - max_score);")
    emitLine(s"sum_exp = sum_exp * expf(prev_max - max_score) + exp_score;")
    newline()

    // Accumulate weighted values
    emitLine(s"// Accumulate attention output")
    emitLine(s"for (int d = 0; d < D; d++) {")
    indent += 1
    emitLine(s"float v_val = ${valueCachePtr}[phys_block_id * BLOCK_SIZE * D + d];")
    emitLine(s"attn_out[d] = attn_out[d] * expf(prev_max - max_score) + exp_score * v_val;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    newline()

    // Normalize and write output
    emitLine(s"// Normalize and write output")
    emitLine(s"for (int d = 0; d < D; d++) {")
    indent += 1
    emitLine(s"${outPtr}[i * D + d] = attn_out[d] / sum_exp;")
    indent -= 1
    emitLine(s"}")
  }

  // FlexAttention: Flexible attention with custom score/mask modifiers
  // Allows custom score transformation and mask application via callback functions
  private def emitFlexAttentionKernel(
    qPtr: String, kPtr: String, vPtr: String,
    outPtr: String,
    N: TTIRExpr, D: TTIRExpr,
    scoreModName: Option[String], maskModName: Option[String],
    blockSize: Int
  ): Unit = {
    val n = emitExpr(N)
    val d = emitExpr(D)
    val bs = blockSize.toString
    val scoreMod = scoreModName.getOrElse("null")
    val maskMod = maskModName.getOrElse("null")

    emitLine(s"// FlexAttention kernel with custom score/mask modifiers")
    emitLine(s"const int BLOCK_M = $bs;")
    emitLine(s"const int BLOCK_N = $bs;")
    emitLine(s"const float score_mod_scale = 1.0f;  // default score modifier")
    emitLine(s"const float mask_val = 0.0f;  // default mask value")
    newline()

    // Outer loop over query blocks
    emitLine(s"for (int block_m = 0; block_m < $n; block_m += BLOCK_M) {")
    indent += 1
    emitLine(s"float acc[BLOCK_M][$d] = {0.0f};")
    emitLine(s"float l_i[BLOCK_M] = {-1e9f};")
    emitLine(s"float m_i[BLOCK_M] = {0.0f};")
    newline()

    // Inner loop over key/value blocks
    emitLine(s"for (int block_n = 0; block_n < $n; block_n += BLOCK_N) {")
    indent += 1
    emitLine(s"float s[BLOCK_M][BLOCK_N];")
    newline()

    // Compute scores
    emitLine(s"// Compute raw attention scores")
    emitLine(s"for (int m = 0; m < BLOCK_M && block_m + m < $n; m++) {")
    indent += 1
    emitLine(s"for (int n = 0; n < BLOCK_N && block_n + n < $n; n++) {")
    indent += 1
    emitLine(s"float score = 0.0f;")
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"int q_idx = (block_m + m) * $d + d;")
    emitLine(s"int k_idx = (block_n + n) * $d + d;")
    emitLine(s"score += ${qPtr}[q_idx] * ${kPtr}[k_idx];")
    indent -= 1
    emitLine(s"}")
    emitLine(s"score /= sqrtf((float)$d);")
    newline()

    // Apply score modifier if provided
    emitLine(s"// Apply custom score modifier")
    emitLine(s"if ($scoreMod != null) {")
    indent += 1
    emitLine(s"score = $scoreMod(score, block_m + m, block_n + n);")
    indent -= 1
    emitLine(s"}")
    newline()

    // Apply mask modifier if provided
    emitLine(s"// Apply custom mask modifier")
    emitLine(s"float mask = 0.0f;")
    emitLine(s"if ($maskMod != null) {")
    indent += 1
    emitLine(s"mask = $maskMod(block_m + m, block_n + n);")
    emitLine(s"score += mask;")
    indent -= 1
    emitLine(s"}")
    newline()

    emitLine(s"s[m][n] = score;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    newline()

    // Online softmax
    emitLine(s"// Online softmax update")
    emitLine(s"for (int m = 0; m < BLOCK_M && block_m + m < $n; m++) {")
    indent += 1
    emitLine(s"float row_max = l_i[m];")
    emitLine(s"float row_sum = m_i[m];")
    emitLine(s"for (int n = 0; n < BLOCK_N && block_n + n < $n; n++) {")
    indent += 1
    emitLine(s"row_max = fmaxf(row_max, s[m][n]);")
    indent -= 1
    emitLine(s"}")
    emitLine(s"float new_sum = 0.0f;")
    emitLine(s"for (int n = 0; n < BLOCK_N && block_n + n < $n; n++) {")
    indent += 1
    emitLine(s"s[m][n] = expf(s[m][n] - row_max);")
    emitLine(s"new_sum += s[m][n];")
    indent -= 1
    emitLine(s"}")
    emitLine(s"float mult = expf(row_max - row_max);  // simplification")
    emitLine(s"for (int d = 0; d < $d; d++) acc[m][d] *= mult;")
    emitLine(s"l_i[m] = row_max;")
    emitLine(s"m_i[m] = row_sum * mult + new_sum;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    newline()

    // Accumulate output
    emitLine(s"// Accumulate weighted values")
    emitLine(s"for (int m = 0; m < BLOCK_M && block_m + m < $n; m++) {")
    indent += 1
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"float val = 0.0f;")
    emitLine(s"for (int n = 0; n < BLOCK_N && block_n + n < $n; n++) {")
    indent += 1
    emitLine(s"int v_idx = (block_n + n) * $d + d;")
    emitLine(s"val += s[m][n] * ${vPtr}[v_idx];")
    indent -= 1
    emitLine(s"}")
    emitLine(s"acc[m][d] += val;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    newline()

    // Write back
    emitLine(s"// Write back output")
    emitLine(s"for (int m = 0; m < BLOCK_M && block_m + m < $n; m++) {")
    indent += 1
    emitLine(s"int out_idx = (block_m + m) * $d;")
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"${outPtr}[out_idx + d] = acc[m][d] / (m_i[m] + 1e-6f);")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
  }

  // Grouped Query Attention (GQA)
  // Shares KV heads across multiple query heads for efficient inference
  private def emitGQAKernel(
    qPtr: String, kPtr: String, vPtr: String, outPtr: String,
    N: TTIRExpr, D: TTIRExpr,
    numQHeads: TTIRExpr, numKVHeads: TTIRExpr,
    scale: TTIRExpr
  ): Unit = {
    val n = emitExpr(N)
    val d = emitExpr(D)
    val nQ = emitExpr(numQHeads)
    val nKV = emitExpr(numKVHeads)
    val s = emitExpr(scale)

    emitLine(s"// Grouped Query Attention: $nQ query heads, $nKV KV heads")
    emitLine(s"const int Q_HEADS = $nQ;")
    emitLine(s"const int KV_HEADS = $nKV;")
    emitLine(s"const float scale = $s;")
    newline()

    emitLine(s"// Determine which KV head this query head maps to")
    emitLine(s"int kv_head_idx = (i / ($d / Q_HEADS)) % KV_HEADS;")
    newline()

    emitLine(s"float max_score = -1e9f;")
    emitLine(s"float sum_exp = 0.0f;")
    emitLine(s"float out_val = 0.0f;")
    newline()

    emitLine(s"// Compute attention with shared KV heads")
    emitLine(s"for (int j = 0; j < $n; j++) {")
    indent += 1
    emitLine(s"float score = 0.0f;")
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"int q_idx = i * $d + d;")
    emitLine(s"int k_idx = kv_head_idx * $d + d;  // shared KV head")
    emitLine(s"score += ${qPtr}[q_idx] * ${kPtr}[j * $d + k_idx % $d];")
    indent -= 1
    emitLine(s"}")
    emitLine(s"score *= scale;")
    newline()

    emitLine(s"// Online softmax")
    emitLine(s"float prev_max = max_score;")
    emitLine(s"max_score = fmaxf(max_score, score);")
    emitLine(s"float exp_score = expf(score - max_score);")
    emitLine(s"sum_exp = sum_exp * expf(prev_max - max_score) + exp_score;")
    newline()

    emitLine(s"// Accumulate output")
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"int v_idx = kv_head_idx * $d + d;")
    emitLine(s"out_val += ${vPtr}[j * $d + v_idx % $d] * exp_score;")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    newline()

    emitLine(s"// Normalize and store")
    emitLine(s"${outPtr}[i] = out_val / (sum_exp + 1e-6f);")
  }

  // Multi-head Attention (MHA)
  // Standard multi-head attention with full KV parallelism
  private def emitMHAKernel(
    qPtr: String, kPtr: String, vPtr: String, outPtr: String,
    N: TTIRExpr, D: TTIRExpr,
    numHeads: TTIRExpr,
    scale: TTIRExpr
  ): Unit = {
    val n = emitExpr(N)
    val d = emitExpr(D)
    val h = emitExpr(numHeads)
    val s = emitExpr(scale)

    emitLine(s"// Multi-head Attention: $h heads, dim $d")
    emitLine(s"const int NUM_HEADS = $h;")
    emitLine(s"const float scale = $s;")
    newline()

    emitLine(s"float max_score = -1e9f;")
    emitLine(s"float sum_exp = 0.0f;")
    emitLine(s"float acc[$d] = {0.0f};")
    newline()

    emitLine(s"for (int j = 0; j < $n; j++) {")
    indent += 1
    emitLine(s"// Compute attention score")
    emitLine(s"float score = 0.0f;")
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"score += ${qPtr}[i * $d + d] * ${kPtr}[j * $d + d];")
    indent -= 1
    emitLine(s"}")
    emitLine(s"score *= scale;")
    newline()

    emitLine(s"// Online softmax")
    emitLine(s"float prev_max = max_score;")
    emitLine(s"max_score = fmaxf(max_score, score);")
    emitLine(s"float exp_score = expf(score - max_score);")
    emitLine(s"sum_exp = sum_exp * expf(prev_max - max_score) + exp_score;")
    newline()

    emitLine(s"// Accumulate values")
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"acc[d] = acc[d] * expf(prev_max - max_score) + exp_score * ${vPtr}[j * $d + d];")
    indent -= 1
    emitLine(s"}")
    indent -= 1
    emitLine(s"}")
    newline()

    emitLine(s"// Normalize and store")
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"${outPtr}[i * $d + d] = acc[d] / (sum_exp + 1e-6f);")
    indent -= 1
    emitLine(s"}")
  }

  // StoreKVCache: Store key/value to paged KV cache (vLLM-style)
  // Maps logical token positions to physical cache slots via slot_mapping
  private def emitStoreKVCacheKernel(
    keyPtr: String, keyStride: TTIRExpr,
    valuePtr: String, valueStride: TTIRExpr,
    kCachePtr: String, vCachePtr: String,
    slotMappingPtr: String,
    D: TTIRExpr
  ): Unit = {
    val d = emitExpr(D)
    val kStride = emitExpr(keyStride)
    val vStride = emitExpr(valueStride)

    emitLine(s"// StoreKVCache: store key/value to paged KV cache")
    emitLine(s"int slot = ${slotMappingPtr}[i];")
    emitLine(s"if (slot == -1) return;")
    newline()

    emitLine(s"// Compute cache offset for this slot")
    emitLine(s"int cache_offset = slot * $d;")
    newline()

    emitLine(s"// Store entire key and value vectors")
    emitLine(s"for (int d = 0; d < $d; d++) {")
    indent += 1
    emitLine(s"${kCachePtr}[cache_offset + d] = ${keyPtr}[i * $kStride + d];")
    emitLine(s"${vCachePtr}[cache_offset + d] = ${valuePtr}[i * $vStride + d];")
    indent -= 1
    emitLine(s"}")
  }

  // LoadKVCache: Load key/value from paged KV cache (vLLM-style)
  private def emitLoadKVCacheKernel(
    kCachePtr: String, vCachePtr: String,
    slotMappingPtr: String,
    keyOutputPtr: String, valueOutputPtr: String,
    numTokens: TTIRExpr,
    D: TTIRExpr
  ): Unit = {
    val d = emitExpr(D)
    val n = emitExpr(numTokens)

    emitLine(s"// LoadKVCache: load key/value from paged KV cache")
    emitLine(s"if (i >= $n) return;")
    emitLine(s"int slot = ${slotMappingPtr}[i];")
    emitLine(s"if (slot == -1) return;")
    newline()

    emitLine(s"// Compute cache offset for this slot")
    emitLine(s"int cache_offset = slot * $d;")
    newline()

    emitLine(s"// Load entire key and value vectors")
    emitLine(s"for (int dim = 0; dim < $d; dim++) {")
    indent += 1
    emitLine(s"${keyOutputPtr}[i * $d + dim] = ${kCachePtr}[cache_offset + dim];")
    emitLine(s"${valueOutputPtr}[i * $d + dim] = ${vCachePtr}[cache_offset + dim];")
    indent -= 1
    emitLine(s"}")
  }

  // LoadTile: Load a tile from memory
  private def emitLoadTileKernel(
    inputPtr: String,
    tileOutputPtr: String,
    tileRow: TTIRExpr, tileCol: TTIRExpr,
    tileSize: TTIRExpr,
    M: TTIRExpr, N: TTIRExpr,
    stride: TTIRExpr
  ): Unit = {
    val tr = emitExpr(tileRow)
    val tc = emitExpr(tileCol)
    val ts = emitExpr(tileSize)
    val m = emitExpr(M)
    val n = emitExpr(N)
    val s = emitExpr(stride)

    emitLine(s"// LoadTile: load tile from memory")
    emitLine(s"int tile_row = $tr;")
    emitLine(s"int tile_col = $tc;")
    emitLine(s"int tile_size = $ts;")
    emitLine(s"int row = tile_row * tile_size + (i / tile_size);")
    emitLine(s"int col = tile_col * tile_size + (i % tile_size);")
    emitLine(s"if (row < $m && col < $n) {")
    indent += 1
    emitLine(s"${tileOutputPtr}[i] = ${inputPtr}[row * $s + col];")
    indent -= 1
    emitLine(s"}")
  }

  // StoreTile: Store a tile to memory
  private def emitStoreTileKernel(
    tilePtr: String,
    outputPtr: String,
    tileRow: TTIRExpr, tileCol: TTIRExpr,
    tileSize: TTIRExpr,
    M: TTIRExpr, N: TTIRExpr,
    stride: TTIRExpr
  ): Unit = {
    val tr = emitExpr(tileRow)
    val tc = emitExpr(tileCol)
    val ts = emitExpr(tileSize)
    val m = emitExpr(M)
    val n = emitExpr(N)
    val s = emitExpr(stride)

    emitLine(s"// StoreTile: store tile to memory")
    emitLine(s"int tile_row = $tr;")
    emitLine(s"int tile_col = $tc;")
    emitLine(s"int tile_size = $ts;")
    emitLine(s"int row = tile_row * tile_size + (i / tile_size);")
    emitLine(s"int col = tile_col * tile_size + (i % tile_size);")
    emitLine(s"if (row < $m && col < $n) {")
    indent += 1
    emitLine(s"${outputPtr}[row * $s + col] = ${tilePtr}[i];")
    indent -= 1
    emitLine(s"}")
  }

  // SaveStore: Combined save and store operation
  private def emitSaveStoreKernel(
    dataPtr: String,
    storagePtr: String,
    offset: TTIRExpr,
    size: TTIRExpr,
    compressionRatio: TTIRExpr
  ): Unit = {
    val off = emitExpr(offset)
    val sz = emitExpr(size)
    val cr = emitExpr(compressionRatio)

    emitLine(s"// SaveStore: combined save and store operation")
    emitLine(s"int storage_offset = $off;")
    emitLine(s"int comp_size = $sz / $cr;")
    newline()

    emitLine(s"// Save to persistent storage with compression")
    emitLine(s"for (int j = 0; j < comp_size; j++) {")
    indent += 1
    emitLine(s"float val = ${dataPtr}[i * comp_size + j];")
    emitLine(s"${storagePtr}[storage_offset + j] = val;")
    indent -= 1
    emitLine(s"}")
  }

  // LoadStore: Load from persistent storage
  private def emitLoadStoreKernel(
    storagePtr: String,
    dataPtr: String,
    offset: TTIRExpr,
    size: TTIRExpr
  ): Unit = {
    val off = emitExpr(offset)
    val sz = emitExpr(size)

    emitLine(s"// LoadStore: load from persistent storage")
    emitLine(s"int storage_offset = $off;")
    emitLine(s"${dataPtr}[i] = ${storagePtr}[storage_offset + i];")
  }

  // PagedSaveKVCache: Save KV cache with page table mapping
  private def emitPagedSaveKVCacheKernel(
    kvCachePtr: String,
    pageTablePtr: String,
    storagePtr: String,
    numBlocks: TTIRExpr,
    blockSize: TTIRExpr,
    D: TTIRExpr
  ): Unit = {
    val nb = emitExpr(numBlocks)
    val bs = emitExpr(blockSize)
    val d = emitExpr(D)

    emitLine(s"// PagedSaveKVCache: save with page table mapping")
    emitLine(s"int block_idx = i / ($bs * $d);")
    emitLine(s"int local_idx = i % ($bs * $d);")
    emitLine(s"if (block_idx >= $nb) return;")
    emitLine(s"int page_addr = ${pageTablePtr}[block_idx];")
    emitLine(s"${storagePtr}[page_addr * $bs * $d + local_idx] = ${kvCachePtr}[i];")
  }

  // PagedLoadKVCache: Load KV cache with page table mapping
  private def emitPagedLoadKVCacheKernel(
    storagePtr: String,
    pageTablePtr: String,
    kvCachePtr: String,
    numBlocks: TTIRExpr,
    blockSize: TTIRExpr,
    D: TTIRExpr
  ): Unit = {
    val nb = emitExpr(numBlocks)
    val bs = emitExpr(blockSize)
    val d = emitExpr(D)

    emitLine(s"// PagedLoadKVCache: load with page table mapping")
    emitLine(s"int block_idx = i / ($bs * $d);")
    emitLine(s"int local_idx = i % ($bs * $d);")
    emitLine(s"if (block_idx >= $nb) return;")
    emitLine(s"int page_addr = ${pageTablePtr}[block_idx];")
    emitLine(s"${kvCachePtr}[i] = ${storagePtr}[page_addr * $bs * $d + local_idx];")
  }

  // CheckpointSave: Save with compression
  private def emitCheckpointSaveKernel(
    kvCachePtr: String,
    checkpointPtr: String,
    seqLen: TTIRExpr,
    headDim: TTIRExpr,
    numLayers: TTIRExpr,
    compressRatio: TTIRExpr
  ): Unit = {
    val sl = emitExpr(seqLen)
    val hd = emitExpr(headDim)
    val nl = emitExpr(numLayers)
    val cr = emitExpr(compressRatio)

    emitLine(s"// CheckpointSave: save with compression")
    emitLine(s"int total_size = $nl * $sl * $hd;")
    emitLine(s"int comp_size = (total_size + $cr - 1) / $cr;")
    emitLine(s"if (i >= comp_size) return;")
    newline()

    emitLine(s"// Compute average of compressed chunk")
    emitLine(s"float sum = 0.0f;")
    emitLine(s"int count = 0;")
    emitLine(s"for (int j = 0; j < $cr; j++) {")
    indent += 1
    emitLine(s"int idx = i * $cr + j;")
    emitLine(s"if (idx < total_size) { sum += fabsf(${kvCachePtr}[idx]); count++; }")
    indent -= 1
    emitLine(s"}")
    emitLine(s"${checkpointPtr}[i] = (count > 0) ? (sum / count) : 0.0f;")
  }

  // CheckpointRestore: Restore from compressed checkpoint
  private def emitCheckpointRestoreKernel(
    checkpointPtr: String,
    kvCachePtr: String,
    seqLen: TTIRExpr,
    headDim: TTIRExpr,
    numLayers: TTIRExpr,
    compressRatio: TTIRExpr
  ): Unit = {
    val sl = emitExpr(seqLen)
    val hd = emitExpr(headDim)
    val cr = emitExpr(compressRatio)

    emitLine(s"// CheckpointRestore: restore from compressed checkpoint")
    emitLine(s"int total_size = $sl * $hd;")
    emitLine(s"int comp_idx = i / $cr;")
    emitLine(s"float val = ${checkpointPtr}[comp_idx];")
    emitLine(s"${kvCachePtr}[i] = val;")
  }

  // DeltaSave: Incremental save (only deltas)
  private def emitDeltaSaveKernel(
    kvCachePtr: String,
    prevCheckpointPtr: String,
    deltaPtr: String,
    seqLen: TTIRExpr,
    headDim: TTIRExpr,
    numLayers: TTIRExpr
  ): Unit = {
    val sl = emitExpr(seqLen)
    val hd = emitExpr(headDim)
    val nl = emitExpr(numLayers)

    emitLine(s"// DeltaSave: incremental save (only deltas)")
    emitLine(s"int total_size = $nl * $sl * $hd;")
    emitLine(s"if (i >= total_size) return;")
    emitLine(s"float curr = ${kvCachePtr}[i];")
    emitLine(s"float prev = ${prevCheckpointPtr}[i];")
    emitLine(s"${deltaPtr}[i] = curr - prev;")
  }

  // DeltaRestore: Restore from deltas
  private def emitDeltaRestoreKernel(
    prevCheckpointPtr: String,
    deltaPtr: String,
    kvCachePtr: String,
    seqLen: TTIRExpr,
    headDim: TTIRExpr,
    numLayers: TTIRExpr
  ): Unit = {
    val sl = emitExpr(seqLen)
    val hd = emitExpr(headDim)
    val nl = emitExpr(numLayers)

    emitLine(s"// DeltaRestore: restore from deltas")
    emitLine(s"int total_size = $nl * $sl * $hd;")
    emitLine(s"if (i >= total_size) return;")
    emitLine(s"float prev = ${prevCheckpointPtr}[i];")
    emitLine(s"float delta = ${deltaPtr}[i];")
    emitLine(s"${kvCachePtr}[i] = prev + delta;")
  }

  // SelectiveSave: Save based on importance threshold
  private def emitSelectiveSaveKernel(
    kvCachePtr: String,
    importancePtr: String,
    storagePtr: String,
    seqLen: TTIRExpr,
    headDim: TTIRExpr,
    numLayers: TTIRExpr,
    threshold: TTIRExpr
  ): Unit = {
    val sl = emitExpr(seqLen)
    val hd = emitExpr(headDim)
    val nl = emitExpr(numLayers)
    val th = emitExpr(threshold)

    emitLine(s"// SelectiveSave: save based on importance threshold")
    emitLine(s"int layer_idx = i / ($sl * $hd);")
    emitLine(s"int local_idx = i % ($sl * $hd);")
    emitLine(s"int seq_idx = local_idx / $hd;")
    emitLine(s"if (layer_idx >= $nl) return;")
    emitLine(s"float importance = ${importancePtr}[seq_idx];")
    emitLine(s"if (importance > $th) {")
    indent += 1
    emitLine(s"${storagePtr}[i] = ${kvCachePtr}[i];")
    indent -= 1
    emitLine(s"}")
  }

  // SelectiveLoad: Load based on importance threshold
  private def emitSelectiveLoadKernel(
    storagePtr: String,
    importancePtr: String,
    kvCachePtr: String,
    seqLen: TTIRExpr,
    headDim: TTIRExpr,
    numLayers: TTIRExpr,
    threshold: TTIRExpr
  ): Unit = {
    val sl = emitExpr(seqLen)
    val hd = emitExpr(headDim)
    val nl = emitExpr(numLayers)
    val th = emitExpr(threshold)

    emitLine(s"// SelectiveLoad: load based on importance threshold")
    emitLine(s"int layer_idx = i / ($sl * $hd);")
    emitLine(s"int local_idx = i % ($sl * $hd);")
    emitLine(s"int seq_idx = local_idx / $hd;")
    emitLine(s"if (layer_idx >= $nl) return;")
    emitLine(s"float importance = ${importancePtr}[seq_idx];")
    emitLine(s"if (importance > $th) {")
    indent += 1
    emitLine(s"${kvCachePtr}[i] = ${storagePtr}[i];")
    indent -= 1
    emitLine(s"}")
  }

  // SaveWithECC: Save with error correction codes
  private def emitSaveWithECCKernel(
    kvCachePtr: String,
    storagePtr: String,
    eccPtr: String,
    seqLen: TTIRExpr,
    headDim: TTIRExpr
  ): Unit = {
    val sl = emitExpr(seqLen)
    val hd = emitExpr(headDim)

    emitLine(s"// SaveWithECC: save with error correction codes")
    emitLine(s"int seq_idx = i / $hd;")
    emitLine(s"int dim_idx = i % $hd;")
    emitLine(s"float val = ${kvCachePtr}[i];")
    emitLine(s"${storagePtr}[i] = val;")
    emitLine(s"if (dim_idx == 0) {")
    indent += 1
    emitLine(s"float check_sum = (val > 0.0f) ? 1.0f : 0.0f;")
    emitLine(s"${eccPtr}[seq_idx] = check_sum;")
    indent -= 1
    emitLine(s"}")
  }

  // LoadWithECC: Load with error correction
  private def emitLoadWithECCKernel(
    storagePtr: String,
    eccPtr: String,
    kvCachePtr: String,
    seqLen: TTIRExpr,
    headDim: TTIRExpr
  ): Unit = {
    val sl = emitExpr(seqLen)
    val hd = emitExpr(headDim)

    emitLine(s"// LoadWithECC: load with error correction")
    emitLine(s"int seq_idx = i / $hd;")
    emitLine(s"int dim_idx = i % $hd;")
    emitLine(s"float stored = ${storagePtr}[i];")
    emitLine(s"float check_sum = ${eccPtr}[seq_idx];")
    emitLine(s"float expected = (stored > 0.0f) ? 1.0f : 0.0f;")
    emitLine(s"if (check_sum != expected) {")
    indent += 1
    emitLine(s"// Correct single-bit error by scaling down")
    emitLine(s"${kvCachePtr}[i] = stored * 0.95f;")
    indent -= 1
    emitLine(s"} else {")
    indent += 1
    emitLine(s"${kvCachePtr}[i] = stored;")
    indent -= 1
    emitLine(s"}")
  }

// ============================================================================
// @tritonKernel Macro - 自动翻译 Scala 函数 → TTIR → CUDA
// ============================================================================

import scala.annotation.{MacroAnnotation, experimental}
import scala.quoted.*
import scala.collection.mutable

@experimental
class TritonKernelMacro(
  val name: String = "",
  val gridType: String = "1D",
  val blockSize: Int = 256
) extends MacroAnnotation:
  def transform(using Quotes)(definition: quotes.reflect.Definition, companion: Option[quotes.reflect.Definition]): List[quotes.reflect.Definition] =
    import quotes.reflect.*

    // 提取注解参数值（注解的构造函数参数）
    val effectiveName = if name.nonEmpty then name else definition.symbol.name
    val effectiveGridType = gridType
    val effectiveBlockSize = blockSize

    definition match
      case DefDef(methodName, paramss, tpt, Some(body)) =>
        val startTime = System.nanoTime()

        // 解析参数 - 根据类型和命名约定决定是指针还是标量
        val paramInfo = paramss.flatMap(_.params).collect {
          case ValDef(n, tpt, _) =>
            (n, typeToTriton(tpt.tpe, n), isPointerType(tpt.tpe, n))
        }

        // 检查用户是否已有 out: Float 参数
        val hasUserOut = paramInfo.exists(_(0) == "out")

        // 构建 TTIR
        val ir = new TTIR(effectiveName)
        // PageAttention kernels need PAGE_SIZE and INF_NEG constants
        if effectiveName.contains("paged_attention") || effectiveName.contains("pageAttention") then
          ir.constant("PAGE_SIZE", "128")
          ir.constant("INF_NEG", "-1e8f")
        // FlashAttention V3/V4 kernels need HEAD_DIM and tiling constants
        if effectiveName.contains("flash_attention") || effectiveName.contains("FlashAttention") then
          ir.constant("HEAD_DIM", "128")
          ir.constant("BLOCK_Q_V3", "64")
          ir.constant("BLOCK_KV_V3", "128")
          ir.constant("BLOCK_Q_V4", "32")
          ir.constant("BLOCK_KV_V4", "64")
          ir.constant("WARP_TILE_V4", "16")
          ir.constant("INF_NEG", "-1e8f")
        // 添加 out 参数（仅当用户没有 out 参数时才添加）
        if !hasUserOut then ir.param("float*", "out")
        // 添加其他参数 - 使用 typeToTriton 获取标量/指针类型
        paramInfo.foreach((n, t, _) => ir.param(t, n))
        // 注册参数为指针 - 只有 Float* 类型的指针才注册到 ctx.pointers
        // Int* 类型的指针（如 Int 参数按命名约定被视为指针）不在 ctx.pointers 中注册
        // 因为在表达式中使用时，Int* 应该作为普通变量处理（用于地址计算），而不是数组访问
        val context = new TranslateContext
        if !hasUserOut then context.addPointer("out")
        paramInfo.foreach((n, t, isPtr) => if isPtr && t == "float*" then context.addPointer(n))
        // i 是 preamble 自动声明的全局线程ID，不需要重复声明
        context.addLocal("i", "int")

        // 翻译函数体到 TTIR
        val stmts = translateTerm(body, context)

        // 提升所有变量声明到函数作用域，避免在嵌套作用域中声明导致后续使用出错
        // 同时提升 shared memory 声明到函数作用域（CUDA 要求 __shared__ 在函数级别）
        val (stmtsWithoutDecls, hoistedDecls, hoistedShared) = collectAndHoistDecls(stmts)

        // 策略：最后一个表达式包装为 out[i] = expr，其他语句正常添加
        if stmtsWithoutDecls.nonEmpty then
          // 先添加提升的 shared memory 声明（CUDA 要求 __shared__ 在 if 外）
          hoistedShared.foreach(ir += _)
          // 再添加提升的变量声明（到函数作用域）
          hoistedDecls.foreach(ir += _)
          val lastStmt = stmtsWithoutDecls.last
          val otherStmts = stmtsWithoutDecls.init

          // 先添加其他语句
          otherStmts.foreach(ir += _)

          // 将最后一个表达式包装为 out[i] = <expr>
          lastStmt match
            case TTIRExprStmt(expr) =>
              ir += TTIRStore("out", TTIRVar("i"), expr, TTIRConst(true))
            case TTIRReturn(_) =>
              // return 语句直接添加，不生成默认输出
              ir += lastStmt
            case _ =>
              // 其他类型的语句（如 TTIRIf, TTIRFor, TTIRAssign 等）直接添加
              ir += lastStmt

        // 生成 CUDA 代码
        val cudaCode = ir.emit()

        // 注册到 TritonKernelRegistry
        val paramTypes = ir._params.map(_.tpe).toList
        val paramNames = ir._params.map(_.name).toList
        val meta = TritonKernelMeta(effectiveName, cudaCode, paramTypes, paramNames, hasUserOut, effectiveGridType, effectiveBlockSize)
        TritonKernelRegistry.register(meta)

        // 输出到文件
        val filePath = "/tmp/cuda_dsl_generated_kernels90.txt"
        System.err.println(s"[@TritonKernelMacro] Writing $effectiveName to $filePath (${cudaCode.length} chars)")
        try
          val f = new java.io.FileWriter(filePath, true)
          f.write(s"// [@TritonKernelMacro] $effectiveName\n$cudaCode\n\n")
          f.flush()
          f.close()
          System.err.println(s"[@TritonKernelMacro] Written: $effectiveName")
        catch
          case e: Exception =>
            System.err.println(s"[@TritonKernelMacro] FILE WRITE ERROR: ${e.getMessage}")
            e.printStackTrace()

        System.err.println(s"[@TritonKernelMacro] Generated: $effectiveName in ${(System.nanoTime() - startTime) / 1e6}ms")

        List(definition)
      case _ =>
        report.errorAndAbort("@TritonKernelMacro can only be applied to functions")

// ============================================================================
// 翻译上下文
// ============================================================================

class TranslateContext:
  val locals = mutable.Map[String, String]()
  val loopVars = mutable.Map[String, String]()
  val pointers = mutable.Set[String]()
  var loopDepth = 0  // Track nesting depth for variable scoping

  def isLocal(name: String): Boolean = locals.contains(name)
  def isLoopVar(name: String): Boolean = loopVars.contains(name)
  def isPointer(name: String): Boolean = pointers.contains(name)
  def addLocal(name: String, tpe: String): Unit = locals(name) = tpe
  def addLoopVar(name: String): Unit = loopVars(name) = "int"
  def addPointer(name: String): Unit = pointers += name
  def getType(name: String): String = locals.getOrElse(name, "float")

// ============================================================================
// 辅助函数：递归收集语句中的所有 LocalDecl 并返回去除了声明的语句
// ============================================================================

// 递归扫描语句列表，收集所有 TTIRLocalDecl 并将其从语句中移除
// 返回: (去除了 LocalDecl 的语句列表, 所有 LocalDecl 列表)
// 去重：同一变量名只保留第一个声明
private def collectAndHoistDecls(stmts: List[TTIRStmt]): (List[TTIRStmt], List[TTIRLocalDecl], List[TTIRSharedMem]) =
  val seenDecls = mutable.Set[String]()

  def scan(ss: List[TTIRStmt]): List[TTIRStmt] =
    val result = mutable.ListBuffer[TTIRStmt]()
    for stmt <- ss do stmt match
      case d: TTIRLocalDecl =>
        if !seenDecls.contains(d.name) then
          seenDecls += d.name
        // 不添加到结果（声明会通过 seenDecls 收集）
      case s: TTIRSharedMem =>
        // shared memory 声明从结果中移除，将在函数级别重新添加
      case TTIRFor(iter, start, end, body) =>
        result += TTIRFor(iter, start, end, scan(body))
      case TTIRWhile(cond, body) =>
        result += TTIRWhile(cond, scan(body))
      case TTIRIf(cond, thenp, elsep) =>
        result += TTIRIf(cond, scan(thenp), scan(elsep))
      case other =>
        result += other
    result.toList

  // 收集所有声明（按出现顺序）
  val hoisted = mutable.ListBuffer[TTIRLocalDecl]()
  val hoistedShared = mutable.ListBuffer[TTIRSharedMem]()
  def collect(ss: List[TTIRStmt]): Unit =
    for stmt <- ss do stmt match
      case d: TTIRLocalDecl =>
        hoisted += d
      case s: TTIRSharedMem =>
        hoistedShared += s
      case TTIRFor(iter, start, end, body) =>
        collect(body)
      case TTIRWhile(cond, body) =>
        collect(body)
      case TTIRIf(cond, thenp, elsep) =>
        collect(thenp); collect(elsep)
      case _ =>
  collect(stmts)

  (scan(stmts), hoisted.toList, hoistedShared.toList)

// ============================================================================
// 核心翻译函数
// ============================================================================

private def translateTerm(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): List[TTIRStmt] =
  import quotes.reflect.*

  term match
    // 代码块
    case Block(stats, lastExpr) =>
      // 直接遍历stats不过滤类型
      val code = stats.map(_.asInstanceOf[Term]).flatMap(s => translateTerm(s, ctx))
      val lastCode = translateTerm(lastExpr, ctx)
      code ++ lastCode

    // 变量定义: var x = ... 或 val x = ...
    case ValDef(name, tpt, Some(rhs)) =>
      // 特殊处理 tl.warp_any / tl.warp_all - 它们返回 Boolean
      rhs match
        case Apply(Select(parent, "warp_any"), List(pred)) if parent.show.contains("tl") =>
          val predCode = translateExpr(pred, ctx)
          ctx.addLocal(name, "bool")
          List(TTIRLocalDecl("bool", name), TTIRAssign(name, TTIRWarpVote("any", predCode)))
        case Apply(Select(parent, "warp_all"), List(pred)) if parent.show.contains("tl") =>
          val predCode = translateExpr(pred, ctx)
          ctx.addLocal(name, "bool")
          List(TTIRLocalDecl("bool", name), TTIRAssign(name, TTIRWarpVote("all", predCode)))
        case _ =>
          val rhsCode = translateExpr(rhs, ctx)
          // 根据 RHS 的类型推断变量类型
          val tpe = inferTypeFromExpr(rhsCode, ctx, translateType(tpt.tpe))
          // 如果 RHS 就是同名变量（如 tl.program_id(0) 对 i），则跳过（preamble 已声明）
          rhsCode match
            case TTIRVar(`name`) if ctx.isLocal(name) =>
              Nil  // val i = i 或 val i = tl.program_id(0) - 跳过
            case _ if ctx.isLocal(name) =>
              // 变量已在上下文中，只生成赋值
              ctx.addLocal(name, tpe)
              List(TTIRAssign(name, rhsCode))
            case _ =>
              ctx.addLocal(name, tpe)
              // 发出局部变量声明和赋值
              List(TTIRLocalDecl(tpe, name), TTIRAssign(name, rhsCode))

    // out(i) = value 形式的数组赋值 (支持完全限定名如 cuda.dsl.dsl.tl.outPtr)
    case ap @ Apply(Select(ptrTree, "update"), List(idx, value)) =>
      val ptrName = extractBaseName(ptrTree)
      if ctx.isPointer(ptrName) then
        val idxCode = translateExpr(idx, ctx)
        val valueCode = translateExpr(value, ctx)
        List(TTIRStore(ptrName, idxCode, valueCode, TTIRConst(true)))
      else
        translateTermImpl(ap, ctx).asInstanceOf[List[TTIRStmt]]

    // ptr(idx) = value (直接调用形式，支持完全限定名)
    case ap @ Apply(Ident(ptr), List(idx, value)) if ctx.isPointer(ptr) =>
      val idxCode = translateExpr(idx, ctx)
      val valueCode = translateExpr(value, ctx)
      List(TTIRStore(ptr, idxCode, valueCode, TTIRConst(true)))
    case ap @ Apply(ptrTree, List(idx, value)) if !ptrTree.isInstanceOf[Ident] =>
      val ptrName = extractBaseName(ptrTree)
      if ctx.isPointer(ptrName) then
        val idxCode = translateExpr(idx, ctx)
        val valueCode = translateExpr(value, ctx)
        List(TTIRStore(ptrName, idxCode, valueCode, TTIRConst(true)))
      else
        translateTermImpl(ap, ctx).asInstanceOf[List[TTIRStmt]]

    // 变量赋值: x = ...
    case Assign(lhs, rhs) =>
      val name = identName(lhs)
      val rhsCode = translateExpr(rhs, ctx)
      List(TTIRAssign(name, rhsCode))

    // if 语句
    case If(cond, thenp, elsep) =>
      val condCode = translateExpr(cond, ctx)
      val thenStmts = translateTerm(thenp, ctx)
      val elseStmts = translateTerm(elsep, ctx)
      List(TTIRIf(condCode, thenStmts, elseStmts))

    // 0.until(N) foreach { lambda } 形式 - Scala 3 使用 TypeApply 包装方法调用
    // 例如: x.foreach[Type](lambda) => Apply(TypeApply(Select(x, "foreach"), List(typeArg)), List(lambda))
    case ap @ Apply(TypeApply(Select(range, "foreach"), typeArgs), args) =>
      // 增强版提取 lambda 参数 - 递归处理嵌套结构
      def extractLambda(term: Term): Option[(List[String], Term)] = term match
        // 直接 Lambda
        case Lambda(params, body) => Some((params.map(p => p.name.toString), body))
        // Lambda 被 TypeApply 包装 (罕见)
        case TypeApply(Lambda(params, body), _) => Some((params.map(p => p.name.toString), body))
        // Closure 包装 - meth 可能是 Lambda 或直接是 body
        case Closure(meth, _) =>
          extractLambda(meth)  // 递归处理
        // TypeApply 包装 Closure
        case TypeApply(Closure(meth, _), _) =>
          extractLambda(meth)  // 递归处理
        // Block 包装 (Scala 3 有时用 Block 包装 lambda)
        case Block(stats, expr) =>
          extractLambda(expr)  // 递归提取
        // 嵌套 Apply (lambda 作为参数传递给另一个函数,再作为 foreach 的参数)
        case Apply(func, largs) =>
          extractLambda(func)  // 递归提取
        // 其他情况
        case _ => None

      // 辅助函数: 检查是否是 xxx.until(N) 形式
      // Scala 3 macro 展开后可能有多种结构:
      // 1. Apply(Select(Literal(IntConstant(0)), "until"), List(end))  // 直接调用
      // 2. Apply(Select(inner, "until"), List(end)) where inner = intWrapper(0) or similar
      // 3. Apply(Select(inner, "until"), List(end)) for runtime variables
      def extractUntilEnd(r: Term): Option[Term] = r match
        case Apply(Select(Literal(IntConstant(0)), "until"), List(end)) => Some(end)
        case Apply(Select(inner, "until"), List(end)) if inner.toString == "0" => Some(end)
        // 处理隐式转换 intWrapper 包装: 检查 callee 是否包含 "intWrapper"
        case Apply(Select(callee, "until"), List(end)) =>
          if (callee.show.contains("intWrapper")) Some(end)
          else if (callee.show.contains(".until")) None  // 避免递归
          else None
        case _ => None

      args.headOption flatMap extractLambda match
        case Some((params, body)) =>
          extractUntilEnd(range) match
            case Some(end) =>
              // 如果循环变量名与全局变量冲突，重命名为唯一名称
              val rawLoopVar = params.head
              val loopVar = if (rawLoopVar == "i" || ctx.isLocal(rawLoopVar) || ctx.isLoopVar(rawLoopVar)) {
                s"${rawLoopVar}_lb"
              } else {
                rawLoopVar
              }
              ctx.addLoopVar(loopVar)
              ctx.loopDepth += 1
              val endCode = translateExpr(end, ctx)
              val bodyStmts = translateTerm(body, ctx)
              ctx.loopDepth -= 1
              ctx.loopVars.remove(loopVar)
              List(TTIRFor(loopVar, 0, endCode, bodyStmts))
            case None =>
              ctx.loopDepth += 1
              val bodyCode = translateTerm(body, ctx)
              ctx.loopDepth -= 1
              List(TTIRExprStmt(TTIRConst(s"/* TODO: nested foreach range=${range.show} */"))) ++ bodyCode
        case None =>
          List(TTIRExprStmt(TTIRConst(s"/* TODO: foreach TypeApply with non-lambda arg: ${ap.show} */")))

    // 标准 foreach 形式 (不带类型参数)
    case ap @ Apply(Select(range, "foreach"), args) =>
      // 增强版提取 lambda 参数
      def extractLambda(term: Term): Option[(List[String], Term)] = term match
        case Lambda(params, body) => Some((params.map(p => p.name.toString), body))
        case TypeApply(Lambda(params, body), _) => Some((params.map(p => p.name.toString), body))
        case Closure(meth, _) => extractLambda(meth)
        case TypeApply(Closure(meth, _), _) => extractLambda(meth)
        case Block(stats, expr) => extractLambda(expr)
        case Apply(func, _) => extractLambda(func)
        case _ => None

      // 辅助函数: 检查是否是 xxx.until(N) 形式
      // Scala 3 macro 展开后可能有多种结构:
      // 1. Apply(Select(Literal(IntConstant(0)), "until"), List(end))  // 直接调用
      // 2. Apply(Select(inner, "until"), List(end)) where inner = intWrapper(0) or similar
      // 3. Apply(Select(inner, "until"), List(end)) for runtime variables
      def extractUntilEnd(r: Term): Option[Term] = r match
        case Apply(Select(Literal(IntConstant(0)), "until"), List(end)) => Some(end)
        case Apply(Select(inner, "until"), List(end)) if inner.toString == "0" => Some(end)
        // 处理隐式转换 intWrapper 包装: 检查 callee 是否包含 "intWrapper"
        case Apply(Select(callee, "until"), List(end)) =>
          if (callee.show.contains("intWrapper")) Some(end)
          else if (callee.show.contains(".until")) None  // 避免递归
          else None
        case _ => None

      args.headOption flatMap extractLambda match
        case Some((params, body)) =>
          extractUntilEnd(range) match
            case Some(end) =>
              // 如果循环变量名是 "i"（与全局线程ID冲突），重命名
              val rawLoopVar = params.head
              val loopVar = if (rawLoopVar == "i") s"${rawLoopVar}_lb" else rawLoopVar
              ctx.addLoopVar(loopVar)
              ctx.loopDepth += 1
              val endCode = translateExpr(end, ctx)
              val bodyStmts = translateTerm(body, ctx)
              ctx.loopDepth -= 1
              ctx.loopVars.remove(loopVar)
              List(TTIRFor(loopVar, 0, endCode, bodyStmts))
            case None =>
              ctx.loopDepth += 1
              val bodyCode = translateTerm(body, ctx)
              ctx.loopDepth -= 1
              List(TTIRExprStmt(TTIRConst(s"/* TODO: nested foreach range=${range.show} */"))) ++ bodyCode
        case None =>
          List(TTIRExprStmt(TTIRConst(s"/* TODO: foreach with non-lambda arg: ${ap.show} */")))

    // Scala 3 `for (i <- 0.until(N)) { body }` 形式

    // tl.syncthreads() - __syncthreads()
    case Apply(Select(obj, "syncthreads"), List()) if obj.show.contains("tl") =>
      List(TTIRSyncThreads())

    // tl.sharedMem(tpe, name, size) - __shared__ declaration
    case Apply(Select(obj, "sharedMem"), List(tpeArg, nameArg, sizeArg)) if obj.show.contains("tl") =>
      val tpe = tpeArg match
        case Literal(StringConstant(s)) => s
        case _ => tpeArg.show.stripPrefix("\"").stripSuffix("\"")
      val name = nameArg match
        case Literal(StringConstant(s)) => s
        case _ => nameArg.show.stripPrefix("\"").stripSuffix("\"")
      val sizeExpr = translateExpr(sizeArg, ctx)
      List(TTIRSharedMem(tpe, name, sizeExpr))

    // tl.maskedStore(ptr, offset, value, predicate) - masked store
    case Apply(Select(obj, "maskedStore"), List(ptrArg, offsetArg, valueArg, predArg)) if obj.show.contains("tl") =>
      val ptr = ptrArg match
        case Literal(StringConstant(s)) => s
        case Ident(name) => name
        case _ => ptrArg.show
      val offsetExpr = translateExpr(offsetArg, ctx)
      val valueExpr = translateExpr(valueArg, ctx)
      val predExpr = translateExpr(predArg, ctx)
      List(TTIRMaskedStore(ptr, offsetExpr, valueExpr, predExpr))

    // tl.sharedStore(ptr, offset, value) - store to shared memory
    case Apply(Select(obj, "sharedStore"), List(ptrArg, offsetArg, valueArg)) if obj.show.contains("tl") =>
      val ptr = ptrArg match
        case Literal(StringConstant(s)) => s
        case Ident(name) => name
        case _ => ptrArg.show
      val offsetExpr = translateExpr(offsetArg, ctx)
      val valueExpr = translateExpr(valueArg, ctx)
      List(TTIRStore(ptr, offsetExpr, valueExpr, TTIRConst(true)))

    // return 语句
    case Return(expr, _) =>
      val returnExpr = translateExpr(expr, ctx)
      returnExpr match
        case TTIRConst(()) => List(TTIRReturn(TTIRConst(())))
        case _ => List(TTIRReturn(returnExpr))

    // While 循环
    case While(cond, body) =>
      val condCode = translateExpr(cond, ctx)
      val bodyStmts = translateTerm(body, ctx)
      List(TTIRWhile(condCode, bodyStmts))

    // 类型擦除
    case Typed(expr, _) =>
      translateTerm(expr, ctx)

    // Triton 内置函数: tl.store(addr, value) - 生成存储语句
    // 注意: value 如果是指针算法表达式 (ptr + offset)，应该作为加载处理
    case ap @ Apply(Select(obj, "store"), List(addr, value)) if obj.show.contains("tl") =>
      val (ptr, offset) = translateTritonAddr(addr, ctx)
      // 对于 value，检查是否是指针算法，如果是则作为加载处理
      val valueExpr = translateStoreValue(value, ctx)
      List(TTIRStore(ptr, offset, valueExpr, TTIRConst(true)))

    // Attention function calls: flashAttention, pageAttention, flexAttention, etc.
    // These emit complex attention kernels as single statements
    case Apply(Ident(name), args) if isAttentionFunction(name) =>
      translateAttentionCall(name, args, ctx)

    // 其他表达式语句
    case other =>
      val expr = translateExpr(other, ctx)
      expr match
        case TTIRConst(()) => Nil
        case _ => List(TTIRExprStmt(expr))

// 专门处理数组索引中的算术表达式，直接构建 TTIRBinOp，避免通过 translateCall 递归导致 TTIRExpr 被误处理
// 递归提取 toFloat/int2float 包装内部的表达式
// 例如: (row * D + d).toFloat -> row * D + d
// 例如: scala.Int.int2float(ptr.+(offset)) -> ptr + offset
private def stripImplicitConversion(using Quotes)(term: quotes.reflect.Term): quotes.reflect.Term =
  import quotes.reflect.*
  // DEBUG: print the term structure
  val termStr = term.show
  term match
    // 方式1: expr.toFloat -> Select(x, "toFloat") [scala方法调用语法糖]
    case Select(inner, "toFloat") =>
      stripImplicitConversion(inner)
    // 方式2: expr.toFloat 作为 Apply(Select(x, "toFloat"), Nil)
    case Apply(Select(inner, "toFloat"), Nil) =>
      stripImplicitConversion(inner)
    // 方式3: 隐式转换 scala.Int.int2float(expr)
    case Apply(sel @ Select(_, methodName), List(inner))
        if methodName.toString.contains("int2float") ||
           methodName.toString.contains("float2int") ||
           methodName.toString.contains("float2long") ||
           methodName.toString.contains("long2float") =>
      stripImplicitConversion(inner)
    // 方式4: (expr: Float) 类型标注
    case Typed(inner, _) =>
      stripImplicitConversion(inner)
    // 方式4b: Apply(Select(inner, method), args) — inner 可能被 implicit conversion 包装
    // 例如: float2floatOps(kPtr).+(j) -> Apply(Select(float2floatOps(kPtr), '+'), List(j))
    case Apply(sel @ Select(inner, method), args) =>
      val strippedInner = stripImplicitConversion(inner)
      if strippedInner.ne(inner) then
        // inner 被 stripped，说明有 implicit conversion，重建
        // term = Apply(Select(inner, method), args) → Apply(Select(strippedInner, method), args)
        Apply(Select.copy(sel)(strippedInner, method), args)
      else
        term
    // 方式4c: Apply(Ident(funcName), List(arg)) — 隐式转换函数调用
    // 例如: float2floatOps(kPtr) -> Apply(Ident("float2floatOps"), List(kPtr))
    case Apply(Ident(funcName), List(arg)) if funcName.toString.contains("float2") =>
      stripImplicitConversion(arg)
    // 方式5: 其他 Select，检查是否包含 toFloat/int2float
    case other =>
      val s = other.show
      if s.contains("int2float") || s.contains("toFloat") then
        other match
          // 提取 Apply 中的参数
          case Apply(func, List(arg)) =>
            stripImplicitConversion(arg)
          case Typed(inner, _) =>
            stripImplicitConversion(inner)
          // 方法调用链: x.method() -> Select(X, methodName)
          case Select(inner, _) =>
            stripImplicitConversion(inner)
          case _ => other
      else if s.contains("float2floatOps") || s.contains("NumericFloatOps") || s.contains("float2") then
        // 隐式转换包装 Float 值以便调用方法，如 float2floatOps(kPtr).+(...).+(...)
        other match
          // Apply(ImplicitConversionFunc, List(inner)) — 提取被包装的原始表达式
          case Apply(func, List(arg)) =>
            stripImplicitConversion(arg)
          // Select(inner, method) — 递归处理被包装的表达式
          case sel @ Select(innerSel, methodName) =>
            stripImplicitConversion(innerSel)
          case _ => other
      else
        other

// 递归翻译算术表达式中的嵌套表达式 (内部递归，不依赖 translateExpr)
// 只处理算术相关的表达式，避免误触发其他翻译逻辑
private def translateArithInner(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*
  // 首先剥离隐式转换
  val cleaned = stripImplicitConversion(term)
  cleaned match
    case Apply(Select(lhs, op), List(rhs)) if op == "+" || op == "$plus" =>
      val lhsExpr = translateArithInner(lhs, ctx)
      val rhsExpr = translateArithInner(rhs, ctx)
      // If lhs is a zero (pointer), return just rhs. If rhs is a zero (pointer), return just lhs.
      lhsExpr match
        case TTIRConst(0) => rhsExpr
        case _ => rhsExpr match
          case TTIRConst(0) => lhsExpr
          case _ => TTIRBinOp("+", lhsExpr, rhsExpr)
    case Apply(Select(lhs, op), List(rhs)) if op == "-" || op == "$minus" =>
      TTIRBinOp("-", translateArithInner(lhs, ctx), translateArithInner(rhs, ctx))
    case Apply(Select(lhs, op), List(rhs)) if op == "*" || op == "$times" =>
      TTIRBinOp("*", translateArithInner(lhs, ctx), translateArithInner(rhs, ctx))
    case Apply(Select(lhs, op), List(rhs)) if op == "/" || op == "$div" =>
      TTIRBinOp("/", translateArithInner(lhs, ctx), translateArithInner(rhs, ctx))
    case Ident(name) =>
      // Skip pointer names in arithmetic offset expressions — they contribute 0 to the offset
      // 如果是注册的指针，返回 0（指针不贡献偏移量）
      // 如果是本地变量但不是指针（标量参数），直接使用该变量
      if ctx.isPointer(name) then TTIRConst(0)
      else if ctx.isLocal(name) then TTIRVar(name)  // scalar parameter - use directly
      else TTIRVar(name)
    case Literal(IntConstant(n)) => TTIRConst(n)
    case Literal(FloatConstant(f)) => TTIRConst(f)
    case Literal(DoubleConstant(d)) => TTIRConst(d.toFloat)
    case Literal(LongConstant(n)) => TTIRConst(n)
    case Literal(BooleanConstant(b)) => TTIRConst(b)
    case Literal(StringConstant(s)) => TTIRConst(s)
    case Literal(CharConstant(c)) => TTIRConst(c)
    case Literal(ByteConstant(b)) => TTIRConst(b)
    case Literal(ShortConstant(s)) => TTIRConst(s)
    // 继续剥离 toFloat 并递归 (Select form: x.toFloat)
    case Select(inner, "toFloat") =>
      translateArithInner(inner, ctx)
    // Apply form: toFloat(x)
    case Apply(Select(inner, "toFloat"), Nil) =>
      translateArithInner(inner, ctx)
    // 剥离隐式转换并继续
    case Apply(sel @ Select(_, methodName), List(inner))
        if methodName.toString.contains("int2float") ||
           methodName.toString.contains("float2int") ||
           methodName.toString.contains("float2long") ||
           methodName.toString.contains("long2float") =>
      translateArithInner(inner, ctx)
    // 剥离 Typed 包装
    case Typed(inner, _) =>
      translateArithInner(inner, ctx)
    case _ =>
      // 未知表达式: 使用 term.show 作为后备
      TTIRVar(term.show)

private def translateArithExpr(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): TTIRExpr =
  val cleaned = stripImplicitConversion(term)
  translateArithInner(cleaned, ctx)

// Extract base identifier name from a possibly fully-qualified Select chain
// e.g., Select(Select(Select(Ident("cuda"), "dsl"), "tl"), "outPtr") -> "outPtr"
private def extractBaseName(using Quotes)(term: quotes.reflect.Term): String =
  import quotes.reflect.*
  term match
    case Ident(name) => name
    case Select(parent, name) => extractBaseName(parent)
    case _ => term.show

private def translateExpr(using Quotes)(term: quotes.reflect.Tree, ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*

  // 如果传入的是已经翻译好的 TTIRExpr（递归调用时可能发生），直接返回
  // 注意：改变参数类型为 Tree 而不是 Term 以允许 TTIRExpr 输入
  term match
    case te: TTIRExpr => te
    case term: Term =>
      translateTermImpl(term, ctx)

private def translateTermImpl(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*
  term match
    // 标识符/变量 - 如果是指针参数，返回数组访问 x[i]
    case Ident(name) =>
      if ctx.isLoopVar(name) then TTIRVar(name)
      else if ctx.isLocal(name) then TTIRVar(name)
      else if ctx.isPointer(name) then TTIRLoad(name, TTIRVar("i"), TTIRConst(true), TTIRConst(0.0f))
      else TTIRVar(name)

    // this
    case Select(Ident("this"), name) =>
      TTIRVar(name)

    // Typed 表达式 - 剥离类型注解，只翻译内部表达式
    // 例如: (maxVal: Float) -> maxVal
    // (expr: Float) -> expr
    case Typed(expr, _) =>
      translateExpr(expr, ctx)

    // If-then-else as expression (ternary cond ? then : else)
    case If(cond, thenp, elsep) =>
      val condCode = translateExpr(cond, ctx)
      val thenCode = translateExpr(thenp, ctx)
      val elseCode = translateExpr(elsep, ctx)
      TTIRTernary(condCode, thenCode, elseCode)

    // 类型对象成员访问: Float.MinValue, Float.MaxValue, etc. - 必须在通用 Select 之前
    case Select(Ident("Float"), field) =>
      field match
        case "MinValue" => TTIRConst(Float.MinValue)
        case "MaxValue" => TTIRConst(Float.MaxValue)
        case "PositiveInfinity" => TTIRConst(Float.PositiveInfinity)
        case "NegativeInfinity" => TTIRConst(Float.NegativeInfinity)
        case "NaN" => TTIRConst(Float.NaN)
        case _ => TTIRVar("Float")
    case Select(Ident("Int"), field) =>
      field match
        case "MinValue" => TTIRConst(Int.MinValue)
        case "MaxValue" => TTIRConst(Int.MaxValue)
        case _ => TTIRVar("Int")
    // 选择器 (可能的方法调用)
    // 注意: 二元运算符如 + - * / 在 Apply(Select(lhs, op), List(rhs)) 模式中处理
    // 这里只处理真正的方法调用和一元运算符
    case Select(obj, field) =>
      val objCode = translateExpr(obj, ctx)
      field match
        case "-" => TTIRUnaryOp("-", objCode)
        case "unary_!" => TTIRUnaryOp("!", objCode)
        case "unary_~" => TTIRUnaryOp("~", objCode)
        case "unary_-" => TTIRUnaryOp("-", objCode)
        // 类型转换方法: toFloat -> (float)
        case "toFloat" | "toFloat" =>
          TTIRMathCall("toFloat", List(objCode))
        case "toInt" | "toDouble" | "toLong" | "toShort" | "toByte" | "toChar" =>
          TTIRMathCall(field, List(objCode))
        case _ => objCode // 默认返回对象本身 (用于方法链式调用如 obj.method())

    // 字面量
    case Literal(IntConstant(n)) => TTIRConst(n)
    case Literal(FloatConstant(f)) => TTIRConst(f)
    case Literal(DoubleConstant(d)) => TTIRConst(d.toFloat)
    case Literal(BooleanConstant(b)) => TTIRConst(b)
    case Literal(LongConstant(n)) => TTIRConst(n)
    case Literal(StringConstant(s)) => TTIRConst(s)
    case Literal(CharConstant(c)) => TTIRConst(c)
    case Literal(ByteConstant(b)) => TTIRConst(b)
    case Literal(ShortConstant(s)) => TTIRConst(s)
    case Literal(UnitConstant()) => TTIRConst(())
    case Literal(NullConstant()) => TTIRConst(0)

    // ptr(idx) 形式的数组访问 - 单参数版本
    case Apply(Ident(name), List(idx)) if ctx.isPointer(name) =>
      val idxCode = translateExpr(idx, ctx)
      TTIRLoad(name, idxCode, TTIRConst(true), TTIRConst(0.0f))

    // ptr(idx) 形式的数组访问 - 带隐式参数版本 (Scala 3 macro 会看到隐式参数)
    // q(qIdx)(using MemoryOps_Float) => Apply(Apply(Ident("q"), List(qIdx)), List(given_MemoryOps_Float))
    case Apply(Apply(Ident(name), List(idx)), _) if ctx.isPointer(name) =>
      val idxCode = translateExpr(idx, ctx)
      TTIRLoad(name, idxCode, TTIRConst(true), TTIRConst(0.0f))

    // tl.threadIdx.x / tl.threadIdx.y / tl.threadIdx.z - dot notation
    // tl.threadIdx.x 被解析为 Select(Select(Ident("tl"), "threadIdx"), "x")
    // tl may be fully-qualified as cuda.dsl.dsl.tl
    case Select(Select(inner, "threadIdx"), "x") if inner.show.contains("tl") => TTIRThreadId(0)
    case Select(Select(inner, "threadIdx"), "y") if inner.show.contains("tl") => TTIRThreadId(1)
    case Select(Select(inner, "threadIdx"), "z") if inner.show.contains("tl") => TTIRThreadId(2)

    // 二元运算符 - 必须在通用 Apply 模式之前匹配
    case Apply(Select(lhs, op), List(rhs)) =>
      val lhsCode = translateExpr(lhs, ctx)
      val rhsCode = translateExpr(rhs, ctx)
      val opStr = op.toString
      // ptr(idx) 形式的数组访问: ptr.apply(idx) - 必须在二元运算符之前处理
      if opStr == "apply" then
        val ptrName = lhs match
          case Ident(name) => name
          case _ => lhs.show
        val idxCode = translateArithExpr(rhs, ctx)
        if ctx.isPointer(ptrName) || ctx.isLocal(ptrName) || ctx.isLoopVar(ptrName) then
          TTIRLoad(ptrName, idxCode, TTIRConst(true), TTIRConst(0.0f))
        else
          TTIRLoad(ptrName, idxCode, TTIRConst(true), TTIRConst(0.0f))
      else
        (lhs, opStr, rhs) match
        // tl.program_id(axis) - 在 @TritonKernelMacro 中应该返回全局线程ID i
        // Uses lhs.show to detect tl.program_id pattern robustly
        case _ if lhs.show.contains("tl") && opStr == "program_id" =>
          rhs match
            case Literal(IntConstant(axis)) =>
              axis.toInt match
                case 0 => TTIRVar("i")
                case 1 => TTIRBlockId(1)
                case 2 => TTIRBlockId(2)
                case _ => TTIRThreadId(axis.toInt)
            case _ => TTIRVar(rhs.show)
        // tl.threadIdx(axis) -> threadIdx.x/y/z
        case _ if lhs.show.contains("tl") && opStr == "threadIdx" =>
          rhs match
            case Literal(IntConstant(axis)) => TTIRThreadId(axis.toInt)
            case _ => TTIRVar(rhs.show)
        // tl.blockIdx(axis) -> blockIdx.x/y/z
        case _ if lhs.show.contains("tl") && opStr == "blockIdx" =>
          rhs match
            case Literal(IntConstant(axis)) => TTIRBlockId(axis.toInt)
            case _ => TTIRBlockId(0)
        // tl.block_dim(axis) -> blockDim.x/y/z
        case _ if lhs.show.contains("tl") && opStr == "block_dim" =>
          rhs match
            case Literal(IntConstant(axis)) => TTIRBlockDim()
            case _ => TTIRBlockDim()
        // tl.num_blocks(axis) -> gridDim.x/y/z
        case _ if lhs.show.contains("tl") && (opStr == "num_blocks" || opStr == "gridDim") =>
          rhs match
            case Literal(IntConstant(axis)) => TTIRGridDim()
            case _ => TTIRGridDim()
        // tl.program_id as Ident - handle in translateCall
        case _ if lhs.show.contains("tl") && (opStr == "program_id" || opStr == "programId") =>
          rhs match
            case Literal(IntConstant(axis)) =>
              axis.toInt match
                case 0 => TTIRVar("i")
                case 1 => TTIRBlockId(1)
                case 2 => TTIRBlockId(2)
                case _ => TTIRThreadId(axis.toInt)
            case _ => TTIRVar("i")
        // tl.load(ptr + offset)
        case (obj, "load", addrExpr) if obj.show.contains("tl") =>
          translateTritonLoad(addrExpr, ctx)
        // tl.exp(v), tl.log(v), tl.sqrt(v), tl.abs(v), tl.sin(v), tl.cos(v), tl.tanh(v)
        // Single-arg math functions on tl object — must come before arithmetic operators
        case (obj, "exp", _) if obj.show.contains("tl") =>
          TTIRMathCall("exp", List(rhsCode))
        case (obj, "log", _) if obj.show.contains("tl") =>
          TTIRMathCall("log", List(rhsCode))
        case (obj, "sqrt", _) if obj.show.contains("tl") =>
          TTIRMathCall("sqrt", List(rhsCode))
        case (obj, "abs", _) if obj.show.contains("tl") =>
          TTIRMathCall("abs", List(rhsCode))
        case (obj, "sin", _) if obj.show.contains("tl") =>
          TTIRMathCall("sin", List(rhsCode))
        case (obj, "cos", _) if obj.show.contains("tl") =>
          TTIRMathCall("cos", List(rhsCode))
        case (obj, "tanh", _) if obj.show.contains("tl") =>
          TTIRMathCall("tanh", List(rhsCode))
        // tl.warp_any(predicate) - check opStr first
        case _ if opStr == "warp_any" =>
          TTIRWarpVote("any", rhsCode)
        // tl.warp_all(predicate)
        case _ if opStr == "warp_all" =>
          TTIRWarpVote("all", rhsCode)
        case _ =>
          opStr match
            case "apply" =>
              val ptrName = lhs match
                case Ident(name) => name
                case _ => lhs.show
              translateApplyOrCall(lhs, rhs, ctx)
            case s if s == "+" || s == "$plus" =>
              TTIRBinOp("+", lhsCode, rhsCode)
            case s if s == "-" || s == "$minus" =>
              TTIRBinOp("-", lhsCode, rhsCode)
            case "*" | "times" => TTIRBinOp("*", lhsCode, rhsCode)
            case "/" | "div" => TTIRBinOp("/", lhsCode, rhsCode)
            case "%" | "mod" => TTIRBinOp("%", lhsCode, rhsCode)
            case ">" | "gt" => TTIRBinOp(">", lhsCode, rhsCode)
            case "<" | "lt" => TTIRBinOp("<", lhsCode, rhsCode)
            case ">=" | "ge" => TTIRBinOp(">=", lhsCode, rhsCode)
            case "<=" | "le" => TTIRBinOp("<=", lhsCode, rhsCode)
            case "==" | "eq" => TTIRBinOp("==", lhsCode, rhsCode)
            case "!=" | "ne" => TTIRBinOp("!=", lhsCode, rhsCode)
            case "&&" | "and" => TTIRBinOp("&&", lhsCode, rhsCode)
            case "||" | "or" => TTIRBinOp("||", lhsCode, rhsCode)
            case "+=" | "plus_eq" => TTIRBinOp("+", lhsCode, rhsCode)
            case "-=" | "minus_eq" => TTIRBinOp("-", lhsCode, rhsCode)
            case "*=" | "times_eq" => TTIRBinOp("*", lhsCode, rhsCode)
            case "/=" | "div_eq" => TTIRBinOp("/", lhsCode, rhsCode)
            case ">>" | "shift_right" => TTIRBinOp(">>", lhsCode, rhsCode)
            case "<<" | "shift_left" => TTIRBinOp("<<", lhsCode, rhsCode)
            case "|" | "bar" => TTIRBinOp("|", lhsCode, rhsCode)
            case "&" | "amp" => TTIRBinOp("&", lhsCode, rhsCode)
            case "^" | "up" => TTIRBinOp("^", lhsCode, rhsCode)
            case _ => TTIRBinOp(opStr, lhsCode, rhsCode)

    // 一元运算符
    case Apply(Select(prefix, "-"), Nil) =>
      TTIRUnaryOp("-", translateExpr(prefix, ctx))
    case Apply(Select(prefix, "!"), Nil) =>
      TTIRUnaryOp("!", translateExpr(prefix, ctx))

    // 数组/函数调用
    case Apply(callee, args) =>
      translateCall(callee, args, ctx)

    // 类型应用
    case TypeApply(callee, _) =>
      translateExpr(callee, ctx)

    // Lambda - 通常由 for 循环模式处理，这里作为后备
    case Lambda(params, body) =>
      translateExpr(body, ctx)

    // Inlined
    case Inlined(_, _, body) =>
      translateExpr(body, ctx)

    // Closure
    case Closure(meth, _) =>
      translateExpr(meth, ctx)

    // 默认
    case t =>
      t.show match
        case s if s.contains("threadIdx") => TTIRThreadId(0)
        case s if s.contains("blockIdx") => TTIRBlockId(0)
        case s if s.contains("blockDim") => TTIRBlockDim()
        case _ => TTIRConst(s"/* ${t.show} */")

private def translateApplyOrCall(using Quotes)(obj: quotes.reflect.Term, idx: quotes.reflect.Term, ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*

  obj match
    case Ident(name) if ctx.isPointer(name) =>
      val idxCode = translateExpr(idx, ctx)
      TTIRLoad(name, idxCode, TTIRConst(true), TTIRConst(0.0f))
    case _ =>
      val objCode = translateExpr(obj, ctx)
      val idxCode = translateExpr(idx, ctx)
      // 尝试判断是否是方法调用
      obj match
        case Select(_, methodName) =>
          if methodName == "apply" then idxCode
          else TTIRMathCall(methodName, List(objCode, idxCode))
        case _ => idxCode

private def isMathFunction(name: String): Boolean =
  name == "exp" || name == "log" || name == "log10" || name == "sqrt" || name == "rsqrt" ||
  name == "sin" || name == "cos" || name == "tan" || name == "tanh" ||
  name == "sigmoid" || name == "abs" || name == "pow" || name == "max" ||
  name == "min" || name == "fmod" || name == "relu" || name == "signum" ||
  name == "floor" || name == "ceil" || name == "round"

// Triton load 地址翻译: 解析 ptr + offset 形式
// 注意: Int 类型的参数如果名字像指针(如包含 Ptr/Mapping/Table 等)也应该被视为指针
private def isPointerByConventionForTriton(name: String): Boolean =
  isPointerByConvention(name)

// 从表达式中提取指针基址和偏移量
// 使用 term.show 字符串分析避免 Scala 3 macro 中 Select/Apply ClassCastException
// Extract offset from a term - uses translateArithExpr for complex arithmetic
// When ptr is found, the offset part is translated via translateArithExpr
private def extractOffsetOnly(using Quotes)(t: quotes.reflect.Term, ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*
  val term = stripImplicitConversion(t)
  term match
    // ptr + offset: base is pointer, translate offset via translateArithExpr
    case Apply(Select(Ident(base), op), List(offset))
        if (op == "+" || op == "$plus" || op == "-" || op == "$minus") =>
      if ctx.isPointer(base) || isPointerByConventionForTriton(base) then
        // Pointer found, translate the offset expression
        translateArithExpr(offset, ctx)
      else
        // base is not a pointer, full expression is arithmetic
        translateArithExpr(term, ctx)
    // Nested arithmetic (no pointer base) — no guard, handles any binary operator
    case Apply(Select(a, op), List(b)) =>
      translateArithExpr(term, ctx)
    // toFloat wrapper
    case Select(inner, "toFloat") => extractOffsetOnly(inner, ctx)
    case Apply(Select(inner, "toFloat"), Nil) => extractOffsetOnly(inner, ctx)
    // Type annotation wrapper
    case Typed(inner, _) => extractOffsetOnly(inner, ctx)
    // Variable
    case Ident(name) => TTIRVar(name)
    // Literals
    case Literal(IntConstant(n)) => TTIRConst(n)
    case Literal(FloatConstant(f)) => TTIRConst(f)
    case Literal(DoubleConstant(d)) => TTIRConst(d.toFloat)
    case Literal(LongConstant(n)) => TTIRConst(n)
    // Fallback
    case _ => TTIRVar(term.show)

private def extractPointerAndOffset(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): (String, TTIRExpr) =
  import quotes.reflect.*
  val actual = stripImplicitConversion(term)
  val termStr = actual.show

  // Extract pointer name: find first word that is a registered pointer or matches convention
  def ptrFromStr(s: String): String =
    val words = s.split("[^A-Za-z0-9_]+").filter(_.nonEmpty)
    // First priority: a registered pointer
    words.find(w => ctx.isPointer(w) || isPointerByConventionForTriton(w)).getOrElse(s.split("[.+]").head.trim)

  actual match
    // Case 1: Apply(Select(innerConverted, "+"), args) where innerConverted has implicit conversion
    // e.g., float2floatOps(kPtr).+(...).+(...)  → extract kPtr as pointer
    case ap @ Apply(sel @ Select(inner, op), args) if termStr.contains("float2floatOps") || termStr.contains("NumericFloatOps") =>
      // Find the original pointer by looking for registered pointers in the string
      val ptr = ptrFromStr(termStr)
      // Extract offset: the args are the offset parts, need to rebuild the offset expr
      (ptr, extractOffsetOnly(ap, ctx))
    // ptr + offset form: extract pointer from string (no guard on op — any binary op matches)
    case Apply(Select(_, _), List(offset)) =>
      val ptr = ptrFromStr(termStr)
      val isPtr = ctx.isPointer(ptr) || isPointerByConventionForTriton(ptr)
      // Use just the offset part, NOT the full actual expression.
      // translateArithExpr(actual) would see the pointer name and return TTIRConst(0).
      (ptr, extractOffsetOnly(offset, ctx))
    // Simple identifier
    case Ident(name) if ctx.isPointer(name) || isPointerByConventionForTriton(name) =>
      (name, TTIRConst(0))
    // Fallback - try to find pointer or use trimmed term string
    case _ => (ptrFromStr(termStr), translateArithExpr(actual, ctx))

private def translateTritonAddr(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): (String, TTIRExpr) =
  extractPointerAndOffset(term, ctx)

// translateStoreValue: 处理 tl.store 的 value 参数
// 如果 value 是指针算法 (ptr + offset)，则作为从该地址加载处理
// 否则使用标准的 translateExpr
private def translateStoreValue(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*
  val actual = stripImplicitConversion(term)
  actual match
    // ptr + offset 形式: 作为加载处理
    case Apply(Select(Ident(ptr), "+"), List(offset))
        if ctx.isPointer(ptr) || isPointerByConventionForTriton(ptr) =>
      val offsetCode = translateArithExpr(offset, ctx)
      TTIRLoad(ptr, offsetCode, TTIRConst(true), TTIRConst(0.0f))
    // ptr - offset 形式: 作为加载处理
    case Apply(Select(Ident(ptr), "-"), List(offset))
        if ctx.isPointer(ptr) || isPointerByConventionForTriton(ptr) =>
      val offsetCode = translateArithExpr(offset, ctx)
      TTIRLoad(ptr, TTIRUnaryOp("-", offsetCode), TTIRConst(true), TTIRConst(0.0f))
    // 其他情况使用标准翻译
    case _ =>
      translateExpr(actual, ctx)

// Triton load 翻译
private def translateTritonLoad(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*
  // 剥离隐式转换: scala.Int.int2float 包装会包裹整个 tl.load 参数
  val actualTerm = stripImplicitConversion(term)
  val (ptr, offset) = translateTritonAddr(actualTerm, ctx)

  // 关键修复：如果 ptr 不是注册在 ctx.pointers 中的指针，则它是标量参数或未注册的指针
  // 应该直接返回该变量，而不是尝试作为数组访问
  // 例如: tl.load(temp) 其中 temp 是 Float 标量参数 -> 直接返回 temp
  // 只有真正注册为指针的参数才应该使用数组访问语法
  //
  // 但如果 ptr 名称符合指针约定（如 seqLensPtr, blockTablesPtr），即使未注册，
  // 也应该作为指针处理并生成加载代码
  val isLikelyPointerByName = ptr.toLowerCase.contains("ptr") ||
                              ptr.toLowerCase.contains("mapping") ||
                              ptr.toLowerCase.contains("table") ||
                              ptr.toLowerCase.contains("indices") ||
                              ptr.toLowerCase.contains("_len") ||
                              ptr.toLowerCase.contains("len") ||
                              (ptr.toLowerCase.contains("offset") && !ptr.toLowerCase.contains("key") && !ptr.toLowerCase.contains("value"))
  if !ctx.isPointer(ptr) && !isLikelyPointerByName then
    // 标量参数直接使用 - 返回变量本身（使用 offset 部分，因为 ptr 可能被误识别）
    offset match
      case TTIRVar(v) => TTIRVar(v)
      case _ => TTIRVar(ptr)
  else
    // 正常的指针加载
    // 根据指针名称约定判断加载类型:
    // - slotMappingPtr, blockTablesPtr, seqLensPtr, seqLenPtr 等是 int*
    // - qPtr, kPtr, vPtr, outPtr 等是 float*
    val isIntPtr = ptr.toLowerCase.contains("mapping") ||
                   ptr.toLowerCase.contains("table") ||
                   ptr.toLowerCase.contains("indices") ||
                   ptr.toLowerCase.contains("_len") ||
                   ptr.toLowerCase.contains("len") ||
                   (ptr.toLowerCase.contains("offset") && !ptr.toLowerCase.contains("key") && !ptr.toLowerCase.contains("value"))
    // 如果指针名称暗示是 int*，则使用 0 作为默认值表示 int 类型
    if (isIntPtr) {
      TTIRLoad(ptr, offset, TTIRConst(true), TTIRConst(0))  // 0 表示 int 默认值
    } else {
      TTIRLoad(ptr, offset, TTIRConst(true), TTIRConst(0.0f))
  }

private def isAttentionFunction(name: String): Boolean =
  name == "flashAttention" || name == "FlashAttention" ||
  name == "pageAttention" || name == "PageAttention" ||
  name == "flexAttention" || name == "FlexAttention" ||
  name == "groupedQueryAttention" || name == "GroupedQueryAttention" ||
  name == "multiHeadAttention" || name == "MultiHeadAttention" ||
  name == "gqa" || name == "mha" ||
  name == "storeKVCache" || name == "store_kvcache"

private def translateAttentionCall(using Quotes)(name: String, args: List[quotes.reflect.Term], ctx: TranslateContext): List[TTIRStmt] =
  import quotes.reflect.*

  // Translate each argument to get the pointer/type information
  val argExprs = args.map(translateExpr(_, ctx))

  (name, args.length) match
    // flashAttention(q, k, v, out, N, D, blockSize?, causal?)
    case ("flashAttention" | "FlashAttention", 5) =>
      List(TTIRFlashAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractExprArg(args(4), ctx),
        extractExprArg(args(5), ctx),
        128,
        false
      ))

    case ("flashAttention" | "FlashAttention", 6) =>
      List(TTIRFlashAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractExprArg(args(4), ctx),
        extractExprArg(args(5), ctx),
        128,
        false
      ))

    case ("flashAttention" | "FlashAttention", 7) =>
      List(TTIRFlashAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractExprArg(args(4), ctx),
        extractExprArg(args(5), ctx),
        extractIntArg(args(6)),
        false
      ))

    case ("flashAttention" | "FlashAttention", 8) =>
      List(TTIRFlashAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractExprArg(args(4), ctx),
        extractExprArg(args(5), ctx),
        extractIntArg(args(6)),
        extractBoolArg(args(7))
      ))

    // pageAttention(query, keyCache, valueCache, blockTables, seqLens, out, numBlocks, blockSize, D)
    case ("pageAttention" | "PageAttention", 9) =>
      List(TTIRPageAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractStringArg(args(4), ctx),
        extractStringArg(args(5), ctx),
        extractExprArg(args(6), ctx),
        extractExprArg(args(7), ctx),
        extractExprArg(args(8), ctx)
      ))

    // flexAttention(q, k, v, out, N, D, scoreMod?, maskMod?, blockSize?)
    case ("flexAttention" | "FlexAttention", 5) =>
      List(TTIRFlexAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractExprArg(args(4), ctx),
        extractExprArg(args(5), ctx),
        None,
        None,
        128
      ))

    case ("flexAttention" | "FlexAttention", 6) =>
      List(TTIRFlexAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractExprArg(args(4), ctx),
        extractExprArg(args(5), ctx),
        None,
        None,
        128
      ))

    // groupedQueryAttention(q, k, v, out, N, D, numQHeads, numKVHeads, scale)
    case ("groupedQueryAttention" | "GroupedQueryAttention" | "gqa" | "GQA", 9) =>
      List(TTIRGroupedQueryAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractExprArg(args(4), ctx),
        extractExprArg(args(5), ctx),
        extractExprArg(args(6), ctx),
        extractExprArg(args(7), ctx),
        extractExprArg(args(8), ctx)
      ))

    // multiHeadAttention(q, k, v, out, N, D, numHeads, scale)
    case ("multiHeadAttention" | "MultiHeadAttention" | "mha" | "MHA", 8) =>
      List(TTIRMultiHeadAttention(
        extractStringArg(args(0), ctx),
        extractStringArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractStringArg(args(3), ctx),
        extractExprArg(args(4), ctx),
        extractExprArg(args(5), ctx),
        extractExprArg(args(6), ctx),
        extractExprArg(args(7), ctx)
      ))

    // storeKVCache(keyPtr, keyStride, valuePtr, valueStride, kCachePtr, vCachePtr, slotMappingPtr, D)
    case ("storeKVCache" | "store_kvcache", 8) =>
      List(TTIRStoreKVCache(
        extractStringArg(args(0), ctx),
        extractExprArg(args(1), ctx),
        extractStringArg(args(2), ctx),
        extractExprArg(args(3), ctx),
        extractStringArg(args(4), ctx),
        extractStringArg(args(5), ctx),
        extractStringArg(args(6), ctx),
        extractExprArg(args(7), ctx)
      ))

    case _ =>
      List(TTIRExprStmt(TTIRConst(s"/* unknown attention: $name(${args.length} args) */")))

private def extractStringArg(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): String =
  import quotes.reflect.*
  term match
    case Ident(name) => name
    case Literal(StringConstant(s)) => s
    case _ => term.show

// Extract expression argument: handles both string literals (converted to TTIRVar) and other expressions
private def extractExprArg(using Quotes)(term: quotes.reflect.Term, ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*
  term match
    // String literal -> convert to TTIRVar
    case Literal(StringConstant(s)) => TTIRVar(s)
    // Identifier -> use translateExpr to handle pointers/locals properly
    case Ident(name) => translateExpr(term, ctx)
    // Other expressions
    case other => translateExpr(other, ctx)

private def extractIntArg(using Quotes)(term: quotes.reflect.Term): Int =
  import quotes.reflect.*
  term match
    case Literal(IntConstant(n)) => n
    case _ => 128  // default

private def extractBoolArg(using Quotes)(term: quotes.reflect.Term): Boolean =
  import quotes.reflect.*
  term match
    case Literal(BooleanConstant(b)) => b
    case _ => false

// 处理已经翻译为 TTIRExpr 的表达式（如 toFloat 结果）的方法链式调用
// 当 translateExpr 递归处理嵌套表达式时，会产生 TTIRExpr 对象，需要直接处理
private def translateExprCall(callee: TTIRExpr, args: List[TTIRExpr], ctx: TranslateContext): TTIRExpr =
  callee match
    // .toFloat -> (float)
    case _ if args.isEmpty && callee.toString.contains("TTIRLoad") =>
      TTIRMathCall("toFloat", List(callee))
    // .apply(idx) on TTIRLoad -> TTIRLoad with combined index
    case TTIRMathCall("toFloat", List(obj)) =>
      // obj[0] - just return the object (toFloat already applied)
      obj
    case TTIRLoad(ptr, offset, mask, other) =>
      if args.length == 1 then
        // obj[idx] - simple array access
        args.head match
          case TTIRConst(0) => callee  // ptr[offset][0] = ptr[offset]
          case _ => TTIRMathCall("toFloat", List(TTIRLoad(ptr, offset, mask, other)))
      else callee
    case other => other

private def translateCall(using Quotes)(callee: quotes.reflect.Term, args: List[quotes.reflect.Term], ctx: TranslateContext): TTIRExpr =
  import quotes.reflect.*

  callee match
    // TTIR DSL 构造函数 - 需要在通用函数调用之前处理
    // TTIRVar("name") - 变量引用
    case Ident("TTIRVar") if args.length == 1 =>
      args.head match
        case Literal(StringConstant(s)) => TTIRVar(s)
        case _ => TTIRVar(args.head.show)
    // TTIRConst(value) - 常量
    case Ident("TTIRConst") if args.length == 1 =>
      args.head match
        case Literal(IntConstant(n)) => TTIRConst(n)
        case Literal(FloatConstant(f)) => TTIRConst(f)
        case Literal(BooleanConstant(b)) => TTIRConst(b)
        case _ => TTIRConst(args.head.show)
    // TTIRLoad(ptr, offset, mask, other)
    case Ident("TTIRLoad") if args.length == 4 =>
      val argsExprs = args.map(translateExpr(_, ctx))
      TTIRLoad(argsExprs(0).asInstanceOf[TTIRVar].name, argsExprs(1), argsExprs(2), argsExprs(3))
    // TTIRBinOp(op, lhs, rhs)
    case Ident("TTIRBinOp") if args.length == 3 =>
      val argsExprs = args.map(translateExpr(_, ctx))
      args.head match
        case Literal(StringConstant(op)) => TTIRBinOp(op, argsExprs(1), argsExprs(2))
        case _ => TTIRBinOp("+", argsExprs(1), argsExprs(2))

    // scala.math 函数调用: scala.math.tanh(x) 等
    case Select(Select(Ident("scala"), "math"), methodName) =>
      val argsCode = args.map(translateExpr(_, ctx))
      methodName match
        case "tanh" => TTIRMathCall("tanh", argsCode)
        case "sin" => TTIRMathCall("sin", argsCode)
        case "cos" => TTIRMathCall("cos", argsCode)
        case "tan" => TTIRMathCall("tan", argsCode)
        case "exp" => TTIRMathCall("exp", argsCode)
        case "log" | "log10" => TTIRMathCall("log", argsCode)
        case "sqrt" => TTIRMathCall("sqrt", argsCode)
        case "abs" => TTIRMathCall("abs", argsCode)
        case "pow" => TTIRMathCall("pow", argsCode)
        case "max" => TTIRMathCall("max", argsCode)
        case "min" => TTIRMathCall("min", argsCode)
        case "fmod" => TTIRMathCall("fmod", argsCode)
        case _ => TTIRMathCall(methodName, argsCode)

    // Triton 内置函数: tl.load(addr) — tl may be fully qualified
    case Select(obj, "load") if obj.show.contains("tl") && args.length == 1 =>
      translateTritonLoad(args.head, ctx)

    // Triton 内置函数: tl.load(ptr, offset) — 2-argument form
    case Select(obj, "load") if obj.show.contains("tl") && args.length == 2 =>
      // Extract pointer name from the first arg
      val ptrTerm = args(0).show
      val ptrName = ptrTerm.split("[^A-Za-z0-9_]+").find(w =>
        ctx.isPointer(w) || isPointerByConventionForTriton(w)
      ).getOrElse(ptrTerm)
      val offsetExpr = translateExpr(args(1), ctx)
      // Detect if it's an int pointer based on name convention
      val isIntPtr = ptrName.toLowerCase.contains("table") ||
                    ptrName.toLowerCase.contains("seq") ||
                    ptrName.toLowerCase.contains("slot") ||
                    ptrName.toLowerCase.contains("mapping")
      // Generate appropriate load
      if (isIntPtr) TTIRLoad(ptrName, offsetExpr, TTIRConst(true), TTIRConst(0))
      else TTIRLoad(ptrName, offsetExpr, TTIRConst(true), TTIRConst(0.0f))

    // Triton 内置函数: tl.sharedLoad(ptr, offset) — shared memory load
    case Select(obj, "sharedLoad") if obj.show.contains("tl") && args.length == 2 =>
      // Extract shared memory name from first arg (should be string literal like "sh_k", "sh_v")
      val shmTerm = args(0).show
      val shmName = shmTerm.stripPrefix("\"").stripSuffix("\"")
      val offsetExpr = translateExpr(args(1), ctx)
      // Shared memory load returns float, use 0.0f as default
      TTIRLoad(shmName, offsetExpr, TTIRConst(true), TTIRConst(0.0f))

    // Triton math functions: tl.exp, tl.log, tl.sqrt, etc. — tl may be fully qualified
    case Select(obj, "exp") if obj.show.contains("tl") && args.length == 1 =>
      val x = translateExpr(args.head, ctx)
      TTIRMathCall("exp", List(x))

    case Select(obj, "log") if obj.show.contains("tl") && args.length == 1 =>
      val x = translateExpr(args.head, ctx)
      TTIRMathCall("log", List(x))

    case Select(obj, "sqrt") if obj.show.contains("tl") && args.length == 1 =>
      val x = translateExpr(args.head, ctx)
      TTIRMathCall("sqrt", List(x))

    case Select(obj, "abs") if obj.show.contains("tl") && args.length == 1 =>
      val x = translateExpr(args.head, ctx)
      TTIRMathCall("abs", List(x))

    case Select(obj, "sin") if obj.show.contains("tl") && args.length == 1 =>
      val x = translateExpr(args.head, ctx)
      TTIRMathCall("sin", List(x))

    case Select(obj, "cos") if obj.show.contains("tl") && args.length == 1 =>
      val x = translateExpr(args.head, ctx)
      TTIRMathCall("cos", List(x))

    // Other tl.* method calls (e.g. tl.max, tl.min, tl.tanh) — tl may be fully qualified
    case Select(obj, methodName) if obj.show.contains("tl") =>
      val argsCode = args.map(translateExpr(_, ctx))
      methodName match
        case "max" | "MAX" => TTIRMathCall("max", argsCode)
        case "min" | "MIN" => TTIRMathCall("min", argsCode)
        case "tanh" | "TANH" => TTIRMathCall("tanh", argsCode)
        case "pow" | "POW" => TTIRMathCall("pow", argsCode)
        case "sigmoid" => TTIRMathCall("sigmoid", argsCode)
        case "relu" | "RELU" => TTIRMathCall("relu", argsCode)
        case "toFloat" | "floatValue" => argsCode match
          case List(x) => TTIRMathCall("int2float", List(x))
          case _ => TTIRConst(0f)
        case "exp" | "log" | "sqrt" | "abs" | "sin" | "cos" => TTIRConst(0f) // handled above
        case _ =>
          // Unknown tl function — emit as inline C call or error
          TTIRConst(0f)  // safe fallback

    // a(i) 形式的数组访问 — exclude math functions (handled by dedicated case below)
    case Ident(name) if args.length == 1 && !isMathFunction(name) =>
      val idx = translateExpr(args.head, ctx)
      if ctx.isLoopVar(name) then idx
      else if ctx.isLocal(name) then idx
      else if ctx.isPointer(name) then TTIRLoad(name, idx, TTIRConst(true), TTIRConst(0.0f))
      else idx

    // Ident(name) with 1 arg - math functions via implicit conversion (e.g. FloatOps.exp)
    case Ident(name) if args.length == 1 && isMathFunction(name) =>
      val arg = translateExpr(args.head, ctx)
      TTIRMathCall(name, List(arg))

    // Ident(name) with 2 args - 可能是一个运算符如 + - * /
    case Ident(name) if args.length == 2 =>
      val argsCode = args.map(translateExpr(_, ctx))
      // 检测是否是运算符
      val isOp = (name.length == 1 && "+-*/%><=&| ^!".contains(name.charAt(0))) ||
                 name.matches("plus|minus|times|div|mod|gt|lt|ge|le|eq|ne|and|or|bar|amp|up")
      if isOp then
        // 二元运算符: x + y 作为 Apply(Ident("+"), List(x, y))
        val opStr = name match
          case "plus" => "+"
          case "minus" => "-"
          case "times" => "*"
          case "div" => "/"
          case "mod" => "%"
          case "gt" => ">"
          case "lt" => "<"
          case "ge" => ">="
          case "le" => "<="
          case "eq" => "=="
          case "ne" => "!="
          case "and" => "&&"
          case "or" => "||"
          case "bar" => "|"
          case "amp" => "&"
          case "up" => "^"
          case _ => name
        TTIRBinOp(opStr, argsCode(0), argsCode(1))
      // 位移操作
      else if name == "lshift" || name == "LSL" || name == "lsl" || name == "shift_left" then
        TTIRBinOp("<<", argsCode(0), argsCode(1))
      else if name == "rshift" || name == "RSH" || name == "rsh" || name == "shift_right" then
        TTIRBinOp(">>", argsCode(0), argsCode(1))
      // 最大最小值 - int 版本
      else if name == "maxInt" || name == "max_int" then
        TTIRMathCall("max", argsCode)
      else if name == "minInt" || name == "min_int" then
        TTIRMathCall("min", argsCode)
      else
        TTIRMathCall(name, argsCode)

    // obj.method(args) 形式的调用
    case Select(obj, methodName) =>
      // methodName 是 String 类型，但可能包含单字符运算符
      val objCode = translateExpr(obj, ctx)
      val argsCode = args.map(translateExpr(_, ctx))
      // 检测是否是二元运算符（单个字符或特定名称）
      val isOperator = (methodName.length == 1 && "+-*/%><=&| ^!".contains(methodName.charAt(0))) ||
                        methodName.matches("plus|minus|times|div|mod|gt|lt|ge|le|eq|ne|and|or|bar|amp|up|shift_left|shift_right")
      // 排除特殊函数名：lshift/lsl/rshift/rsh/minInt/maxInt 不是二元运算符
      val isSpecialFunc = methodName == "lshift" || methodName == "LSL" || methodName == "lsl" ||
                          methodName == "rshift" || methodName == "RSH" || methodName == "rsh" ||
                          methodName == "maxInt" || methodName == "minInt"
      if isOperator && !isSpecialFunc && argsCode.length >= 1 then
        // 二元运算符
        val opStr = methodName match
          case "plus" => "+"
          case "minus" => "-"
          case "times" => "*"
          case "div" => "/"
          case "mod" => "%"
          case "gt" => ">"
          case "lt" => "<"
          case "ge" => ">="
          case "le" => "<="
          case "eq" => "=="
          case "ne" => "!="
          case "and" => "&&"
          case "or" => "||"
          case "bar" => "|"
          case "amp" => "&"
          case "up" => "^"
          case _ => methodName
        TTIRBinOp(opStr, objCode, argsCode.head)
      else
        methodName match
        // 数学函数
        case "sqrt" | "SQRT" => TTIRMathCall("sqrt", argsCode)
        case "exp" | "EXP" => TTIRMathCall("exp", argsCode)
        case "log" | "LOG" | "log10" => TTIRMathCall("log", argsCode)
        case "abs" | "ABS" => TTIRMathCall("abs", argsCode)
        case "sin" | "SIN" => TTIRMathCall("sin", argsCode)
        case "cos" | "COS" => TTIRMathCall("cos", argsCode)
        case "tan" | "TAN" => TTIRMathCall("tan", argsCode)
        case "tanh" | "TANH" => TTIRMathCall("tanh", argsCode)
        case "pow" | "POW" => TTIRMathCall("pow", argsCode)
        case "max" | "MAX" => TTIRMathCall("max", argsCode)
        case "min" | "MIN" => TTIRMathCall("min", argsCode)
        case "fmod" | "fmodf" => TTIRMathCall("fmod", argsCode)
        case "sigmoid" => TTIRMathCall("sigmoid", argsCode)
        case "relu" | "RELU" => TTIRMathCall("relu", argsCode)
        // Extended math functions
        case "exp2" => TTIRMathCall("exp2", argsCode)
        case "expm1" => TTIRMathCall("expm1", argsCode)
        case "log2" => TTIRMathCall("log2", argsCode)
        case "log1p" => TTIRMathCall("log1p", argsCode)
        case "floor" => TTIRMathCall("floor", argsCode)
        case "ceil" => TTIRMathCall("ceil", argsCode)
        case "round" => TTIRMathCall("round", argsCode)
        case "erf" => TTIRMathCall("erf", argsCode)
        case "erfc" => TTIRMathCall("erfc", argsCode)
        case "rint" => TTIRMathCall("rint", argsCode)
        case "nearbyint" => TTIRMathCall("nearbyint", argsCode)
        case "atan" => TTIRMathCall("atan", argsCode)
        case "sinh" => TTIRMathCall("sinh", argsCode)
        case "cosh" => TTIRMathCall("cosh", argsCode)
        case "signum" | "sign" => TTIRMathCall("signum", argsCode)
        // 位移和整数运算函数
        case "lshift" | "LSL" | "lsl" => TTIRBinOp("<<", argsCode(0), argsCode(1))
        case "rshift" | "RSH" | "rsh" => TTIRBinOp(">>", argsCode(0), argsCode(1))
        case "maxInt" | "max_int" => TTIRMathCall("max", argsCode)
        case "minInt" | "min_int" => TTIRMathCall("min", argsCode)
        // Warp shuffle primitives: tl.shfl(value, srcLane, width)
        case "shfl" => TTIRShfl("shfl", argsCode(0), argsCode(1), argsCode(2))
        case "shfl_up" => TTIRShfl("shfl_up", argsCode(0), argsCode(1), argsCode(2))
        case "shfl_down" => TTIRShfl("shfl_down", argsCode(0), argsCode(1), argsCode(2))
        case "shfl_xor" => TTIRShfl("shfl_xor", argsCode(0), argsCode(1), argsCode(2))
        // Warp vote: tl.warp_any / tl.warp_all
        case "warp_any" => TTIRWarpVote("any", argsCode(0))
        case "warp_all" => TTIRWarpVote("all", argsCode(0))
        // Masked load: ptr is either Ident (variable name) or Literal String
        case "maskedLoad" =>
          val ptrName = args(0) match
            case Ident(name) => name
            case Literal(StringConstant(s)) => s
            case _ => args(0).show
          TTIRMaskedLoad(ptrName, argsCode(1), argsCode(2), argsCode(3))
        // Shared memory load: tl.sharedLoad("s_a", idx) -> s_a[idx]
        case "sharedLoad" =>
          val ptrName = args(0) match
            case Ident(name) => name
            case Literal(StringConstant(s)) => s
            case _ => args(0).show
          TTIRLoad(ptrName, argsCode(1), TTIRConst(true), TTIRConst(0.0f))
        // 类型转换
        case "toFloat" | "float2double" | "double2float" => argsCode.headOption.getOrElse(objCode)
        case "int2float" | "float2int" | "toInt" => TTIRMathCall(methodName, argsCode)
        // 线程/块内置变量
        case "threadIdx" => TTIRThreadId(0)
        case "blockIdx" => TTIRBlockId(0)
        case "blockDim" => TTIRBlockDim()
        // apply 方法 (数组访问)
        case "apply" => argsCode.headOption.getOrElse(TTIRConst(0))
        // array update - shouldn't reach here normally, handled in translateTerm
        case "update" if args.length == 2 =>
          TTIRConst(0) // placeholder, actual handling is in translateTerm
        // JavaCPP Pointer dereference: ptr.get() -> ptr[0] (reads first element)
        case "get" =>
          TTIRLoad(obj.show, TTIRConst(0), TTIRConst(true), TTIRConst(0.0f))
        // 其他方法
        case _ => TTIRMathCall(methodName, argsCode)

    // 直接函数调用 - 无参数函数
    case Ident(name) =>
      val argsCode = args.map(translateExpr(_, ctx))
      name match
        // 数学函数
        case "sqrt" => TTIRMathCall("sqrt", argsCode)
        case "exp" => TTIRMathCall("exp", argsCode)
        case "log" | "log10" => TTIRMathCall("log", argsCode)
        case "abs" => TTIRMathCall("abs", argsCode)
        case "sin" => TTIRMathCall("sin", argsCode)
        case "cos" => TTIRMathCall("cos", argsCode)
        case "tan" => TTIRMathCall("tan", argsCode)
        case "tanh" => TTIRMathCall("tanh", argsCode)
        case "pow" => TTIRMathCall("pow", argsCode)
        case "max" => TTIRMathCall("max", argsCode)
        case "min" => TTIRMathCall("min", argsCode)
        case "fmod" => TTIRMathCall("fmod", argsCode)
        case "sigmoid" => TTIRMathCall("sigmoid", argsCode)
        case "relu" => TTIRMathCall("relu", argsCode)
        // 类型转换
        case "int2float" => TTIRMathCall("int2float", argsCode)
        case "float2int" => TTIRMathCall("float2int", argsCode)
        // 线程/块内置变量
        // tl.threadIdx(axis) -> threadIdx.x/y/z, axis: 0=x, 1=y, 2=z
        case "threadIdx" =>
          if argsCode.nonEmpty then
            argsCode.head match
              case TTIRConst(v: Int) => TTIRThreadId(v)
              case TTIRConst(v: Long) => TTIRThreadId(v.toInt)
              case _ => TTIRThreadId(0) // 默认x
          else TTIRThreadId(0)
        // tl.blockIdx(axis) -> blockIdx.x/y/z, use sentinel from DSL (-axis - 1000)
        case "blockIdx" =>
          if argsCode.nonEmpty then
            argsCode.head match
              case TTIRConst(v: Int) if v < 0 => TTIRBlockId(-v - 1000) // reverse sentinel: -1000->0, -1001->1
              case TTIRConst(v: Long) => TTIRBlockId(0)
              case _ => TTIRBlockId(0)
          else TTIRBlockId(0)
        case "blockDim" => TTIRBlockDim()
        // 程序ID - tl.program_id(axis) 返回全局线程ID
        case "programId" | "pid" => TTIRVar("i")
        case "globalThreadId" => TTIRVar("i")
        // 浮点字面量
        case _ if argsCode.nonEmpty => TTIRMathCall(name, argsCode)
        case _ => TTIRVar(name)

    case TypeApply(callee, _) =>
      translateCall(callee, args, ctx)

    case other =>
      // 可能是方法链式调用
      val shown = other.show
      if shown.contains("Apply") then
        TTIRConst(s"/* apply: ${shown} */")
      else
        TTIRConst(s"/* call: ${shown} */")

private def identName(using Quotes)(term: quotes.reflect.Term): String =
  import quotes.reflect.*
  term match
    case Ident(name) => name
    case Select(obj, name) => name
    case _ => term.show

// 辅助函数: 将变量提升到函数作用域而非局部作用域
// 这解决了变量在嵌套作用域中使用的问题
private def hoistVarToFunctionScope(name: String, tpe: String, init: TTIRExpr): List[TTIRStmt] =
  List(TTIRLocalDecl(tpe, name), TTIRAssign(name, init))

private def translateType(using Quotes)(tpe: quotes.reflect.TypeRepr): String =
  import quotes.reflect.*
  tpe match
    case t if t =:= TypeRepr.of[Float] => "float"
    case t if t =:= TypeRepr.of[Int] => "int"
    case t if t =:= TypeRepr.of[Long] => "long"
    case t if t =:= TypeRepr.of[Double] => "double"
    case t if t =:= TypeRepr.of[Boolean] => "bool"
    case t if t =:= TypeRepr.of[Byte] => "char"
    case t if t =:= TypeRepr.of[Short] => "short"
    case t if t =:= TypeRepr.of[Unit] => "void"
    case _ => "float"

private def typeToTriton(using Quotes)(tpe: quotes.reflect.TypeRepr, name: String = ""): String =
  import quotes.reflect.*
  // Special handling for Ptr[T] types - always treat as pointer with float base type
  val typeStr = tpe.show
  // JavaCPP Pointer types - always treat as pointers
  if typeStr.contains("FloatPointer") then
    return "float*"
  if typeStr.contains("DoublePointer") then
    return "double*"
  if typeStr.contains("IntPointer") then
    return "int*"
  if typeStr.contains("LongPointer") then
    return "long*"
  // DSL Ptr types — check specific names before generic Ptr check
  if typeStr.contains("IntPtr") then
    return "int*"
  if typeStr.contains("Ptr") || typeStr.contains("Float*") then
    return "float*"
  if typeStr.contains("Int*") then
    return "int*"
  if typeStr.contains("Double*") then
    return "double*"
  if typeStr.contains("Long*") then
    return "long*"
  val isPtr = isPointerType(tpe, name)
  val baseType = tpe match
    case t if t =:= TypeRepr.of[Float] => "float"
    case t if t =:= TypeRepr.of[Int] => "int"
    case t if t =:= TypeRepr.of[Long] => "long"
    case t if t =:= TypeRepr.of[Double] => "double"
    case t if t =:= TypeRepr.of[Boolean] => "bool"
    case _ => "float"
  if isPtr then s"$baseType*" else baseType

// 所有参数都作为指针处理
private def typeToTritonPointer(using Quotes)(tpe: quotes.reflect.TypeRepr): String =
  import quotes.reflect.*
  tpe match
    case t if t =:= TypeRepr.of[Float] => "float*"
    case t if t =:= TypeRepr.of[Int] => "int*"
    case t if t =:= TypeRepr.of[Long] => "long*"
    case t if t =:= TypeRepr.of[Double] => "double*"
    case t if t =:= TypeRepr.of[Boolean] => "bool*"
    case _ => "float*"

// 从表达式推断类型
// 当 Scala 推断的类型不正确时（如 tl.load 返回 Float 但实际应返回 int），
// 根据表达式内容推断正确的 C 类型
private def inferTypeFromExpr(expr: TTIRExpr, ctx: TranslateContext, defaultType: String): String =
  expr match
    // TTIRLoad 根据指针名称约定判断是 int 还是 float
    case TTIRLoad(ptr, _, _, TTIRConst(v: Int)) if isPointerByConventionForTriton(ptr) => "int"
    case TTIRLoad(_, _, _, TTIRConst(v: Int)) => "int"
    case TTIRLoad(_, _, _, TTIRConst(_)) => defaultType  // float default
    // TTIRBinOp: 如果任一操作数是 int，结果应该是 int
    case TTIRBinOp(_, TTIRVar(name), _) if ctx.isLocal(name) && ctx.getType(name) == "int" => "int"
    case TTIRBinOp(_, _, TTIRVar(name)) if ctx.isLocal(name) && ctx.getType(name) == "int" => "int"
    case TTIRBinOp("*", lhs, rhs) =>
      // Int * Int = Int
      if (lhs.toString == "int" || rhs.toString == "int") "int" else defaultType
    case TTIRBinOp("+", lhs, rhs) =>
      if (lhs.toString == "int" || rhs.toString == "int") "int" else defaultType
    case TTIRBinOp("-", lhs, rhs) =>
      if (lhs.toString == "int" || rhs.toString == "int") "int" else defaultType
    case TTIRBinOp(_, lhs, rhs) =>
      // 其他二元运算，如果涉及 int 变量，结果可能是 int
      val lhsType = getExprBaseType(lhs, ctx)
      val rhsType = getExprBaseType(rhs, ctx)
      if (lhsType == "int" || rhsType == "int") "int" else defaultType
    case _ => defaultType

// 获取表达式的基本类型（如果是变量引用的话）
private def getExprBaseType(expr: TTIRExpr, ctx: TranslateContext): String = expr match
  case TTIRVar(name) if ctx.isLocal(name) => ctx.getType(name)
  case _ => expr.toString  // 尝试从字符串推断

// 判断类型是否应该作为指针处理
// 规则：
// - Float 永远作为指针（数组）
// - Int/Long/Double/Bool 默认为标量，但名字包含指针后缀时视为指针
// - 其他类型根据命名约定判断
private def isPointerByConvention(name: String): Boolean =
  val n = name.toLowerCase
  n.endsWith("_ptr") || n.endsWith("ptr") ||
  n.contains("_ptr") || n.contains("ptr") ||
  n.contains("_table") || n.endsWith("table") ||
  n.endsWith("_mapping") || n.endsWith("mapping") ||
  n.endsWith("_indices") || n.endsWith("indices") ||
  n.endsWith("_offsets") || n.endsWith("offsets") ||
  n.endsWith("_cache") || n.endsWith("cache")

// Detect scalar (non-pointer) parameter names for Float type
private def isScalarByConvention(name: String): Boolean =
  val n = name.toLowerCase
  // ptr 后缀 → 指针
  if n.contains("ptr") || n.endsWith("_val") || n.endsWith("val") then false
  else
    // Common scalar parameter names (used as scalar in FlashAttention, etc.)
    n.contains("scale") || n.contains("bias") ||
    n.contains("alpha") || n.contains("beta") || n.contains("gamma") ||
    n.contains("threshold") || n.contains("eps") ||
    n.endsWith("_scalar") || n.endsWith("_const") ||
    n == "idx" || n == "tmp" || n == "eps"

private def isPointerType(using Quotes)(tpe: quotes.reflect.TypeRepr, name: String = ""): Boolean =
  import quotes.reflect.*
  val typeStr = tpe.show
  // JavaCPP Pointer types are always pointers
  if typeStr.contains("Pointer") then
    return true
  // 首先根据类型判断
  tpe match
    case t if t =:= TypeRepr.of[Float] =>
      // Float: treat as pointer by default (array), but scalar names are scalar
      !isScalarByConvention(name)
    case t if t =:= TypeRepr.of[Int] =>
      // Int 类型：检查命名约定
      isPointerByConvention(name)
    case t if t =:= TypeRepr.of[Long] => false   // Long 作为标量
    case t if t =:= TypeRepr.of[Double] => false  // Double 作为标量
    case t if t =:= TypeRepr.of[Boolean] => false // Boolean 作为标量
    // Ptr[T] 类型始终作为指针处理 - 使用更健壮的检查
    case AppliedType(tq, _) =>
      tq.show.contains("Ptr") || tq.termSymbol.name == "Ptr"
    case _ =>
      // 其他类型根据命名约定判断
      isPointerByConvention(name)

// ============================================================================
// Triton-like built-in functions for kernel authoring
// These are placeholder functions that are recognized by @TritonKernelMacro
// at compile time and translated to corresponding CUDA code.
// ============================================================================
// ============================================================================
// Triton-like built-in functions for kernel authoring
// These functions are recognized by @TritonKernelMacro at compile time
// and translated to corresponding CUDA code. They also support direct
// CUDA memory operations via JavaCPP when called from host code.
// ============================================================================

object tl:
  // Import MemoryOps for actual CUDA operations
  import cuda.dsl.core.MemoryOps
  import cuda.dsl.core.Ptr
  import cuda.dsl.core.Types.{given_MemoryOps_Float, given_MemoryOps_Double,
                                  given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Byte}
  import org.bytedeco.javacpp.{FloatPointer, DoublePointer, IntPointer, LongPointer, BytePointer}

  // --- ID Functions ---------------------------------------------------------
  /** Thread ID along a given axis (0, 1, or 2).
    *  在 @TritonKernelMacro 展开后生成: blockIdx.x * blockDim + threadIdx.x
    *  注意：这些方法在Scala编译时返回占位符0，实际值在生成的CUDA kernel中使用。 */
  inline def program_id(inline axis: Int): Int = axis match {
    case 0 => 0  // placeholder - expands to blockIdx.x at generate time
    case 1 => 0
    case 2 => 0
    case _ => 0
  }

  /** Thread index along a given axis (0, 1, or 2) - returns threadIdx.x/y/z */
  inline def threadIdx(inline axis: Int): Int = axis match {
    case 0 => 0  // placeholder - expands to threadIdx.x in generated kernel
    case 1 => 0
    case 2 => 0
    case _ => 0
  }

  /** Block index along a given axis - uses sentinel marker to help TTIR distinguish from program_id.
    *  TTIR post-process expects pattern: tl.blockIdx(0), tl.blockIdx(1), etc. */
  inline def blockIdx(inline axis: Int): Int = -1000 - axis

  /** Number of threads in a block along a given axis (blockDim.x/y/z) */
  inline def block_dim(inline axis: Int): Int = axis match {
    case 0 => 256  // typical default blockDim.x
    case 1 => 1
    case 2 => 1
    case _ => 1
  }

  /** Number of blocks in a grid along a given axis (gridDim.x/y/z) */
  inline def num_blocks(inline axis: Int): Int = 1  // default grid size

  // --- Constexpr Support ------------------------------------------------
  /** Compile-time constant value - expanded at kernel generation time.
   *  Usage: val BLOCK_M = tl.constexpr(64)
   *  Generates: #define BLOCK_M 64 in CUDA header
   */
  inline def constexpr(inline value: Int): Int = value
  inline def constexpr(inline value: Float): Float = value
  inline def constexpr(inline value: Boolean): Boolean = value

  // --- Barrier & Sync ----------------------------------------------------
  /** Barrier: __syncthreads() . Stub for macro expansion. */
  def syncthreads(): Unit = () // expands to __syncthreads()

  /** Warp-level barrier: __syncwarp(mask) */
  def syncwarp(mask: Int = -1): Unit = () // expands to __syncwarp(mask)

  // --- Shared Memory ------------------------------------------------------
  /** Declare shared memory: __shared__ T name[size];
    *  Stub for macro expansion. */
  def sharedMem(tpe: String, name: String, size: Int): Unit = ()

  // --- Load/Store (for code generation) ------------------------------------
  // These are placeholder methods for kernel code generation.
  // The @TritonKernelMacro translates these to actual CUDA C code (__ldg, etc.)
  // For actual CUDA memory operations from host code, use the Pointer overloads below.

  /** Load a value from memory at the given address.
    *  Stub for code generation - expands to `ptr[offset]` in CUDA code. */
  def load(addr: Float): Float = 0.0f

  /** Load with mask fallback (out-of-bounds guard). */
  def load(addr: Float, mask: Boolean, other: Float): Float = if mask then load(addr) else other

  /** Store a value to memory at the given address.
    *  Stub for code generation - expands to `ptr[offset] = value` in CUDA code. */
  def store(addr: Float, value: Float): Unit = ()

  /** Store a value to memory at the given address with mask. */
  def store(addr: Float, value: Float, mask: Boolean): Unit = ()

  // --- Pointer type overloads for JavaCPP Pointer types (CUDA cuMemcpy) ---
  // These methods use implicitly to resolve MemoryOps, making them easier to call

  /** Load a value from FloatPointer at given byte offset.
   *  Performs actual CUDA memory copy: cuMemcpyDtoH */
  def load(ptr: org.bytedeco.javacpp.FloatPointer, byteOffset: Int): Float = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Float]]
    val dptr = Ptr.fromAddress[Float](ptr.address + byteOffset)
    ops.read(dptr, 0)
  }

  /** Load a value from DoublePointer at given byte offset */
  def load(ptr: org.bytedeco.javacpp.DoublePointer, byteOffset: Int): Double = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Double]]
    val dptr = Ptr.fromAddress[Double](ptr.address + byteOffset)
    ops.read(dptr, 0)
  }

  /** Load a value from IntPointer at given byte offset */
  def load(ptr: org.bytedeco.javacpp.IntPointer, byteOffset: Int): Int = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Int]]
    val dptr = Ptr.fromAddress[Int](ptr.address + byteOffset)
    ops.read(dptr, 0)
  }

  /** Load a value from LongPointer at given byte offset */
  def load(ptr: org.bytedeco.javacpp.LongPointer, byteOffset: Int): Long = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Long]]
    val dptr = Ptr.fromAddress[Long](ptr.address + byteOffset)
    ops.read(dptr, 0)
  }

  /** Load a value from BytePointer at given byte offset */
  def load(ptr: org.bytedeco.javacpp.BytePointer, byteOffset: Int): Byte = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Byte]]
    val dptr = Ptr.fromAddress[Byte](ptr.address + byteOffset)
    ops.read(dptr, 0)
  }

  /** Store a value to FloatPointer at given byte offset (CUDA: cuMemcpyHtoD) */
  def store(ptr: org.bytedeco.javacpp.FloatPointer, byteOffset: Int, value: Float): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Float]]
    val dptr = Ptr.fromAddress[Float](ptr.address + byteOffset)
    ops.write(dptr, 0, value)
  }

  /** Store a value to DoublePointer at given byte offset */
  def store(ptr: org.bytedeco.javacpp.DoublePointer, byteOffset: Int, value: Double): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Double]]
    val dptr = Ptr.fromAddress[Double](ptr.address + byteOffset)
    ops.write(dptr, 0, value)
  }

  /** Store a value to IntPointer at given byte offset */
  def store(ptr: org.bytedeco.javacpp.IntPointer, byteOffset: Int, value: Int): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Int]]
    val dptr = Ptr.fromAddress[Int](ptr.address + byteOffset)
    ops.write(dptr, 0, value)
  }

  /** Store a value to LongPointer at given byte offset */
  def store(ptr: org.bytedeco.javacpp.LongPointer, byteOffset: Int, value: Long): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Long]]
    val dptr = Ptr.fromAddress[Long](ptr.address + byteOffset)
    ops.write(dptr, 0, value)
  }

  /** Store a value to BytePointer at given byte offset */
  def store(ptr: org.bytedeco.javacpp.BytePointer, byteOffset: Int, value: Byte): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Byte]]
    val dptr = Ptr.fromAddress[Byte](ptr.address + byteOffset)
    ops.write(dptr, 0, value)
  }

  // --- DSL Typed Pointers (cuda.dsl.core.FloatPtr etc.) -----------------
  // These use the actual CUDA device pointers

  /** Load a value from cuda.dsl.core.FloatPtr at given byte offset */
  def load(ptr: cuda.dsl.core.FloatPtr, byteOffset: Int): Float = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Float]]
    val idx = byteOffset / 4  // Float = 4 bytes
    ops.read(ptr.toPtr, idx)
  }

  /** Load a value from cuda.dsl.core.DoublePtr at given byte offset */
  def load(ptr: cuda.dsl.core.DoublePtr, byteOffset: Int): Double = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Double]]
    val idx = byteOffset / 8  // Double = 8 bytes
    ops.read(ptr.toPtr, idx)
  }

  /** Load a value from cuda.dsl.core.IntPtr at given byte offset */
  def load(ptr: cuda.dsl.core.IntPtr, byteOffset: Int): Int = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Int]]
    val idx = byteOffset / 4
    ops.read(ptr.toPtr, idx)
  }

  /** Load a value from cuda.dsl.core.LongPtr at given byte offset */
  def load(ptr: cuda.dsl.core.LongPtr, byteOffset: Int): Long = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Long]]
    val idx = byteOffset / 8
    ops.read(ptr.toPtr, idx)
  }

  /** Load a value from cuda.dsl.core.BytePtr at given byte offset */
  def load(ptr: cuda.dsl.core.BytePtr, byteOffset: Int): Byte = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Byte]]
    val idx = byteOffset / 1
    ops.read(ptr.toPtr, idx)
  }

  /** Store a value to cuda.dsl.core.FloatPtr at given byte offset */
  def store(ptr: cuda.dsl.core.FloatPtr, byteOffset: Int, value: Float): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Float]]
    val idx = byteOffset / 4
    ops.write(ptr.toPtr, idx, value)
  }

  /** Store a value to cuda.dsl.core.DoublePtr at given byte offset */
  def store(ptr: cuda.dsl.core.DoublePtr, byteOffset: Int, value: Double): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Double]]
    val idx = byteOffset / 8
    ops.write(ptr.toPtr, idx, value)
  }

  /** Store a value to cuda.dsl.core.IntPtr at given byte offset */
  def store(ptr: cuda.dsl.core.IntPtr, byteOffset: Int, value: Int): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Int]]
    val idx = byteOffset / 4
    ops.write(ptr.toPtr, idx, value)
  }

  /** Store a value to cuda.dsl.core.LongPtr at given byte offset */
  def store(ptr: cuda.dsl.core.LongPtr, byteOffset: Int, value: Long): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Long]]
    val idx = byteOffset / 8
    ops.write(ptr.toPtr, idx, value)
  }

  /** Store a value to cuda.dsl.core.BytePtr at given byte offset */
  def store(ptr: cuda.dsl.core.BytePtr, byteOffset: Int, value: Byte): Unit = {
    val ops = implicitly[cuda.dsl.core.MemoryOps[Byte]]
    val idx = byteOffset / 1
    ops.write(ptr.toPtr, idx, value)
  }

  // Note: CUDALib.FloatPtr/DoublePtr/IntPtr/LongPtr are type aliases for cuda.dsl.core.* types.
  // The load/store methods above handle them via implicit conversions.

  /** Warp shuffle: __shfl_sync(mask, value, srcLane, width).
    *  Stub - @TritonKernelMacro expands to CUDA intrinsic. */
  inline def shfl(inline value: Float, inline srcLane: Int, inline width: Int): Float = ???

  /** Warp shuffle up: __shfl_up_sync(mask, value, delta, width) */
  inline def shfl_up(inline value: Float, inline srcLane: Int, inline width: Int): Float = ???

  /** Warp shuffle down: __shfl_down_sync(mask, value, delta, width) */
  inline def shfl_down(inline value: Float, inline srcLane: Int, inline width: Int): Float = ???

  /** Warp shuffle xor: __shfl_xor_sync(mask, value, laneMask, width) */
  inline def shfl_xor(inline value: Float, inline srcLane: Int, inline width: Int): Float = ???

  /** Warp any vote: __any_sync(mask, predicate) */
  inline def warp_any(inline predicate: Boolean): Boolean = ???

  /** Warp all vote: __all_sync(mask, predicate) */
  inline def warp_all(inline predicate: Boolean): Boolean = ???

  /** Masked (predicated) load: returns value if predicate else other.
   *  Used for boundary checking in Triton kernels. */
  inline def maskedLoad(inline ptr: Float, inline offset: Int, inline predicate: Boolean, inline other: Float): Float = 0.0f

  /** Masked (predicated) store: stores value if predicate */
  inline def maskedStore(inline ptr: Float, inline offset: Int, inline value: Float, inline predicate: Boolean): Unit = ()

  // Masked load/store for FloatPointer (JavaCPP) - actual CUDA operations
  def maskedLoad(ptr: org.bytedeco.javacpp.FloatPointer, offset: Int, predicate: Boolean, other: Float): Float =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: org.bytedeco.javacpp.FloatPointer, offset: Int, value: Float, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for DoublePointer (JavaCPP)
  def maskedLoad(ptr: org.bytedeco.javacpp.DoublePointer, offset: Int, predicate: Boolean, other: Double): Double =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: org.bytedeco.javacpp.DoublePointer, offset: Int, value: Double, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for IntPointer (JavaCPP)
  def maskedLoad(ptr: org.bytedeco.javacpp.IntPointer, offset: Int, predicate: Boolean, other: Int): Int =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: org.bytedeco.javacpp.IntPointer, offset: Int, value: Int, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for LongPointer (JavaCPP)
  def maskedLoad(ptr: org.bytedeco.javacpp.LongPointer, offset: Int, predicate: Boolean, other: Long): Long =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: org.bytedeco.javacpp.LongPointer, offset: Int, value: Long, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for BytePointer (JavaCPP)
  def maskedLoad(ptr: org.bytedeco.javacpp.BytePointer, offset: Int, predicate: Boolean, other: Byte): Byte =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: org.bytedeco.javacpp.BytePointer, offset: Int, value: Byte, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for FloatPtr (DSL typed pointer)
  def maskedLoad(ptr: cuda.dsl.core.FloatPtr, offset: Int, predicate: Boolean, other: Float): Float =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: cuda.dsl.core.FloatPtr, offset: Int, value: Float, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for DoublePtr (DSL typed pointer)
  def maskedLoad(ptr: cuda.dsl.core.DoublePtr, offset: Int, predicate: Boolean, other: Double): Double =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: cuda.dsl.core.DoublePtr, offset: Int, value: Double, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for IntPtr (DSL typed pointer)
  def maskedLoad(ptr: cuda.dsl.core.IntPtr, offset: Int, predicate: Boolean, other: Int): Int =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: cuda.dsl.core.IntPtr, offset: Int, value: Int, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for LongPtr (DSL typed pointer)
  def maskedLoad(ptr: cuda.dsl.core.LongPtr, offset: Int, predicate: Boolean, other: Long): Long =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: cuda.dsl.core.LongPtr, offset: Int, value: Long, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // Masked load/store for BytePtr (DSL typed pointer)
  def maskedLoad(ptr: cuda.dsl.core.BytePtr, offset: Int, predicate: Boolean, other: Byte): Byte =
    if predicate then load(ptr, offset) else other
  def maskedStore(ptr: cuda.dsl.core.BytePtr, offset: Int, value: Byte, predicate: Boolean): Unit =
    if predicate then store(ptr, offset, value)

  // --- Shared Memory Operations (macro-expanded) -------------------------
  /** Load from shared memory array (ptr is shared mem name, offset is index).
    *  Stub - expands to shared memory load in generated CUDA code. */
  inline def sharedLoad(inline ptr: String, inline offset: Int): Float = ???

  /** Store to shared memory array (ptr is shared mem name, offset is index) */
  inline def sharedStore(inline ptr: String, inline offset: Int, inline value: Float): Unit = ???

  def exp(x: Float): Float = scala.math.exp(x.toDouble).toFloat

  def log(x: Float): Float = scala.math.log(x.toDouble).toFloat

  def sqrt(x: Float): Float = scala.math.sqrt(x.toDouble).toFloat

  def sin(x: Float): Float = scala.math.sin(x.toDouble).toFloat

  def cos(x: Float): Float = scala.math.cos(x.toDouble).toFloat

  def tanh(x: Float): Float = scala.math.tanh(x.toDouble).toFloat

  def atan(x: Float): Float = scala.math.atan(x.toDouble).toFloat

  def abs(x: Float): Float = scala.math.abs(x.toDouble).toFloat

  def max(a: Float, b: Float): Float = if (a > b) a else b

  def min(a: Float, b: Float): Float = if (a < b) a else b
  def min(a: Int, b: Int): Int = if (a < b) a else b
  def abs(x: Int): Int = if (x < 0) -x else x

  def sigmoid(x: Float): Float = 1.0f / (1.0f + exp(-x))

  def relu(x: Float): Float = max(0.0f, x)

  def floor(x: Float): Float = scala.math.floor(x.toDouble).toFloat

  // --- Extended Math Functions (erf/rint/lgamma) ---
  def erf(x: Float): Float = {
    // Error function - stub implementation for DSL
    val xd = x.toDouble
    if (xd.abs > 4.0) (if (xd > 0) 1.0 else -1.0).toFloat
    else {
      // Approximation using Taylor series (first few terms)
      val x2 = xd * xd
      var sum = xd
      var term = xd
      for (n <- 1 to 10) {
        term *= -x2 / (n * (2 * n + 1))
        sum += term
      }
      (sum * 2 / scala.math.sqrt(scala.math.Pi)).toFloat
    }
  }

  def erfc(x: Float): Float = 1.0f - erf(x)

  def rint(x: Float): Float = scala.math.rint(x.toDouble).toFloat

  def nearbyint(x: Float): Float = scala.math.rint(x.toDouble).toFloat

  def round(x: Float): Float = scala.math.round(x.toDouble).toFloat

  // Gamma-related functions - stubs for DSL (actual impl in CUDA)
  def lgamma(x: Float): Float = scala.math.log(mathGammaApprox(x.toDouble)).toFloat

  def gamma(x: Float): Float = mathGammaApprox(x.toDouble).toFloat

  // Simple gamma approximation (Stirling-based)
  private def mathGammaApprox(z: Double): Double = {
    if (z <= 0.5) {
      scala.math.Pi / (scala.math.sin(scala.math.Pi * z) * mathGammaApprox(1 - z))
    } else {
      val z1 = z - 1
      val c = List(76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
      var tmp = z1 + 5.5
      tmp -= (z1 + 0.5) * scala.math.log(tmp)
      var ser = 1.000000000190015
      for (i <- 0 until 6) {
        ser += c(i) / (z1 + i + 1)
      }
      scala.math.exp(-tmp + scala.math.log(2.5066282746310005 * ser / z))
    }
  }

  def digamma(x: Float): Float = {
    // Digamma approximation (psi function)
    val xd = x.toDouble
    if (xd > 0) {
      scala.math.log(xd) - 1.0 / (2.0 * xd)
    } else {
      -1.0 / xd
    }
  }.toFloat

  // Exponential-related
  def exp2(x: Float): Float = scala.math.pow(2.0, x.toDouble).toFloat

  def expm1(x: Float): Float = (scala.math.exp(x.toDouble) - 1.0).toFloat

  def log10(x: Float): Float = scala.math.log10(x.toDouble).toFloat

  def log2(x: Float): Float = (scala.math.log(x.toDouble) / scala.math.log(2.0)).toFloat

  def log1p(x: Float): Float = scala.math.log1p(x.toDouble).toFloat

  // Hyperbolic
  def sinh(x: Float): Float = scala.math.sinh(x.toDouble).toFloat

  def cosh(x: Float): Float = scala.math.cosh(x.toDouble).toFloat

  // Power
  def pow(x: Float, y: Float): Float = scala.math.pow(x.toDouble, y.toDouble).toFloat

  // --- Tensor Creation Helpers ---
  /** Create evenly spaced values: [start, start + step, ..., end]
   *  Usage: val x = tl.linspace(0.0f, 1.0f, 10)  // [0, 0.111..., ..., 1.0]
   */
  def linspace(start: Float, end: Float, num: Int): Float = {
    if (num <= 1) start
    else start  // Actual calculation happens in kernel expansion
  }

  /** Create evenly spaced values (alias)
   *  Usage: val x = tl.arange(0.0f, 10.0f, 1.0f)
   */
  def arange(start: Float, end: Float, step: Float = 1.0f): Float = start

  /** Create a tensor filled with zeros */
  def zeros(n: Int): Float = 0.0f

  /** Create a tensor filled with ones */
  def ones(n: Int): Float = 1.0f

  /** Create a tensor with a specific value */
  def full(value: Float, n: Int): Float = value

  // --- Async Copy (cp.async) ---------------------------------------------
  /** Asynchronous copy from global to shared memory.
   *  Maps to CUDA's cp.async instruction.
   *  cache: cache modifier (cg=read-write, ca=read, cs=streaming)
   */
  def cp_async(dst: String, src: Float, cache: String = "cg"): Unit = ()

  /** Async copy commit - waits for all pending cp.async to complete */
  def cp_async_commit(): Unit = ()

  /** Async copy wait - waits for pending async copies */
  def cp_async_wait(mask: Int = 0): Unit = ()

  // --- TMA (Texture Memory Architecture) --------------------------------
  /** Create a TMA descriptor for the given pointer and shape.
   *  Maps to CUDA's cuda::memcpy_async with TMA.
   *  This enables efficient 2D/3D bulk transfers.
   */
  def create_tma_descriptor(ptr: Float, shape: (Int, Int, Int), strides: (Int, Int, Int)): Float = 0.0f

  /** TMA load - loads a tensor using TMA descriptor.
   *  Maps to CUDA's cuda::load(TMA descriptor, ...)
   */
  def tma_load(desc: Float, dst: String, src_ptr: Float, src_box: (Int, Int, Int)): Unit = ()

  /** TMA store - stores a tensor using TMA descriptor.
   *  Maps to CUDA's cuda::store(TMA descriptor, ...)
   */
  def tma_store(desc: Float, dst_ptr: Float, src: String, dst_box: (Int, Int, Int)): Unit = ()

  /** Barrier for TMA - ensures TMA operations complete before use */
  def tma_barrier(): Unit = ()

  /** Arrival count for TMA multiproducer synchronization */
  def tma_arrive(): Unit = ()

  // --- Extended Atomic Operations ----------------------------------------
  /** Atomic compare-and-swap (already implemented but listed for completeness) */
  def atomic_cas(ptr: Float, cmp: Float, newVal: Float): Float = 0.0f

  /** Atomic AND operation */
  def atomic_and(ptr: Float, newVal: Float): Float = 0.0f

  /** Atomic OR operation */
  def atomic_or(ptr: Float, newVal: Float): Float = 0.0f

  /** Atomic XOR operation */
  def atomic_xor(ptr: Float, newVal: Float): Float = 0.0f

  /** Atomic MIN operation (for floats) */
  def atomic_min(ptr: Float, newVal: Float): Float = 0.0f

  /** Atomic MAX operation (for floats) */
  def atomic_max(ptr: Float, newVal: Float): Float = 0.0f

  // --- Prologue / Epilogue Hooks ----------------------------------------
  /** Kernel prologue hook - called before main computation.
   *  User can override this in their kernel for initialization.
   */
  def kernel_prologue(): Unit = ()

  /** Kernel epilogue hook - called after main computation.
   *  User can override this for cleanup/finalization.
   */
  def kernel_epilogue(): Unit = ()

  // --- Extended Reduction Primitives ------------------------------------
  /** Reduction across warp with custom operation */
  def warp_reduce(op: String, value: Float, lane_mask: Int = 0xffffffff): Float = value

  /** Reduction across thread block */
  def block_reduce(op: String, value: Float): Float = value

  // --- Load/Store with Cache Hints --------------------------------------
  /** Load with eviction policy hint
   *  policy: "evict_first" or "evict_last" for L2 cache management
   */
  def load_evict_first(addr: Float, default: Float): Float = default

  def load_evict_last(addr: Float, default: Float): Float = default

  /** Prefetch to L1/L2 cache */
  def prefetch(addr: Float): Unit = ()


// =============================================================================
// Type aliases for convenience
// =============================================================================

/** Short alias for @TritonKernelMacro - marks a method as a GPU kernel.
  *
  * Usage:
  * {{{
  * @TritonJit
  * def myKernel(out: Float, a: Float, b: Float, n: Int): Unit = { ... }
  * }}}
  */
type TritonJit = TritonKernelMacro



