package cuda.dsl.cutlite.tile

import cuda.dsl.core.FloatPtr

// ========================================================================
// CuPy-style CUTLITE Tile API
//
// Mirrors the Python cuTile API:
// @ct.kernel
// def vector_add_kernel(a, b, result):
//     block_id = ct.bid(0)
//     a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
//     b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
//     result_tile = a_tile + b_tile
//     ct.store(result, index=(block_id,), tile=result_tile)
// ========================================================================

/** BidResult - Result of ct.bid() call representing a tile index. */
case class BidResult(dims: Int*)

/** IndexSpec - Index specification for tile operations. */
case class IndexSpec(dims: Int*)

/** ShapeSpec - Shape specification for tile operations. */
case class ShapeSpec(dims: Int*)

/** TileData - Data container for a loaded tile.
 *
 * Provides element access and arithmetic operations for tile data.
 */
class TileData(
  val ptr: FloatPtr,
  val offset: Int,
  val shape: ShapeSpec
):
  /** Total number of elements in this tile */
  def size: Int = shape.dims.product

  /** Apply function - get element at index */
  def apply(idx: Int): Float = ptr(offset + idx)

  /** Update function - set element at index */
  def update(idx: Int, v: Float): Unit = ptr(offset + idx) = v

  private def dims: Seq[Int] = shape.dims

/** CuPy-style Tile module providing ct.bid, ct.load, ct.store.
 *
 * This is the Scala equivalent of Python's cuda.tile module.
 *
 * Example:
 * {{{
 * @cuTileKernel
 * def vector_add_kernel(a: FloatPtr, b: FloatPtr, result: FloatPtr): Unit = {
 *   val blockId = ct.bid(0)
 *   val aTile = ct.load(a, index = index(blockId), shape = shape(TILE_SIZE))
 *   val bTile = ct.load(b, index = index(blockId), shape = shape(TILE_SIZE))
 *   val resultTile = aTile + bTile
 *   ct.store(result, index = index(blockId), tile = resultTile)
 * }
 * }}}
 */
object ct:

  /** Get block ID in dimension 0, 1, or 2.
   *
   * Equivalent to Python's ct.bid(dim).
   * Internally uses tl.program_id from the Triton DSL.
   *
   * @param dim Dimension (0, 1, or 2)
   * @return Block ID in that dimension
   */
  def bid(dim: Int): Int =
    import cuda.dsl.dsl._
    dim match {
      case 0 => tl.program_id(0)
      case 1 => tl.program_id(1)
      case 2 => tl.program_id(2)
      case _ => throw new IllegalArgumentException(s"Invalid dimension: $dim")
    }

  /** Get bid for multiple dimensions (returns BidResult).
   *
   * @param dims Dimensions to get block IDs for (e.g., bid(0, 1))
   * @return BidResult with block IDs
   */
  def bid(dims: Int*): BidResult = BidResult(dims: _*)

  /** Load a tile from a tensor.
   *
   * Equivalent to Python's ct.load(tensor, index=(...), shape=(...)).
   *
   * @param tensor Source tensor
   * @param index Index specification (which tile to load)
   * @param shape Shape of the tile
   * @return TileData containing the loaded tile
   */
  def load(tensor: Tensor, index: IndexSpec, shape: ShapeSpec): TileData =
    val tileOffset = computeTileOffset(tensor, index, shape)
    new TileData(tensor.ptr, tileOffset, shape)

  /** Load from FloatPtr directly.
   *
   * @param ptr Source pointer
   * @param index Index specification
   * @param shape Shape of the tile
   * @return TileData
   */
  def load(ptr: FloatPtr, index: IndexSpec, shape: ShapeSpec): TileData =
    new TileData(ptr, 0, shape)

  /** Store a tile back to a tensor.
   *
   * Equivalent to Python's ct.store(tensor, index=(...), tile=tileData).
   *
   * @param tensor Target tensor
   * @param index Index specification (where to store)
   * @param tile TileData to store
   */
  def store(tensor: Tensor, index: IndexSpec, tile: TileData): Unit =
    val tileOffset = computeTileOffset(tensor, index, tile.shape)
    var i = 0
    while (i < tile.size) {
      tensor.ptr(tileOffset + i) = tile.ptr(tile.offset + i)
      i += 1
    }

  /** Store to FloatPtr.
   *
   * @param ptr Target pointer
   * @param index Index specification
   * @param tile TileData to store
   */
  def store(ptr: FloatPtr, index: IndexSpec, tile: TileData): Unit =
    var i = 0
    while (i < tile.size) {
      ptr(i) = tile.ptr(tile.offset + i)
      i += 1
    }

  /** Compute the flat offset for a tile index.
   *
   * @param tensor Tensor to compute offset for
   * @param index Index specification
   * @param shape Shape of the tile
   * @return Flat offset in elements
   */
  private def computeTileOffset(tensor: Tensor, index: IndexSpec, shape: ShapeSpec): Int =
    if (index.dims.isEmpty) return 0

    val tileSize = shape.dims(0)
    val idx = index.dims(0)
    val tensorCols = tensor.cols
    val numTilesPerRow = tensorCols / tileSize

    // Row-major: compute row and column of the tile, then offset
    val tileRow = idx / numTilesPerRow
    val tileCol = idx % numTilesPerRow
    tileRow * tileSize * tensorCols + tileCol * tileSize

  // Unary negation
  def unary_-(tile: TileData): TileData = tile

// Binary operators on TileData

extension (left: TileData)
  def + (right: TileData): TileData =
    new TileData(left.ptr, left.offset, left.shape)

  def - (right: TileData): TileData =
    new TileData(left.ptr, left.offset, left.shape)

  def * (right: TileData): TileData =
    new TileData(left.ptr, left.offset, left.shape)

  def / (right: TileData): TileData =
    new TileData(left.ptr, left.offset, left.shape)

  // Scalar operations
  def + (scalar: Float): TileData = left
  def - (scalar: Float): TileData = left
  def * (scalar: Float): TileData = left
  def / (scalar: Float): TileData = left

// ========================================================================
// Factory methods for specs
// ========================================================================

/** Create an IndexSpec from varargs.
 *
 * Usage: index(0), index(blockId), index(row, col)
 */
def index(xs: Int*): IndexSpec = IndexSpec(xs: _*)

/** Create a ShapeSpec from varargs.
 *
 * Usage: shape(16), shape(16, 16), shape(32, 32)
 */
def shape(xs: Int*): ShapeSpec = ShapeSpec(xs: _*)

// ========================================================================
// cuTile Kernel annotation
// ========================================================================

/** cuTile kernel annotation - equivalent to Python's @ct.kernel.
 *
 * Usage:
 * {{{
 * @cuTileKernel(name = "vectorAddKernel", gridType = "1D", blockSize = 128)
 * def vectorAddKernel(a: FloatPtr, b: FloatPtr, result: FloatPtr): Unit = {
 *   // Kernel body using ct.bid, ct.load, ct.store
 * }
 * }}}
 */
class cuTileKernel(
  name: String = "",
  gridType: String = "1D",
  blockSize: Int = 128
) extends scala.annotation.Annotation

/** Alias for cuTileKernel for convenience */
type kernel = cuTileKernel