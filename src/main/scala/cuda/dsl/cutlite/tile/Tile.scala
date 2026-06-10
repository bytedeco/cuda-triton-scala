package cuda.dsl.cutlite.tile

import cuda.dsl.core.FloatPtr

// ========================================================================
// Tile Shape - Type-level representation of tile dimensions
// ========================================================================

/** Tile shape with type-level dimensions.
 *
 * Represents the shape of a tile at the type level.
 */
sealed trait TileShape:
  type Rows <: Int
  type Cols <: Int

/** Dynamic tile shape (for runtime-sized tiles). */
class DynamicTileShape(val rows: Int, val cols: Int) extends TileShape:
  type Rows = Int
  type Cols = Int

/** Fixed tile shape (compile-time known dimensions). */
class FixedTileShape[R <: Int, C <: Int] extends TileShape:
  type Rows = R
  type Cols = C

// ========================================================================
// Tile - Core tile abstraction
// ========================================================================

/** Tile - A tile (block) of data from a tensor.
 *
 * Similar to CUTLITE's Tile abstraction.
 * A tile is a view into a portion of a tensor with a specific shape.
 *
 * @param ptr Pointer to the underlying data
 * @param offset Starting offset in the tensor
 * @param rows Number of rows in this tile
 * @param cols Number of columns in this tile
 */
class Tile(
  val ptr: FloatPtr,
  val offset: Int,
  val rows: Int,
  val cols: Int
):
  /** Total number of elements in this tile */
  def size: Int = rows * cols

  /** Stride (number of elements per row) */
  def stride: Int = cols

  /** Load element at position (i, j) within the tile */
  def apply(i: Int, j: Int): Float = ptr(offset + i * cols + j)

  /** Store element at position (i, j) within the tile */
  def update(i: Int, j: Int, value: Float): Unit =
    ptr(offset + i * cols + j) = value

  /** Copy this tile to a new buffer */
  def toArray: Array[Float] =
    val arr = new Array[Float](size)
    var i = 0
    while (i < rows) {
      var j = 0
      while (j < cols) {
        arr(i * cols + j) = apply(i, j)
        j += 1
      }
      i += 1
    }
    arr

  /** Fill tile with a constant value */
  def fill(value: Float): Unit =
    var i = 0
    while (i < rows) {
      var j = 0
      while (j < cols) {
        update(i, j, value)
        j += 1
      }
      i += 1
    }

  /** Map operation over tile elements */
  def map(f: Float => Float): Tile =
    val result = FloatPtr.alloc(size)
    var i = 0
    while (i < rows) {
      var j = 0
      while (j < cols) {
        result(i * cols + j) = f(apply(i, j))
        j += 1
      }
      i += 1
    }
    new Tile(result, 0, rows, cols)

  /** Zip with another tile element-wise */
  def zip(other: Tile)(f: (Float, Float) => Float): Tile =
    require(rows == other.rows && cols == other.cols)
    val result = FloatPtr.alloc(size)
    var i = 0
    while (i < rows) {
      var j = 0
      while (j < cols) {
        result(i * cols + j) = f(apply(i, j), other(i, j))
        j += 1
      }
      i += 1
    }
    new Tile(result, 0, rows, cols)

  // Arithmetic operations
  def +(other: Tile): Tile = zip(other)(_ + _)
  def -(other: Tile): Tile = zip(other)(_ - _)
  def *(other: Tile): Tile = zip(other)(_ * _)
  def /(other: Tile): Tile = zip(other)(_ / _)

/** Companion object for Tile */
object Tile:
  /** Create a new tile */
  def apply(ptr: FloatPtr, rows: Int, cols: Int): Tile =
    new Tile(ptr, 0, rows, cols)

  /** Create a tile with explicit offset */
  def apply(ptr: FloatPtr, offset: Int, rows: Int, cols: Int): Tile =
    new Tile(ptr, offset, rows, cols)

  /** Common tile sizes */
  object TileSize:
    val TILE_8: (Int, Int) = (8, 8)
    val TILE_16: (Int, Int) = (16, 16)
    val TILE_32: (Int, Int) = (32, 32)
    val TILE_64: (Int, Int) = (64, 64)
    val TILE_128: (Int, Int) = (128, 128)
    val TILE_16x64: (Int, Int) = (16, 64)
    val TILE_64x16: (Int, Int) = (64, 16)

// ========================================================================
// TiledView - View a tensor as a collection of tiles
// ========================================================================

/** TiledView - Splits a tensor into tiles for block-wise processing.
 *
 * Similar to CUTLITE's TiledView.
 *
 * @param ptr Pointer to the underlying tensor data
 * @param tensorRows Total rows in the tensor
 * @param tensorCols Total columns in the tensor
 * @param tileRows Rows per tile
 * @param tileCols Columns per tile
 */
class TiledView(
  val ptr: FloatPtr,
  val tensorRows: Int,
  val tensorCols: Int,
  val tileRows: Int,
  val tileCols: Int
):
  require(tensorRows % tileRows == 0, s"$tensorRows not divisible by $tileRows")
  require(tensorCols % tileCols == 0, s"$tensorCols not divisible by $tileCols")

  /** Number of tiles along rows */
  def numTilesRow: Int = tensorRows / tileRows

  /** Number of tiles along columns */
  def numTilesCol: Int = tensorCols / tileCols

  /** Total number of tiles */
  def numTiles: Int = numTilesRow * numTilesCol

  /** Total elements in tensor */
  def size: Int = tensorRows * tensorCols

  /** Get tile at position (tileRow, tileCol) */
  def tile(tileRow: Int, tileCol: Int): Tile =
    require(tileRow >= 0 && tileRow < numTilesRow)
    require(tileCol >= 0 && tileCol < numTilesCol)
    val offset = tileRow * tileRows * tensorCols + tileCol * tileCols
    new Tile(ptr, offset, tileRows, tileCols)

  /** Get tile by linear index */
  def tile(index: Int): Tile =
    val tileRow = index / numTilesCol
    val tileCol = index % numTilesCol
    tile(tileRow, tileCol)

  /** Iterate over all tiles row-major */
  def iterator: Iterator[Tile] = new Iterator[Tile]:
    private var idx = 0
    def hasNext: Boolean = idx < numTiles
    def next(): Tile =
      val t = tile(idx)
      idx += 1
      t

  /** Iterate over tiles with their coordinates */
  def tiles: Iterator[(Int, Int, Tile)] = new Iterator[(Int, Int, Tile)]:
    private var i = 0
    private var j = 0
    def hasNext: Boolean = i < numTilesRow
    def next(): (Int, Int, Tile) =
      val result = (i, j, tile(i, j))
      j += 1
      if (j >= numTilesCol) {
        j = 0
        i += 1
      }
      result

  /** Partition into tiles */
  def partition(): Iterator[Tile] = iterator

/** Companion object for TiledView */
object TiledView:
  /** Create a TiledView with automatic tile size adjustment */
  def apply(ptr: FloatPtr, tensorRows: Int, tensorCols: Int, tileSize: Int = 128): TiledView =
    val tilesRow = (tensorRows + tileSize - 1) / tileSize
    val tilesCol = (tensorCols + tileSize - 1) / tileSize
    val actualTileRows = tensorRows / tilesRow
    val actualTileCols = tensorCols / tilesCol
    new TiledView(ptr, tensorRows, tensorCols, actualTileRows, actualTileCols)

  /** Create a TiledView with explicit tile shape */
  def apply(ptr: FloatPtr, tensorRows: Int, tensorCols: Int, tileRows: Int, tileCols: Int): TiledView =
    new TiledView(ptr, tensorRows, tensorCols, tileRows, tileCols)

// ========================================================================
// load_tile_like - Load a tile matching another tile's shape
// ========================================================================

/** Load a tile from a tensor that matches the shape of the reference tile.
 *
 * This is the core CUTLITE operation that loads a tile of data
 * from a tensor, using another tile's shape as a template.
 *
 * @param tensor The source tensor
 * @param like The reference tile whose shape should be matched
 * @return A tile with the same shape as `like`
 */
def load_tile_like(tensor: Tensor, like: Tile): Tile =
  new Tile(tensor.ptr, like.offset, like.rows, like.cols)

/** Load a tile from a TiledView at a specific position.
 *
 * @param view The TiledView to load from
 * @param tileRow Row index of the tile
 * @param tileCol Column index of the tile
 * @return The tile at the specified position
 */
def load_tile(view: TiledView, tileRow: Int, tileCol: Int): Tile =
  view.tile(tileRow, tileCol)

// ========================================================================
// Tensor - High-level tensor abstraction
// ========================================================================

/** Tensor - A tensor with shape information.
 *
 * Represents a multi-dimensional array with compile-time or runtime shape.
 */
class Tensor(
  val ptr: FloatPtr,
  val rows: Int,
  val cols: Int
):
  require(rows > 0 && cols > 0, "Tensor dimensions must be positive")

  def size: Int = rows * cols

  def apply(i: Int, j: Int): Float = ptr(i * cols + j)
  def update(i: Int, j: Int, value: Float): Unit = ptr(i * cols + j) = value
  def apply(idx: Int): Float = ptr(idx)
  def update(idx: Int, value: Float): Unit = ptr(idx) = value

  /** Create a TiledView of this tensor with the given tile size */
  def tiled(tileSize: Int = 128): TiledView =
    TiledView(ptr, rows, cols, tileSize)

  /** Create a TiledView with explicit tile shape */
  def tiled(tileRows: Int, tileCols: Int): TiledView =
    TiledView(ptr, rows, cols, tileRows, tileCols)

  /** Load a tile matching another tile's shape */
  def loadTileLike(like: Tile): Tile =
    load_tile_like(this, like)

  /** Copy to a new tensor */
  def copy(): Tensor =
    val newPtr = FloatPtr.alloc(size)
    var i = 0
    while (i < size) {
      newPtr(i) = ptr(i)
      i += 1
    }
    new Tensor(newPtr, rows, cols)

  /** Fill with a constant value */
  def fill(value: Float): Unit =
    var i = 0
    while (i < size) {
      ptr(i) = value
      i += 1
    }

  /** Apply a function element-wise */
  def map(f: Float => Float): Tensor =
    val result = FloatPtr.alloc(size)
    var i = 0
    while (i < size) {
      result(i) = f(ptr(i))
      i += 1
    }
    new Tensor(result, rows, cols)

  override def toString: String = s"Tensor($rows x $cols)"

/** Companion object for Tensor */
object Tensor:
  /** Create a tensor filled with zeros */
  def zeros(rows: Int, cols: Int): Tensor =
    val ptr = FloatPtr.alloc(rows * cols)
    var i = 0
    while (i < rows * cols) {
      ptr(i) = 0.0f
      i += 1
    }
    new Tensor(ptr, rows, cols)

  /** Create a tensor filled with ones */
  def ones(rows: Int, cols: Int): Tensor =
    val ptr = FloatPtr.alloc(rows * cols)
    var i = 0
    while (i < rows * cols) {
      ptr(i) = 1.0f
      i += 1
    }
    new Tensor(ptr, rows, cols)

  /** Create a tensor with random values */
  def rand(rows: Int, cols: Int): Tensor =
    val ptr = FloatPtr.alloc(rows * cols)
    var i = 0
    while (i < rows * cols) {
      ptr(i) = scala.util.Random.nextFloat()
      i += 1
    }
    new Tensor(ptr, rows, cols)

// ========================================================================
// Store operations
// ========================================================================

/** Store a tile back to a tensor (in-place modification).
 *
 * @param tensor The target tensor
 * @param tile The tile to store
 * @param offset Starting offset (default 0)
 */
def store(tensor: Tensor, tile: Tile, offset: Int = 0): Unit =
  require(tile.size + offset <= tensor.size)
  var i = 0
  while (i < tile.size) {
    tensor.ptr(offset + i) = tile.ptr(tile.offset + i)
    i += 1
  }

// ========================================================================
// CUTLITE-style module DSL
// ========================================================================

/** CUTLITE module - A collection of kernel entry functions.
 *
 * This is a marker trait for organizing CUTLITE operations into modules.
 * Similar to the `#[cutile::module]` decorator in CUTLITE Python/Rust.
 *
 * Example:
 * {{{
 * object MyKernels extends CutliteModule:
 *   // Kernel definitions go here
 * }}}
 */
trait CutliteModule

/** Entry point marker for CUTLITE kernel functions.
 *
 * Similar to `#[cutile::entry()]` in CUTLITE Python/Rust.
 *
 * Usage:
 * {{{
 * @entry
 * def myKernel(z: Tensor, x: Tensor, y: Tensor): Unit = {
 *   val tx = x.loadTileLike(z.tiled.head)
 *   val ty = y.loadTileLike(z.tiled.head)
 *   store(z, tx + ty)
 * }
 * }}}
 */
class entry extends scala.annotation.Annotation
