/** Tile module - Core Tile and TiledView abstractions.
 *
 * Provides CUTLITE-style tile abstractions:
 * - Tile: A block of data with shape
 * - TiledView: A tensor partitioned into tiles
 * - Tensor: High-level tensor with shape
 * - load_tile_like: Load a tile matching another's shape
 * - store: Write tile back to tensor
 *
 * Also provides CuPy-style API in the `ct` object:
 * - ct.bid(dim): Get block ID in a dimension
 * - ct.load(tensor, index=(...), shape=(...)): Load a tile
 * - ct.store(tensor, index=(...), tile=...): Store a tile back
 * - @cuTileKernel: Kernel annotation equivalent to Python's @ct.kernel
 *
 * Example usage:
 * {{{
 * import cuda.dsl.cutlite.tile._
 *
 * // Create a tensor and tile it
 * val x = Tensor.rand(1024, 1024)
 * val view = x.tiled(128)
 *
 * // Process each tile
 * for (tile <- view) {
 *   val result = tile.map(_ * 2.0f)
 *   // ...
 * }
 *
 * // Load tile like another
 * val z = Tensor.zeros(128, 128)
 * val tx = load_tile_like(x, z.tiled.head)
 *
 * // CuPy-style API
 * import cuda.dsl.cutlite.tile.ct._
 * val blockId = ct.bid(0)
 * val tile = ct.load(x, index = index(blockId), shape = shape(128))
 * ct.store(z, index = index(blockId), tile = tile)
 * }}}
 */
package cuda.dsl.cutlite.tile


