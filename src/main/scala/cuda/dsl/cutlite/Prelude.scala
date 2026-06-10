package cuda.dsl.cutlite

import cuda.dsl.cutlite.{CutliteMath, CutliteReduce}

/** CUTLITE Prelude - Import all CUTLITE operations with convenient implicits.
 *
 * Usage:
 * {{{
 * import cuda.dsl.cutlite.prelude._
 *
 * // Now you have access to all CUTLITE operations
 * }}}
 */
object Prelude:

  // Import core operations
  val Math = CutliteMath
  val Reduce = CutliteReduce

// ========================================================================
// Type alias re-exports
// ========================================================================

/** Type alias for backward compatibility with older code */
type Tensor = cuda.dsl.cutlite.tile.Tensor

/** Type alias for Tile */
type Tile = cuda.dsl.cutlite.tile.Tile

/** Type alias for TiledView */
type TiledView = cuda.dsl.cutlite.tile.TiledView