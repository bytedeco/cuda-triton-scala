package cuda.dsl.dsl

import scala.quoted.*

/** Core AST nodes for the Triton-like DSL */
sealed trait Expr[T]

// ============================================================================
// Literals and Variables
// ============================================================================

case class Const[T](value: T) extends Expr[T]
case class Var[T](name: String) extends Expr[T]
case class ProgId(axis: Int) extends Expr[Int]
case class BlockDim() extends Expr[Int]
case class BlockId(axis: Int) extends Expr[Int]
case class ThreadId(axis: Int) extends Expr[Int]

// ============================================================================
// Arithmetic Operations
// ============================================================================

case class BinOp[T](op: String, lhs: Expr[T], rhs: Expr[T]) extends Expr[T]
case class UnaryOp[T](op: String, arg: Expr[T]) extends Expr[T]

// ============================================================================
// Memory Operations (Tile-level)
// ============================================================================

case class Load[T](ptr: Expr[Ptr[T]], mask: Expr[Boolean], other: Expr[T]) extends Expr[Tile[T]]
case class Store[T](ptr: Expr[Ptr[T]], value: Expr[Tile[T]], mask: Expr[Boolean]) extends Stmt
case class TileVar[T](name: String, shape: List[Int]) extends Expr[Tile[T]]

// ============================================================================
// Control Flow
// ============================================================================

case class ForLoop(iter: String, start: Int, end: Int, step: Int, body: List[Stmt]) extends Stmt
case class IfThenElse(cond: Expr[Boolean], thenp: List[Stmt], elsep: List[Stmt]) extends Stmt
case class Reduce(maximize: Boolean, name: String, input: Expr[Float], init: Float, body: List[Stmt]) extends Expr[Float]
case class MathCall(op: String, args: List[Expr[Float]]) extends Expr[Float]
case class Call(op: String, args: List[Expr[?]]) extends Expr[?]

// ============================================================================
// Statements
// ============================================================================

sealed trait Stmt
case class ValDef[T](name: String, value: Expr[T]) extends Stmt
case class Assign[T](lhs: String, rhs: Expr[T]) extends Stmt
case class ExprStmt[T](expr: Expr[T]) extends Stmt
case class Barrier() extends Stmt

// ============================================================================
// Types
// ============================================================================

sealed trait Type
object FloatType extends Type
object IntType extends Type
object BoolType extends Type
object VoidType extends Type
case class TileType(base: Type, shape: List[Int]) extends Type
case class PtrType(base: Type) extends Type

// ============================================================================
// Implicit helpers
// ============================================================================

object Expr:
  given Conversion[Int, Const[Int]] = Const(_)
  given Conversion[Float, Const[Float]] = Const(_)
  given Conversion[Boolean, Const[Boolean]] = Const(_)

/** A tile of data (M x N) */
case class Tile[T](elems: Array[T], shape: List[Int]):
  def apply(i: Int, j: Int): T = elems(i * shape(1) + j)
  def update(i: Int, j: Int, v: T): Unit = elems(i * shape(1) + j) = v

/** Device pointer */
case class Ptr[T](addr: Long)
