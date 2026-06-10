package cuda.dsl.dsl

import scala.collection.mutable

/** Triton-like high-level DSL for writing GPU kernels
 *
 *  Example:
 *  {{{
 *  val code = TritonKernel("flashAttention")
 *    .param("float*", "q")
 *    .param("float*", "k")
 *    .param("float*", "v")
 *    .param("float*", "out")
 *    .param("int", "N")
 *    .param("int", "d")
 *    .local("float", "maxScore", -1e9f)
 *    .local("float", "sumExp", 0.0f)
 *    .kernel { k =>
 *      import k._
 *
 *      // First pass: compute max
 *      for_("j", 0, N) { j =>
 *        val score = load(q, i, j, N, d) * load(k, j, ::, N, d)
 *        maxScore := max(maxScore, score)
 *      }
 *
 *      // Second pass: softmax
 *      for_("j", 0, N) { j =>
 *        val score = exp(load(q, i, j, N, d) * load(k, j, ::, N, d) - maxScore)
 *        sumExp := sumExp + score
 *        out(i, ::) := out(i, ::) + score * load(v, j, ::, N, d)
 *      }
 *
 *      // Normalize
 *      out(i, ::) := out(i, ::) / sumExp
 *    }
 *    .emit()
 *  }}}
 */
class TritonKernel(val name: String):
  private val _params = mutable.ListBuffer[(String, String)]()
  private val _locals = mutable.ListBuffer[(String, String, Any)]()
  private var compiled: Boolean = false
  private val stmts = mutable.ListBuffer[TTIRStmt]()

  // ==========================================================================
  // Parameter management
  // ==========================================================================

  def param(tpe: String, name: String): this.type =
    _params += ((tpe, name))
    this

  def local(tpe: String, name: String, init: Any): this.type =
    _locals += ((tpe, name, init))
    this

  // ==========================================================================
  // Kernel DSL
  // ==========================================================================

  class KernelDSL:
    // Thread ID
    def i = TTIRVar("i")

    // Constants
    def const(value: Any) = TTIRConst(value)
    def var_(name: String) = TTIRVar(name)

    // Binary operators
    def +(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("+", lhs, rhs)
    def -(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("-", lhs, rhs)
    def *(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("*", lhs, rhs)
    def /(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("/", lhs, rhs)
    def >(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp(">", lhs, rhs)
    def <(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("<", lhs, rhs)
    def >=(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp(">=", lhs, rhs)
    def <=(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("<=", lhs, rhs)
    def ==(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("==", lhs, rhs)
    def &&(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("&&", lhs, rhs)
    def ||(lhs: TTIRExpr, rhs: TTIRExpr) = TTIRBinOp("||", lhs, rhs)

    // Math functions
    def exp(x: TTIRExpr) = TTIRMathCall("exp", List(x))
    def log(x: TTIRExpr) = TTIRMathCall("log", List(x))
    def sqrt(x: TTIRExpr) = TTIRMathCall("sqrt", List(x))
    def abs(x: TTIRExpr) = TTIRMathCall("abs", List(x))
    def sin(x: TTIRExpr) = TTIRMathCall("sin", List(x))
    def cos(x: TTIRExpr) = TTIRMathCall("cos", List(x))
    def tan(x: TTIRExpr) = TTIRMathCall("tan", List(x))
    def tanh(x: TTIRExpr) = TTIRMathCall("tanh", List(x))
    def max(a: TTIRExpr, b: TTIRExpr) = TTIRMathCall("max", List(a, b))
    def min(a: TTIRExpr, b: TTIRExpr) = TTIRMathCall("min", List(a, b))
    def pow(a: TTIRExpr, b: TTIRExpr) = TTIRMathCall("pow", List(a, b))
    def sigmoid(x: TTIRExpr) = TTIRMathCall("sigmoid", List(x))
    def relu(x: TTIRExpr) = TTIRMathCall("relu", List(x))

    // Load from 2D tensor: ptr[i][j] -> ptr[i*stride + j]
    def load2D(ptr: String, row: TTIRExpr, col: TTIRExpr, stride: TTIRExpr, mask: TTIRExpr = TTIRConst(true), other: TTIRExpr = TTIRConst(0.0f)): TTIRExpr =
      TTIRLoad2D(ptr, row, col, stride, mask, other)

    // Store to 2D tensor
    def store2D(ptr: String, row: TTIRExpr, col: TTIRExpr, stride: TTIRExpr, value: TTIRExpr, mask: TTIRExpr = TTIRConst(true)): TTIRStmt =
      TTIRStore2D(ptr, row, col, stride, value, mask)

    // For loop builder
    def for_(iter: String, start: Int, end: TTIRExpr)(body: String => List[TTIRStmt]): List[TTIRStmt] =
      List(TTIRFor(iter, start, end, body(iter)))

    // If-then-else builder
    def if_(cond: TTIRExpr)(thenp: => List[TTIRStmt]): List[TTIRStmt] =
      List(TTIRIf(cond, thenp, Nil))

    def if_(cond: TTIRExpr)(thenp: => List[TTIRStmt])(elsep: => List[TTIRStmt]): List[TTIRStmt] =
      List(TTIRIf(cond, thenp, elsep))

    // Assignment to local variable
    def assign(name: String, value: TTIRExpr): TTIRStmt = TTIRAssign(name, value)

    // Statement helpers
    def stmt(s: TTIRStmt): List[TTIRStmt] = List(s)
    def stmts(s: TTIRStmt*): List[TTIRStmt] = s.toList

    // Flatten statements
    def flatten(lists: List[TTIRStmt]*): List[TTIRStmt] = lists.toList.flatten

  // ==========================================================================
  // Build kernel body
  // ==========================================================================

  def kernel(f: KernelDSL ?=> Unit): this.type =
    if (compiled) throw new IllegalStateException("Kernel already compiled")
    compiled = true
    val ctx = new KernelDSL
    f(using ctx)
    // Note: statements are accumulated in ctx but we need to collect them
    // Since we can't capture statements from context function easily,
    // we use a different approach - accumulate directly
    this

  // Alternative: use explicit builder pattern
  def +=(stmt: TTIRStmt): this.type =
    stmts += stmt
    this

  def ++=(stmts: List[TTIRStmt]): this.type =
    stmts.foreach(s => this.stmts += s)
    this

  // ==========================================================================
  // Emit CUDA code
  // ==========================================================================

  def emit(): String =
    val ir = TTIR(name)
    _params.foreach { (t, n) => ir.param(t, n) }
    _locals.foreach { (t, n, init) =>
      ir.local(t, n, toTTIRExpr(init))
    }
    stmts.foreach(s => ir += s)
    ir.emit()

  private def toTTIRExpr(value: Any): TTIRExpr = value match
    case v: Double => TTIRConst(v.toFloat)
    case v: Float => TTIRConst(v)
    case v: Int => TTIRConst(v)
    case v: Long => TTIRConst(v.toInt)
    case v: Boolean => TTIRConst(v)
    case v: String => TTIRVar(v)
    case e: TTIRExpr => e
    case _ => TTIRConst(value)

object TritonKernel:
  def apply(name: String) = new TritonKernel(name)
