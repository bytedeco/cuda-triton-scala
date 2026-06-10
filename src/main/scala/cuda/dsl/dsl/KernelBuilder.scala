package cuda.dsl.dsl

/** Simple DSL for building CUDA kernels programmatically
 *
 *  Example:
 *  {{{
 *  val code = Kernel("flashAttention")
 *    .param("float*", "q")
 *    .param("float*", "k")
 *    .param("float*", "v")
 *    .param("float*", "out")
 *    .param("int", "N")
 *    .param("int", "d")
 *    .local("float", "maxScore", "0.0f")
 *    .local("float", "sumExp", "0.0f")
 *    .forLoop("j", 0, "N", 1)(
 *      _.assign("score", "score + q[i] * k[j]"),
 *      _.if_("score > maxScore")(
 *        _.assign("maxScore", "score")
 *      )
 *    )
 *    .emit()
 *
 *  println(code)
 *  }}}
 */
class Kernel(val name: String):

  private val params = collection.mutable.ListBuffer[(String, String)]()
  private val locals = collection.mutable.ListBuffer[(String, String, String)]()
  private val statements = collection.mutable.ListBuffer[String]()

  // ==========================================================================
  // Parameter management
  // ==========================================================================

  def param(typ: String, name: String): this.type =
    params += ((typ, name))
    this

  def local(typ: String, name: String, init: String): this.type =
    locals += ((typ, name, init))
    this

  // ==========================================================================
  // Statement builders
  // ==========================================================================

  def assign(lhs: String, rhs: String): this.type =
    statements += s"$lhs = $rhs;"
    this

  def if_(cond: String)(thenBlock: => Unit)(using this.type): this.type =
    statements += s"if ($cond) {"
    thenBlock
    statements += "}"
    this

  def forLoop(iter: String, start: String, end: String, step: Int)(stmts: (this.type => Unit)*): this.type =
    statements += s"for (int $iter = $start; $iter < $end; $iter += $step) {"
    stmts.foreach(s => s(this))
    statements += "}"
    this

  def expr(code: String): this.type =
    statements += s"$code;"
    this

  def emit(): String =
    val sb = new StringBuilder

    // Kernel signature
    sb.append(s"extern \"C\" __global__ void $name(")
    sb.append("\n")
    params.foreach { (t, n) =>
      sb.append(s"    $t $n,\n")
    }
    sb.append("    int n)\n")
    sb.append("{\n")

    // Thread ID setup
    sb.append("    int i = blockIdx.x * blockDim.x + threadIdx.x;\n")
    sb.append("    if (i >= n) return;\n")
    sb.append("\n")

    // Local variables
    locals.foreach { (t, n, init) =>
      sb.append(s"    $t $n = $init;\n")
    }
    if (locals.nonEmpty) sb.append("\n")

    // Statements
    statements.foreach { s =>
      sb.append(s"    $s\n")
    }

    sb.append("}\n")

    sb.toString()
end Kernel

object Kernel:
  def apply(name: String) = new Kernel(name)
