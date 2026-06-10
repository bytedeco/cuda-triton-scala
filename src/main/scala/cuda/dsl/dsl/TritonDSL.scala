package cuda.dsl.dsl

/** Simple Triton-like DSL - converts Scala code to CUDA code
 *
 *  Example:
 *  {{{
 *  val code = Triton("kernel")
 *    .param("float*", "x")
 *    .param("float*", "y")
 *    .param("float*", "out")
 *    .stmt("out[i] = x[i] + y[i];")
 *    .emit()
 *  }}}
 */
class Triton(val name: String):
  private val _params = collection.mutable.ListBuffer[(String, String)]()
  private val _stmts = collection.mutable.ListBuffer[String]()

  def param(tpe: String, n: String): this.type =
    _params += ((tpe, n))
    this

  def stmt(s: String): this.type =
    _stmts += s
    this

  def +=(s: String): this.type = stmt(s)

  def emit(): String =
    val sb = new StringBuilder
    sb.append(s"extern \"C\" __global__ void $name(")
    sb.append("\n")
    _params.foreach { (t, n) =>
      sb.append(s"    $t $n,\n")
    }
    sb.append("    int n)\n")
    sb.append("{\n")
    sb.append("    int i = blockIdx.x * blockDim.x + threadIdx.x;\n")
    sb.append("    if (i >= n) return;\n")
    sb.append("\n")
    _stmts.foreach { s =>
      sb.append(s"    $s\n")
    }
    sb.append("}\n")
    sb.toString()

object Triton:
  def apply(name: String) = new Triton(name)
