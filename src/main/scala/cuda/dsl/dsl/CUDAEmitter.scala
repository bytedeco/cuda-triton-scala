package cuda.dsl.dsl

/** Emits CUDA code from the AST */
class CUDAEmitter {

  private var indent = 0
  private val sb = new StringBuilder

  private def line(s: String = ""): Unit =
    sb.append("  " * indent)
    if (s.nonEmpty) sb.append(s)
    sb.append("\n")

  private def indentBlock(f: => Unit): Unit =
    indent += 1
    f
    indent -= 1

  // ==========================================================================
  // Entry point
  // ==========================================================================

  def emitKernel(name: String, params: List[(String, String)], stmts: List[Stmt]): String =
    sb.clear()
    indent = 0

    // Kernel signature
    line(s"extern \"C\" __global__ void $name(")
    indentBlock {
      params.foreach { (n, t) =>
        line(s"$t $n,")
      }
      line("int n)")
    }
    line("{")

    indentBlock {
      // Thread ID setup
      line("int i = blockIdx.x * blockDim.x + threadIdx.x;")
      line("if (i >= n) return;")
      line("")

      // Emit statements
      stmts.foreach(emitStmt)
    }

    line("}")

    sb.toString

  // ==========================================================================
  // Statement emission
  // ==========================================================================

  private def emitStmt(stmt: Stmt): Unit = stmt match
    case ValDef(name, value) =>
      val v = emitExpr(value)
      line(s"float $name = $v;")

    case Assign(lhs, rhs) =>
      val r = emitExpr(rhs)
      line(s"$lhs = $r;")

    case ExprStmt(expr) =>
      val e = emitExpr(expr)
      if (!isUnitExpr(expr)) line(s"$e;")

    case ForLoop(iter, start, end, step, body) =>
      line(s"for (int $iter = $start; $iter < $end; $iter += $step) {")
      indentBlock {
        body.foreach(emitStmt)
      }
      line("}")

    case IfThenElse(cond, thenp, elsep) =>
      val c = emitExpr(cond)
      line(s"if ($c) {")
      indentBlock {
        thenp.foreach(emitStmt)
      }
      if (elsep.nonEmpty) {
        line("} else {")
        indentBlock {
          elsep.foreach(emitStmt)
        }
      }
      line("}")

    case Barrier() =>
      line("__syncthreads();")

    case Reduce(maximize, name, input, init, body) =>
      // Emit reduction logic
      line(s"float $name = ${emitExpr(init)};")
      line(s"for (int _i = 0; _i < n; _i++) {")
      indentBlock {
        line(s"float ${name}_val = ${emitExpr(input).replace("tile_i", "_i")};")
        val ext = if maximize then s"fmaxf($name, ${name}_val)" else s"fminf($name, ${name}_val)"
        line(s"$name = $ext;")
      }
      line("}")

  // ==========================================================================
  // Expression emission
  // ==========================================================================

  private def isUnitExpr(expr: Expr[?]): Boolean = expr match
    case Const(()) => true
    case _ => false

  private def emitExpr[T](expr: Expr[T]): String = expr match
    case Const(v) => v.toString

    case Var(name) => name

    case ProgId(axis) => s"blockIdx.${"xyz".charAt(axis)}"

    case BlockDim() => s"blockDim.${"xyz".charAt(0)}"

    case BlockId(axis) => s"blockIdx.${"xyz".charAt(axis)}"

    case ThreadId(axis) => s"threadIdx.${"xyz".charAt(axis)}"

    case BinOp("+", lhs, rhs) => s"(${emitExpr(lhs)} + ${emitExpr(rhs)})"
    case BinOp("-", lhs, rhs) => s"(${emitExpr(lhs)} - ${emitExpr(rhs)})"
    case BinOp("*", lhs, rhs) => s"(${emitExpr(lhs)} * ${emitExpr(rhs)})"
    case BinOp("/", lhs, rhs) => s"(${emitExpr(lhs)} / ${emitExpr(rhs)})"
    case BinOp("%", lhs, rhs) => s"fmodf(${emitExpr(lhs)}, ${emitExpr(rhs)})"
    case BinOp("&&", lhs, rhs) => s"(${emitExpr(lhs)} && ${emitExpr(rhs)})"
    case BinOp("||", lhs, rhs) => s"(${emitExpr(lhs)} || ${emitExpr(rhs)})"
    case BinOp(">", lhs, rhs) => s"(${emitExpr(lhs)} > ${emitExpr(rhs)})"
    case BinOp("<", lhs, rhs) => s"(${emitExpr(lhs)} < ${emitExpr(rhs)})"
    case BinOp(">=", lhs, rhs) => s"(${emitExpr(lhs)} >= ${emitExpr(rhs)})"
    case BinOp("<=", lhs, rhs) => s"(${emitExpr(lhs)} <= ${emitExpr(rhs)})"
    case BinOp("==", lhs, rhs) => s"(${emitExpr(lhs)} == ${emitExpr(rhs)})"
    case BinOp("!=", lhs, rhs) => s"(${emitExpr(lhs)} != ${emitExpr(rhs)})"
    case BinOp(op, lhs, rhs) => s"(${emitExpr(lhs)} $op ${emitExpr(rhs)})"

    case UnaryOp("!", arg) => s"(!${emitExpr(arg)})"
    case UnaryOp("-", arg) => s"(-${emitExpr(arg)})"
    case UnaryOp(op, arg) => s"($op${emitExpr(arg)})"

    case Load(ptr, mask, other) =>
      s"(${emitExpr(mask)} ? ${emitExpr(ptr)}[i] : ${emitExpr(other)})"

    case Store(ptr, value, mask) =>
      s"if (${emitExpr(mask)}) ${emitExpr(ptr)}[i] = ${emitExpr(value)}"

    case Call("exp", List(x)) => s"expf(${emitExpr(x)})"
    case Call("log", List(x)) => s"logf(${emitExpr(x)})"
    case Call("sqrt", List(x)) => s"sqrtf(${emitExpr(x)})"
    case Call("abs", List(x)) => s"fabsf(${emitExpr(x)})"
    case Call("sin", List(x)) => s"sinf(${emitExpr(x)})"
    case Call("cos", List(x)) => s"cosf(${emitExpr(x)})"
    case Call("tan", List(x)) => s"tanf(${emitExpr(x)})"
    case Call("tanh", List(x)) => s"tanhf(${emitExpr(x)})"
    case Call("max", List(x, y)) => s"fmaxf(${emitExpr(x)}, ${emitExpr(y)})"
    case Call("min", List(x, y)) => s"fminf(${emitExpr(x)}, ${emitExpr(y)})"
    case Call("dot", List(x, y)) => s"(${emitExpr(x)} * ${emitExpr(y)})"
    case Call(op, args) => s"$op(${args.map(emitExpr).mkString(", ")})"

    case MathCall("exp", List(x)) => s"expf(${emitExpr(x)})"
    case MathCall("log", List(x)) => s"logf(${emitExpr(x)})"
    case MathCall("sqrt", List(x)) => s"sqrtf(${emitExpr(x)})"
    case MathCall("abs", List(x)) => s"fabsf(${emitExpr(x)})"
    case MathCall("sin", List(x)) => s"sinf(${emitExpr(x)})"
    case MathCall("cos", List(x)) => s"cosf(${emitExpr(x)})"
    case MathCall("tan", List(x)) => s"tanf(${emitExpr(x)})"
    case MathCall("tanh", List(x)) => s"tanhf(${emitExpr(x)})"
    case MathCall("max", List(x, y)) => s"fmaxf(${emitExpr(x)}, ${emitExpr(y)})"
    case MathCall("min", List(x, y)) => s"fminf(${emitExpr(x)}, ${emitExpr(y)})"
    case MathCall("pow", List(x, y)) => s"powf(${emitExpr(x)}, ${emitExpr(y)})"
    case MathCall(op, args) => s"${op}f(${args.map(emitExpr).mkString(", ")})"

    case TileVar(name, shape) => name

    case _ => s"/* TODO: ${expr.toString} */"
}
