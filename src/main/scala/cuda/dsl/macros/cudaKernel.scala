package cuda.dsl.macros

import scala.annotation.{MacroAnnotation, experimental}
import scala.quoted.*
import cuda.dsl.core.Ptr

@experimental
class cudaKernel extends MacroAnnotation:
  def transform(using Quotes)(definition: quotes.reflect.Definition, companion: Option[quotes.reflect.Definition]): List[quotes.reflect.Definition] =
    import quotes.reflect.*
    definition match
      case DefDef(name, paramss, tpt, Some(rhs)) =>
        val startTime = System.nanoTime()

        val paramInfo = paramss.flatMap(_.params).collect {
          case ValDef(n, tpt, _) => (n, typeToCudaType(tpt.tpe), isPointerType(tpt.tpe))
        }

        val pointerParams = paramInfo.filter(_._3).map(_._1).toSet
        val scalarParams = paramInfo.filter(!_._3).map(_._1).toSet
        val outParam = paramInfo.find((n, _, isPtr) => isPtr && n == "out").map(_._1)

        val bodyCode = translateTerm(rhs, pointerParams, scalarParams, outParam)
        val trimmedBody = bodyCode.trim
        val needsAssign = !trimmedBody.startsWith("out[") &&
                          !trimmedBody.startsWith("output[") &&
                          !trimmedBody.startsWith("if") &&
                          !trimmedBody.startsWith("for") &&
                          !trimmedBody.contains(";")
        val wrappedBody = if (needsAssign) s"out[i] = $trimmedBody;" else bodyCode

        val threadIdVar = "int i = blockIdx.x * blockDim.x + threadIdx.x;"
        val boundsCheck = "if (i < n)"

        val cudaCode = s"""
// ============================================================================
// CUDA Kernel: ${name}
// ============================================================================
extern "C" __global__ void ${name}(${paramInfo.map((n, t, _) => s"$t $n").mkString(", ")}, float* out, int n) {
    ${threadIdVar}
    ${boundsCheck} {
        ${wrappedBody}
    }
}
// ============================================================================
""".trim

        try
          val f = new java.io.FileWriter("/tmp/cuda_dsl_generated_kernels.txt", true)
          f.write(s"[cudaKernel] ${name}\n${cudaCode}\n\n")
          f.close()
        catch
          case e: Exception =>
            System.err.println(s"[@cudaKernel] Could not write to file: ${e.getMessage}")

        val endTime = System.nanoTime()
        System.err.println(s"[@cudaKernel] Generated: ${name} in ${(endTime - startTime) / 1e6}ms")

        List(definition)
      case _ =>
        report.errorAndAbort("@cudaKernel can only be applied to functions")

@experimental
class cudaOperator extends MacroAnnotation:
  def transform(using Quotes)(definition: quotes.reflect.Definition, companion: Option[quotes.reflect.Definition]): List[quotes.reflect.Definition] =
    import quotes.reflect.*
    definition match
      case DefDef(name, paramss, tpt, Some(rhs)) =>
        val paramInfo = paramss.flatMap(_.params).collect {
          case ValDef(n, tpt, _) => (n, typeToCudaType(tpt.tpe), isPointerType(tpt.tpe))
        }

        val pointerParams = paramInfo.filter(_._3).map(_._1).toSet
        val scalarParams = paramInfo.filter(!_._3).map(_._1).toSet

        val bodyCode = translateTerm(rhs, pointerParams, scalarParams)

        val cudaCode = s"""
// ============================================================================
// CUDA Operator: ${name}_op
// ============================================================================
extern "C" __global__ void ${name}_op(${paramInfo.map((n, t, _) => s"$t $n").mkString(", ")}) {
    ${bodyCode}
}
// ============================================================================
""".trim

        try
          val f = new java.io.FileWriter("/tmp/cuda_dsl_generated_operators.txt", true)
          f.write(s"[cudaOperator] ${name}_op\n${cudaCode}\n\n")
          f.close()
        catch
          case e: Exception =>
            System.err.println(s"[@cudaOperator] Could not write to file: ${e.getMessage}")

        System.err.println(s"[@cudaOperator] Generated: ${name}_op")

        List(definition)
      case _ =>
        report.errorAndAbort("@cudaOperator can only be applied to functions")

private def isPointerType(using Quotes)(tpe: quotes.reflect.TypeRepr): Boolean = {
  import quotes.reflect.*
  tpe match {
    case t if t <:< TypeRepr.of[Ptr[?]] => true
    case t if t <:< TypeRepr.of[Array[?]] => true
    case t if t =:= TypeRepr.of[Float] => true
    case _ => false
  }
}

private def translateTerm(using Quotes)(
  term: quotes.reflect.Term,
  pointerParams: Set[String],
  scalarParams: Set[String],
  outParam: Option[String] = None,
  loopVarTypes: Map[String, String] = Map.empty
): String = {
  import quotes.reflect.*
  term match {
    case Block(stats, lastExpr) =>
      val statCode = stats.collect { case s: Term => translateTerm(s, pointerParams, scalarParams, outParam, loopVarTypes) }.filter(_.nonEmpty)
      val lastCode = translateTerm(lastExpr, pointerParams, scalarParams, outParam, loopVarTypes)
      if (statCode.isEmpty) lastCode
      else s"${statCode.mkString("\n")}\n$lastCode"

    case ValDef(name, tpt, Some(rhs)) =>
      val t = translateLocalType(tpt.tpe)
      val rhsCode = translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)
      s"$t $name = $rhsCode"

    case Assign(lhs, rhs) =>
      s"${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} = ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)}"

    case If(cond, thenp, elsep) =>
      s"(${translateTerm(cond, pointerParams, scalarParams, outParam, loopVarTypes)} ? ${translateTerm(thenp, pointerParams, scalarParams, outParam, loopVarTypes)} : ${translateTerm(elsep, pointerParams, scalarParams, outParam, loopVarTypes)})"

    case Apply(Select(lhs, "+"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} + ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "-"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} - ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "*"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} * ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "/"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} / ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "%"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} % ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, ">"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} > ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "<"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} < ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, ">="), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} >= ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "<="), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} <= ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "=="), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} == ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "!="), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} != ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "&&"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} && ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "||"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} || ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "|"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} | ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "&"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} & ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "^"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} ^ ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, ">>"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} >> ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"
    case Apply(Select(lhs, "<<"), List(rhs)) =>
      s"(${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} << ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)})"

    case Apply(Select(lhs, "+="), List(rhs)) =>
      s"${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} += ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)}"
    case Apply(Select(lhs, "-="), List(rhs)) =>
      s"${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} -= ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)}"
    case Apply(Select(lhs, "*="), List(rhs)) =>
      s"${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} *= ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)}"
    case Apply(Select(lhs, "/="), List(rhs)) =>
      s"${translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)} /= ${translateTerm(rhs, pointerParams, scalarParams, outParam, loopVarTypes)}"

    case Apply(Select(obj, "apply"), List(idx)) =>
      val result = s"${translateTerm(obj, pointerParams, scalarParams, outParam, loopVarTypes)}[${translateTerm(idx, pointerParams, scalarParams, outParam, loopVarTypes)}]"
      result
    // Handle implicit arguments (Scala 3 macro expands q(idx) with implicit as q.apply(idx)(using MemoryOps_Float))
    case Apply(Apply(Select(obj, "apply"), List(idx)), _) =>
      val result = s"${translateTerm(obj, pointerParams, scalarParams, outParam, loopVarTypes)}[${translateTerm(idx, pointerParams, scalarParams, outParam, loopVarTypes)}]"
      result
    case Apply(Select(obj, "update"), List(idx, value)) =>
      s"${translateTerm(obj, pointerParams, scalarParams, outParam, loopVarTypes)}[${translateTerm(idx, pointerParams, scalarParams, outParam, loopVarTypes)}] = ${translateTerm(value, pointerParams, scalarParams, outParam, loopVarTypes)}"

    case Apply(TypeApply(Select(range, "foreach"), _), List(Lambda(params, body))) =>
      val loopVar = params match
        case List(ValDef(name, _, _)) => name
        case _ => "i"
      translateForeach(range, loopVar, body, pointerParams, scalarParams, outParam, loopVarTypes)

    case Apply(Select(range, "foreach"), List(Lambda(params, body))) =>
      val loopVar = params match
        case List(ValDef(name, _, _)) => name
        case _ => "i"
      translateForeach(range, loopVar, body, pointerParams, scalarParams, outParam, loopVarTypes)

    case Apply(callee, args) =>
      translateCall(callee, args, pointerParams, scalarParams, outParam, loopVarTypes)

    case Select(qualifier, fieldName) =>
      val qualCode = translateTerm(qualifier, pointerParams, scalarParams, outParam, loopVarTypes)
      fieldName match {
        case "+" => s"($qualCode + "
        case "-" => s"($qualCode - "
        case "*" => s"($qualCode * "
        case "/" => s"($qualCode / "
        case "%" => s"($qualCode % "
        case ">" => s"($qualCode > "
        case "<" => s"($qualCode < "
        case ">=" => s"($qualCode >= "
        case "<=" => s"($qualCode <= "
        case "==" => s"($qualCode == "
        case "!=" => s"($qualCode != "
        case "&&" => s"($qualCode && "
        case "||" => s"($qualCode || "
        case "|" => s"($qualCode | "
        case "&" => s"($qualCode & "
        case "^" => s"($qualCode ^ "
        case "<<" => s"($qualCode << "
        case ">>" => s"($qualCode >> "
        case "unary_-" => s"(-$qualCode)"
        case "unary_+" => s"(+$qualCode)"
        case "unary_!" => s"(!$qualCode)"
        case "unary_~" => s"(~$qualCode)"
        case "toFloat" | "toFloatConversion" => s"((float)$qualCode)"
        case "toDouble" | "float2double" => s"((double)$qualCode)"
        case "double2float" => s"((float)$qualCode)"
        case _ => s"$qualCode.$fieldName"
      }

    case Literal(IntConstant(n)) => n.toString
    case Literal(LongConstant(n)) => s"${n}L"
    case Literal(FloatConstant(f)) => translateFloat(f)
    case Literal(DoubleConstant(d)) => s"${d}f"
    case Literal(BooleanConstant(b)) => if (b) "true" else "false"
    case Literal(StringConstant(s)) => s"\"$s\""
    case Literal(CharConstant(c)) => s"'$c'"
    case Literal(ByteConstant(b)) => b.toString
    case Literal(ShortConstant(s)) => s.toString
    case Literal(UnitConstant()) => ""
    case Literal(NullConstant()) => "NULL"

    case Ident(name) =>
      if pointerParams.contains(name) then s"${name}[i]"
      else if loopVarTypes.contains(name) then name
      else
        name match {
          case "Nil" | "None" => "0"
          case "true" | "True" => "true"
          case "false" | "False" => "false"
          case _ => name
        }

    case Lambda(_, body) => translateTerm(body, pointerParams, scalarParams, outParam, loopVarTypes)

    case TypeApply(callee, _) => translateTerm(callee, pointerParams, scalarParams, outParam, loopVarTypes)

    case Closure(meth, _) => translateTerm(meth, pointerParams, scalarParams, outParam, loopVarTypes)

    case Inlined(_, _, body) => translateTerm(body, pointerParams, scalarParams, outParam, loopVarTypes)

    case Return(expr, _) => s"return ${translateTerm(expr, pointerParams, scalarParams, outParam, loopVarTypes)}"

    case Typed(expr, _) => translateTerm(expr, pointerParams, scalarParams, outParam, loopVarTypes)

    case Wildcard() => "_"

    case _ => term.show
  }
}

private def translateForeach(using Quotes)(
  range: quotes.reflect.Term,
  loopVar: String,
  body: quotes.reflect.Term,
  pointerParams: Set[String],
  scalarParams: Set[String],
  outParam: Option[String],
  loopVarTypes: Map[String, String]
): String = {
  import quotes.reflect.*

  range match {
    case Apply(Select(innerApply, "until"), List(end)) if innerApply.show.contains("intWrapper") =>
      innerApply match
        case Apply(Ident("intWrapper"), List(start)) =>
          val startCode = translateTerm(start, pointerParams, scalarParams, outParam, loopVarTypes)
          val endCode = translateTerm(end, pointerParams, scalarParams, outParam, loopVarTypes)
          val newLoopVarTypes = loopVarTypes + (loopVar -> "int")
          val bodyCode = translateTerm(body, pointerParams, scalarParams, outParam, newLoopVarTypes)
          s"for (int $loopVar = $startCode; $loopVar < $endCode; $loopVar++) {\n$bodyCode\n}"

    case Apply(Select(start, "until"), List(end)) =>
      val startCode = translateTerm(start, pointerParams, scalarParams, outParam, loopVarTypes)
      val endCode = translateTerm(end, pointerParams, scalarParams, outParam, loopVarTypes)
      val newLoopVarTypes = loopVarTypes + (loopVar -> "int")
      val bodyCode = translateTerm(body, pointerParams, scalarParams, outParam, newLoopVarTypes)
      s"for (int $loopVar = $startCode; $loopVar < $endCode; $loopVar++) {\n$bodyCode\n}"

    case Apply(Select(start, "to"), List(end)) =>
      val startCode = translateTerm(start, pointerParams, scalarParams, outParam, loopVarTypes)
      val endCode = translateTerm(end, pointerParams, scalarParams, outParam, loopVarTypes)
      val newLoopVarTypes = loopVarTypes + (loopVar -> "int")
      val bodyCode = translateTerm(body, pointerParams, scalarParams, outParam, newLoopVarTypes)
      s"for (int $loopVar = $startCode; $loopVar <= $endCode; $loopVar++) {\n$bodyCode\n}"

    case Apply(Select(start, "until"), List(end, step)) =>
      val startCode = translateTerm(start, pointerParams, scalarParams, outParam, loopVarTypes)
      val endCode = translateTerm(end, pointerParams, scalarParams, outParam, loopVarTypes)
      val stepCode = translateTerm(step, pointerParams, scalarParams, outParam, loopVarTypes)
      val newLoopVarTypes = loopVarTypes + (loopVar -> "int")
      val bodyCode = translateTerm(body, pointerParams, scalarParams, outParam, newLoopVarTypes)
      s"for (int $loopVar = $startCode; $loopVar < $endCode; $loopVar += $stepCode) {\n$bodyCode\n}"

    case other =>
      val rangeCode = translateTerm(other, pointerParams, scalarParams, outParam, loopVarTypes)
      s"$rangeCode foreach not supported"
  }
}

private def translateCall(using Quotes)(
  callee: quotes.reflect.Term,
  args: List[quotes.reflect.Term],
  pointerParams: Set[String],
  scalarParams: Set[String],
  outParam: Option[String],
  loopVarTypes: Map[String, String]
): String = {
  import quotes.reflect.*

  callee match {
    case Select(lhs, "intWrapper") =>
      args.map(translateTerm(_, pointerParams, scalarParams, outParam, loopVarTypes)).mkString(", ")

    case Select(lhs, methodName) =>
      val lhsCode = translateTerm(lhs, pointerParams, scalarParams, outParam, loopVarTypes)
      val argsCode = args.map(translateTerm(_, pointerParams, scalarParams, outParam, loopVarTypes)).mkString(", ")
      methodName match {
        case "maxInt" | "MAXINT" => s"max($argsCode)"
        case "minInt" | "MININT" => s"min($argsCode)"
        case "sqrt" | "SQRT" => s"sqrtf($argsCode)"
        case "exp" | "EXP" => s"expf($argsCode)"
        case "log" | "LOG" => s"logf($argsCode)"
        case "log10" | "LOG10" => s"log10f($argsCode)"
        case "abs" | "ABS" => s"fabsf($argsCode)"
        case "max" | "MAX" => s"fmaxf($argsCode)"
        case "min" | "MIN" => s"fminf($argsCode)"
        case "sin" | "SIN" => s"sinf($argsCode)"
        case "cos" | "COS" => s"cosf($argsCode)"
        case "tan" | "TAN" => s"tanf($argsCode)"
        case "pow" | "POW" => s"powf($argsCode)"
        case "fmod" | "FMOD" => s"fmodf($argsCode)"
        case "tanh" | "TANH" => s"tanhf($argsCode)"
        case "sigmoid" => s"(1.0f / (1.0f + expf(-$argsCode)))"
        case "relu" | "RELU" => s"fmaxf($argsCode, 0.0f)"
        case "toFloat" | "toFloatConversion" | "toDouble" | "float2double" | "double2float" => argsCode
        case "IntToFloat" | "int2float" => s"((float)$argsCode)"
        case "FloatToInt" | "float2int" => s"((int)$argsCode)"
        case "threadIdx" => "threadIdx.x"
        case "blockIdx" => "blockIdx.x"
        case "blockDim" => "blockDim.x"
        case "gridDim" => "gridDim.x"
        case "warpSize" => "warpSize"
        case _ => s"($lhsCode$methodName($argsCode))"
      }

    case Ident(name) =>
      val argsCode = args.map(translateTerm(_, pointerParams, scalarParams, outParam, loopVarTypes)).mkString(", ")
      name match {
        case "maxInt" | "MAXINT" => s"max($argsCode)"
        case "minInt" | "MININT" => s"min($argsCode)"
        case "sqrt" | "SQRT" => s"sqrtf($argsCode)"
        case "exp" | "EXP" => s"expf($argsCode)"
        case "log" | "LOG" => s"logf($argsCode)"
        case "log10" | "LOG10" => s"log10f($argsCode)"
        case "abs" | "ABS" => s"fabsf($argsCode)"
        case "max" | "MAX" => s"fmaxf($argsCode)"
        case "min" | "MIN" => s"fminf($argsCode)"
        case "sin" | "SIN" => s"sinf($argsCode)"
        case "cos" | "COS" => s"cosf($argsCode)"
        case "tan" | "TAN" => s"tanf($argsCode)"
        case "pow" | "POW" => s"powf($argsCode)"
        case "fmod" | "FMOD" => s"fmodf($argsCode)"
        case "tanh" | "TANH" => s"tanhf($argsCode)"
        case "sigmoid" => s"(1.0f / (1.0f + expf(-$argsCode)))"
        case "relu" | "RELU" => s"fmaxf($argsCode, 0.0f)"
        case "toFloat" | "toFloatConversion" | "toDouble" | "float2double" | "double2float" => argsCode
        case "threadIdx" => "threadIdx.x"
        case "blockIdx" => "blockIdx.x"
        case "blockDim" => "blockDim.x"
        case "gridDim" => "gridDim.x"
        case "warpSize" => "warpSize"
        case _ => s"$name($argsCode)"
      }

    case TypeApply(callee, _) => translateCall(callee, args, pointerParams, scalarParams, outParam, loopVarTypes)

    case other =>
      val calleeCode = translateTerm(other, pointerParams, scalarParams, outParam, loopVarTypes)
      val argsCode = args.map(translateTerm(_, pointerParams, scalarParams, outParam, loopVarTypes)).mkString(", ")
      s"$calleeCode($argsCode)"
  }
}

private def translateFloat(f: Float): String = {
  if (f.isNaN) "NAN"
  else if (f.isInfinite) if (f > 0) "INFINITY" else "-INFINITY"
  else s"${f}f"
}

private def translateLocalType(using Quotes)(tpe: quotes.reflect.TypeRepr): String = {
  import quotes.reflect.*
  tpe match {
    case t if t =:= TypeRepr.of[Float] => "float"
    case t if t =:= TypeRepr.of[Int] => "int"
    case t if t =:= TypeRepr.of[Long] => "long"
    case t if t =:= TypeRepr.of[Double] => "double"
    case t if t =:= TypeRepr.of[Boolean] => "bool"
    case t if t =:= TypeRepr.of[Byte] => "char"
    case t if t =:= TypeRepr.of[Short] => "short"
    case t if t =:= TypeRepr.of[Unit] => "void"
    case _ => "float"
  }
}

def typeToCudaType(using Quotes)(tpe: quotes.reflect.TypeRepr): String = {
  import quotes.reflect.*
  tpe match {
    case t if t <:< TypeRepr.of[Ptr[?]] => "float*"
    case t if t <:< TypeRepr.of[Array[?]] => "float*"
    case t if t =:= TypeRepr.of[Float] => "float*"
    case t if t =:= TypeRepr.of[Int] => "int"
    case t if t =:= TypeRepr.of[Long] => "long"
    case t if t =:= TypeRepr.of[Double] => "double"
    case t if t =:= TypeRepr.of[Boolean] => "bool"
    case _ => "float*"
  }
}
