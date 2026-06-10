package cuda.dsl.operators.elementwise

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Basic element-wise arithmetic operators.
 *  These operators perform element-wise operations on arrays.
 */
object BasicOps {

  /** Element-wise addition: c = a + b */
  @cudaOperator
  def add(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) + b(i)
  }

  /** Element-wise addition with scalar: c = a + scalar */
  @cudaOperator
  def addScalar(a: Ptr[Float], scalar: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) + scalar
  }

  /** Element-wise subtraction: c = a - b */
  @cudaOperator
  def sub(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) - b(i)
  }

  /** Element-wise subtraction with scalar: c = a - scalar */
  @cudaOperator
  def subScalar(a: Ptr[Float], scalar: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) - scalar
  }

  /** Element-wise multiplication: c = a * b */
  @cudaOperator
  def mul(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) * b(i)
  }

  /** Element-wise multiplication with scalar: c = a * scalar */
  @cudaOperator
  def mulScalar(a: Ptr[Float], scalar: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) * scalar
  }

  /** Element-wise division: c = a / b */
  @cudaOperator
  def div(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) / b(i)
  }

  /** Element-wise division with scalar: c = a / scalar */
  @cudaOperator
  def divScalar(a: Ptr[Float], scalar: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) / scalar
  }

  /** Element-wise negation: c = -a */
  @cudaOperator
  def neg(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = -a(i)
  }

  /** Element-wise absolute value: c = abs(a) */
  @cudaOperator
  def abs(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = if (v < 0.0f) -v else v
    }
  }

  /** Element-wise reciprocal: c = 1 / a */
  @cudaOperator
  def reciprocal(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = 1.0f / a(i)
  }

  /** Fused multiply-add: out = a * b + c */
  @cudaOperator
  def fma(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], out: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) out(i) = a(i) * b(i) + c(i)
  }

  /** Fused multiply-add with scalar: out = a * scalar + c */
  @cudaOperator
  def fmaScalar(a: Ptr[Float], scalar: Float, c: Ptr[Float], out: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) out(i) = a(i) * scalar + c(i)
  }

  /** Square: c = a * a */
  @cudaOperator
  def square(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = v * v
    }
  }

  /** Cube: c = a * a * a */
  @cudaOperator
  def cube(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = v * v * v
    }
  }

  /** Floor: c = floor(a) */
  @cudaOperator
  def floor(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.floor(a(i)).toFloat
  }

  /** Ceiling: c = ceil(a) */
  @cudaOperator
  def ceil(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.ceil(a(i)).toFloat
  }

  /** Truncate: c = trunc(a) */
  @cudaOperator
  def trunc(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i).toInt.toFloat
  }

  /** Round: c = round(a) */
  @cudaOperator
  def round(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.round(a(i)).toFloat
  }
}

/** Math function operators ( transcendental functions ) */
object MathOps {

  /** Square root: c = sqrt(a) */
  @cudaOperator
  def sqrt(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.sqrt(a(i)).toFloat
  }

  /** Reciprocal square root: c = rsqrt(a) */
  @cudaOperator
  def rsqrt(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = 1.0f / scala.math.sqrt(a(i)).toFloat
  }

  /** Cube root: c = cbrt(a) */
  @cudaOperator
  def cbrt(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.cbrt(a(i)).toFloat
  }

  /** Exponential: c = exp(a) */
  @cudaOperator
  def exp(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.exp(a(i)).toFloat
  }

  /** Base-2 exponential: c = exp2(a) */
  @cudaOperator
  def exp2(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.pow(2, a(i)).toFloat
  }

  /** Base-10 exponential: c = exp10(a) */
  @cudaOperator
  def exp10(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.pow(10, a(i)).toFloat
  }

  /** Natural logarithm: c = log(a) */
  @cudaOperator
  def log(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n && a(i) > 0) c(i) = scala.math.log(a(i)).toFloat
  }

  /** Base-2 logarithm: c = log2(a) */
  @cudaOperator
  def log2(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n && a(i) > 0) c(i) = scala.math.log(a(i) / scala.math.log(2).toFloat).toFloat
  }

  /** Base-10 logarithm: c = log10(a) */
  @cudaOperator
  def log10(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n && a(i) > 0) c(i) = scala.math.log10(a(i)).toFloat
  }

  /** Sine: c = sin(a) */
  @cudaOperator
  def sin(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.sin(a(i)).toFloat
  }

  /** Cosine: c = cos(a) */
  @cudaOperator
  def cos(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.cos(a(i)).toFloat
  }

  /** Tangent: c = tan(a) */
  @cudaOperator
  def tan(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.tan(a(i)).toFloat
  }

  /** Arc sine: c = asin(a) */
  @cudaOperator
  def asin(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n && a(i) >= -1.0f && a(i) <= 1.0f) c(i) = scala.math.asin(a(i)).toFloat
  }

  /** Arc cosine: c = acos(a) */
  @cudaOperator
  def acos(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n && a(i) >= -1.0f && a(i) <= 1.0f) c(i) = scala.math.acos(a(i)).toFloat
  }

  /** Arc tangent: c = atan(a) */
  @cudaOperator
  def atan(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.atan(a(i)).toFloat
  }

  /** Arc tangent2: c = atan2(a, b) */
  @cudaOperator
  def atan2(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.atan2(a(i), b(i)).toFloat
  }

  /** Hyperbolic sine: c = sinh(a) */
  @cudaOperator
  def sinh(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.sinh(a(i)).toFloat
  }

  /** Hyperbolic cosine: c = cosh(a) */
  @cudaOperator
  def cosh(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.cosh(a(i)).toFloat
  }

  /** Hyperbolic tangent: c = tanh(a) */
  @cudaOperator
  def tanh(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.tanh(a(i)).toFloat
  }

  /** Arc hyperbolic sine: c = asinh(a) */
  @cudaOperator
  def asinh(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val x = a(i)
      c(i) = scala.math.log(x + scala.math.sqrt(x * x + 1.0f)).toFloat
    }
  }

  /** Arc hyperbolic cosine: c = acosh(a) */
  @cudaOperator
  def acosh(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n && a(i) >= 1.0f) {
      val x = a(i)
      c(i) = scala.math.log(x + scala.math.sqrt(x * x - 1.0f)).toFloat
    }
  }

  /** Arc hyperbolic tangent: c = atanh(a) */
  @cudaOperator
  def atanh(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n && a(i) > -1.0f && a(i) < 1.0f) {
      val x = a(i)
      c(i) = (0.5f * scala.math.log((1.0f + x) / (1.0f - x))).toFloat
    }
  }

  /** Power function: c = pow(a, b) */
  @cudaOperator
  def pow(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.pow(a(i), b(i)).toFloat
  }

  /** Power with scalar: c = pow(a, scalar) */
  @cudaOperator
  def powScalar(a: Ptr[Float], scalar: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.pow(a(i), scalar).toFloat
  }

  /** Floating-point remainder: c = fmod(a, b) */
  @cudaOperator
  def fmod(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) % b(i)
  }

  /** Copy sign: c = copysign(a, b) */
  @cudaOperator
  def copysign(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val sign = if (b(i) < 0) -1.0f else 1.0f
      c(i) = scala.math.abs(a(i)) * sign
    }
  }

  /** Fractional part: c = fract(a) = a - floor(a) */
  @cudaOperator
  def fract(a: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = v - scala.math.floor(v).toFloat
    }
  }
}

/** Comparison operators */
object ComparisonOps {

  /** Less than: c = a < b */
  @cudaOperator
  def lt(a: Ptr[Float], b: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) < b(i)
  }

  /** Less than or equal: c = a <= b */
  @cudaOperator
  def le(a: Ptr[Float], b: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) <= b(i)
  }

  /** Greater than: c = a > b */
  @cudaOperator
  def gt(a: Ptr[Float], b: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) > b(i)
  }

  /** Greater than or equal: c = a >= b */
  @cudaOperator
  def ge(a: Ptr[Float], b: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) >= b(i)
  }

  /** Equal: c = a == b */
  @cudaOperator
  def eq(a: Ptr[Float], b: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) == b(i)
  }

  /** Not equal: c = a != b */
  @cudaOperator
  def ne(a: Ptr[Float], b: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) != b(i)
  }

  /** Maximum: c = max(a, b) */
  @cudaOperator
  def max(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val av = a(i)
      val bv = b(i)
      c(i) = if (av > bv) av else bv
    }
  }

  /** Minimum: c = min(a, b) */
  @cudaOperator
  def min(a: Ptr[Float], b: Ptr[Float], c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val av = a(i)
      val bv = b(i)
      c(i) = if (av < bv) av else bv
    }
  }

  /** Clamp: c = clamp(a, min_val, max_val) */
  @cudaOperator
  def clamp(a: Ptr[Float], minVal: Float, maxVal: Float, c: Ptr[Float], n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val v = a(i)
      c(i) = if (v < minVal) minVal else if (v > maxVal) maxVal else v
    }
  }

  /** Is NaN: c = isnan(a) */
  @cudaOperator
  def isnan(a: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i).isNaN
  }

  /** Is infinity: c = isinf(a) */
  @cudaOperator
  def isinf(a: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i).isInfinite
  }

  /** Is finite: c = isfinite(a) */
  @cudaOperator
  def isfinite(a: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = !a(i).isNaN && !a(i).isInfinite
  }

  /** Sign bit: c = signbit(a) */
  @cudaOperator
  def signbit(a: Ptr[Float], c: Ptr[Bool], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) < 0.0f
  }
}

/** Double-precision versions of basic operators */
object DoubleOps {

  @cudaOperator
  def add(a: Ptr[Double], b: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) + b(i)
  }

  @cudaOperator
  def sub(a: Ptr[Double], b: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) - b(i)
  }

  @cudaOperator
  def mul(a: Ptr[Double], b: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) * b(i)
  }

  @cudaOperator
  def div(a: Ptr[Double], b: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) / b(i)
  }

  @cudaOperator
  def sqrt(a: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.sqrt(a(i))
  }

  @cudaOperator
  def exp(a: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.exp(a(i))
  }

  @cudaOperator
  def log(a: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.log(a(i))
  }

  @cudaOperator
  def sin(a: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.sin(a(i))
  }

  @cudaOperator
  def cos(a: Ptr[Double], c: Ptr[Double], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = scala.math.cos(a(i))
  }
}

/** Integer arithmetic operators */
object IntOps {

  @cudaOperator
  def add(a: Ptr[Int], b: Ptr[Int], c: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) + b(i)
  }

  @cudaOperator
  def sub(a: Ptr[Int], b: Ptr[Int], c: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) - b(i)
  }

  @cudaOperator
  def mul(a: Ptr[Int], b: Ptr[Int], c: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = a(i) * b(i)
  }

  @cudaOperator
  def div(a: Ptr[Int], b: Ptr[Int], c: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n && b(i) != 0) c(i) = a(i) / b(i)
  }

  @cudaOperator
  def mod(a: Ptr[Int], b: Ptr[Int], c: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n && b(i) != 0) c(i) = a(i) % b(i)
  }

  @cudaOperator
  def max(a: Ptr[Int], b: Ptr[Int], c: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = if (a(i) > b(i)) a(i) else b(i)
  }

  @cudaOperator
  def min(a: Ptr[Int], b: Ptr[Int], c: Ptr[Int], n: Int): Unit = {
    val i = programId(0)
    if (i < n) c(i) = if (a(i) < b(i)) a(i) else b(i)
  }
}
