package cuda.dsl.operators.special

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Special mathematical functions.
 *  Includes error function, Bessel functions, gamma functions, etc.
 */
object SpecialOps {

  /** Error function (erf) - Taylor series approximation */
  @cudaOperator
  def erf(x: Float): Float = {
    // Approximation formula
    val t = 1.0f / (1.0f + 0.5f * scala.math.abs(x).toFloat)
    val tau = t * scala.math.exp(-x * x - 1.26551223 +
      t * 1.00002368 +
      t * t * 0.37409196 +
      t * t * t * 0.09678418 +
      t * t * t * t * (-0.15195108) +
      t * t * t * t * t * 0.21047258 +
      t * t * t * t * t * t * (-0.13908744) +
      t * t * t * t * t * t * t * 0.0232235
    ).toFloat

    val sign = if (x >= 0) 1.0f else -1.0f
    sign * (1.0f - tau)
  }

  /** Complementary error function (erfc) */
  @cudaOperator
  def erfc(x: Float): Float = {
    1.0f - erf(x)
  }

  /** Inverse error function (approximation) */
  @cudaOperator
  def erfinv(x: Float): Float = {
    val w = -scala.math.log((1.0f - x) * (1.0f + x)).toFloat
    val p = if (w < 5.0f) {
      w - 2.5f
    } else {
      w - 3.0f
    }

    val a = 0.886226899f + p * (-1.64535962f + p * 0.07934044f)
    val b = 1.0f + p * (-1.13438333f + p * 0.09591176f)

    x * (a / b)
  }

  /** Gamma function (Lanczos approximation) */
  @cudaOperator
  def tgamma(x: Float): Float = {
    if (x <= 0 && x == x.floor) return Float.NaN

    val g = 4.7421875f
    val p = Array(
      0.99999999999980993f,
      676.5203681218851f,
      -1259.1392167224028f,
      771.32342877765313f,
      -176.61502916214059f,
      12.507343278686905f,
      -0.13857109526572012f,
      9.9843695780195716e-6f,
      1.5056327351493116e-7f
    )

    if (x < 0.5f) {
      scala.math.Pi.toFloat / (scala.math.sin(scala.math.Pi * x).toFloat * tgamma(1.0f - x))
    } else {
      val x1 = x - 1.0f
      var A = p(0)
      val T = x1 + g + 0.5f

      for (i <- 1 until 9) {
        A += p(i) / (x1 + i)
      }

      scala.math.sqrt(2 * scala.math.Pi).toFloat * scala.math.pow(T, x1 + 0.5f).toFloat * scala.math.exp(-T).toFloat * A
    }
  }

  /** Log gamma function */
  @cudaOperator
  def lgamma(x: Float): Float = {
    scala.math.log(scala.math.abs(tgamma(x))).toFloat
  }

  /** Digamma function (psi) */
  @cudaOperator
  def digamma(x: Float): Float = {
    // Approximation using series expansion
    val small = 1.0e-6f
    val pi = scala.math.Pi.toFloat

    if (x < 0.0f) {
      -digamma(1.0f - x) - pi / scala.math.tan(pi * x).toFloat
    } else if (x < small) {
      -scala.math.log(x).toFloat - 0.5772156649f
    } else {
      var sum = -scala.math.log(x).toFloat - 0.5772156649f
      var term = x * x
      var k = 1.0f

      while (k <= 100) {
        sum += term / k - term / (k + 1.0f)
        k += 1.0f
        term *= x * x
      }
      sum
    }
  }

  /** Trigamma function */
  @cudaOperator
  def trigamma(x: Float): Float = {
    val small = 1.0e-6f

    if (x < small) {
      1.0f / (x * x) + scala.math.Pi.toFloat * scala.math.Pi.toFloat / 6.0f
    } else {
      var sum = 1.0f / (x * x)
      var term = x * x
      var k = 1.0f

      while (k <= 100) {
        term *= x * x
        sum += 1.0f / ((k + x) * (k + x))
        k += 1.0f
      }
      1.0f / x + sum / 2.0f
    }
  }
}

/** Bessel functions */
object BesselOps {

  /** Bessel J0 (first kind, order 0) */
  @cudaOperator
  def j0(x: Float): Float = {
    val ax = scala.math.abs(x).toFloat
    var result: Float = 0

    if (ax < 8.0f) {
      val y = x * x
      val ans1 = 57568490574.0f + y * (-13362590354.0f + y * (651619640.7f +
        y * (-11214424.18f + y * (77392.33017f + y * (-184.9052456f)))))
      val ans2 = 57568490411.0f + y * (1029532985.0f + y * (9494680.718f +
        y * (59272.64853f + y * (267.8532712f + y * 1.0f))))

      result = ans1 / ans2
    } else {
      val z = 8.0f / ax
      val y = z * z
      val xx = ax - 0.785398164f

      val ans1 = 1.0f + y * (-0.1098628627e-2f + y * (0.2734510407e-4f +
        y * (-0.2073370639e-5f + y * 0.2093887211e-6f)))
      val ans2 = -0.1562499995e-1f + y * (0.1430488765e-3f +
        y * (-0.6911147651e-5f + y * (0.7621095161e-6f - y * 0.934935152e-7f)))

      result = scala.math.sqrt(0.636619772f / ax).toFloat *
        (scala.math.cos(xx).toFloat * ans1 - z * scala.math.sin(xx).toFloat * ans2)
    }
    result
  }

  /** Bessel J1 (first kind, order 1) */
  @cudaOperator
  def j1(x: Float): Float = {
    val ax = scala.math.abs(x).toFloat
    var result: Float = 0

    if (ax < 8.0f) {
      val y = x * x
      val ans1 = x * (72362614232.0f + y * (-7895059235.0f + y * (242396853.1f +
        y * (2972611.439f + y * (15704.48260f + y * (30.16036606f))))))
      val ans2 = 144725228442.0f + y * (2300535178.0f + y * (18583304.74f +
        y * (99447.43394f + y * (376.9991397f + y * 1.0f))))

      result = ans1 / ans2
    } else {
      val z = 8.0f / ax
      val y = z * z
      val xx = ax - 2.356194491f

      val ans1 = 1.0f + y * (0.183105e-2f + y * (-0.3516396496e-4f +
        y * (0.2457520174e-5f + y * (-0.240337019e-6f))))
      val ans2 = 0.04687499995f + y * (-0.2002690873e-3f +
        y * (0.8449199096e-5f + y * (-0.88228987e-6f + y * 0.105787412e-6f)))

      result = scala.math.sqrt(0.636619772f / ax).toFloat *
        (scala.math.cos(xx).toFloat * ans1 - z * scala.math.sin(xx).toFloat * ans2)

      if (x < 0.0f) result = -result
    }
    result
  }

  /** Bessel Jn (first kind, order n) */
  @cudaOperator
  def jn(n: Int, x: Float): Float = {
    if (n == 0) j0(x)
    else if (n == 1) j1(x)
    else if (x == 0.0f) 0.0f
    else {
      var jsum = false
      val ax = scala.math.abs(x).toFloat

      if (ax > n.toFloat) {
        var bjm = j0(ax)
        var bj = j1(ax)
        var j = 1

        while (j < n) {
          val bjm1 = bjm
          bjm = bj
          bj = (2 * j / ax) * bj - bjm1
          j += 1
        }
        if (x < 0.0f && (n & 1) == 1) bj = -bj
        bj
      } else {
        val tox = 2.0f / ax
        val m = 2 * ((n + scala.math.sqrt(n.toFloat * 40.0f).toInt) / 2)
        var j = m

        var bjm = 0.0f
        var bj = 1.0f
        var bjp = 0.0f
        var sum = 0.0f

        while (j > 0) {
          bjp = (2 * j / tox) * bj - bjm
          bjm = bj
          bj = bjp
          if (scala.math.abs(bj) > 1.0e10f) {
            bj *= 1.0e-10f
            bjm *= 1.0e-10f
            sum *= 1.0e-10f
          }
          if (jsum) sum += bj
          j -= 1
        }
        sum *= 2.0f

        jsum = !jsum
        if (jsum) sum += bj

        val result = bjm / sum
        if (x < 0.0f && (n & 1) == 1) -result else result
      }
    }
  }

  /** Bessel Y0 (second kind, order 0) */
  @cudaOperator
  def y0(x: Float): Float = {
    if (x < 8.0f) {
      val y = x * x
      val ans1 = -2957821389.0f + y * (7062834065.0f + y * (-512359803.6f +
        y * (10879881.29f + y * (86327.92757f))))
      val ans2 = 40076544269.0f + y * (745249964.8f + y * (7189466.438f +
        y * (47447.26470f + y * (226.1030244f + y * 1.0f))))

      (ans1 / ans2) + 0.636619772f * j0(x) * scala.math.log(x).toFloat
    } else {
      val z = 8.0f / x
      val y = z * z
      val xx = x - 0.785398164f

      val ans1 = 1.0f + y * (-0.1098628627e-2f + y * (0.2734510407e-4f +
        y * (-0.2073370639e-5f + y * 0.2093887211e-6f)))
      val ans2 = -0.1562499995e-1f + y * (0.1430488765e-3f +
        y * (-0.6911147651e-5f + y * (0.7621095161e-6f + y * (-0.934945152e-7f))))

      scala.math.sqrt(0.636619772f / x).toFloat *
        (scala.math.sin(xx).toFloat * ans1 + z * scala.math.cos(xx).toFloat * ans2)
    }
  }

  /** Bessel Y1 (second kind, order 1) */
  @cudaOperator
  def y1(x: Float): Float = {
    if (x < 8.0f) {
      val y = x * x
      val ans1 = x * (-0.4900604943e13f + y * (0.1275274390e13f +
        y * (-0.5153438139e11f + y * (0.7349264551e9f + y * (-0.3787705362e7f)))))
      val ans2 = -0.2495804571e14f + y * (0.4244819660e13f +
        y * (-0.3653699889e11f + y * (0.2305535170e9f + y * (-0.6544550693e7f + y * 1.0f))))

      (ans1 / ans2) + 0.636619772f * (j1(x) * scala.math.log(x).toFloat - 1.0f / x)
    } else {
      val z = 8.0f / x
      val y = z * z
      val xx = x - 2.356194491f

      val ans1 = 1.0f + y * (0.183105e-2f + y * (-0.3516396496e-4f +
        y * (0.2457520174e-5f + y * (-0.240337019e-6f))))
      val ans2 = 0.04687499995f + y * (-0.2002690873e-3f +
        y * (0.8449199096e-5f + y * (-0.88228987e-6f + y * 0.105787412e-6f)))

      scala.math.sqrt(0.636619772f / x).toFloat *
        (scala.math.sin(xx).toFloat * ans1 + z * scala.math.cos(xx).toFloat * ans2)
    }
  }

  /** Bessel Yn (second kind, order n) */
  @cudaOperator
  def yn(n: Int, x: Float): Float = {
    if (n == 0) y0(x)
    else if (n == 1) y1(x)
    else {
      var bym = y0(x)
      var by = y1(x)
      var j = 1

      while (j < n) {
        val byp = (2 * j / x) * by - bym
        bym = by
        by = byp
        j += 1
      }
      by
    }
  }
}

/** Sigmoid and related functions */
object LogisticOps {

  /** Logit function (inverse sigmoid) */
  @cudaOperator
  def logit(p: Float): Float = {
    scala.math.log(p / (1.0f - p)).toFloat
  }

  /** Log log sigmoid */
  @cudaOperator
  def logLogSigmoid(x: Float): Float = {
    if (x >= 0) {
      -x - scala.math.log(1.0f + scala.math.exp(-x)).toFloat
    } else {
      -scala.math.log(1.0f + scala.math.exp(x)).toFloat
    }
  }

  /** Hard sigmoid */
  @cudaOperator
  def hardSigmoid(x: Float): Float = {
    (x + 3.0f) / 6.0f match
      case v if v < 0 => 0.0f
      case v if v > 1 => 1.0f
      case v => v
  }

  /** Soft sign */
  @cudaOperator
  def softSign(x: Float): Float = {
    x / (1.0f + scala.math.abs(x)).toFloat
  }
}
