import jdk.incubator.vector.{FloatVector, VectorOperators, VectorSpecies}

/** Java Vector API benchmark implementation
 *  Uses SIMD instructions via jdk.incubator.vector for acceleration
 */
object VectorAPIBenchmark {

  // Get the preferred vector species - let type be inferred
  private val SPECIES = FloatVector.SPECIES_PREFERRED

  // Work with primitive float arrays (scala.Float is java.lang.Float boxed)
  def flashAttention(q: Array[Float], k: Array[Float], v: Array[Float],
                     B: Int, H: Int, N: Int, d: Int): Array[Float] = {
    val Dh = math.sqrt(d.toFloat).toFloat
    val output = new Array[Float](B * H * N * d)
    val vecLen = SPECIES.length()

    for (b <- 0 until B; h <- 0 until H) {
      for (i <- 0 until N) {
        // First pass: compute max using SIMD
        var maxVal = Float.NegativeInfinity
        var j = 0
        val limit = N - vecLen

        while (j <= limit) {
          val qIdx = (((b * H + h) * N) + i) * d
          val kIdx = (((b * H + h) * N) + j) * d

          // Vectorized dot product using fromArray with offset only
          var sum = FloatVector.zero(SPECIES)
          var d_ = 0
          while (d_ < d) {
            val qv = FloatVector.fromArray(SPECIES, q, qIdx + d_)
            val kv = FloatVector.fromArray(SPECIES, k, kIdx + d_)
            sum = sum.add(qv.mul(kv))
            d_ += vecLen
          }
          val dotProd = sum.reduceLanes(VectorOperators.ADD)
          val score = dotProd / Dh
          if (score > maxVal) maxVal = score
          j += vecLen
        }

        // Handle remainder
        while (j < N) {
          var dotProd = 0f
          var d_ = 0
          val qIdx = (((b * H + h) * N) + i) * d
          val kIdx = (((b * H + h) * N) + j) * d
          while (d_ < d) {
            dotProd += q(qIdx + d_) * k(kIdx + d_)
            d_ += 1
          }
          val score = dotProd / Dh
          if (score > maxVal) maxVal = score
          j += 1
        }

        // Second pass: compute sumExp
        var sumExp = 0f
        j = 0
        while (j <= limit) {
          val qIdx = (((b * H + h) * N) + i) * d
          val kIdx = (((b * H + h) * N) + j) * d

          var sum = FloatVector.zero(SPECIES)
          var d_ = 0
          while (d_ < d) {
            val qv = FloatVector.fromArray(SPECIES, q, qIdx + d_)
            val kv = FloatVector.fromArray(SPECIES, k, kIdx + d_)
            sum = sum.add(qv.mul(kv))
            d_ += vecLen
          }
          val dotProd = sum.reduceLanes(VectorOperators.ADD)
          val score = dotProd / Dh
          sumExp += math.exp((score - maxVal).toDouble).toFloat
          j += vecLen
        }

        while (j < N) {
          var dotProd = 0f
          var d_ = 0
          val qIdx = (((b * H + h) * N) + i) * d
          val kIdx = (((b * H + h) * N) + j) * d
          while (d_ < d) {
            dotProd += q(qIdx + d_) * k(kIdx + d_)
            d_ += 1
          }
          val score = dotProd / Dh
          sumExp += math.exp((score - maxVal).toDouble).toFloat
          j += 1
        }

        // Third pass: compute output
        for (d_ <- 0 until d) {
          var result = 0f
          j = 0
          while (j <= limit) {
            val qIdx = (((b * H + h) * N) + i) * d
            val kIdx = (((b * H + h) * N) + j) * d
            val vIdx = (((b * H + h) * N) + j) * d

            var sum = FloatVector.zero(SPECIES)
            var dd = 0
            while (dd < d) {
              val qv = FloatVector.fromArray(SPECIES, q, qIdx + dd)
              val kv = FloatVector.fromArray(SPECIES, k, kIdx + dd)
              sum = sum.add(qv.mul(kv))
              dd += vecLen
            }
            val dotProd = sum.reduceLanes(VectorOperators.ADD)
            val score = dotProd / Dh
            val weight = math.exp((score - maxVal).toDouble).toFloat / sumExp
            result += weight * v(vIdx + d_)
            j += vecLen
          }

          while (j < N) {
            var dotProd = 0f
            var dd = 0
            val qIdx = (((b * H + h) * N) + i) * d
            val kIdx = (((b * H + h) * N) + j) * d
            val vIdx = (((b * H + h) * N) + j) * d
            while (dd < d) {
              dotProd += q(qIdx + dd) * k(kIdx + dd)
              dd += 1
            }
            val score = dotProd / Dh
            val weight = math.exp((score - maxVal).toDouble).toFloat / sumExp
            result += weight * v(vIdx + d_)
            j += 1
          }

          output((((b * H + h) * N) + i) * d + d_) = result
        }
      }
    }
    output
  }

  def pageAttention(q: Array[Float], k: Array[Float], v: Array[Float],
                   B: Int, H: Int, N: Int, d: Int): Array[Float] = {
    flashAttention(q, k, v, B, H, N, d)
  }

  def flexAttention(q: Array[Float], k: Array[Float], v: Array[Float],
                    B: Int, H: Int, N: Int, d: Int): Array[Float] = {
    val Dh = math.sqrt(d.toFloat).toFloat
    val output = new Array[Float](B * H * N * d)

    for (b <- 0 until B; h <- 0 until H) {
      for (i <- 0 until N) {
        var blockMax = Float.NegativeInfinity
        for (j <- 0 until N) {
          var dotProd = 0f
          var d_ = 0
          val qIdx = (((b * H + h) * N) + i) * d
          val kIdx = (((b * H + h) * N) + j) * d
          while (d_ < d) {
            dotProd += q(qIdx + d_) * k(kIdx + d_)
            d_ += 1
          }
          val score = math.max(-50f, math.min(50f, dotProd / Dh))
          if (score > blockMax) blockMax = score
        }

        var blockSum = 0f
        for (j <- 0 until N) {
          var dotProd = 0f
          var d_ = 0
          val qIdx = (((b * H + h) * N) + i) * d
          val kIdx = (((b * H + h) * N) + j) * d
          while (d_ < d) {
            dotProd += q(qIdx + d_) * k(kIdx + d_)
            d_ += 1
          }
          val score = math.max(-50f, math.min(50f, dotProd / Dh))
          blockSum += math.exp((score - blockMax).toDouble).toFloat
        }

        for (d_ <- 0 until d) {
          var result = 0f
          for (j <- 0 until N) {
            var dotProd = 0f
            var dd = 0
            val qIdx = (((b * H + h) * N) + i) * d
            val kIdx = (((b * H + h) * N) + j) * d
            val vIdx = (((b * H + h) * N) + j) * d
            while (dd < d) {
              dotProd += q(qIdx + dd) * k(kIdx + dd)
              dd += 1
            }
            val score = math.max(-50f, math.min(50f, dotProd / Dh))
            val weight = math.exp((score - blockMax).toDouble).toFloat / blockSum
            result += weight * v(vIdx + d_)
          }
          output((((b * H + h) * N) + i) * d + d_) = result
        }
      }
    }
    output
  }
}
