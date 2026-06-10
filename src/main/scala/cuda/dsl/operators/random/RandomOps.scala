package cuda.dsl.operators.random

import cuda.dsl.core.*
import cuda.dsl.core.Types.{Bool, given_MemoryOps_Float, given_MemoryOps_Double, given_MemoryOps_Int, given_MemoryOps_Long, given_MemoryOps_Bool}
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.DSL.programId

/** Random number generation operations.
 *  Provides various probability distributions for GPU computing.
 */
object RandomOps {

  /** Linear congruential generator (LCG) state */
  class LCGState(seed: Long) {
    var state = seed

    def next(): Long = {
      state = (state * 1103515245 + 12345) & 0x7fffffff
      state
    }

    def nextFloat(): Float = next() / 0x7fffffff.toFloat
    def nextDouble(): Double = next() / 0x7fffffff.toDouble
  }

  /** XORWOW state for CUDA-style random */
  class XORWOWState(var x: Int, var y: Int, var z: Int, var w: Int, var v: Int, var d: Int) {
    def next(): Long = {
      val t = ((x ^ (x >>> 2)) ^ ((y ^ (y << 1)) << 1)) ^ ((v ^ (v << 1)) >>> 3) ^ (w >>> 3)
      x = y; y = z; z = w; w = v; v = t; d = (d + 362437) & 0x7fffffff
      (d.toLong << 32) | (t & 0xffffffffL)
    }
  }

  /** Mersenne Twister state (simplified) */
  class MTState(seed: Int) {
    private val N = 624
    private val state = new Array[Int](N)
    private var idx = N

    state(0) = seed
    for (i <- 1 until N) {
      state(i) = (1812433253 * (state(i - 1) ^ (state(i - 1) >>> 30)) + i) & 0xffffffff
    }

    def twist(): Unit = {
      for (i <- 0 until N) {
        val bits = (state(i) & 0x80000000) | (state((i + 1) % N) & 0x7fffffff)
        val out = state((i + 397) % N) ^ (bits >>> 1)
        state(i) = out ^ ((bits & 1) * 0x9908b0df)
      }
      idx = 0
    }

    def nextInt(): Int = {
      if (idx >= N) twist()
      var y = state(idx)
      y ^= (y >>> 11)
      y ^= ((y << 7) & 0x9d2c5680)
      y ^= ((y << 15) & 0xefc60000)
      y ^= (y >>> 18)
      idx += 1
      y
    }

    def nextFloat(): Float = (nextInt() >>> 8) / 16777216.0f
  }

  /** Uniform random [0, 1) */
  @cudaOperator
  def rand(seed: Int, n: Int): Float = {
    val lcg = new LCGState(seed + programId(0))
    lcg.nextFloat()
  }

  /** Normal distribution (Box-Muller) */
  @cudaOperator
  def randn(seed: Int, n: Int): Float = {
    val lcg1 = new LCGState(seed + programId(0))
    val lcg2 = new LCGState(seed + programId(0) + 12345)

    val u1 = lcg1.nextFloat()
    val u2 = lcg2.nextFloat()

    val z0 = scala.math.sqrt(-2.0 * scala.math.log(u1)).toFloat * scala.math.cos(2 * scala.math.Pi * u2).toFloat
    z0
  }

  /** Uniform integers [low, high) */
  @cudaOperator
  def randInt(seed: Int, low: Int, high: Int, n: Int): Int = {
    val lcg = new LCGState(seed + programId(0))
    val range = high - low
    low + (lcg.next() % range).toInt
  }

  /** Bernoulli distribution */
  @cudaOperator
  def randBernoulli(seed: Int, p: Float, n: Int): Bool = {
    val lcg = new LCGState(seed + programId(0))
    lcg.nextFloat() < p
  }

  /** Exponential distribution */
  @cudaOperator
  def randExp(seed: Int, lambda: Float, n: Int): Float = {
    val lcg = new LCGState(seed + programId(0))
    val u = lcg.nextFloat()
    -scala.math.log(1 - u).toFloat / lambda
  }

  /** Gamma distribution (shape, scale) */
  @cudaOperator
  def randGamma(seed: Int, shape: Float, scale: Float, n: Int): Float = {
    // Marsaglia and Tsang's method for gamma distribution
    val lcg = new LCGState(seed + programId(0))

    if (shape >= 1.0f) {
      val d = shape - 1.0f / 3.0f
      val c = 1.0f / scala.math.sqrt(9.0f * d).toFloat
      var x: Float = 0
      var v: Float = 0
      var valid = false

      while (!valid) {
        // Generate x and v using Box-Muller
        x = randn(seed + programId(0), 1)
        v = 1.0f + c * x

        if (v > 0) {
          v = v * v * v
          val u = lcg.nextFloat()
          valid = u < 1 - 0.0331f * (x * x) * (x * x)
        }
      }
      d * v * scale
    } else {
      // For shape < 1, use shape + 1
      randGamma(seed, shape + 1.0f, scale, n) * scala.math.pow(lcg.nextFloat(), 1.0f / shape).toFloat
    }
  }

  /** Beta distribution */
  @cudaOperator
  def randBeta(seed: Int, alpha: Float, beta: Float, n: Int): Float = {
    val gamma1 = randGamma(seed, alpha, 1.0f, n)
    val gamma2 = randGamma(seed + 12345, beta, 1.0f, n)
    gamma1 / (gamma1 + gamma2)
  }

  /** Poisson distribution */
  @cudaOperator
  def randPoisson(seed: Int, lambda: Float, n: Int): Int = {
    val lcg = new LCGState(seed + programId(0))
    val L = scala.math.exp(-lambda).toFloat
    var k = 0
    var p = 1.0f

    while (p > L) {
      k += 1
      p *= lcg.nextFloat()
    }
    k - 1
  }

  /** Geometric distribution */
  @cudaOperator
  def randGeometric(seed: Int, p: Float, n: Int): Int = {
    val lcg = new LCGState(seed + programId(0))
    val u = lcg.nextFloat()
    (scala.math.log(1 - u) / scala.math.log(1 - p)).toInt + 1
  }
}

/** Random fill operations */
object RandomFillOps {

  /** Fill with uniform random values */
  @cudaOperator
  def fillRand(output: Ptr[Float], seed: Int, n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val lcg = new RandomOps.LCGState(seed + i)
      output(i) = lcg.nextFloat()
    }
  }

  /** Fill with normal random values */
  @cudaOperator
  def fillRandn(output: Ptr[Float], seed: Int, n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      output(i) = RandomOps.randn(seed, n)
    }
  }

  /** Fill with random integers in range */
  @cudaOperator
  def fillRandInt(output: Ptr[Int], seed: Int, low: Int, high: Int, n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val lcg = new RandomOps.LCGState(seed + i)
      output(i) = low + (lcg.next() % (high - low)).toInt
    }
  }

  /** Fill with Bernoulli random values */
  @cudaOperator
  def fillRandBernoulli(output: Ptr[Bool], seed: Int, p: Float, n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val lcg = new RandomOps.LCGState(seed + i)
      output(i) = lcg.nextFloat() < p
    }
  }

  /** Random dropout mask */
  @cudaOperator
  def dropoutMask(output: Ptr[Float], rate: Float, scale: Float, seed: Int, n: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      val lcg = new RandomOps.LCGState(seed + i)
      output(i) = if (lcg.nextFloat() > rate) scale else 0.0f
    }
  }

  /** Random permutation */
  @cudaOperator
  def randomPermutation(output: Ptr[Int], n: Int, seed: Int): Unit = {
    val i = programId(0)
    if (i < n) {
      // Fisher-Yates shuffle (simplified)
      val lcg = new RandomOps.LCGState(seed + i)
      output(i) = i
    }

    // Perform swaps
    for (i <- n - 1 to 1 by -1) {
      val lcg = new RandomOps.LCGState(seed + i)
      val j = (lcg.nextFloat() * (i + 1)).toInt
      val temp = output(i)
      output(i) = output(j)
      output(j) = temp
    }
  }

  /** Random shuffle of array indices */
  @cudaOperator
  def shuffleIndices(indices: Ptr[Int], n: Int, seed: Int): Unit = {
    val lcg = new RandomOps.LCGState(seed)
    for (i <- n - 1 to 1 by -1) {
      val j = (lcg.nextFloat() * (i + 1)).toInt
      val temp = indices(i)
      indices(i) = indices(j)
      indices(j) = temp
    }
  }
}

/** Hash functions for random-like operations */
object HashOps {

  /** Jenkin's hash */
  @cudaOperator
  def jenkinsHash(input: Ptr[Int], n: Int): Int = {
    var hash = 0
    for (i <- 0 until n) {
      hash += input(i)
      hash = (hash + (hash << 10)) ^ (hash >>> 6)
    }
    hash = (hash + (hash << 3)) ^ (hash >>> 11)
    hash = (hash + (hash << 15))
    hash
  }

  /** MurmurHash (simplified) */
  @cudaOperator
  def murmurHash(key: Int, seed: Int): Int = {
    var h = seed
    val k = key
    h ^= k
    h ^= (k >>> 16)
    h *= 0x85ebca6b
    h ^= (h >>> 13)
    h *= 0xc2b2ae35
    h ^= (h >>> 16)
    h
  }

  /** Simple hash for floats */
  @cudaOperator
  def hashFloat(value: Float, seed: Int): Int = {
    murmurHash(java.lang.Float.floatToIntBits(value), seed)
  }
}
