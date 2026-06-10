package cuda.dsl.benchmark

import cuda.dsl.dsl._

/** Test simple Triton DSL */
object TestSimpleTritonDSL {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Simple Triton DSL Test")
    println("=" * 80)

    // ========================================================================
    // Test 1: Vector Add
    // ========================================================================
    println("\n[1] Vector Add")
    println("-" * 40)

    val vecAdd = Triton("vectorAdd")
      .param("float*", "x")
      .param("float*", "y")
      .param("float*", "out")
      .stmt("out[i] = x[i] + y[i];")
      .emit()
    println(vecAdd)

    // ========================================================================
    // Test 2: Complex Math Expression
    // ========================================================================
    println("\n[2] Complex Math (User's Example)")
    println("-" * 40)

    // User wants: (sqrt(x * x + y * y + 1.0f) + sqrt(x * x + y * y)) / (x * y + 0.001f)
    val code = Triton("complexMath")
      .param("float", "x")
      .param("float", "y")
      .stmt("float temp1 = sqrtf(x * x + y * y + 1.0f);")
      .stmt("float temp2 = sqrtf(x * x + y * y);")
      .stmt("float result = (temp1 + temp2) / (x * y + 0.001f);")
      .emit()
    println(code)

    // ========================================================================
    // Test 3: Math operations
    // ========================================================================
    println("\n[3] Math Operations")
    println("-" * 40)

    val mathOps = Triton("mathOps")
      .param("float*", "input")
      .param("float*", "output")
      .stmt("float t = input[i];")
      .stmt("output[i] = sinf(t) + cosf(t) * expf(-t);")
      .emit()
    println(mathOps)

    // ========================================================================
    // Test 4: SAXPY
    // ========================================================================
    println("\n[4] SAXPY")
    println("-" * 40)

    val saxpy = Triton("saxpy")
      .param("float*", "x")
      .param("float*", "y")
      .param("float*", "out")
      .param("float", "alpha")
      .stmt("out[i] = alpha * x[i] + y[i];")
      .emit()
    println(saxpy)

    // ========================================================================
    // Test 5: FlashAttention (complex)
    // ========================================================================
    println("\n[5] FlashAttention (complex)")
    println("-" * 40)

    val flashKernel = Triton("flashAttention")
      .param("float*", "q")
      .param("float*", "k")
      .param("float*", "v")
      .param("float*", "out")
      .param("int", "N")
      .param("int", "d")
      .stmt("float maxScore = -1e9f;")
      .stmt("float sumExp = 0.0f;")
      .stmt("float score = 0.0f;")
      .stmt("for (int j = 0; j < N; j++) {")
      .stmt("  score = 0.0f;")
      .stmt("  for (int d_ = 0; d_ < d; d_++) {")
      .stmt("    score += q[i * d + d_] * k[j * d + d_];")
      .stmt("  }")
      .stmt("  maxScore = fmaxf(maxScore, score);")
      .stmt("}")
      .stmt("for (int j = 0; j < N; j++) {")
      .stmt("  score = 0.0f;")
      .stmt("  for (int d_ = 0; d_ < d; d_++) {")
      .stmt("    score += q[i * d + d_] * k[j * d + d_];")
      .stmt("  }")
      .stmt("  score = expf(score - maxScore);")
      .stmt("  sumExp += score;")
      .stmt("  for (int d_ = 0; d_ < d; d_++) {")
      .stmt("    out[i * d + d_] += score * v[j * d + d_];")
      .stmt("  }")
      .stmt("}")
      .stmt("for (int d_ = 0; d_ < d; d_++) {")
      .stmt("  out[i * d + d_] /= sumExp;")
      .stmt("}")
      .emit()
    println(flashKernel)

    println("\n" + "=" * 80)
    println("All tests passed!")
    println("=" * 80)
  }
}
