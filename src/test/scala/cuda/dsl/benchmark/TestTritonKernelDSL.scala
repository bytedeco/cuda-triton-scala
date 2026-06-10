package cuda.dsl.benchmark

import cuda.dsl.dsl._
import TTIRDSL._

/** Test TritonKernel high-level DSL for FlashAttention and PageAttention */
object TestTritonKernelDSL {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("TritonKernel DSL - FlashAttention & PageAttention")
    println("=" * 80)

    // ========================================================================
    // FlashAttention using TritonKernel DSL
    // ========================================================================
    println("\n[1] FlashAttention Kernel (TritonKernel DSL)")
    println("-" * 40)

    val N = TTIRVar("N")
    val d = TTIRVar("d")
    val i = TTIRVar("i")
    val stride = d

    val flashKernel = TritonKernel("flashAttention")
      .param("float*", "q")
      .param("float*", "k")
      .param("float*", "v")
      .param("float*", "out")
      .param("int", "N")
      .param("int", "d")
      .local("float", "maxScore", -1e9f)
      .local("float", "sumExp", 0.0f)
      .local("float", "score", 0.0f)

    // First pass: compute max score
    flashKernel += TTIRFor("j", 0, N, List(
      TTIRAssign("score", TTIRConst(0.0f)),
      TTIRFor("d_", 0, d, List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("q", i, TTIRVar("d_"), stride),
            load2D("k", TTIRVar("j"), TTIRVar("d_"), stride)
          )))
      )),
      TTIRIf(TTIRBinOp(">", TTIRVar("score"), TTIRVar("maxScore")),
        List(TTIRAssign("maxScore", TTIRVar("score"))),
        Nil)
    ))

    // Second pass: softmax and accumulate
    flashKernel += TTIRFor("j", 0, N, List(
      TTIRAssign("score", TTIRConst(0.0f)),
      TTIRFor("d_", 0, d, List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("q", i, TTIRVar("d_"), stride),
            load2D("k", TTIRVar("j"), TTIRVar("d_"), stride)
          )))
      )),
      TTIRAssign("score", TTIRMathCall("exp", List(TTIRBinOp("-", TTIRVar("score"), TTIRVar("maxScore"))))),
      TTIRAssign("sumExp", TTIRBinOp("+", TTIRVar("sumExp"), TTIRVar("score"))),
      TTIRFor("d_", 0, d, List(
        TTIRStore2D("out", i, TTIRVar("d_"), stride,
          TTIRBinOp("+",
            load2D("out", i, TTIRVar("d_"), stride),
            TTIRBinOp("*", TTIRVar("score"), load2D("v", TTIRVar("j"), TTIRVar("d_"), stride))
          ), TTIRConst(true))
      ))
    ))

    // Normalize
    flashKernel += TTIRFor("d_", 0, d, List(
      TTIRStore2D("out", i, TTIRVar("d_"), stride,
        TTIRBinOp("/", load2D("out", i, TTIRVar("d_"), stride), TTIRVar("sumExp")),
        TTIRConst(true))
    ))

    println(flashKernel.emit())

    // ========================================================================
    // PageAttention using TritonKernel DSL
    // ========================================================================
    println("\n" + "=" * 80)
    println("[2] PageAttention Kernel (TritonKernel DSL)")
    println("-" * 40)

    val numBlocks = TTIRVar("num_blocks")
    val D = TTIRVar("D")

    val pageKernel = TritonKernel("pageAttention")
      .param("float*", "key_cache")
      .param("float*", "value_cache")
      .param("int*", "block_tables")
      .param("float*", "query")
      .param("float*", "out")
      .param("int", "num_blocks")
      .param("int", "D")
      .local("float", "maxScore", -1e9f)
      .local("float", "sumExp", 0.0f)
      .local("float", "score", 0.0f)

    // First pass: compute max
    pageKernel += TTIRFor("block_idx", 0, numBlocks, List(
      TTIRAssign("score", TTIRConst(0.0f)),
      TTIRFor("d_", 0, D, List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("query", TTIRConst(0), TTIRVar("d_"), D),
            load2D("key_cache", TTIRVar("block_idx"), TTIRVar("d_"), D)
          )))
      )),
      TTIRIf(TTIRBinOp(">", TTIRVar("score"), TTIRVar("maxScore")),
        List(TTIRAssign("maxScore", TTIRVar("score"))),
        Nil)
    ))

    // Second pass: softmax and accumulate
    pageKernel += TTIRFor("block_idx", 0, numBlocks, List(
      TTIRAssign("score", TTIRConst(0.0f)),
      TTIRFor("d_", 0, D, List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("query", TTIRConst(0), TTIRVar("d_"), D),
            load2D("key_cache", TTIRVar("block_idx"), TTIRVar("d_"), D)
          )))
      )),
      TTIRAssign("score", TTIRMathCall("exp", List(TTIRBinOp("-", TTIRVar("score"), TTIRVar("maxScore"))))),
      TTIRAssign("sumExp", TTIRBinOp("+", TTIRVar("sumExp"), TTIRVar("score"))),
      TTIRFor("d_", 0, D, List(
        TTIRStore2D("out", TTIRConst(0), TTIRVar("d_"), D,
          TTIRBinOp("+",
            load2D("out", TTIRConst(0), TTIRVar("d_"), D),
            TTIRBinOp("*", TTIRVar("score"),
              load2D("value_cache", TTIRVar("block_idx"), TTIRVar("d_"), D))
            ), TTIRConst(true))
      ))
    ))

    // Normalize
    pageKernel += TTIRFor("d_", 0, D, List(
      TTIRStore2D("out", TTIRConst(0), TTIRVar("d_"), D,
        TTIRBinOp("/", load2D("out", TTIRConst(0), TTIRVar("d_"), D), TTIRVar("sumExp")),
        TTIRConst(true))
    ))

    println(pageKernel.emit())

    // ========================================================================
    // Simpler Vector Add using TritonKernel DSL
    // ========================================================================
    println("\n" + "=" * 80)
    println("[3] Simple Vector Add (TritonKernel DSL)")
    println("-" * 40)

    val vecAddKernel = TritonKernel("vectorAdd")
      .param("float*", "a")
      .param("float*", "b")
      .param("float*", "out")
      .param("int", "n")

    vecAddKernel += TTIRStore("out", TTIRVar("i"),
      TTIRBinOp("+",
        TTIRLoad("a", TTIRVar("i"), TTIRConst(true), TTIRConst(0.0f)),
        TTIRLoad("b", TTIRVar("i"), TTIRConst(true), TTIRConst(0.0f))
      ), TTIRConst(true))

    println(vecAddKernel.emit())

    // ========================================================================
    // SAXPY using TritonKernel DSL
    // ========================================================================
    println("\n" + "=" * 80)
    println("[4] SAXPY Kernel (TritonKernel DSL)")
    println("-" * 40)

    val saxpyKernel = TritonKernel("saxpy")
      .param("float*", "x")
      .param("float*", "y")
      .param("float*", "out")
      .param("float", "alpha")
      .local("float", "result", 0.0f)

    saxpyKernel += TTIRFor("j", 0, TTIRVar("n"), List(
      TTIRAssign("result", TTIRBinOp("+", TTIRVar("result"),
        TTIRBinOp("*",
          TTIRVar("alpha"),
          TTIRBinOp("+",
            TTIRLoad("x", TTIRVar("j"), TTIRConst(true), TTIRConst(0.0f)),
            TTIRLoad("y", TTIRVar("j"), TTIRConst(true), TTIRConst(0.0f))
          )
        )
      ))
    ))

    saxpyKernel += TTIRStore("out", TTIRVar("i"), TTIRVar("result"), TTIRConst(true))

    println(saxpyKernel.emit())

    println("\n" + "=" * 80)
    println("All kernels generated successfully!")
    println("=" * 80)
  }
}
