package cuda.dsl.benchmark

import cuda.dsl.dsl._
import TTIRDSL._

/** Test FlashAttention and PageAttention kernel generation */
object TestAttentionKernels {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("FlashAttention & PageAttention Kernel Generation")
    println("=" * 80)

    // ========================================================================
    // FlashAttention Kernel
    // ========================================================================
    println("\n[1] FlashAttention Kernel")
    println("-" * 40)

    val N = TTIRVar("N")
    val D = TTIRVar("d")
    val stride = D

    val flashKernel = TTIR("flashAttention")
      .param("float*", "q")      // Query [N, D]
      .param("float*", "k")      // Keys [N, D]
      .param("float*", "v")      // Values [N, D]
      .param("float*", "out")    // Output [N, D]
      .param("int", "N")
      .param("int", "d")
      .local("float", "maxScore", TTIRConst(-1e9f))
      .local("float", "sumExp", TTIRConst(0.0f))
      .local("float", "score", TTIRConst(0.0f))

    val iVar = TTIRVar("i")

    // First pass: compute max score
    flashKernel += TTIRFor("j", 0, N, List(
      TTIRAssign("score", TTIRConst(0.0f)),
      TTIRFor("d_", 0, D, List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("q", iVar, TTIRVar("d_"), stride),
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
      TTIRFor("d_", 0, D, List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("q", iVar, TTIRVar("d_"), stride),
            load2D("k", TTIRVar("j"), TTIRVar("d_"), stride)
          )))
      )),
      TTIRAssign("score", TTIRMathCall("exp", List(TTIRBinOp("-", TTIRVar("score"), TTIRVar("maxScore"))))),
      TTIRAssign("sumExp", TTIRBinOp("+", TTIRVar("sumExp"), TTIRVar("score"))),
      TTIRFor("d_", 0, D, List(
        TTIRStore2D("out", iVar, TTIRVar("d_"), stride,
          TTIRBinOp("+",
            load2D("out", iVar, TTIRVar("d_"), stride),
            TTIRBinOp("*", TTIRVar("score"), load2D("v", TTIRVar("j"), TTIRVar("d_"), stride))
          ), TTIRConst(true))
      ))
    ))

    // Normalize
    flashKernel += TTIRFor("d_", 0, D, List(
      TTIRStore2D("out", iVar, TTIRVar("d_"), stride,
        TTIRBinOp("/", load2D("out", iVar, TTIRVar("d_"), stride), TTIRVar("sumExp")),
        TTIRConst(true))
    ))

    println(flashKernel.emit())

    // ========================================================================
    // PageAttention Kernel (vLLM-style PagedAttention)
    // ========================================================================
    println("\n" + "=" * 80)
    println("[2] PageAttention Kernel (vLLM-style)")
    println("-" * 40)

    val pageKernel = TTIR("pageAttention")
      .param("float*", "key_cache")   // KV cache [num_blocks, block_size, D]
      .param("float*", "value_cache")
      .param("int*", "block_tables")   // Block indices for each sequence
      .param("int*", "seq_lens")      // Sequence lengths
      .param("float*", "query")       // Query [D]
      .param("float*", "out")         // Output [D]
      .param("int", "num_blocks")
      .param("int", "block_size")
      .param("int", "D")
      .local("float", "maxScore", TTIRConst(-1e9f))
      .local("float", "sumExp", TTIRConst(0.0f))
      .local("float", "score", TTIRConst(0.0f))
      .local("float", "attn", TTIRConst(0.0f))

    val numBlocks = TTIRVar("num_blocks")
    val blockSize = TTIRVar("block_size")

    // Iterate over blocks
    pageKernel += TTIRFor("block_idx", 0, numBlocks, List(
      // Get physical block id
      TTIRAssign("score", TTIRConst(0.0f)),
      // Dot product with block
      TTIRFor("d_", 0, TTIRVar("D"), List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("query", TTIRConst(0), TTIRVar("d_"), TTIRVar("D")),
            load2D("key_cache", TTIRVar("block_idx"), TTIRVar("d_"), TTIRVar("D"))
          )))
      )),
      // Softmax
      TTIRIf(TTIRBinOp(">", TTIRVar("score"), TTIRVar("maxScore")),
        List(TTIRAssign("maxScore", TTIRVar("score"))),
        Nil)
    ))

    // Second pass: compute attention and accumulate
    pageKernel += TTIRFor("block_idx", 0, numBlocks, List(
      TTIRAssign("score", TTIRConst(0.0f)),
      TTIRFor("d_", 0, TTIRVar("D"), List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("query", TTIRConst(0), TTIRVar("d_"), TTIRVar("D")),
            load2D("key_cache", TTIRVar("block_idx"), TTIRVar("d_"), TTIRVar("D"))
          )))
      )),
      TTIRAssign("score", TTIRMathCall("exp", List(TTIRBinOp("-", TTIRVar("score"), TTIRVar("maxScore"))))),
      TTIRAssign("sumExp", TTIRBinOp("+", TTIRVar("sumExp"), TTIRVar("score"))),
      // Accumulate output: out[d_] += score * value_cache[block_idx][d_]
      TTIRFor("d_", 0, TTIRVar("D"), List(
        TTIRStore2D("out", TTIRConst(0), TTIRVar("d_"), TTIRVar("D"),
          TTIRBinOp("+",
            load2D("out", TTIRConst(0), TTIRVar("d_"), TTIRVar("D")),
            TTIRBinOp("*", TTIRVar("score"),
              load2D("value_cache", TTIRVar("block_idx"), TTIRVar("d_"), TTIRVar("D")))
          ), TTIRConst(true))
      ))
    ))

    // Normalize
    pageKernel += TTIRFor("d_", 0, TTIRVar("D"), List(
      TTIRStore2D("out", TTIRConst(0), TTIRVar("d_"), TTIRVar("D"),
        TTIRBinOp("/", load2D("out", TTIRConst(0), TTIRVar("d_"), TTIRVar("D")), TTIRVar("sumExp")),
        TTIRConst(true))
    ))

    println(pageKernel.emit())

    // ========================================================================
    // Grouped Query Attention (GQA)
    // ========================================================================
    println("\n" + "=" * 80)
    println("[3] Grouped Query Attention (GQA)")
    println("-" * 40)

    val gqaKernel = TTIR("groupedQueryAttention")
      .param("float*", "q")           // Query [N, D]
      .param("float*", "k")           // Keys [num_kv_heads, D]
      .param("float*", "v")           // Values [num_kv_heads, D]
      .param("float*", "out")         // Output [N, D]
      .param("int", "N")
      .param("int", "D")
      .param("int", "num_q_heads")    // Num query heads
      .param("int", "num_kv_heads")  // Num KV heads (smaller)
      .param("float", "scale")
      .local("float", "maxScore", TTIRConst(-1e9f))
      .local("float", "sumExp", TTIRConst(0.0f))
      .local("float", "score", TTIRConst(0.0f))

    val numQHeads = TTIRVar("num_q_heads")
    val numKVHeads = TTIRVar("num_kv_heads")
    val scale = TTIRConst(1.0f) // Would use scale param in real impl

    // GQA: each query head attends to fewer KV heads
    gqaKernel += TTIRFor("kv_idx", 0, numKVHeads, List(
      TTIRAssign("score", TTIRConst(0.0f)),
      TTIRFor("d_", 0, D, List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("q", iVar, TTIRVar("d_"), D),
            load2D("k", TTIRVar("kv_idx"), TTIRVar("d_"), D)
          )))
      )),
      TTIRAssign("score", TTIRBinOp("*", TTIRVar("score"), scale)),
      TTIRIf(TTIRBinOp(">", TTIRVar("score"), TTIRVar("maxScore")),
        List(TTIRAssign("maxScore", TTIRVar("score"))),
        Nil)
    ))

    gqaKernel += TTIRFor("kv_idx", 0, numKVHeads, List(
      TTIRAssign("score", TTIRConst(0.0f)),
      TTIRFor("d_", 0, D, List(
        TTIRAssign("score", TTIRBinOp("+", TTIRVar("score"),
          TTIRBinOp("*",
            load2D("q", iVar, TTIRVar("d_"), D),
            load2D("k", TTIRVar("kv_idx"), TTIRVar("d_"), D)
          )))
      )),
      TTIRAssign("score", TTIRMathCall("exp", List(TTIRBinOp("-", TTIRBinOp("*", TTIRVar("score"), scale), TTIRVar("maxScore"))))),
      TTIRAssign("sumExp", TTIRBinOp("+", TTIRVar("sumExp"), TTIRVar("score"))),
      TTIRFor("d_", 0, D, List(
        TTIRStore2D("out", iVar, TTIRVar("d_"), D,
          TTIRBinOp("+",
            load2D("out", iVar, TTIRVar("d_"), D),
            TTIRBinOp("*", TTIRVar("score"), load2D("v", TTIRVar("kv_idx"), TTIRVar("d_"), D))
          ), TTIRConst(true))
      ))
    ))

    gqaKernel += TTIRFor("d_", 0, D, List(
      TTIRStore2D("out", iVar, TTIRVar("d_"), D,
        TTIRBinOp("/", load2D("out", iVar, TTIRVar("d_"), D), TTIRVar("sumExp")),
        TTIRConst(true))
    ))

    println(gqaKernel.emit())

    println("\n" + "=" * 80)
    println("All attention kernels generated successfully!")
    println("=" * 80)
  }
}
