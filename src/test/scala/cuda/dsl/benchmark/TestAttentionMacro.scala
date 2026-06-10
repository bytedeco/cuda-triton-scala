package cuda.dsl.benchmark

import cuda.dsl.dsl._
import TTIRDSL._

/** Test Attention IR nodes and DSL helpers
 */
object TestAttentionMacro {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Attention Kernel Tests (TTIR DSL)")
    println("=" * 80)
    println()

    // ========================================================================
    // Test 1: Direct TTIR FlashAttention
    // ========================================================================
    println("[1] FlashAttention Kernel")
    println("-" * 40)

    val flashKernel = TTIR("flashAttentionKernel")
      .param("float*", "q")
      .param("float*", "k")
      .param("float*", "v")
      .param("float*", "out")
      .param("int", "N")
      .param("int", "D")

    flashKernel += flashAttention("q", "k", "v", "out", TTIRVar("N"), TTIRVar("D"))
    println(flashKernel.emit())

    // ========================================================================
    // Test 2: Direct TTIR PageAttention
    // ========================================================================
    println("\n[2] PageAttention Kernel")
    println("-" * 40)

    val pageKernel = TTIR("pageAttentionKernel")
      .param("float*", "query")
      .param("float*", "key_cache")
      .param("float*", "value_cache")
      .param("int*", "block_tables")
      .param("int*", "seq_lens")
      .param("float*", "out")
      .param("int", "num_blocks")
      .param("int", "block_size")
      .param("int", "D")

    pageKernel += pageAttention(
      "query", "key_cache", "value_cache",
      "block_tables", "seq_lens", "out",
      TTIRVar("num_blocks"), TTIRVar("block_size"), TTIRVar("D")
    )
    println(pageKernel.emit())

    // ========================================================================
    // Test 3: Direct TTIR FlexAttention
    // ========================================================================
    println("\n[3] FlexAttention Kernel")
    println("-" * 40)

    val flexKernel = TTIR("flexAttentionKernel")
      .param("float*", "q")
      .param("float*", "k")
      .param("float*", "v")
      .param("float*", "out")
      .param("int", "N")
      .param("int", "D")

    flexKernel += flexAttention("q", "k", "v", "out", TTIRVar("N"), TTIRVar("D"))
    println(flexKernel.emit())

    // ========================================================================
    // Test 4: Grouped Query Attention
    // ========================================================================
    println("\n[4] Grouped Query Attention Kernel")
    println("-" * 40)

    val gqaKernel = TTIR("groupedQueryAttentionKernel")
      .param("float*", "q")
      .param("float*", "k")
      .param("float*", "v")
      .param("float*", "out")
      .param("int", "N")
      .param("int", "D")
      .param("int", "numQHeads")
      .param("int", "numKVHeads")
      .param("float", "scale")

    gqaKernel += groupedQueryAttention(
      "q", "k", "v", "out",
      TTIRVar("N"), TTIRVar("D"),
      TTIRVar("numQHeads"), TTIRVar("numKVHeads"),
      TTIRVar("scale")
    )
    println(gqaKernel.emit())

    // ========================================================================
    // Test 5: Multi-Head Attention
    // ========================================================================
    println("\n[5] Multi-Head Attention Kernel")
    println("-" * 40)

    val mhaKernel = TTIR("multiHeadAttentionKernel")
      .param("float*", "q")
      .param("float*", "k")
      .param("float*", "v")
      .param("float*", "out")
      .param("int", "N")
      .param("int", "D")
      .param("int", "numHeads")
      .param("float", "scale")

    mhaKernel += multiHeadAttention(
      "q", "k", "v", "out",
      TTIRVar("N"), TTIRVar("D"),
      TTIRVar("numHeads"), TTIRVar("scale")
    )
    println(mhaKernel.emit())

    println("\n" + "=" * 80)
    println("All attention kernel tests completed!")
    println("=" * 80)
  }
}