package cuda.dsl.benchmark

import cuda.dsl.dsl._
import TTIRDSL._

/** Test StoreKVCache kernel for vLLM-style paged KV cache */
object TestStoreKVCache {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("StoreKVCache Kernel Tests")
    println("=" * 80)
    println()

    // ========================================================================
    // Test 1: Basic StoreKVCache (vLLM-style)
    // ========================================================================
    println("[1] StoreKVCache Kernel (vLLM-style)")
    println("-" * 40)

    val storeKVCacheKernel = TTIR("storeKVCache")
      .param("float*", "key_ptr")
      .param("int", "key_stride")
      .param("float*", "value_ptr")
      .param("int", "value_stride")
      .param("float*", "k_cache_ptr")
      .param("float*", "v_cache_ptr")
      .param("int*", "slot_mapping_ptr")
      .param("int", "D")

    storeKVCacheKernel += storeKVCache(
      "key_ptr", TTIRVar("key_stride"),
      "value_ptr", TTIRVar("value_stride"),
      "k_cache_ptr", "v_cache_ptr",
      "slot_mapping_ptr",
      TTIRVar("D")
    )
    println(storeKVCacheKernel.emit())

    // ========================================================================
    // Test 2: StoreKVCache with block_size parameter
    // ========================================================================
    println("\n[2] StoreKVCache with block_size")
    println("-" * 40)

    val storeKVCacheKernel2 = TTIR("storeKVCacheBlock")
      .param("float*", "key_ptr")
      .param("int", "key_stride")
      .param("float*", "value_ptr")
      .param("int", "value_stride")
      .param("float*", "k_cache_ptr")
      .param("float*", "v_cache_ptr")
      .param("int*", "slot_mapping_ptr")
      .param("int", "block_size")
      .param("int", "D")

    storeKVCacheKernel2 += storeKVCache(
      "key_ptr", TTIRVar("key_stride"),
      "value_ptr", TTIRVar("value_stride"),
      "k_cache_ptr", "v_cache_ptr",
      "slot_mapping_ptr",
      TTIRVar("D")
    )
    println(storeKVCacheKernel2.emit())

    println("\n" + "=" * 80)
    println("All StoreKVCache kernel tests completed!")
    println("=" * 80)
  }
}