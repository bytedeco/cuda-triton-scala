package cuda.dsl.benchmark

import cuda.dsl.dsl._
import TTIRDSL._

/** Test @TritonKernelMacro for StoreKVCache */
object TestStoreKVCacheMacro {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("@TritonKernelMacro: StoreKVCache Kernel")
    println("=" * 80)
    println("\nCheck /tmp/cuda_dsl_generated_kernels.txt for generated CUDA code.")
    println("=" * 80)

    // StoreKVCache kernel using @TritonKernelMacro
    // All parameters are treated as pointers by the macro
    @TritonKernelMacro
    def storeKVCacheKernel(
        keyPtr: Float, keyStride: Int,
        valuePtr: Float, valueStride: Int,
        kCachePtr: Float, vCachePtr: Float,
        slotMappingPtr: Int,
        D: Int): Unit = {
      // 调用 storeKVCache DSL helper，会被翻译成 TTIRStoreKVCache
      storeKVCache(keyPtr, keyStride, valuePtr, valueStride, kCachePtr, vCachePtr, slotMappingPtr, D)
    }

    println("\nStoreKVCache kernel defined with @TritonKernelMacro")
  }
}