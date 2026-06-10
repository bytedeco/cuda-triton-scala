package cuda.dsl.dsl

import cuda.dsl.core._
import cuda.dsl.core.dim3
import cuda.dsl.runtime._
import org.bytedeco.cuda.global.cudart.*
import org.bytedeco.javacpp._

/** Triton Kernel Runner - 自动编译和执行 DSL 生成的 CUDA kernel
 *
 * 完整流程:
 * 1. @TritonKernelMacro 生成 CUDA C++ 代码并注册到 KernelRegistry
 * 2. KernelRegistry 缓存编译后的 PTX 模块
 * 3. runKernel 执行编译后的 kernel
 */
object TritonKernelRunner {

  // 内核缓存: name -> (CUDA source, compiled kernel)
  private val kernelCache = scala.collection.mutable.Map[String, CompiledKernel]()
  private val sourceCache = scala.collection.mutable.Map[String, String]()
  def removeSource(name: String): Unit = sourceCache.remove(name)

  // 获取运行时
  private def getRuntime(): DeviceRuntime = DeviceSelector.getRuntime()

  /** 注册 kernel 的 CUDA 源代码
   *  由 @TritonKernelMacro 在编译时调用
   */
  def registerKernelSource(name: String, cudaSource: String): Unit = {
    sourceCache(name) = cudaSource
    println(s"[TritonKernelRunner] Registered: $name (${cudaSource.length} chars)")
  }

  /** 编译 kernel (如果尚未编译)
   *  运行时调用
   */
  def compileKernel(name: String): CompiledKernel = {
    kernelCache.getOrElseUpdate(name, {
      val runtime = getRuntime()
      val source = sourceCache.getOrElse(name, {
        throw new RuntimeException(s"Kernel source for '$name' not found in registry")
      })

      println(s"[TritonKernelRunner] Compiling: $name")
      val kernel = runtime.compileKernel(name, source)
      if (kernel == null) {
        throw new RuntimeException(s"Failed to compile kernel: $name")
      }
      println(s"[TritonKernelRunner] Compiled successfully: $name")
      kernel
    })
  }

  /** 检查 kernel 是否已注册
   */
  def isRegistered(name: String): Boolean = sourceCache.contains(name)

  /** 检查 kernel 是否已编译
   */
  def isCompiled(name: String): Boolean = kernelCache.contains(name)

  /** 运行 kernel
   *  @param name kernel 名称
   *  @param grid Grid 维度
   *  @param block Block 维度
   *  @param args kernel 参数 (指针)
   */
  def launchKernel(name: String, grid: dim3, block: dim3, args: Pointer*): Unit = {
    val runtime = getRuntime()
    val kernel = compileKernel(name)
    runtime.launchKernel(kernel, grid, block, args)
    runtime.synchronize()
  }

  /** 便捷方法: 运行 1D kernel
   */
  def launchKernel1D(name: String, n: Int, blockSize: Int = 256)(args: Pointer*): Unit = {
    val grid = dim3((n + blockSize - 1) / blockSize, 1, 1)
    launchKernel(name, grid, dim3(blockSize, 1, 1), args: _*)
  }

  /** 便捷方法: 运行 2D kernel
   */
  def launchKernel2D(name: String, gridX: Int, gridY: Int, blockX: Int, blockY: Int)(args: Pointer*): Unit = {
    launchKernel(name, dim3(gridX, gridY, 1), dim3(blockX, blockY, 1), args: _*)
  }

  /** 清除所有缓存的 kernel
   */
  def clearCache(): Unit = {
    kernelCache.clear()
    println("[TritonKernelRunner] Cache cleared")
  }

  /** 获取已注册 kernel 的源代码
   */
  def getSource(name: String): Option[String] = sourceCache.get(name)

  /** 打印所有已注册的 kernel
   */
  def listKernels(): Unit = {
    println(s"[TritonKernelRunner] Registered kernels: ${sourceCache.size}")
    sourceCache.foreach { (name, source) =>
      val status = if (kernelCache.contains(name)) "compiled" else "source only"
      println(s"  - $name ($status, ${source.length} chars)")
    }
  }
}

/** Triton Kernel 元数据 - 保存 kernel 的签名信息
 */
case class TritonKernelMeta(
  name: String,
  source: String,
  paramTypes: List[String],  // List of "float*" or "int" etc
  paramNames: List[String],
  hasUserOut: Boolean,
  gridType: String = "1D",    // 1D, 2D, 3D
  blockSize: Int = 256
)

/** Triton Kernel 注册表
 */
object TritonKernelRegistry {
  private val kernels = scala.collection.mutable.Map[String, TritonKernelMeta]()

  def register(meta: TritonKernelMeta): Unit = {
    kernels(meta.name) = meta
    TritonKernelRunner.registerKernelSource(meta.name, meta.source)
    println(s"[TritonKernelRegistry] Registered: ${meta.name}")
  }

  def get(name: String): Option[TritonKernelMeta] = kernels.get(name)

  def listAll(): Seq[TritonKernelMeta] = kernels.values.toSeq

  def exists(name: String): Boolean = kernels.contains(name)

  def unregister(name: String): Unit = {
    kernels.remove(name)
    TritonKernelRunner.removeSource(name)
  }
}
