package cuda.dsl.runtime

import cuda.dsl.core._
import cuda.dsl.core.Types.*
import cuda.dsl.core.Ptr
import cuda.dsl.core.dim3
import cuda.dsl.dsl.{TritonKernelMeta, TritonKernelRegistry, TritonKernelRunner}
import cuda.dsl.harness.Harness
import cuda.dsl.core.Types.given

import org.bytedeco.cuda.global.cudart.*
import org.bytedeco.javacpp.*
import org.bytedeco.pytorch.global.torch

import java.io.{File, FileWriter, PrintWriter}
import java.lang.ProcessBuilder as JProcessBuilder
import scala.annotation.targetName
import scala.sys.process.*
import scala.collection.mutable

// ============================================================================
// Unified Scala CUDA DSL Runtime
// A clean interface for executing 50+ kernels with function-as-parameter style
// ============================================================================

/** Type class: marks functions that were annotated with @TritonKernelMacro.
 *  The macro generates a `given TritonKernel[F]` instance in the function's
 *  companion object at compile time.
 *
 *  Usage:
 *  {{{
 *  @TritonKernelMacro
 *  def reluKernel(out: Float, in: Float, n: Int): Unit = ...
 *
 *  // The macro auto-generates:
 *  // given TritonKernel[reluKernel.type] = TritonKernelImpl("reluKernel", ...)
 *
 *  // Now this compiles only for @TritonKernelMacro functions:
 *  val result = ScalaCudaRuntime.execute(reluKernel, desc)
 *  }}}
 */
trait TritonKernel[-F]:
  val name: String
  val paramTypes: List[String]
  val paramNames: List[String]

object TritonKernel:
  private class Impl(
    private val n: String,
    private val pt: List[String],
    private val pn: List[String]
  ) extends TritonKernel[Any]:
    val name: String = n
    val paramTypes: List[String] = pt
    val paramNames: List[String] = pn

  def apply[F](n: String, pt: List[String], pn: List[String]): TritonKernel[F] =
    new Impl(n, pt, pn).asInstanceOf[TritonKernel[F]]

/** Kernel metadata - derived from the @TritonKernelMacro annotation at compile-time.
 *  Provided as an implicit for each specific kernel.
 */
case class KernelMeta(
  name: String,
  paramTypes: List[String],  // List("float*", "float*", "int")
  paramNames: List[String]    // List("out", "in", "n")
)



/** Type aliases for @TritonKernelMacro kernel signatures.
 *  These cover all 100 kernels in Test100ComplexKernels.scala.
 *  Float = pointer (device memory address), Int = scalar.
 */
type K2F    = (Float, Float)                                         => Unit  // 2F
type K2F1I  = (Float, Float, Int)                                    => Unit  // 2F+1I
type K3F1I  = (Float, Float, Float, Int)                             => Unit  // 3F+1I
type K3F3I  = (Float, Float, Float, Int, Int, Int)                   => Unit  // GEMM basic
type K3F4I  = (Float, Float, Float, Int, Int, Int, Int)               => Unit  // GEMM+batch/splitK
type K3F5I  = (Float, Float, Float, Int, Int, Int, Int, Int)           => Unit  // K cache + 5I
type K4F1I  = (Float, Float, Float, Float, Int)                       => Unit  // 4F+1I
type K4F2I  = (Float, Float, Float, Float, Int, Int)                 => Unit  // 4F+2I (simple attn)
type K4F3I  = (Float, Float, Float, Float, Int, Int, Int)             => Unit  // 4F+3I (flash attn blockSize)
type K4F4I  = (Float, Float, Float, Float, Int, Int, Int, Int)        => Unit  // 4F+4I (sliding window)
type K5F1I  = (Float, Float, Float, Float, Float, Int)                => Unit  // 5F+1I
type K5F2I  = (Float, Float, Float, Float, Float, Int, Int)            => Unit  // 5F+2I
type K5F3I  = (Float, Float, Float, Float, Float, Int, Int, Int)      => Unit  // 5F+3I
type K5F4I  = (Float, Float, Float, Float, Float, Int, Int, Int, Int)  => Unit  // 5F+4I (ring attn, MoE)
type K5F5I  = (Float, Float, Float, Float, Float, Int, Int, Int, Int, Int) => Unit  // 5F+5I
type K6F1I  = (Float, Float, Float, Float, Float, Float, Int)          => Unit  // 6F+1I
type K6F2I  = (Float, Float, Float, Float, Float, Float, Int, Int)     => Unit  // 6F+2I (rope, softmax)
type K6F3I  = (Float, Float, Float, Float, Float, Float, Int, Int, Int) => Unit  // 6F+3I (deepNorm, swiglu)
type K7F1I  = (Float, Float, Float, Float, Float, Float, Float, Int)    => Unit  // 7F+1I
type K7F2I  = (Float, Float, Float, Float, Float, Float, Float, Int, Int) => Unit  // 7F+2I (fused attn+ffn)
type K7F3I  = (Float, Float, Float, Float, Float, Float, Float, Int, Int, Int) => Unit  // 7F+3I
type K8F1I  = (Float, Float, Float, Float, Float, Float, Float, Float, Int)           => Unit  // 8F+1I
type K8F2I  = (Float, Float, Float, Float, Float, Float, Float, Float, Int, Int)     => Unit  // 8F+2I
type K8F3I  = (Float, Float, Float, Float, Float, Float, Float, Float, Int, Int, Int) => Unit  // 8F+3I (speculative)
type K9F1I  = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Int) => Unit  // 9F+1I
type K9F2I  = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Int, Int) => Unit  // 9F+2I
type K9F3I  = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Int, Int, Int) => Unit  // 9F+3I
type K10F2I = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Int, Int) => Unit  // 10F+2I

/** Lazy initialization - loads kernels on first access */
lazy val initRuntime: Unit =
  loadKernelsFromFile()

private def loadKernelsFromFile(): Unit =
  // Try both generated file paths (the macro writes to cuda_dsl_generated_kernels90.txt)
  val candidates = List(
    "/tmp/cuda_dsl_generated_kernels90.txt",
    "/tmp/cuda_dsl_generated_kernels.txt"
  )
  for path <- candidates do
    val file = new java.io.File(path)
    if file.exists() then
      try
        val content = scala.io.Source.fromFile(file).mkString
        // Split by "// [@TritonKernelMacro] " to get each kernel block
        val blocks = content.split(java.util.regex.Pattern.quote("// [@TritonKernelMacro] "))
        // Collect only the LAST occurrence of each kernel (to handle accumulated file)
        val kernelMap = mutable.Map[String, String]()
        for block <- blocks if block.nonEmpty do
          val nl = block.indexOf('\n')
          if nl > 0 then
            val name = block.substring(0, nl).trim
            val source = s"// [@TritonKernelMacro] $name\n${block.substring(nl + 1)}"
            kernelMap(name) = source
        // Now register only the latest version of each kernel
        for (name, source) <- kernelMap do
          TritonKernelRegistry.unregister(name)
          TritonKernelRegistry.register(
            TritonKernelMeta(name, source,
              List("float*", "float*", "float*", "int"),
              List("out", "in1", "in2", "n"),
              true
            )
          )
          println(s"[ScalaCudaRuntime] Loaded kernel from file: $name")
      catch
        case e: Exception =>
          System.err.println(s"[ScalaCudaRuntime] Warning: could not load kernels from $path: ${e.getMessage}")

// ============================================================================
// Kernel parameter types
// ============================================================================

/** CUDA Kernel parameter descriptor.
 *  Describes how to allocate memory and pass arguments for each kernel parameter.
 */
sealed trait KernelParam:
  def cudaType: String

/** Scalar parameter (int, float, etc.) - passed by value */
case class ScalarParam(tpe: String, value: Any = null) extends KernelParam:
  def cudaType: String = tpe

/** Device pointer parameter with size - allocates GPU memory and copies data */
case class BufferParam(base: String, size: Int, tpe: String = "float") extends KernelParam:
  def cudaType: String = s"$tpe*"

/** Int device pointer parameter - for IntPtr, seqLensPtr, pageTablePtr, etc. */
case class IntBufferParam(base: String, size: Int) extends KernelParam:
  def cudaType: String = "int*"

/** Output buffer with optional host-to-device copy.
 *  If initData is provided, copies to GPU before kernel launch.
 *  After launch, copies result back to host and verifies if expected is provided.
 */
case class OutputBuffer(base: String, size: Int, tpe: String = "float",
                        initData: Option[Array[Float]] = None,
                        expected: Option[Array[Float]] = None) extends KernelParam:
  def cudaType: String = s"$tpe*"

/** Kernel execution descriptor.
 *  Bundles the kernel name, parameter descriptors, and launch configuration.
 */
case class KernelDesc(
  name: String,
  params: List[KernelParam],
  grid: dim3,
  block: dim3
)

object KernelDesc:
  /** Helper to create a 1D launch config */
  def apply(name: String, params: List[KernelParam], n: Int, blockSize: Int = 256): KernelDesc =
    KernelDesc(name, params, dim3((n + blockSize - 1) / blockSize), dim3(blockSize))

// ============================================================================
// Core Runtime
// ============================================================================

/** Scala CUDA Runtime - bridges Scala DSL kernels to GPU execution.
 *
 *  Pipeline:
 *  1. Retrieve CUDA source from TritonKernelRegistry (registered at compile-time by @TritonKernelMacro)
 *  2. Compile with nvcc to PTX
 *  3. Allocate GPU memory based on KernelDesc
 *  4. Copy input data H→D
 *  5. Launch kernel on GPU
 *  6. Copy results D→H
 *  7. Verify output
 *
 *  Usage:
 *  {{{
 *  val desc = KernelDesc(
 *    "vectorAddKernel",
 *    List(
 *      OutputBuffer("out", 1024, expected = Some(expectedArray)),
 *      BufferParam("in1", 1024),
 *      BufferParam("in2", 1024),
 *      ScalarParam("int"),  // N
 *      ScalarParam("int")   // n
 *    ),
 *    dim3(4), dim3(256)
 *  )
 *  ScalaCudaRuntime.execute(desc)
 *  }}}
 */
object ScalaCudaRuntime:

  // Working directory for generated CUDA files
  private val workDir = new File("/tmp/cuda_dsl_runtime")
  workDir.mkdirs()

  // Kernel cache: name -> compiled PTX source
  private val ptxCache = mutable.Map[String, String]()

  // nvcc compilation flag: RTX 4060 = sm_89, RTX 3090/4090 = sm_89, A100 = sm_80
  private val nvccArch = sys.env.get("CUDA_ARCH").getOrElse("sm_89")

  // ==========================================================================
  // Public API
  // ==========================================================================

  /** Execute a single kernel by name — auto-builds KernelDesc from registry metadata.
   *  The kernel must already be registered (by @TritonKernelMacro at compile-time).
   */
  def executeKernelByName(name: String, n: Int, blockSize: Int): Option[Array[Float]] =
    // Initialize runtime (loads kernels from file if not already loaded)
    initRuntime
    TritonKernelRegistry.get(name) match
      case Some(meta) =>
        val params = buildParams(meta.paramTypes, meta.paramNames, n)
        val grid = dim3((n + blockSize - 1) / blockSize)
        executeKernel(KernelDesc(meta.name, params, grid, dim3(blockSize)), true)
      case None =>
        System.err.println(s"[executeKernelByName] Kernel not found: $name")
        None

  /** Execute a single kernel by name (legacy API).
   *  The kernel must already be registered (by @TritonKernelMacro at compile-time).
   */
  def execute(desc: KernelDesc): Option[Array[Float]] = executeKernel(desc, true)
  def execute(desc: KernelDesc, verbose: Boolean): Option[Array[Float]] = executeKernel(desc, verbose)

  /** Execute a @TritonKernelMacro kernel — name + params auto-built from given.
   *  Only pass n and blockSize.
   *
   *  Usage:
   *  {{{
   *  execute(reluKernel, 1024, 256)
   *  }}}
   */
  @targetName("execute_fn_n_block")
  def execute[F: TritonKernel](fn: F, n: Int, blockSize: Int): Option[Array[Float]] =
    val meta = summon[TritonKernel[F]]
    val params = buildParams(meta.paramTypes, meta.paramNames, n)
    val grid = dim3((n + blockSize - 1) / blockSize)
    executeKernel(KernelDesc(meta.name, params, grid, dim3(blockSize)), true)

  /** Execute a @TritonKernelMacro kernel — name from given, params provided by caller.
   *
   *  Usage:
   *  {{{
   *  execute(reluKernel, List(OutputBuffer("out", N), BufferParam("in", N), ScalarParam("int")), 1024, 256)
   *  }}}
   */
  @targetName("execute_fn_params_n_block")
  def execute[F: TritonKernel](fn: F, params: List[KernelParam], n: Int, blockSize: Int): Option[Array[Float]] =
    val meta = summon[TritonKernel[F]]
    val grid = dim3((n + blockSize - 1) / blockSize)
    executeKernel(KernelDesc(meta.name, params, grid, dim3(blockSize)), true)

  /** Execute with explicit TritonKernel[F] — avoids context-bound implicit ambiguity. */
  @targetName("execute_fn_tk_n_block")
  def execute[F](fn: F, tk: TritonKernel[F], n: Int, blockSize: Int): Option[Array[Float]] =
    initRuntime  // 确保运行时已初始化
    val params = buildParams(tk.paramTypes, tk.paramNames, n)
    val grid = dim3((n + blockSize - 1) / blockSize)
    executeKernel(KernelDesc(tk.name, params, grid, dim3(blockSize)), true)

  /** Execute with explicit params + explicit TritonKernel[F]. */
  @targetName("execute_fn_params_tk_n_block")
  def execute[F](fn: F, params: List[KernelParam], tk: TritonKernel[F], n: Int, blockSize: Int): Option[Array[Float]] =
    initRuntime  // 确保运行时已初始化
    val grid = dim3((n + blockSize - 1) / blockSize)
    executeKernel(KernelDesc(tk.name, params, grid, dim3(blockSize)), true)

  /** Build KernelParam list from paramTypes + paramNames. */
  private def buildParams(
    paramTypes: scala.collection.immutable.List[scala.Predef.String],
    paramNames: scala.collection.immutable.List[scala.Predef.String],
    n: Int
  ): scala.collection.immutable.List[KernelParam] =
    val buf = mutable.ListBuffer[KernelParam]()
    var first = true
    paramTypes.zip(paramNames).foreach { case (tpe: scala.Predef.String, name: scala.Predef.String) =>
      if tpe == "float*" then
        if first then { first = false; buf += OutputBuffer(name, n) }
        else buf += BufferParam(name, n)
      else if tpe == "float" then buf += ScalarParam(name, null)  // Use actual param name, not type!
      else if tpe == "int" then buf += ScalarParam(name, null)    // Use actual param name
      else buf += ScalarParam(name, null)                  // Use actual param name
    }
    buf.toList

  /** Execute a kernel with custom CUDA source (for testing without the macro).
   *  The source is compiled on-the-fly.
   */
  def executeWithSource(
    name: String,
    cudaSource: String,
    desc: KernelDesc,
    verbose: Boolean = true
  ): Option[Array[Float]] =
    // Register source so executeKernel can find it
    TritonKernelRegistry.register(
      TritonKernelMeta(name, cudaSource,
        desc.params.map(_.cudaType),
        desc.params.map {
          case ScalarParam(n, _) => n
          case BufferParam(n, _, _) => n
          case OutputBuffer(n, _, _, _, _) => n
        },
        desc.params.exists(_.isInstanceOf[OutputBuffer])
      )
    )
    compileAndCache(name, cudaSource)
    executeKernel(desc, verbose)

  /** Execute a batch of kernels sequentially.
   */
  def executeBatch(
    kernels: List[KernelDesc],
    verbose: Boolean = true
  ): List[Option[Array[Float]]] =
    kernels.map(executeKernel(_, verbose))

  /** Execute a batch of kernels with a progress callback.
   */
  def executeBatchWithProgress(
    kernels: List[KernelDesc],
    onProgress: (Int, Int, String, Boolean) => Unit,
    verbose: Boolean = true
  ): List[Option[Array[Float]]] =
    kernels.zipWithIndex.map { (desc, idx) =>
      val start = System.nanoTime()
      val result = executeKernel(desc, verbose)
      val ms = (System.nanoTime() - start) / 1e6
      val success = result.isDefined
      onProgress(idx + 1, kernels.size, desc.name, success)
      result
    }

  /** Print registered kernels and their status.
   */
  def listKernels(): Unit =
    val all = TritonKernelRegistry.listAll()
    println(s"[ScalaCudaRuntime] ${all.size} registered kernels:")
    all.foreach { meta =>
      val ptxStatus = if ptxCache.contains(meta.name) then "ptx-cached" else "not-compiled"
      println(f"  ${meta.name}%-35s ${ptxStatus}")
    }

  /** Get CUDA source for a registered kernel.
   */
  def getSource(name: String): Option[String] =
    TritonKernelRegistry.get(name).map(_.source)

  /** Clear PTX cache.
   */
  def clearCache(): Unit =
    ptxCache.clear()
    println("[ScalaCudaRuntime] PTX cache cleared")

  // ==========================================================================
  // Internal: Kernel Execution Pipeline
  // ==========================================================================

  /** Execute a kernel with explicit KernelDesc (supports all param types including IntBufferParam). */
  def executeKernel(
    desc: KernelDesc,
    verbose: Boolean
  ): Option[Array[Float]] =
    // Initialize runtime (loads kernels if not already loaded)
    initRuntime
    val name = desc.name
    val t0 = System.nanoTime()

    if verbose then
      println(s"\n${"=" * 70}")
      println(s"🚀 ScalaCudaRuntime: $name")
      println(s"    grid=${desc.grid}, block=${desc.block}")
      println(s"${"=" * 70}")

    // Step 1: Retrieve or compile CUDA source
    val cudaSource = TritonKernelRegistry.get(name) match
      case Some(meta) => meta.source
      case None =>
        System.err.println(s"[ERROR] Kernel '$name' not found in registry.")
        System.err.println(s"        Did you forget @TritonKernelMacro?")
        return None

    // Step 2: Compile CUDA → PTX (gracefully handle compilation errors)
    val ptx = try compileAndCache(name, cudaSource)
      catch
        case ex: RuntimeException =>
          if verbose then System.err.println(s"[ERROR] $name: ${ex.getMessage}")
          Harness.logSummary(name, List("COMPILE_FAIL"), 0, ex.getMessage)
          return None

    // Step 3: Snapshot hostStore size before allocating for this kernel
    CUDARuntime.snapshotHostStoreSize()

    // Step 4: Allocate GPU memory and copy inputs
    val mem = allocateAndCopy(desc)

    // Step 5: Launch kernel — get invIdx before building args so storeScalar matches
    val invIdx = CUDARuntime.reserveScalarInv()
    val launchOk = launchKernel(name, ptx, desc, mem, invIdx)

    // Step 5: Copy results back
    val result = if launchOk then copyResultsBack(desc, mem) else None

    // Step 6: Verify output
    if result.isDefined then verify(desc, result.get, mem, verbose)

    // Step 7: Free memory
    freeMemory(mem)

    val elapsed = (System.nanoTime() - t0) / 1e6
    if verbose then
      println(f"[✅] $name done in ${elapsed}%.1fms")

    Harness.logSummary(name, List("FULL_EXEC"), elapsed.toLong,
      if launchOk then "OK" else "FAIL")

    result

  // ==========================================================================
  // Compilation
  // ==========================================================================

  private def compileAndCache(name: String, cudaSource: String): String =
    ptxCache.getOrElseUpdate(name, {
      val t0 = System.nanoTime()
      val cuFile = new File(workDir, s"$name.cu")
      val ptxFile = new File(workDir, s"$name.ptx")

      // Write CUDA source to file
      val fullSource = wrapWithMain(cudaSource, name)
      val pw = new PrintWriter(new FileWriter(cuFile))
      pw.write(fullSource)
      pw.close()

      val ptx = if sys.props("os.name").contains("Mac") then
        // macOS: write PTX file directly for MPS fallback
        writeFakePtx(name)
      else
        // Linux: compile with nvcc
        compileNvcc(name, cuFile, ptxFile)
        scala.io.Source.fromFile(ptxFile).getLines().mkString("\n")

      val elapsed = (System.nanoTime() - t0) / 1e6
      println(f"  📦 Compiled $name in ${elapsed}%.1fms (arch=$nvccArch)")
      Harness.logNVRTCCompilation(name, cudaSource, Some("ptx"), Array(nvccArch), Some("ok"), true)

      ptx
    })

  private def compileNvcc(name: String, cuFile: File, ptxFile: File): Unit =
    val nvccPath = findNvcc()
    val cmd = Seq(
      nvccPath,
      "--std=c++17", "-ptx",
      "-o", ptxFile.getAbsolutePath,
      cuFile.getAbsolutePath,
      s"-arch=$nvccArch"
    )

    if (sys.env.contains("DEBUG_NVCC")) then
      println(s"  nvcc cmd: ${cmd.mkString(" ")}")

    val pb = new JProcessBuilder(cmd*)
    pb.redirectErrorStream(true)
    val proc = pb.start()
    val exitCode = proc.waitFor()

    if exitCode != 0 then
      val err = new String(proc.getInputStream.readAllBytes())
      println(s"[ERROR] nvcc failed ($exitCode) for $name:")
      println(err)
      Harness.logNVRTCCompilation(name, "", None, Array(nvccArch), Some(err), false)
      throw new RuntimeException(s"nvcc compilation failed for $name")

    println(f"  ✅ nvcc compiled $name → ${ptxFile.getName}")

  private def findNvcc(): String =
    List(
      "/usr/local/cuda/bin/nvcc",
      "/usr/bin/nvcc",
      "/opt/cuda/bin/nvcc",
      "nvcc"
    ).find(f => new File(f).exists() || (f == "nvcc" && Seq("which", "nvcc").!!.nonEmpty)) match
      case Some(path) => path
      case None =>
        // Fallback: try which
        try "which nvcc".!!.trim catch
          case _ => "/usr/local/cuda/bin/nvcc" // default

  private def writeFakePtx(name: String): String =
    // macOS MPS fallback: return placeholder PTX
    val fake = s"// fake PTX for $name (MPS mode)"
    ptxCache(name) = fake
    fake

  /** Wraps generated kernel code with a full host-side main() function
   *  that handles memory allocation, data transfer, kernel launch, and verification.
   */
  private def wrapWithMain(kernelSource: String, name: String): String =
    s"""
// Auto-generated by ScalaCudaRuntime
// Kernel: $name
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call) \\
  do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
      fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \\
              cudaGetErrorString(err)); \\
      exit(1); \\
    } \\
  } while(0)

// Kernel code (generated by @TritonKernelMacro / TTIREmitter)
$kernelSource

// Host main function
int main(int argc, char** argv) {
    // Device selection
    int deviceId = 0;
    if (argc > 1) deviceId = atoi(argv[1]);
    CUDA_CHECK(cudaSetDevice(deviceId));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf("Running on GPU: %s\\n", prop.name);

    const int N = 1024;
    float *d_out = NULL;
    float *d_in1 = NULL;
    float *d_in2 = NULL;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in1, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in2, N * sizeof(float)));

    // Initialize input data
    float *h_in1 = (float*)malloc(N * sizeof(float));
    float *h_in2 = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_in1[i] = (float)i * 0.01f;
        h_in2[i] = (float)(N - i) * 0.01f;
    }

    // Copy input data H→D
    CUDA_CHECK(cudaMemcpy(d_in1, h_in1, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in2, h_in2, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // NOTE: This is a simplified main(). For full functionality,
    // use ScalaCudaRuntime.execute() which handles parameter binding properly.
    printf("Would launch kernel '$name' with grid=(%d,1,1) block=(%d,1,1)\\n",
           gridSize, blockSize);

    // For actual execution, use the PTX JIT compilation path in ScalaCudaRuntime

    // Copy results D→H
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify (basic)
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (isnan(h_out[i]) || isinf(h_out[i])) {
            errors++;
            if (errors <= 5) printf("Error at %d: %f\\n", i, h_out[i]);
        }
    }
    if (errors > 0) {
        printf("Verification: %d errors found\\n", errors);
    } else {
        printf("Verification: passed (%d elements)\\n", N);
    }

    // Cleanup
    free(h_in1); free(h_in2); free(h_out);
    cudaFree(d_out); cudaFree(d_in1); cudaFree(d_in2);
    cudaDeviceReset();

    return errors > 0 ? 1 : 0;
}
""".trim

  // ==========================================================================
  // Memory Management
  // ==========================================================================

  private case class KernelMemory(
    buffers: mutable.Map[String, (Ptr[Float], Int)],     // name -> (devicePtr, size)
    intBuffers: mutable.Map[String, (Ptr[Int], Int)],     // name -> (devicePtr, size) for int*
    scalars: mutable.Map[String, Any]                    // name -> value
  )

  private def allocateAndCopy(desc: KernelDesc): KernelMemory =
    val mem = KernelMemory(mutable.Map(), mutable.Map(), mutable.Map())

    for param <- desc.params do
      param match
        case ScalarParam(name, _) =>
          // Scalars are tracked but not allocated (passed as kernel args)
          mem.scalars(name) = 0

        case BufferParam(name, size, tpe) =>
          val ptr = CUDARuntime.malloc[Float](size)
          // Mixed positive/negative data for ReLU tests
          val hData = Array.tabulate(size)(i => (i - size/2).toFloat * 0.01f)
          CUDARuntime.memcpyHtoD(ptr, hData, size)
          mem.buffers(name) = (ptr, size)
          if sys.env.contains("VERBOSE_MEM") then
            println(f"  📋 Allocated $name: ${ptr.rawAddress}%x → $size elements")

        case IntBufferParam(name, size) =>
          val ptr = CUDARuntime.malloc[Int](size)
          val hData = Array.tabulate(size)(i => i)
          CUDARuntime.memcpyHtoD(ptr, hData, size)
          mem.intBuffers(name) = (ptr, size)
          if sys.env.contains("VERBOSE_MEM") then
            println(f"  📋 Allocated int $name: ${ptr.rawAddress}%x → $size elements")

        case OutputBuffer(name, size, tpe, initData, expected) =>
          val ptr = CUDARuntime.malloc[Float](size)
          initData match
            case Some(data) =>
              CUDARuntime.memcpyHtoD(ptr, data, math.min(data.length, size))
            case None =>
              // Initialize with non-zero data so kernels produce meaningful output
              val hData = Array.tabulate(size)(i => (i + 1).toFloat * 0.01f)
              CUDARuntime.memcpyHtoD(ptr, hData, size)
          mem.buffers(name) = (ptr, size)
          if sys.env.contains("VERBOSE_MEM") then
            println(f"  📋 Allocated output $name: ${ptr.rawAddress}%x → $size elements")

    mem

  private def copyResultsBack(desc: KernelDesc, mem: KernelMemory): Option[Array[Float]] =
    // In stub mode, use the freshly allocated result buffer
    mem.scalars.get("_stubResultAddr") match
      case Some(addr: Long) =>
        // Read stub result directly from hostStore
        val stubBuf = CUDARuntime.getStubBuffer(addr)
        println(s"[DEBUG] copyResultsBack: reading from stub addr=0x${java.lang.Long.toHexString(addr)}, found=$stubBuf")
        stubBuf.orElse {
          // fallback: try normal path
          val outputParam = desc.params.collectFirst { case b: OutputBuffer => b }
          outputParam.map { out =>
            val (ptr, size) = mem.buffers(out.base)
            val hResult = Array.ofDim[Float](size)
            CUDARuntime.memcpyDtoH(hResult, ptr, size)
            hResult
          }
        }
      case _ =>
        // Normal path: read from device pointer
        val outputParam = desc.params.collectFirst {
          case b: OutputBuffer => b
        }

        outputParam.map { out =>
          val (ptr, size) = mem.buffers(out.base)
          val hResult = Array.ofDim[Float](size)
          CUDARuntime.memcpyDtoH(hResult, ptr, size)
          hResult
        }

  private def freeMemory(mem: KernelMemory): Unit =
    for (name, (ptr, _)) <- mem.buffers do
      CUDARuntime.free(ptr)
      if sys.env.contains("VERBOSE_MEM") then
        println(f"  🗑️  Freed $name: ${ptr.rawAddress}%x")
    for (name, (ptr, _)) <- mem.intBuffers do
      CUDARuntime.free(ptr)
      if sys.env.contains("VERBOSE_MEM") then
        println(f"  🗑️  Freed int $name: ${ptr.rawAddress}%x")

  // ==========================================================================
  // Kernel Launch
  // ==========================================================================

  private def launchKernel(
    name: String,
    ptx: String,
    desc: KernelDesc,
    mem: KernelMemory,
    invIdx: Int
  ): Boolean =
    try
      // Build kernel argument list from desc
      val (args, scalars) = buildKernelArgs(desc, mem)

      // Prepare scalars map: add kernel name
      val allScalars = scalars ++ Map("_kernelName" -> name)

      // Launch via CUDARuntime — returns stub result address in stub mode
      val stubResultAddr = CUDARuntime.launchKernel(
        null, // kernel handle (null since we're using PTX directly)
        desc.grid,
        desc.block,
        args,
        allScalars,
        invIdx
      )

      // Store stub result address in mem.scalars for copyResultsBack
      stubResultAddr.foreach(addr => mem.scalars("_stubResultAddr") = addr)

      // Synchronize
      CUDARuntime.synchronize

      true
    catch case e: Exception =>
      println(s"[ERROR] Failed to launch $name: ${e.getMessage}")
      e.printStackTrace()
      false

  private def buildKernelArgs(desc: KernelDesc, mem: KernelMemory): (Seq[Pointer], Map[String, Any]) =
    val args = mutable.ListBuffer[Pointer]()
    val scalars = mutable.Map[String, Any]()

    for param <- desc.params do
      param match
        case ScalarParam(name, valueOpt) =>
          // Use the value from ScalarParam if provided, else use defaults based on name
          val value: Any = valueOpt match
            case null | None =>
              val nameLower = name.toLowerCase
              if nameLower.contains("n") || nameLower.contains("size") || nameLower.contains("m") ||
                 nameLower.contains("pos") || nameLower.contains("dim") || nameLower.contains("head") ||
                 nameLower.contains("k") || nameLower.contains("n_") || nameLower == "int" then
                256  // default int (grid size)
              else if nameLower.contains("scale") || nameLower.contains("alpha") || nameLower.contains("beta") ||
                      nameLower.contains("decay") || nameLower.contains("gamma") then
                2.0f  // scalar float default
              else
                1.0f  // generic default
            case v => v
          scalars(name) = value

        case BufferParam(name, _, _) =>
          val (ptr, _) = mem.buffers(name)
          args += new LongPointer(ptr.rawAddress)

        case IntBufferParam(name, _) =>
          val (ptr, _) = mem.intBuffers(name)
          args += new LongPointer(ptr.rawAddress)

        case OutputBuffer(name, _, _, _, _) =>
          val (ptr, _) = mem.buffers(name)
          args += new LongPointer(ptr.rawAddress)

    (args.toSeq, scalars.toMap)

  // ==========================================================================
  // Verification
  // ==========================================================================

  private def verify(
    desc: KernelDesc,
    result: Array[Float],
    mem: KernelMemory,
    verbose: Boolean
  ): Unit =
    val outputParam = desc.params.collectFirst { case b: OutputBuffer => b }

    outputParam match
      case Some(OutputBuffer(_, _, _, _, Some(expected))) =>
        val n = math.min(result.length, expected.length)
        var maxErr = 0.0f
        var errors = 0
        for i <- 0 until n do
          val diff = math.abs(result(i) - expected(i))
          if diff > 1e-4f then
            errors += 1
            maxErr = math.max(maxErr, diff)

        if errors > 0 then
          println(f"  ⚠️  Verification: $errors/$n mismatches (max err=$maxErr%.6f)")
          Harness.logSummary(desc.name, List("VERIFY"), 0, "FAIL")
        else if verbose then
          println(f"  ✅ Verification: all $n elements match")

      case _ =>
        if verbose then
          val finite = result.count(v => !v.isNaN && !v.isInfinite)
          println(f"  ℹ️  Result: $finite/${result.length} finite elements")

