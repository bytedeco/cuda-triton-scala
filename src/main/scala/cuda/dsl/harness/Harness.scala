package cuda.dsl.harness

import java.io.{File, FileWriter, PrintWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/** Harness system for observing and logging CUDA DSL operations.
  * All outputs are written to harness_output directory for debugging and analysis.
  *
  * The harness follows a pipeline paradigm with these stages:
  * 1. KERNEL_GENERATION - Scala kernel code generation via macros
  * 2. CODE_TRANSLATION - Translation to CUDA C++ code
  * 3. NVRTC_COMPILATION - NVRTC compilation to PTX
  * 4. MODULE_LOADING - Loading compiled module via cuModuleLoadData
  * 5. FUNCTION_LOOKUP - Getting function handle via cuModuleGetFunction
  * 6. KERNEL_LAUNCH - Launching kernel via cuLaunchKernel
  * 7. MEMORY_OPS - Memory allocation and transfer operations
  * 8. EXECUTION_COMPLETE - Kernel execution finished
  */
object Harness {

  private val harnessDir = new File("/home/muller/IdeaProjects/cuda-dsl/harness_output")

  // Ensure harness directory exists
  harnessDir.mkdirs()

  private val timestamp: String = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))

  // Stage-specific log files
  private val kernelGenLog = new File(harnessDir, s"01_kernel_generation_$timestamp.log")
  private val codeTransLog = new File(harnessDir, s"02_code_translation_$timestamp.log")
  private val nvrtcLog = new File(harnessDir, s"03_nvrtc_compilation_$timestamp.log")
  private val moduleLog = new File(harnessDir, s"04_module_loading_$timestamp.log")
  private val functionLog = new File(harnessDir, s"05_function_lookup_$timestamp.log")
  private val launchLog = new File(harnessDir, s"06_kernel_launch_$timestamp.log")
  private val memoryLog = new File(harnessDir, s"07_memory_ops_$timestamp.log")
  private val executionLog = new File(harnessDir, s"08_execution_$timestamp.log")
  private val summaryLog = new File(harnessDir, s"09_summary_$timestamp.log")

  private val allLogs = List(
    kernelGenLog, codeTransLog, nvrtcLog, moduleLog,
    functionLog, launchLog, memoryLog, executionLog, summaryLog
  )

  // ============================================================================
  // Harness Stages
  // ============================================================================

  sealed abstract class Stage(val name: String, val logFile: File) {
    override def toString: String = name
  }

  object Stage {
    case object KERNEL_GENERATION extends Stage("KERNEL_GENERATION", kernelGenLog)
    case object CODE_TRANSLATION extends Stage("CODE_TRANSLATION", codeTransLog)
    case object NVRTC_COMPILATION extends Stage("NVRTC_COMPILATION", nvrtcLog)
    case object MODULE_LOADING extends Stage("MODULE_LOADING", moduleLog)
    case object FUNCTION_LOOKUP extends Stage("FUNCTION_LOOKUP", functionLog)
    case object KERNEL_LAUNCH extends Stage("KERNEL_LAUNCH", launchLog)
    case object MEMORY_OPS extends Stage("MEMORY_OPS", memoryLog)
    case object EXECUTION_COMPLETE extends Stage("EXECUTION_COMPLETE", executionLog)
  }

  // ============================================================================
  // Logging Methods
  // ============================================================================

  private def writeToFile(file: File, content: String): Unit = {
    val writer = new PrintWriter(new FileWriter(file, true))
    try {
      writer.println(content)
    } finally {
      writer.close()
    }
  }

  private def formatEntry(stage: Stage, kernelName: String, message: String, data: Option[String] = None): String = {
    val sb = new StringBuilder()
    sb.append(s"[${stage.name}] ${LocalDateTime.now()} | Kernel: $kernelName | $message\n")
    data.foreach { d =>
      sb.append("-" * 80 + "\n")
      sb.append(d)
      sb.append("\n" + "-" * 80 + "\n")
    }
    sb.toString()
  }

  /** Log a kernel generation event */
  def logKernelGeneration(kernelName: String, scalaCode: String, generatedCudaCode: String): Unit = {
    val entry = formatEntry(
      Stage.KERNEL_GENERATION,
      kernelName,
      "Generated CUDA code from Scala DSL",
      Some(s"SCALA_CODE:\n$scalaCode\n\nGENERATED_CUDA_CODE:\n$generatedCudaCode")
    )
    writeToFile(kernelGenLog, entry)
  }

  /** Log code translation event */
  def logCodeTranslation(kernelName: String, inputCode: String, outputCode: String): Unit = {
    val entry = formatEntry(
      Stage.CODE_TRANSLATION,
      kernelName,
      "Translated kernel code",
      Some(s"INPUT:\n$inputCode\n\nOUTPUT:\n$outputCode")
    )
    writeToFile(codeTransLog, entry)
  }

  /** Log NVRTC compilation event */
  def logNVRTCCompilation(kernelName: String, source: String, ptx: Option[String], compileOptions: Array[String], log: Option[String], success: Boolean): Unit = {
    val message = if (success) "NVRTC compilation SUCCEEDED" else "NVRTC compilation FAILED"
    val entry = formatEntry(
      Stage.NVRTC_COMPILATION,
      kernelName,
      message,
      Some(s"COMPILE_OPTIONS: ${compileOptions.mkString(", ")}\n\nPTX_CODE:\n${ptx.getOrElse("(not generated)")}\n\nCOMPILATION_LOG:\n${log.getOrElse("N/A")}")
    )
    writeToFile(nvrtcLog, entry)
  }

  /** Log module loading event (cuModuleLoadData) */
  def logModuleLoading(kernelName: String, module: String, ptxSize: Long, success: Boolean): Unit = {
    val message = if (success) "cuModuleLoadData SUCCEEDED" else "cuModuleLoadData FAILED"
    val entry = formatEntry(
      Stage.MODULE_LOADING,
      kernelName,
      message,
      Some(s"MODULE: $module\nPTX_SIZE: $ptxSize bytes")
    )
    writeToFile(moduleLog, entry)
  }

  /** Log function lookup event (cuModuleGetFunction) */
  def logFunctionLookup(kernelName: String, moduleName: String, functionName: String, success: Boolean): Unit = {
    val message = if (success) "cuModuleGetFunction SUCCEEDED" else "cuModuleGetFunction FAILED"
    val entry = formatEntry(
      Stage.FUNCTION_LOOKUP,
      kernelName,
      message,
      Some(s"MODULE: $moduleName\nFUNCTION: $functionName")
    )
    writeToFile(functionLog, entry)
  }

  /** Log kernel launch event (cuLaunchKernel) */
  def logKernelLaunch(kernelName: String, function: String, grid: String, block: String, sharedMem: Long, stream: String, success: Boolean): Unit = {
    val message = if (success) "cuLaunchKernel SUCCEEDED" else "cuLaunchKernel FAILED"
    val entry = formatEntry(
      Stage.KERNEL_LAUNCH,
      kernelName,
      message,
      Some(s"FUNCTION: $function\nGRID: $grid\nBLOCK: $block\nSHARED_MEM: $sharedMem bytes\nSTREAM: $stream")
    )
    writeToFile(launchLog, entry)
  }

  /** Log memory operation */
  def logMemoryOp(op: String, ptr: Long, size: Long, device: Int, success: Boolean): Unit = {
    val kernelName = op
    val message = if (success) s"$op SUCCEEDED" else s"$op FAILED"
    val entry = formatEntry(
      Stage.MEMORY_OPS,
      kernelName,
      message,
      Some(s"PTR: 0x${java.lang.Long.toHexString(ptr)}\nSIZE: $size bytes\nDEVICE: $device")
    )
    writeToFile(memoryLog, entry)
  }

  /** Log memory copy operation */
  def logMemcpy(kind: String, dst: Long, src: Long, size: Long, success: Boolean): Unit = {
    val entry = formatEntry(
      Stage.MEMORY_OPS,
      "memcpy",
      s"$kind ${if (success) "SUCCEEDED" else "FAILED"}",
      Some(s"DST: 0x${java.lang.Long.toHexString(dst)}\nSRC: 0x${java.lang.Long.toHexString(src)}\nSIZE: $size bytes")
    )
    writeToFile(memoryLog, entry)
  }

  /** Log execution completion */
  def logExecutionComplete(kernelName: String, durationMs: Long, grid: String, block: String, blocksCompleted: Int): Unit = {
    val entry = formatEntry(
      Stage.EXECUTION_COMPLETE,
      kernelName,
      s"Execution completed in ${durationMs}ms",
      Some(s"GRID: $grid\nBLOCK: $block\nBLOCKS_COMPLETED: $blocksCompleted\nDURATION: $durationMs ms")
    )
    writeToFile(executionLog, entry)
  }

  /** Write summary entry */
  def logSummary(kernelName: String, stages: List[String], totalTimeMs: Long, status: String): Unit = {
    val entry = formatEntry(
      Stage.EXECUTION_COMPLETE,
      kernelName,
      s"SUMMARY | Status: $status | Total Time: ${totalTimeMs}ms",
      Some(s"STAGES:\n${stages.zipWithIndex.map { case (s, i) => s"  ${i + 1}. $s" }.mkString("\n")}")
    )
    writeToFile(summaryLog, entry)
  }

  // ============================================================================
  // Global Summary
  // ============================================================================

  private val globalSummary = new File(harnessDir, "global_summary.md")

  def logGlobalSummary(operation: String, details: Map[String, String]): Unit = {
    val sb = new StringBuilder()
    sb.append(s"\n## $operation - ${LocalDateTime.now()}\n")
    details.foreach { case (k, v) =>
      sb.append(s"- **$k**: $v\n")
    }
    writeToFile(globalSummary, sb.toString())
  }

  // ============================================================================
  // Clear Logs
  // ============================================================================

  def clearAllLogs(): Unit = {
    allLogs.foreach { f =>
      if (f.exists()) f.delete()
    }
  }
}
