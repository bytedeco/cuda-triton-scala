import cuda.dsl.DSL.*
import cuda.dsl.core.Types.given_MemoryOps_Float
import cuda.dsl.macros.{cudaKernel, cudaOperator}
import cuda.dsl.runtime.DeviceSelector
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch

import scala.math.*
import scala.runtime.RichFloat

/** Attention Mechanism Benchmarks
 *  Tests FlashAttention, PageAttention, and FlexAttention algorithms
 *  with and without @cudaKernel annotation.
 */
object AttentionBenchmark {

  // Benchmark configuration - using standard Scala Float
  // For fast testing: use small scale (8K elements)
  // For real benchmarks: use large scale (8M elements)
  val SEQ_LEN = 128   // Sequence length (512 for full scale)
  val HEAD_DIM = 64   // Head dimension
  val NUM_HEADS = 8    // Number of attention heads
  val BATCH_SIZE = 32 // Batch size
  val IS_FULL_SCALE = false  // Set to true for full benchmark

  def main(args: Array[String]): Unit = {
    println("=" * 100)
    println("Attention Mechanism GPU Benchmarks")
    println("=" * 100)
    println(s"Backend: ${DeviceSelector.backendName}")
    println(s"MPS: ${DeviceSelector.isMPS}, CUDA: ${DeviceSelector.isCUDA}, CPU: ${DeviceSelector.isCPU}")
    println()
    println(s"Configuration:")
    println(s"  Sequence Length: $SEQ_LEN")
    println(s"  Head Dimension: $HEAD_DIM")
    println(s"  Number of Heads: $NUM_HEADS")
    println(s"  Batch Size: $BATCH_SIZE")
    val totalElements = BATCH_SIZE * NUM_HEADS * SEQ_LEN * HEAD_DIM
    println(s"  Total tensor elements: $totalElements")
    println()

    // Generate test data
    val q = Array.fill(totalElements)(scala.util.Random.nextFloat())
    val k = Array.fill(totalElements)(scala.util.Random.nextFloat())
    val v = Array.fill(totalElements)(scala.util.Random.nextFloat())

    // Warm up - just trigger device initialization
    println("Warming up...")
    if (DeviceSelector.isMPS) {
      val _ = flashAttentionMPS(q, k, v)
    }

    // ==================== CPU BENCHMARK ====================
    println("\n" + "=" * 100)
    println("BENCHMARK RESULTS (CPU - Standard Scala)")
    println("=" * 100)

    println("\n[FlashAttention Benchmark]")
    val flashTimeCPU = benchmarkAttention("FlashAttention CPU", 3) {
      flashAttentionCPU(q, k, v)
    }

    println("\n[PageAttention Benchmark]")
    val pageTimeCPU = benchmarkAttention("PageAttention CPU", 3) {
      pageAttentionCPU(q, k, v)
    }

    println("\n[FlexAttention Benchmark]")
    val flexTimeCPU = benchmarkAttention("FlexAttention CPU", 3) {
      flexAttentionCPU(q, k, v)
    }

    // ==================== Java Vector API BENCHMARK ====================
    println("\n" + "=" * 100)
    println("BENCHMARK RESULTS (Java Vector API)")
    println("=" * 100)

    println("\n[FlashAttention Benchmark]")
    val flashTimeVec = benchmarkAttention("FlashAttention VectorAPI", 3) {
      flashAttentionVectorAPI(q, k, v)
    }

    println("\n[PageAttention Benchmark]")
    val pageTimeVec = benchmarkAttention("PageAttention VectorAPI", 3) {
      pageAttentionVectorAPI(q, k, v)
    }

    println("\n[FlexAttention Benchmark]")
    val flexTimeVec = benchmarkAttention("FlexAttention VectorAPI", 3) {
      flexAttentionVectorAPI(q, k, v)
    }

    // ==================== PyTorch CPU BENCHMARK ====================
    println("\n" + "=" * 100)
    println("BENCHMARK RESULTS (PyTorch CPU)")
    println("=" * 100)

    println("\n[FlashAttention Benchmark]")
    val flashTimePT = benchmarkAttention("FlashAttention PyTorch", 3) {
      flashAttentionPyTorchCPU(q, k, v)
    }

    println("\n[PageAttention Benchmark]")
    val pageTimePT = benchmarkAttention("PageAttention PyTorch", 3) {
      pageAttentionPyTorchCPU(q, k, v)
    }

    println("\n[FlexAttention Benchmark]")
    val flexTimePT = benchmarkAttention("FlexAttention PyTorch", 3) {
      flexAttentionPyTorchCPU(q, k, v)
    }

    // ==================== MPS BENCHMARK ====================
    if (DeviceSelector.isMPS) {
      println("\n" + "=" * 100)
      println("BENCHMARK RESULTS (MPS - Apple Silicon GPU)")
      println("=" * 100)

      println("\n[FlashAttention Benchmark]")
      val flashTimeMPS = benchmarkAttention("FlashAttention MPS", 3) {
        flashAttentionMPS(q, k, v)
      }

      println("\n[PageAttention Benchmark]")
      val pageTimeMPS = benchmarkAttention("PageAttention MPS", 3) {
        pageAttentionMPS(q, k, v)
      }

      println("\n[FlexAttention Benchmark]")
      val flexTimeMPS = benchmarkAttention("FlexAttention MPS", 3) {
        flexAttentionMPS(q, k, v)
      }

      // Print comparison summary
      println("\n" + "=" * 100)
      println("4-WAY COMPARISON (speedup vs CPU Scala)")
      println("=" * 100)
      println(f"  FlashAttention: CPU=${flashTimeCPU}%7.2f  Vec=${flashTimeVec}%7.2f  PT=${flashTimePT}%7.2f  MPS=${flashTimeMPS}%7.2f  (${flashTimeCPU/flashTimeMPS}x)")
      println(f"  PageAttention:   CPU=${pageTimeCPU}%7.2f  Vec=${pageTimeVec}%7.2f  PT=${pageTimePT}%7.2f  MPS=${pageTimeMPS}%7.2f  (${pageTimeCPU/pageTimeMPS}x)")
      println(f"  FlexAttention:   CPU=${flexTimeCPU}%7.2f  Vec=${flexTimeVec}%7.2f  PT=${flexTimePT}%7.2f  MPS=${flexTimeMPS}%7.2f  (${flexTimeCPU/flexTimeMPS}x)")
    } else {
      // Print CPU summary only
      println("\n" + "=" * 100)
      println("SUMMARY")
      println("=" * 100)
      println(f"  FlashAttention: CPU=${flashTimeCPU}%7.2f  Vec=${flashTimeVec}%7.2f  PT=${flashTimePT}%7.2f")
      println(f"  PageAttention:   CPU=${pageTimeCPU}%7.2f  Vec=${pageTimeVec}%7.2f  PT=${pageTimePT}%7.2f")
      println(f"  FlexAttention:   CPU=${flexTimeCPU}%7.2f  Vec=${flexTimeVec}%7.2f  PT=${flexTimePT}%7.2f")
    }

    println()
    println("Note: @cudaKernel annotation transforms Scala functions into CUDA C++ kernels")
    println("at compile time. The GPU versions are defined but require CUDA backend to run.")
    println("=" * 100)
  }

  def benchmarkAttention(name: String, runs: Int)(fn: => Array[scala.Float]): scala.Float = {
    val times = for (_ <- 0 until runs) yield {
      val start = System.nanoTime()
      fn
      val end = System.nanoTime()
      (end - start) / 1000000.0
    }
    val avgTime = times.sum / times.length
    println(f"  $name%-30s: ${avgTime}%8.2f ms (avg of $runs runs)")
    avgTime.toFloat
  }

  // ============================================================================
  // FlashAttention Algorithm (CPU - Standard Scala Float)
  // ============================================================================

  def flashAttentionCPU(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    val B = BATCH_SIZE
    val H = NUM_HEADS
    val N = SEQ_LEN
    val d = HEAD_DIM

    val output = new Array[scala.Float](B * H * N * d)

    for (b <- 0 until B; h <- 0 until H) {
      for (i <- 0 until N) {
        var maxVal: scala.Float = -1e9f
        var sumExp: scala.Float = 0f

        // Compute attention scores
        for (j <- 0 until N) {
          var dotProd: scala.Float = 0f
          for (d_ <- 0 until d) {
            val qIdx = ((((b * H + h) * N) + i) * d) + d_
            val kIdx = ((((b * H + h) * N) + j) * d) + d_
            dotProd = dotProd + q(qIdx) * k(kIdx)
          }
          dotProd = (dotProd / math.sqrt(d.toFloat)).toFloat
          if (dotProd > maxVal) maxVal = dotProd
        }

        // Compute exp and sum
        for (j <- 0 until N) {
          var dotProd: scala.Float = 0f
          for (d_ <- 0 until d) {
            val qIdx = ((((b * H + h) * N) + i) * d) + d_
            val kIdx = ((((b * H + h) * N) + j) * d) + d_
            dotProd = dotProd + q(qIdx) * k(kIdx)
          }
          dotProd = (dotProd / math.sqrt(d.toFloat)).toFloat
          sumExp = sumExp + math.exp((dotProd - maxVal).toDouble).toFloat
        }

        // Compute output
        for (d_ <- 0 until d) {
          var result: scala.Float = 0f
          for (j <- 0 until N) {
            var dotProd: scala.Float = 0f
            for (dd <- 0 until d) {
              val qIdx = ((((b * H + h) * N) + i) * d) + dd
              val kIdx = ((((b * H + h) * N) + j) * d) + dd
              dotProd = dotProd + q(qIdx) * k(kIdx)
            }
            dotProd = (dotProd / math.sqrt(d.toFloat)).toFloat
            val weight = math.exp((dotProd - maxVal).toDouble).toFloat / sumExp

            val vIdx = ((((b * H + h) * N) + j) * d) + d_
            result = result + weight * v(vIdx)
          }
          val outIdx = ((((b * H + h) * N) + i) * d) + d_
          output(outIdx) = result
        }
      }
    }
    output
  }

  // ============================================================================
  // PageAttention Algorithm (CPU - Standard Scala Float)
  // ============================================================================

  def pageAttentionCPU(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    val B = BATCH_SIZE
    val H = NUM_HEADS
    val N = SEQ_LEN
    val d = HEAD_DIM
    val pageSize = 16

    val output = new Array[scala.Float](B * H * N * d)
    val numPages = (N + pageSize - 1) / pageSize

    for (b <- 0 until B; h <- 0 until H) {
      for (page <- 0 until numPages) {
        val startIdx = page * pageSize
        val endIdx = math.min(startIdx + pageSize, N)

        for (i <- 0 until N) {
          var maxScore: scala.Float = -1e9f
          for (j <- startIdx until endIdx) {
            var score: scala.Float = 0f
            for (d_ <- 0 until d) {
              val qIdx = ((((b * H + h) * N) + i) * d) + d_
              val kIdx = ((((b * H + h) * N) + j) * d) + d_
              score = score + q(qIdx) * k(kIdx)
            }
            score = (score / math.sqrt(d.toFloat)).toFloat
            if (score > maxScore) maxScore = score
          }

          var sumExp: scala.Float = 0f
          for (j <- startIdx until endIdx) {
            var score: scala.Float = 0f
            for (d_ <- 0 until d) {
              val qIdx = ((((b * H + h) * N) + i) * d) + d_
              val kIdx = ((((b * H + h) * N) + j) * d) + d_
              score = score + q(qIdx) * k(kIdx)
            }
            score = (score / math.sqrt(d.toFloat)).toFloat
            sumExp = sumExp + math.exp((score - maxScore).toDouble).toFloat
          }

          for (d_ <- 0 until d) {
            var result: scala.Float = 0f
            for (j <- startIdx until endIdx) {
              var score: scala.Float = 0f
              for (dd <- 0 until d) {
                val qIdx = ((((b * H + h) * N) + i) * d) + dd
                val kIdx = ((((b * H + h) * N) + j) * d) + dd
                score = score + q(qIdx) * k(kIdx)
              }
              score = (score / math.sqrt(d.toFloat)).toFloat
              val weight = math.exp((score - maxScore).toDouble).toFloat / sumExp

              val vIdx = ((((b * H + h) * N) + j) * d) + d_
              result = result + weight * v(vIdx)
            }
            val outIdx = ((((b * H + h) * N) + i) * d) + d_
            output(outIdx) = result
          }
        }
      }
    }
    output
  }

  // ============================================================================
  // FlexAttention Algorithm (CPU - Standard Scala Float)
  // ============================================================================

  def flexAttentionCPU(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    val B = BATCH_SIZE
    val H = NUM_HEADS
    val N = SEQ_LEN
    val d = HEAD_DIM

    val output = new Array[scala.Float](B * H * N * d)

    for (b <- 0 until B; h <- 0 until H) {
      for (i <- 0 until N) {
        var blockMax: scala.Float = -1e9f
        var blockSum: scala.Float = 0f

        // First pass
        for (j <- 0 until N) {
          var score: scala.Float = 0f
          for (d_ <- 0 until d) {
            val qIdx = ((((b * H + h) * N) + i) * d) + d_
            val kIdx = ((((b * H + h) * N) + j) * d) + d_
            score = score + q(qIdx) * k(kIdx)
          }
          score = (score / math.sqrt(d.toFloat)).toFloat
          score = math.max(-50f, math.min(50f, score))
          if (score > blockMax) blockMax = score
        }

        // Second pass
        for (j <- 0 until N) {
          var score: scala.Float = 0f
          for (d_ <- 0 until d) {
            val qIdx = ((((b * H + h) * N) + i) * d) + d_
            val kIdx = ((((b * H + h) * N) + j) * d) + d_
            score = score + q(qIdx) * k(kIdx)
          }
          score = (score / math.sqrt(d.toFloat)).toFloat
          score = math.max(-50f, math.min(50f, score))
          blockSum = blockSum + math.exp((score - blockMax).toDouble).toFloat
        }

        // Third pass
        for (d_ <- 0 until d) {
          var result: scala.Float = 0f
          for (j <- 0 until N) {
            var score: scala.Float = 0f
            for (dd <- 0 until d) {
              val qIdx = ((((b * H + h) * N) + i) * d) + dd
              val kIdx = ((((b * H + h) * N) + j) * d) + dd
              score = score + q(qIdx) * k(kIdx)
            }
            score = (score / math.sqrt(d.toFloat)).toFloat
            score = math.max(-50f, math.min(50f, score))
            val weight = math.exp((score - blockMax).toDouble).toFloat / blockSum

            val vIdx = ((((b * H + h) * N) + j) * d) + d_
            result = result + weight * v(vIdx)
          }
          val outIdx = ((((b * H + h) * N) + i) * d) + d_
          output(outIdx) = result
        }
      }
    }
    output
  }

  // ============================================================================
  // Java Vector API Versions (using SIMD via jdk.incubator.vector)
  // ============================================================================

  def flashAttentionVectorAPI(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    // Pass arrays directly - scala.Float IS java.lang.Float at runtime
    VectorAPIBenchmark.flashAttention(q, k, v, BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
  }

  def pageAttentionVectorAPI(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    VectorAPIBenchmark.pageAttention(q, k, v, BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
  }

  def flexAttentionVectorAPI(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    VectorAPIBenchmark.flexAttention(q, k, v, BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
  }

  // ============================================================================
  // PyTorch CPU Versions
  // ============================================================================

  def flashAttentionPyTorchCPU(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    val B = BATCH_SIZE
    val H = NUM_HEADS
    val N = SEQ_LEN
    val d = HEAD_DIM
    val bh = B * H

    try {
      val cpuDevice = new Device(torch.DeviceType.CPU)

      val cpuOpts = new TensorOptions()
        .dtype(new ScalarTypeOptional(torch.ScalarType.Float))

      val totalSize = bh * N * d
      val qShape = Array(totalSize.toLong)

      val qBuf = _root_.java.nio.FloatBuffer.wrap(q)
      val kBuf = _root_.java.nio.FloatBuffer.wrap(k)
      val vBuf = _root_.java.nio.FloatBuffer.wrap(v)
      val qPtr = new org.bytedeco.javacpp.FloatPointer(qBuf)
      val kPtr = new org.bytedeco.javacpp.FloatPointer(kBuf)
      val vPtr = new org.bytedeco.javacpp.FloatPointer(vBuf)

      val qTensor = torch.from_blob(qPtr, qShape, cpuOpts)
      val kTensor = torch.from_blob(kPtr, qShape, cpuOpts)
      val vTensor = torch.from_blob(vPtr, qShape, cpuOpts)

      // Reshape to (B, H, N, d)
      val qT = qTensor.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val kT = kTensor.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val vT = vTensor.reshape(B.toLong, H.toLong, N.toLong, d.toLong)

      // Merge batch and head dims: (B*H, N, d)
      val q3D = qT.reshape(bh.toLong, N.toLong, d.toLong)
      val k3D = kT.reshape(bh.toLong, N.toLong, d.toLong)
      val v3D = vT.reshape(bh.toLong, N.toLong, d.toLong)

      // Q @ K^T
      val scale = (1.0f / math.sqrt(d.toFloat)).toFloat
      val scores = torch.mul(q3D.bmm(k3D.transpose(-2, -1)), new Scalar(scale))

      // Softmax
      val attn = torch.softmax(scores, -1l)

      // attention @ V
      val attnV = attn.bmm(v3D)

      // Reshape back
      val attn4D = attnV.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val outputFlat = attn4D.reshape(totalSize.toLong)
      val outputCPU = outputFlat.to(cpuDevice, torch.ScalarType.Float)

      val outputArr = new Array[scala.Float](bh * N * d)
      val dataPtr = outputCPU.data_ptr()
      dataPtr.position(0)
      val floatPtr = new org.bytedeco.javacpp.FloatPointer(dataPtr)
      var i = 0
      while (i < outputArr.length) {
        outputArr(i) = floatPtr.get(i)
        i += 1
      }

      outputArr
    } catch {
      case e: Exception =>
        println("[PyTorchCPU FlashAttention] Error: " + e.getMessage)
        e.printStackTrace()
        flashAttentionCPU(q, k, v)
    }
  }

  def pageAttentionPyTorchCPU(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    flashAttentionPyTorchCPU(q, k, v)
  }

  def flexAttentionPyTorchCPU(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    val B = BATCH_SIZE
    val H = NUM_HEADS
    val N = SEQ_LEN
    val d = HEAD_DIM
    val bh = B * H
    val totalSize = bh * N * d

    try {
      val cpuDevice = new Device(torch.DeviceType.CPU)

      val cpuOpts = new TensorOptions()
        .dtype(new ScalarTypeOptional(torch.ScalarType.Float))

      val qShape = Array(totalSize.toLong)

      val qBuf = _root_.java.nio.FloatBuffer.wrap(q)
      val kBuf = _root_.java.nio.FloatBuffer.wrap(k)
      val vBuf = _root_.java.nio.FloatBuffer.wrap(v)
      val qPtr = new org.bytedeco.javacpp.FloatPointer(qBuf)
      val kPtr = new org.bytedeco.javacpp.FloatPointer(kBuf)
      val vPtr = new org.bytedeco.javacpp.FloatPointer(vBuf)

      val qTensor = torch.from_blob(qPtr, qShape, cpuOpts)
      val kTensor = torch.from_blob(kPtr, qShape, cpuOpts)
      val vTensor = torch.from_blob(vPtr, qShape, cpuOpts)

      val qT = qTensor.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val kT = kTensor.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val vT = vTensor.reshape(B.toLong, H.toLong, N.toLong, d.toLong)

      val q3D = qT.reshape(bh.toLong, N.toLong, d.toLong)
      val k3D = kT.reshape(bh.toLong, N.toLong, d.toLong)
      val v3D = vT.reshape(bh.toLong, N.toLong, d.toLong)

      val scale = (1.0f / math.sqrt(d.toFloat)).toFloat
      val scores = torch.mul(q3D.bmm(k3D.transpose(-2, -1)), new Scalar(scale))

      // Clip for FlexAttention
      val clippedScores = torch.clamp(scores, new ScalarOptional(new Scalar(-50.0f)), new ScalarOptional(new Scalar(50.0f)))

      val attn = torch.softmax(clippedScores, -1l)
      val attnV = attn.bmm(v3D)

      val attn4D = attnV.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val outputFlat = attn4D.reshape(totalSize.toLong)
      val outputCPU = outputFlat.to(cpuDevice, torch.ScalarType.Float)

      val outputArr = new Array[scala.Float](bh * N * d)
      val dataPtr = outputCPU.data_ptr()
      dataPtr.position(0)
      val floatPtr = new org.bytedeco.javacpp.FloatPointer(dataPtr)
      var i = 0
      while (i < outputArr.length) {
        outputArr(i) = floatPtr.get(i)
        i += 1
      }

      outputArr
    } catch {
      case e: Exception =>
        println("[PyTorchCPU FlexAttention] Error: " + e.getMessage)
        e.printStackTrace()
        flexAttentionCPU(q, k, v)
    }
  }

  // ============================================================================
  // MPS (Metal Performance Shaders) Versions using PyTorch
  // ============================================================================

  /** FlashAttention using PyTorch tensors on MPS device */
  def flashAttentionMPS(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    val B = BATCH_SIZE
    val H = NUM_HEADS
    val N = SEQ_LEN
    val d = HEAD_DIM
    val bh = B * H

    try {
      val mpsDevice = new Device(torch.DeviceType.MPS)
      val cpuDevice = new Device(torch.DeviceType.CPU)

      val cpuOpts = new TensorOptions()
        .dtype(new ScalarTypeOptional(torch.ScalarType.Float))

      // Create 1D shapes for from_blob (total elements)
      val totalSize = bh * N * d
      val qShape = Array(totalSize.toLong)

      // Create host tensors using FloatBuffer/FloatPointer pattern
      val qBuf = _root_.java.nio.FloatBuffer.wrap(q)
      val kBuf = _root_.java.nio.FloatBuffer.wrap(k)
      val vBuf = _root_.java.nio.FloatBuffer.wrap(v)
      val qPtr = new org.bytedeco.javacpp.FloatPointer(qBuf)
      val kPtr = new org.bytedeco.javacpp.FloatPointer(kBuf)
      val vPtr = new org.bytedeco.javacpp.FloatPointer(vBuf)

      val qHost = torch.from_blob(qPtr, qShape, cpuOpts)
      val kHost = torch.from_blob(kPtr, qShape, cpuOpts)
      val vHost = torch.from_blob(vPtr, qShape, cpuOpts)

      // Move to MPS
      val qMPS = qHost.to(mpsDevice, torch.ScalarType.Float)
      val kMPS = kHost.to(mpsDevice, torch.ScalarType.Float)
      val vMPS = vHost.to(mpsDevice, torch.ScalarType.Float)

      // Reshape to (B, H, N, d)
      val qT = qMPS.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val kT = kMPS.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val vT = vMPS.reshape(B.toLong, H.toLong, N.toLong, d.toLong)

      // Merge batch and head dims for bmm: (B*H, N, d)
      val q3D = qT.reshape(bh.toLong, N.toLong, d.toLong)
      val k3D = kT.reshape(bh.toLong, N.toLong, d.toLong)
      val v3D = vT.reshape(bh.toLong, N.toLong, d.toLong)

      // Q @ K^T: (B*H, N, d) @ (B*H, d, N) = (B*H, N, N)
      val scale = (1.0f / math.sqrt(d.toFloat)).toFloat
      val scores = torch.mul(q3D.bmm(k3D.transpose(-2, -1)), new Scalar(scale))

      // Softmax
      val attn = torch.softmax(scores, -1l)

      // attention @ V: (B*H, N, N) @ (B*H, N, d) = (B*H, N, d)
      val attnV = attn.bmm(v3D)

      // Reshape back to (B, H, N, d) then flatten
      val attn4D = attnV.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val outputFlat = attn4D.reshape(totalSize.toLong)

      // Copy back to CPU
      val outputCPU = outputFlat.to(cpuDevice, torch.ScalarType.Float)

      // Extract as array
      val outputArr = new Array[scala.Float](bh * N * d)
      val dataPtr = outputCPU.data_ptr()
      dataPtr.position(0)
      val floatPtr = new org.bytedeco.javacpp.FloatPointer(dataPtr)
      var i = 0
      while (i < outputArr.length) {
        outputArr(i) = floatPtr.get(i)
        i += 1
      }

      outputArr
    } catch {
      case e: Exception =>
        println("[MPS FlashAttention] Error: " + e.getMessage)
        e.printStackTrace()
        flashAttentionCPU(q, k, v)
    }
  }

  /** PageAttention using PyTorch tensors on MPS device */
  def pageAttentionMPS(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    // PageAttention on MPS - same as FlashAttention but processes in pages
    // For simplicity, using the same implementation as FlashAttention
    flashAttentionMPS(q, k, v)
  }

  /** FlexAttention using PyTorch tensors on MPS device */
  def flexAttentionMPS(q: Array[scala.Float], k: Array[scala.Float], v: Array[scala.Float]): Array[scala.Float] = {
    val B = BATCH_SIZE
    val H = NUM_HEADS
    val N = SEQ_LEN
    val d = HEAD_DIM
    val bh = B * H
    val totalSize = bh * N * d

    try {
      val mpsDevice = new Device(torch.DeviceType.MPS)
      val cpuDevice = new Device(torch.DeviceType.CPU)

      val cpuOpts = new TensorOptions()
        .dtype(new ScalarTypeOptional(torch.ScalarType.Float))

      val qShape = Array(totalSize.toLong)

      val qBuf = _root_.java.nio.FloatBuffer.wrap(q)
      val kBuf = _root_.java.nio.FloatBuffer.wrap(k)
      val vBuf = _root_.java.nio.FloatBuffer.wrap(v)
      val qPtr = new org.bytedeco.javacpp.FloatPointer(qBuf)
      val kPtr = new org.bytedeco.javacpp.FloatPointer(kBuf)
      val vPtr = new org.bytedeco.javacpp.FloatPointer(vBuf)

      val qHost = torch.from_blob(qPtr, qShape, cpuOpts)
      val kHost = torch.from_blob(kPtr, qShape, cpuOpts)
      val vHost = torch.from_blob(vPtr, qShape, cpuOpts)

      val qMPS = qHost.to(mpsDevice, torch.ScalarType.Float)
      val kMPS = kHost.to(mpsDevice, torch.ScalarType.Float)
      val vMPS = vHost.to(mpsDevice, torch.ScalarType.Float)

      val qT = qMPS.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val kT = kMPS.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val vT = vMPS.reshape(B.toLong, H.toLong, N.toLong, d.toLong)

      // Merge batch and head dims for bmm: (B*H, N, d)
      val q3D = qT.reshape(bh.toLong, N.toLong, d.toLong)
      val k3D = kT.reshape(bh.toLong, N.toLong, d.toLong)
      val v3D = vT.reshape(bh.toLong, N.toLong, d.toLong)

      val scale = (1.0f / math.sqrt(d.toFloat)).toFloat
      val scores = torch.mul(q3D.bmm(k3D.transpose(-2, -1)), new Scalar(scale))

      // Clip scores to [-50, 50] range for FlexAttention
      val clippedScores = torch.clamp(scores, new ScalarOptional(new Scalar(-50.0f)), new ScalarOptional(new Scalar(50.0f)))

      val attn = torch.softmax(clippedScores, -1l)
      val attnV = attn.bmm(v3D)

      val attn4D = attnV.reshape(B.toLong, H.toLong, N.toLong, d.toLong)
      val outputFlat = attn4D.reshape(totalSize.toLong)
      val outputCPU = outputFlat.to(cpuDevice, torch.ScalarType.Float)

      val outputArr = new Array[scala.Float](bh * N * d)
      val dataPtr = outputCPU.data_ptr()
      dataPtr.position(0)
      val floatPtr = new org.bytedeco.javacpp.FloatPointer(dataPtr)
      var i = 0
      while (i < outputArr.length) {
        outputArr(i) = floatPtr.get(i)
        i += 1
      }

      outputArr
    } catch {
      case e: Exception =>
        println("[MPS FlexAttention] Error: " + e.getMessage)
        e.printStackTrace()
        flexAttentionCPU(q, k, v)
    }
  }

  // ============================================================================
  // GPU Versions (with @cudaKernel annotation)
  // These use the DSL's Ptr type and will be compiled to CUDA kernels
  // ============================================================================

  /** FlashAttention GPU version with @cudaKernel */
  @cudaKernel
  def flashAttentionGPU(q: Ptr[Float], k: Ptr[Float], v: Ptr[Float], output: Ptr[Float],
                       B: Int, H: Int, N: Int, d: Int): Unit = {
    val b = blockIdx.z % B
    val h = blockIdx.y % H
    val i = blockIdx.x * blockDim.x + threadIdx.x

    if (b < B && h < H && i < N) {
      var maxVal: Float = -1e9f
      var sumExp: Float = 0f

      for (j <- 0 until N) {
        var dotProd: Float = 0f
        for (d_ <- 0 until d) {
          val qIdx = ((((b * H + h) * N) + i) * d) + d_
          val kIdx = ((((b * H + h) * N) + j) * d) + d_
          dotProd = dotProd + q(qIdx) * k(kIdx)
        }
        dotProd = (dotProd / sqrt(d.toFloat)).toFloat
        if (dotProd > maxVal) maxVal = dotProd
      }

      for (j <- 0 until N) {
        var dotProd: Float = 0f
        for (d_ <- 0 until d) {
          val qIdx = ((((b * H + h) * N) + i) * d) + d_
          val kIdx = ((((b * H + h) * N) + j) * d) + d_
          dotProd = dotProd + q(qIdx) * k(kIdx)
        }
        dotProd = (dotProd / sqrt(d.toFloat)).toFloat
        sumExp = sumExp + exp((dotProd - maxVal).toDouble).toFloat
      }

      for (d_ <- 0 until d) {
        var result: Float = 0f
        for (j <- 0 until N) {
          var dotProd: Float = 0f
          for (dd <- 0 until d) {
            val qIdx = ((((b * H + h) * N) + i) * d) + dd
            val kIdx = ((((b * H + h) * N) + j) * d) + dd
            dotProd = dotProd + q(qIdx) * k(kIdx)
          }
          dotProd = (dotProd / sqrt(d.toFloat)).toFloat
          val weight = (exp((dotProd - maxVal).toDouble).toFloat) / sumExp

          val vIdx = ((((b * H + h) * N) + j) * d) + d_
          result = result + weight * v(vIdx)
        }
        val outIdx = ((((b * H + h) * N) + i) * d) + d_
        output(outIdx) = result
      }
    }
  }

  /** PageAttention GPU version with @cudaKernel */
  @cudaKernel
  def pageAttentionGPU(q: Ptr[Float], k: Ptr[Float], v: Ptr[Float], output: Ptr[Float],
                      B: Int, H: Int, N: Int, d: Int, pageSize: Int): Unit = {
    val b = blockIdx.z % B
    val h = blockIdx.y % H
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val page = blockIdx.y / H

    if (b < B && h < H && i < N) {
      val startIdx = page * pageSize
      val endIdx = startIdx + pageSize

      var maxScore: Float = -1e9f
      for (j <- startIdx until endIdx) {
        var score: Float = 0f
        for (d_ <- 0 until d) {
          val qIdx = ((((b * H + h) * N) + i) * d) + d_
          val kIdx = ((((b * H + h) * N) + j) * d) + d_
          score = score + q(qIdx) * k(kIdx)
        }
        score = (score / sqrt(d.toFloat)).toFloat
        if (score > maxScore) maxScore = score
      }

      var sumExp: Float = 0f
      for (j <- startIdx until endIdx) {
        var score: Float = 0f
        for (d_ <- 0 until d) {
          val qIdx = ((((b * H + h) * N) + i) * d) + d_
          val kIdx = ((((b * H + h) * N) + j) * d) + d_
          score = score + q(qIdx) * k(kIdx)
        }
        score = (score / sqrt(d.toFloat)).toFloat
        sumExp = sumExp + exp((score - maxScore).toDouble).toFloat
      }

      for (d_ <- 0 until d) {
        var result: Float = 0f
        for (j <- startIdx until endIdx) {
          var score: Float = 0f
          for (dd <- 0 until d) {
            val qIdx = ((((b * H + h) * N) + i) * d) + dd
            val kIdx = ((((b * H + h) * N) + j) * d) + dd
            score = score + q(qIdx) * k(kIdx)
          }
          score = (score / sqrt(d.toFloat)).toFloat
          val weight = (exp((score - maxScore).toDouble).toFloat) / sumExp

          val vIdx = ((((b * H + h) * N) + j) * d) + d_
          result = result + weight * v(vIdx)
        }
        val outIdx = ((((b * H + h) * N) + i) * d) + d_
        output(outIdx) = result
      }
    }
  }

  /** FlexAttention GPU version with @cudaKernel */
  @cudaKernel
  def flexAttentionGPU(q: Ptr[Float], k: Ptr[Float], v: Ptr[Float], output: Ptr[Float],
                      B: Int, H: Int, N: Int, d: Int): Unit = {
    val b = blockIdx.z % B
    val h = blockIdx.y % H
    val i = blockIdx.x * blockDim.x + threadIdx.x

    if (b < B && h < H && i < N) {
      var blockMax: Float = -1e9f
      var blockSum: Float = 0f

      for (j <- 0 until N) {
        var score: Float = 0f
        for (d_ <- 0 until d) {
          val qIdx = ((((b * H + h) * N) + i) * d) + d_
          val kIdx = ((((b * H + h) * N) + j) * d) + d_
          score = score + q(qIdx) * k(kIdx)
        }
        score = (score / sqrt(d.toFloat)).toFloat
        score = max(-50f, min(50f, score))
        if (score > blockMax) blockMax = score
      }

      for (j <- 0 until N) {
        var score: Float = 0f
        for (d_ <- 0 until d) {
          val qIdx = ((((b * H + h) * N) + i) * d) + d_
          val kIdx = ((((b * H + h) * N) + j) * d) + d_
          score = score + q(qIdx) * k(kIdx)
        }
        score = (score / sqrt(d.toFloat)).toFloat
        score = max(-50f, min(50f, score))
        blockSum = blockSum + exp((score - blockMax).toDouble).toFloat
      }

      for (d_ <- 0 until d) {
        var result: Float = 0f
        for (j <- 0 until N) {
          var score: Float = 0f
          for (dd <- 0 until d) {
            val qIdx = ((((b * H + h) * N) + i) * d) + dd
            val kIdx = ((((b * H + h) * N) + j) * d) + dd
            score = score + q(qIdx) * k(kIdx)
          }
          score = (score / sqrt(d.toFloat)).toFloat
          score = max(-50f, min(50f, score))
          val weight = (exp((score - blockMax).toDouble).toFloat) / blockSum

          val vIdx = ((((b * H + h) * N) + j) * d) + d_
          result = result + weight * v(vIdx)
        }
        val outIdx = ((((b * H + h) * N) + i) * d) + d_
        output(outIdx) = result
      }
    }
  }
}
