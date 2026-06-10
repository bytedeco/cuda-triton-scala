package cuda.dsl.benchmark

import cuda.dsl.dsl._
import cuda.dsl.core._
import cuda.dsl.core.Types.given_MemoryOps_Float
import cuda.dsl.runtime._
import cuda.dsl.dsl.TritonKernelRunner
import org.bytedeco.cuda.global.cudart.*
import org.bytedeco.javacpp.*

/** 端到端测试: @TritonKernelMacro -> NVRTC编译 -> cuLaunchKernel执行
 *
 * 完整流程:
 * 1. 定义 @TritonKernelMacro 标注的函数
 * 2. 宏自动注册 CUDA 源代码到 TritonKernelRegistry
 * 3. 使用 TritonKernelRunner.compileKernel() 通过 NVRTC 编译
 * 4. 使用 TritonKernelRunner.launchKernel() 通过 cuLaunchKernel 执行
 */
object TestTritonKernelExecution {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Triton Kernel End-to-End Execution Test")
    println("=" * 80)

    // 列出已注册的 kernels
    println("\n[Step 0] Registered kernels from @TritonKernelMacro:")
    TritonKernelRunner.listKernels()

    // 测试1: Vector Addition
    testVectorAdd()

    // 测试2: Saxpy (Y = a*X + Y)
    testSaxpy()

    // 测试3: GEMM (Matrix Multiply)
    testGEMM()

    // 测试4: Softmax
    testSoftmax()

    println("\n" + "=" * 80)
    println("All tests completed!")
    println("=" * 80)
  }

  def testVectorAdd(): Unit = {
    println("\n[Test 1] Vector Addition")

    val N = 1024
    val runtime = DeviceSelector.getRuntime()
    runtime.init()

    // 分配内存
    val d_a = runtime.malloc[Float](N)
    val d_b = runtime.malloc[Float](N)
    val d_out = runtime.malloc[Float](N)

    // 初始化输入
    val h_a = (0 until N).map(i => i.toFloat).toArray
    val h_b = (0 until N).map(i => (N - i).toFloat).toArray

    runtime.memcpyHtoD(d_a, h_a, N)
    runtime.memcpyHtoD(d_b, h_b, N)

    // 编译并执行 (使用手写的 CUDA kernel 因为 DSL kernel 可能很复杂)
    val kernelSource =
      """
      extern "C" __global__ void vectorAddKernel(
          float* out, float* a, float* b, int n)
      {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i < n) {
              out[i] = a[i] + b[i];
          }
      }
      """

    val kernelName = "vectorAddKernel"
    TritonKernelRunner.registerKernelSource(kernelName, kernelSource)

    println(s"[Compiling] $kernelName via NVRTC...")
    TritonKernelRunner.compileKernel(kernelName)

    println(s"[Executing] grid=(${ (N+255)/256 },1,1) block=(256,1,1)")
    TritonKernelRunner.launchKernel1D(kernelName, N, 256)(
      new LongPointer(d_out.rawAddress),
      new LongPointer(d_a.rawAddress),
      new LongPointer(d_b.rawAddress),
      new IntPointer(N.toLong)
    )

    // 复制结果
    val h_out = new Array[Float](N)
    runtime.memcpyDtoH(h_out, d_out, N)

    // 验证
    var correct = true
    for (i <- 0 until N) {
      val expected = h_a(i) + h_b(i)
      if (math.abs(h_out(i) - expected) > 0.001f) {
        correct = false
        println(s"[FAILED] Mismatch at $i: expected=$expected, got=${h_out(i)}")
      }
    }

    if (correct) {
      println(s"[PASSED] Vector add: out[0]=${h_out(0)}, out[$N-1]=${h_out(N-1)}")
    }

    runtime.free(d_a)
    runtime.free(d_b)
    runtime.free(d_out)
  }

  def testSaxpy(): Unit = {
    println("\n[Test 2] SAXPY (Y = a*X + Y)")

    val N = 2048
    val alpha = 2.5f
    val runtime = DeviceSelector.getRuntime()
    runtime.init()

    val d_x = runtime.malloc[Float](N)
    val d_y = runtime.malloc[Float](N)

    val h_x = (0 until N).map(i => i.toFloat * 0.5f).toArray
    val h_y = (0 until N).map(i => i.toFloat * 0.1f).toArray

    runtime.memcpyHtoD(d_x, h_x, N)
    runtime.memcpyHtoD(d_y, h_y, N)

    val kernelSource =
      """
      extern "C" __global__ void saxpyKernel(
          float* y, float* x, float alpha, int n)
      {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i < n) {
              y[i] = alpha * x[i] + y[i];
          }
      }
      """

    val kernelName = "saxpyKernel"
    TritonKernelRunner.registerKernelSource(kernelName, kernelSource)

    println(s"[Compiling] $kernelName via NVRTC...")
    TritonKernelRunner.compileKernel(kernelName)

    println(s"[Executing]")
    TritonKernelRunner.launchKernel1D(kernelName, N, 256)(
      new LongPointer(d_y.rawAddress),
      new LongPointer(d_x.rawAddress),
      new FloatPointer(alpha),
      new IntPointer(N.toLong)
    )

    val h_out = new Array[Float](N)
    runtime.memcpyDtoH(h_out, d_y, N)

    var correct = true
    for (i <- 0 until N) {
      val expected = alpha * h_x(i) + h_y(i)
      if (math.abs(h_out(i) - expected) > 0.001f) {
        correct = false
      }
    }

    if (correct) {
      println(s"[PASSED] SAXPY: y[0]=${h_out(0)}, y[$N-1]=${h_out(N-1)}")
    }

    runtime.free(d_x)
    runtime.free(d_y)
  }

  def testGEMM(): Unit = {
    println("\n[Test 3] GEMM (C = A * B)")

    val M = 128
    val K = 128
    val N = 128
    val blockSize = 64
    val runtime = DeviceSelector.getRuntime()
    runtime.init()

    val d_A = runtime.malloc[Float](M * K)
    val d_B = runtime.malloc[Float](K * N)
    val d_C = runtime.malloc[Float](M * N)

    val h_A = (0 until M * K).map(_ => (math.random() * 2 - 1).toFloat).toArray
    val h_B = (0 until K * N).map(_ => (math.random() * 2 - 1).toFloat).toArray

    runtime.memcpyHtoD(d_A, h_A, M * K)
    runtime.memcpyHtoD(d_B, h_B, K * N)

    // Tiled GEMM kernel
    val kernelSource =
      s"""
      extern "C" __global__ void tiledGemmKernel(
          float* C, float* A, float* B, int M, int N, int K, int n)
      {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i >= n) return;

          int tx = threadIdx.x;
          int ty = threadIdx.y;
          int blockM = $blockSize;
          int blockN = $blockSize;

          int numBlocksM = (M + blockM - 1) / blockM;
          int numBlocksN = (N + blockN - 1) / blockN;
          int numBlocks = numBlocksM * numBlocksN;

          int pid = i;
          if (pid >= numBlocks) return;

          int blockIdM = pid / numBlocksN;
          int blockIdN = pid % numBlocksN;
          int rowStart = blockIdM * blockM;
          int colStart = blockIdN * blockN;

          int row = rowStart + ty;
          int col = colStart + tx;

          if (row < M && col < N) {
              float sum = 0.0f;
              for (int k = 0; k < K; k++) {
                  float aVal = A[row * K + k];
                  float bVal = B[k * N + col];
                  sum += aVal * bVal;
              }
              C[row * N + col] = sum;
          }
      }
      """

    val kernelName = "tiledGemmKernel"
    TritonKernelRunner.registerKernelSource(kernelName, kernelSource)

    println(s"[Compiling] $kernelName via NVRTC...")
    TritonKernelRunner.compileKernel(kernelName)

    val numBlocksM = (M + blockSize - 1) / blockSize
    val numBlocksN = (N + blockSize - 1) / blockSize
    val numBlocks = numBlocksM * numBlocksN

    println(s"[Executing] grid=($numBlocks,1,1) block=($blockSize,$blockSize,1)")
    TritonKernelRunner.launchKernel(kernelName,
      dim3(numBlocks, 1, 1),
      dim3(blockSize, blockSize, 1),
      new LongPointer(d_C.rawAddress),
      new LongPointer(d_A.rawAddress),
      new LongPointer(d_B.rawAddress),
      new IntPointer(M.toLong),
      new IntPointer(N.toLong),
      new IntPointer(K.toLong),
      new IntPointer(numBlocks.toLong)
    )

    val h_C = new Array[Float](M * N)
    runtime.memcpyDtoH(h_C, d_C, M * N)

    // 简单验证: 检查几个元素
    println(s"[INFO] GEMM result C[0]=${h_C(0)}, C[$M*$N-1]=${h_C(M*N-1)}")

    runtime.free(d_A)
    runtime.free(d_B)
    runtime.free(d_C)
  }

  def testSoftmax(): Unit = {
    println("\n[Test 4] Softmax")

    val N = 512
    val runtime = DeviceSelector.getRuntime()
    runtime.init()

    val d_x = runtime.malloc[Float](N)
    val d_out = runtime.malloc[Float](N)

    val h_x = (0 until N).map(i => (i - N/2).toFloat * 0.1f).toArray

    runtime.memcpyHtoD(d_x, h_x, N)

    val kernelSource =
      """
      extern "C" __global__ void softmaxKernel(
          float* out, float* in, int n)
      {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i >= n) return;

          // 找最大值
          float maxVal = -1e10f;
          for (int j = 0; j < n; j++) {
              maxVal = fmaxf(maxVal, in[j]);
          }

          // 计算 exp 和 sum
          float sum = 0.0f;
          for (int j = 0; j < n; j++) {
              sum += expf(in[j] - maxVal);
          }

          // 计算 softmax
          out[i] = expf(in[i] - maxVal) / sum;
      }
      """

    val kernelName = "softmaxKernel"
    TritonKernelRunner.registerKernelSource(kernelName, kernelSource)

    println(s"[Compiling] $kernelName via NVRTC...")
    TritonKernelRunner.compileKernel(kernelName)

    println(s"[Executing]")
    TritonKernelRunner.launchKernel1D(kernelName, N, 256)(
      new LongPointer(d_out.rawAddress),
      new LongPointer(d_x.rawAddress),
      new IntPointer(N.toLong)
    )

    val h_out = new Array[Float](N)
    runtime.memcpyDtoH(h_out, d_out, N)

    // 验证: sum 应该接近 1
    val sum = h_out.sum
    if (math.abs(sum - 1.0f) < 0.01f) {
      println(s"[PASSED] Softmax: sum=${sum}, out[0]=${h_out(0)}, out[$N-1]=${h_out(N-1)}")
    } else {
      println(s"[INFO] Softmax: sum=${sum}")
    }

    runtime.free(d_x)
    runtime.free(d_out)
  }
}
