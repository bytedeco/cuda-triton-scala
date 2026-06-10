package cuda.dsl.benchmark

import cuda.dsl.core._
import cuda.dsl.core.dim3
import cuda.dsl.runtime._
import cuda.dsl.core.Types.given_MemoryOps_Float
import org.bytedeco.cuda.global.cudart._
import org.bytedeco.javacpp._
import scala.math._

/** 验证编译后的CUDA kernel运行结果是否符合预期，不能都是0.0 */

object TestKVCacheVerification {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("CUDA Kernel Runtime Result Verification")
    println("验证返回值不能都是0.0")
    println("=" * 80)

    val runtime = DeviceSelector.getRuntime()
    runtime.init()

    if (!runtime.isAvailable) {
      println("[ERROR] CUDA不可用")
      return
    }

    var passed = 0
    var failed = 0

    // 测试1: Tile Extract
    if (testTileExtract(runtime)) passed += 1 else failed += 1

    // 测试2: Tile Insert
    if (testTileInsert(runtime)) passed += 1 else failed += 1

    // 测试3: KVCache Update
    if (testKVCacheUpdate(runtime)) passed += 1 else failed += 1

    // 测试4: KVCache RoPE
    if (testKVCacheRoPE(runtime)) passed += 1 else failed += 1

    // 测试5: PageAttention
    if (testPageAttention(runtime)) passed += 1 else failed += 1

    // 测试6: FlashAttention
    if (testFlashAttention(runtime)) passed += 1 else failed += 1

    // 测试7: FlexAttention
    if (testFlexAttention(runtime)) passed += 1 else failed += 1

    // 测试8: SaveStore操作
    if (testSaveStore(runtime)) passed += 1 else failed += 1

    // 测试9: Token操作
    if (testTokenEmbedding(runtime)) passed += 1 else failed += 1

    // 测试10: GEMM操作
    if (testGEMM(runtime)) passed += 1 else failed += 1

    runtime.synchronize()

    println("\n" + "=" * 80)
    println(s"结果: $passed 通过, $failed 失败 (共10个测试)")
    if (failed == 0) println("全部通过!")
    println("=" * 80)
  }

  // ==========================================
  // Test 1: Tile Extract
  // ==========================================
  def testTileExtract(rt: DeviceRuntime): Boolean = {
    println("\n[Test 1] Tile Extract")

    val M, N = 64
    val tileSize = 16

    val kernelSrc = """
    |extern "C" __global__ void tileExtractKernel(
    |    float* tile, float* input,
    |    int tileRow, int tileCol, int tileSize,
    |    int M, int N, int stride)
    |{
    |    int tx = threadIdx.x;
    |    int ty = threadIdx.y;
    |    if (ty < tileSize && tx < tileSize) {
    |        int row = tileRow * tileSize + ty;
    |        int col = tileCol * tileSize + tx;
    |        if (row < M && col < N) {
    |            tile[ty * tileSize + tx] = input[row * stride + col] * 2.0f;
    |        }
    |    }
    |}
    """.stripMargin

    val kernel = rt.compileKernel("tileExtractKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val d_tile = rt.malloc[Float](tileSize * tileSize)
    val d_input = rt.malloc[Float](M * N)

    // 初始化输入为非零值
    val h_input = (0 until M * N).map(i => ((i % 50) + 1).toFloat).toArray
    rt.memcpyHtoD(d_input, h_input, M * N)

    val grid = dim3(1, 1, 1)
    val block = dim3(tileSize, tileSize, 1)

    val args = Seq[Pointer](
      new LongPointer(d_tile.rawAddress),
      new LongPointer(d_input.rawAddress),
      new IntPointer(0L), new IntPointer(0L),
      new IntPointer(tileSize.toLong),
      new IntPointer(M.toLong), new IntPointer(N.toLong),
      new IntPointer(N.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val h_tile = new Array[Float](tileSize * tileSize)
    rt.memcpyDtoH(h_tile, d_tile, tileSize * tileSize)

    rt.free(d_tile)
    rt.free(d_input)

    val nonZero = h_tile.count(abs(_) > 0.01f)
    val first = h_tile(0)

    if (nonZero == 0) {
      println(s"[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] $nonZero/${tileSize*tileSize} 非零, 首值=$first")
    true
  }

  // ==========================================
  // Test 2: Tile Insert
  // ==========================================
  def testTileInsert(rt: DeviceRuntime): Boolean = {
    println("\n[Test 2] Tile Insert")

    val M, N, tileSize = 64

    val kernelSrc = """
    |extern "C" __global__ void tileInsertKernel(
    |    float* output, float* tile,
    |    int tileRow, int tileCol, int tileSize,
    |    int M, int N, int stride)
    |{
    |    int tx = threadIdx.x;
    |    int ty = threadIdx.y;
    |    if (ty < tileSize && tx < tileSize) {
    |        int row = tileRow * tileSize + ty;
    |        int col = tileCol * tileSize + tx;
    |        if (row < M && col < N) {
    |            output[row * stride + col] = tile[ty * tileSize + tx] * 3.0f;
    |        }
    |    }
    |}
    """.stripMargin

    val kernel = rt.compileKernel("tileInsertKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val d_output = rt.malloc[Float](M * N)
    val d_tile = rt.malloc[Float](tileSize * tileSize)

    // 初始化数据
    val h_tile = (0 until tileSize * tileSize).map(i => (i + 1).toFloat).toArray
    rt.memcpyHtoD(d_tile, h_tile, tileSize * tileSize)

    val grid = dim3(1, 1, 1)
    val block = dim3(tileSize, tileSize, 1)

    val args = Seq[Pointer](
      new LongPointer(d_output.rawAddress),
      new LongPointer(d_tile.rawAddress),
      new IntPointer(0L), new IntPointer(0L),
      new IntPointer(tileSize.toLong),
      new IntPointer(M.toLong), new IntPointer(N.toLong),
      new IntPointer(N.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val h_result = new Array[Float](M * N)
    rt.memcpyDtoH(h_result, d_output, M * N)

    rt.free(d_output)
    rt.free(d_tile)

    val nonZero = h_result.count(abs(_) > 0.01f)
    val first = h_result(0)

    if (nonZero == 0) {
      println(s"[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] $nonZero 非零, 首值=$first")
    true
  }

  // ==========================================
  // Test 3: KVCache Update
  // ==========================================
  def testKVCacheUpdate(rt: DeviceRuntime): Boolean = {
    println("\n[Test 3] KVCache Update")

    val seqLen, headDim = 64

    val kernelSrc = """
    |extern "C" __global__ void kvCacheUpdateKernel(
    |    float* kOut, float* vOut, float* kIn, float* vIn,
    |    int seqLen, int headDim)
    |{
    |    int i = blockIdx.x * blockDim.x + threadIdx.x;
    |    if (i >= seqLen * headDim) return;
    |    kOut[i] = kIn[i] * 1.1f;
    |    vOut[i] = vIn[i] * 0.9f;
    |}
    """.stripMargin

    val kernel = rt.compileKernel("kvCacheUpdateKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val size = seqLen * headDim
    val d_kIn = rt.malloc[Float](size)
    val d_vIn = rt.malloc[Float](size)
    val d_kOut = rt.malloc[Float](size)
    val d_vOut = rt.malloc[Float](size)

    val h_kIn = (0 until size).map(i => (i + 1).toFloat).toArray
    val h_vIn = (0 until size).map(i => (i + 1).toFloat).toArray
    rt.memcpyHtoD(d_kIn, h_kIn, size)
    rt.memcpyHtoD(d_vIn, h_vIn, size)

    val grid = dim3((size + 255) / 256, 1, 1)
    val block = dim3(256, 1, 1)

    val args = Seq[Pointer](
      new LongPointer(d_kOut.rawAddress),
      new LongPointer(d_vOut.rawAddress),
      new LongPointer(d_kIn.rawAddress),
      new LongPointer(d_vIn.rawAddress),
      new IntPointer(seqLen.toLong),
      new IntPointer(headDim.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val h_kOut = new Array[Float](size)
    val h_vOut = new Array[Float](size)
    rt.memcpyDtoH(h_kOut, d_kOut, size)
    rt.memcpyDtoH(h_vOut, d_vOut, size)

    rt.free(d_kIn); rt.free(d_vIn)
    rt.free(d_kOut); rt.free(d_vOut)

    val nonZero = h_kOut.count(abs(_) > 0.01f) + h_vOut.count(abs(_) > 0.01f)
    val k0 = h_kOut(0)
    val v0 = h_vOut(0)

    if (nonZero == 0) {
      println("[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] k[0]=$k0, v[0]=$v0")
    true
  }

  // ==========================================
  // Test 4: KVCache RoPE
  // ==========================================
  def testKVCacheRoPE(rt: DeviceRuntime): Boolean = {
    println("\n[Test 4] KVCache RoPE")

    val seqLen, headDim = 64

    val kernelSrc = """
    |extern "C" __global__ void kvRoPEKernel(
    |    float* out, float* in,
    |    int seqLen, int headDim)
    |{
    |    int i = blockIdx.x * blockDim.x + threadIdx.x;
    |    if (i >= seqLen * headDim) return;
    |    int s = i / headDim;
    |    int h = i % headDim;
    |    float angle = s / powf(10000.0f, (2.0f * h) / (float)headDim);
    |    float cosA = cosf(angle);
    |    float sinA = sinf(angle);
    |    float val = in[i];
    |    // RoPE公式
    |    out[i] = (h % 2 == 0) ? val * cosA : val * sinA;
    |}
    """.stripMargin

    val kernel = rt.compileKernel("kvRoPEKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val size = seqLen * headDim
    val dIn = rt.malloc[Float](size)
    val dOut = rt.malloc[Float](size)

    val hIn = (0 until size).map(i => 1.0f).toArray
    rt.memcpyHtoD(dIn, hIn, size)

    val grid = dim3((size + 255) / 256, 1, 1)
    val block = dim3(256, 1, 1)

    val args = Seq[Pointer](
      new LongPointer(dOut.rawAddress),
      new LongPointer(dIn.rawAddress),
      new IntPointer(seqLen.toLong),
      new IntPointer(headDim.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val hOut = new Array[Float](size)
    rt.memcpyDtoH(hOut, dOut, size)

    rt.free(dIn); rt.free(dOut)

    val nonZero = hOut.count(abs(_) > 0.0001f)
    val first = hOut(0)

    if (nonZero == 0) {
      println("[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] $nonZero/$size 非零, 首值=$first")
    true
  }

  // ==========================================
  // Test 5: PageAttention
  // ==========================================
  def testPageAttention(rt: DeviceRuntime): Boolean = {
    println("\n[Test 5] PageAttention")

    val numBlocks, blockSize, headDim = 16

    val kernelSrc = """
    |extern "C" __global__ void pageAttnKernel(
    |    float* out, float* q, float* kCache, float* vCache,
    |    int numBlocks, int blockSize, int headDim)
    |{
    |    int i = blockIdx.x * blockDim.x + threadIdx.x;
    |    if (i >= numBlocks * headDim) return;
    |    int blockId = i / headDim;
    |    int hd = i % headDim;
    |    float sum = 0.0f;
    |    // 从各block累加attention score
    |    for (int b = 0; b < numBlocks; b++) {
    |        sum += kCache[b * blockSize * headDim + blockId * headDim + hd];
    |    }
    |    out[i] = q[i] * sum / sqrtf((float)headDim);
    |}
    """.stripMargin

    val kernel = rt.compileKernel("pageAttnKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val qSize = numBlocks * headDim
    val cacheSize = numBlocks * blockSize * headDim

    val dQ = rt.malloc[Float](qSize)
    val dKCache = rt.malloc[Float](cacheSize)
    val dOut = rt.malloc[Float](qSize)

    val hQ = (0 until qSize).map(_ => 1.0f).toArray
    val hK = (0 until cacheSize).map(_ => 0.5f).toArray
    rt.memcpyHtoD(dQ, hQ, qSize)
    rt.memcpyHtoD(dKCache, hK, cacheSize)

    val grid = dim3((qSize + 255) / 256, 1, 1)
    val block = dim3(256, 1, 1)

    val args = Seq[Pointer](
      new LongPointer(dOut.rawAddress),
      new LongPointer(dQ.rawAddress),
      new LongPointer(dKCache.rawAddress),
      new LongPointer(dKCache.rawAddress),
      new IntPointer(numBlocks.toLong),
      new IntPointer(blockSize.toLong),
      new IntPointer(headDim.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val hOut = new Array[Float](qSize)
    rt.memcpyDtoH(hOut, dOut, qSize)

    rt.free(dQ); rt.free(dKCache); rt.free(dOut)

    val nonZero = hOut.count(abs(_) > 0.01f)
    val first = hOut(0)

    if (nonZero == 0) {
      println("[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] $nonZero/$qSize 非零, out[0]=$first")
    true
  }

  // ==========================================
  // Test 6: FlashAttention
  // ==========================================
  def testFlashAttention(rt: DeviceRuntime): Boolean = {
    println("\n[Test 6] FlashAttention")

    val B, N, D = 4

    val kernelSrc = """
    |extern "C" __global__ void flashAttnKernel(
    |    float* out, float* q, float* k, float* v,
    |    int B, int N, int D)
    |{
    |    int row = blockIdx.x;
    |    if (row >= B) return;
    |    float scale = 1.0f / sqrtf((float)D);
    |    // 简化的flash attention
    |    for (int j = 0; j < N; j++) {
    |        float score = 0.0f;
    |        for (int d = 0; d < D; d++) {
    |            score += q[row * D + d] * k[j * D + d];
    |        }
    |        score *= scale;
    |        for (int d = 0; d < D; d++) {
    |            out[row * D + d] += score * v[j * D + d];
    |        }
    |    }
    |}
    """.stripMargin

    val kernel = rt.compileKernel("flashAttnKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val qSize = B * D
    val kSize = N * D
    val vSize = N * D
    val outSize = B * D

    val dQ = rt.malloc[Float](qSize)
    val dK = rt.malloc[Float](kSize)
    val dV = rt.malloc[Float](vSize)
    val dOut = rt.malloc[Float](outSize)

    val hQ = (0 until qSize).map(i => 0.1f).toArray
    val hK = (0 until kSize).map(i => 0.1f).toArray
    val hV = (0 until vSize).map(i => (i + 1).toFloat * 0.01f).toArray
    rt.memcpyHtoD(dQ, hQ, qSize)
    rt.memcpyHtoD(dK, hK, kSize)
    rt.memcpyHtoD(dV, hV, vSize)

    val grid = dim3(B, 1, 1)
    val block = dim3(32, 1, 1)

    val args = Seq[Pointer](
      new LongPointer(dOut.rawAddress),
      new LongPointer(dQ.rawAddress),
      new LongPointer(dK.rawAddress),
      new LongPointer(dV.rawAddress),
      new IntPointer(B.toLong),
      new IntPointer(N.toLong),
      new IntPointer(D.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val hOut = new Array[Float](outSize)
    rt.memcpyDtoH(hOut, dOut, outSize)

    rt.free(dQ); rt.free(dK); rt.free(dV); rt.free(dOut)

    val nonZero = hOut.count(abs(_) > 0.001f)
    val first = hOut(0)

    if (nonZero == 0) {
      println("[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] $nonZero/$outSize 非零, out[0]=$first")
    true
  }

  // ==========================================
  // Test 7: FlexAttention
  // ==========================================
  def testFlexAttention(rt: DeviceRuntime): Boolean = {
    println("\n[Test 7] FlexAttention")

    val B, N, D = 4

    val kernelSrc = """
    |extern "C" __global__ void flexAttnKernel(
    |    float* out, float* q, float* k, float* v,
    |    int B, int N, int D)
    |{
    |    int i = blockIdx.x * blockDim.x + threadIdx.x;
    |    if (i >= B * D) return;
    |    // Flex attention: 自定义score计算
    |    float score = 0.0f;
    |    for (int j = 0; j < N; j++) {
    |        float qk = q[i] * k[j * D + (i % D)];
    |        score += qk + v[j * D + (i % D)];
    |    }
    |    out[i] = score;
    |}
    """.stripMargin

    val kernel = rt.compileKernel("flexAttnKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val qSize = B * D
    val kSize = N * D
    val vSize = N * D
    val outSize = B * D

    val dQ = rt.malloc[Float](qSize)
    val dK = rt.malloc[Float](kSize)
    val dV = rt.malloc[Float](vSize)
    val dOut = rt.malloc[Float](outSize)

    val hQ = (0 until qSize).map(_ => 1.0f).toArray
    val hK = (0 until kSize).map(_ => 1.0f).toArray
    val hV = (0 until vSize).map(i => (i + 1).toFloat).toArray
    rt.memcpyHtoD(dQ, hQ, qSize)
    rt.memcpyHtoD(dK, hK, kSize)
    rt.memcpyHtoD(dV, hV, vSize)

    val grid = dim3(1, 1, 1)
    val block = dim3(256, 1, 1)

    val args = Seq[Pointer](
      new LongPointer(dOut.rawAddress),
      new LongPointer(dQ.rawAddress),
      new LongPointer(dK.rawAddress),
      new LongPointer(dV.rawAddress),
      new IntPointer(B.toLong),
      new IntPointer(N.toLong),
      new IntPointer(D.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val hOut = new Array[Float](outSize)
    rt.memcpyDtoH(hOut, dOut, outSize)

    rt.free(dQ); rt.free(dK); rt.free(dV); rt.free(dOut)

    val nonZero = hOut.count(abs(_) > 0.001f)
    val first = hOut(0)

    if (nonZero == 0) {
      println("[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] $nonZero/$outSize 非零, out[0]=$first")
    true
  }

  // ==========================================
  // Test 8: SaveStore (with ECC simulation)
  // ==========================================
  def testSaveStore(rt: DeviceRuntime): Boolean = {
    println("\n[Test 8] SaveStore ECC")

    val size = 256

    val kernelSrc = """
    |extern "C" __global__ void saveStoreKernel(
    |    float* out, float* in, float* ecc, int size)
    |{
    |    int i = blockIdx.x * blockDim.x + threadIdx.x;
    |    if (i >= size) return;
    |    // 保存数据和ECC校验
    |    out[i] = in[i];
    |    if (i < size / 8) {
    |        float sum = 0.0f;
    |        for (int j = 0; j < 8; j++) sum += in[i * 8 + j];
    |        ecc[i] = sum / 8.0f;  // 简单checksum
    |    }
    |}
    """.stripMargin

    val kernel = rt.compileKernel("saveStoreKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val dIn = rt.malloc[Float](size)
    val dOut = rt.malloc[Float](size)
    val dEcc = rt.malloc[Float](size / 8)

    val hIn = (0 until size).map(i => (i + 1).toFloat).toArray
    rt.memcpyHtoD(dIn, hIn, size)

    val grid = dim3(1, 1, 1)
    val block = dim3(256, 1, 1)

    val args = Seq[Pointer](
      new LongPointer(dOut.rawAddress),
      new LongPointer(dIn.rawAddress),
      new LongPointer(dEcc.rawAddress),
      new IntPointer(size.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val hOut = new Array[Float](size)
    val hEcc = new Array[Float](size / 8)
    rt.memcpyDtoH(hOut, dOut, size)
    rt.memcpyDtoH(hEcc, dEcc, size / 8)

    rt.free(dIn); rt.free(dOut); rt.free(dEcc)

    val nonZero = hOut.count(abs(_) > 0.01f) + hEcc.count(abs(_) > 0.01f)
    val first = hOut(0)
    val ecc0 = hEcc(0)

    if (nonZero == 0) {
      println("[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] out[0]=$first, ecc[0]=$ecc0")
    true
  }

  // ==========================================
  // Test 9: Token Embedding
  // ==========================================
  def testTokenEmbedding(rt: DeviceRuntime): Boolean = {
    println("\n[Test 9] Token Embedding")

    val vocabSize, embedDim, seqLen = 64

    val kernelSrc = """
    |extern "C" __global__ void embedKernel(
    |    float* out, float* table, int* tokens, int seqLen, int embedDim)
    |{
    |    int i = blockIdx.x * blockDim.x + threadIdx.x;
    |    if (i >= seqLen * embedDim) return;
    |    int t = tokens[i / embedDim];
    |    int e = i % embedDim;
    |    out[i] = table[t * embedDim + e];
    |}
    """.stripMargin

    val kernel = rt.compileKernel("embedKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val tableSize = vocabSize * embedDim
    val outSize = seqLen * embedDim

    val dTable = rt.malloc[Float](tableSize)
    val dTokens = rt.malloc[Float](seqLen)
    val dOut = rt.malloc[Float](outSize)

    val hTable = (0 until tableSize).map(i => (i % 10).toFloat).toArray
    val hTokens = (0 until seqLen).map(i => (i % vocabSize).toFloat).toArray
    rt.memcpyHtoD(dTable, hTable, tableSize)
    rt.memcpyHtoD(dTokens, hTokens, seqLen)

    val grid = dim3(1, 1, 1)
    val block = dim3(256, 1, 1)

    val args = Seq[Pointer](
      new LongPointer(dOut.rawAddress),
      new LongPointer(dTable.rawAddress),
      new LongPointer(dTokens.rawAddress),
      new IntPointer(seqLen.toLong),
      new IntPointer(embedDim.toLong)
    )

    rt.launchKernel(kernel, grid, block, args)
    rt.synchronize()

    val hOut = new Array[Float](outSize)
    rt.memcpyDtoH(hOut, dOut, outSize)

    rt.free(dTable); rt.free(dTokens); rt.free(dOut)

    val nonZero = hOut.count(abs(_) > 0.01f)
    val first = hOut(0)

    if (nonZero == 0) {
      println("[FAIL] 输出全是0")
      return false
    }

    println(s"[PASS] $nonZero/$outSize 非零, embedding[0]=$first")
    true
  }

  // ==========================================
  // Test 10: GEMM
  // ==========================================
  def testGEMM(rt: DeviceRuntime): Boolean = {
    println("\n[Test 10] GEMM")

    val M, K, N = 32

    val kernelSrc = """
    |extern "C" __global__ void gemmKernel(
    |    float* C, float* A, float* B, int M, int K, int N)
    |{
    |    int row = blockIdx.y * blockDim.y + threadIdx.y;
    |    int col = blockIdx.x * blockDim.x + threadIdx.x;
    |    if (row >= M || col >= N) return;
    |    float sum = 0.0f;
    |    for (int k = 0; k < K; k++) {
    |        sum += A[row * K + k] * B[k * N + col];
    |    }
    |    C[row * N + col] = sum;
    |}
    """.stripMargin

    val kernel = rt.compileKernel("gemmKernel", kernelSrc)
    if (kernel == null) {
      println("[FAIL] 编译失败")
      return false
    }

    val aSize = M * K
    val bSize = K * N
    val cSize = M * N

    val dA = rt.malloc[Float](aSize)
    val dB = rt.malloc[Float](bSize)
    val dC = rt.malloc[Float](cSize)

    // 初始化非零矩阵
    val hA = (0 until aSize).map(i => 1.0f).toArray
    val hB = (0 until bSize).map(i => 1.0f).toArray
    rt.memcpyHtoD(dA, hA, aSize)
    rt.memcpyHtoD(dB, hB, bSize)

    val grid = dim3((N + 15) / 16, (M + 15) / 16, 1)
    val block = dim3(16, 16, 1)

    val args = Seq[Pointer](
      new LongPointer(dC.rawAddress),
      new LongPointer(dA.rawAddress),
      new LongPointer(dB.rawAddress),
      new IntPointer(M.toLong),
      new IntPointer(K.toLong),
      new IntPointer(N.toLong)
    )

    // Pass scalars explicitly so stub can read them in GPU mode
    rt.launchKernel(kernel, grid, block, args, Map("M" -> M, "K" -> K, "N" -> N))
    rt.synchronize()

    val hC = new Array[Float](cSize)
    rt.memcpyDtoH(hC, dC, cSize)

    rt.free(dA); rt.free(dB); rt.free(dC)

    val nonZero = hC.count(abs(_) > 0.01f)
    val first = hC(0)

    if (nonZero == 0) {
      println("[FAIL] 输出全是0")
      return false
    }

    // 期望值 = K * 1.0f * 1.0f = K
    val expected = K.toFloat
    if (abs(first - expected) > 0.1f) {
      println(s"[FAIL] 期望=$expected, 实际=$first")
      return false
    }

    println(s"[PASS] C[0]=$first (期望=$expected)")
    true
  }
}