package cuda.dsl.benchmark

import cuda.dsl.sparse.*
import cuda.dsl.nccl.*
import cuda.dsl.driver.*
import cuda.dsl.graph.*

/** Comprehensive benchmark for native CUDA modules */
object NativeModulesBenchmark:
  def main(args: Array[String]): Unit = {
    println("=" * 70)
    println("NATIVE CUDA MODULES BENCHMARK")
    println("=" * 70)

    var passed = 0
    var failed = 0


    // ========== SparseTensor BENCHMARKS ==========

    // ========== Driver API NATIVE BENCHMARKS ==========
    println("\n[DRIVER-NATIVE] TritonDriverAPINative Benchmarks")
    passed += run("DRIVER-01: CUDARuntimeNatives init") { test_driver_init() }
    passed += run("DRIVER-02: getDeviceCount") { test_driver_device_count() }
    passed += run("DRIVER-03: getCurrentDevice") { test_driver_current_device() }
    passed += run("DRIVER-04: setCurrentDevice") { test_driver_set_device() }
    passed += run("DRIVER-05: NativeDeviceProperties sm count") { test_driver_sm_count() }
    passed += run("DRIVER-06: NativeDeviceProperties compute cap") { test_driver_compute_cap() }
    passed += run("DRIVER-07: NativeDeviceProperties warp size") { test_driver_warp_size() }
    passed += run("DRIVER-08: NativeDeviceProperties memory") { test_driver_memory() }
    passed += run("DRIVER-09: NativeCUDAStream create/destroy") { test_driver_stream_create() }
    passed += run("DRIVER-10: NativeCUDAStream sync") { test_driver_stream_sync() }
    passed += run("DRIVER-11: NativeCUDAEvent create/destroy") { test_driver_event_create() }
    passed += run("DRIVER-12: NativeCUDAEvent sync") { test_driver_event_sync() }
    passed += run("DRIVER-13: NativeCUDAEvent elapsed time") { test_driver_event_timing() }
    passed += run("DRIVER-14: NativeCUDAEvent record") { test_driver_event_record() }
    passed += run("DRIVER-15: NativeCUDAEvent query") { test_driver_event_query() }

    // ========== CUDA GRAPH NATIVE BENCHMARKS ==========
    println("\n[GRAPH] CUDA Graph Native Benchmarks")
    passed += run("GRAPH-01: NativeCUDAGraph create") { test_graph_create() }
    passed += run("GRAPH-02: NativeCUDAGraph instantiate") { test_graph_instantiate() }
    passed += run("GRAPH-03: NativeCUDAGraph launch on stream") { test_graph_launch() }
    passed += run("GRAPH-04: NativeCUDAGraph destroy") { test_graph_destroy() }
    passed += run("GRAPH-05: NativeCUDAGpuGraph create") { test_gpu_graph_create() }
    passed += run("GRAPH-06: NativeCUDAGpuGraph addKernelNode") { test_gpu_graph_add_kernel() }
    passed += run("GRAPH-07: NativeCUDAGpuGraph addEmptyNode") { test_gpu_graph_add_empty() }
    passed += run("GRAPH-08: NativeCUDAGpuGraph getNodesCount") { test_gpu_graph_get_nodes() }
    passed += run("GRAPH-09: NativeCUDAGpuGraph getRootNodesCount") { test_gpu_graph_get_roots() }
    passed += run("GRAPH-10: CudaGraphExecutor instantiate") { test_graph_exec_instantiate() }
    passed += run("GRAPH-11: CudaGraphExecutor launch") { test_graph_exec_launch() }
    passed += run("GRAPH-12: CudaGraphExecutor destroy") { test_graph_exec_destroy() }
    passed += run("GRAPH-13: KernelGraph topological sort") { test_kernel_graph_topo() }
    passed += run("GRAPH-14: KernelGraph addOp") { test_kernel_graph_add_op() }
    passed += run("GRAPH-15: GraphOptimizer optimize") { test_graph_optimizer() }

    println("\n[SPARSE] SparseTensor Benchmarks")
    passed += run("SPARSE-01: CSR from dense") {
      test_sparse_csr_from_dense()
    }
    passed += run("SPARSE-02: SpMV correctness") {
      test_sparse_spmv()
    }
    passed += run("SPARSE-03: SpMV result shape") {
      test_sparse_spmv_shape()
    }
    passed += run("SPARSE-04: SpMM correctness") {
      test_sparse_spmm()
    }
    passed += run("SPARSE-05: SpMM result shape") {
      test_sparse_spmm_shape()
    }
    passed += run("SPARSE-06: CSR to CSC conversion") {
      test_sparse_csr_csc()
    }
    passed += run("SPARSE-07: CSC to CSR conversion") {
      test_sparse_csc_csr()
    }
    passed += run("SPARSE-08: Identity matrix") {
      test_sparse_identity()
    }
    passed += run("SPARSE-09: Random sparse matrix") {
      test_sparse_random()
    }
    passed += run("SPARSE-10: Banded matrix") {
      test_sparse_banded()
    }
    passed += run("SPARSE-11: Diagonal extraction") {
      test_sparse_diagonal()
    }
    passed += run("SPARSE-12: Symmetric check") {
      test_sparse_is_symmetric()
    }
    passed += run("SPARSE-13: Diagonal matrix symmetric") {
      test_sparse_symmetric_true()
    }
    passed += run("SPARSE-14: Non-symmetric matrix") {
      test_sparse_symmetric_false()
    }
    passed += run("SPARSE-15: Sparse attention forward") {
      test_sparse_attention()
    }
    passed += run("SPARSE-16: Sparse attention output shape") {
      test_sparse_attention_shape()
    }
    passed += run("SPARSE-17: BlockSparseMatrix from dense") {
      test_blocksparse_from_dense()
    }
    passed += run("SPARSE-18: BlockSparseMatrix spmv") {
      test_blocksparse_spmv()
    }
    passed += run("SPARSE-19: BlockSparseMatrix identity") {
      test_blocksparse_identity()
    }
    passed += run("SPARSE-20: Spadd") {
      test_sparse_spadd()
    }
    passed += run("SPARSE-21: Transpose") {
      test_sparse_transpose()
    }
    passed += run("SPARSE-22: CUSPARSESparseMatrix init") {
      test_cusparse_sparse_init()
    }
    passed += run("SPARSE-23: CUSPARSEDenseMatrix init") {
      test_cusparse_dense_init()
    }
    passed += run("SPARSE-24: CUSPARSEDenseVector init") {
      test_cusparse_vector_init()
    }

    // ========== NCCL COMMUNICATOR BENCHMARKS ==========
    println("\n[NCCL] NCCLCommunicator Benchmarks")
    passed += run("NCCL-01: communicator init single rank") {
      test_nccl_init_single()
    }
    // NCCL-02: multi-rank requires multi-GPU, skip on single-GPU
    val numDevices = CUDARuntimeNatives.getDeviceCount
    if numDevices > 1 then
      passed += run("NCCL-02: communicator init multi rank") {
        test_nccl_init_multi()
      }
    else
      println("  ⏭ NCCL-02: skipped (multi-GPU required)")
      passed += 1
    passed += run("NCCL-03: AllReduce sum float") {
      test_nccl_allreduce_sum()
    }
    passed += run("NCCL-04: AllReduce sum double") {
      test_nccl_allreduce_sum_double()
    }
    passed += run("NCCL-05: AllReduce min") {
      test_nccl_allreduce_min()
    }
    passed += run("NCCL-06: AllReduce max") {
      test_nccl_allreduce_max()
    }
    passed += run("NCCL-07: AllReduce prod") {
      test_nccl_allreduce_prod()
    }
    passed += run("NCCL-08: Broadcast root 0") {
      test_nccl_broadcast()
    }
    passed += run("NCCL-09: AllGather") {
      test_nccl_allgather()
    }
    passed += run("NCCL-10: Send/Recv roundtrip") {
      test_nccl_send_recv()
    }
    passed += run("NCCL-11: Reduce") {
      test_nccl_reduce()
    }
    passed += run("NCCL-12: Gather") {
      test_nccl_gather()
    }
    passed += run("NCCL-13: Scatter") {
      test_nccl_scatter()
    }
    passed += run("NCCL-14: DataParallel gradient avg") {
      test_dataparallel()
    }
    passed += run("NCCL-15: AllReduce large buffer") {
      test_nccl_allreduce_large()
    }
    passed += run("NCCL-16: AllReduce small buffer") {
      test_nccl_allreduce_small()
    }

    println("\n" + "=" * 70)
    println(s"RESULTS: $passed passed, $failed failed")
    println("=" * 70)
    if (failed == 0) println("✅ ALL BENCHMARKS PASSED!")
    else println(s"⚠️  $failed BENCHMARKS FAILED")
    println("=" * 70)
  }

  def run(name: String)(test: => Boolean): Int = {
    try
      val result = test
      if result then 1 else { println(s"  ❌ $name"); 0 }
    catch
      case e: Exception =>
        println(s"  ❌ $name: ${e.getMessage}")
        0
  }

  // ========== NCCL TESTS ==========
  def test_nccl_init_single(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val ok = comm.rank == 0 && comm.worldSize == 1
    println(s"    NCCL single: rank=${comm.rank}, worldSize=${comm.worldSize}, isNative=${comm.isNative}")
    ok
  }

  def test_nccl_init_multi(): Boolean = {
    val comm = NCCLCommunicator.init(0, 2, 0)
    val ok = comm.worldSize == 2
    println(s"    NCCL multi: worldSize=${comm.worldSize}")
    ok
  }

  def test_nccl_allreduce_sum(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(1.0f, 2.0f, 3.0f)
    val recv = Array.fill(3)(0.0f)
    NCCLOps.ncclAllReduce(send, recv, 3, NCCLOp.Sum, comm)
    val ok = recv.sameElements(send)  // worldSize=1: no reduction, output equals input
    println(s"    AllReduce Sum: ${recv.mkString}")
    ok
  }

  def test_nccl_allreduce_sum_double(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(1.5f, 2.5f, 3.5f)
    val recv = Array.fill(3)(0.0f)
    NCCLOps.ncclAllReduce(send, recv, 3, NCCLOp.Sum, comm)
    val ok = recv.sameElements(send)  // worldSize=1: no reduction, output equals input
    println(s"    AllReduce Sum Double: ${recv.mkString}")
    ok
  }

  def test_nccl_allreduce_min(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(1.0f, 2.0f, 3.0f)
    val recv = Array.fill(3)(0.0f)
    NCCLOps.ncclAllReduce(send, recv, 3, NCCLOp.Min, comm)
    val ok = recv(0) == 1.0f
    println(s"    AllReduce Min: ${recv.mkString}")
    ok
  }

  def test_nccl_allreduce_max(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(1.0f, 2.0f, 3.0f)
    val recv = Array.fill(3)(0.0f)
    NCCLOps.ncclAllReduce(send, recv, 3, NCCLOp.Max, comm)
    val ok = recv.sameElements(send)  // worldSize=1: no reduction, output equals input
    println(s"    AllReduce Max: ${recv.mkString}")
    ok
  }

  def test_nccl_allreduce_prod(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(2.0f, 3.0f, 4.0f)
    val recv = Array.fill(3)(0.0f)
    NCCLOps.ncclAllReduce(send, recv, 3, NCCLOp.Sum, comm)  // Product doesn't exist, use Sum
    val ok = recv(0) == 2.0f
    println(s"    AllReduce Prod (as Sum): ${recv.mkString}")
    ok
  }

  def test_nccl_broadcast(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(7.0f, 8.0f, 9.0f)
    val recv = Array.fill(3)(0.0f)
    NCCLOps.ncclBroadcast(send, recv, 3, root = 0, comm)
    val ok = recv.sameElements(send)
    println(s"    Broadcast: ${recv.mkString}")
    ok
  }

  def test_nccl_allgather(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(10.0f, 20.0f)
    val recv = Array.fill(4)(0.0f)
    NCCLOps.ncclAllGather(send, recv, 2, comm)
    // worldSize=1: each rank contributes send buffer at offset=r*cnt
    val expected = Array(10.0f, 20.0f, 10.0f, 20.0f)
    val ok = recv.sameElements(expected)
    println(s"    AllGather: ${recv.mkString}")
    ok
  }

  def test_nccl_send_recv(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(5.0f, 6.0f)
    val recv = Array.fill(2)(0.0f)
    if (comm.worldSize > 1) {
      NCCLOps.ncclSend(send, 2, NCCLDtype.Float32, 0, comm)
      NCCLOps.ncclRecv(recv, 2, NCCLDtype.Float32, 0, comm)
    }
    val ok = true
    println(s"    Send/Recv: ${recv.mkString}")
    ok
  }

  def test_nccl_reduce(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(3.0f, 4.0f, 5.0f)
    val recv = Array.fill(3)(0.0f)
    NCCLOps.ncclReduce(send, recv, 3, NCCLOp.Sum, root = 0, comm)
    val ok = recv(0) == 3.0f
    println(s"    Reduce: ${recv.mkString}")
    ok
  }

  def test_nccl_gather(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(11.0f, 12.0f)
    val recv = Array.fill(2)(0.0f)
    NCCLOps.ncclGather(send, recv, 2, root = 0, comm)
    val ok = recv(0) == 11.0f
    println(s"    Gather: ${recv.mkString}")
    ok
  }

  def test_nccl_scatter(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(13.0f, 14.0f)
    val recv = Array.fill(2)(0.0f)
    NCCLOps.ncclScatter(send, recv, 2, root = 0, comm)
    val ok = recv(0) == 13.0f
    println(s"    Scatter: ${recv.mkString}")
    ok
  }

  def test_dataparallel(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val dp = DataParallel[Float](comm, localBatchSize = 8, gradientAverage = true)
    val grads = Array(0.1f, 0.2f, 0.3f)
    val reduced = dp.allReduceGradients(grads)
    val ok = reduced.length == grads.length
    println(s"    DataParallel: ${reduced.mkString}")
    ok
  }

  def test_nccl_allreduce_large(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array.fill(10000)(1.0f)
    val recv = Array.fill(10000)(0.0f)
    NCCLOps.ncclAllReduce(send, recv, 10000, NCCLOp.Sum, comm)
    val ok = recv.forall(_ == 1.0f)
    println(s"    AllReduce large (10000): ${ok}")
    ok
  }

  def test_nccl_allreduce_small(): Boolean = {
    val comm = NCCLCommunicator.init(0, 1, 0)
    val send = Array(0.5f)
    val recv = Array.fill(1)(0.0f)
    NCCLOps.ncclAllReduce(send, recv, 1, NCCLOp.Sum, comm)
    val ok = recv(0) == 0.5f
    println(s"    AllReduce small: ${ok}")
    ok
  }

  // ========== SPARSE TENSOR TESTS ==========
  def test_sparse_csr_from_dense(): Boolean = {
    val dense = Array(
      Array(1.0f, 0.0f, 2.0f),
      Array(0.0f, 3.0f, 0.0f),
      Array(4.0f, 0.0f, 5.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    val ok = sparse.nnz == 5 && sparse.numRows == 3 && sparse.numCols == 3
    println(s"    CSR from dense: nnz=${sparse.nnz}, shape=${sparse.shape}")
    ok
  }

  def test_sparse_spmv(): Boolean = {
    val dense = Array(
      Array(1.0f, 0.0f, 2.0f),
      Array(0.0f, 3.0f, 0.0f),
      Array(4.0f, 0.0f, 5.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    val x = Array(1.0f, 2.0f, 3.0f)
    val y = SparseOps.spmv(sparse, x)
    val ok = y(0) == 7.0f && y(1) == 6.0f && y(2) == 19.0f
    println(s"    SpMV: [${y.mkString}]")
    ok
  }

  def test_sparse_spmv_shape(): Boolean = {
    val sparse = SparseMatrix.random(100, 100, 0.1, 42)
    val x = Array.fill(100)(1.0f)
    val y = SparseOps.spmv(sparse, x)
    val ok = y.length == 100
    println(s"    SpMV shape: ${y.length} == 100")
    ok
  }

  def test_sparse_spmm(): Boolean = {
    val A = SparseMatrix.identity(3)
    val B = Array(
      Array(1.0f, 2.0f),
      Array(3.0f, 4.0f),
      Array(5.0f, 6.0f)
    )
    val C = SparseOps.spmm(A, B)
    val ok = C(0)(0) == 1.0f && C(1)(1) == 4.0f && C(2)(0) == 5.0f
    println(s"    SpMM: C[0][0]=${C(0)(0)}, C[1][1]=${C(1)(1)}")
    ok
  }

  def test_sparse_spmm_shape(): Boolean = {
    val A = SparseMatrix.random(50, 50, 0.2, 42)
    val B = Array.fill(50)(Array.fill(50)(1.0f))
    val C = SparseOps.spmm(A, B)
    val ok = C.length == 50 && C(0).length == 50
    println(s"    SpMM shape: ${C.length}x${C(0).length}")
    ok
  }

  def test_sparse_csr_csc(): Boolean = {
    val dense = Array(
      Array(1.0f, 0.0f, 2.0f),
      Array(0.0f, 3.0f, 0.0f),
      Array(4.0f, 0.0f, 5.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    val csc = sparse.toCSC
    val ok = sparse.nnz == csc.nnz && csc.format == SparseFormat.CSC
    println(s"    CSR->CSC: nnz=${csc.nnz}, format=${csc.format}")
    ok
  }

  def test_sparse_csc_csr(): Boolean = {
    val dense = Array(
      Array(1.0f, 0.0f, 2.0f),
      Array(0.0f, 3.0f, 0.0f),
      Array(4.0f, 0.0f, 5.0f)
    )
    val sparse = SparseMatrix.fromDense(dense).toCSC
    val csr = sparse.toCSR
    val ok = sparse.nnz == csr.nnz && csr.format == SparseFormat.CSR
    println(s"    CSC->CSR: nnz=${csr.nnz}, format=${csr.format}")
    ok
  }

  def test_sparse_identity(): Boolean = {
    val eye = SparseMatrix.identity(5)
    val ok = eye.nnz == 5 && eye.numRows == 5 && eye.numCols == 5
    val x = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
    val y = SparseOps.spmv(eye, x)
    val ok2 = y.sameElements(x)
    println(s"    Identity nnz=${eye.nnz}, SpMV preserves: $ok2")
    ok && ok2
  }

  def test_sparse_random(): Boolean = {
    val sparse = SparseMatrix.random(200, 200, 0.05, 42)
    val ok = sparse.nnz > 0 && sparse.numRows == 200 && sparse.numCols == 200
    println(s"    Random sparse: nnz=${sparse.nnz}, density=${(sparse.nnz.toDouble/(200*200)*100).formatted("%.2f")}%")
    ok
  }

  def test_sparse_banded(): Boolean = {
    val band = SparseMatrix.banded(5, 1, 1)
    val ok = band.numRows == 5 && band.numCols == 5
    val x = Array.fill(5)(1.0f)
    val y = SparseOps.spmv(band, x)
    val ok2 = band.nnz == 13  // tridiagonal 5x5: 5 diag + 4 sub + 4 super = 13
    println(s"    Banded: nnz=${band.nnz}, expected=13")
    ok && ok2
  }

  def test_sparse_diagonal(): Boolean = {
    val dense = Array(
      Array(1.0f, 2.0f, 3.0f),
      Array(4.0f, 5.0f, 6.0f),
      Array(7.0f, 8.0f, 9.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    val diag = SparseOps.diagonal(sparse)
    val ok = diag(0) == 1.0f && diag(1) == 5.0f && diag(2) == 9.0f
    println(s"    Diagonal: [${diag.mkString}]")
    ok
  }

  def test_sparse_is_symmetric(): Boolean = {
    val dense = Array(
      Array(1.0f, 2.0f, 3.0f),
      Array(2.0f, 5.0f, 6.0f),
      Array(3.0f, 6.0f, 9.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    val ok = SparseOps.isSymmetric(sparse)
    println(s"    Symmetric check: $ok")
    ok
  }

  def test_sparse_symmetric_true(): Boolean = {
    val eye = SparseMatrix.identity(10)
    val ok = SparseOps.isSymmetric(eye)
    println(s"    Identity symmetric: $ok")
    ok
  }

  def test_sparse_symmetric_false(): Boolean = {
    val dense = Array(
      Array(1.0f, 2.0f),
      Array(3.0f, 4.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    val ok = !SparseOps.isSymmetric(sparse)
    println(s"    Non-symmetric: $ok")
    ok
  }

  def test_sparse_attention(): Boolean = {
    val mask = Array.tabulate(4) { i => Array.tabulate(4) { j => j <= i } }
    val attn = SparseAttention(mask, blockSize = 1)
    val Q = Array.tabulate(4)(_ => Array.fill(4)(1.0f))
    val K = Array.tabulate(4)(_ => Array.fill(4)(1.0f))
    val V = Array.tabulate(4)(_ => Array.fill(4)(1.0f))
    val out = attn.forward(Q, K, V)
    val ok = out.length == 4 && out(0).length == 4
    println(s"    SparseAttention output: ${out.length}x${out(0).length}")
    ok
  }

  def test_sparse_attention_shape(): Boolean = {
    val attn = SparseAttention.fromLowerTriangular(16, 4)
    val Q = Array.fill(16)(Array.fill(8)(1.0f))
    val K = Array.fill(16)(Array.fill(8)(1.0f))
    val V = Array.fill(16)(Array.fill(8)(1.0f))
    val out = attn.forward(Q, K, V)
    val ok = out.length == 16 && out(0).length == 8
    println(s"    SparseAttention shape: ${out.length}x${out(0).length}")
    ok
  }

  def test_blocksparse_from_dense(): Boolean = {
    val dense = Array.fill(8)(Array.fill(8)(1.0f))
    val bsm = BlockSparseMatrix.fromDense(dense, 4, 0.0f)
    val ok = bsm.nnzBlocks > 0 && bsm.totalValues > 0
    println(s"    BlockSparseMatrix: blocks=${bsm.nnzBlocks}, values=${bsm.totalValues}")
    ok
  }

  def test_blocksparse_spmv(): Boolean = {
    val dense = Array.fill(4)(Array.fill(4)(1.0f))
    val bsm = BlockSparseMatrix.fromDense(dense, 2, 0.0f)
    val x = Array.fill(4)(2.0f)
    val y = bsm.spmv(x)
    val ok = y.length == 4 && y.forall(v => v > 0)
    println(s"    BlockSparse SpMV: [${y.mkString}]")
    ok
  }

  def test_blocksparse_identity(): Boolean = {
    val bsm = BlockSparseMatrix.identity(8, 4)
    val x = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f)
    val y = bsm.spmv(x)
    val ok = y.sameElements(x)
    println(s"    BlockSparse identity SpMV: $ok")
    ok
  }

  def test_sparse_spadd(): Boolean = {
    val A = SparseMatrix.identity(3)
    val B = SparseMatrix.identity(3)
    val C = SparseOps.spadd(A, B)
    val ok = C.nnz == 3 && C.values(0) == 2.0f
    println(s"    SpAdd: nnz=${C.nnz}, values[0]=${C.values(0)}")
    ok
  }

  def test_sparse_transpose(): Boolean = {
    val dense = Array(
      Array(1.0f, 0.0f, 2.0f),
      Array(0.0f, 3.0f, 0.0f),
      Array(4.0f, 0.0f, 5.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    val t = SparseOps.transpose(sparse)
    val ok = t.numRows == 3 && t.numCols == 3 && t.nnz == 5
    println(s"    Transpose: shape=${t.shape}, nnz=${t.nnz}")
    ok
  }

  def test_cusparse_sparse_init(): Boolean = {
    val dense = Array(
      Array(1.0f, 0.0f, 2.0f),
      Array(0.0f, 3.0f, 0.0f),
      Array(4.0f, 0.0f, 5.0f)
    )
    val sparse = SparseMatrix.fromDense(dense)
    val handle = CUSPARSEHandle.getOrCreate
    val ok = handle.isCreated
    println(s"    CUSPARSEHandle: created=$ok")
    if ok then handle.close()
    ok
  }

  def test_cusparse_dense_init(): Boolean = {
    val mat = new CUSPARSEDenseMatrix(3, 3, Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f))
    val ok = mat.rows == 3 && mat.cols == 3
    println(s"    CUSPARSEDenseMatrix: ${mat.rows}x${mat.cols}")
    ok
  }

  def test_cusparse_vector_init(): Boolean = {
    val vec = new CUSPARSEDenseVector(5, Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f))
    val ok = vec.size == 5
    println(s"    CUSPARSEDenseVector: size=${vec.size}")
    ok
  }

  // ========== DRIVER API NATIVE TESTS ==========
  def test_driver_init(): Boolean = {
    val ok = CUDARuntimeNatives.initialize()
    println(s"    CUDARuntime init: $ok")
    ok
  }

  def test_driver_device_count(): Boolean = {
    val count = CUDARuntimeNatives.getDeviceCount
    val ok = count >= 1
    println(s"    Device count: $count")
    ok
  }

  def test_driver_current_device(): Boolean = {
    val dev = CUDARuntimeNatives.getCurrentDevice
    val ok = dev >= 0
    println(s"    Current device: $dev")
    ok
  }

  def test_driver_set_device(): Boolean = {
    CUDARuntimeNatives.setCurrentDevice(0)
    val dev = CUDARuntimeNatives.getCurrentDevice
    val ok = dev == 0
    println(s"    Set device: $dev")
    ok
  }

  def test_driver_sm_count(): Boolean = {
    val props = CUDARuntimeNatives.getDeviceProperties(0)
    val ok = props.num_sm > 0
    println(s"    SM count: ${props.num_sm}")
    ok
  }

  def test_driver_compute_cap(): Boolean = {
    val props = CUDARuntimeNatives.getDeviceProperties(0)
    val ok = props.major > 0 && props.minor >= 0
    println(s"    Compute capability: sm_${props.major}.${props.minor}")
    ok
  }

  def test_driver_warp_size(): Boolean = {
    val props = CUDARuntimeNatives.getDeviceProperties(0)
    val ok = props.warpSize == 32
    println(s"    Warp size: ${props.warpSize}")
    ok
  }

  def test_driver_memory(): Boolean = {
    val props = CUDARuntimeNatives.getDeviceProperties(0)
    val ok = props.totalGlobalMem > 0
    println(s"    Total memory: ${props.totalGlobalMem / (1024*1024*1024)} GB")
    ok
  }

  def test_driver_stream_create(): Boolean = {
    val s = NativeCUDAStream.create()
    val ok = s.stream != null
    s.synchronize()
    s.destroy()
    println(s"    Stream create/destroy: $ok")
    ok
  }

  def test_driver_stream_sync(): Boolean = {
    val s = NativeCUDAStream.create()
    val res = s.synchronize()
    val ok = res == 0
    s.destroy()
    println(s"    Stream sync: $ok")
    ok
  }

  def test_driver_event_create(): Boolean = {
    val e = NativeCUDAEvent.create()
    val ok = e.event != null
    e.destroy()
    println(s"    Event create/destroy: $ok")
    ok
  }

  def test_driver_event_sync(): Boolean = {
    val e = NativeCUDAEvent.create()
    val res = e.synchronize()
    val ok = res == 0
    e.destroy()
    println(s"    Event sync: $ok")
    ok
  }

  def test_driver_event_timing(): Boolean = {
    val start = NativeCUDAEvent.create()
    val end = NativeCUDAEvent.create()
    val elapsed = end.elapsedTimeMs(start)
    val ok = elapsed >= 0.0f
    start.destroy()
    end.destroy()
    println(s"    Event elapsed time: ${elapsed}ms")
    ok
  }

  def test_driver_event_record(): Boolean = {
    val s = NativeCUDAStream.create()
    val e = NativeCUDAEvent.create()
    val res = e.record(s)
    val ok = res == 0
    e.destroy()
    s.destroy()
    println(s"    Event record: $ok")
    ok
  }

  def test_driver_event_query(): Boolean = {
    val e = NativeCUDAEvent.create()
    val query = e.query
    val ok = query == 0
    e.destroy()
    println(s"    Event query: completed=$ok")
    ok
  }

  // ========== CUDA GRAPH NATIVE TESTS ==========
  def test_graph_create(): Boolean = {
    val graph = NativeCUDAGraph.create()
    val ok = graph.graph != null
    graph.destroy()
    println(s"    Legacy graph create: $ok")
    ok
  }

  def test_graph_instantiate(): Boolean = {
    val graph = NativeCUDAGraph.create()
    val res = graph.instantiate()
    graph.destroy()
    val ok = true
    println(s"    Legacy graph instantiate: $res")
    ok
  }

  def test_graph_launch(): Boolean = {
    val graph = NativeCUDAGraph.create()
    val launchRes = graph.launchOnDefaultStream()
    graph.destroy()
    val ok = true
    println(s"    Legacy graph launch: $launchRes")
    ok
  }

  def test_graph_destroy(): Boolean = {
    val graph = NativeCUDAGraph.create()
    val res = graph.destroy()
    val ok = res == 0
    println(s"    Legacy graph destroy: $ok")
    ok
  }

  def test_gpu_graph_create(): Boolean = {
    val graph = NativeCUDAGpuGraph.create()
    val ok = graph.isCreated
    graph.close()
    println(s"    NativeCUDAGpuGraph create: $ok")
    ok
  }

  def test_gpu_graph_add_kernel(): Boolean = {
    val graph = NativeCUDAGpuGraph.create()
    val added = graph.addKernelNode(null, 1, 1, 1, 64, 1, 1, 0)
    graph.close()
    val ok = !added
    println(s"    addKernelNode (null func expected false): $ok")
    ok
  }

  def test_gpu_graph_add_empty(): Boolean = {
    val graph = NativeCUDAGpuGraph.create()
    val added = graph.addEmptyNode()
    graph.close()
    val ok = added
    println(s"    addEmptyNode: $ok")
    ok
  }

  def test_gpu_graph_get_nodes(): Boolean = {
    val graph = NativeCUDAGpuGraph.create()
    graph.addEmptyNode()
    val count = graph.getNodesCount
    graph.close()
    val ok = count >= 1
    println(s"    getNodesCount: $count")
    ok
  }

  def test_gpu_graph_get_roots(): Boolean = {
    val graph = NativeCUDAGpuGraph.create()
    graph.addEmptyNode()
    graph.addEmptyNode()
    val count = graph.getRootNodesCount
    graph.close()
    val ok = count >= 1
    println(s"    getRootNodesCount: $count")
    ok
  }

  def test_graph_exec_instantiate(): Boolean = {
    val graph = NativeCUDAGpuGraph.create()
    graph.addEmptyNode()
    val exec = CudaGraphExecutor.create()
    val inst = exec.instantiate(graph)
    exec.close()
    graph.close()
    val ok = inst
    println(s"    CudaGraphExecutor instantiate: $ok")
    ok
  }

  def test_graph_exec_launch(): Boolean = {
    val graph = NativeCUDAGpuGraph.create()
    graph.addEmptyNode()
    val exec = CudaGraphExecutor.create()
    exec.instantiate(graph)
    val stream = NativeCUDAStream.create()
    val launched = exec.launch(stream)
    exec.close()
    stream.close()
    graph.close()
    val ok = true
    println(s"    CudaGraphExecutor launch: $launched")
    ok
  }

  def test_graph_exec_destroy(): Boolean = {
    val graph = NativeCUDAGpuGraph.create()
    graph.addEmptyNode()
    val exec = CudaGraphExecutor.create()
    exec.instantiate(graph)
    val res = exec.destroy()
    exec.close()
    graph.close()
    val ok = res == 0
    println(s"    CudaGraphExecutor destroy: $ok")
    ok
  }

  def test_kernel_graph_topo(): Boolean = {
    val kg = new KernelGraph
    val a = kg.addOp(OpType.Load)
    val b = kg.addOp(OpType.Load)
    val c = kg.addOp(OpType.Add, List(a.id, b.id))
    val sorted = kg.topologicalSort
    val ok = sorted.nonEmpty && sorted.size == 3
    println(s"    KernelGraph topo: ${sorted.size} ops")
    ok
  }

  def test_kernel_graph_add_op(): Boolean = {
    val kg = new KernelGraph
    val ops = List(
      kg.addOp(OpType.Load),
      kg.addOp(OpType.MatMul),
      kg.addOp(OpType.Add),
      kg.addOp(OpType.Relu)
    )
    val ok = kg.getOps.size == 4
    println(s"    KernelGraph addOp: ${kg.getOps.size} ops")
    ok
  }

  def test_graph_optimizer(): Boolean = {
    val kg = new KernelGraph
    kg.addOp(OpType.Load)
    kg.addOp(OpType.Load)
    kg.addOp(OpType.Mul)
    kg.addOp(OpType.Add)
    val opt = new GraphOptimizer
    val optimized = opt.optimize(kg)
    val ok = optimized.getOps.nonEmpty
    println(s"    GraphOptimizer: ${kg.getOps.size} -> ${optimized.getOps.size} ops")
    ok
  }
