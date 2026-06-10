package cuda.dsl.collections

import cuda.dsl.DSL._
import cuda.dsl.collections.Implicits._
import cuda.dsl.core.Types.given_MemoryOps_Float
import cuda.dsl.core.Types.given_MemoryOps_Int
import cuda.dsl.core.Types.given_compileTime_Float

/** Test program for GPU Collection operations */
@main def TestGPUCollection(): Unit = {
  println("=" * 60)
  println("GPU Collection Test")
  println("=" * 60)
  println(s"Backend: ${DeviceSelector.backendName}")
  println(s"MPS: ${DeviceSelector.isMPS}, CUDA: ${DeviceSelector.isCUDA}, CPU: ${DeviceSelector.isCPU}")
  println()

  // Test 1: GPUArray from Array
  println("Test 1: GPUArray from Array")
  val arr = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
  val gpuArr = GPUArray.fromArray(arr)
  println(s"  Created: ${gpuArr}")
  println()

  // Test 2: Map operation
  println("Test 2: Map (x * 2)")
  val mapped = gpuArr.map(_ * 2.0f)
  val mappedResult = mapped.toArray
  println(s"  Input:  ${arr.mkString(", ")}")
  println(s"  Output: ${mappedResult.mkString(", ")}")
  assert(mappedResult.sameElements(Array(2.0f, 4.0f, 6.0f, 8.0f, 10.0f)), "Map * 2 failed")
  println("  PASSED!")
  println()

  // Test 3: Filter operation
  println("Test 3: Filter (> 3)")
  val filtered = gpuArr.filter(_ > 3.0f)
  val filteredResult = filtered.toArray
  println(s"  Input:  ${arr.mkString(", ")}")
  println(s"  Output: ${filteredResult.mkString(", ")}")
  assert(filteredResult.sameElements(Array(4.0f, 5.0f)), "Filter > 3 failed")
  println("  PASSED!")
  println()

  // Test 4: Reduce (sum)
  println("Test 4: Reduce (sum)")
  val sum = gpuArr.reduce(_ + _)
  println(s"  Input:  ${arr.mkString(", ")}")
  println(s"  Sum:    $sum")
  assert(sum == 15.0f, s"Sum failed: expected 15.0f, got $sum")
  println("  PASSED!")
  println()

  // Test 5: Chained operations
  println("Test 5: Chained operations (map -> filter -> reduce)")
  val chained = gpuArr
    .map(_ * 2.0f)
    .filter(_ > 5.0f)
    .reduce(_ + _)
  println(s"  Input:  ${arr.mkString(", ")}")
  println(s"  (x * 2).filter(_ > 5).sum = $chained")
  val expected = arr.map(_ * 2.0f).filter(_ > 5.0f).sum
  assert(chained == expected, s"Chained failed: expected $expected, got $chained")
  println("  PASSED!")
  println()

  // Test 6: Using implicit .enable_gpu
  println("Test 6: Using implicit .enable_gpu")
  val seqResult = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
    .enable_gpu
    .map(_ + 1.0f)
    .filter(_ > 3.0f)
    .toSeq
  println(s"  Input:  List(1, 2, 3, 4, 5)")
  println(s"  Output: ${seqResult.mkString(", ")}")
  // map(_ + 1.0f): 2, 3, 4, 5, 6
  // filter(_ > 3.0f): 4, 5, 6
  assert(seqResult == Seq(4.0f, 5.0f, 6.0f), s"Implicit enable_gpu failed: expected Seq(4, 5, 6), got $seqResult")
  println("  PASSED!")
  println()

  // Test 7: GPUSeq
  println("Test 7: GPUSeq operations")
  val gpuSeq = GPUSeq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
  val seqSum = gpuSeq.reduce(_ + _)
  println(s"  GPUSeq sum: $seqSum")
  assert(seqSum == 15.0f, s"GPUSeq sum failed")
  println("  PASSED!")
  println()

  // Test 8: Int type (only works on CPU backend, MPS only supports Float)
  if (DeviceSelector.backendName == "CPU") {
    println("Test 8: Int type operations (CPU only)")
    val intArr = Array(1, 2, 3, 4, 5)
    val intGpuArr = GPUArray.fromArray(intArr)
    val intMapped = intGpuArr.map(_ * 2)
    val intResult = intMapped.toArray
    println(s"  Input:  ${intArr.mkString(", ")}")
    println(s"  Output: ${intResult.mkString(", ")}")
    assert(intResult.sameElements(Array(2, 4, 6, 8, 10)), "Int map failed")
    println("  PASSED!")
  } else {
    println("Test 8: Int type operations - SKIPPED (MPS only supports Float)")
  }
  println()

  // Test 9: Zero initialization
  println("Test 9: GPUArray.zeros")
  val zerosArr = GPUArray.zeros[Float](3)
  val zerosResult = zerosArr.toArray
  println(s"  zeros(3): ${zerosResult.mkString(", ")}")
  assert(zerosResult.sameElements(Array(0.0f, 0.0f, 0.0f)), "Zeros failed")
  println("  PASSED!")
  println()

  // Cleanup
  gpuArr.free()
  mapped.free()
  filtered.free()
  zerosArr.free()
  println("All tests passed!")
  println("=" * 60)
}
