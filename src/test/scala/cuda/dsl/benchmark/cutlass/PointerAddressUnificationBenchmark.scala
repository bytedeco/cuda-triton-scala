package cuda.dsl.benchmark.cutlass

import cuda.dsl.core.*
import org.bytedeco.javacpp.*

/** Scala benchmark verifying pointer address unification.
  *
  * Tests that all Ptr types (IntPtr, FloatPtr, LongPtr, DoublePtr, BytePtr, PointerPtr)
  * correctly unify addresses between:
  * 1. The internal Ptr[T] raw address
  * 2. The JavaCPP parent Pointer's internal address field
  * 3. The rawAddress() method
  *
  * This is critical for correct memory access — if these don't match,
  * memory reads/writes will go to the wrong location.
  */
object PointerAddressUnificationBenchmark:

  private var passed = 0
  private var failed = 0

  // =============================================================================
  // Test Results
  // =============================================================================

  case class TestResult(name: String, passed: Boolean, details: String)

  private val results = scala.collection.mutable.ListBuffer[TestResult]()

  // =============================================================================
  // Test Suite 1: Individual Ptr Address Consistency
  // =============================================================================

  def testIntPtrAddressConsistency(): TestResult =
    try
      val testAddr = 0x1234567890L
      val intPtr = IntPtr.fromAddress(testAddr)

      val ptrAddress = intPtr.rawAddress
      val parentAddress = intPtr.address()

      if ptrAddress == testAddr && parentAddress == testAddr then
        passed += 1
        TestResult("IntPtr Address Consistency", true,
          s"deviceAddress=0x${ptrAddress.toHexString}, Pointer.address()=0x${parentAddress.toHexString}")
      else
        failed += 1
        TestResult("IntPtr Address Consistency", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got deviceAddress=0x${ptrAddress.toHexString}, parent=0x${parentAddress.toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("IntPtr Address Consistency", false, s"EXCEPTION: ${e.getMessage}")

  def testFloatPtrAddressConsistency(): TestResult =
    try
      val testAddr = 0xABCDEF0000L
      val floatPtr = FloatPtr.fromAddress(testAddr)

      val ptrAddress = floatPtr.rawAddress
      val parentAddress = floatPtr.address()

      if ptrAddress == testAddr && parentAddress == testAddr then
        passed += 1
        TestResult("FloatPtr Address Consistency", true,
          s"deviceAddress=0x${ptrAddress.toHexString}, Pointer.address()=0x${parentAddress.toHexString}")
      else
        failed += 1
        TestResult("FloatPtr Address Consistency", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got deviceAddress=0x${ptrAddress.toHexString}, parent=0x${parentAddress.toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("FloatPtr Address Consistency", false, s"EXCEPTION: ${e.getMessage}")

  def testLongPtrAddressConsistency(): TestResult =
    try
      val testAddr = 0x111122223333L
      val longPtr = LongPtr.fromAddress(testAddr)

      val ptrAddress = longPtr.rawAddress
      val parentAddress = longPtr.address()

      if ptrAddress == testAddr && parentAddress == testAddr then
        passed += 1
        TestResult("LongPtr Address Consistency", true,
          s"deviceAddress=0x${ptrAddress.toHexString}, Pointer.address()=0x${parentAddress.toHexString}")
      else
        failed += 1
        TestResult("LongPtr Address Consistency", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got deviceAddress=0x${ptrAddress.toHexString}, parent=0x${parentAddress.toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("LongPtr Address Consistency", false, s"EXCEPTION: ${e.getMessage}")

  def testDoublePtrAddressConsistency(): TestResult =
    try
      val testAddr = 0xAAAABBBBCCCCL
      val doublePtr = DoublePtr.fromAddress(testAddr)

      val ptrAddress = doublePtr.rawAddress
      val parentAddress = doublePtr.address()

      if ptrAddress == testAddr && parentAddress == testAddr then
        passed += 1
        TestResult("DoublePtr Address Consistency", true,
          s"deviceAddress=0x${ptrAddress.toHexString}, Pointer.address()=0x${parentAddress.toHexString}")
      else
        failed += 1
        TestResult("DoublePtr Address Consistency", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got deviceAddress=0x${ptrAddress.toHexString}, parent=0x${parentAddress.toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("DoublePtr Address Consistency", false, s"EXCEPTION: ${e.getMessage}")

  def testBytePtrAddressConsistency(): TestResult =
    try
      val testAddr = 0xDEADBEEF0000L
      val bytePtr = BytePtr.fromAddress(testAddr)

      val ptrAddress = bytePtr.rawAddress
      val parentAddress = bytePtr.address()

      if ptrAddress == testAddr && parentAddress == testAddr then
        passed += 1
        TestResult("BytePtr Address Consistency", true,
          s"deviceAddress=0x${ptrAddress.toHexString}, Pointer.address()=0x${parentAddress.toHexString}")
      else
        failed += 1
        TestResult("BytePtr Address Consistency", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got deviceAddress=0x${ptrAddress.toHexString}, parent=0x${parentAddress.toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("BytePtr Address Consistency", false, s"EXCEPTION: ${e.getMessage}")

  def testPointerPtrAddressConsistency(): TestResult =
    try
      val testAddr = 0xCAFEBABE0000L
      val pointerPtr = PointerPtr.fromAddress(testAddr)

      val ptrAddress = pointerPtr.rawAddress
      val parentAddress = pointerPtr.address()

      if ptrAddress == testAddr && parentAddress == testAddr then
        passed += 1
        TestResult("PointerPtr Address Consistency", true,
          s"deviceAddress=0x${ptrAddress.toHexString}, Pointer.address()=0x${parentAddress.toHexString}")
      else
        failed += 1
        TestResult("PointerPtr Address Consistency", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got deviceAddress=0x${ptrAddress.toHexString}, parent=0x${parentAddress.toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("PointerPtr Address Consistency", false, s"EXCEPTION: ${e.getMessage}")

  // =============================================================================
  // Test Suite 2: Cross-Type Address Casting
  // =============================================================================

  def testIntToFloatCast(): TestResult =
    try
      val testAddr = 0x12345678L
      val intPtr = IntPtr.fromAddress(testAddr)
      val floatPtr = FloatPtr.fromAddress(testAddr)

      if intPtr.rawAddress == floatPtr.rawAddress && intPtr.address() == floatPtr.address() then
        passed += 1
        TestResult("IntPtr -> FloatPtr Cross-Type", true,
          s"Both point to 0x${testAddr.toHexString}")
      else
        failed += 1
        TestResult("IntPtr -> FloatPtr Cross-Type", false,
          s"MISMATCH: IntPtr=0x${intPtr.rawAddress.toHexString}, FloatPtr=0x${floatPtr.rawAddress.toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("IntPtr -> FloatPtr Cross-Type", false, s"EXCEPTION: ${e.getMessage}")

  def testAllTypesSameAddress(): TestResult =
    try
      val testAddr = 0xDEADBEEFL
      val intPtr = IntPtr.fromAddress(testAddr)
      val floatPtr = FloatPtr.fromAddress(testAddr)
      val longPtr = LongPtr.fromAddress(testAddr)
      val doublePtr = DoublePtr.fromAddress(testAddr)
      val bytePtr = BytePtr.fromAddress(testAddr)
      val pointerPtr = PointerPtr.fromAddress(testAddr)

      val allParentSame = intPtr.address() == floatPtr.address()
        && floatPtr.address() == longPtr.address()
        && longPtr.address() == doublePtr.address()
        && doublePtr.address() == bytePtr.address()
        && bytePtr.address() == pointerPtr.address()
        && pointerPtr.address() == testAddr

      val allRawSame = intPtr.rawAddress == floatPtr.rawAddress
        && floatPtr.rawAddress == longPtr.rawAddress
        && longPtr.rawAddress == doublePtr.rawAddress
        && doublePtr.rawAddress == bytePtr.rawAddress
        && bytePtr.rawAddress == pointerPtr.rawAddress
        && pointerPtr.rawAddress == testAddr

      if allParentSame && allRawSame then
        passed += 1
        TestResult("All 6 Ptr Types -> Same Address", true,
          s"All types consistently point to 0x${testAddr.toHexString}")
      else
        failed += 1
        TestResult("All 6 Ptr Types -> Same Address", false,
          s"MISMATCH: parent=${intPtr.address()}, raw=${intPtr.rawAddress}")
    catch case e: Exception =>
      failed += 1
      TestResult("All 6 Ptr Types -> Same Address", false, s"EXCEPTION: ${e.getMessage}")

  def testBytePtrConversions(): TestResult =
    try
      val testAddr = 0x9876543210L
      val bytePtr = BytePtr.fromAddress(testAddr)

      val asFloat = bytePtr.asFloatPtr
      val asInt = bytePtr.asIntPtr
      val asLong = bytePtr.asLongPtr
      val asDouble = bytePtr.asDoublePtr

      val allMatch = asFloat.rawAddress == testAddr
        && asInt.rawAddress == testAddr
        && asLong.rawAddress == testAddr
        && asDouble.rawAddress == testAddr
        && asFloat.address() == testAddr
        && asInt.address() == testAddr
        && asLong.address() == testAddr
        && asDouble.address() == testAddr

      if allMatch then
        passed += 1
        TestResult("BytePtr -> TypedPtr Conversions", true,
          s"All conversions preserve address 0x${testAddr.toHexString}")
      else
        failed += 1
        TestResult("BytePtr -> TypedPtr Conversions", false,
          s"MISMATCH after conversion")
    catch case e: Exception =>
      failed += 1
      TestResult("BytePtr -> TypedPtr Conversions", false, s"EXCEPTION: ${e.getMessage}")

  // =============================================================================
  // Test Suite 3: Edge Cases
  // =============================================================================

  def testZeroAddress(): TestResult =
    try
      val intPtr = IntPtr.fromAddress(0L)
      val floatPtr = FloatPtr.fromAddress(0L)

      if intPtr.rawAddress == 0 && intPtr.address() == 0
        && floatPtr.rawAddress == 0 && floatPtr.address() == 0 then
        passed += 1
        TestResult("Zero Address Handling", true, "Zero address correctly preserved")
      else
        failed += 1
        TestResult("Zero Address Handling", false,
          s"MISMATCH: IntPtr=${intPtr.rawAddress}, FloatPtr=${floatPtr.rawAddress}")
    catch case e: Exception =>
      failed += 1
      TestResult("Zero Address Handling", false, s"EXCEPTION: ${e.getMessage}")

  def testMaxAddress(): TestResult =
    try
      val maxAddr = 0x7FFFFFFFFFFFFFFFL
      val floatPtr = FloatPtr.fromAddress(maxAddr)
      val doublePtr = DoublePtr.fromAddress(maxAddr)

      if floatPtr.rawAddress == maxAddr && floatPtr.address() == maxAddr
        && doublePtr.rawAddress == maxAddr && doublePtr.address() == maxAddr then
        passed += 1
        TestResult("Maximum 64-bit Address", true,
          s"Max address 0x${maxAddr.toHexString} correctly preserved")
      else
        failed += 1
        TestResult("Maximum 64-bit Address", false,
          s"MISMATCH: FloatPtr=${floatPtr.rawAddress}, DoublePtr=${doublePtr.rawAddress}")
    catch case e: Exception =>
      failed += 1
      TestResult("Maximum 64-bit Address", false, s"EXCEPTION: ${e.getMessage}")

  def testNullEquivalentAddress(): TestResult =
    try
      val nullAddr = 1L
      val intPtr = IntPtr.fromAddress(nullAddr)

      if intPtr.rawAddress == nullAddr && intPtr.address() == nullAddr then
        passed += 1
        TestResult("NULL-equivalent Address (1)", true,
          s"NULL-equivalent 0x1 correctly preserved")
      else
        failed += 1
        TestResult("NULL-equivalent Address (1)", false,
          s"MISMATCH: expected=0x1, got ${intPtr.rawAddress}")
    catch case e: Exception =>
      failed += 1
      TestResult("NULL-equivalent Address (1)", false, s"EXCEPTION: ${e.getMessage}")

  // =============================================================================
  // Test Suite 4: Implicit Conversions to JavaCPP Pointer
  // =============================================================================

  def testIntPtrToJavaCPPIntPointer(): TestResult =
    try
      val testAddr = 0x55555555L
      val intPtr = IntPtr.fromAddress(testAddr)

      // Use implicit conversion to JavaCPP IntPointer
      val javaPtr: IntPointer = intPtr

      if javaPtr.address() == testAddr then
        passed += 1
        TestResult("IntPtr -> JavaCPP IntPointer", true,
          s"JavaCPP IntPointer.address() = 0x${javaPtr.address().toHexString}")
      else
        failed += 1
        TestResult("IntPtr -> JavaCPP IntPointer", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got 0x${javaPtr.address().toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("IntPtr -> JavaCPP IntPointer", false, s"EXCEPTION: ${e.getMessage}")

  def testFloatPtrToJavaCPPFloatPointer(): TestResult =
    try
      val testAddr = 0x66666666L
      val floatPtr = FloatPtr.fromAddress(testAddr)

      val javaPtr: FloatPointer = floatPtr

      if javaPtr.address() == testAddr then
        passed += 1
        TestResult("FloatPtr -> JavaCPP FloatPointer", true,
          s"JavaCPP FloatPointer.address() = 0x${javaPtr.address().toHexString}")
      else
        failed += 1
        TestResult("FloatPtr -> JavaCPP FloatPointer", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got 0x${javaPtr.address().toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("FloatPtr -> JavaCPP FloatPointer", false, s"EXCEPTION: ${e.getMessage}")

  def testPtrToJavaCPPBytePointer(): TestResult =
    try
      val testAddr = 0x77777777L
      val ptr = Ptr.fromAddress[Float](testAddr)
      val javaPtr: BytePointer = ptr.toPointer

      if javaPtr.address() == testAddr then
        passed += 1
        TestResult("Ptr[T].toPointer -> JavaCPP BytePointer", true,
          s"BytePointer.address() = 0x${javaPtr.address().toHexString}")
      else
        failed += 1
        TestResult("Ptr[T].toPointer -> JavaCPP BytePointer", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got 0x${javaPtr.address().toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("Ptr[T].toPointer -> JavaCPP BytePointer", false, s"EXCEPTION: ${e.getMessage}")

  // =============================================================================
  // Test Suite 5: PointerPtr toPointerPointer
  // =============================================================================

  def testPointerPtrToPointerPointer(): TestResult =
    try
      val testAddr = 0xAAAAAAAAAAAL
      val pointerPtr = PointerPtr.fromAddress(testAddr)

      val pp = pointerPtr.toPointerPointer

      if pp.address() == testAddr then
        passed += 1
        TestResult("PointerPtr -> PointerPointer", true,
          s"PointerPointer.address() = 0x${pp.address().toHexString}")
      else
        failed += 1
        TestResult("PointerPtr -> PointerPointer", false,
          s"MISMATCH: expected=0x${testAddr.toHexString}, got 0x${pp.address().toHexString}")
    catch case e: Exception =>
      failed += 1
      TestResult("PointerPtr -> PointerPointer", false, s"EXCEPTION: ${e.getMessage}")

  // =============================================================================
  // Test Suite 6: Multiple Iteration Stress
  // =============================================================================

  def testMultipleIterations(): TestResult =
    try
      var allConsistent = true
      val iterations = 1000

      for i <- 0 until iterations do
        val addr = (i * 0x1000L).abs

        val intPtr = IntPtr.fromAddress(addr)
        val floatPtr = FloatPtr.fromAddress(addr)
        val longPtr = LongPtr.fromAddress(addr)
        val doublePtr = DoublePtr.fromAddress(addr)
        val bytePtr = BytePtr.fromAddress(addr)
        val pointerPtr = PointerPtr.fromAddress(addr)

        if intPtr.rawAddress != addr || intPtr.address() != addr then allConsistent = false
        if floatPtr.rawAddress != addr || floatPtr.address() != addr then allConsistent = false
        if longPtr.rawAddress != addr || longPtr.address() != addr then allConsistent = false
        if doublePtr.rawAddress != addr || doublePtr.address() != addr then allConsistent = false
        if bytePtr.rawAddress != addr || bytePtr.address() != addr then allConsistent = false
        if pointerPtr.rawAddress != addr || pointerPtr.address() != addr then allConsistent = false

      if allConsistent then
        passed += 1
        TestResult(s"Multiple Iterations ($iterations)", true,
          s"All $iterations iterations consistent")
      else
        failed += 1
        TestResult(s"Multiple Iterations ($iterations)", false,
          s"Inconsistency detected in $iterations iterations")
    catch case e: Exception =>
      failed += 1
      TestResult(s"Multiple Iterations", false, s"EXCEPTION: ${e.getMessage}")

  // =============================================================================
  // Main
  // =============================================================================

  def main(args: Array[String]): Unit =
    println("=" * 80)
    println("SCALA POINTER ADDRESS UNIFICATION BENCHMARK")
    println("=" * 80)
    println("Verifies that all Ptr types unify addresses between:")
    println("  1. Internal Ptr[T] raw address")
    println("  2. JavaCPP parent Pointer.address()")
    println("  3. rawAddress() method")
    println()

    // Test Suite 1
    println("=== Test Suite 1: Individual Ptr Address Consistency ===")
    results.addOne(testIntPtrAddressConsistency())
    results.addOne(testFloatPtrAddressConsistency())
    results.addOne(testLongPtrAddressConsistency())
    results.addOne(testDoublePtrAddressConsistency())
    results.addOne(testBytePtrAddressConsistency())
    results.addOne(testPointerPtrAddressConsistency())

    // Test Suite 2
    println("\n=== Test Suite 2: Cross-Type Address Casting ===")
    results.addOne(testIntToFloatCast())
    results.addOne(testAllTypesSameAddress())
    results.addOne(testBytePtrConversions())

    // Test Suite 3
    println("\n=== Test Suite 3: Edge Cases ===")
    results.addOne(testZeroAddress())
    results.addOne(testMaxAddress())
    results.addOne(testNullEquivalentAddress())

    // Test Suite 4
    println("\n=== Test Suite 4: Implicit Conversions to JavaCPP Pointer ===")
    results.addOne(testIntPtrToJavaCPPIntPointer())
    results.addOne(testFloatPtrToJavaCPPFloatPointer())
    results.addOne(testPtrToJavaCPPBytePointer())

    // Test Suite 5
    println("\n=== Test Suite 5: PointerPtr to PointerPointer ===")
    results.addOne(testPointerPtrToPointerPointer())

    // Test Suite 6
    println("\n=== Test Suite 6: Multiple Iteration Stress ===")
    results.addOne(testMultipleIterations())

    // Print summary
    println("\n" + "=" * 80)
    println("TEST SUMMARY")
    println("=" * 80)

    results.foreach: r =>
      val marker = if r.passed then "[OK]" else "[!!]"
      val status = if r.passed then "PASS" else "FAIL"
      println(f"$marker%-4s $status%-4s ${r.name}%-45s ${r.details}")

    println("-" * 80)
    println(f"Total: ${results.size} tests | Passed: $passed | Failed: $failed")
    println("=" * 80)

    if failed == 0 then
      println("VERDICT: ALL POINTER ADDRESSES ARE UNIFIED AND CONSISTENT")
      println("         All Ptr types correctly maintain address consistency.")
    else
      println("VERDICT: ADDRESS INCONSISTENCY DETECTED!")
      println("         Some pointer types have mismatched addresses.")
      println("         THIS CAN CAUSE WRONG MEMORY ACCESS - FIX IMMEDIATELY!")
    println("=" * 80)
