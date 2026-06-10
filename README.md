[![cuda-triton-scala Banner](doc/img/banner.png)]
# cuda-triton-scala: Scala + CUDA + Triton Integration Examples

[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Scala](https://img.shields.io/badge/scala-3.x-DC322F?style=for-the-badge)](https://www.scala-lang.org/)
[![Triton](https://img.shields.io/badge/Triton-Inference-orange?style=for-the-badge)](https://github.com/triton-inference-server/server)

[English](README.md) | [简体中文](README_zh.md)

## Table of Contents

- [Introduction](#introduction)
- [Goals and Scope](#goals-and-scope)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation & Build](#installation--build)
- [Quick Start Examples](#quick-start-examples)
- [Project Layout](#project-layout)
- [Native Integration Patterns](#native-integration-patterns)
- [Triton Integration Guide](#triton-integration-guide)
- [Model Packaging for Triton](#model-packaging-for-triton)
- [Performance & Benchmarking](#performance--benchmarking)
- [Testing & CI Recommendations](#testing--ci-recommendations)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Contributing](#contributing)
- [License](#license)

## Introduction

`cuda-triton-scala` is a practical example repository that demonstrates common patterns to integrate Scala applications with native GPU-accelerated code (CUDA), and how to serve models via NVIDIA Triton Inference Server. The project is intentionally small and modular so that developers can adapt specific parts — native bindings, data encoding, model packaging, or client code — to their own production setups.

This README documents the repository layout, explains different interoperability strategies (JavaCPP / JNI / gRPC), shows how to prepare models for Triton, provides runnable examples, and includes troubleshooting tips for common native runtime issues.

## Goals and Scope

The primary goals of this project are:

- Provide a reproducible sbt-based build that compiles Scala code that interacts with native libraries
- Show multiple integration patterns for calling GPU-accelerated code from Scala (JavaCPP wrapper, JNI, external process)
- Demonstrate client-side code that communicates with Triton using HTTP/gRPC
- Provide example model packaging and minimal model config templates for Triton
- Offer guidelines and sample commands to benchmark and profile end-to-end latency and throughput

The repository intentionally avoids including large model binaries. Instead, it provides templates and small synthetic examples that are sufficient for local testing and CI.

## Key Features

- Scala 3-friendly project skeleton and sbt build
- Examples for calling native CUDA helpers from Scala (JavaCPP/JNI examples)
- Triton client examples (HTTP/gRPC) written in Scala
- Model repository templates and config examples for Triton
- Quick benchmarking scripts and suggestions for profiling GPU-bound workloads
- Troubleshooting section covering library loading, LD_LIBRARY_PATH, CUDA version mismatches and driver issues

## Requirements

- JDK 17 or newer (JDK 21 recommended for latest features)
- sbt 1.8+ (configured via `build.sbt`) — ensure sbt is installed and available on PATH
- Scala 3.x (project sources target Scala 3)
- CUDA toolkit and NVIDIA driver if you want to run GPU-backed examples locally
- Docker (optional) to run Triton Inference Server containers
- curl / grpcurl for quick manual requests to Triton during testing

Notes:

- The project compiles on the JVM without GPU present. Native examples will fail at runtime if required CUDA libraries or driver mismatches exist. See Troubleshooting for ways to run CPU-only or to mock native calls for CI.
- Some examples may rely on JavaCPP or other native-binding libraries; these dependencies are declared in `build.sbt` and are usually auto-extracted at runtime.

## Installation & Build

1. Clone the repository

```bash
git clone https://github.com/your-repo/cuda-triton-scala.git
cd cuda-triton-scala
```

2. Configure environment variables (if using GPU/native libs)

```bash
# Optional: point to CUDA toolkit/lib locations if not standard
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Optional: increase Java native memory or set java.library.path
export JAVA_TOOL_OPTIONS="-Djava.library.path=./native"
```

3. Build and compile with sbt

```bash
sbt clean compile
```

4. Run unit tests (if present)

```bash
sbt test
```

Dependency notes:

- This project intentionally keeps external dependencies minimal. If you need specific JavaCPP or other native bindings, add them to `build.sbt` as library dependencies or to your corporate artifact repository.

## Quick Start Examples

Below are runnable examples to verify local setup. Replace class names with actual mains found in `src/main/scala/examples`.

### 1) Simple CPU-only Scala example

This example demonstrates a purely Scala-based workload and is useful to confirm the sbt build and basic runtime.

```bash
sbt "runMain examples.SimpleCpuExample"
```

Expected output: info-level logs and a short printed summary of operations.

### 2) Local CUDA helper example

This example demonstrates calling a native library (CUDA helper) via JavaCPP or JNI. Make sure `LD_LIBRARY_PATH` or `java.library.path` points to the native library location.

```bash
# If native libs are in ./native
export LD_LIBRARY_PATH=$(pwd)/native:$LD_LIBRARY_PATH
sbt "runMain examples.SimpleCudaExample"
```

Typical failure modes: UnsatisfiedLinkError (native lib not found), or IllegalStateException when CUDA driver/device mismatch occurs. See Troubleshooting.

### 3) Triton client example (HTTP)

This example shows how to send a simple infer request to a running Triton server using HTTP API from Scala.

```bash
# Ensure Triton server is running and model repo contains 'simple_model'
sbt "runMain examples.TritonHttpClientExample"
```

### 4) Triton client example (gRPC)

```bash
sbt "runMain examples.TritonGrpcClientExample"
```

These examples include compact helper classes to create tensors and to decode model outputs into Scala collections.

## Project Layout

```
cuda-triton-scala/
├── README.md                        # English README (this file)
├── README_zh.md                     # Chinese README
├── build.sbt                        # sbt build definition
├── project/                         # sbt project config
├── src/
│   ├── main/
│   │   ├── scala/
│   │   │   ├── cuda/                # Native/CUDA integration helpers & wrappers
│   │   │   │   ├── CudaHelpers.scala
│   │   │   │   ├── CudaWrapper.java  # Optional JNI helper
│   │   │   │   └── NativeLoader.scala
│   │   │   ├── triton/              # Triton client utilities (HTTP/gRPC)
│   │   │   │   ├── TritonHttpClient.scala
│   │   │   │   ├── TritonGrpcClient.scala
│   │   │   │   └── TritonUtils.scala
│   │   │   └── examples/            # Example mains
│   │   │       ├── SimpleCpuExample.scala
│   │   │       ├── SimpleCudaExample.scala
│   │   │       ├── TritonHttpClientExample.scala
│   │   │       └── TritonGrpcClientExample.scala
│   │   └── resources/
│   │       ├── native/              # Optional bundled native libraries
│   │       └── triton_model_templates/ # model repo templates and config.pbtxt
│   └── test/
│       └── scala/
│           └── UnitTests.scala
└── scripts/
    ├── run_triton.sh                # Helper to launch triton (docker)
    └── benchmark.sh                 # Quick throughput/latency benchmark harness
```

Adjust the tree above to match your actual code layout. The example names are suggestions; use real classes present in your repository.

## Native Integration Patterns

This repository demonstrates multiple ways to integrate native GPU-accelerated code from Scala:

1. JavaCPP (recommended when prebuilt JavaCPP presets exist)
   - Advantage: minimal JNI boilerplate, auto-extraction of native libraries by Maven/Gradle/sbt if using bytedeco presets
   - Pattern: add JavaCPP preset dependency, call native APIs as regular Java classes

2. JNI (manual)
   - Advantage: full control over native API surface, optimized memory sharing
   - Pattern: write small JNI C/C++ wrapper that exposes constrained API, load with System.loadLibrary, call via Java methods

3. External process + RPC (isolated native process)
   - Advantage: isolates native crashes, simplifies JVM-native boundary
   - Pattern: start a small native server (e.g., gRPC or simple socket) and call from Scala, useful when native dependency management is hard in JVM

4. Process-level integration with Triton (preferred for serving)
   - Advantage: production-ready inference server with versioning, model control and multi-framework support
   - Pattern: package model into Triton model repository, start Triton, and call via HTTP/gRPC from Scala client

### Memory and Data Passing Considerations

- Avoid unnecessary copies: use direct/nio buffers or JavaCPP pointers when passing large tensors
- Zero-copy between native and JVM is possible but requires careful lifecycle management to avoid premature GC or use-after-free
- Be explicit about tensor dtype, endianness, and shape conventions when encoding/decoding

## Triton (triton-lang) Integration Guide

This repository targets OpenAI's Triton (the CUDA DSL, https://github.com/triton-lang/triton). Note: this is *not* NVIDIA's Triton Inference Server. OpenAI's Triton is a CUDA DSL primarily used for writing high-performance GPU kernels. This project demonstrates a Scala-first workflow:

- Author kernels using the Scala DSL / macros provided by this project (see `cuda.dsl.triton.jit`) or supply precompiled PTX/CUBIN kernels produced by your toolchain.
- At runtime, use JavaCPP-CUDA (Driver API) from Scala to load the compiled kernel module, allocate device memory, copy inputs/outputs, and launch kernels.

Why Scala-first?

- The codebase includes a Scala macro-based DSL (`cuda.dsl.dsl.TritonKernelMacro`) that can annotate Scala methods to be treated as Triton-style kernels. This enables expressing kernels in Scala source and compiling them to GPU code during your build or a separate native compilation step.
- Using JavaCPP-CUDA for the runtime ensures the JVM process can directly control the CUDA Driver API without writing manual JNI wrappers.

High-level workflow (no Python required):

1. Write kernel logic using the Scala DSL annotation `@cuda.dsl.triton.jit` or maintain precompiled PTX/CUBIN artifacts in `src/main/resources/native/kernels/`.
2. Build your Scala project; if you use a separate native compilation step, ensure produced PTX/CUBIN files are copied into resources or an accessible native directory.
3. From Scala, use JavaCPP's CUDA Driver API to:
   - Initialize CUDA (cuInit)
   - Create CUDA context
   - Load module (cuModuleLoad or cuModuleLoadData)
   - Get function handle (cuModuleGetFunction)
   - Allocate device memory and copy inputs (cuMemAlloc / cuMemcpyHtoD)
   - Launch kernel (cuLaunchKernel) with appropriate grid/block and stream
   - Copy outputs back (cuMemcpyDtoH)

This README contains concrete Scala examples showing how to perform the Driver API calls (see `src/main/scala/examples`).

Supported artifact workflows

- Precompiled PTX/CUBIN: Good for production deployment, compile kernels once for target architectures and deploy with the JVM artifact.
- On-device JIT (if your toolchain supports it): You can dynamically load PTX at runtime using cuModuleLoadData and launch kernels immediately.

Resource locations

- Put compiled artifacts in `src/main/resources/native/kernels/` to package them with your JAR.
- Or place them in an external `native/kernels/` directory and set `NATIVE_KERNEL_DIR` env var to point to that folder at runtime.

Security note

- Loading and running GPU kernels involves executing native code on the host GPU. Ensure compiled kernels come from trusted sources and validate input sizes before kernel launches to prevent out-of-bounds memory operations.


## Model Packaging for Triton

1. Export your model to a supported backend (ONNX, TensorRT plan, PyTorch traced model, TensorFlow SavedModel, etc.)
2. Create `config.pbtxt` with correct input/output specs
3. Place model artifact(s) under version folder (e.g., `models/my_model/1/`)
4. Optionally add `labels.txt`, `tokenizer` files, or pre/post-processing scripts

This repository includes `triton_model_templates/` containing minimal config templates for common backends.

## Performance & Benchmarking

### Microbenchmarks

- Use the `scripts/benchmark.sh` to run simple throughput tests. The script runs a client loop issuing requests and measuring latencies and QPS.
- For GPU-bound models, warm up the GPU by running several iterations before measurement.

### Profiling

- CPU profiling: use async-profiler, VisualVM, or your preferred Java profiler
- GPU profiling: use NVIDIA Nsight Systems or Nsight Compute to capture GPU kernel timelines and utilization

### Common Performance Tips

- Use batching where appropriate (increase model `max_batch_size` and client-side batching)
- For Triton, configure model instance groups (CPU vs GPU) and instance count for optimal throughput
- Use appropriate tensor layout and dtype (FP16/INT8 when acceptable) to decrease memory bandwidth
- Avoid synchronous file I/O on the request path; preload model artifacts and tokenizers at startup

## Testing & CI Recommendations

- Unit tests: mock native calls so CI can run without GPU
- Integration tests: include a small CPU-only model for Triton integration tests in CI
- Containerized CI: use Docker images with CPU-only Triton or mock servers for end-to-end checks

Sample CI matrix:

- JDK 17 + sbt compile + unit tests (Linux)
- JDK 17 + sbt compile + unit tests (macOS)
- Optional: Integration test job that starts Triton in Docker (CPU-only) and runs `TritonHttpClientExample`

## Troubleshooting & FAQ

Q: `UnsatisfiedLinkError` when loading native library

A: Ensure `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) contains the directory with your native `.so`/`.dylib`. Alternatively, set `-Djava.library.path` JVM flag to point to the directory. If using JavaCPP presets, ensure the correct artifact coordinates (platform classifier) are included in `build.sbt`.

Q: CUDA driver version mismatch or `CUDA_ERROR_UNKNOWN`

A: Verify the installed NVIDIA driver version is compatible with the CUDA toolkit version you are using. Use `nvidia-smi` to inspect driver and GPU info. If running inside Docker, ensure `--gpus` is available and the container image supports your driver.

Q: Triton returns model not ready or fails to load

A: Check Triton server logs for load errors. Common issues: missing model files, wrong `config.pbtxt`, unsupported model backend, or model requires a specific library (e.g., TensorRT) that is not installed.

Q: Performance is worse than expected

A: Check GPU utilization with `nvidia-smi` and Nsight tools. Possible causes: CPU bottleneck on preprocessing, small batch sizes, PCIe transfer overhead, or single-threaded client code.

Q: How to debug incompatible native symbols

A: Build native libraries with `-fPIC` and matching compilers for your platform. For JNI, confirm `javah`/jni headers and package names match generated natives.

## Examples Deep Dive (Code Snippets)

Below are illustrative Scala snippets to show typical usage patterns. These snippets are simplified; see actual examples in `src/main/scala/examples`.

### Native loader helper (Scala)

```scala
package cuda

object NativeLoader {
  def load(name: String): Unit = {
    val libPath = sys.env.get("NATIVE_LIB_DIR").getOrElse("./native")
    System.load(s"$libPath/lib$name.so")
  }
}
```

### JavaCPP-style call (pseudo)

```scala
import org.bytedeco.javacpp.Pointer

object CudaOps {
  def addVectors(a: Array[Float], b: Array[Float]): Array[Float] = {
    // Assuming a native helper was generated by JavaCPP
    val n = a.length
    val out = new Array[Float](n)
    NativeCuda.add(n, a, b, out)
    out
  }
}
```

### Triton HTTP client example (pseudo)

```scala
import scalaj.http.Http
import io.circe._

object TritonHttpClient {
  def infer(model: String, input: Array[Float]): Array[Float] = {
    val payload = Map("inputs" -> List(Map("name" -> "input__0", "shape" -> List(1, input.length), "data" -> input.toList)))
    val resp = Http(s"http://localhost:8000/v2/models/$model/infer").postData(io.circe.parser.parser(payload)).asString
    // parse response and return output array
    Array.emptyFloatArray
  }
}
```

## Contributing

Contributions are very welcome. Please follow the steps below to contribute:

1. Fork this repository
2. Create a new branch with a descriptive name
3. Add tests and update documentation for your changes
4. Run `sbt compile` and `sbt test` locally
5. Open a Pull Request describing your changes

Code style:

- Prefer idiomatic Scala 3 syntax
- Keep changes localized and provide tests for crucial logic
- When adding native code, include build instructions and platform notes in the PR description

Maintainers will review PRs and request changes if needed. For large changes (new integration paths, major refactors), open an issue first to discuss design.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*Last updated: 2026-06-10*

## Appendix A — Example build.sbt

Below is a minimal `build.sbt` that demonstrates how to declare typical dependencies you may want for this project. Adjust versions to match your environment.

```scala
ThisBuild / scalaVersion     := "3.2.0"
ThisBuild / version          := "0.1.0"
ThisBuild / organization     := "com.example"

lazy val root = (project in file("."))
  .settings(
    name := "cuda-triton-scala",
    libraryDependencies ++= Seq(
      "org.scala-lang" %% "scala3-library" % scalaVersion.value,
      "org.scalaj" %% "scalaj-http" % "2.4.2",
      "io.circe" %% "circe-core" % "0.14.3",
      "io.circe" %% "circe-parser" % "0.14.3",
      // Optional: JavaCPP and PyTorch / CUDA bindings if you choose to use them
      // "org.bytedeco" % "javacpp" % "1.5.13",
      // "org.bytedeco" % "pytorch" % "2.1.0-1.5.13",
    ),
    javacOptions ++= Seq("-Xlint:unchecked", "-Xlint:deprecation")
  )
```

Notes:
- Uncomment JavaCPP or other native-binding dependencies only if you plan to use those libraries and ensure platform classifiers are correct for your OS/CPU/GPU.

## Appendix B — Notes on Triton (triton-lang) vs other serving systems

This project focuses on OpenAI's Triton (triton-lang) CUDA DSL and a Scala-first runtime using JavaCPP-CUDA. If you are instead deploying models with a separate model server (for example NVIDIA Triton Inference Server), you will need a different model repository layout and `config.pbtxt` files specific to that server.

The examples in this repository are scoped to:

- Producing or consuming PTX/CUBIN artifacts (triton-lang or other toolchains)
- Using JavaCPP-CUDA Driver API to load and launch kernels from Scala

If you require examples for server-based deployment (ONNX/TensorRT and model repositories), consult the corresponding server documentation; those topics are outside the main focus of this codebase.

## Appendix C — Long-form Troubleshooting Guide

This section provides step-by-step checks and commands to diagnose common errors encountered when integrating Scala, native libraries, and Triton.

1) Native library load failures (UnsatisfiedLinkError)

- Confirm library file exists:
  - Linux: `ls -l native/libmylib.so`
  - macOS: `ls -l native/libmylib.dylib`
- Check `LD_LIBRARY_PATH` or `java.library.path`:
  - `echo $LD_LIBRARY_PATH`
  - `java -XshowSettings:properties -version 2>&1 | grep java.library.path -A1`
- Use `ldd` to inspect dependencies: `ldd native/libmylib.so` and ensure all symbols resolved.

2) GPU driver / CUDA mismatch

- Check GPU driver & CUDA runtime:
  - `nvidia-smi`
  - `nvcc --version`
- If running in Docker, ensure `--gpus all` and compatible base image. If driver mismatch persists, use CPU-only mode for development or rebuild native code against local CUDA headers.

3) Triton model fails to load

- Inspect Triton server logs (otherwise start Triton with `--log-verbose=1`):
  - `docker logs <triton-container-id>`
- Check model format / backend dependencies (e.g. TensorRT must be available for TensorRT plans). Verify `config.pbtxt` input/output names match model artifacts.

4) Strange numeric differences / precision issues

- Confirm dtype usage (float32 vs float16 vs int8) in client and model config. Small numerical differences may be due to quantization or precision conversion.

5) End-to-end latency spikes

- Collect both client-side and server-side traces. On Triton, enable metrics and tracing; on client, measure request time including serialization time.
- For GPU-bound models, enable profiler (Nsight) to inspect kernel launch times and memory copies.

## Appendix D — Example: Load PTX/CUBIN and launch Triton DSL kernel from Scala

This example demonstrates the recommended Scala workflow for running OpenAI Triton (triton-lang) kernels from a JVM process using JavaCPP-CUDA Driver APIs. It assumes you have a compiled PTX or CUBIN kernel artifact available (packaged in `src/main/resources/native/kernels/` or an external folder set via `NATIVE_KERNEL_DIR`).

Steps:

1. Ensure your kernel artifact (e.g., `add_kernel.ptx`) is available in `src/main/resources/native/kernels/` or `native/kernels/` in the project root.

2. Build the project so resources are packaged:

```bash
sbt clean compile
```

3. Run the Scala launcher example which uses JavaCPP-CUDA to load the PTX and launch the kernel:

```bash
sbt "runMain examples.TritonKernelLauncherExample"
```

What the example does (high level):

- Calls `cuInit(0)` and creates a CUDA context
- Loads kernel module via `cuModuleLoadData` pointing at the PTX bytes loaded from resources
- Allocates GPU buffers via `cuMemAlloc`
- Copies input arrays to device memory using `cuMemcpyHtoD`
- Launches the kernel with `cuLaunchKernel` and the chosen grid/block configuration
- Copies output data back with `cuMemcpyDtoH` and prints results

See `src/main/scala/examples/TritonKernelLauncherExample.scala` for the full example code. The example is intentionally small and checks for common errors (module load, kernel not found, launch failure) and prints brief diagnostics to help debug device/driver mismatches.


## Appendix E — Sample Benchmark Output (synthetic)

```
============================================================
cuda-triton-scala Benchmark Suite
============================================================

--- Simple ONNX HTTP Benchmark ---
Requests: 1000, Batch size: 4
Avg latency (client): 12.4 ms
P95 latency: 34.7 ms
Throughput: 322 req/s

--- TensorRT Local Benchmark ---
Requests: 10000, Batch size: 16
Avg latency: 3.7 ms
P95 latency: 10.2 ms
Throughput: 4200 req/s
```

Note: these numbers are synthetic and for illustration only. Actual numbers depend on model, hardware, and configuration.

## Appendix F — Glossary

- Triton: In this repository, "Triton" refers to OpenAI's Triton (triton-lang), a CUDA DSL for writing GPU kernels. This is different from NVIDIA's Triton Inference Server.
- JavaCPP: A Java library that generates JNI bindings to native C/C++ code
- JNI: Java Native Interface
- ONNX: Open Neural Network Exchange format
- TensorRT: NVIDIA library for high-performance inference on NVIDIA GPUs

## Acknowledgements

This documentation and repository structure borrow organizational ideas from various open-source projects that integrate JVM-based languages with native inference stacks. Thank you to those communities for inspiration and shared tooling.

---

If you'd like, I can also:

- Add concrete implementations of `examples.SimpleCudaExample` and `examples.TritonHttpClientExample` (small runnable mains) so the README examples are directly usable; or
- Add a minimal `LICENSE` file and a small `doc/img/banner.png` placeholder to remove the static warnings; or
- Update the example main class names in README to match actual classes found in your repository—if you tell me where the example mains are I can make the README commands exact.

## Maintainers & Contact

If you have questions about the repository, feel free to open an issue on GitHub or contact the maintainers listed in the repository metadata. For urgent questions, you may also add a comment on the relevant PR.

## Changelog (high level)

- 2026-06-10: Initial expanded documentation and examples; added Triton templates and troubleshooting guide.

---


