![cuda-triton-scala 横幅](doc/img/banner.png)
# cuda-triton-scala：Scala + CUDA + Triton 集成示例

[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Scala](https://img.shields.io/badge/scala-3.x-DC322F?style=for-the-badge)](https://www.scala-lang.org/)
[![Triton](https://img.shields.io/badge/Triton-Inference-orange?style=for-the-badge)](https://github.com/triton-inference-server/server)

[English](README.md) | [简体中文](README_zh.md)

## 目录

- [简介](#简介)
- [目标与范围](#目标与范围)
- [主要特性](#主要特性)
- [依赖与要求](#依赖与要求)
- [安装与构建](#安装与构建)
- [快速启动示例](#快速启动示例)
- [工程结构](#工程结构)
- [本地与原生集成模式](#本地与原生集成模式)
- [Triton 集成指南](#triton-集成指南)
- [模型打包示例](#模型打包示例)
- [性能与基准测试](#性能与基准测试)
- [测试与 CI 建议](#测试与-ci-建议)
- [故障排查与常见问题](#故障排查与常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 简介

`cuda-triton-scala` 是一个演示如何在 Scala 生态中与 GPU 本地内核互操作的示例仓库，重点展示通过 JavaCPP-CUDA 驱动 API 从 JVM 加载并启动由 Triton DSL（triton-lang）风格或其它工具链生成的 PTX/CUBIN 内核的流程。

仓库包含：Scala 层的 Triton 风格 DSL（注解/宏）、本地 native 调用示例（JavaCPP/JNI）、以及在 JVM 中加载与执行预编译 GPU 内核的完整示例与诊断指南。

## 目标与范围

本项目的目标包括：

- 提供一个基于 sbt 的可复现构建示例，展示 Scala 与本地库（CUDA）互操作的基本模式；
- 演示 JavaCPP、JNI 及外部进程三种常见互操作方案的权衡与实现；
- 提供 Scala 层的 Triton 客户端示例，以及最小模型仓库模板，便于在本地快速验证服务化路径；
- 分享性能调优、采样规约和 profiling 的实用建议；
- 给出在 CI 中如何在无 GPU 环境下执行单元/集成测试的建议（mock、CPU-only 测试模型等）。

本仓库不包含大型模型二进制，而是提供可替换的模板与小型示例以便快速上手。

## 主要特性

- Scala 3 友好的项目结构与 sbt 构建
 - JavaCPP / JNI 原生调用示例（包含加载与错误处理示例）
 - Scala-first 的 Triton DSL 支持（注解/宏）和如何把内核编译为 PTX/CUBIN 并在 JVM 中加载
 - JavaCPP-CUDA Driver API 的运行时示例（加载模块、分配内存、拷贝、启动 kernel）
- 性能基准脚本、部署建议与故障排查指南

## 依赖与要求

- JDK 17 或更高
- sbt 1.8+（或 `build.sbt` 中声明的兼容版本）
- Scala 3.x
- CUDA 工具包及相应 NVIDIA 驱动（运行 GPU 示例时必需）
- Docker（可选，用于本地运行 Triton 容器）
- curl / grpcurl（可用于手动测试 Triton REST/gRPC 接口）

说明：

- 项目可在没有 GPU 的环境中编译（JVM 层代码），但运行 CUDA 示例需要正确安装驱动与 native 库；
- 如果使用 JavaCPP 预编译的二进制（bytedeco presets），相关本地库通常会被依赖管理工具在运行时解压并加载；在某些系统上需要手动调整 `LD_LIBRARY_PATH`。

## 安装与构建

## 贡献

欢迎贡献！流程建议：

1. Fork 本仓库
2. 新建 feature 分支
3. 编写并提交代码
4. 提交 Pull Request

请保持代码风格一致，并为关键功能添加示例或说明。

## 许可证

本项目采用 MIT 许可证，详见 `LICENSE` 文件。

---

*最后更新：2026-06-10*

## 附录 A — 示例 build.sbt

下面是一个示例 `build.sbt`，用于展示常见依赖与配置。请根据你的环境适当调整依赖版本。

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
      // 可选：JavaCPP / PyTorch bindings，如果需要请启用并确保平台分类器(platform classifier)
      // "org.bytedeco" % "javacpp" % "1.5.13",
    ),
    javacOptions ++= Seq("-Xlint:unchecked", "-Xlint:deprecation")
  )
```

## 附录 B — Triton 配置示例（扩展）

### 1) 带动态 shape 和批处理的 ONNX 配置

```
name: "resnet50_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]
    reshape: { shape: [ 3, 224, 224 ] }
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
instance_group [ { kind: KIND_GPU, count: 1 } ]
dynamic_batching { preferred_batch_size: [4,8,16], max_queue_delay_microseconds: 1000 }
```

### 2) TensorRT 示例（含优化）

```
name: "my_tensorrt_model"
platform: "tensorrt_plan"
max_batch_size: 16
input [ { name: "input", data_type: TYPE_FP32, dims: [ 1, 3, 224, 224 ] } ]
output [ { name: "output", data_type: TYPE_FP32, dims: [ 1000 ] } ]
instance_group { kind: KIND_GPU, count: 1 }
optimization { priority: PRIORITY_MAX }
```

## 附录 C — 详细故障排查

下面给出更系统的诊断步骤，方便你在出现问题时快速定位原因：

1) 本地 native 库加载失败

- 检查文件是否存在：`ls -l native/libmylib.so`（Linux）
- 检查环境变量：`echo $LD_LIBRARY_PATH` 或 `java -XshowSettings:properties -version 2>&1 | grep java.library.path -A1`
- 使用 `ldd` 查看依赖：`ldd native/libmylib.so` 并确保所有依赖都解析成功

2) CUDA 驱动或运行时不匹配

- 使用 `nvidia-smi` 查看驱动版本
- 使用 `nvcc --version` 查看 CUDA 版本
- 容器中运行时需确保宿主机驱动与容器镜像兼容

3) Triton 模型加载失败

- 查看 Triton 日志：`docker logs <triton-container-id>` 或在启动时增加 `--log-verbose` 参数
- 检查 `config.pbtxt` 是否与模型文件匹配（输入名、输出名、dims）

4) 精度与数值差异

- 检查 dtype（FP32/FP16/INT8）是否在客户端与模型端一致；量化带来的数值偏差为正常现象，需要评估精度损失

5) 延迟波动大

- 在客户端与服务器两侧收集时间戳；在 GPU 上使用 Nsight 进行内核级采样

## 附录 D — 示例：在 Scala 中加载 PTX/CUBIN 并启动 Triton 风格内核

本示例说明如何在 JVM 进程中使用 JavaCPP-CUDA 驱动 API 来加载预编译的 PTX 或 CUBIN，并启动内核（无需 Python）。

步骤：

1. 将编译好的 PTX/CUBIN（例如 `add_kernel.ptx`）放置于 `src/main/resources/native/kernels/` 或项目根的 `native/kernels/`，也可以通过环境变量 `NATIVE_KERNEL_DIR` 指定外部路径；
2. 编译项目以确保资源被打包：

```bash
sbt clean compile
```

3. 运行 Scala 启动器示例：

```bash
sbt "runMain examples.TritonKernelLauncherExample"
```

该示例会加载资源并演示 Driver API 的推荐调用序列（`cuInit`、`cuModuleLoadData`、`cuModuleGetFunction`、`cuMemAlloc`、`cuMemcpyHtoD`、`cuLaunchKernel`、`cuMemcpyDtoH` 等）。

## 附录 E — 合成基准示例输出

```
============================================================
cuda-triton-scala 基准套件
============================================================

--- ONNX HTTP 基准 ---
Requests: 1000, Batch size: 4
平均延迟: 12.4 ms
P95 延迟: 34.7 ms
吞吐量: 322 req/s

--- TensorRT 本地基准 ---
Requests: 10000, Batch size: 16
平均延迟: 3.7 ms
P95 延迟: 10.2 ms
吞吐量: 4200 req/s
```

提示：以上为合成数据，仅供参考，真实性能受模型、硬件与配置影响。

## 附录 F — 术语表

- Triton：NVIDIA Triton Inference Server（推理服务）
- JavaCPP：生成 Java 与 C++ 绑定的工具
- JNI：Java Native Interface
- ONNX：开放神经网络交换格式
- TensorRT：NVIDIA 的高性能推理库

## 致谢

本仓库的文档结构和示例借鉴了多方开源项目的思路，感谢这些社区提供的工具和最佳实践。

---

如果你愿意，我可以：

- 添加 `examples.SimpleCudaExample` 与 `examples.TritonHttpClientExample` 的可运行主类；
- 添加最小的 `LICENSE` 文件与一个 `doc/img/banner.png` 占位图以消除 README 的静态警告；
- 或者根据仓库中的实际主类名更新 README 中的示例命令。

```bash
sbt clean compile
```

4. 运行测试（如有）

```bash
sbt test
```

依赖说明：项目尽量保持外部依赖最小，如需额外的 JavaCPP preset、TensorRT 或其他本地依赖，请在 `build.sbt` 中添加对应依赖。

## 快速启动示例

下面列出的示例用于验证本地环境与示例代码是否工作。请根据 `src/main/scala/examples` 中的实际类名替换命令中的主类。

### 1) 纯 CPU 示例（验证 sbt 构建与运行）

```bash
sbt "runMain examples.SimpleCpuExample"
```

期望输出：有限的日志与运行信息，演示纯 Scala 业务逻辑。

### 2) 本地 CUDA 示例（JavaCPP/JNI）

确保 native 库已放置在 `LD_LIBRARY_PATH` 或 `java.library.path` 可访问的位置。

```bash
export LD_LIBRARY_PATH=$(pwd)/native:$LD_LIBRARY_PATH
sbt "runMain examples.SimpleCudaExample"
```

常见错误：`UnsatisfiedLinkError`（找不到 native 库）、CUDA 驱动/设备不匹配等。参见故障排查。

### 3) 在 Scala 中加载并启动 Triton 风格内核（Driver API）

运行我们在仓库中提供的启动器示例，该示例会检测并（在可用时）使用 JavaCPP-CUDA Driver API 加载内核并启动：

```bash
sbt "runMain examples.TritonKernelLauncherExample"
```

### 4) JavaCPP-CUDA 示例

演示如何检测 JavaCPP-CUDA 驱动类并展示调用片段（不会直接执行内核），用于快速检查运行时依赖：

```bash
sbt "runMain examples.JavaCppCudaExample"
```

示例文件位于 `src/main/scala/examples/`，可以直接参考或修改以适配你自己的内核与参数传递逻辑。

## 工程结构（建议）

```
cuda-triton-scala/
├── README.md
├── README_zh.md
├── build.sbt
├── project/
├── src/
│   ├── main/
│   │   ├── scala/
│   │   │   ├── cuda/
│   │   │   │   ├── CudaHelpers.scala
│   │   │   │   ├── NativeLoader.scala
│   │   │   │   └── CudaBindings.java
│   │   │   ├── triton/
│   │   │   │   ├── TritonHttpClient.scala
│   │   │   │   ├── TritonGrpcClient.scala
│   │   │   │   └── TritonUtils.scala
│   │   │   └── examples/
│   │   │       ├── SimpleCpuExample.scala
│   │   │       ├── SimpleCudaExample.scala
│   │   │       ├── TritonHttpClientExample.scala
│   │   │       └── TritonGrpcClientExample.scala
│   │   └── resources/
│   │       ├── native/
│   │       └── triton_model_templates/
│   └── test/
│       └── scala/
└── scripts/
    ├── run_triton.sh
    └── benchmark.sh
```

以上仅为推荐结构，请根据实际代码与模块划分进行调整。

## 本地与原生集成模式

本仓库展示了几种常见的 JVM <-> 原生互操作模式，适用于不同的约束和需求：

1. JavaCPP（推荐存在成熟预设时）
   - 优点：自动管理 native 二进制、较少 JNI 栈代码
   - 使用场景：已有 JavaCPP 预设或可通过 JavaCPP 生成绑定的库

2. JNI（手工）
   - 优点：对接口与数据布局完全可控
   - 使用场景：需要高性能、零拷贝或复杂内存管理的场合

3. 外部进程（RPC）
   - 优点：隔离原生崩溃、简化 JVM 与本地库版本依赖
   - 使用场景：原生依赖复杂、需要独立生命周期或容器化运行时

4. Triton 服务化（生产建议）
   - 优点：提供模型版本管理、并发控制、自动批处理与多后端支持
   - 使用场景：需要稳定、可扩展且易于运维的推理服务

### 数据传输与内存考虑

- 避免不必要的数据拷贝：使用 DirectByteBuffer 或 JavaCPP 指针传递大数组；
- 零拷贝需要严格的生命周期管理，避免 JVM GC 导致的内存问题；
- 明确 tensor 的 dtype、shape 与 layout 规范，避免端到端数据解释错误。

## Triton 集成指南

仓库中包含了用于在 Triton 上部署与测试模型的示例和模板。下面说明常见流程。

### Triton 模型仓库布局

Trition 要求如下目录结构：

```
models/
└── my_model/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

模型文件应放在版本目录（如 `1/`）下，`config.pbtxt` 描述输入、输出和其他模型选项。

### 最小 `config.pbtxt` 示例

```
name: "simple_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]
```

根据模型实际输入输出调整 `platform`、`dims`、`data_type` 等字段。

（注意：本仓库面向 OpenAI 的 triton-lang（Triton CUDA DSL）和 Scala 驱动层示例；并不依赖 NVIDIA Triton Inference Server 的模型仓库与 HTTP/gRPC 接口。若你确实需要使用 Triton Server，请参考官方文档或其他专门的示例仓库。）

### 从 Scala 加载并启动 PTX/CUBIN（Driver API）

关键步骤：

1. 读取并加载 PTX/CUBIN 二进制（例如从 classpath 的 `native/kernels` 目录或外部 `NATIVE_KERNEL_DIR`）；
2. 调用 CUDA Driver API（`cuInit`, `cuDeviceGet`, `cuCtxCreate`）；
3. 使用 `cuModuleLoadData` / `cuModuleLoad` 加载内核模块；
4. 使用 `cuModuleGetFunction` 获取内核函数句柄；
5. 使用 `cuMemAlloc` 分配设备内存并用 `cuMemcpyHtoD` 传输输入；
6. 使用 `cuLaunchKernel` 启动内核（传入 grid/block、参数指针与 stream）；
7. 使用 `cuMemcpyDtoH` 将输出拷回主机，最后释放资源。

仓库中的 `examples/TritonKernelLauncherExample.scala` 给出一个完整的启动流程示例（基于 JavaCPP-CUDA 驱动 API）。

## 模型打包示例

1. 将模型导出为 Triton 支持的格式（ONNX、TensorRT plan、TorchScript 等）；
2. 编写并检查 `config.pbtxt`；
3. 将模型放入版本目录（例如 `models/my_model/1/`）；
4. 可选：加入 tokenizer、词表或前/后处理脚本。

仓库 `triton_model_templates/` 中包含若干常见后端的配置模版，便于复制与修改。

## 性能与基准测试

### 微基准

- 使用 `scripts/benchmark.sh` 进行简单吞吐量/延迟测试；
- GPU 模型应先进行若干次热身以使 JIT/内核等准备就绪；

### Profiling

- CPU 层：使用 async-profiler 或 VisualVM；
- GPU 层：使用 Nsight Systems/Compute，关注内核执行与内存带宽；

### 性能优化建议

- 使用合适的批次（batching），对 Triton 配置 `max_batch_size`；
- 调整 Triton `instance_group` 和实例数量以提高吞吐；
- 考虑使用 FP16/INT8 来减少带宽和显存占用（测试精度影响）；
- 避免在请求路径中进行重量级 I/O 或同步阻塞操作。

## 测试与 CI 建议

- 单元测试：对依赖 native 的部分使用 mock 或抽象层以便在无 GPU 环境下运行；
- 集成测试：在 CI 中使用 CPU-only 的 Triton 镜像或在单独的集成 job 中运行带有容器的测试；
- CI 矩阵示例：
  - JDK 17 + sbt compile + unit tests（Linux）
  - JDK 17 + sbt compile + unit tests（macOS）
  - 可选：Integration job 启动 Triton（CPU）并运行 `TritonHttpClientExample`

## 故障排查与常见问题

Q: `UnsatisfiedLinkError` / 无法加载 native 库？

A: 确认 `LD_LIBRARY_PATH`（Linux）或 `DYLD_LIBRARY_PATH`（macOS）包含 native 库目录，或设置 `-Djava.library.path`。如果使用 JavaCPP，请确保依赖正确并且平台 classifier 匹配当前系统。

Q: CUDA 驱动不匹配 / `CUDA_ERROR_UNKNOWN`？

A: 检查 `nvidia-smi` 输出确认驱动版本与 CUDA 兼容性。容器中运行时请确认宿主驱动与容器镜像兼容。

Q: Triton 报 model not ready 或加载失败？

A: 查看 Triton 日志，检查模型文件、`config.pbtxt` 是否正确，确认 Triton 镜像包含模型所需的后端库（例如 TensorRT）。

Q: 性能不如预期？

A: 使用 `nvidia-smi` / Nsight 检查 GPU 利用率。常见瓶颈：预处理占 CPU、批次过小、PCIe 传输开销或单线程客户端。

Q: JNI 符号不匹配？

A: 确保 native 编译时使用 `-fPIC` 并使用与 JVM 兼容的头文件和符号，确认包名与签名与 Java 端一致。

## 示例代码片段（重点说明）

以下片段为示例性代码，实际请查看 `src/main/scala/examples` 下完整实现。

### NativeLoader（Scala）示例

```scala
package cuda

object NativeLoader {
  def load(name: String): Unit = {
    val libDir = sys.env.getOrElse("NATIVE_LIB_DIR", "./native")
    val path = s"$libDir/lib$name.so"
    System.load(path)
  }
}
```

### JavaCPP 风格调用（伪代码）

```scala
object CudaOps {
  def add(a: Array[Float], b: Array[Float]): Array[Float] = {
    val n = a.length
    val out = new Array[Float](n)
    NativeCuda.add(n, a, b, out) // 假设 JavaCPP 绑定已生成
    out
  }
}
```

### Triton HTTP 客户端示例（伪代码）

```scala
import scalaj.http.Http

object TritonHttpClient {
  def infer(model: String, input: Array[Float]): Array[Float] = {
    val payload = Map("inputs" -> List(Map("name" -> "input__0", "shape" -> List(1, input.length), "data" -> input.toList)))
    val resp = Http(s"http://localhost:8000/v2/models/$model/infer").postData(payload.toString).asString
    // 解析响应并返回输出数组
    Array.emptyFloatArray
  }
}
```

## 贡献指南

欢迎贡献！请按照下面流程提交变更：

1. Fork 仓库
2. 创建 feature 分支并实现变更
3. 添加测试与更新文档
4. 在本地运行 `sbt compile` 与 `sbt test`
5. 提交 PR 并在 PR 描述中说明变更内容及测试方式

关于代码风格：

- 使用 Scala 3 现代语法与 idiomatic 风格；
- 大变更（例如新增原生互操作模式或重大重构）请先开 Issue 讨论；
- 添加 native 代码时请在 PR 中注明支持的平台与构建步骤。

## 许可证

本项目采用 MIT 许可证，详见 `LICENSE` 文件。

---

*最后更新：2026-06-10*

本仓库借鉴了 Torch-RecHub-Scala 的文档风格，提供清晰的使用说明、示例代码和常见问题提示，便于在本地开发、测试以及向 Triton 部署模型。

## 亮点

- 支持 Scala 3 的项目结构和 sbt 构建
- 提供调用 CUDA 加速本地代码的最小适配层（Java/JavaCPP/JNI 等可选实现）
- 含 Triton 客户端示例，用于演示模型请求与返回的处理
- 关注可复现的本地开发流程与简单部署路径

## 依赖与要求

- Scala 3.x
- sbt 1.8+
- JDK 17+
- CUDA 工具包与驱动（用于在 GPU 上运行示例）
- （可选）NVIDIA Triton Inference Server 用于模型上线和低延迟服务

注意：本项目的本地运行依赖于系统的原生驱动与库。即使在没有 GPU 的环境下也可以编译通过，但 GPU 示例需要正确安装 CUDA 与驱动。

## 快速开始

1. 克隆仓库

```bash
git clone https://github.com/your-repo/cuda-triton-scala.git
cd cuda-triton-scala
```

2. 使用 sbt 编译

```bash
sbt compile
```

3. 运行示例（将示例主类替换为 `src/main/scala/examples` 中实际的类名）

```bash
sbt "runMain examples.SimpleCudaExample"
```

如果示例涉及 Triton，请确保 Triton 服务已启动，且模型仓库已正确配置。

## 工程结构

```
cuda-triton-scala/
├── README.md                # 英文文档
├── README_zh.md             # 中文文档
├── build.sbt                # sbt 配置
├── src/main/scala/          # Scala 源码
│   ├── cuda/                # 与 CUDA 本地调用相关的适配代码
│   ├── triton/              # Triton 客户端工具（可选）
│   └── examples/            # 示例程序
└── src/main/resources/      # 原生库、配置、模型模板等
```

请根据项目实际代码调整路径与模块名称。

## 示例

- `examples.SimpleCudaExample` — 载入本地 CUDA 帮助方法并运行一个小型 GPU 计算（如果可用）；
- `examples.TritonClientExample` — 演示如何将推理请求发送到本地 Triton 服务并处理返回结果。

通过 sbt 运行示例，或直接查看 `src/main/scala/examples` 下的代码以了解配置细节。

## 与 Triton 一起使用

1. 按照 Triton 的仓库布局和模型配置规范准备模型仓库。
2. 本地启动 Triton（使用 Docker 或二进制），并将模型仓库挂载或指向 Triton 的 --model-repository 参数。
3. 使用仓库中提供的 Scala Triton 客户端工具构建并发送推理请求。

示例：使用 Docker 启动 Triton（仅示例，请根据实际镜像与版本调整）：

```bash
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /path/to/model/repo:/models nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
```

## 贡献

欢迎贡献！流程建议：

1. Fork 本仓库
2. 新建 feature 分支
3. 编写并提交代码
4. 提交 Pull Request

请保持代码风格一致，并为关键功能添加示例或说明。

## 许可证

本项目采用 MIT 许可证，详见 `LICENSE` 文件。

---

*最后更新：2026-06-10*

