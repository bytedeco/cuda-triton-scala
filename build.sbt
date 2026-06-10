// cuda-dsl module: Pure Scala 3 DSL (no Java sources)
// Artifact: com.cuda-dsl:cuda-dsl_3.8.3:0.1.0

name := "cuda-triton-scala"

crossScalaVersions := Seq("3.8.4")

// Compile ONLY Scala sources (no Java)
//Compile / sources := {
//  val base = (sourceDirectory in Compile).value
//  (base / "scala") ** "*.scala" get
//}

// Override java-output-version from sbt-typelevel plugin
// Also remove -Xkind-projector:underscores to use Scala 3 native ? wildcard syntax
scalacOptions ~= { opts =>
  opts.filterNot(s => s.startsWith("-java-output-version") || s.contains("kind-projector")) ++
    Seq("-java-output-version:17", "-experimental")
}

fork := true
javaOptions += "--add-modules=jdk.incubator.vector"
javaOptions += "--add-exports=jdk.incubator.vector/jdk.incubator.vector=ALL-UNNAMED"

// 对应你的 pom 仓库：central-snapshots
resolvers += (
  "central-snapshots" at "https://central.sonatype.com/repository/maven-snapshots"
  ).withAllowInsecureProtocol(true) // 解决 http 警告

// 开启 SNAPSHOT 每次都更新（对应 updatePolicy=always）

// 阿里云仓库（支持 snapshot，速度最快）
resolvers ++= Seq(
  "aliyun-snapshot" at "https://maven.aliyun.com/repository/public",
  Resolver.sonatypeRepo("snapshots")
)

// 依赖（完全对应你的 pom）
libraryDependencies ++= Seq(
  // PyTorch GPU
  //  "org.bytedeco" % "pytorch-platform-gpu" % "2.11.0-1.5.14",
  "org.javassist" % "javassist" % "3.31.0-GA",
  // JavaCPP
  "org.bytedeco" % "javacpp" % "1.5.14-SNAPSHOT",
  "org.bytedeco" % "javacpp" % "1.5.14-SNAPSHOT" classifier "linux-x86_64",

  // OpenBLAS
  "org.bytedeco" % "openblas" % "0.3.32-1.5.14-SNAPSHOT",
  "org.bytedeco" % "openblas" % "0.3.32-1.5.14-SNAPSHOT" classifier "linux-x86_64",

  // PyTorch
  "org.bytedeco" % "pytorch" % "2.12.0-1.5.14-SNAPSHOT",
  "org.bytedeco" % "pytorch" % "2.12.0-1.5.14-SNAPSHOT" classifier "linux-x86_64",

  "org.bytedeco" % "pytorch-platform-gpu" %  "2.12.0-1.5.14-SNAPSHOT" excludeAll(
    ExclusionRule("org.bytedeco", "cuda-platform"),
    ExclusionRule("org.bytedeco", "javacpp-platform"),
    ExclusionRule("org.bytedeco", "openblas-platform"),
    //      ExclusionRule("org.bytedeco", "pytorch")//,"windows-x86_64" todo do not open ，or cuda not work！！！
  ),
  // CUDA
  "org.bytedeco" % "cuda" % "13.2-9.21-1.5.14-SNAPSHOT",
  "org.bytedeco" % "cuda" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",

  "org.bytedeco" % "cuda-redist" % "13.2-9.21-1.5.14-SNAPSHOT",
  "org.bytedeco" % "cuda-redist-cublas" % "13.2-9.21-1.5.14-SNAPSHOT",
  "org.bytedeco" % "cuda-redist-cudnn" % "13.2-9.21-1.5.14-SNAPSHOT",
  "org.bytedeco" % "cuda-redist-cusolver" % "13.2-9.21-1.5.14-SNAPSHOT",
  "org.bytedeco" % "cuda-redist-cusparse" % "13.2-9.21-1.5.14-SNAPSHOT",
  "org.bytedeco" % "cuda-redist-npp" % "13.2-9.21-1.5.14-SNAPSHOT",
  "org.bytedeco" % "cuda-redist-nccl" % "13.2-9.21-1.5.14-SNAPSHOT",
  "org.bytedeco" % "cuda-redist-nvcomp" % "13.2-9.21-1.5.14-SNAPSHOT",

  "org.bytedeco" % "cuda-redist" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-cublas" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-cudnn" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-cusolver" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-cusparse" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-npp" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-nccl" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-nvcomp" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",



  // OpenCV
  "org.bytedeco" % "opencv" % "4.13.0-1.5.14-SNAPSHOT",
  "org.bytedeco" % "opencv" % "4.13.0-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  //  "org.bytedeco" % "opencv-platform" % "4.13.0-1.5.14-SNAPSHOT",

  // FFmpeg
  "org.bytedeco" % "ffmpeg" % "8.1-1.5.14-SNAPSHOT",
  "org.bytedeco" % "ffmpeg" % "8.1-1.5.14-SNAPSHOT" classifier "linux-x86_64",
  //  "org.bytedeco" % "ffmpeg-platform" % "8.1-1.5.14-SNAPSHOT"
  // Gson
  "com.google.code.gson" % "gson" % "2.14.0",


)
//libraryDependencies ++= Seq(
//  "org.javassist" % "javassist" % "3.31.0-GA",
//  "org.bytedeco" % "javacpp" % "1.5.13" ,
//  "org.bytedeco" % "openblas" % "0.3.31-1.5.13"  ,
//  "org.bytedeco" % "pytorch" % "2.10.0-1.5.13" ,
//  "org.bytedeco" % "pytorch-platform" % "2.10.0-1.5.13" % "provided",
//  "org.bytedeco" % "pytorch-platform-gpu" % "2.10.0-1.5.13" % "provided",
//  "org.bytedeco" % "opencv" % "4.13.0-1.5.13" % "provided",
//  "org.bytedeco" % "ffmpeg" % "8.0.1-1.5.13" % "provided",
//  "org.bytedeco" % "cuda" % "13.1-9.19-1.5.13" % "compile",
//  "org.bytedeco" % "cuda-redist-cublas" % "13.1-9.19-1.5.13" % "compile",
//  "org.bytedeco" % "cuda-redist-cudnn" % "13.1-9.19-1.5.13" % "compile",
//  "org.bytedeco" % "cuda-redist-cusolver" % "13.1-9.19-1.5.13" % "compile",
//  "org.bytedeco" % "cuda-redist-cusparse" % "13.1-9.19-1.5.13" % "compile",
//  "org.bytedeco" % "cuda-redist-nccl" % "13.1-9.19-1.5.13" % "compile",
//  "org.bytedeco" % "cuda-redist-npp" % "13.1-9.19-1.5.13" % "compile",
//  "org.bytedeco" % "cuda-redist-nvcomp" % "13.1-9.19-1.5.13" % "compile"
//)
