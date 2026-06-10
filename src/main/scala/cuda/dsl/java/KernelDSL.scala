package cuda.dsl.java

import cuda.dsl.dsl._

/** Simple Java API for generating CUDA kernels using Scala TTIR
 *
 *  Usage from Java:
 *  {{{
 *  String cuda = KernelDSL.create("vectorAdd")
 *      .param("float*", "out")
 *      .param("float*", "a")
 *      .param("float*", "b")
 *      .param("int", "n")
 *      .stmt("if (i >= n) return;")
 *      .stmt("out[i] = a[i] + b[i];")
 *      .emit();
 *  }}}
 */
object KernelDSL {

  class KernelBuilder(private val name: String) {

    private val paramList = scala.collection.mutable.ListBuffer[(String, String)]()
    private val stmtList = scala.collection.mutable.ListBuffer[String]()

    def param(tpe: String, name: String): KernelBuilder = {
      paramList += ((tpe, name))
      this
    }

    def paramFloatPointer(name: String): KernelBuilder = param("float*", name)
    def paramIntPointer(name: String): KernelBuilder = param("int*", name)
    def paramFloat(name: String): KernelBuilder = param("float", name)
    def paramInt(name: String): KernelBuilder = param("int", name)
    def paramLong(name: String): KernelBuilder = param("long", name)
    def paramDouble(name: String): KernelBuilder = param("double", name)

    def stmt(code: String): KernelBuilder = {
      stmtList += code
      this
    }

    def body(lines: String*): KernelBuilder = {
      stmtList ++= lines
      this
    }

    def emit(): String = {
      val sb = new StringBuilder
      sb ++= s"extern \"C\" __global__ void $name("
      sb ++= paramList.map(p => s"${p._1} ${p._2}").mkString(", ")
      sb ++= ") {\n"
      sb ++= "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
      stmtList.foreach(s => sb ++= s"    $s\n")
      sb ++= "}\n"
      sb.toString
    }

    def emitWithGridType(gridType: String): String = {
      val sb = new StringBuilder
      sb ++= s"extern \"C\" __global__ void $name("
      sb ++= paramList.map(p => s"${p._1} ${p._2}").mkString(", ")
      sb ++= ") {\n"

      gridType match {
        case "1D" =>
          sb ++= "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        case "2D" =>
          sb ++= "    int i = blockIdx.y * gridDim.x + blockIdx.x;\n"
          sb ++= "    int j = threadIdx.y;\n"
        case "3D" =>
          sb ++= "    int i = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;\n"
          sb ++= "    int j = threadIdx.y;\n"
          sb ++= "    int k = threadIdx.z;\n"
      }

      stmtList.foreach(s => sb ++= s"    $s\n")
      sb ++= "}\n"
      sb.toString
    }

    // Registry support
    def register(): Unit = {
      TritonKernelRunner.registerKernelSource(name, emit())
    }
  }

  def create(name: String): KernelBuilder = new KernelBuilder(name)

  // Convenience method for simple kernels
  def vectorAdd(): String = {
    create("vectorAdd")
      .paramFloatPointer("out")
      .paramFloatPointer("a")
      .paramFloatPointer("b")
      .paramInt("n")
      .stmt("if (i >= n) return;")
      .stmt("out[i] = a[i] + b[i];")
      .emit()
  }

  def relu(): String = {
    create("relu")
      .paramFloatPointer("out")
      .paramFloatPointer("in")
      .paramInt("n")
      .stmt("if (i >= n) return;")
      .stmt("out[i] = fmaxf(in[i], 0.0f);")
      .emit()
  }

  def layerNorm(): String = {
    create("layerNorm")
      .paramFloatPointer("out")
      .paramFloatPointer("in")
      .paramFloatPointer("gamma")
      .paramFloatPointer("beta")
      .paramInt("n")
      .stmt("if (i >= n) return;")
      .stmt("float mean = 0.0f;")
      .stmt("for (int j = 0; j < n; j++) mean += in[j];")
      .stmt("mean /= n;")
      .stmt("float x_norm = (in[i] - mean) / sqrtf(mean + 1e-5f);")
      .stmt("out[i] = gamma[i] * x_norm + beta[i];")
      .emit()
  }
}