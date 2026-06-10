package cuda.dsl.annotations

import scala.annotation.{StaticAnnotation, compileTimeOnly}

/** Annotation to enable GPU execution for a function.
 *  When applied, all collection operations within the function will be
 *  executed on the GPU when possible.
 *
 *  Example:
 *  {{{
 *  @enable_gpu
 *  def process(seq: GPUSeq[Float]): Float = {
 *    seq.map(_ * 2.0f).filter(_ > 0).sum
 *  }
 *  }}}
 */
@compileTimeOnly("enable_gpu must be processed by compiler plugin or macro")
class enable_gpu extends StaticAnnotation {
  inline def apply(inline body: Any): Any = body
}
