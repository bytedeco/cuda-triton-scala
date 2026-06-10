package cuda.dsl

/** Package-level type aliases for kernel function signatures.
  * These allow shorthand like TritonKernel[K3F1I] instead of full function types.
  * Format: K{N}F{M}I = (Float×N, Int×M) => Unit
  */
package object runtime:
  type K2F    = (Float, Float)                                              => Unit
  type K2F1I  = (Float, Float, Int)                                         => Unit
  type K2F2I  = (Float, Float, Int, Int)                                    => Unit
  type K3F1I  = (Float, Float, Float, Int)                                  => Unit
  type K3F2I  = (Float, Float, Float, Int, Int)                             => Unit
  type K3F3I  = (Float, Float, Float, Int, Int, Int)                        => Unit
  type K3F4I  = (Float, Float, Float, Int, Int, Int, Int)                   => Unit
  type K3F5I  = (Float, Float, Float, Int, Int, Int, Int, Int)              => Unit
  type K4F1I  = (Float, Float, Float, Float, Int)                          => Unit
  type K4F2I  = (Float, Float, Float, Float, Int, Int)                     => Unit
  type K4F3I  = (Float, Float, Float, Float, Int, Int, Int)                 => Unit
  type K4F4I  = (Float, Float, Float, Float, Int, Int, Int, Int)            => Unit
  type K5F1I  = (Float, Float, Float, Float, Float, Int)                   => Unit
  type K5F2I  = (Float, Float, Float, Float, Float, Int, Int)               => Unit
  type K5F3I  = (Float, Float, Float, Float, Float, Int, Int, Int)           => Unit
  type K5F4I  = (Float, Float, Float, Float, Float, Int, Int, Int, Int)     => Unit
  type K5F5I  = (Float, Float, Float, Float, Float, Int, Int, Int, Int, Int) => Unit
  type K6F1I  = (Float, Float, Float, Float, Float, Float, Int)              => Unit
  type K6F2I  = (Float, Float, Float, Float, Float, Float, Int, Int)        => Unit
  type K6F3I  = (Float, Float, Float, Float, Float, Float, Int, Int, Int)    => Unit
  type K7F1I  = (Float, Float, Float, Float, Float, Float, Float, Int)      => Unit
  type K7F2I  = (Float, Float, Float, Float, Float, Float, Float, Int, Int) => Unit
  type K7F3I  = (Float, Float, Float, Float, Float, Float, Float, Int, Int, Int) => Unit
  type K8F1I  = (Float, Float, Float, Float, Float, Float, Float, Float, Int) => Unit
  type K8F2I  = (Float, Float, Float, Float, Float, Float, Float, Float, Int, Int) => Unit
  type K8F3I  = (Float, Float, Float, Float, Float, Float, Float, Float, Int, Int, Int) => Unit
  type K8F4I  = (Float, Float, Float, Float, Float, Float, Float, Float, Int, Int, Int, Int) => Unit
  type K9F1I  = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Int) => Unit
  type K9F2I  = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Int, Int) => Unit
  type K9F3I  = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Int, Int, Int) => Unit
  type K10F2I = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Int, Int) => Unit
