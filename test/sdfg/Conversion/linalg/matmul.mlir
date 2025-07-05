// RUN: sdfg-opt --linalg-to-sdfg --allow-unregistered-dialect %s | FileCheck %s

// CHECK: func.func @main() -> tensor<256x256xf32> {
// CHECK:   %0 = tensor.empty() : tensor<256x16xf32>
// CHECK:   %1 = tensor.empty() : tensor<16x256xf32>
// CHECK:   %2 = tensor.empty() : tensor<256x256xf32>
// CHECK:   %3 = sdfg.library_node "math.gemm" ins %0, %1 : tensor<256x16xf32>, tensor<16x256xf32> outs %2 : tensor<256x256xf32> -> tensor<256x256xf32>
// CHECK:   return %3 : tensor<256x256xf32>
// CHECK: }
func.func @main() -> tensor<256x256xf32> {
  %0 = tensor.empty() : tensor<256x16xf32>
  %1 = tensor.empty() : tensor<16x256xf32>
  %2 = tensor.empty() : tensor<256x256xf32>
  %3 = linalg.matmul ins(%0, %1 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %3 : tensor<256x256xf32>
}
