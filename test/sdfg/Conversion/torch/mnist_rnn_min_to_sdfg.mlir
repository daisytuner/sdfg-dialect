// RUN: sdfg-opt --torch-to-sdfg --allow-unregistered-dialect %s | FileCheck %s

// Minimal RNN-style model exercising torch.constant.none → sdfg.scalar lowering.

// CHECK: module {
// CHECK:   sdfg.sdfg @main(%arg0: !sdfg.array<4 x !sdfg.array<1 x !sdfg.array<3 x f32>>>>) -> !sdfg.array<4 x !sdfg.array<1 x !sdfg.array<4 x f32>>> {
// CHECK:     %[[NONE:.*]] = sdfg.alloca : !sdfg.scalar<none>
// CHECK:     %[[W:.*]] = sdfg.alloca {value = dense_resource<w> : tensor<1x16x3xf32>} : !sdfg.array<1 x !sdfg.array<16 x !sdfg.array<3 x f32>>>
// CHECK:     %[[R:.*]] = sdfg.alloca {value = dense_resource<r> : tensor<1x16x4xf32>} : !sdfg.array<1 x !sdfg.array<16 x !sdfg.array<4 x f32>>>
// CHECK:     %[[OUT:.*]]:3 = sdfg.library_node "LSTM"  %arg0, %[[W]], %[[R]], %[[NONE]] : {{.*}} {hidden_size = 4 : si64}
// CHECK:     sdfg.return %[[OUT]]#0 : {{.*}}
// CHECK:   }
// CHECK: }

module {
  func.func @main(%arg0: !torch.vtensor<[4,1,3],f32>) -> !torch.vtensor<[4,1,4],f32> {
    %none = torch.constant.none
    %w = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<w> : tensor<1x16x3xf32>} : () -> !torch.vtensor<[1,16,3],f32>
    %r = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<r> : tensor<1x16,4>,f32>} : () -> !torch.vtensor<[1,16,4],f32>
    %0:3 = torch.operator "onnx.LSTM"(%arg0, %w, %r, %none) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[4,1,3],f32>, !torch.vtensor<[1,16,3],f32>, !torch.vtensor<[1,16,4],f32>, !torch.none) -> (!torch.vtensor<[4,1,4],f32>, !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],f32>)
    return %0#0 : !torch.vtensor<[4,1,4],f32>
  }
} 