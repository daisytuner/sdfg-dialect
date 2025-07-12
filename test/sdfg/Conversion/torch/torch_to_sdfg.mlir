// RUN: sdfg-opt --torch-to-sdfg --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @main(%arg0: !torch.vtensor<[1,3,224,224],f32>, %arg1: !torch.vtensor<[64,3,3,3],f32>, %arg2: !torch.vtensor<[64],f32>) -> !torch.vtensor<[1,64,55,55],f32> {
    // CHECK: %c = torch.operator "onnx.Constant"
    %c = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1_w_0> : tensor<64x3x3x3xf32>} : () -> !torch.vtensor<[64,3,3,3],f32>
    // CHECK: %0 = sdfg.library_node "onnx.Conv"
    %0 = torch.operator "onnx.Conv"(%arg0, %c, %arg2) {torch.onnx.kernel_shape = [3 : si64, 3 : si64]} : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,3,3],f32>, !torch.vtensor<[64],f32>) -> !torch.vtensor<[1,64,111,111],f32>
    // CHECK: %1 = sdfg.library_node "onnx.Relu"
    %1 = torch.operator "onnx.Relu"(%0) : (!torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,111,111],f32>
    // CHECK: %2 = sdfg.library_node "onnx.MaxPool"
    %2 = torch.operator "onnx.MaxPool"(%1) {torch.onnx.kernel_shape = [3 : si64, 3 : si64]} : (!torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,55,55],f32>
    return %2 : !torch.vtensor<[1,64,55,55],f32>
  }
} 