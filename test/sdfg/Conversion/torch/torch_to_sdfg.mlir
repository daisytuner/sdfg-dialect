// RUN: sdfg-opt --torch-to-sdfg --allow-unregistered-dialect %s | FileCheck %s

// CHECK: module {
// CHECK:   sdfg.sdfg @main(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,64,55,55],f32> {
// CHECK:     %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1_w_0> : tensor<64x3x3x3xf32>} : () -> !torch.vtensor<[64,3,3,3],f32>
// CHECK:     %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1_b_0> : tensor<64xf32>} : () -> !torch.vtensor<[64],f32>
// CHECK:     %2 = sdfg.library_node "onnx.Conv" ins %arg0, %0, %1 : !torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,3,3],f32>, !torch.vtensor<[64],f32> outs : -> !torch.vtensor<[1,64,111,111],f32> {kernel_shape = [3 : si64, 3 : si64], pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], strides = [2 : si64, 2 : si64]}
// CHECK:     %3 = sdfg.library_node "onnx.Relu" ins %2 : !torch.vtensor<[1,64,111,111],f32> outs : -> !torch.vtensor<[1,64,111,111],f32>
// CHECK:     %4 = sdfg.library_node "onnx.MaxPool" ins %3 : !torch.vtensor<[1,64,111,111],f32> outs : -> !torch.vtensor<[1,64,55,55],f32> {kernel_shape = [3 : si64, 3 : si64], pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], strides = [2 : si64, 2 : si64]}
// CHECK:     sdfg.return %4 : !torch.vtensor<[1,64,55,55],f32>
// CHECK:   }
// CHECK: }

module {
  func.func @main(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,64,55,55],f32> {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1_w_0> : tensor<64x3x3x3xf32>} : () -> !torch.vtensor<[64,3,3,3],f32> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1_b_0> : tensor<64xf32>} : () -> !torch.vtensor<[64],f32>
    %2 = torch.operator "onnx.Conv"(%arg0, %0, %1) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [2: si64, 2 : si64]} : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,3,3],f32>, !torch.vtensor<[64],f32>) -> !torch.vtensor<[1,64,111,111],f32> 
    %3 = torch.operator "onnx.Relu"(%2) : (!torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,111,111],f32> 
    %4 = torch.operator "onnx.MaxPool"(%3) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,55,55],f32> 
    return %4 : !torch.vtensor<[1,64,55,55],f32>
  }
} 