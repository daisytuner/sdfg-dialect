// RUN: sdfg-opt --torch-to-sdfg --allow-unregistered-dialect %s | FileCheck %s

// CHECK: module {
// CHECK:   sdfg.sdfg @main(%arg0: !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>) -> !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<55 x !sdfg.array<55 x f32>>>> {
// CHECK:     %0 = sdfg.alloca {value = dense_resource<conv1_w_0> : tensor<64x3x3x3xf32>} : !sdfg.array<64 x !sdfg.array<3 x !sdfg.array<3 x !sdfg.array<3 x f32>>>>
// CHECK:     %1 = sdfg.alloca {value = dense_resource<conv1_b_0> : tensor<64xf32>} : !sdfg.array<64 x f32>
// CHECK:     %2 = sdfg.library_node "Conv" %arg0, %0, %1 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>, !sdfg.array<64 x !sdfg.array<3 x !sdfg.array<3 x !sdfg.array<3 x f32>>>>, !sdfg.array<64 x f32> -> !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<111 x !sdfg.array<111 x f32>>>> {kernel_shape = [3 : si64, 3 : si64], pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], strides = [2 : si64, 2 : si64]}
// CHECK:     %3 = sdfg.library_node "Relu" %2 : !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<111 x !sdfg.array<111 x f32>>>> -> !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<111 x !sdfg.array<111 x f32>>>>
// CHECK:     %4 = sdfg.library_node "MaxPool" %3 : !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<111 x !sdfg.array<111 x f32>>>> -> !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<55 x !sdfg.array<55 x f32>>>> {kernel_shape = [3 : si64, 3 : si64], pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], strides = [2 : si64, 2 : si64]}
// CHECK:     %5 = sdfg.alloca {value = dense<5.000000e-01> : tensor<f32>} : f32
// CHECK:     %6:2 = sdfg.library_node "Dropout" %4, %5 : !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<55 x !sdfg.array<55 x f32>>>>, f32 -> !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<55 x !sdfg.array<55 x f32>>>>, !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<55 x !sdfg.array<55 x i1>>>>
// CHECK:     sdfg.return %6#0 : !sdfg.array<1 x !sdfg.array<64 x !sdfg.array<55 x !sdfg.array<55 x f32>>>>
// CHECK:   }
// CHECK: }

module {
  func.func @main(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,64,55,55],f32> {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1_w_0> : tensor<64x3x3x3xf32>} : () -> !torch.vtensor<[64,3,3,3],f32> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1_b_0> : tensor<64xf32>} : () -> !torch.vtensor<[64],f32>
    %2 = torch.operator "onnx.Conv"(%arg0, %0, %1) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [2: si64, 2 : si64]} : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,3,3],f32>, !torch.vtensor<[64],f32>) -> !torch.vtensor<[1,64,111,111],f32> 
    %3 = torch.operator "onnx.Relu"(%2) : (!torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,111,111],f32> 
    %4 = torch.operator "onnx.MaxPool"(%3) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,55,55],f32> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<5.000000e-01> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %6:2 = torch.operator "onnx.Dropout"(%4, %5) : (!torch.vtensor<[1,64,55,55],f32>, !torch.vtensor<[],f32>) -> (!torch.vtensor<[1,64,55,55],f32>, !torch.vtensor<[1,64,55,55],i1>) 
    return %6#0 : !torch.vtensor<[1,64,55,55],f32>
  }
} 
