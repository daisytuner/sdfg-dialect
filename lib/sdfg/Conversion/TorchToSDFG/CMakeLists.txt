add_mlir_dialect_library(
  TorchToSDFG ConvertTorchToSDFG.cpp ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sdfg/Conversion/TorchToSDFG DEPENDS
  MLIRTorchToSDFGPassIncGen)

target_link_libraries(TorchToSDFG PUBLIC MLIRIR)
target_include_directories(TorchToSDFG PRIVATE ${PROJECT_SOURCE_DIR}/3rdParty/iree/third_party/torch-mlir/include)
target_include_directories(TorchToSDFG PRIVATE ${PROJECT_BINARY_DIR}/3rdParty/iree/compiler/plugins/input/Torch) 