add_mlir_dialect_library(
  TorchToSDFG ConvertTorchToSDFG.cpp ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sdfg/Conversion/TorchToSDFG DEPENDS
  MLIRTorchToSDFGPassIncGen)

target_link_libraries(TorchToSDFG PUBLIC MLIRIR) 