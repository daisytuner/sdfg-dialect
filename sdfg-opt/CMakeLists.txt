get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSDFG
        LinalgToSDFG
        TorchToSDFG
        # IREE Torch-MLIR targets
        iree::compiler::plugins::input::Torch::InputConversion
        iree::compiler::plugins::input::Torch::torch-mlir::TorchConversionDialectIR
        iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectIR
        iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectPasses
        iree::compiler::plugins::input::Torch::torch-mlir::TorchOnnxToTorchPasses
        iree::compiler::plugins::input::Torch::torch-mlir::ConversionPasses
        iree::compiler::plugins::input::Torch::torch-mlir-dialects::TMTensorDialectIR
        )
add_llvm_executable(sdfg-opt sdfg-opt.cpp)

llvm_update_compile_flags(sdfg-opt)
target_link_libraries(sdfg-opt PRIVATE ${LIBS})
