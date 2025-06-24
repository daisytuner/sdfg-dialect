get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSDFG
        )
add_llvm_executable(sdfg-opt sdfg-opt.cpp)

llvm_update_compile_flags(sdfg-opt)
target_link_libraries(sdfg-opt PRIVATE ${LIBS})
