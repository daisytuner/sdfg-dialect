get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSDFG
        ExportSDFG
        )
add_llvm_executable(sdfg-export sdfg-export.cpp)

llvm_update_compile_flags(sdfg-export)
target_link_libraries(sdfg-export PRIVATE ${LIBS}) 