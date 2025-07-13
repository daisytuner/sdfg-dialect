set(LLVM_TARGET_DEFINITIONS ExportSDFG/Passes.td)
mlir_tablegen(ExportSDFG/Passes.h.inc -gen-pass-decls -name ExportSDFG)
add_public_tablegen_target(MLIRExportSDFGPassIncGen) 