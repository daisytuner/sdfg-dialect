#ifndef SDFG_Analysis_ExportSDFG
#define SDFG_Analysis_ExportSDFG

include "mlir/Pass/PassBase.td"
include "sdfg/Dialect/SDFGDialect.td"

//===----------------------------------------------------------------------===//
// Export SDFG Pass Definition
//===----------------------------------------------------------------------===//

def ExportSDFGPass : Pass<"export-sdfg", "ModuleOp"> {
  let summary = "Walk all sdfg.sdfg nodes and exports an SDFG";
  let constructor = "mlir::sdfg::analysis::createExportSDFGPass()";
  let dependentDialects = ["mlir::sdfg::SDFGDialect"];
}

#endif // SDFG_Analysis_ExportSDFG 