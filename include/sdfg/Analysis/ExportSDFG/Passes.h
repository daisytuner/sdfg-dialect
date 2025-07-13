#ifndef SDFG_Analysis_ExportSDFG_Passes_H
#define SDFG_Analysis_ExportSDFG_Passes_H

#include "mlir/Pass/Pass.h"

namespace mlir::sdfg::analysis {

// Factory to create the ExportSDFG pass.
std::unique_ptr<mlir::Pass> createExportSDFGPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "sdfg/Analysis/ExportSDFG/Passes.h.inc"

} // namespace mlir::sdfg::analysis

#endif // SDFG_Analysis_ExportSDFG_Passes_H 