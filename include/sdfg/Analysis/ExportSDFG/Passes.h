#ifndef SDFG_Analysis_ExportSDFG_Passes_H
#define SDFG_Analysis_ExportSDFG_Passes_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::sdfg::analysis {

// Global variable for output prefix
extern std::string g_outputPrefix;

// Factory to create the ExportSDFG pass.
std::unique_ptr<mlir::Pass> createExportSDFGPass();

// Set the output prefix for the ExportSDFG pass.
void setExportSDFGOutputPrefix(const std::string& prefix);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "sdfg/Analysis/ExportSDFG/Passes.h.inc"

} // namespace mlir::sdfg::analysis

#endif // SDFG_Analysis_ExportSDFG_Passes_H 