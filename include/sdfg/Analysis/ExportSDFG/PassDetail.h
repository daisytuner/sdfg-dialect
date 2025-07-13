#ifndef SDFG_Analysis_ExportSDFG_PassDetail_H
#define SDFG_Analysis_ExportSDFG_PassDetail_H

#include "sdfg/Dialect/SDFGDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sdfg {
namespace analysis {

// Generate the code for base classes.
#define GEN_PASS_CLASSES
#include "sdfg/Analysis/ExportSDFG/Passes.h.inc"

} // namespace analysis
} // namespace sdfg
} // namespace mlir

#endif // SDFG_Analysis_ExportSDFG_PassDetail_H 