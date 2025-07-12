#ifndef SDFG_Conversion_TorchToSDFG_PassDetail_H
#define SDFG_Conversion_TorchToSDFG_PassDetail_H

#include "sdfg/Dialect/SDFGDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sdfg {
namespace conversion {

/// Generate the code for base classes.
#define GEN_PASS_CLASSES
#include "sdfg/Conversion/TorchToSDFG/Passes.h.inc"

} // namespace conversion
} // namespace sdfg
} // end namespace mlir

#endif // SDFG_Conversion_TorchToSDFG_PassDetail_H 