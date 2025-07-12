#ifndef SDFG_Conversion_TorchToSDFG_H
#define SDFG_Conversion_TorchToSDFG_H

#include "mlir/Pass/Pass.h"

namespace mlir::sdfg::conversion {

/// Creates a Torch to sdfg converting pass
std::unique_ptr<Pass> createTorchToSDFGPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "sdfg/Conversion/TorchToSDFG/Passes.h.inc"

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_TorchToSDFG_H 