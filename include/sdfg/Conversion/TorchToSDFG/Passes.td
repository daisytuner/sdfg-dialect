#ifndef SDFG_Conversion_TorchToSDFG
#define SDFG_Conversion_TorchToSDFG

include "mlir/Pass/PassBase.td"
include "sdfg/Dialect/SDFGDialect.td"

/// Define Torch to SDFG Pass.
def TorchToSDFGPass : Pass<"torch-to-sdfg", "ModuleOp"> {
  let summary = "Convert Torch dialect to SDFG dialect";
  let constructor = "mlir::sdfg::conversion::createTorchToSDFGPass()";
  let dependentDialects = ["mlir::sdfg::SDFGDialect"];
}

#endif // SDFG_Conversion_TorchToSDFG 