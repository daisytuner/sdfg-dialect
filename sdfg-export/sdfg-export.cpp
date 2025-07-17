#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/InitAllDialects.h"
#include "llvm/Support/ToolOutputFile.h"

#include "sdfg/Dialect/SDFGDialect.h"
#include "sdfg/Dialect/SDFGOpsDialect.cpp.inc"
#include "sdfg/Analysis/ExportSDFG/Passes.h"

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/serializer/json_serializer.h>

int main(int argc, char **argv) {

  // Register MLIR core dialects and all others.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::sdfg::SDFGDialect>();

  // Register our SDFG exporter pass.
  mlir::sdfg::analysis::registerExportSDFGPasses();

  sdfg::codegen::register_default_dispatchers();
  sdfg::serializer::register_default_serializers();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SDFG exporter\n", registry));
} 