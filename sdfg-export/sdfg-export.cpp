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
#include <cstdlib>
#include <vector>
#include <string>

int main(int argc, char **argv) {

  // Register MLIR core dialects and all others.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::sdfg::SDFGDialect>();

  // Register our SDFG exporter pass.
  mlir::sdfg::analysis::registerExportSDFGPasses();

  sdfg::codegen::register_default_dispatchers();
  sdfg::serializer::register_default_serializers();

  // Parse command line arguments manually to extract -o option
  std::string outputPrefix;
  std::vector<std::string> mlirArgs;
  
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      outputPrefix = argv[i + 1];
      i++; // Skip the next argument as well
    } else if (arg.substr(0, 2) == "-o" && arg.length() > 2) {
      // Handle -o=value format
      outputPrefix = arg.substr(2);
    } else {
      mlirArgs.push_back(arg);
    }
  }

  // Set the output prefix if provided
  if (!outputPrefix.empty()) {
    mlir::sdfg::analysis::setExportSDFGOutputPrefix(outputPrefix);
  }

  // Convert back to argc/argv format for MLIR
  std::vector<char*> mlirArgv;
  mlirArgv.push_back(argv[0]); // program name
  for (const auto& arg : mlirArgs) {
    mlirArgv.push_back(const_cast<char*>(arg.c_str()));
  }
  int mlirArgc = mlirArgv.size();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(mlirArgc, mlirArgv.data(), "SDFG exporter\n", registry));
} 