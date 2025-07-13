#include "sdfg/Analysis/ExportSDFG/PassDetail.h"
#include "sdfg/Analysis/ExportSDFG/Passes.h"

#include "sdfg/Dialect/SDFGOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::sdfg;
using namespace mlir::sdfg::analysis;

namespace {

struct ExportSDFGPass : public ExportSDFGPassBase<ExportSDFGPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](SDFGNode sdfgNode) {
      llvm::outs() << "=== SDFG Node: ";
      if (auto symAttr = sdfgNode->getAttrOfType<StringAttr>("sym_name"))
        llvm::outs() << symAttr.getValue();
      else
        llvm::outs() << "<unnamed>";
      llvm::outs() << " ===\n";

      // Fetch function type held in attribute.
      auto funcTypeAttr = sdfgNode->getAttrOfType<TypeAttr>("function_type");
      if (funcTypeAttr) {
        auto funcType = funcTypeAttr.getValue().cast<FunctionType>();
        llvm::outs() << "  Num Inputs : " << funcType.getNumInputs() << "\n";
        llvm::outs() << "  Num Results: " << funcType.getNumResults() << "\n";
      }

      // Count operations inside the SDFG body region.
      size_t opCount = 0;
      sdfgNode.getBody().walk([&](Operation *innerOp) {
        // Skip the region terminator
        if (isa<ReturnOp>(innerOp))
          return;
        ++opCount;
      });
      llvm::outs() << "  Contained ops: " << opCount << "\n";

      // TODO: Extend to gather and print more metadata.
    });
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::sdfg::analysis::createExportSDFGPass() {
  return std::make_unique<ExportSDFGPass>();
} 