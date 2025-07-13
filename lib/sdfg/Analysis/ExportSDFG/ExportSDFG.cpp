#include "sdfg/Analysis/ExportSDFG/PassDetail.h"
#include "sdfg/Analysis/ExportSDFG/Passes.h"

#include "sdfg/Dialect/SDFGOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/serializer/json_serializer.h"

namespace {

struct ExportSDFGPass : public mlir::sdfg::analysis::ExportSDFGPassBase<ExportSDFGPass> {
  void visit_alloca(mlir::sdfg::AllocaOp allocaOp) {
    llvm::outs() << "AllocaOp\n";
  }

  void visit_library_node(mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    llvm::outs() << "LibraryNode\n";
  }

  void visit_return(mlir::sdfg::ReturnOp returnOp) {
    llvm::outs() << "ReturnOp\n";
  }

  std::unique_ptr<sdfg::StructuredSDFG> simplify(std::unique_ptr<sdfg::StructuredSDFG>& sdfg) {
    // TODO: Implement
    return std::move(sdfg);
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    module.walk([&](mlir::sdfg::SDFGNode sdfgNode) {
      std::string sdfgName;
      if (auto symAttr = sdfgNode->getAttrOfType<mlir::StringAttr>("sym_name"))
        sdfgName = symAttr.getValue();
      else
        sdfgName = "_unnamed_";

      sdfg::builder::StructuredSDFGBuilder builder(sdfgName, sdfg::FunctionType_CPU);

      // Visit arguments of the SDFG function.
      auto funcTypeAttr = sdfgNode->getAttrOfType<mlir::TypeAttr>("function_type");
      if (funcTypeAttr) {
        auto funcType = funcTypeAttr.getValue().cast<mlir::FunctionType>();
        for (auto arg : funcType.getInputs()) {
          // Add name and type to sdfg
          // builder.add_container(name, type, true);
        }
      }

      // Visit all operations in the SDFG body region.
      sdfgNode.getBody().walk([&](mlir::Operation *innerOp) {
        if (auto allocaOp = dyn_cast<mlir::sdfg::AllocaOp>(innerOp)) {
          visit_alloca(allocaOp);
        } else if (auto libraryNodeOp = dyn_cast<mlir::sdfg::LibraryNodeOp>(innerOp)) {
          visit_library_node(libraryNodeOp);
        } else if (auto returnOp = dyn_cast<mlir::sdfg::ReturnOp>(innerOp)) {
          visit_return(returnOp);
        } else {
          throw std::runtime_error("Unsupported operation: " +
                                   innerOp->getName().getStringRef().str());
        }
      });

      // Finish SDFG
      auto sdfg = builder.move();

      // Simplify SDFG
      sdfg = simplify(sdfg);

      // Serialize SDFG to JSON
      sdfg::serializer::JSONSerializer serializer;
      auto j = serializer.serialize(sdfg);
      std::filesystem::path sdfgPath = sdfgName + ".json";

      std::ofstream ofs(sdfgPath);
      if (!ofs.is_open()) {
          throw std::runtime_error("Failed to open file: " + sdfgPath.string());
      }
      ofs << j.dump(2);
      ofs.close();

    });
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::sdfg::analysis::createExportSDFGPass() {
  return std::make_unique<ExportSDFGPass>();
} 