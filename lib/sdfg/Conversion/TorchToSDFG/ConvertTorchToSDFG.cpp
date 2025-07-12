#include "sdfg/Conversion/TorchToSDFG/PassDetail.h"
#include "sdfg/Conversion/TorchToSDFG/Passes.h"
#include "sdfg/Dialect/SDFGDialect.h"
#include "sdfg/Dialect/SDFGOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace sdfg;
using namespace conversion;

namespace {

struct TorchOperatorToLibraryNodePattern : public RewritePattern {
  TorchOperatorToLibraryNodePattern(MLIRContext *context)
      : RewritePattern("torch.operator", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only match torch.operator ops
    if (!op->getName().getStringRef().starts_with("torch.operator"))
      return failure();

    // Get the operator name, e.g., "onnx.Conv"
    std::string opName;
    if (auto attr = op->getAttrOfType<StringAttr>("name")) {
      opName = attr.getValue().str();
    } else {
      // Try to parse from the op's custom format: torch.operator "onnx.Conv"
      auto opStr = op->getName().getStringRef().str();
      auto quotePos = opStr.find('"');
      if (quotePos != std::string::npos) {
        auto endQuote = opStr.find('"', quotePos + 1);
        if (endQuote != std::string::npos)
          opName = opStr.substr(quotePos + 1, endQuote - quotePos - 1);
      }
    }

    // If this is a torch.operator "onnx.Constant", do not lower
    if (opName == "onnx.Constant")
      return failure();

    // Lower to sdfg.library_node
    SmallVector<Value> inputs, outputs;
    for (auto operand : op->getOperands())
      inputs.push_back(operand);
    // Don't add results as output operands - this creates a dominance issue
    // The library node should only have inputs and return results

    // Extract relevant attributes from the torch operator
    SmallVector<NamedAttribute> attributes;
    for (auto attr : op->getAttrs()) {
      // Skip the "name" attribute as it's handled by the code field
      if (attr.getName() == "name")
        continue;
      
      // Remove 'torch.onnx.' prefix if present
      auto nameStr = attr.getName().str();
      StringRef nameRef = nameStr;
      if (nameRef.starts_with("torch.onnx.")) {
        nameRef = nameRef.drop_front(strlen("torch.onnx."));
      }
      attributes.push_back(NamedAttribute(mlir::StringAttr::get(op->getContext(), nameRef), attr.getValue()));
    }

    // Use the opName as the code for the library node
    auto libraryNode = rewriter.create<LibraryNodeOp>(
        op->getLoc(),
        op->getResultTypes(),
        rewriter.getStringAttr(opName),
        inputs,
        outputs);

    // Add the extracted attributes to the library node
    for (auto attr : attributes) {
      libraryNode->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.replaceOp(op, libraryNode.getResults());
    return success();
  }
};

struct TorchToSDFGPass : public sdfg::conversion::TorchToSDFGPassBase<TorchToSDFGPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<TorchOperatorToLibraryNodePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::sdfg::conversion::createTorchToSDFGPass() {
  return std::make_unique<TorchToSDFGPass>();
} 