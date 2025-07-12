#include "sdfg/Conversion/TorchToSDFG/PassDetail.h"
#include "sdfg/Conversion/TorchToSDFG/Passes.h"
#include "sdfg/Dialect/SDFGDialect.h"
#include "sdfg/Dialect/SDFGOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace sdfg;
using namespace conversion;

namespace {

// Lower `torch.operator "onnx.Constant"` to `sdfg.alloca` while forwarding the
// constant payload via a `value` attribute.
struct TorchConstantToAllocaPattern : public RewritePattern {
  TorchConstantToAllocaPattern(MLIRContext *context)
      : RewritePattern("torch.operator", /*benefit=*/2, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // We are only interested in `torch.operator "onnx.Constant"`.
    if (!op->getName().getStringRef().starts_with("torch.operator"))
      return failure();

    // Extract the operator name (e.g., "onnx.Constant").  This follows the
    // same logic as the library-node pattern below.
    std::string opName;
    if (auto attr = op->getAttrOfType<StringAttr>("name")) {
      opName = attr.getValue().str();
    } else {
      auto opStr = op->getName().getStringRef().str();
      auto quotePos = opStr.find('"');
      if (quotePos != std::string::npos) {
        auto endQuote = opStr.find('"', quotePos + 1);
        if (endQuote != std::string::npos)
          opName = opStr.substr(quotePos + 1, endQuote - quotePos - 1);
      }
    }

    if (opName != "onnx.Constant")
      return failure();

    // Acquire the constant payload attribute (typically a DenseResourceAttr).
    Attribute valueAttr = op->getAttr("torch.onnx.value");

    if (!valueAttr)
      return failure();

    // Generate the `sdfg.alloca` op producing the same type as the constant.
    Type resultType = op->getResult(0).getType();

    auto alloca = rewriter.create<sdfg::AllocaOp>(op->getLoc(), resultType, valueAttr);

    rewriter.replaceOp(op, alloca.getResult());
    return success();
  }
};

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
    MLIRContext *ctx = &getContext();

    // 1. Lower torch.operator operations to sdfg.library_node.
    RewritePatternSet patterns(ctx);
    patterns.add<TorchConstantToAllocaPattern, TorchOperatorToLibraryNodePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // 2. Convert func.func operations to sdfg.sdfg and their terminators to
    //    sdfg.return.  This happens after the pattern rewrite above so that the
    //    body of the function already contains library nodes instead of torch
    //    operators.
    for (auto funcOp : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      // Prepare a rewriter anchored at the current func.func op.
      PatternRewriter rewriter(ctx);
      rewriter.setInsertionPoint(funcOp);

      // Create the new sdfg.sdfg operation with the same signature and name.
      auto funcType = funcOp.getFunctionType();
      auto nameAttr = rewriter.getStringAttr(funcOp.getName());
      auto typeAttr = TypeAttr::get(funcType);

      // The SDFG op takes optional argument/result attribute arrays.  Forward
      // them if they exist.
      ArrayAttr argAttrs = funcOp.getAllArgAttrs();
      ArrayAttr resAttrs = funcOp.getAllResultAttrs();

      auto sdfgOp = rewriter.create<SDFGNode>(funcOp.getLoc(), nameAttr, typeAttr,
                                              argAttrs ? argAttrs : ArrayAttr(),
                                              resAttrs ? resAttrs : ArrayAttr());

      // Move the entire region body from the original function into the SDFG
      // op.  First, remove the automatically inserted empty block in the SDFG
      // op so we can splice the original block(s) directly.
      {
        Region &sdfgRegion = sdfgOp.getBody();
        if (!sdfgRegion.empty())
          rewriter.eraseBlock(&sdfgRegion.front());
      }

      // Inline the region using the rewriter helper to maintain bookkeeping.
      rewriter.inlineRegionBefore(funcOp.getBody(), sdfgOp.getBody(),
                                  sdfgOp.getBody().end());

      // Convert any func.return terminators inside the newly inlined region.
      sdfgOp.walk([&](func::ReturnOp retOp) {
        rewriter.setInsertionPoint(retOp);
        rewriter.replaceOpWithNewOp<sdfg::ReturnOp>(retOp, retOp.getOperands());
      });

      // Finally erase the original function.
      rewriter.eraseOp(funcOp);
    }
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::sdfg::conversion::createTorchToSDFGPass() {
  return std::make_unique<TorchToSDFGPass>();
} 